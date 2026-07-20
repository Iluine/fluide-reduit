"""Tests F1 — sonde EPS (N1), §A19 (pocCascade2phys c3d13a1).

Ce que ces tests protègent :
  1. la RÈGLE Q2 est appliquée telle qu'elle est gravée — le PLUS GRAND
     EPS sous le seuil, et AUCUN par défaut si personne ne passe ;
  2. k reste FIGÉ — balayer k perceptuellement atteindrait la condition
     de réveil σ_ω, la garde le refuse fail-loud ;
  3. le VIVANT est bien ce que le CPU sait (remontée seuillée,
     incrémentale, L1 incluse), pas une copie déguisée de la vérité —
     sans quoi la sonde mesurerait zéro et retiendrait n'importe quel EPS ;
  4. l'observable passe par les primitives EXISTANTES et vit en espace
     instrument, jamais sur l'état."""
import json

import numpy as np
import pytest

from scripts.run_f1_ma_quater import PipelineMaQuater, slots_v4
from scripts.run_f1_sonde_eps import (
    CADENCE_FIGEE,
    CADENCE_MESURE,
    CADENCE_VERIFICATION,
    CADENCES_ADMISES,
    FACTEUR_INDETERMINATION,
    LABEL_INDETERMINE,
    LABEL_MECANISME_EN_QUESTION,
    LABEL_PEREMPTION,
    N_REPETITIONS_PLANCHER,
    SEUIL_DISCRIMINATION_RELATIVE,
    TABLE_BRANCHES,
    VERITE_COURANTE,
    VERITE_TRANSPORTEE,
    CHAMP_SEDIMENT,
    EPS_BALAYAGE,
    EPS_EN_VIGUEUR,
    SEUIL_IC_BAS,
    ReconstructeurNiveau0,
    _slot_observe,
    champ_verite,
    branche_decision_k,
    delta_chi_readout,
    discrimination_du_balayage,
    exiger_cadence_admise,
    exiger_cadence_figee,
    exiger_verification_instrument,
    lecture_innocuite,
    lecture_mecanique,
    plancher_bruit_replicat,
    verifier_instrument,
)
from src.albedo import albedo, delta_chi
from src.f1_gpu.backend import cupy_disponible
from src.f1_gpu.pyramide import GeometriePyramide
from src.f1_gpu.transferts import TransfertComptable

gpu_requis = pytest.mark.skipif(
    not cupy_disponible(), reason="cupy/device CUDA indisponible")

GEO_PETITE = GeometriePyramide(n_fov=64, n_niv=10, n0=8,
                               slots=slots_v4(10), emboitee=True)


def _pipeline(cp, eps: float, k: int = CADENCE_FIGEE):
    transferts = TransfertComptable(cp)
    return PipelineMaQuater(cp, GEO_PETITE, transferts, k=k, eps=eps,
                            capturer_coefficients=True)


# ----- 1. la règle Q2, telle qu'elle est gravée -----

def _mesure(eps: float, dchi_max: float, octets: float = 1000.0,
            dchi_max_n1: float | None = None) -> dict:
    """Mesure factice au format v2 : les DEUX vérités (§A21-complément).

    `dchi_max_n1` par défaut égal à vérité(n) — le cas neutre, où la
    frame de retard ne coûte rien et où (iii-bis) ne peut pas tirer."""
    if dchi_max_n1 is None:
        dchi_max_n1 = dchi_max
    par_verite = {
        VERITE_COURANTE: {"delta_chi_max": dchi_max,
                          "delta_chi_max_sans_decimation": dchi_max},
        VERITE_TRANSPORTEE: {"delta_chi_max": dchi_max_n1,
                             "delta_chi_max_sans_decimation": dchi_max_n1},
    }
    return {
        "eps": eps,
        "delta_chi": {"delta_chi_max": dchi_max,
                      "verite_de_la_regle": VERITE_COURANTE,
                      "par_verite": par_verite,
                      "prix_peremption_une_frame": dchi_max - dchi_max_n1},
        "chrono": {"octets_par_frame_median": octets},
    }


def _verification(valide: bool = True) -> dict:
    """Vérification d'instrument factice — DUE avant toute lecture."""
    return {"instrument_valide": valide, "cadence_verification": 1,
            "eps_passants_a_k1": [1e-5] if valide else [],
            "lecture_preecrite": "x", "nota_peremption_minimale": "y",
            "discrimination": {"a_discrimine": valide}, "portee": "z"}


def _lire(mesures: list[dict], valide: bool = True,
          plancher: float = 0.0) -> dict:
    return lecture_mecanique(mesures, _verification(valide),
                             plancher)


def test_retient_le_plus_grand_eps_sous_le_seuil():
    """La règle ne dit pas « le meilleur Δχ » mais « le PLUS GRAND EPS »
    qui passe : c'est le plus économique en trafic."""
    lecture = _lire([
        _mesure(1e-5, 0.001), _mesure(1e-4, 0.010),
        _mesure(3e-4, 0.030), _mesure(1e-3, 0.070)])
    assert lecture["eps_retenu"] == 3e-4
    assert lecture["autre_remonte"] is False
    assert lecture["eps_satisfaisants"] == [1e-5, 1e-4, 3e-4]


def test_aucun_eps_retenu_par_defaut_si_aucun_ne_passe():
    """Y compris 1e-5 : AUTRE remonté, et surtout AUCUN repli — « jamais
    de choix après courbe »."""
    lecture = _lire([
        _mesure(1e-5, 0.08), _mesure(1e-4, 0.12)])
    assert lecture["eps_retenu"] is None
    assert lecture["autre_remonte"] is True
    assert "AUCUN EPS retenu par défaut" in (
        lecture["branche_preecrite_si_aucun"])


def test_le_seuil_est_strict_et_vaut_ic_bas():
    """< 0.0603, pas <=. La lecture porte sur l'IC entier (§A13)."""
    assert SEUIL_IC_BAS == 0.0603
    assert _lire([_mesure(1e-4, SEUIL_IC_BAS)])["eps_retenu"] is None
    assert _lire([_mesure(1e-4, SEUIL_IC_BAS - 1e-9)])["eps_retenu"] == 1e-4


def test_le_balayage_grave_est_reproduit():
    assert EPS_BALAYAGE == (1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
    assert EPS_EN_VIGUEUR == 1e-4
    assert EPS_EN_VIGUEUR in EPS_BALAYAGE


# ----- le balayage a-t-il seulement DISCRIMINÉ ? -----

def test_signale_un_balayage_qui_ne_discrimine_pas():
    """Le trafic varie d'un facteur énorme mais Δχ ne bouge pas :
    l'observable ne VOIT pas EPS. La note dit désormais POURQUOI ce test
    était ininterprétable en v1 — le décalage spatial du reconstructeur y
    plaquait un plancher insensible à tout — et que l'observable est
    corrigé (§A21-complément)."""
    mesures = [_mesure(1e-5, 0.0806, octets=200000.0),
               _mesure(1e-2, 0.0805, octets=100.0)]
    diagnostic = discrimination_du_balayage(mesures, plancher=0.0)
    assert diagnostic["a_discrimine"] is False
    assert diagnostic["critere_forme"] is False        # la FORME manque
    assert diagnostic["critere_echelle"] is True       # l'ÉCHELLE suffit
    assert diagnostic["amplitude_relative"] < 0.01
    assert diagnostic["rapport_octets"] == pytest.approx(2000.0)
    assert "décalage spatial" in diagnostic["note"]
    assert "aucune valeur de la v1 n'est reconduite" in diagnostic["note"]


def test_la_discrimination_exige_les_deux_criteres():
    """(A) §A22 : un rapport spectaculaire entre deux nombres tous deux
    DANS LE BRUIT ne discrimine pas. La FORME seule ne suffit plus."""
    mesures = [_mesure(1e-5, 1e-6), _mesure(1e-2, 1e-4)]
    assert discrimination_du_balayage(mesures, plancher=0.0)[
        "a_discrimine"] is True                # hors bruit : discrimine
    serre = discrimination_du_balayage(mesures, plancher=1e-3)
    assert serre["critere_forme"] is True      # 99 % d'amplitude relative
    assert serre["critere_echelle"] is False   # mais sous le plancher
    assert serre["a_discrimine"] is False


def test_sans_plancher_la_discrimination_reste_indeterminee():
    """Plutôt que de prononcer sur la FORME seule — la faute de la v1 —
    l'absence de plancher laisse la discrimination indéterminée."""
    diagnostic = discrimination_du_balayage(
        [_mesure(1e-5, 0.010), _mesure(1e-2, 0.090)])
    assert diagnostic["plancher_evalue"] is False
    assert diagnostic["critere_forme"] is True
    assert diagnostic["a_discrimine"] is False


def test_le_seuil_relatif_est_conserve_et_justifie():
    """Le critère de FORME est SANS ÉCHELLE : il ne devient pas faux
    quand l'observable passe de ~0.082 à ~1e-4. C'est son isolement qui
    était le défaut, pas sa valeur."""
    assert SEUIL_DISCRIMINATION_RELATIVE == 0.05
    note = discrimination_du_balayage(
        [_mesure(1e-5, 0.01), _mesure(1e-2, 0.09)], plancher=0.0)["note"]
    assert "FORME" in note and "ÉCHELLE" in note
    assert "réplicat" in note


def test_aucune_explication_refutee_nest_portee_comme_vraie():
    """Garde anti-régression, élargie à la v2 : ni l'explication par la
    péremption (falsifiée par le bras k=1), ni le « plancher structurel »
    (expliqué depuis : c'était le décalage spatial) ne doivent revenir
    comme des faits. Le module doit en outre porter son statut de
    SUPERSEDED, faute de quoi une lecture v1 pourrait être reprise."""
    import scripts.run_f1_sonde_eps as sonde
    diagnostic = discrimination_du_balayage(
        [_mesure(1e-5, 0.0806), _mesure(1e-2, 0.0805)], plancher=0.0)
    assert "gouverné par la PÉREMPTION" not in diagnostic["note"]
    assert "PLANCHER STRUCTUREL" not in diagnostic["note"]
    doc = " ".join(sonde.__doc__.split())
    assert "SUPERSEDED" in doc
    assert "Aucune lecture de la v1 n'est reconduite" in doc


def test_signale_un_balayage_qui_discrimine():
    """Contre-épreuve : si Δχ suit EPS, le diagnostic le dit."""
    diagnostic = discrimination_du_balayage(
        [_mesure(1e-5, 0.010), _mesure(1e-2, 0.090)], plancher=0.0)
    assert diagnostic["a_discrimine"] is True
    assert diagnostic["amplitude_relative"] > 0.5


def test_la_lecture_porte_le_diagnostic_de_discrimination():
    lecture = _lire([_mesure(1e-5, 0.0806), _mesure(1e-2, 0.0805)])
    assert lecture["discrimination_du_balayage"]["a_discrimine"] is False
    assert lecture["autre_remonte"] is True     # les deux dépassent


# ----- (A) la vérification d'instrument, DUE avant toute lecture -----

def test_aucune_lecture_sans_verification_enregistree():
    """Câblage « muette => remonter » (B9) : sans le PASS/FAIL de la
    vérification, aucune lecture k=4 n'est produite. Une lecture sans
    instrument vérifié serait un chiffre dont on ne saurait pas s'il
    mesure EPS ou la péremption."""
    for absente in (None, {}, {"autre_chose": 1}):
        with pytest.raises(RuntimeError, match="SONDE MUETTE"):
            lecture_mecanique([_mesure(1e-4, 0.01)], absente)
    with pytest.raises(RuntimeError, match="SONDE MUETTE"):
        exiger_verification_instrument(None)


def test_instrument_valide_si_k1_discrimine_et_un_eps_passe():
    """Lecture pré-écrite (i) : Δχ répond à EPS ET au moins un EPS passe."""
    valide, details = verifier_instrument(
        [_mesure(1e-5, 0.010), _mesure(1e-2, 0.090)], plancher=0.0)
    assert valide is True
    assert details["instrument_valide"] is True
    assert details["eps_passants_a_k1"] == [1e-5]
    assert "VALIDE" in details["lecture_preecrite"]
    # (C) §A22 : la conclusion ne dépend plus de la discrimination.
    assert "DIAGNOSTIC" in details["lecture_preecrite"]


def test_sonde_muette_si_k1_ne_discrimine_pas():
    """Lecture pré-écrite (ii) : AUTRE D'INSTRUMENT — ne rien régler."""
    valide, details = verifier_instrument(
        [_mesure(1e-5, 0.0806), _mesure(1e-2, 0.0805)], plancher=0.0)
    assert valide is False
    assert "MUETTE" in details["lecture_preecrite"]
    assert "ne rien régler" in details["lecture_preecrite"].lower()
    assert details["branche"] == "ii"


def test_branche_iii_mecanisme_de_remontee_en_question():
    """(iii) : le balayage DISCRIMINE mais aucun EPS ne passe, sur AUCUNE
    des deux vérités. Ni sonde muette (elle discrimine), ni retard
    coupable (le retirer ne sauve aucun EPS) — le résidu pointe AILLEURS.

    Le plancher de bruit est fourni : sans lui la discrimination reste
    indéterminée (§A22)."""
    valide, details = verifier_instrument(
        [_mesure(1e-5, 0.070), _mesure(1e-2, 0.200)], plancher=0.0)
    assert valide is False
    assert details["discrimination"]["a_discrimine"] is True
    assert details["eps_passants_a_k1"] == []
    assert details["branche"] == "iii"
    assert details["label"] == LABEL_MECANISME_EN_QUESTION
    assert "pointe AILLEURS" in details["lecture_preecrite"]


def test_branche_iii_bis_designe_le_retard_et_non_le_mecanisme():
    """(iii-bis), §A22 : aucun EPS ne passe sur vérité(n) MAIS au moins un
    passe sur vérité(n−1). Retirer la seule frame de retard suffirait donc
    à satisfaire la règle — la cause est LE RETARD, et le mécanisme de
    remontée est EXONÉRÉ."""
    valide, details = verifier_instrument(
        [_mesure(1e-5, 0.070, dchi_max_n1=0.010),
         _mesure(1e-2, 0.200, dchi_max_n1=0.150)], plancher=0.0)
    assert valide is False
    assert details["branche"] == "iii-bis"
    assert details["label"] == LABEL_PEREMPTION
    assert details["eps_passants"][VERITE_COURANTE] == []
    assert details["eps_passants"][VERITE_TRANSPORTEE] == [1e-5]
    assert "la cause est LE RETARD" in details["lecture_preecrite"]
    assert "EXONÉRÉ" in details["lecture_preecrite"]
    # (iii-bis) n'est PAS une attribution de mécanisme : rien à armer.
    assert details["attributions_branche_iii"] is None


def test_iii_bis_prime_meme_si_le_balayage_ne_discrimine_pas():
    """ORDRE REMONTÉ COMME CHOIX : (iii-bis) repose sur une mesure
    DIRECTE (le prix du retard), pas sur la pente du balayage. Une sonde
    peut être muette sur EPS tout en mesurant parfaitement ce prix — les
    deux axes sont indépendants, et « la cause est le retard » est plus
    informatif que « sonde muette »."""
    _, details = verifier_instrument(
        [_mesure(1e-5, 0.0806, dchi_max_n1=0.010),
         _mesure(1e-2, 0.0805, dchi_max_n1=0.010)], plancher=0.0)
    assert details["discrimination"]["a_discrimine"] is False
    assert details["branche"] == "iii-bis"


def test_la_discrimination_ne_gate_quen_cas_dechec():
    """(C) §A22 : au moins un EPS passe ⇒ l'instrument SUFFIT, la
    discrimination passe en DIAGNOSTIC. Aucun ne passe ⇒ elle GATE."""
    _, passe = verifier_instrument(
        [_mesure(1e-5, 0.0100), _mesure(1e-2, 0.0101)], plancher=0.0)
    assert passe["branche"] == "i"                  # malgré une pente nulle
    assert passe["discrimination"]["a_discrimine"] is False
    assert passe["discrimination_gate"] is False
    assert "DIAGNOSTIC" in passe["role_de_la_discrimination"]

    _, echoue = verifier_instrument(
        [_mesure(1e-5, 0.0806), _mesure(1e-2, 0.0805)], plancher=0.0)
    assert echoue["discrimination_gate"] is True
    assert "GATE" in echoue["role_de_la_discrimination"]


def test_lattribution_a_est_desormais_armee_par_la_verite_transportee():
    """(a) « le retard d'une frame suffit à lui seul » avait pour
    falsificateur « comparer contre la vérité DÉCALÉE d'une frame ». La v2
    la reporte à chaque frame : (a) est ARMÉE, et (iii-bis) est sa
    lecture. (b) reste NON armée, son bras étant gaté."""
    _, details = verifier_instrument(
        [_mesure(1e-5, 0.070), _mesure(1e-2, 0.200)], plancher=0.0)
    attributions = details["attributions_branche_iii"]
    retard = attributions["a_retard_dune_frame"]
    derive = attributions["b_derive_reference_incrementale"]
    assert retard["arme"] is True
    assert "vérité(n−1)" in retard["arme_par"]
    assert derive["arme"] is False
    assert "GATÉ" in derive["statut_du_bras"]


def test_les_attributions_nexistent_que_pour_la_branche_iii():
    """(i), (ii) et (iii-bis) ne portent aucune attribution de mécanisme —
    chacune a sa propre lecture."""
    cas = [
        [_mesure(1e-5, 0.010), _mesure(1e-2, 0.090)],            # (i)
        [_mesure(1e-5, 0.0806), _mesure(1e-2, 0.0805)],          # (ii)
        [_mesure(1e-5, 0.070, dchi_max_n1=0.010),                # (iii-bis)
         _mesure(1e-2, 0.200, dchi_max_n1=0.150)],
    ]
    for mesures in cas:
        _, details = verifier_instrument(mesures, plancher=0.0)
        assert details["attributions_branche_iii"] is None


def test_la_table_de_branches_est_close_sur_cinq_issues():
    """Les CINQ issues sont exhaustives et mutuellement exclusives, et le
    test les exerce TOUTES (§A22-complément).

    Le plancher vaut 0 partout sauf pour l'indétermination, où il est
    choisi assez grand pour que le balayage entier tienne dans 3× le
    bruit."""
    cas = {
        "i": ([_mesure(1e-5, 0.010), _mesure(1e-2, 0.090)], 0.0),
        "iii-bis": ([_mesure(1e-5, 0.070, dchi_max_n1=0.010),
                     _mesure(1e-2, 0.200, dchi_max_n1=0.150)], 0.0),
        "iii": ([_mesure(1e-5, 0.070), _mesure(1e-2, 0.200)], 0.0),
        "ii": ([_mesure(1e-5, 0.0806), _mesure(1e-2, 0.0805)], 0.0),
        # amplitude 0.08 < 3 × 0.05 = 0.15 : rien n'est résolu.
        "indetermine": ([_mesure(1e-5, 0.010), _mesure(1e-2, 0.090)], 0.05),
    }
    observees = set()
    for attendue, (mesures, plancher) in cas.items():
        _, details = verifier_instrument(mesures, plancher=plancher)
        assert details["branche"] == attendue, attendue
        assert details["label"] in TABLE_BRANCHES[attendue]
        observees.add(details["branche"])
    assert observees == {"i", "ii", "iii", "iii-bis", "indetermine"}
    assert set(TABLE_BRANCHES) == {"i", "ii", "iii", "iii-bis",
                                   "indetermine"}


def test_lindetermination_precede_toutes_les_branches():
    """La règle est en TÊTE de l'arbre : elle recouvre même (i). Le MÊME
    jeu de mesures — où un EPS passe — bascule d'INSTRUMENT VALIDE à
    INDÉTERMINÉ selon le seul plancher."""
    mesures = [_mesure(1e-5, 0.010), _mesure(1e-2, 0.090)]
    _, tranche = verifier_instrument(mesures, plancher=0.0)
    assert tranche["branche"] == "i"
    _, noye = verifier_instrument(mesures, plancher=0.05)
    assert noye["branche"] == "indetermine"
    assert noye["label"] == LABEL_INDETERMINE
    assert noye["instrument_valide"] is False
    assert "AUCUNE branche" in noye["lecture_preecrite"]
    # ... et les attributions de mécanisme ne sont pas portées non plus.
    assert noye["attributions_branche_iii"] is None


def test_le_seuil_dindetermination_vaut_trois_fois_le_plancher():
    """Frontière exacte, fixée AVANT les chiffres : strictement en deçà
    de 3 × plancher, on ne prononce pas ; à partir de 3 ×, on prononce."""
    assert FACTEUR_INDETERMINATION == 3.0
    plancher = 0.01
    juste_en_deca = [_mesure(1e-5, 0.010),
                     _mesure(1e-2, 0.010 + 3 * plancher - 1e-9)]
    juste_au_dela = [_mesure(1e-5, 0.010),
                     _mesure(1e-2, 0.010 + 3 * plancher)]
    _, a = verifier_instrument(juste_en_deca, plancher=plancher)
    _, b = verifier_instrument(juste_au_dela, plancher=plancher)
    assert a["branche"] == "indetermine"
    assert b["branche"] != "indetermine"


def test_sans_plancher_evalue_rien_nest_prononce():
    """On ne prononce pas sur un bruit inconnu : l'absence de plancher
    donne INDÉTERMINÉ, pas une branche par défaut."""
    _, details = verifier_instrument(
        [_mesure(1e-5, 0.010), _mesure(1e-2, 0.090)])
    assert details["branche"] == "indetermine"
    assert "bruit inconnu" in details["indetermination"]["motif"]


def test_le_plancher_exige_trois_repetitions():
    """Deux passes ne donnaient qu'UN tirage de la dispersion. Le driver
    refuse fail-loud en deçà de trois (§A22-complément)."""
    assert N_REPETITIONS_PLANCHER == 3
    repetition = _mesure(1e-4, 0.010)["delta_chi"]
    for n in (1, 2):
        with pytest.raises(RuntimeError, match="répétition"):
            plancher_bruit_replicat([repetition] * n)


def test_le_plancher_est_la_dispersion_des_repetitions():
    """Plancher = plus grand écart observé : la DISPERSION (max − min) de
    chaque grandeur, prise sur les deux vérités."""
    bruit = plancher_bruit_replicat([
        _mesure(1e-4, 0.0100)["delta_chi"],
        _mesure(1e-4, 0.0104)["delta_chi"],
        _mesure(1e-4, 0.0101)["delta_chi"]])
    assert bruit["plancher_delta_chi"] == pytest.approx(0.0004, abs=1e-12)
    assert bruit["n_repetitions"] == 3
    assert bruit["deterministe"] is False


def test_un_plancher_nul_est_dit_deterministe():
    """Trois passes identiques : la chaîne est DÉTERMINISTE, ce qui ne
    veut pas dire infiniment précise — et c'est dit."""
    identique = _mesure(1e-4, 0.010)["delta_chi"]
    bruit = plancher_bruit_replicat([identique] * 3)
    assert bruit["plancher_delta_chi"] == 0.0
    assert bruit["deterministe"] is True
    assert "DÉTERMINISTE" in bruit["note_si_nul"]


def test_chaque_branche_porte_sa_lecture_preecrite():
    for branche, texte in TABLE_BRANCHES.items():
        assert texte.startswith(f"({branche})")


def test_la_verification_nomme_la_frame_de_retard():
    """k=1 conserve la frame de retard du compteur : péremption MINIMALE,
    pas nulle — et c'est dit au JSON."""
    _, details = verifier_instrument([_mesure(1e-5, 0.01)])
    assert "frame de retard" in details["nota_peremption_minimale"]
    assert "PAS NULLE" in details["nota_peremption_minimale"]
    assert "n'est PAS un balayage de k" in details["portee"]


def test_les_deux_cadences_sont_admises_et_pas_une_de_plus():
    """k ∈ {1 (vérif), 4 (mesure)} — toute autre valeur serait un
    balayage de k, donc le réveil σ_ω."""
    assert CADENCES_ADMISES == (CADENCE_VERIFICATION, CADENCE_MESURE)
    assert (CADENCE_VERIFICATION, CADENCE_MESURE) == (1, 4)
    exiger_cadence_admise(1)
    exiger_cadence_admise(4)
    for k in (0, 2, 3, 8):
        with pytest.raises(RuntimeError, match="σ_ω"):
            exiger_cadence_admise(k)


def test_la_lecture_porte_la_verification():
    lecture = _lire([_mesure(1e-4, 0.01)])
    assert lecture["verification_instrument"]["instrument_valide"] is True


# ----- (B) l'échelle de lecture est ÉPINGLÉE -----

def test_la_regle_lit_le_decime_et_le_non_decime_est_diagnostic():
    lecture = _lire([_mesure(1e-4, 0.01)])
    echelle = lecture["echelle_de_lecture"]
    assert "DÉCIMÉ" in echelle["grandeur_de_la_regle"]
    assert "DIAGNOSTIC SEUL" in echelle["non_decime"]
    assert "pas monotone" in echelle["non_decime"]


# ----- (C) innocuité établie / non établie, jamais nocivité -----

def test_innocuite_etablie_sous_le_seuil():
    innocuite = lecture_innocuite(0.010)
    assert innocuite["innocuite_etablie"] is True
    assert innocuite["formulation"] == "innocuité ÉTABLIE"


def test_innocuite_non_etablie_au_dessus_et_jamais_nocivite():
    """Fait de logique : le joueur ne voit jamais la référence."""
    innocuite = lecture_innocuite(0.090)
    assert innocuite["innocuite_etablie"] is False
    assert innocuite["formulation"] == "innocuité NON ÉTABLIE"
    assert "nocivité" not in innocuite["formulation"]
    assert "JAMAIS" in innocuite["jamais_nocivite"]
    assert "cohérence proche/lointain" in innocuite["jamais_nocivite"]


def test_la_lecture_porte_linnocuite_sur_le_pire_eps():
    """L'innocuité se lit sur le Δχ le PLUS grand du balayage."""
    lecture = _lire([_mesure(1e-5, 0.010), _mesure(1e-2, 0.090)])
    assert lecture["innocuite"]["delta_chi_max_decime"] == 0.090
    assert lecture["innocuite"]["innocuite_etablie"] is False


# ----- (D) la branche pré-écrite, REPORTÉE -----

def test_branche_applicable_si_instrument_valide_et_aucun_eps():
    branche = branche_decision_k(instrument_valide=True, eps_retenu=None)
    assert branche["applicable"] is True
    assert "la décision passe à k" in branche["enonce"]
    assert "RÉVEIL σ_ω" in branche["enonce"]


def test_branche_non_applicable_si_un_eps_passe_ou_instrument_muet():
    assert branche_decision_k(True, 1e-4)["applicable"] is False
    assert branche_decision_k(False, None)["applicable"] is False


def test_la_branche_est_reportee_jamais_declenchee():
    """Le driver ne réveille rien : la décision appartient à Romain."""
    branche = branche_decision_k(True, None)
    assert "REPORTÉE" in branche["statut"]
    assert "ne la déclenche pas" in branche["statut"]
    assert "appartient à Romain" in branche["statut"]


def test_la_lecture_porte_la_branche():
    lecture = _lire([_mesure(1e-5, 0.08), _mesure(1e-2, 0.09)])
    assert lecture["autre_remonte"] is True
    assert lecture["branche_preecrite_decision_k"]["applicable"] is True


# ----- 2. k reste FIGÉ (réveil σ_ω) -----

def test_k_est_fige_et_son_balayage_refuse():
    """Balayer k perceptuellement = décider le cadencement avec la
    question perceptuelle en main = condition de réveil σ_ω ATTEINTE."""
    assert CADENCE_FIGEE == 4
    exiger_cadence_figee(4)                     # ne lève pas
    for k in (1, 2, 8):
        with pytest.raises(RuntimeError, match="INTERDIT"):
            exiger_cadence_figee(k)


def test_la_lecture_porte_lexclusion_de_k():
    lecture = _lire([_mesure(1e-4, 0.01)])
    assert lecture["cadence_mesure"] == 4
    assert "σ_ω" in lecture["exclusion_k"]
    assert "INTERDIT" in lecture["exclusion_k"]


# ----- 3. le vivant est bien ce que le CPU sait -----

@gpu_requis
def test_le_vivant_part_des_references_pas_de_la_verite():
    """Le CPU part de ce qu'il a descendu. S'il partait de la vérité, la
    sonde mesurerait zéro et retiendrait n'importe quel EPS."""
    import cupy as cp
    pipeline = _pipeline(cp, 1e-4)
    reconstructeur = ReconstructeurNiveau0(pipeline)
    indice, position = _slot_observe(pipeline)
    vivant = reconstructeur.champ(indice, position)
    verite = champ_verite(pipeline, indice, position)
    assert not np.array_equal(vivant, verite)


@gpu_requis
def test_le_vivant_avance_par_les_coefficients_remontes():
    """L'incrément du schéma B4 : ce qui remonte est appliqué."""
    import cupy as cp
    pipeline = _pipeline(cp, 1e-4)
    reconstructeur = ReconstructeurNiveau0(pipeline)
    indice, position = _slot_observe(pipeline)
    avant = reconstructeur.champ(indice, position).copy()
    for _ in range(6):
        pipeline.frame()
        reconstructeur.appliquer(pipeline.coefficients_frame)
    assert not np.array_equal(reconstructeur.champ(indice, position), avant)


@gpu_requis
def test_le_seuil_gouverne_bien_le_volume_remonte():
    """Le MÉCANISME d'EPS, vérifié sans ambiguïté : plus le seuil est
    haut, moins il remonte de coefficients. (Que cela SE VOIE ou non au
    readout est une autre question — c'est ce que la sonde mesure, et le
    diagnostic de discrimination le dira.)"""
    import cupy as cp
    volumes = {}
    for eps in (1e-5, 1e-3, 1e-1):
        pipeline = _pipeline(cp, eps)
        total = 0
        for _ in range(8):
            pipeline.frame()
            total += sum(valeurs.size
                         for _, valeurs, _ in pipeline.coefficients_frame)
        volumes[eps] = total
    assert volumes[1e-5] > volumes[1e-3] > volumes[1e-1]


@gpu_requis
def test_les_coefficients_atterrissent_dans_le_bon_slot():
    """Les indices émis par le kernel sont relatifs à la TRANCHE du tour,
    pas au groupe : sans l'offset du segment, tout retomberait dans le
    slot 0 et le vivant des autres slots n'avancerait JAMAIS — une sonde
    silencieusement fausse. Le round-robin doit donc toucher des slots
    DIFFÉRENTS au fil des tours."""
    import cupy as cp
    pipeline = _pipeline(cp, 1e-5)
    groupe = pipeline.fenetres[0]
    elements_par_slot = int(groupe.shape[1] * 4 * groupe.shape[-1]
                            * groupe.shape[-2])
    slots_touches: set[int] = set()
    for _ in range(CADENCE_FIGEE):
        pipeline.frame()
        for indice_groupe, _, indices in pipeline.coefficients_frame:
            if indice_groupe == 0 and indices.size:
                slots_touches.update(
                    np.unique(indices // elements_par_slot).tolist())
    assert len(slots_touches) > 1


@gpu_requis
def test_la_peremption_l1_fait_partie_du_regime():
    """L1 k=4 étalée est INCLUSE : une fenêtre ne remonte qu'une frame sur
    k, donc certaines frames n'apportent aucun coefficient au slot
    observé — la péremption est mesurée, pas écartée."""
    import cupy as cp
    pipeline = _pipeline(cp, 1e-5)
    groupes_vus = []
    for _ in range(CADENCE_FIGEE):
        pipeline.frame()
        groupes_vus.append({c[0] for c in pipeline.coefficients_frame})
    assert pipeline.k == CADENCE_FIGEE
    assert any(groupes_vus)          # il remonte quelque chose
    assert len(pipeline.coefficients_frame) <= len(pipeline.fenetres)


# ----- 4. l'observable : primitives existantes, espace instrument -----

def test_delta_chi_readout_passe_par_les_primitives_du_depot():
    """albedo puis delta_chi(...)['max_carrier'] — le même chemin que
    l'arc A, pas une réécriture."""
    rng = np.random.default_rng(11)
    verite = np.abs(rng.normal(0.05, 0.01, (64, 64)))
    vivant = verite + 0.002 * rng.normal(size=(64, 64))
    from scripts.run_f1_sonde_eps import S_HALF_OP
    attendu = float(delta_chi(albedo(np.maximum(vivant, 0.0), S_HALF_OP),
                              albedo(verite, S_HALF_OP))["max_carrier"])
    assert delta_chi_readout(vivant, verite, 1) == pytest.approx(attendu)


def test_delta_chi_est_nul_quand_le_vivant_egale_la_verite():
    """Cas dégénéré : aucun écart de readout si les deux coïncident."""
    rng = np.random.default_rng(13)
    champ = np.abs(rng.normal(0.05, 0.01, (64, 64)))
    assert delta_chi_readout(champ, champ, 1) == pytest.approx(0.0)
    assert delta_chi_readout(champ, champ, 2) == pytest.approx(0.0)


def test_la_decimation_est_une_moyenne_de_blocs():
    """« Décimation exacte (moyennes Harten) » = `downsample` du dépôt."""
    from src.multiresolution import downsample
    champ = np.arange(64, dtype=np.float64).reshape(8, 8)
    attendu = downsample(champ, 2)
    assert attendu.shape == (4, 4)
    assert attendu[0, 0] == pytest.approx(champ[:2, :2].mean())


def test_lobservable_lit_le_champ_sediment():
    """Le readout albedo porte sur s, le 4e champ."""
    assert CHAMP_SEDIMENT == 3


# ----- portée -----

def test_la_sonde_ne_prononce_rien_et_nachete_pas_n2():
    lecture = _lire([_mesure(1e-4, 0.01)])
    texte = json.dumps(lecture, ensure_ascii=False)
    for interdit in ("MORT", "PASS", "VERDICT"):
        assert interdit not in texte
    assert "N2 (M-b) N'EST PAS achetée" in lecture["rappel_portee"]
    assert "Aucun verdict" in lecture["rappel_portee"]


# ----- plomberie GPU -----

@gpu_requis
def test_la_sonde_traverse_le_pipeline_complet():
    """Plomberie (taille minuscule, JAMAIS une mesure) : le pipeline V4
    tourne, le vivant se reconstruit, l'observable se calcule."""
    import cupy as cp
    pipeline = _pipeline(cp, 1e-4)
    reconstructeur = ReconstructeurNiveau0(pipeline)
    indice, position = _slot_observe(pipeline)
    serie = []
    for _ in range(6):
        pipeline.frame()
        reconstructeur.appliquer(pipeline.coefficients_frame)
        serie.append(delta_chi_readout(
            reconstructeur.champ(indice, position),
            champ_verite(pipeline, indice, position), 2))
    assert len(serie) == 6
    assert all(np.isfinite(valeur) and valeur >= 0.0 for valeur in serie)


@gpu_requis
def test_la_capture_est_desactivee_par_defaut():
    """Le coût de la capture ne doit jamais entrer dans une passe
    chronométrée : elle est opt-in."""
    import cupy as cp
    transferts = TransfertComptable(cp)
    pipeline = PipelineMaQuater(cp, GEO_PETITE, transferts, k=CADENCE_FIGEE)
    assert pipeline.capturer_coefficients is False
    for _ in range(5):
        pipeline.frame()
    assert pipeline.coefficients_frame == []
