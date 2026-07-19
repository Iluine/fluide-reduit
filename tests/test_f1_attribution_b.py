"""Tests F1 — attribution (b) armée (§A19-lecture-diagnostic,
pocCascade2phys 4062747).

Ce que ces tests protègent :
  1. la portée — deux protocoles et deux seulement, k figé, eps = 0
     traité comme un APPAREIL et jamais comme un régime retenable ;
  2. le témoin EST une remontée pleine non incrémentale : à eps = 0 le
     kernel L3 INTOUCHÉ écrase la référence par l'état ;
  3. le plancher de bruit est MESURÉ (réplicat), jamais supposé, et
     l'effondrement exige DEUX conditions — facteur ET écart au-dessus du
     bruit ;
  4. le VERROU amont : tant que l'invariant B4 est rompu, rien n'est
     prononcé et le verdict est refusé fail-loud à la consommation."""
import json

import numpy as np
import pytest

from scripts.run_f1_attribution_b import (
    CADENCE_FIGEE_ATTRIBUTION,
    EPS_COURANT,
    EPS_TEMOIN,
    FACTEUR_EFFONDREMENT_ATTRIBUTION,
    LECTURE_EFFONDREMENT,
    LECTURE_PERSISTANCE,
    OBSERVABLE_LIVRE,
    OBSERVABLE_RECONSTRUIT,
    PROTOCOLES,
    VUE_DISCRIMINANTE,
    VUES,
    champ_livre_de_comptes,
    exiger_attribution_prononcable,
    exiger_rien_regle,
    lecture_attribution,
    plancher_bruit,
    verrou_invariant,
)
from scripts.run_f1_diagnostic_instrument import ReconstructeurAvecCouverture
from scripts.run_f1_ma_quater import PipelineMaQuater, slots_v4
from scripts.run_f1_sonde_eps import (
    CHAMP_SEDIMENT,
    _slot_observe,
    champ_verite,
    delta_chi_readout,
)
from src.f1_gpu.backend import cupy_disponible, vers_cpu
from src.f1_gpu.pyramide import GeometriePyramide
from src.f1_gpu.transferts import TransfertComptable

gpu_requis = pytest.mark.skipif(
    not cupy_disponible(), reason="cupy/device CUDA indisponible")

GEO_PETITE = GeometriePyramide(n_fov=64, n_niv=10, n0=8,
                               slots=slots_v4(10), emboitee=True)


def _bras(par_vue: dict, invariant_max: float = 0.0) -> dict:
    """Bras synthétique : mêmes chiffres sur les deux observables."""
    vues = {observable: {
        vue: {"max": valeur, "median": valeur * 0.5,
              **({"fraction_couverte_mediane": 0.3}
                 if vue != "complet" else {})}
        for vue, valeur in par_vue.items()}
        for observable in (OBSERVABLE_RECONSTRUIT, OBSERVABLE_LIVRE)}
    return {"vues": vues,
            "invariant_b4": {"max": invariant_max, "median": invariant_max,
                             "ecart_brut_max_diagnostic": invariant_max}}


# ----- 1. la portée : rien n'est réglé -----

def test_deux_protocoles_et_deux_seulement():
    """eps = 1e-4 pour le régime, eps = 0 pour le témoin. Toute autre
    valeur serait un balayage déguisé en falsificateur."""
    assert EPS_COURANT == 1e-4
    assert EPS_TEMOIN == 0.0
    exiger_rien_regle(EPS_COURANT, CADENCE_FIGEE_ATTRIBUTION)
    exiger_rien_regle(EPS_TEMOIN, CADENCE_FIGEE_ATTRIBUTION)
    for eps in (1e-5, 3e-4, 1e-3, 1e-2):
        with pytest.raises(RuntimeError, match="hors des deux protocoles"):
            exiger_rien_regle(eps, CADENCE_FIGEE_ATTRIBUTION)


def test_la_cadence_reste_figee():
    assert CADENCE_FIGEE_ATTRIBUTION == 4
    for k in (1, 2, 3, 8):
        with pytest.raises(RuntimeError, match="σ_ω"):
            exiger_rien_regle(EPS_COURANT, k)


def test_eps_zero_est_un_appareil_jamais_un_regime():
    """Le témoin transfère l'état entier des slots remontés : il ne peut
    pas devenir une configuration, et le protocole le dit."""
    temoin = PROTOCOLES["temoin"]
    assert "APPAREIL DE MESURE" in temoin["pas_un_reglage"]
    assert "jamais une configuration" in temoin["pas_un_reglage"]


def test_aucune_lecture_ne_propose_un_reglage():
    lecture = lecture_attribution(
        {"courant": _bras({"complet": 0.08, "cycle": 0.04, "cumule": 0.09}),
         "temoin": _bras({"complet": 0.01, "cycle": 0.005, "cumule": 0.01})},
        verrou_invariant({"courant": _bras({}), "temoin": _bras({})}, 0.0),
        0.0)
    texte = json.dumps(lecture, ensure_ascii=False)
    for interdit in ("MORT", "VERDICT", "EPS retenu"):
        assert interdit not in texte
    assert "rien n'est réglé" in lecture["rappel_portee"]


# ----- 2. le témoin est bien une remontée PLEINE non incrémentale -----

def test_le_temoin_est_documente_comme_ecrasement_complet():
    """À eps = 0 : |d| >= 0 toujours vrai, donc toute cellule émet, et
    reference += (etat − reference) = etat. L'incrément DEVIENT
    l'écrasement complet — kernel INTOUCHÉ."""
    pourquoi = PROTOCOLES["temoin"]["pourquoi_eps_zero"]
    assert "toute cellule émet" in pourquoi
    assert "reference = etat" in pourquoi
    assert "INTOUCHÉ" in pourquoi


@gpu_requis
def test_a_eps_zero_la_reference_devient_letat():
    """La propriété est VÉRIFIÉE sur le device, pas seulement affirmée :
    sur les slots remontés, le livre de comptes égale l'état."""
    import cupy as cp
    pipeline = PipelineMaQuater(cp, GEO_PETITE, TransfertComptable(cp),
                                k=CADENCE_FIGEE_ATTRIBUTION, eps=EPS_TEMOIN,
                                capturer_coefficients=True)
    indice, position = _slot_observe(pipeline)
    ecarts = []
    for _ in range(2 * CADENCE_FIGEE_ATTRIBUTION):
        pipeline.frame()
        ecarts.append(float(np.max(np.abs(
            champ_livre_de_comptes(pipeline, indice, position)
            - champ_verite(pipeline, indice, position)))))
    # Au tour du slot, la référence a été écrasée par l'état : l'écart
    # retombe à zéro EXACT au moins une fois par cycle.
    assert min(ecarts) == 0.0


@gpu_requis
def test_a_eps_courant_la_reference_ne_rejoint_pas_letat():
    """Contre-épreuve : avec le seuil, la référence ne colle JAMAIS
    exactement — c'est précisément le schéma que le témoin retire."""
    import cupy as cp
    pipeline = PipelineMaQuater(cp, GEO_PETITE, TransfertComptable(cp),
                                k=CADENCE_FIGEE_ATTRIBUTION, eps=EPS_COURANT,
                                capturer_coefficients=True)
    indice, position = _slot_observe(pipeline)
    ecarts = []
    for _ in range(2 * CADENCE_FIGEE_ATTRIBUTION):
        pipeline.frame()
        ecarts.append(float(np.max(np.abs(
            champ_livre_de_comptes(pipeline, indice, position)
            - champ_verite(pipeline, indice, position)))))
    assert min(ecarts) > 0.0


# ----- 3. plancher de bruit MESURÉ, et deux conditions pour effondrer -----

def test_le_plancher_de_bruit_exige_un_replicat():
    """Sans deux passes identiques, « au-dessus du bruit » n'a pas de
    sens — c'est la faute de seuil corrigée au §A19-lecture-diagnostic."""
    with pytest.raises(RuntimeError, match="réplicat est DÛ"):
        plancher_bruit([_bras({vue: 0.08 for vue in VUES})])


def test_le_plancher_de_bruit_est_lecart_entre_replicats():
    bruit = plancher_bruit([
        _bras({"complet": 0.0800, "cycle": 0.0400, "cumule": 0.0900}),
        _bras({"complet": 0.0803, "cycle": 0.0400, "cumule": 0.0902})])
    assert bruit["plancher_delta_chi"] == pytest.approx(0.0003, abs=1e-9)
    assert bruit["deterministe"] is False
    assert "D-1(a)" in bruit["justification"]


def test_un_plancher_nul_est_dit_deterministe_pas_infiniment_precis():
    identique = {"complet": 0.08, "cycle": 0.04, "cumule": 0.09}
    bruit = plancher_bruit([_bras(identique), _bras(identique)])
    assert bruit["plancher_delta_chi"] == 0.0
    assert bruit["deterministe"] is True
    assert "DÉTERMINISTE" in bruit["note_si_nul"]
    assert "infiniment précise" in bruit["note_si_nul"]


def test_le_facteur_deffondrement_est_fige_avant_le_run():
    assert FACTEUR_EFFONDREMENT_ATTRIBUTION == 0.5


def test_un_rapport_spectaculaire_dans_le_bruit_ne_dit_rien():
    """DEUX conditions, pas une : le témoin a beau valoir le dixième du
    courant, si l'écart absolu est sous le bruit il n'y a rien à lire."""
    bras = {
        "courant": _bras({vue: 0.0010 for vue in VUES}),
        "temoin": _bras({vue: 0.0001 for vue in VUES}),
    }
    lecture = lecture_attribution(
        bras, verrou_invariant(bras, 0.01), plancher=0.01)
    comparaison = lecture["comparaisons_par_vue"][VUE_DISCRIMINANTE]
    assert comparaison["sous_le_facteur"] is True
    assert comparaison["au_dessus_du_bruit"] is False
    assert comparaison["seffondre"] is False


def test_effondrement_designe_la_derive_de_la_reference():
    bras = {
        "courant": _bras({vue: 0.0800 for vue in VUES}),
        "temoin": _bras({vue: 0.0050 for vue in VUES}),
    }
    lecture = lecture_attribution(
        bras, verrou_invariant(bras, 1.0), plancher=0.0001)
    assert lecture["prononcable"] is True
    assert lecture["label"] == "RÉFÉRENCE INCRÉMENTALE QUI DÉRIVE"
    assert lecture["lecture"] == LECTURE_EFFONDREMENT
    assert "fovéa-z" in lecture["lecture"]


def test_persistance_renvoie_a_lexigence_de_fidelite():
    bras = {
        "courant": _bras({vue: 0.0800 for vue in VUES}),
        "temoin": _bras({vue: 0.0790 for vue in VUES}),
    }
    lecture = lecture_attribution(
        bras, verrou_invariant(bras, 1.0), plancher=0.0001)
    assert lecture["prononcable"] is True
    assert lecture["label"] == "ERREUR INHÉRENTE AU GROSSIER"
    assert lecture["lecture"] == LECTURE_PERSISTANCE
    assert "EXIGENCE DE FIDÉLITÉ" in lecture["lecture"]


def test_la_vue_discriminante_est_le_cumule():
    """Gravé : c'est la comparaison des CUMULÉS qui discrimine. Les trois
    vues sont reportées, mais une seule décide."""
    assert VUE_DISCRIMINANTE == "cumule"
    assert VUES == ("complet", "cycle", "cumule")
    bras = {
        "courant": _bras({"complet": 0.08, "cycle": 0.08, "cumule": 0.08}),
        # seul le cumulé s'effondre : la lecture doit suivre le cumulé.
        "temoin": _bras({"complet": 0.079, "cycle": 0.079, "cumule": 0.005}),
    }
    lecture = lecture_attribution(
        bras, verrou_invariant(bras, 1.0), plancher=0.0001)
    assert lecture["comparaisons_par_vue"]["complet"]["seffondre"] is False
    assert lecture["comparaisons_par_vue"]["cumule"]["seffondre"] is True
    assert lecture["label"] == "RÉFÉRENCE INCRÉMENTALE QUI DÉRIVE"


def test_lecart_de_domaine_est_reporte_jamais_suppose_nul():
    """Seule la vue COMPLÈTE a un domaine identique entre les bras ; les
    fractions couvertes des deux bras sont reportées côte à côte."""
    bras = {
        "courant": _bras({vue: 0.08 for vue in VUES}),
        "temoin": _bras({vue: 0.01 for vue in VUES}),
    }
    lecture = lecture_attribution(
        bras, verrou_invariant(bras, 1.0), plancher=0.0)
    assert set(lecture["fractions_couvertes"]) == {"courant", "temoin"}
    assert "domaine IDENTIQUE" in lecture["note_domaine"]
    assert "jamais supposé nul" in lecture["note_domaine"]


# ----- 4. le VERROU amont : ne rien prononcer sur une reconstruction fausse -----

def test_invariant_tenu_laisse_prononcer():
    bras = {"courant": _bras({vue: 0.08 for vue in VUES}, invariant_max=0.0),
            "temoin": _bras({vue: 0.01 for vue in VUES}, invariant_max=0.0)}
    verrou = verrou_invariant(bras, plancher=0.0)
    assert verrou["respecte"] is True
    exiger_attribution_prononcable(verrou)


def test_invariant_rompu_refuse_toute_attribution():
    """Si le vivant n'est pas la connaissance du CPU, les deux bras
    portent le MÊME défaut de reconstruction et sont indistinguables : la
    soustraction ne mesurerait plus le schéma."""
    bras = {"courant": _bras({vue: 0.08 for vue in VUES}, invariant_max=0.08),
            "temoin": _bras({vue: 0.01 for vue in VUES}, invariant_max=0.08)}
    verrou = verrou_invariant(bras, plancher=0.0001)
    assert verrou["respecte"] is False
    assert verrou["tenu_par_bras"] == {"courant": False, "temoin": False}
    with pytest.raises(RuntimeError, match="NON PRONONÇABLE"):
        exiger_attribution_prononcable(verrou)


def test_lecture_refusee_ne_porte_aucune_branche_preecrite():
    """Le verrou ne se contente pas de lever un drapeau : la lecture
    pré-écrite n'est PAS appliquée, et les deux bras sont tout de même
    reportés pour que la contamination soit lisible."""
    bras = {"courant": _bras({vue: 0.08 for vue in VUES}, invariant_max=0.08),
            "temoin": _bras({vue: 0.005 for vue in VUES}, invariant_max=0.08)}
    verrou = verrou_invariant(bras, plancher=0.0001)
    lecture = lecture_attribution(bras, verrou, plancher=0.0001)
    assert lecture["prononcable"] is False
    assert lecture["label"] == "ATTRIBUTION NON PRONONÇABLE"
    assert lecture["lecture"] not in (LECTURE_EFFONDREMENT,
                                      LECTURE_PERSISTANCE)
    assert "indistinguables" in lecture["lecture"]
    # les chiffres restent reportés : le refus ne cache pas la mesure
    assert lecture["comparaisons_par_vue"][VUE_DISCRIMINANTE][
        "courant_max"] == 0.08


@gpu_requis
def test_le_livre_de_comptes_est_bien_la_reference_device():
    """Plomberie : `champ_livre_de_comptes` lit le champ sédiment de la
    `reference` du device — l'objet que le schéma B4 dit être la
    connaissance du CPU, et rien d'autre."""
    import cupy as cp
    pipeline = PipelineMaQuater(cp, GEO_PETITE, TransfertComptable(cp),
                                k=CADENCE_FIGEE_ATTRIBUTION, eps=EPS_COURANT,
                                capturer_coefficients=True)
    indice, position = _slot_observe(pipeline)
    pipeline.frame()
    livre = champ_livre_de_comptes(pipeline, indice, position)
    attendu = vers_cpu(
        pipeline.references[indice][position, 0, CHAMP_SEDIMENT])
    assert livre.shape == (GEO_PETITE.n_fov, GEO_PETITE.n_fov)
    np.testing.assert_array_equal(livre, attendu.astype(np.float64))


@gpu_requis
def test_linvariant_b4_separe_reconstruction_et_livre_de_comptes():
    """Le verrou a un objet RÉEL à mesurer : le vivant reconstruit et le
    livre de comptes sont deux tableaux distincts, que seul le respect du
    schéma peut rendre égaux."""
    import cupy as cp
    pipeline = PipelineMaQuater(cp, GEO_PETITE, TransfertComptable(cp),
                                k=CADENCE_FIGEE_ATTRIBUTION, eps=EPS_COURANT,
                                capturer_coefficients=True)
    reconstructeur = ReconstructeurAvecCouverture(pipeline)
    indice, position = _slot_observe(pipeline)
    # À l'initialisation le CPU part des références : ils COÏNCIDENT.
    assert delta_chi_readout(reconstructeur.champ(indice, position),
                             champ_livre_de_comptes(pipeline, indice,
                                                    position), 1) == 0.0
    for _ in range(4):
        pipeline.frame()
        reconstructeur.appliquer(pipeline.coefficients_frame)
    ecart = delta_chi_readout(
        reconstructeur.champ(indice, position),
        champ_livre_de_comptes(pipeline, indice, position), 1)
    assert np.isfinite(ecart)
