"""Tests F1 — micro-sondes d'ouverture de séance s1 et s2 (§A17 G1,
pocCascade2phys b1f9cf9).

Le risque propre à s2 : elle RÉÉCRIT l'orchestration Python autour du
kernel (batch inter-niveaux, CFL laissée on-device). Réécrire un chemin
mesuré, c'est risquer d'OMETTRE du travail — un harnais complaisant qui
rendrait A_batché artificiellement bas et ferait conclure « c'était
l'orchestration » à tort. Deux verrous l'interdisent :
  - `reduction_cfl_on_device` rapatriée == `reduction_cfl` du jetable :
    aucune passe de la réduction n'a sauté ;
  - `pas_f_fusionne_async` produit un état BIT-IDENTIQUE à
    `pas_f_fusionne` : le calcul est le même, seule la barrière tombe.
Le reste couvre les lectures, la dépendance s1 -> s2, et la structure des
groupes."""
import json

import numpy as np
import pytest

from scripts.run_f1_s1_ancre_1sys import (
    ANCRE_2SYS_MESUREE_MS,
    ANCRE_MODELE_MS,
    CELLULES,
    N_SLOTS_C4_V2,
    TOLERANCE_RELATIVE,
    lecture_ancre,
)
from scripts.run_f1_s1_ancre_1sys import (
    lecture_mecanique as lecture_s1,
)
from scripts.run_f1_s2_batche import (
    ANCRE_2SYS_MS,
    LANCEMENTS_BRAS_A_PER_NIVEAU,
    LANCEMENTS_PAR_FRAME,
    N_BLOCS_1SYS,
    N_BLOCS_2SYS,
    TOL_CFL_RELATIVE,
    lire_ancre_s1,
    reduction_cfl_on_device,
)
from scripts.run_f1_s2_batche import (
    lecture_mecanique as lecture_s2,
)
from src.f1_gpu.backend import cupy_disponible, vers_cpu
from src.f1_gpu.substrat_jetable import (
    CFL_JETABLE,
    GRAVITE,
    _desing,
    etat_initial_jetable,
    reduction_cfl,
)

gpu_requis = pytest.mark.skipif(
    not cupy_disponible(), reason="cupy/device CUDA indisponible")


# ----- verrou : la CFL on-device n'omet AUCUNE passe -----

def test_le_travail_lourd_de_la_reduction_est_bit_identique():
    """LE verrou anti-omission : `smax` — sqrt, désingularisation, deux
    maximum, la réduction `.max()`, c'est-à-dire TOUT le travail — est
    bit-identique à celui du jetable. Le test le recalcule passe par
    passe et vérifie que diviser ce smax en f64 redonne EXACTEMENT
    `reduction_cfl`, et en f32 EXACTEMENT `reduction_cfl_on_device`.
    Autrement dit : rien n'a été omis, seule la précision de la division
    finale diffère (cf. docstring de `reduction_cfl_on_device`)."""
    q = etat_initial_jetable(2, 12, graine=41)
    h, hu, hv = q[:, :, 0], q[:, :, 1], q[:, :, 2]
    c = np.sqrt(GRAVITE * np.maximum(h, 0.0))
    u, v = _desing(h, hu, np), _desing(h, hv, np)
    smax = np.maximum(np.abs(u) + c, np.abs(v) + c).max()

    assert CFL_JETABLE / max(float(smax), 1e-12) == reduction_cfl(q, np)
    assert (float(CFL_JETABLE / np.maximum(smax, 1e-12))
            == float(reduction_cfl_on_device(q, np)))


def test_reduction_on_device_concorde_a_la_tolerance_nommee_numpy():
    """Le dt lui-même concorde au dernier bit f32 près — il n'est PAS
    consommé (dt figé), l'état final est bit-identique par ailleurs."""
    q = etat_initial_jetable(2, 12, graine=41)
    obtenu, attendu = float(reduction_cfl_on_device(q, np)), reduction_cfl(q, np)
    assert abs(obtenu - attendu) / attendu <= TOL_CFL_RELATIVE


@gpu_requis
def test_reduction_on_device_concorde_a_la_tolerance_nommee_gpu():
    """Même exigence sur le device, où la barrière supprimée existe
    réellement."""
    import cupy as cp
    q = cp.asarray(etat_initial_jetable(3, 16, graine=43))
    obtenu, attendu = float(reduction_cfl_on_device(q, cp)), reduction_cfl(q, cp)
    assert abs(obtenu - attendu) / attendu <= TOL_CFL_RELATIVE


@gpu_requis
def test_reduction_on_device_ne_rapatrie_rien():
    """Le résultat reste un TABLEAU device : c'est ce qui supprime la
    barrière. S'il redevenait un float Python, la sync serait de retour
    et s2 mesurerait exactement l'artefact qu'elle veut supprimer."""
    import cupy as cp
    q = cp.asarray(etat_initial_jetable(2, 12, graine=47))
    resultat = reduction_cfl_on_device(q, cp)
    assert isinstance(resultat, cp.ndarray)
    assert resultat.shape == ()
    assert not isinstance(resultat, float)


def test_reduction_on_device_reste_auditable():
    """« Non synchronisante » ne veut pas dire « perdue » : la valeur est
    rapatriable à la demande, ce que le driver fait UNE fois hors série."""
    q = etat_initial_jetable(1, 10, graine=53)
    valeur = float(reduction_cfl_on_device(q, np))
    assert np.isfinite(valeur) and valeur > 0.0


# ----- verrou : le pas batché est BIT-IDENTIQUE au pas de référence -----

@gpu_requis
@pytest.mark.parametrize("n_systemes", [1, 2])
def test_pas_async_bit_identique_au_pas_fusionne(n_systemes):
    """Le calcul est le MÊME : mêmes lancements, mêmes arguments, même dt
    figé. Si un étage ou une passe avait sauté, l'état divergerait."""
    import cupy as cp

    from scripts.run_f1_s2_batche import pas_f_fusionne_async
    from src.f1_gpu.substrat_fusionne import pas_f_fusionne

    q_cpu = np.ascontiguousarray(
        etat_initial_jetable(3, 12, graine=59)[:, :n_systemes])
    reference = cp.asarray(q_cpu)
    essai = cp.asarray(q_cpu)
    pas_f_fusionne(reference, cp, sortie=reference,
                   tampon_etage=cp.empty_like(reference))
    pas_f_fusionne_async(essai, cp, essai, cp.empty_like(essai))
    assert np.array_equal(vers_cpu(essai), vers_cpu(reference))


@gpu_requis
def test_pas_async_bit_identique_sur_plusieurs_pas():
    """Composition : l'identité tient sur 3 pas (un écart d'1 ULP au pas
    1 se verrait amplifié)."""
    import cupy as cp

    from scripts.run_f1_s2_batche import pas_f_fusionne_async
    from src.f1_gpu.substrat_fusionne import pas_f_fusionne

    q_cpu = etat_initial_jetable(2, 10, graine=61)
    reference, essai = cp.asarray(q_cpu), cp.asarray(q_cpu)
    tampon_r, tampon_e = cp.empty_like(reference), cp.empty_like(essai)
    for _ in range(3):
        pas_f_fusionne(reference, cp, sortie=reference, tampon_etage=tampon_r)
        pas_f_fusionne_async(essai, cp, essai, tampon_e)
    assert np.array_equal(vers_cpu(essai), vers_cpu(reference))


@gpu_requis
def test_pas_async_garde_les_memes_shapes_que_loriginal():
    """L'élargissement S ∈ {1, 2} de S1 est reconduit, et rien d'autre
    n'est accepté."""
    import cupy as cp

    from scripts.run_f1_s2_batche import pas_f_fusionne_async
    for n_systemes in (0, 3):
        q = cp.zeros((1, n_systemes, 4, 8, 8), dtype=cp.float32)
        with pytest.raises(ValueError, match="E4a"):
            pas_f_fusionne_async(q, cp, q, cp.empty_like(q))
    q64 = cp.zeros((1, 2, 4, 8, 8), dtype=cp.float64)
    with pytest.raises(ValueError, match="float32"):
        pas_f_fusionne_async(q64, cp, q64, cp.empty_like(q64))


# ----- s1 : les deux cellules et la lecture des ancres -----

def test_s1_mesure_bien_les_deux_cellules_gravees():
    """(a) B=1 isolé et (b) B=7 batché — les 7 slots c=4 de V2."""
    assert CELLULES == (("a_isole", 1), ("b_batche", N_SLOTS_C4_V2))
    assert N_SLOTS_C4_V2 == 7


def test_constantes_gravees_des_sondes():
    """Aucun chiffre ne bouge : l'étiquette structurelle contestée
    (0.843), l'ancre 2-systèmes réellement mesurée en M-a′ (1.685, dont
    0.843 était censée être la moitié), et la tolérance ±10 % de §A17."""
    assert ANCRE_MODELE_MS == 0.843
    assert ANCRE_2SYS_MESUREE_MS == 1.685
    assert ANCRE_MODELE_MS * 2 == pytest.approx(ANCRE_2SYS_MESUREE_MS,
                                                abs=0.001)
    assert TOLERANCE_RELATIVE == 0.10
    assert ANCRE_2SYS_MS == ANCRE_2SYS_MESUREE_MS   # s1 et s2 d'accord


def test_s1_ancre_par_slot_est_la_grandeur_du_modele():
    """Le modèle raisonne PAR SLOT : la lecture divise donc la médiane
    par le nombre de blocs. Vérifié sur l'arithmétique de lecture."""
    lecture = lecture_ancre(0.843)
    assert lecture["dans_tolerance"] is True
    assert lecture["ecart_relatif"] == pytest.approx(0.0)
    assert lecture["rapport_au_2sys_mesure"] == pytest.approx(
        0.843 / ANCRE_2SYS_MESUREE_MS)


def test_s1_linearite_fausse_hors_tolerance():
    """Hors ±10 % : la linéarité en systèmes est FAUSSE, le modèle se
    re-calibre sur mesures. Le cas suspecté (occupancy) est un bloc
    1-système qui coûte NETTEMENT plus que la moitié d'un 2-systèmes."""
    lecture = lecture_ancre(1.400)
    assert lecture["dans_tolerance"] is False
    assert lecture["ecart_relatif"] > TOLERANCE_RELATIVE
    assert lecture["rapport_au_2sys_mesure"] > 0.5


def test_s1_designe_lancre_batchee_pour_s2():
    """s2 batche 7 blocs en UN lancement : c'est l'ancre (b) qui la
    concerne, pas l'unitaire (a)."""
    cellules = {"a_isole": {"ancre_par_slot_ms": 1.20},
                "b_batche": {"ancre_par_slot_ms": 0.90}}
    lecture = lecture_s1(cellules)
    assert lecture["ancre_retenue_pour_s2"]["valeur_ms"] == 0.90
    assert "b" in lecture["ancre_retenue_pour_s2"]["origine"]
    assert lecture["toutes_dans_tolerance"] is False   # 1.20 hors ±10 %


def test_s1_ne_prononce_rien():
    cellules = {"a_isole": {"ancre_par_slot_ms": 0.85},
                "b_batche": {"ancre_par_slot_ms": 0.84}}
    texte = json.dumps(lecture_s1(cellules), ensure_ascii=False)
    for interdit in ("MORT", "PASS", "VERDICT"):
        assert interdit not in texte


# ----- s2 : dépendance à s1, structure des groupes, lecture -----

def test_s2_refuse_de_demarrer_sans_le_json_de_s1(tmp_path):
    """L'ordre s1 -> s2 est une DÉPENDANCE : sans l'ancre mesurée, s2
    n'a pas de prédiction re-calibrée à confronter. Fail-loud, et le
    message dit quoi lancer."""
    with pytest.raises(RuntimeError, match="run_f1_s1_ancre_1sys"):
        lire_ancre_s1(tmp_path / "absent.json")


def test_s2_refuse_un_json_s1_incomplet(tmp_path):
    chemin = tmp_path / "s1.json"
    chemin.write_text(json.dumps({"lecture_mecanique": {}}),
                      encoding="utf-8")
    with pytest.raises(RuntimeError, match="illisible ou incomplet"):
        lire_ancre_s1(chemin)


def test_s2_refuse_une_ancre_absurde(tmp_path):
    chemin = tmp_path / "s1.json"
    chemin.write_text(json.dumps({"lecture_mecanique": {
        "ancre_retenue_pour_s2": {"valeur_ms": 0.0, "origine": "x"}}}),
        encoding="utf-8")
    with pytest.raises(RuntimeError, match="non exploitable"):
        lire_ancre_s1(chemin)


def test_s2_lit_lancre_mesuree_sans_la_recopier(tmp_path):
    """La valeur vient du JSON, pas d'une constante du driver — un
    chiffre recopié serait non traçable."""
    chemin = tmp_path / "s1.json"
    chemin.write_text(json.dumps({"lecture_mecanique": {
        "ancre_retenue_pour_s2": {"valeur_ms": 1.234,
                                  "origine": "cellule (b)"}}}),
        encoding="utf-8")
    ancre = lire_ancre_s1(chemin)
    assert ancre["ancre_1sys_ms"] == 1.234
    assert str(chemin) == ancre["source"]


def test_s2_orchestration_est_celle_de_v2():
    """5 blocs 2-systèmes + 7 blocs 1-système = 17 blocs — exactement V2,
    en 4 lancements au lieu des 18 du bras A."""
    assert N_BLOCS_2SYS * 2 + N_BLOCS_1SYS * 1 == 17
    assert LANCEMENTS_PAR_FRAME == 4
    assert LANCEMENTS_BRAS_A_PER_NIVEAU == 18


def test_s2_prediction_recalibree_suit_la_formule_gravee():
    """5 x 1.685 + 7 x ancre_s1 — l'ancre 2-sys reste celle de M-a′."""
    ancre = {"ancre_1sys_ms": 0.900, "origine": "cellule (b)",
             "source": "s1.json"}
    attendu = 5 * ANCRE_2SYS_MS + 7 * 0.900
    lecture = lecture_s2(attendu, ancre)
    assert lecture["prediction_recalibree"]["valeur_ms"] == pytest.approx(
        attendu)
    assert lecture["ecart_relatif"] == pytest.approx(0.0)
    assert lecture["dans_tolerance"] is True


def test_s2_hors_tolerance_est_un_autre():
    """Hors ±10 % : coût architectural résiduel non compris — AUTRE
    remonté, pas de design à l'aveugle."""
    ancre = {"ancre_1sys_ms": 0.843, "origine": "b", "source": "s1.json"}
    prediction = 5 * ANCRE_2SYS_MS + 7 * 0.843
    lecture = lecture_s2(prediction * 1.5, ancre)
    assert lecture["dans_tolerance"] is False
    assert "AUTRE" in lecture["branche_preecrite_si_hors_tolerance"]


def test_s2_budget_machinerie_est_reporte_non_interprete():
    """budget = 16.7 - A_batché, reporté tel quel — y compris NÉGATIF,
    sans que le driver commente."""
    ancre = {"ancre_1sys_ms": 0.843, "origine": "b", "source": "s1.json"}
    budget = lecture_s2(14.0, ancre)["budget_machinerie"]
    assert budget["valeur_ms"] == pytest.approx(2.7)
    assert "jamais interprété" in budget["note"]
    budget_negatif = lecture_s2(20.0, ancre)["budget_machinerie"]
    assert budget_negatif["valeur_ms"] == pytest.approx(-3.3)


def test_s2_nomme_ce_quelle_ne_departage_pas():
    """Trois artefacts tombent ensemble (lancements, syncs CFL,
    orchestration CPU per-niveau) : la lecture doit interdire la
    sur-lecture « c'était les lancements »."""
    ancre = {"ancre_1sys_ms": 0.843, "origine": "b", "source": "s1.json"}
    limite = lecture_s2(14.0, ancre)["limite_de_lecture"]
    assert "CUMULÉ" in limite
    assert "jamais leurs parts" in limite
    assert "non pré-enregistrés" in limite


def test_s2_ne_prononce_rien():
    ancre = {"ancre_1sys_ms": 0.843, "origine": "b", "source": "s1.json"}
    texte = json.dumps(lecture_s2(14.0, ancre), ensure_ascii=False)
    for interdit in ("MORT", "PASS", "VERDICT", "escalade décidée"):
        assert interdit not in texte
    assert "ANCRE le design" in texte


# ----- plomberie GPU : le bras A batché tourne vraiment -----

@gpu_requis
def test_groupes_batches_tournent_et_restent_auditables():
    """Plomberie (taille minuscule, JAMAIS une mesure) : les deux groupes
    traversent le kernel, l'état reste fini, les références restent
    allouées (résidence du bras A) et l'audit CFL est exploitable."""
    import cupy as cp

    from scripts.run_f1_s2_batche import GroupesBatches
    groupes = GroupesBatches(cp, n_blocs_2sys=2, n_blocs_1sys=3, n_fov=8)
    assert groupes.n_blocs() == 2 * 2 + 3          # 7 blocs
    assert len(groupes.fenetres) == 2              # 2 appels => 4 lancements
    for _ in range(3):
        groupes.frame()
    audit = groupes.audit_cfl()
    assert audit["toutes_finies_positives"] is True
    assert len(audit["dt_cfl_par_groupe"]) == 2
    for bloc in groupes.fenetres:
        assert bool(cp.isfinite(bloc).all())
    # références préallouées : structure conservée malgré la remontée OFF
    assert len(groupes.references) == 2
    assert groupes.octets_etat() == 2 * sum(
        int(b.nbytes) for b in groupes.fenetres)


@gpu_requis
def test_groupes_batches_ne_synchronisent_pas_par_frame():
    """Le dt CFL conservé est un tableau device : la frame n'a rapatrié
    aucun scalaire (c'est tout l'objet de s2)."""
    import cupy as cp

    from scripts.run_f1_s2_batche import GroupesBatches
    groupes = GroupesBatches(cp, n_blocs_2sys=1, n_blocs_1sys=1, n_fov=8)
    groupes.frame()
    for dt in groupes.dt_cfl:
        assert isinstance(dt, cp.ndarray)
        assert dt.shape == ()
