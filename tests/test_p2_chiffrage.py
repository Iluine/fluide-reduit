"""P2 / pré-requis P3 — tests du runner de chiffrage et des deux gardes de
session (`scripts/run_p2_chiffrage.py`, `scripts/run_arcC_session.py`).

Sources qui font foi : `claude/prereg-p2-chiffrage-rendu.md` (bandes, branches),
`claude/prereg-p3-transport-pin.md` (pré-requis), §A38, §A37.

Ce qui est testé ici est la LOGIQUE : classement de bande, prononcé mécanique
des branches, gardes de comparabilité et liaison sidecar↔log. Les CHIFFRES,
eux, sont verdict-grade et appartiennent à Romain (iluin-tworings3, terminal
natif, 300 appels) — aucun test ne mesure quoi que ce soit."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from scripts.run_p2_chiffrage import (BANDE_CELLULE1_MS, BANDE_CELLULE2_MS, MARGE_V4_MS,
                                      N_APPELS_GRAVE, N_CHAUFFE_GRAVE, SEUIL_TIMINGS_P3_MS,
                                      VERDICT_AUTRE, VERDICT_DANS_BANDE, champ_equivalence,
                                      chronometre, classe_bande, empreinte_r1,
                                      etages_srgb_quantif_cupy, lecture_cellule1,
                                      lecture_cellule2)
from scripts.run_arcC_session import (GEOMETRIE_DU_PIN, assert_geometrie_comparable,
                                      sha256_fichier)
from src.arcC_rendu import quantifie_uint8, srgb_encode
from src.f1_gpu.backend import cupy_disponible

gpu_requis = pytest.mark.skipif(
    not cupy_disponible(), reason="cupy/device CUDA indisponible")


# --- Les constantes SONT celles du prereg ------------------------------------


def test_constantes_recopiees_du_prereg():
    """Les bandes et le protocole se COPIENT, ne s'ajustent pas (garde-fou de
    l'ordre de mission). Ce test est le verrou de cette copie : si quelqu'un
    élargit une bande pour faire passer un chiffre, il casse ici d'abord."""
    assert BANDE_CELLULE1_MS == (0.05, 5.0)
    assert BANDE_CELLULE2_MS == (0.02, 0.5)
    assert SEUIL_TIMINGS_P3_MS == 5.0
    assert MARGE_V4_MS == 2.199
    assert (N_APPELS_GRAVE, N_CHAUFFE_GRAVE) == (300, 30)


# --- Classement de bande -----------------------------------------------------


@pytest.mark.parametrize("mediane,attendu", [
    (0.05, VERDICT_DANS_BANDE),      # borne basse INCLUSE
    (1.0, VERDICT_DANS_BANDE),
    (5.0, VERDICT_DANS_BANDE),       # borne haute INCLUSE
    (0.049, VERDICT_AUTRE),
    (5.001, VERDICT_AUTRE),
])
def test_classe_bande_cellule1(mediane, attendu):
    """Bornes INCLUSES, des deux côtés. Hors bande = `AUTRE`, remonté tel
    quel — jamais rattrapé par un arrondi de confort."""
    assert classe_bande(mediane, BANDE_CELLULE1_MS) == attendu


@pytest.mark.parametrize("mediane,attendu", [
    (0.02, VERDICT_DANS_BANDE), (0.1, VERDICT_DANS_BANDE), (0.5, VERDICT_DANS_BANDE),
    (0.019, VERDICT_AUTRE), (0.51, VERDICT_AUTRE),
])
def test_classe_bande_cellule2(mediane, attendu):
    assert classe_bande(mediane, BANDE_CELLULE2_MS) == attendu


# --- Prononcé mécanique des lectures pré-écrites -----------------------------


@pytest.mark.parametrize("mediane,branche", [(0.001, "1"), (5.0, "1"), (5.0001, "2"),
                                             (50.0, "2")])
def test_lecture_cellule1_est_totale(mediane, branche):
    """La lecture de la cellule 1 est TOTALE : `≤ 5 ms` ou `> 5 ms`, aucune
    zone muette, aucune décision à l'œil. Consommateur nommé (D17) :
    l'intégrité des timings de présentation du régime sévère (§C0)."""
    assert lecture_cellule1(mediane)["branche"] == branche


@pytest.mark.parametrize("mediane,branche", [
    (0.001, "1"),        # sous la bande : toujours ≪ marge
    (0.5, "1"),          # haut de la bande gravée = 0.227 x marge
    (0.5001, None),      # zone SANS nombre gravé : rien n'est prononcé
    (2.198, None),
    (2.199, "2"),        # la marge V4 elle-même
    (10.0, "2"),
])
def test_lecture_cellule2_ne_prononce_pas_dans_la_zone_non_gravee(mediane, branche):
    """« ≪ » n'est pas un opérateur : le prereg dit « médiane ≪ marge V4 » pour
    la branche 1 et « comparable ou supérieure » pour la branche 2, sans aucun
    nombre entre les deux.

    La règle appliquée n'invente RIEN — elle n'utilise que des chiffres déjà
    gravés : branche 1 jusqu'au HAUT de la bande de cette cellule (0.5 ms, soit
    0.227 x la marge, où « ≪ » est garanti par la bande elle-même), branche 2 à
    partir de la marge (2.199 ms). Entre les deux, **rien n'est prononcé** et le
    ratio est surfacé : prononcer là serait choisir un seuil, ce que la mission
    interdit."""
    lecture = lecture_cellule2(mediane)
    assert lecture["branche"] == branche
    assert lecture["ratio_sur_marge_v4"] == pytest.approx(mediane / MARGE_V4_MS)


def test_lecture_cellule2_branche1_porte_son_interdiction():
    """La branche 1 est une lecture GRAVÉE D'AVANCE, y compris son refus : la
    dette se DÉPLACE (vers la composition et R2+), elle ne rétrécit pas. Le
    texte prononcé doit porter l'interdiction, sinon la lecture serait lue
    comme « le rendu tient »."""
    texte = lecture_cellule2(0.1)["texte"]
    assert "INTERDICTION" in texte
    assert "DEPLACE" in texte


# --- Chronométrie ------------------------------------------------------------


def test_chronometre_compte_les_appels_et_la_chauffe():
    """La chauffe n'entre PAS dans les statistiques, et il y a exactement
    `n_appels` mesures. Test de comptage, pas de mesure : les chiffres du
    prereg sont verdict-grade et n'appartiennent pas aux tests."""
    appels = {"n": 0}

    def appel():
        appels["n"] += 1

    mesure = chronometre(appel, n_appels=7, n_chauffe=3)
    assert appels["n"] == 10
    assert mesure["n_appels"] == 7 and mesure["n_chauffe"] == 3
    assert mesure["min_ms"] <= mesure["mediane_ms"] <= mesure["max_ms"]


def test_chronometre_synchronise_avant_chaque_lecture_dhorloge():
    """Tradition B6 : `synchronise` est appelé avant CHAQUE lecture d'horloge,
    les deux — sans la synchro d'entrée, la file laissée par l'itération
    précédente s'imputerait à l'itération courante. Donc 2 synchros par appel
    mesuré, et aucune pendant la chauffe."""
    compte = {"sync": 0}
    mesure = chronometre(lambda: None, n_appels=5, n_chauffe=2,
                         synchronise=lambda: compte.__setitem__("sync", compte["sync"] + 1))
    assert compte["sync"] == 2 * mesure["n_appels"]


# --- Champ d'équivalence -----------------------------------------------------


def test_champ_equivalence_couvre_les_deux_branches_srgb():
    """La rampe couvre la bande ENTIÈRE, les deux extrémités EXACTES et la
    branche linéaire de sRGB — ce qu'un tirage aléatoire ne garantit pas.
    C'est ce qui donne un sens à « tolérance zéro »."""
    champ = champ_equivalence()
    assert champ.shape == (1080, 1920) and champ.dtype == np.float64
    assert champ.min() == 0.0 and champ.max() == 1.0
    assert int((champ <= 0.0031308).sum()) > 1000    # branche linéaire échantillonnée


@gpu_requis
def test_equivalence_sonde_cupy_contre_r1_numpy():
    """L'ÉQUIVALENCE, tolérance ZÉRO (prereg cellule 2) — sur un ÉCHANTILLON
    du champ test (le plein 1920x1080 est pour le runner, pas pour la suite).

    Ce test ne présuppose AUCUNE branche : l'arrondi f32/f64 de la puissance
    1/2.4 PEUT casser le bit-exact, et si c'est le cas le chiffre doit être
    remonté, pas toléré. Il vérifie donc que la sonde est DÉCIDABLE (elle rend
    des uint8 comparables et des valeurs encodées dans [0,1]) et IMPRIME
    l'écart mesuré."""
    import cupy as cp

    champ = champ_equivalence()[::37, ::41].copy()
    attendu = quantifie_uint8(srgb_encode(champ))
    obtenu_gpu, encode_gpu = etages_srgb_quantif_cupy(cp, cp.asarray(champ, dtype=cp.float32))
    obtenu = cp.asnumpy(obtenu_gpu)

    n_diff = int((attendu != obtenu).sum())
    ecart = int(np.abs(attendu.astype(np.int16) - obtenu.astype(np.int16)).max())
    v_min = float(cp.asnumpy(encode_gpu.min()))
    v_max = float(cp.asnumpy(encode_gpu.max()))
    print(f"\n[equivalence f32 vs f64] {n_diff}/{attendu.size} pixels differents, "
          f"ecart max {ecart} niveau(x), encode f32 dans [{v_min:.9f}, {v_max:.9f}]")

    assert obtenu.dtype == np.uint8 and obtenu.shape == attendu.shape
    assert 0.0 <= v_min and v_max <= 1.0, (
        "l'encodage f32 sort de [0,1] : `astype(uint8)` enroulerait silencieusement, "
        "exactement la classe de defaut que la garde fail-loud de R1 refuse.")


def test_empreinte_r1_est_celle_du_verrou():
    """Le runner recalcule l'empreinte de R1 dans son JSON. Elle doit être
    celle que `tests/test_arcC_rendu.py` verrouille : le chiffrage porte sur
    l'objet daté, sinon P2 chiffrerait autre chose que ce que P3 traversera."""
    from tests.test_arcC_rendu import LONGUEUR_ATTENDUE, SHA256_ATTENDU

    empreinte = empreinte_r1()
    assert empreinte["sha256"] == SHA256_ATTENDU
    assert empreinte["octets"] == LONGUEUR_ATTENDUE


# --- Pré-requis P3 : garde de COMPARABILITÉ ----------------------------------


def test_garde_comparabilite_leve_hors_geometrie_du_pin():
    """§A38-CORRECTION : une session humaine hors `pic-csf` changerait DEUX
    choses à la fois (chemin de rendu ET taille d'affichage) ; l'écart mesuré
    ne serait attribuable ni à l'un ni à l'autre. Le message est un AIGUILLAGE :
    défaut, cause, correction exacte, et l'acte qu'aucune donnée n'a été
    produite."""
    with pytest.raises(RuntimeError) as exc:
        assert_geometrie_comparable("plafond", est_session_humaine=True)
    message = str(exc.value)
    assert "--geometrie pic-csf" in message
    assert "AUCUNE donnée n'est produite" in message


def test_garde_comparabilite_passe_a_la_geometrie_du_pin():
    """La campagne fondatrice du pin (manifeste 2026-07-05, `pic-csf`) serait
    passée — par identité. C'est la propriété qui rend la garde légitime, là où
    l'ancienne garde d'acuité l'aurait refusée (ratio 1.71)."""
    assert GEOMETRIE_DU_PIN == "pic-csf"
    assert_geometrie_comparable(GEOMETRIE_DU_PIN, est_session_humaine=True)


@pytest.mark.parametrize("geometrie", ("pic-csf", "plafond"))
def test_garde_comparabilite_no_op_hors_session_humaine(geometrie):
    """Replay et sujet synthétique : INCHANGÉS, à TOUTE géométrie — rien ne
    s'affiche, aucune donnée humaine n'est produite, il n'y a rien à comparer
    à l'IC gravé."""
    assert_geometrie_comparable(geometrie, est_session_humaine=False)


def test_ancienne_garde_dacuite_aurait_refuse_la_campagne_du_pin():
    """Le FAIT qui a fait remplacer la garde, gardé sous test pour qu'il ne se
    re-découvre pas : à pic-CSF, `ppd` s'annule et la cellule vaut
    `(porteuse/c_deg_cible)·60/64` = 1.71875 arcmin — indépendamment de l'écran
    et de la distance. Une garde « cellule ≥ seuil d'acuité (1.0) » aurait donc
    refusé TOUTE session pic-CSF, y compris celle qui a produit le pin."""
    from src.arcC_calibration import observation_cellule_pic_csf, pixels_par_degre

    for distance_mm in (400.0, 600.0, 750.0, 1500.0):
        ppd = pixels_par_degre(1920.0, 310.0, distance_mm)
        obs = observation_cellule_pic_csf(ppd)
        assert obs["cellule_arcmin_pic_csf"] == pytest.approx(1.71875, abs=6e-3)
        assert obs["ratio_cellule_sur_seuil"] > 1.0
    # Et le report reste un CHIFFRE, jamais un booléen (§C7 pièce 3).
    assert not any(isinstance(v, bool) for v in obs.values())


# --- Pré-requis P3 : liaison sidecar <-> log ---------------------------------


def _essai_journal():
    """UN essai, au schéma RÉEL de `src/arcC_abx.py` (intouché) — construit
    depuis son propre `from_dict`, jamais réimplémenté ici."""
    from src.arcC_abx import EssaiJournal

    return EssaiJournal.from_dict(dict(
        numero_staircase=0, indice_essai=0, regime="severe", seed_source=101,
        L_source=10, budget=32, t=0.5, delta_chi_mesure=0.1, type_essai="roving",
        reponse="A", correct=True, latence_s=1.25, niveau_delta_chi=0.1,
        est_renversement=False))


def test_sha256_fichier_egale_le_log_relu(tmp_path):
    """La liaison sidecar↔log sur un log RÉELLEMENT écrit par le harnais
    gravé : le sha que le sidecar consignera égale celui du fichier relu,
    calculé indépendamment par `hashlib`. `arcC_abx.py` est INTOUCHÉ — la
    liaison est un post-traitement de coquille, pas un changement de format."""
    from src.arcC_abx import ecrit_log_jsonl, lit_log_jsonl

    chemin = tmp_path / "session_0_severe.jsonl"
    ecrit_log_jsonl(chemin, [_essai_journal()])
    assert sha256_fichier(chemin) == hashlib.sha256(chemin.read_bytes()).hexdigest()
    assert len(lit_log_jsonl(chemin)) == 1     # c'est bien un log d'essais relisible


def test_log_altere_dun_octet_est_detectable(tmp_path):
    """Un log altéré d'UN SEUL octet donne un sha différent : c'est toute la
    raison d'être de la liaison — l'appariement sidecar↔log cesse de reposer
    sur le nom de fichier."""
    chemin = tmp_path / "session.jsonl"
    chemin.write_text('{"reponse": "A"}\n', encoding="utf-8")
    avant = sha256_fichier(chemin)
    chemin.write_text('{"reponse": "B"}\n', encoding="utf-8")
    assert sha256_fichier(chemin) != avant
