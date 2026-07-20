"""Tests F1 — driver de la borne L3 (achat A, §A18-complément-2,
pocCascade2phys d720da6).

Ce que ces tests protègent : (i) les DEUX bras ont la même structure
d'allocation — sinon la différence mesurerait aussi une différence de
résidence, pas seulement le travail d'émission ; (ii) le transfert suit
bien le retard d'une frame ; (iii) l'arithmétique de lecture est celle du
protocole ; (iv) le driver ne prononce rien."""
import json

import numpy as np
import pytest

from scripts.run_f1_borne_l3 import (
    A_BATCHE_GRAVE_MS,
    CIBLE_REMONTEE_MS,
    REMONTEE_CUPY_MS,
    BrasBorne,
    lecture_mecanique,
    mesurer_ecart_sec_mouille,
)
from scripts.run_f1_ma_ter import B_FRAME_MS
from src.f1_gpu.backend import cupy_disponible
from src.f1_gpu.pyramide import EPS_DETAIL, GeometriePyramide, Slot
from src.f1_gpu.transferts import TransfertComptable

gpu_requis = pytest.mark.skipif(
    not cupy_disponible(), reason="cupy/device CUDA indisponible")

GEO_TEST = GeometriePyramide(
    n_fov=8, n_niv=4, n0=16,
    slots=(Slot(niveau=1, c=4, role="fovea"),
           Slot(niveau=2, c=4, role="fovea"),
           Slot(niveau=3, c=8, role="fovea"),
           Slot(niveau=3, c=8, role="energie")))


# ----- les deux bras ont la MÊME structure -----

@gpu_requis
def test_les_deux_bras_allouent_les_references():
    """On retire du travail, jamais de la structure : le bras REF alloue
    ses références bien qu'il n'émette pas. Sans cela, A_L3 − A_REF
    mélangerait le travail d'émission et un écart de résidence."""
    import cupy as cp
    ref = BrasBorne(cp, GEO_TEST, avec_emission=False)
    l3 = BrasBorne(cp, GEO_TEST, avec_emission=True,
                   transferts=TransfertComptable(cp))
    assert len(ref.references) == len(l3.references)
    for a, b in zip(ref.references, l3.references):
        assert a.shape == b.shape
    assert ref.n_blocs() == l3.n_blocs()
    assert ref.compacteurs == []          # seul L3 émet
    assert len(l3.compacteurs) == len(l3.fenetres)


@gpu_requis
def test_la_residence_de_l3_ajoute_les_buffers_demission():
    """Les buffers d'émission sont dimensionnés au pire cas (choix 5) :
    la résidence de L3 dépasse donc celle de REF, et c'est reporté."""
    import cupy as cp
    ref = BrasBorne(cp, GEO_TEST, avec_emission=False)
    l3 = BrasBorne(cp, GEO_TEST, avec_emission=True,
                   transferts=TransfertComptable(cp))
    assert l3.octets_etat() > ref.octets_etat()


@gpu_requis
def test_les_references_sont_decalees_sinon_rien_nemet():
    """Sans écart initial entre fenêtre et référence, la borne mesurerait
    une remontée vide — le driver décale donc la référence."""
    import cupy as cp
    bras = BrasBorne(cp, GEO_TEST, avec_emission=True,
                     transferts=TransfertComptable(cp))
    for fenetre, reference in zip(bras.fenetres, bras.references):
        assert not bool((fenetre == reference).all())


# ----- le transfert suit le retard d'une frame -----

@gpu_requis
def test_la_premiere_frame_ne_transfere_rien():
    """Retard d'une frame (choix 6) : à la première frame aucun compteur
    n'est encore arrivé, donc rien n'est transféré — la frame est dans le
    warmup, exclu de la série (B6)."""
    import cupy as cp
    transferts = TransfertComptable(cp)
    bras = BrasBorne(cp, GEO_TEST, avec_emission=True,
                     transferts=transferts)
    bras.frame()
    bilan = transferts.frame_suivante()
    assert bilan["d2h_octets"] == 0


@gpu_requis
def test_les_frames_suivantes_transferent():
    """Une fois le compteur arrivé, le transfert a lieu — deux D2H par
    groupe (valeurs f32 puis indices u32), comme le schéma B4."""
    import cupy as cp
    transferts = TransfertComptable(cp)
    bras = BrasBorne(cp, GEO_TEST, avec_emission=True,
                     transferts=transferts)
    for _ in range(4):
        bras.frame()
        bilan = transferts.frame_suivante()
    assert bilan["d2h_octets"] > 0
    assert bilan["d2h_n"] == 2 * len(bras.fenetres)


@gpu_requis
def test_le_bras_ref_ne_transfere_jamais():
    """Le bras de référence mesure F seul : aucun trafic."""
    import cupy as cp
    transferts = TransfertComptable(cp)
    bras = BrasBorne(cp, GEO_TEST, avec_emission=False)
    for _ in range(4):
        bras.frame()
        bilan = transferts.frame_suivante()
    assert bilan["d2h_octets"] == 0
    assert bilan["h2d_octets"] == 0


@gpu_requis
def test_les_tailles_vues_sont_enregistrees_par_frame():
    """Le diagnostic du retard repose sur la VARIABILITÉ des tailles ; un
    cumul serait tautologique (chaque frame transfère ce que la
    précédente a émis)."""
    import cupy as cp
    bras = BrasBorne(cp, GEO_TEST, avec_emission=True,
                     transferts=TransfertComptable(cp))
    for _ in range(5):
        bras.frame()
    for compacteur in bras.compacteurs:
        assert len(compacteur.tailles_vues) == 5
        assert compacteur.tailles_vues[0] == 0      # aucun précédent
        assert max(compacteur.tailles_vues) > 0     # puis ça émet


@gpu_requis
def test_le_compteur_final_est_releve_hors_chrono():
    """La dernière frame n'a pas de suivante pour être transférée : son
    compteur est relevé à part, par la seule lecture bloquante du bras."""
    import cupy as cp
    bras = BrasBorne(cp, GEO_TEST, avec_emission=True,
                     transferts=TransfertComptable(cp))
    for _ in range(3):
        bras.frame()
    bras.relever_emissions()
    assert sum(c.compteur_final for c in bras.compacteurs) > 0


@gpu_requis
def test_diagnostic_retard_signale_la_stabilite():
    """Sur un état qui se répète, les tailles doivent être stables — donc
    le retard d'une frame est neutre, et le diagnostic le dit."""
    import cupy as cp
    bras = BrasBorne(cp, GEO_TEST, avec_emission=True,
                     transferts=TransfertComptable(cp))
    for _ in range(6):
        bras.frame()
    diagnostic = bras.compacteurs[0].diagnostic_retard()
    assert diagnostic["frames_observees"] == 6
    assert diagnostic["taille_max"] > 0
    assert isinstance(diagnostic["tailles_stables"], bool)
    # §A21 : la note ne rassure plus par l'incrémentalité (nota RETIRÉ),
    # elle dit la bijection que le ping-pong garantit.
    assert "PING-PONG" in diagnostic["note"]
    assert "une fois et une seule" in diagnostic["note"]


# ----- arithmétique de lecture -----

def _bras_factice(mediane: float, stables: bool = True,
                  avec_emission: bool = False) -> dict:
    resultat = {"frame_time": {"mediane_ms": mediane, "p99_ms": mediane}}
    if avec_emission:
        resultat["retard_transfert"] = [
            {"frames_observees": 300, "taille_min": 100,
             "taille_max": 100 if stables else 180, "taille_mediane": 100,
             "tailles_stables": stables, "compteur_final_non_transfere": 100,
             "note": "incrémental"}]
    return resultat


def _sec_mouille(ecart: float = -0.05) -> dict:
    return {"n_blocs": 3, "mediane_mouille_ms": 1.0,
            "mediane_second_systeme_sec_ms": 1.0 + ecart,
            "ecart_relatif": ecart, "note": "reporté, non appliqué"}


def test_remontee_est_la_difference_des_bras():
    lecture = lecture_mecanique(
        _bras_factice(14.4),
        _bras_factice(16.0, avec_emission=True),
        _sec_mouille())
    assert lecture["remontee_l3_ms"] == pytest.approx(1.6)
    assert lecture["comparaisons"]["remontee_cupy_gravee_ms"] == 8.523
    assert lecture["comparaisons"]["cible_ms"] == 1.5
    assert lecture["comparaisons"]["facteur_vs_cupy"] == pytest.approx(
        8.523 / 1.6)


def test_derive_machine_compare_au_grave():
    """Le bras REF re-mesure F seul : un écart notable signalerait une
    dérive de machine, pas un effet de L3."""
    lecture = lecture_mecanique(
        _bras_factice(14.432),
        _bras_factice(16.0, avec_emission=True), _sec_mouille())
    assert lecture["derive_machine"]["ecart_relatif"] == pytest.approx(0.0)
    assert lecture["derive_machine"]["a_batche_grave_ms"] == A_BATCHE_GRAVE_MS


def test_retard_transfert_reporte_la_variabilite():
    """Le diagnostic porte sur la VARIABILITÉ des tailles, pas sur un
    cumul — celui-ci serait tautologique, chaque frame transférant ce que
    la précédente a émis."""
    stable = lecture_mecanique(
        _bras_factice(14.4), _bras_factice(16.0, avec_emission=True),
        _sec_mouille())["retard_transfert"]
    assert stable["toutes_tailles_stables"] is True
    assert "tautologique" in stable["note"]

    instable = lecture_mecanique(
        _bras_factice(14.4),
        _bras_factice(16.0, stables=False, avec_emission=True),
        _sec_mouille())["retard_transfert"]
    assert instable["toutes_tailles_stables"] is False


def test_budget_restant_peut_etre_negatif_et_se_lit_tel_quel():
    lecture = lecture_mecanique(
        _bras_factice(14.4), _bras_factice(17.5, avec_emission=True),
        _sec_mouille())
    budget = lecture["budget_restant"]
    assert budget["valeur_ms"] == pytest.approx(B_FRAME_MS - 17.5)
    assert "PAS M-a-quater" in budget["note"]


def test_facteur_non_calcule_si_remontee_nulle_ou_negative():
    """Une remontée nulle ou négative ne produit pas un facteur absurde."""
    lecture = lecture_mecanique(
        _bras_factice(16.0), _bras_factice(16.0, avec_emission=True),
        _sec_mouille())
    assert lecture["comparaisons"]["facteur_vs_cupy"] is None


def test_lecture_porte_lanti_minoration():
    lecture = lecture_mecanique(
        _bras_factice(14.4), _bras_factice(16.0, avec_emission=True),
        _sec_mouille(ecart=-0.12))
    assert lecture["ecart_sec_mouille"]["ecart_relatif"] == -0.12


# ----- portée : le driver ne prononce rien -----

def test_le_driver_ne_prononce_rien():
    lecture = lecture_mecanique(
        _bras_factice(14.4), _bras_factice(16.0, avec_emission=True),
        _sec_mouille())
    texte = json.dumps(lecture, ensure_ascii=False)
    for interdit in ("MORT", "PASS", "VERDICT"):
        assert interdit not in texte
    rappel = lecture["rappel_portee"]
    assert "JETABLE" in rappel
    assert "dernière escalade" in rappel
    assert "décision de Romain" in lecture["comparaisons"]["note"]


def test_constantes_gravees():
    """Aucun chiffre ne bouge."""
    assert REMONTEE_CUPY_MS == 8.523
    assert CIBLE_REMONTEE_MS == 1.5
    assert A_BATCHE_GRAVE_MS == 14.432
    assert EPS_DETAIL == 1e-4


# ----- vérification anti-minoration -----

@gpu_requis
def test_mesure_sec_mouille_produit_les_deux_medianes():
    """La micro-mesure compare un second système mouillé et sec, et
    reporte l'écart — elle ne l'applique pas à la borne."""
    import cupy as cp
    resultat = mesurer_ecart_sec_mouille(cp, GEO_TEST, n_blocs=1)
    assert resultat["mediane_mouille_ms"] > 0
    assert resultat["mediane_second_systeme_sec_ms"] > 0
    assert np.isfinite(resultat["ecart_relatif"])
    assert "non appliqué à la borne" in resultat["note"]
