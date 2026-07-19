"""Tests F1 — diagnostic d'instrument D-1 / D-2 (§A19-lecture-sonde-EPS,
pocCascade2phys 07ef6cc).

Ce que ces tests protègent :
  1. le diagnostic ne RÈGLE rien — la branche (ii) est appliquée, EPS et
     k sont figés et toute autre valeur est refusée fail-loud ;
  2. le test à blanc est un vrai contrôle négatif (vivant := vérité ⇒ Δχ
     nul EXACT) et ses deux lectures sont pré-écrites ;
  3. la restriction ne compare QUE les cellules couvertes, sans casser la
     structure spectrale ;
  4. D-2 PROUVE que l'appareil de reconstruction est hors boucle, plutôt
     que de l'affirmer."""
import json

import numpy as np
import pytest

from scripts.run_f1_diagnostic_instrument import (
    CADENCE_FIGEE_DIAGNOSTIC,
    ECART_A_ATTRIBUER_MS,
    EPS_FIGE,
    FACTEUR_EFFONDREMENT,
    MEDIANE_MA_QUATER_MS,
    MEDIANE_SONDE_MS,
    PLANCHER_OBSERVE,
    SEUIL_MORT_MB,
    TOLERANCE_ABSORPTION_MS,
    ReconstructeurAvecCouverture,
    champ_restreint,
    exiger_rien_regle,
    lecture_d1,
    lecture_d2,
)
from scripts.run_f1_ma_quater import PipelineMaQuater, slots_v4
from scripts.run_f1_sonde_eps import (
    SEUIL_IC_BAS,
    _slot_observe,
    champ_verite,
    delta_chi_readout,
)
from src.f1_gpu.backend import cupy_disponible
from src.f1_gpu.pyramide import GeometriePyramide
from src.f1_gpu.transferts import TransfertComptable

gpu_requis = pytest.mark.skipif(
    not cupy_disponible(), reason="cupy/device CUDA indisponible")

GEO_PETITE = GeometriePyramide(n_fov=64, n_niv=10, n0=8,
                               slots=slots_v4(10), emboitee=True)


# ----- 1. le diagnostic ne règle RIEN -----

def test_eps_et_k_restent_figes():
    """La branche (ii) est APPLIQUÉE : ce driver diagnostique
    l'instrument, il ne cherche pas un meilleur EPS."""
    assert EPS_FIGE == 1e-4
    assert CADENCE_FIGEE_DIAGNOSTIC == 4
    exiger_rien_regle(EPS_FIGE, CADENCE_FIGEE_DIAGNOSTIC)
    for eps in (1e-5, 1e-3, 1e-2):
        with pytest.raises(RuntimeError, match="rien n'est réglé"):
            exiger_rien_regle(eps, CADENCE_FIGEE_DIAGNOSTIC)
    for k in (2, 3, 8):
        with pytest.raises(RuntimeError, match="σ_ω"):
            exiger_rien_regle(EPS_FIGE, k)


def test_les_chiffres_graves_sont_reproduits():
    assert PLANCHER_OBSERVE == 0.069
    assert SEUIL_MORT_MB == 0.0733
    assert MEDIANE_SONDE_MS == 17.249
    assert MEDIANE_MA_QUATER_MS == 16.589
    assert ECART_A_ATTRIBUER_MS == pytest.approx(0.66, abs=0.001)
    assert PLANCHER_OBSERVE < SEUIL_MORT_MB       # 6 % sous le seuil


# ----- 2. D-1(a) : le test à blanc est un vrai contrôle négatif -----

def test_le_test_a_blanc_est_exactement_nul():
    """vivant := vérité ⇒ Δχ = 0 EXACT par le chemin complet (décimation,
    albedo, delta_chi). Un contrôle négatif ne peut pas « mieux
    réussir » que zéro ; il ne peut que révéler un défaut."""
    rng = np.random.default_rng(17)
    champ = np.abs(rng.normal(0.05, 0.01, (64, 64)))
    for facteur in (1, 2, 4):
        assert delta_chi_readout(champ, champ, facteur) == 0.0


def test_lecture_a_blanc_nul_dit_absence_de_plancher_readout():
    mesure = {
        "test_a_blanc": {"max": 0.0, "median": 0.0, "exactement_nul": True},
        "comparaison_complete": {"max": 0.069},
        "comparaison_restreinte_cycle": {"max": 0.004},
        "comparaison_restreinte_cumulee": {"max": 0.025},
    }
    lecture = lecture_d1(mesure)["a_plancher_de_readout"]
    assert lecture["exactement_nul"] is True
    assert "aucun plancher de readout" in lecture["lecture"]


def test_lecture_a_blanc_non_nul_dit_plancher_transferable():
    """Tout résidu serait une propriété des PRIMITIVES, donc transférable
    à M-b qui les partage — c'est ce qui rend le contrôle gatant."""
    mesure = {
        "test_a_blanc": {"max": 0.01, "median": 0.01,
                         "exactement_nul": False},
        "comparaison_complete": {"max": 0.069},
        "comparaison_restreinte_cycle": {"max": 0.060},
        "comparaison_restreinte_cumulee": {"max": 0.065},
    }
    lecture = lecture_d1(mesure)["a_plancher_de_readout"]
    assert lecture["exactement_nul"] is False
    assert "TRANSFÉRABLE à M-b" in lecture["lecture"]
    assert str(SEUIL_MORT_MB) in lecture["lecture"]


# ----- 3. D-1(b) : la restriction aux cellules couvertes -----

def test_champ_restreint_annule_lecart_hors_couverture():
    """Les cellules non couvertes prennent la vérité : leur écart est nul
    et seules les couvertes contribuent. La grille reste pleine — on ne
    peut pas faire de FFT sur un sous-ensemble troué."""
    vivant = np.array([[1.0, 2.0], [3.0, 4.0]])
    verite = np.array([[1.5, 2.5], [3.5, 4.5]])
    couvert = np.array([[True, False], [False, True]])
    restreint = champ_restreint(vivant, verite, couvert)
    assert restreint.shape == vivant.shape
    np.testing.assert_array_equal(restreint,
                                  np.array([[1.0, 2.5], [3.5, 4.0]]))


def test_restriction_totale_equivaut_au_champ_complet():
    """Contre-épreuve : si tout est couvert, la restriction ne change
    rien — elle ne peut pas fabriquer d'amélioration à elle seule."""
    rng = np.random.default_rng(19)
    verite = np.abs(rng.normal(0.05, 0.01, (32, 32)))
    vivant = verite + 0.003 * rng.normal(size=(32, 32))
    tout = np.ones_like(verite, dtype=bool)
    assert delta_chi_readout(champ_restreint(vivant, verite, tout),
                             verite, 1) == pytest.approx(
        delta_chi_readout(vivant, verite, 1))


def test_restriction_vide_donne_un_ecart_nul():
    """Autre borne : si rien n'est couvert, il ne reste aucun écart."""
    rng = np.random.default_rng(23)
    verite = np.abs(rng.normal(0.05, 0.01, (32, 32)))
    vivant = verite + 0.01 * rng.normal(size=(32, 32))
    rien = np.zeros_like(verite, dtype=bool)
    assert delta_chi_readout(champ_restreint(vivant, verite, rien),
                             verite, 1) == 0.0


def test_le_seuil_deffondrement_est_nomme_avant_le_run():
    """Le gravé dit « si le plancher s'effondre » sans le chiffrer : le
    facteur est fixé ici, AVANT la donnée."""
    assert FACTEUR_EFFONDREMENT == 0.5
    mesure = {
        "test_a_blanc": {"max": 0.0, "median": 0.0, "exactement_nul": True},
        "comparaison_complete": {"max": 0.069},
        "comparaison_restreinte_cycle": {"max": 0.004},
        "comparaison_restreinte_cumulee": {"max": 0.025},
    }
    couverture = lecture_d1(mesure)["b_plancher_de_couverture"]
    assert couverture["seuil_effondrement"] == pytest.approx(0.0345)
    assert couverture["plancher_seffondre"] == {"cycle": True,
                                                "cumule": True}
    assert "COUVERTURE" in couverture["lecture"]
    assert "NON transférable" in couverture["lecture"]


def test_plancher_persistant_ne_conclut_pas_a_la_couverture():
    """Si la restriction ne fait pas tomber le plancher, la couverture
    n'explique rien — et le driver le dit sans inventer autre chose."""
    mesure = {
        "test_a_blanc": {"max": 0.0, "median": 0.0, "exactement_nul": True},
        "comparaison_complete": {"max": 0.069},
        "comparaison_restreinte_cycle": {"max": 0.068},
        "comparaison_restreinte_cumulee": {"max": 0.067},
    }
    couverture = lecture_d1(mesure)["b_plancher_de_couverture"]
    assert couverture["plancher_seffondre"] == {"cycle": False,
                                                "cumule": False}
    assert "PERSISTE" in couverture["lecture"]
    assert "reste à attribuer" in couverture["lecture"]


def test_le_seuil_ic_bas_est_reporte_sans_rouvrir_eps():
    """Que le restreint repasse sous 0.0603 est REPORTÉ, et rien de
    plus : la branche (ii) reste appliquée."""
    mesure = {
        "test_a_blanc": {"max": 0.0, "median": 0.0, "exactement_nul": True},
        "comparaison_complete": {"max": 0.069},
        "comparaison_restreinte_cycle": {"max": 0.004},
        "comparaison_restreinte_cumulee": {"max": 0.025},
    }
    couverture = lecture_d1(mesure)["b_plancher_de_couverture"]
    assert couverture["seuil_ic_bas"] == SEUIL_IC_BAS
    assert couverture["restreint_sous_le_seuil"] == {"cycle": True,
                                                     "cumule": True}
    assert "ne change RIEN" in couverture["note_seuil"]
    assert "Aucune relecture d'EPS" in couverture["note_seuil"]


def test_d1_ne_tire_aucune_autre_conclusion():
    mesure = {
        "test_a_blanc": {"max": 0.0, "median": 0.0, "exactement_nul": True},
        "comparaison_complete": {"max": 0.069},
        "comparaison_restreinte_cycle": {"max": 0.004},
        "comparaison_restreinte_cumulee": {"max": 0.025},
    }
    lecture = lecture_d1(mesure)
    assert "RIEN d'autre" in lecture["rappel_portee"]
    texte = json.dumps(lecture, ensure_ascii=False)
    for interdit in ("MORT", "PASS", "VERDICT"):
        assert interdit not in texte


# ----- 4. D-2 : l'appareil est hors boucle, et c'est PROUVÉ -----

def _mesure_d2(mediane: float) -> dict:
    return {
        "frame_time": {"mediane_ms": mediane, "p99_ms": mediane + 1.0},
        "transferts_median_ms": 2.7,
        "preuve_appareil_hors_boucle": {
            "capture_desactivee": True, "aucun_coefficient_retenu": True,
            "note": "PROUVÉ, pas affirmé"},
    }


def test_ecart_absorbe_designe_lappareil():
    lecture = lecture_d2(_mesure_d2(MEDIANE_MA_QUATER_MS + 0.05))
    assert lecture["ecart_absorbe"] is True
    assert "APPAREIL" in lecture["lecture"]
    assert "sans conséquence" in lecture["lecture"]


def test_ecart_persistant_designe_un_terme_de_production():
    """La marge de V4 est alors entamée, et le contrôle T1 de M-b
    (verdictal) s'appliquera à ce total."""
    lecture = lecture_d2(_mesure_d2(MEDIANE_SONDE_MS))
    assert lecture["ecart_absorbe"] is False
    assert "TERME DE PRODUCTION NON COMPTÉ" in lecture["lecture"]
    assert "contrôle T1" in lecture["lecture"]
    assert lecture["ecart_restant_ms"] == pytest.approx(
        ECART_A_ATTRIBUER_MS, abs=0.001)


def test_la_tolerance_dabsorption_est_nommee_avant_le_run():
    assert TOLERANCE_ABSORPTION_MS == 0.15
    limite = MEDIANE_MA_QUATER_MS + TOLERANCE_ABSORPTION_MS
    assert lecture_d2(_mesure_d2(limite))["ecart_absorbe"] is True
    assert lecture_d2(_mesure_d2(limite + 1e-6))["ecart_absorbe"] is False


def test_la_preuve_hors_boucle_est_portee_a_la_lecture():
    lecture = lecture_d2(_mesure_d2(16.6))
    preuve = lecture["preuve_appareil_hors_boucle"]
    assert preuve["capture_desactivee"] is True
    assert preuve["aucun_coefficient_retenu"] is True


# ----- plomberie GPU -----

@gpu_requis
def test_la_couverture_est_suivie_et_reste_partielle():
    """Le suivi distingue bien couvert et non couvert : si tout était
    couvert, D-1(b) ne pourrait rien montrer."""
    import cupy as cp
    pipeline = PipelineMaQuater(cp, GEO_PETITE, TransfertComptable(cp),
                                k=CADENCE_FIGEE_DIAGNOSTIC, eps=EPS_FIGE,
                                capturer_coefficients=True)
    reconstructeur = ReconstructeurAvecCouverture(pipeline)
    indice, position = _slot_observe(pipeline)
    for numero in range(8):
        if numero % CADENCE_FIGEE_DIAGNOSTIC == 0:
            reconstructeur.nouveau_cycle()
        pipeline.frame()
        reconstructeur.appliquer(pipeline.coefficients_frame)
    cycle = reconstructeur.masque(False, indice, position)
    cumule = reconstructeur.masque(True, indice, position)
    assert 0.0 < cycle.mean() < 1.0          # partielle, pas triviale
    assert cumule.mean() >= cycle.mean()     # le cumul englobe le cycle


@gpu_requis
def test_le_cycle_se_reinitialise():
    import cupy as cp
    pipeline = PipelineMaQuater(cp, GEO_PETITE, TransfertComptable(cp),
                                k=CADENCE_FIGEE_DIAGNOSTIC, eps=EPS_FIGE,
                                capturer_coefficients=True)
    reconstructeur = ReconstructeurAvecCouverture(pipeline)
    for _ in range(4):
        pipeline.frame()
        reconstructeur.appliquer(pipeline.coefficients_frame)
    assert any(masque.any() for masque in reconstructeur.couvert_cycle)
    reconstructeur.nouveau_cycle()
    assert not any(masque.any() for masque in reconstructeur.couvert_cycle)
    assert any(masque.any() for masque in reconstructeur.couvert_cumule)


@gpu_requis
def test_le_plancher_se_lit_sur_le_chemin_reel():
    """Plomberie : le chemin complet (couverture, restriction, readout)
    tourne sur le device et produit des nombres finis."""
    import cupy as cp
    pipeline = PipelineMaQuater(cp, GEO_PETITE, TransfertComptable(cp),
                                k=CADENCE_FIGEE_DIAGNOSTIC, eps=EPS_FIGE,
                                capturer_coefficients=True)
    reconstructeur = ReconstructeurAvecCouverture(pipeline)
    indice, position = _slot_observe(pipeline)
    for _ in range(6):
        pipeline.frame()
        reconstructeur.appliquer(pipeline.coefficients_frame)
    vivant = reconstructeur.champ(indice, position)
    verite = champ_verite(pipeline, indice, position)
    couvert = reconstructeur.masque(True, indice, position)
    complet = delta_chi_readout(vivant, verite, 2)
    restreint = delta_chi_readout(
        champ_restreint(vivant, verite, couvert), verite, 2)
    assert np.isfinite(complet) and np.isfinite(restreint)
    assert delta_chi_readout(verite, verite, 2) == 0.0     # à blanc
