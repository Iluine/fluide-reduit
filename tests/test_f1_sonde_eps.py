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
    CHAMP_SEDIMENT,
    EPS_BALAYAGE,
    EPS_EN_VIGUEUR,
    SEUIL_IC_BAS,
    ReconstructeurNiveau0,
    _slot_observe,
    champ_verite,
    delta_chi_readout,
    discrimination_du_balayage,
    exiger_cadence_figee,
    lecture_mecanique,
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

def _mesure(eps: float, dchi_max: float, octets: float = 1000.0) -> dict:
    return {"eps": eps, "delta_chi": {"delta_chi_max": dchi_max},
            "chrono": {"octets_par_frame_median": octets}}


def test_retient_le_plus_grand_eps_sous_le_seuil():
    """La règle ne dit pas « le meilleur Δχ » mais « le PLUS GRAND EPS »
    qui passe : c'est le plus économique en trafic."""
    lecture = lecture_mecanique([
        _mesure(1e-5, 0.001), _mesure(1e-4, 0.010),
        _mesure(3e-4, 0.030), _mesure(1e-3, 0.070)])
    assert lecture["eps_retenu"] == 3e-4
    assert lecture["autre_remonte"] is False
    assert lecture["eps_satisfaisants"] == [1e-5, 1e-4, 3e-4]


def test_aucun_eps_retenu_par_defaut_si_aucun_ne_passe():
    """Y compris 1e-5 : AUTRE remonté, et surtout AUCUN repli — « jamais
    de choix après courbe »."""
    lecture = lecture_mecanique([
        _mesure(1e-5, 0.08), _mesure(1e-4, 0.12)])
    assert lecture["eps_retenu"] is None
    assert lecture["autre_remonte"] is True
    assert "AUCUN EPS retenu par défaut" in (
        lecture["branche_preecrite_si_aucun"])


def test_le_seuil_est_strict_et_vaut_ic_bas():
    """< 0.0603, pas <=. La lecture porte sur l'IC entier (§A13)."""
    assert SEUIL_IC_BAS == 0.0603
    assert lecture_mecanique([_mesure(1e-4, SEUIL_IC_BAS)])[
        "eps_retenu"] is None
    assert lecture_mecanique([_mesure(1e-4, SEUIL_IC_BAS - 1e-9)])[
        "eps_retenu"] == 1e-4


def test_le_balayage_grave_est_reproduit():
    assert EPS_BALAYAGE == (1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
    assert EPS_EN_VIGUEUR == 1e-4
    assert EPS_EN_VIGUEUR in EPS_BALAYAGE


# ----- le balayage a-t-il seulement DISCRIMINÉ ? -----

def test_signale_un_balayage_qui_ne_discrimine_pas():
    """Le cas mesuré au build : le trafic varie d'un facteur énorme, mais
    Δχ ne bouge pas — l'observable est alors gouverné par la PÉREMPTION
    L1, pas par EPS. Un AUTRE ne voudrait alors PAS dire « EPS trop
    grand », et la lecture doit pouvoir le voir."""
    mesures = [_mesure(1e-5, 0.0806, octets=200000.0),
               _mesure(1e-2, 0.0805, octets=100.0)]
    diagnostic = discrimination_du_balayage(mesures)
    assert diagnostic["a_discrimine"] is False
    assert diagnostic["amplitude_relative"] < 0.01
    assert diagnostic["rapport_octets"] == pytest.approx(2000.0)
    assert "PÉREMPTION L1" in diagnostic["note"]


def test_signale_un_balayage_qui_discrimine():
    """Contre-épreuve : si Δχ suit EPS, le diagnostic le dit."""
    diagnostic = discrimination_du_balayage([
        _mesure(1e-5, 0.010), _mesure(1e-2, 0.090)])
    assert diagnostic["a_discrimine"] is True
    assert diagnostic["amplitude_relative"] > 0.5


def test_la_lecture_porte_le_diagnostic_de_discrimination():
    lecture = lecture_mecanique([_mesure(1e-5, 0.0806),
                                 _mesure(1e-2, 0.0805)])
    assert lecture["discrimination_du_balayage"]["a_discrimine"] is False
    assert lecture["autre_remonte"] is True     # les deux dépassent


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
    lecture = lecture_mecanique([_mesure(1e-4, 0.01)])
    assert lecture["cadence_figee"] == 4
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
    lecture = lecture_mecanique([_mesure(1e-4, 0.01)])
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
