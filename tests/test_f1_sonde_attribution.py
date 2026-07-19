"""Tests F1 — sonde d'attribution du résidu M-a-ter
(§A16-lecture-M-a-ter, pocCascade2phys a2440a9).

Ce que ces tests protègent : (i) les étages désactivés retirent du
TRAVAIL et RIEN de la structure — c'est la validité même de la
décomposition ; (ii) la config des deux bras est la V2 EXACTE, pas une
V2 approchée ; (iii) l'arithmétique de décomposition est celle qui est
gravée. Backend numpy (B1) sauf la plomberie GPU, marquée."""
import numpy as np
import pytest

from scripts.run_f1_ma_ter import slots_v2
from scripts.run_f1_sonde_attribution import (
    DELTA_X_IMMOBILE,
    DELTA_X_MOBILE,
    V2_CALCUL_MS,
    V2_FULL_MEDIANE_MS,
    V2_TRANSFERTS_MS,
    lecture_mecanique,
    postes_residuels_bras_a,
)
from src.f1_gpu.backend import cupy_disponible
from src.f1_gpu.pyramide import GeometriePyramide, PyramideFovea
from src.f1_gpu.transferts import TransfertComptable

GEO = GeometriePyramide(n_fov=8, n_niv=4, n0=16)


def _pyramide(remontee_active: bool):
    transferts = TransfertComptable(np)
    pyramide = PyramideFovea(np, GEO, transferts,
                             remontee_active=remontee_active)
    transferts.frame_suivante()
    return pyramide, transferts


# ----- l'OFF retire du TRAVAIL, pas de la STRUCTURE -----

def test_remontee_off_ne_retire_aucune_allocation():
    """B2 identique : `references[j]` reste préalloué à shape pleine, et
    l'état préalloué total est le MÊME qu'avec la remontée active. Une
    pyramide amputée de travail, jamais une pyramide plus petite."""
    avec, _ = _pyramide(remontee_active=True)
    sans, _ = _pyramide(remontee_active=False)
    assert sans.octets_etat() == avec.octets_etat()
    for j in GEO.niveaux_gpu:
        assert sans.references[j].shape == avec.references[j].shape
        assert sans.references[j].shape == sans.fenetres[j].shape


def test_remontee_off_supprime_les_d2h_et_fige_la_reference():
    """Le TRAVAIL retiré : plus aucun D2H, et la référence n'est plus
    mise à jour (l'étage incrémental est bien hors service)."""
    pyramide, transferts = _pyramide(remontee_active=False)
    references_avant = {j: pyramide.references[j].copy()
                        for j in GEO.niveaux_gpu}
    diag = pyramide.frame(delta_x=DELTA_X_MOBILE)
    bilan = transferts.frame_suivante()
    assert diag["remontee_active"] is False
    assert diag["nnz_par_niveau"] == {}
    assert bilan["d2h_octets"] == 0
    assert bilan["d2h_n"] == 0
    for j in GEO.niveaux_gpu:
        assert np.array_equal(pyramide.references[j], references_avant[j])


def test_remontee_active_reste_le_defaut():
    """`False` n'est JAMAIS un défaut : M-a, M-c et M-a-ter gardent la
    remontée. Non-régression du reste du harnais."""
    pyramide, transferts = _pyramide(remontee_active=True)
    assert pyramide.remontee_active is True
    diag = pyramide.frame(delta_x=DELTA_X_MOBILE)
    bilan = transferts.frame_suivante()
    assert diag["remontee_active"] is True
    assert set(diag["nnz_par_niveau"]) == {"1", "2", "3"}
    assert bilan["d2h_octets"] > 0


# ----- bras A : fovéa immobile -----

def test_bras_a_immobile_ne_descend_rien_et_ne_deplace_rien():
    """Fovéa IMMOBILE = `delta_x=0` : aucun niveau ne bouge, donc aucune
    re-prédiction CPU et aucune H2D. Combiné à la remontée OFF, la frame
    du bras A est du F PUR — c'est ce qui rend A comparable à la bande."""
    pyramide, transferts = _pyramide(remontee_active=False)
    diag = pyramide.frame(delta_x=DELTA_X_IMMOBILE)
    bilan = transferts.frame_suivante()
    assert diag["niveaux_deplaces"] == []
    assert bilan["h2d_octets"] == 0
    assert bilan["d2h_octets"] == 0


def test_bras_a_immobile_fait_quand_meme_avancer_f():
    """L'immobilité retire le déplacement, pas le CALCUL : F s'applique
    bien à chaque frame (sinon la sonde mesurerait le vide)."""
    pyramide, _ = _pyramide(remontee_active=False)
    avant = {j: pyramide.fenetres[j].copy() for j in GEO.niveaux_gpu}
    for _ in range(3):
        pyramide.frame(delta_x=DELTA_X_IMMOBILE)
    change = [not np.array_equal(pyramide.fenetres[j], avant[j])
              for j in GEO.niveaux_gpu]
    assert any(change)
    assert all(np.isfinite(pyramide.fenetres[j]).all()
               for j in GEO.niveaux_gpu)


def test_bras_b_mobile_descend_mais_ne_remonte_pas():
    """Bras B = le bras A plus la prédiction de déplacement : H2D
    présentes, D2H toujours nulles. C'est cet écart qui EST B − A."""
    pyramide, transferts = _pyramide(remontee_active=False)
    pyramide.frame(delta_x=DELTA_X_MOBILE)
    bilan = transferts.frame_suivante()
    assert bilan["h2d_octets"] > 0
    assert bilan["d2h_octets"] == 0


# ----- la config des bras est la V2 EXACTE -----

def test_la_sonde_mesure_la_v2_exacte():
    """Les slots viennent du driver M-a-ter : l'identité de config est
    structurelle, pas recopiée. 12 slots, 17 blocs, énergie S3 2+1."""
    geo = GeometriePyramide(n_fov=512, n_niv=10, slots=slots_v2())
    resume = geo.resume_slots()
    assert resume["n_slots_total"] == 12
    assert resume["blocs_actifs_gpu"] == 17
    assert resume["par_niveau"]["9"]["roles"] == [
        "fovea", "energie", "energie"]
    assert resume["par_niveau"]["8"]["roles"] == ["fovea", "energie"]


# ----- l'arithmétique de décomposition gravée -----

def _bras(mediane_ms: float) -> dict:
    return {"frame_time": {"mediane_ms": mediane_ms, "p99_ms": mediane_ms}}


def test_decomposition_suit_la_formule_gravee():
    """coût(prédiction) = B − A ; coût(remontée) = V2_full − B, avec
    V2_full = 31.097 ms GRAVÉ (non rejoué)."""
    lecture = lecture_mecanique(_bras(14.0), _bras(20.0))
    decomposition = lecture["decomposition_ms"]
    assert decomposition["f_seul_bras_a"] == 14.0
    assert decomposition["prediction_deplacement_b_moins_a"] == pytest.approx(
        6.0)
    assert decomposition["remontee_full_moins_b"] == pytest.approx(11.097)
    assert decomposition["somme_de_controle"] == pytest.approx(
        V2_FULL_MEDIANE_MS)


def test_attribution_confirmee_si_a_dans_la_bande():
    """ATTENDU PRÉ-ÉCRIT : A ∈ [12.18, 16.48] ⇒ attribution CONFIRMÉE."""
    lecture = lecture_mecanique(_bras(14.3), _bras(20.0))
    attendu = lecture["attendu_preecrit_bras_a"]
    assert attendu["a_dans_bande"] is True
    assert "CONFIRMÉE" in attendu["branche_si_dans_bande"]


def test_hypothese_machinerie_fausse_si_a_hors_bande():
    """Hors bande ⇒ l'hypothèse machinerie est FAUSSE : chercher ailleurs
    (AUTRE remonté). Le cas qui compte : A proche des 30.484 ms mesurés
    en M-a-ter signifierait que F seul porte déjà tout le résidu."""
    lecture = lecture_mecanique(_bras(30.0), _bras(30.5))
    attendu = lecture["attendu_preecrit_bras_a"]
    assert attendu["a_dans_bande"] is False
    assert "FAUSSE" in attendu["branche_si_hors_bande"]


def test_la_sonde_ne_prononce_rien():
    """Portée : aucun verdict, aucune escalade, aucun design — le cap
    (« pas de tapis roulant ») est rappelé dans la lecture elle-même, et
    aucun mot de verdict n'apparaît nulle part dans la lecture."""
    import json
    lecture = lecture_mecanique(_bras(14.0), _bras(20.0))
    rappel = lecture["rappel_portee"]
    assert "attribution seule" in rappel.lower()
    assert "SÉPARÉE" in rappel
    texte = json.dumps(lecture, ensure_ascii=False)
    for interdit in ("MORT", "PASS", "VERDICT", "escalade décidée"):
        assert interdit not in texte


def test_postes_residuels_du_bras_a_sont_comptes():
    """« F seul » = la pyramide privée de déplacement et de remontée, PAS
    le kernel nu. Les postes per-NIVEAU restants (9 appels, 18 lancements,
    9 réductions CFL synchrones contre 1/2/1 pour l'ancre M-a′) sont
    comptés dans le JSON — si A sort hors bande, ce sont eux les suspects
    immédiats, et le modèle per-slot ne les voit pas."""
    geo = GeometriePyramide(n_fov=512, n_niv=10, slots=slots_v2())
    postes = postes_residuels_bras_a(geo)
    assert postes["appels_pas_f_par_frame"] == 9
    assert postes["lancements_kernel_par_frame"] == 18
    assert postes["reductions_cfl_synchrones_par_frame"] == 9
    assert postes["reference_m_a_prime"]["reductions_cfl_synchrones"] == 1


def test_ancrage_grave_non_rejoue():
    """Les chiffres de M-a-ter sont figés ici, pas recalculés — un
    glissement se verrait."""
    assert V2_FULL_MEDIANE_MS == 31.097
    assert V2_CALCUL_MS == 30.484
    assert V2_TRANSFERTS_MS == 0.613
    assert V2_FULL_MEDIANE_MS - V2_TRANSFERTS_MS == pytest.approx(
        V2_CALCUL_MS, abs=0.001)


# ----- plomberie GPU : les deux bras tournent vraiment -----

@pytest.mark.skipif(not cupy_disponible(),
                    reason="cupy/device CUDA indisponible")
@pytest.mark.parametrize(
    "delta_x, remontee", [(DELTA_X_IMMOBILE, False), (DELTA_X_MOBILE, False)])
def test_bras_tourne_sur_gpu_avec_kernel_fusionne(delta_x, remontee):
    """Plomberie (taille minuscule, JAMAIS une mesure) : chaque bras
    traverse réellement le kernel fusionné sur une pyramide à c mixte."""
    import cupy as cp

    from scripts.run_f1_ma_ter import ApplicateurFusionne
    from src.f1_gpu.pyramide import Slot
    slots = (Slot(niveau=1, c=4, role="fovea"),
             Slot(niveau=2, c=8, role="fovea"),
             Slot(niveau=3, c=8, role="fovea"),
             Slot(niveau=3, c=8, role="energie"))
    geo = GeometriePyramide(n_fov=8, n_niv=4, n0=16, slots=slots)
    transferts = TransfertComptable(cp)
    pyramide = PyramideFovea(cp, geo, transferts,
                             pas_f=ApplicateurFusionne(),
                             remontee_active=remontee)
    transferts.frame_suivante()
    for _ in range(3):
        pyramide.frame(delta_x=delta_x)
        bilan = transferts.frame_suivante()
    assert bilan["d2h_octets"] == 0            # remontée OFF dans les deux
    for j in geo.niveaux_gpu:
        assert bool(cp.isfinite(pyramide.fenetres[j]).all())
