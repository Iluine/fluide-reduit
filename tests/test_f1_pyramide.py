"""Tests F1 tranche-1 — pyramide + fovéa mobile + schéma diff (B2/B3/B4,
`src/f1_gpu/pyramide.py`). Backend numpy (B1), tailles minuscules (VM)."""
import numpy as np
import pytest

from src.f1_gpu.pyramide import GeometriePyramide, PyramideFovea
from src.f1_gpu.transferts import TransfertComptable

# Géométrie de test : n0=16, n_fov=8, 3 niveaux (0 CPU + 2 GPU), γ₂=3.
GEO = GeometriePyramide(n_fov=8, n_niv=3, n0=16)
OCTETS_FENETRE_NIVEAU = 3 * 2 * 4 * 8 * 8 * 4   # bloc (3,2,4,8,8) f32
# Géométrie §A15-complément-3 (« les 3 fenêtres bougent ») : la descente
# de régime est la colonne des TROIS fenêtres du niveau (lockstep).
OCTETS_COLONNE_NIVEAU = 3 * 2 * 4 * 8 * 1 * 4   # bloc (3,2,4,8,1) f32


def _pyramide(eps_detail: float):
    transferts = TransfertComptable(np)
    pyramide = PyramideFovea(np, GEO, transferts, eps_detail=eps_detail)
    transferts.frame_suivante()  # clôt la descente initiale (B4)
    return pyramide, transferts


def test_geometrie_niveaux_et_cellules():
    assert list(GEO.niveaux_gpu) == [1, 2]
    assert GEO.niveau_fin == 2
    assert GEO.cellules_actives_gpu == 3 * 8 * 8 * 2
    assert GEO.cote_monde(2) == 64


def test_origines_alignement_dyadique():
    """B3 : l'origine du niveau j avance d'1 cellule toutes les 2^(J−j)
    frames du balayage E4c."""
    centre = GEO.centre_fin_initial()
    assert centre == 32
    assert GEO.origines(2, centre)[0] == (28, 28)
    assert GEO.origines(1, centre)[0] == (12, 12)
    # +1 cellule fine : le niveau fin bouge, le niveau 1 non.
    assert GEO.origines(2, centre + 1)[0][1] == 29
    assert GEO.origines(1, centre + 1)[0][1] == 12
    assert GEO.origines(1, centre + 2)[0][1] == 13


def test_fenetres_lockstep_avec_le_balayage():
    """Géométrie rétablie §A15-complément-3 (décision Romain : « les 3
    fenêtres bougent ») : les γ₂ fenêtres d'un niveau partagent le même x
    et translatent en LOCKSTEP avec le balayage — majorant de cadence."""
    c0 = GEO.centre_fin_initial()
    for j in GEO.niveaux_gpu:
        init = GEO.origines(j, c0)
        apres = GEO.origines(j, c0 + 7)
        assert len({ox for _, ox in apres}) == 1     # x commun aux 3
        assert apres[0][1] > init[0][1]              # ... qui a avancé
        for avant, maintenant in zip(init, apres):
            assert maintenant[0] == avant[0]         # offsets y inchangés
            assert maintenant[1] - avant[1] == apres[0][1] - init[0][1]


def test_n_niv_minimal_fail_loud():
    with pytest.raises(ValueError):
        GeometriePyramide(n_fov=8, n_niv=1, n0=16)


def test_descente_initiale_comptee_et_reference_egale():
    pyramide, transferts = _pyramide(eps_detail=1e30)
    assert transferts.bilans[0]["h2d_octets"] == 2 * OCTETS_FENETRE_NIVEAU
    for j in GEO.niveaux_gpu:
        assert np.array_equal(pyramide.fenetres[j], pyramide.references[j])


def test_frame_descente_colonnes_entrantes_seules():
    """B4 : à 1 cellule fine/frame, seule la colonne entrante des niveaux
    DÉPLACÉS descend (frame 1 : niveau fin seul ; frame 2 : les deux)."""
    pyramide, transferts = _pyramide(eps_detail=1e30)
    diag = pyramide.frame(delta_x=1)
    transferts.frame_suivante()
    assert diag["niveaux_deplaces"] == [2]
    assert transferts.bilans[-1]["h2d_octets"] == OCTETS_COLONNE_NIVEAU
    diag = pyramide.frame(delta_x=1)
    transferts.frame_suivante()
    assert diag["niveaux_deplaces"] == [1, 2]
    assert transferts.bilans[-1]["h2d_octets"] == 2 * OCTETS_COLONNE_NIVEAU


def test_remontee_seuil_infini_vide_et_seuil_zero_dense():
    """B4/E4d : seuil infini -> 0 octet de valeurs remontées ; seuil 0 ->
    remontée DENSE = valeurs f32 + indices u32 de toutes les fenêtres."""
    pyramide, transferts = _pyramide(eps_detail=1e30)
    diag = pyramide.frame(delta_x=1)
    transferts.frame_suivante()
    assert all(nnz == 0 for nnz in diag["nnz_par_niveau"].values())
    assert transferts.bilans[-1]["d2h_octets"] == 0

    pyramide, transferts = _pyramide(eps_detail=0.0)
    diag = pyramide.frame(delta_x=1)
    transferts.frame_suivante()
    n_coeffs = 3 * 2 * 4 * 8 * 8
    assert all(nnz == n_coeffs for nnz in diag["nnz_par_niveau"].values())
    assert transferts.bilans[-1]["d2h_octets"] == 2 * n_coeffs * (4 + 4)


def test_reference_incrementale_apres_remontee():
    """B4 : après remontée à seuil 0 (tout remonte), la référence égale la
    fenêtre — le CPU « sait » exactement ce qui est remonté."""
    pyramide, _ = _pyramide(eps_detail=0.0)
    pyramide.frame(delta_x=1)
    for j in GEO.niveaux_gpu:
        np.testing.assert_allclose(
            pyramide.references[j], pyramide.fenetres[j], atol=1e-6)


def test_zero_realloc_des_buffers_etat():
    """B2/E4b : les buffers d'état gardent leur identité à travers les
    frames (aucune réallocation d'état)."""
    pyramide, _ = _pyramide(eps_detail=1e-4)
    identites = {j: id(pyramide.fenetres[j]) for j in GEO.niveaux_gpu}
    for _ in range(5):
        pyramide.frame(delta_x=1)
    for j in GEO.niveaux_gpu:
        assert id(pyramide.fenetres[j]) == identites[j]


def test_saut_de_fenetre_re_prediction_pleine():
    """E4c (diagnostic non-verdictal) : un saut >= n_fov re-prédit la
    fenêtre ENTIÈRE des niveaux qui sautent."""
    pyramide, transferts = _pyramide(eps_detail=1e30)
    diag = pyramide.frame(delta_x=GEO.n_fov)
    transferts.frame_suivante()
    assert 2 in diag["niveaux_deplaces"]
    # Niveau fin : saut plein des 3 fenêtres (8 >= n_fov) ; niveau 1 :
    # dx = 4 colonnes du niveau (lockstep §A15-complément-3).
    attendu = OCTETS_FENETRE_NIVEAU + 4 * OCTETS_COLONNE_NIVEAU
    assert transferts.bilans[-1]["h2d_octets"] == attendu


def test_recul_interdit():
    pyramide, _ = _pyramide(eps_detail=1e30)
    with pytest.raises(ValueError):
        pyramide.frame(delta_x=-1)


def test_octets_etat_et_dense_equivalent():
    pyramide, _ = _pyramide(eps_detail=1e-4)
    # fenêtres + références : 2 buffers x 2 niveaux x bloc f32.
    assert pyramide.octets_etat() == 2 * 2 * OCTETS_FENETRE_NIVEAU
    assert (pyramide.octets_dense_equivalent_par_frame()
            == 2 * OCTETS_FENETRE_NIVEAU)
