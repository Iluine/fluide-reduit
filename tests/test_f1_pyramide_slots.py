"""Tests F1 — configuration par LISTE DE SLOTS (§A16, re-épinglage du
quadruplet §6, pocCascade2phys 38a2739). Backend numpy (B1), tailles
minuscules (VM). Les tests de la géométrie γ₂ dense vivent dans
`test_f1_pyramide.py` (non-régression : le défaut ne bouge pas)."""
import numpy as np
import pytest

from src.f1_gpu.pyramide import (
    GAMMA2,
    GeometriePyramide,
    PyramideFovea,
    Slot,
    slots_uniformes,
)
from src.f1_gpu.transferts import TransfertComptable

N_FOV = 8
N0 = 16
N_NIV = 4          # niveau 0 CPU + niveaux GPU 1, 2, 3

# Config hétérogène de test (le motif V2 en réduction) : niveau fin 3 à
# c=8 avec une fenêtre d'énergie, niveau 2 à c=8, niveau 1 à c=4.
SLOTS_HETEROGENES = (
    Slot(niveau=1, c=4, role="fovea"),
    Slot(niveau=2, c=8, role="fovea"),
    Slot(niveau=3, c=8, role="fovea"),
    Slot(niveau=3, c=8, role="energie"),
)


def _geo(slots=SLOTS_HETEROGENES) -> GeometriePyramide:
    return GeometriePyramide(n_fov=N_FOV, n_niv=N_NIV, n0=N0, slots=slots)


def _pyramide(geo: GeometriePyramide, eps_detail: float = 1e30):
    transferts = TransfertComptable(np)
    pyramide = PyramideFovea(np, geo, transferts, eps_detail=eps_detail)
    transferts.frame_suivante()  # clôt la descente initiale (B4)
    return pyramide, transferts


# ----- le défaut ne bouge pas (rétro-compatibilité M-a/M-c) -----

def test_defaut_est_lenveloppe_dense_gamma2():
    """`slots` non fourni => γ₂ slots c=8 par niveau GPU : la géométrie
    tranche-1 au bit près (M-a/M-c inchangés)."""
    geo = GeometriePyramide(n_fov=N_FOV, n_niv=N_NIV, n0=N0)
    assert geo.slots == slots_uniformes(N_NIV, GAMMA2, c=8)
    for j in geo.niveaux_gpu:
        assert geo.n_slots(j) == GAMMA2
        assert geo.c_du_niveau(j) == 8
        assert geo.n_systemes_du_niveau(j) == 2
    assert geo.blocs_actifs_gpu == GAMMA2 * 2 * 3
    assert geo.cellules_actives_gpu == GAMMA2 * N_FOV * N_FOV * 3


# ----- comptes (cellules, blocs, slots) -----

def test_comptes_de_cellules_blocs_et_slots():
    """Cellules = slots · n_fov² (INDÉPENDANT de c) ; blocs = Σ systèmes
    (c'est LUI qui est linéaire dans le coût prédit, §A16)."""
    geo = _geo()
    assert geo.n_slots_total == 4
    assert geo.cellules_actives_gpu == 4 * N_FOV * N_FOV
    # 1 slot c=4 (1 bloc) + 3 slots c=8 (2 blocs chacun) = 7 blocs.
    assert geo.blocs_actifs_gpu == 7


def test_c_par_niveau_et_systemes():
    geo = _geo()
    assert (geo.c_du_niveau(1), geo.n_systemes_du_niveau(1)) == (4, 1)
    assert (geo.c_du_niveau(2), geo.n_systemes_du_niveau(2)) == (8, 2)
    assert (geo.c_du_niveau(3), geo.n_systemes_du_niveau(3)) == (8, 2)
    assert geo.n_slots(3) == 2 and geo.n_slots(1) == 1


def test_resume_slots_par_role():
    resume = _geo().resume_slots()
    assert resume["par_role"]["fovea"] == {"n_slots": 3, "n_blocs": 5}
    assert resume["par_role"]["energie"] == {"n_slots": 1, "n_blocs": 2}
    assert resume["par_niveau"]["3"]["roles"] == ["fovea", "energie"]


# ----- géométrie B3 : une origine par slot, offsets inchangés -----

def test_origines_une_par_slot_offsets_b3():
    """Le slot d'indice 0 prend l'offset B3 nul (fovéal) ; le suivant
    +n_fov (énergie). La fovéa mobile E4c est INCHANGÉE."""
    geo = _geo()
    centre = geo.centre_fin_initial()
    assert len(geo.origines(1, centre)) == 1        # 1 slot au niveau 1
    origines_3 = geo.origines(3, centre)
    assert len(origines_3) == 2                     # 2 slots au niveau 3
    (oy_fov, ox_fov), (oy_ener, ox_ener) = origines_3
    assert ox_ener == ox_fov                        # x commun (lockstep)
    assert oy_ener - oy_fov == N_FOV                # offset B3 +n_fov


def test_lockstep_conserve_avec_slots():
    """Géométrie 4e0e719 conservée : les fenêtres d'un niveau partagent le
    x et avancent ensemble avec le balayage."""
    geo = _geo()
    c0 = geo.centre_fin_initial()
    init, apres = geo.origines(3, c0), geo.origines(3, c0 + 7)
    assert len({ox for _, ox in apres}) == 1
    assert apres[0][1] > init[0][1]
    for avant, maintenant in zip(init, apres):
        assert maintenant[0] == avant[0]            # offsets y inchangés
        assert maintenant[1] - avant[1] == apres[0][1] - init[0][1]


# ----- gardes fail-loud -----

def test_garde_c_hors_valeurs_gravees():
    with pytest.raises(ValueError, match="c=6"):
        Slot(niveau=1, c=6)


def test_garde_role_inconnu():
    with pytest.raises(ValueError, match="role"):
        Slot(niveau=1, c=8, role="halo")


def test_garde_slot_hors_niveaux_gpu():
    """Le niveau 0 est CPU (§5) : aucun slot ne peut s'y placer."""
    with pytest.raises(ValueError, match="hors"):
        GeometriePyramide(n_fov=N_FOV, n_niv=N_NIV, n0=N0,
                          slots=(Slot(niveau=0, c=8),))


def test_garde_niveau_gpu_sans_slot():
    """Une pyramide trouée n'est pas mesurable."""
    with pytest.raises(ValueError, match="sans aucun slot"):
        GeometriePyramide(n_fov=N_FOV, n_niv=N_NIV, n0=N0,
                          slots=(Slot(niveau=1, c=8), Slot(niveau=2, c=8)))


def test_garde_trop_de_slots_pour_les_offsets_b3():
    """B3 ne définit que 3 offsets (0, ±n_fov) — un 4e serait une
    géométrie NEUVE, non mesurée."""
    slots = tuple(Slot(niveau=j, c=8, role="fovea" if k == 0 else "energie")
                  for j in (1, 2, 3) for k in range(4 if j == 3 else 1))
    with pytest.raises(ValueError, match="offsets B3"):
        GeometriePyramide(n_fov=N_FOV, n_niv=N_NIV, n0=N0, slots=slots)


def test_garde_c_heterogene_dans_un_niveau():
    """Les fenêtres d'un niveau sont batchées dans UN buffer : c doit y
    être uniforme (sinon la shape n'est pas rectangulaire)."""
    slots = (Slot(niveau=1, c=8), Slot(niveau=2, c=8),
             Slot(niveau=3, c=8),
             Slot(niveau=3, c=4, role="energie"))
    with pytest.raises(ValueError, match="hétérogène"):
        GeometriePyramide(n_fov=N_FOV, n_niv=N_NIV, n0=N0, slots=slots)


def test_garde_slot_zero_doit_etre_foveal():
    """L'offset B3 nul est la fenêtre fovéale — l'ordre est load-bearing."""
    slots = (Slot(niveau=1, c=8), Slot(niveau=2, c=8),
             Slot(niveau=3, c=8, role="energie"),
             Slot(niveau=3, c=8, role="fovea"))
    with pytest.raises(ValueError, match="FOVÉALE"):
        GeometriePyramide(n_fov=N_FOV, n_niv=N_NIV, n0=N0, slots=slots)


# ----- buffers et frame sous configuration hétérogène -----

def test_shapes_des_buffers_par_niveau():
    """(n_slots(j), n_systemes(j), 4, n_fov, n_fov) — le c du niveau pilote
    la 2e dimension."""
    geo = _geo()
    pyramide, _ = _pyramide(geo)
    assert pyramide.fenetres[1].shape == (1, 1, 4, N_FOV, N_FOV)
    assert pyramide.fenetres[2].shape == (1, 2, 4, N_FOV, N_FOV)
    assert pyramide.fenetres[3].shape == (2, 2, 4, N_FOV, N_FOV)
    for j in geo.niveaux_gpu:
        assert pyramide.references[j].shape == pyramide.fenetres[j].shape


def test_descente_initiale_compte_les_octets_par_c():
    """La descente initiale descend n_slots·n_sys·4·n_fov² f32 par niveau
    — un niveau c=4 descend MOITIÉ moins qu'un c=8 à slots égaux."""
    geo = _geo()
    _, transferts = _pyramide(geo)
    octets_bloc = 4 * N_FOV * N_FOV * 4       # 4 champs f32 d'un système
    attendu = (1 * 1 + 1 * 2 + 2 * 2) * octets_bloc
    assert transferts.bilans[0]["h2d_octets"] == attendu


def test_frame_tourne_sous_configuration_heterogene():
    """Le F jetable accepte 1 ou 2 systèmes : la frame complète (descente,
    F, remontée) tourne sur une pyramide à c mixte."""
    geo = _geo()
    pyramide, transferts = _pyramide(geo, eps_detail=1e-4)
    identites = {j: id(pyramide.fenetres[j]) for j in geo.niveaux_gpu}
    for _ in range(3):
        diag = pyramide.frame(delta_x=1)
        transferts.frame_suivante()
    assert set(diag["nnz_par_niveau"]) == {"1", "2", "3"}
    assert all(np.isfinite(pyramide.fenetres[j]).all()
               for j in geo.niveaux_gpu)
    for j in geo.niveaux_gpu:                  # B2 : zéro réallocation
        assert id(pyramide.fenetres[j]) == identites[j]


def test_octets_etat_et_dense_equivalent_suivent_les_blocs():
    geo = _geo()
    pyramide, _ = _pyramide(geo)
    octets_bloc = 4 * N_FOV * N_FOV * 4
    # fenêtres + références = 2 buffers, 7 blocs au total.
    assert pyramide.octets_etat() == 2 * geo.blocs_actifs_gpu * octets_bloc
    assert (pyramide.octets_dense_equivalent_par_frame()
            == geo.blocs_actifs_gpu * octets_bloc)
