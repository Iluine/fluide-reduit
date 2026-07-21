"""Tests F1 — M-b tranche-2 : le BRAS PRODUCTION (§A31-build).

Ce que ces tests protègent :
  1. la fovéa mobile E4c : ~1 cellule par épisode, ~24 sur les 6 émissions
     (~un tiers du domaine) — c'est la portée déjà gravée dans la lecture ;
  2. la composition 2-niveaux : c'est ELLE la source (c), et elle doit être
     exactement ce qu'elle prétend (fovéa fine dedans, grossier upsamplé
     dehors) ;
  3. le CANAL option 1 : la connaissance du CPU est la `reference` mise à
     jour par la remontée SEUILLÉE — jamais une copie qui puisse diverger.
     Les deux bouts sont vérifiés : seuil infini ⇒ CPU muet ; seuil nul et
     budget large ⇒ CPU exact ; budget saturé ⇒ seuls les plus gros |Δ|
     passent (c'est là que le cap 10 % mord) ;
  4. l'invariant liant : rien de neuf dans `pas_f_fidele`, la production
     ORCHESTRE le kernel figé de C1 via `pas_deux_niveaux`."""
import numpy as np
import pytest

from config import GridConfig
from src.f1_gpu.backend import cupy_disponible
from src.f1_gpu.cellule_mb import N_EPISODES, budget_k_fen, centres_cellule
from src.f1_gpu.production_fidele import (
    EPS_PRODUCTION,
    N_FOVEA,
    PAS_FOVEA,
    composer_plein,
    evoluer_histoire_production,
    position_fovea,
    remonter_seuillee,
)
from src.sediment import SedimentParams, default_terrain

gpu_requis = pytest.mark.skipif(
    not cupy_disponible(), reason="cupy/device CUDA indisponible")

N = 64


def _terrain():
    return default_terrain(GridConfig()).astype(np.float32)


# ----- 1. la fovéa mobile E4c -----

def test_fovea_avance_d_une_cellule_par_episode():
    assert PAS_FOVEA == 1
    oy0, ox0 = position_fovea(0)
    oy1, ox1 = position_fovea(1)
    assert oy1 == oy0 and ox1 == ox0 + 1


def test_fovea_parcourt_un_tiers_du_domaine_sur_la_cellule():
    """La portée gravée dans la lecture dit ~24 cellules, ~un tiers de 64.
    Le code doit la RENDRE VRAIE, sinon la portée ment."""
    _, ox_debut = position_fovea(0)
    _, ox_fin = position_fovea(N_EPISODES - 1)
    parcours = ox_fin - ox_debut
    assert parcours == N_EPISODES - 1                 # ~24 cellules
    assert 0.25 * N <= parcours <= 0.42 * N           # ~un tiers


def test_fovea_reste_dans_le_domaine_sur_toute_la_cellule():
    for ep in range(N_EPISODES):
        oy, ox = position_fovea(ep)
        assert 0 <= oy <= N - N_FOVEA
        assert 0 <= ox <= N - N_FOVEA


# ----- 2. la composition 2-niveaux = la source (c) -----

@gpu_requis
def test_composer_fovea_fine_dedans_grossier_upsample_dehors():
    import cupy as cp
    from src.f1_gpu.fovea_2niveaux import decimer
    rng = np.random.default_rng(4)
    q = cp.asarray(rng.random((1, 1, 3, N, N)).astype(np.float32))
    qc = decimer(q, cp)
    oy, ox = 16, 8
    qf = cp.ascontiguousarray(q[:, :, :, oy:oy + N_FOVEA, ox:ox + N_FOVEA])
    plein = composer_plein(qc, qf, oy, ox, cp)
    assert plein.shape == (1, 1, 3, N, N)
    # dedans : la fovéa fine, à l'identique
    assert float(cp.max(cp.abs(
        plein[:, :, :, oy:oy + N_FOVEA, ox:ox + N_FOVEA] - qf))) == 0.0
    # dehors : le grossier upsamplé (coin haut-gauche, hors fovéa)
    assert float(cp.abs(plein[0, 0, 0, 0, 0] - qc[0, 0, 0, 0, 0])) == 0.0


# ----- 3. le canal option 1 : seuil, budget, cap -----

def test_canal_muet_si_le_seuil_est_infini():
    """Seuil infini ⇒ rien n'est transporté ⇒ la connaissance ne bouge pas."""
    champ = np.full((8, 8), 0.5, dtype=np.float32)
    ref = np.zeros((8, 8), dtype=np.float32)
    n = remonter_seuillee(champ, ref, eps=1e9, budget=64)
    assert n == 0
    assert float(np.max(np.abs(ref))) == 0.0


def test_canal_exact_si_seuil_nul_et_budget_large():
    """Seuil nul + budget large ⇒ la connaissance EST le champ. C'est la
    borne haute du canal : sans perte, option 1 est exacte."""
    rng = np.random.default_rng(5)
    champ = rng.random((8, 8)).astype(np.float32)
    ref = np.zeros((8, 8), dtype=np.float32)
    n = remonter_seuillee(champ, ref, eps=0.0, budget=64)
    assert n == 64
    np.testing.assert_array_equal(ref, champ)


def test_le_cap_ne_laisse_passer_que_les_plus_gros_ecarts():
    """Budget saturé ⇒ seuls les plus gros |Δ| passent, et la connaissance
    n'est mise à jour QUE là. C'est là que le cap 10 % mord."""
    champ = np.array([[0.1, 0.9], [0.5, 0.2]], dtype=np.float32)
    ref = np.zeros((2, 2), dtype=np.float32)
    n = remonter_seuillee(champ, ref, eps=0.0, budget=1)
    assert n == 1
    assert ref[0, 1] == pytest.approx(0.9)      # le plus gros seul
    assert float(ref[0, 0]) == 0.0 and float(ref[1, 1]) == 0.0


def test_remontee_incrementale_accumule():
    """La remontée est INCRÉMENTALE (schéma B4) : elle transporte l'écart au
    connu, pas le champ — deux passes convergent."""
    rng = np.random.default_rng(6)
    champ = rng.random((8, 8)).astype(np.float32)
    ref = np.zeros((8, 8), dtype=np.float32)
    remonter_seuillee(champ, ref, eps=0.0, budget=32)
    remonter_seuillee(champ, ref, eps=0.0, budget=32)
    np.testing.assert_allclose(ref, champ, rtol=0, atol=1e-7)


def test_eps_production_est_celui_grave():
    assert EPS_PRODUCTION == 1e-2


def test_budget_de_la_cellule_est_le_cap_dix_pourcent():
    assert budget_k_fen(N * N) == int(0.10 * N * N)


# ----- 4. l'histoire production, et l'invariant liant -----

@gpu_requis
def test_histoire_production_rend_la_connaissance_aux_emissions():
    import cupy as cp
    b0 = _terrain()
    params = SedimentParams(N_settle=20)
    centres = centres_cellule(101, 4)
    out = evoluer_histoire_production(101, 4, b0, cp, centres, params,
                                      checkpoints={2, 4})
    assert set(out) == {2, 4}
    for k in out:
        assert out[k].shape == (N, N)
        assert np.isfinite(out[k]).all()


@gpu_requis
def test_connaissance_muette_si_le_seuil_est_infini():
    """Bout à bout : seuil infini ⇒ le CPU ne sait RIEN, même si le live a
    évolué. Prouve que la connaissance rendue est bien la `reference` du
    canal (option 1), pas le champ live recopié."""
    import cupy as cp
    b0 = _terrain()
    params = SedimentParams(N_settle=20)
    out = evoluer_histoire_production(101, 2, b0, cp, centres_cellule(101, 2),
                                      params, checkpoints={2}, eps=1e9)
    assert float(np.max(np.abs(out[2]))) == 0.0


def test_invariant_liant_production():
    """Rien de neuf dans `pas_f_fidele` : la production ORCHESTRE le kernel
    figé de C1 via `pas_deux_niveaux` (identité, pas diff)."""
    import src.f1_gpu.production_fidele as prod
    from src.f1_gpu.fovea_2niveaux import pas_deux_niveaux
    from src.f1_gpu.substrat_fidele import pas_f_fidele
    from src.f1_gpu.temoin_fidele import pas_f_fidele as pas_temoin
    assert prod.pas_deux_niveaux is pas_deux_niveaux
    assert pas_temoin is pas_f_fidele        # les deux bras, le MÊME F
