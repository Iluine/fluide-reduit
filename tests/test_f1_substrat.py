"""Tests F1 tranche-1 — F jetable (B5/E4a, `src/f1_gpu/substrat_jetable.py`).

Backend numpy (B1) : on teste la LOGIQUE du motif de coût — shapes/dtype,
équilibre au repos (hérité du schéma réel), indépendance des 2 systèmes
(E4a), plancher sec. Aucune mesure ici (VM)."""
import numpy as np

from src.f1_gpu.substrat_jetable import (
    DRY_EPS,
    etat_initial_jetable,
    pas_f_jetable,
    reduction_cfl,
)


def _etat_repos(n_fenetres: int, n: int) -> np.ndarray:
    """(B, 2, 4, n, n) au repos : h = 1, hu = hv = 0, s = 0.2 uniforme."""
    q = np.zeros((n_fenetres, 2, 4, n, n), dtype=np.float32)
    q[:, :, 0] = 1.0
    q[:, :, 3] = 0.2
    return q


def test_etat_initial_shape_dtype():
    q = etat_initial_jetable(3, 16, graine=7)
    assert q.shape == (3, 2, 4, 16, 16)
    assert q.dtype == np.float32
    assert np.all(np.isfinite(q))
    assert np.all(q[:, :, 0] > 0.5)  # h autour de 1, jamais sec


def test_pas_preserve_shape_dtype_et_finitude():
    q = etat_initial_jetable(2, 16, graine=3)
    sortie, dt_cfl = pas_f_jetable(q, np)
    assert sortie.shape == q.shape
    assert sortie.dtype == np.float32
    assert np.all(np.isfinite(sortie))
    assert dt_cfl > 0.0


def test_repos_bien_equilibre():
    """η constante, u = 0 : le motif hérite la C-property du schéma réel —
    l'état au repos reste au repos (tolérance f32)."""
    q = _etat_repos(2, 12)
    sortie, _ = pas_f_jetable(q, np)
    np.testing.assert_allclose(sortie, q, atol=1e-6)


def test_systemes_independants():
    """E4a : les 2 systèmes (h, hu, hv, s) sont sous stencil complet mais
    INDÉPENDANTS — perturber le système 0 ne change pas un bit du
    système 1 (arithmétique CPU déterministe : comparaison exacte)."""
    q_mixte = _etat_repos(1, 12)
    q_mixte[:, 0, 0] += 0.1 * np.random.default_rng(5).random(
        (1, 12, 12)).astype(np.float32)
    q_repos = _etat_repos(1, 12)
    sortie_mixte, _ = pas_f_jetable(q_mixte, np)
    sortie_repos, _ = pas_f_jetable(q_repos, np)
    assert np.array_equal(sortie_mixte[:, 1], sortie_repos[:, 1])
    assert not np.array_equal(sortie_mixte[:, 0], sortie_repos[:, 0])


def test_plancher_sec_zero_les_quatre_champs():
    q = _etat_repos(1, 8)
    q[:, :, 0, :4, :] = DRY_EPS / 2  # moitié haute sous le seuil sec
    q[:, :, 1, :4, :] = 0.3
    q[:, :, 3, :4, :] = 0.5
    sortie, _ = pas_f_jetable(q, np)
    sec = sortie[:, :, 0, :2, :] <= DRY_EPS  # cœur de la zone sèche
    assert np.all(sortie[:, :, 1, :2, :][sec] == 0.0)
    assert np.all(sortie[:, :, 3, :2, :][sec] == 0.0)


def test_sortie_in_place_autorisee():
    """B2/E4b : `sortie is q` (écriture dans le tampon d'état) donne le
    même résultat que la sortie fraîche."""
    q = etat_initial_jetable(1, 10, graine=11)
    attendu, _ = pas_f_jetable(q, np)
    meme, _ = pas_f_jetable(q, np, sortie=q)
    assert meme is q
    assert np.array_equal(meme, attendu)


def test_reduction_cfl_positive_et_bornee():
    q = _etat_repos(1, 8)
    dt = reduction_cfl(q, np)
    # h = 1, u = 0 -> smax = sqrt(g) ~ 3.13 ; dt = 0.4 / smax ~ 0.128.
    assert 0.05 < dt < 0.5
