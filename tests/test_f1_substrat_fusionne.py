"""Tests F1 — M-a′ : équivalence de motif jetable ↔ fusionné
(`src/f1_gpu/substrat_fusionne.py`, §A15-lecture-M-a).

Le TEST-CONTRAT de M-a′ : le fusionné RÉORDONNE le même motif E4a, il
n'omet rien — sur une même petite fenêtre, `pas_f_fusionne` doit
reproduire `pas_f_jetable` à la tolérance f32 NOMMÉE (TOL_EQUIVALENCE_*).
GPU requis (RawKernel) — sauté proprement sinon. Tailles minuscules :
plomberie/équivalence, JAMAIS une mesure."""
import numpy as np
import pytest

from src.f1_gpu.backend import cupy_disponible, vers_cpu
from src.f1_gpu.substrat_jetable import etat_initial_jetable, pas_f_jetable

pytestmark = pytest.mark.skipif(
    not cupy_disponible(), reason="cupy/device CUDA indisponible")


@pytest.fixture(scope="module")
def cp():
    import cupy
    return cupy


def _fusionne(cp, q_cpu: np.ndarray) -> tuple:
    from src.f1_gpu.substrat_fusionne import pas_f_fusionne
    sortie, dt_cfl = pas_f_fusionne(cp.asarray(q_cpu), cp)
    return vers_cpu(sortie), dt_cfl


def test_equivalence_motif_un_pas(cp):
    """Un pas : fusionné == jetable (référence CPU f32 déterministe) à
    TOL_EQUIVALENCE — le réordonnancement f32 est le seul écart admis."""
    from src.f1_gpu.substrat_fusionne import (
        TOL_EQUIVALENCE_ATOL,
        TOL_EQUIVALENCE_RTOL,
    )
    q = etat_initial_jetable(2, 12, graine=3)
    attendu, dt_ref = pas_f_jetable(q, np)
    obtenu, dt_fus = _fusionne(cp, q)
    np.testing.assert_allclose(obtenu, attendu,
                               rtol=TOL_EQUIVALENCE_RTOL,
                               atol=TOL_EQUIVALENCE_ATOL)
    assert dt_fus == pytest.approx(dt_ref, rel=1e-5)


def _etat_dense(n_fenetres: int, n: int, graine: int) -> np.ndarray:
    """Champ DENSE (review 29/07, M9) : gradients non nuls PARTOUT, hu/hv non
    nuls. `etat_initial_jetable` construit par `np.kron` des blocs constants
    6×6 à vitesses nulles : `minmod` y vaut 0 au premier pas et la
    reconstruction MUSCL — la partie délicate du portage — contribuait ZÉRO
    au test-contrat qui porte l'ancre M-a′. C'est structurellement le champ
    pauvre que `run_p2_chiffrage.py` démasque en toutes lettres ; ici on le
    refuse aussi. Sinusoïdes déphasées par système + petit cisaillement :
    lisse (stable sur 3 pas), mais AUCUN voisinage constant."""
    rng = np.random.default_rng(graine)
    yy, xx = (a.astype(np.float64) / n for a in np.mgrid[0:n, 0:n])
    q = np.zeros((n_fenetres, 2, 4, n, n), dtype=np.float32)
    for f in range(n_fenetres):
        for s in range(2):
            ph = rng.uniform(0.0, 2.0 * np.pi, size=6)
            h = (1.0 + 0.10 * np.sin(2.0 * np.pi * yy + ph[0])
                 * np.cos(2.0 * np.pi * xx + ph[1])
                 + 0.05 * np.sin(4.0 * np.pi * (yy + xx) + ph[2]))
            q[f, s, 0] = h
            q[f, s, 1] = h * 0.05 * np.sin(2.0 * np.pi * xx + ph[3])
            q[f, s, 2] = h * 0.04 * np.cos(2.0 * np.pi * yy + ph[4])
            q[f, s, 3] = 0.05 * (1.2 + np.sin(2.0 * np.pi * yy + ph[5]))
    # Le champ EST dense — aucune première différence nulle sur h : si ce
    # garde-fou casse, le test re-teste un champ à blocs sans le dire.
    assert not np.any(np.diff(q[:, :, 0], axis=-1) == 0.0)
    assert not np.any(np.diff(q[:, :, 0], axis=-2) == 0.0)
    return q


def test_equivalence_motif_un_pas_champ_dense(cp):
    """M9 : le contrat M-a′ rejoué là où MUSCL travaille — pentes non nulles
    partout, vitesses non nulles. C'est CE test qui éprouve la partie
    délicate du portage ; la variante kron ci-dessus reste comme cas
    dégénéré (C-property des blocs constants)."""
    from src.f1_gpu.substrat_fusionne import (
        TOL_EQUIVALENCE_ATOL,
        TOL_EQUIVALENCE_RTOL,
    )
    q = _etat_dense(2, 12, graine=3)
    attendu, dt_ref = pas_f_jetable(q, np)
    obtenu, dt_fus = _fusionne(cp, q)
    assert not np.array_equal(obtenu, q[:, :, :]), "le pas doit faire du travail"
    np.testing.assert_allclose(obtenu, attendu,
                               rtol=TOL_EQUIVALENCE_RTOL,
                               atol=TOL_EQUIVALENCE_ATOL)
    assert dt_fus == pytest.approx(dt_ref, rel=1e-5)


def test_equivalence_motif_pas_composes_champ_dense(cp):
    """M9, composé : trois pas sur champ dense — l'écart de
    réordonnancement ne diverge pas là où les limiteurs mordent."""
    from src.f1_gpu.substrat_fusionne import (
        TOL_EQUIVALENCE_ATOL,
        TOL_EQUIVALENCE_RTOL,
        pas_f_fusionne,
    )
    q_cpu = _etat_dense(1, 10, graine=11)
    q_gpu = cp.asarray(q_cpu)
    for _ in range(3):
        q_cpu, _ = pas_f_jetable(q_cpu, np)
        q_gpu, _ = pas_f_fusionne(q_gpu, cp)
    np.testing.assert_allclose(vers_cpu(q_gpu), q_cpu,
                               rtol=10 * TOL_EQUIVALENCE_RTOL,
                               atol=10 * TOL_EQUIVALENCE_ATOL)


def test_equivalence_motif_pas_composes(cp):
    """Trois pas composés : l'écart de réordonnancement ne diverge pas
    (tolérance x10 sur la composition, nommée ici)."""
    from src.f1_gpu.substrat_fusionne import (
        TOL_EQUIVALENCE_ATOL,
        TOL_EQUIVALENCE_RTOL,
        pas_f_fusionne,
    )
    q_cpu = etat_initial_jetable(1, 10, graine=11)
    q_gpu = cp.asarray(q_cpu)
    for _ in range(3):
        q_cpu, _ = pas_f_jetable(q_cpu, np)
        q_gpu, _ = pas_f_fusionne(q_gpu, cp)
    np.testing.assert_allclose(vers_cpu(q_gpu), q_cpu,
                               rtol=10 * TOL_EQUIVALENCE_RTOL,
                               atol=10 * TOL_EQUIVALENCE_ATOL)


def test_repos_bien_equilibre(cp):
    """η constante, u = 0 : le fusionné hérite la C-property du motif."""
    q = np.zeros((1, 2, 4, 12, 12), dtype=np.float32)
    q[:, :, 0] = 1.0
    q[:, :, 3] = 0.2
    obtenu, _ = _fusionne(cp, q)
    np.testing.assert_allclose(obtenu, q, atol=1e-6)


def test_systemes_independants_bitwise(cp):
    """E4a : le kernel confine chaque thread à son bloc (fenêtre,
    système) — perturber le système 0 ne change pas UN BIT du système 1
    (kernel déterministe, comparaison exacte)."""
    q_repos = np.zeros((1, 2, 4, 12, 12), dtype=np.float32)
    q_repos[:, :, 0] = 1.0
    q_repos[:, :, 3] = 0.2
    q_mixte = q_repos.copy()
    q_mixte[:, 0, 0] += 0.1 * np.random.default_rng(5).random(
        (1, 12, 12)).astype(np.float32)
    sortie_mixte, _ = _fusionne(cp, q_mixte)
    sortie_repos, _ = _fusionne(cp, q_repos)
    assert np.array_equal(sortie_mixte[:, 1], sortie_repos[:, 1])
    assert not np.array_equal(sortie_mixte[:, 0], sortie_repos[:, 0])


def test_sortie_in_place_bitwise(cp):
    from src.f1_gpu.substrat_fusionne import pas_f_fusionne
    q_cpu = etat_initial_jetable(1, 10, graine=7)
    attendu, _ = _fusionne(cp, q_cpu)
    q_gpu = cp.asarray(q_cpu)
    meme, _ = pas_f_fusionne(q_gpu, cp, sortie=q_gpu)
    assert meme is q_gpu
    assert np.array_equal(vers_cpu(meme), attendu)


def test_shapes_invalides_fail_loud(cp):
    from src.f1_gpu.substrat_fusionne import pas_f_fusionne
    with pytest.raises(ValueError, match="float32"):
        pas_f_fusionne(cp.zeros((1, 2, 4, 8, 8), dtype=cp.float64), cp)
    with pytest.raises(ValueError, match="E4a"):
        pas_f_fusionne(cp.zeros((1, 3, 4, 8, 8), dtype=cp.float32), cp)
