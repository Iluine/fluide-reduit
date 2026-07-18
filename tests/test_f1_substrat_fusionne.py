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
