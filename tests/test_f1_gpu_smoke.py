"""Tests F1 tranche-1 — SMOKE GPU (chemin cupy réel, B1).

Sautés proprement si cupy/le device sont indisponibles. Tailles minuscules
et séries courtes : ce sont des tests de PLOMBERIE (le chemin cuda-Events,
les transferts réels, une frame de pyramide sur device) — jamais des
mesures. Les chiffres produits ici n'ont AUCUNE valeur de lecture."""
import numpy as np
import pytest

from src.f1_gpu.backend import cupy_disponible, vers_cpu

pytestmark = pytest.mark.skipif(
    not cupy_disponible(), reason="cupy/device CUDA indisponible")


@pytest.fixture(scope="module")
def cp():
    import cupy
    return cupy


def test_chrono_cuda_events(cp):
    from src.f1_gpu.chrono import chronometrer_frames
    tampon = cp.zeros((64, 64), dtype=cp.float32)

    def frame():
        cp.multiply(tampon, 1.0, out=tampon)

    resultat = chronometrer_frames(cp, frame, warmup=2, serie=5)
    assert len(resultat["serie_ms"]) == 5
    assert all(ms >= 0.0 for ms in resultat["serie_ms"])


def test_transferts_aller_retour_bit_exact(cp):
    from src.f1_gpu.transferts import TransfertComptable
    comptable = TransfertComptable(cp)
    source = np.arange(32, dtype=np.float32)
    device = comptable.descendre(source)
    retour = comptable.remonter(device)
    bilan = comptable.frame_suivante()
    assert np.array_equal(retour, source)
    assert bilan["h2d_octets"] == bilan["d2h_octets"] == 128
    assert bilan["h2d_ms"] >= 0.0 and bilan["d2h_ms"] >= 0.0


def test_pyramide_frame_sur_device(cp):
    from src.f1_gpu.pyramide import GeometriePyramide, PyramideFovea
    from src.f1_gpu.transferts import TransfertComptable
    geo = GeometriePyramide(n_fov=8, n_niv=3, n0=16)
    transferts = TransfertComptable(cp)
    pyramide = PyramideFovea(cp, geo, transferts)
    transferts.frame_suivante()
    diag = pyramide.frame(delta_x=1)
    transferts.frame_suivante()
    assert diag["niveaux_deplaces"] == [2]
    for j in geo.niveaux_gpu:
        assert np.all(np.isfinite(vers_cpu(pyramide.fenetres[j])))
    assert transferts.bilans[-1]["h2d_octets"] == 3 * 2 * 4 * 8 * 4


def test_substrat_cupy_coincide_avec_numpy_en_tolerance(cp):
    """Le MÊME code B1 sous les deux backends : pas d'exigence bit-exacte
    (le non-déterminisme f32 GPU est un DROIT, Option A) — mais un écart
    au-delà du bruit f32 signerait une erreur de plomberie."""
    from src.f1_gpu.substrat_jetable import etat_initial_jetable, pas_f_jetable
    q_cpu = etat_initial_jetable(1, 12, graine=3)
    sortie_cpu, _ = pas_f_jetable(q_cpu, np)
    sortie_gpu, _ = pas_f_jetable(cp.asarray(q_cpu), cp)
    np.testing.assert_allclose(vers_cpu(sortie_gpu), sortie_cpu,
                               rtol=1e-5, atol=1e-6)
