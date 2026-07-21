"""Tests F1 — M-b tranche-2 : fovéation 2 niveaux + halo-parent rafraîchi
entre étages RK2 (§A31).

Ce que ces tests protègent :
  1. la C-PROPERTY tient sur la structure 2-niveaux (fovéa à halo-parent au
     repos ⇒ pas de vitesse parasite — le bien-équilibré traverse le bord
     fovéa/grossier) ;
  2. le REFRESH inter-étage est load-bearing : refreshed ≠ halo figé (la
     signature du 0,186 de C1) ;
  3. l'invariant liant tient : l'opérateur intérieur est le kernel de C1,
     inchangé (empreinte 6dd207ca) — le refresh est ICI, pas dans C1."""
import hashlib

import numpy as np
import pytest

from config import GridConfig
from src.f1_gpu.backend import cupy_disponible
from src.f1_gpu.fovea_2niveaux import (
    DECIMATION,
    construire_halo,
    decimer,
    pas_deux_niveaux,
)
from src.sediment import default_terrain

gpu_requis = pytest.mark.skipif(
    not cupy_disponible(), reason="cupy/device CUDA indisponible")

N: int = 64
NF: int = 32          # fovéa
OY, OX = 16, 16       # position fovéa (intérieure)


def _terrain():
    return default_terrain(GridConfig()).astype(np.float32)


# ----- 1. décimation et halo -----

@gpu_requis
def test_decimer_moyenne_2x2():
    import cupy as cp
    a = cp.arange(1 * 1 * 1 * 4 * 4, dtype=cp.float32).reshape(1, 1, 1, 4, 4)
    d = decimer(a, cp)
    assert d.shape == (1, 1, 1, 2, 2)
    assert float(d[0, 0, 0, 0, 0]) == pytest.approx((0 + 1 + 4 + 5) / 4)
    assert DECIMATION == 2


@gpu_requis
def test_construire_halo_interieur_fin_anneau_grossier():
    """Halo padé de 2 : intérieur = fovéa fine ; anneau ±2 = grossier
    upsamplé (continuation parent, pas la fovéa)."""
    import cupy as cp
    b0 = _terrain()
    h = (b0.max() + 0.3 - b0)
    qf = cp.asarray(np.stack([h, np.zeros_like(h), np.zeros_like(h)]
                             ).reshape(1, 1, 3, N, N))
    bf = cp.asarray(b0.reshape(1, 1, N, N))
    qc = decimer(qf, cp)
    bc = decimer(bf.reshape(1, 1, 1, N, N), cp)[:, :, 0]
    qff = cp.ascontiguousarray(qf[:, :, :, OY:OY + NF, OX:OX + NF])
    bff = cp.ascontiguousarray(bf[:, :, OY:OY + NF, OX:OX + NF])
    halo_q, halo_b = construire_halo(qff, bff, qc, bc, OY, OX, cp)
    assert halo_q.shape == (1, 1, 3, NF + 4, NF + 4)
    # intérieur == la fovéa
    assert float(cp.max(cp.abs(halo_q[:, :, :, 2:NF + 2, 2:NF + 2] - qff))) == 0.0


# ----- 2. C-property sur la structure 2-niveaux -----

@gpu_requis
def test_c_property_deux_niveaux():
    """Fovéa (halo-parent) + grossier au repos ⇒ pas de vitesse parasite :
    le bien-équilibré traverse le bord fovéa/grossier."""
    import cupy as cp
    b0 = _terrain()
    h = (b0.max() + 0.3 - b0)                       # plein mouillé
    qf = cp.asarray(np.stack([h, np.zeros_like(h), np.zeros_like(h)]
                             ).reshape(1, 1, 3, N, N))
    bf = cp.asarray(b0.reshape(1, 1, N, N))
    qc = decimer(qf, cp)
    bc = decimer(bf.reshape(1, 1, 1, N, N), cp)[:, :, 0]
    qff = cp.ascontiguousarray(qf[:, :, :, OY:OY + NF, OX:OX + NF])
    bff = cp.ascontiguousarray(bf[:, :, OY:OY + NF, OX:OX + NF])
    c2, f2 = pas_deux_niveaux(qc, bc, qff, bff, OY, OX, cp, dt=0.01)
    assert float(cp.max(cp.abs(f2[0, 0, 1:3]))) < 1e-6    # fovéa
    assert float(cp.max(cp.abs(c2[0, 0, 1:3]))) < 1e-6    # grossier


# ----- 3. le refresh inter-étage est load-bearing -----

@gpu_requis
def test_refresh_inter_etage_matter():
    """refreshed ≠ halo FIGÉ : sans refresh, le halo se périme au 2e étage
    (la signature du 0,186 de C1). Le refresh corrige la cause."""
    import cupy as cp
    from src.f1_gpu.substrat_fidele import pas_f_fidele

    b0 = _terrain()
    rng = np.random.default_rng(3)
    eta = float(b0.min() + 0.4 * (b0.max() - b0.min()))
    h = (np.maximum(eta - b0, 0.0) + 0.02 * rng.random((N, N))).astype(
        np.float32)
    hu = (0.1 * rng.standard_normal((N, N)) * (h > 1e-3)).astype(np.float32)
    hv = (0.1 * rng.standard_normal((N, N)) * (h > 1e-3)).astype(np.float32)
    qf = cp.asarray(np.stack([h, hu, hv]).reshape(1, 1, 3, N, N))
    bf = cp.asarray(b0.reshape(1, 1, N, N))
    qc = decimer(qf, cp)
    bc = decimer(bf.reshape(1, 1, 1, N, N), cp)[:, :, 0]
    qff = cp.ascontiguousarray(qf[:, :, :, OY:OY + NF, OX:OX + NF])
    bff = cp.ascontiguousarray(bf[:, :, OY:OY + NF, OX:OX + NF])

    _, f_ref = pas_deux_niveaux(qc, bc, qff, bff, OY, OX, cp, dt=0.01)
    halo_q, halo_b = construire_halo(qff, bff, qc, bc, OY, OX, cp)
    f_stale, _ = pas_f_fidele(qff, bff, cp, dt=0.01, mode=1,
                              halo_q=halo_q, halo_b=halo_b)
    assert float(cp.max(cp.abs(f_ref - f_stale))) > 1e-3   # le refresh compte


# ----- 4. invariant liant : le kernel de C1, inchangé -----

def test_operateur_interieur_est_le_kernel_de_c1():
    """Le refresh est ICI (couche t2), pas dans C1 : l'empreinte de
    _SOURCE_FIDELE ne bouge pas, et fovea_2niveaux importe le kernel de C1."""
    import src.f1_gpu.fovea_2niveaux as f2
    from src.f1_gpu.substrat_fidele import (
        _SOURCE_FIDELE,
        _obtenir_kernel_fidele,
    )
    assert hashlib.sha256(_SOURCE_FIDELE.encode()).hexdigest() == (
        "6dd207cae0a9c23a6a042db825a3259051e8491c1f118d614e52a4b4ebc6265d")
    # fovea_2niveaux ORCHESTRE le kernel de C1 (même objet)
    assert f2._obtenir_kernel_fidele is _obtenir_kernel_fidele
