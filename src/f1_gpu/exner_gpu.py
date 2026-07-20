"""F1 — M-b tranche-1 / C2 : intégration Exner sur GPU.

Portage GPU de `_exner_step` (src/sediment.py) — dépôt/érosion type Exner
sur les cellules mouillées, POINTWISE (aucun stencil). Le sédiment `s` est
mis à jour ; il n'est PAS advecté par le flux (§A28) — c'est ce kernel, pas
le F fidèle, qui le fait évoluer.

    wet = h > 10·DRY_EPS   (seuil Exner conservateur, cf. _exner_step)
    θ = u² + v²
    ds/dt = 1[wet]·(k_d·h·1[θ<θc]·(1−θ/θc) − k_e·s·1[θ>θc]·(θ/θc−1))
    s ← max(s + dt·ds/dt, 0)

CADENCE DÉRIVÉE DU REDERIVE, PAS CHOISIE (consigne §A29-C1). `run_episode`
fait ~300 pas Exner pour ~600 pas wetdry (`save_every = 2`) : UN Exner tous
les DEUX pas. `CADENCE_EXNER` LIT `SedimentParams.save_every` — la choisir
au build ferait comparer deux physiques à tranche-2 (l'invariant « même code
F » ne couvre pas la cadence). Conséquence sur la bande T1 : le poste Exner
est amorti par 2 (poste par-Exner ÷ cadence), il passe de [1,5 ; 3,0] à
[0,75 ; 1,5] ms/frame.

Kernels FIGÉS et C1 intouchés : ce module est un `_SOURCE_EXNER` NEUF, avec
son propre verrou d'empreinte (§A29-C — on ne refait pas le coup de L3)."""
from __future__ import annotations

import hashlib

import numpy as np

from src.f1_gpu.substrat_jetable import DRY_EPS
from src.sediment import KD_CALIBRE_V2, KE_CALIBRE_V2, SedimentParams

# Cadence LUE dans la structure du rederive (save_every), jamais choisie :
# un pas Exner tous les `CADENCE_EXNER` pas wetdry.
CADENCE_EXNER: int = SedimentParams().save_every        # = 2
THETA_C: float = SedimentParams().theta_c               # = 0.5
_WET_EXNER: float = 10.0 * DRY_EPS                       # seuil Exner-mouillé

_NOM_KERNEL_EXNER: str = "exner_step"

_SOURCE_EXNER: str = r"""
#define DRY_EPS """ + f"{DRY_EPS}f" + r"""
#define WET_EXNER """ + f"{_WET_EXNER}f" + r"""
#define THETA_C """ + f"{THETA_C}f" + r"""
#define K_D """ + f"{KD_CALIBRE_V2}f" + r"""
#define K_E """ + f"{KE_CALIBRE_V2}f" + r"""

/* Un pas d'Euler explicite Exner sur (h, hu, hv) -> s. Pointwise : un thread
   = une cellule. s lu et écrit sur place (aucune dépendance inter-cellule).
   q layout (B, S, 3, n, n) ; s layout (B, S, n, n). */
extern "C" __global__ void """ + _NOM_KERNEL_EXNER + r"""(
        const float* q, float* s, const float dt, const int n,
        const long total) {
    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) return;
    long nn = (long)n * n;
    long bloc = t / nn;
    long local = t % nn;
    long idxq = bloc * 3 * nn + local;
    long idxs = bloc * nn + local;

    float h  = q[idxq];
    float hu = q[idxq + nn];
    float hv = q[idxq + 2 * nn];
    int wet = (h > WET_EXNER);
    float u = wet ? hu / h : 0.0f;
    float v = wet ? hv / h : 0.0f;
    float theta = u * u + v * v;
    float depot   = (wet && (theta < THETA_C))
                    ? K_D * h * (1.0f - theta / THETA_C) : 0.0f;
    float erosion = (wet && (theta > THETA_C))
                    ? K_E * s[idxs] * (theta / THETA_C - 1.0f) : 0.0f;
    float sn = s[idxs] + dt * (depot - erosion);
    s[idxs] = fmaxf(sn, 0.0f);
}
"""

# VERROU D'EMPREINTE (§A29-C), dès le premier commit.
LONGUEUR_SOURCE_EXNER: int = len(_SOURCE_EXNER)
SHA256_SOURCE_EXNER: str = hashlib.sha256(_SOURCE_EXNER.encode()).hexdigest()

_TAILLE_BLOC: int = 256
_cache_kernel_exner = None


def _obtenir_kernel_exner(cp):
    global _cache_kernel_exner
    if _cache_kernel_exner is None:
        _cache_kernel_exner = cp.RawKernel(_SOURCE_EXNER, _NOM_KERNEL_EXNER)
    return _cache_kernel_exner


def pas_exner(q, s, cp, dt: float) -> None:
    """Un pas Exner sur place : met à jour `s` (B, S, n, n) depuis l'état
    évolué `q` (B, S, 3, n, n). Ne retourne rien — `s` est modifié.

    APPELÉ UNE FOIS TOUS LES `CADENCE_EXNER` PAS (§A29-C1) — l'appelant
    (driver/C5) tient le compteur ; ce module ne décide pas la cadence, il
    la reçoit dérivée."""
    if q.dtype != cp.float32 or s.dtype != cp.float32:
        raise ValueError("pas_exner : q et s doivent être float32.")
    if q.ndim != 5 or q.shape[2] != 3:
        raise ValueError(
            f"pas_exner : q shape {q.shape} != (B, S, 3, n, n) — état évolué "
            "(h, hu, hv).")
    n = int(q.shape[-1])
    if s.shape != (q.shape[0], q.shape[1], n, n):
        raise ValueError(
            f"pas_exner : s shape {s.shape} != {(q.shape[0], q.shape[1], n, n)}.")
    total = int(q.shape[0] * q.shape[1] * n * n)
    kernel = _obtenir_kernel_exner(cp)
    grille = ((total + _TAILLE_BLOC - 1) // _TAILLE_BLOC,)
    kernel(grille, (_TAILLE_BLOC,),
           (cp.ascontiguousarray(q), s, np.float32(dt), np.int32(n),
            np.int64(total)))


def cadence_exner_derivee() -> dict:
    """Rend la cadence LUE dans `run_episode` (structure du rederive), pour
    que le driver la reporte et ne la choisisse pas. ~300 Exner pour ~600
    wetdry ⇒ un tous les save_every pas."""
    p = SedimentParams()
    n_wetdry = p.N_settle
    n_exner = len(range(p.save_every, p.N_settle + 1, p.save_every))
    return {
        "save_every": p.save_every,
        "n_wetdry_par_episode": n_wetdry,
        "n_exner_par_episode": n_exner,
        "cadence_exner_sur_wetdry": n_exner / n_wetdry,
        "un_exner_tous_les": n_wetdry // max(n_exner, 1),
        "source": "SedimentParams.save_every (rederive), non choisie (§A29-C1)",
    }
