"""F1 — M-b tranche-2 : le BRAS TÉMOIN, isolant la source (a) (§A31).

Le témoin est le fidèle F GPU f32 au 64² PLEIN DOMAINE, UN SEUL NIVEAU, murs
réfléchissants — configuration IDENTIQUE au rederive (`run_episode`), seule
l'arithmétique diffère (f32 vs f64). Pas de pyramide, pas de L3, pas de L1.
Il isole (a) l'arithmétique : si le témoin PASSE, une traversée en production
n'est pas due à f32, donc pas guérie par Option B (§A31, lecture 3 branches).

Objection de C1 tombée d'elle-même (§A31) : le bord du témoin est le VRAI
bord du domaine 64², donc RÉFLÉCHISSANT est physiquement correct —
l'interdiction visait les bords INTÉRIEURS de fenêtre.

FIDÉLITÉ DU PORTAGE. Le kernel de C1 EST le pas SSP-RK2 de
`simulate_wetdry_o2` (Heun, plancher sec par étage, mêmes fonctions device).
Le témoin reproduit donc `run_episode` : b_eff = b0 + s (gelé) ; pulse ;
N_settle pas wetdry à dt CFL-adaptatif (kernel figé via `pas_f_fidele`,
réfléchissant, invariant liant) ; intégration Exner sur les snapshots à la
cadence `save_every` LUE. Seul l'arithmétique change.

INVARIANT LIANT : `pas_f_fidele` est LE MÊME OBJET qu'en tranche-1. Cadence
Exner LUE de `save_every`, jamais choisie. Aucun kernel neuf."""
from __future__ import annotations

import numpy as np

from config import GridConfig
from src.f1_gpu.exner_gpu import CADENCE_EXNER, pas_exner
from src.f1_gpu.substrat_fidele import (
    MODE_REFLECHISSANT,
    pas_f_fidele,
    reduction_cfl_fidele,
)
from src.sediment import SedimentParams, _initial_pulse, _T_END_RELAX


def _pulse_gpu(b_eff_shape, centre_frac, params, cp):
    """Pulse gaussien (h0, u=v=0) — calculé sur CPU (déterministe, comme
    `_initial_pulse`) puis transféré. b_eff_shape = (H, W)."""
    grid = GridConfig(H=b_eff_shape[0], W=b_eff_shape[1])
    h0, hu0, hv0 = _initial_pulse(grid, centre_frac, params)
    q = np.stack([h0, hu0, hv0]).astype(np.float32)
    return cp.asarray(q.reshape(1, 1, 3, b_eff_shape[0], b_eff_shape[1]))


def evoluer_episode_temoin(s, b0, centre_frac, cp,
                           params: SedimentParams = SedimentParams(),
                           n_settle: int | None = None):
    """Un épisode TÉMOIN sur GPU f32, plein domaine 64², réfléchissant.
    Reproduit `run_episode` : pulse -> relaxation wetdry (N_settle pas, dt
    CFL-adaptatif, borné à t_end) -> Exner sur snapshots. `s`, `b0` numpy
    f32 (H, W) ; retourne le nouveau `s` numpy f32 (H, W).

    `n_settle` surcharge params.N_settle (tests réduits) — la cadence Exner
    reste LUE de save_every."""
    n_settle = params.N_settle if n_settle is None else n_settle
    H, W = b0.shape
    b_eff = cp.asarray((b0 + s).astype(np.float32).reshape(1, 1, H, W))
    q = _pulse_gpu((H, W), centre_frac, params, cp)
    tampon = cp.empty_like(q)

    # relaxation : dt CFL-adaptatif borné à t_end (comme le CPU)
    t = 0.0
    snapshots, temps = [q.copy()], [0.0]
    for _ in range(n_settle):
        dt = reduction_cfl_fidele(q, cp)
        if t + dt > _T_END_RELAX:
            dt = _T_END_RELAX - t
        q, _ = pas_f_fidele(q, b_eff, cp, dt=dt, sortie=q,
                            tampon_etage=tampon, mode=MODE_REFLECHISSANT)
        t += dt
        snapshots.append(q.copy())
        temps.append(t)

    # Exner sur les snapshots (cadence save_every LUE), s persistant
    s_dev = cp.asarray(s.astype(np.float32).reshape(1, 1, H, W))
    for i in range(CADENCE_EXNER, n_settle + 1, CADENCE_EXNER):
        dt_exner = float(temps[i] - temps[i - CADENCE_EXNER])
        pas_exner(snapshots[i], s_dev, cp, dt_exner)
    return cp.asnumpy(s_dev)[0, 0]


def evoluer_histoire_temoin(seed: int, n_episodes: int, b0, cp,
                            centres, params: SedimentParams = SedimentParams(),
                            checkpoints: set[int] | None = None) -> dict:
    """Séquence de `n_episodes` épisodes témoins, centres FOURNIS (mêmes que
    le rederive `run_history` — le témoin ne les tire pas, il les reçoit,
    pour rester config-identique). Retourne {n: s_gpu} aux checkpoints."""
    if checkpoints is None:
        checkpoints = {n_episodes}
    s = np.zeros_like(b0, dtype=np.float32)
    out: dict[int, object] = {}
    for ep in range(1, n_episodes + 1):
        s = evoluer_episode_temoin(s, b0, centres[ep - 1], cp, params)
        if ep in checkpoints:
            out[ep] = s.copy()
    return out
