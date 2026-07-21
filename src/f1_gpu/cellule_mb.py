"""F1 — M-b tranche-2 : la CELLULE §A15 et l'observable Δχ (§A31-build).

Ce module ne fait évoluer RIEN : il porte ce que les DEUX bras (production
et témoin) partagent, pour que la comparaison soit apples-to-apples (§A31-
build) — même histoire, mêmes instants d'émission, même observable.

LA CELLULE, GRAVÉE ET RECONDUITE (§A15) : 3 seeds {101, 102, 103}, Δt = 4,
6 émissions ⇒ 24 épisodes, commits fenêtrés, k_fen aire-proportionnel,
cap 10 %.

LE VERROU DES CENTRES. Le rederive `run_history` est INTOUCHÉ : on ne peut
pas lui demander ses centres, il ne les rend pas. Les deux bras doivent
pourtant parcourir EXACTEMENT la même histoire, sans quoi Δχ mesurerait un
écart d'histoire et non de fidélité. `centres_cellule` reproduit donc son
tirage (`default_rng(seed)`, uniforme dans [0,15 ; 0,85]²) — et un test
VÉRIFIE l'identité contre `run_history` lui-même plutôt que de la supposer.

L'OBSERVABLE. Δχ readout : `albedo` au point d'opération gravé S_HALF_OP,
puis `delta_chi(...)["max_carrier"]`. Primitives EXISTANTES, importées et
non réécrites — espace instrument, jamais l'état (règle 2)."""
from __future__ import annotations

import numpy as np

from scripts.run_arcA_revalidate import S_HALF_OP
from src.albedo import albedo, delta_chi

# ── la cellule §A15, reconduite sans changement ───────────────────────────
GRAINES: tuple[int, ...] = (101, 102, 103)
DELTA_T: int = 4                       # épisodes entre deux émissions
N_EMISSIONS: int = 6
N_EPISODES: int = DELTA_T * N_EMISSIONS                    # 24
EMISSIONS: tuple[int, ...] = tuple(
    DELTA_T * (i + 1) for i in range(N_EMISSIONS))         # 4, 8, ..., 24

# Cap anti-trivialité : un commit ne porte pas plus de 10 % de la pleine
# résolution de sa fenêtre (§A15).
CAP_COMMIT: float = 0.10


def centres_cellule(seed: int, n_episodes: int) -> list[tuple[float, float]]:
    """Les centres de pulse de l'histoire `seed`, IDENTIQUES à ceux que le
    rederive intouché tire (`run_history` : `default_rng(seed)` uniforme
    dans la boîte [0,15 ; 0,85]², un tirage (y, x) par épisode).

    Les deux bras les REÇOIVENT — ils ne les tirent pas — pour que la
    comparaison porte sur la fidélité et non sur l'histoire."""
    rng = np.random.default_rng(seed)
    return [(float(rng.uniform(0.15, 0.85)), float(rng.uniform(0.15, 0.85)))
            for _ in range(n_episodes)]


def dchi(champ_candidat: np.ndarray, champ_reference: np.ndarray) -> float:
    """L'observable gravé : Δχ readout au point d'opération S_HALF_OP,
    critère `max_carrier`. `dchi(x, x) = 0.0` par construction."""
    return float(delta_chi(albedo(np.asarray(champ_candidat, dtype=np.float64),
                                  S_HALF_OP),
                           albedo(np.asarray(champ_reference, dtype=np.float64),
                                  S_HALF_OP))["max_carrier"])


def budget_k_fen(aire: int) -> int:
    """Budget de floats d'un commit fenêtré : k_fen AIRE-PROPORTIONNEL, borné
    par le cap 10 % (§A15). Plancher à 1 float — une fenêtre minuscule doit
    rester muette par SEUIL, jamais par arrondi du budget."""
    return max(1, int(CAP_COMMIT * aire))
