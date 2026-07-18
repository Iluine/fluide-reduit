"""F1 tranche-1 — composant (f) : chrono GPU (choix B6, cf. `__init__.py`).

Protocole gravé (§A15, « Instrument ») : « Chrono GPU : synchronisation
explicite avant/après mesure ; warmup ≥ 30 frames EXCLU ; série ≥ 300
frames ; médiane ET p99 reportées toutes deux (chiffres inconfortables en
évidence). »

B6 opérationnalisé :
  - cupy : une paire de cuda-Events par frame (`get_elapsed_time`, précision
    device), synchronisation device AVANT le warmup, AVANT la série, et par
    frame (event.synchronize() sur l'event de fin — la frame est mesurée
    bord à bord, aucun recouvrement) ;
  - numpy (tests VM uniquement, B1) : `time.perf_counter` — ce chemin ne
    produit JAMAIS une lecture ;
  - médiane = numpy.percentile(., 50), p99 = numpy.percentile(., 99),
    interpolation linéaire (défaut numpy, nommé) ;
  - AUCUN timestamp dans les données retournées."""
from __future__ import annotations

import time

import numpy as np

from src.f1_gpu.backend import est_cupy, synchroniser

WARMUP_FRAMES: int = 30   # gravé §A15 : warmup >= 30 EXCLU de la série
SERIE_FRAMES: int = 300   # gravé §A15 : série >= 300


def stats_ms(echantillons_ms: list[float]) -> dict:
    """Statistiques B6 d'une série (ms) : médiane ET p99 (les deux reportées,
    gravé), min/max/n en appui. Série vide -> ValueError (fail-loud)."""
    if not echantillons_ms:
        raise ValueError("stats_ms : série vide.")
    arr = np.asarray(echantillons_ms, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mediane_ms": float(np.percentile(arr, 50)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def chronometrer_frames(xp, frame_fn, warmup: int = WARMUP_FRAMES,
                        serie: int = SERIE_FRAMES) -> dict:
    """Chronomètre `frame_fn()` (une frame complète, sans argument) :
    `warmup` appels EXCLUS puis `serie` appels mesurés un à un (B6).

    Retourne `{"serie_ms": [...], "stats": stats_ms(...),
    "warmup_ms": [...]}` — le warmup est reporté à part (diagnostic,
    jamais agrégé dans les stats). AUCUN timestamp."""
    if warmup < 0 or serie <= 0:
        raise ValueError(
            f"chronometrer_frames : warmup={warmup} / serie={serie} invalides.")

    if est_cupy(xp):
        mesurer = _mesurer_frame_cupy(xp)
    else:
        mesurer = _mesurer_frame_cpu

    warmup_ms: list[float] = []
    synchroniser(xp)
    for _ in range(warmup):
        warmup_ms.append(mesurer(frame_fn))

    serie_ms: list[float] = []
    synchroniser(xp)
    for _ in range(serie):
        serie_ms.append(mesurer(frame_fn))
    synchroniser(xp)

    return {"serie_ms": serie_ms, "stats": stats_ms(serie_ms),
            "warmup_ms": warmup_ms}


def _mesurer_frame_cupy(cp):
    """Fabrique le mesureur cuda-Events (une paire d'events réutilisée —
    aucune allocation par frame, B2)."""
    debut = cp.cuda.Event()
    fin = cp.cuda.Event()

    def mesurer(frame_fn) -> float:
        debut.record()
        frame_fn()
        fin.record()
        fin.synchronize()
        return float(cp.cuda.get_elapsed_time(debut, fin))

    return mesurer


def _mesurer_frame_cpu(frame_fn) -> float:
    """Mesureur perf_counter — backend numpy, TESTS UNIQUEMENT (B1)."""
    t0 = time.perf_counter()
    frame_fn()
    return (time.perf_counter() - t0) * 1e3
