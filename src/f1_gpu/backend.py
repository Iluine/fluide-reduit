"""F1 tranche-1 — sélection du backend tableau (choix B1, cf. `__init__.py`).

B1 — BACKEND INJECTÉ : les composants du harnais prennent un module tableau
`xp` (l'API numpy). Pour la MESURE, `xp` est cupy EXCLUSIVEMENT (les drivers
`run_f1_*.py` appellent `exiger_cupy()` avant toute lecture) ; le backend
numpy n'existe que pour tester la LOGIQUE en VM (pas de mesure valide,
kill 45 s). CuPy mime l'API numpy (raison du choix T6/E2) : le code des
composants est identique sous les deux backends, seuls le chrono et les
transferts ont des chemins spécifiques (cuda-Events vs perf_counter).

L'import de cupy est PARESSEUX : importer ce module (et donc les tests VM)
ne touche jamais cupy."""
from __future__ import annotations

import numpy as np

_BACKENDS: tuple[str, ...] = ("numpy", "cupy")


def obtenir_xp(backend: str):
    """Renvoie le module tableau du backend demandé ("numpy" | "cupy").
    L'import de cupy n'a lieu qu'ici (paresseux, B1)."""
    if backend == "numpy":
        return np
    if backend == "cupy":
        import cupy
        return cupy
    raise ValueError(
        f"obtenir_xp : backend inconnu {backend!r} (attendu un de {_BACKENDS}).")


def est_cupy(xp) -> bool:
    """True ssi `xp` est le module cupy (sans l'importer)."""
    return getattr(xp, "__name__", "") == "cupy"


def cupy_disponible() -> bool:
    """True ssi cupy est importable ET un device CUDA répond à un calcul
    réel (un `getDeviceCount` seul peut réussir sans runtime utilisable)."""
    try:
        import cupy
        cupy.arange(2, dtype=cupy.float32).sum().item()
        return True
    except Exception:
        return False


def exiger_cupy():
    """Garde des drivers de MESURE : renvoie le module cupy, ou RuntimeError
    fail-loud si le chemin GPU n'est pas utilisable (jamais de repli numpy
    silencieux — une mesure numpy serait un faux chiffre, B1)."""
    if not cupy_disponible():
        raise RuntimeError(
            "exiger_cupy : chemin GPU cupy indisponible — AUCUNE mesure "
            "possible (le repli numpy est interdit pour la mesure, B1). "
            "Machine attendue : iluin-tworings3, terminal natif.")
    import cupy
    return cupy


def vers_cpu(tableau) -> np.ndarray:
    """Rapatriement D2H BRUT, hors comptage `TransfertComptable` — réservé
    aux tests et aux diagnostics (jamais dans une boucle de frame mesurée :
    tout transfert du chemin mesuré passe par `TransfertComptable`, (c))."""
    if hasattr(tableau, "get"):
        return tableau.get()
    return np.asarray(tableau)


def synchroniser(xp) -> None:
    """Barrière device (no-op sous numpy). À appeler avant/après toute
    section chronométrée côté hôte (B6 : sync explicite)."""
    if est_cupy(xp):
        xp.cuda.runtime.deviceSynchronize()
