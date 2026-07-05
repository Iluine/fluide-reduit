"""Arc C / session — sélection et GARDE du backend matplotlib (plomberie session live).

Une session HUMAINE (§C3) exige un backend GUI : une fenêtre affichée + la capture
clavier a/b. Sur un venv sans binding Qt (ni tkinter), matplotlib se rabat sur
**Agg** (non-interactif) : aucune fenêtre, et `plt.waitforbuttonpress` ne capte
JAMAIS de touche -> la session boucle sans fin (le mode d'échec observé au 1er
pré-vol B). Ce module :
  1. `selectionne_backend_qt` : tente de basculer sur un backend Qt interactif
     (exige `PySide6`/`PyQt`), à appeler AVANT toute création de figure ;
  2. `assert_backend_interactif` : GARDE PUR qui LÈVE clair en session live si le
     backend reste non-interactif -- jamais un hang silencieux.

Rien ici n'importe matplotlib au chargement (import paresseux dans les fonctions),
pour rester testable sans effet de bord de backend."""
from __future__ import annotations

# Backends matplotlib SANS fenêtre : une session humaine ne peut pas tourner dessus.
BACKENDS_NON_INTERACTIFS: frozenset[str] = frozenset(
    {"agg", "pdf", "ps", "svg", "template", "cairo", "pgf"})


def selectionne_backend_qt() -> str | None:
    """Tente `matplotlib.use("QtAgg")` (exige un binding Qt : PySide6/PyQt6/PyQt5).
    Renvoie le nom du backend actif après tentative, ou `None` si aucun binding Qt
    n'est disponible (le défaut, souvent Agg, reste alors -- `assert_backend_
    interactif` lèvera en session live). À appeler AVANT toute création de figure
    (bascule via `switch_backend`, sûre tant qu'aucune figure n'existe encore)."""
    import matplotlib
    try:
        matplotlib.use("QtAgg")
        return matplotlib.get_backend()
    except Exception:
        return None


def assert_backend_interactif(backend: str, *, est_replay: bool,
                              display: str | None = None) -> None:
    """GARDE PUR. En session LIVE (`est_replay=False`), lève un `RuntimeError`
    CLAIR si `backend` est non-interactif (pas de fenêtre) -- au lieu de laisser
    la session boucler sans fin sur `waitforbuttonpress`. En `--replay` (aucune
    capture humaine), tout backend convient (no-op).

    Le message est un AIGUILLAGE : il nomme le défaut, la cause, la correction
    exacte, et acte qu'AUCUNE donnée n'a été produite (pas un demi-état)."""
    if est_replay:
        return
    if backend.strip().lower() in BACKENDS_NON_INTERACTIFS:
        raise RuntimeError(
            f"Backend matplotlib NON-INTERACTIF ({backend!r}) -- une session humaine exige une "
            "fenêtre GUI (affichage des stimuli + capture clavier a/b). Ce venv n'a aucun backend "
            "GUI (matplotlib se rabat sur Agg) : installe un binding Qt puis relance --\n"
            "    uv pip install --python .venv PySide6\n"
            f"(DISPLAY={display!r} -- un serveur X/Wayland est aussi requis côté machine ; "
            "l'init Qt/xcb a été vérifiée sur ce poste.) AUCUN essai n'a été présenté, AUCUNE "
            "donnée n'est produite -- rien à reprendre, seulement à réinstaller.")
