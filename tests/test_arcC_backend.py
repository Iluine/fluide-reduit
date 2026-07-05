"""Arc C — garde de backend matplotlib (`src/arcC_backend.py`).

Teste le GARDE PUR `assert_backend_interactif` (aucune dépendance matplotlib :
la sélection réelle du backend et l'affichage sont de la glue non-testable, cf.
docstring module). Le garde empêche le mode d'échec observé au 1er pré-vol B :
une session live sur backend Agg (non-interactif) qui boucle sans fin sur
`waitforbuttonpress` sans jamais capter de touche."""
from __future__ import annotations

import pytest

from src.arcC_backend import BACKENDS_NON_INTERACTIFS, assert_backend_interactif


def test_session_live_sur_agg_leve_clair():
    """Le cœur : session LIVE + backend non-interactif -> RuntimeError CLAIR
    (nomme le backend, la correction PySide6, le DISPLAY, et acte AUCUNE
    donnée)."""
    with pytest.raises(RuntimeError) as exc:
        assert_backend_interactif("agg", est_replay=False, display=":0")
    msg = str(exc.value)
    assert "agg" in msg.lower()
    assert "PySide6" in msg          # la correction exacte
    assert ":0" in msg               # le DISPLAY, pour diagnostiquer
    assert "AUCUNE" in msg or "aucune" in msg.lower()  # aucune donnée produite


@pytest.mark.parametrize("backend", sorted(BACKENDS_NON_INTERACTIFS))
def test_tous_les_backends_non_interactifs_levent_en_live(backend):
    with pytest.raises(RuntimeError):
        assert_backend_interactif(backend, est_replay=False, display=":0")


@pytest.mark.parametrize("backend", ["QtAgg", "Qt5Agg", "TkAgg", "GTK4Agg", "MacOSX"])
def test_backend_interactif_ne_leve_pas(backend):
    """Un backend GUI (fenêtre possible) passe -- session live OK."""
    assert_backend_interactif(backend, est_replay=False, display=":0")


def test_replay_ne_leve_jamais_meme_sur_agg():
    """En `--replay` (aucune capture humaine), tout backend convient : le garde
    est un no-op, y compris sur Agg."""
    assert_backend_interactif("agg", est_replay=True, display=None)
    assert_backend_interactif("QtAgg", est_replay=True, display=None)


def test_casse_insensible_et_espaces():
    """`get_backend()` peut renvoyer 'Agg'/' agg ' -- la comparaison normalise."""
    with pytest.raises(RuntimeError):
        assert_backend_interactif("  Agg ", est_replay=False, display=":0")
