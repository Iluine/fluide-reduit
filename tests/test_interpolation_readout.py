"""Tests — INTERPOLATION DE READOUT (`src/f1_gpu/interpolation_readout.py`),
les six verrous du §7 du spec vérifiés MÉCANIQUEMENT.

Backend numpy (B1), tailles minuscules (VM) : on teste ici la LOGIQUE et les
GARDES, JAMAIS un coût. `I` ne se mesure pas ici — §A61 l'interdit tant que
l'instrument n'est pas qualifié, et §A63 ne l'a pas qualifié.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.f1_gpu.interpolation_readout import (
    ALPHA_EXTRAPOLER,
    ALPHA_INTERPOLER,
    INDICE_CHAMP_S,
    TamponReadout,
)
from src.f1_gpu.pyramide import GeometriePyramide, PyramideFovea, Slot
from src.f1_gpu.transferts import TransfertComptable

N_FOV = 8
N0 = 16
N_NIV = 4

SLOTS = (
    Slot(niveau=1, c=4, role="fovea"),
    Slot(niveau=2, c=8, role="fovea"),
    Slot(niveau=3, c=8, role="fovea"),
)


def _pyramide():
    geo = GeometriePyramide(n_fov=N_FOV, n_niv=N_NIV, n0=N0, slots=SLOTS)
    transferts = TransfertComptable(np)
    pyr = PyramideFovea(np, geo, transferts, eps_detail=1e30)
    transferts.frame_suivante()
    return pyr


def _peindre_s(pyr, valeur: float) -> None:
    """Peint le champ `s` de toutes les fenêtres avec une valeur constante."""
    for j in pyr.geo.niveaux_gpu:
        pyr.fenetres[j][:, :, INDICE_CHAMP_S, :, :] = float(valeur)


def test_alpha_zero_rend_s_prev():
    """Verrou (a) — borne basse. Propriété du NOYAU : les images exactes
    CONTOURNENT le noyau en production (§3 du spec)."""
    pyr = _pyramide()
    _peindre_s(pyr, 2.0)
    tampon = TamponReadout(pyr)
    tampon.capturer(pyr)          # s_prev <- 2.0
    _peindre_s(pyr, 7.0)          # s_cur  <- 7.0
    vue = tampon.melanger(pyr, 0.0)
    for j in pyr.geo.niveaux_gpu:
        assert np.array_equal(vue.fenetres[j][:, :, 0], np.full_like(
            vue.fenetres[j][:, :, 0], 2.0))


def test_alpha_un_rend_s_cur():
    """Verrou (a) — borne haute."""
    pyr = _pyramide()
    _peindre_s(pyr, 2.0)
    tampon = TamponReadout(pyr)
    tampon.capturer(pyr)
    _peindre_s(pyr, 7.0)
    vue = tampon.melanger(pyr, 1.0)
    for j in pyr.geo.niveaux_gpu:
        assert np.array_equal(vue.fenetres[j][:, :, 0], np.full_like(
            vue.fenetres[j][:, :, 0], 7.0))


def test_melange_a_un_demi():
    """Le mode INTERPOLER : α = ½ entre deux états RÉELS."""
    pyr = _pyramide()
    _peindre_s(pyr, 2.0)
    tampon = TamponReadout(pyr)
    tampon.capturer(pyr)
    _peindre_s(pyr, 8.0)
    vue = tampon.melanger(pyr, ALPHA_INTERPOLER)
    for j in pyr.geo.niveaux_gpu:
        assert np.allclose(vue.fenetres[j][:, :, 0], 5.0)


def test_les_deux_modes_partagent_un_seul_chemin():
    """Verrou (b) — §2 du spec MÉCANISÉ.

    `I` est annoncé INVARIANT AU MODE par construction. Non vérifié, cet
    énoncé est une garde promise, donc une garde absente (§A62-bis-2). Ici
    on l'exécute : à `α` ÉGAL, les deux appels doivent rendre les MÊMES
    OCTETS — s'il existait un chemin propre à un mode, ils différeraient."""
    pyr = _pyramide()
    _peindre_s(pyr, 3.0)
    tampon = TamponReadout(pyr)
    tampon.capturer(pyr)
    _peindre_s(pyr, 9.0)

    for alpha in (ALPHA_INTERPOLER, ALPHA_EXTRAPOLER):
        premier = {j: np.array(tampon.melanger(pyr, alpha).fenetres[j],
                               copy=True)
                   for j in pyr.geo.niveaux_gpu}
        second = {j: np.array(tampon.melanger(pyr, alpha).fenetres[j],
                              copy=True)
                  for j in pyr.geo.niveaux_gpu}
        for j in pyr.geo.niveaux_gpu:
            assert premier[j].tobytes() == second[j].tobytes(), (
                f"le mode α={alpha} n'est pas déterministe au niveau {j}")


def test_extrapoler_depasse_et_le_clamp_compte():
    """Verrou (d) — §5 du spec. Le clamp COMPTE et REPORTE, il ne lève pas."""
    pyr = _pyramide()
    _peindre_s(pyr, 10.0)
    tampon = TamponReadout(pyr)
    tampon.capturer(pyr)          # s_prev = 10
    _peindre_s(pyr, 1.0)          # s_cur  = 1  ⇒ α=1½ donne −3,5
    vue = tampon.melanger(pyr, ALPHA_EXTRAPOLER)

    assert vue.clamps > 0, "l'extrapolation devait passer sous zéro"
    for j in pyr.geo.niveaux_gpu:
        assert (vue.fenetres[j] >= 0.0).all(), "le clamp n'a pas tenu"


def test_interpoler_ne_depasse_jamais():
    """Verrou (d), face FAVORABLE — un critère se vérifie aussi sur le cas
    qui doit PASSER (§A52), sinon il punit la qualité qu'il contrôle."""
    pyr = _pyramide()
    _peindre_s(pyr, 10.0)
    tampon = TamponReadout(pyr)
    tampon.capturer(pyr)
    _peindre_s(pyr, 1.0)
    vue = tampon.melanger(pyr, ALPHA_INTERPOLER)
    assert vue.clamps == 0, "α=½ entre deux états positifs ne peut pas clamper"
