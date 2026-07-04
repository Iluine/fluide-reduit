"""Tests Task 4 (spec GRAVÉE, `docs/superpowers/plans/arc-a-manche1.md`, citée
verbatim dans `.superpowers/sdd/refond-task2-m0bis-brief.md`) : compresseur/
régénérateur `src/summary.py`. Champ fin = grille 64x64 (GridConfig par défaut) :
size_floats attendu 1041 (ℓ=1), 273 (ℓ=2), 81 (ℓ=3)."""
from __future__ import annotations

import hashlib
import inspect
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.summary import Summary, regenerate, size_floats, summarize

ROOT = Path(__file__).resolve().parents[1]
_N_SOUS_DOMAINES = 4


def _sous_masses(field: np.ndarray) -> list[float]:
    """Réplique indépendante (test) du découpage 4x4 en sous-domaines, pour
    vérifier les invariants restitués sans dépendre de l'implémentation interne."""
    H, W = field.shape
    bh, bw = H // _N_SOUS_DOMAINES, W // _N_SOUS_DOMAINES
    return [float(field[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw].sum())
            for i in range(_N_SOUS_DOMAINES) for j in range(_N_SOUS_DOMAINES)]


# --- Tailles exactes (spec gravée) -------------------------------------------


@pytest.mark.parametrize("level,expected", [(1, 1041), (2, 273), (3, 81)])
def test_size_floats_tailles_exactes(level, expected):
    field = np.random.default_rng(0).random((64, 64))
    summary = summarize(field, level)
    assert size_floats(summary) == expected


def test_summary_champ_fin_est_4096():
    field = np.random.default_rng(0).random((64, 64))
    assert field.size == 4096


# --- Forme / contenu du résumé (sanity TDD) ----------------------------------


@pytest.mark.parametrize("level,expected_coarse_shape", [(1, (32, 32)), (2, (16, 16)),
                                                         (3, (8, 8))])
def test_summarize_coarse_shape(level, expected_coarse_shape):
    field = np.random.default_rng(1).random((64, 64))
    summary = summarize(field, level)
    assert summary.coarse.shape == expected_coarse_shape
    assert summary.level == level
    assert summary.shape == (64, 64)
    assert isinstance(summary, Summary)


def test_summarize_invariants_total_mass_matches_field_sum():
    field = np.random.default_rng(2).random((64, 64)) * 3.0
    summary = summarize(field, level=2)
    assert summary.invariants[0] == pytest.approx(float(field.sum()), rel=1e-12)


def test_summarize_invariants_submasses_match_reference_split():
    field = np.random.default_rng(3).random((64, 64)) * 2.0
    summary = summarize(field, level=1)
    assert len(summary.invariants) == 17
    ref = _sous_masses(field)
    for stored, expected in zip(summary.invariants[1:], ref):
        assert stored == pytest.approx(expected, rel=1e-12)


def test_summarize_block_average_on_uniform_block():
    """Sur un champ constant par blocs 2x2, la moyenne dyadique ℓ=1 restitue
    exactement la valeur du bloc."""
    field = np.zeros((64, 64))
    field[0:2, 0:2] = 7.0
    summary = summarize(field, level=1)
    assert summary.coarse[0, 0] == pytest.approx(7.0)


# --- M-A3 : déterminisme / identité cross-process ----------------------------


def test_ma3_double_regeneration_ecart_relatif_borne():
    field = np.random.default_rng(4).random((64, 64)) * 5.0
    summary = summarize(field, level=2)
    r1 = regenerate(summary)
    r2 = regenerate(summary)
    assert np.array_equal(r1, r2)
    denom = np.maximum(np.abs(r1), 1e-300)
    rel = np.abs(r1 - r2) / denom
    assert float(np.max(rel)) <= 1e-12


_MA3_SUBPROCESS_SCRIPT = """
import hashlib
import numpy as np
from src.summary import summarize, regenerate

field = np.random.default_rng(4).random((64, 64)) * 5.0
summary = summarize(field, level=2)
result = regenerate(summary)
print(hashlib.sha256(result.tobytes()).hexdigest())
"""


def test_ma3_identite_cross_process_via_hash():
    out1 = subprocess.run([sys.executable, "-c", _MA3_SUBPROCESS_SCRIPT], cwd=ROOT,
                          capture_output=True, text=True, check=True)
    out2 = subprocess.run([sys.executable, "-c", _MA3_SUBPROCESS_SCRIPT], cwd=ROOT,
                          capture_output=True, text=True, check=True)
    h1, h2 = out1.stdout.strip(), out2.stdout.strip()
    assert len(h1) == 64  # hex sha256
    assert h1 == h2


# --- Anti-fuite ---------------------------------------------------------------


def test_regenerate_signature_a_un_seul_parametre():
    sig = inspect.signature(regenerate)
    assert len(sig.parameters) == 1


def test_regenerate_apres_mutation_decoy_est_inchange():
    """`summarize` ne doit garder aucune référence vivante vers le champ vrai :
    muter le champ d'origine (décoy) APRÈS le résumé ne doit rien changer au
    résultat de `regenerate`."""
    field = np.random.default_rng(5).random((64, 64))
    decoy = field.copy()
    summary = summarize(decoy, level=3)
    result1 = regenerate(summary)
    decoy[:] = 0.0
    decoy[:, :] = 999.0
    result2 = regenerate(summary)
    assert np.array_equal(result1, result2)


# --- Invariants restitués (masses 4x4 à 1e-10 relatif) -----------------------


@pytest.mark.parametrize("level", [1, 2, 3])
def test_invariants_masses_restituees_a_1e10_relatif(level):
    field = np.random.default_rng(6).random((64, 64)) * 4.0 + 0.1
    summary = summarize(field, level)
    regen = regenerate(summary)
    masses_vraies = _sous_masses(field)
    masses_regen = _sous_masses(regen)
    for mv, mr in zip(masses_vraies, masses_regen):
        rel = abs(mr - mv) / abs(mv) if mv != 0.0 else abs(mr)
        assert rel <= 1e-10, f"masse sous-domaine non restituée : vrai={mv} regen={mr}"


# --- Round-trip champ constant = exact ---------------------------------------


@pytest.mark.parametrize("level", [1, 2, 3])
def test_round_trip_champ_constant_est_exact(level):
    # Constante dyadique (4.0 = 2^2) : la moyenne par blocs (sommes/divisions par
    # puissances de 2) et le rescale (ratio exactement 1.0) sont alors bit-exacts
    # en IEEE754 -- pas de bruit d'arrondi résiduel à masquer par une tolérance.
    # (Une décimale arbitraire comme 3.7 laisse ~1 ULP de bruit sur une moyenne à
    # 64 termes : propriété universelle de l'arithmétique flottante, pas un défaut
    # de cette implémentation -- vérifié empiriquement avant d'écrire ce test.)
    field = np.full((64, 64), 4.0, dtype=np.float64)
    summary = summarize(field, level)
    regen = regenerate(summary)
    assert np.array_equal(regen, field)


def test_regenerate_output_shape_and_nonnegative():
    field = np.random.default_rng(7).normal(loc=1.0, scale=0.5, size=(64, 64))
    summary = summarize(field, level=2)
    regen = regenerate(summary)
    assert regen.shape == (64, 64)
    assert np.all(regen >= 0.0)
