"""Tests Task 4 (spec GRAVÉE, `docs/superpowers/plans/arc-a-manche1.md`, citée
verbatim dans `.superpowers/sdd/refond-task2-m0bis-brief.md`) : compresseur/
régénérateur `src/summary.py`. Champ fin = grille 64x64 (GridConfig par défaut) :
size_floats attendu 1041 (ℓ=1), 273 (ℓ=2), 81 (ℓ=3).

Amendement instrument (arbitrage `9bcb09a`, pocCascade2phys, gravé 2026-07-04) :
`regenerate` est désormais l'upsampling CONSTANT-PAR-BLOCS (seule projection
exacte de la famille sur un champ positif sparse avec clip) -- l'ancien
bilinéaire est conservé sous `regenerer_bilineaire` (audit seulement, plus
jamais utilisé par les mesures). Les tests génériques (M-A3, anti-fuite,
tailles, round-trip constant, invariants restitués) portent sur le NOUVEAU
`regenerate` ; un sous-ensemble minimal (M-A3 + tailles) est dupliqué sur
`regenerer_bilineaire` pour le garder vivant."""
from __future__ import annotations

import hashlib
import inspect
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.summary import Summary, regenerate, regenerer_bilineaire, size_floats, summarize

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


# --- Champs de test dédiés à la nouvelle spec (positifs, SPARSE + aléatoire) --


def _champ_sparse_positif(seed: int, taux_occupation: float = 0.05) -> np.ndarray:
    """Champ (64, 64) positif SPARSE (typique du substrat sédiment : majoritairement
    nul, quelques dépôts positifs épars) -- pas une simple constante."""
    rng = np.random.default_rng(seed)
    field = np.zeros((64, 64), dtype=np.float64)
    mask = rng.random((64, 64)) < taux_occupation
    field[mask] = rng.random(int(mask.sum())) * 3.0 + 0.05
    return field


def _champ_aleatoire_positif(seed: int) -> np.ndarray:
    """Champ (64, 64) positif dense, aléatoire (pas sparse) -- deuxième famille de
    champs de test exigée par la spec amendée."""
    return np.random.default_rng(seed).random((64, 64)) * 2.5 + 0.01


_CHAMPS_TEST_NOUVELLE_SPEC = {
    "sparse": _champ_sparse_positif(101),
    "aleatoire_positif": _champ_aleatoire_positif(202),
}


# --- S∘R = id bit-à-bit (NOUVEAU, spec amendée constant-par-blocs) -----------


@pytest.mark.parametrize("nom_champ", list(_CHAMPS_TEST_NOUVELLE_SPEC))
@pytest.mark.parametrize("level", [1, 2, 3])
def test_summarize_regenerate_summarize_identite_bit_exacte(nom_champ, level):
    """S∘R = id, spec amendée (constant-par-blocs = seule projection exacte de
    la famille sur un champ positif sparse avec clip) :
    - `coarse` de `summarize(regenerate(summarize(x, ℓ)), ℓ)` == `coarse` de
      `summarize(x, ℓ)` STRICT bit-à-bit (np.array_equal). Argument (gravé,
      arbitrage 9bcb09a) : chaque bloc régénéré est rempli d'une valeur
      UNIQUE (celle du coarse) ; ré-sommer/diviser N=2^(2ℓ) copies BIT-
      IDENTIQUES d'un même float64 par des additions successives par
      puissances de 2 (N est une puissance de 4, donc de 2) est une opération
      exacte en IEEE-754 -- aucun bruit d'arrondi ne peut apparaître.
    - Les invariants (masses), eux, sont recalculés en resommant les valeurs
      du champ RÉGÉNÉRÉ (block-constant) au lieu du champ D'ORIGINE (valeurs
      arbitraires) -- l'ORDRE de sommation (et les opérandes) diffèrent d'un
      calcul à l'autre, donc l'égalité stricte n'est PAS garantie : on la
      teste à 1e-12 relatif (documenté ICI, choix délibéré vs np.array_equal
      utilisé pour le coarse)."""
    field = _CHAMPS_TEST_NOUVELLE_SPEC[nom_champ]
    s1 = summarize(field, level)
    regen = regenerate(s1)
    s2 = summarize(regen, level)

    assert np.array_equal(s2.coarse, s1.coarse), (
        f"S∘R = id violé sur le coarse (champ={nom_champ}, level={level}) : "
        "la projection constant-par-blocs devait être exacte bit-à-bit.")

    for idx, (attendu, obtenu) in enumerate(zip(s1.invariants, s2.invariants)):
        if attendu == 0.0:
            assert abs(obtenu) <= 1e-12, (
                f"invariant[{idx}] attendu nul, obtenu {obtenu} (champ={nom_champ}, "
                f"level={level}).")
        else:
            rel = abs(obtenu - attendu) / abs(attendu)
            assert rel <= 1e-12, (
                f"invariant[{idx}] : écart relatif {rel} > 1e-12 (champ={nom_champ}, "
                f"level={level}, attendu={attendu}, obtenu={obtenu}).")


# --- Idempotence bit-à-bit R∘S∘R∘S = R∘S (NOUVEAU) ---------------------------


@pytest.mark.parametrize("nom_champ", list(_CHAMPS_TEST_NOUVELLE_SPEC))
@pytest.mark.parametrize("level", [1, 2, 3])
def test_regenerate_idempotence_bit_exacte(nom_champ, level):
    """`regenerate(summarize(regenerate(summarize(x, ℓ)), ℓ))` ==
    `regenerate(summarize(x, ℓ))`, STRICT (np.array_equal) -- conséquence
    directe de S∘R = id bit-à-bit sur le coarse (le second passage repart
    d'un coarse identique, donc régénère un champ identique)."""
    field = _CHAMPS_TEST_NOUVELLE_SPEC[nom_champ]
    r1 = regenerate(summarize(field, level))
    r2 = regenerate(summarize(r1, level))
    assert np.array_equal(r1, r2), (
        f"idempotence violée (champ={nom_champ}, level={level}).")


def test_regenerate_est_bien_constant_par_blocs():
    """Sanity directe (pas seulement via S∘R) : chaque bloc 2^level x 2^level
    du champ régénéré est EXACTEMENT constant, de valeur égale au coarse
    correspondant -- comportement caractéristique du constant-par-blocs,
    absent du bilinéaire."""
    field = _champ_sparse_positif(303)
    for level in (1, 2, 3):
        summary = summarize(field, level)
        regen = regenerate(summary)
        bs = 2 ** level
        Hc, Wc = summary.coarse.shape
        blocs = regen.reshape(Hc, bs, Wc, bs)
        for i in range(Hc):
            for j in range(Wc):
                bloc = blocs[i, :, j, :]
                assert np.all(bloc == summary.coarse[i, j]), (
                    f"bloc ({i},{j}) non constant à level={level}.")


def test_regenerate_leve_si_masse_incoherente():
    """La vérification (sans rescale) doit lever `AssertionError` si les
    invariants stockés dans le `Summary` sont incohérents avec le coarse (ici
    falsifiés à la main) -- prouve que la vérification est réellement
    exercée, pas un no-op silencieux."""
    field = _champ_sparse_positif(404)
    summary = summarize(field, level=2)
    invariants_casses = summary.invariants.copy()
    invariants_casses[1] += 10.0   # casse la masse du sous-domaine (0,0)
    summary_casse = Summary(coarse=summary.coarse, invariants=invariants_casses,
                            level=summary.level, shape=summary.shape)
    with pytest.raises(AssertionError):
        regenerate(summary_casse)


# --- Bilinéaire d'audit (`regenerer_bilineaire`) -- minimum vital -----------
# Garder vivant l'ancien régénérateur (reproductibilité M-0bis / run v1) sans
# dupliquer toute la suite : M-A3 (déterminisme) + tailles (forme/level).


def test_regenerer_bilineaire_double_regeneration_est_deterministe():
    field = np.random.default_rng(4).random((64, 64)) * 5.0
    summary = summarize(field, level=2)
    r1 = regenerer_bilineaire(summary)
    r2 = regenerer_bilineaire(summary)
    assert np.array_equal(r1, r2)


@pytest.mark.parametrize("level", [1, 2, 3])
def test_regenerer_bilineaire_output_shape_and_nonnegative(level):
    field = np.random.default_rng(7).random((64, 64)) * 3.0 + 0.1
    summary = summarize(field, level)
    regen = regenerer_bilineaire(summary)
    assert regen.shape == (64, 64)
    assert np.all(regen >= 0.0)
