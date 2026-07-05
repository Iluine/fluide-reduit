"""Arc A / extension 16L₀ -- tests du Task 1 (histoires L=160), cf. brief
`.superpowers/sdd/16L0-task1-brief.md`. RAPIDES -- AUCUN run complet de 160
épisodes ici (seul `test_replay_prefixe_bit_identique` lance un run réel,
préfixe COURT L=10, ~1 min CPU -- coût accepté, c'est le sanity 2 de la
manche 1 rejoué à l'identique)."""
from __future__ import annotations

import numpy as np

import scripts.run_arcA_histories as m1
import scripts.run_arcA_histories_16L0 as m16
from src.sediment import run_history


# =============================================================================
# Configuration : constantes RÉUTILISÉES par import, jamais recalculées
# =============================================================================


def test_config_checkpoints():
    assert m16.N_EPISODES == 160
    assert m16.CHECKPOINTS == (10, 20, 40, 80, 160)

    assert m16._npz_path(101).name == "h160_s101.npz"
    assert m16._npz_path_original(101).name == "h_s101.npz"
    assert m16._npz_path(101).parent == m16.OUT_DIR

    # Identité d'objet (PAS une copie recalculée) avec le module manche 1.
    assert m16.B0 is m1.B0
    assert m16.PARAMS is m1.PARAMS
    assert m16.S_HALF_OP is m1.S_HALF_OP
    assert m16.GRID is m1.GRID
    assert m16.RELIEF is m1.RELIEF
    assert m16.SEEDS is m1.SEEDS
    assert m16.OUT_DIR is m1.OUT_DIR
    assert m16._f_ordre is m1._f_ordre


# =============================================================================
# Preuve de non-régression sur le replay réel (~1 min CPU, coût accepté)
# =============================================================================


def test_replay_prefixe_bit_identique():
    """Sanity 2 de la manche 1 rejoué à l'identique : `run_history(101, 10,
    B0, PARAMS, {10})` (10 épisodes, ~1 min CPU) doit être bit-à-bit
    identique à `s_L10` de l'ORIGINAL `h_s101.npz` -- même rng, même
    physique déterministe, aucune horloge. Preuve directe (pas supposée) que
    le run 160 reproduira ces mêmes préfixes."""
    original = m16._load_history_npz(m16._npz_path_original(101))["s_L10"]

    replay = run_history(101, 10, m16.B0, m16.PARAMS, checkpoints={10})

    assert np.array_equal(replay[10], original)


# =============================================================================
# Fonction de comparaison bit-à-bit -- test UNITAIRE, données synthétiques
# =============================================================================


def test_verify_detecte_mismatch():
    a = np.arange(16, dtype=np.float64).reshape(4, 4)
    b_altere = a.copy()
    b_altere[0, 0] += 1.0  # altération délibérée, un seul bit de différence

    data_nouveau = dict(s_L10=a, s_L20=a, s_L40=a, s_L80=a)
    data_original_ok = dict(s_L10=a.copy(), s_L20=a.copy(), s_L40=a.copy(), s_L80=a.copy())
    data_original_altere = dict(s_L10=b_altere, s_L20=a.copy(), s_L40=a.copy(), s_L80=a.copy())

    resultat_ok = m16._compare_checkpoints_bit_a_bit(data_nouveau, data_original_ok)
    resultat_fail = m16._compare_checkpoints_bit_a_bit(data_nouveau, data_original_altere)

    assert resultat_ok["verdict"] == "PASS"
    assert all(resultat_ok["detail"].values())

    assert resultat_fail["verdict"] == "BLOCKED"
    assert resultat_fail["detail"]["s_L10"] is False
    assert resultat_fail["detail"]["s_L20"] is True  # les autres checkpoints, eux, concordent


# =============================================================================
# Garde anti-écrasement des originaux manche 1
# =============================================================================


def test_ne_clobbe_pas_original(monkeypatch, tmp_path):
    """`run_one_seed` écrit exclusivement `h160_s{seed}.npz` -- jamais
    `h_s{seed}.npz` (les originaux manche 1). Vérifié fonctionnellement : le
    répertoire de sortie est redirigé vers un `tmp_path` vide, `run_history`
    est remplacé par une fabrique déterministe SANS simulation (aucun run
    réel), puis on constate que le SEUL fichier écrit porte le préfixe
    `h160_`."""
    def _run_history_factice(seed, n_episodes, b0, params, checkpoints=None):
        return {L: np.zeros_like(b0) for L in checkpoints}

    monkeypatch.setattr(m16, "OUT_DIR", tmp_path)
    monkeypatch.setattr(m16, "run_history", _run_history_factice)

    path_ecrit = m16.run_one_seed(101)

    fichiers = sorted(p.name for p in tmp_path.iterdir())
    assert fichiers == ["h160_s101.npz"]
    assert path_ecrit.name == "h160_s101.npz"
    assert not (tmp_path / "h_s101.npz").exists()
