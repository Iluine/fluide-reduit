"""Arc A / extension 16L₀ -- Task 1 : histoires L=160 (générateur + vérification
bit-identité).

Contrat : `PREREGISTRATION.md` (pocCascade2phys) §A12, cf. brief
`.superpowers/sdd/16L0-task1-brief.md`. `run_history` (`src/sediment.py`) est
GELÉ -- ce module l'IMPORTE et l'APPELLE avec `n_episodes=160`, ne le modifie
jamais.

Le fait qui dé-risque la tâche : les checkpoints de `run_history` sont des
PRÉFIXES de la MÊME trajectoire `default_rng(seed)` -- un run de 160 épisodes
traverse exactement les mêmes états à 10/20/40/80 qu'un run de 80 (même rng,
même physique déterministe, aucune horloge). Donc `s_L{10,20,40,80}` d'un run
à 160 sont bit-identiques PAR CONSTRUCTION aux npz manche 1 déjà commités
(`outputs/arcA/histories/h_s{seed}.npz`). Ce script PROUVE cette identité --
il ne l'espère pas.

Miroir strict de `scripts/run_arcA_histories.py` (Task 3 manche 1) : mêmes
constantes RÉUTILISÉES PAR IMPORT (jamais redéfinies), seule la longueur
d'histoire change (160 au lieu de 80) et un cinquième checkpoint apparaît
(160). Substrat v2 GELÉ (`SedimentParams()`, `default_terrain`), seeds
{101..105} -- identique à la manche 1.

Exécution PHASÉE par seed (même impératif d'infrastructure que la manche 1,
~2x plus long : un run de 160 épisodes ≈ 16 min CPU) :
  .venv/bin/python scripts/run_arcA_histories_16L0.py --seed 101
  .venv/bin/python scripts/run_arcA_histories_16L0.py --seed 102
  .venv/bin/python scripts/run_arcA_histories_16L0.py --seed 103
  .venv/bin/python scripts/run_arcA_histories_16L0.py --seed 104
  .venv/bin/python scripts/run_arcA_histories_16L0.py --seed 105
  .venv/bin/python scripts/run_arcA_histories_16L0.py --verify

`--verify` (une fois les 5 `h160_s{seed}.npz` écrits) exécute DEUX sanity,
échec de l'un ou l'autre = BLOCKED (SystemExit, rien n'est maquillé) :

  1. Non-régression bit-à-bit (LA garde centrale) -- pour CHAQUE seed,
     `s_L{10,20,40,80}` du npz 160 doit être `np.array_equal` (aucune
     tolérance) aux mêmes clés de l'ORIGINAL `h_s{seed}.npz` manche 1. Un
     seul désaccord, sur une seule seed, sur un seul checkpoint -> BLOCKED.
  2. Path-dependence au readout à L=160 -- `_f_ordre` (réutilisée telle
     quelle depuis `scripts.run_arcA_histories`, PAS réimplémentée) entre
     albedo(s_L160) des seeds 101/102, au point d'opération S_HALF_OP,
     JND=5% -> doit être > 0 : les deux histoires de forçage distinctes
     restent perceptiblement distinctes à la longueur 160 (pas seulement
     un écart L2 sur l'état, interdit comme critère par le contrat).

Sorties : `outputs/arcA/histories/h160_s{seed}.npz` (NOUVEAU, préfixe distinct
des originaux `h_s{seed}.npz` -- jamais touchés, ni en lecture-écriture ni en
écrasement) et `outputs/arcA/histories/verify_16L0.json`."""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.run_arcA_histories import (
    B0,
    GRID,
    JND_SANITY,
    OUT_DIR,
    PARAMS,
    RELIEF,
    S_HALF_OP,
    SEEDS,
    _f_ordre,
    _SANITY_PAIR,
    _versions,
)
from src.albedo import albedo
from src.sediment import run_history

VERIFY_PATH = OUT_DIR / "verify_16L0.json"

# Extension 16L₀ : mêmes seeds/substrat que la manche 1 (importés ci-dessus,
# jamais redéfinis), histoire deux fois plus longue + un cinquième checkpoint.
N_EPISODES: int = 160
CHECKPOINTS: tuple[int, ...] = (10, 20, 40, 80, 160)

# Checkpoints comparés au sanity 1 (non-régression) -- les 4 préfixes communs
# avec la manche 1 ; L=160 n'a pas d'équivalent dans les npz originaux.
_CHECKPOINTS_COMMUNS: tuple[int, ...] = (10, 20, 40, 80)

# Pas de sanity « reproductibilité » dans `verify()` (seulement les DEUX sanity
# du brief, ci-dessous) -- le replay court (~1 min CPU, seed=101, L=10) qui
# prouvait la reproductibilité manche 1 est ici rejoué directement dans
# `tests/test_arcA_histories_16L0.py::test_replay_prefixe_bit_identique`
# (preuve au coût minimal, hors de ce module).


def _npz_path(seed: int) -> Path:
    """Chemin du npz 160 épisodes -- préfixe `h160_`, JAMAIS `h_` (les
    originaux manche 1 ne sont ni lus en écriture, ni écrasés par ce module)."""
    return OUT_DIR / f"h160_s{seed}.npz"


def _npz_path_original(seed: int) -> Path:
    """Chemin de l'ORIGINAL manche 1 (`h_s{seed}.npz`) -- lu en LECTURE SEULE
    par `--verify` (sanity 1), jamais ouvert en écriture par ce module."""
    return OUT_DIR / f"h_s{seed}.npz"


def run_one_seed(seed: int) -> Path:
    """Une histoire NESTED de 160 épisodes pour `seed` : un unique appel
    `run_history` produit les 5 checkpoints demandés (préfixes de la même
    rng). Écrit `outputs/arcA/histories/h160_s{seed}.npz`."""
    t0 = time.time()
    hist = run_history(seed, N_EPISODES, B0, PARAMS, checkpoints=set(CHECKPOINTS))
    elapsed = time.time() - t0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = _npz_path(seed)
    np.savez_compressed(
        path,
        s_L10=np.asarray(hist[10], dtype=np.float64),
        s_L20=np.asarray(hist[20], dtype=np.float64),
        s_L40=np.asarray(hist[40], dtype=np.float64),
        s_L80=np.asarray(hist[80], dtype=np.float64),
        s_L160=np.asarray(hist[160], dtype=np.float64),
        b0=np.asarray(B0, dtype=np.float64),
        seed=np.array(seed),
        params_json=np.array(json.dumps(asdict(PARAMS))),
        versions_json=np.array(json.dumps(_versions())),
    )
    print(f"[seed={seed}] histoire NESTED (160 épisodes, checkpoints "
          f"{list(CHECKPOINTS)}) écrite -> {path}  [{elapsed:.1f}s]")
    return path


def _load_history_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _compare_checkpoints_bit_a_bit(
    data_nouveau: dict, data_original: dict,
    checkpoints: tuple[int, ...] = _CHECKPOINTS_COMMUNS,
) -> dict:
    """Cœur PUR (aucune I/O, aucune RNG) du sanity 1 : compare `s_L{L}` de deux
    dictionnaires de champs déjà chargés, `np.array_equal` (bit-exact, AUCUNE
    tolérance) checkpoint par checkpoint. Séparé de `_compare_prefixes_bit_a_bit`
    pour être testable UNITAIREMENT sur des données synthétiques (§tests)."""
    detail = {f"s_L{L}": bool(np.array_equal(data_nouveau[f"s_L{L}"],
                                             data_original[f"s_L{L}"]))
              for L in checkpoints}
    verdict = "PASS" if all(detail.values()) else "BLOCKED"
    return dict(detail=detail, verdict=verdict)


def _compare_prefixes_bit_a_bit(seed: int) -> dict:
    """Sanity 1 pour UNE seed : charge `h160_s{seed}.npz` (nouveau) et
    `h_s{seed}.npz` (ORIGINAL manche 1, lecture seule) puis délègue la
    comparaison au cœur pur ci-dessus."""
    data_nouveau = _load_history_npz(_npz_path(seed))
    data_original = _load_history_npz(_npz_path_original(seed))
    result = _compare_checkpoints_bit_a_bit(data_nouveau, data_original)
    result["seed"] = seed
    return result


def verify() -> dict:
    """Sanity 16L₀ (§Task 1), à lancer une fois les 5 npz 160 écrits :
    (1) non-régression bit-à-bit des 4 préfixes communs à la manche 1, POUR
    CHAQUE seed ; (2) path-dependence au readout albedo à L=160 entre les
    seeds 101/102. Écrit `verify_16L0.json`, lève SystemExit(1) si un sanity
    échoue (BLOCKED -- rien n'est maquillé)."""
    missing = [s for s in SEEDS if not _npz_path(s).exists()]
    if missing:
        raise RuntimeError(
            f"--verify : npz 160 manquants pour les seeds {missing} -- lancer "
            "--seed pour chacun d'abord (les 5 histoires doivent exister).")
    missing_originaux = [s for s in SEEDS if not _npz_path_original(s).exists()]
    if missing_originaux:
        raise RuntimeError(
            f"--verify : npz ORIGINAUX manche 1 manquants pour les seeds "
            f"{missing_originaux} -- le sanity 1 (non-régression) ne peut "
            "pas comparer sans eux.")

    # --- Sanity 1 : non-régression bit-à-bit, POUR CHAQUE seed ---
    par_seed = [_compare_prefixes_bit_a_bit(s) for s in SEEDS]
    sanity_1_pass = all(r["verdict"] == "PASS" for r in par_seed)

    # --- Sanity 2 : path-dependence au readout, à L=160 ---
    data_a = _load_history_npz(_npz_path(_SANITY_PAIR[0]))
    data_b = _load_history_npz(_npz_path(_SANITY_PAIR[1]))
    a_readout = albedo(data_a["s_L160"], S_HALF_OP)
    b_readout = albedo(data_b["s_L160"], S_HALF_OP)
    f_val = _f_ordre(a_readout, b_readout, JND_SANITY)
    sanity_2_pass = f_val > 0.0

    verdict = "PASS" if (sanity_1_pass and sanity_2_pass) else "BLOCKED"

    result = dict(
        meta=dict(
            seeds=list(SEEDS),
            n_episodes=N_EPISODES,
            checkpoints=list(CHECKPOINTS),
            checkpoints_communs_manche1=list(_CHECKPOINTS_COMMUNS),
            grid=dict(H=GRID.H, W=GRID.W, dx=GRID.dx, dy=GRID.dy),
            relief=RELIEF,
            s_half_op=S_HALF_OP,
            kd_calibre_v2=PARAMS.k_d,
            ke_calibre_v2=PARAMS.k_e,
        ),
        sanity_1_non_regression_bit_a_bit=dict(
            par_seed=par_seed,
            condition="np.array_equal(h160_s{seed}, h_s{seed}) pour L in "
                      f"{list(_CHECKPOINTS_COMMUNS)}, bit-a-bit, toutes seeds",
            verdict="PASS" if sanity_1_pass else "FAIL",
        ),
        sanity_2_path_dependence_L160=dict(
            paire=list(_SANITY_PAIR),
            checkpoint="s_L160",
            jnd=JND_SANITY,
            s_half_op=S_HALF_OP,
            f_ordre=f_val,
            condition="f_ordre > 0",
            verdict="PASS" if sanity_2_pass else "FAIL",
        ),
        verdict=verdict,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VERIFY_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print("=" * 78)
    print("ARC A / EXTENSION 16L0 -- TASK 1 -- VERIFY histoires L=160")
    print("=" * 78)
    for r in par_seed:
        print(f"[1] seed={r['seed']} non-régression bit-à-bit "
              f"{list(_CHECKPOINTS_COMMUNS)} -> {r['verdict']}")
    print(f"[2] path-dependence readout : f_ordre(seeds={_SANITY_PAIR} @ s_L160, "
          f"S_HALF_op={S_HALF_OP:.4f}, JND=5%) = {f_val:.6f} "
          f"(condition > 0) -> {'PASS' if sanity_2_pass else 'FAIL'}")
    print("-" * 78)
    print(f"VERDICT -> {verdict}")
    print(f"[REPORT] -> {VERIFY_PATH}")

    if verdict != "PASS":
        raise SystemExit(
            f"BLOCKED : au moins un sanity a échoué (non_regression="
            f"{'PASS' if sanity_1_pass else 'FAIL'}, path_dependence="
            f"{'PASS' if sanity_2_pass else 'FAIL'}) -- cf. {VERIFY_PATH}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Histoires L=160 Arc A extension 16L0 Task 1 (substrat v2 gelé).")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=None,
                        help="Lance l'histoire NESTED de 160 épisodes pour cette "
                             "seed de forçage et écrit son npz h160_.")
    parser.add_argument("--verify", action="store_true",
                        help="Sanity (non-régression bit-à-bit + path-dependence "
                             "à L=160), APRÈS que les 5 npz h160_ existent.")
    args = parser.parse_args()

    if args.seed is None and not args.verify:
        parser.error("préciser --seed N (N dans {101..105}) ou --verify")

    if args.seed is not None:
        run_one_seed(args.seed)
    if args.verify:
        verify()


if __name__ == "__main__":
    main()
