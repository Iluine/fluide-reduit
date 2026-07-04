"""Arc A re-fondation / Task 3 -- harnais d'histoires (manche 1, relance sur v2).

Contrat : plan `docs/superpowers/plans/arc-a-manche1.md` Task 3 + section
AMENDEMENTS (relance sur v2, gravée au contrat `8e02dd6`). Extrait verbatim du
plan :

    Seeds de forçage : {101, 102, 103, 104, 105}. Histoires NESTED : un run de
    80 épisodes par seed, checkpoints s à L in {10, 20, 40, 80} (préfixes = les
    histoires courtes).
    Sauvegarde outputs/arcA/histories/h_s{seed}.npz : s aux 4 checkpoints, b0,
    params (asdict), seed, versions. Committé.
    Sanity dans le script : deux seeds -> f(readout) > 0 entre leurs s finaux
    (path-dependence au readout) ; relance d'un checkpoint -> bit-à-bit
    identique (reproductibilité).

Ce script est un HARNAIS MINCE : `run_history` de `src/sediment.py` fait déjà
tout le travail physique (pulse -> relaxation `simulate_wetdry_o2` -> intégration
Exner -> checkpoints), y compris la notion d'histoire NESTED -- les 4 checkpoints
{10,20,40,80} d'un même seed sont des PRÉFIXES de la MÊME trajectoire `rng(seed)`,
produits en un seul appel `run_history(seed, 80, b0, params, checkpoints={...})`.
Aucune boucle d'épisodes n'est réimplémentée ici, aucune fonction existante n'est
modifiée. Substrat v2 GELÉ : `SedimentParams()` par défaut (KD_CALIBRE_V2,
KE_CALIBRE_V2), `default_terrain` (b0 FIGÉ) -- tels quels, aucun nouveau paramètre.

Exécution PHASÉE par seed (impératif d'infrastructure -- un run ≈ 8 min CPU,
80 épisodes à ~4-5 s/épisode) : `--seed N` (N dans {101..105}) écrit
`outputs/arcA/histories/h_s{N}.npz` pour CETTE seed de forçage uniquement.
`--verify` (à lancer seulement une fois que les 5 npz existent) exécute les deux
sanity du plan :

  1. Path-dependence au readout -- entre les seeds 101 et 102, `f_ordre`
     (convention symétrique M-0, réutilisée telle quelle depuis
     `scripts/run_arcA_m0.py::_f_ordre`, PAS réimplémentée) entre
     albedo(s_L80) des deux seeds, au point d'opération S_HALF = 0.05*relief
     (convention Task 1 / M-0), JND = 5 % -> doit être > 0 : les deux histoires
     de forçage distinctes laissent un écart PERCEPTIBLE au readout (pas
     seulement un écart L2 sur l'état, interdit comme critère par le contrat).
  2. Reproductibilité -- relance `run_history(101, 10, b0, params, {10})`
     (10 épisodes, préfixe court, ~1 min CPU) -> le champ obtenu doit être
     bit-à-bit identique (`np.array_equal`, aucune tolérance) à `s_L10` du npz
     déjà écrit pour la seed 101 : même rng, même physique, aucune horloge dans
     le calcul -- seule garantie de déterminisme testée directement sur les
     artefacts committés (pas seulement supposée).

Échec d'un sanity = STOP : verdict BLOCKED imprimé + écrit dans `verify.json`,
rien n'est maquillé ni retenté avec une tolérance assouplie ou un seuil déplacé.

Usage (seeds SÉQUENTIELLEMENT, en avant-plan, jamais en parallèle -- le CPU est
le goulot) :
  .venv/bin/python scripts/run_arcA_histories.py --seed 101
  .venv/bin/python scripts/run_arcA_histories.py --seed 102
  .venv/bin/python scripts/run_arcA_histories.py --seed 103
  .venv/bin/python scripts/run_arcA_histories.py --seed 104
  .venv/bin/python scripts/run_arcA_histories.py --seed 105
  .venv/bin/python scripts/run_arcA_histories.py --verify

Sorties : `outputs/arcA/histories/h_s{seed}.npz` (un par seed, committé -- artefact
load-bearing consommé par les tâches de mesure/verdict suivantes) et
`outputs/arcA/histories/verify.json` (verdict des deux sanity + tableau imprimé).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from config import GridConfig
from scripts.run_arcA_m0 import _f_ordre
from src.albedo import albedo
from src.sediment import SedimentParams, default_terrain, run_history

OUT_DIR = ROOT / "outputs" / "arcA" / "histories"
VERIFY_PATH = OUT_DIR / "verify.json"

# Seeds de forçage du plan (PAS le seed 7 du protocole M-0) -- histoires NESTED
# de 80 épisodes, checkpoints aux 4 longueurs de préfixe demandées.
SEEDS: tuple[int, ...] = (101, 102, 103, 104, 105)
N_EPISODES: int = 80
CHECKPOINTS: tuple[int, ...] = (10, 20, 40, 80)

# Substrat v2 GELÉ (identique à `run_arcA_revalidate.py`/`run_arcA_m0.py`, non
# importé de là pour ne pas coupler ce harnais à un autre script -- même
# construction déterministe, valeurs bit-identiques).
GRID = GridConfig()
B0 = default_terrain(GRID)
RELIEF: float = float(B0.max() - B0.min())
PARAMS = SedimentParams()
S_HALF_OP: float = 0.05 * RELIEF  # point d'opération (convention Task 1 / M-0)

JND_SANITY: float = 0.05          # JND = 5 % (sanity §Task 3, readout path-dependence)
_SANITY_PAIR: tuple[int, int] = (101, 102)
_SANITY_REPLAY_SEED: int = 101
_SANITY_REPLAY_L: int = 10


def _npz_path(seed: int) -> Path:
    return OUT_DIR / f"h_s{seed}.npz"


def _versions() -> dict[str, str]:
    """Versions numpy + python, pour reproductibilité de l'artefact (pas du calcul :
    aucune branche de `run_history`/`_exner_step`/etc. ne dépend de ces versions)."""
    return dict(numpy=np.__version__, python=platform.python_version())


def run_one_seed(seed: int) -> Path:
    """Une histoire NESTED de 80 épisodes pour `seed` : un unique appel
    `run_history` produit les 4 checkpoints demandés (préfixes de la même rng).
    Écrit `outputs/arcA/histories/h_s{seed}.npz`."""
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
        b0=np.asarray(B0, dtype=np.float64),
        seed=np.array(seed),
        params_json=np.array(json.dumps(asdict(PARAMS))),
        versions_json=np.array(json.dumps(_versions())),
    )
    print(f"[seed={seed}] histoire NESTED (80 épisodes, checkpoints "
          f"{list(CHECKPOINTS)}) écrite -> {path}  [{elapsed:.1f}s]")
    return path


def _load_history_npz(seed: int) -> dict:
    with np.load(_npz_path(seed), allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def verify() -> dict:
    """Sanity du plan (§Task 3), à lancer une fois les 5 npz écrits : (1)
    path-dependence au readout albedo entre les seeds 101/102, (2) reproductibilité
    d'un checkpoint par relance de `run_history`. Écrit `verify.json`, lève
    SystemExit(1) si un sanity échoue (BLOCKED -- rien n'est maquillé)."""
    missing = [s for s in SEEDS if not _npz_path(s).exists()]
    if missing:
        raise RuntimeError(
            f"--verify : npz manquants pour les seeds {missing} -- lancer "
            "--seed pour chacun d'abord (les 5 histoires doivent exister).")

    data_a, data_b = (_load_history_npz(s) for s in _SANITY_PAIR)

    # --- Sanity 1 : path-dependence au readout (albedo, S_HALF op, JND=5%) ---
    a_readout = albedo(data_a["s_L80"], S_HALF_OP)
    b_readout = albedo(data_b["s_L80"], S_HALF_OP)
    f_val = _f_ordre(a_readout, b_readout, JND_SANITY)
    sanity_1_pass = f_val > 0.0

    # --- Sanity 2 : reproductibilité (relance run_history, bit-à-bit) ---
    t0 = time.time()
    hist_replay = run_history(_SANITY_REPLAY_SEED, _SANITY_REPLAY_L, B0, PARAMS,
                              checkpoints={_SANITY_REPLAY_L})
    elapsed_replay = time.time() - t0
    s_l10_original = _load_history_npz(_SANITY_REPLAY_SEED)[f"s_L{_SANITY_REPLAY_L}"]
    sanity_2_pass = bool(np.array_equal(hist_replay[_SANITY_REPLAY_L], s_l10_original))

    verdict = "PASS" if (sanity_1_pass and sanity_2_pass) else "BLOCKED"

    result = dict(
        meta=dict(
            seeds=list(SEEDS),
            n_episodes=N_EPISODES,
            checkpoints=list(CHECKPOINTS),
            grid=dict(H=GRID.H, W=GRID.W, dx=GRID.dx, dy=GRID.dy),
            relief=RELIEF,
            s_half_op=S_HALF_OP,
            kd_calibre_v2=PARAMS.k_d,
            ke_calibre_v2=PARAMS.k_e,
        ),
        sanity_1_path_dependence=dict(
            paire=list(_SANITY_PAIR),
            checkpoint="s_L80",
            jnd=JND_SANITY,
            s_half_op=S_HALF_OP,
            f_ordre=f_val,
            condition="f_ordre > 0",
            verdict="PASS" if sanity_1_pass else "FAIL",
        ),
        sanity_2_reproductibilite=dict(
            seed=_SANITY_REPLAY_SEED,
            checkpoint_relance=_SANITY_REPLAY_L,
            array_equal=sanity_2_pass,
            elapsed_seconds=elapsed_replay,
            condition="np.array_equal(replay, s_L{L} du npz) bit-a-bit",
            verdict="PASS" if sanity_2_pass else "FAIL",
        ),
        verdict=verdict,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VERIFY_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print("=" * 78)
    print("ARC A / TASK 3 -- VERIFY harnais d'histoires")
    print("=" * 78)
    print(f"[1] path-dependence readout : f_ordre(seeds={_SANITY_PAIR} @ s_L80, "
          f"S_HALF_op={S_HALF_OP:.4f}, JND=5%) = {f_val:.6f} "
          f"(condition > 0) -> {'PASS' if sanity_1_pass else 'FAIL'}")
    print(f"[2] reproductibilité : replay(seed={_SANITY_REPLAY_SEED}, "
          f"L={_SANITY_REPLAY_L}) == s_L{_SANITY_REPLAY_L} du npz -> "
          f"{'PASS' if sanity_2_pass else 'FAIL'}  [{elapsed_replay:.1f}s]")
    print("-" * 78)
    print(f"VERDICT -> {verdict}")
    print(f"[REPORT] -> {VERIFY_PATH}")

    if verdict != "PASS":
        raise SystemExit(
            f"BLOCKED : au moins un sanity a échoué ({result['sanity_1_path_dependence']['verdict']}, "
            f"{result['sanity_2_reproductibilite']['verdict']}) -- cf. {VERIFY_PATH}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Harnais d'histoires Arc A manche 1 Task 3 (substrat v2 gelé).")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=None,
                        help="Lance l'histoire NESTED de 80 épisodes pour cette "
                             "seed de forçage et écrit son npz.")
    parser.add_argument("--verify", action="store_true",
                        help="Sanity (path-dependence + reproductibilité), APRÈS "
                             "que les 5 npz existent.")
    args = parser.parse_args()

    if args.seed is None and not args.verify:
        parser.error("préciser --seed N (N dans {101..105}) ou --verify")

    if args.seed is not None:
        run_one_seed(args.seed)
    if args.verify:
        verify()


if __name__ == "__main__":
    main()
