"""Arc A / Task 5 -- Harnais de mesure M-A1/M-A2 (manche 1, relance sur v2).

Contrat : plan `docs/superpowers/plans/arc-a-manche1.md` Task 5 + section
AMENDEMENTS (contrôles synthétiques, gravés au contrat `8e02dd6`), reproduits
verbatim dans `.superpowers/sdd/refond-task5-measure-brief.md`. Ce script MESURE
et STOCKE -- aucun verdict, aucune lecture d'attendu (Task 6 lit `measures.npz`
et prononce le verdict k*(L)).

Pour chaque (L ∈ {10,20,40,80}, seed ∈ {101..105}, ℓ ∈ {1,2,3}), sur le champ
VRAI d'un bras :
  - **M-A1** (instantané) : Δχ(albedo(regenerate(summarize(champ_vrai, ℓ))),
    albedo(champ_vrai)) au point d'op S_HALF_OP, critère = "max_carrier" de
    `src.albedo.delta_chi`.
  - **M-A2** (sous dynamique) : rollout d'UN épisode -- centre = celui de
    l'épisode L+1 de la même seed de forçage (cf. `_center_rollout` ci-dessous)
    -- depuis s_regen(ℓ) ET depuis champ_vrai (`run_episode_trajectoire`,
    src/sediment.py) ; Δχ à CHAQUE snapshot entre les deux trajectoires
    d'albedo ; statistique = max le long du rollout.
  - `sous_jnd` = max(M-A1, M-A2) < JND, JND ∈ {2,3,4,5} % (calcul, pas verdict).

Trois bras (amendement (ii), remplace §A11) -- même pipeline pour les trois,
seul le CHAMP VRAI change :
  - `v2`   : champ vrai = s_L du npz d'histoire (Task 3). Grille COMPLÈTE
             L ∈ {10,20,40,80} x seeds {101..105} x ℓ ∈ {1,2,3}.
  - `ferm` (contrôle-fermable) : champ vrai = `regenerate(summarize(s_L, 3))`.
             Grille L ∈ {10, 80} x seeds x ℓ. Attendu (lu en Task 6) : k*=81
             partout -- non vérifié ici.
  - `shuf` (contrôle-infermable) : champ vrai = permutation des 4096 cellules
             de s_L par `default_rng(9001 + 1000*L + seed).permutation(4096)`,
             reshape (64, 64). Grille L ∈ {10, 80} x seeds x ℓ. Attendu (Task 6) :
             k*=∞ partout -- non vérifié ici.

Centre du rollout (épisode L+1 de la seed de forçage) -- reproduit VERBATIM le
motif de tirage de `run_history` (src/sediment.py) : `rng = default_rng(seed)`
puis, par épisode, `(float(rng.uniform(0.15,0.85)), float(rng.uniform(0.15,0.85)))`
(x puis y). On tire L+1 paires ; le centre du rollout est la (L+1)-ième. AUCUNE
autre primitive de tirage. Sanity `_sanity_prefixe_tirage` (gratuite, pas de
physique rejouée) : pour chaque seed, les L+1 paires tirées pour L préfixent
bit-à-bit celles tirées pour tout L' > L -- lève RuntimeError (pas un assert nu)
si violée, exécutée avant tout calcul de chaque phase `--arm/--L`.

Optimisation autorisée (physique bit-identique, cf. brief) : par (bras, L, seed),
le rollout depuis le champ VRAI est calculé UNE fois et partagé entre les 3 ℓ ;
seul le rollout depuis s_regen(ℓ) est refait par ℓ -- 4 rollouts par (bras, L,
seed) au lieu de 6.

Exécution PHASÉE (impératif d'infrastructure -- un appel ≈ 3-5 min, avant-plan,
timeout 600000, PAS d'arrière-plan) :
  .venv/bin/python scripts/run_arcA_measure.py --arm v2   --L 10
  .venv/bin/python scripts/run_arcA_measure.py --arm v2   --L 20
  .venv/bin/python scripts/run_arcA_measure.py --arm v2   --L 40
  .venv/bin/python scripts/run_arcA_measure.py --arm v2   --L 80
  .venv/bin/python scripts/run_arcA_measure.py --arm ferm --L 10
  .venv/bin/python scripts/run_arcA_measure.py --arm ferm --L 80
  .venv/bin/python scripts/run_arcA_measure.py --arm shuf --L 10
  .venv/bin/python scripts/run_arcA_measure.py --arm shuf --L 80
  .venv/bin/python scripts/run_arcA_measure.py --assemble

`--only-seed N` (optionnel) restreint une phase à une seule seed -- usage :
vérifier le déterminisme par double run bit-à-bit d'une cellule témoin (ex.
`--arm v2 --L 10 --only-seed 101`, lancé deux fois, comparer les npz produits).
N'écrit PAS le fichier-partie canonique consommé par `--assemble` (suffixe
`_only{seed}_check`), pour ne jamais corrompre silencieusement la grille réelle.

Sorties :
  - `outputs/arcA/measures_parts/{arm}_L{L}.npz` (intermédiaires, committables) :
    ma1/ma2 shape (nSeed, nLevel), seeds, levels, centres de rollout (audit).
  - `outputs/arcA/measures.npz` (assemblage final, committé) : par bras,
    `ma1_{arm}`/`ma2_{arm}` shape (nL, nSeed, nℓ) float64 (max_carrier de
    delta_chi), `sous_jnd_{arm}` shape (nL, nSeed, nℓ, nJND) bool. NaN pour les
    cellules hors grille d'un bras (ferm/shuf à L ∈ {20,40}) ; `sous_jnd` vaut
    False (pas NaN, dtype bool) à ces cellules -- convention "indéfini = False",
    documentée ici, non interprétée. `meta_json` : L, seeds, niveaux, JND,
    S_HALF_OP, tailles floats {1041,273,81}, cap 409.6 (10% de 4096, pour Task 6),
    versions.

Aucun verdict n'est prononcé ni imprimé par ce script : les tableaux imprimés
sont un dump d'audit humain des mesures, sans lecture des attendus des
contrôles (Task 6 seul lit `measures.npz` et prononce k*(L) / PASS-FAIL).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.run_arcA_revalidate import B0, GRID, PARAMS, RELIEF, S_HALF_OP
from src.albedo import albedo, delta_chi
from src.sediment import run_episode_trajectoire
from src.summary import regenerate, size_floats, summarize

HIST_DIR = ROOT / "outputs" / "arcA" / "histories"
OUT_DIR = ROOT / "outputs" / "arcA"
PARTS_DIR = OUT_DIR / "measures_parts"
MEASURES_PATH = OUT_DIR / "measures.npz"

# --- Grille de mesure (plan Task 5 + amendement (ii)) -----------------------
SEEDS: tuple[int, ...] = (101, 102, 103, 104, 105)
L_LIST: tuple[int, ...] = (10, 20, 40, 80)           # grille bras v2 (complète)
L_LIST_CONTROL: tuple[int, ...] = (10, 80)           # grille bras ferm/shuf
LEVELS: tuple[int, ...] = (1, 2, 3)
JND_LIST: tuple[float, ...] = (0.02, 0.03, 0.04, 0.05)
ARMS: tuple[str, ...] = ("v2", "ferm", "shuf")
_ARM_L_LIST: dict[str, tuple[int, ...]] = {
    "v2": L_LIST, "ferm": L_LIST_CONTROL, "shuf": L_LIST_CONTROL,
}
# Cap 10% de 4096 floats (utilisé par le verdict k*(80) de Task 6 -- reporté
# ici en méta uniquement, aucun usage de verdict dans CE script).
CAP_FLOATS_10PCT: float = 409.6


def _part_path(arm: str, L: int) -> Path:
    return PARTS_DIR / f"{arm}_L{L}.npz"


def _versions() -> dict[str, str]:
    return dict(numpy=np.__version__, python=platform.python_version())


# --- Motif de tirage des centres (verbatim run_history, src/sediment.py) ----


def _draw_rollout_pairs(seed: int, n_pairs: int) -> list[tuple[float, float]]:
    """Réplique VERBATIM le motif de tirage de `run_history` (src/sediment.py,
    boucle `for ep in range(1, n_episodes+1): center = (rng.uniform(0.15,0.85),
    rng.uniform(0.15,0.85))`) : `rng = default_rng(seed)` FRAIS à chaque appel
    (pas de continuation d'état -- permet la sanity de préfixe ci-dessous),
    puis `n_pairs` tirages (x, y) indépendants uniformes dans [0.15, 0.85]²
    (x puis y par paire). AUCUNE autre primitive de tirage."""
    rng = np.random.default_rng(seed)
    return [(float(rng.uniform(0.15, 0.85)), float(rng.uniform(0.15, 0.85)))
            for _ in range(n_pairs)]


def _center_rollout(seed: int, L: int) -> tuple[float, float]:
    """Centre du pulse de l'épisode L+1 (même seed de forçage que l'histoire
    tronquée à L) : la (L+1)-ième paire du motif de tirage de `run_history`."""
    return _draw_rollout_pairs(seed, L + 1)[-1]


def _sanity_prefixe_tirage() -> None:
    """Sanity GRATUITE (brief §Task 5 -- rejouer l'histoire complète serait
    trop cher) : pour chaque seed, les L+1 paires tirées pour L doivent
    préfixer BIT-À-BIT les paires tirées pour tout L' > L de `L_LIST` --
    cohérence interne du motif de tirage, sans rejouer aucune physique. Lève
    RuntimeError (pas un assert nu) si violée : STOP immédiat, protocole de
    tirage rompu."""
    L_max = max(L_LIST)
    for seed in SEEDS:
        pairs_max = _draw_rollout_pairs(seed, L_max + 1)
        for L in L_LIST:
            pairs_L = _draw_rollout_pairs(seed, L + 1)
            if pairs_L != pairs_max[: L + 1]:
                raise RuntimeError(
                    f"Sanity de tirage rompue : seed={seed}, L={L} -- les {L + 1} "
                    f"paires tirées ne préfixent pas bit-à-bit le tirage à "
                    f"L_max={L_max}. Motif de tirage incohérent -- STOP, ne pas "
                    "poursuivre le calcul.")


# --- Chargement des histoires (Task 3) --------------------------------------


def _load_history(seed: int) -> dict:
    path = HIST_DIR / f"h_s{seed}.npz"
    if not path.exists():
        raise RuntimeError(
            f"Histoire manquante pour seed={seed} : {path} -- Task 3 "
            "(scripts/run_arcA_histories.py) doit être exécutée avant Task 5.")
    with np.load(path, allow_pickle=False) as data:
        out = {k: data[k] for k in data.files}
    if not np.array_equal(out["b0"], B0):
        raise RuntimeError(
            f"seed={seed} : b0 du npz d'histoire diffère de B0 "
            "(scripts/run_arcA_revalidate.py) -- substrat incohérent, STOP.")
    return out


# --- Champ VRAI par bras (amendement (ii)) ----------------------------------


def _field_true(arm: str, L: int, seed: int, s_L: np.ndarray) -> np.ndarray:
    """Champ VRAI du bras : v2 = `s_L` tel quel ; ferm =
    `regenerate(summarize(s_L, 3))` ; shuf = permutation des 4096 cellules de
    `s_L` par `default_rng(9001 + 1000*L + seed).permutation(4096)`, reshape
    à la forme d'origine. Pour ferm/shuf, `s_L` désigne toujours le champ
    ORIGINAL du npz d'histoire (pas un résultat intermédiaire d'un autre
    niveau) -- construit UNE fois par (bras, L, seed), en amont de la boucle
    sur ℓ qui applique ENSUITE le même pipeline summarize/regenerate à ce
    champ vrai (identique au bras v2 -- c'est la définition du contrôle)."""
    if arm == "v2":
        return s_L
    if arm == "ferm":
        return regenerate(summarize(s_L, 3))
    if arm == "shuf":
        if s_L.size != 4096:
            raise RuntimeError(
                f"shuf : champ de taille {s_L.size} != 4096 attendu (grille "
                "64x64) -- protocole du contrôle non applicable tel quel.")
        rng = np.random.default_rng(9001 + 1000 * L + seed)
        perm = rng.permutation(4096)
        return s_L.flatten()[perm].reshape(s_L.shape)
    raise ValueError(f"bras inconnu : {arm!r} (attendu un de {ARMS}).")


# --- Mesures M-A1 / M-A2 -----------------------------------------------------


def _max_carrier(a_cand: np.ndarray, a_ref: np.ndarray) -> float:
    """Δχ(a_cand, a_ref)["max_carrier"] -- max sur les bandes porteuses de la
    référence (cf. `src.albedo.delta_chi`), en float Python natif (JSON/npz-safe)."""
    return float(delta_chi(a_cand, a_ref)["max_carrier"])


def _measure_cell(field_true: np.ndarray, seed: int, L: int
                  ) -> tuple[dict[int, float], dict[int, float], tuple[float, float]]:
    """M-A1 et M-A2 pour les 3 niveaux, à un (bras, L, seed) déjà résolu en
    `field_true`. Le rollout depuis `field_true` (`run_episode_trajectoire`,
    MÊME centre pour les 3 niveaux) est calculé UNE fois et partagé ; seul le
    rollout depuis s_regen(ℓ) est refait par niveau -- 4 rollouts au lieu de 6
    (optimisation autorisée, physique bit-identique). Retourne (ma1, ma2 par
    niveau, centre du rollout utilisé -- pour audit)."""
    center = _center_rollout(seed, L)
    traj_true = run_episode_trajectoire(field_true, B0, center, PARAMS)
    a_true_traj = [albedo(s, S_HALF_OP) for s in traj_true]

    ma1: dict[int, float] = {}
    ma2: dict[int, float] = {}
    for level in LEVELS:
        s_regen = regenerate(summarize(field_true, level))
        # M-A1 = Δχ(albedo(s_regen), albedo(field_true)) -- a_true_traj[0] est
        # bit-identique à albedo(field_true) (traj_true[0] = field_true, "s0
        # en tête" -- docstring run_episode_trajectoire), pas de recalcul.
        ma1[level] = _max_carrier(albedo(s_regen, S_HALF_OP), a_true_traj[0])

        traj_regen = run_episode_trajectoire(s_regen, B0, center, PARAMS)
        if len(traj_regen) != len(traj_true):
            raise RuntimeError(
                f"M-A2 : longueurs de trajectoire différentes (regen="
                f"{len(traj_regen)}, vrai={len(traj_true)}) -- seed={seed}, "
                f"L={L}, level={level}. Incohérence de rollout, STOP.")
        vals = [_max_carrier(albedo(sr, S_HALF_OP), a_true_traj[i])
                for i, sr in enumerate(traj_regen)]
        ma2[level] = float(max(vals))
    return ma1, ma2, center


# --- Phase de mesure (--arm/--L) ---------------------------------------------


def phase_measure(arm: str, L: int, only_seed: int | None = None) -> Path:
    valid_L = _ARM_L_LIST[arm]
    if L not in valid_L:
        raise RuntimeError(
            f"--arm {arm} --L {L} : L doit être dans {valid_L} pour ce bras.")
    _sanity_prefixe_tirage()

    seeds_to_run = [only_seed] if only_seed is not None else list(SEEDS)
    t0 = time.time()
    ma1_arr = np.full((len(seeds_to_run), len(LEVELS)), np.nan, dtype=np.float64)
    ma2_arr = np.full((len(seeds_to_run), len(LEVELS)), np.nan, dtype=np.float64)
    centers: list[tuple[float, float]] = []

    print(f"[{arm} L={L}] seeds={seeds_to_run}")
    for i, seed in enumerate(seeds_to_run):
        ts = time.time()
        hist = _load_history(seed)
        s_L = hist[f"s_L{L}"]
        field_true = _field_true(arm, L, seed, s_L)
        ma1, ma2, center = _measure_cell(field_true, seed, L)
        centers.append(center)
        for j, level in enumerate(LEVELS):
            ma1_arr[i, j] = ma1[level]
            ma2_arr[i, j] = ma2[level]
        ma1_fmt = {lv: round(ma1[lv], 5) for lv in LEVELS}
        ma2_fmt = {lv: round(ma2[lv], 5) for lv in LEVELS}
        print(f"  seed={seed}  centre_rollout=({center[0]:.4f},{center[1]:.4f})  "
              f"ma1={ma1_fmt}  ma2={ma2_fmt}  [{time.time() - ts:.1f}s]")
    elapsed = time.time() - t0

    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    if only_seed is not None:
        out_path = PARTS_DIR / f"{arm}_L{L}_only{only_seed}_check.npz"
    else:
        out_path = _part_path(arm, L)
    np.savez_compressed(
        out_path,
        arm=np.array(arm), L=np.array(L),
        seeds=np.array(seeds_to_run), levels=np.array(LEVELS),
        ma1=ma1_arr, ma2=ma2_arr,
        centers=np.array(centers, dtype=np.float64),
        elapsed_seconds=np.array(elapsed),
        versions_json=np.array(json.dumps(_versions())),
    )
    print(f"[{arm} L={L}] écrit -> {out_path}  [{elapsed:.1f}s total]")
    return out_path


# --- Phase d'assemblage (--assemble) -----------------------------------------


def phase_assemble() -> Path:
    nL, nSeed, nLevel, nJnd = len(L_LIST), len(SEEDS), len(LEVELS), len(JND_LIST)
    data_out: dict[str, np.ndarray] = {}

    for arm in ARMS:
        ma1 = np.full((nL, nSeed, nLevel), np.nan, dtype=np.float64)
        ma2 = np.full((nL, nSeed, nLevel), np.nan, dtype=np.float64)
        for L in _ARM_L_LIST[arm]:
            path = _part_path(arm, L)
            if not path.exists():
                raise RuntimeError(
                    f"--assemble : partie manquante {path} -- lancer "
                    f"'--arm {arm} --L {L}' d'abord.")
            with np.load(path, allow_pickle=False) as d:
                part_seeds = [int(s) for s in d["seeds"]]
                if part_seeds != list(SEEDS):
                    raise RuntimeError(
                        f"{path} : seeds {part_seeds} != grille attendue "
                        f"{list(SEEDS)} -- partie incomplète ou produite avec "
                        "--only-seed (ne consomme pas ce fichier).")
                part_levels = [int(lv) for lv in d["levels"]]
                if part_levels != list(LEVELS):
                    raise RuntimeError(
                        f"{path} : niveaux {part_levels} != {list(LEVELS)}.")
                iL = L_LIST.index(L)
                ma1[iL] = d["ma1"]
                ma2[iL] = d["ma2"]

        max_val = np.maximum(ma1, ma2)          # NaN se propage où indéfini
        is_nan = np.isnan(max_val)
        sous_jnd = np.zeros((nL, nSeed, nLevel, nJnd), dtype=bool)
        for k, jnd in enumerate(JND_LIST):
            with np.errstate(invalid="ignore"):
                cond = max_val < jnd
            # NaN (hors grille du bras) -> False explicite (dtype bool ne
            # porte pas NaN) ; convention documentée dans le docstring module.
            sous_jnd[..., k] = np.where(is_nan, False, cond)

        data_out[f"ma1_{arm}"] = ma1
        data_out[f"ma2_{arm}"] = ma2
        data_out[f"sous_jnd_{arm}"] = sous_jnd

    tailles_floats = {
        str(level): size_floats(summarize(np.zeros((GRID.H, GRID.W)), level))
        for level in LEVELS
    }
    meta = dict(
        L=list(L_LIST),
        L_par_bras={arm: list(_ARM_L_LIST[arm]) for arm in ARMS},
        seeds=list(SEEDS),
        niveaux=list(LEVELS),
        jnd=list(JND_LIST),
        s_half_op=S_HALF_OP,
        relief=RELIEF,
        kd_calibre_v2=PARAMS.k_d,
        ke_calibre_v2=PARAMS.k_e,
        grid=dict(H=GRID.H, W=GRID.W, dx=GRID.dx, dy=GRID.dy),
        tailles_floats=tailles_floats,
        champ_fin_floats=GRID.H * GRID.W,
        cap_floats_10pct=CAP_FLOATS_10PCT,
        versions=_versions(),
    )
    data_out["meta_json"] = np.array(json.dumps(meta, indent=2, ensure_ascii=False))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(MEASURES_PATH, **data_out)

    # -- Tableaux d'audit humain -- AUCUNE interprétation, AUCUN verdict.
    print("=" * 78)
    print("ARC A / TASK 5 -- MESURES M-A1/M-A2 (assemblage, dump d'audit)")
    print("=" * 78)
    for arm in ARMS:
        ma1 = data_out[f"ma1_{arm}"]
        ma2 = data_out[f"ma2_{arm}"]
        print(f"-- bras {arm} (L couverts : {list(_ARM_L_LIST[arm])}) --")
        for iL, L in enumerate(L_LIST):
            if L not in _ARM_L_LIST[arm]:
                print(f"  L={L:3d} : hors grille de ce bras (NaN)")
                continue
            for iS, seed in enumerate(SEEDS):
                cells = "  ".join(
                    f"l={level}:ma1={ma1[iL, iS, iLv]:.5f}/ma2={ma2[iL, iS, iLv]:.5f}"
                    for iLv, level in enumerate(LEVELS)
                )
                print(f"  L={L:3d} seed={seed} : {cells}")
    print("-" * 78)
    for arm in ARMS:
        ma1 = data_out[f"ma1_{arm}"]
        n_nan = int(np.isnan(ma1).sum())
        n_total = ma1.size
        n_attendu_nan = (nL - len(_ARM_L_LIST[arm])) * nSeed * nLevel
        print(f"bras {arm} : NaN dans ma1 = {n_nan}/{n_total} (attendu {n_attendu_nan})")
    print(f"[REPORT] -> {MEASURES_PATH}")
    return MEASURES_PATH


# --- CLI ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=ARMS, default=None,
                        help="Bras à mesurer (v2, ferm, shuf).")
    parser.add_argument("--L", type=int, default=None,
                        help="Longueur d'histoire (10/20/40/80 pour v2 ; 10/80 "
                             "pour ferm/shuf).")
    parser.add_argument("--only-seed", type=int, default=None, choices=SEEDS,
                        help="Restreint la phase à cette seule seed -- usage "
                             "déterminisme (double run bit-à-bit) ; n'écrit PAS "
                             "le fichier-partie canonique consommé par --assemble.")
    parser.add_argument("--assemble", action="store_true",
                        help="Fusionne les intermédiaires de "
                             "outputs/arcA/measures_parts/ en measures.npz.")
    args = parser.parse_args()

    if args.assemble:
        phase_assemble()
        return
    if args.arm is None or args.L is None:
        parser.error("préciser --arm {v2,ferm,shuf} --L N, ou --assemble")
    phase_measure(args.arm, args.L, only_seed=args.only_seed)


if __name__ == "__main__":
    main()
