"""Arc A / manche 1, famille 2 (quadtree) -- harnais de mesure M-A1/M-A2.

Contrat : « Arbitrage fork famille-vs-manche-2 » (PREREGISTRATION.md,
pocCascade2phys, commit `a8ed659`, section « Spec opérationnelle du
quadtree ») + « AMENDEMENT FAMILLE 2 » du plan (`docs/superpowers/plans/
arc-a-manche1.md`, commit `78fbe0c`), reproduits verbatim dans
`.superpowers/sdd/refond-task9-harnais-qt-brief.md`. Ce script MESURE et
STOCKE -- aucun verdict, aucune lecture d'attendu (`run_arcA_verdict_qt.py`
lit `measures_qt.npz` et prononce le verdict).

MÊME architecture que `scripts/run_arcA_measure.py` (famille 1, INTOUCHÉ) --
helpers RÉUTILISABLES importés de ce module (`_draw_rollout_pairs`,
`_center_rollout`, `_sanity_prefixe_tirage`, `_load_history`, `_max_carrier`,
et les constantes de grille SEEDS/L_LIST/L_LIST_CONTROL/JND_LIST/ARMS/
CAP_FLOATS_10PCT -- IDENTIQUES entre les deux familles, cf. brief). Seul
change l'AXE DES TAILLES : budgets {32, 64, 128, 256, 400, 1024, 2048}
(`src/summary_quadtree.py` : `summarize_qt`/`regenerate_qt`) remplace
ℓ ∈ {1, 2, 3} (`src/summary.py`).

Pour chaque (L ∈ {10,20,40,80}, seed ∈ {101..105}, budget ∈ BUDGETS), sur le
champ VRAI d'un bras :
  - **M-A1** (instantané) : Δχ(albedo(regenerate_qt(summarize_qt(champ_vrai,
    budget))), albedo(champ_vrai)) au point d'op S_HALF_OP, critère =
    "max_carrier" de `src.albedo.delta_chi` (INSTRUMENT INCHANGÉ).
  - **M-A2** (sous dynamique) : rollout d'UN épisode -- centre = celui de
    l'épisode L+1 de la même seed de forçage (`_center_rollout`, importé,
    NON réimplémenté) -- depuis s_regen(budget) ET depuis champ_vrai
    (`run_episode_trajectoire`, src/sediment.py) ; Δχ à CHAQUE snapshot entre
    les deux trajectoires d'albedo ; statistique = max le long du rollout.
  - `sous_jnd` = max(M-A1, M-A2) < JND, JND ∈ {2,3,4,5} % (calcul, pas verdict).

Trois bras -- même pipeline pour les trois, seul le CHAMP VRAI change (champ
VRAI = `s_L` du npz d'histoire pour v2 ; DÉRIVÉ de `s_L` pour les contrôles) :
  - `v2`   : champ vrai = s_L tel quel. Grille COMPLÈTE L ∈ {10,20,40,80} x
             seeds {101..105} x budgets {32,64,128,256,400,1024,2048}.
  - `ferm` (contrôle-fermable) : champ vrai =
             `regenerate_qt(summarize_qt(s_L, budget=32))` -- le champ vrai
             est déjà EXACTEMENT représentable par l'arbre à budget 32 (S∘R
             strict, cf. `src/summary_quadtree.py`) : tout budget >= 32 de la
             grille doit donc fermer EXACT (Δχ=0), et k* doit retenir le PLUS
             PETIT budget de la grille, soit 32 -- attendu lu par le verdict,
             NON vérifié ici. Grille L ∈ {10, 80} x seeds x budgets.
  - `shuf` (contrôle-infermable, définition INCHANGÉE vs famille 1) : champ
             vrai = permutation des 4096 cellules de s_L par
             `default_rng(9001 + 1000*L + seed).permutation(4096)`, reshape
             (64, 64). Grille L ∈ {10, 80} x seeds x budgets. Attendu (lu par
             le verdict) : k*=∞ partout -- non vérifié ici.

Centre du rollout, sanity de préfixe de tirage, chargement des histoires :
IMPORTÉS tels quels de `run_arcA_measure` (mêmes fonctions, mêmes garanties,
cf. docstrings de ce module -- pas réimplémentés ici).

Optimisation autorisée (physique bit-identique, cf. brief) : par (bras, L,
seed), le rollout depuis le champ VRAI est calculé UNE fois et partagé entre
les 7 budgets ; seul le rollout depuis s_regen(budget) est refait par budget
-- 8 rollouts par (bras, L, seed) au lieu de 14. Total attendu sur le run
complet : (4 L x 5 seeds [v2] + 2x2 L x 5 seeds [ferm+shuf]) x 8 rollouts
= 40 cellules x 8 = 320 rollouts ("~320 épisodes" du brief).

Exécution PHASÉE (impératif d'infrastructure -- avant-plan, timeout 600000,
PAS d'arrière-plan) :
  .venv/bin/python scripts/run_arcA_measure_qt.py --arm v2   --L 10
  .venv/bin/python scripts/run_arcA_measure_qt.py --arm v2   --L 20
  .venv/bin/python scripts/run_arcA_measure_qt.py --arm v2   --L 40
  .venv/bin/python scripts/run_arcA_measure_qt.py --arm v2   --L 80
  .venv/bin/python scripts/run_arcA_measure_qt.py --arm ferm --L 10
  .venv/bin/python scripts/run_arcA_measure_qt.py --arm ferm --L 80
  .venv/bin/python scripts/run_arcA_measure_qt.py --arm shuf --L 10
  .venv/bin/python scripts/run_arcA_measure_qt.py --arm shuf --L 80
  .venv/bin/python scripts/run_arcA_measure_qt.py --assemble

`--only-seed N` (optionnel) restreint une phase à une seule seed -- usage :
vérifier le déterminisme par double run bit-à-bit d'une cellule témoin.
N'écrit PAS le fichier-partie canonique consommé par `--assemble` (suffixe
`_only{seed}_check`), pour ne jamais corrompre silencieusement la grille réelle.

Sorties :
  - `outputs/arcA/measures_qt_parts/{arm}_L{L}.npz` (intermédiaires) :
    ma1/ma2 shape (nSeed, nBudget), seeds, budgets, centres de rollout (audit).
  - `outputs/arcA/measures_qt.npz` (assemblage final) : par bras,
    `ma1_{arm}`/`ma2_{arm}` shape (nL, nSeed, nBudget) float64 (max_carrier de
    delta_chi), `sous_jnd_{arm}` shape (nL, nSeed, nBudget, nJND) bool. NaN
    pour les cellules hors grille d'un bras (ferm/shuf à L ∈ {20,40}) ;
    `sous_jnd` vaut False (pas NaN, dtype bool) à ces cellules -- convention
    "indéfini = False", documentée ici, non interprétée (IDENTIQUE famille 1).
    `meta_json` : L, seeds, budgets, JND, S_HALF_OP, cap 409.6, versions.

Aucun verdict n'est prononcé ni imprimé par ce script : les tableaux imprimés
sont un dump d'audit humain des mesures, sans lecture des attendus des
contrôles (`run_arcA_verdict_qt.py` seul lit `measures_qt.npz` et prononce
k*(L) / PASS-FAIL / issue mappée).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.run_arcA_measure import (ARMS, CAP_FLOATS_10PCT, JND_LIST,
                                      L_LIST, L_LIST_CONTROL, SEEDS,
                                      _center_rollout, _load_history,
                                      _max_carrier, _sanity_prefixe_tirage,
                                      _versions)
from scripts.run_arcA_revalidate import B0, GRID, PARAMS, RELIEF, S_HALF_OP
from src.albedo import albedo
from src.sediment import run_episode_trajectoire
from src.summary_quadtree import regenerate_qt, summarize_qt

HIST_DIR = ROOT / "outputs" / "arcA" / "histories"
OUT_DIR = ROOT / "outputs" / "arcA"
PARTS_DIR = OUT_DIR / "measures_qt_parts"
MEASURES_PATH = OUT_DIR / "measures_qt.npz"

# --- Grille de mesure (IDENTIQUE famille 1 pour L/seeds/JND/bras, importée
# ci-dessus -- SEUL l'axe des tailles change : budgets, pas niveaux) ---------
BUDGETS: tuple[int, ...] = (32, 64, 128, 256, 400, 1024, 2048)
_ARM_L_LIST: dict[str, tuple[int, ...]] = {
    "v2": L_LIST, "ferm": L_LIST_CONTROL, "shuf": L_LIST_CONTROL,
}
BUDGET_FERM: int = 32  # amendement famille 2 : R(S(s_L, 32)) -- pas de niveau 3


def _part_path(arm: str, L: int) -> Path:
    return PARTS_DIR / f"{arm}_L{L}.npz"


# --- Champ VRAI par bras (amendement famille 2) ------------------------------


def _field_true_qt(arm: str, L: int, seed: int, s_L: np.ndarray) -> np.ndarray:
    """Champ VRAI du bras : v2 = `s_L` tel quel ; ferm =
    `regenerate_qt(summarize_qt(s_L, budget=32))` (le plus petit budget de la
    grille -- amendement famille 2) ; shuf = permutation des 4096 cellules de
    `s_L` par `default_rng(9001 + 1000*L + seed).permutation(4096)`, reshape
    à la forme d'origine -- définition INCHANGÉE vs famille 1 (reproduite ici
    car `run_arcA_measure._field_true` bifurque sur `regenerate`/`summarize`
    de la famille 1 pour le bras ferm -- pas un pipeline qui s'importe
    proprement pour cette famille). Pour ferm/shuf, `s_L` désigne toujours le
    champ ORIGINAL du npz d'histoire, construit UNE fois par (bras, L, seed),
    en amont de la boucle sur les budgets qui applique ENSUITE le même
    pipeline summarize_qt/regenerate_qt à ce champ vrai (identique au bras
    v2 -- c'est la définition du contrôle)."""
    if arm == "v2":
        return s_L
    if arm == "ferm":
        return regenerate_qt(summarize_qt(s_L, budget=BUDGET_FERM))
    if arm == "shuf":
        if s_L.size != 4096:
            raise RuntimeError(
                f"shuf : champ de taille {s_L.size} != 4096 attendu (grille "
                "64x64) -- protocole du contrôle non applicable tel quel.")
        rng = np.random.default_rng(9001 + 1000 * L + seed)
        perm = rng.permutation(4096)
        return s_L.flatten()[perm].reshape(s_L.shape)
    raise ValueError(f"bras inconnu : {arm!r} (attendu un de {ARMS}).")


# --- Mesures M-A1 / M-A2 (axe budgets) ---------------------------------------


def _measure_cell_qt(field_true: np.ndarray, seed: int, L: int
                     ) -> tuple[dict[int, float], dict[int, float], tuple[float, float]]:
    """M-A1 et M-A2 pour les 7 budgets, à un (bras, L, seed) déjà résolu en
    `field_true`. Le rollout depuis `field_true` (`run_episode_trajectoire`,
    MÊME centre pour les 7 budgets) est calculé UNE fois et partagé ; seul le
    rollout depuis s_regen(budget) est refait par budget -- 8 rollouts au
    lieu de 14 (optimisation autorisée, physique bit-identique, symétrique de
    `run_arcA_measure._measure_cell`). Retourne (ma1, ma2 par budget, centre
    du rollout utilisé -- pour audit)."""
    center = _center_rollout(seed, L)
    traj_true = run_episode_trajectoire(field_true, B0, center, PARAMS)
    a_true_traj = [albedo(s, S_HALF_OP) for s in traj_true]

    ma1: dict[int, float] = {}
    ma2: dict[int, float] = {}
    for budget in BUDGETS:
        s_regen = regenerate_qt(summarize_qt(field_true, budget))
        # M-A1 = Δχ(albedo(s_regen), albedo(field_true)) -- a_true_traj[0] est
        # bit-identique à albedo(field_true) (traj_true[0] = field_true, "s0
        # en tête" -- docstring run_episode_trajectoire), pas de recalcul.
        ma1[budget] = _max_carrier(albedo(s_regen, S_HALF_OP), a_true_traj[0])

        traj_regen = run_episode_trajectoire(s_regen, B0, center, PARAMS)
        if len(traj_regen) != len(traj_true):
            raise RuntimeError(
                f"M-A2 : longueurs de trajectoire différentes (regen="
                f"{len(traj_regen)}, vrai={len(traj_true)}) -- seed={seed}, "
                f"L={L}, budget={budget}. Incohérence de rollout, STOP.")
        vals = [_max_carrier(albedo(sr, S_HALF_OP), a_true_traj[i])
                for i, sr in enumerate(traj_regen)]
        ma2[budget] = float(max(vals))
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
    ma1_arr = np.full((len(seeds_to_run), len(BUDGETS)), np.nan, dtype=np.float64)
    ma2_arr = np.full((len(seeds_to_run), len(BUDGETS)), np.nan, dtype=np.float64)
    centers: list[tuple[float, float]] = []

    print(f"[{arm} L={L}] seeds={seeds_to_run}")
    for i, seed in enumerate(seeds_to_run):
        ts = time.time()
        hist = _load_history(seed)
        s_L = hist[f"s_L{L}"]
        field_true = _field_true_qt(arm, L, seed, s_L)
        ma1, ma2, center = _measure_cell_qt(field_true, seed, L)
        centers.append(center)
        for j, budget in enumerate(BUDGETS):
            ma1_arr[i, j] = ma1[budget]
            ma2_arr[i, j] = ma2[budget]
        ma1_fmt = {b: round(ma1[b], 5) for b in BUDGETS}
        ma2_fmt = {b: round(ma2[b], 5) for b in BUDGETS}
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
        seeds=np.array(seeds_to_run), budgets=np.array(BUDGETS),
        ma1=ma1_arr, ma2=ma2_arr,
        centers=np.array(centers, dtype=np.float64),
        elapsed_seconds=np.array(elapsed),
        versions_json=np.array(json.dumps(_versions())),
    )
    print(f"[{arm} L={L}] écrit -> {out_path}  [{elapsed:.1f}s total]")
    return out_path


# --- Phase d'assemblage (--assemble) -----------------------------------------


def phase_assemble() -> Path:
    nL, nSeed, nBudget, nJnd = len(L_LIST), len(SEEDS), len(BUDGETS), len(JND_LIST)
    data_out: dict[str, np.ndarray] = {}

    for arm in ARMS:
        ma1 = np.full((nL, nSeed, nBudget), np.nan, dtype=np.float64)
        ma2 = np.full((nL, nSeed, nBudget), np.nan, dtype=np.float64)
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
                part_budgets = [int(b) for b in d["budgets"]]
                if part_budgets != list(BUDGETS):
                    raise RuntimeError(
                        f"{path} : budgets {part_budgets} != {list(BUDGETS)}.")
                iL = L_LIST.index(L)
                ma1[iL] = d["ma1"]
                ma2[iL] = d["ma2"]

        max_val = np.maximum(ma1, ma2)          # NaN se propage où indéfini
        is_nan = np.isnan(max_val)
        sous_jnd = np.zeros((nL, nSeed, nBudget, nJnd), dtype=bool)
        for k, jnd in enumerate(JND_LIST):
            with np.errstate(invalid="ignore"):
                cond = max_val < jnd
            # NaN (hors grille du bras) -> False explicite (dtype bool ne
            # porte pas NaN) ; convention documentée dans le docstring module.
            sous_jnd[..., k] = np.where(is_nan, False, cond)

        data_out[f"ma1_{arm}"] = ma1
        data_out[f"ma2_{arm}"] = ma2
        data_out[f"sous_jnd_{arm}"] = sous_jnd

    # NOTE (documentée, ne pas "corriger") : contrairement à la famille 1 où
    # chaque niveau produit une taille FIXE (1041/273/81, indépendante du
    # champ), la famille 2 est ADAPTATIVE -- la taille RÉELLEMENT atteinte par
    # `summarize_qt(field, budget)` dépend du contenu du champ (<= budget,
    # souvent <) : il n'existe donc PAS de correspondance budget -> taille
    # fixe analogue à `tailles_floats` de la famille 1 (une taille calculée
    # sur un champ nul, par exemple, serait 2 pour TOUS les budgets -- non
    # représentative). Seul `budget` (le CAP passé à `summarize_qt`, dans les
    # mêmes unités floats-équivalents que `size_floats_qt`) est reporté ici.
    meta = dict(
        L=list(L_LIST),
        L_par_bras={arm: list(_ARM_L_LIST[arm]) for arm in ARMS},
        seeds=list(SEEDS),
        budgets=list(BUDGETS),
        jnd=list(JND_LIST),
        s_half_op=S_HALF_OP,
        relief=RELIEF,
        kd_calibre_v2=PARAMS.k_d,
        ke_calibre_v2=PARAMS.k_e,
        grid=dict(H=GRID.H, W=GRID.W, dx=GRID.dx, dy=GRID.dy),
        champ_fin_floats=GRID.H * GRID.W,
        cap_floats_10pct=CAP_FLOATS_10PCT,
        budget_ferm=BUDGET_FERM,
        versions=_versions(),
    )
    data_out["meta_json"] = np.array(json.dumps(meta, indent=2, ensure_ascii=False))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(MEASURES_PATH, **data_out)

    # -- Tableaux d'audit humain -- AUCUNE interprétation, AUCUN verdict.
    print("=" * 78)
    print("ARC A / FAMILLE 2 (QUADTREE) -- MESURES M-A1/M-A2 (assemblage, dump d'audit)")
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
                    f"b={budget}:ma1={ma1[iL, iS, iB]:.5f}/ma2={ma2[iL, iS, iB]:.5f}"
                    for iB, budget in enumerate(BUDGETS))
                print(f"  L={L:3d} seed={seed} : {cells}")
    print("-" * 78)
    for arm in ARMS:
        ma1 = data_out[f"ma1_{arm}"]
        n_nan = int(np.isnan(ma1).sum())
        n_total = ma1.size
        n_attendu_nan = (nL - len(_ARM_L_LIST[arm])) * nSeed * nBudget
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
                             "outputs/arcA/measures_qt_parts/ en measures_qt.npz.")
    args = parser.parse_args()

    if args.assemble:
        phase_assemble()
        return
    if args.arm is None or args.L is None:
        parser.error("préciser --arm {v2,ferm,shuf} --L N, ou --assemble")
    phase_measure(args.arm, args.L, only_seed=args.only_seed)


if __name__ == "__main__":
    main()
