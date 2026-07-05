"""Arc A / extension 16L₀ -- Task 2 : mesures M-A1/M-A2 à L=160 + contrôles
ferm/shuf à L∈{10,160}, reproduit verbatim dans `.superpowers/sdd/
16L0-task2-brief.md`. Contrat : `PREREGISTRATION.md` (pocCascade2phys) §A12,
commit `3fd72aa`.

**Couche ADDITIVE.** `L_LIST=(10,20,40,80)` (`scripts.run_arcA_measure`) est
gelé et partagé par toute la chaîne manche 1 -- ce module ne le MUTE pas, ne
touche NI `measures_qt.npz`, NI `h_s{seed}.npz`, NI `h160_s{seed}.npz` (lus
en lecture seule), NI `run_arcA_measure*.py`. Il écrit un npz SÉPARÉ,
`outputs/arcA/measures_160.npz`, ne contenant QUE l'axe L=160 (bras v2) et
les contrôles aux bornes L∈{10,160} (bras ferm/shuf). Task 3 (verdict 16L₀)
agrafera ce npz à `measures_qt.npz`.

Instrument famille-2 (quadtree) GELÉ, réutilisé PAR IMPORT, jamais
réimplémenté : `_measure_cell_qt`/`_field_true_qt`/`BUDGETS`/`BUDGET_FERM`
(`scripts.run_arcA_measure_qt`) ; `SEEDS`/`JND_LIST`/`CAP_FLOATS_10PCT`/
`_versions` (`scripts.run_arcA_measure`) ; `sous_jnd_a_jnd`
(`scripts.run_arcC_surface_kstar_pins`, déjà verrouillée bit-à-bit dans
`tests/test_arcC_surface_kstar_pins.py`) ; `_npz_path` (`scripts.
run_arcA_histories_16L0`, Task 1 -- garantit le MÊME chemin `h160_s{seed}.npz`
que celui écrit par Task 1, sans dupliquer la convention de nommage).

Pour chaque seed ∈ {101..105}, charge l'histoire NESTED L=160 (`h160_s
{seed}.npz`, Task 1) puis mesure 5 cellules (mêmes M-A1/M-A2 que la manche 1,
cf. docstring `run_arcA_measure_qt._measure_cell_qt` -- non reproduit ici) :
  - **v2 à L=160** : champ vrai = `s_L160` tel quel.
  - **ferm à L∈{10,160}** : champ vrai = `regenerate_qt(summarize_qt(s_L,
    BUDGET_FERM))` (`_field_true_qt`), `s_L` = `s_L10` ou `s_L160` du MÊME
    npz `h160_`.
  - **shuf à L∈{10,160}** : champ vrai = permutation déterministe des 4096
    cellules de `s_L` (`_field_true_qt`, définition inchangée vs famille 1).

Sortie : `outputs/arcA/measures_160.npz` -- MÊME layout que `measures_qt.npz`
(axes L-en-tête, seed, budget[, JND]), mais L-grille RESTREINTE par bras :
  - `ma1_v2`/`ma2_v2` : (1, 5, 7), L-grille = [160].
  - `sous_jnd_v2` : (1, 5, 7, 4), aux 4 JND de `JND_LIST`.
  - `ma1_ferm`/`ma2_ferm`, `ma1_shuf`/`ma2_shuf` : (2, 5, 7), L-grille = [10, 160].
  - `sous_jnd_ferm`/`sous_jnd_shuf` : (2, 5, 7, 4).
  - `meta_json` : `L_v2`, `L_control`, `budgets`, `jnd`, `seeds`, `s_half_op`,
    `cap`, `budget_ferm`, `versions`. AUCUNE horloge/timestamp (déterminisme :
    un double run doit produire des arrays identiques).

Aucun verdict, aucune pente, aucun k* : uniquement les mesures brutes ma1/ma2
+ sous_jnd (même statut que `measures_qt.npz`) -- Task 3 lit ce npz et
prononce le verdict 16L₀.

Coût : ~8 rollouts/cellule (optimisation partagée de `_measure_cell_qt`,
identique famille 2) x 25 cellules (5 v2 + 10 ferm + 10 shuf) -- run complet
~30 min, lancé par le contrôleur en arrière-plan, PAS depuis les tests.

Usage :
  .venv/bin/python scripts/run_arcA_measure_16L0.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.run_arcA_histories_16L0 import _npz_path as _npz_path_h160
from scripts.run_arcA_measure import CAP_FLOATS_10PCT, JND_LIST, SEEDS, _versions
from scripts.run_arcA_measure_qt import (BUDGET_FERM, BUDGETS, _field_true_qt,
                                         _measure_cell_qt)
from scripts.run_arcA_revalidate import B0, S_HALF_OP
from scripts.run_arcC_surface_kstar_pins import sous_jnd_a_jnd

OUT_DIR = ROOT / "outputs" / "arcA"
MEASURES_160_PATH = OUT_DIR / "measures_160.npz"

# --- Grille de mesure de l'extension (§A12) ----------------------------------
L_V2: tuple[int, ...] = (160,)          # bras v2 : nouvel axe seul
L_CONTROL: tuple[int, ...] = (10, 160)  # bras ferm/shuf : bornes inter-écrans


# --- Chargement de l'histoire L=160 (Task 1, lecture seule) ------------------


def _load_history_160(seed: int) -> dict:
    """Charge `h160_s{seed}.npz` (Task 1) -- lecture seule, jamais réécrit ni
    déplacé. `_npz_path_h160` (importé de `run_arcA_histories_16L0`) garantit
    le MÊME chemin que celui utilisé par Task 1 pour écrire ce fichier. Sanity
    substrat (miroir de `run_arcA_measure._load_history`) : `b0` du npz doit
    être bit-identique à `B0` (`run_arcA_revalidate`, substrat v2 gelé)."""
    path = _npz_path_h160(seed)
    if not path.exists():
        raise RuntimeError(
            f"Histoire L=160 manquante pour seed={seed} : {path} -- Task 1 "
            "(scripts/run_arcA_histories_16L0.py --seed puis --verify) doit "
            "avoir tourné avant cette mesure. Rien n'est fabriqué : BLOCKED.")
    with np.load(path, allow_pickle=False) as data:
        out = {k: data[k] for k in data.files}
    if not np.array_equal(out["b0"], B0):
        raise RuntimeError(
            f"seed={seed} : b0 de {path} diffère de B0 (run_arcA_revalidate) "
            "-- substrat incohérent, STOP.")
    return out


# --- Une cellule (bras, L, seed) ---------------------------------------------


def _mesurer_cellule(arm: str, L: int, seed: int, hist: dict
                     ) -> tuple[dict[int, float], dict[int, float], tuple[float, float]]:
    """Dérive le champ VRAI du bras (`_field_true_qt`, gelé) puis appelle
    `_measure_cell_qt` (gelé) -- AUCUNE logique de mesure ici, seulement le
    branchement bras -> champ vrai -> instrument famille-2. `hist` est le
    dict npz déjà chargé pour `seed` (`s_L10`, `s_L160`, ...)."""
    s_L = hist[f"s_L{L}"]
    field_true = _field_true_qt(arm, L, seed, s_L)
    return _measure_cell_qt(field_true, seed, L)


# --- Assemblage pur (aucun I/O, aucune mesure) -- testable sur du synthétique -


def _assemble_grille(mesures: dict[tuple[int, int], tuple[dict[int, float], dict[int, float]]],
                      L_list: tuple[int, ...]
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Assemble des cellules déjà mesurées `{(L, seed): (ma1_dict, ma2_dict)}`
    (dict budget -> float, MÊME forme que le retour de `_measure_cell_qt`) en
    tableaux `(nL, nSeed, nBudget)` float64 -- axes dans l'ordre `L_list` x
    `SEEDS` x `BUDGETS`. Fonction PURE : aucune dépendance à l'instrument,
    aucun I/O -- c'est ce qui permet de la tester sur des ma1/ma2
    SYNTHÉTIQUES sans lancer aucune mesure réelle."""
    nL, nSeed, nBudget = len(L_list), len(SEEDS), len(BUDGETS)
    ma1 = np.full((nL, nSeed, nBudget), np.nan, dtype=np.float64)
    ma2 = np.full((nL, nSeed, nBudget), np.nan, dtype=np.float64)
    for iL, L in enumerate(L_list):
        for iS, seed in enumerate(SEEDS):
            ma1_dict, ma2_dict = mesures[(L, seed)]
            for iB, budget in enumerate(BUDGETS):
                ma1[iL, iS, iB] = ma1_dict[budget]
                ma2[iL, iS, iB] = ma2_dict[budget]
    return ma1, ma2


def _construit_sous_jnd(ma1: np.ndarray, ma2: np.ndarray) -> np.ndarray:
    """`sous_jnd` aux 4 JND de `JND_LIST`, en RÉUTILISANT `sous_jnd_a_jnd`
    (`run_arcC_surface_kstar_pins`, déjà verrouillée bit-à-bit) -- pas de
    ré-implémentation de la règle max(M-A1,M-A2)<JND ici. `ma1`/`ma2` peuvent
    être SYNTHÉTIQUES (test) ou issus de `_assemble_grille` (run réel)."""
    nJnd = len(JND_LIST)
    sous_jnd = np.zeros(ma1.shape + (nJnd,), dtype=bool)
    for k, jnd in enumerate(JND_LIST):
        sous_jnd[..., k] = sous_jnd_a_jnd(ma1, ma2, jnd)
    return sous_jnd


# --- main() autonome ----------------------------------------------------------


def main() -> Path:
    t0 = time.time()
    mesures_v2: dict[tuple[int, int], tuple[dict, dict]] = {}
    mesures_ferm: dict[tuple[int, int], tuple[dict, dict]] = {}
    mesures_shuf: dict[tuple[int, int], tuple[dict, dict]] = {}

    print(f"[16L0 mesure] seeds={list(SEEDS)}  L_v2={list(L_V2)}  "
          f"L_control={list(L_CONTROL)}")
    for seed in SEEDS:
        ts = time.time()
        hist = _load_history_160(seed)

        ma1, ma2, center = _mesurer_cellule("v2", 160, seed, hist)
        mesures_v2[(160, seed)] = (ma1, ma2)
        print(f"  seed={seed} v2 L=160  centre={center}  "
              f"ma1={ {b: round(ma1[b], 5) for b in BUDGETS} }")

        for L in L_CONTROL:
            ma1f, ma2f, cf = _mesurer_cellule("ferm", L, seed, hist)
            mesures_ferm[(L, seed)] = (ma1f, ma2f)
            print(f"  seed={seed} ferm L={L}  centre={cf}  "
                  f"ma1={ {b: round(ma1f[b], 5) for b in BUDGETS} }")

            ma1s, ma2s, cs = _mesurer_cellule("shuf", L, seed, hist)
            mesures_shuf[(L, seed)] = (ma1s, ma2s)
            print(f"  seed={seed} shuf L={L}  centre={cs}  "
                  f"ma1={ {b: round(ma1s[b], 5) for b in BUDGETS} }")
        print(f"  seed={seed} terminée [{time.time() - ts:.1f}s]")

    ma1_v2, ma2_v2 = _assemble_grille(mesures_v2, L_V2)
    ma1_ferm, ma2_ferm = _assemble_grille(mesures_ferm, L_CONTROL)
    ma1_shuf, ma2_shuf = _assemble_grille(mesures_shuf, L_CONTROL)

    sous_jnd_v2 = _construit_sous_jnd(ma1_v2, ma2_v2)
    sous_jnd_ferm = _construit_sous_jnd(ma1_ferm, ma2_ferm)
    sous_jnd_shuf = _construit_sous_jnd(ma1_shuf, ma2_shuf)

    # Déterminisme : AUCUNE horloge dans les valeurs sauvées (meta_json ci-dessous
    # ne porte ni timestamp ni elapsed_seconds -- seulement des constantes gelées).
    meta = dict(
        L_v2=list(L_V2),
        L_control=list(L_CONTROL),
        budgets=list(BUDGETS),
        jnd=list(JND_LIST),
        seeds=list(SEEDS),
        s_half_op=S_HALF_OP,
        cap=CAP_FLOATS_10PCT,
        budget_ferm=BUDGET_FERM,
        versions=_versions(),
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        MEASURES_160_PATH,
        ma1_v2=ma1_v2, ma2_v2=ma2_v2, sous_jnd_v2=sous_jnd_v2,
        ma1_ferm=ma1_ferm, ma2_ferm=ma2_ferm, sous_jnd_ferm=sous_jnd_ferm,
        ma1_shuf=ma1_shuf, ma2_shuf=ma2_shuf, sous_jnd_shuf=sous_jnd_shuf,
        meta_json=np.array(json.dumps(meta, indent=2, ensure_ascii=False)),
    )
    elapsed = time.time() - t0
    print("-" * 78)
    print(f"[REPORT] -> {MEASURES_160_PATH}  [{elapsed:.1f}s total]")
    return MEASURES_160_PATH


if __name__ == "__main__":
    main()
