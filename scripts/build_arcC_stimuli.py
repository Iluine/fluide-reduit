"""Arc C / Task 1 -- construction du catalogue de stimuli spatiaux (post-
processing pur, aucune physique, ~secondes).

Contrat : addendum §C3 de PREREGISTRATION.md (pocCascade2phys, commits
`c494d50` + `5d13bc4`), reproduit verbatim dans
`.superpowers/sdd/arcC-task1-stimuli-brief.md`. Ce script ne fait AUCUN
re-run de simulation : il lit les 20 champs de sédiment déjà commités de la
manche 1 (`outputs/arcA/histories/h_s{101..105}.npz`, clés `s_L{10,20,40,80}`)
et mesure, pour chacun d'eux et pour chaque budget de la grille manche 1
(famille 2, quadtree), la courbe Δχ(t) du mélange `stim(t) = (1-t)*s_true +
t*régénéré(budget)` (`src.arcC_stimuli.courbe_delta_chi`) sur
`ts = linspace(0, 1, 11)`.

Grille : 5 seeds x 4 L x 7 budgets x 11 valeurs de t = 20 champs sources x 7
budgets, chacun avec sa courbe Δχ(t) mesurée -- AUCUNE hypothèse de monotonie
ni de linéarité (spec §C3 : mesuré, pas supposé).

Sortie : `outputs/arcC/stimuli_catalogue.npz` (commité), clés :
  - `courbes`      : (nSeed=5, nL=4, nBudget=7, nTs=11) float64 -- Δχ(t) mesuré.
  - `delta_chi_1`  : (nSeed, nL, nBudget) float64 -- Δχ(t=1) = `courbes[..., -1]`,
                     dupliqué pour un accès direct (= M-A1 manche 1 du budget).
  - `seeds`, `L`, `budgets`, `ts` : axes de la grille (int64/int64/int64/float64).
  - `meta_json`    : seeds, L, budgets, ts, S_HALF_OP, versions (numpy/python).

Imprime un résumé (plage de Δχ(1) atteignable par budget, min/max sur les 20
champs) et une COMPARAISON INFORMATIVE (non gravée) : Δχ(1) à (L=10,
seed=101, budget) vs `ma1_v2` de `outputs/arcA/measures_qt.npz` à la même
cellule -- doivent coïncider (même instrument, même régénérateur) ; un écart
serait un bug à remonter, pas à absorber (cf. brief).

Usage :
  .venv/bin/python scripts/build_arcC_stimuli.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.run_arcA_measure import _versions
from scripts.run_arcA_revalidate import S_HALF_OP
from src.arcC_stimuli import courbe_delta_chi

HIST_DIR = ROOT / "outputs" / "arcA" / "histories"
OUT_DIR = ROOT / "outputs" / "arcC"
OUT_PATH = OUT_DIR / "stimuli_catalogue.npz"
MEASURES_QT_PATH = ROOT / "outputs" / "arcA" / "measures_qt.npz"

# Grille manche 1 -- IMPORTÉE en dur ici (mêmes valeurs que
# `scripts.run_arcA_measure_qt`, cf. brief : "Réutilise-les comme axe des
# budgets" -- pas de réimport croisé de script à script pour ces constantes
# simples, afin de ne pas coupler ce script à l'exécution de la mesure M-A1).
SEEDS: tuple[int, ...] = (101, 102, 103, 104, 105)
L_LIST: tuple[int, ...] = (10, 20, 40, 80)
BUDGETS: tuple[int, ...] = (32, 64, 128, 256, 400, 1024, 2048)
TS: np.ndarray = np.linspace(0.0, 1.0, 11)


def _charger_s_true(seed: int, L: int) -> np.ndarray:
    with np.load(HIST_DIR / f"h_s{seed}.npz") as d:
        return np.array(d[f"s_L{L}"], dtype=np.float64)


def construire_catalogue() -> Path:
    nSeed, nL, nBudget, nTs = len(SEEDS), len(L_LIST), len(BUDGETS), len(TS)
    courbes = np.empty((nSeed, nL, nBudget, nTs), dtype=np.float64)

    for iSeed, seed in enumerate(SEEDS):
        for iL, L in enumerate(L_LIST):
            s_true = _charger_s_true(seed, L)
            for iBudget, budget in enumerate(BUDGETS):
                courbes[iSeed, iL, iBudget, :] = courbe_delta_chi(s_true, budget, TS)

    delta_chi_1 = courbes[..., -1].copy()

    meta = dict(
        seeds=list(SEEDS), L=list(L_LIST), budgets=list(BUDGETS),
        ts=[float(t) for t in TS], s_half_op=S_HALF_OP,
        versions=_versions())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        courbes=courbes, delta_chi_1=delta_chi_1,
        seeds=np.array(SEEDS, dtype=np.int64),
        L=np.array(L_LIST, dtype=np.int64),
        budgets=np.array(BUDGETS, dtype=np.int64),
        ts=TS.astype(np.float64),
        meta_json=np.array(json.dumps(meta, indent=2, ensure_ascii=False)))

    _resume(delta_chi_1)
    _comparaison_informative(delta_chi_1)
    print(f"[REPORT] -> {OUT_PATH}")
    return OUT_PATH


def _resume(delta_chi_1: np.ndarray) -> None:
    """Plage de Δχ(1) atteignable par budget (min/max sur les 20 champs) --
    pour repérer d'un coup d'œil quels budgets bracketent la plage JND
    candidate [~1-6 %]."""
    print("=" * 78)
    print("ARC C / TASK 1 -- CATALOGUE DE STIMULI SPATIAUX (Δχ(1) par budget, "
          "min/max sur les 20 champs sources)")
    print("=" * 78)
    for iBudget, budget in enumerate(BUDGETS):
        vals = delta_chi_1[:, :, iBudget]
        print(f"  budget={budget:5d} : Δχ(1) ∈ [{vals.min() * 100:6.3f}%, "
              f"{vals.max() * 100:6.3f}%]")


def _comparaison_informative(delta_chi_1: np.ndarray) -> None:
    """Comparaison INFORMATIVE (imprimée, non gravée) : Δχ(1) à (L=10,
    seed=101, budget) vs `ma1_v2` de `outputs/arcA/measures_qt.npz` à la même
    cellule -- doivent coïncider (même instrument, même régénérateur). Un
    écart est un bug à remonter, jamais absorbé silencieusement."""
    print("-" * 78)
    print("Comparaison informative -- Δχ(1) [ce catalogue] vs ma1_v2 [manche 1] "
          "-- L=10, seed=101 :")
    if not MEASURES_QT_PATH.exists():
        print(f"  (absent : {MEASURES_QT_PATH} introuvable -- comparaison sautée)")
        return
    with np.load(MEASURES_QT_PATH, allow_pickle=False) as d:
        ma1_v2 = d["ma1_v2"]
    iL, iSeed = L_LIST.index(10), SEEDS.index(101)
    ecart_max = 0.0
    for iBudget, budget in enumerate(BUDGETS):
        ici = float(delta_chi_1[iSeed, iL, iBudget])
        manche1 = float(ma1_v2[iL, iSeed, iBudget])
        ecart = abs(ici - manche1)
        ecart_max = max(ecart_max, ecart)
        marqueur = "OK" if ecart <= 1e-12 else "*** ÉCART ***"
        print(f"  budget={budget:5d} : ici={ici:.12f}  manche1={manche1:.12f}  "
              f"écart={ecart:.3e}  [{marqueur}]")
    if ecart_max <= 1e-12:
        print("  -> coïncidence exacte (à 1e-12) sur les 7 budgets.")
    else:
        print(f"  -> ÉCART MAXIMAL {ecart_max:.3e} > 1e-12 -- BUG À REMONTER, "
              "ne pas absorber.")


def main() -> None:
    construire_catalogue()


if __name__ == "__main__":
    main()
