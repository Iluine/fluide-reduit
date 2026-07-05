"""Arc C / Task C-1a — Task 2 : re-seuillage de la surface k*(L) famille-2
quadtree aux JND RÉELS du pin humain, reproduit verbatim dans `.superpowers/
sdd/arcC-c1a-task2-kstar-brief.md`.

**Re-lecture MÉCANIQUE de mesures déjà archivées — ZÉRO simulation, ZÉRO
RNG, ZÉRO nouvelle donnée.** La surface k*(L, JND) de la manche 1 (famille 2
quadtree) n'a été tabulée qu'à la grille JND ∈ {2,3,4,5} % (`outputs/arcA/
verdict_qt.json`). Le pin humain réel (§C11, campagne ABX+staircase) tombe
HORS grille, aux deux régimes sévère/laxiste. Ce script recalcule k*(L) aux
JND EXACTS du pin (`outputs/arcC/pins_spatial.json`, lu au runtime — jamais
codé en dur) en RÉUTILISANT tel quel `k_star_seed_qt`/`k_star_groupe_qt` de
`scripts.run_arcA_verdict_qt` (famille 2, JAMAIS réimplémentées ici).

**INTERDIT load-bearing (contrainte de mission)** : aucune lecture de
verdict (cellule 1/2/3, PASS/FAIL/INDÉTERMINÉ, issue mappée), aucune pente,
aucun bootstrap. Ce script produit LA TABLE k*, rien d'autre — la lecture
§C4 et la pente 16L₀ vivent ailleurs, gatées. N'importe ni `verdict_A3_*`,
ni `issue_mappee`, ni `pente_bootstrap_qt`.

Le seul morceau écrit ici (pas importé, car règle inline non factorisée) :
`sous_jnd_a_jnd`, réplique verbatim de la règle de `run_arcA_measure_qt.py`
(l.284-292) — max(M-A1, M-A2) < JND, convention NaN -> False (indéfini =
False). Verrouillée bit-à-bit par `tests/test_arcC_surface_kstar_pins.py`
(`test_recalcul_sous_jnd_reproduit_stocke`, compare aux 4 JND de grille avec
`sous_jnd_v2` stocké dans `measures_qt.npz`).

Pour chaque régime (sévère, laxiste) et chaque rôle (ic_bas, pin, ic_haut) —
donc 6 JND — et chaque L ∈ {10,20,40,80} : k*(L) = médiane inter-seeds +
clause 2×JND (`k_star_groupe_qt`, §A2 exact), plus les annotations
`non_discriminant` (`_non_discriminant`, budget médian ∈ {1024,2048}) et
`sup_cap` (k* fini et > `CAP_FLOATS` = 409.6). L'écart sévère↔laxiste EN
FLOATS au pin (`delta_lax_moins_sev = k_lax - k_sev`, signé) est calculé
séparément, par L.

Usage :
  .venv/bin/python scripts/run_arcC_surface_kstar_pins.py

Sortie : `outputs/arcC/surface_kstar_aux_pins.json`. Déterministe (aucune
RNG) — un double run produit un JSON identique."""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.run_arcA_measure import JND_LIST, L_LIST, SEEDS
from scripts.run_arcA_measure_qt import BUDGETS, MEASURES_PATH
from scripts.run_arcA_verdict import _to_jsonable
from scripts.run_arcA_verdict_qt import (BUDGETS_NON_DISCRIMINANTS, CAP_FLOATS,
                                         _non_discriminant, k_star_groupe_qt,
                                         k_star_seed_qt)

OUT_DIR = ROOT / "outputs" / "arcC"
PINS_PATH = OUT_DIR / "pins_spatial.json"
SURFACE_PATH = OUT_DIR / "surface_kstar_aux_pins.json"

REGIMES: tuple[str, ...] = ("severe", "laxiste")
ROLES: tuple[str, ...] = ("ic_bas", "pin", "ic_haut")
# Étiquettes du plan (arrondies) — PAS la source de vérité, seulement portées
# dans le JSON pour lisibilité humaine ; la valeur EXACTE vient de
# `pins_spatial.json` (cf. `charge_pins`).
LABELS_PCT: dict[str, tuple[str, str, str]] = {
    "severe": ("6.0", "7.3", "8.7"),
    "laxiste": ("9.4", "11.5", "13.9"),
}


# --- Le seul morceau écrit ici : réplique verbatim de run_arcA_measure_qt.py ---


def sous_jnd_a_jnd(ma1: np.ndarray, ma2: np.ndarray, jnd: float) -> np.ndarray:
    """Réplique EXACTE de la règle inline de `run_arcA_measure_qt.py`
    (l.284-292) : sous_jnd = max(M-A1, M-A2) < JND, avec convention NaN ->
    False (indéfini = False). Verrouillée bit-à-bit par
    `test_recalcul_sous_jnd_reproduit_stocke` (comparaison aux 4 JND de
    grille avec `sous_jnd_v2` stocké dans `measures_qt.npz`)."""
    max_val = np.maximum(ma1, ma2)
    is_nan = np.isnan(max_val)
    with np.errstate(invalid="ignore"):
        cond = max_val < jnd
    return np.where(is_nan, False, cond)


# --- Chargement des pins réels (source unique de vérité, §C11) --------------


def charge_pins(pins_path: Path = PINS_PATH) -> dict[str, dict[str, float]]:
    """Charge les JND réels (ic_bas, pin, ic_haut) des deux régimes depuis
    `pins_spatial.json` (fait foi, §C11) — lu au RUNTIME, jamais codé en dur,
    pour éviter toute dérive vs les étiquettes arrondies du plan. Retourne
    `{regime: {"ic_bas": float, "pin": float, "ic_haut": float}}`."""
    data = json.loads(Path(pins_path).read_text())
    regimes = data["regimes"]
    pins: dict[str, dict[str, float]] = {}
    for regime in REGIMES:
        r = regimes[regime]
        ic_bas, ic_haut = r["ic"]
        pins[regime] = dict(ic_bas=float(ic_bas), pin=float(r["jnd"]), ic_haut=float(ic_haut))
    return pins


# --- Cœur du calcul : k*(L) à un JND donné, réutilisant k_star_*_qt --------


def kstar_par_L(ma1_v2: np.ndarray, ma2_v2: np.ndarray, jnd: float) -> dict[int, dict]:
    """k*(L) pour un JND donné, sur les 4 L de `L_LIST` — RÉUTILISE
    `k_star_seed_qt`/`k_star_groupe_qt` (famille 2, verdict quadtree, jamais
    réimplémentées). Retourne, par L : médiane inter-seeds + clause 2×JND,
    k* par seed, et les annotations `non_discriminant`/`sup_cap`
    (`_non_discriminant`/`CAP_FLOATS` réutilisés)."""
    sj = sous_jnd_a_jnd(ma1_v2, ma2_v2, jnd)
    out: dict[int, dict] = {}
    for iL, L in enumerate(L_LIST):
        par_seed = {seed: k_star_seed_qt(sj[iL, iS]) for iS, seed in enumerate(SEEDS)}
        k_med, audit = k_star_groupe_qt(sj[iL], ma1_v2[iL], ma2_v2[iL], jnd)
        out[L] = dict(
            mediane=k_med,
            par_seed=par_seed,
            non_discriminant=_non_discriminant(k_med),
            sup_cap=bool(math.isfinite(k_med) and k_med > CAP_FLOATS),
            audit=audit,
        )
    return out


def construit_surface(ma1_v2: np.ndarray, ma2_v2: np.ndarray,
                      pins: dict[str, dict[str, float]]) -> dict:
    """Surface complète : {régime: {rôle: {"jnd": ..., "k_star_par_L": ...}}},
    pour les 2 régimes x 3 rôles x 4 L (24 cellules)."""
    surface: dict = {}
    for regime in REGIMES:
        surface[regime] = {}
        for role in ROLES:
            jnd = pins[regime][role]
            surface[regime][role] = dict(jnd=jnd, k_star_par_L=kstar_par_L(ma1_v2, ma2_v2, jnd))
    return surface


def construit_ecart(surface: dict) -> dict[int, dict]:
    """Écart sévère↔laxiste EN FLOATS, au pin (rôle "pin") de chaque régime,
    pour chaque L : delta = k_lax - k_sev (signé). Si l'un des deux est ∞,
    le delta est indéfini (`None`, sérialisé `null`), noté explicitement."""
    ecart: dict[int, dict] = {}
    for L in L_LIST:
        k_sev = surface["severe"]["pin"]["k_star_par_L"][L]["mediane"]
        k_lax = surface["laxiste"]["pin"]["k_star_par_L"][L]["mediane"]
        if math.isfinite(k_sev) and math.isfinite(k_lax):
            delta: float | None = k_lax - k_sev
            note = None
        else:
            delta = None
            note = "delta indefini -- au moins un des deux k* est infini"
        ecart[L] = dict(k_star_severe=k_sev, k_star_laxiste=k_lax,
                        delta_lax_moins_sev=delta, note=note)
    return ecart


def construit_meta(pins: dict[str, dict[str, float]], measures_path: Path,
                   pins_path: Path) -> dict:
    pins_utilises = {
        regime: dict(**{role: pins[regime][role] for role in ROLES},
                    labels_pct=list(LABELS_PCT[regime]))
        for regime in REGIMES
    }
    return dict(
        description=("Re-seuillage surface k*(L) famille-2 quadtree aux pins reels "
                     "Arc C (C-1a Task 2). Aucun verdict, aucune pente."),
        source_measures=str(Path(measures_path).relative_to(ROOT)),
        source_pins=str(Path(pins_path).relative_to(ROOT)),
        verdict_reutilise=("scripts/run_arcA_verdict_qt.py (k_star_seed_qt, "
                           "k_star_groupe_qt) — famille 2"),
        budgets=list(BUDGETS),
        cap_floats_10pct=CAP_FLOATS,
        budgets_non_discriminants=list(BUDGETS_NON_DISCRIMINANTS),
        L=list(L_LIST),
        seeds=list(SEEDS),
        pins_utilises=pins_utilises,
        note_exact_vs_arrondi=("JND EXACTS de pins_spatial.json (fait foi §C11) ; "
                               "les % du plan sont des etiquettes."),
        non_regression=dict(kstar_L80_a_5pct=400.0,
                            sous_jnd_reproduit_aux_4_jnd_grille=True),
        convention_serialisation="float infini -> 'inf'",
    )


def construit_resultat(measures_path: Path = MEASURES_PATH,
                       pins_path: Path = PINS_PATH) -> dict:
    """Assemble la structure complète (meta + surface + écart) depuis les
    mesures archivées (`measures_qt.npz`) et les pins réels
    (`pins_spatial.json`). Aucune RNG — déterministe par construction."""
    with np.load(measures_path, allow_pickle=False) as d:
        ma1_v2 = np.asarray(d["ma1_v2"])
        ma2_v2 = np.asarray(d["ma2_v2"])
    pins = charge_pins(pins_path)
    surface = construit_surface(ma1_v2, ma2_v2, pins)
    ecart = construit_ecart(surface)
    meta = construit_meta(pins, measures_path, pins_path)
    return dict(meta=meta, surface_kstar_aux_pins=surface,
               ecart_severe_laxiste_floats_au_pin=ecart)


def _imprime_table(resultat: dict) -> None:
    surface = resultat["surface_kstar_aux_pins"]
    ecart = resultat["ecart_severe_laxiste_floats_au_pin"]
    print("=== Surface k*(L) aux pins reels (famille 2 quadtree) ===")
    for regime in REGIMES:
        for role in ROLES:
            cellule = surface[regime][role]
            jnd = cellule["jnd"]
            ligne = ", ".join(
                f"L={L}: k*={cellule['k_star_par_L'][L]['mediane']}"
                for L in L_LIST)
            print(f"[{regime}/{role}] jnd={jnd:.8f} -- {ligne}")
    print("=== Ecart severe<->laxiste (floats, au pin) ===")
    for L in L_LIST:
        e = ecart[L]
        print(f"L={L}: severe={e['k_star_severe']} laxiste={e['k_star_laxiste']} "
             f"delta={e['delta_lax_moins_sev']}")


def main() -> None:
    t0 = time.time()
    resultat = construit_resultat()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SURFACE_PATH.write_text(json.dumps(_to_jsonable(resultat), indent=2, ensure_ascii=False))
    _imprime_table(resultat)
    print(f"[REPORT] -> {SURFACE_PATH} ({time.time() - t0:.2f}s)")


if __name__ == "__main__":
    main()
