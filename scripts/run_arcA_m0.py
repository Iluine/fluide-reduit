"""Arc A re-fondation / Task 1 -- M-0 (§A8) : borne d'histoire-readout du substrat v2.

Claim testé : sur le substrat sédiment v2 GELÉ (KD_CALIBRE_V2, KE_CALIBRE_V2,
`SedimentParams()` par défaut), les deux histoires même-multiset d'ordre inversé
du protocole (a) du gate v2 (`scripts/run_arcA_revalidate.py::phase_a`) laissent-elles
un écart PERCEPTIBLE au readout ALBEDO (`src/albedo.py::albedo`), au-delà de ce que
corr(s_direct, s_inversé) (diagnostic, référence 0.9446) laisse deviner ?

`f_ordre(jnd, s_half)` = fraction du domaine où |A_direct − A_inversé| / ⟨A⟩ > jnd,
⟨A⟩ = moyenne de (A_direct + A_inversé)/2 sur le domaine, balayé sur
JND ∈ {2,3,4,5} % × S_HALF ∈ {0.005,0.01,0.05,0.10,0.20}·relief (même convention que
le gate v2). Cellule PORTEUSE du verdict mécanique : (JND=5 %, S_HALF=0.05·relief).

Lectures pré-écrites (§A8, contrat PREREGISTRATION.md commit 598fabd du dépôt
contrat pocCascade2phys) :
  - f_ordre(porteuse) <= 1 % -> LECTURE 1 : v2 confirmé mort au readout -> bras nul
    propre, build §A10 peut démarrer.
  - f_ordre(porteuse) >  1 % -> LECTURE 2 : le readout AMPLIFIE une histoire que
    corr sous-estime -> STOP, remonter (la hiérarchie corr<->readout serait
    inversée, le gate §A9 doit être repensé avant tout build).
AUCUNE interprétation au-delà de cette règle mécanique -- les deux issues sont
également acceptables, rien n'est ajusté pour privilégier l'une ou l'autre.

Protocole des deux histoires : STRICTEMENT celui du critère (a) du gate v2, réutilisé
(pas dupliqué) depuis `scripts/run_arcA_revalidate.py` -- `_draw_centers`,
`_run_sequence`, `GRID`/`B0`/`PARAMS`/`RELIEF`/`S_HALF_FRACTIONS`/`S_HALF_OP` importés
TELS QUELS (mêmes constantes gelées, mêmes seeds ; ce fichier n'est ni modifié ni
ré-exécuté ici).

Écart de convention (documenté, pas une ambiguïté) : `f_fraction` de `src/albedo.py`
normalise par la moyenne d'un UNIQUE champ de référence ⟨y⟩ (l'un des deux membres
comparés est "la" référence). Ici, direct et inversé jouent un rôle SYMÉTRIQUE --
aucun des deux n'est privilégié -- et le contrat §A8 demande explicitement
⟨A⟩ = moyenne de (A_direct+A_inversé)/2. Cette convention diffère donc de celle de
`f_fraction` (le dénominateur n'est pas le même champ) : `f_ordre` est réimplémentée
ci-dessous, en miroir de `f_fraction` (même garde-fou du cas dégénéré ⟨A⟩=0), mais
avec le dénominateur symétrique prescrit par le contrat. Pas de STOP ici : le
contrat donne la formule exacte, il n'y a pas d'ambiguïté sur ce qu'il faut calculer,
seulement une non-réutilisation directe de `f_fraction`.

Coût réel : 2 histoires × 10 épisodes (~4-5 s/épisode CPU mesuré) = ~20 épisodes,
~2-4 min CPU -- un seul run, pas de décomposition en phases (contrairement à
`run_arcA_revalidate.py`, qui doit composer ~160 épisodes et dépasse la limite d'un
appel Bash).

Usage :
  .venv/bin/python scripts/run_arcA_m0.py

Sortie : `outputs/arcA/m0_v2_readout.json` (+ tableau imprimé). Déterministe (seeds
figés, aucune horloge dans le calcul -- seul `elapsed_seconds` est un artefact de
mesure, hors calcul)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.run_arcA_revalidate import (B0, GRID, PARAMS, RELIEF, S_HALF_FRACTIONS,
                                         S_HALF_OP, _draw_centers, _run_sequence)
from src.albedo import albedo, relief_shaded

OUT_DIR = ROOT / "outputs" / "arcA"
OUT_PATH = OUT_DIR / "m0_v2_readout.json"

# Protocole (a) du gate v2 -- IDENTIQUE à `run_arcA_revalidate.phase_a` (mêmes
# seed/n_episodes/boîte que `scripts/run_arcA_revalidate.py::phase_a`).
N_EPISODES: int = 10
SEED: int = 7
BOX: tuple[float, float] = (0.15, 0.85)

# Balayage §A8 (verbatim task-1-brief.md).
JND_PCT_LIST: tuple[int, ...] = (2, 3, 4, 5)

# Seuil FIGÉ (contrat §A8) -- ne JAMAIS ajuster, quel que soit le résultat.
THRESH_F_ORDRE_PORTEUSE: float = 0.01
# Dernière mesure v2 gravée au journal pocCascade2phys (itération de co-calibration,
# commit ec384b9) -- diagnostic rapporté, pas un seuil.
CORR_REFERENCE_ORIGINE: float = 0.9446


def _f_ordre(a_direct: np.ndarray, a_inverse: np.ndarray, jnd: float) -> float:
    """Fraction du domaine où |a_direct − a_inversé| / ⟨(a_direct+a_inversé)/2⟩ > jnd
    (convention symétrique §A8 -- cf. docstring module pour l'écart avec
    `f_fraction`). Même garde-fou 0/0 que `f_fraction` : dénominateur nul ->
    rapport nul là où l'écart est nul, +inf sinon (pas de 0/0 silencieux)."""
    diff = np.abs(a_direct - a_inverse)
    denom = float(np.mean((a_direct + a_inverse) / 2.0))
    if denom == 0.0:
        rel = np.where(diff == 0.0, 0.0, np.inf)
    else:
        rel = diff / denom
    return float(np.mean(rel > jnd))


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    centers = _draw_centers(rng, N_EPISODES, BOX, BOX)
    s_direct = _run_sequence(centers)
    s_inverse = _run_sequence(list(reversed(centers)))

    corr = float(np.corrcoef(s_direct.ravel(), s_inverse.ravel())[0, 1])

    balayage: dict[str, dict[str, float]] = {}
    for frac in S_HALF_FRACTIONS:
        s_half = frac * RELIEF
        a_direct = albedo(s_direct, s_half)
        a_inverse = albedo(s_inverse, s_half)
        balayage[f"{frac:.3f}"] = {
            str(jnd_pct): _f_ordre(a_direct, a_inverse, jnd_pct / 100.0)
            for jnd_pct in JND_PCT_LIST
        }

    # Diagnostic (non porteur) : plancher de canal -- même comparaison direct/inversé
    # mais sur le readout RELIEF (pas de S_HALF, l'ombrage ne dépend que de b0+s).
    relief_direct = relief_shaded(B0 + s_direct)
    relief_inverse = relief_shaded(B0 + s_inverse)
    f_ordre_relief = {
        str(jnd_pct): _f_ordre(relief_direct, relief_inverse, jnd_pct / 100.0)
        for jnd_pct in JND_PCT_LIST
    }

    s_half_porteuse_key = f"{0.05:.3f}"
    f_ordre_porteur = balayage[s_half_porteuse_key]["5"]
    verdict = "lecture_1" if f_ordre_porteur <= THRESH_F_ORDRE_PORTEUSE else "lecture_2"

    elapsed = time.time() - t0

    result = dict(
        meta=dict(
            grid=dict(H=GRID.H, W=GRID.W, dx=GRID.dx, dy=GRID.dy),
            relief=RELIEF,
            kd_calibre_v2=PARAMS.k_d,
            ke_calibre_v2=PARAMS.k_e,
            n_episodes=N_EPISODES,
            seed=SEED,
            box=list(BOX),
            centers=[list(c) for c in centers],
            jnd_pct_list=list(JND_PCT_LIST),
            s_half_fractions=list(S_HALF_FRACTIONS),
            s_half_op_fraction=0.05,
            s_half_op=S_HALF_OP,
            elapsed_seconds=elapsed,
        ),
        diagnostics=dict(
            corr_direct_inverse=corr,
            corr_reference_origine=CORR_REFERENCE_ORIGINE,
            f_ordre_relief_shaded_by_jnd_pct=f_ordre_relief,
        ),
        balayage=balayage,
        cellule_porteuse=dict(
            jnd_pct=5,
            s_half_fraction=0.05,
            f_ordre=f_ordre_porteur,
            seuil=THRESH_F_ORDRE_PORTEUSE,
            condition="f_ordre <= seuil",
        ),
        verdict=verdict,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print("=" * 78)
    print("ARC A RE-FONDATION / TASK 1 -- M-0 (§A8) : borne d'histoire-readout v2")
    print("=" * 78)
    print(f"corr(s_direct, s_inversé) = {corr:.4f}  (référence origine {CORR_REFERENCE_ORIGINE})")
    header = f"{'S_HALF (*relief)':>18} | " + " | ".join(f"JND={j}%" for j in JND_PCT_LIST)
    print(header)
    for frac in S_HALF_FRACTIONS:
        key = f"{frac:.3f}"
        row = balayage[key]
        print(f"{key:>18} | " + " | ".join(f"{row[str(j)]:.4f}" for j in JND_PCT_LIST))
    print(f"f_ordre (relief ombré, plancher de canal) = {f_ordre_relief}")
    print("-" * 78)
    print(f"Cellule porteuse (JND=5%, S_HALF=0.05*relief) : f_ordre={f_ordre_porteur:.4f} "
          f"(seuil <= {THRESH_F_ORDRE_PORTEUSE}) -> {verdict.upper()}")
    if verdict == "lecture_1":
        print("[LECTURE 1] v2 confirmé mort au readout -> bras nul propre, build §A10 lancé.")
    else:
        print("[LECTURE 2] STOP -- le readout amplifie une histoire que corr sous-estime : "
              "la hiérarchie corr<->readout doit être repensée (gate §A9) avant tout build. "
              "Remonter au contrôleur.")
    print(f"Temps de calcul : {elapsed:.1f}s")
    print(f"[REPORT] -> {OUT_PATH}")


if __name__ == "__main__":
    main()
