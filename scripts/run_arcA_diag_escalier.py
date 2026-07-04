"""Arc A / itération instrument -- diagnostic « escalier » (clause 2, gravée
9bcb09a, NON-BLOQUANT -- AUCUN verdict).

Motif (verbatim de la clause) : avec le constant-par-blocs, le contrôle-fermable
devient exact au round-trip -- son M-A2 ne sonde plus rien. Ce diagnostic chiffre
le prix des blocs sur UNE cellule GRAVÉE : **(L=10, seed=101, ℓ=3)** -- rollout
depuis s_ferm (bloqué) vs depuis s_L (lisse), trajectoire complète Δχ(t)
sérialisée. Lecture rapportée :
  - Δχ(0)      : part compression STATIQUE (écart d'albedo avant toute dynamique),
  - max_t Δχ(t) : écart maximal AVEC dynamique le long du rollout,
  - rapport max/Δχ(0) : la contribution de la dynamique-sur-marches seule.
Si le verdict de la manche tombe en famille-insuffisante, ce chiffre démêle la
part escalier de la part compression (clause 2 : « sans lui on rejouerait le
débat du plancher un cran plus loin »).

Protocole EXACT de `run_arcA_measure` (helpers importés, PAS réimplémentés) :
centre du rollout = épisode L+1 de la même seed de forçage (`_center_rollout`),
histoire chargée par `_load_history` (sanity b0 incluse), Δχ par `_max_carrier`
(delta_chi max_carrier, même instrument que Task 5), sanity de préfixe de tirage
exécutée avant tout calcul.

Usage :
  .venv/bin/python scripts/run_arcA_diag_escalier.py

Sortie : `outputs/arcA/diag_escalier.json` (trajectoire complète + les trois
chiffres) -- déterministe (physique pure, mêmes primitives que Task 5).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.run_arcA_measure import (_center_rollout, _load_history, _max_carrier,
                                      _sanity_prefixe_tirage)
from scripts.run_arcA_revalidate import B0, PARAMS, S_HALF_OP
from src.albedo import albedo
from src.sediment import run_episode_trajectoire
from src.summary import regenerate, summarize

OUT_DIR = ROOT / "outputs" / "arcA"
DIAG_PATH = OUT_DIR / "diag_escalier.json"

# Cellule GRAVÉE (clause 2, arbitrage 9bcb09a) -- pas un paramètre.
L_GRAVE: int = 10
SEED_GRAVE: int = 101
LEVEL_GRAVE: int = 3


def main() -> None:
    t0 = time.time()
    _sanity_prefixe_tirage()  # gratuite, protocole Task 5 -- avant tout calcul

    hist = _load_history(SEED_GRAVE)
    s_L = hist[f"s_L{L_GRAVE}"]
    s_ferm = regenerate(summarize(s_L, LEVEL_GRAVE))   # NOUVEAU regenerate
    center = _center_rollout(SEED_GRAVE, L_GRAVE)      # épisode L+1, protocole exact

    traj_lisse = run_episode_trajectoire(s_L, B0, center, PARAMS)
    traj_bloque = run_episode_trajectoire(s_ferm, B0, center, PARAMS)
    if len(traj_lisse) != len(traj_bloque):
        raise RuntimeError(
            f"diag escalier : longueurs de trajectoire différentes (lisse="
            f"{len(traj_lisse)}, bloqué={len(traj_bloque)}) -- incohérence de "
            "rollout, STOP.")

    dchi = [_max_carrier(albedo(sb, S_HALF_OP), albedo(sl, S_HALF_OP))
            for sb, sl in zip(traj_bloque, traj_lisse)]

    dchi0 = float(dchi[0])
    dchi_max = float(max(dchi))
    t_max = int(np.argmax(dchi))
    rapport = dchi_max / dchi0 if dchi0 != 0.0 else float("inf")

    resultat = dict(
        cellule=dict(L=L_GRAVE, seed=SEED_GRAVE, level=LEVEL_GRAVE),
        centre_rollout=list(center),
        s_half_op=S_HALF_OP,
        n_snapshots=len(dchi),
        delta_chi_t=[float(v) for v in dchi],       # trajectoire COMPLÈTE
        delta_chi_0=dchi0,                          # part compression statique
        delta_chi_max=dchi_max,                     # max le long du rollout
        t_du_max=t_max,
        rapport_max_sur_0=rapport,                  # dynamique-sur-marches seule
        elapsed_seconds=time.time() - t0,
        note="Diagnostic clause 2 (gravée 9bcb09a), NON-BLOQUANT -- aucun verdict.",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_PATH.write_text(json.dumps(resultat, indent=2, ensure_ascii=False))

    print("=" * 78)
    print("ARC A -- DIAGNOSTIC ESCALIER (clause 2, non-bloquant, aucun verdict)")
    print("=" * 78)
    print(f"Cellule gravée : L={L_GRAVE}, seed={SEED_GRAVE}, ℓ={LEVEL_GRAVE} -- "
          f"centre rollout=({center[0]:.4f},{center[1]:.4f}), "
          f"{len(dchi)} snapshots.")
    print(f"  Δχ(0)          = {dchi0:.6f}   (part compression statique)")
    print(f"  max_t Δχ(t)    = {dchi_max:.6f}   (atteint à t={t_max})")
    print(f"  rapport max/Δχ(0) = {rapport:.4f}   (contribution dynamique-sur-marches)")
    print(f"[REPORT] -> {DIAG_PATH}")
    print(f"Temps de calcul : {resultat['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
