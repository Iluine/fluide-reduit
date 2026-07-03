"""Arc A / diagnostic (NON-gate) — pourquoi le gate de re-validation (a) FAIL sur la
path-dependence : corr(s_A, s_B) = 0.9929 pour deux histoires de 10 pulses aux mêmes
centres (default_rng(7)) mais ordre inversé, contre un seuil <= 0.7 (référence
d'origine : 0.39). Ce script ne CORRIGE rien : il MESURE, pour instruire l'hypothèse
que la loi de dépôt/érosion reconstruite (`src/sediment.py::_exner_step`, θ_c=0.5,
k_e=k_d/5) serait quasi dépôt-seul et/ou dominée par la phase calme (eau stagnante
dans les cuvettes pendant la relaxation) -> dépôt ≈ temps de résidence de l'eau ≈
terrain-déterminé ≈ commutatif à l'ordre des pulses.

Réplique fidèlement UNE histoire (seed 7, 10 épisodes, `SedimentParams` par défaut,
terrain `default_terrain` par défaut, centres tirés exactement comme `run_history` /
`scripts/run_arcA_revalidate.py::phase_a` : `default_rng(7)`, puis pour chaque
épisode `(rng.uniform(0.15,0.85), rng.uniform(0.15,0.85))`), en appelant le solveur
(`_relax_episode`) et la loi Exner (`_exner_step`) telles quelles importées de
`src.sediment` — AUCUNE modification de `src/`, aucun re-réglage de constante figée.
La boucle sur les snapshots sous-échantillonnés (`save_every`) reproduit EXACTEMENT
celle de `src.sediment._integrate_exner` ; le nouvel état `s` à chaque snapshot est
calculé par le vrai `_exner_step` importé (bit-identique à la production). Les
quantités décomposées (dépôt/érosion séparés, masque mouillé, θ) qui ne sont PAS
exposées par `_exner_step` sont recalculées ici avec la MÊME formule (recopiée du
docstring de `_exner_step`), uniquement à des fins de mesure — elles n'influencent
jamais la trajectoire réelle de `s`, qui reste pilotée par l'appel au vrai
`_exner_step`.

Quatre mesures (cf. mission du contrôleur) :
  1. Part d'érosion active : sur tous les (snapshot, cellule mouillée) de l'histoire,
     fraction où θ > θ_c ; et par épisode, ratio masse érodée / masse déposée
     (Σ|terme dépôt|·dt vs Σ|terme érosion|·dt, les deux termes de `_exner_step`).
  2. Répartition temporelle du dépôt : part du dépôt total accumulée dans les
     snapshots « calmes » (max(θ) sur le domaine < θ_c/10) vs « actifs » (le reste) ;
     nombre moyen de snapshots calmes/actifs par épisode.
  3. Terrain-déterminisme : corrélation de Pearson entre s_final (10 épisodes,
     ordre direct) et la carte de résidence d'eau W = Σ_épisodes Σ_snapshots h
     (aplaties, mêmes snapshots sous-échantillonnés que (1)/(2)) ; et corr entre
     s_final et la résidence du DERNIER épisode seul.
  4. Sensibilité à l'ordre du terme d'érosion : rejoue la même histoire (mêmes 10
     centres, un seul tirage `default_rng(7)`) avec k_e=0 (érosion coupée — RUN
     DIAGNOSTIC via `dataclasses.replace(SedimentParams(), k_e=0.0)`, pas un
     re-réglage du substrat) et rapporte corr(s_ordre_direct, s_ordre_inversé) pour
     k_e nominal et k_e=0. Si couper l'érosion ne change presque pas cette
     corrélation, le FAIL de path-dependence ne vient pas du ratio k_e/k_d.

Coût : 4 histoires de 10 épisodes = 40 épisodes (~3-5 min mesuré). Écrit
`outputs/arcA/diag_pathdep.json` + imprime un résumé lisible.

Usage : .venv/bin/python scripts/diag_arcA_pathdep.py
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from config import GridConfig
from src.sediment import SedimentParams, _DRY_EPS, _exner_step, _relax_episode, default_terrain

OUT_DIR = ROOT / "outputs" / "arcA"

SEED: int = 7
N_EPISODES: int = 10
GRID = GridConfig()
B0 = default_terrain(GRID)
PARAMS_NOMINAL = SedimentParams()
PARAMS_KE0 = replace(PARAMS_NOMINAL, k_e=0.0)

# Référence du gate (scripts/run_arcA_revalidate.py::phase_a, step_a.json), pour
# vérifier que ce script reproduit fidèlement l'histoire du gate (sanity check, pas
# une mesure en soi).
_REF_GATE_CORR_NOMINAL: float = 0.9928766659230924


def _draw_centers(seed: int, n: int) -> list[tuple[float, float]]:
    """Centres tirés EXACTEMENT comme `src.sediment.run_history` /
    `run_arcA_revalidate.phase_a` : `default_rng(seed)`, puis par épisode
    (x, y) ~ U(0.15,0.85) i.i.d., x avant y."""
    rng = np.random.default_rng(seed)
    return [(float(rng.uniform(0.15, 0.85)), float(rng.uniform(0.15, 0.85))) for _ in range(n)]


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def _run_episode_diag(s: np.ndarray, b0: np.ndarray, center_frac: tuple[float, float],
                      params: SedimentParams, collect_diag: bool
                      ) -> tuple[np.ndarray, dict | None]:
    """Un épisode complet, réplique de `src.sediment.run_episode` +
    `_integrate_exner`, mais avec la boucle sur les snapshots déroulée ici pour
    pouvoir mesurer les quantités décomposées. L'état `s` retourné est calculé par
    le VRAI `_exner_step` importé (bit-identique à `run_episode`). Si
    `collect_diag`, retourne aussi un dict de diagnostics par snapshot agrégés sur
    l'épisode ; sinon (2e/3e/4e histoires, seul l'état final compte) `None`."""
    b_eff = b0 + s
    times, hs, hus, hvs = _relax_episode(b_eff, center_frac, params)
    n = len(times)
    s_local = s.copy()

    diag = None
    if collect_diag:
        diag = dict(
            n_snapshots=0, n_wet_total=0, n_active_total=0,
            depot_mass_ep=0.0, erosion_mass_ep=0.0,
            depot_calm=0.0, depot_active=0.0,
            n_calm=0, n_active_snap=0,
            h_sum=np.zeros_like(b0, dtype=np.float64),
        )

    for i in range(params.save_every, n, params.save_every):
        dt = float(times[i] - times[i - params.save_every])
        h_i, hu_i, hv_i = hs[i], hus[i], hvs[i]

        if collect_diag:
            # --- Recopie EXACTE de la formule de src.sediment._exner_step (lue,
            # pas éditée) pour exposer dépôt/érosion séparément et θ/wet, à des
            # fins de mesure uniquement -- la trajectoire réelle de s_local plus
            # bas vient du vrai _exner_step importé.
            wet = h_i > 10.0 * _DRY_EPS
            u = np.zeros_like(h_i)
            v = np.zeros_like(h_i)
            u[wet] = hu_i[wet] / h_i[wet]
            v[wet] = hv_i[wet] / h_i[wet]
            theta = u ** 2 + v ** 2
            depot = wet * params.k_d * h_i * (theta < params.theta_c) * (1.0 - theta / params.theta_c)
            erosion = wet * params.k_e * s_local * (theta > params.theta_c) * (theta / params.theta_c - 1.0)
            depot_mass = float(np.sum(depot) * dt)
            erosion_mass = float(np.sum(erosion) * dt)
            n_wet = int(wet.sum())
            n_active = int(np.sum(wet & (theta > params.theta_c)))
            max_theta = float(theta.max())  # cellules sèches : theta=0 (init), n'affecte pas le max
            is_calm = max_theta < params.theta_c / 10.0

            diag["n_snapshots"] += 1
            diag["n_wet_total"] += n_wet
            diag["n_active_total"] += n_active
            diag["depot_mass_ep"] += depot_mass
            diag["erosion_mass_ep"] += erosion_mass
            if is_calm:
                diag["n_calm"] += 1
                diag["depot_calm"] += depot_mass
            else:
                diag["n_active_snap"] += 1
                diag["depot_active"] += depot_mass
            diag["h_sum"] += h_i

        # État réel : vrai _exner_step importé, bit-identique à run_episode/_integrate_exner.
        s_local = _exner_step(s_local, h_i, hu_i, hv_i, dt, params)

    return s_local, diag


def _run_history_diag(centers: list[tuple[float, float]], params: SedimentParams,
                      collect_diag: bool
                      ) -> tuple[np.ndarray, list[dict] | None]:
    """Séquence de `len(centers)` épisodes dans l'ORDRE donné (réplique de
    `run_history`, mais centres explicites pour pouvoir rejouer un ordre inversé
    avec le MÊME tirage — même pattern que `run_arcA_revalidate._run_sequence`)."""
    s = np.zeros_like(B0, dtype=np.float64)
    episodes_diag: list[dict] | None = [] if collect_diag else None
    for center in centers:
        s, ep_diag = _run_episode_diag(s, B0, center, params, collect_diag)
        if collect_diag:
            episodes_diag.append(ep_diag)
    return s, episodes_diag


def main() -> None:
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    centers = _draw_centers(SEED, N_EPISODES)
    print(f"[diag] seed={SEED} n_episodes={N_EPISODES} centres tirés de default_rng({SEED}).")

    # --- Histoire principale : ordre direct, k_e nominal, AVEC diagnostics ------
    t0 = time.time()
    s_direct_nominal, episodes_diag = _run_history_diag(centers, PARAMS_NOMINAL, collect_diag=True)
    t_direct_nominal = time.time() - t0
    print(f"[diag] histoire directe (nominal, avec diagnostics) : {t_direct_nominal:.1f}s")

    # --- 3 histoires additionnelles : état final seulement -----------------------
    t0 = time.time()
    s_reversed_nominal, _ = _run_history_diag(list(reversed(centers)), PARAMS_NOMINAL, collect_diag=False)
    t_reversed_nominal = time.time() - t0
    print(f"[diag] histoire inversée (nominal) : {t_reversed_nominal:.1f}s")

    t0 = time.time()
    s_direct_ke0, _ = _run_history_diag(centers, PARAMS_KE0, collect_diag=False)
    t_direct_ke0 = time.time() - t0
    print(f"[diag] histoire directe (k_e=0) : {t_direct_ke0:.1f}s")

    t0 = time.time()
    s_reversed_ke0, _ = _run_history_diag(list(reversed(centers)), PARAMS_KE0, collect_diag=False)
    t_reversed_ke0 = time.time() - t0
    print(f"[diag] histoire inversée (k_e=0) : {t_reversed_ke0:.1f}s")

    # ============================= Mesure 1 ====================================
    n_wet_total = sum(e["n_wet_total"] for e in episodes_diag)
    n_active_total = sum(e["n_active_total"] for e in episodes_diag)
    global_active_fraction = n_active_total / n_wet_total if n_wet_total > 0 else float("nan")
    per_ep_ratio = [
        (e["erosion_mass_ep"] / e["depot_mass_ep"] if e["depot_mass_ep"] > 0 else float("nan"))
        for e in episodes_diag
    ]
    total_depot_mass = sum(e["depot_mass_ep"] for e in episodes_diag)
    total_erosion_mass = sum(e["erosion_mass_ep"] for e in episodes_diag)
    aggregate_ratio = total_erosion_mass / total_depot_mass if total_depot_mass > 0 else float("nan")

    measure1 = dict(
        global_fraction_wet_cells_active=global_active_fraction,
        global_n_wet_total=n_wet_total,
        global_n_active_total=n_active_total,
        per_episode_ratio_erosion_over_depot=per_ep_ratio,
        per_episode_depot_mass=[e["depot_mass_ep"] for e in episodes_diag],
        per_episode_erosion_mass=[e["erosion_mass_ep"] for e in episodes_diag],
        total_depot_mass=total_depot_mass,
        total_erosion_mass=total_erosion_mass,
        aggregate_ratio_erosion_over_depot=aggregate_ratio,
    )

    # ============================= Mesure 2 ====================================
    total_depot_calm = sum(e["depot_calm"] for e in episodes_diag)
    total_depot_active = sum(e["depot_active"] for e in episodes_diag)
    total_depot_calm_active = total_depot_calm + total_depot_active
    frac_calm = total_depot_calm / total_depot_calm_active if total_depot_calm_active > 0 else float("nan")
    frac_active = total_depot_active / total_depot_calm_active if total_depot_calm_active > 0 else float("nan")
    mean_n_calm = float(np.mean([e["n_calm"] for e in episodes_diag]))
    mean_n_active_snap = float(np.mean([e["n_active_snap"] for e in episodes_diag]))

    measure2 = dict(
        total_depot_calm=total_depot_calm,
        total_depot_active=total_depot_active,
        fraction_depot_in_calm_snapshots=frac_calm,
        fraction_depot_in_active_snapshots=frac_active,
        mean_n_calm_snapshots_per_episode=mean_n_calm,
        mean_n_active_snapshots_per_episode=mean_n_active_snap,
        per_episode_n_calm=[e["n_calm"] for e in episodes_diag],
        per_episode_n_active_snap=[e["n_active_snap"] for e in episodes_diag],
        per_episode_n_snapshots=[e["n_snapshots"] for e in episodes_diag],
    )

    # ============================= Mesure 3 ====================================
    W_all = np.zeros_like(B0, dtype=np.float64)
    for e in episodes_diag:
        W_all += e["h_sum"]
    W_last = episodes_diag[-1]["h_sum"]

    corr_sfinal_W_all = _pearson(s_direct_nominal, W_all)
    corr_sfinal_W_last = _pearson(s_direct_nominal, W_last)

    measure3 = dict(
        corr_sfinal_W_all_episodes=corr_sfinal_W_all,
        corr_sfinal_W_last_episode=corr_sfinal_W_last,
        s_final_stats=dict(min=float(s_direct_nominal.min()), max=float(s_direct_nominal.max()),
                           mean=float(s_direct_nominal.mean()), sum=float(s_direct_nominal.sum())),
        W_all_stats=dict(min=float(W_all.min()), max=float(W_all.max()),
                         mean=float(W_all.mean()), sum=float(W_all.sum())),
        W_last_stats=dict(min=float(W_last.min()), max=float(W_last.max()),
                          mean=float(W_last.mean()), sum=float(W_last.sum())),
    )

    # ============================= Mesure 4 ====================================
    corr_nominal = _pearson(s_direct_nominal, s_reversed_nominal)
    corr_ke0 = _pearson(s_direct_ke0, s_reversed_ke0)

    measure4 = dict(
        corr_nominal_direct_vs_reversed=corr_nominal,
        corr_ke0_direct_vs_reversed=corr_ke0,
        k_e_nominal=PARAMS_NOMINAL.k_e,
        k_e_zero=PARAMS_KE0.k_e,
        delta_corr_ke0_minus_nominal=corr_ke0 - corr_nominal,
        sanity_check_vs_gate=dict(
            corr_reference_gate_step_a=_REF_GATE_CORR_NOMINAL,
            corr_measured_here=corr_nominal,
            delta=corr_nominal - _REF_GATE_CORR_NOMINAL,
        ),
    )

    elapsed_total = time.time() - t_start

    result = dict(
        meta=dict(
            seed=SEED, n_episodes=N_EPISODES,
            grid=dict(H=GRID.H, W=GRID.W, dx=GRID.dx, dy=GRID.dy),
            k_d=PARAMS_NOMINAL.k_d, k_e_nominal=PARAMS_NOMINAL.k_e, theta_c=PARAMS_NOMINAL.theta_c,
            centers=[list(c) for c in centers],
            elapsed_seconds=dict(direct_nominal=t_direct_nominal, reversed_nominal=t_reversed_nominal,
                                 direct_ke0=t_direct_ke0, reversed_ke0=t_reversed_ke0,
                                 total=elapsed_total),
        ),
        measure1_erosion_active=measure1,
        measure2_temporal_deposit=measure2,
        measure3_terrain_determinism=measure3,
        measure4_erosion_order_sensitivity=measure4,
    )

    (OUT_DIR / "diag_pathdep.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print("\n" + "=" * 78)
    print("DIAGNOSTIC ARC A -- MÉCANISME DE LA PATH-DEPENDENCE (substrat sédiment)")
    print("=" * 78)
    print(f"(1) Érosion active : fraction (snapshot,cellule mouillée) avec θ>θ_c = "
          f"{global_active_fraction:.4%}  (n_wet={n_wet_total}, n_active={n_active_total})")
    print(f"    Ratio masse érodée/déposée -- par épisode : "
          f"{['%.4f' % r for r in per_ep_ratio]}")
    print(f"    Ratio agrégé (Σérosion/Σdépôt, 10 épisodes) = {aggregate_ratio:.4f}")
    print(f"(2) Dépôt en snapshots calmes (max θ < θ_c/10) : {frac_calm:.4%}  vs actifs : "
          f"{frac_active:.4%}")
    print(f"    Nb moyen de snapshots calmes/actifs par épisode : "
          f"{mean_n_calm:.1f} / {mean_n_active_snap:.1f}  (sur {episodes_diag[0]['n_snapshots']} au total)")
    print(f"(3) corr(s_final, W=Σ_épisodes Σ_snapshots h) = {corr_sfinal_W_all:.4f}")
    print(f"    corr(s_final, W_dernier_épisode_seul)      = {corr_sfinal_W_last:.4f}")
    print(f"(4) corr(s_direct,s_inversé) k_e nominal={PARAMS_NOMINAL.k_e:.6f} : {corr_nominal:.4f}")
    print(f"    corr(s_direct,s_inversé) k_e=0 (érosion coupée)         : {corr_ke0:.4f}")
    print(f"    Δcorr (k_e=0 − nominal) = {corr_ke0 - corr_nominal:+.4f}")
    print(f"    Sanity check vs gate (step_a.json, corr={_REF_GATE_CORR_NOMINAL:.4f}) : "
          f"mesuré ici = {corr_nominal:.4f}  (Δ={corr_nominal - _REF_GATE_CORR_NOMINAL:+.6f})")
    print("-" * 78)
    print(f"Temps total : {elapsed_total:.1f}s")
    print(f"[REPORT] -> {OUT_DIR / 'diag_pathdep.json'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
