"""Arc A / Task 2 — GATE de re-validation (BLOQUANT) du substrat sédiment reconstruit.

Claim testé : le substrat sédiment reconstruit (Task 1, dépôt/érosion Exner sur
`simulate_wetdry_o2` figé) reproduit les TROIS propriétés gravées au journal
d'origine (pocCascade2phys, 2026-07-03) :
  (a) path-dependence — l'ORDRE des pulses affecte l'état final (pas seulement le
      multiset de centres) ;
  (b) survie en readout albedo — un signal invisible en relief (ombrage) reste
      détectable en albedo (readout sédiment, saturant) ;
  (c) closure f(k) — deux histoires jumelles (préfixe divergent inversé, suffixe
      commun) reconvergent quand le préfixe divergent raccourcit (k grandit).

Seuils FIGÉS AVANT exécution (`.superpowers/sdd/task-2-brief.md`, section « Task 2 »),
appliqués MÉCANIQUEMENT — un seuil manqué est un RÉSULTAT à rapporter, jamais
retouché ni ajusté a posteriori :
  (a) corr(s_A, s_B) <= 0.7                              (référence d'origine : 0.39)
  (b) f_albedo(point d'op, JND=5%) >= 2%  ET  f_albedo(S_HALF) > f_relief pour
      TOUT S_HALF dans le balayage {0.005,0.01,0.05,0.10,0.20}*relief
                                                           (référence : 3.5-10% >> 1.2%)
  (c) f(k) non-croissante (tolérance +0.002, bruit de grille) ET f(k_max) < 5%,
      aux DEUX géométries (overlap : bande y in [0.4,0.6] ; séparée : boîte pleine)

Protocole exact (tirage des centres, construction du pulse simultané) : cf.
RÉSOLUTIONS D'AMBIGUÏTÉ du contrôleur dans task-2-brief.md. En résumé :
  (a) séquence A = 10 centres de default_rng(7) dans [0.15,0.85]² ; séquence B =
      la même liste INVERSÉE. corr = Pearson(s_A.ravel(), s_B.ravel()).
  (b) s_ordre = s_A de (a). Le pulse « simultané-même-masse » = UN épisode dont
      le pulse initial est la SOMME des 10 gaussiennes (mêmes centres/sigma/h_p),
      relaxé + intégré Exner exactement comme un épisode normal (réutilise les
      fonctions internes de src.sediment, cf. `_run_simultaneous` ci-dessous —
      `run_episode` n'accepte qu'un centre unique). s rescalé multiplicativement
      pour Σs_sim = Σs_ordre.
  (c) pour chaque géométrie et k in {1,4,7,10} : un unique rng(seed) tire 10
      centres ; préfixe = les 10-k premiers, suffixe = les k derniers. Histoire X
      = préfixe+suffixe (= le tirage original, dans l'ordre). Histoire Y =
      préfixe INVERSÉ + même suffixe. f(k) = f_fraction(albedo(s_X,S_HALF_op),
      albedo(s_Y,S_HALF_op), jnd=0.05). Seeds : 1000+k (overlap), 2000+k (séparée).

Coût réel (10 épisodes/histoire, grille 64x64, ~4-5s/épisode CPU mesuré) : (a)
~20 épisodes, (b) ~1 relaxation (pulse 10x plus massif), (c) ~140 épisodes (2
géométries x 4 k x 2 histoires, avec réutilisation bit-exacte de l'histoire
k=n_episodes où le préfixe est vide) -> largement > 600s (limite d'un appel
Bash) : le script est décomposé en phases relançables, chacune persistant son
résultat intermédiaire sous `outputs/arcA/` (durabilité + reprise).

Usage :
  .venv/bin/python scripts/run_arcA_revalidate.py --phase a
  .venv/bin/python scripts/run_arcA_revalidate.py --phase b
  .venv/bin/python scripts/run_arcA_revalidate.py --phase c_overlap
  .venv/bin/python scripts/run_arcA_revalidate.py --phase c_separee
  .venv/bin/python scripts/run_arcA_revalidate.py --phase report
  .venv/bin/python scripts/run_arcA_revalidate.py --phase all --smoke   # plomberie, PAS un verdict

Le script n'ajuste JAMAIS un seuil ni un paramètre physique, quel que soit le
verdict. Sorties : `outputs/arcA/revalidation.json` (+ tableau imprimé) et les
intermédiaires `outputs/arcA/step_{a,b,c_overlap,c_separee}.{npz,json}`.
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

from config import GridConfig
from src.albedo import albedo, f_fraction, relief_shaded
from src.sediment import (SedimentParams, _T_END_RELAX, _integrate_exner,
                          default_terrain, run_episode)
from src.solver_wetdry import simulate_wetdry_o2

OUT_DIR = ROOT / "outputs" / "arcA"

# Tolérance numérique pour "non-croissante" en (c) : bruit de grille (RÉSOLUTION
# D'AMBIGUÏTÉ du contrôleur, task-2-brief.md) — comparaison entre points
# consécutifs f(k_i) <= f(k_{i-1}) + _NONINCREASING_TOL.
_NONINCREASING_TOL: float = 0.002

GRID = GridConfig()
B0 = default_terrain(GRID)
RELIEF = float(B0.max() - B0.min())
PARAMS = SedimentParams()
S_HALF_FRACTIONS: tuple[float, ...] = (0.005, 0.01, 0.05, 0.10, 0.20)
S_HALF_OP: float = 0.05 * RELIEF  # point d'opération (Task 1)

# Seuils FIGÉS (task-2-brief.md, section Task 2) — ne JAMAIS ajuster ici.
THRESH_A_CORR: float = 0.7
THRESH_B_FALBEDO_OP: float = 0.02
THRESH_C_FLAST: float = 0.05

_GEOMETRIES: dict[str, dict] = {
    "overlap": dict(x_range=(0.15, 0.85), y_range=(0.4, 0.6), seed_base=1000),
    "separee": dict(x_range=(0.15, 0.85), y_range=(0.15, 0.85), seed_base=2000),
}


# --- Construction des centres / séquences ------------------------------------


def _draw_centers(rng: np.random.Generator, n: int,
                  x_range: tuple[float, float],
                  y_range: tuple[float, float]) -> list[tuple[float, float]]:
    """Tire `n` centres (x_frac, y_frac) i.i.d. uniformes, x puis y par centre
    (même pattern d'appel que `run_history`, bornes généralisées à la géométrie)."""
    return [(float(rng.uniform(*x_range)), float(rng.uniform(*y_range))) for _ in range(n)]


def _run_sequence(centers: list[tuple[float, float]]) -> np.ndarray:
    """Boucle `run_episode` sur une liste de centres explicite, s=0 initial.
    (Résolution du contrôleur : séquences de centres CONTRÔLÉES -> tirage +
    boucle manuels ; `run_history` n'est ni éditée ni utilisée ici, sa signature
    ne permettant pas de centres explicites.)"""
    s = np.zeros_like(B0, dtype=np.float64)
    for c in centers:
        s = run_episode(s, B0, c, PARAMS)
    return s


def _gaussian_pulse(grid: GridConfig, center_frac: tuple[float, float],
                    sigma: float, amplitude: float) -> np.ndarray:
    """Monticule gaussien isolé — réplique EXACTE de la formule interne de
    `src.sediment._initial_pulse` (même convention de grille : centre en
    fractions * (dim-1), r2 en cellules²). Exposée séparément pour permettre la
    SOMME de plusieurs monticules : `run_episode`/`_relax_episode` n'acceptent
    qu'un centre unique, or (b) exige un pulse simultané multi-centres
    (superposition), cf. RÉSOLUTION D'AMBIGUÏTÉ."""
    H, W = grid.H, grid.W
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    cxf, cyf = center_frac
    cx, cy = cxf * (W - 1), cyf * (H - 1)
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    return amplitude * np.exp(-r2 / (2.0 * sigma ** 2))


def _run_simultaneous(centers: list[tuple[float, float]]) -> np.ndarray:
    """UN épisode dont le pulse initial = superposition des monticules aux
    `centers` (somme des gaussiennes, mêmes sigma_pulse/h_p que les épisodes
    ordonnés), s=0 initial. Relaxation + intégration Exner IDENTIQUES à celles
    de `src.sediment.run_episode` (réutilise `_T_END_RELAX`/`_integrate_exner`
    telles quelles, même garde-fou de troncature à N_settle) — seul le pulse
    initial diffère (un centre unique -> N centres superposés)."""
    h0 = np.zeros_like(B0, dtype=np.float64)
    for c in centers:
        h0 = h0 + _gaussian_pulse(GRID, c, PARAMS.sigma_pulse, PARAMS.h_p)
    hu0 = np.zeros_like(h0)
    hv0 = np.zeros_like(h0)
    times, hs, hus, hvs = simulate_wetdry_o2(h0, hu0, hv0, B0, GRID, t_end=_T_END_RELAX)
    n_available = len(times) - 1
    if n_available < PARAMS.N_settle:
        raise RuntimeError(
            f"_T_END_RELAX={_T_END_RELAX} insuffisant pour le pulse simultané "
            f"({len(centers)} centres superposés) : {n_available} pas acceptés, "
            f"{PARAMS.N_settle} requis. Ne PAS augmenter silencieusement -- "
            "remonter au contrôleur (paramètre figé).")
    n = PARAMS.N_settle + 1
    s0 = np.zeros_like(B0, dtype=np.float64)
    return _integrate_exner(s0, times[:n], hs[:n], hus[:n], hvs[:n], PARAMS)


# --- Phases --------------------------------------------------------------


def phase_a(n_episodes: int) -> dict:
    """(a) path-dependence : corr(s_A, s_B) avec B = A à l'ordre inversé."""
    t0 = time.time()
    rng = np.random.default_rng(7)
    centers = _draw_centers(rng, n_episodes, (0.15, 0.85), (0.15, 0.85))
    s_A = _run_sequence(centers)
    s_B = _run_sequence(list(reversed(centers)))
    corr = float(np.corrcoef(s_A.ravel(), s_B.ravel())[0, 1])
    elapsed = time.time() - t0
    verdict = "PASS" if corr <= THRESH_A_CORR else "FAIL"
    result = dict(n_episodes=n_episodes, seed=7, centers=[list(c) for c in centers],
                  corr=corr, threshold=THRESH_A_CORR, reference_origine=0.39,
                  verdict=verdict, elapsed_seconds=elapsed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_DIR / "step_a.npz", s_A=s_A, s_B=s_B)
    (OUT_DIR / "step_a.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[A] corr(s_A,s_B)={corr:.4f} (seuil <= {THRESH_A_CORR}) -> {verdict}  [{elapsed:.1f}s]")
    return result


def phase_b(n_episodes: int) -> dict:
    """(b) survie albedo : ordre (s_A de (a)) vs simultané-même-masse."""
    t0 = time.time()
    step_a = json.loads((OUT_DIR / "step_a.json").read_text())
    if step_a["n_episodes"] != n_episodes:
        raise RuntimeError(
            f"phase b: n_episodes={n_episodes} != step_a.n_episodes="
            f"{step_a['n_episodes']} -- relancer la phase a avec le même réglage d'abord.")
    centers = [tuple(c) for c in step_a["centers"]]
    npz_a = np.load(OUT_DIR / "step_a.npz")
    s_ordre = npz_a["s_A"]

    s_sim_raw = _run_simultaneous(centers)
    raw_sum = float(s_sim_raw.sum())
    if raw_sum == 0.0:
        raise RuntimeError("phase b: s_sim_raw.sum()==0 -- rescale mal défini (0/0).")
    scale = float(s_ordre.sum() / raw_sum)
    s_sim = s_sim_raw * scale

    f_relief = f_fraction(relief_shaded(B0 + s_ordre), relief_shaded(B0 + s_sim), 0.05)
    f_albedo_by_frac: dict[str, float] = {}
    for frac in S_HALF_FRACTIONS:
        s_half = frac * RELIEF
        f_albedo_by_frac[f"{frac:.3f}"] = f_fraction(
            albedo(s_ordre, s_half), albedo(s_sim, s_half), 0.05)
    f_albedo_op = f_albedo_by_frac[f"{0.05:.3f}"]

    cond_op = f_albedo_op >= THRESH_B_FALBEDO_OP
    cond_dominance = all(v > f_relief for v in f_albedo_by_frac.values())
    verdict = "PASS" if (cond_op and cond_dominance) else "FAIL"

    elapsed = time.time() - t0
    result = dict(n_episodes=n_episodes, s_half_op_fraction=0.05, s_half_op=S_HALF_OP,
                  scale_rescale=scale, sum_s_ordre=float(s_ordre.sum()), sum_s_sim_raw=raw_sum,
                  f_relief=f_relief, f_albedo_by_shalf_fraction=f_albedo_by_frac,
                  f_albedo_op=f_albedo_op, threshold_f_albedo_op=THRESH_B_FALBEDO_OP,
                  cond_op_ge_2pct=cond_op, cond_albedo_dominates_relief_all_shalf=cond_dominance,
                  reference_origine="3.5-10% >> 1.2%", verdict=verdict, elapsed_seconds=elapsed)
    np.savez(OUT_DIR / "step_b.npz", s_sim=s_sim, s_sim_raw=s_sim_raw)
    (OUT_DIR / "step_b.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[B] f_albedo_op={f_albedo_op:.4f} (seuil >= {THRESH_B_FALBEDO_OP}), "
          f"f_relief={f_relief:.4f}, f_albedo par S_HALF={f_albedo_by_frac} -> {verdict}"
          f"  [{elapsed:.1f}s]")
    return result


def phase_c(geometry: str, n_episodes: int, k_list: tuple[int, ...]) -> dict:
    """(c) closure f(k) pour une géométrie donnée."""
    t0 = time.time()
    geo = _GEOMETRIES[geometry]
    f_by_k: dict[int, float] = {}
    for k in k_list:
        seed = geo["seed_base"] + k
        rng = np.random.default_rng(seed)
        centers_full = _draw_centers(rng, n_episodes, geo["x_range"], geo["y_range"])
        prefix = centers_full[: n_episodes - k]
        suffix = centers_full[n_episodes - k:]
        centers_X = prefix + suffix
        centers_Y = list(reversed(prefix)) + suffix
        s_X = _run_sequence(centers_X)
        if prefix == list(reversed(prefix)):
            # préfixe vide ou singleton -> X et Y bit-identiques par construction
            # (cf. RÉSOLUTION D'AMBIGUÏTÉ : f(k=n_episodes)=0) ; réutiliser évite
            # un run redondant de n_episodes épisodes.
            s_Y = s_X
        else:
            s_Y = _run_sequence(centers_Y)
        f_by_k[k] = f_fraction(albedo(s_X, S_HALF_OP), albedo(s_Y, S_HALF_OP), 0.05)

    ks_sorted = sorted(k_list)
    fk_sorted = [f_by_k[k] for k in ks_sorted]
    noncroissante = all(fk_sorted[i + 1] <= fk_sorted[i] + _NONINCREASING_TOL
                        for i in range(len(fk_sorted) - 1))
    f_last = f_by_k[ks_sorted[-1]]
    cond_f_last = f_last < THRESH_C_FLAST
    verdict = "PASS" if (noncroissante and cond_f_last) else "FAIL"

    elapsed = time.time() - t0
    result = dict(geometry=geometry, n_episodes=n_episodes, k_list=list(ks_sorted),
                  x_range=list(geo["x_range"]), y_range=list(geo["y_range"]),
                  f_by_k={str(k): v for k, v in f_by_k.items()},
                  noncroissante_tol=_NONINCREASING_TOL, cond_noncroissante=noncroissante,
                  f_last_k=ks_sorted[-1], f_last=f_last, threshold_f_last=THRESH_C_FLAST,
                  cond_f_last_lt_seuil=cond_f_last, verdict=verdict, elapsed_seconds=elapsed)
    (OUT_DIR / f"step_c_{geometry}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[C/{geometry}] f(k)={result['f_by_k']}  non-croissante(tol {_NONINCREASING_TOL})="
          f"{noncroissante}  f({ks_sorted[-1]})={f_last:.4f} < {THRESH_C_FLAST} -> {verdict}"
          f"  [{elapsed:.1f}s]")
    return result


def phase_report() -> dict:
    """Assemble les 4 intermédiaires, calcule les verdicts (a)/(b)/(c) et le
    verdict global, écrit `revalidation.json`, imprime le tableau lisible."""
    a = json.loads((OUT_DIR / "step_a.json").read_text())
    b = json.loads((OUT_DIR / "step_b.json").read_text())
    c_overlap = json.loads((OUT_DIR / "step_c_overlap.json").read_text())
    c_separee = json.loads((OUT_DIR / "step_c_separee.json").read_text())

    c_verdict = "PASS" if (c_overlap["verdict"] == "PASS"
                           and c_separee["verdict"] == "PASS") else "FAIL"
    global_verdict = "PASS" if (a["verdict"] == "PASS" and b["verdict"] == "PASS"
                                and c_verdict == "PASS") else "FAIL"
    total_elapsed = (a.get("elapsed_seconds", 0.0) + b.get("elapsed_seconds", 0.0)
                     + c_overlap.get("elapsed_seconds", 0.0) + c_separee.get("elapsed_seconds", 0.0))

    report = dict(
        meta=dict(grid=dict(H=GRID.H, W=GRID.W, dx=GRID.dx, dy=GRID.dy), relief=RELIEF,
                  kd_calibre=PARAMS.k_d, s_half_op=S_HALF_OP,
                  total_elapsed_seconds=total_elapsed),
        a_path_dependence=a,
        b_survie_albedo=b,
        c_closure_fk=dict(overlap=c_overlap, separee=c_separee, verdict=c_verdict),
        verdict_global=global_verdict,
    )
    (OUT_DIR / "revalidation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print("\n" + "=" * 78)
    print("ARC A / TASK 2 -- GATE DE RE-VALIDATION")
    print("=" * 78)
    print(f"(a) path-dependence   corr(s_A,s_B) = {a['corr']:.4f}  (seuil <= {a['threshold']})"
          f"  -> {a['verdict']}")
    print(f"(b) survie albedo     f_albedo_op = {b['f_albedo_op']:.4f}  (seuil >= "
          f"{b['threshold_f_albedo_op']}) ; f_relief = {b['f_relief']:.4f}  -> {b['verdict']}")
    for frac, val in b["f_albedo_by_shalf_fraction"].items():
        tag = "> f_relief OK" if val > b["f_relief"] else "<= f_relief !!"
        print(f"      S_HALF={float(frac):.3f}*relief : f_albedo={val:.4f}  ({tag})")
    for geom_name, geom in (("overlap", c_overlap), ("separee", c_separee)):
        print(f"(c) closure f(k) [{geom_name}]  f={geom['f_by_k']}  non-croissante="
              f"{geom['cond_noncroissante']}  f({geom['f_last_k']})={geom['f_last']:.4f} < "
              f"{geom['threshold_f_last']}  -> {geom['verdict']}")
    print("-" * 78)
    print(f"VERDICT GLOBAL : {global_verdict}")
    print(f"Temps de calcul cumulé (phases a+b+c) : {total_elapsed:.1f}s")
    print("=" * 78)
    if global_verdict != "PASS":
        print("[GATE] Au moins un critère manqué -> STOP. Ne pas lancer les tâches "
              "suivantes du plan Arc A, ne PAS retoucher les paramètres.")
    print(f"[REPORT] -> {OUT_DIR / 'revalidation.json'}")
    return report


# --- CLI -----------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=["a", "b", "c_overlap", "c_separee", "report", "all"],
                        default="all")
    parser.add_argument("--smoke", action="store_true",
                        help="Plomberie uniquement : n_episodes/k_list réduits. "
                             "NE constitue PAS un verdict de gate.")
    args = parser.parse_args()

    n_episodes = 3 if args.smoke else 10
    k_list: tuple[int, ...] = (1, n_episodes) if args.smoke else (1, 4, 7, 10)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        print(f"[SMOKE] n_episodes={n_episodes}, k_list={k_list} -- plomberie seulement, "
              "PAS un verdict de gate.")

    if args.phase in ("a", "all"):
        phase_a(n_episodes)
    if args.phase in ("b", "all"):
        phase_b(n_episodes)
    if args.phase in ("c_overlap", "all"):
        phase_c("overlap", n_episodes, k_list)
    if args.phase in ("c_separee", "all"):
        phase_c("separee", n_episodes, k_list)
    if args.phase in ("report", "all"):
        phase_report()


if __name__ == "__main__":
    main()
