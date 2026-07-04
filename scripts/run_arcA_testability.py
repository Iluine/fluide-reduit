"""M-0bis — Sonde de testabilité de la manche 1 sur le substrat v2 (Arc A).

Contrat qui fait foi : entrée « M-0bis PRÉ-ENREGISTRÉE » de `PREREGISTRATION.md`
(dépôt pocCascade2phys, commit `ae1c926`), reproduite dans
`.superpowers/sdd/refond-task2-m0bis-brief.md`. Aucun seuil ni protocole n'est
modifiable après première mesure.

Claim testé : le résumé grossier déterministe-régénérable `src/summary.py`
(ℓ ∈ {1,2,3}) tient-il le readout albedo sous le JND le plus sévère de la plage
[2,5] % -- à la fois INSTANTANÉMENT (M-A1) et sous un pas de dynamique (M-A2-mini)
-- sur le substrat v2 GELÉ (`SedimentParams()` par défaut), pour les DEUX histoires
même-multiset d'ordre inversé du protocole (a) du gate v2 ?

Champs sondés : `s_direct` et `s_inversé`, reproduits EXACTEMENT via les imports
de `scripts/run_arcA_m0.py` (BOX, N_EPISODES, SEED) et `scripts/run_arcA_revalidate.py`
(B0, GRID, PARAMS, RELIEF, S_HALF_OP, `_draw_centers`, `_run_sequence`) -- ni l'un
ni l'autre de ces deux scripts n'est modifié ni ré-exécuté ici.

M-A1 instantané, pour chaque histoire x ℓ :
  delta_chi(albedo(regenerate(summarize(s, ℓ)), S_HALF_OP), albedo(s, S_HALF_OP))
(critère = clé "max_carrier" = max sur bandes porteuses de `src.albedo.delta_chi`).

M-A2-mini, pour chaque histoire x ℓ : centre du rollout = 11e tirage de la même rng
seed 7 (10 premiers centres = ceux de M-0/du protocole (a), tirés en CONTINUANT la
même rng -- identité bit-à-bit avec M-0 vérifiée par assertion ci-dessous, PAS
seulement supposée). Depuis `s_regen = regenerate(summarize(s, ℓ))` ET depuis
`s_vrai = s`, un épisode complet (`run_episode_trajectoire`, additif dans
`src/sediment.py`, MÊME physique que `run_episode`) au 11e centre, en récupérant
les `s` intermédiaires à CHAQUE snapshot d'intégration Exner. Delta_chi entre les
deux trajectoires d'albedo à chaque snapshot k ; statistique = max le long du
rollout (+ argmax rapporté pour audit).

Verdict mécanique (seuils FIGÉS, lectures pré-écrites) :
  - Niveaux SOUS CAP = ℓ=3 (81 floats) et ℓ=2 (273) ; ℓ=1 (1041) hors-cap, indicatif.
  - T_triviale ssi pour TOUS les niveaux sous cap ET les DEUX histoires :
    max(M-A1, M-A2-mini) < 0.02 (JND le plus sévère de la plage [2,5] %).
  - T_testable sinon.
AUCUNE interprétation au-delà de cette règle mécanique -- les deux issues sont
également acceptables, rien n'est ajusté pour privilégier l'une ou l'autre.

Diagnostics rapportés (non porteurs) : f_close par histoire x ℓ (convention M-0,
`_f_ordre` de `scripts/run_arcA_m0.py` réutilisée telle quelle -- formule
symétrique identique) entre A_regen et A_vrai (comparaison INSTANTANÉE, même paire
que M-A1), jnd ∈ {2,3,4,5} %, au point d'opération S_HALF_OP.

Coût réel : 2 histoires x 10 épisodes (~2-4 min, partagé entre M-A1 et M-A2-mini)
+ 2 histoires x 3 niveaux x 2 trajectoires x 1 épisode (~40 s) -- quelques minutes
au total, un seul run (pas de décomposition en phases).

Usage :
  .venv/bin/python scripts/run_arcA_testability.py

Sortie : `outputs/arcA/testability_v2.json` (+ tableau imprimé). Déterministe
(seeds figés, aucune horloge dans le calcul -- seul `elapsed_seconds` est un
artefact de mesure, hors calcul) : un double run produit un JSON identique hors
`elapsed_seconds`."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.run_arcA_m0 import BOX, N_EPISODES, SEED, _f_ordre
from scripts.run_arcA_revalidate import (B0, GRID, PARAMS, RELIEF, S_HALF_OP,
                                         _draw_centers, _run_sequence)
from src.albedo import albedo, delta_chi
from src.sediment import run_episode_trajectoire
from src.summary import regenerate, size_floats, summarize

OUT_DIR = ROOT / "outputs" / "arcA"
OUT_PATH = OUT_DIR / "testability_v2.json"

LEVELS: tuple[int, ...] = (1, 2, 3)
# Niveaux "sous cap" (spec M-0bis) -- ℓ=1 (1041 floats, > 4096*10% n'a pas de sens
# comme "compression" utile) est hors-cap, reporté seulement à titre indicatif.
CAP_LEVELS: tuple[int, ...] = (2, 3)
# Balayage f_close (diagnostic non porteur, convention M-0).
JND_PCT_LIST_FCLOSE: tuple[int, ...] = (2, 3, 4, 5)
# Seuil FIGÉ du verdict mécanique -- JND le plus sévère de la plage [2,5] %.
THRESH_TESTABILITE: float = 0.02


def _delta_chi_albedo(s_regen: np.ndarray, s_vrai: np.ndarray) -> dict:
    """Δχ(albedo(s_regen, S_HALF_OP), albedo(s_vrai, S_HALF_OP)) -- dict complet
    (cf. `src.albedo.delta_chi`) : {"delta_chi": {...}, "carriers": {...},
    "max_carrier": float}. Valeurs converties en types Python natifs (le calcul
    interne de `chi_bands`/`delta_chi` produit des scalaires `numpy.float64`/
    `numpy.bool`, non sérialisables JSON tels quels) -- `src/albedo.py` n'est PAS
    modifié, la conversion se fait ici, à la frontière de sérialisation."""
    a_regen = albedo(s_regen, S_HALF_OP)
    a_vrai = albedo(s_vrai, S_HALF_OP)
    d = delta_chi(a_regen, a_vrai)
    return dict(
        delta_chi={k: float(v) for k, v in d["delta_chi"].items()},
        carriers={k: bool(v) for k, v in d["carriers"].items()},
        max_carrier=float(d["max_carrier"]),
    )


def _m_a1(s_vrai: np.ndarray, level: int) -> dict:
    """M-A1 instantané (cf. docstring module)."""
    s_regen = regenerate(summarize(s_vrai, level))
    return _delta_chi_albedo(s_regen, s_vrai)


def _m_a2_mini(s_vrai: np.ndarray, level: int,
              center_rollout: tuple[float, float]) -> dict:
    """M-A2-mini (cf. docstring module) : max le long du rollout + argmax (audit)."""
    s_regen = regenerate(summarize(s_vrai, level))
    traj_regen = run_episode_trajectoire(s_regen, B0, center_rollout, PARAMS)
    traj_vrai = run_episode_trajectoire(s_vrai, B0, center_rollout, PARAMS)
    assert len(traj_regen) == len(traj_vrai), (
        "M-A2-mini : les deux trajectoires (regen/vrai) n'ont pas la même longueur "
        f"-- {len(traj_regen)} vs {len(traj_vrai)}.")
    dchi_par_snapshot = [_delta_chi_albedo(sr, sv) for sr, sv in zip(traj_regen, traj_vrai)]
    max_carriers = [d["max_carrier"] for d in dchi_par_snapshot]
    argmax = int(np.argmax(max_carriers))
    d_argmax = dchi_par_snapshot[argmax]
    return dict(max_carrier=max_carriers[argmax], argmax_snapshot=argmax,
                n_snapshots=len(max_carriers), delta_chi=d_argmax["delta_chi"],
                carriers=d_argmax["carriers"])


def _f_close(s_vrai: np.ndarray, level: int) -> dict[str, float]:
    """Diagnostic (non porteur) : f_close par histoire x ℓ, convention M-0
    (`_f_ordre` réutilisée telle quelle, formule symétrique identique) entre
    A_regen et A_vrai (comparaison INSTANTANÉE, même paire que M-A1)."""
    s_regen = regenerate(summarize(s_vrai, level))
    a_regen = albedo(s_regen, S_HALF_OP)
    a_vrai = albedo(s_vrai, S_HALF_OP)
    return {str(jnd): _f_ordre(a_regen, a_vrai, jnd / 100.0) for jnd in JND_PCT_LIST_FCLOSE}


def main() -> None:
    t0 = time.time()

    # Histoires directe/inversée -- protocole (a) du gate v2, réutilisé bit-à-bit
    # (mêmes seed/n_episodes/boîte que scripts/run_arcA_m0.py / _revalidate.py::phase_a).
    # 11e centre : MÊME rng continuée (pas de reset) -- garantit bit-à-bit les 10
    # premiers centres identiques à M-0, vérifié par assertion ci-dessous.
    rng = np.random.default_rng(SEED)
    centers_10 = _draw_centers(rng, N_EPISODES, BOX, BOX)
    center_rollout = _draw_centers(rng, 1, BOX, BOX)[0]

    rng_ref = np.random.default_rng(SEED)
    centers_10_ref = _draw_centers(rng_ref, N_EPISODES, BOX, BOX)
    assert centers_10 == centers_10_ref, (
        "Les 10 premiers centres du tirage à 11 ne correspondent pas bit-à-bit à "
        "ceux de M-0 -- protocole rompu, STOP.")

    s_direct = _run_sequence(centers_10)
    s_inverse = _run_sequence(list(reversed(centers_10)))
    histoires: dict[str, np.ndarray] = {"direct": s_direct, "inverse": s_inverse}

    tailles: dict[str, int] = {
        str(level): size_floats(summarize(s_direct, level)) for level in LEVELS
    }

    m_a1: dict[str, dict[str, dict]] = {}
    m_a2_mini: dict[str, dict[str, dict]] = {}
    f_close: dict[str, dict[str, dict[str, float]]] = {}
    for nom, s_vrai in histoires.items():
        m_a1[nom] = {}
        m_a2_mini[nom] = {}
        f_close[nom] = {}
        for level in LEVELS:
            m_a1[nom][str(level)] = _m_a1(s_vrai, level)
            m_a2_mini[nom][str(level)] = _m_a2_mini(s_vrai, level, center_rollout)
            f_close[nom][str(level)] = _f_close(s_vrai, level)

    # Verdict mécanique (seuils FIGÉS) -- niveaux sous cap = {2,3}, DEUX histoires.
    conditions_sous_cap: dict[str, dict[str, bool]] = {}
    tous_sous_seuil = True
    for nom in histoires:
        conditions_sous_cap[nom] = {}
        for level in CAP_LEVELS:
            valeur = max(m_a1[nom][str(level)]["max_carrier"],
                        m_a2_mini[nom][str(level)]["max_carrier"])
            cond = valeur < THRESH_TESTABILITE
            conditions_sous_cap[nom][str(level)] = cond
            tous_sous_seuil = tous_sous_seuil and cond

    verdict = "T_triviale" if tous_sous_seuil else "T_testable"

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
            centers_10=[list(c) for c in centers_10],
            center_rollout=list(center_rollout),
            s_half_op=S_HALF_OP,
            levels=list(LEVELS),
            cap_levels=list(CAP_LEVELS),
            tailles_floats=tailles,
            champ_fin_floats=GRID.H * GRID.W,
            seuil_testabilite=THRESH_TESTABILITE,
            elapsed_seconds=elapsed,
        ),
        m_a1=m_a1,
        m_a2_mini=m_a2_mini,
        f_close=f_close,
        conditions_sous_cap=conditions_sous_cap,
        verdict=verdict,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print("=" * 78)
    print("M-0bis -- SONDE DE TESTABILITÉ MANCHE 1 (substrat v2)")
    print("=" * 78)
    print(f"Tailles (floats) : {tailles}  (champ fin = {GRID.H * GRID.W})")
    print(f"11e centre (rollout, IDENTIQUE pour les deux histoires) : {center_rollout}")
    for nom in histoires:
        print(f"-- histoire {nom} --")
        for level in LEVELS:
            a1 = m_a1[nom][str(level)]["max_carrier"]
            a2 = m_a2_mini[nom][str(level)]["max_carrier"]
            cap = "sous cap" if level in CAP_LEVELS else "hors cap (indicatif)"
            fclose5 = f_close[nom][str(level)]["5"]
            print(f"  l={level} ({tailles[str(level)]} floats, {cap}) : "
                  f"M-A1={a1:.5f}  M-A2-mini={a2:.5f}  max={max(a1, a2):.5f}  "
                  f"f_close(5%)={fclose5:.4f}")
    print("-" * 78)
    print(f"Conditions par (histoire, niveau sous cap) [< {THRESH_TESTABILITE}] : "
          f"{conditions_sous_cap}")
    print(f"Verdict mécanique : {verdict}")
    print(f"Temps de calcul : {elapsed:.1f}s")
    print(f"[REPORT] -> {OUT_PATH}")


if __name__ == "__main__":
    main()
