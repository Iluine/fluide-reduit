"""F1 — M-b : le BRAS EPS = 0, discriminateur PAR CANAL (§A32-décomposition).

POURQUOI CE BRAS. La décomposition spatiale a montré que les DEUX régions
échouent séparément (18/18 de part et d'autre du pin). Elle ne pouvait pas
aller plus loin : les deux régions passent par LE MÊME canal, et un défaut de
canal ne se voit pas dans une coupe spatiale. Il faut un discriminateur par
CANAL — celui-ci.

L'ÉCHELLE DES TROIS BRAS, une fois celui-ci ajouté :
    témoin      = (a) f32 seul                  — plein domaine, un niveau
    EPS = 0     = (a) + (c) structure fovéale   — canal TRANSPARENT
    production  = (a) + (b) + (c)               — canal seuillé à 1e-2
La différence EPS0 − témoin isole (c) ; production − EPS0 isole (b).

────────────────────────────────────────────────────────────────────────
CE QUI EMPÊCHERAIT CE BRAS D'ÊTRE UN VRAI PLANCHER — dit AVANT de courir
────────────────────────────────────────────────────────────────────────
1. **LE CAP 10 % MORDRAIT À LA PLACE DU SEUIL.** À EPS = 0, `|d| >= 0` est
   vrai PARTOUT : 4096 candidats se présentent pour `budget_k_fen(4096)` =
   409 places. Avec le budget de la cellule, ce bras mesurerait le CAP, pas
   la structure — et ne trancherait rien. D'où `BUDGET_PLEIN = H·W` :
   toutes les cellules passent, le canal ne retire plus rien. C'est vérifié
   par test, pas supposé.
2. **CAPACITÉ DE BUFFER : sans objet ici, et c'est une LIMITE À NOMMER.** La
   remontée de ce pipeline est le modèle numpy du canal (`remonter_seuillee`,
   option 1), pas le compacteur GPU `CompacteurL3` avec son `taille_max`. Ce
   bras établit donc un plancher pour LE CANAL TEL QUE MODÉLISÉ ; il ne dit
   rien d'une saturation de buffer du L3 réel, qui n'est pas exercé ici.
3. **RÉSIDU D'ARRONDI f32.** Le schéma incrémental `ref += (champ − ref)`
   laisse ~1e-8 relatif (mesuré 1,5e-11 absolu). Sept ordres sous le pin :
   le canal ne peut pas porter le plancher.
4. **CADENCE.** La remontée est faite à CHAQUE épisode et les émissions
   tombent en fin d'épisode : à budget plein, la connaissance n'est jamais
   périmée. Aucune staleness ne se déguise en plancher.

────────────────────────────────────────────────────────────────────────
LECTURES PRÉ-ÉCRITES, CÂBLÉES
────────────────────────────────────────────────────────────────────────
  PASSE  ⇒ la cause est la BANDE MORTE du seuil ; le remède est EPS, et
           l'échelle à laquelle §A23 l'avait validé (état synthétisé V4)
           est À RE-POSER.
  ÉCHOUE ⇒ le plancher est STRUCTUREL (décimation + halo) ; l'architecture
           fovéale telle que construite ne peut pas satisfaire É2 —
           question de SPEC, pas de réglage.

CE BRAS ATTRIBUE, IL NE RÉHABILITE PAS. Le verdict MORT-b n'est pas rouvert :
la production reste ce qu'elle est, et ce bras ne la remplace pas. Il dit
seulement OÙ est la cause. Porté dans la lecture.

Usage :  .venv/bin/python scripts/run_f1_mb_t2_eps0.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import GridConfig  # noqa: E402
from src.f1_gpu.backend import exiger_cupy  # noqa: E402
from src.f1_gpu.cellule_mb import (  # noqa: E402
    EMISSIONS,
    GRAINES,
    N_EPISODES,
    budget_k_fen,
    centres_cellule,
    dchi,
)
from src.f1_gpu.memoire_instrument import (  # noqa: E402
    armer_plafond_as,
    vm_size_go,
)
from src.f1_gpu.mort_b_lecture import PIN_SEVERE  # noqa: E402
from src.f1_gpu.production_fidele import (  # noqa: E402
    EPS_PRODUCTION,
    evoluer_histoire_production,
)
from src.f1_gpu.rederive_borne import ConsommateurRederive  # noqa: E402
from src.sediment import SedimentParams, default_terrain  # noqa: E402

OUT_DIR = ROOT / "outputs" / "f1"
OUT_JSON_PATH = OUT_DIR / "mb_t2_eps0.json"
LECTURE_T2 = ROOT / "claude" / "lectures" / "mb_t2.lecture.json"
MARGE_AS_GO: float = 6.0

EPS_ZERO: float = 0.0


def mesurer_seed_eps0(seed: int, b0, cp,
                      params: SedimentParams = SedimentParams()) -> dict:
    """Le bras EPS = 0 d'un seed, contre le MÊME rederive et le MÊME
    observable que les deux autres bras (apples-to-apples reconduit).

    Le budget est FORCÉ à `b0.size` — sans quoi le cap 10 % mordrait à la
    place du seuil et le bras ne trancherait rien (cf. docstring §1)."""
    emissions = set(EMISSIONS)
    centres = centres_cellule(seed, N_EPISODES)
    rederive = ConsommateurRederive(params).histoire(
        b0.astype(float), centres, checkpoints=emissions)
    eps0 = evoluer_histoire_production(
        seed, N_EPISODES, b0, cp, centres, params, checkpoints=emissions,
        eps=EPS_ZERO, budget=b0.size)
    par_emission = {n: dchi(eps0[n], rederive[n]) for n in EMISSIONS}
    return {"par_emission": par_emission,
            "eps0": max(par_emission.values())}


def lecture_eps0(par_seed: dict, publie: dict) -> dict:
    """La lecture PRÉ-ÉCRITE, appliquée sans interprétation."""
    traversants = [s for s, m in par_seed.items() if m["eps0"] > PIN_SEVERE]
    passe = not traversants
    if passe:
        verdict = (
            "LE BRAS EPS=0 PASSE ⇒ la cause est la BANDE MORTE du seuil. Le "
            "remède est EPS — et l'échelle à laquelle §A23 l'avait validé "
            "(état synthétisé V4) est À RE-POSER : validé là, il ne l'est "
            "pas ici.")
    else:
        verdict = (
            f"LE BRAS EPS=0 ÉCHOUE ({len(traversants)}/{len(par_seed)} seeds "
            f"traversent : {sorted(traversants)}) ⇒ le plancher est "
            "STRUCTUREL (décimation + halo). Resserrer EPS ne peut pas le "
            "franchir : à seuil NUL et budget PLEIN, le canal ne retire déjà "
            "plus rien. L'architecture fovéale telle que construite ne peut "
            "pas satisfaire É2 — question de SPEC, pas de réglage.")
    # ── attribution PAR ÉMISSION, jamais par différence de maxima ──
    # Les maxima de série des trois bras tombent à des émissions DIFFÉRENTES
    # (seed 101 : EPS0 culmine à l'ém. 12, la production à l'ém. 24) :
    # les soustraire mélangerait des instants. On apparie donc émission par
    # émission.
    attribution, structure, canal, n_traverse, n_total = {}, [], [], 0, 0
    for s, m in par_seed.items():
        pub = publie[str(s)]["par_emission"]
        lignes = {}
        for n, e in m["par_emission"].items():
            t = pub[str(n)]["temoin"]
            p = pub[str(n)]["production"]
            structure.append(e - t)
            canal.append(p - e)
            n_total += 1
            n_traverse += int(e > PIN_SEVERE)
            lignes[n] = {"temoin_a": t, "eps0_a_plus_c": e,
                         "production_a_b_c": p,
                         "structure_c": e - t, "canal_b": p - e}
        attribution[s] = lignes
    canal_negatif = [x for x in canal if x < 0]
    return {
        "pin_severe": PIN_SEVERE,
        "eps_du_bras": EPS_ZERO,
        "eps_de_production": EPS_PRODUCTION,
        "budget_force": "H·W (plein)",
        "budget_de_la_cellule_refuse": budget_k_fen(64 * 64),
        "seeds_traversants": sorted(traversants),
        "eps0_traverse_par_emission": [n_traverse, n_total],
        "verdict_eps0": verdict,
        "attribution_par_emission": attribution,
        "structure_c_min_max": [min(structure), max(structure)],
        "canal_b_min_max": [min(canal), max(canal)],
        "canal_b_negatif_sur_n": [len(canal_negatif), n_total],
        "non_monotonie_du_canal": (
            f"L'échelle (a) ⊂ (a)+(c) ⊂ (a)+(b)+(c) n'est PAS monotone : sur "
            f"{len(canal_negatif)}/{n_total} émissions, retirer le seuil "
            "AGGRAVE Δχ (EPS0 > production). Le seuil ne fait donc pas que "
            "perdre de l'information — il filtre aussi des écarts de faible "
            "amplitude, et Δχ étant un rapport spectral, tout laisser passer "
            "peut injecter du désaccord. « Part du canal » est donc un écart "
            "signé, pas une contribution additive."),
        "ce_bras_attribue_il_ne_rehabilite_pas": (
            "Le verdict MORT-b n'est PAS rouvert. Ce bras ne remplace pas la "
            "production et ne la réhabilite pas : il dit OÙ est la cause. La "
            "production reste ce qu'elle est, et Option A reste morte."),
        "portee_canal_modelise": (
            "le plancher établi vaut pour LE CANAL TEL QUE MODÉLISÉ "
            "(`remonter_seuillee`, option 1 numpy). Le compacteur GPU L3 et "
            "sa capacité de buffer ne sont PAS exercés ici — une saturation "
            "de buffer du L3 réel resterait hors de cette mesure."),
        "portee_transparence": (
            "à EPS=0 et budget plein, la transparence du canal est à "
            "l'ARRONDI f32 près (~1e-8 relatif, mesuré), non bit-exacte — "
            "sept ordres sous le pin, donc le canal ne peut pas porter le "
            "plancher."),
    }


def _relire_mesure() -> dict:
    """Relit la mesure déjà acquise (les émissions ne sont pas re-calculées).
    Permet de corriger une LECTURE sans re-mesurer — la mesure et sa lecture
    sont deux objets distincts."""
    d = json.loads(OUT_JSON_PATH.read_text(encoding="utf-8"))["par_seed"]
    return {int(s): {"par_emission": {int(n): v for n, v in
                                      m["par_emission"].items()},
                     "eps0": m["eps0"]} for s, m in d.items()}


def main(relire: bool = False) -> None:
    publie = json.loads(LECTURE_T2.read_text(encoding="utf-8"))["par_seed"]
    if relire:
        print("  [relecture] mesure déjà acquise — aucune ré-exécution",
              flush=True)
        par_seed = _relire_mesure()
    else:
        cp = exiger_cupy()
        plafond = armer_plafond_as(vm_size_go() + MARGE_AS_GO)
        print(f"  [plafond AS] {plafond['plafond_go']:.1f} Go", flush=True)
        print(f"  [budget] FORCÉ à 4096 (plein) — celui de la cellule "
              f"({budget_k_fen(64 * 64)}) ferait mordre le CAP à la place du "
              f"seuil", flush=True)
        b0 = default_terrain(GridConfig()).astype("float32")
        par_seed = {}
        for seed in GRAINES:
            print(f"  seed {seed} : rederive + bras EPS=0 ({N_EPISODES} "
                  f"épisodes) ...", flush=True)
            par_seed[seed] = mesurer_seed_eps0(seed, b0, cp)
            cp.get_default_memory_pool().free_all_blocks()

    lecture = lecture_eps0(par_seed, publie)
    document = {
        "meta": {"mesure": ("M-b — bras EPS=0, discriminateur par canal "
                            "(§A32-décomposition, pocCascade2phys d7fd6ee)")},
        "par_seed": {str(s): m for s, m in par_seed.items()},
        "lecture_mecanique": lecture,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON_PATH.write_text(json.dumps(document, indent=2, ensure_ascii=False),
                             encoding="utf-8")

    print("=" * 78)
    print("F1 -- M-b : BRAS EPS = 0 (discriminateur par canal)")
    print("=" * 78)
    print(f"  {'seed/em':>8} {'témoin(a)':>10} {'EPS0(a+c)':>10} "
          f"{'prod(a+b+c)':>12} {'(c)':>9} {'(b)':>10}")
    for seed in GRAINES:
        for n in EMISSIONS:
            a = lecture["attribution_par_emission"][seed][n]
            print(f"  {seed}/{n:<3} {a['temoin_a']:10.5f} "
                  f"{a['eps0_a_plus_c']:10.5f} {a['production_a_b_c']:12.5f} "
                  f"{a['structure_c']:9.5f} {a['canal_b']:+10.5f}")
    nt, tot = lecture["eps0_traverse_par_emission"]
    print(f"\n  pin sévère = {PIN_SEVERE}  |  seeds traversants : "
          f"{lecture['seeds_traversants']}  |  émissions traversantes : "
          f"{nt}/{tot}")
    print(f"  [non-monotonie] {lecture['non_monotonie_du_canal']}")
    print(f"\n  VERDICT : {lecture['verdict_eps0']}")
    print(f"  [portée] {lecture['ce_bras_attribue_il_ne_rehabilite_pas']}")
    print(f"  [portée] {lecture['portee_canal_modelise']}")
    print("  Le driver REPORTE — la décision revient à Romain.")
    print(f"[REPORT] -> {OUT_JSON_PATH}")
    print("POINT D'ARRÊT OBLIGATOIRE : lecture remontée à Romain.")


if __name__ == "__main__":
    main(relire="--relire" in sys.argv)
