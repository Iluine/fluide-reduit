"""F1 — M-b : DÉCOMPOSITION SPATIALE de l'écart de T2 (§A32).

DIAGNOSTIC, PAS UNE MESURE NEUVE. Le verdict MORT-b n'est pas rouvert : ni
l'observable ni le critère ne bougent. La question est étroite — l'écart
0,70–0,87 vient-il de la FOVÉA (25 % de l'aire, pleine résolution) ou de la
PÉRIPHÉRIE (75 %, décimée 2×) ?

────────────────────────────────────────────────────────────────────────
LES CHAMPS NE SONT PAS SUR DISQUE — CE QUE JE FAIS À LA PLACE
────────────────────────────────────────────────────────────────────────
Le driver de T2 n'a persisté que les SCALAIRES Δχ, pas les champs. « Sur les
champs déjà capturés » n'est donc pas littéralement exécutable. Les trois
bras étant DÉTERMINISTES (centres reçus, rederive f64, GPU f32 reproductible
— testé), je les RE-MATÉRIALISE et je PROUVE l'identité : chaque Δχ plein
recalculé doit retomber, AU BIT PRÈS, sur celui publié dans
`claude/lectures/mb_t2.lecture.json`. Si un seul diffère, la décomposition
est refusée fail-loud plutôt que reportée sur des champs qui ne seraient pas
ceux du verdict.

Ce n'est donc pas un run neuf au sens du gravé : aucune donnée nouvelle
n'entre, et la preuve d'identité le garantit au lieu de le promettre.

────────────────────────────────────────────────────────────────────────
CE QUI EST MESURÉ, ET LA RÉSERVE
────────────────────────────────────────────────────────────────────────
Voir `src/f1_gpu/decomposition_spatiale.py` : on masque la DIFFÉRENCE, pas le
domaine (restreindre `delta_chi` à une sous-fenêtre est illégitime — piège
§A14, cf. le module). Réserve portée avec le résultat : le masquage fuit au
bord, les deux Δχ partiels ne s'additionnent pas ; d'où la partition
d'énergie, exactement additive, en second regard.

Usage :  .venv/bin/python scripts/run_f1_mb_t2_decomposition.py"""
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
    centres_cellule,
    dchi,
)
from src.f1_gpu.decomposition_spatiale import (  # noqa: E402
    decomposer,
    fidelite_structurelle,
    masque_fenetre,
    partition_energie,
)
from src.f1_gpu.mort_b_lecture import PIN_SEVERE  # noqa: E402
from src.f1_gpu.memoire_instrument import (  # noqa: E402
    armer_plafond_as,
    vm_size_go,
)
from src.f1_gpu.production_fidele import (  # noqa: E402
    EPS_PRODUCTION,
    N_FOVEA,
    evoluer_histoire_production,
    position_fovea,
)
from src.f1_gpu.rederive_borne import ConsommateurRederive  # noqa: E402
from src.f1_gpu.temoin_fidele import evoluer_histoire_temoin  # noqa: E402
from src.sediment import SedimentParams, default_terrain  # noqa: E402

OUT_DIR = ROOT / "outputs" / "f1"
OUT_JSON_PATH = OUT_DIR / "mb_t2_decomposition.json"
LECTURE_T2 = ROOT / "claude" / "lectures" / "mb_t2.lecture.json"
MARGE_AS_GO: float = 6.0


def champs_du_seed(seed: int, b0, cp,
                   params: SedimentParams = SedimentParams()) -> dict:
    """Re-matérialise les trois champs aux émissions (déterministe)."""
    emissions = set(EMISSIONS)
    centres = centres_cellule(seed, N_EPISODES)
    return {
        "rederive": ConsommateurRederive(params).histoire(
            b0.astype(float), centres, checkpoints=emissions),
        "production": evoluer_histoire_production(
            seed, N_EPISODES, b0, cp, centres, params,
            checkpoints=emissions, eps=EPS_PRODUCTION),
        "temoin": evoluer_histoire_temoin(
            seed, N_EPISODES, b0, cp, centres, params,
            checkpoints=emissions),
    }


def exiger_identite(seed: int, champs: dict, publie: dict) -> dict:
    """PREUVE D'IDENTITÉ : les Δχ recalculés doivent retomber AU BIT PRÈS sur
    ceux publiés. Sinon, refus fail-loud — décomposer des champs qui ne sont
    pas ceux du verdict ne dirait rien."""
    ecarts = {}
    for n in EMISSIONS:
        for bras in ("production", "temoin"):
            recalcule = dchi(champs[bras][n], champs["rederive"][n])
            attendu = publie["par_emission"][str(n)][bras]
            if recalcule != attendu:
                ecarts[f"{bras}@{n}"] = {"recalcule": recalcule,
                                         "publie": attendu}
    if ecarts:
        raise RuntimeError(
            f"IDENTITÉ ROMPUE (seed {seed}) : les champs re-matérialisés ne "
            f"reproduisent pas les Δχ publiés — {ecarts}. La décomposition "
            "est REFUSÉE : elle porterait sur d'autres champs que le verdict.")
    return {"seed": seed, "emissions_verifiees": len(EMISSIONS),
            "identite": "BIT-EXACTE"}


def decomposer_seed(seed: int, champs: dict) -> dict:
    """Décomposition fovéa / périphérie, par émission."""
    n_grille = champs["rederive"][EMISSIONS[0]].shape[0]
    par_emission = {}
    for n in EMISSIONS:
        oy, ox = position_fovea(n - 1, n=n_grille)
        masque = masque_fenetre(n_grille, oy, ox, N_FOVEA)
        d = decomposer(champs["production"][n], champs["rederive"][n], masque)
        e = partition_energie(champs["production"][n],
                              champs["rederive"][n], masque)
        f = fidelite_structurelle(champs["production"][n],
                                  champs["rederive"][n])
        par_emission[n] = {"fovea_oy_ox": [oy, ox], "spectral": d,
                           "energie": e, "fidelite": f}
    return par_emission


def lecture_decomposition(par_seed: dict) -> dict:
    """Les lectures PRÉ-ÉCRITES (§A32), appliquées sans interprétation.

    LE CONTREFACTUEL. L'hybride « fovéa seule » EST le champ d'un système
    dont la périphérie serait EXACTEMENT juste : son Δχ dit donc ce que
    vaudrait le critère si r_fovea réparait parfaitement la périphérie.
    Symétriquement pour l'hybride « périphérie seule » et un EPS parfait sur
    la fovéa. Chaque hybride est donc lu CONTRE LE PIN, et pas seulement
    contre l'autre. (Approximation nommée : réparer une région changerait
    aussi l'autre par le couplage du halo ; l'hybride le néglige.)"""
    ratios, fov, per, cor = [], [], [], []
    for m in par_seed.values():
        for v in m.values():
            e, s = v["energie"], v["spectral"]
            f, p = (e["energie_par_cellule_fovea"],
                    e["energie_par_cellule_peripherie"])
            ratios.append(p / f if f > 0 else float("inf"))
            fov.append(s["dchi_fovea"])
            per.append(s["dchi_peripherie"])
            cor.append(v["fidelite"]["correlation"])
    ratio_median = sorted(ratios)[len(ratios) // 2]
    bandes = [v["spectral"]["bande_dominante"]
              for m in par_seed.values() for v in m.values()]
    n = len(fov)
    fov_traverse = sum(x > PIN_SEVERE for x in fov)
    per_traverse = sum(x > PIN_SEVERE for x in per)
    if fov_traverse == 0 and per_traverse > 0:
        verdict = ("CAUSE (c) LA DÉCIMATION — fovéa propre, périphérie "
                   "saturée. Le remède est le CRITÈRE appliqué à la "
                   "périphérie, donc r_fovea.")
    elif fov_traverse > 0 and per_traverse == 0:
        verdict = ("CAUSE (b) EPS et/ou le canal — la fovéa seule traverse. "
                   "EPS se resserre.")
    elif fov_traverse > 0 and per_traverse > 0:
        verdict = (
            f"LES DEUX RÉGIONS TRAVERSENT SÉPARÉMENT ({fov_traverse}/{n} "
            f"fovéa, {per_traverse}/{n} périphérie) — aucune des deux "
            "lectures pré-écrites ne s'applique seule. Par le contrefactuel : "
            "réparer parfaitement la périphérie (r_fovea) laisserait Δχ à "
            f"{min(fov):.3f}–{max(fov):.3f}, et réparer parfaitement la "
            f"fovéa (EPS) le laisserait à {min(per):.3f}–{max(per):.3f} — "
            f"des deux côtés AU-DESSUS du pin {PIN_SEVERE}. AUCUN remède "
            "unilatéral ne rachète le gate (iii).")
    else:
        verdict = "AUCUNE région ne traverse séparément — à remonter."
    return {
        "ratio_energie_par_cellule_peripherie_sur_fovea_median": ratio_median,
        "bandes_dominantes": sorted(set(b for b in bandes if b)),
        "fovea_traverse_sur_n": [fov_traverse, n],
        "peripherie_traverse_sur_n": [per_traverse, n],
        "dchi_fovea_min_max": [min(fov), max(fov)],
        "dchi_peripherie_min_max": [min(per), max(per)],
        "correlation_min_max": [min(cor), max(cor)],
        "verdict_decomposition": verdict,
        "lecture_pre_ecrite": (
            "fovéa propre + périphérie saturée ⇒ cause (c) LA DÉCIMATION — "
            "le remède n'est ni Option B ni EPS, c'est le CRITÈRE appliqué à "
            "la périphérie, donc r_fovea. Fovéa également dégradée ⇒ cause "
            "(b) EPS et/ou le canal, EPS se resserre."),
        "reserve_masquage": (
            "les Δχ partiels ne s'additionnent PAS au Δχ plein : masquer la "
            "différence introduit une discontinuité au bord de la fovéa, qui "
            "fuit en spectre, et Δχ n'est de toute façon pas additif "
            "(rapports de RMS). Ils se lisent l'un contre l'autre. La "
            "partition d'énergie, elle, est exactement additive — c'est elle "
            "qui départage."),
        "reserve_fovea_mobile": (
            "la fovéa est MOBILE : le champ à l'émission n a été accumulé sur "
            "n épisodes où elle occupait n positions distinctes. Le masque "
            "porte sa position à l'épisode n — le cœur fovéal de l'émission, "
            "pas l'union des positions visitées."),
        "portee": (
            "DIAGNOSTIC sur données déjà acquises, identité bit-exacte "
            "prouvée contre les Δχ publiés. Ni l'observable ni le critère ne "
            "bougent ; le verdict MORT-b n'est pas rouvert."),
    }


def main() -> None:
    cp = exiger_cupy()
    plafond = armer_plafond_as(vm_size_go() + MARGE_AS_GO)
    print(f"  [plafond AS] {plafond['plafond_go']:.1f} Go", flush=True)
    publie = json.loads(LECTURE_T2.read_text(encoding="utf-8"))["par_seed"]
    b0 = default_terrain(GridConfig()).astype("float32")

    par_seed, identites = {}, []
    for seed in GRAINES:
        print(f"  seed {seed} : re-matérialisation des trois champs ...",
              flush=True)
        champs = champs_du_seed(seed, b0, cp)
        identites.append(exiger_identite(seed, champs, publie[str(seed)]))
        print(f"    identité BIT-EXACTE sur {len(EMISSIONS)} émissions",
              flush=True)
        par_seed[seed] = decomposer_seed(seed, champs)
        cp.get_default_memory_pool().free_all_blocks()

    lecture = lecture_decomposition(par_seed)
    document = {
        "meta": {"mesure": ("M-b — décomposition spatiale de l'écart T2 "
                            "(§A32, pocCascade2phys 0ef658e)"),
                 "identite_champs": identites,
                 "n_fovea": N_FOVEA, "aire_fovea_pct": 100 * N_FOVEA ** 2 / 64 ** 2},
        "par_seed": {str(s): {str(n): v for n, v in m.items()}
                     for s, m in par_seed.items()},
        "lecture_mecanique": lecture,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON_PATH.write_text(json.dumps(document, indent=2, ensure_ascii=False),
                             encoding="utf-8")

    print("=" * 78)
    print("F1 -- M-b : DÉCOMPOSITION SPATIALE (fovéa / périphérie)")
    print("=" * 78)
    for seed in GRAINES:
        print(f"  seed {seed} :")
        for n in EMISSIONS:
            v = par_seed[seed][n]
            s, e = v["spectral"], v["energie"]
            print(f"    em {n:>2} | Δχ plein={s['dchi_plein']:.5f}  "
                  f"fovéa={s['dchi_fovea']:.5f}  "
                  f"périph={s['dchi_peripherie']:.5f}  "
                  f"| E/cell péri/fov="
                  f"{e['energie_par_cellule_peripherie'] / e['energie_par_cellule_fovea']:7.2f}"
                  f"  bande={s['bande_dominante']}")
    print(f"\n  ratio médian E/cellule périphérie/fovéa = "
          f"{lecture['ratio_energie_par_cellule_peripherie_sur_fovea_median']:.2f}")
    print(f"  bandes dominantes : {lecture['bandes_dominantes']}")
    cmin, cmax = lecture["correlation_min_max"]
    print(f"  corrélation production/rederive : {cmin:.4f} .. {cmax:.4f}")
    print(f"  fovéa seule > pin : {lecture['fovea_traverse_sur_n']}  |  "
          f"périphérie seule > pin : {lecture['peripherie_traverse_sur_n']}")
    print(f"\n  VERDICT DÉCOMPOSITION : {lecture['verdict_decomposition']}")
    print(f"  [réserve] {lecture['reserve_masquage']}")
    print("  Le driver REPORTE — la décision revient à Romain.")
    print(f"[REPORT] -> {OUT_JSON_PATH}")
    print("POINT D'ARRÊT OBLIGATOIRE : lecture remontée à Romain.")


if __name__ == "__main__":
    main()
