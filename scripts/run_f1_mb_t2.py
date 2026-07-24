"""F1 — M-b TRANCHE-2 : driver du gate (iii), la FIDÉLITÉ (§A31-build).

T1 était le TEMPS ; T2 est la FIDÉLITÉ. Ici le rederive CPU f64 INTOUCHÉ
(`run_history`) est le juge, et la question est : ce que le système SAIT
diverge-t-il de la vérité au-delà du pin sévère 0,0733 ?

DEUX BRAS, LA MÊME CELLULE, LE MÊME OBSERVABLE (§A31-build) :
  - **PRODUCTION** — fovéation 2-niveaux + Exner cadencé + remontée L3
    seuillée à EPS 1e-2, connaissance CPU option 1. Porte les TROIS sources
    d'écart : (a) f32, (b) le seuil, (c) la structure fovéale ;
  - **TÉMOIN** — fidèle F GPU f32, 64² plein domaine, un seul niveau, murs
    réfléchissants. Ne porte QUE (a).
Mêmes centres (ceux du rederive), mêmes émissions, même Δχ, même vérité-sol.
Apples-to-apples : leur DIFFÉRENCE est exactement (b)+(c), et c'est ce qui
rend la lecture à trois branches décidable — Option B « tout-déterministe »
ne guérit que (a).

RÉSERVE SUR LE TÉMOIN (§A31-build) : au build, le témoin a montré 6e-5
d'écart d'ÉTAT contre le rederive f64, trois ordres sous le pin. La branche
« les deux traversent » en devient très improbable. Mais 6e-5 n'est PAS la
mesure Δχ pré-enregistrée — l'instrument diffère, le témoin tourne comme
prévu et sa branche se lit sur SA mesure.

GATE (iii) RE-SCOPÉ AVANT LE RUN (§A31-build) : « M-b sans mort À 64², sur
fovéation 2-niveaux ». Le contrat à l'échelle V4 devient une TRANSPOSITION
NOMMÉE, non mesurée (son falsificateur exigerait un rederive V4, interdit
par « intouché »). Un PASS ne s'énonce donc PAS « le contrat tient à
l'échelle V4 » — et cette limite VOYAGE DANS la lecture, comme les portées.

Kernels FIGÉS et C1–C5 intouchés ; k reste 4 ; `attribution_b` gaté ; pas de
miroir CPU ; rederive INTOUCHÉ ; invariant liant par IDENTITÉ.

Machine : iluin-tworings3, terminal natif UNIQUEMENT. Usage :
  .venv/bin/python scripts/run_f1_mb_t2.py

POINT D'ARRÊT OBLIGATOIRE AVANT LE RUN (§A31) : ce driver est CONSTRUIT mais
NON LANCÉ tant que l'assemblage n'est pas endossé."""
from __future__ import annotations

import hashlib
import json
import sys
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import GridConfig  # noqa: E402
from src.f1_gpu.backend import exiger_cupy  # noqa: E402
from src.f1_gpu.cellule_mb import (  # noqa: E402
    CAP_COMMIT,
    DELTA_T,
    EMISSIONS,
    GRAINES,
    N_EPISODES,
    centres_cellule,
    dchi,
)
from src.f1_gpu.exner_gpu import CADENCE_EXNER  # noqa: E402
from src.f1_gpu.memoire_instrument import (  # noqa: E402
    SondeMemoire,
    armer_plafond_as,
    lire_jalons,
    pic_par_phase,
    vm_size_go,
)
from src.f1_gpu.mort_b_lecture import (  # noqa: E402
    LABEL_OPTION_A_TIENT,
    lecture_mort_b,
    portees_de_la_mesure,
)
from src.f1_gpu.production_fidele import (  # noqa: E402
    EPS_PRODUCTION,
    N_FOVEA,
    evoluer_histoire_production,
)
from src.f1_gpu.rederive_borne import ConsommateurRederive  # noqa: E402
from src.f1_gpu.temoin_fidele import evoluer_histoire_temoin  # noqa: E402
from src.f1_gpu.verifs import exiger_pass  # noqa: E402
from src.sediment import SedimentParams, default_terrain  # noqa: E402

OUT_DIR = ROOT / "outputs" / "f1"
OUT_JSON_PATH = OUT_DIR / "mb_t2.json"
MEMOIRE_PATH = OUT_DIR / "mb_t2_memoire.jsonl"

# Marge au-dessus du socle d'espace d'adressage réservé par CUDA. Le pic
# mesuré du rederive borné est ~2,2 Go ; 6 Go laissent large tout en
# hard-stoppant très en-deçà des 23–25 Go qui ont tué les deux premiers runs.
MARGE_AS_GO: float = 6.0

# T2 mesure la FIDÉLITÉ contre le rederive CPU f64 : la vérification
# load-bearing est le GEL BIT-EXACT du chemin rederive sur CETTE machine
# (vérif #3, §A15). Si le rederive avait bougé ici, la vérité-sol serait
# fausse et Δχ ne voudrait rien dire. Les instruments de TEMPS (vram,
# chrono, PCIe) sont ceux de T1 — les exiger ici sur-gaterait.
VERIFS_EXIGEES: tuple[str, ...] = ("gel_bit_exact",)


def maxima_de_serie(production: dict, temoin: dict, rederive: dict) -> dict:
    """MORT-b est un MAX DE SÉRIE : Δχ est évalué à CHAQUE émission, pour les
    DEUX bras, contre le MÊME rederive, avec le MÊME observable — puis le
    maximum sur la série est retenu par bras.

    `production` : {émission: connaissance_cpu} ; `temoin` : {émission: s} ;
    `rederive` : {émission: s_f64}. Une seule émission traversante suffit à
    faire traverser le seed."""
    par_emission = {
        n: {"production": dchi(production[n], rederive[n]),
            "temoin": dchi(temoin[n], rederive[n])}
        for n in EMISSIONS}
    return {
        "par_emission": par_emission,
        "production": max(v["production"] for v in par_emission.values()),
        "temoin": max(v["temoin"] for v in par_emission.values()),
    }


def mesurer_seed(seed: int, b0, cp,
                 params: SedimentParams = SedimentParams(),
                 sonde: SondeMemoire | None = None) -> dict:
    """Les trois bras d'UN seed, sur la MÊME histoire. Les centres sont tirés
    UNE fois et donnés aux trois — le rederive, lui, les retire identiquement
    (verrou testé dans `cellule_mb`).

    Le rederive est consommé BORNÉ (§A31-run) : il reste INTOUCHÉ, mais son
    `t_end` est réduit au plus petit barreau suffisant, ce qui rend le MÊME
    résultat (bit-exact, testé) pour 2,2 Go au lieu de 25.

    `sonde` : chaque bras est une phase persistée — si le processus meurt, le
    disque dit lequel le tenait et à quel pic."""
    emissions = set(EMISSIONS)
    centres = centres_cellule(seed, N_EPISODES)

    def _phase(bras):
        return (sonde.phase(bras, f"seed={seed}") if sonde is not None
                else nullcontext())

    with _phase("rederive"):
        rederive = ConsommateurRederive(params).histoire(
            b0.astype(float), centres, checkpoints=emissions)
    with _phase("production"):
        production = evoluer_histoire_production(seed, N_EPISODES, b0, cp,
                                                 centres, params,
                                                 checkpoints=emissions,
                                                 eps=EPS_PRODUCTION)
    with _phase("temoin"):
        temoin = evoluer_histoire_temoin(seed, N_EPISODES, b0, cp, centres,
                                         params, checkpoints=emissions)
    return maxima_de_serie(production, temoin, rederive)


def lecture_t2(par_seed: dict[int, dict]) -> dict:
    """La lecture MÉCANIQUE de T2 : la décision pré-écrite à trois branches,
    ses portées, et ce qu'un PASS a le droit de dire. Le driver REPORTE — la
    décision revient à Romain."""
    mort_b = lecture_mort_b({s: {"production": m["production"],
                                 "temoin": m["temoin"]}
                             for s, m in par_seed.items()})
    pass_prononce = mort_b["verdict"] == LABEL_OPTION_A_TIENT
    return {
        "mort_b": mort_b,
        "cellule": {"graines": list(GRAINES), "delta_t": DELTA_T,
                    "emissions": list(EMISSIONS), "n_episodes": N_EPISODES,
                    "cap_commit": CAP_COMMIT},
        "ce_que_ce_pass_dit": (
            "gate (iii) re-scopé (§A31-build) : un PASS s'énonce « M-b sans "
            "mort À 64², sur fovéation 2-niveaux » et RIEN de plus. Il ne "
            "peut PAS s'énoncer « le contrat tient à l'échelle V4 » : la "
            "transposition à V4 est NOMMÉE, non mesurée, et son "
            "falsificateur exigerait un rederive à l'échelle V4 qu'« "
            "intouché » interdit."
            + ("" if pass_prononce else
               " (Ici aucun PASS n'est prononcé — la limite est reportée "
               "quand même, elle ne dépend pas du résultat.)")),
        "portees": portees_de_la_mesure(),
        "verrous": {
            "kernels": ("F fidèle _SOURCE_FIDELE 6dd207ca (C1) INTOUCHÉ ; "
                        "figés e18015f5/9533a130 ; Exner 3533fd0b — aucun "
                        "kernel neuf en tranche-2"),
            "rederive": ("`run_history` CPU f64 INTOUCHÉ — la vérité-sol ; "
                         "les centres des deux bras sont vérifiés contre lui"),
            "invariant_liant": (
                "`t1.pas_f_fidele is t2.pas_f_fidele` — identité, pas diff. "
                "Le halo-parent est rafraîchi entre étages RK2 AU DRIVER "
                "(couche t2), jamais dans `pas_f_fidele`"),
            "canal": (f"remontée L3 seuillée EPS {EPS_PRODUCTION}, "
                      f"connaissance CPU option 1 (la `reference`), cap "
                      f"{CAP_COMMIT:.0%} ; pas de miroir CPU"),
            "cadence_exner": f"LUE de save_every = {CADENCE_EXNER}",
        },
    }


def empreinte_config() -> str:
    """Ce qui rendrait deux seeds INCOMPARABLES s'il bougeait entre deux
    runs : le seuil du canal, la géométrie fovéale, la cadence Exner, la
    cellule, et les empreintes des kernels. Une reprise n'est licite que sous
    empreinte identique."""
    from src.f1_gpu.exner_gpu import _SOURCE_EXNER
    from src.f1_gpu.substrat_fidele import _SOURCE_FIDELE
    graine = json.dumps({
        "eps": EPS_PRODUCTION, "n_fovea": N_FOVEA,
        "cadence_exner": CADENCE_EXNER, "n_episodes": N_EPISODES,
        "emissions": list(EMISSIONS), "graines": list(GRAINES),
        "cap": CAP_COMMIT,
        "kernels": [hashlib.sha256(s.encode()).hexdigest()
                    for s in (_SOURCE_FIDELE, _SOURCE_EXNER)],
    }, sort_keys=True)
    return hashlib.sha256(graine.encode()).hexdigest()[:16]


def seeds_a_mesurer(document_existant: dict | None,
                    empreinte: str) -> tuple[list[int], dict]:
    """Reprise après coupure : quels seeds restent à mesurer, et lesquels
    sont repris. Un report d'une AUTRE config est ignoré en bloc — mélanger
    des seeds mesurés sous deux configs produirait une lecture fausse."""
    if not document_existant or document_existant.get("empreinte") != empreinte:
        return list(GRAINES), {}
    repris = {int(s): m
              for s, m in (document_existant.get("par_seed") or {}).items()
              if int(s) in GRAINES}
    return [s for s in GRAINES if s not in repris], repris


def document_partiel(par_seed: dict[int, dict], empreinte: str) -> dict:
    """Report d'un run INCOMPLET. Il ne porte AUCUNE lecture : un verdict lu
    sur 2 seeds sur 3 serait faux (MORT-b est par-seed — le seed manquant
    peut être celui qui traverse). Le statut le dit en toutes lettres."""
    return {
        "statut": (f"PARTIEL — {len(par_seed)}/{len(GRAINES)} seeds mesurés, "
                   "AUCUNE lecture prononcée (MORT-b est par-seed : un seed "
                   "manquant peut être celui qui traverse). Relancer le "
                   "driver reprend les seeds manquants."),
        "empreinte": empreinte,
        "par_seed": {str(s): m for s, m in par_seed.items()},
        "lecture_mecanique": None,
    }


def document_t2(par_seed: dict[int, dict], lecture: dict,
                memoire: dict | None = None) -> dict:
    """Le report. EXTRAIT de `main` à dessein : il s'exécute APRÈS la mesure,
    donc une faute ici détruirait un run déjà payé — il doit être exerçable à
    vide (cf. tests)."""
    return {
        "meta": {
            "mesure": ("M-b tranche-2 / gate (iii) — fidélité à 64², "
                       "fovéation 2-niveaux (§A31, pocCascade2phys a8a7639 ; "
                       "gate re-scopé §A31-build)"),
            "config": {"n": 64, "n_fovea": N_FOVEA,
                       "eps_production": EPS_PRODUCTION,
                       "cadence_exner": CADENCE_EXNER,
                       "bord_fovea": "halo-parent, rafraîchi entre étages RK2",
                       "bord_grossier": "réfléchissant (vrai bord, = rederive)"},
            "verifs_exigees": list(VERIFS_EXIGEES),
        },
        "statut": f"COMPLET — {len(par_seed)}/{len(GRAINES)} seeds mesurés",
        "empreinte": empreinte_config(),
        "par_seed": {str(s): m for s, m in par_seed.items()},
        # pic mémoire PAR BRAS ET PAR PHASE : l'instrument voyage avec la
        # mesure (§A31-run — un échec doit laisser un chiffre).
        "memoire_pic_mo": {f"{bras}|{phase}": v
                           for (bras, phase), v in (memoire or {}).items()},
        "lecture_mecanique": lecture,
    }


def imprimer_lecture(lecture: dict) -> None:
    """L'affichage de la lecture. EXTRAIT de `main` pour la même raison que
    `document_t2` : il vient après la mesure et doit être exerçable à vide."""
    mb = lecture["mort_b"]
    print("=" * 78)
    print("F1 -- M-b TRANCHE-2 / gate (iii) (lecture mécanique)")
    print("=" * 78)
    for seed in GRAINES:
        b = mb["par_seed"][seed]
        print(f"  seed {seed} : production Δχ_max = "
              f"{b['production_dchi_max']:.5f} | témoin Δχ_max = "
              f"{b['temoin_dchi_max']:.5f}")
        print(f"    branche : {b['label']}")
    print(f"  pin sévère = {mb['pin_severe']}  |  seeds traversants : "
          f"{mb['seeds_traversants']}")
    print(f"  VERDICT (trois branches pré-écrites) : {mb['verdict']}")
    if mb["anomalies"]:
        print(f"  ANOMALIES à remonter : {mb['anomalies']}")
    print(f"  [portée] {lecture['ce_que_ce_pass_dit']}")
    print("  Le driver REPORTE — la décision revient à Romain.")


def main() -> None:
    """Le run de T2. GATÉ : n'est lancé qu'après endossement de l'assemblage.

    Ordre voulu : le gate d'instrument AVANT toute mesure (une tranche muette
    ne doit pas coûter les minutes de run), le report écrit APRÈS CHAQUE SEED
    (une coupure ne coûte plus qu'un seed, et relancer reprend là où ça s'est
    arrêté), l'affichage en dernier (la mesure est sauvée même s'il fâche).

    Un run repris ne prononce de lecture QUE complet : MORT-b est par-seed,
    le seed manquant peut être celui qui traverse."""
    cp = exiger_cupy()
    exiger_pass(VERIFS_EXIGEES)
    # Plafond armé APRÈS l'init CUDA (qui réserve ~6,4 Go d'espace
    # d'adressage) : + MARGE_AS_GO au-dessus du socle réel. Le pic mesuré du
    # rederive borné est de 2,2 Go — le plafond est une PREUVE, pas une
    # béquille : sous lui, un débordement lève au lieu de tuer en silence.
    plafond = armer_plafond_as(vm_size_go() + MARGE_AS_GO)
    print(f"  [plafond AS] {plafond['plafond_go']:.1f} Go "
          f"(socle CUDA réservé : {plafond['reserve_avant_go']:.1f} Go)",
          flush=True)
    b0 = default_terrain(GridConfig()).astype("float32")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    empreinte = empreinte_config()
    existant = (json.loads(OUT_JSON_PATH.read_text(encoding="utf-8"))
                if OUT_JSON_PATH.exists() else None)
    a_faire, par_seed = seeds_a_mesurer(existant, empreinte)
    if par_seed:
        print(f"  [reprise] seeds déjà mesurés sous la même empreinte "
              f"{empreinte} : {sorted(par_seed)}", flush=True)

    sonde = SondeMemoire(MEMOIRE_PATH, cp=cp)
    for seed in a_faire:
        print(f"  seed {seed} : rederive borné + production + témoin "
              f"({N_EPISODES} épisodes) ...", flush=True)
        par_seed[seed] = mesurer_seed(seed, b0, cp, sonde=sonde)
        # report AVANT le seed suivant : une coupure ne perd plus le mesuré
        OUT_JSON_PATH.write_text(
            json.dumps(document_partiel(par_seed, empreinte), indent=2,
                       ensure_ascii=False), encoding="utf-8")
        cp.get_default_memory_pool().free_all_blocks()

    memoire = pic_par_phase(lire_jalons(MEMOIRE_PATH))
    lecture = lecture_t2(par_seed)
    OUT_JSON_PATH.write_text(
        json.dumps(document_t2(par_seed, lecture, memoire), indent=2,
                   ensure_ascii=False), encoding="utf-8")
    imprimer_lecture(lecture)
    for (bras, phase), v in sorted(memoire.items()):
        print(f"  [mémoire] {bras:11s} {phase:12s} pic RSS = "
              f"{v['rss_pic_mo']:7.0f} Mo")
    print(f"[REPORT] -> {OUT_JSON_PATH}")
    print(f"[MÉMOIRE] -> {MEMOIRE_PATH}")
    print("POINT D'ARRÊT OBLIGATOIRE : lecture remontée à Romain.")


if __name__ == "__main__":
    main()
