"""F1 tranche-1 — driver M-d : coût de LOAD du ledger (§4 de la spec).

Protocole gravé (§A15 M-d) : ledger de référence 1 h simulée ≈ 3600
commits ≈ 11.5 Mo, généré par la tranche (cf. `run_f1_md_generer.py`,
E6). MORT-d (T5, GRAVÉ — LECTURE MÉCANIQUE uniquement) : temps de load
> 30 s SANS snapshot-racine.

Load = E7/B8 gravé : parse+validation INTÉGRALE + regenerate(dernier
commit) + replay du segment courant (évoluteur = `run_episode`, cœur
intouché ; segment vide à Δt=1) + reprise du vivant (H2D de l'état,
compté). Consignes E7 endossées, appliquées ici :
  1. MORT-d re-scopé DIT EN FACE : 30 s borne le coût du LEDGER (parse,
     intégrité, regenerate), pas celui de la physique ;
  2. le diagnostic snapshot-au-milieu est RETOURNÉ en test de la claim
     E7 : ATTENDU PRÉ-ENREGISTRÉ = écart de load avec/sans snapshot ≈ 0.
     Écart MATÉRIEL (B8, seuil d'instrument figé : |Δ| > max(10 % du load
     sans snapshot, 0.5 s)) => AUTRE, à remonter — le diagnostic falsifie
     E7 au lieu d'être vidé par lui ;
  3. la compaction snapshot-racine achète du disque/de la rétention, PAS
     du temps de load (note de spec consignée, §4 non réécrit).

Répétitions (nommé, à endosser) : 5 loads par ledger — le PREMIER (cache
disque froid du point de vue du processus) reporté À PART, la lecture
mécanique porte sur la MÉDIANE des 5 ; les deux chiffres sont en évidence.

Câblage B9 : PASS #1/#3/#4 exigés. Sortie JSON SANS timestamp.
Machine : iluin-tworings3, terminal natif UNIQUEMENT. Usage :
  .venv/bin/python scripts/run_f1_md.py

POINT D'ARRÊT OBLIGATOIRE après le run : lecture remontée à Romain."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_arcA_m2_registre import centres_episodes  # noqa: E402
from scripts.run_arcA_revalidate import B0, PARAMS  # noqa: E402
from scripts.run_f1_md_generer import (  # noqa: E402
    GRAINE,
    LEDGER_AVEC_PATH,
    LEDGER_SANS_PATH,
    N_COMMITS,
)
from src.f1_gpu.backend import exiger_cupy  # noqa: E402
from src.f1_gpu.chrono import stats_ms  # noqa: E402
from src.f1_gpu.ledger import charger_ledger  # noqa: E402
from src.f1_gpu.verifs import exiger_pass  # noqa: E402
from src.sediment import run_episode  # noqa: E402

OUT_DIR = ROOT / "outputs" / "f1"
OUT_JSON_PATH = OUT_DIR / "md_load.json"

# Chiffre GRAVÉ (T5) -- lecture mécanique seulement.
T_LOAD_S: float = 30.0
REPETITIONS: int = 5
# Seuil d'instrument B8 (écart matériel avec/sans snapshot) -- figé.
ECART_RELATIF: float = 0.10
ECART_PLANCHER_S: float = 0.5

VERIFS_EXIGEES: tuple[str, ...] = (
    "vram_concordance", "gel_bit_exact", "bande_passante_a_vide")


def _evoluteur_production(etat, entree: dict):
    """Évoluteur B8 du segment courant : l'événement seedé (a) rejoue son
    épisode via le cœur intouché (mêmes centres du seed, motif VERBATIM)."""
    centres = centres_episodes(GRAINE, N_COMMITS)
    return run_episode(etat, B0, centres[entree["t_sim"] - 1], PARAMS)


def mesurer_loads(cp, repertoire: Path) -> dict:
    """5 loads chronométrés (perf_counter — le load est une opération
    hôte+H2D à l'échelle de la seconde ; la reprise du vivant est
    synchronisée device avant l'arrêt du chrono)."""
    durees_s: list[float] = []
    dernier: dict = {}

    def televerser(etat) -> None:
        cp.asarray(etat)
        cp.cuda.runtime.deviceSynchronize()

    for _ in range(REPETITIONS):
        t0 = time.perf_counter()
        dernier = charger_ledger(repertoire, evoluer=_evoluteur_production,
                                 televerser=televerser)
        durees_s.append(time.perf_counter() - t0)
    dernier.pop("etat")
    stats = stats_ms([d * 1e3 for d in durees_s])
    return {"repertoire": str(repertoire.relative_to(ROOT)),
            "durees_s": durees_s,
            "premier_froid_s": durees_s[0],
            "mediane_s": stats["mediane_ms"] / 1e3,
            "p99_s": stats["p99_ms"] / 1e3,
            "contenu": dernier}


def main() -> None:
    cp = exiger_cupy()
    exiger_pass(VERIFS_EXIGEES)
    for repertoire in (LEDGER_SANS_PATH, LEDGER_AVEC_PATH):
        if not (repertoire / "index.jsonl").exists():
            raise RuntimeError(
                f"run_f1_md : ledger absent ({repertoire}) — générer d'abord "
                "(scripts/run_f1_md_generer.py, run de nuit E6).")

    print("load SANS snapshot ...", flush=True)
    sans = mesurer_loads(cp, LEDGER_SANS_PATH)
    print("load AVEC snapshot ...", flush=True)
    avec = mesurer_loads(cp, LEDGER_AVEC_PATH)

    ecart_s = abs(avec["mediane_s"] - sans["mediane_s"])
    seuil_materiel_s = max(ECART_RELATIF * sans["mediane_s"],
                           ECART_PLANCHER_S)
    lecture = {
        "seuil_grave_load_s": T_LOAD_S,
        "load_sans_snapshot_mediane_s": sans["mediane_s"],
        "load_sans_snapshot_premier_froid_s": sans["premier_froid_s"],
        "load_depasse_seuil": bool(sans["mediane_s"] > T_LOAD_S),
        "diagnostic_e7_snapshot": {
            "attendu_preenregistre": "écart de load avec/sans snapshot ≈ 0",
            "load_avec_snapshot_mediane_s": avec["mediane_s"],
            "ecart_s": ecart_s,
            "seuil_materiel_s": seuil_materiel_s,
            "ecart_materiel": bool(ecart_s > seuil_materiel_s),
            "autre_a_remonter": bool(ecart_s > seuil_materiel_s),
        },
    }
    document = {
        "meta": {
            "mesure": "M-d coût de load du ledger (§A15)",
            "load": "E7/B8 : parse intégral + regenerate(dernier commit) + "
                    "segment courant + reprise du vivant (H2D)",
            "repetitions": REPETITIONS,
            "verifs_exigees": list(VERIFS_EXIGEES),
        },
        "sans_snapshot": sans,
        "avec_snapshot": avec,
        "lecture_mecanique": lecture,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON_PATH.write_text(
        json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 78)
    print("F1 TRANCHE-1 -- M-d : coût de load du ledger (lecture mécanique)")
    print("=" * 78)
    print(f"  SANS snapshot : médiane={sans['mediane_s']:.3f} s  "
          f"(1er/froid={sans['premier_froid_s']:.3f} s)  "
          f"seuil gravé={T_LOAD_S} s  >seuil={lecture['load_depasse_seuil']}")
    print(f"  AVEC snapshot : médiane={avec['mediane_s']:.3f} s  "
          f"écart={ecart_s:.3f} s  (seuil matériel {seuil_materiel_s:.3f} s)"
          f"  AUTRE à remonter="
          f"{lecture['diagnostic_e7_snapshot']['autre_a_remonter']}")
    print(f"  contenu : {sans['contenu']['n_entrees']} entrées, "
          f"segment courant {sans['contenu']['n_segment']} épisode(s), "
          f"{sans['contenu']['octets_data'] / 1e6:.1f} Mo de payloads")
    print(f"[REPORT] -> {OUT_JSON_PATH}")
    print("POINT D'ARRÊT OBLIGATOIRE : lecture remontée à Romain, aucun "
          "verdict prononcé ici, aucun enchaînement automatique.")


if __name__ == "__main__":
    main()
