"""F1 — MICRO-SONDE s1 : ancre du bloc 1-SYSTÈME (§A17 G1, pocCascade2phys
b1f9cf9 ; brouillon de séance claude/seance-design-f-machinerie-2026-07-19).

CE QUE s1 MESURE, ET POURQUOI ELLE EXISTE : le modèle de coût suppose
qu'un slot c=4 vaut 0.843 ms, soit la MOITIÉ d'un slot c=8 (1.685 ms
mesuré en M-a′). Cette moitié n'a JAMAIS été mesurée — elle porte
l'étiquette « linéarité structurelle en systèmes ». La sonde
d'attribution (§A16-lecture-sonde-attribution) l'a rangée première parmi
les suspects du +6.11 ms : un bloc 1-système occupe 262k threads là où un
2-systèmes en occupe 524k, et rien ne garantit qu'un GPU à demi rempli
coûte la moitié. s1 remplace l'étiquette par DEUX ancres mesurées.

LES DEUX CELLULES (protocole M-a′-style, kernel INTOUCHÉ, 1 niveau) :
  - (a) B=1, S=1 — UN slot c=4 isolé : l'ancre unitaire, le cas le plus
    défavorable à l'occupancy ;
  - (b) B=7, S=1 — les 7 slots c=4 de V2 en UN lancement : l'ancre telle
    que s2 l'utilisera (c'est cette valeur-là, pas l'unitaire, que la
    prédiction re-calibrée de s2 consomme).

ATTENDU PRÉ-ÉCRIT (gravé §A17 G1) : les deux ancres sont lues contre
0.843 ms ± 10 %. Hors tolérance ⇒ la linéarité en systèmes est FAUSSE et
le modèle per-slot se re-calibre sur mesures. Les ancres sont reportées
PAR SLOT (médiane / nombre de slots), puisque c'est la grandeur du modèle.

Ce driver REPORTE. Il ne prononce aucun verdict, ne décide aucun design
et n'enchaîne rien — la séance §A17 consomme ses chiffres.

DÉPENDANCE : s1 doit tourner AVANT s2 (`run_f1_s2_batche.py` exige ce
JSON et refuse de démarrer sans lui).

Kernel CUDA, chrono B6 et substrats : INTOUCHÉS (diff vide ; empreinte du
kernel e8fcaad47db0af010728d5c5d878e1629191bfef4e0382fa80aa3cefd9f27f04).
Vérifs #1/#2 reconduites, natif. Sortie JSON SANS timestamp.

Machine : iluin-tworings3, terminal natif UNIQUEMENT. Usage :
  .venv/bin/python scripts/run_f1_s1_ancre_1sys.py

POINT D'ARRÊT OBLIGATOIRE après le run : lecture remontée à Romain."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_f1_ma_ter import VERIFS_EXIGEES, liberer_vram  # noqa: E402
from src.f1_gpu.backend import exiger_cupy  # noqa: E402
from src.f1_gpu.chrono import (  # noqa: E402
    SERIE_FRAMES,
    WARMUP_FRAMES,
    chronometrer_frames,
)
from src.f1_gpu.pyramide import GRAINE_MONDE  # noqa: E402
from src.f1_gpu.substrat_fusionne import pas_f_fusionne  # noqa: E402
from src.f1_gpu.substrat_jetable import etat_initial_jetable  # noqa: E402
from src.f1_gpu.verifs import exiger_pass, vram_nvidia_smi_octets  # noqa: E402

OUT_DIR = ROOT / "outputs" / "f1"
OUT_JSON_PATH = OUT_DIR / "s1_ancre_1sys.json"

N_FOV: int = 512                 # fenêtre du quadruplet (§6 / §A16)
N_SLOTS_C4_V2: int = 7           # les 7 slots c=4 de V2 (fovéale supérieure)

# Étiquette « structurelle » que s1 remplace par des MESURES (§A16) :
# 1 slot ≡ 1.685 ms à c=8, « donc » 0.843 ms à c=4 — la moitié jamais mesurée.
ANCRE_MODELE_MS: float = 0.843
ANCRE_2SYS_MESUREE_MS: float = 1.685      # M-a′, celle-là EST mesurée
TOLERANCE_RELATIVE: float = 0.10          # ±10 %, gravé §A17 G1

# Les deux cellules : (nom, B = nombre de blocs 1-système).
CELLULES: tuple[tuple[str, int], ...] = (
    ("a_isole", 1),
    ("b_batche", N_SLOTS_C4_V2),
)


def etat_mono_systeme(cp, n_blocs: int, n_fov: int = N_FOV):
    """État (B, 1, 4, n, n) f32 sur device — le générateur jetable de
    M-a′, tranché à UN système. `ascontiguousarray` côté CPU : la tranche
    `[:, :1]` d'un (B, 2, 4, n, n) ne l'est pas, et le kernel exige du
    contigu (il indexe des blocs aplatis)."""
    import numpy as np
    q = np.ascontiguousarray(
        etat_initial_jetable(n_blocs, n_fov, GRAINE_MONDE)[:, :1])
    return cp.asarray(q)


def mesurer_cellule(cp, nom: str, n_blocs: int) -> dict:
    """Une cellule : B blocs 1-système, UN appel `pas_f_fusionne` par
    frame (2 lancements de kernel), chrono B6. Transferts hors-mesure —
    l'état monte une fois avant le chrono, comme M-a′."""
    q = etat_mono_systeme(cp, n_blocs)
    tampon_etage = cp.empty_like(q)          # B2 : préalloué, zéro realloc

    def frame() -> None:
        pas_f_fusionne(q, cp, sortie=q, tampon_etage=tampon_etage)

    chrono = chronometrer_frames(cp, frame)

    mempool = cp.get_default_memory_pool()
    mediane = chrono["stats"]["mediane_ms"]
    return {
        "nom": nom,
        "cellule": {"n_blocs": n_blocs, "n_systemes": 1, "n_fov": N_FOV,
                    "n_niveaux": 1, "lancements_kernel_par_frame": 2},
        "frames": {"warmup_exclu": WARMUP_FRAMES, "serie": SERIE_FRAMES},
        "frame_time": chrono["stats"],
        "ancre_par_slot_ms": mediane / n_blocs,
        "residence": {
            "mempool_used_octets": int(mempool.used_bytes()),
            "mempool_total_octets": int(mempool.total_bytes()),
            "nvidia_smi_octets": vram_nvidia_smi_octets(),
        },
    }


def lecture_ancre(ancre_ms: float) -> dict:
    """Lecture mécanique d'une ancre contre l'étiquette 0.843 ± 10 %."""
    ecart_relatif = (ancre_ms - ANCRE_MODELE_MS) / ANCRE_MODELE_MS
    return {
        "ancre_par_slot_ms": ancre_ms,
        "etiquette_structurelle_ms": ANCRE_MODELE_MS,
        "ecart_relatif": ecart_relatif,
        "dans_tolerance": bool(abs(ecart_relatif) <= TOLERANCE_RELATIVE),
        "rapport_au_2sys_mesure": ancre_ms / ANCRE_2SYS_MESUREE_MS,
    }


def lecture_mecanique(cellules: dict[str, dict]) -> dict:
    """Consommation MÉCANIQUE des attendus §A17 G1 — le driver REPORTE."""
    lectures = {nom: lecture_ancre(c["ancre_par_slot_ms"])
                for nom, c in cellules.items()}
    toutes_dans_tolerance = all(lec["dans_tolerance"]
                                for lec in lectures.values())
    return {
        "tolerance_relative": TOLERANCE_RELATIVE,
        "par_cellule": lectures,
        "ancre_retenue_pour_s2": {
            "valeur_ms": lectures["b_batche"]["ancre_par_slot_ms"],
            "origine": ("cellule (b) B=7 batchée — c'est l'orchestration "
                        "que s2 emploie ; l'ancre isolée (a) est reportée "
                        "pour la lecture d'occupancy, pas consommée par s2"),
        },
        "branche_preecrite_si_dans_tolerance": (
            "la linéarité en systèmes TIENT à ±10 % — l'étiquette "
            "structurelle est confirmée par la mesure"),
        "branche_preecrite_si_hors_tolerance": (
            "la linéarité en systèmes est FAUSSE : le modèle per-slot se "
            "re-calibre sur les ancres MESURÉES (§A17 G1)"),
        "toutes_dans_tolerance": bool(toutes_dans_tolerance),
        "rappel_portee": (
            "micro-sonde d'ouverture de séance : elle ANCRE le design, "
            "elle ne le décide pas. Aucun verdict, aucun levier tranché."),
    }


def main() -> None:
    cp = exiger_cupy()
    exiger_pass(VERIFS_EXIGEES)

    cellules: list[dict] = []
    for nom, n_blocs in CELLULES:
        print(f"cellule {nom} (B={n_blocs}, S=1) ...", flush=True)
        cellules.append(mesurer_cellule(cp, nom, n_blocs))
        liberer_vram(cp)

    par_nom = {c["nom"]: c for c in cellules}
    lecture = lecture_mecanique(par_nom)

    document = {
        "meta": {
            "mesure": "s1 — ancre du bloc 1-système "
                      "(§A17 G1, pocCascade2phys b1f9cf9)",
            "protocole": {
                "style": "M-a′ (1 niveau, transferts hors-mesure, natif)",
                "F": "kernel FUSIONNÉ M-a′ (substrat_fusionne, INTOUCHÉ)",
                "chrono": "B6 (cuda-Events, warmup exclu, médiane ET p99)",
                "note_cfl": ("`pas_f_fusionne` est employé TEL QUEL : sa "
                             "réduction CFL synchronise une fois par "
                             "frame, exactement comme dans l'ancre M-a′ à "
                             "laquelle ces mesures se comparent"),
            },
            "etiquette_remplacee": {
                "ancre_2sys_mesuree_ms": ANCRE_2SYS_MESUREE_MS,
                "ancre_1sys_supposee_ms": ANCRE_MODELE_MS,
                "note": ("la moitié n'a jamais été mesurée — c'est "
                         "précisément ce que s1 corrige"),
            },
            "verifs_exigees": list(VERIFS_EXIGEES),
        },
        "cellules": cellules,
        "lecture_mecanique": lecture,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON_PATH.write_text(
        json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 78)
    print("F1 -- s1 : ancre du bloc 1-système (lecture mécanique)")
    print("=" * 78)
    for c in cellules:
        lec = lecture["par_cellule"][c["nom"]]
        print(f"  {c['nom']:<10} B={c['cellule']['n_blocs']:<2} "
              f"médiane={c['frame_time']['mediane_ms']:7.3f} ms  "
              f"p99={c['frame_time']['p99_ms']:7.3f} ms  "
              f"ancre/slot={lec['ancre_par_slot_ms']:.3f} ms  "
              f"écart={lec['ecart_relatif']:+.1%}  "
              f"dans ±10 %={lec['dans_tolerance']}")
    retenue = lecture["ancre_retenue_pour_s2"]
    print(f"  étiquette remplacée : {ANCRE_MODELE_MS} ms/slot "
          f"(supposée moitié de {ANCRE_2SYS_MESUREE_MS} mesurés)")
    print(f"  ancre retenue pour s2 : {retenue['valeur_ms']:.3f} ms/slot "
          f"(cellule b, batchée)")
    print("  RAPPEL : micro-sonde d'ouverture — elle ANCRE le design, "
          "elle ne le décide pas.")
    print(f"[REPORT] -> {OUT_JSON_PATH}")
    print("POINT D'ARRÊT OBLIGATOIRE : lecture remontée à Romain. "
          "s2 (run_f1_s2_batche.py) consomme ce JSON.")


if __name__ == "__main__":
    main()
