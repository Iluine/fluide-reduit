"""F1 tranche-1 — driver M-a : frame-time vs L_eff (le mot « moteur »).

Protocole gravé (§A15 M-a) : pyramide active à l'enveloppe §6 (c=8, f32,
n_fov=512, N_niv=10, γ₂≈3), F représentatif chaque frame sur les fenêtres
actives, fovéa MOBILE (E4c : balayage 1 cellule fine/frame). Scan :
N_niv ∈ {2, 4, 7, 10} x n_fov ∈ {256, 512}. Chrono B6 : cuda-Events,
sync explicite, warmup 30 EXCLU, série 300, médiane ET p99.

MORT-a (T2, GRAVÉ — consommé ici en LECTURE MÉCANIQUE uniquement, jamais
un verdict prononcé) : médiane frame-time > 16.7 ms à la cellule
d'enveloppe (512, 10). p99 : reportée, NON verdictale en v1. Gate §6
reconduit (pas une mort) : résidence mesurée > 1.35 Go => REMONTER
(re-épinglage du quadruplet = décision, jamais glissement).

Câblage B9 : exige les PASS #1/#3/#4 du registre AVANT toute mesure ;
la vérif #2 (sensibilité du chrono, 256->512 à N_niv=10) est calculée
DEPUIS le scan et enregistrée AVANT d'émettre la lecture — si elle
échoue, RuntimeError : tranche muette, aucune lecture.

Résidence (nommé) : la grandeur du gate §6 = `total_bytes()` du mempool
CuPy (l'allocateur du stack, validé concordant par la vérif #1) ;
nvidia-smi reporté en appui. Sortie JSON SANS timestamp.

Machine : iluin-tworings3, terminal natif UNIQUEMENT. Usage :
  .venv/bin/python scripts/run_f1_ma.py

POINT D'ARRÊT OBLIGATOIRE après le run : lecture remontée à Romain,
aucun enchaînement automatique."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.f1_gpu.backend import exiger_cupy  # noqa: E402
from src.f1_gpu.chrono import (  # noqa: E402
    SERIE_FRAMES,
    WARMUP_FRAMES,
    chronometrer_frames,
)
from src.f1_gpu.pyramide import (  # noqa: E402
    EPS_DETAIL,
    GeometriePyramide,
    PyramideFovea,
)
from src.f1_gpu.transferts import TransfertComptable  # noqa: E402
from src.f1_gpu.verifs import (  # noqa: E402
    enregistrer_verif,
    exiger_pass,
    verifier_sensibilite_chrono,
    vram_nvidia_smi_octets,
)

OUT_DIR = ROOT / "outputs" / "f1"
OUT_JSON_PATH = OUT_DIR / "ma_scan.json"

# Cellules du scan (gravées §A15 M-a) + cellule d'enveloppe.
SCAN_N_FOV: tuple[int, ...] = (256, 512)
SCAN_N_NIV: tuple[int, ...] = (2, 4, 7, 10)
CELLULE_ENVELOPPE: tuple[int, int] = (512, 10)

# Chiffres GRAVÉS (T2, gate §6) -- consommés en lecture mécanique SEULEMENT.
B_FRAME_MS: float = 16.7
GATE_RESIDENCE_OCTETS: int = int(1.35 * 1024 ** 3)

VERIFS_EXIGEES: tuple[str, ...] = (
    "vram_concordance", "gel_bit_exact", "bande_passante_a_vide")


def liberer_vram(cp) -> None:
    """Rend la VRAM au driver entre cellules (résidence par cellule
    propre) — appelé APRÈS le retour de `mesurer_cellule` (les buffers
    d'état sont locaux à la cellule, morts au retour)."""
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def mesurer_cellule(cp, n_fov: int, n_niv: int) -> dict:
    """Une cellule du scan : pyramide + fovéa mobile (E4c), chrono B6.
    La résidence est photographiée AVANT libération (fin de cellule)."""
    transferts = TransfertComptable(cp)
    geo = GeometriePyramide(n_fov=n_fov, n_niv=n_niv)
    pyramide = PyramideFovea(cp, geo, transferts)
    transferts.frame_suivante()  # clôt la descente initiale (hors régime, B4)

    def frame() -> None:
        pyramide.frame(delta_x=1)
        transferts.frame_suivante()

    chrono = chronometrer_frames(cp, frame)

    mempool = cp.get_default_memory_pool()
    cellule = {
        "n_fov": n_fov, "n_niv": n_niv,
        "cellules_actives_gpu": geo.cellules_actives_gpu,
        "frames": {"warmup_exclu": WARMUP_FRAMES, "serie": SERIE_FRAMES},
        "frame_time": chrono["stats"],
        "transferts_regime": transferts.stats_regime(
            exclure_premiers=1 + WARMUP_FRAMES),
        "residence": {
            "mempool_used_octets": int(mempool.used_bytes()),
            "mempool_total_octets": int(mempool.total_bytes()),
            "etat_prealloue_octets": pyramide.octets_etat(),
            "nvidia_smi_octets": vram_nvidia_smi_octets(),
        },
    }
    return cellule


def main() -> None:
    cp = exiger_cupy()
    exiger_pass(VERIFS_EXIGEES)

    cellules: list[dict] = []
    for n_niv in SCAN_N_NIV:
        for n_fov in SCAN_N_FOV:
            print(f"cellule n_fov={n_fov} n_niv={n_niv} ...", flush=True)
            cellules.append(mesurer_cellule(cp, n_fov, n_niv))
            liberer_vram(cp)

    par_cle = {(c["n_fov"], c["n_niv"]): c for c in cellules}

    # Vérif #2 -- AVANT toute lecture (B9), à N_niv de l'enveloppe.
    n_niv_env = CELLULE_ENVELOPPE[1]
    passe_2, details_2 = verifier_sensibilite_chrono(
        par_cle[(256, n_niv_env)]["frame_time"]["mediane_ms"],
        par_cle[(512, n_niv_env)]["frame_time"]["mediane_ms"])
    enregistrer_verif("sensibilite_chrono", passe_2, details_2)
    if not passe_2:
        raise RuntimeError(
            "TRANCHE MUETTE (vérif #2) : le chrono ne bouge pas de "
            f"256->512 ({details_2}) -- REMONTER, aucune lecture.")

    enveloppe = par_cle[CELLULE_ENVELOPPE]
    mediane_env = enveloppe["frame_time"]["mediane_ms"]
    residence_env = enveloppe["residence"]["mempool_total_octets"]
    lecture = {
        "seuil_grave_ms": B_FRAME_MS,
        "cellule_enveloppe": {"n_fov": CELLULE_ENVELOPPE[0],
                              "n_niv": CELLULE_ENVELOPPE[1]},
        "mediane_enveloppe_ms": mediane_env,
        "p99_enveloppe_ms": enveloppe["frame_time"]["p99_ms"],
        "mediane_depasse_seuil": bool(mediane_env > B_FRAME_MS),
        "gate_residence": {
            "seuil_octets": GATE_RESIDENCE_OCTETS,
            "residence_mempool_total_octets": residence_env,
            "declenche": bool(residence_env > GATE_RESIDENCE_OCTETS),
        },
    }

    document = {
        "meta": {
            "mesure": "M-a frame-time vs L_eff (§A15)",
            "choix": {"E4a": "(h,hu,hv,s)x2 stencil complet",
                      "E4b": "buffers preallocues + mempool (B2)",
                      "E4c": "balayage 1 cellule fine/frame",
                      "E4d": "remontee detail seuille (B4)",
                      "eps_detail": EPS_DETAIL},
            "verifs_exigees": list(VERIFS_EXIGEES) + ["sensibilite_chrono"],
        },
        "cellules": cellules,
        "lecture_mecanique": lecture,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON_PATH.write_text(
        json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 78)
    print("F1 TRANCHE-1 -- M-a : frame-time vs L_eff (lecture mécanique)")
    print("=" * 78)
    for c in cellules:
        ft = c["frame_time"]
        print(f"  n_fov={c['n_fov']:>3}  N_niv={c['n_niv']:>2}  "
              f"médiane={ft['mediane_ms']:.3f} ms  p99={ft['p99_ms']:.3f} ms")
    print(f"  ENVELOPPE (512, 10) : médiane={mediane_env:.3f} ms  "
          f"p99={lecture['p99_enveloppe_ms']:.3f} ms  "
          f"seuil gravé={B_FRAME_MS} ms  "
          f"médiane>seuil={lecture['mediane_depasse_seuil']}")
    print(f"  gate §6 : résidence={residence_env / 1024 ** 3:.3f} Go  "
          f"seuil=1.35 Go  déclenché={lecture['gate_residence']['declenche']}")
    print(f"[REPORT] -> {OUT_JSON_PATH}")
    print("POINT D'ARRÊT OBLIGATOIRE : lecture remontée à Romain, aucun "
          "verdict prononcé ici, aucun enchaînement automatique.")


if __name__ == "__main__":
    main()
