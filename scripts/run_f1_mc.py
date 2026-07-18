"""F1 tranche-1 — driver M-c : débit PCIe réel du schéma diff (topologie §5).

Protocole gravé (§A15 M-c) : « Compter les octets/frame réels du trafic
CPU<->GPU : descente (prédiction Harten des fenêtres actives) + remontée
(coefficients de détail). Enveloppe attendue : échelle n_fov² — brut
~c·b·n_fov² ≈ 8.4 Mo/direction/frame plein-fovéa ; le schéma diff doit
faire MIEUX que le plein. »

MORT-c (T4, GRAVÉ — consommé en LECTURE MÉCANIQUE uniquement) :
  1. temps de transfert mesuré > 25 % x 16.7 ms = 4.2 ms à la cellule
     d'enveloppe (512, 10) ;
  2. FUITE D'ÉCHELLE : le trafic croît avec la taille du monde au lieu de
     n_fov² — opérationnalisé (gravé au lancement de tranche) :
     N_niv 10 -> 11 à fovéa fixe (un niveau de plus = monde doublé), le
     trafic/frame doit rester ~constant.

Seuil d'INSTRUMENT nommé par ce build (à endosser, pas re-réglable après
lecture) : SEUIL_FUITE_RATIO = 1.5 — un ratio octets(11)/octets(10) >= 1.5
signe un trafic qui suit le monde (un doublement ferait ~2.0) ; la valeur
STRUCTURELLE attendue du schéma B3/B4 est ~(11−1)/(10−1) ≈ 1.11 (un niveau
de fenêtres en plus). Le ratio brut est reporté, le seuil ne fait que le
mécaniser.

Câblage B9 : PASS #1/#3/#4 exigés (le #4 est le dénominateur : la bande
passante à vide vit déjà au registre). Sortie JSON SANS timestamp.

Machine : iluin-tworings3, terminal natif UNIQUEMENT. Usage :
  .venv/bin/python scripts/run_f1_mc.py

POINT D'ARRÊT OBLIGATOIRE après le run : lecture remontée à Romain."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_f1_ma import liberer_vram  # noqa: E402
from src.f1_gpu.backend import exiger_cupy  # noqa: E402
from src.f1_gpu.chrono import SERIE_FRAMES, WARMUP_FRAMES  # noqa: E402
from src.f1_gpu.pyramide import (  # noqa: E402
    EPS_DETAIL,
    GeometriePyramide,
    PyramideFovea,
)
from src.f1_gpu.transferts import TransfertComptable  # noqa: E402
from src.f1_gpu.verifs import exiger_pass  # noqa: E402

OUT_DIR = ROOT / "outputs" / "f1"
OUT_JSON_PATH = OUT_DIR / "mc_pcie.json"

CELLULE_ENVELOPPE: tuple[int, int] = (512, 10)
CELLULE_ECHELLE: tuple[int, int] = (512, 11)   # MORT-c(2) : monde doublé

# Chiffres GRAVÉS (T4) -- lecture mécanique seulement.
SEUIL_TRANSFERT_MS: float = 4.2
# Référence plein-fovéa (MESURÉ-par-calcul, note VRAM) : c·b·n_fov².
PLEIN_OCTETS_PAR_DIRECTION: int = 8 * 4 * 512 * 512
# Seuil d'instrument nommé (cf. docstring) -- PAS un chiffre gravé §A15.
SEUIL_FUITE_RATIO: float = 1.5

VERIFS_EXIGEES: tuple[str, ...] = (
    "vram_concordance", "gel_bit_exact", "bande_passante_a_vide")


def mesurer_cellule(cp, n_fov: int, n_niv: int) -> dict:
    """Une cellule M-c : même pipeline que M-a (le trafic EST celui de la
    frame de mesure), comptage par `TransfertComptable`, régime établi =
    série après warmup (warmup + descente initiale EXCLUS, B4)."""
    transferts = TransfertComptable(cp)
    geo = GeometriePyramide(n_fov=n_fov, n_niv=n_niv)
    pyramide = PyramideFovea(cp, geo, transferts)
    transferts.frame_suivante()
    nnz_dernier: dict = {}
    for _ in range(WARMUP_FRAMES + SERIE_FRAMES):
        diag = pyramide.frame(delta_x=1)
        transferts.frame_suivante()
        nnz_dernier = diag["nnz_par_niveau"]
    cellule = {
        "n_fov": n_fov, "n_niv": n_niv,
        "transferts_regime": transferts.stats_regime(
            exclure_premiers=1 + WARMUP_FRAMES),
        "octets_dense_equivalent_par_frame":
            pyramide.octets_dense_equivalent_par_frame(),
        "nnz_par_niveau_derniere_frame": nnz_dernier,
    }
    return cellule


def main() -> None:
    cp = exiger_cupy()
    exiger_pass(VERIFS_EXIGEES)

    print(f"cellule enveloppe {CELLULE_ENVELOPPE} ...", flush=True)
    enveloppe = mesurer_cellule(cp, *CELLULE_ENVELOPPE)
    liberer_vram(cp)
    print(f"cellule échelle {CELLULE_ECHELLE} ...", flush=True)
    echelle = mesurer_cellule(cp, *CELLULE_ECHELLE)
    liberer_vram(cp)

    stats_env = enveloppe["transferts_regime"]
    stats_ech = echelle["transferts_regime"]
    transfert_ms = stats_env["transfert_total_ms"]["mediane_ms"]
    octets_env = stats_env["transfert_total_octets"]["mediane"]
    octets_ech = stats_ech["transfert_total_octets"]["mediane"]
    ratio_echelle = octets_ech / max(octets_env, 1.0)
    plein_total = 2 * PLEIN_OCTETS_PAR_DIRECTION

    lecture = {
        "seuil_grave_transfert_ms": SEUIL_TRANSFERT_MS,
        "transfert_total_mediane_ms": transfert_ms,
        "transfert_total_p99_ms": stats_env["transfert_total_ms"]["p99_ms"],
        "transfert_depasse_seuil": bool(transfert_ms > SEUIL_TRANSFERT_MS),
        "plein_reference": {
            "octets_par_direction": PLEIN_OCTETS_PAR_DIRECTION,
            "octets_total": plein_total,
            "diff_mediane_octets": octets_env,
            "ratio_diff_sur_plein": octets_env / plein_total,
            "diff_bat_le_plein": bool(octets_env < plein_total),
        },
        "fuite_echelle": {
            "octets_mediane_n_niv_10": octets_env,
            "octets_mediane_n_niv_11": octets_ech,
            "ratio": ratio_echelle,
            "attendu_structurel": (CELLULE_ECHELLE[1] - 1)
                                  / (CELLULE_ENVELOPPE[1] - 1),
            "seuil_instrument_ratio": SEUIL_FUITE_RATIO,
            "fuite_detectee": bool(ratio_echelle >= SEUIL_FUITE_RATIO),
        },
    }
    document = {
        "meta": {
            "mesure": "M-c débit PCIe réel du schéma diff (§A15)",
            "choix": {"B4": "descente colonnes entrantes + remontée détail "
                            "seuillé (fenêtres touchées, chaque frame)",
                      "eps_detail": EPS_DETAIL},
            "verifs_exigees": list(VERIFS_EXIGEES),
            "frames": {"warmup_exclu": WARMUP_FRAMES, "serie": SERIE_FRAMES},
        },
        "cellule_enveloppe": enveloppe,
        "cellule_echelle": echelle,
        "lecture_mecanique": lecture,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON_PATH.write_text(
        json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 78)
    print("F1 TRANCHE-1 -- M-c : débit PCIe du schéma diff (lecture mécanique)")
    print("=" * 78)
    print(f"  transfert total (512,10) : médiane={transfert_ms:.3f} ms  "
          f"p99={lecture['transfert_total_p99_ms']:.3f} ms  "
          f"seuil gravé={SEUIL_TRANSFERT_MS} ms  "
          f">seuil={lecture['transfert_depasse_seuil']}")
    print(f"  octets/frame médiane : diff={octets_env / 1e6:.3f} Mo  "
          f"plein={plein_total / 1e6:.3f} Mo  "
          f"diff<plein={lecture['plein_reference']['diff_bat_le_plein']}")
    print(f"  échelle N_niv 10->11 : ratio={ratio_echelle:.3f}  "
          f"(structurel ~{lecture['fuite_echelle']['attendu_structurel']:.2f},"
          f" seuil instrument {SEUIL_FUITE_RATIO})  "
          f"fuite={lecture['fuite_echelle']['fuite_detectee']}")
    print(f"[REPORT] -> {OUT_JSON_PATH}")
    print("POINT D'ARRÊT OBLIGATOIRE : lecture remontée à Romain, aucun "
          "verdict prononcé ici, aucun enchaînement automatique.")


if __name__ == "__main__":
    main()
