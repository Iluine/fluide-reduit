"""F1 — driver M-a′ : micro-mesure de la BORNE D'IMPLÉMENTATION (fusionné).

Pré-enregistrement gravé (§A15-lecture-M-a, pocCascade2phys 6e9ee46,
2026-07-19) — cité : « RawKernel CUDA fusionné du MÊME motif E4a
(2 systèmes (h,hu,hv,s), stencil wetdry O2 complet, 2 étages SSP-RK2,
réduction CFL payée non consommée, dt figé, bathymétrie plate) —
1 niveau, 3 fenêtres n_fov=512, chrono B6 (warmup 30 exclu, série 300,
médiane ET p99), même base que M-a (calcul, transferts hors-mesure).
Machine-instrument, natif. »

CRITÈRE PRÉ-ÉCRIT (gravé — consommé ici en LECTURE MÉCANIQUE, ce driver
REPORTE et ne prononce RIEN) : médiane par-niveau > 16.7/9 ≈ 1.856 ms
=> MORT-a CONFIRMÉE au niveau ARCHITECTURE (à cette enveloppe, ce
quadruplet §6) — l'extrapolation ×9 niveaux est étiquetée TRANSPOSITION
(la linéarité en niveaux est MESURÉE sur le naïf, pas sur le fusionné).
Si < 1.856 ms : MORT-a RE-SCOPÉE à l'implémentation naïve ; le règlement
COMPLET de M-a exige alors une décision neuve (re-mesure enveloppe pleine
avec F fusionné + fovéa mobile + E4d) qui ne s'enchaîne PAS
automatiquement.

CAP ANTI-TAPIS-ROULANT GRAVÉ : dernière escalade d'implémentation — au
delà du RawKernel fusionné (CUDA Graphs, exotique), toute escalade est
INTERDITE. Ce driver est le DERNIER instrument de la question M-a.

Transferts hors-mesure (protocole) : l'état monte UNE fois avant le
chrono ; la frame mesurée est du calcul pur (le trafic du schéma diff
appartient à M-a/M-c, déjà mesurés/à mesurer — pas à cette borne).

Câblage B9 : exige les 4 PASS du registre (l'instrument complet est armé
depuis M-a). Sortie JSON SANS timestamp.

Machine : iluin-tworings3, terminal natif UNIQUEMENT. Usage :
  .venv/bin/python scripts/run_f1_ma_prime.py

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
from src.f1_gpu.pyramide import GAMMA2, GRAINE_MONDE  # noqa: E402
from src.f1_gpu.substrat_fusionne import (  # noqa: E402
    TOL_EQUIVALENCE_ATOL,
    TOL_EQUIVALENCE_RTOL,
    pas_f_fusionne,
)
from src.f1_gpu.substrat_jetable import etat_initial_jetable  # noqa: E402
from src.f1_gpu.verifs import exiger_pass, vram_nvidia_smi_octets  # noqa: E402

OUT_DIR = ROOT / "outputs" / "f1"
OUT_JSON_PATH = OUT_DIR / "ma_prime.json"

N_FOV: int = 512          # gravé : 1 niveau, 3 fenêtres n_fov=512
N_FENETRES: int = GAMMA2  # 3 (γ₂, §6)

# Seuil PRÉ-ÉCRIT (§A15-lecture-M-a) : budget par-niveau de l'enveloppe
# (N_niv=10 => 9 niveaux GPU). Consommé en lecture mécanique SEULEMENT.
SEUIL_PREREG_MS: float = 16.7 / 9  # ≈ 1.856 ms/niveau

VERIFS_EXIGEES: tuple[str, ...] = (
    "vram_concordance", "sensibilite_chrono", "gel_bit_exact",
    "bande_passante_a_vide")


def main() -> None:
    cp = exiger_cupy()
    exiger_pass(VERIFS_EXIGEES)

    # État : même générateur jetable que M-a ; montée UNE fois, hors chrono.
    q = cp.asarray(etat_initial_jetable(N_FENETRES, N_FOV, GRAINE_MONDE))
    tampon_etage = cp.empty_like(q)  # B2 : préalloué, zéro realloc en frame

    def frame() -> None:
        pas_f_fusionne(q, cp, sortie=q, tampon_etage=tampon_etage)

    # La compilation NVRTC du kernel a lieu à la 1re frame du warmup
    # (exclue de la série, B6).
    chrono = chronometrer_frames(cp, frame)

    mempool = cp.get_default_memory_pool()
    mediane = chrono["stats"]["mediane_ms"]
    lecture = {
        "seuil_prereg_ms_par_niveau": SEUIL_PREREG_MS,
        "mediane_ms_par_niveau": mediane,
        "p99_ms_par_niveau": chrono["stats"]["p99_ms"],
        "mediane_depasse_seuil": bool(mediane > SEUIL_PREREG_MS),
        "branche_preecrite_si_depasse": (
            "MORT-a CONFIRMÉE au niveau ARCHITECTURE (extrapolation x9 "
            "niveaux = TRANSPOSITION, linéarité mesurée sur le naïf)"),
        "branche_preecrite_si_sous_seuil": (
            "MORT-a RE-SCOPÉE à l'implémentation naïve ; règlement complet "
            "de M-a = décision neuve (re-mesure enveloppe pleine fusionné + "
            "fovéa mobile + E4d), PAS d'enchaînement automatique"),
        "cap_grave": "dernière escalade d'implémentation — aucune autre "
                     "(pas de CUDA Graphs, pas d'exotique)",
    }
    document = {
        "meta": {
            "mesure": "M-a' borne d'implémentation, RawKernel fusionné "
                      "(§A15-lecture-M-a, 6e9ee46)",
            "cellule": {"n_fov": N_FOV, "n_fenetres": N_FENETRES,
                        "n_niveaux_mesures": 1},
            "frames": {"warmup_exclu": WARMUP_FRAMES, "serie": SERIE_FRAMES},
            "transferts": "hors-mesure (calcul pur — protocole gravé)",
            "equivalence_motif": {
                "test": "tests/test_f1_substrat_fusionne.py",
                "rtol": TOL_EQUIVALENCE_RTOL,
                "atol": TOL_EQUIVALENCE_ATOL},
            "verifs_exigees": list(VERIFS_EXIGEES),
        },
        "frame_time": chrono["stats"],
        "residence_appui": {
            "mempool_used_octets": int(mempool.used_bytes()),
            "mempool_total_octets": int(mempool.total_bytes()),
            "nvidia_smi_octets": vram_nvidia_smi_octets(),
        },
        "lecture_mecanique": lecture,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON_PATH.write_text(
        json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 78)
    print("F1 -- M-a' : borne d'implémentation fusionnée (lecture mécanique)")
    print("=" * 78)
    print(f"  cellule : 1 niveau, {N_FENETRES} fenêtres {N_FOV}² "
          f"(calcul pur, transferts hors-mesure)")
    print(f"  médiane={mediane:.3f} ms  "
          f"p99={lecture['p99_ms_par_niveau']:.3f} ms  "
          f"seuil pré-écrit={SEUIL_PREREG_MS:.3f} ms/niveau  "
          f"médiane>seuil={lecture['mediane_depasse_seuil']}")
    print("  RAPPEL gravé : >seuil => mort ARCHITECTURALE confirmée ; "
          "<seuil => re-scope naïf, décision neuve requise (jamais "
          "d'enchaînement). CAP : dernière escalade.")
    print(f"[REPORT] -> {OUT_JSON_PATH}")
    print("POINT D'ARRÊT OBLIGATOIRE : lecture remontée à Romain, aucun "
          "verdict prononcé ici, aucun enchaînement automatique.")


if __name__ == "__main__":
    main()
