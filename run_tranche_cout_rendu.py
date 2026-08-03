"""DRIVER DE MESURE — tranche : coût du gather-plancher et résidence VRAM.

Exécute le protocole de `claude/prereg-tranche-cout-rendu-2026-08-03.md`, écrit
et endossé AVANT ce fichier. Le driver n'a AUCUNE latitude : configuration,
tailles d'écran, découpage du chronomètre et branches sont fixés là-bas. Ce
fichier les applique, il ne les choisit pas.

    « Choisir `r_fovea` pour que le budget passe = fabriquer le verdict »
    — PREREGISTRATION.md:1032-1033

CE QUI EST MESURÉ : la résidence VRAM de la pyramide V4 (M-1), le coût du gather
du chemin-de-coût par taille d'écran (M-2), le coût du verrou de la garde 2
(M-3). CE QUI NE L'EST PAS : la fidélité (impossible par construction, garde 2),
`frame()`, le readout, l'ombrage, l'éclairage, la composition réelle.

ANTI-SURCLAME (§A48 garde 4) : ce driver produit « le gather-plancher coûte X »,
JAMAIS « le rendu coûte X ». La chaîne de caractères du verdict est écrite en
dur ci-dessous pour que personne n'ait à s'en souvenir.

Usage : `.venv/bin/python run_tranche_cout_rendu.py`
"""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np

from src.f1_gpu.backend import exiger_cupy, synchroniser
from src.f1_gpu.chemin_de_cout import (
    gather_chemin_de_cout,
    violations_consignees,
)
from src.f1_gpu.chrono import SERIE_FRAMES, WARMUP_FRAMES, chronometrer_frames
from src.f1_gpu.pyramide import GeometriePyramide, PyramideFovea, Slot
from src.f1_gpu.transferts import TransfertComptable

# ----- configuration V4, épinglée par le pré-enregistrement §2 --------------

N_FOV = 512
N_NIV = 10          # niveaux GPU 1..9, fin = 9 (PREREGISTRATION.md:5386)
N0 = 256
NIVEAU_FIN = N_NIV - 1

SLOTS_V4: tuple[Slot, ...] = (
    # fovéale : 2 niveaux fins à c=8, 7 supérieurs à c=4 (:4214)
    tuple(Slot(niveau=j, c=8, role="fovea") for j in (NIVEAU_FIN, NIVEAU_FIN - 1))
    + tuple(Slot(niveau=j, c=4, role="fovea") for j in range(1, NIVEAU_FIN - 1))
    # énergie : 2 slots au niveau le plus fin — [INFÉRENCE DE SESSION, préreg §2]
    + (Slot(niveau=NIVEAU_FIN, c=8, role="energie"),
       Slot(niveau=NIVEAU_FIN, c=8, role="energie"))
)

COTES_ECRAN: tuple[int, ...] = (512, 1024, 1440, 1920)
COTE_TEMOIN = 512  # témoin de dégénérescence, préreg §2

ARTEFACT = Path("claude/lectures/tranche-cout-rendu-2026-08-03.json")


def _vram_mo(cp) -> tuple[float, float]:
    libre, total = cp.cuda.runtime.memGetInfo()
    return libre / 2**20, total / 2**20


def _mesure_1_residence(cp, pyramide, vram_avant: float) -> dict:
    """M-1 — deux nombres à ne pas confondre (préreg §3) : la part déterministe
    (`octets_etat`) et le tout (`memGetInfo`, mempool compris). L'écart est
    lui-même un résultat : aucun compte sur papier ne le prédisait."""
    synchroniser(cp)
    libre_apres, _ = _vram_mo(cp)
    deterministe_mo = pyramide.octets_etat() / 2**20
    total_mo = vram_avant - libre_apres
    return {
        "octets_etat_mo": deterministe_mo,
        "delta_memgetinfo_mo": total_mo,
        "ecart_mempool_mo": total_mo - deterministe_mo,
        "cellules_actives_gpu": int(pyramide.geo.cellules_actives_gpu),
        "n_slots_total": int(pyramide.geo.n_slots_total),
    }


def _mesure_2_gather(cp, pyramide) -> dict:
    """M-2 — coût du gather SEUL, par taille d'écran. Protocole gravé §A15 :
    warmup >= 30 EXCLU, série >= 300, médiane ET p99."""
    resultats: dict[str, dict] = {}
    for cote in COTES_ECRAN:
        def une_frame(c=cote):
            gather_chemin_de_cout(pyramide, c)

        mesure = chronometrer_frames(cp, une_frame,
                                     warmup=WARMUP_FRAMES, serie=SERIE_FRAMES)
        stats = mesure["stats"]
        stats["cote_px"] = cote
        stats["pixels"] = cote * cote
        stats["ns_par_pixel_median"] = (
            stats["mediane_ms"] * 1e6 / (cote * cote))
        resultats[str(cote)] = stats
        print(f"  cote={cote:5d} : médiane {stats['mediane_ms']:8.3f} ms  "
              f"p99 {stats['p99_ms']:8.3f} ms  "
              f"({stats['ns_par_pixel_median']:6.2f} ns/px)")
    return resultats


def _mesure_3_verrou(cp) -> dict:
    """M-3 — coût du verrou de la garde 2, mesuré à PART et déclaré (préreg §3).
    Le module l'annonce : « jamais supposé négligeable »."""
    from src.f1_gpu.chemin_de_cout import _verrouiller_appelant

    mesure = chronometrer_frames(cp, _verrouiller_appelant,
                                 warmup=WARMUP_FRAMES, serie=SERIE_FRAMES)
    return mesure["stats"]


def _mesure_2b_gather_kernel(cp, pyramide) -> dict:
    """M-2b — gather en UN kernel (préreg §5-bis). Arithmétique identique à
    M-2, équivalence testée bit à bit. Le kernel est construit HORS chrono ;
    les métadonnées de slots sont en cache, fait déclaré avec le chiffre."""
    from src.f1_gpu.chemin_de_cout import GatherKernel

    gk = GatherKernel(pyramide)
    gk.gather(COTES_ECRAN[0])  # compilation JIT, hors mesure

    resultats: dict[str, dict] = {}
    for cote in COTES_ECRAN:
        def une_frame(c=cote):
            gk.gather(c)

        stats = chronometrer_frames(cp, une_frame, warmup=WARMUP_FRAMES,
                                    serie=SERIE_FRAMES)["stats"]
        stats["cote_px"] = cote
        stats["pixels"] = cote * cote
        stats["ns_par_pixel_median"] = stats["mediane_ms"] * 1e6 / (cote * cote)
        resultats[str(cote)] = stats
        print(f"  cote={cote:5d} : médiane {stats['mediane_ms']:8.3f} ms  "
              f"p99 {stats['p99_ms']:8.3f} ms  "
              f"({stats['ns_par_pixel_median']:6.2f} ns/px)")
    return resultats


def _mesure_4_controle_couverture(cp, cote: int) -> dict:
    """M-4 — coût du contrôle fail-loud de couverture.

    POSTE NON ISOLÉ PAR LE PRÉ-ENREGISTREMENT — apparu à l'écriture du driver.
    `gather_chemin_de_cout` termine par `int(xp.isnan(ecran).sum())`, qui force
    une SYNCHRONISATION device->hôte à chaque appel. M-2 le mesure donc sans le
    dire. On applique ici le principe que le préreg §3 posait pour M-3 : isoler,
    déclarer, et soustraire explicitement si le poste pèse.

    Le contrôle n'est PAS retiré du gather : un moteur réel ne le ferait pas à
    chaque frame, mais l'affaiblir pour embellir un chiffre serait exactement la
    faute que le protocole existe pour empêcher."""
    ecran = cp.zeros((cote, cote), dtype=cp.float32)

    def controle():
        int(cp.isnan(ecran).sum())

    mesure = chronometrer_frames(cp, controle,
                                 warmup=WARMUP_FRAMES, serie=SERIE_FRAMES)
    stats = mesure["stats"]
    stats["cote_px"] = cote
    return stats


def _mesure_5_temoin_bande_passante(cp, cote: int) -> dict:
    """M-5 — TÉMOIN D'IMPLÉMENTATION. Coût d'une écriture pleine de l'écran,
    sans boucle Python, sans indexation avancée : le minimum que TOUTE
    implémentation doit payer en bande passante pour produire ces pixels.

    SECOND POSTE NON PRÉVU PAR LE PRÉ-ENREGISTREMENT, et celui-ci décide de
    l'interprétation. La garde 1 de §A48 impose un plancher sur l'ARITHMÉTIQUE
    (« naïf dans l'opérateur ») — l'opérateur de couture réel en fera plus. Elle
    ne dit rien de l'IMPLÉMENTATION : le gather écrit ici boucle en Python sur
    11 slots avec indexation avancée, donc il est vraisemblablement TRÈS
    au-dessus du minimum atteignable. Les deux écarts vont en SENS OPPOSÉS.

    Sans ce témoin, un ratio M-2/M-5 élevé serait lu comme un coût structurel
    alors qu'il mesurerait la lenteur du code de session. Avec lui, la lecture
    devient un ENCADREMENT : le coût structurel est entre M-5 (bande passante
    incompressible) et M-2 (cette implémentation-ci)."""
    source = cp.ones((cote, cote), dtype=cp.float32)
    ecran = cp.empty((cote, cote), dtype=cp.float32)

    def copie():
        ecran[:] = source

    mesure = chronometrer_frames(cp, copie,
                                 warmup=WARMUP_FRAMES, serie=SERIE_FRAMES)
    stats = mesure["stats"]
    stats["cote_px"] = cote
    return stats


def _loi_par_pixel(m2: dict) -> dict:
    """Régression du coût sur le nombre de pixels — la LOI, dont 1920x1080 se
    déduit sans que la forme de l'écran ait à être tranchée (préreg §2)."""
    pixels = np.array([m2[str(c)]["pixels"] for c in COTES_ECRAN], dtype=float)
    ms = np.array([m2[str(c)]["mediane_ms"] for c in COTES_ECRAN], dtype=float)
    pente, ordonnee = np.polyfit(pixels, ms, 1)
    residus = ms - (pente * pixels + ordonnee)
    return {
        "ns_par_pixel": float(pente * 1e6),
        "cout_fixe_ms": float(ordonnee),
        "residu_max_ms": float(np.abs(residus).max()),
        "deduction_1920x1080_ms": float(pente * 1920 * 1080 + ordonnee),
    }


def main() -> int:
    cp = exiger_cupy()  # fail-loud : aucune mesure numpy (backend.py:48-54)

    print("TRANCHE — coût du gather-plancher et résidence VRAM")
    print(f"protocole : claude/prereg-tranche-cout-rendu-2026-08-03.md")
    print(f"warmup={WARMUP_FRAMES} EXCLU, série={SERIE_FRAMES}, "
          "médiane ET p99 (gravé §A15)\n")

    libre_avant, total = _vram_mo(cp)
    print(f"VRAM : {libre_avant:.0f} Mo libres / {total:.0f} Mo totaux")

    geo = GeometriePyramide(n_fov=N_FOV, n_niv=N_NIV, n0=N0, slots=SLOTS_V4)
    transferts = TransfertComptable(cp)
    pyramide = PyramideFovea(cp, geo, transferts)
    transferts.frame_suivante()  # clôt la descente initiale (B4, hors-série)

    print("\nM-1 — résidence")
    m1 = _mesure_1_residence(cp, pyramide, libre_avant)
    print(f"  octets_etat      : {m1['octets_etat_mo']:8.1f} Mo  "
          f"(prédit préreg : ~126 Mo)")
    print(f"  memGetInfo delta : {m1['delta_memgetinfo_mo']:8.1f} Mo")
    print(f"  écart mempool    : {m1['ecart_mempool_mo']:8.1f} Mo")

    print("\nM-2 — gather")
    m2 = _mesure_2_gather(cp, pyramide)
    loi = _loi_par_pixel(m2)
    print(f"  loi : {loi['ns_par_pixel']:.2f} ns/px + "
          f"{loi['cout_fixe_ms']:.3f} ms fixe "
          f"(résidu max {loi['residu_max_ms']:.3f} ms)")
    print(f"  => 1920x1080 déduit : {loi['deduction_1920x1080_ms']:.3f} ms")

    print("\nM-2b — gather en UN kernel (préreg §5-bis)")
    m2b = _mesure_2b_gather_kernel(cp, pyramide)
    loi_b = _loi_par_pixel(m2b)
    print(f"  loi : {loi_b['ns_par_pixel']:.2f} ns/px + "
          f"{loi_b['cout_fixe_ms']:.3f} ms fixe "
          f"(résidu max {loi_b['residu_max_ms']:.3f} ms)")
    print(f"  => 1920x1080 déduit : {loi_b['deduction_1920x1080_ms']:.3f} ms")

    ref_ms = m2[str(COTES_ECRAN[-1])]["mediane_ms"]
    ref_b_ms = m2b[str(COTES_ECRAN[-1])]["mediane_ms"]

    print("\nM-3 — verrou (garde 2)")
    m3 = _mesure_3_verrou(cp)
    part3 = 100.0 * m3["mediane_ms"] / ref_ms
    part3b = 100.0 * m3["mediane_ms"] / ref_b_ms
    print(f"  médiane {m3['mediane_ms']:.6f} ms — {part3:.3f} % de M-2, "
          f"{part3b:.3f} % de M-2b")

    print("\nM-4 — contrôle fail-loud de couverture (poste non prévu au préreg)")
    m4 = _mesure_4_controle_couverture(cp, COTES_ECRAN[-1])
    part4 = 100.0 * m4["mediane_ms"] / ref_ms
    part4b = 100.0 * m4["mediane_ms"] / ref_b_ms
    print(f"  médiane {m4['mediane_ms']:.6f} ms — {part4:.3f} % de M-2, "
          f"**{part4b:.1f} % de M-2b**")
    print(f"  gather kernel NET du contrôle à {COTES_ECRAN[-1]} : "
          f"{ref_b_ms - m4['mediane_ms']:.3f} ms")

    print("\nM-5 — témoin de bande passante (poste non prévu au préreg)")
    m5 = _mesure_5_temoin_bande_passante(cp, COTES_ECRAN[-1])
    ratio = ref_ms / max(m5["mediane_ms"], 1e-12)
    ref_b_ms = m2b[str(COTES_ECRAN[-1])]["mediane_ms"]
    ratio_b = ref_b_ms / max(m5["mediane_ms"], 1e-12)
    print(f"  écriture pleine {COTES_ECRAN[-1]}² : {m5['mediane_ms']:.3f} ms")
    print(f"  ratio M-2  / M-5 : {ratio:7.1f}x  (boucle Python)")
    print(f"  ratio M-2b / M-5 : {ratio_b:7.1f}x  (kernel unique)"
          f"  — seuil de prononçabilité : 10x (préreg §5-bis)")

    # ----- GARDE DE LECTURE 1 (préreg §5) : la trace AVANT le chiffre -------
    violations = violations_consignees()
    if violations:
        print(f"\n*** CHIFFRE NUL : {len(violations)} violation(s) de la "
              f"garde 2 consignée(s) : {violations}. Un chemin perceptuel a "
              "atteint le chemin-de-coût — rien ne garantit ce qui a été "
              "mesuré (préreg §5.1).")
        return 2

    # ----- témoin de dégénérescence (préreg §2) ----------------------------
    ns_temoin = m2[str(COTE_TEMOIN)]["ns_par_pixel_median"]
    ns_grand = m2[str(COTES_ECRAN[-1])]["ns_par_pixel_median"]
    ecart_temoin = abs(ns_temoin - ns_grand) / max(ns_grand, 1e-12)
    temoin_distingue = ecart_temoin > 0.10

    artefact = {
        "protocole": "claude/prereg-tranche-cout-rendu-2026-08-03.md",
        "machine": {
            "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "vram_totale_mo": total,
            "vram_libre_avant_mo": libre_avant,
            "cupy": cp.__version__,
            "python": platform.python_version(),
        },
        "configuration": {
            "quadruplet": "V4",
            "n_fov": N_FOV, "n_niv": N_NIV, "n0": N0,
            "slots": [{"niveau": s.niveau, "c": s.c, "role": s.role}
                      for s in SLOTS_V4],
            "placement_energie": "2 au niveau le plus fin "
                                 "[INFÉRENCE DE SESSION, préreg §2]",
        },
        "m1_residence": m1,
        "m2_gather": m2,
        "m2_loi": loi,
        "m2b_gather_kernel": m2b,
        "m2b_loi": loi_b,
        "m2b_note": ("Arithmétique IDENTIQUE à M-2, équivalence testée bit à "
                     "bit. Métadonnées de slots en CACHE : coût amorti à zéro "
                     "à fovéa immobile, PAS en régime mobile. M-2b est donc un "
                     "plancher de plus."),
        "ratio_m2b_sur_m5": ratio_b,
        "m3_verrou": m3,
        "m4_controle_couverture": m4,
        "m4_note": ("Poste NON isolé par le pré-enregistrement, apparu à "
                    "l'écriture du driver : le contrôle fail-loud force une "
                    "synchronisation device->hôte par appel, donc M-2 "
                    "l'inclut. Isolé et déclaré ici, jamais retiré du gather."),
        "part_verrou_pct_m2": part3, "part_verrou_pct_m2b": part3b,
        "part_controle_pct_m2": part4, "part_controle_pct_m2b": part4b,
        "m5_temoin_bande_passante": m5,
        "m5_note": ("TÉMOIN D'IMPLÉMENTATION, non prévu au préreg. La garde 1 "
                    "de §A48 impose un plancher sur l'ARITHMÉTIQUE, pas sur "
                    "l'implémentation : ce gather boucle en Python sur 11 "
                    "slots. Le coût structurel est ENCADRÉ entre M-5 (bande "
                    "passante incompressible) et M-2 (cette implémentation). "
                    "Un ratio élevé mesure la lenteur du code de session, pas "
                    "la structure."),
        "ratio_m2_sur_m5": ratio,
        "temoin_degenerescence": {
            "ns_par_pixel_temoin_512": ns_temoin,
            "ns_par_pixel_grand": ns_grand,
            "ecart_relatif": ecart_temoin,
            "se_distingue": bool(temoin_distingue),
        },
        "violations_garde_2": list(violations),
        "portee": ("LE GATHER-PLANCHER coûte ces chiffres — JAMAIS « le rendu "
                   "coûte X » (§A48 garde 4). Plancher de la classe gather : "
                   "l'opérateur de couture réel ajoutera de l'arithmétique. La "
                   "dette « mesurable quand le compositeur existera » reste "
                   "OUVERTE."),
    }

    ARTEFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTEFACT.write_text(json.dumps(artefact, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\nArtefact écrit : {ARTEFACT}")

    if not temoin_distingue:
        print("\n*** INDÉTERMINÉE : le témoin 512 ne se distingue pas des "
              f"grandes tailles (écart {ecart_temoin:.1%} <= 10 %). La mesure "
              "ne mesure pas ce qu'on croit (préreg §4, branche "
              "transversale).")
        return 3

    # ----- garde de prononçabilité, fixée AVANT le run (préreg §5-bis) -----
    if ratio_b >= 10.0:
        print(f"\n*** INDÉTERMINÉE : ratio M-2b/M-5 = {ratio_b:.1f}x >= 10. "
              "L'implémentation domine encore le coût structurel ; aucune "
              "branche G n'est prononçable (préreg §5-bis).")
        return 4

    print("\nAucune branche n'est prononcée ici : la lecture appartient au "
          "journal, contre les branches écrites AVANT (préreg §4).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
