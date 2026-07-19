"""F1 — driver M-a-ter : frame-time du QUADRUPLET-JEU V2 (re-épinglage §A16).

Pré-enregistrement gravé (pocCascade2phys PREREGISTRATION.md §A16, commit
38a2739, 2026-07-19) — cité : « Protocole : harnais tranche-1 + kernel
fusionné M-a′, config = V2 (liste de slots (niveau, c) paramétrique —
extension mineure du driver requise, revue avant run), fovéa mobile E4c,
chrono B6, vérifs #1/#2 reconduites, natif. »

CONFIG V2 GRAVÉE (R1, §A16) : fovéale 9 slots (2 niveaux fins à c=8, 7
supérieurs à c=4) + 3 slots énergie c=8 aux niveaux fins. 12 slots,
17 blocs (fenêtre, système), ≈3.15 M cellules. Coût prédit 14.33 ms
(9.271 fovéale + 5.055 énergie), marge 14.2 %.

RÉPARTITION DES 3 SLOTS ÉNERGIE — LECTURE NOMMÉE : §A16 grave « 3 slots
énergie c=8 fins » sans répartir entre les 2 niveaux fins. La géométrie B3
(INCHANGÉE) ne définit que 2 offsets d'énergie par niveau (±n_fov) : la
seule répartition compatible sans inventer un 3e offset est donc 2 au
niveau le plus fin + 1 au second. Retenue ici, reportée dans le JSON —
aucune géométrie neuve, aucun offset inventé.

CRITÈRES PRÉ-ÉCRITS (gravés — consommés ici en LECTURE MÉCANIQUE ; ce
driver REPORTE et ne prononce RIEN) :
  - MORT : médiane frame-time > 16.7 ms => le quadruplet-jeu V2 est mort
    (seuil T2, INCHANGÉ — aucun seuil ne bouge) ;
  - modèle : médiane hors [12.18, 16.48] ms (14.33 ± 15 %, R3) => AUTRE
    remonté (le modèle linéaire en slots serait faux — information
    capitale, PAS un échec) ;
  - COHÉRENCE NOMMÉE (R3) : la bande entière est SOUS 16.7 — si le modèle
    tient, M-a-ter passe ; une mort impliquerait AUSSI un AUTRE du modèle ;
  - gate §6 reconduit (pas une mort) : résidence > 1.35 Go => REMONTER.

S2 — LECTURE R3 PRÉCISÉE (§A16-complément, 03f2534, gravé AVANT toute
donnée) : le modèle (14.33 ms) prédit du CALCUL PUR (ancre M-a′, qui
mesurait transferts hors-mesure) ; la frame du harnais tranche-1 inclut
les transferts (~1.2–1.5 ms à V2, M-c réduit de 27 à 12 fenêtres
`[calculé]`). Les deux critères portent donc sur des grandeurs
DIFFÉRENTES, et c'est gravé avant le run — sans cette précision, un AUTRE
prévu d'avance aurait vidé le test du modèle :
  - **bande [12.18, 16.48] appliquée à la COMPOSANTE CALCUL** = médiane
    frame − médiane des transferts, les DEUX reportées en évidence ;
  - **MORT sur la médiane frame COMPLÈTE > 16.7 ms** (T2 est un budget de
    frame, transferts compris).

DIAGNOSTIC NON-VERDICTAL : la config V1 (c=4 partout, 14 slots ×1 système)
est mesurée dans le MÊME run — coût marginal nul, information pour la
spec. Elle ne porte AUCUN critère. Sa répartition d'énergie (5 slots) suit
la même règle B3 : les niveaux les plus fins d'abord, 2 par niveau.

Câblage B9 : exige les PASS #1 (vram_concordance) et #2
(sensibilite_chrono) du registre — « vérifs #1/#2 reconduites » (§A16) :
elles sont RELUES du registre écrit par M-a, jamais recalculées ici.
Sortie JSON SANS timestamp.

Machine : iluin-tworings3, terminal natif UNIQUEMENT. Usage :
  .venv/bin/python scripts/run_f1_ma_ter.py

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
    Slot,
)
from src.f1_gpu.substrat_fusionne import pas_f_fusionne  # noqa: E402
from src.f1_gpu.transferts import TransfertComptable  # noqa: E402
from src.f1_gpu.verifs import exiger_pass, vram_nvidia_smi_octets  # noqa: E402

OUT_DIR = ROOT / "outputs" / "f1"
OUT_JSON_PATH = OUT_DIR / "ma_ter.json"

# Scénario épinglé v1-jeu (R2, §A16) : n = 512 cellules d'arc, J = 9
# niveaux sous h0 => N_niv = 10 (niveau 0 CPU + 9 niveaux GPU).
N_FOV: int = 512
N_NIV: int = 10

# Chiffres GRAVÉS (T2, R3, gate §6) — consommés en lecture mécanique SEULEMENT.
B_FRAME_MS: float = 16.7
BANDE_MODELE_MS: tuple[float, float] = (12.18, 16.48)
COUT_PREDIT_GRAVE_MS: float = 14.33
GATE_RESIDENCE_OCTETS: int = int(1.35 * 1024 ** 3)

# Ancre du modèle de coût (§A16) : « 1 slot ≡ fenêtre 512² = 1.685 ms à
# c=8, 0.843 ms à c=4 [MESURÉ M-a′ + linéarité structurelle en systèmes] ».
MS_PAR_SLOT: dict[int, float] = {8: 1.685, 4: 0.843}

# Vérifs #1/#2 reconduites (§A16) — relues du registre, jamais recalculées.
VERIFS_EXIGEES: tuple[str, ...] = ("vram_concordance", "sensibilite_chrono")

N_SLOTS_ENERGIE_V2: int = 3
N_SLOTS_ENERGIE_V1: int = 5
MAX_ENERGIE_PAR_NIVEAU: int = 2   # B3 : offsets ±n_fov, pas un de plus


def _slots_energie(n_niv: int, n_energie: int, c: int) -> list[Slot]:
    """Place `n_energie` slots d'énergie aux niveaux les plus FINS, au plus
    MAX_ENERGIE_PAR_NIVEAU par niveau (B3 : la géométrie ne définit que 2
    offsets d'énergie, ±n_fov). Règle nommée — §A16 ne répartit pas."""
    slots: list[Slot] = []
    niveau = n_niv - 1
    restant = n_energie
    while restant > 0:
        if niveau < 1:
            raise ValueError(
                f"_slots_energie : {n_energie} slots d'énergie ne tiennent "
                f"pas sur {n_niv - 1} niveaux GPU à "
                f"{MAX_ENERGIE_PAR_NIVEAU}/niveau (B3).")
        pour_ce_niveau = min(MAX_ENERGIE_PAR_NIVEAU, restant)
        slots.extend(Slot(niveau=niveau, c=c, role="energie")
                     for _ in range(pour_ce_niveau))
        restant -= pour_ce_niveau
        niveau -= 1
    return slots


def slots_v2(n_niv: int = N_NIV) -> tuple[Slot, ...]:
    """Config V2 GRAVÉE (R1, §A16) : fovéale 9 slots — les 2 niveaux les
    plus FINS à c=8, les 7 supérieurs à c=4 — + 3 slots énergie c=8 aux
    niveaux fins (2 au plus fin, 1 au second : lecture B3, cf. docstring
    du module). 12 slots, 17 blocs, 14.33 ms prédits."""
    niveaux_fins = (n_niv - 1, n_niv - 2)
    slots = [Slot(niveau=j, c=8 if j in niveaux_fins else 4, role="fovea")
             for j in range(1, n_niv)]
    slots += _slots_energie(n_niv, N_SLOTS_ENERGIE_V2, c=8)
    return tuple(slots)


def slots_v1(n_niv: int = N_NIV) -> tuple[Slot, ...]:
    """Config V1 DIAGNOSTIQUE (§A16 : « V1 (c=4 partout, 14 slots) mesurée
    aussi — coût marginal nul, information spec ») : 9 slots fovéale + 5
    slots énergie, c=4 partout. 14 slots, 14 blocs, 11.80 ms prédits.
    AUCUN critère ne porte sur cette config."""
    slots = [Slot(niveau=j, c=4, role="fovea") for j in range(1, n_niv)]
    slots += _slots_energie(n_niv, N_SLOTS_ENERGIE_V1, c=4)
    return tuple(slots)


def cout_predit_ms(geo: GeometriePyramide) -> float:
    """Coût prédit par le modèle §A16 : somme des ancres par slot (le
    modèle est LINÉAIRE en slots — c'est LUI que la bande R3 teste)."""
    return sum(MS_PAR_SLOT[slot.c] for slot in geo.slots)


class ApplicateurFusionne:
    """Applicateur de F pour M-a-ter : le kernel FUSIONNÉ de M-a′
    (`pas_f_fusionne`) appliqué aux fenêtres d'un niveau, TELLES QUELLES.

    S1 (§A16-complément) ayant élargi la garde à {1, 2} systèmes, le
    buffer d'un niveau part au kernel sans reshape ni regroupement : un
    slot c=4 EST un bloc à un système — c'est le design V2, et son coût
    est un objet de la mesure. Regrouper deux slots c=4 en un appel à 2
    systèmes aurait changé le nombre de lancements, donc la mesure.

    B2 : le tampon d'étage est préalloué par SHAPE au premier appel (donc
    au warmup du chrono, EXCLU de la série) et réutilisé — zéro
    réallocation d'état en série. Deux niveaux de même shape partagent le
    tampon : il n'est qu'un temporaire INTRA-appel, aucun état ne survit
    d'un niveau à l'autre (tous les lancements passent par le stream par
    défaut, donc l'ordre d'exécution suit l'ordre d'émission)."""

    def __init__(self):
        self._tampons: dict[tuple, object] = {}

    def __call__(self, fenetres, xp, sortie):
        tampon = self._tampons.get(fenetres.shape)
        if tampon is None:
            tampon = xp.empty_like(fenetres)
            self._tampons[fenetres.shape] = tampon
        return pas_f_fusionne(fenetres, xp, sortie=sortie,
                              tampon_etage=tampon)


def liberer_vram(cp) -> None:
    """Rend la VRAM au driver entre configs (résidence par config propre)
    — appelé APRÈS le retour de `mesurer_config`."""
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def mesurer_config(cp, nom: str, geo: GeometriePyramide) -> dict:
    """Une config : pyramide à slots + fovéa mobile (E4c) + F FUSIONNÉ,
    chrono B6. La résidence est photographiée AVANT libération."""
    transferts = TransfertComptable(cp)
    pyramide = PyramideFovea(cp, geo, transferts,
                             pas_f=ApplicateurFusionne())
    transferts.frame_suivante()  # clôt la descente initiale (hors régime, B4)

    def frame() -> None:
        pyramide.frame(delta_x=1)
        transferts.frame_suivante()

    chrono = chronometrer_frames(cp, frame)

    mempool = cp.get_default_memory_pool()
    return {
        "nom": nom,
        "config": geo.resume_slots(),
        "cout_predit_ms": cout_predit_ms(geo),
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


def lecture_mecanique(v2: dict) -> dict:
    """Consommation MÉCANIQUE des critères gravés §A16 + §A16-complément
    (S2) sur la config V2 — aucun verdict n'est prononcé ici (le driver
    REPORTE). Les deux critères portent sur des grandeurs DIFFÉRENTES :
    la MORT sur la frame COMPLÈTE, la bande du modèle sur le CALCUL seul
    (frame − transferts)."""
    mediane_frame = v2["frame_time"]["mediane_ms"]
    mediane_transferts = (
        v2["transferts_regime"]["transfert_total_ms"]["mediane_ms"])
    mediane_calcul = mediane_frame - mediane_transferts
    residence = v2["residence"]["mempool_total_octets"]
    bas, haut = BANDE_MODELE_MS
    return {
        "seuil_grave_ms": B_FRAME_MS,
        "mediane_frame_complete_ms": mediane_frame,
        "mediane_transferts_ms": mediane_transferts,
        "mediane_calcul_ms": mediane_calcul,
        "p99_frame_complete_ms": v2["frame_time"]["p99_ms"],
        "decomposition_s2": (
            "MORT sur la frame COMPLÈTE (T2 = budget de frame, transferts "
            "compris) ; bande du modèle sur le CALCUL seul (l'ancre M-a′ "
            "mesurait transferts hors-mesure) — §A16-complément S2"),
        "mediane_depasse_seuil": bool(mediane_frame > B_FRAME_MS),
        "branche_preecrite_si_depasse": (
            "MORT du quadruplet-jeu V2 (seuil T2 inchangé), prononcée sur "
            "la frame COMPLÈTE"),
        "modele_de_cout": {
            "cout_predit_grave_ms": COUT_PREDIT_GRAVE_MS,
            "cout_predit_recalcule_ms": v2["cout_predit_ms"],
            "bande_ms": [bas, haut],
            "grandeur_testee": "composante CALCUL (frame − transferts)",
            "mediane_calcul_ms": mediane_calcul,
            "calcul_hors_bande": bool(not bas <= mediane_calcul <= haut),
            "branche_preecrite_si_hors_bande": (
                "AUTRE remonté : le modèle linéaire en slots est faux — "
                "information capitale, PAS un échec (§A16 R3)"),
        },
        "gate_residence": {
            "seuil_octets": GATE_RESIDENCE_OCTETS,
            "residence_mempool_total_octets": residence,
            "declenche": bool(residence > GATE_RESIDENCE_OCTETS),
            "note": "gate §6 reconduit — REMONTER, ce n'est PAS une mort",
        },
    }


def main() -> None:
    cp = exiger_cupy()
    exiger_pass(VERIFS_EXIGEES)

    geos = {"V2": GeometriePyramide(n_fov=N_FOV, n_niv=N_NIV,
                                    slots=slots_v2()),
            "V1_diagnostic": GeometriePyramide(n_fov=N_FOV, n_niv=N_NIV,
                                               slots=slots_v1())}

    configs: list[dict] = []
    for nom, geo in geos.items():
        print(f"config {nom} ...", flush=True)
        configs.append(mesurer_config(cp, nom, geo))
        liberer_vram(cp)

    par_nom = {c["nom"]: c for c in configs}
    lecture = lecture_mecanique(par_nom["V2"])

    document = {
        "meta": {
            "mesure": "M-a-ter frame-time du quadruplet-jeu V2 (§A16 "
                      "38a2739 + §A16-complément 03f2534)",
            "protocole": {
                "harnais": "tranche-1 (pyramide + fovéa mobile E4c + B4)",
                "F": "kernel FUSIONNÉ M-a′ — kernel CUDA à diff VIDE, "
                     "garde élargie à {1, 2} systèmes (S1, 3 verrous "
                     "testés : tests/test_f1_substrat_fusionne_s1.py)",
                "scenario": "v1-jeu R2 : 2D mono-observateur, FOV 20°, "
                            "n=512 d'arc, J=9, monde h0 500k, 3050 Ti",
                "chrono": "B6 (cuda-Events, warmup exclu, médiane ET p99)",
                "transferts": (
                    "DANS la frame mesurée (harnais tranche-1 complet) ; "
                    "S2 : MORT sur la frame COMPLÈTE, bande du modèle sur "
                    "la composante CALCUL (frame − transferts)"),
            },
            "cellule": {"n_fov": N_FOV, "n_niv": N_NIV},
            "choix": {"E4a": "(h,hu,hv,s)x{1,2} selon c du slot",
                      "E4b": "buffers preallocues + mempool (B2)",
                      "E4c": "balayage 1 cellule fine/frame",
                      "E4d": "remontee detail seuille (B4)",
                      "eps_detail": EPS_DETAIL},
            "repartition_energie": (
                "§A16 ne répartit pas les slots d'énergie entre niveaux "
                "fins ; B3 n'offrant que 2 offsets (±n_fov) par niveau, la "
                "seule répartition sans géométrie neuve est retenue : les "
                "niveaux les plus fins d'abord, 2 par niveau"),
            "verifs_exigees": list(VERIFS_EXIGEES),
            "diagnostic_non_verdictal": (
                "V1 (c=4 partout, 14 slots) mesurée dans le même run — "
                "AUCUN critère ne porte sur elle"),
        },
        "configs": configs,
        "lecture_mecanique": lecture,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON_PATH.write_text(
        json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 78)
    print("F1 -- M-a-ter : quadruplet-jeu V2 (lecture mécanique)")
    print("=" * 78)
    for c in configs:
        ft = c["frame_time"]
        cfg = c["config"]
        print(f"  {c['nom']:<14} {cfg['n_slots_total']:>2} slots / "
              f"{cfg['blocs_actifs_gpu']:>2} blocs  "
              f"prédit={c['cout_predit_ms']:6.3f} ms  "
              f"médiane={ft['mediane_ms']:.3f} ms  p99={ft['p99_ms']:.3f} ms")
    modele = lecture["modele_de_cout"]
    print("-" * 78)
    print("  V2 -- S2 : les deux critères portent sur des grandeurs "
          "DIFFÉRENTES")
    print(f"    frame COMPLÈTE = {lecture['mediane_frame_complete_ms']:.3f} ms"
          f"   (dont transferts {lecture['mediane_transferts_ms']:.3f} ms)")
    print(f"    composante CALCUL = "
          f"{lecture['mediane_calcul_ms']:.3f} ms  "
          f"= frame − transferts")
    print(f"    MORT   : frame complète > {B_FRAME_MS} ms  -> "
          f"{lecture['mediane_depasse_seuil']}")
    print(f"    MODÈLE : calcul dans {modele['bande_ms']} ms  -> "
          f"hors bande = {modele['calcul_hors_bande']}")
    print(f"  gate §6 : résidence="
          f"{lecture['gate_residence']['residence_mempool_total_octets'] / 1024 ** 3:.3f}"
          f" Go  seuil=1.35 Go  "
          f"déclenché={lecture['gate_residence']['declenche']}")
    print("  RAPPEL gravé (R3 + S2) : la bande porte sur le CALCUL et est "
          "entièrement sous 16.7 ; la mort porte sur la frame COMPLÈTE — "
          "l'écart entre les deux EST le trafic, reporté ci-dessus.")
    print(f"[REPORT] -> {OUT_JSON_PATH}")
    print("POINT D'ARRÊT OBLIGATOIRE : lecture remontée à Romain, aucun "
          "verdict prononcé ici, aucun enchaînement automatique.")


if __name__ == "__main__":
    main()
