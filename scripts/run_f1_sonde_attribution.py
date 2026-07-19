"""F1 — SONDE D'ATTRIBUTION du résidu M-a-ter (§A16-lecture-M-a-ter,
pocCascade2phys a2440a9, pré-enregistrée AVANT tout code de sonde).

CE QUI EST DÉJÀ LU (gravé, non rejoué ici) : V2 frame complète 31.097 ms
> 16.7 ⇒ MORT du candidat V2 ; composante calcul 30.484 ms hors bande
[12.18, 16.48] ⇒ AUTRE — le modèle linéaire en slots a un terme manquant
de la taille de F lui-même (×2.13). Transferts V2 : 0.613 ms. Le suspect
nommé au diagnostic (étiquette HYPOTHÈSE ARITHMÉTIQUE sur 2 points) est
la MACHINERIE DE PYRAMIDE en CuPy non fusionné — prédiction Harten,
extraction/seuillage/compaction B4, mise à jour de référence.

CETTE SONDE ATTRIBUE, ELLE NE REMÉDIE PAS. Aucune escalade n'est décidée
ici : le cap interdit toute nouvelle escalade DU F, et « fusionner la
chose suivante » est précisément le tapis roulant que le cap craignait.
Le remède (design E4d ou borne fusionnée de la machinerie) est une
décision SÉPARÉE, postérieure à cette lecture.

LES DEUX BRAS (config V2 EXACTE — mêmes 12 slots §A16, répartition
énergie S3 2+1, importés du driver M-a-ter pour que l'identité de config
soit structurelle et non recopiée) :
  - BRAS A — fovéa IMMOBILE, remontée B4 OFF : F fusionné seul dans la
    boucle pyramide. ATTENDU PRÉ-ÉCRIT : A ∈ [12.18, 16.48] (le modèle F)
    ⇒ attribution CONFIRMÉE ; hors bande ⇒ l'hypothèse machinerie est
    FAUSSE, chercher ailleurs (AUTRE remonté).
  - BRAS B — fovéa MOBILE E4c, remontée B4 OFF : isole la prédiction de
    déplacement.
Décomposition mécanique gravée : coût(remontée) = V2_full − B ;
coût(prédiction) = B − A. Les TROIS grandeurs sont reportées.

COMMENT L'« OFF » EST IMPLÉMENTÉ (load-bearing — on retire du TRAVAIL,
PAS de la structure) :
  - fovéa IMMOBILE = `frame(delta_x=0)`. Le balayage E4c s'arrête ; aucun
    niveau ne bouge, donc aucune re-prédiction CPU et aucune H2D. Rien
    n'est désalloué : les fenêtres et les références gardent leurs shapes
    et leur résidence.
  - remontée B4 OFF = `PyramideFovea(..., remontee_active=False)`. Sont
    retirés : la soustraction `fen − ref`, le seuillage, l'indexation
    booléenne et le `flatnonzero` (compaction), les DEUX D2H, la mise à
    jour incrémentale de la référence. NE SONT PAS retirés : le buffer
    `references[j]`, PRÉALLOUÉ à sa shape pleine (B2 identique, même
    résidence, même pression mempool), la boucle de niveaux, et F qui
    s'applique aux mêmes buffers aux mêmes shapes.
Autrement dit les deux bras mesurent la MÊME pyramide que M-a-ter, privée
d'étages de travail — jamais une pyramide plus petite. Contrôle reporté :
les transferts du bras A doivent être nuls, ceux du bras B sans D2H.

CE QUE LE BRAS A CONTIENT ENCORE — À LIRE AVANT D'INTERPRÉTER A. « F seul »
signifie « la pyramide privée de déplacement et de remontée », PAS « le
kernel nu ». Restent dans A, et n'étaient PAS dans l'ancre M-a′ (qui
mesurait UN niveau) :
  1. **9 appels `pas_f_fusionne`** (un par niveau GPU) au lieu d'un seul,
     donc **18 lancements de kernel** au lieu de 2 — l'overhead de
     lancement est multiplié par 9 ;
  2. **9 réductions CFL SYNCHRONES** : `reduction_cfl` finit par
     `float(...max())`, un rapatriement scalaire qui synchronise le
     device. Neuf points de synchronisation par frame là où l'ancre en
     avait un. La réduction est « payée non consommée » par protocole
     (B5/E4a) — elle est donc légitimement dans la mesure, mais son coût
     n'est PAS linéaire en slots ;
  3. l'orchestration CPU par niveau (calcul des origines B3, indexation
     des buffers), négligeable en soi mais présente.
Conséquence pour la lecture, nommée AVANT le run : si A sort hors bande,
« l'hypothèse machinerie est FAUSSE » ne veut pas dire « le modèle F est
faux » — ces trois postes-là deviennent les suspects immédiats, et ils
sont per-niveau, pas per-slot. Le JSON les compte explicitement
(`postes_residuels_bras_a`) pour que la lecture ne les oublie pas.

Kernel CUDA, chrono B6 et substrat : INTOUCHÉS (diff vide). Vérifs #1/#2
reconduites, natif. Sortie JSON SANS timestamp.

Machine : iluin-tworings3, terminal natif UNIQUEMENT. Usage :
  .venv/bin/python scripts/run_f1_sonde_attribution.py

POINT D'ARRÊT OBLIGATOIRE après le run : lecture remontée à Romain, aucun
verdict, aucune décision d'escalade ou de design enchaînée."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_f1_ma_ter import (  # noqa: E402
    BANDE_MODELE_MS,
    N_FOV,
    N_NIV,
    VERIFS_EXIGEES,
    ApplicateurFusionne,
    cout_predit_ms,
    liberer_vram,
    slots_v2,
)
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
from src.f1_gpu.verifs import exiger_pass, vram_nvidia_smi_octets  # noqa: E402

OUT_DIR = ROOT / "outputs" / "f1"
OUT_JSON_PATH = OUT_DIR / "sonde_attribution.json"

# Chiffres GRAVÉS de la lecture M-a-ter (§A16-lecture-M-a-ter, a2440a9) —
# NON rejoués ici : la décomposition s'y adosse telle quelle.
V2_FULL_MEDIANE_MS: float = 31.097     # frame complète mesurée (MORT)
V2_CALCUL_MS: float = 30.484           # frame − transferts (AUTRE)
V2_TRANSFERTS_MS: float = 0.613        # schéma diff à 12 fenêtres

# Balayage de chaque bras (E4c : 1 cellule fine/frame ; 0 = fovéa immobile).
DELTA_X_IMMOBILE: int = 0
DELTA_X_MOBILE: int = 1


def mesurer_bras(cp, nom: str, geo: GeometriePyramide, delta_x: int,
                 remontee_active: bool) -> dict:
    """Un bras de la sonde : pyramide V2 + F FUSIONNÉ, chrono B6. Les
    étages désactivés retirent du TRAVAIL, jamais de l'allocation — la
    résidence est photographiée pour l'attester."""
    transferts = TransfertComptable(cp)
    pyramide = PyramideFovea(cp, geo, transferts,
                             pas_f=ApplicateurFusionne(),
                             remontee_active=remontee_active)
    transferts.frame_suivante()  # clôt la descente initiale (hors régime, B4)

    def frame() -> None:
        pyramide.frame(delta_x=delta_x)
        transferts.frame_suivante()

    chrono = chronometrer_frames(cp, frame)

    mempool = cp.get_default_memory_pool()
    regime = transferts.stats_regime(exclure_premiers=1 + WARMUP_FRAMES)
    return {
        "nom": nom,
        "etages": {"fovea_mobile": bool(delta_x > 0), "delta_x": delta_x,
                   "remontee_b4": remontee_active},
        "config": geo.resume_slots(),
        "cout_predit_f_ms": cout_predit_ms(geo),
        "frames": {"warmup_exclu": WARMUP_FRAMES, "serie": SERIE_FRAMES},
        "frame_time": chrono["stats"],
        "transferts_regime": regime,
        "controle_etages_off": {
            "h2d_octets_median": regime["h2d_octets"]["mediane"],
            "d2h_octets_median": regime["d2h_octets"]["mediane"],
            "attendu": ("bras A : h2d ET d2h nuls (fovéa immobile, "
                        "remontée OFF) ; bras B : d2h nul, h2d = colonnes "
                        "entrantes"),
        },
        "residence": {
            "mempool_used_octets": int(mempool.used_bytes()),
            "mempool_total_octets": int(mempool.total_bytes()),
            "etat_prealloue_octets": pyramide.octets_etat(),
            "nvidia_smi_octets": vram_nvidia_smi_octets(),
        },
    }


def postes_residuels_bras_a(geo: GeometriePyramide) -> dict:
    """Ce que le bras A contient EN PLUS du kernel nu, compté — tout est
    per-NIVEAU, donc invisible au modèle qui est per-SLOT. Sert la lecture
    si A sort hors bande (cf. docstring du module)."""
    n_niveaux = len(list(geo.niveaux_gpu))
    return {
        "appels_pas_f_par_frame": n_niveaux,
        "lancements_kernel_par_frame": 2 * n_niveaux,
        "reductions_cfl_synchrones_par_frame": n_niveaux,
        "reference_m_a_prime": {"appels_pas_f": 1, "lancements_kernel": 2,
                                "reductions_cfl_synchrones": 1},
        "note": ("ces postes sont per-NIVEAU (9x l'ancre M-a′), pas "
                 "per-slot — le modèle linéaire en slots ne les voit pas"),
    }


def lecture_mecanique(bras_a: dict, bras_b: dict) -> dict:
    """Consommation MÉCANIQUE de la sonde pré-enregistrée — le driver
    REPORTE, il ne prononce RIEN (ni verdict, ni escalade, ni design)."""
    a_ms = bras_a["frame_time"]["mediane_ms"]
    b_ms = bras_b["frame_time"]["mediane_ms"]
    bas, haut = BANDE_MODELE_MS
    cout_prediction = b_ms - a_ms
    cout_remontee = V2_FULL_MEDIANE_MS - b_ms
    return {
        "bande_modele_ms": [bas, haut],
        "v2_full_grave_ms": V2_FULL_MEDIANE_MS,
        "decomposition_ms": {
            "f_seul_bras_a": a_ms,
            "prediction_deplacement_b_moins_a": cout_prediction,
            "remontee_full_moins_b": cout_remontee,
            "somme_de_controle": a_ms + cout_prediction + cout_remontee,
            "note": ("la somme reconstitue V2_full par construction — "
                     "c'est un contrôle d'arithmétique, PAS une mesure "
                     "indépendante"),
        },
        "bras_b_ms": b_ms,
        "attendu_preecrit_bras_a": {
            "a_dans_bande": bool(bas <= a_ms <= haut),
            "branche_si_dans_bande": (
                "attribution CONFIRMÉE : F seul retrouve le modèle, le "
                "résidu appartient à la MACHINERIE de pyramide"),
            "branche_si_hors_bande": (
                "l'hypothèse machinerie est FAUSSE — chercher ailleurs, "
                "AUTRE remonté (§A16-lecture-M-a-ter)"),
        },
        "rappel_portee": (
            "SONDE = attribution seule. Aucun verdict, aucune escalade, "
            "aucun design décidés ici ; le remède est une décision "
            "SÉPARÉE (cap : pas de tapis roulant)."),
    }


def main() -> None:
    cp = exiger_cupy()
    exiger_pass(VERIFS_EXIGEES)

    geo = GeometriePyramide(n_fov=N_FOV, n_niv=N_NIV, slots=slots_v2())
    plan = (("A_f_seul", DELTA_X_IMMOBILE, False),
            ("B_avec_prediction", DELTA_X_MOBILE, False))

    bras: list[dict] = []
    for nom, delta_x, remontee in plan:
        print(f"bras {nom} (delta_x={delta_x}, remontée={remontee}) ...",
              flush=True)
        bras.append(mesurer_bras(cp, nom, geo, delta_x, remontee))
        liberer_vram(cp)

    par_nom = {b["nom"]: b for b in bras}
    lecture = lecture_mecanique(par_nom["A_f_seul"],
                                par_nom["B_avec_prediction"])

    document = {
        "meta": {
            "mesure": "Sonde d'attribution du résidu M-a-ter "
                      "(§A16-lecture-M-a-ter, pocCascade2phys a2440a9)",
            "protocole": {
                "config": "V2 EXACTE (12 slots §A16, énergie S3 2+1) — "
                          "slots importés du driver M-a-ter",
                "F": "kernel FUSIONNÉ M-a′ (substrat_fusionne, INTOUCHÉ)",
                "chrono": "B6 (cuda-Events, warmup exclu, médiane ET p99)",
                "off_implemente": (
                    "fovéa immobile = delta_x=0 ; remontée OFF = flag "
                    "`remontee_active=False`. Les deux retirent du "
                    "TRAVAIL, aucune allocation : `references[j]` reste "
                    "préalloué à shape pleine (B2 identique)"),
            },
            "cellule": {"n_fov": N_FOV, "n_niv": N_NIV},
            "choix": {"E4a": "(h,hu,hv,s)x{1,2} selon c du slot",
                      "E4b": "buffers preallocues + mempool (B2)",
                      "E4c": "balayage 1 cellule fine/frame (bras B seul)",
                      "E4d": "remontee DESACTIVEE dans les deux bras",
                      "eps_detail": EPS_DETAIL},
            "ancrage_grave": {
                "v2_full_ms": V2_FULL_MEDIANE_MS,
                "v2_calcul_ms": V2_CALCUL_MS,
                "v2_transferts_ms": V2_TRANSFERTS_MS,
                "note": "lus au run M-a-ter, NON rejoués ici",
            },
            "verifs_exigees": list(VERIFS_EXIGEES),
            "postes_residuels_bras_a": postes_residuels_bras_a(geo),
        },
        "bras": bras,
        "lecture_mecanique": lecture,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON_PATH.write_text(
        json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")

    decomposition = lecture["decomposition_ms"]
    print("=" * 78)
    print("F1 -- SONDE D'ATTRIBUTION du résidu M-a-ter (lecture mécanique)")
    print("=" * 78)
    for b in bras:
        ft = b["frame_time"]
        etages = b["etages"]
        print(f"  {b['nom']:<18} fovéa "
              f"{'MOBILE ' if etages['fovea_mobile'] else 'IMMOBILE'}  "
              f"remontée={'ON' if etages['remontee_b4'] else 'OFF'}  "
              f"médiane={ft['mediane_ms']:.3f} ms  p99={ft['p99_ms']:.3f} ms")
    print("-" * 78)
    print("  DÉCOMPOSITION (les trois postes) :")
    print(f"    F seul (bras A)              = "
          f"{decomposition['f_seul_bras_a']:.3f} ms")
    print(f"    prédiction déplacement (B−A) = "
          f"{decomposition['prediction_deplacement_b_moins_a']:.3f} ms")
    print(f"    remontée B4 (full−B)         = "
          f"{decomposition['remontee_full_moins_b']:.3f} ms   "
          f"[full gravé = {V2_FULL_MEDIANE_MS} ms]")
    attendu = lecture["attendu_preecrit_bras_a"]
    print(f"  ATTENDU PRÉ-ÉCRIT : A dans {lecture['bande_modele_ms']} ms  -> "
          f"{attendu['a_dans_bande']}")
    print("  RAPPEL : sonde = ATTRIBUTION seule — aucun verdict, aucune "
          "escalade, aucun design décidés ici.")
    print(f"[REPORT] -> {OUT_JSON_PATH}")
    print("POINT D'ARRÊT OBLIGATOIRE : lecture remontée à Romain, aucun "
          "enchaînement automatique.")


if __name__ == "__main__":
    main()
