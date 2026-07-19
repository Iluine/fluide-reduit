"""F1 — BRAS s2-MOBILE : la prédiction de déplacement sous orchestration
BATCHÉE (§A17-lecture-s1-s2 G2 amendé, pocCascade2phys 61641c2).

POURQUOI CE BRAS EXISTE : s2 a montré que le modèle F re-calibré TIENT
une fois l'orchestration guérie (A_batché = 14.432 vs 14.341 prédits,
+0.6 %), ce qui laisse un budget machinerie de 2.268 ms. Or la prédiction
de déplacement a été mesurée à 2.135 ms — soit le budget ENTIER — mais
sous l'ANCIENNE orchestration per-niveau, précisément la maladie que s2
vient de guérir pour F. Ce bras est le falsificateur pré-nommé (L4) : il
re-mesure la prédiction avec l'orchestration batchée.

CE QUI CHANGE ET CE QUI NE CHANGE PAS PAR RAPPORT À s2 :
  - IDENTIQUE : la config V2 (12 slots, 17 blocs), l'orchestration
    batchée (2 groupes, 4 lancements/frame), la réduction CFL laissée
    on-device, la remontée B4 OFF, l'état initial (mêmes appels au
    générateur jetable, même graine) ;
  - AJOUTÉ : la fovéa MOBILE E4c (balayage 1 cellule fine/frame). Les
    colonnes entrantes des niveaux déplacés sont donc prédites et
    descendues — c'est ELLE qu'on mesure, par différence.

CE QUE LE DÉPLACEMENT PAIE ICI, INTÉGRALEMENT (aucun travail omis) :
pour CHAQUE niveau déplacé — et l'alignement dyadique fait que le niveau
j n'avance qu'une frame sur 2^(J−j) — on paie (1) la prédiction voisin
CPU des colonnes entrantes depuis le niveau 0, via la MÊME fonction que
`PyramideFovea` (`pyramide.predire_bloc_cpu`, appelée et non
ré-implémentée : c'est ce qui rend l'omission impossible), (2) UNE
descente H2D par niveau déplacé, batchée sur les slots de ce niveau,
(3) le `roll` et l'écriture des colonnes entrantes sur la fenêtre ET sur
la référence — la pyramide fait les deux HORS de l'étage de remontée,
donc les omettre serait sous-compter. Un test compare, frame par frame,
les octets et le nombre d'appels H2D de ce bras à ceux de `PyramideFovea`
mobile : ils doivent être IDENTIQUES.

LE CHOIX D'ORCHESTRATION, NOMMÉ : dans chaque groupe, les slots sont
rangés PAR NIVEAU, de sorte que les slots d'un même niveau occupent une
tranche CONTIGUË du buffer. C'est ce qui permet une seule H2D par niveau,
exactement comme la pyramide per-niveau. L'ordre des blocs à l'intérieur
d'un batch ne change pas le travail de F (le kernel traite des blocs
aplatis indépendants), donc la soustraction A_mobile − A_batché reste
propre. `plan_groupes` construit ce rangement et le JSON le reporte.

CE QUE LA MÉDIANE CAPTURE — limite de lecture, nommée avant le run :
l'alignement dyadique B3 fait que le niveau j n'avance qu'une frame sur
2^(J−j). Sur la série, le niveau le plus fin bouge à CHAQUE frame, le
suivant une sur deux, ... et le plus grossier une sur 256. La frame
MÉDIANE ne déplace donc qu'un ou deux niveaux fins, pas les neuf : la
prédiction lue par la médiane est celle d'une frame typique, ni la
moyenne sur tous les niveaux, ni le pire cas — lequel vit dans le p99,
reporté à côté. C'est exactement le régime du bras B auquel elle se
compare (même géométrie, même cadence), donc la soustraction reste
propre ; mais lire « prédiction_batchée » comme « le coût de déplacer
toute la pyramide » serait faux. Le JSON reporte la cadence théorique
par niveau (`cadence_deplacement`) pour que la nuance soit sous les yeux.

LECTURE MÉCANIQUE (gravée §A17-lecture-s1-s2) :
    prédiction_batchée = A_mobile − A_batché
    cible L1+L3        = 16.7 − A_mobile
A_batché est LU dans le JSON de s2, jamais recopié — ce driver refuse de
démarrer sans lui. Les deux grandeurs sont REPORTÉES SANS
INTERPRÉTATION : la décision — le design est-il jouable à cette cible, ou
faut-il un candidat plus mince / la porte 33.3 par la porte de devant —
revient à Romain à la lecture. Aucun verdict ici.

Kernel CUDA, chrono B6 et substrats : INTOUCHÉS (diff vide ; bloc
_SOURCE à l'empreinte
e8fcaad47db0af010728d5c5d878e1629191bfef4e0382fa80aa3cefd9f27f04).
Vérifs #1/#2 reconduites, natif. Sortie JSON SANS timestamp.

Machine : iluin-tworings3, terminal natif UNIQUEMENT. Usage (APRÈS s2) :
  .venv/bin/python scripts/run_f1_s2_mobile.py

POINT D'ARRÊT OBLIGATOIRE après le run : lecture remontée à Romain."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_f1_ma_ter import (  # noqa: E402
    B_FRAME_MS,
    N_NIV,
    VERIFS_EXIGEES,
    liberer_vram,
    slots_v2,
)
from scripts.run_f1_s1_ancre_1sys import N_FOV, etat_mono_systeme  # noqa: E402
from scripts.run_f1_s2_batche import (  # noqa: E402
    LANCEMENTS_PAR_FRAME,
    pas_f_fusionne_async,
)
from scripts.run_f1_s2_batche import OUT_JSON_PATH as S2_JSON_PATH  # noqa: E402
from src.f1_gpu.backend import exiger_cupy  # noqa: E402
from src.f1_gpu.chrono import (  # noqa: E402
    SERIE_FRAMES,
    WARMUP_FRAMES,
    chronometrer_frames,
)
from src.f1_gpu.pyramide import (  # noqa: E402
    GRAINE_MONDE,
    GeometriePyramide,
    monde0_jetable,
    predire_bloc_cpu,
)
from src.f1_gpu.substrat_jetable import etat_initial_jetable  # noqa: E402
from src.f1_gpu.transferts import TransfertComptable  # noqa: E402
from src.f1_gpu.verifs import exiger_pass, vram_nvidia_smi_octets  # noqa: E402

OUT_DIR = ROOT / "outputs" / "f1"
OUT_JSON_PATH = OUT_DIR / "s2_mobile.json"

DELTA_X_MOBILE: int = 1        # E4c : 1 cellule fine/frame

# Référence NON verdictale : la prédiction telle que mesurée sous
# l'ancienne orchestration per-niveau (§A16-lecture-sonde-attribution).
PREDICTION_PER_NIVEAU_MS: float = 2.135


def plan_groupes(geo: GeometriePyramide) -> list[dict]:
    """Range les slots par nombre de systèmes, et DANS chaque groupe par
    NIVEAU — de sorte que les slots d'un niveau forment une tranche
    contiguë. C'est ce rangement qui permet UNE descente H2D par niveau
    déplacé, comme la pyramide per-niveau."""
    plans: list[dict] = []
    for n_systemes in (2, 1):
        niveaux = [j for j in geo.niveaux_gpu
                   if geo.n_systemes_du_niveau(j) == n_systemes]
        tranches: dict[int, tuple[int, int]] = {}
        curseur = 0
        for j in niveaux:
            tranches[j] = (curseur, curseur + geo.n_slots(j))
            curseur += geo.n_slots(j)
        if curseur:
            plans.append({"n_systemes": n_systemes, "niveaux": niveaux,
                          "tranches": tranches, "n_slots": curseur})
    return plans


def cadence_deplacement(geo: GeometriePyramide) -> dict:
    """Période de déplacement de chaque niveau, en frames (alignement
    dyadique B3 : le niveau j avance une frame sur 2^(J−j) quand la fovéa
    balaie 1 cellule fine/frame). Sert la lecture : la médiane ne voit
    que les niveaux à petite période."""
    periodes = {str(j): 2 ** (geo.niveau_fin - j) for j in geo.niveaux_gpu}
    return {
        "periode_par_niveau_frames": periodes,
        "niveaux_a_chaque_frame": [j for j in geo.niveaux_gpu
                                   if periodes[str(j)] == 1],
        "niveaux_moins_dune_fois_sur_la_serie": [
            j for j in geo.niveaux_gpu
            if periodes[str(j)] > SERIE_FRAMES],
        "note": ("la frame MÉDIANE ne déplace que les niveaux à petite "
                 "période — la prédiction lue est celle d'une frame "
                 "typique, pas du pire cas (p99) ni d'une moyenne sur "
                 "tous les niveaux"),
    }


class PyramideBatcheeMobile:
    """Le bras s2 (batché, CFL on-device, remontée OFF) AVEC la fovéa
    mobile E4c. Les buffers sont ceux de s2 ; la géométrie B3, les
    origines et la prédiction sont celles de `PyramideFovea` — appelées,
    jamais réécrites.

    B2 : fenêtres, tampons d'étage et références sont préalloués à la
    construction. Les références ne sont jamais LUES (remontée OFF) mais
    elles sont bien ÉCRITES par le déplacement — la pyramide le fait hors
    de l'étage de remontée, et ce travail est donc payé ici aussi."""

    def __init__(self, cp, geo: GeometriePyramide,
                 transferts: TransfertComptable,
                 graine: int = GRAINE_MONDE,
                 pas_f=pas_f_fusionne_async):
        # `pas_f` est injectable pour que la MACHINERIE DE DÉPLACEMENT
        # (prédiction, descentes, roll/écritures) soit testable sous
        # numpy, sans device : c'est ce chemin-là que le verrou
        # « aucun travail omis » compare à `PyramideFovea`.
        self.cp = cp
        self.geo = geo
        self.transferts = transferts
        self.pas_f = pas_f
        self.plans = plan_groupes(geo)
        self.monde0 = monde0_jetable(geo, graine)
        self.centre_fin = geo.centre_fin_initial()
        self.fenetres: list = []
        self.tampons: list = []
        self.references: list = []
        for plan in self.plans:
            # État initial IDENTIQUE à s2 (mêmes appels, même graine) :
            # la soustraction A_mobile − A_batché n'isole ainsi que le
            # déplacement, pas une différence de contenu.
            if plan["n_systemes"] == 2:
                bloc = cp.asarray(etat_initial_jetable(
                    plan["n_slots"], geo.n_fov, graine))
            else:
                bloc = etat_mono_systeme(cp, plan["n_slots"], geo.n_fov)
            self.fenetres.append(bloc)
            self.tampons.append(cp.empty_like(bloc))
            self.references.append(cp.array(bloc, copy=True))
        self._origines = {j: geo.origines(j, self.centre_fin)
                          for j in geo.niveaux_gpu}
        self.dt_cfl: list = [None] * len(self.plans)

    def frame(self, delta_x: int = DELTA_X_MOBILE) -> dict:
        """Une frame : déplacement (prédiction + descente + roll/écriture
        sur fenêtre ET référence), puis F batché en 4 lancements. Pas de
        remontée."""
        cp = self.cp
        geo = self.geo
        if delta_x < 0:
            raise ValueError(f"frame : delta_x={delta_x} < 0 (balayage +x).")
        self.centre_fin += delta_x
        niveaux_deplaces: list[int] = []

        for indice, plan in enumerate(self.plans):
            fenetre_groupe = self.fenetres[indice]
            reference_groupe = self.references[indice]
            for j in plan["niveaux"]:
                nouvelles = geo.origines(j, self.centre_fin)
                dx = nouvelles[0][1] - self._origines[j][0][1]
                if dx < 0:
                    raise RuntimeError(
                        f"frame : recul d'origine au niveau {j} (dx={dx}) — "
                        "balayage E4c strictement +x.")
                if dx > 0:
                    niveaux_deplaces.append(j)
                    debut, fin = plan["tranches"][j]
                    vue_fenetre = fenetre_groupe[debut:fin]
                    vue_reference = reference_groupe[debut:fin]
                    if dx >= geo.n_fov:
                        # Saut de fenêtre (diagnostic E4c) : pleine.
                        bloc = predire_bloc_cpu(
                            self.monde0, geo, j, nouvelles, 0, geo.n_fov,
                            plan["n_systemes"])
                        entrant = self.transferts.descendre(bloc)
                        vue_fenetre[...] = entrant
                        vue_reference[...] = entrant
                    else:
                        vue_fenetre[...] = cp.roll(vue_fenetre, -dx, axis=-1)
                        vue_reference[...] = cp.roll(
                            vue_reference, -dx, axis=-1)
                        bloc = predire_bloc_cpu(
                            self.monde0, geo, j, nouvelles,
                            geo.n_fov - dx, geo.n_fov, plan["n_systemes"])
                        entrant = self.transferts.descendre(bloc)
                        vue_fenetre[..., geo.n_fov - dx:] = entrant
                        vue_reference[..., geo.n_fov - dx:] = entrant
                self._origines[j] = nouvelles

        # F batché : un appel par groupe, 2 étages chacun (4 lancements).
        for indice, (fen, tampon) in enumerate(
                zip(self.fenetres, self.tampons)):
            _, dt_cfl = self.pas_f(fen, self.cp, fen, tampon)
            self.dt_cfl[indice] = dt_cfl

        return {"niveaux_deplaces": sorted(niveaux_deplaces),
                "remontee_active": False}

    def audit_cfl(self) -> dict:
        """Rapatriement UNIQUE hors chrono (même discipline que s2)."""
        valeurs = [float(dt) if dt is not None else None
                   for dt in self.dt_cfl]
        return {
            "dt_cfl_par_groupe": valeurs,
            "toutes_finies_positives": bool(
                all(v is not None and np.isfinite(v) and v > 0.0
                    for v in valeurs)),
        }

    def octets_etat(self) -> int:
        return int(sum(b.nbytes for b in self.fenetres)
                   + sum(b.nbytes for b in self.references))

    def n_blocs(self) -> int:
        return int(sum(b.shape[0] * b.shape[1] for b in self.fenetres))


def lire_a_batche_s2(chemin: Path = S2_JSON_PATH) -> dict:
    """A_batché MESURÉ, lu dans le JSON de s2 — jamais recopié. Fail-loud
    si s2 n'a pas tourné : la prédiction batchée est une DIFFÉRENCE, elle
    n'existe pas sans son terme de gauche."""
    if not chemin.exists():
        raise RuntimeError(
            f"le bras mobile exige A_batché, mesuré par s2, absent "
            f"({chemin}). Lancer d'abord `.venv/bin/python scripts/"
            "run_f1_s2_batche.py` — la prédiction batchée est la "
            "DIFFÉRENCE A_mobile − A_batché ; recopier le terme à la main "
            "serait un chiffre non traçable. Aucune mesure ici.")
    document = json.loads(chemin.read_text(encoding="utf-8"))
    try:
        valeur = float(document["lecture_mecanique"]["a_batche_ms"])
    except (KeyError, TypeError, ValueError) as erreur:
        raise RuntimeError(
            f"bras mobile : JSON s2 illisible ou incomplet ({chemin}) — "
            f"{erreur!r}. Aucune mesure.") from erreur
    if not np.isfinite(valeur) or valeur <= 0.0:
        raise RuntimeError(
            f"bras mobile : A_batché non exploitable ({valeur}). "
            "Aucune mesure.")
    return {"a_batche_ms": valeur, "source": str(chemin)}


def lecture_mecanique(a_mobile_ms: float, a_batche: dict) -> dict:
    """Consommation MÉCANIQUE de la lecture gravée — les deux grandeurs
    sont REPORTÉES, aucune n'est interprétée."""
    a_batche_ms = a_batche["a_batche_ms"]
    prediction_batchee = a_mobile_ms - a_batche_ms
    cible = B_FRAME_MS - a_mobile_ms
    return {
        "a_mobile_ms": a_mobile_ms,
        "a_batche_ms": a_batche_ms,
        "a_batche_source": a_batche["source"],
        "prediction_batchee": {
            "formule": "A_mobile - A_batché",
            "valeur_ms": prediction_batchee,
            "reference_per_niveau_ms": PREDICTION_PER_NIVEAU_MS,
            "note": ("la référence per-niveau (§A16-lecture-sonde) est "
                     "reportée pour situer, PAS pour conclure : ce driver "
                     "ne compare ni ne commente"),
        },
        "cible_l1_l3": {
            "formule": f"{B_FRAME_MS} - A_mobile",
            "valeur_ms": cible,
            "seuil_t2_ms": B_FRAME_MS,
            "note": ("REPORTÉE sans interprétation. Ce qu'elle implique — "
                     "design jouable, candidat plus mince, ou porte 33.3 "
                     "par la porte de devant — est une décision de Romain "
                     "à la lecture, pas une sortie de ce driver. Une cible "
                     "nulle ou négative se lit telle quelle."),
        },
        "rappel_portee": (
            "bras de sonde (L4, G2 amendé) : il MESURE la prédiction sous "
            "orchestration batchée. Aucun verdict, aucun levier tranché, "
            "aucune escalade — le kernel est intouché."),
    }


def main() -> None:
    cp = exiger_cupy()
    exiger_pass(VERIFS_EXIGEES)
    a_batche = lire_a_batche_s2()      # fail-loud AVANT toute allocation

    print(f"A_batché lu de s2 : {a_batche['a_batche_ms']:.3f} ms", flush=True)
    geo = GeometriePyramide(n_fov=N_FOV, n_niv=N_NIV, slots=slots_v2())
    transferts = TransfertComptable(cp)
    pyramide = PyramideBatcheeMobile(cp, geo, transferts)
    transferts.frame_suivante()   # clôt l'initialisation (hors régime, B4)
    print(f"bras mobile : {pyramide.n_blocs()} blocs, "
          f"{LANCEMENTS_PAR_FRAME} lancements/frame, fovéa MOBILE ...",
          flush=True)

    def frame() -> None:
        pyramide.frame(delta_x=DELTA_X_MOBILE)
        transferts.frame_suivante()

    chrono = chronometrer_frames(cp, frame)
    audit = pyramide.audit_cfl()      # rapatriement UNIQUE, hors chrono

    mempool = cp.get_default_memory_pool()
    a_mobile_ms = chrono["stats"]["mediane_ms"]
    lecture = lecture_mecanique(a_mobile_ms, a_batche)

    document = {
        "meta": {
            "mesure": "bras s2-mobile — prédiction de déplacement sous "
                      "orchestration batchée (§A17-lecture-s1-s2 G2 "
                      "amendé, pocCascade2phys 61641c2)",
            "protocole": {
                "config": "V2 EXACTE (12 slots, 17 blocs) — slots importés",
                "orchestration": ("IDENTIQUE à s2 : 2 groupes, "
                                  f"{LANCEMENTS_PAR_FRAME} lancements/"
                                  "frame, CFL on-device non synchronisante"),
                "fovea": f"MOBILE E4c, delta_x={DELTA_X_MOBILE}",
                "remontee_b4": False,
                "etat_initial": ("mêmes appels au générateur jetable que "
                                 "s2, même graine — la soustraction "
                                 "n'isole que le déplacement"),
                "prediction": ("`pyramide.predire_bloc_cpu` APPELÉE (non "
                               "ré-implémentée) ; UNE H2D par niveau "
                               "déplacé ; roll et écriture des colonnes "
                               "entrantes sur la fenêtre ET la référence"),
                "rangement": ("slots rangés par niveau dans chaque groupe "
                              "— tranches contiguës, donc une seule H2D "
                              "par niveau comme la pyramide"),
                "chrono": "B6 (cuda-Events, warmup exclu, médiane ET p99)",
            },
            "groupes": [{"n_systemes": p["n_systemes"],
                         "n_slots": p["n_slots"],
                         "niveaux": p["niveaux"],
                         "tranches": {str(j): list(t)
                                      for j, t in p["tranches"].items()}}
                        for p in pyramide.plans],
            "cellule": {"n_fov": N_FOV, "n_niv": N_NIV},
            "frames": {"warmup_exclu": WARMUP_FRAMES, "serie": SERIE_FRAMES},
            "cadence_deplacement": cadence_deplacement(geo),
            "verifs_exigees": list(VERIFS_EXIGEES),
        },
        "frame_time": chrono["stats"],
        "transferts_regime": transferts.stats_regime(
            exclure_premiers=1 + WARMUP_FRAMES),
        "audit_cfl": audit,
        "residence": {
            "mempool_used_octets": int(mempool.used_bytes()),
            "mempool_total_octets": int(mempool.total_bytes()),
            "etat_prealloue_octets": pyramide.octets_etat(),
            "nvidia_smi_octets": vram_nvidia_smi_octets(),
        },
        "lecture_mecanique": lecture,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON_PATH.write_text(
        json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    liberer_vram(cp)

    prediction = lecture["prediction_batchee"]
    cible = lecture["cible_l1_l3"]
    print("=" * 78)
    print("F1 -- bras s2-mobile : prédiction sous batch (lecture mécanique)")
    print("=" * 78)
    print(f"  A_mobile  = {a_mobile_ms:.3f} ms   "
          f"p99 = {chrono['stats']['p99_ms']:.3f} ms")
    print(f"  A_batché  = {lecture['a_batche_ms']:.3f} ms  (lu de s2)")
    print(f"  prédiction batchée = {prediction['valeur_ms']:.3f} ms   "
          f"[per-niveau mesurée auparavant : "
          f"{PREDICTION_PER_NIVEAU_MS} ms]")
    print(f"  cible L1+L3        = {cible['valeur_ms']:.3f} ms   "
          f"(= {B_FRAME_MS} - A_mobile)")
    print(f"  audit CFL : {audit['dt_cfl_par_groupe']}  "
          f"finies>0 = {audit['toutes_finies_positives']}")
    print("  Les DEUX grandeurs sont REPORTÉES SANS INTERPRÉTATION — la "
          "décision revient à Romain à la lecture.")
    print(f"[REPORT] -> {OUT_JSON_PATH}")
    print("POINT D'ARRÊT OBLIGATOIRE : lecture remontée à Romain, aucun "
          "verdict, aucun enchaînement automatique.")


if __name__ == "__main__":
    main()
