"""F1 — M-a-quater : le pipeline COMPLET du candidat V4 (achat B,
§A18-lecture-borne-L3, pocCascade2phys 7c4a4ca).

CE QUE M-a-quater ASSEMBLE, et d'où vient chaque pièce :
  - **V4 EMBOÎTÉE** (propriété E) : 11 slots — 9 fovéale (2 niveaux fins
    c=8, 7 supérieurs c=4) + 2 énergie c=8 aux quarts libres de la
    parente fovéale, ZÉRO tour d'ancêtres. Les offsets ±n_fov/2 tombent
    exactement aux bornes de la couverture (identité
    oy_j − 2·oy_parent = n_fov/2, exacte à tous les niveaux) ;
  - **CHAMPS D'ÉCHELLE** (P2) : les systèmes au-delà de c_parent sont à
    contenu grossier nul, donc prédits par remplissage-ZÉRO exact ;
  - **PRÉDICTION GPU-SIDE** : la propriété E rend caducs les cinq replis
    de s3 — 10 slots descendent parent→enfant, le niveau 1 SEUL reste CPU
    (son parent est le monde 0). Prouvé par test, pas supposé ;
  - **REMONTÉE L3 + L1 ÉTALÉE** : le kernel L3 tel que mesuré à la borne,
    INTOUCHÉ, appliqué en ROUND-ROBIN k=4.
  - F batché, CFL on-device, chrono B6, fovéa mobile E4c.

────────────────────────────────────────────────────────────────────────
L1 EST ÉTALÉE, ET C'EST UNE CONSIGNE D'INTÉGRITÉ, PAS UNE OPTIMISATION
────────────────────────────────────────────────────────────────────────
Une remontée en RAFALE — tout remonter une frame sur k — ferait passer la
MÉDIANE (k−1 frames sur k sans remontée) tout en laissant un stutter
d'environ 20 ms invisible au critère. Le piège est gravé
(§A18-lecture-borne-L3) et fermé ici : **round-robin, ~11/k fenêtres par
frame, jamais toutes**. Le coût est lissé, le critère médian reste
honnête, et l'amortissement est le même. Un test vérifie qu'aucune frame
ne remonte tout, et qu'un cycle complet couvre chaque slot EXACTEMENT une
fois. La p99 est reportée EN ÉVIDENCE : c'est là que se lirait un stutter
résiduel.

CHOIX D'IMPLÉMENTATION NON COUVERT PAR LE GRAVÉ, REMONTÉ : étaler L1
oblige à faire cohabiter, dans la même frame, le kernel L3 (sur les slots
qui remontent ce tour) et le kernel fusionné (sur les autres). Les deux
kernels étant INTOUCHÉS — `emettre` y est un scalaire uniforme, pas un
masque par slot — la seule façon de les mêler sans y toucher est de
SCINDER chaque groupe en segments contigus : [avant le tour], [le tour],
[après le tour]. Conséquence mesurable : **8 à 12 lancements par frame au
lieu des 4 de s2**. Ordre de grandeur du surcoût : un lancement vaut
quelques microsecondes, donc 4 à 8 lancements de plus ≈ 0.03–0.08 ms —
négligeable devant 14 ms, et sans commune mesure avec les 6.11 ms de s2,
qui venaient surtout des NEUF synchronisations CFL (ici il n'y en a
aucune : la réduction reste on-device). Le compte réel est reporté à
chaque frame dans le JSON (`lancements_par_frame`) plutôt qu'estimé.
Les alternatives ont été écartées : réordonner les buffers à chaque tour
coûterait des copies, et passer un masque par slot exigerait de modifier
le kernel L3 — ce que le cap interdit.

────────────────────────────────────────────────────────────────────────
LE DEUXIÈME SYSTÈME RESTE MOUILLÉ — l'anti-minoration a MORDU
────────────────────────────────────────────────────────────────────────
La borne L3 a mesuré **−16.9 %** entre un second système sec et mouillé :
la consigne était load-bearing, un système nul coûte franchement moins
(warp sans divergence). L'état du harnais garde donc ses DEUX systèmes
mouillés. La prédiction, elle, applique le remplissage-zéro conforme à
P2 — mais un seul slot est concerné (le fovéal du niveau 8, dont le
parent est c=4), et ses colonnes entrantes assèchent progressivement son
système 1. Le driver MESURE cette fraction et la reporte
(`assechement_champs_echelle`) : c'est le prix, chiffré, de la conformité
P2 sur ce slot, et il doit être sous les yeux à la lecture.

LECTURE MÉCANIQUE (gravée) :
  - **MORT : médiane frame COMPLÈTE > 16.7 ms** (T2 inchangé) ;
  - **bande [14.2, 15.9]** — recalculée avec L1 k=4, remplace
    [13.9, 17.2] ; hors bande = AUTRE ;
  - **parts F / prédiction / remontée / transferts, étiquetées [DÉRIVÉ]**
    des ancres gravées et du comptage exact des transferts — la série de
    sondes reste CLOSE, aucun bras n'est ajouté ;
  - **p99 REPORTÉE EN ÉVIDENCE**.
Le driver REPORTE. La décision — V4 vit, ou le repli 33.3 devient LA
décision — revient à Romain.

Modules mesurés INTOUCHÉS : kernel F fusionné (empreinte
e8fcaad47db0af010728d5c5d878e1629191bfef4e0382fa80aa3cefd9f27f04) et
kernel L3 tel que mesuré à la borne.

Machine : iluin-tworings3, terminal natif UNIQUEMENT. Usage :
  .venv/bin/python scripts/run_f1_ma_quater.py

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
)
from scripts.run_f1_s1_ancre_1sys import N_FOV, etat_mono_systeme  # noqa: E402
from scripts.run_f1_s2_batche import (  # noqa: E402
    pas_f_fusionne_async,
    reduction_cfl_on_device,
)
from scripts.run_f1_s2_mobile import DELTA_X_MOBILE, plan_groupes  # noqa: E402
from scripts.run_f1_s3_pred_gpu import predire_colonnes_gpu  # noqa: E402
from src.f1_gpu.backend import exiger_cupy  # noqa: E402
from src.f1_gpu.chrono import (  # noqa: E402
    SERIE_FRAMES,
    WARMUP_FRAMES,
    chronometrer_frames,
)
from src.f1_gpu.pyramide import (  # noqa: E402
    EPS_DETAIL,
    GRAINE_MONDE,
    GeometriePyramide,
    Slot,
    monde0_jetable,
    predire_bloc_cpu,
)
from src.f1_gpu.substrat_jetable import etat_initial_jetable  # noqa: E402
from src.f1_gpu.substrat_l3 import (  # noqa: E402
    CompacteurL3,
    pas_f_l3,
    taille_sortie_max,
)
from src.f1_gpu.transferts import TransfertComptable  # noqa: E402
from src.f1_gpu.verifs import exiger_pass, vram_nvidia_smi_octets  # noqa: E402

OUT_DIR = ROOT / "outputs" / "f1"
OUT_JSON_PATH = OUT_DIR / "ma_quater.json"

# Chiffres GRAVÉS — consommés en lecture mécanique SEULEMENT.
# BANDE_MS RECOMPOSÉE à EPS=1e-2 (§A24, b6c8807). L'ancienne [14.2, 15.9]
# incluait des transferts ~2.4 ms ; à 1e-2 ils tombent à 0.110 et elle ne
# testait plus rien. Recomposition depuis les ancres mesurées :
#   point (prédiction≈0) = F 12.655 + remontée 5.122/4=1.280 + transferts
#                          0.110 = 14.045 ms ;
#   bande d'incertitude : ± (dérive chrono 0.18 + amortissement L1 0.10),
#   la prédiction étant un PLANCHER unilatéral [0, ~0.5] (elle ne peut
#   qu'ajouter). D'où [14.045 − 0.28 ; 14.045 + 0.28 + 0.5] ≈ [13.8, 14.8].
# Le mesuré 14.490 y tombe (+0.445 au-dessus du plancher, = la prédiction
# solde, petite et positive — pas un AUTRE ; cf. rapport §A24).
BANDE_MS: tuple[float, float] = (13.8, 14.8)
F_BATCHE_V4_MS: float = 12.655                 # ancres 1.685 / 0.845
REMONTEE_L3_MESUREE_MS: float = 5.122          # borne L3
CADENCE_L1: int = 4                            # k, armé round-robin
GATE_VRAM_GO: float = 1.35                     # gate de résidence, gravé
N_SLOTS_ENERGIE_V4: int = 2
NIVEAU_PARENT_CPU: int = 1

# ─── SEUIL DE PRODUCTION EPS = 1e-2, APPLIQUÉ (§A23, pocCascade2phys
# b6c8807) ────────────────────────────────────────────────────────────
# La sonde EPS v2 a proposé 1e-2 (les six EPS du balayage passent,
# innocuité ÉTABLIE à 24× sous ic_bas) ; Romain l'APPLIQUE ici. C'est le
# seul point où le régime de production diffère du paramètre d'instrument
# EPS_DETAIL (1e-4), délibérément INCHANGÉ : lui est l'ancre gravée sous
# laquelle la borne L3 fut mesurée, et EPS_EN_VIGUEUR (sonde) est le
# régime FIGÉ qu'importent D-1/D-2 et attribution_b. Aucune de ces ancres
# gelées ne peut porter la valeur de production courante sans réécrire une
# mesure acquise — d'où cette constante distincte (choix d'implémentation
# remonté, le gravé nommant « EPS_EN_VIGUEUR »).
#
# TROIS PORTÉES GRAVÉES (§A23), à garder sous les yeux à toute lecture :
#   1. INNOCENT ≠ OPTIMAL — le balayage ne borne pas par le haut : tous
#      ses points passent, donc 1e-2 est le plus grand TESTÉ, pas un
#      optimum. Où EPS cesserait d'être innocent n'est pas mesuré.
#   2. LE CANAL ≠ LE LOINTAIN — la sonde mesure la fidélité du CANAL
#      (seuil, cadence, transfert), option 1. L'argument de sûreté du
#      lointain (miroir CPU, option 2) reste NON mesuré : « le canal est
#      innocent à 1e-2 » ne dit pas « le lointain est fidèle à 1e-2 ».
#   3. SCOPÉ À CE SUBSTRAT — vaut pour le substrat F1/V4 mesuré, pas
#      au-delà.
EPS_PRODUCTION: float = 1e-2

# Ancres MESURÉES (§A18 P3 : « ancres MESURÉES : 1.685 / 0.845 »). Le
# 2-systèmes vient de M-a′ ; le 1-système vient de s1 BATCHÉE — 0.845, et
# NON l'étiquette structurelle 0.843 que le modèle supposait avant s1.
# C'est cette paire, et elle seule, qui donne les 12.655 ms gravés.
ANCRES_MESUREES_MS: dict[int, float] = {8: 1.685, 4: 0.845}


def slots_v4(n_niv: int = N_NIV) -> tuple[Slot, ...]:
    """Candidat V4 (§A18 P3) : fovéale 9 slots — les 2 niveaux les plus
    fins à c=8, les 7 supérieurs à c=4 — plus 2 slots d'énergie c=8 au
    niveau le PLUS FIN, dont la parente est le fovéal du niveau
    au-dessus. 11 slots, 15 blocs, F batché 12.655 ms prédits."""
    niveaux_fins = (n_niv - 1, n_niv - 2)
    slots = [Slot(niveau=j, c=8 if j in niveaux_fins else 4, role="fovea")
             for j in range(1, n_niv)]
    slots += [Slot(niveau=n_niv - 1, c=8, role="energie")
              for _ in range(N_SLOTS_ENERGIE_V4)]
    return tuple(slots)


def geometrie_v4(n_fov: int = N_FOV, n_niv: int = N_NIV
                 ) -> GeometriePyramide:
    """La géométrie EMBOÎTÉE du candidat — propriété E satisfaite."""
    return GeometriePyramide(n_fov=n_fov, n_niv=n_niv, slots=slots_v4(n_niv),
                             emboitee=True)


def cout_predit_f_ms(geo: GeometriePyramide) -> float:
    """F batché prédit, sur les ancres MESURÉES — 12.655 ms gravés."""
    return sum(ANCRES_MESUREES_MS[slot.c] for slot in geo.slots)


# ----- L1 étalée : round-robin sur tranches CONTIGUËS -----

def plan_round_robin(plans_groupes: list[dict], k: int = CADENCE_L1
                     ) -> list[list[tuple[int, int]]]:
    """Répartit les slots de chaque groupe en `k` tours de tranches
    CONTIGUËS — `[(debut, fin), ...]` par groupe, pour chaque tour.

    Contiguïté voulue : elle permet d'appeler le kernel L3 sur une VUE du
    groupe (les slots qui remontent ce tour) et le kernel fusionné sur le
    reste, sans réordonner les buffers ni toucher aux kernels.

    Étalement voulu : chaque tour prend ~n/k slots de CHAQUE groupe, donc
    aucune frame ne remonte tout — c'est la consigne d'intégrité, pas une
    optimisation."""
    if k < 1:
        raise ValueError(f"plan_round_robin : k={k} < 1.")
    tours: list[list[tuple[int, int]]] = []
    for tour in range(k):
        tranches = []
        for plan in plans_groupes:
            total = plan["n_slots"]
            debut = (total * tour) // k
            fin = (total * (tour + 1)) // k
            tranches.append((debut, fin))
        tours.append(tranches)
    return tours


def diagnostic_round_robin(plans_groupes: list[dict],
                           k: int = CADENCE_L1) -> dict:
    """Preuve auditable de l'étalement : couverture exacte sur un cycle,
    et aucune frame ne remonte tout."""
    tours = plan_round_robin(plans_groupes, k)
    slots_par_tour = [sum(fin - debut for debut, fin in tranches)
                      for tranches in tours]
    total = sum(plan["n_slots"] for plan in plans_groupes)
    couverture: list[int] = []
    for indice, plan in enumerate(plans_groupes):
        vus: list[int] = []
        for tranches in tours:
            debut, fin = tranches[indice]
            vus.extend(range(debut, fin))
        couverture.append(len(vus) == plan["n_slots"]
                          and sorted(vus) == list(range(plan["n_slots"])))
    return {
        "k": k,
        "slots_total": total,
        "slots_par_tour": slots_par_tour,
        "aucune_frame_ne_remonte_tout": bool(
            all(n < total for n in slots_par_tour)),
        "cycle_couvre_chaque_slot_une_fois": bool(all(couverture)),
        "note": ("L1 ÉTALÉE (consigne d'intégrité §A18-lecture-borne-L3) : "
                 "une rafale ferait passer la MÉDIANE en laissant un "
                 "stutter invisible au critère. Le round-robin lisse le "
                 "coût — même amortissement, critère honnête. La p99 est "
                 "reportée en évidence."),
    }


# ----- prédiction : GPU-side partout sauf le niveau 1 -----

def plan_prediction_v4(geo: GeometriePyramide,
                       centre_fin: int | None = None) -> dict:
    """Par slot : descente parent→enfant GPU-side, ou CPU pour le seul
    niveau 1. La propriété E garantit une parente couvrante à tout slot
    de niveau >= 2 — les cinq replis de s3 sont donc CADUCS, et
    `slots_sans_parente()` doit être vide (vérifié, pas supposé)."""
    if centre_fin is None:
        centre_fin = geo.centre_fin_initial()
    orphelins = geo.slots_sans_parente(centre_fin)
    if orphelins:
        raise RuntimeError(
            f"plan_prediction_v4 : propriété E violée pour {orphelins} — "
            "la géométrie n'est pas un arbre, la descente parent->enfant "
            "n'est pas définie. Aucune mesure.")
    plan: dict[int, dict] = {}
    for j in geo.niveaux_gpu:
        if j == NIVEAU_PARENT_CPU:
            plan[j] = {"gpu": {}, "cpu": list(range(geo.n_slots(j)))}
            continue
        plan[j] = {
            "gpu": {i: geo.parente_couvrante(j, i, centre_fin)
                    for i in range(geo.n_slots(j))},
            "cpu": [],
            # Systèmes de l'enfant sans parent : champs d'échelle (P2),
            # prédits par remplissage-ZÉRO exact.
            "systemes_parent": geo.n_systemes_du_niveau(j - 1),
            "systemes_enfant": geo.n_systemes_du_niveau(j),
        }
    return plan


def resume_prediction(geo: GeometriePyramide) -> dict:
    plan = plan_prediction_v4(geo)
    gpu = [(j, i) for j, e in plan.items() for i in e["gpu"]]
    cpu = [(j, i) for j, e in plan.items() for i in e["cpu"]]
    echelle = [j for j, e in plan.items()
               if e.get("systemes_enfant", 0) > e.get("systemes_parent", 0)]
    return {
        "n_slots_gpu": len(gpu),
        "n_slots_cpu": len(cpu),
        "slots_gpu": [{"niveau": j, "slot": i} for j, i in gpu],
        "slots_cpu": [{"niveau": j, "slot": i} for j, i in cpu],
        "niveaux_a_champs_echelle": echelle,
        "propriete_e_satisfaite": geo.slots_sans_parente() == [],
        "note": ("la propriété E rend caducs les cinq replis de s3 : seul "
                 "le niveau 1 reste CPU (son parent est le monde 0)"),
    }


class PipelineMaQuater:
    """Le pipeline complet V4 : déplacement + prédiction (GPU-side ou CPU)
    + F batché + remontée L3 étalée. Les buffers, groupes et tampons
    suivent l'orchestration de s2 ; les références sont préallouées comme
    partout (B2 : on retire du travail, jamais de la structure)."""

    def __init__(self, cp, geo: GeometriePyramide,
                 transferts: TransfertComptable, k: int = CADENCE_L1,
                 graine: int = GRAINE_MONDE, eps: float = EPS_DETAIL,
                 capturer_coefficients: bool = False):
        # `capturer_coefficients` (sonde EPS §A19) : conserve les
        # coefficients remontés de chaque frame pour qu'un consommateur
        # reconstruise le niveau 0 VIVANT. Coûteux (rapatriement retenu
        # en mémoire) — JAMAIS activé dans une passe chronométrée.
        self.capturer_coefficients = bool(capturer_coefficients)
        self.coefficients_frame: list = []
        self.cp = cp
        self.geo = geo
        self.transferts = transferts
        self.eps = float(eps)
        self.k = int(k)
        self.plans = plan_groupes(geo)
        self.tours = plan_round_robin(self.plans, self.k)
        self.plan_prediction = plan_prediction_v4(geo)
        self.monde0 = monde0_jetable(geo, graine)
        self.centre_fin = geo.centre_fin_initial()
        self.frame_courante = 0
        self.fenetres: list = []
        self.tampons: list = []
        self.references: list = []
        for plan in self.plans:
            # Second système MOUILLÉ (anti-minoration : −16.9 % mesuré).
            if plan["n_systemes"] == 2:
                bloc = cp.asarray(etat_initial_jetable(
                    plan["n_slots"], geo.n_fov, graine))
            else:
                bloc = etat_mono_systeme(cp, plan["n_slots"], geo.n_fov)
            self.fenetres.append(bloc)
            self.tampons.append(cp.empty_like(bloc))
            self.references.append(bloc * cp.float32(0.999))
        self.compacteurs = [CompacteurL3(cp, taille_sortie_max(bloc))
                            for bloc in self.fenetres]
        self._origines = {j: geo.origines(j, self.centre_fin)
                          for j in geo.niveaux_gpu}
        self._localisation = self._construire_localisation()
        self.cellules_echelle_nulles = 0
        self.cellules_echelle_total = 0
        self.lancements_derniere_frame = 0

    def _construire_localisation(self) -> dict:
        localisation = {}
        for indice, plan in enumerate(self.plans):
            for j, (debut, _) in plan["tranches"].items():
                for i in range(self.geo.n_slots(j)):
                    localisation[(j, i)] = (indice, debut + i)
        return localisation

    def _vue_slot(self, buffers: list, j: int, i: int):
        indice, position = self._localisation[(j, i)]
        return buffers[indice][position]

    # ----- déplacement + prédiction -----

    def _deplacer(self, delta_x: int) -> dict:
        cp, geo = self.cp, self.geo
        self.centre_fin += delta_x
        niveaux_deplaces: list[int] = []
        colonnes_gpu = colonnes_cpu = 0

        for j in geo.niveaux_gpu:
            nouvelles = geo.origines(j, self.centre_fin)
            dx = nouvelles[0][1] - self._origines[j][0][1]
            if dx < 0:
                raise RuntimeError(
                    f"frame : recul d'origine au niveau {j} (dx={dx}).")
            if dx > 0:
                niveaux_deplaces.append(j)
                plein = dx >= geo.n_fov
                x_debut = 0 if plein else geo.n_fov - dx
                entree = self.plan_prediction[j]

                if not plein:
                    for i in range(geo.n_slots(j)):
                        for buffers in (self.fenetres, self.references):
                            vue = self._vue_slot(buffers, j, i)
                            vue[...] = cp.roll(vue, -dx, axis=-1)

                for i, index_parent in entree["gpu"].items():
                    oy, ox = nouvelles[i]
                    oy_p, ox_p = geo.origines(j - 1, self.centre_fin)[
                        index_parent]
                    parent = self._vue_slot(self.fenetres, j - 1,
                                            index_parent)
                    entrant = predire_colonnes_gpu(
                        parent, cp, geo, oy, ox, oy_p, ox_p, x_debut,
                        geo.n_fov)
                    n_parent = entree["systemes_parent"]
                    n_enfant = entree["systemes_enfant"]
                    for buffers in (self.fenetres, self.references):
                        vue = self._vue_slot(buffers, j, i)
                        vue[:n_parent, ..., x_debut:] = entrant[:n_parent]
                        if n_enfant > n_parent:
                            # CHAMPS D'ÉCHELLE (P2) : les systèmes au-delà
                            # de c_parent sont à contenu grossier nul —
                            # remplissage-ZÉRO exact, par définition.
                            vue[n_parent:, ..., x_debut:] = 0.0
                    if n_enfant > n_parent:
                        largeur = geo.n_fov - x_debut
                        self.cellules_echelle_nulles += largeur
                        self.cellules_echelle_total += geo.n_fov
                    colonnes_gpu += geo.n_fov - x_debut

                if entree["cpu"]:
                    origines_cpu = [nouvelles[i] for i in entree["cpu"]]
                    bloc = predire_bloc_cpu(
                        self.monde0, geo, j, origines_cpu, x_debut,
                        geo.n_fov, geo.n_systemes_du_niveau(j))
                    entrant = self.transferts.descendre(bloc)
                    for rang, i in enumerate(entree["cpu"]):
                        for buffers in (self.fenetres, self.references):
                            vue = self._vue_slot(buffers, j, i)
                            vue[..., x_debut:] = entrant[rang]
                        colonnes_cpu += geo.n_fov - x_debut
            self._origines[j] = nouvelles
        return {"niveaux_deplaces": sorted(niveaux_deplaces),
                "colonnes_gpu": colonnes_gpu, "colonnes_cpu": colonnes_cpu}

    # ----- F batché + remontée L3 étalée -----

    def _appliquer_f(self) -> int:
        """F sur chaque groupe : le kernel L3 (émettant) sur la TRANCHE du
        tour, le kernel fusionné (n'émettant pas) sur le reste. Les deux
        kernels sont INTOUCHÉS — c'est l'orchestration qui étale."""
        cp = self.cp
        tranches = self.tours[self.frame_courante % self.k]
        lancements = 0
        for indice, (fen, tampon) in enumerate(
                zip(self.fenetres, self.tampons)):
            debut, fin = tranches[indice]
            compacteur = self.compacteurs[indice]
            reference = self.references[indice]
            taille = compacteur.taille_precedente()
            compacteur.noter_taille(taille)
            # Les indices émis par le kernel sont relatifs au tableau
            # qu'on lui PASSE — donc à la tranche du tour, pas au groupe.
            # Tout consommateur qui reconstruit depuis ces indices doit
            # ajouter l'offset du segment émetteur, sinon il écrit dans
            # le mauvais slot (le premier) sans que rien ne le signale.
            elements_par_slot = int(fen.shape[1] * 4 * fen.shape[-1]
                                    * fen.shape[-2])
            # L'offset est confié au compacteur, qui le fait voyager AVEC
            # les données du même tour (§A21-complément). Depuis le
            # ping-pong, les données transférées sont celles de n−1 : les
            # indexer par l'offset de n écrirait dans le mauvais slot.
            compacteur.noter_offset(int(debut * elements_par_slot))
            for segment_debut, segment_fin, emet in (
                    (0, debut, False), (debut, fin, True),
                    (fin, fen.shape[0], False)):
                if segment_fin <= segment_debut:
                    continue
                vue = fen[segment_debut:segment_fin]
                vue_tampon = tampon[segment_debut:segment_fin]
                if emet:
                    pas_f_l3(vue, cp, reference[segment_debut:segment_fin],
                             compacteur, sortie=vue, tampon_etage=vue_tampon,
                             eps=self.eps,
                             reduction=reduction_cfl_on_device)
                else:
                    pas_f_fusionne_async(vue, cp, vue, vue_tampon)
                lancements += 2          # deux étages RK2 par appel
            if taille:
                valeurs, indices = compacteur.vues_a_transferer(taille)
                valeurs_cpu = self.transferts.remonter(valeurs)
                indices_cpu = self.transferts.remonter(indices)
                if self.capturer_coefficients:
                    self.coefficients_frame.append(
                        (indice, valeurs_cpu,
                         indices_cpu.astype(np.int64)
                         + compacteur.offset_precedent()))
            compacteur.cloturer_frame()
        return lancements

    def frame(self, delta_x: int = DELTA_X_MOBILE,
              observateur=None) -> dict:
        """Une frame complète. `observateur`, s'il est fourni, est appelé
        UNE fois, APRÈS le déplacement et AVANT l'émission (§A21-complément).

        Pourquoi cet instant précis, et pas un autre : c'est le SEUL où
        `reference` porte la connaissance du CPU DANS LES COORDONNÉES DE
        LA FRAME COURANTE. Le déplacement vient d'y appliquer le roll et
        les colonnes prédites ; l'émission de cette frame n'y a pas encore
        été absorbée. Lue plus tôt, la connaissance est dans les
        coordonnées de la frame précédente ; lue plus tard, elle contient
        déjà ce que le CPU ne recevra qu'à la frame suivante. Confondre
        ces instants, c'est refabriquer le décalage spatial qui a produit
        le plancher 0.082 (§A19-CORRECTION).

        JAMAIS employé dans une passe CHRONOMÉTRÉE — même discipline que
        `capturer_coefficients`. Le coût quand il vaut None est un test
        d'identité par frame, sans commune mesure avec les ~16 ms."""
        self.coefficients_frame = []
        diagnostic = self._deplacer(delta_x)
        if observateur is not None:
            observateur(self)
        self.lancements_derniere_frame = self._appliquer_f()
        self.frame_courante += 1
        diagnostic["lancements"] = self.lancements_derniere_frame
        diagnostic["tour_l1"] = (self.frame_courante - 1) % self.k
        return diagnostic

    # ----- comptes rendus -----

    def assechement_champs_echelle(self) -> dict:
        """Prix CHIFFRÉ de la conformité P2 sur le seul slot concerné (le
        fovéal dont le parent porte moins de systèmes) : ses colonnes
        entrantes arrivent à zéro, donc son second système s'assèche au
        fil du balayage. La borne L3 a mesuré −16.9 % entre sec et
        mouillé : cette fraction dit de combien ce slot-là peut minorer.
        Reportée, jamais corrigée en douce."""
        fraction = (self.cellules_echelle_nulles
                    / max(self.cellules_echelle_total, 1))
        return {
            "colonnes_mises_a_zero": self.cellules_echelle_nulles,
            "colonnes_de_reference": self.cellules_echelle_total,
            "fraction_asséchée": float(min(fraction, 1.0)),
            "ecart_sec_mouille_mesure": -0.169,
            "note": ("conformité P2 : les systèmes au-delà de c_parent "
                     "sont nuls par définition. L'état du harnais reste "
                     "MOUILLÉ (anti-minoration load-bearing) ; seul ce "
                     "slot s'assèche par sa prédiction. ATTÉNUATION "
                     "NOMMÉE : le chemin est vivant, F repropage depuis "
                     "les voisins mouillés dès la frame suivante — la "
                     "colonne prédite à zéro ne le reste pas. Le −16.9 % "
                     "mesurait un système ENTIÈREMENT sec : il borne "
                     "l'effet par le haut, il ne le décrit pas. Fraction "
                     "et atténuation reportées, jamais corrigées."),
        }

    def audit_cfl(self) -> dict:
        return {"compacteurs": len(self.compacteurs),
                "tailles_par_groupe": [c.diagnostic_retard()
                                       for c in self.compacteurs]}

    def octets_etat(self) -> int:
        total = sum(b.nbytes for b in self.fenetres)
        total += sum(b.nbytes for b in self.references)
        for compacteur in self.compacteurs:
            # Les DEUX jeux du ping-pong (§A21) : le doublement des
            # buffers d'émission doit se voir dans la résidence.
            total += compacteur.octets_buffers()
        return int(total)

    def octets_surcout_pingpong(self) -> int:
        """Prix du correctif §A21, isolé : le SECOND jeu de buffers
        d'émission, soit la moitié de leur total. Reporté à part plutôt
        que fondu dans `octets_etat` — un correctif qui coûte de la VRAM
        doit se lire comme tel à la lecture de la résidence."""
        return int(sum(c.octets_buffers() for c in self.compacteurs) // 2)

    def n_blocs(self) -> int:
        return int(sum(b.shape[0] * b.shape[1] for b in self.fenetres))


def parts_derivees(mediane_ms: float, transferts_ms: float,
                   k: int = CADENCE_L1) -> dict:
    """Parts F / prédiction / remontée / transferts, toutes étiquetées
    [DÉRIVÉ] : la série de sondes reste CLOSE, aucun bras n'est ajouté.
    F vient de l'ancre V4, la remontée de la borne L3 amortie par k, les
    transferts sont COMPTÉS exactement, et la prédiction est le SOLDE."""
    remontee = REMONTEE_L3_MESUREE_MS / k
    prediction = mediane_ms - F_BATCHE_V4_MS - remontee - transferts_ms
    return {
        "f_batche_ms": F_BATCHE_V4_MS,
        "remontee_l3_amortie_ms": remontee,
        "transferts_ms": transferts_ms,
        "prediction_solde_ms": prediction,
        "etiquette": "[DÉRIVÉ]",
        "note": ("F = ancre V4 (1.685/0.845) ; remontée = borne L3 5.122 "
                 f"amortie par k={k} ; transferts COMPTÉS ; prédiction = "
                 "SOLDE. Aucun bras ajouté — la série de sondes reste "
                 "close. La MORT porte sur la médiane MESURÉE, pas sur "
                 "ces parts."),
    }


def lecture_mecanique(chrono_stats: dict, transferts_ms: float,
                      prediction: dict, round_robin: dict,
                      assechement: dict, k: int = CADENCE_L1) -> dict:
    """Consommation MÉCANIQUE des critères gravés — le driver REPORTE."""
    mediane = chrono_stats["mediane_ms"]
    p99 = chrono_stats["p99_ms"]
    bas, haut = BANDE_MS
    return {
        "mediane_frame_complete_ms": mediane,
        "p99_frame_complete_ms": p99,
        "p99_en_evidence": {
            "valeur_ms": p99,
            "ecart_a_la_mediane_ms": p99 - mediane,
            "note": ("REPORTÉE EN ÉVIDENCE : c'est là que se lirait un "
                     "stutter. L1 est ÉTALÉE précisément pour qu'aucune "
                     "frame ne concentre la remontée — une p99 très "
                     "au-dessus de la médiane contredirait l'étalement."),
        },
        "mort": {
            "seuil_ms": B_FRAME_MS,
            "declenchee": bool(mediane > B_FRAME_MS),
            "branche_preecrite": (
                "V4 mort ⇒ le repli 33.3 devient LA décision à prendre "
                "(jamais un enchaînement)"),
        },
        "bande_modele": {
            "bande_ms": [bas, haut],
            "hors_bande": bool(not bas <= mediane <= haut),
            "origine": ("RECOMPOSÉE à EPS=1e-2 (§A24) : F 12.655 + "
                        "remontée 1.280 + transferts 0.110, prédiction "
                        "plancher unilatéral ; remplace [14.2, 15.9] qui "
                        "incluait ~2.4 ms de transferts"),
            "branche_preecrite_si_hors_bande": "AUTRE remonté",
        },
        "parts": parts_derivees(mediane, transferts_ms, k),
        "prediction_gpu_side": prediction,
        "l1_round_robin": round_robin,
        "assechement_champs_echelle": assechement,
        "rappel_portee": (
            "M-a-quater REPORTE. La décision — V4 vit, ou le repli 33.3 "
            "devient LA décision — revient à Romain à la lecture. Kernels "
            "F et L3 intouchés."),
    }


def main() -> None:
    cp = exiger_cupy()
    exiger_pass(VERIFS_EXIGEES)

    geo = geometrie_v4()
    prediction = resume_prediction(geo)
    transferts = TransfertComptable(cp)
    # EPS = 1e-2 APPLIQUÉ (§A23) : le régime de production, pas le défaut
    # d'instrument. Passé explicitement pour que la production ne dépende
    # pas d'une valeur par défaut restée à 1e-4 pour les bornes gelées.
    pipeline = PipelineMaQuater(cp, geo, transferts, k=CADENCE_L1,
                                eps=EPS_PRODUCTION)
    round_robin = diagnostic_round_robin(pipeline.plans, CADENCE_L1)
    transferts.frame_suivante()

    print(f"V4 emboîtée : {geo.n_slots_total} slots / {geo.blocs_actifs_gpu} "
          f"blocs ; propriété E = {prediction['propriete_e_satisfaite']} ; "
          f"EPS_PRODUCTION={EPS_PRODUCTION:g} (§A23)", flush=True)
    print(f"prédiction : {prediction['n_slots_gpu']} GPU-side / "
          f"{prediction['n_slots_cpu']} CPU ; L1 k={CADENCE_L1} étalée "
          f"{round_robin['slots_par_tour']} slots/tour", flush=True)

    def frame() -> None:
        pipeline.frame(delta_x=DELTA_X_MOBILE)
        transferts.frame_suivante()

    chrono = chronometrer_frames(cp, frame)

    regime = transferts.stats_regime(exclure_premiers=1 + WARMUP_FRAMES)
    transferts_ms = regime["transfert_total_ms"]["mediane_ms"]
    assechement = pipeline.assechement_champs_echelle()
    lecture = lecture_mecanique(chrono["stats"], transferts_ms, prediction,
                                round_robin, assechement, CADENCE_L1)

    mempool = cp.get_default_memory_pool()
    document = {
        "meta": {
            "mesure": "M-a-quater — pipeline complet V4 (achat B, "
                      "§A18-lecture-borne-L3, pocCascade2phys 7c4a4ca)",
            "candidat": {
                "nom": "V4 emboîtée",
                "n_slots": geo.n_slots_total,
                "n_blocs": geo.blocs_actifs_gpu,
                "cout_predit_f_ms": cout_predit_f_ms(geo),
                "propriete_e": "arbre — zéro tour d'ancêtres",
            },
            "protocole": {
                "geometrie": ("EMBOÎTÉE : énergie aux quarts libres de la "
                              "parente fovéale, offsets ±n_fov/2"),
                "prediction": "GPU-side parent→enfant, niveau 1 seul CPU",
                "champs_echelle": ("systèmes au-delà de c_parent : "
                                   "remplissage-ZÉRO exact (P2)"),
                "second_systeme": ("MOUILLÉ dans le harnais — "
                                   "anti-minoration, −16.9 % mesuré"),
                "remontee": (f"kernel L3 INTOUCHÉ, L1 ÉTALÉE round-robin "
                             f"k={CADENCE_L1}"),
                "eps_production": (
                    f"{EPS_PRODUCTION:g} APPLIQUÉ (§A23, b6c8807) — trois "
                    "portées : innocent≠optimal (balayage non borné par le "
                    "haut) ; CANAL≠lointain (miroir CPU non mesuré) ; "
                    "scopé à ce substrat"),
                "kernels": ("F fusionné et L3 tel que mesuré à la borne — "
                            "INTOUCHÉS (git diff + empreintes _SOURCE / "
                            "_SOURCE_L3)"),
                "chrono": "B6 (cuda-Events, warmup exclu, médiane ET p99)",
            },
            "cellule": {"n_fov": N_FOV, "n_niv": N_NIV},
            "frames": {"warmup_exclu": WARMUP_FRAMES, "serie": SERIE_FRAMES},
            "lancements_par_frame": pipeline.lancements_derniere_frame,
            "verifs_exigees": list(VERIFS_EXIGEES),
        },
        "frame_time": chrono["stats"],
        "transferts_regime": regime,
        "retard_transfert": pipeline.audit_cfl(),
        "residence": {
            "mempool_used_octets": int(mempool.used_bytes()),
            "mempool_total_octets": int(mempool.total_bytes()),
            "etat_prealloue_octets": pipeline.octets_etat(),
            "nvidia_smi_octets": vram_nvidia_smi_octets(),
            "surcout_pingpong_octets": pipeline.octets_surcout_pingpong(),
            "gate_go": GATE_VRAM_GO,
            "note_pingpong": (
                "le correctif §A21 (buffers d'émission en PING-PONG) "
                "double le dimensionnement au pire cas des buffers "
                "d'émission. Le surcoût est reporté À PART, jamais fondu "
                f"dans le total ; le gate reste {GATE_VRAM_GO} Go."),
        },
        "lecture_mecanique": lecture,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON_PATH.write_text(
        json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    liberer_vram(cp)

    parts = lecture["parts"]
    print("=" * 78)
    print("F1 -- M-a-quater : pipeline complet V4 (lecture mécanique)")
    print("=" * 78)
    print(f"  médiane frame COMPLÈTE = "
          f"{lecture['mediane_frame_complete_ms']:.3f} ms")
    print(f"  p99 (EN ÉVIDENCE)      = "
          f"{lecture['p99_frame_complete_ms']:.3f} ms   "
          f"(+{lecture['p99_en_evidence']['ecart_a_la_mediane_ms']:.3f} "
          f"vs médiane)")
    print(f"  MORT   : médiane > {B_FRAME_MS} ms  -> "
          f"{lecture['mort']['declenchee']}")
    print(f"  MODÈLE : bande {BANDE_MS} ms  -> hors bande = "
          f"{lecture['bande_modele']['hors_bande']}")
    print(f"  parts [DÉRIVÉ] : F={parts['f_batche_ms']:.3f}  "
          f"prédiction={parts['prediction_solde_ms']:.3f}  "
          f"remontée={parts['remontee_l3_amortie_ms']:.3f}  "
          f"transferts={parts['transferts_ms']:.3f} ms")
    print(f"  L1 k={CADENCE_L1} : {round_robin['slots_par_tour']} slots/tour ; "
          f"aucune frame ne remonte tout = "
          f"{round_robin['aucune_frame_ne_remonte_tout']}")
    print(f"  assèchement champs d'échelle : "
          f"{assechement['fraction_asséchée']:.1%} "
          f"[anti-minoration, reporté]")
    print("  Le driver REPORTE — la décision revient à Romain.")
    print(f"[REPORT] -> {OUT_JSON_PATH}")
    print("POINT D'ARRÊT OBLIGATOIRE : lecture remontée à Romain, aucun "
          "verdict, aucun enchaînement automatique.")


if __name__ == "__main__":
    main()
