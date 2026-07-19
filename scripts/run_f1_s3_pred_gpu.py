"""F1 — BRAS s3 : prédiction des colonnes entrantes GPU-SIDE
(§A17-lecture-s2-mobile, L4-design, pocCascade2phys 47e0586).

POURQUOI CE BRAS : s2-mobile a mesuré la prédiction batchée à 2.819 ms —
elle MONTE par rapport aux 2.135 per-niveau — et la cible L1+L3 est
sortie NÉGATIVE (−0.551). Mais le harnais prédit TOUTES les colonnes
côté CPU depuis le niveau 0 (choix B4), alors que la spec §5 prévoit la
descente parent→enfant, les parents vivant sur GPU dès le niveau 2. Le
2.819 mesure donc une SIMPLIFICATION DE HARNAIS, majorant de la
topologie spécifiée. s3 mesure la topologie spécifiée.

CHEMIN VIVANT (Option A, gravée) : les parents sont les fenêtres GPU
TELLES QU'ELLES VIVENT — déjà avancées par F. La descente lit donc de
l'état vivant en f32, ce qui est légitime et voulu ; ce n'est pas une
re-prédiction depuis un monde figé.

CE QUI CALCULE LA PRÉDICTION GPU : du CuPy simple — une indexation
avancée `parent[..., iy[:, None], ix[None, :]]` sur la fenêtre du slot
parent, exactement le motif « voisin » (M6) de la prédiction CPU, avec
un facteur 2 au lieu de 2^j. AUCUN kernel n'est écrit pour cela, et le
kernel F reste INTOUCHÉ (bloc _SOURCE à l'empreinte
e8fcaad47db0af010728d5c5d878e1629191bfef4e0382fa80aa3cefd9f27f04).
Le choix du CuPy simple est délibéré : un kernel dédié serait une
escalade d'implémentation non pré-enregistrée, et la sonde doit mesurer
le DESIGN (où vit la prédiction), pas une optimisation de plus.

────────────────────────────────────────────────────────────────────────
FAIT GÉOMÉTRIQUE VÉRIFIÉ AU BUILD — À LIRE AVANT D'INTERPRÉTER s3
────────────────────────────────────────────────────────────────────────
Le pré-enregistrement suppose que tout slot de niveau >= 2 a un parent
GPU. C'est VRAI pour les slots FOVÉAUX, FAUX pour les slots d'ÉNERGIE :
vérifié sur 400 positions du balayage (résultat stable, jamais
intermittent) —
  - les 8 slots fovéaux des niveaux 2..9 ont TOUJOURS un parent couvrant ;
  - les 3 slots d'ÉNERGIE (niveau 8 x1, niveau 9 x2) n'en ont JAMAIS.
    Cause : B3 place les fenêtres d'énergie à ±n_fov EN Y du même niveau ;
    la région parente qu'elles réclament tombe hors de toute fenêtre du
    niveau parent, qui n'a pas de fenêtre à cet endroit. La pyramide V2
    n'est pas « fermée » pour la descente parent→enfant sur l'énergie.
  - de plus, le niveau 8 (c=8, 2 systèmes) a pour parent le niveau 7
    (c=4, 1 système) : le parent ne porte pas les deux systèmes que
    l'enfant réclame.
CE QUE CE BUILD EN FAIT : chaque slot prend le chemin que la topologie
PERMET, jamais un chemin inventé. 7 slots passent GPU-side (les fovéaux
des niveaux 2..7 et 9) ; 5 gardent le chemin CPU+H2D (le niveau 1 —
gravé —, le fovéal du niveau 8 pour cause de systèmes, et les 3 slots
d'énergie). Aucun contenu n'est fabriqué pour combler un parent absent :
ce serait inventer de la donnée, et dupliquer un système pour en remplir
deux minorerait la lecture. Le majorant honnête est retenu.
CONSÉQUENCE POUR LA LECTURE, chiffrée ici pour qu'elle ne surprenne
personne : les slots restés CPU sont ceux des niveaux 8 et 9 — c'est-à-
dire les niveaux qui bougent le PLUS souvent (périodes 2 et 1). Sur une
frame MÉDIANE, seul le niveau 9 bouge : 1 slot y passe GPU-side et 2
restent CPU. La baisse attendue sur la médiane est donc PARTIELLE par
construction. Le JSON reporte `couverture_gpu` (quels slots, pour quelle
raison, et la part GPU d'une frame médiane) — ce driver ne conclut pas :
il rend le fait lisible.
────────────────────────────────────────────────────────────────────────

CE QUI NE CHANGE PAS PAR RAPPORT À s2-mobile : la config V2 (12 slots,
17 blocs), l'orchestration batchée (2 groupes, 4 lancements/frame), la
CFL on-device, la remontée B4 OFF, la fovéa mobile E4c, l'état initial.
Seul le CHEMIN de la prédiction change — c'est ce que la soustraction
isole.

VERROUS (même discipline que s2 / s2-mobile) :
  (a) AUCUNE OMISSION — les colonnes prédites, slot par slot et frame par
      frame, sont exactement celles de `PyramideFovea` ; les écritures
      touchent la fenêtre ET la référence comme elle ; contre-épreuve que
      la fenêtre de test couvre bien tous les niveaux ;
  (b) ÉQUIVALENCE à la prédiction CPU : sur un parent cohérent (celui
      qu'on obtient en descendant depuis le monde 0), la descente
      parent→enfant donne le MÊME bloc que la prédiction directe, à la
      tolérance f32 nommée `TOL_PREDICTION_*`. C'est la propriété que le
      docstring de `pyramide` énonçait déjà : « les octets descendus sont
      identiques à une chaîne niveau-à-niveau » ;
  (c) le niveau 1 paie bien son chemin CPU + H2D (il n'a pas de parent
      GPU, et son trafic doit rester visible).

LECTURE MÉCANIQUE (gravée) : prédiction_gpu = A_s3 − A_batché ;
cible L1+L3 = 16.7 − A_s3. A_batché est LU du JSON de s2 et A_mobile du
JSON de s2-mobile — jamais recopiés ; les deux dépendances sont
fail-loud. Les grandeurs sont REPORTÉES SANS INTERPRÉTATION : la décision
— design jouable, candidat aminci, ou porte 33.3 par la porte de devant —
revient à Romain à la lecture.

Chrono B6, vérifs #1/#2, natif. Sortie JSON SANS timestamp.

Machine : iluin-tworings3, terminal natif UNIQUEMENT. Usage (APRÈS s2 et
s2-mobile) :
  .venv/bin/python scripts/run_f1_s3_pred_gpu.py

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
from scripts.run_f1_s2_mobile import (  # noqa: E402
    DELTA_X_MOBILE,
    cadence_deplacement,
    plan_groupes,
)
from scripts.run_f1_s2_mobile import OUT_JSON_PATH as S2_MOBILE_JSON_PATH  # noqa: E402
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
OUT_JSON_PATH = OUT_DIR / "s3_pred_gpu.json"

# Tolérance de l'équivalence descente-parent vs prédiction directe. La
# prédiction voisin est une SÉLECTION d'échantillons (aucune arithmétique),
# donc l'égalité est en principe EXACTE ; la tolérance est nommée par
# prudence de protocole, et le test exige l'exactitude quand elle tient.
TOL_PREDICTION_RTOL: float = 1e-6
TOL_PREDICTION_ATOL: float = 1e-7

NIVEAU_PARENT_CPU: int = 1      # seul niveau sans parent GPU (gravé §5)

# Raisons de repli sur le chemin CPU (reportées telles quelles).
RAISON_NIVEAU_1: str = "niveau 1 : parent = monde 0 (CPU), gravé §5"
RAISON_PAS_DE_PARENT: str = (
    "aucune fenêtre du niveau parent ne couvre la région requise "
    "(fenêtre d'énergie à ±n_fov : B3 n'en place pas au parent)")
RAISON_SYSTEMES: str = (
    "le parent porte moins de systèmes que l'enfant (c dégressif) — "
    "dupliquer un système pour en remplir deux minorerait la lecture")


def _region_parente(oy: int, ox: int, n_fov: int) -> tuple[int, int, int, int]:
    """Bornes INCLUSIVES (y0, y1, x0, x1) des cellules parentes que la
    fenêtre (oy, ox) réclame — facteur 2 exactement (parent = niveau−1)."""
    return oy // 2, (oy + n_fov - 1) // 2, ox // 2, (ox + n_fov - 1) // 2


def _parent_couvrant(geo: GeometriePyramide, j: int, oy: int, ox: int,
                     origines_parent: list[tuple[int, int]]) -> int | None:
    """Index du slot parent dont la fenêtre contient TOUTE la région
    requise, ou None. Une couverture partielle ne compte pas : lire à
    cheval sur deux fenêtres n'est pas la descente que la spec décrit."""
    y0, y1, x0, x1 = _region_parente(oy, ox, geo.n_fov)
    for index, (poy, pox) in enumerate(origines_parent):
        if (poy <= y0 and y1 < poy + geo.n_fov
                and pox <= x0 and x1 < pox + geo.n_fov):
            return index
    return None


def plan_prediction(geo: GeometriePyramide, centre_fin: int | None = None
                    ) -> dict:
    """Pour chaque slot : chemin GPU (avec son slot parent) ou CPU, avec
    la RAISON du repli. Le plan est stable au cours du balayage (vérifié
    au build sur 400 positions) ; une garde runtime le revérifie à chaque
    prédiction GPU."""
    if centre_fin is None:
        centre_fin = geo.centre_fin_initial()
    plan: dict[int, dict] = {}
    for j in geo.niveaux_gpu:
        origines = geo.origines(j, centre_fin)
        entree: dict = {"gpu": {}, "cpu": [], "raisons": {}}
        if j == NIVEAU_PARENT_CPU:
            entree["cpu"] = list(range(len(origines)))
            entree["raisons"] = {i: RAISON_NIVEAU_1 for i in entree["cpu"]}
            plan[j] = entree
            continue
        origines_parent = geo.origines(j - 1, centre_fin)
        memes_systemes = (geo.n_systemes_du_niveau(j)
                          == geo.n_systemes_du_niveau(j - 1))
        for i, (oy, ox) in enumerate(origines):
            index_parent = _parent_couvrant(geo, j, oy, ox, origines_parent)
            if index_parent is None:
                entree["cpu"].append(i)
                entree["raisons"][i] = RAISON_PAS_DE_PARENT
            elif not memes_systemes:
                entree["cpu"].append(i)
                entree["raisons"][i] = RAISON_SYSTEMES
            else:
                entree["gpu"][i] = index_parent
        plan[j] = entree
    return plan


def resume_couverture(geo: GeometriePyramide) -> dict:
    """Description auditable du plan : quels slots GPU, quels slots CPU et
    pourquoi, plus la part GPU d'une frame MÉDIANE (seuls les niveaux à
    petite période y bougent — cf. `cadence_deplacement`)."""
    plan = plan_prediction(geo)
    gpu = [(j, i) for j, e in plan.items() for i in e["gpu"]]
    cpu = [(j, i) for j, e in plan.items() for i in e["cpu"]]
    periodes = cadence_deplacement(geo)["periode_par_niveau_frames"]
    niveaux_frame_mediane = [j for j in geo.niveaux_gpu
                             if periodes[str(j)] == 1]
    gpu_median = sum(len(plan[j]["gpu"]) for j in niveaux_frame_mediane)
    cpu_median = sum(len(plan[j]["cpu"]) for j in niveaux_frame_mediane)
    return {
        "slots_gpu": [{"niveau": j, "slot": i} for j, i in gpu],
        "slots_cpu": [{"niveau": j, "slot": i,
                       "raison": plan[j]["raisons"][i]} for j, i in cpu],
        "n_slots_gpu": len(gpu),
        "n_slots_cpu": len(cpu),
        "frame_mediane": {
            "niveaux_deplaces": niveaux_frame_mediane,
            "slots_gpu": gpu_median,
            "slots_cpu": cpu_median,
            "note": ("les slots restés CPU sont ceux des niveaux les plus "
                     "fins, qui bougent le plus souvent : la baisse sur la "
                     "MÉDIANE est partielle par construction"),
        },
    }


def predire_colonnes_gpu(parent, cp, geo: GeometriePyramide, oy: int, ox: int,
                         oy_parent: int, ox_parent: int, x_debut: int,
                         x_fin: int):
    """Descente parent→enfant SUR GPU : prédiction voisin (motif M6) avec
    un facteur 2, lue dans la fenêtre du slot parent — CuPy simple,
    aucune écriture de kernel.

    `parent` : (n_systemes, 4, n_fov, n_fov) GPU, l'état VIVANT du parent.
    Retourne (n_systemes, 4, n_fov, x_fin−x_debut).

    Fail-loud si un indice sort de la fenêtre parente : une prédiction
    silencieusement repliée (wrap-around de l'indexation) fabriquerait des
    valeurs fausses sans rien signaler."""
    iy = (oy + cp.arange(geo.n_fov)) // 2 - oy_parent
    ix = (ox + cp.arange(x_debut, x_fin)) // 2 - ox_parent
    if not (0 <= int(iy.min()) and int(iy.max()) < geo.n_fov
            and 0 <= int(ix.min()) and int(ix.max()) < geo.n_fov):
        raise RuntimeError(
            "predire_colonnes_gpu : la région requise sort de la fenêtre "
            f"parente (y[{int(iy.min())},{int(iy.max())}] "
            f"x[{int(ix.min())},{int(ix.max())}] hors [0,{geo.n_fov})) — "
            "le plan de prédiction ne tient plus. Aucune mesure.")
    return parent[..., iy[:, None], ix[None, :]]


class PyramideBatcheePredGpu:
    """Le bras s2-mobile avec la prédiction GPU-side là où la topologie la
    permet. Buffers, groupes, orchestration et F sont ceux de s2-mobile ;
    seul le CHEMIN de la prédiction change."""

    def __init__(self, cp, geo: GeometriePyramide,
                 transferts: TransfertComptable,
                 graine: int = GRAINE_MONDE,
                 pas_f=pas_f_fusionne_async):
        self.cp = cp
        self.geo = geo
        self.transferts = transferts
        self.pas_f = pas_f
        self.plans = plan_groupes(geo)
        self.plan_prediction = plan_prediction(geo)
        self.monde0 = monde0_jetable(geo, graine)
        self.centre_fin = geo.centre_fin_initial()
        self.fenetres: list = []
        self.tampons: list = []
        self.references: list = []
        for plan in self.plans:
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
        self._localisation = self._construire_localisation()
        self.dt_cfl: list = [None] * len(self.plans)

    def _construire_localisation(self) -> dict[tuple[int, int], tuple[int, int]]:
        """(niveau, slot) -> (indice de groupe, index dans le groupe)."""
        localisation: dict[tuple[int, int], tuple[int, int]] = {}
        for indice, plan in enumerate(self.plans):
            for j, (debut, _) in plan["tranches"].items():
                for i in range(self.geo.n_slots(j)):
                    localisation[(j, i)] = (indice, debut + i)
        return localisation

    def _vue_slot(self, buffers: list, j: int, i: int):
        indice, position = self._localisation[(j, i)]
        return buffers[indice][position]

    def frame(self, delta_x: int = DELTA_X_MOBILE) -> dict:
        """Une frame : déplacement (prédiction GPU-side ou CPU selon le
        plan, puis roll et écritures fenêtre ET référence), puis F batché
        en 4 lancements. Pas de remontée."""
        cp = self.cp
        geo = self.geo
        if delta_x < 0:
            raise ValueError(f"frame : delta_x={delta_x} < 0 (balayage +x).")
        self.centre_fin += delta_x
        niveaux_deplaces: list[int] = []
        colonnes_gpu = 0
        colonnes_cpu = 0

        for j in geo.niveaux_gpu:
            nouvelles = geo.origines(j, self.centre_fin)
            dx = nouvelles[0][1] - self._origines[j][0][1]
            if dx < 0:
                raise RuntimeError(
                    f"frame : recul d'origine au niveau {j} (dx={dx}) — "
                    "balayage E4c strictement +x.")
            if dx > 0:
                niveaux_deplaces.append(j)
                plein = dx >= geo.n_fov
                x_debut = 0 if plein else geo.n_fov - dx
                entree = self.plan_prediction[j]

                # 1. roll de TOUTES les fenêtres du niveau (fenêtre ET
                #    référence) — comme la pyramide, hors remontée.
                if not plein:
                    for i in range(geo.n_slots(j)):
                        for buffers in (self.fenetres, self.references):
                            vue = self._vue_slot(buffers, j, i)
                            vue[...] = cp.roll(vue, -dx, axis=-1)

                # 2. slots GPU-side : descente parent -> enfant.
                for i, index_parent in entree["gpu"].items():
                    oy, ox = nouvelles[i]
                    oy_p, ox_p = geo.origines(j - 1, self.centre_fin)[
                        index_parent]
                    parent = self._vue_slot(self.fenetres, j - 1,
                                            index_parent)
                    entrant = predire_colonnes_gpu(
                        parent, cp, geo, oy, ox, oy_p, ox_p,
                        x_debut, geo.n_fov)
                    for buffers in (self.fenetres, self.references):
                        vue = self._vue_slot(buffers, j, i)
                        vue[..., x_debut:] = entrant
                    colonnes_gpu += geo.n_fov - x_debut

                # 3. slots CPU : prédiction depuis le monde 0 + H2D,
                #    batchée sur les slots CPU du niveau (contigus).
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

        for indice, (fen, tampon) in enumerate(
                zip(self.fenetres, self.tampons)):
            _, dt_cfl = self.pas_f(fen, self.cp, fen, tampon)
            self.dt_cfl[indice] = dt_cfl

        return {"niveaux_deplaces": sorted(niveaux_deplaces),
                "colonnes_gpu": colonnes_gpu, "colonnes_cpu": colonnes_cpu,
                "remontee_active": False}

    def audit_cfl(self) -> dict:
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


def _lire_mesure(chemin: Path, cle: str, quoi: str, script: str) -> dict:
    """Lecture fail-loud d'une mesure antérieure — jamais recopiée."""
    if not chemin.exists():
        raise RuntimeError(
            f"le bras s3 exige {quoi}, absent ({chemin}). Lancer d'abord "
            f"`.venv/bin/python scripts/{script}` — recopier le chiffre à "
            "la main serait non traçable. Aucune mesure ici.")
    document = json.loads(chemin.read_text(encoding="utf-8"))
    try:
        valeur = float(document["lecture_mecanique"][cle])
    except (KeyError, TypeError, ValueError) as erreur:
        raise RuntimeError(
            f"bras s3 : JSON illisible ou incomplet ({chemin}) — "
            f"{erreur!r}. Aucune mesure.") from erreur
    if not np.isfinite(valeur) or valeur <= 0.0:
        raise RuntimeError(
            f"bras s3 : {quoi} non exploitable ({valeur}). Aucune mesure.")
    return {"valeur_ms": valeur, "source": str(chemin)}


def lire_a_batche(chemin: Path = S2_JSON_PATH) -> dict:
    return _lire_mesure(chemin, "a_batche_ms", "A_batché (mesuré par s2)",
                        "run_f1_s2_batche.py")


def lire_a_mobile(chemin: Path = S2_MOBILE_JSON_PATH) -> dict:
    return _lire_mesure(chemin, "a_mobile_ms",
                        "A_mobile (mesuré par s2-mobile)",
                        "run_f1_s2_mobile.py")


def lecture_mecanique(a_s3_ms: float, a_batche: dict, a_mobile: dict,
                      couverture: dict) -> dict:
    """Consommation MÉCANIQUE de la lecture gravée — REPORTÉE, jamais
    interprétée."""
    a_batche_ms = a_batche["valeur_ms"]
    a_mobile_ms = a_mobile["valeur_ms"]
    prediction_gpu = a_s3_ms - a_batche_ms
    prediction_cpu = a_mobile_ms - a_batche_ms
    return {
        "a_s3_ms": a_s3_ms,
        "a_batche_ms": a_batche_ms,
        "a_batche_source": a_batche["source"],
        "a_mobile_ms": a_mobile_ms,
        "a_mobile_source": a_mobile["source"],
        "prediction_gpu": {
            "formule": "A_s3 - A_batché",
            "valeur_ms": prediction_gpu,
            "prediction_cpu_side_ms": prediction_cpu,
            "note": ("la prédiction CPU-side de s2-mobile est reportée "
                     "pour situer, PAS pour conclure : ce driver ne "
                     "compare ni ne commente. Sa lecture doit tenir compte "
                     "de `couverture_gpu` — une part des slots reste "
                     "CPU faute de parent couvrant"),
        },
        "cible_l1_l3": {
            "formule": f"{B_FRAME_MS} - A_s3",
            "valeur_ms": B_FRAME_MS - a_s3_ms,
            "seuil_t2_ms": B_FRAME_MS,
            "note": ("REPORTÉE sans interprétation. Ce qu'elle implique — "
                     "design jouable, candidat aminci, ou porte 33.3 par "
                     "la porte de devant — est une décision de Romain à la "
                     "lecture, pas une sortie de ce driver. Une cible "
                     "nulle ou négative se lit telle quelle."),
        },
        "couverture_gpu": couverture,
        "rappel_portee": (
            "bras de sonde (L4-design) : il MESURE la prédiction GPU-side "
            "là où la topologie la permet. Aucun verdict, aucun levier "
            "tranché, aucune escalade — le kernel F est intouché."),
    }


def main() -> None:
    cp = exiger_cupy()
    exiger_pass(VERIFS_EXIGEES)
    a_batche = lire_a_batche()        # fail-loud AVANT toute allocation
    a_mobile = lire_a_mobile()

    print(f"A_batché lu de s2 : {a_batche['valeur_ms']:.3f} ms ; "
          f"A_mobile lu de s2-mobile : {a_mobile['valeur_ms']:.3f} ms",
          flush=True)
    geo = GeometriePyramide(n_fov=N_FOV, n_niv=N_NIV, slots=slots_v2())
    couverture = resume_couverture(geo)
    print(f"prédiction GPU-side : {couverture['n_slots_gpu']} slots ; "
          f"CPU : {couverture['n_slots_cpu']} "
          f"(frame médiane : {couverture['frame_mediane']['slots_gpu']} GPU / "
          f"{couverture['frame_mediane']['slots_cpu']} CPU)", flush=True)

    transferts = TransfertComptable(cp)
    pyramide = PyramideBatcheePredGpu(cp, geo, transferts)
    transferts.frame_suivante()

    def frame() -> None:
        pyramide.frame(delta_x=DELTA_X_MOBILE)
        transferts.frame_suivante()

    chrono = chronometrer_frames(cp, frame)
    audit = pyramide.audit_cfl()

    mempool = cp.get_default_memory_pool()
    a_s3_ms = chrono["stats"]["mediane_ms"]
    lecture = lecture_mecanique(a_s3_ms, a_batche, a_mobile, couverture)

    document = {
        "meta": {
            "mesure": "bras s3 — prédiction GPU-side (§A17-lecture-"
                      "s2-mobile, L4-design, pocCascade2phys 47e0586)",
            "protocole": {
                "config": "V2 EXACTE (12 slots, 17 blocs) — slots importés",
                "base": "s2-mobile (batché, CFL on-device, remontée OFF, "
                        "fovéa mobile E4c) — seul le CHEMIN de la "
                        "prédiction change",
                "prediction_gpu": ("descente parent->enfant en CuPy simple "
                                   "(indexation avancée, motif voisin M6, "
                                   "facteur 2) — aucun kernel écrit"),
                "chemin_vivant": "Option A : les parents sont l'état GPU "
                                 "vivant, f32, déjà avancé par F",
                "kernel_f": "INTOUCHÉ (empreinte e8fcaad4...7f04)",
                "chrono": "B6 (cuda-Events, warmup exclu, médiane ET p99)",
            },
            "cellule": {"n_fov": N_FOV, "n_niv": N_NIV},
            "groupes": [{"n_systemes": p["n_systemes"],
                         "n_slots": p["n_slots"], "niveaux": p["niveaux"]}
                        for p in pyramide.plans],
            "lancements_par_frame": LANCEMENTS_PAR_FRAME,
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

    prediction = lecture["prediction_gpu"]
    cible = lecture["cible_l1_l3"]
    print("=" * 78)
    print("F1 -- bras s3 : prédiction GPU-side (lecture mécanique)")
    print("=" * 78)
    print(f"  A_s3      = {a_s3_ms:.3f} ms   "
          f"p99 = {chrono['stats']['p99_ms']:.3f} ms")
    print(f"  A_batché  = {lecture['a_batche_ms']:.3f} ms  (lu de s2)")
    print(f"  prédiction GPU-side = {prediction['valeur_ms']:.3f} ms   "
          f"[CPU-side, s2-mobile : "
          f"{prediction['prediction_cpu_side_ms']:.3f} ms]")
    print(f"  cible L1+L3         = {cible['valeur_ms']:.3f} ms   "
          f"(= {B_FRAME_MS} - A_s3)")
    mediane = couverture["frame_mediane"]
    print(f"  couverture : {couverture['n_slots_gpu']} slots GPU / "
          f"{couverture['n_slots_cpu']} CPU ; frame médiane "
          f"{mediane['slots_gpu']} GPU / {mediane['slots_cpu']} CPU")
    print(f"  audit CFL : {audit['dt_cfl_par_groupe']}  "
          f"finies>0 = {audit['toutes_finies_positives']}")
    print("  Les grandeurs sont REPORTÉES SANS INTERPRÉTATION — la "
          "décision revient à Romain à la lecture.")
    print(f"[REPORT] -> {OUT_JSON_PATH}")
    print("POINT D'ARRÊT OBLIGATOIRE : lecture remontée à Romain, aucun "
          "verdict, aucun enchaînement automatique.")


if __name__ == "__main__":
    main()
