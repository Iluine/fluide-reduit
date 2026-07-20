"""F1 — SONDE EPS (N1), VERSION 2 : PRÉ-ENREGISTREMENT NEUF, à endosser
AVANT tout run (§A21-complément, pocCascade2phys f943e9a).

**L'ANCIEN PRÉ-ENREGISTREMENT EST SUPERSEDED (§A19-CORRECTION).** Il
n'est pas amendé, il est remplacé : son observable était faux. Le
reconstructeur entretenait une copie CPU indépendante qui n'appliquait ni
le roll ni les colonnes prédites que le device écrit dans `references` à
chaque déplacement de fovéa. La sonde comparait donc des régions
spatialement DÉCALÉES. Le plancher 0.082, insensible à EPS comme à k et
resté inexpliqué, était CELA — pas une infidélité du grossier. Aucune
lecture de la v1 n'est reconduite.

────────────────────────────────────────────────────────────────────────
CE QUE LA SONDE COMPARE, MAINTENANT
────────────────────────────────────────────────────────────────────────
  - **VÉRITÉ** : niveau 0 par DÉCIMATION EXACTE (moyennes Harten,
    `multiresolution.downsample`) de l'état fin courant ;
  - **CONNAISSANCE DU CPU** : LUE sur `reference` (option 1, tranchée par
    Romain). Le schéma B4 pose que `reference` EST le modèle de ce que le
    CPU sait ; depuis le ping-pong (§A21) tout ce qui est émis est
    transféré une fois et une seule, donc l'identité est EXACTE à une
    frame près. Une copie indépendante peut diverger du device — une
    lecture directe ne le peut pas. C'est tout l'objet du correctif ;
  - **OBSERVABLE** : Δχ readout — `albedo` au point d'opération S_HALF_OP
    puis `delta_chi(...)["max_carrier"]`, primitives EXISTANTES, importées
    et non réécrites. Espace instrument, JAMAIS l'état (règle 2).

L'INSTANT DE LECTURE EST GRAVÉ, parce qu'il est la cause du désastre
précédent : la connaissance est capturée APRÈS le déplacement et AVANT
l'émission de la frame. C'est le SEUL instant où `reference` porte la
connaissance du CPU dans les coordonnées de la frame courante.

────────────────────────────────────────────────────────────────────────
LES DEUX VÉRITÉS, DÉCLARÉES AVANT LE RUN
────────────────────────────────────────────────────────────────────────
Consigne gravée : à la frame n le CPU connaît l'état de n−1. Contre
quelle vérité comparer doit donc être DIT d'avance, sous peine de
refabriquer un artefact. Les deux sont déclarées, et TOUTES DEUX
reportées — le surcoût est marginal (un champ capturé de plus, une
évaluation de Δχ de plus, sur la même trajectoire) :

  - **vérité(n)** — K(n) contre S(n), l'état MAINTENANT. Canal ET
    péremption acceptée. **ELLE PORTE LA RÈGLE Q2.** Justification : la
    règle décide d'INNOCUITÉ PERCEPTUELLE, et le joueur ne perçoit pas un
    canal — il perçoit un écart à l'instant présent. Lire la règle sur la
    vérité transportée reviendrait à s'accorder gratuitement la frame de
    retard que le joueur, lui, subit ;
  - **vérité(n−1)** — K(n) contre S̃(n−1), l'état d'où la connaissance
    provient, transporté dans les mêmes coordonnées. Isole la fidélité du
    CANAL SEUL. **DIAGNOSTIC, jamais la règle.**

Leur ÉCART est le prix de la frame de retard, reporté comme tel : en cas
d'échec il dit s'il faut incriminer le SEUIL ou le RETARD, et cette
distinction est précisément ce que l'attribution (a) — nommée, non armée
— demanderait.

────────────────────────────────────────────────────────────────────────
PORTÉE, ET CE QUE LA SONDE NE MESURE PAS
────────────────────────────────────────────────────────────────────────
Le device inscrit dans `reference` les colonnes qu'il a PRÉDITES
GPU-side. Les lire CRÉDITE donc le CPU d'une prédiction identique à celle
du device. **La sonde mesure la fidélité du CANAL — seuil, cadence,
transfert — et NON celle de la prédiction propre au CPU.** C'est une
limite déclarée, pas un détail : mesurer la seconde exige le MIROIR CPU
(option 2), nommé comme travail de PRODUCTION et délibérément non
construit. Un résultat de cette sonde ne peut donc jamais s'énoncer « le
lointain est fidèle », seulement « le canal l'est ».

────────────────────────────────────────────────────────────────────────
RÈGLE DE DÉCISION (Q2), RECONDUITE POUR L'OBSERVABLE CORRIGÉ
────────────────────────────────────────────────────────────────────────
**EPS retenu = le PLUS GRAND EPS dont le Δχ MAX de série, lu sur
vérité(n) et à l'échelle DÉCIMÉE du niveau 0, reste < 0.0603 (ic_bas)** —
lecture sur l'IC entier, discipline §A13. Si aucun EPS du balayage ne
satisfait, 1e-5 compris : **AUTRE remonté, aucun EPS retenu par défaut.**
Jamais de choix après courbe.

ÉCHELLE ÉPINGLÉE : la règle lit le Δχ DÉCIMÉ — le grossier comparé à ce
que le grossier devrait être, à sa propre résolution. Le non décimé est
reporté en DIAGNOSTIC SEUL. La relation n'est pas monotone : décimer
DÉPLACE les bandes porteuses que `delta_chi` est seul à lire.

LOGIQUE DU CRITÈRE, inchangée et toujours un fait de logique : le joueur
ne voit JAMAIS la référence. Δχ < 0.0603 ⇒ **innocuité ÉTABLIE** ;
Δχ > 0.0603 ⇒ **innocuité NON ÉTABLIE** — et JAMAIS « nocivité établie ».

────────────────────────────────────────────────────────────────────────
VÉRIFICATION D'INSTRUMENT ET TABLE DE BRANCHES, RECONDUITES
────────────────────────────────────────────────────────────────────────
Le même balayage est d'abord joué à **k=1**, et `lecture_mecanique`
REFUSE de produire la lecture k=4 sans son PASS/FAIL enregistré (câblage
B9). Une sonde muette se détecte AVANT, pas après (§A14). La table reste
COMPLÈTE et CLOSE :
  **(i)** k=1 discrimine ET au moins un EPS passe ⇒ **INSTRUMENT VALIDE** ;
  **(ii)** k=1 ne discrimine pas ⇒ **SONDE MUETTE sur EPS** — ne rien
      régler, remonter ;
  **(iii)** k=1 DISCRIMINE mais AUCUN EPS ne passe ⇒ **MÉCANISME DE
      REMONTÉE EN QUESTION**, avec ses deux attributions nommées et non
      armées.
NOTA : à k=1 subsiste la frame de retard du compteur (choix 6) — la
vérification teste la discrimination sous péremption MINIMALE, PAS NULLE.

**CE N'EST PAS UN BALAYAGE DE k** : deux valeurs seulement, k=4 (mesure,
figée) et k=1 (vérification). Toute DÉCISION sur k reste gatée σ_ω (R4).

────────────────────────────────────────────────────────────────────────
DEUX PASSES, ET CE QUI A ALLÉGÉ LA SECONDE
────────────────────────────────────────────────────────────────────────
La passe CHRONO tourne le pipeline nu (B6 : warmup exclu, série, médiane
ET p99). La passe Δχ rejoue la MÊME trajectoire hors de toute mesure de
temps. Les deux ne partagent aucun chiffre.

L'option 1 SUPPRIME la capture de coefficients dont la v1 avait besoin :
l'observable se lit sur le device. La passe Δχ est donc plus légère — et
c'est cohérent avec la lecture D-2, qui attribuait +0.822 ms à l'appareil
de sonde.

────────────────────────────────────────────────────────────────────────
CE QUI EST FIGÉ TANT QUE RIEN N'EST ENDOSSÉ
────────────────────────────────────────────────────────────────────────
EPS reste 1e-4, k reste 4, **rien n'est réglé**. Kernels F et L3
INTOUCHÉS dans leur code CUDA ; les deux empreintes, sur la CHAÎNE
EXTRAITE et non sur les fichiers, sont re-calculables :
    sha256(substrat_fusionne._SOURCE)  = e18015f5...f30b4  (8223 car.)
    sha256(substrat_l3._SOURCE_L3)     = 9533a130...781a   (9635 car.)

Machine : iluin-tworings3, terminal natif UNIQUEMENT. Usage :
  .venv/bin/python scripts/run_f1_sonde_eps.py

**POINT D'ARRÊT : ce pré-enregistrement est remonté pour ENDOSSEMENT.
AUCUN run avant. N2 (M-b) n'est pas achetée.**"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_f1_ma_quater import (  # noqa: E402
    CADENCE_L1,
    PipelineMaQuater,
    geometrie_v4,
)
from scripts.run_f1_ma_ter import VERIFS_EXIGEES, liberer_vram  # noqa: E402
from scripts.run_f1_s2_mobile import DELTA_X_MOBILE  # noqa: E402
from scripts.run_arcA_revalidate import S_HALF_OP  # noqa: E402
from src.albedo import albedo, delta_chi  # noqa: E402
from src.f1_gpu.backend import exiger_cupy, vers_cpu  # noqa: E402
from src.f1_gpu.chrono import (  # noqa: E402
    SERIE_FRAMES,
    WARMUP_FRAMES,
    chronometrer_frames,
)
from src.f1_gpu.pyramide import N0_DEFAUT  # noqa: E402
from src.f1_gpu.transferts import TransfertComptable  # noqa: E402
from src.f1_gpu.verifs import exiger_pass, vram_nvidia_smi_octets  # noqa: E402
from src.multiresolution import downsample  # noqa: E402

OUT_DIR = ROOT / "outputs" / "f1"
OUT_JSON_PATH = OUT_DIR / "sonde_eps.json"

# Balayage GRAVÉ (§A19). 1e-4 est la valeur en vigueur, [NON-ANCRÉ].
EPS_BALAYAGE: tuple[float, ...] = (1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
EPS_EN_VIGUEUR: float = 1e-4

# Seuil de la règle Q2 : borne BASSE de l'IC du JND (discipline §A13).
SEUIL_IC_BAS: float = 0.0603

# k de MESURE (§A19, figé) et k de VÉRIFICATION D'INSTRUMENT (§A19-complément
# A). DEUX valeurs, et deux seulement : ce n'est PAS un balayage de k, et ce
# driver ne prononce RIEN sur le cadencement — toute décision sur k reste
# gatée σ_ω (R4).
CADENCE_MESURE: int = CADENCE_L1          # 4
CADENCE_VERIFICATION: int = 1
CADENCES_ADMISES: tuple[int, ...] = (CADENCE_VERIFICATION, CADENCE_MESURE)
CADENCE_FIGEE: int = CADENCE_MESURE        # nom historique, conservé

CHAMP_SEDIMENT: int = 3          # (h, hu, hv, s) — s porte le readout

# Le pré-enregistrement v2 est REMONTÉ, pas encore endossé. Tant que ce
# drapeau est faux, le driver REFUSE de tourner : « aucun run avant
# endossement » devient mécanique au lieu d'être une promesse. C'est
# Romain qui le lève, avec le commit qui endosse.
ENDOSSEMENT_PREREG_V2: bool = False


def exiger_endossement() -> None:
    """Verrou de procédure : un pré-enregistrement remonté n'est pas un
    pré-enregistrement endossé. Mesurer avant endossement laisserait la
    règle se former après la donnée."""
    if not ENDOSSEMENT_PREREG_V2:
        raise RuntimeError(
            "SONDE EPS v2 : pré-enregistrement REMONTÉ, NON ENDOSSÉ "
            "(§A21-complément). Aucun run n'est produit avant endossement "
            "— l'ancien pré-enregistrement est superseded et la règle ne "
            "doit pas pouvoir se former après la donnée. Lever "
            "ENDOSSEMENT_PREREG_V2 est une décision de Romain.")


def exiger_cadence_admise(k: int) -> None:
    """Garde d'intégrité : SEULES deux cadences existent ici — celle de
    la mesure (4, figée) et celle de la VÉRIFICATION D'INSTRUMENT (1).
    Toute autre valeur serait un balayage de k, c'est-à-dire décider le
    cadencement avec la question perceptuelle en main : la condition de
    réveil σ_ω (R4) serait ATTEINTE. Refusé."""
    if k not in CADENCES_ADMISES:
        raise RuntimeError(
            f"sonde EPS : k={k} hors des cadences admises "
            f"{CADENCES_ADMISES} — {CADENCE_MESURE} pour la MESURE, "
            f"{CADENCE_VERIFICATION} pour la VÉRIFICATION D'INSTRUMENT. "
            "Toute autre valeur serait un balayage de k, donc une "
            "décision de cadencement prise avec la question perceptuelle "
            "en main : réveil σ_ω (R4) ATTEINT. Bouger k est une décision "
            "explicite de Romain, jamais un effet de bord de sonde.")


def exiger_cadence_figee(k: int) -> None:
    """Alias historique — la cadence de MESURE, elle, reste figée à 4."""
    if k != CADENCE_MESURE:
        raise RuntimeError(
            f"sonde EPS : k={k} != {CADENCE_MESURE}. Le balayage de k est "
            "INTERDIT (§A19) pour la MESURE ; seule la vérification "
            f"d'instrument emploie k={CADENCE_VERIFICATION} "
            "(§A19-complément A).")


class ReconstructeurNiveau0:
    """**SUPERSEDED (§A19-CORRECTION). NE PAS EMPLOYER POUR UNE LECTURE
    NEUVE.** Conservé parce que des lectures ACQUISES en dépendent
    (`run_f1_diagnostic_instrument.py`) et qu'on ne réécrit pas le passé.

    Ce reconstructeur entretenait une copie CPU INDÉPENDANTE, alimentée
    par les seuls coefficients reçus. Il ne faisait NI le roll NI les
    colonnes prédites que le device applique à `references` à chaque
    déplacement de fovéa. Il comparait donc des régions spatiales
    DÉCALÉES, et c'est ce décalage — non une infidélité du grossier — qui
    produisait le plancher 0.082, insensible à EPS comme à k.

    Il est remplacé par `ConnaissanceCpu` (option 1, §A21-complément) :
    une copie indépendante peut diverger du device, une lecture directe
    ne le peut pas."""

    def __init__(self, pipeline: PipelineMaQuater):
        # Le CPU part de ce qu'il a descendu : les références initiales.
        self.vivant = [vers_cpu(reference).astype(np.float64)
                       for reference in pipeline.references]

    def appliquer(self, coefficients: list) -> None:
        """Applique les coefficients d'une frame (indices aplatis dans le
        groupe, valeurs f32) — l'incrément du schéma B4."""
        for indice_groupe, valeurs, indices in coefficients:
            if valeurs.size:
                cible = self.vivant[indice_groupe]
                np.add.at(cible.reshape(-1), indices, valeurs)

    def champ(self, indice_groupe: int, position: int) -> np.ndarray:
        """Champ sédiment d'un slot, vu par le CPU."""
        return self.vivant[indice_groupe][position, 0, CHAMP_SEDIMENT]


# ----- OPTION 1 : la connaissance du CPU se LIT, elle ne se refait pas -----

VERITE_COURANTE: str = "verite_n"                 # porte la RÈGLE Q2
VERITE_TRANSPORTEE: str = "verite_n_moins_1"      # DIAGNOSTIC de canal


class ConnaissanceCpu:
    """Ce que le CPU sait, LU SUR `reference` — option 1, tranchée §A21.

    Le schéma B4 pose que `reference` EST le modèle de la connaissance du
    CPU : le kernel émet d = etat − reference, transmet d, puis fait
    reference += d. Depuis le ping-pong (§A21), tout ce qui est émis est
    transféré une fois et une seule ; l'identité est donc EXACTE, à un
    décalage d'une frame près. La lire plutôt que la refaire supprime par
    construction toute possibilité de divergence — c'est là tout l'objet
    du correctif.

    S'emploie comme OBSERVATEUR de `PipelineMaQuater.frame` : appelée
    après le déplacement et avant l'émission, elle capture au vol les
    DEUX champs dont la sonde a besoin, dans les MÊMES coordonnées :

      - `connaissance` = K(n) : `reference` transportée dans les
        coordonnées de la frame n, avant que l'émission de n n'y entre.
        C'est ce que le CPU détient à la frame n ;
      - `verite_transportee` = S̃(n−1) : l'état de la frame n−1, roulé et
        complété par les colonnes prédites, donc lui aussi dans les
        coordonnées de n.

    La vérité COURANTE S(n) se lit après la frame, par `champ_verite`.

    PORTÉE DÉCLARÉE, ET C'EST UNE LIMITE, PAS UN DÉTAIL : le device
    inscrit dans `reference` les colonnes qu'il a PRÉDITES GPU-side. La
    lire, c'est donc CRÉDITER le CPU d'une prédiction identique à celle du
    device. La sonde mesure ainsi la fidélité du CANAL — seuil, cadence,
    transfert — et NON celle de la prédiction propre au CPU. Mesurer
    cette dernière exige le miroir CPU (option 2), NOMMÉ comme travail de
    PRODUCTION et délibérément non construit ici."""

    def __init__(self, indice_groupe: int, position: int):
        self.indice_groupe = indice_groupe
        self.position = position
        self.connaissance: np.ndarray | None = None
        self.verite_transportee: np.ndarray | None = None

    def _lire(self, buffers: list) -> np.ndarray:
        bloc = buffers[self.indice_groupe][self.position, 0, CHAMP_SEDIMENT]
        return vers_cpu(bloc).astype(np.float64)

    def __call__(self, pipeline: PipelineMaQuater) -> None:
        """Point d'arrêt : après le déplacement, avant l'émission."""
        self.connaissance = self._lire(pipeline.references)
        self.verite_transportee = self._lire(pipeline.fenetres)


def champ_verite(pipeline: PipelineMaQuater, indice_groupe: int,
                 position: int) -> np.ndarray:
    """Champ sédiment tel qu'il EST sur le device (la vérité)."""
    bloc = pipeline.fenetres[indice_groupe][position, 0, CHAMP_SEDIMENT]
    return vers_cpu(bloc).astype(np.float64)


def delta_chi_readout(vivant: np.ndarray, verite: np.ndarray,
                      facteur: int) -> float:
    """Δχ readout entre vivant et vérité, décimés d'un facteur commun.

    `albedo` puis `delta_chi(...)["max_carrier"]` : les primitives du
    dépôt, importées telles quelles. Espace INSTRUMENT — aucune
    comparaison sur l'état."""
    if facteur > 1:
        vivant = downsample(vivant, facteur)
        verite = downsample(verite, facteur)
    a_vivant = albedo(np.maximum(vivant, 0.0), S_HALF_OP)
    a_verite = albedo(np.maximum(verite, 0.0), S_HALF_OP)
    return float(delta_chi(a_vivant, a_verite)["max_carrier"])


def _slot_observe(pipeline: PipelineMaQuater) -> tuple[int, int]:
    """Le slot FOVÉAL du niveau le plus fin — celui que le regard porte,
    et le plus exigeant : c'est là que la perte de détail se verrait."""
    return pipeline._localisation[(pipeline.geo.niveau_fin, 0)]


def mesurer_chrono(cp, eps: float, k: int) -> dict:
    """Passe CHRONO : le pipeline nu, aucun rapatriement d'observable."""
    geo = geometrie_v4()
    transferts = TransfertComptable(cp)
    pipeline = PipelineMaQuater(cp, geo, transferts, k=k, eps=eps)
    transferts.frame_suivante()

    def frame() -> None:
        pipeline.frame(delta_x=DELTA_X_MOBILE)
        transferts.frame_suivante()

    chrono = chronometrer_frames(cp, frame)
    regime = transferts.stats_regime(exclure_premiers=1 + WARMUP_FRAMES)
    return {
        "frame_time": chrono["stats"],
        "octets_par_frame_median": (regime["h2d_octets"]["mediane"]
                                    + regime["d2h_octets"]["mediane"]),
        "temps_transfert_median_ms": regime["transfert_total_ms"][
            "mediane_ms"],
        "transferts_regime": regime,
    }


def mesurer_delta_chi(cp, eps: float, k: int, frames: int,
                      facteur_decimation: int) -> dict:
    """Passe Δχ : même trajectoire, observable évalué HORS chrono, et la
    connaissance du CPU LUE sur `reference` (option 1, §A21-complément).

    LES DEUX VÉRITÉS SONT DÉCLARÉES ICI, AVANT LE RUN, et toutes deux
    reportées — leur coût relatif est marginal (un champ de plus capturé
    par frame, une évaluation de Δχ de plus) et les CONFONDRE
    refabriquerait l'artefact :

      - **vérité(n)** — K(n) contre S(n), l'état MAINTENANT. C'est ce que
        le joueur voit : canal ET péremption acceptée. **Elle porte la
        RÈGLE Q2** parce que la règle décide d'innocuité perceptuelle, et
        que le joueur ne perçoit pas un canal, il perçoit un écart à
        l'instant présent ;
      - **vérité(n−1)** — K(n) contre S̃(n−1), l'état d'où la connaissance
        provient, transporté dans les mêmes coordonnées. Elle isole la
        fidélité du CANAL SEUL (seuil, cadence, transfert), sans la frame
        de retard. **Diagnostic**, jamais la règle.

    L'écart entre les deux EST le prix de la péremption d'une frame ; il
    est reporté comme tel, ce qui rend lisible, en cas d'échec, s'il faut
    incriminer le seuil ou le retard.

    Aucune capture de coefficients n'est nécessaire : l'observable se lit
    sur le device. La passe est donc plus légère que la précédente."""
    geo = geometrie_v4()
    transferts = TransfertComptable(cp)
    pipeline = PipelineMaQuater(cp, geo, transferts, k=k, eps=eps)
    indice_groupe, position = _slot_observe(pipeline)
    observateur = ConnaissanceCpu(indice_groupe, position)

    series: dict[str, dict[str, list[float]]] = {
        verite: {"decime": [], "plein": []}
        for verite in (VERITE_COURANTE, VERITE_TRANSPORTEE)}

    for _ in range(frames):
        pipeline.frame(delta_x=DELTA_X_MOBILE, observateur=observateur)
        connaissance = observateur.connaissance
        verites = {
            VERITE_COURANTE: champ_verite(pipeline, indice_groupe, position),
            VERITE_TRANSPORTEE: observateur.verite_transportee,
        }
        for nom, verite in verites.items():
            series[nom]["decime"].append(
                delta_chi_readout(connaissance, verite, facteur_decimation))
            series[nom]["plein"].append(
                delta_chi_readout(connaissance, verite, 1))

    def resume(valeurs: dict) -> dict:
        return {
            "delta_chi_max": float(np.max(valeurs["decime"])),
            "delta_chi_median": float(np.percentile(valeurs["decime"], 50)),
            "delta_chi_max_sans_decimation": float(np.max(valeurs["plein"])),
            "delta_chi_median_sans_decimation": float(
                np.percentile(valeurs["plein"], 50)),
        }

    courante = resume(series[VERITE_COURANTE])
    transportee = resume(series[VERITE_TRANSPORTEE])
    return {
        "frames_evaluees": frames,
        "facteur_decimation": facteur_decimation,
        # La RÈGLE lit ici, et seulement ici.
        **courante,
        "verite_de_la_regle": VERITE_COURANTE,
        "par_verite": {VERITE_COURANTE: courante,
                       VERITE_TRANSPORTEE: transportee},
        "prix_peremption_une_frame": (courante["delta_chi_max"]
                                      - transportee["delta_chi_max"]),
        "note_verites": (
            f"{VERITE_COURANTE} (K(n) contre S(n)) porte la RÈGLE : c'est "
            "ce que le JOUEUR voit — canal ET péremption acceptée. "
            f"{VERITE_TRANSPORTEE} (K(n) contre S̃(n−1)) isole le CANAL "
            "seul et reste un DIAGNOSTIC. Leur écart est le prix de la "
            "frame de retard ; il dit, en cas d'échec, s'il faut "
            "incriminer le seuil ou le retard. Les confondre "
            "refabriquerait l'artefact du décalage spatial."),
        "note": ("le décimé porte la RÈGLE (le niveau 0 est l'observable "
                 "du joueur hors fovéa) ; le non-décimé est reporté à "
                 "côté. Aucun des deux n'est un majorant de l'autre par "
                 "construction : décimer ne fait pas qu'atténuer, cela "
                 "DÉPLACE les bandes porteuses, et delta_chi ne lit que "
                 "les porteuses. L'écart se lit tel quel."),
    }


SEUIL_DISCRIMINATION_RELATIVE: float = 0.05    # 5 % d'amplitude


def discrimination_du_balayage(mesures: list[dict]) -> dict:
    """Le balayage a-t-il seulement DISCRIMINÉ ? Amplitude relative du Δχ
    max entre le plus petit et le plus grand EPS.

    Si elle est négligeable, l'observable n'est pas gouverné par EPS mais
    par la péremption L1 (k figé) : un AUTRE remonté signifierait alors
    « à ce k, le niveau 0 vivant est infidèle quel que soit EPS », et NON
    « EPS est trop grand ». Diagnostic REPORTÉ — le driver ne conclut
    pas, et il ne balaie pas k pour autant (interdit, §A19)."""
    valeurs = [m["delta_chi"]["delta_chi_max"] for m in mesures]
    coefficients = [m["chrono"]["octets_par_frame_median"] for m in mesures]
    if not valeurs:
        return {"amplitude_relative": 0.0, "a_discrimine": False}
    plus_bas, plus_haut = min(valeurs), max(valeurs)
    amplitude = (plus_haut - plus_bas) / plus_haut if plus_haut > 0 else 0.0
    octets_min, octets_max = min(coefficients), max(coefficients)
    return {
        "delta_chi_max_min": plus_bas,
        "delta_chi_max_max": plus_haut,
        "amplitude_relative": float(amplitude),
        "seuil_discrimination": SEUIL_DISCRIMINATION_RELATIVE,
        "a_discrimine": bool(amplitude >= SEUIL_DISCRIMINATION_RELATIVE),
        "octets_par_frame_min": octets_min,
        "octets_par_frame_max": octets_max,
        "rapport_octets": (float(octets_max / octets_min)
                           if octets_min > 0 else None),
        "note": ("si le trafic varie fortement mais que Δχ ne bouge pas, "
                 "l'observable ne VOIT pas EPS. NOTA v2 : en v1 ce test "
                 "était rendu ininterprétable par le DÉCALAGE SPATIAL du "
                 "reconstructeur, qui plaquait un plancher insensible à "
                 "tout (0.082). L'observable est corrigé (option 1, "
                 "§A21-complément) et la discrimination redevient "
                 "lisible ; aucune valeur de la v1 n'est reconduite. "
                 "Diagnostic reporté, aucune conclusion tirée ici ; k "
                 "n'est pas balayé."),
    }


# TABLE DE BRANCHES, close §A19-complément-2 — aucune lecture ne peut
# tomber hors table. La validité STRICTE (discriminer ET qu'un EPS passe)
# a exposé un troisième cas, fermé ici AVANT toute donnée.
LABEL_INSTRUMENT_VALIDE: str = "INSTRUMENT VALIDE"
LABEL_SONDE_MUETTE: str = "SONDE MUETTE sur EPS"
LABEL_MECANISME_EN_QUESTION: str = "MÉCANISME DE REMONTÉE EN QUESTION"

TABLE_BRANCHES: dict[str, str] = {
    "i": ("(i) INSTRUMENT VALIDE — l'observable répond à EPS sous "
          "péremption minimale et au moins un EPS passe. Un AUTRE à k=4 "
          "est alors attribuable à la PÉREMPTION, pas au seuil."),
    "ii": ("(ii) SONDE MUETTE sur EPS — AUTRE D'INSTRUMENT : l'observable "
           "ne répond pas à EPS même sous péremption minimale. Ne rien "
           "régler, remonter."),
    "iii": ("(iii) MÉCANISME DE REMONTÉE EN QUESTION — k=1 DISCRIMINE "
            "mais AUCUN EPS ne passe, 1e-5 compris. Ni sonde muette "
            "(elle discrimine), ni k coupable (la péremption est "
            "minimale) : à k=1 avec EPS=1e-5 la remontée est quasi sans "
            "perte, donc un résidu au-dessus du seuil pointe AILLEURS."),
}

# Les deux attributions de la branche (iii) : NOMMÉES, NON ARMÉES.
# Les armer est une décision de Romain à la lecture, jamais un
# enchaînement de ce driver.
ATTRIBUTIONS_MECANISME: dict = {
    "a_retard_dune_frame": {
        "enonce": "le retard d'une frame suffit à lui seul",
        "falsificateur": ("comparer contre la vérité DÉCALÉE d'une "
                          "frame"),
        "arme": False,
    },
    "b_derive_reference_incrementale": {
        "enonce": "la référence incrémentale DÉRIVE",
        "falsificateur": ("comparer contre une remontée PLEINE, non "
                          "incrémentale"),
        "arme": False,
    },
    "statut": ("NOMMÉES, NON ARMÉES — l'armement de l'une ou l'autre est "
               "une décision de Romain à la lecture, jamais un "
               "enchaînement. Ce driver ne mesure ni l'une ni l'autre."),
}


def verifier_instrument(mesures_k1: list[dict]) -> tuple[bool, dict]:
    """VÉRIFICATION D'INSTRUMENT (§A19-complément A) — bras k=1, même
    balayage EPS, DUE AVANT toute lecture. Précédent §A14 : une sonde
    muette se détecte AVANT, pas après.

    Elle répond à une seule question : l'observable RÉPOND-IL à EPS quand
    la péremption est minimale ? Deux lectures PRÉ-ÉCRITES :
      (i) k=1 DISCRIMINE — Δχ répond à EPS **et** au moins un EPS passe
          sous le seuil ⇒ **instrument VALIDE**, et l'AUTRE éventuel à
          k=4 est alors attribuable à la PÉREMPTION ;
      (ii) k=1 ne discrimine pas non plus ⇒ **sonde MUETTE sur EPS,
          AUTRE D'INSTRUMENT — ne rien régler, remonter.**

    NOTA gravé : à k=1 subsiste la frame de retard du compteur (choix 6
    de la borne L3, endossé). La vérification teste donc la
    discrimination sous péremption MINIMALE, PAS NULLE — c'est une borne
    inférieure de péremption, pas son absence.

    Ce bras n'est PAS un balayage de k et ne décide aucun cadencement."""
    discrimination = discrimination_du_balayage(mesures_k1)
    passants = [m["eps"] for m in mesures_k1
                if m["delta_chi"]["delta_chi_max"] < SEUIL_IC_BAS]
    repond = discrimination["a_discrimine"]
    valide = bool(repond and passants)
    if valide:
        branche, label = "i", LABEL_INSTRUMENT_VALIDE
    elif not repond:
        branche, label = "ii", LABEL_SONDE_MUETTE
    else:
        branche, label = "iii", LABEL_MECANISME_EN_QUESTION
    details = {
        "cadence_verification": CADENCE_VERIFICATION,
        "discrimination": discrimination,
        "eps_passants_a_k1": passants,
        "instrument_valide": valide,
        "branche": branche,
        "label": label,
        "lecture_preecrite": TABLE_BRANCHES[branche],
        "attributions_branche_iii": (ATTRIBUTIONS_MECANISME
                                     if branche == "iii" else None),
        "table_close": (
            "table complète et CLOSE (§A19-complément-2) : aucune lecture "
            "ne peut tomber hors table — (i) discrimine + un EPS passe ; "
            "(ii) ne discrimine pas ; (iii) discrimine sans qu'aucun EPS "
            "ne passe"),
        "nota_peremption_minimale": (
            f"à k={CADENCE_VERIFICATION} subsiste la frame de retard du "
            "compteur (choix 6 de la borne L3, endossé) : la "
            "vérification teste la discrimination sous péremption "
            "MINIMALE, PAS NULLE"),
        "portee": (
            "ce bras n'est PAS un balayage de k et ne décide aucun "
            "cadencement ; toute DÉCISION sur k reste gatée σ_ω (R4)"),
    }
    return valide, details


def exiger_verification_instrument(verification: dict | None) -> None:
    """Câblage « tranche muette => remonter » (B9) : AUCUNE lecture k=4
    n'est produite sans le PASS/FAIL de la vérification d'instrument
    ENREGISTRÉ. Une lecture sans instrument vérifié serait un chiffre
    dont on ne saurait pas s'il mesure EPS ou la péremption."""
    if not verification or "instrument_valide" not in verification:
        raise RuntimeError(
            "SONDE MUETTE (§A19-complément A) : la vérification "
            "d'instrument (bras k=1) est DUE AVANT toute lecture, et son "
            "PASS/FAIL n'est pas enregistré. Aucune lecture k=4 n'est "
            "produite — remonter.")


def lecture_innocuite(delta_chi_max: float) -> dict:
    """LOGIQUE DU CRITÈRE, gravée (§A19-complément C) — c'est un fait de
    logique, pas une préférence : le joueur ne voit JAMAIS la référence.

    Δχ < 0.0603 ⇒ **innocuité ÉTABLIE**. Δχ > 0.0603 ⇒ **innocuité NON
    ÉTABLIE** — et JAMAIS « nocivité établie ». Un AUTRE ne justifie donc
    pas de resserrer EPS : le seul indice interne dont le joueur dispose
    est la cohérence proche/lointain, et c'est une MESURE DIFFÉRENTE,
    nommée et NON armée."""
    etablie = bool(delta_chi_max < SEUIL_IC_BAS)
    return {
        "delta_chi_max_decime": delta_chi_max,
        "seuil": SEUIL_IC_BAS,
        "innocuite_etablie": etablie,
        "formulation": ("innocuité ÉTABLIE" if etablie
                        else "innocuité NON ÉTABLIE"),
        "jamais_nocivite": (
            "au-dessus du seuil on ne conclut JAMAIS à une « nocivité "
            "établie » : le joueur ne voit jamais la référence. Le seul "
            "indice interne serait la cohérence proche/lointain — mesure "
            "DIFFÉRENTE, nommée, NON armée."),
    }


def branche_decision_k(instrument_valide: bool, eps_retenu: float | None
                       ) -> dict:
    """BRANCHE PRÉ-ÉCRITE (§A19-complément D) — REPORTÉE, jamais
    déclenchée par ce driver.

    Si l'instrument est VALIDÉ mais qu'aucun EPS ne passe à k=4, alors la
    décision passe à k, et le RÉVEIL σ_ω devient LA décision explicite à
    prendre (R4 : le cadencement se déciderait avec une question
    perceptuelle en main). L'EPS retenu serait alors celui que k=1
    valide ; le budget transferts se traite ensuite."""
    applicable = bool(instrument_valide and eps_retenu is None)
    return {
        "applicable": applicable,
        "enonce": (
            "instrument VALIDÉ + aucun EPS ne passe à k=4 ⇒ la décision "
            "passe à k, et le RÉVEIL σ_ω devient LA décision explicite à "
            "prendre (R4). L'EPS retenu serait celui que k=1 valide ; le "
            "budget transferts se traite ensuite."),
        "statut": (
            "REPORTÉE — ce driver ne la déclenche pas et ne réveille "
            "rien. Aucun réveil silencieux, aucune reformulation après "
            "lecture : la décision appartient à Romain."),
    }


def lecture_mecanique(mesures: list[dict],
                      verification: dict | None = None) -> dict:
    """Règle Q2 appliquée SANS interprétation : le PLUS GRAND EPS dont le
    Δχ **DÉCIMÉ** max de série reste < 0.0603. Si aucun ne satisfait,
    AUTRE — et aucun EPS retenu par défaut.

    La vérification d'instrument (bras k=1) est DUE : sans son PASS/FAIL
    enregistré, aucune lecture n'est produite (§A19-complément A)."""
    exiger_verification_instrument(verification)
    satisfaisants = [m for m in mesures
                     if m["delta_chi"]["delta_chi_max"] < SEUIL_IC_BAS]
    retenu = max((m["eps"] for m in satisfaisants), default=None)
    instrument_valide = bool(verification["instrument_valide"])
    pire = max((m["delta_chi"]["delta_chi_max"] for m in mesures),
               default=0.0)
    return {
        "seuil_ic_bas": SEUIL_IC_BAS,
        "regle": ("EPS retenu = le PLUS GRAND EPS dont le Δχ DÉCIMÉ max "
                  f"de série, lu sur {VERITE_COURANTE}, reste < 0.0603 "
                  "(ic_bas) — lecture sur l'IC entier"),
        # (B) — échelle ET vérité de lecture, toutes deux ÉPINGLÉES.
        "echelle_de_lecture": {
            "grandeur_de_la_regle": "delta_chi_max (DÉCIMÉ, échelle du "
                                    "niveau 0)",
            "motif": ("le grossier comparé à ce que le grossier devrait "
                      "être, à sa propre résolution — c'est l'objet qui "
                      "existe et sera consommé"),
            "non_decime": ("delta_chi_max_sans_decimation — DIAGNOSTIC "
                           "SEUL, jamais la règle : la décimation déplace "
                           "les bandes porteuses que delta_chi lit, la "
                           "relation n'est pas monotone"),
        },
        "verite_de_lecture": {
            "verite_de_la_regle": VERITE_COURANTE,
            "motif": ("la règle décide d'INNOCUITÉ PERCEPTUELLE, et le "
                      "joueur ne perçoit pas un canal — il perçoit un "
                      "écart à l'instant présent. Lire la règle sur la "
                      "vérité transportée s'accorderait gratuitement la "
                      "frame de retard que le joueur subit."),
            "verite_diagnostique": VERITE_TRANSPORTEE,
            "usage_du_diagnostic": (
                "isole la fidélité du CANAL seul (seuil, cadence, "
                "transfert). Leur écart est le prix de la frame de "
                "retard : en cas d'échec, il dit s'il faut incriminer le "
                "SEUIL ou le RETARD. Jamais la règle."),
            "declaree_avant_run": True,
        },
        "portee_option_1": (
            "la connaissance du CPU est LUE sur `reference`, qui contient "
            "les colonnes PRÉDITES GPU-side : la sonde crédite donc le CPU "
            "d'une prédiction identique à celle du device. Elle mesure la "
            "fidélité du CANAL, jamais celle de la prédiction propre au "
            "CPU — laquelle exigerait le miroir CPU (option 2), nommé "
            "comme travail de PRODUCTION et non construit. Aucun résultat "
            "d'ici ne peut s'énoncer « le lointain est fidèle », seulement "
            "« le canal l'est »."),
        "eps_retenu": retenu,
        "eps_satisfaisants": [m["eps"] for m in satisfaisants],
        "autre_remonte": bool(retenu is None),
        "branche_preecrite_si_aucun": (
            "AUTRE remonté, AUCUN EPS retenu par défaut — jamais de choix "
            "après courbe"),
        # (A) — la vérification d'instrument, DUE avant cette lecture.
        "verification_instrument": verification,
        "discrimination_du_balayage": discrimination_du_balayage(mesures),
        # (C) — logique du critère : innocuité, jamais nocivité.
        "innocuite": lecture_innocuite(pire),
        # (D) — branche pré-écrite, REPORTÉE et non déclenchée.
        "branche_preecrite_decision_k": branche_decision_k(
            instrument_valide, retenu),
        "eps_en_vigueur": EPS_EN_VIGUEUR,
        "cadence_mesure": CADENCE_MESURE,
        "cadence_verification": CADENCE_VERIFICATION,
        "exclusion_k": (
            "balayage de k INTERDIT (§A19) : SEULES deux cadences "
            f"existent ici — {CADENCE_MESURE} pour la mesure (figée, "
            f"[NON-ANCRÉ, par budget]) et {CADENCE_VERIFICATION} pour la "
            "VÉRIFICATION D'INSTRUMENT. Ce driver ne prononce RIEN sur le "
            "cadencement ; toute décision sur k reste gatée σ_ω (R4)."),
        "rappel_portee": (
            "sonde N1 : elle mesure la fidélité du niveau 0 vivant, rien "
            "d'autre. N2 (M-b) N'EST PAS achetée — elle vient après cette "
            "lecture, avec son propre chiffrage. Aucun verdict ici."),
    }


def main() -> None:
    # AVANT toute allocation : un pré-enregistrement remonté n'est pas
    # endossé, et rien ne doit tourner tant qu'il ne l'est pas.
    exiger_endossement()
    cp = exiger_cupy()
    exiger_pass(VERIFS_EXIGEES)
    exiger_cadence_figee(CADENCE_FIGEE)

    geo = geometrie_v4()
    facteur = max(geo.n_fov // N0_DEFAUT, 1)
    print(f"sonde EPS : {len(EPS_BALAYAGE)} valeurs, décimation ×{facteur} "
          f"(fenêtre {geo.n_fov}² -> {N0_DEFAUT}²)", flush=True)

    def balayer(k: int, etiquette: str) -> list[dict]:
        exiger_cadence_admise(k)
        resultats: list[dict] = []
        for eps in EPS_BALAYAGE:
            print(f"  [{etiquette} k={k}] EPS={eps:g} : chrono ...",
                  flush=True)
            chrono = mesurer_chrono(cp, eps, k)
            liberer_vram(cp)
            print(f"  [{etiquette} k={k}] EPS={eps:g} : Δχ ...", flush=True)
            observable = mesurer_delta_chi(cp, eps, k, SERIE_FRAMES, facteur)
            liberer_vram(cp)
            resultats.append({"eps": eps, "chrono": chrono,
                              "delta_chi": observable})
        return resultats

    # (A) La VÉRIFICATION D'INSTRUMENT est DUE AVANT toute lecture : sans
    # son PASS/FAIL, `lecture_mecanique` refuse de produire quoi que ce
    # soit. Une sonde muette se détecte AVANT, pas après (§A14).
    mesures_k1 = balayer(CADENCE_VERIFICATION, "VÉRIF")
    instrument_valide, verification = verifier_instrument(mesures_k1)
    print(f"  vérification d'instrument (k={CADENCE_VERIFICATION}) : "
          f"{'VALIDE' if instrument_valide else 'SONDE MUETTE'}", flush=True)

    mesures = balayer(CADENCE_MESURE, "MESURE")
    lecture = lecture_mecanique(mesures, verification)
    mempool = cp.get_default_memory_pool()

    document = {
        "meta": {
            "mesure": ("sonde EPS (N1) VERSION 2 — fidélité du CANAL de "
                       "remontée (§A21-complément, pocCascade2phys "
                       "f943e9a)"),
            "supersede": (
                "l'ancien pré-enregistrement (§A19) est SUPERSEDED, pas "
                "amendé : son observable était faux (décalage spatial du "
                "reconstructeur). AUCUNE lecture de la v1 n'est "
                "reconduite — ni le plancher 0.082, ni la branche (ii)."),
            "protocole": {
                "pipeline": ("V4 complet : géométrie emboîtée, prédiction "
                             "GPU-side, L3, L1 k=4 étalée, fovéa mobile"),
                "verite": "niveau 0 par décimation exacte (moyennes Harten)",
                "connaissance_cpu": (
                    "LUE sur `reference` (option 1, tranchée §A21) au SEUL "
                    "instant où elle est dans les coordonnées de la frame "
                    "courante : après le déplacement, avant l'émission. "
                    "Une copie indépendante peut diverger du device ; une "
                    "lecture directe ne le peut pas"),
                "deux_verites": (
                    f"DÉCLARÉES AVANT LE RUN et toutes deux reportées. "
                    f"{VERITE_COURANTE} (K(n) contre S(n)) PORTE LA RÈGLE "
                    "— c'est ce que le joueur voit, canal ET péremption "
                    f"acceptée. {VERITE_TRANSPORTEE} (K(n) contre S̃(n−1)) "
                    "isole le CANAL seul et reste DIAGNOSTIC. Leur écart "
                    "est le prix de la frame de retard"),
                "observable": ("Δχ readout : albedo(S_HALF_OP) puis "
                               "delta_chi(...)['max_carrier'] — primitives "
                               "EXISTANTES, espace instrument, jamais "
                               "l'état"),
                "deux_passes": ("chrono nu d'un côté, Δχ hors chrono de "
                                "l'autre ; aucun chiffre partagé. L'option "
                                "1 supprime la capture de coefficients de "
                                "la v1 : la passe Δχ est plus légère"),
                "echelle_de_lecture": (
                    f"ÉPINGLÉE : la RÈGLE lit le Δχ DÉCIMÉ ×{facteur} à "
                    f"l'échelle du niveau 0 ({N0_DEFAUT}²) — le grossier "
                    "comparé à ce que le grossier devrait être. Le Δχ SANS "
                    "décimation est reporté en DIAGNOSTIC SEUL"),
                "verification_instrument": (
                    f"bras k={CADENCE_VERIFICATION} sur le MÊME balayage, "
                    "DUE avant toute lecture ; k=1 conserve la frame de "
                    "retard du compteur — péremption MINIMALE, pas nulle. "
                    "Ce n'est PAS un balayage de k"),
                "portee": (
                    "mesure la fidélité du CANAL (seuil, cadence, "
                    "transfert), PAS celle de la prédiction propre au CPU "
                    "— `reference` contient les colonnes prédites "
                    "GPU-side. Le miroir CPU (option 2) est nommé comme "
                    "travail de PRODUCTION et non construit"),
                "chrono": "B6 (warmup exclu, médiane ET p99)",
                "kernels": (
                    "F et L3 INTOUCHÉS dans leur code CUDA — empreintes "
                    "sur la CHAÎNE EXTRAITE : sha256(_SOURCE)=e18015f5..."
                    "f30b4 (8223 car.), sha256(_SOURCE_L3)=9533a130...781a "
                    "(9635 car.)"),
            },
            "balayage_eps": list(EPS_BALAYAGE),
            "cadence_l1": CADENCE_FIGEE,
            "s_half_op": S_HALF_OP,
            "frames": {"warmup_exclu": WARMUP_FRAMES, "serie": SERIE_FRAMES},
            "verifs_exigees": list(VERIFS_EXIGEES),
        },
        "mesures": mesures,
        "mesures_verification_k1": mesures_k1,
        "residence": {
            "mempool_total_octets": int(mempool.total_bytes()),
            "nvidia_smi_octets": vram_nvidia_smi_octets(),
        },
        "lecture_mecanique": lecture,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON_PATH.write_text(
        json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 78)
    print("F1 -- SONDE EPS (N1) : fidélité du niveau 0 vivant")
    print("=" * 78)
    print(f"  {'EPS':>8}  {'Δχ max':>8} {'Δχ méd':>8}  "
          f"{'octets/f':>10} {'transf':>7}  {'médiane':>8} {'p99':>8}")
    for mesure in mesures:
        observable, chrono = mesure["delta_chi"], mesure["chrono"]
        stats = chrono["frame_time"]
        print(f"  {mesure['eps']:>8.0e}  "
              f"{observable['delta_chi_max']:>8.4f} "
              f"{observable['delta_chi_median']:>8.4f}  "
              f"{chrono['octets_par_frame_median']:>10.0f} "
              f"{chrono['temps_transfert_median_ms']:>7.3f}  "
              f"{stats['mediane_ms']:>8.3f} {stats['p99_ms']:>8.3f}")
    verif = lecture["verification_instrument"]
    print("-" * 78)
    print(f"  VÉRIFICATION D'INSTRUMENT (k={CADENCE_VERIFICATION}, DUE) : "
          f"branche ({verif['branche']}) {verif['label']}")
    print(f"    {verif['lecture_preecrite']}")
    if verif["attributions_branche_iii"]:
        attributions = verif["attributions_branche_iii"]
        print("    deux attributions NOMMÉES, NON ARMÉES :")
        for cle in ("a_retard_dune_frame", "b_derive_reference_incrementale"):
            entree = attributions[cle]
            print(f"      - {entree['enonce']} ; falsificateur : "
                  f"{entree['falsificateur']}")
        print(f"      ({attributions['statut']})")
    print(f"    nota : {verif['nota_peremption_minimale']}")
    print(f"  seuil ic_bas = {SEUIL_IC_BAS}  (règle : le PLUS GRAND EPS "
          f"dont Δχ DÉCIMÉ max < seuil ; le non décimé est un "
          f"DIAGNOSTIC)")
    innocuite = lecture["innocuite"]
    print(f"  LOGIQUE (C) : Δχ max décimé = "
          f"{innocuite['delta_chi_max_decime']:.4f} -> "
          f"{innocuite['formulation']}")
    print("    (au-dessus du seuil on ne conclut JAMAIS à une nocivité : "
          "le joueur ne voit jamais la référence)")
    if lecture["autre_remonte"]:
        print("  -> AUCUN EPS ne satisfait : AUTRE remonté, aucun EPS "
              "retenu par défaut.")
    else:
        print(f"  -> EPS retenu = {lecture['eps_retenu']:g}  "
              f"(en vigueur : {EPS_EN_VIGUEUR:g})")
    discrimination = lecture["discrimination_du_balayage"]
    print(f"  discrimination du balayage : Δχ varie de "
          f"{discrimination['amplitude_relative']:.2%} pour un trafic ×"
          f"{discrimination['rapport_octets'] or float('nan'):.0f}  -> "
          f"a discriminé = {discrimination['a_discrimine']}")
    if not discrimination["a_discrimine"]:
        print("     (Δχ insensible à EPS : l'observable ne VOIT pas le "
              "seuil — un AUTRE ne dirait PAS « EPS trop grand ». "
              "L'explication « péremption L1 » est FALSIFIÉE par k=1 ; "
              "plancher structurel, D-1/D-2 tranchent.)")
    branche = lecture["branche_preecrite_decision_k"]
    if branche["applicable"]:
        print("  BRANCHE PRÉ-ÉCRITE (D) APPLICABLE — REPORTÉE, non "
              "déclenchée :")
        print(f"    {branche['enonce']}")
    print(f"  cadences : mesure k={CADENCE_MESURE} (figée), vérification "
          f"k={CADENCE_VERIFICATION} — PAS un balayage de k ; aucune "
          "décision de cadencement prononcée ici (§A19).")
    print(f"[REPORT] -> {OUT_JSON_PATH}")
    print("POINT D'ARRÊT OBLIGATOIRE : lecture remontée à Romain. N2 (M-b) "
          "n'est PAS achetée — chiffrage propre à venir.")


if __name__ == "__main__":
    main()
