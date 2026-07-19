"""F1 — SONDE EPS (N1) : à quel EPS_DETAIL la remontée seuillée cesse-t-elle
d'être fidèle AU READOUT ? (§A19, pocCascade2phys c3d13a1).

CE QUE N1 MESURE, ET LA CONCESSION QUI L'A FAIT NAÎTRE : la formule du
§A18-lecture — « balayage EPS gratuit dans le run M-b » — était FAUSSE sur
l'observable. EPS_DETAIL gouverne la fidélité du NIVEAU 0 VIVANT, pas le
contrat live↔rederive ; il n'atteint le Δχ de M-b qu'au second ordre. La
sonde corrigée est AUTONOME : elle compare directement les deux niveaux 0.

  - **VÉRITÉ** : niveau 0 par DÉCIMATION EXACTE (moyennes Harten,
    `multiresolution.downsample`) de l'état fin courant ;
  - **VIVANT** : niveau 0 reconstruit par la remontée SEUILLÉE à EPS,
    INCRÉMENTALE, **L1 k=4 étalée INCLUSE** — la péremption
    d'amortissement fait partie du régime de production, pas un défaut
    qu'on écarterait pour flatter la mesure ;
  - **OBSERVABLE** : Δχ readout — `albedo` au point d'opération S_HALF_OP
    puis `delta_chi(...)["max_carrier"]`, les primitives EXISTANTES,
    importées et non réécrites. Espace instrument, JAMAIS l'état : aucune
    comparaison L2 sur les champs (règle 2 du dépôt).

RÈGLE DE DÉCISION PRÉ-ENREGISTRÉE (Q2), appliquée SANS interprétation :
**EPS retenu = le plus grand EPS dont le Δχ MAX de série reste < 0.0603
(ic_bas)** — lecture sur l'IC entier, discipline §A13. Si aucun EPS du
balayage ne satisfait, 1e-5 compris : **AUTRE remonté, aucun EPS retenu
par défaut**. Jamais de choix après courbe.

**VÉRIFICATION D'INSTRUMENT (A), DUE AVANT TOUTE LECTURE** — le même
balayage EPS est d'abord joué à **k=1**, et `lecture_mecanique` REFUSE de
produire la lecture k=4 sans son PASS/FAIL enregistré (câblage « muette ⇒
remonter », B9). Une sonde muette se détecte AVANT, pas après (§A14).
TABLE DE BRANCHES COMPLÈTE ET CLOSE (§A19-complément-2) — aucune lecture
ne peut tomber hors table :
  **(i)** k=1 discrimine ET au moins un EPS passe ⇒ **INSTRUMENT VALIDE** ;
      un AUTRE à k=4 est alors attribuable à la PÉREMPTION, pas au seuil ;
  **(ii)** k=1 ne discrimine pas ⇒ **SONDE MUETTE sur EPS**, AUTRE
      D'INSTRUMENT — ne rien régler, remonter ;
  **(iii)** k=1 DISCRIMINE mais AUCUN EPS ne passe, 1e-5 compris ⇒
      **MÉCANISME DE REMONTÉE EN QUESTION**. Ni sonde muette (elle
      discrimine), ni k coupable (la péremption est minimale) : à k=1
      avec EPS=1e-5 la remontée est quasi sans perte, donc un résidu
      au-dessus du seuil pointe AILLEURS. DEUX ATTRIBUTIONS NOMMÉES, NON
      ARMÉES — (a) le retard d'une frame suffit à lui seul, falsificateur
      = comparer contre la vérité DÉCALÉE d'une frame ; (b) la référence
      incrémentale DÉRIVE, falsificateur = comparer contre une remontée
      PLEINE non incrémentale. Les armer est une décision de Romain à la
      lecture, jamais un enchaînement : ce driver porte le LABEL, il ne
      mesure ni l'une ni l'autre.
NOTA : à k=1 subsiste la frame de retard du compteur (choix 6 de la borne
L3) — la vérification teste la discrimination sous péremption MINIMALE,
PAS NULLE. C'est précisément ce qui rend (iii) lisible : la péremption
restante est minimale, pas nulle, et l'attribution (a) la nomme.

**CE N'EST PAS UN BALAYAGE DE k** : deux valeurs seulement existent ici,
k=4 (mesure, figée) et k=1 (vérification), et ce driver ne prononce RIEN
sur le cadencement. Toute DÉCISION sur k reste gatée σ_ω (R4) et
appartient à Romain — le driver refuse fail-loud toute autre valeur.

DEUX PASSES PAR EPS, et c'est délibéré : reconstruire le vivant exige de
rapatrier des coefficients et la vérité, ce qui fausserait un chrono. La
passe CHRONO tourne donc le pipeline nu (B6 : warmup exclu, série,
médiane ET p99) ; la passe Δχ rejoue la MÊME trajectoire (même graine,
même géométrie, même k) en évaluant l'observable hors de toute mesure de
temps. Les deux passes ne partagent aucun chiffre.

**ÉCHELLE DE LECTURE ÉPINGLÉE (B)** : la règle Q2 lit le Δχ **DÉCIMÉ à
l'échelle du niveau 0** — le grossier comparé à ce que le grossier
devrait être, à sa propre résolution : c'est l'objet qui existe et qui
sera consommé. Le Δχ NON décimé est reporté en **DIAGNOSTIC SEUL**, et
n'entre dans aucune décision. La relation entre les deux n'est pas
monotone : la décimation ne fait pas qu'atténuer, elle DÉPLACE les bandes
porteuses que `delta_chi` est seul à lire (vérifié au build : 0.038
décimé contre 0.022 à pleine résolution sur un cas).

**LOGIQUE DU CRITÈRE (C)**, fait de logique et non préférence : le joueur
ne voit JAMAIS la référence. Donc Δχ < 0.0603 ⇒ **innocuité ÉTABLIE** ;
Δχ > 0.0603 ⇒ **innocuité NON ÉTABLIE** — et JAMAIS « nocivité établie ».
Un AUTRE ne justifie donc PAS de resserrer EPS ; le seul indice interne
dont le joueur disposerait est la cohérence proche/lointain, et c'est une
MESURE DIFFÉRENTE, nommée et NON armée.

**BRANCHE PRÉ-ÉCRITE (D), REPORTÉE** : instrument VALIDÉ + aucun EPS ne
passe à k=4 ⇒ la décision passe à k, et le RÉVEIL σ_ω devient LA décision
explicite à prendre. Ce driver la REPORTE au JSON ; il ne la déclenche
pas et ne réveille rien.

────────────────────────────────────────────────────────────────────────
FAIT MESURÉ AU BUILD, REMONTÉ AVANT LE RUN : la péremption L1 pourrait
DOMINER le seuil EPS — auquel cas cette sonde ne discriminerait pas
────────────────────────────────────────────────────────────────────────
Sur une configuration RÉDUITE (n_fov=64, n0=8, 12 frames — PAS la cellule
de mesure), le balayage complet a été exercé au build. Résultat : le
nombre de coefficients remontés varie d'un facteur ~2000 entre EPS=1e-5
et EPS=1e-2, mais le Δχ max ne bouge que de **0.07 %** (0.0806 → 0.0805),
et reste AU-DESSUS du seuil 0.0603 pour TOUS les EPS.

Mécanisme, structurel et non accidentel : avec L1 k=4, une fenêtre ne
remonte qu'une frame sur quatre. Entre deux remontées, F fait avancer
l'état sans que le CPU en sache rien, et cette DÉRIVE domine largement la
troncature du seuil. Autrement dit l'observable mesure surtout la
péremption — dont le paramètre, k, est FIGÉ et dont le balayage est
INTERDIT ici.

Ce que cela implique pour la lecture, dit AVANT le run : si le
comportement persiste à la cellule réelle, la règle Q2 remontera
vraisemblablement AUTRE — mais cet AUTRE ne signifiera PAS « EPS est trop
grand ». Il signifiera « à k=4, le niveau 0 vivant est infidèle
indépendamment d'EPS ». Confondre les deux conduirait à resserrer EPS
sans effet, en payant du trafic pour rien.

Ces chiffres viennent d'une config réduite et ne préjugent pas de la
cellule de mesure — ils ne sont PAS reportés comme résultat. Le driver
instrumente le fait : `discrimination_du_balayage` mesure, sur la vraie
série, l'amplitude de Δχ entre le plus petit et le plus grand EPS, et
lève un drapeau si elle est négligeable. La lecture verra donc d'elle-même
si le balayage a discriminé quoi que ce soit.

Kernels F et L3, chrono, substrats : INTOUCHÉS. Sortie JSON SANS
timestamp.

Machine : iluin-tworings3, terminal natif UNIQUEMENT. Usage :
  .venv/bin/python scripts/run_f1_sonde_eps.py

POINT D'ARRÊT OBLIGATOIRE après le run : lecture remontée à Romain. N2
(M-b) N'EST PAS achetée — elle vient après cette lecture, avec son propre
chiffrage."""
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
    """Le niveau 0 VIVANT : ce que le CPU sait réellement, reconstruit à
    partir des seuls coefficients qui lui sont REMONTÉS — donc seuillés à
    EPS et périmés par L1 (une fenêtre ne remonte qu'une frame sur k).

    Le schéma B4 est incrémental : chaque coefficient reçu s'ajoute à ce
    que le CPU détenait. Les résidus sous le seuil s'accumulent côté GPU
    et repassent le seuil plus tard — rien n'est perdu, mais le vivant
    RETARDE sur la vérité, et c'est exactement ce que la sonde mesure."""

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
    """Passe Δχ : même trajectoire, observable évalué HORS chrono."""
    geo = geometrie_v4()
    transferts = TransfertComptable(cp)
    pipeline = PipelineMaQuater(cp, geo, transferts, k=k, eps=eps,
                                capturer_coefficients=True)
    reconstructeur = ReconstructeurNiveau0(pipeline)
    indice_groupe, position = _slot_observe(pipeline)

    serie_decimee: list[float] = []
    serie_pleine: list[float] = []
    for _ in range(frames):
        pipeline.frame(delta_x=DELTA_X_MOBILE)
        reconstructeur.appliquer(pipeline.coefficients_frame)
        vivant = reconstructeur.champ(indice_groupe, position)
        verite = champ_verite(pipeline, indice_groupe, position)
        serie_decimee.append(
            delta_chi_readout(vivant, verite, facteur_decimation))
        serie_pleine.append(delta_chi_readout(vivant, verite, 1))
    return {
        "frames_evaluees": frames,
        "facteur_decimation": facteur_decimation,
        "delta_chi_max": float(np.max(serie_decimee)),
        "delta_chi_median": float(np.percentile(serie_decimee, 50)),
        "delta_chi_max_sans_decimation": float(np.max(serie_pleine)),
        "delta_chi_median_sans_decimation": float(
            np.percentile(serie_pleine, 50)),
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
                 "l'observable est gouverné par la PÉREMPTION L1 (k figé), "
                 "pas par EPS. Un AUTRE voudrait alors dire « à ce k, le "
                 "niveau 0 vivant est infidèle quel que soit EPS » — et "
                 "NON « EPS est trop grand ». Diagnostic reporté, aucune "
                 "conclusion tirée ici ; k n'est pas balayé (§A19)."),
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
                  "de série reste < 0.0603 (ic_bas) — lecture sur l'IC "
                  "entier"),
        # (B) — l'échelle de lecture est ÉPINGLÉE, et dite ici.
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
            "mesure": "sonde EPS (N1) — fidélité du niveau 0 vivant "
                      "(§A19, pocCascade2phys c3d13a1)",
            "protocole": {
                "pipeline": ("V4 complet : géométrie emboîtée, prédiction "
                             "GPU-side, L3, L1 k=4 étalée, fovéa mobile"),
                "verite": "niveau 0 par décimation exacte (moyennes Harten)",
                "vivant": ("niveau 0 reconstruit par la remontée seuillée, "
                           "incrémentale, L1 k=4 INCLUSE (la péremption "
                           "fait partie du régime)"),
                "observable": ("Δχ readout : albedo(S_HALF_OP) puis "
                               "delta_chi(...)['max_carrier'] — primitives "
                               "EXISTANTES, espace instrument, jamais "
                               "l'état"),
                "deux_passes": ("chrono nu d'un côté, Δχ hors chrono de "
                                "l'autre — reconstruire le vivant exige "
                                "des rapatriements qui fausseraient le "
                                "temps ; aucun chiffre partagé"),
                "echelle_de_lecture": (
                    f"ÉPINGLÉE (§A19-complément B) : la RÈGLE lit le Δχ "
                    f"DÉCIMÉ ×{facteur} à l'échelle du niveau 0 "
                    f"({N0_DEFAUT}²) — le grossier comparé à ce que le "
                    "grossier devrait être. Le Δχ SANS décimation est "
                    "reporté en DIAGNOSTIC SEUL"),
                "verification_instrument": (
                    f"bras k={CADENCE_VERIFICATION} sur le MÊME balayage, "
                    "DUE avant toute lecture (§A19-complément A) ; k=1 "
                    "conserve la frame de retard du compteur — péremption "
                    "MINIMALE, pas nulle. Ce n'est PAS un balayage de k"),
                "chrono": "B6 (warmup exclu, médiane ET p99)",
                "kernels": "F fusionné et L3 INTOUCHÉS",
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
        print("     (Δχ insensible à EPS : l'observable est gouverné par la "
              "PÉREMPTION L1, pas par le seuil — un AUTRE ne dirait PAS "
              "« EPS trop grand ».)")
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
