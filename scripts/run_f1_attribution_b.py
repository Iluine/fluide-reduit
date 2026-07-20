"""F1 — ATTRIBUTION (b) ARMÉE : la référence incrémentale DÉRIVE-t-elle ?
(§A19-lecture-diagnostic, pocCascade2phys 4062747).

Le falsificateur était NOMMÉ dès la table de branches (§A19-complément-2,
attribution `b_derive_reference_incrementale`) : **comparer contre une
remontée PLEINE, non incrémentale**. Il est armé ici.

  - **BRAS TÉMOIN** — remontée SANS seuillage et SANS référence
    incrémentale : le CPU reçoit l'ÉTAT COMPLET des cellules remontées ;
  - **BRAS COURANT** — le schéma incrémental seuillé tel que mesuré ;
  - mêmes cellules, même cadence (k=4), même série, même état initial
    (même graine, même géométrie, même trajectoire) : la soustraction
    n'isole que le SCHÉMA.

────────────────────────────────────────────────────────────────────────
CE QUI FAIT DU TÉMOIN UNE REMONTÉE PLEINE — et pourquoi ce n'est pas un
réglage d'EPS déguisé
────────────────────────────────────────────────────────────────────────
Le témoin s'obtient EXACTEMENT en posant eps = 0 sur le kernel L3
INTOUCHÉ. Le kernel retient un détail dès que `|d| >= eps` ; à eps = 0 la
condition est toujours vraie, donc TOUTE cellule émet, et la mise à jour
incrémentale `reference[p] += d` avec d = etat − reference donne
`reference = etat` : l'incrément DEVIENT l'écrasement complet. Le témoin
est donc la remontée pleine non incrémentale au sens strict, obtenue sans
toucher une ligne de kernel.

eps = 0 n'est PAS un candidat de réglage et ne peut pas le devenir : il
transfère l'état entier des slots remontés à chaque tour. C'est un
APPAREIL DE MESURE, pas une configuration — et `exiger_rien_regle` le
traite comme tel, en refusant fail-loud toute valeur qui ne soit ni le
courant (1e-4, figé) ni le témoin (0). k reste FIGÉ à 4 : rien n'est
réglé ici non plus.

────────────────────────────────────────────────────────────────────────
VERROU AMONT — POURQUOI CE DRIVER PEUT REFUSER DE PRONONCER
────────────────────────────────────────────────────────────────────────
Le schéma B4 pose une IDENTITÉ : la `reference` côté device EST le modèle
de « ce que le CPU sait » — le kernel émet d = etat − reference, transmet
d, puis fait reference += d. Le vivant reconstruit côté CPU doit donc
égaler `reference` au readout. Si l'identité casse, le « vivant » n'est
PAS la connaissance du CPU, et les deux bras ne mesurent plus le schéma
mais le défaut de reconstruction — qui les affecte tous les deux et les
rendrait indistinguables.

Ce driver MESURE cette identité (`verrou_invariant`), en espace
INSTRUMENT — Δχ entre le vivant reconstruit et le livre de comptes, pas
une norme sur l'état (règle 2 du dépôt) ; l'écart brut est reporté à côté
en DIAGNOSTIC seul. Tant qu'elle est rompue au-delà du plancher de bruit,
l'attribution est **NON PRONONÇABLE** : les deux bras sont reportés, la
lecture pré-écrite n'est PAS appliquée, et `exiger_attribution_prononcable`
refuse fail-loud toute consommation du verdict (câblage B9, « tranche
muette ⇒ remonter »).

────────────────────────────────────────────────────────────────────────
LECTURE MÉCANIQUE PRÉ-ÉCRITE (gravée), appliquée SANS interprétation
────────────────────────────────────────────────────────────────────────
  - l'erreur **S'EFFONDRE** sur le bras témoin ⇒ **LA RÉFÉRENCE
    INCRÉMENTALE DÉRIVE** — défaut de mécanisme, load-bearing pour tout le
    lointain de la fovéa-z ;
  - elle **PERSISTE** ⇒ **ERREUR INHÉRENTE au grossier**, et c'est
    l'exigence de fidélité qui est à redéfinir.
Les TROIS vues (complet / cycle / cumulé) sont reportées pour les DEUX
bras ; c'est la comparaison des CUMULÉS qui discrimine (gravé).

NOTE DE DOMAINE, reportée et non corrigée en douce : seule la vue
COMPLÈTE a un domaine IDENTIQUE entre les deux bras. Les vues cycle et
cumulé se restreignent aux cellules qu'un coefficient a touchées, et le
témoin en touche mécaniquement plus (il n'a pas de seuil) : leur
soustraction compare des domaines différents. Les fractions couvertes des
deux bras sont donc reportées CÔTE À CÔTE, pour que l'écart de domaine
soit lisible à la lecture plutôt que supposé nul.

────────────────────────────────────────────────────────────────────────
SEUILS FIGÉS AVANT LE RUN, ET LE PLANCHER DE BRUIT QUI LES JUSTIFIE
────────────────────────────────────────────────────────────────────────
**Plancher de bruit côté Δχ — MESURÉ, pas supposé.** D-1(a) a établi que
le readout est PROPRE : Δχ à blanc exactement nul. Il n'y a donc aucun
plancher d'instrument à retrancher, et le seul bruit qui reste est la
reproductibilité de bout en bout. Elle est MESURÉE ici par un RÉPLICAT :
le bras courant est joué DEUX FOIS, à graine, géométrie, cadence et série
identiques. Tout ce qui sépare les deux passes est du bruit par
définition — c'est l'exact analogue Δχ de la dérive machine (~0.18 ms)
côté temps. Le plancher retenu est l'écart max observé entre réplicats,
sur les trois vues. S'il ressort NUL, cela dit que la chaîne est
déterministe, PAS qu'elle est infiniment précise : le facteur relatif
porte alors seul la décision, et le driver le dit.

**Effondrement** (`FACTEUR_EFFONDREMENT_ATTRIBUTION`, 0.5) : l'erreur est
tenue pour effondrée si le témoin tombe sous la MOITIÉ du courant **ET**
si l'écart absolu entre les deux dépasse le plancher de bruit mesuré. Les
deux conditions, pas une : un rapport spectaculaire entre deux nombres
tous deux dans le bruit ne dit rien.

────────────────────────────────────────────────────────────────────────
PORTÉE
────────────────────────────────────────────────────────────────────────
EPS reste FIGÉ à 1e-4 pour le régime, k reste FIGÉ à 4, **rien n'est
réglé**. Kernels F et L3, chrono, substrats, pyramide : INTOUCHÉS —
preuve par `git diff` sur `src/`, et non par empreinte.

Le seul sha256 réellement VERROUILLÉ par la suite, re-calculable par qui
l'exige, est celui du kernel F :
    sha256(src.f1_gpu.substrat_fusionne._SOURCE.encode()) =
    e18015f57e14263414239b18c51d25cd335250f58dde2ccb892a6e2c1d6f30b4
Il porte sur la CHAÎNE CUDA EXTRAITE (`_SOURCE`, 8223 caractères), PAS
sur le fichier entier. Deux tests l'imposent
(`test_f1_substrat_fusionne_s1.py`, `test_f1_substrat_l3.py`). Le kernel
L3 (`_SOURCE_L3`) n'a PAS de verrou d'empreinte : il est tenu par
l'équivalence structurelle (réutilisation du préfixe du fusionné) et par
les verrous d'état bit-identique.

Les primitives de readout et le reconstructeur de la sonde sont RÉUTILISÉS
tels quels, jamais réécrits. Sortie JSON SANS timestamp.

Machine : iluin-tworings3, terminal natif UNIQUEMENT. Usage :
  .venv/bin/python scripts/run_f1_attribution_b.py

POINT D'ARRÊT OBLIGATOIRE après le run : lecture remontée à Romain."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_f1_diagnostic_instrument import (  # noqa: E402
    ReconstructeurAvecCouverture,
    champ_restreint,
)
from scripts.run_f1_ma_quater import (  # noqa: E402
    CADENCE_L1,
    PipelineMaQuater,
    geometrie_v4,
)
from scripts.run_f1_ma_ter import VERIFS_EXIGEES, liberer_vram  # noqa: E402
from scripts.run_f1_s2_mobile import DELTA_X_MOBILE  # noqa: E402
from scripts.run_f1_sonde_eps import (  # noqa: E402
    CHAMP_SEDIMENT,
    EPS_EN_VIGUEUR,
    SEUIL_IC_BAS,
    _slot_observe,
    champ_verite,
    delta_chi_readout,
)
from src.f1_gpu.backend import exiger_cupy, vers_cpu  # noqa: E402
from src.f1_gpu.chrono import SERIE_FRAMES, WARMUP_FRAMES  # noqa: E402
from src.f1_gpu.pyramide import N0_DEFAUT  # noqa: E402
from src.f1_gpu.transferts import TransfertComptable  # noqa: E402
from src.f1_gpu.verifs import exiger_pass, vram_nvidia_smi_octets  # noqa: E402

OUT_DIR = ROOT / "outputs" / "f1"
OUT_JSON_PATH = OUT_DIR / "attribution_b.json"

# Les DEUX protocoles, et deux seulement. Le témoin n'est pas un réglage :
# à eps = 0 le kernel L3 émet TOUTE cellule et `reference += d` devient
# l'écrasement complet — c'est la remontée pleine non incrémentale, au
# sens strict, obtenue sans toucher au kernel.
EPS_COURANT: float = EPS_EN_VIGUEUR        # 1e-4, régime figé
EPS_TEMOIN: float = 0.0                    # falsificateur, PAS un candidat
EPS_ADMIS: tuple[float, ...] = (EPS_COURANT, EPS_TEMOIN)
CADENCE_FIGEE_ATTRIBUTION: int = CADENCE_L1

PROTOCOLES: dict[str, dict] = {
    "courant": {
        "eps": EPS_COURANT,
        "enonce": "schéma incrémental SEUILLÉ, tel que mesuré",
    },
    "temoin": {
        "eps": EPS_TEMOIN,
        "enonce": ("remontée PLEINE : sans seuillage et sans référence "
                   "incrémentale — le CPU reçoit l'état COMPLET des "
                   "cellules remontées"),
        "pourquoi_eps_zero": (
            "à eps = 0 la condition |d| >= eps du kernel L3 est toujours "
            "vraie : toute cellule émet, et reference += (etat − "
            "reference) donne reference = etat. L'incrément DEVIENT "
            "l'écrasement complet — remontée pleine non incrémentale au "
            "sens strict, kernel INTOUCHÉ."),
        "pas_un_reglage": (
            "eps = 0 transfère l'état entier des slots remontés à chaque "
            "tour : c'est un APPAREIL DE MESURE, jamais une configuration "
            "retenable. Aucune lecture ne peut le proposer comme régime."),
    },
}

# Seuil d'effondrement, FIGÉ AVANT LE RUN et nommé au protocole. Distinct
# de celui de D-1(b) : là-bas on comparait un champ restreint à lui-même
# complet, ici on compare DEUX BRAS. Deux conditions, pas une — voir
# `lecture_attribution`.
FACTEUR_EFFONDREMENT_ATTRIBUTION: float = 0.5

# Le réplicat qui MESURE le plancher de bruit : le bras courant joué deux
# fois à l'identique. Ce n'est pas un bras de plus au sens du protocole —
# c'est l'étalon de bruit, sans lequel « au-dessus du bruit » n'a pas de
# sens (correction §A19-lecture-diagnostic : la faute de seuil de D-2
# venait d'un seuil placé SOUS le bruit connu).
N_PASSES_REPLICAT: int = 2

VUES: tuple[str, ...] = ("complet", "cycle", "cumule")
VUE_DISCRIMINANTE: str = "cumule"          # gravé
VUE_DOMAINE_IDENTIQUE: str = "complet"

OBSERVABLE_RECONSTRUIT: str = "reconstruite"
OBSERVABLE_LIVRE: str = "livre_de_comptes"


def exiger_rien_regle(eps: float, k: int) -> None:
    """Garde de portée, RECONDUITE : ce driver ne règle RIEN.

    Deux valeurs d'EPS existent ici — le régime figé (1e-4) et le témoin
    (0), qui est un appareil de mesure et non un candidat. Toute autre
    valeur serait un balayage déguisé en falsificateur. k reste FIGÉ à 4 :
    bouger la cadence avec une question perceptuelle en main ATTEINDRAIT
    la condition de réveil σ_ω (R4)."""
    if eps not in EPS_ADMIS:
        raise RuntimeError(
            f"attribution (b) : eps={eps} hors des deux protocoles "
            f"{EPS_ADMIS} — {EPS_COURANT:g} pour le bras COURANT (régime "
            f"figé), {EPS_TEMOIN:g} pour le bras TÉMOIN (remontée pleine, "
            "appareil de mesure). Rien n'est réglé ici.")
    if k != CADENCE_FIGEE_ATTRIBUTION:
        raise RuntimeError(
            f"attribution (b) : k={k} != {CADENCE_FIGEE_ATTRIBUTION}. La "
            "cadence est FIGÉE ; la bouger avec la question perceptuelle "
            "en main atteindrait la condition de réveil σ_ω (R4). C'est "
            "une décision explicite de Romain, jamais un effet de bord.")


def champ_livre_de_comptes(pipeline: PipelineMaQuater, indice_groupe: int,
                           position: int) -> np.ndarray:
    """Le LIVRE DE COMPTES : `reference` côté device, c'est-à-dire ce que
    le schéma B4 AFFIRME que le CPU sait — le kernel émet d = etat −
    reference puis fait reference += d.

    Lu ici pour une seule raison : confronter l'affirmation du schéma à ce
    que le CPU détient réellement. L'écart entre les deux n'est imputable
    ni au seuil ni à la cadence — il est imputable à la RECONSTRUCTION."""
    bloc = pipeline.references[indice_groupe][position, 0, CHAMP_SEDIMENT]
    return vers_cpu(bloc).astype(np.float64)


def resume(serie: list[float]) -> dict:
    return {"max": float(np.max(serie)),
            "median": float(np.percentile(serie, 50))}


def mesurer_bras(cp, eps: float, frames: int, facteur: int,
                 k: int = CADENCE_FIGEE_ATTRIBUTION) -> dict:
    """Un bras : même graine, même géométrie, même cadence, même série,
    même trajectoire — seul le SCHÉMA change (eps).

    Deux observables sont évalués sur la MÊME trajectoire :
      - `reconstruite`  : Δχ(vivant reconstruit côté CPU, vérité) — ce que
        le consommateur du lointain détient réellement ;
      - `livre_de_comptes` : Δχ(reference device, vérité) — ce que le
        schéma AFFIRME qu'il détient.
    Leur écart est l'objet du verrou amont."""
    exiger_rien_regle(eps, k)
    geo = geometrie_v4()
    transferts = TransfertComptable(cp)
    pipeline = PipelineMaQuater(cp, geo, transferts, k=k, eps=eps,
                                capturer_coefficients=True)
    reconstructeur = ReconstructeurAvecCouverture(pipeline)
    indice, position = _slot_observe(pipeline)

    series: dict[str, dict[str, list[float]]] = {
        observable: {vue: [] for vue in VUES}
        for observable in (OBSERVABLE_RECONSTRUIT, OBSERVABLE_LIVRE)}
    fractions: dict[str, list[float]] = {"cycle": [], "cumule": []}
    invariant: list[float] = []
    invariant_brut: list[float] = []

    for numero in range(frames):
        if numero % k == 0:
            reconstructeur.nouveau_cycle()
        pipeline.frame(delta_x=DELTA_X_MOBILE)
        reconstructeur.appliquer(pipeline.coefficients_frame)

        verite = champ_verite(pipeline, indice, position)
        champs = {
            OBSERVABLE_RECONSTRUIT: reconstructeur.champ(indice, position),
            OBSERVABLE_LIVRE: champ_livre_de_comptes(pipeline, indice,
                                                     position),
        }
        for observable, champ in champs.items():
            series[observable]["complet"].append(
                delta_chi_readout(champ, verite, facteur))
        for vue, cumule in (("cycle", False), ("cumule", True)):
            couvert = reconstructeur.masque(cumule, indice, position)
            fractions[vue].append(float(couvert.mean()))
            for observable, champ in champs.items():
                series[observable][vue].append(delta_chi_readout(
                    champ_restreint(champ, verite, couvert), verite,
                    facteur))

        # VERROU : le vivant reconstruit doit égaler le livre de comptes.
        # Mesuré en espace INSTRUMENT (règle 2 : aucune norme sur l'état
        # ne sert de critère) ; l'écart brut suit en DIAGNOSTIC seul.
        invariant.append(delta_chi_readout(
            champs[OBSERVABLE_RECONSTRUIT], champs[OBSERVABLE_LIVRE],
            facteur))
        invariant_brut.append(float(np.max(np.abs(
            champs[OBSERVABLE_RECONSTRUIT] - champs[OBSERVABLE_LIVRE]))))

    return {
        "eps": eps,
        "frames": frames,
        "cadence": k,
        "vues": {
            observable: {
                vue: ({**resume(valeurs),
                       "fraction_couverte_mediane": float(
                           np.percentile(fractions[vue], 50))}
                      if vue in fractions else resume(valeurs))
                for vue, valeurs in par_vue.items()}
            for observable, par_vue in series.items()},
        "invariant_b4": {
            **resume(invariant),
            "ecart_brut_max_diagnostic": float(np.max(invariant_brut)),
            "enonce": ("Δχ(vivant reconstruit, livre de comptes) — le "
                       "schéma B4 pose qu'ils sont le MÊME objet"),
        },
        "retard_transfert": pipeline.audit_cfl(),
    }


def plancher_bruit(passes_courantes: list[dict]) -> dict:
    """Plancher de bruit côté Δχ, MESURÉ par réplicat — le bras courant
    joué deux fois à l'identique.

    Justification, dite ici plutôt que supposée : D-1(a) a établi que le
    readout est PROPRE (Δχ à blanc exactement nul), donc il n'y a aucun
    plancher d'instrument à retrancher. Le seul bruit restant est la
    reproductibilité de bout en bout, et c'est elle qu'on mesure. Analogue
    exact de la dérive machine (~0.18 ms) côté temps."""
    if len(passes_courantes) < 2:
        raise RuntimeError(
            "plancher de bruit : le réplicat est DÛ — sans deux passes "
            "identiques, « au-dessus du bruit » n'a pas de sens. C'est "
            "précisément la faute de seuil corrigée au "
            "§A19-lecture-diagnostic.")
    premiere, seconde = passes_courantes[0], passes_courantes[1]
    ecarts: dict[str, float] = {}
    for observable in (OBSERVABLE_RECONSTRUIT, OBSERVABLE_LIVRE):
        for vue in VUES:
            ecarts[f"{observable}.{vue}"] = abs(
                premiere["vues"][observable][vue]["max"]
                - seconde["vues"][observable][vue]["max"])
    plancher = max(ecarts.values())
    return {
        "plancher_delta_chi": float(plancher),
        "ecarts_par_vue": ecarts,
        "n_passes": len(passes_courantes),
        "mesure": "réplicat du bras courant, tout identique",
        "justification": (
            "D-1(a) a établi le readout PROPRE (Δχ à blanc exactement "
            "nul) : aucun plancher d'instrument à retrancher. Le seul "
            "bruit restant est la reproductibilité de bout en bout, "
            "MESURÉE ici — analogue Δχ de la dérive machine (~0.18 ms) "
            "côté temps."),
        "note_si_nul": (
            "un plancher NUL dit que la chaîne est DÉTERMINISTE, pas "
            "qu'elle est infiniment précise : le facteur relatif porte "
            "alors seul la décision, et c'est dit."),
        "deterministe": bool(plancher == 0.0),
    }


def verrou_invariant(bras: dict[str, dict], plancher: float) -> dict:
    """VERROU AMONT (câblage B9). Le schéma B4 pose que la `reference`
    device EST la connaissance du CPU. Si le vivant reconstruit s'en
    écarte au-delà du bruit, alors les deux bras ne mesurent plus le
    schéma mais le défaut de RECONSTRUCTION — qui les affecte tous les
    deux et les rendrait indistinguables. Aucune attribution n'est
    prononçable dans cet état."""
    ecarts = {nom: mesure["invariant_b4"]["max"]
              for nom, mesure in bras.items()}
    tenu = {nom: bool(valeur <= plancher) for nom, valeur in ecarts.items()}
    respecte = all(tenu.values())
    return {
        "invariant": ("vivant reconstruit ≡ livre de comptes (reference "
                      "device) — le schéma B4 pose qu'ils sont le MÊME "
                      "objet"),
        "delta_chi_ecart_max": ecarts,
        "plancher_bruit": plancher,
        "tenu_par_bras": tenu,
        "respecte": bool(respecte),
        "consequence_si_rompu": (
            "le « vivant » n'est PAS la connaissance du CPU : il porte en "
            "plus le défaut de reconstruction, qui frappe les DEUX bras "
            "de la même façon et les rend indistinguables. La "
            "soustraction n'isolerait plus le schéma — elle mesurerait la "
            "reconstruction. ATTRIBUTION NON PRONONÇABLE, remonter."),
    }


def exiger_attribution_prononcable(verrou: dict) -> None:
    """Câblage « tranche muette ⇒ remonter » (B9) : aucun verdict
    d'attribution n'est consommable tant que l'invariant est rompu."""
    if not verrou.get("respecte"):
        raise RuntimeError(
            "ATTRIBUTION NON PRONONÇABLE : l'invariant du schéma B4 "
            "(vivant reconstruit ≡ livre de comptes) est ROMPU — "
            f"Δχ d'écart {verrou['delta_chi_ecart_max']} au-dessus du "
            f"plancher de bruit {verrou['plancher_bruit']}. Les deux bras "
            "portent le même défaut de reconstruction et sont "
            "indistinguables : la soustraction ne mesurerait pas le "
            "schéma. Remonter, ne rien prononcer.")


LECTURE_EFFONDREMENT: str = (
    "l'erreur S'EFFONDRE sur le bras témoin ⇒ LA RÉFÉRENCE INCRÉMENTALE "
    "DÉRIVE — défaut de mécanisme, load-bearing pour tout le lointain de "
    "la fovéa-z")
LECTURE_PERSISTANCE: str = (
    "l'erreur PERSISTE sur le bras témoin ⇒ ERREUR INHÉRENTE au grossier, "
    "et c'est l'EXIGENCE DE FIDÉLITÉ qui est à redéfinir")


def lecture_attribution(bras: dict[str, dict], verrou: dict,
                        plancher: float) -> dict:
    """Lecture MÉCANIQUE pré-écrite, appliquée sans interprétation — et
    REFUSÉE tant que le verrou amont n'est pas tenu."""
    def valeur(nom_bras: str, vue: str) -> float:
        return bras[nom_bras]["vues"][OBSERVABLE_RECONSTRUIT][vue]["max"]

    comparaisons = {}
    for vue in VUES:
        courant, temoin = valeur("courant", vue), valeur("temoin", vue)
        seuil = courant * FACTEUR_EFFONDREMENT_ATTRIBUTION
        # DEUX conditions, pas une : un rapport spectaculaire entre deux
        # nombres tous deux dans le bruit ne dit rien.
        comparaisons[vue] = {
            "courant_max": courant,
            "temoin_max": temoin,
            "seuil_effondrement": seuil,
            "ecart_absolu": courant - temoin,
            "sous_le_facteur": bool(temoin <= seuil),
            "au_dessus_du_bruit": bool((courant - temoin) > plancher),
            "seffondre": bool(temoin <= seuil
                              and (courant - temoin) > plancher),
        }

    discriminante = comparaisons[VUE_DISCRIMINANTE]
    prononcable = bool(verrou["respecte"])
    return {
        "prononcable": prononcable,
        "label": ("ATTRIBUTION NON PRONONÇABLE" if not prononcable else
                  ("RÉFÉRENCE INCRÉMENTALE QUI DÉRIVE"
                   if discriminante["seffondre"] else
                   "ERREUR INHÉRENTE AU GROSSIER")),
        "lecture": (
            verrou["consequence_si_rompu"] if not prononcable else
            (LECTURE_EFFONDREMENT if discriminante["seffondre"]
             else LECTURE_PERSISTANCE)),
        "vue_discriminante": VUE_DISCRIMINANTE,
        "observable_de_la_regle": OBSERVABLE_RECONSTRUIT,
        "note_observable": (
            "la règle porte sur l'observable RECONSTRUIT — ce que le "
            "consommateur du lointain détient réellement. Le livre de "
            "comptes est mesuré et REPORTÉ (bras.*.vues.livre_de_comptes) "
            "mais AUCUNE branche ne lui est appliquée : substituer "
            "l'observable après avoir vu le pré-enregistré échouer serait "
            "un observable post-hoc. Le changer est une décision de "
            "Romain, jamais un effet de bord de ce driver."),
        "comparaisons_par_vue": comparaisons,
        "facteur_effondrement": FACTEUR_EFFONDREMENT_ATTRIBUTION,
        "plancher_bruit_delta_chi": plancher,
        "verrou_invariant": verrou,
        "note_domaine": (
            "seule la vue COMPLÈTE a un domaine IDENTIQUE entre les deux "
            "bras. Les vues cycle et cumulé se restreignent aux cellules "
            "qu'un coefficient a touchées, et le témoin en touche "
            "mécaniquement PLUS (il n'a pas de seuil) : leur soustraction "
            "compare des domaines différents. Les fractions couvertes des "
            "deux bras sont reportées côte à côte pour que l'écart de "
            "domaine soit LISIBLE, jamais supposé nul."),
        "fractions_couvertes": {
            nom: {vue: mesure["vues"][OBSERVABLE_RECONSTRUIT][vue].get(
                "fraction_couverte_mediane")
                for vue in ("cycle", "cumule")}
            for nom, mesure in bras.items()},
        "seuil_ic_bas": SEUIL_IC_BAS,
        "rappel_portee": (
            "ce driver REPORTE. EPS reste figé à 1e-4 pour le régime, k "
            "reste figé à 4, rien n'est réglé — et eps = 0 est un "
            "appareil de mesure, jamais un régime proposable. La décision "
            "appartient à Romain."),
    }


def main() -> None:
    cp = exiger_cupy()
    exiger_pass(VERIFS_EXIGEES)
    exiger_rien_regle(EPS_COURANT, CADENCE_FIGEE_ATTRIBUTION)
    exiger_rien_regle(EPS_TEMOIN, CADENCE_FIGEE_ATTRIBUTION)

    geo = geometrie_v4()
    facteur = max(geo.n_fov // N0_DEFAUT, 1)
    print(f"attribution (b) : COURANT eps={EPS_COURANT:g} contre TÉMOIN "
          f"eps={EPS_TEMOIN:g} (remontée pleine), k="
          f"{CADENCE_FIGEE_ATTRIBUTION} FIGÉ", flush=True)

    passes_courantes: list[dict] = []
    for numero in range(N_PASSES_REPLICAT):
        print(f"  bras COURANT, passe {numero + 1}/{N_PASSES_REPLICAT} "
              f"(réplicat = étalon de bruit) ...", flush=True)
        passes_courantes.append(
            mesurer_bras(cp, EPS_COURANT, SERIE_FRAMES, facteur,
                         CADENCE_FIGEE_ATTRIBUTION))
        liberer_vram(cp)

    print("  bras TÉMOIN (remontée pleine, non incrémentale) ...", flush=True)
    bras = {"courant": passes_courantes[0],
            "temoin": mesurer_bras(cp, EPS_TEMOIN, SERIE_FRAMES, facteur,
                                   CADENCE_FIGEE_ATTRIBUTION)}
    liberer_vram(cp)

    bruit = plancher_bruit(passes_courantes)
    verrou = verrou_invariant(bras, bruit["plancher_delta_chi"])
    lecture = lecture_attribution(bras, verrou, bruit["plancher_delta_chi"])
    mempool = cp.get_default_memory_pool()

    document = {
        "meta": {
            "mesure": ("attribution (b) — la référence incrémentale "
                       "dérive-t-elle ? (§A19-lecture-diagnostic, "
                       "pocCascade2phys 4062747)"),
            "protocoles": PROTOCOLES,
            "portee": (
                "falsificateur de l'attribution (b), NOMMÉ dès la table "
                "de branches. EPS reste figé à 1e-4 pour le régime, k "
                "reste figé à 4, RIEN n'est réglé — eps = 0 est un "
                "appareil de mesure, jamais un régime proposable."),
            "invariance_du_protocole": (
                "mêmes cellules, même cadence, même série, même état "
                "initial (même graine, même géométrie, même trajectoire) "
                "— seul le SCHÉMA change"),
            "plancher_bruit": bruit,
            "cellule": {"n_fov": geo.n_fov, "n_niv": geo.n_niv,
                        "decimation": facteur},
            "frames": {"warmup_exclu": WARMUP_FRAMES, "serie": SERIE_FRAMES},
            "kernels": ("F, L3, chrono, substrats, pyramide INTOUCHÉS — "
                        "preuve par git diff sur src/, pas par empreinte. "
                        "Seul sha256 réellement verrouillé, portant sur la "
                        "chaîne CUDA EXTRAITE et non sur le fichier : "
                        "sha256(substrat_fusionne._SOURCE) = e18015f5..."
                        "f30b4"),
            "verifs_exigees": list(VERIFS_EXIGEES),
        },
        "bras": bras,
        "passes_courantes": passes_courantes,
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
    print("F1 -- ATTRIBUTION (b) : la référence incrémentale dérive-t-elle ?")
    print("=" * 78)
    print(f"  plancher de bruit Δχ (réplicat MESURÉ) = "
          f"{bruit['plancher_delta_chi']:.6f}")
    if bruit["deterministe"]:
        print(f"    {bruit['note_si_nul']}")
    print(f"  {'vue':>9}  {'courant':>9} {'témoin':>9}  {'seuil':>9}  "
          f"{'effondre':>9}")
    for vue in VUES:
        comparaison = lecture["comparaisons_par_vue"][vue]
        marque = " <-- discriminante" if vue == VUE_DISCRIMINANTE else ""
        print(f"  {vue:>9}  {comparaison['courant_max']:>9.6f} "
              f"{comparaison['temoin_max']:>9.6f}  "
              f"{comparaison['seuil_effondrement']:>9.6f}  "
              f"{str(comparaison['seffondre']):>9}{marque}")
    print(f"  fractions couvertes (domaines) : "
          f"{lecture['fractions_couvertes']}")
    print(f"    {lecture['note_domaine']}")
    print("-" * 78)
    print(f"  VERROU B4 : invariant respecté = {verrou['respecte']}  "
          f"(Δχ d'écart {verrou['delta_chi_ecart_max']})")
    print(f"  -> {lecture['label']}")
    print(f"     {lecture['lecture']}")
    print("  Le driver REPORTE — aucun réglage, aucune autre conclusion.")
    print(f"[REPORT] -> {OUT_JSON_PATH}")
    print("POINT D'ARRÊT OBLIGATOIRE : lecture remontée à Romain.")


if __name__ == "__main__":
    main()
