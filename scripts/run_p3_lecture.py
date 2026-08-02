"""P3 — LECTURE MÉCANIQUE du transport du pin à travers R1.

Pré-enregistrement qui fait foi : `claude/prereg-p3-transport-pin.md` (ENDOSSÉ,
gravé §A38). Ce script PRONONCE ; il n'interprète pas. Sa raison d'être est
négative : **sans lui, la session finirait en lecture à la main** — et une
lecture à la main sur un chiffre qu'on vient de produire soi-même n'est pas une
mesure.

ORDRE DU PRONONCÉ, et l'ordre est le contrat :

  1. GARDES DE VALIDITÉ, AVANT toute lecture de seuil — chaque sidecar lié à
     son log par sha256 (chantier 3) ; `validite` §C5 de chaque staircase ;
     la chaîne `--conditions` est ANCRÉE sur `date_session` (date + heure ±2 h,
     zéro marqueur de gabarit — §A41) ; l'ordre des chemins est bien
     (r1, r1, viridis, r1). Puis, dans la lecture du bras : CV <= 30 % (§C5).
  2. LE TÉMOIN D'ABORD — la staircase viridis en 3e position. Son seuil doit
     tomber dans l'IC gravé du pin, sinon la session est INDÉTERMINÉE et **les
     seuils R1 ne sont ni imprimés ni écrits** : quand l'instrument est
     invalide, on ne regarde pas le résultat. C'est la seule protection
     efficace contre soi-même.
  3. Témoin valide ⇒ IC des trois staircases R1, branches du prereg prononcées
     telles quelles.
  4. Écriture de `outputs/arcC/p3_transport_pin.lecture.json`. La version
     `claude/lectures/` se recopie APRÈS verdict, jamais par la machine.

POURQUOI LA GARDE DES CONDITIONS EST ANCRÉE SUR `date_session` (§A41). La
version « chevrons seuls » a laissé passer les ellipses du 26/07 (§A40), et
toute garde purement MORPHOLOGIQUE reste contournable (crochets, « p. ex. »,
copié-collé d'une observation d'hier). La vérité de référence est
`date_session` — un horodatage machine, jamais tapé par un humain : la chaîne
`--conditions` doit contenir la DATE du jour de session et une HEURE cohérente
avec cet horodatage à ±2 h. C'est la garde qui aurait attrapé à la fois la
fuite du gabarit, la session de nuit du 26/07 et le fait de conditions faux du
journal (§A41). Les deux pré-vols du 25/07 ET la session P3 du 26/07 sont ses
champs d'essai posthumes : tous trois doivent être REFUSÉS par elle. NOTE DE
PROVENANCE : cette garde et la garde de dispersion ci-dessous sont POSTÉRIEURES
au verdict P3 du 26/07 (review du 29/07, constats M2/M3) — le verdict archivé
(`claude/lectures/p3_transport_pin.lecture.json`) reste le prononcé qui fait
foi ; rejouer la lecture sur le manifeste du 26/07 refuse désormais la session
à la garde des conditions au lieu du témoin, INDÉTERMINÉE dans les deux cas.

MÉTHODE D'IC — RÉTABLIE (§A38-CORRECTION-4). Le prereg disait « même méthode
d'IC (mean±2SEM combiné) » ; l'IC GRAVÉ du pin, [6.03, 8.67] %, est en fait
l'IC PRIMAIRE **min/max** (`pins_spatial.json` : `ic_combine` est ABSENT, la
branche à-cheval n'a jamais tourné). La lettre était fautive, pas la mesure :
**min/max DÉCIDE des deux côtés** — le bras R1 est réduit par la méthode même
qui a produit son référent, et « même méthode d'IC » redevient vrai au sens
littéral. Le `mean±2SEM` reste calculé et SURFACÉ en diagnostic, jamais jugeant
(cf. `DIAGNOSTIC_IC`)."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.arcC_abx import (BASE_SEED_P3PRIME, evalue_dispersion, heures_declarees_minutes,
                          verifie_observation_conditions)
from src.arcC_scelle import descelle_seuils
from scripts.run_arcC_orchestration import PLAN_P3PRIME, pauses_inter_staircases
from scripts.run_arcC_pins import calcule_ic, calcule_ic_combine

# --- Constantes RECOPIÉES (unités = FRACTION : 0.04 = 4 %) -------------------

# Le pin gravé, §C11/§C12, artefact `outputs/arcC/pins_spatial.json` (campagne
# du 2026-07-05, harnais 3d2a6ed8). Valeurs PLEINE PRÉCISION de l'artefact ; un
# test vérifie qu'elles arrondissent bien au 7.33 % / [6.03, 8.67] du prereg.
PIN_JND_SEV: float = 0.07334065957385666
PIN_IC: tuple[float, float] = (0.06033229334102097, 0.08673568621862042)

# La disposition GRAVÉE des staircases (prereg P3, précisions du 25/07 soir).
CHEMINS_ATTENDUS: tuple[str, ...] = ("r1", "r1", "viridis", "r1")
REGIME_P3: str = "severe"
CHEMIN_TEMOIN: str = "viridis"
CHEMIN_MESURE: str = "r1"
# Le bras R1 en compte TROIS (prereg P3 : « l'IC des TROIS staircases R1 »),
# soit exactement le nombre de staircases de la campagne du pin.
N_STAIRCASES_R1: int = 3

VERDICT_COMPATIBLE: str = "TRANSPORT_COMPATIBLE_AVEC_1"
VERDICT_DIFFERENT: str = "TRANSPORT_DIFFERENT_DE_1"
VERDICT_INDETERMINEE: str = "INDETERMINEE"

# Le bras R1 est réduit par MIN/MAX — la méthode du pin telle que son artefact
# la déclare (§A38-CORRECTION-4). Le mean±2SEM est calculé quand même et
# surfacé : il documente ce que l'autre méthode aurait donné, et il garde
# visible, dans chaque rapport, la trace de la lettre fautive. Chiffre de
# référence : sur les trois seuils DU PIN, min/max donne [6.03, 8.67] % (l'IC
# gravé) et mean±2SEM aurait donné [5.81, 8.86] % — plus large, donc favorable
# au recouvrement, donc à la branche 1, la moins dérangeante. C'est ce biais-là
# que la bascule retire.
DIAGNOSTIC_IC: str = (
    "DECIDEUR = min/max des seuils, la methode du pin telle que "
    "pins_spatial.json la declare (methode_ic_primaire = min_max_seuils ; "
    "'ic_combine' ABSENT, branche a-cheval jamais active) -- les DEUX cotes de "
    "la comparaison sont donc reduits par la MEME methode (§A38-CORRECTION-4). "
    "Le mean+-2SEM ci-dessous est SURFACE en diagnostic, il ne juge rien : sur "
    "les seuils du pin eux-memes il aurait donne [5.81, 8.86] % au lieu de "
    "[6.03, 8.67] % -- plus large, donc favorable au recouvrement.")


# --- Résolution des fichiers -------------------------------------------------


def resout_chemin(chemin_enregistre: str, racine: Path) -> Path | None:
    """Résout un chemin consigné au manifeste. Les campagnes écrivent des
    chemins tels quels ; si les sorties ont été DÉPLACÉES depuis (c'est le cas
    des deux pré-vols du 25/07, rangés dans leur propre dossier), le chemin
    enregistré ne résout plus.

    On tente donc le chemin tel quel, puis le même NOM DE FICHIER à côté du
    manifeste. `None` si aucun des deux n'existe — et c'est alors une garde qui
    échoue, jamais un silence : sans le fichier, il n'y a rien à vérifier."""
    direct = Path(chemin_enregistre)
    if direct.exists():
        return direct
    voisin = racine / direct.name
    return voisin if voisin.exists() else None


def sha256_fichier(chemin: Path) -> str:
    """sha256 d'un fichier lu en OCTETS — même calcul que celui qui a écrit
    `sha256_log` à la clôture de session (`scripts/run_arcC_session.py`)."""
    return hashlib.sha256(chemin.read_bytes()).hexdigest()


# --- Gardes de validité (elles passent AVANT tout seuil) ---------------------


def garde_conditions(manifeste: dict) -> dict:
    """Les conditions de session sont une OBSERVATION DATÉE, pas un gabarit —
    la garde est ANCRÉE sur `date_session`, l'horodatage machine que personne
    ne tape (§A41 : « la machine lit, la session ne recopie pas »). La logique
    vit dans `verifie_observation_conditions` (src/arcC_abx.py, IMPORTÉE,
    jamais recopiée) : date du jour exigée dans la chaîne, heure déclarée
    cohérente avec `date_session` à ±2 h, zéro marqueur de gabarit dans
    `conditions` ET `luminosite`."""
    validite = manifeste.get("conditions_validite") or {}
    conditions = validite.get("conditions")
    luminosite = validite.get("luminosite")
    texte = "" if conditions is None else str(conditions)

    date_session = None
    motifs: list[str] = []
    try:
        date_session = datetime.fromisoformat(str(manifeste.get("date_session")))
    except (TypeError, ValueError):
        motifs.append("date_session ILLISIBLE au manifeste -- rien pour ancrer la garde")
    if date_session is not None:
        # Portée (mission 8a.4) : la clause (d) — plage horaire — est GRAVÉE
        # pour P3′ ; le mode P3 HISTORIQUE relit son manifeste archivé avec
        # les clauses (a)-(c) de son époque, plage désactivée EXPLICITEMENT.
        motifs.extend(verifie_observation_conditions(conditions, luminosite, date_session,
                                                     plage_horaire=None))
    return dict(nom="conditions_ancrees_date_session", ok=(not motifs),
                conditions=texte, luminosite=luminosite,
                heures_declarees_minutes=heures_declarees_minutes(texte),
                date_session=(date_session.isoformat() if date_session else None),
                motif=(None if not motifs else " ; ".join(motifs)))


def garde_chemins(manifeste: dict) -> dict:
    """L'ordre des chemins doit être EXACTEMENT la disposition gravée
    (r1, r1, viridis, r1). Un témoin ailleurs ne serait plus le témoin
    pré-enregistré, et la session mesurerait une autre chose sous le même nom."""
    chemins = (manifeste.get("config_affichage") or {}).get("chemins_staircases")
    obtenu = tuple(chemins) if chemins else None
    return dict(nom="ordre_des_chemins", ok=(obtenu == CHEMINS_ATTENDUS),
                attendu=list(CHEMINS_ATTENDUS), obtenu=(list(obtenu) if obtenu else None),
                motif=(None if obtenu == CHEMINS_ATTENDUS else
                       f"disposition {obtenu} != disposition gravee {CHEMINS_ATTENDUS}"))


def garde_validite_c5(sessions: list[dict]) -> dict:
    """Validité §C5 de CHAQUE staircase (taux de catch >= 90 %). Une seule
    staircase invalide suffit : la moyenne d'un bras avec une staircase
    invalide n'est pas un seuil, c'est un mélange."""
    invalides = [s["numero_staircase"] for s in sessions
                 if not (s.get("validite") or {}).get("valide")]
    if not sessions:
        motif = "aucune staircase dans le regime lu"
    elif invalides:
        motif = f"staircases INVALIDES (§C5, taux de catch < 90 %) : {invalides}"
    else:
        motif = None
    return dict(nom="validite_c5", ok=(not invalides and bool(sessions)),
                n_staircases=len(sessions), staircases_invalides=invalides, motif=motif)


def garde_sidecars(sessions: list[dict], racine: Path) -> dict:
    """Chaque staircase doit avoir son sidecar, et le `sha256_log` du sidecar
    doit être celui de SON log, recalculé ici (chantier 3).

    C'est ce qui interdit l'appariement par nom de fichier : un log réécrit,
    tronqué ou interverti depuis la session devient visible. Le sidecar doit
    aussi déclarer le MÊME chemin de rendu que le manifeste — sinon on ne
    saurait plus quel bras a traversé quoi."""
    details, ok = [], True
    for session in sessions:
        k = session["numero_staircase"]
        log = resout_chemin(session.get("log_path", ""), racine)
        sidecar_path = resout_chemin(session.get("conditions_path", "") or "", racine)
        entree = dict(numero_staircase=k, log_trouve=bool(log),
                      sidecar_trouve=bool(sidecar_path))
        if log is None or sidecar_path is None:
            entree["motif"] = "log ou sidecar INTROUVABLE"
            ok = False
        else:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            attendu = sha256_fichier(log)
            entree.update(sha256_log_sidecar=sidecar.get("sha256_log"),
                          sha256_log_recalcule=attendu,
                          chemin_sidecar=(sidecar.get("provenance_rendu") or {}).get(
                              "chemin_rendu"),
                          chemin_manifeste=session.get("chemin_rendu"))
            if sidecar.get("sha256_log") != attendu:
                entree["motif"] = "sha256_log du sidecar != sha du log relu"
                ok = False
            elif entree["chemin_sidecar"] != entree["chemin_manifeste"]:
                entree["motif"] = "chemin de rendu : sidecar et manifeste se contredisent"
                ok = False
        details.append(entree)
    return dict(nom="liaison_sidecar_log", ok=(ok and bool(sessions)), details=details,
                motif=(None if ok and sessions else
                       "au moins une staircase n'est pas liee a son log"))


def evalue_gardes(manifeste: dict, sessions: list[dict], racine: Path) -> dict:
    """TOUTES les gardes sont évaluées, jamais court-circuitées à la première
    qui tombe : un rapport qui s'arrête au premier échec cache les suivants, et
    la session serait relancée une fois par garde."""
    gardes = [garde_conditions(manifeste), garde_chemins(manifeste),
              garde_validite_c5(sessions), garde_sidecars(sessions, racine)]
    echouees = [g["nom"] for g in gardes if not g["ok"]]
    return dict(toutes_ok=(not echouees), echouees=echouees, gardes=gardes)


# --- Lecture du témoin, puis du transport ------------------------------------


def _seuils_complets(sessions: list[dict], chemin: str) -> list[float]:
    """Seuils des staircases COMPLÈTES d'un bras — même filtre que le pin
    (`construit_regime_pins` : `complet` et `seuil is not None`). Un escalier
    non convergent n'a pas de seuil à moyenner."""
    return [s["seuil"] for s in sessions
            if s.get("chemin_rendu") == chemin and s.get("complet") and s.get("seuil") is not None]


def lecture_temoin(sessions: list[dict]) -> dict:
    """LE TÉMOIN D'ABORD (prereg P3, tradition §A31) : la staircase viridis
    doit retomber dans l'IC gravé du pin. Sinon, le sujet ou l'écran a dérivé
    depuis Arc C, et AUCUNE lecture de transport n'est prononçable — un écart
    R1-vs-pin serait inattribuable (chemin ? dérive ?)."""
    seuils = _seuils_complets(sessions, CHEMIN_TEMOIN)
    if len(seuils) != 1:
        return dict(ok=False, n_staircases_temoin=len(seuils), seuil=None,
                    motif=("le bras temoin doit avoir EXACTEMENT une staircase complete "
                           f"(trouve {len(seuils)}) -- escalier non convergent ou "
                           "disposition inattendue"))
    seuil = float(seuils[0])
    dans_ic = PIN_IC[0] <= seuil <= PIN_IC[1]
    return dict(ok=dans_ic, seuil=seuil, seuil_pct=100.0 * seuil,
                ic_pin=list(PIN_IC), ic_pin_pct=[100.0 * b for b in PIN_IC],
                motif=(None if dans_ic else
                       f"seuil temoin {100.0 * seuil:.2f} % HORS de l'IC grave du pin "
                       f"[{100.0 * PIN_IC[0]:.2f}, {100.0 * PIN_IC[1]:.2f}] % -- le sujet ou "
                       "l'ecran a derive depuis Arc C"))


def lecture_transport(seuils_r1: list[float]) -> dict:
    """Les branches du prereg P3, prononcées mécaniquement.

    Le DÉCIDEUR est l'IC **min/max** (`calcule_ic`, IMPORTÉ de `run_arcC_pins`,
    jamais réimplémenté) : la méthode du pin telle que son artefact la déclare
    (§A38-CORRECTION-4). Les deux côtés de la comparaison sont ainsi réduits par
    la même méthode. Le `mean±2SEM` est calculé aussi, et SURFACÉ SEULEMENT — il
    n'entre dans aucune décision (cf. `DIAGNOSTIC_IC`).

    Le bras R1 doit avoir ses TROIS staircases complètes. Deux ne suffisent
    pas : le prereg lit « l'IC des TROIS staircases R1 », et un min/max sur un
    bras amputé est un intervalle sur autre chose. Un seul seuil rendrait même
    l'intervalle dégénéré `[x, x]` — un IC ne se fabrique pas.

    GARDE DE DISPERSION (§C5, review 29/07 M3) : la méthodologie du pin exige
    CV <= 30 % (`evalue_dispersion`, IMPORTÉE, jamais réimplémentée) — le pin
    aurait été INDÉTERMINÉ au-delà ; un bras R1 plus dispersé que ce que le
    pin se serait permis ne se lit pas contre le pin. Fait de référence : le
    bras R1 du 26/07 était à CV 42.7 % — cette garde n'existait pas alors."""
    if len(seuils_r1) != N_STAIRCASES_R1:
        return dict(verdict=VERDICT_INDETERMINEE, branche="3", ic=None,
                    n_staircases=len(seuils_r1),
                    motif=(f"le bras R1 a {len(seuils_r1)} staircase(s) COMPLETE(s), il en "
                           f"faut {N_STAIRCASES_R1} -- escalier non convergent (prereg P3, "
                           "lecture 3). Un IC ne se fabrique pas sur un bras ampute."))
    dispersion = evalue_dispersion(seuils_r1)
    cv = dispersion["dispersion_relative"]
    dispersion_rapport = dict(
        n=dispersion["n"], cv=(None if not np.isfinite(cv) else float(cv)),
        cv_pct=(None if not np.isfinite(cv) else 100.0 * float(cv)),
        seuil_cv=dispersion["seuil_dispersion_relative"], coherent=dispersion["coherent"])
    if not dispersion["coherent"]:
        return dict(verdict=VERDICT_INDETERMINEE, branche="3", ic=None,
                    n_staircases=len(seuils_r1), dispersion=dispersion_rapport,
                    motif=(f"DISPERSION du bras R1 hors clous : CV = {100.0 * cv:.1f} % > "
                           f"{100.0 * dispersion['seuil_dispersion_relative']:.0f} % (§C5) -- "
                           "la methode du pin aurait rendu ce bras INDETERMINE ; il ne se lit "
                           "pas contre le pin."))
    ic = calcule_ic(seuils_r1)
    ic_diagnostic = calcule_ic_combine(seuils_r1)
    jnd = float(np.mean(seuils_r1))
    recouvre = not (ic[1] < PIN_IC[0] or ic[0] > PIN_IC[1])
    commun = dict(n_staircases=len(seuils_r1), seuils=[float(s) for s in seuils_r1],
                  dispersion=dispersion_rapport,
                  jnd_r1=jnd, jnd_r1_pct=100.0 * jnd,
                  ic_minmax_decideur=ic, ic_minmax_decideur_pct=[100.0 * b for b in ic],
                  ic_mean2sem_surface=ic_diagnostic, ic_pin=list(PIN_IC),
                  methode_ic="min_max_seuils (celle du pin)", diagnostic_ic=DIAGNOSTIC_IC)
    if recouvre:
        return dict(commun, verdict=VERDICT_COMPATIBLE, branche="1", texte=(
            "IC(R1) recoupe l'IC grave [6.03, 8.67] % ⇒ TRANSPORT COMPATIBLE AVEC 1. "
            "Consequences gravees d'avance : la condition « transport ≈ 1 » du caveat D14 "
            "est ETABLIE ; la SECONDE condition (« ponderation plate ») reste NON MESUREE "
            "-- P3 presente un stimulus foveal (~2°), il ne dit RIEN de l'excentricite ; "
            "la scission D14 ne se referme donc PAS ici (elle attendrait r_fovea, gate). "
            "Le gate (iii') se mesure avec le pin grave tel quel a travers R1 : le juge se "
            "durcit, le contrat ne s'affaiblit pas."))
    facteur = jnd / PIN_JND_SEV
    # « IC par les bornes » : les bornes de l'IC DÉCIDEUR, divisées par le pin.
    return dict(commun, verdict=VERDICT_DIFFERENT, branche="2",
                transport_T=facteur,
                transport_T_ic=[ic[0] / PIN_JND_SEV, ic[1] / PIN_JND_SEV],
                direction=("seuil PLUS HAUT a travers R1 (moins sensible)" if facteur > 1.0
                           else "seuil PLUS BAS a travers R1 (plus sensible)"),
                texte=(
                    f"IC(R1) DISJOINT de l'IC grave ⇒ TRANSPORT ≠ 1, T = {facteur:.4f} "
                    "(IC par les bornes). Consequences gravees : le gate (iii') recoit SON "
                    "echelle (jnd_sev^R1) ; r_fovea recoit son consommateur. AUCUN seuil "
                    "d'etat ne bouge -- 0.0733 et tous les seuils T2/F1 restent les seuils "
                    "de l'INSTRUMENT Δχ-albedo, graves pour leur config exacte. "
                    "jnd_sev^R1 est un pin NOUVEAU (le pin-projection), il ne remplace "
                    "rien retroactivement."))


# --- Prononcé complet --------------------------------------------------------


def prononce(manifeste: dict, racine: Path) -> dict:
    """Le prononcé entier, dans l'ordre du contrat. Les seuils R1 n'entrent
    dans le rapport QUE si les gardes ET le témoin sont passés : quand
    l'instrument est invalide, le résultat ne se regarde pas — ni à l'écran, ni
    dans le JSON, où il serait tout aussi consultable."""
    sessions = ((manifeste.get("regimes") or {}).get(REGIME_P3) or {}).get("sessions") or []
    rapport = dict(
        prereg="claude/prereg-p3-transport-pin.md (ENDOSSE, §A38)",
        date=datetime.now().isoformat(), machine=platform.node(),
        regime=REGIME_P3, n_staircases=len(sessions),
        pin=dict(jnd=PIN_JND_SEV, jnd_pct=100.0 * PIN_JND_SEV, ic=list(PIN_IC),
                 ic_pct=[100.0 * b for b in PIN_IC], methode_ic_du_pin="min_max_seuils"),
        gardes=evalue_gardes(manifeste, sessions, racine))

    if not rapport["gardes"]["toutes_ok"]:
        rapport.update(verdict=VERDICT_INDETERMINEE, branche="3", motif=(
            "GARDES DE VALIDITE ECHOUEES : " + ", ".join(rapport["gardes"]["echouees"]) +
            ". Aucun seuil n'est lu -- ni le temoin, ni R1."))
        return rapport

    temoin = lecture_temoin(sessions)
    rapport["temoin"] = temoin
    if not temoin["ok"]:
        rapport.update(verdict=VERDICT_INDETERMINEE, branche="3", motif=(
            "TEMOIN INVALIDE : " + str(temoin["motif"]) + ". Les seuils R1 ne sont PAS lus "
            "-- un ecart R1-vs-pin serait inattribuable (chemin ? derive ?)."))
        return rapport

    transport = lecture_transport(_seuils_complets(sessions, CHEMIN_MESURE))
    rapport["r1"] = transport
    rapport["verdict"] = transport["verdict"]
    rapport["branche"] = transport["branche"]
    rapport["motif"] = transport.get("texte") or transport.get("motif")
    return rapport


# --- Mode P3′ (mission chantier 8b ; prereg P3′ v2.2 ENDOSSÉ §A42) -----------

# Marge d'équivalence D-P3′-2, TRANCHÉE §A42 : λ = 1.5 — « établie à un
# facteur <= 1.5 près ». [1/1.5, 1.5] est une fenêtre de largeur 2.25 face à
# un IC_T attendu de largeur ~2.5-3 : la branche 1b n'est atteignable que par
# une session à faible dispersion centrée sur 1 (prereg v2.1).
LAMBDA_EQUIV: float = 1.5

VERDICT_NON_DISTINGUE: str = "TRANSPORT_NON_DISTINGUE_DE_1"
VERDICT_EQUIVALENT: str = "TRANSPORT_EQUIVALENT_A_1_MARGE_LAMBDA"
VERDICT_DIFFERENT_ET_BORNE: str = "TRANSPORT_DIFFERENT_DE_1_ET_EQUIVALENT_A_LAMBDA"


def garde_protocole_p3prime(manifeste: dict) -> dict:
    """Le manifeste doit se DÉCLARER p3prime ET porter son bloc de scellement
    — un manifeste P3 historique (ou pré-P3′) lu en mode P3′ meurt ici, à
    voix haute, jamais par un KeyError trois gardes plus loin."""
    protocole = manifeste.get("protocole")
    scellement = manifeste.get("scellement_seuils") or {}
    ok = (protocole == "p3prime" and bool(scellement.get("chemin"))
          and bool(scellement.get("sha256")))
    return dict(nom="protocole_p3prime", ok=ok, protocole=protocole,
                motif=(None if ok else
                       f"manifeste protocole={protocole!r} et scellement="
                       f"{'present' if scellement else 'ABSENT'} -- pas une session P3′ "
                       "(prereg v2.2 : seuils SCELLÉS, protocole déclaré)"))


def garde_conditions_p3prime(manifeste: dict) -> dict:
    """Garde 1 (a)-(d) du prereg P3′ : la version ancrée §A41 PLUS la clause
    (d) — plage horaire gravée [09:00, 19:00] (§A42 [v2.1]), portée par le
    DÉFAUT de `verifie_observation_conditions` (un exemplaire, jamais
    recopié). L'instant vient de `date_session`, l'horodatage machine."""
    validite = manifeste.get("conditions_validite") or {}
    conditions = validite.get("conditions")
    luminosite = validite.get("luminosite")
    texte = "" if conditions is None else str(conditions)
    date_session = None
    motifs: list[str] = []
    try:
        date_session = datetime.fromisoformat(str(manifeste.get("date_session")))
    except (TypeError, ValueError):
        motifs.append("date_session ILLISIBLE au manifeste -- rien pour ancrer la garde")
    if date_session is not None:
        motifs.extend(verifie_observation_conditions(conditions, luminosite, date_session))
    return dict(nom="conditions_ancrees_a_d", ok=(not motifs),
                conditions=texte, luminosite=luminosite,
                date_session=(date_session.isoformat() if date_session else None),
                motif=(None if not motifs else " ; ".join(motifs)))


def garde_graine_p3prime(manifeste: dict) -> dict:
    """Garde de graine à la LECTURE : ÉGALITÉ SEULE au `base_seed` gravé
    (§A42-COMPLÉMENT) — elle pinne tout : une graine quelconque (12345) ET
    20260705 tombent par l'égalité. La liste des brûlées ne mord PAS ici :
    sinon l'acte de brûlage rendrait l'archive P3′ illisible — la
    re-dérivabilité promise exige qu'une archive à 20260729 reste re-jouable
    après brûlage. (La liste mord à L'ENTRÉE : `valide_graine_entree_p3prime`.)"""
    base_seed = manifeste.get("base_seed")
    ok = (base_seed == BASE_SEED_P3PRIME)
    return dict(nom="graine_egalite_gravee", ok=ok, base_seed=base_seed,
                base_seed_grave=BASE_SEED_P3PRIME,
                motif=(None if ok else
                       f"base_seed {base_seed!r} != {BASE_SEED_P3PRIME} (graine GRAVÉE du "
                       "prereg P3′ v2.2, §A42) -- une session P3′ ne se lit que sur SA "
                       "graine"))


def garde_chemins_p3prime(manifeste: dict) -> dict:
    """L'ordre des chemins == le plan D-P3′-1 GRAVÉ (V,R,R,V,V,R, §A42),
    IMPORTÉ de l'orchestration — un plan déplacé en silence n'est plus le
    protocole (l'acquis de P3)."""
    chemins = (manifeste.get("config_affichage") or {}).get("chemins_staircases")
    obtenu = tuple(chemins) if chemins else None
    ok = (obtenu == PLAN_P3PRIME)
    return dict(nom="plan_d_p3prime_1", ok=ok, attendu=list(PLAN_P3PRIME),
                obtenu=(list(obtenu) if obtenu else None),
                motif=(None if ok else
                       f"plan {obtenu} != plan D-P3′-1 gravé {PLAN_P3PRIME} (§A42)"))


def garde_bras_complets(sessions: list[dict]) -> dict:
    """Clause du bras complet, étendue aux DEUX bras (prereg P3′) : trois
    staircases COMPLÈTES par bras, comptées sur le drapeau `complet` — les
    seuils, eux, restent SCELLÉS à ce stade."""
    complets = {"viridis": 0, "r1": 0}
    for s in sessions:
        if s.get("complet") and s.get("chemin_rendu") in complets:
            complets[s["chemin_rendu"]] += 1
    ok = (complets["viridis"] == 3 and complets["r1"] == 3)
    return dict(nom="bras_complets_3_3", ok=ok, complets=complets,
                motif=(None if ok else
                       f"bras amputé : viridis {complets['viridis']}/3, r1 "
                       f"{complets['r1']}/3 COMPLÈTES -- un bras amputé ne se lit pas "
                       "(verdict INDÉTERMINÉE, prereg P3′)"))


def _seuils_par_bras(sessions: list[dict], seuils_descelles: dict) -> dict[str, list[float]]:
    """Apparie les seuils DESCELLÉS aux bras via `numero_staircase` et
    `chemin_rendu` du manifeste — staircases complètes seulement, dans
    l'ordre des positions (le même filtre que le pin)."""
    bras: dict[str, list[float]] = {"viridis": [], "r1": []}
    for s in sorted(sessions, key=lambda x: x["numero_staircase"]):
        seuil = seuils_descelles.get(str(s["numero_staircase"]))
        if s.get("complet") and seuil is not None and s.get("chemin_rendu") in bras:
            bras[s["chemin_rendu"]].append(float(seuil))
    return bras


def prononce_p3prime(manifeste: dict, racine: Path) -> dict:
    """Le prononcé P3′ entier, dans l'ordre du contrat (mission 8b).

    TOUTES les gardes sont ÉVALUÉES et rapportées, aucune ne court-circuite
    les autres (§A38-CORRECTION-4, choix endossé : une mesure humaine ne se
    relance pas une garde à la fois) ; l'ordre ne gouverne que le PRONONCÉ du
    motif principal — et les seuils ne se DESCELLENT que TOUT VERT. La garde
    de dispersion (CV <= 30 % PAR BRAS, les deux bras) vient APRÈS le
    descellement par nécessité (elle lit les seuils) ; si elle tombe, les
    seuils ne sont NI imprimés NI écrits — seul le CV, un ratio, est surfacé.

    Branches (§A42 + §A42-COMPLÉMENT, partition CLOSE sur la grille
    inclusion x recouvrement, bornes INCLUSES) :
      1a  recouvrement sans inclusion  -> NON DISTINGUÉ de 1 (rien ne s'établit)
      1b  recouvrement ET inclusion    -> ÉQUIVALENT à 1 à λ près (D14 établie)
      2   disjoints sans inclusion     -> ≠ 1, direction et facteur
      2-équiv  disjoints ET inclusion  -> PRONONCÉ DOUBLE (résolution β)
      3   toute garde rouge            -> INDÉTERMINÉE"""
    sessions = ((manifeste.get("regimes") or {}).get(REGIME_P3) or {}).get("sessions") or []
    rapport = dict(
        prereg="claude/prereg-p3prime-transport-intra-session.md (ENDOSSÉ v2.2, §A42 + "
               "§A42-COMPLÉMENT)",
        protocole="p3prime", date=datetime.now().isoformat(), machine=platform.node(),
        regime=REGIME_P3, n_staircases=len(sessions),
        lambda_equiv=LAMBDA_EQUIV,
        gardes=None)

    gardes = [garde_protocole_p3prime(manifeste),
              garde_conditions_p3prime(manifeste),
              garde_graine_p3prime(manifeste),
              garde_chemins_p3prime(manifeste),
              garde_validite_c5(sessions),
              garde_bras_complets(sessions),
              garde_sidecars(sessions, racine)]
    echouees = [g["nom"] for g in gardes if not g["ok"]]
    rapport["gardes"] = dict(toutes_ok=(not echouees), echouees=echouees, gardes=gardes)

    if echouees:
        rapport.update(verdict=VERDICT_INDETERMINEE, branche="3", motif=(
            "GARDES DE VALIDITE ECHOUEES : " + ", ".join(echouees) +
            ". Les seuils restent SCELLÉS -- ni lus, ni imprimés, ni écrits."))
        return rapport

    scellement = manifeste["scellement_seuils"]
    chemin_scelle = resout_chemin(scellement["chemin"], racine)
    if chemin_scelle is None:
        rapport.update(verdict=VERDICT_INDETERMINEE, branche="3", motif=(
            f"fichier scellé INTROUVABLE ({scellement['chemin']}) -- rien à desceller."))
        return rapport
    seuils_regime = descelle_seuils(chemin_scelle, scellement["sha256"]).get(REGIME_P3, {})
    bras = _seuils_par_bras(sessions, seuils_regime)

    # Garde de dispersion, PAR BRAS, les deux bras (§C5 / prereg v2 garde 4).
    dispersions = {}
    for nom_bras, seuils in bras.items():
        d = evalue_dispersion(seuils)
        cv = d["dispersion_relative"]
        dispersions[nom_bras] = dict(
            n=d["n"], cv=(None if not np.isfinite(cv) else float(cv)),
            cv_pct=(None if not np.isfinite(cv) else 100.0 * float(cv)),
            seuil_cv=d["seuil_dispersion_relative"], coherent=bool(d["coherent"]))
    rapport["dispersion_par_bras"] = dispersions
    incoherents = [nom for nom, d in dispersions.items() if not d["coherent"]]
    if incoherents:
        rapport.update(verdict=VERDICT_INDETERMINEE, branche="3", motif=(
            "DISPERSION hors clous (CV > 30 %, §C5) sur bras : " + ", ".join(incoherents) +
            ". Les seuils ne sont NI imprimés NI écrits -- seul le CV est surfacé."))
        return rapport

    # Gardes toutes vertes, dispersion comprise : lecture des branches.
    ic_v, ic_r = calcule_ic(bras["viridis"]), calcule_ic(bras["r1"])
    jnd_v, jnd_r = float(np.mean(bras["viridis"])), float(np.mean(bras["r1"]))
    transport_t = jnd_r / jnd_v
    ic_t = [ic_r[0] / ic_v[1], ic_r[1] / ic_v[0]]
    recouvre = not (ic_r[1] < ic_v[0] or ic_r[0] > ic_v[1])
    # CONVENTION DE BORNE gravée (mission 8b) : ⊂ à bornes INCLUSES — la
    # symétrie avec la convention du pin (« un témoin pile à la borne n'est
    # pas une dérive ») ; sur des seuils mesurés, l'égalité exacte est de
    # mesure nulle, la convention pèse sur le déterminisme du prononcé.
    inclus = (ic_t[0] >= 1.0 / LAMBDA_EQUIV) and (ic_t[1] <= LAMBDA_EQUIV)

    rapport["bras"] = {
        nom: dict(seuils=[float(s) for s in seuils], jnd=float(np.mean(seuils)),
                  ic_minmax_decideur=calcule_ic(seuils),
                  ic_mean2sem_surface=calcule_ic_combine(seuils))
        for nom, seuils in bras.items()}
    rapport["transport"] = dict(
        T=transport_t, ic_T=ic_t, recouvrement=recouvre, inclusion_lambda=inclus,
        lambda_equiv=LAMBDA_EQUIV, fenetre=[1.0 / LAMBDA_EQUIV, LAMBDA_EQUIV],
        bornes="INCLUSES (convention gravée, mission 8b)")
    # Diagnostics SURFACÉS, jamais jugés.
    rapport["diagnostics"] = dict(
        derive_viridis_vs_pin=dict(
            jnd_viridis_du_jour=jnd_v, pin_grave=PIN_JND_SEV, ic_pin=list(PIN_IC),
            note="SURFACÉ (prereg P3′) : documente le chemin du sujet depuis Arc C -- "
                 "jamais un critère, le référent du transport est INTRA-session."),
        seuils_par_position=[
            dict(position=s["numero_staircase"], chemin=s.get("chemin_rendu"),
                 seuil=seuils_regime.get(str(s["numero_staircase"])))
            for s in sorted(sessions, key=lambda x: x["numero_staircase"])],
        # Mission 8a.6 / M2 : la FENÊTRE de la session — durée de chaque
        # staircase et pauses libres entre elles. SURFACÉES, JAMAIS JUGÉES :
        # aucune borne n'est gravée, aucune garde n'en dépend, aucun verdict
        # ne s'y adosse. Elles disent dans quel temps la mesure a couru, ce
        # qui est ce qu'il faut pour LIRE la dérive intra-session déjà
        # surfacée par `seuils_par_position` — pas pour la juger.
        temps_de_session=dict(
            duree_par_staircase=[
                dict(position=s["numero_staircase"], duree_s=s.get("duree_s"),
                     debut=s.get("horodatage_debut"), fin=s.get("horodatage_fin"))
                for s in sorted(sessions, key=lambda x: x["numero_staircase"])],
            pauses_inter_staircases=pauses_inter_staircases(
                sorted(sessions, key=lambda x: x["numero_staircase"])),
            note="SURFACÉ (mission 8a.6) : pauses LIBRES au prereg -- aucune borne, "
                 "aucune garde n'en dépend, jamais jugées. Vide sur un manifeste "
                 "d'avant ce chantier (aucun horodatage à lire, rien n'est fabriqué)."),
        mean2sem="surfacé par bras ci-dessus, jamais décideur")

    direction = ("seuil PLUS HAUT a travers R1 (moins sensible)" if transport_t > 1.0
                 else "seuil PLUS BAS a travers R1 (plus sensible)")
    if recouvre and not inclus:
        rapport.update(verdict=VERDICT_NON_DISTINGUE, branche="1a", texte=(
            "IC_R recoupe IC_V sans inclusion de IC_T dans [1/λ, λ] ⇒ TRANSPORT NON "
            "DISTINGUÉ DE 1 (α ≈ 0.10, puissance chiffrée au prereg). C'est un NON-REJET, "
            "pas une équivalence : la condition « transport ≈ 1 » du caveat D14 N'EST PAS "
            "ÉTABLIE -- elle reste ouverte, consignée telle quelle. Le gate (iii′) ne "
            "reçoit rien ; l'anti-surclame voyage inchangé."))
    elif recouvre and inclus:
        rapport.update(verdict=VERDICT_EQUIVALENT, branche="1b", texte=(
            f"IC_T ⊂ [1/λ, λ] (λ = {LAMBDA_EQUIV}, D-P3′-2 §A42, bornes incluses) ET "
            "recouvrement ⇒ TRANSPORT ÉQUIVALENT À 1 à la marge gravée. C'est la SEULE "
            "branche qui ÉTABLIT la condition du caveat D14 -- version intra-session, "
            "bornée par λ en toutes lettres (« établie à un facteur <= 1.5 près »). "
            "L'anti-surclame VOYAGE : la pondération d'excentricité n'est PAS mesurée, "
            "la scission D14 ne se referme pas ici ; le gate (iii′) mesure avec le pin "
            "d'instrument via R1."))
    elif not recouvre and inclus:
        rapport.update(verdict=VERDICT_DIFFERENT_ET_BORNE, branche="2-equiv",
                       transport_T=transport_t, transport_T_ic=ic_t, direction=direction,
                       texte=(
            f"IC disjoints ET IC_T ⊆ [1/λ, λ] ⇒ PRONONCÉ DOUBLE (§A42-COMPLÉMENT, "
            f"résolution β) : TRANSPORT ≠ 1, T = {transport_t:.4f} ({direction}), ET "
            "équivalence à λ établie AU SENS BORNÉ -- la condition D14 est établie à "
            f"λ = {LAMBDA_EQUIV} près, ET le gate (iii′) reçoit son échelle relative ; "
            "r_fovea reçoit son consommateur ; le prononcé double voyage ENTIER, aucune "
            "moitié ne se cite seule. AUCUN seuil d'état ne bouge."))
    else:
        rapport.update(verdict=VERDICT_DIFFERENT, branche="2",
                       transport_T=transport_t, transport_T_ic=ic_t, direction=direction,
                       texte=(
            f"IC disjoints sans inclusion ⇒ TRANSPORT ≠ 1, T = {transport_t:.4f} "
            f"({direction}), IC_T = [{ic_t[0]:.4f}, {ic_t[1]:.4f}]. Le gate (iii′) reçoit "
            "SON échelle relative ; r_fovea reçoit son consommateur ; AUCUN seuil d'état "
            "ne bouge -- le pin gravé reste le pin du harnais, jnd du jour et T sont des "
            "grandeurs NOUVELLES."))
    rapport["motif"] = rapport["texte"]
    return rapport


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocole", choices=["historique", "p3prime"],
                        default="historique",
                        help="'historique' = lecture P3 (prereg §A38) ; 'p3prime' = "
                             "lecture P3′ (prereg v2.2 §A42 : gardes (a)-(d), graine par "
                             "égalité, plan D-P3′-1, CV par bras, branches 1a/1b/2/"
                             "2-équiv/3, seuils descellés tout-vert seulement).")
    parser.add_argument("--manifeste", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.manifeste is None:
        args.manifeste = ROOT / "outputs" / "arcC" / (
            "manifeste_p3prime.json" if args.protocole == "p3prime" else "manifeste_p3.json")
    if args.out is None:
        args.out = ROOT / "outputs" / "arcC" / (
            "p3prime_transport.lecture.json" if args.protocole == "p3prime"
            else "p3_transport_pin.lecture.json")

    manifeste = json.loads(args.manifeste.read_text(encoding="utf-8"))
    racine = args.manifeste.resolve().parent

    if args.protocole == "p3prime":
        rapport = prononce_p3prime(manifeste, racine)
        print("=" * 78)
        print("P3′ -- LECTURE MECANIQUE DU TRANSPORT INTRA-SESSION (prereg v2.2, §A42)")
        print("=" * 78)
        print(f"manifeste = {args.manifeste}")
        print("\n[gardes -- TOUTES evaluees, jamais court-circuitees]")
        for garde in rapport["gardes"]["gardes"]:
            etat = "OK  " if garde["ok"] else "ECHEC"
            print(f"  {etat} {garde['nom']}"
                  + (f"  -- {garde['motif']}" if garde.get("motif") else ""))
        if "dispersion_par_bras" in rapport:
            print("\n[dispersion par bras (§C5, CV <= 30 %)]")
            for nom, d in rapport["dispersion_par_bras"].items():
                cv = "?" if d["cv_pct"] is None else f"{d['cv_pct']:.1f} %"
                print(f"  {nom:8s} n={d['n']}  CV={cv}  -> "
                      f"{'coherent' if d['coherent'] else 'HORS CLOUS'}")
        if "transport" in rapport:
            tr = rapport["transport"]
            print(f"\n[transport]  T = {tr['T']:.4f}  IC_T = [{tr['ic_T'][0]:.4f}, "
                  f"{tr['ic_T'][1]:.4f}]  fenetre λ = {tr['fenetre']}  "
                  f"recouvrement={tr['recouvrement']}  inclusion={tr['inclusion_lambda']}")
        print(f"\nVERDICT : {rapport['verdict']}  (branche {rapport.get('branche')})")
        print(f"  {rapport.get('motif')}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rapport, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        print(f"\n[REPORT] -> {args.out}")
        print("POINT D'ARRET : la lecture versionnee (claude/lectures/) se recopie APRES "
              "verdict, jamais par la machine. Le verdict remonte a Romain.")
        return

    rapport = prononce(manifeste, racine)

    print("=" * 78)
    print("P3 -- LECTURE MECANIQUE DU TRANSPORT DU PIN")
    print("=" * 78)
    print(f"manifeste = {args.manifeste}")
    print(f"pin grave : jnd_sev {100.0 * PIN_JND_SEV:.2f} % "
          f"IC [{100.0 * PIN_IC[0]:.2f}, {100.0 * PIN_IC[1]:.2f}] % (methode : min/max)")
    print("\n[gardes]")
    for garde in rapport["gardes"]["gardes"]:
        etat = "OK  " if garde["ok"] else "ECHEC"
        print(f"  {etat} {garde['nom']}" + (f"  -- {garde['motif']}" if garde.get("motif")
                                            else ""))
    if "temoin" in rapport:
        temoin = rapport["temoin"]
        print("\n[temoin viridis]")
        if temoin.get("seuil") is not None:
            print(f"  seuil = {temoin['seuil_pct']:.2f} %  "
                  f"IC pin [{100.0 * PIN_IC[0]:.2f}, {100.0 * PIN_IC[1]:.2f}] %  -> "
                  f"{'DANS' if temoin['ok'] else 'HORS'}")
        else:
            print(f"  {temoin['motif']}")
    if "r1" in rapport:
        r1 = rapport["r1"]
        print("\n[bras R1]")
        print(f"  n = {r1['n_staircases']}  jnd_R1 = {r1['jnd_r1_pct']:.2f} %")
        print(f"  IC min/max   (DECIDEUR) = [{r1['ic_minmax_decideur_pct'][0]:.2f}, "
              f"{r1['ic_minmax_decideur_pct'][1]:.2f}] %")
        surface = r1["ic_mean2sem_surface"]
        print(f"  IC mean+-2SEM (SURFACE) = [{100.0 * surface[0]:.2f}, "
              f"{100.0 * surface[1]:.2f}] %")
        print(f"  [DIAGNOSTIC] {DIAGNOSTIC_IC}")
    print(f"\nVERDICT : {rapport['verdict']}  (branche {rapport.get('branche')})")
    print(f"  {rapport.get('motif')}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rapport, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[REPORT] -> {args.out}")
    print("POINT D'ARRET : la lecture versionnee (claude/lectures/"
          "p3_transport_pin.lecture.json) se recopie APRES verdict, jamais par la machine.")


if __name__ == "__main__":
    main()
