#!/usr/bin/env python
"""DRIVER — RE-QUALIFICATION DE L'INSTRUMENT 3D, v2.

Exécute le protocole de
`claude/prereg-requalification-instrument-3d-v2-2026-08-22.md`, endossé par
les commits `4a7b9a9` (corps) et `f9f57fc` (amendement : dimensionnement par
côté, test de bit d'I-q6, mécanisation de la garde 4). **Le protocole
complet est celui de `f9f57fc`** — écrit AVANT ce fichier.

C'est de la qualification d'APPAREIL : par la règle 1 de `CLAUDE.md` et par
le §7 du prereg, rien ici ne prononce quoi que ce soit sur V4, la cadence,
le côté de fenêtre ou le budget.

Ce que v2 renverse contre v1 (`run_requalification_instrument_3d.py`, que ce
fichier NE remplace PAS — il porte le protocole que §A61 a disqualifié, et
l'écraser effacerait la trace de ce qui a été tenté) : v1 cherchait le
warmup qui atteint le plateau. §A61 a établi qu'il n'y a pas de plateau.
v2 ne cherche donc aucune convergence : elle mesure si un CONTRASTE APPARIÉ
EN RANG est répétable malgré la dérive, et avec quelle incertitude. Aucun
`T_conv`, aucune médiane glissante, aucune valeur attendue externe.

`src/f1_gpu/chrono.py` n'est PAS modifié (garde 1). La trace temporelle est
reconstruite HORS de lui, par somme cumulée des durées qu'il rend déjà, mise
à l'échelle du temps mur mesuré autour de l'appel — approximation DÉCLARÉE,
identique à v1 : elle répartit uniformément le surcoût Python entre frames.

Usage : `.venv/bin/python run_requalification_instrument_3d_v2.py`
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import subprocess
import sys
import threading
import time

import numpy as np

from src.f1_gpu import substrat_fusionne_3d as fusionne_3d
from src.f1_gpu.chrono import SERIE_FRAMES, WARMUP_FRAMES, chronometrer_frames
from src.f1_gpu.substrat_fusionne_3d_param import (
    empreinte_param,
    pas_f_fusionne_3d_param,
)
from src.f1_gpu.substrat_jetable_3d import etat_initial_jetable_3d

# ------------------------------------------------------------- provenance
PREREG = "claude/prereg-requalification-instrument-3d-v2-2026-08-22.md"
COMMIT_PREREG = "ab954f6"          # protocole COMPLET (4a7b9a9 + f9f57fc + ab954f6)
SORTIE = pathlib.Path("claude/lectures/requalification-instrument-3d-v2-2026-08-22.json")

# --------------------------------------------------- constantes du prereg
# Toutes viennent du document ; aucune n'est choisie ici. La section qui
# grave chacune est nommée — un lecteur doit pouvoir remonter au texte.
N_SCALAIRES, N_STATIQUES = 3, 2                     # §2 vocabulaire réel
N_CHAMPS = 4 + N_SCALAIRES + N_STATIQUES            # §2 « 9 champs »
N_FENETRES = 3                                      # §2 « 3 fenêtres »
GRAINE = 12345                                      # état jetable, hors décision

COTES = (64, 128, 32)                               # §2 le cycle, sens ALLER
DUREE_MIN_BLOC_S = 3.0                              # §2 max(300 frames, 3 s)
SEUIL_FROID_C = 55.0                                # §2 « froid » est mesuré
PLAFOND_REFROIDISSEMENT_S = 180.0                   # §2 plafond du refroidissement
PERIODE_SONDE_S = 0.05                              # §2 relevé ~20 Hz

SEUIL_DISPERSION = 0.02                             # §4 Q-A, 3x la concordance §A53
SEUIL_QC = 0.15                                     # §4 Q-C, ampleur de §A60
SEUIL_FORME = 5.5                                   # §3 DÉRIVÉ du budget de FP
BUDGET_FP_GLOBAL = 0.10                             # §4 CONVENTION : le FP GLOBAL
FACTEUR_USAGE_SIGMA = 3                             # §4 CONVENTION (règle 3·σ_r)
SEUIL_SONDE = 0.02                                  # I-q1
SEUIL_DERIVE_VUE = 0.01                             # I-q2
PRIOR_PENTE_PCT_PAR_S = 0.017                       # §1 prior de dérive
FACTEUR_IQ5 = 2.0                                   # I-q5 : 2x le prior
BIT_SW_POWER_CAP = 0x4                              # I-q6 : test de BIT (amendement)

# §2 ordre d'exécution, gravé. (nom, ordre des côtés, départ thermique)
PLAN_CYCLES = (
    ("M-c1", "ALLER", "froid"),
    ("M-c2", "RETOUR", "chaud"),
    ("M-c3", "RETOUR", "froid"),
    ("M-c4", "ALLER", "chaud"),
)
N_CYCLES_PAR_RUN = 4                                # §2
N_BLOCS_M_S = 6                                     # §2 M-s : 6 blocs 64³ alternés
DUREE_M_D_S = 180.0                                 # §2 M-d : série continue 180 s

# Contrastes CANONIQUES : le sens ALLER. Toute paire temporelle est lue dans
# ce sens, en inversant le rapport si elle a été parcourue à l'envers —
# sans quoi la moyenne géométrique agrégerait des inverses (I-q1 a montré ce
# que coûte une orientation non gravée).
CONTRASTES_CANONIQUES = ((64, 128), (128, 32), (32, 64))

_REQUETE_SMI = ("clocks.sm,clocks.max.sm,temperature.gpu,power.draw,"
                "clocks_event_reasons.active")


class MesureImpossible(RuntimeError):
    """Le run s'arrête AVANT de produire un chiffre (garde 8 : fail-loud,
    jamais d'`assert` nu, jamais de PASS/FAIL par défaut)."""


def _echo(m: str) -> None:
    print(m, flush=True)


def _git(c: list[str]) -> str:
    return subprocess.run(["git", *c], capture_output=True, text=True).stdout.strip()


def _lire_smi() -> dict | None:
    try:
        sortie = subprocess.run(
            ["nvidia-smi", f"--query-gpu={_REQUETE_SMI}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip()
        champs = [c.strip() for c in sortie.split(",")]
        return {"sm_mhz": float(champs[0]), "sm_max_mhz": float(champs[1]),
                "temperature_c": float(champs[2]), "puissance_w": float(champs[3]),
                "raisons": champs[4]}
    except Exception:
        return None


def _raisons_en_entier(brut: str) -> int | None:
    """I-q6 (amendement `f9f57fc`) : le throttle se lit par test de BIT sur
    la valeur ENTIÈRE, jamais par comparaison de chaîne. Rendre None plutôt
    que 0 quand la valeur est illisible — 0 signifierait « aucun throttle »,
    ce qui est un PASS par défaut (garde 8)."""
    try:
        return int(str(brut).strip(), 16)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------- les gardes

def garde_kernels_intouches() -> dict:
    """Garde 3 : `git diff` vide sur les kernels, empreintes dans l'artefact."""
    diff = _git(["diff", "--stat", "HEAD", "--", "src/f1_gpu/"])
    if diff:
        raise MesureImpossible(f"src/f1_gpu modifié :\n{diff}")
    return {
        "empreinte_source_3d": hashlib.sha256(
            fusionne_3d._SOURCE_3D.encode("utf-8")).hexdigest(),
        "empreinte_param_3_2": empreinte_param(N_SCALAIRES, N_STATIQUES),
        "git_diff_src": "vide",
    }


def garde_prereg_endosse() -> dict:
    """Le protocole exécuté doit être celui que l'histoire git endosse.

    Cette garde n'était pas demandée par le prereg : elle mécanise
    l'incident du 22/08 (le prereg amendé APRÈS son commit a vécu 5 min 28 s
    dans une version que HEAD ne portait pas). Une garde promise est une
    garde absente — celle-ci est codée. Elle échoue si le prereg sur disque
    diffère de sa version dans `COMMIT_PREREG`, ou s'il est non commité."""
    chemin = pathlib.Path(PREREG)
    if not chemin.exists():
        raise MesureImpossible(f"prereg absent du disque : {PREREG}")
    disque = chemin.read_bytes()
    commite = subprocess.run(["git", "show", f"{COMMIT_PREREG}:{PREREG}"],
                             capture_output=True)
    if commite.returncode != 0:
        raise MesureImpossible(
            f"{PREREG} introuvable dans {COMMIT_PREREG} — le protocole "
            "exécuté ne serait endossé par aucun commit.")
    if disque != commite.stdout:
        raise MesureImpossible(
            f"le prereg sur disque DIFFÈRE de {COMMIT_PREREG} — le driver "
            "porterait du protocole que l'histoire n'endosse pas (incident "
            "du 22/08, mécanisé ici).")
    sale = _git(["status", "--porcelain", "--", PREREG])
    if sale:
        raise MesureImpossible(f"prereg non propre dans l'arbre : {sale}")
    return {
        "prereg": PREREG,
        "commit_prereg": COMMIT_PREREG,
        "sha256_prereg": hashlib.sha256(disque).hexdigest(),
        "identique_au_commit": True,
    }


def garde_artefact_vierge() -> None:
    """On n'écrase JAMAIS un artefact verdict-grade."""
    if SORTIE.exists():
        raise MesureImpossible(f"{SORTIE} existe déjà — on n'écrase pas.")


# -------------------------------------------------------------- la sonde

class SondeHorloge:
    """Sonde UNIQUE de session (garde 4, amendement `f9f57fc`).

    Un démarrage/arrêt par bloc insérerait jusqu'à `join(timeout)` de GPU au
    repos entre deux blocs « enchaînés », ce que la garde 4 interdit. Elle
    tourne donc en continu et MARQUE les frontières.

    Elle n'est SUSPENDUE qu'autour des blocs `OFF` de `M-s`, pour un coût
    d'au plus une période, consigné en idle intentionnel. Un `Thread` arrêté
    ne se redémarre pas : la reprise en recrée un, et les relevés
    s'accumulent dans la MÊME instance — la sonde reste l'instrument unique
    de la session, y compris comme mesure du critère « froid », même si
    `I-q1` la retire de la chaîne de verdict."""

    def __init__(self):
        self.releves: list[dict] = []
        self.marques: list[dict] = []
        self.suspensions: list[dict] = []
        self._thread: threading.Thread | None = None
        self._continuer = threading.Event()

    def _boucle(self):
        while self._continuer.is_set():
            valeur = _lire_smi()
            if valeur is not None:
                self.releves.append({"t": time.perf_counter(), **valeur})
            time.sleep(PERIODE_SONDE_S)

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def demarrer(self) -> None:
        if self.active:
            return
        self._continuer.set()
        self._thread = threading.Thread(target=self._boucle, daemon=True)
        self._thread.start()

    def suspendre(self, pourquoi: str) -> float:
        """Rend la durée de l'arrêt — idle INTENTIONNEL, mesuré, consigné."""
        if not self.active:
            return 0.0
        t0 = time.perf_counter()
        self._continuer.clear()
        self._thread.join(timeout=2.0)
        self._thread = None
        duree = time.perf_counter() - t0
        self.suspensions.append({"quoi": "suspension", "pourquoi": pourquoi,
                                 "duree_s": duree, "intentionnel": True})
        return duree

    def reprendre(self, pourquoi: str) -> float:
        t0 = time.perf_counter()
        self.demarrer()
        duree = time.perf_counter() - t0
        self.suspensions.append({"quoi": "reprise", "pourquoi": pourquoi,
                                 "duree_s": duree, "intentionnel": True})
        return duree

    def marquer(self, quoi: str, **details) -> None:
        """Marquer ne dépend pas du thread : une frontière reste datée même
        pendant une suspension."""
        self.marques.append({"t": time.perf_counter(), "quoi": quoi, **details})


def releves_entre(sonde: SondeHorloge | None, t0: float, t1: float) -> list[dict]:
    if sonde is None:
        return []
    return [r for r in sonde.releves if t0 <= r["t"] <= t1]


def _sonde_ou_none(sonde, active: bool):
    return sonde if active else None


# ------------------------------------------------- refroidissement mesuré

def refroidir(cp, sonde: SondeHorloge | None, quoi: str) -> dict:
    """§2 : « froid » se définit par une MESURE, pas un décret — idle jusqu'à
    température ≤ 55 °C, plafonné à 180 s, température atteinte consignée."""
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    avant = _lire_smi()
    atteint = False
    while time.perf_counter() - t0 < PLAFOND_REFROIDISSEMENT_S:
        valeur = _lire_smi()
        if valeur is not None and valeur["temperature_c"] <= SEUIL_FROID_C:
            atteint = True
            break
        time.sleep(1.0)
    apres = _lire_smi()
    duree = time.perf_counter() - t0
    if sonde is not None:
        sonde.marquer("refroidissement", pour=quoi, duree_s=duree,
                      seuil_atteint=atteint)
    return {"pour": quoi, "duree_s": duree, "seuil_atteint": atteint,
            "seuil_c": SEUIL_FROID_C, "plafond_s": PLAFOND_REFROIDISSEMENT_S,
            "temperature_avant_c": avant["temperature_c"] if avant else None,
            "temperature_apres_c": apres["temperature_c"] if apres else None}


# ------------------------------------- états pré-alloués, dimensionnement

def preallouer(cp) -> dict:
    """Garde 4 : états PRÉ-ALLOUÉS par côté, ré-initialisés device-to-device
    à chaque bloc. Sans cela, chaque bloc rouvrirait un trou d'allocation
    entre deux blocs « enchaînés ». La copie de ré-initialisation est
    chronométrée et consignée."""
    etats = {}
    for cote in COTES:
        cpu = etat_initial_jetable_3d(N_FENETRES, 1, cote, GRAINE,
                                      n_champs=N_CHAMPS)
        reference = cp.asarray(cpu)
        etats[cote] = {
            "reference": reference,
            "q": cp.array(reference, copy=True),
            "tampon": cp.array(reference, copy=True),
            "octets": int(reference.nbytes),
        }
    return etats


def reinitialiser(cp, etat: dict) -> float:
    """Copie device→device, chronométrée (garde 4)."""
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    etat["q"][...] = etat["reference"]
    etat["tampon"][...] = etat["reference"]
    cp.cuda.Device().synchronize()
    return time.perf_counter() - t0


def dimensionner(cp, etats: dict, sonde: SondeHorloge | None) -> dict:
    """§2 (amendement) : le dimensionnement est fixé PAR CÔTÉ, UNE FOIS, en
    tête de session — pas par bloc. Deux blocs du même côté ont ainsi un
    TRAVAIL identique ; un dimensionnement par bloc les rendrait
    inappariables (positions différentes dans la dérive ET travail
    différent).

    Le biais est celui que le prereg nomme : ce pré-chrono s'exécute dans le
    transitoire, donc `t_chapeau` est LENT et les blocs peuvent durer moins
    de 3 s au régime rapide. Le plancher de 300 frames, lui, tient toujours.
    Le pré-chrono est CONSIGNÉ et NON CONSOMMÉ, et il précède le premier
    refroidissement."""
    plan = {}
    for cote in COTES:
        etat = etats[cote]
        reinitialiser(cp, etat)

        def frame(_etat=etat):
            pas_f_fusionne_3d_param(_etat["q"], cp, sortie=_etat["q"],
                                    tampon_etage=_etat["tampon"],
                                    n_statiques=N_STATIQUES)

        if sonde is not None:
            sonde.marquer("pre_chrono", cote=cote)
        pre = chronometrer_frames(cp, frame, warmup=WARMUP_FRAMES,
                                  serie=SERIE_FRAMES)
        t_frame_ms = pre["stats"]["mediane_ms"]
        serie = max(SERIE_FRAMES,
                    math.ceil(DUREE_MIN_BLOC_S * 1000.0 / t_frame_ms))
        plan[cote] = {
            "serie": serie,
            "warmup": WARMUP_FRAMES,
            "t_frame_pre_chrono_ms": t_frame_ms,
            "duree_bloc_nominale_s": serie * t_frame_ms / 1000.0,
            "biais_declare": ("pré-chrono exécuté dans le transitoire : "
                              "t_frame est LENT, donc la série peut durer "
                              "moins de 3 s au régime rapide ; le plancher "
                              "de 300 frames tient."),
        }
        _echo(f"  dimensionnement {cote}³ : {t_frame_ms:8.4f} ms/frame -> "
              f"série {serie} ({plan[cote]['duree_bloc_nominale_s']:.2f} s nominaux)")
    return plan


# ------------------------------------------------------------- un bloc

def mesurer_bloc(cp, etats: dict, plan: dict, sonde: SondeHorloge | None,
                 nom: str, cote: int, sonde_active: bool) -> dict:
    """L'unité du protocole (§2) : warmup 30 frames exclues, puis la série
    dimensionnée pour ce côté, médiane intra-bloc."""
    etat = etats[cote]
    duree_reinit = reinitialiser(cp, etat)

    def frame():
        pas_f_fusionne_3d_param(etat["q"], cp, sortie=etat["q"],
                                tampon_etage=etat["tampon"],
                                n_statiques=N_STATIQUES)

    if sonde is not None:
        sonde.marquer("bloc_debut", nom=nom, cote=cote, sonde_active=sonde_active)
    t0 = time.perf_counter()
    resultat = chronometrer_frames(cp, frame, warmup=plan[cote]["warmup"],
                                   serie=plan[cote]["serie"])
    t1 = time.perf_counter()
    if sonde is not None:
        sonde.marquer("bloc_fin", nom=nom, cote=cote)

    return {
        "nom": nom, "cote": cote, "sonde_active": sonde_active,
        "serie": plan[cote]["serie"], "warmup": plan[cote]["warmup"],
        "duree_reinit_s": duree_reinit,
        "t_mur_debut": t0, "t_mur_fin": t1,
        "duree_mur_s": t1 - t0,
        "mediane_ms": resultat["stats"]["mediane_ms"],
        "stats": resultat["stats"],
        "warmup_ms": list(resultat["warmup_ms"]),
        "serie_ms": list(resultat["serie_ms"]),
        "releves": releves_entre(sonde, t0, t1) if sonde_active else [],
    }


# ------------------------------------------------- exécution des mesures

def executer_m_s(cp, etats, plan, sonde, nom: str) -> dict:
    """§2 `M-s` / `M-s'` : 6 blocs 64³, sonde ON/OFF ALTERNÉE par bloc.

    Les blocs `OFF` exigent que la sonde soit réellement coupée : c'est la
    seule exception de la garde 4, et son coût (au plus une période) est
    consigné en idle intentionnel."""
    blocs = []
    for i in range(N_BLOCS_M_S):
        active = (i % 2 == 0)
        if active and not sonde.active:
            sonde.reprendre(f"{nom}-b{i} : bloc ON")
        if not active and sonde.active:
            sonde.suspendre(f"{nom}-b{i} : bloc OFF")
        blocs.append(mesurer_bloc(cp, etats, plan, sonde, f"{nom}-b{i}",
                                  64, active))
    if not sonde.active:
        sonde.reprendre(f"{nom} : fin de session de sonde")
    return {"nom": nom, "blocs": blocs}


def executer_run_cycles(cp, etats, plan, sonde, nom: str, ordre: str,
                        thermique: str) -> dict:
    """§2 : 4 cycles enchaînés, SANS repos GPU entre blocs. L'écart mur
    entre fin d'un bloc et début du suivant est consigné à chaque frontière
    (garde 4) — un « enchaîné » qui ne l'est pas doit se voir."""
    cotes = COTES if ordre == "ALLER" else tuple(reversed(COTES))
    blocs, frontieres = [], []
    for cycle in range(N_CYCLES_PAR_RUN):
        for cote in cotes:
            if blocs:
                frontieres.append({
                    "apres": blocs[-1]["nom"],
                    "ecart_mur_s": time.perf_counter() - blocs[-1]["t_mur_fin"]})
            blocs.append(mesurer_bloc(cp, etats, plan, sonde,
                                      f"{nom}-c{cycle}-{cote}", cote, True))
    return {"nom": nom, "ordre": ordre, "thermique": thermique,
            "blocs": blocs, "frontieres": frontieres}


def executer_m_d(cp, etats, plan, sonde) -> dict:
    """§2 `M-d` : série continue 64³ de 180 s — la NON-STATIONNARITÉ
    elle-même. Dimensionnée en frames depuis le pré-chrono du côté 64."""
    t_frame_ms = plan[64]["t_frame_pre_chrono_ms"]
    serie = max(SERIE_FRAMES, math.ceil(DUREE_M_D_S * 1000.0 / t_frame_ms))
    etat = etats[64]
    duree_reinit = reinitialiser(cp, etat)

    def frame():
        pas_f_fusionne_3d_param(etat["q"], cp, sortie=etat["q"],
                                tampon_etage=etat["tampon"],
                                n_statiques=N_STATIQUES)

    sonde.marquer("m_d_debut", serie=serie)
    t0 = time.perf_counter()
    resultat = chronometrer_frames(cp, frame, warmup=WARMUP_FRAMES, serie=serie)
    t1 = time.perf_counter()
    sonde.marquer("m_d_fin")
    return {"nom": "M-d", "cote": 64, "serie": serie,
            "duree_reinit_s": duree_reinit,
            "t_mur_debut": t0, "t_mur_fin": t1, "duree_mur_s": t1 - t0,
            "serie_ms": list(resultat["serie_ms"]),
            "warmup_ms": list(resultat["warmup_ms"]),
            "stats": resultat["stats"],
            "releves": releves_entre(sonde, t0, t1)}


# ------------------------------------------------------- outils de lecture

def _iqr_demi(x: np.ndarray) -> float:
    """§3 : l'écart interquartile/2 DÉCIDE, SEUL. Valeur ABSOLUE, dans
    l'unité des `r_i` — à normaliser avant toute comparaison à un seuil."""
    return float((np.percentile(x, 75) - np.percentile(x, 25)) / 2.0)


def _dispersion_relative(x: np.ndarray) -> float:
    """La dispersion qui DÉCIDE est RELATIVE, et elle doit l'être.

    Le seuil de 2 % du §4 est relatif par construction : il vaut 3x la
    concordance de 0,63 % de `§A53`, qui est un écart relatif. Or les trois
    contrastes vivent à des échelles séparées de trois ordres de grandeur
    (`r` ≈ 8,4 pour 64→128, ≈ 0,017 pour 128→32). Comparer un
    interquartile ABSOLU à ce seuil rendrait 128→32 increvable et 64→128
    condamné d'avance — le verdict dépendrait de l'échelle du contraste,
    pas de l'instrument. Trouvé par l'exercice sur données synthétiques,
    pas à l'oeil."""
    centre = float(np.median(x))
    if centre == 0.0:
        return float("nan")
    return _iqr_demi(x) / abs(centre)


def _etendue_demi(x: np.ndarray) -> float:
    """§3 : l'étendue/2 est un TEST DE FORME, pas un second juge."""
    return float((np.max(x) - np.min(x)) / 2.0)


def _test_de_forme(x: np.ndarray) -> dict:
    """§3 : si (étendue/2)/(interquartile/2) dépasse 4,4, le contraste est
    INDÉTERMINÉ et les blocs extrêmes sont nommés. Le seuil est une
    CONVENTION déclarée (p90 gaussien i.i.d., stable n = 12–16)."""
    iqr, etendue = _iqr_demi(x), _etendue_demi(x)
    if iqr <= 0.0:
        # Rapport indéfini. Deux cas OPPOSÉS, que confondre punirait la
        # qualité que le critère contrôle (§A52) : dispersion TOTALEMENT
        # nulle = rien à juger, aucune queue possible ; corps central nul
        # mais étendue non nulle = forme extrême, sortie sûre (garde 8).
        if etendue <= 0.0:
            return {"rapport": None, "declenche": False, "iqr_demi": iqr,
                    "etendue_demi": etendue,
                    "note": ("dispersion nulle des deux côtés : aucune queue "
                             "possible, le test de forme n'a rien à dire")}
        return {"rapport": None, "declenche": True, "iqr_demi": iqr,
                "etendue_demi": etendue,
                "note": ("corps central nul mais étendue non nulle : forme "
                         "extrême -> INDÉTERMINÉ (sortie sûre)")}
    rapport = etendue / iqr
    return {"rapport": rapport, "declenche": bool(rapport > SEUIL_FORME),
            "iqr_demi": iqr, "etendue_demi": etendue, "seuil": SEUIL_FORME}


def _pente_theil_sen(t_s: np.ndarray, y: np.ndarray) -> float:
    """Pente en %/s, estimée par la MÉDIANE des pentes deux-à-deux
    (Theil-Sen) sur des médianes de tranches — robuste aux extrêmes, ce que
    des moindres carrés sur la trace brute ne seraient pas.

    Le prereg exige « un estimateur robuste à la dérive » (§A61-5) sans
    nommer la forme pour `M-d` : ce choix est DÉCLARÉ ici et consigné dans
    l'artefact."""
    if t_s.size < 2:
        return float("nan")
    pentes = []
    for i in range(t_s.size):
        for j in range(i + 1, t_s.size):
            dt = t_s[j] - t_s[i]
            if dt > 0:
                pentes.append((y[j] - y[i]) / dt)
    if not pentes:
        return float("nan")
    base = float(np.median(y))
    return float(np.median(pentes)) / base * 100.0 if base > 0 else float("nan")


def axe_temps_ms(bloc: dict) -> np.ndarray:
    """Trace temporelle reconstruite HORS de `chrono.py` (garde 1) : somme
    cumulée des durées, mise à l'échelle du temps mur mesuré autour de
    l'appel. Approximation DÉCLARÉE — elle répartit uniformément le surcoût
    Python entre frames."""
    a = np.asarray(bloc["serie_ms"], dtype=float)
    cumul = np.cumsum(a)
    duree_mur = bloc["duree_mur_s"] * 1000.0
    echelle = duree_mur / cumul[-1] if cumul[-1] > 0 else 1.0
    return cumul * echelle


# ------------------------------------------ les rapports appariés (§3)

def paires_orientees(run: dict) -> list[dict]:
    """Chaque paire de blocs ADJACENTS dans le temps donne un rapport
    apparié, lu dans le sens CANONIQUE du contraste.

    L'orientation est le point qu'`I-q1` a rendu non négociable : en ordre
    RETOUR, la paire (128, 64) parcourt le contraste 64→128 à l'envers ; son
    rapport doit être INVERSÉ, sinon l'agrégation mélangerait des inverses
    et lirait un signe décidé par le sens de parcours."""
    paires = []
    blocs = run["blocs"]
    for a, b in zip(blocs, blocs[1:]):
        ca, cb = a["cote"], b["cote"]
        if (ca, cb) in CONTRASTES_CANONIQUES:
            contraste, r = (ca, cb), b["mediane_ms"] / a["mediane_ms"]
            sens = "direct"
        elif (cb, ca) in CONTRASTES_CANONIQUES:
            contraste, r = (cb, ca), a["mediane_ms"] / b["mediane_ms"]
            sens = "inverse"
        else:
            raise MesureImpossible(
                f"paire ({ca}, {cb}) hors des contrastes canoniques "
                f"{CONTRASTES_CANONIQUES} — le plan a dérivé du prereg.")
        paires.append({
            "contraste": f"{contraste[0]}->{contraste[1]}",
            "ordre": run["ordre"], "thermique": run["thermique"],
            "sens_de_parcours": sens, "r": r,
            "bloc_amont": a["nom"], "bloc_aval": b["nom"],
            # §3 : la durée MILIEU-À-MILIEU est ce qui porte la borne de
            # biais. Mesurée ici, jamais supposée (les 12,5 s du prereg sont
            # NOMINAUX — volet mécanisé d'I-q5).
            "milieu_a_milieu_s": (
                (b["t_mur_debut"] + b["t_mur_fin"]) / 2.0
                - (a["t_mur_debut"] + a["t_mur_fin"]) / 2.0),
        })
    return paires


def _par_thermique(paires: list[dict]) -> dict:
    """Q-A exige l'accord froid/chaud au meme titre que aller/retour. §2
    rappelle ce que ce contraste achete : l'etat thermique reste confondu
    avec le rang a +1, structurellement — la lecture est BORNEE, pas
    orthogonale au temps."""
    sortie = {}
    for etat in ("froid", "chaud"):
        r = np.array([p["r"] for p in paires if p["thermique"] == etat],
                     dtype=float)
        sortie[etat] = {"n": int(r.size),
                        "mediane": float(np.median(r)) if r.size else None,
                        "iqr_demi_relative": (_dispersion_relative(r)
                                              if r.size else None)}
    return sortie


def _ecart_froid_chaud(paires: list[dict]) -> float | None:
    par = _par_thermique(paires)
    f, c = par["froid"]["mediane"], par["chaud"]["mediane"]
    if f is None or c is None or c <= 0:
        return None
    return abs(f / c - 1.0)


def lire_contrastes(runs: list[dict]) -> dict:
    """Lecture finale = MOYENNE GÉOMÉTRIQUE des deux lectures orientées
    (§3, point 2) : `sqrt(r_ALLER · r_RETOUR)`. Le premier ordre de la
    dérive s'annule alors par construction. Une médiane sur l'UNION n'aurait
    pas cette propriété — elle est calculée et consignée, jamais décidante."""
    toutes = [p for run in runs for p in paires_orientees(run)]
    lecture = {}
    for contraste in (f"{a}->{b}" for a, b in CONTRASTES_CANONIQUES):
        du_contraste = [p for p in toutes if p["contraste"] == contraste]
        r_tous = np.array([p["r"] for p in du_contraste], dtype=float)
        par_ordre = {}
        for ordre in ("ALLER", "RETOUR"):
            r_ordre = np.array([p["r"] for p in du_contraste
                                if p["ordre"] == ordre], dtype=float)
            par_ordre[ordre] = {
                "n": int(r_ordre.size),
                "mediane": float(np.median(r_ordre)) if r_ordre.size else None,
                "iqr_demi_relative": (_dispersion_relative(r_ordre)
                                      if r_ordre.size else None),
            }
        m_aller, m_retour = par_ordre["ALLER"]["mediane"], par_ordre["RETOUR"]["mediane"]
        if m_aller is None or m_retour is None or m_aller <= 0 or m_retour <= 0:
            raise MesureImpossible(
                f"contraste {contraste} : un des deux ordres est vide ou "
                "non positif — la moyenne géométrique n'est pas définie.")
        geometrique = math.sqrt(m_aller * m_retour)
        # AMBIGUÏTÉ DU PREREG, tranchée ici du côté LITTÉRAL et déclarée :
        # §4 écrit « dispersion des r_i ≤ 2 % PAR CONTRASTE », sans dire
        # « par ordre ». La décision se prend donc sur l'UNION des deux
        # ordres ; la dispersion par ordre est consignée à côté.
        lecture[contraste] = {
            "n_paires": int(r_tous.size),
            "lecture_geometrique": geometrique,
            "lecture_par_ordre": par_ordre,
            "mediane_union_non_decidante": float(np.median(r_tous)),
            "dispersion_iqr_demi_absolue": _iqr_demi(r_tous),
            "dispersion_iqr_demi": _dispersion_relative(r_tous),
            "test_de_forme": _test_de_forme(r_tous),
            "ecart_aller_retour": abs(m_aller / m_retour - 1.0),
            "lecture_par_thermique": _par_thermique(du_contraste),
            "ecart_froid_chaud": _ecart_froid_chaud(du_contraste),
            "milieu_a_milieu_s": {
                "median": float(np.median([p["milieu_a_milieu_s"]
                                           for p in du_contraste])),
                "max": float(np.max([p["milieu_a_milieu_s"]
                                     for p in du_contraste])),
            },
            "paires": du_contraste,
        }
    return lecture


# ------------------------------------------- les indéterminations (§5)

def i_q1(m_s: dict, m_s_prime: dict) -> dict:
    """Rapports ORIENTÉS `med(ON)/med(OFF)` par paire adjacente, quel que
    soit l'ordre temporel des deux blocs — sans quoi la garde serait AVEUGLE
    par construction. Critère : `|médiane(r_i) − 1| > 2 %`, l'écart de la
    MÉDIANE, jamais la médiane des écarts (qui mord sur du bruit pur)."""
    def ecart(session: dict) -> dict:
        blocs = session["blocs"]
        r = []
        for a, b in zip(blocs, blocs[1:]):
            on, off = (a, b) if a["sonde_active"] else (b, a)
            if on["sonde_active"] == off["sonde_active"]:
                raise MesureImpossible(
                    f"{session['nom']} : deux blocs adjacents de même état "
                    "de sonde — l'alternance ON/OFF a été rompue.")
            r.append(on["mediane_ms"] / off["mediane_ms"])
        med = float(np.median(r))
        return {"n_paires": len(r), "rapports": r, "mediane": med,
                "ecart_de_la_mediane": abs(med - 1.0),
                "perturbe": bool(abs(med - 1.0) > SEUIL_SONDE)}

    a, b = ecart(m_s), ecart(m_s_prime)
    accord = (a["perturbe"] == b["perturbe"])
    # Désaccord entre M-s et M-s' -> verdict INDÉTERMINÉ et RETRAIT par
    # défaut. Branche MÉCANISÉE, pas laissée à la relecture (§A62-bis-2).
    retrait = (not accord) or a["perturbe"] or b["perturbe"]
    return {"M-s": a, "M-s_prime": b, "accord": accord,
            "verdict": ("INDÉTERMINÉ — M-s et M-s' ne concluent pas de même"
                        if not accord else
                        ("la sonde PERTURBE" if a["perturbe"]
                         else "la sonde est neutre")),
            "retrait_applique": retrait,
            "consequence": ("pente contre le TEMPS et I-q5 restent lisibles ; "
                            "pente contre la TEMPÉRATURE, I-q6 et la "
                            "puissance du test de forme deviennent NON "
                            "LISIBLES" if retrait else "aucune"),
            "seuil": SEUIL_SONDE}


def i_q2(runs: list[dict]) -> dict:
    """La dérive doit être VUE pour être annulée. Écart absolu entre le
    premier et le dernier bloc d'un même côté, DANS un run.

    §5 (amendement `ab954f6`) : la condition s'évalue PAR CONTRASTE —
    `max(écart du côté X, écart du côté Y) < 1 %` suspend CE contraste, et
    le global avec lui. L'écart d'un côté est pris au MAXIMUM sur les
    quatre runs : le critère demande que la dérive ait été vue QUELQUE PART ;
    exiger qu'elle le soit partout punirait un run calme."""
    detail, par_cote = [], {c: [] for c in COTES}
    for run in runs:
        for cote in COTES:
            du_cote = [b for b in run["blocs"] if b["cote"] == cote]
            if len(du_cote) >= 2:
                e = abs(du_cote[-1]["mediane_ms"] / du_cote[0]["mediane_ms"] - 1.0)
                par_cote[cote].append(e)
                detail.append({"run": run["nom"], "cote": cote, "ecart": e})
    ecart_cote = {c: (max(v) if v else 0.0) for c, v in par_cote.items()}
    par_contraste = {}
    for x, y in CONTRASTES_CANONIQUES:
        vu = max(ecart_cote[x], ecart_cote[y])
        par_contraste[f"{x}->{y}"] = {
            "ecart_cote_amont": ecart_cote[x], "ecart_cote_aval": ecart_cote[y],
            "derive_vue": vu, "suspendu": bool(vu < SEUIL_DERIVE_VUE)}
    suspendus = [n for n, v in par_contraste.items() if v["suspendu"]]
    return {"detail": detail, "ecart_par_cote": ecart_cote,
            "par_contraste": par_contraste, "suspendus": suspendus,
            "seuil": SEUIL_DERIVE_VUE,
            "derive_non_vue": bool(suspendus),
            "ecart_max": max(ecart_cote.values()) if ecart_cote else 0.0,
            "consequence": ("verdict SUSPENDU sur " + ", ".join(suspendus) +
                            " et sur le global : σ_r n'est pas gravée, §8 "
                            "reste fermé" if suspendus else "aucune")}


def i_q3(contrastes: dict, biais_nominal: float) -> dict:
    """L'ordre ne doit rien porter. AMBIGUÏTÉ DU PREREG, déclarée : « > 2 %
    (au-delà du biais pré-dérivé de §3) » peut se lire seuil = 2 %, ou
    seuil = 2 % + biais. Lecture retenue — la STRICTE (2 %), le biais étant
    consigné à côté pour que la lecture reste possible."""
    par_contraste = {}
    for nom, c in contrastes.items():
        par_contraste[nom] = {
            "ecart_aller_retour": c["ecart_aller_retour"],
            "seuil": SEUIL_DISPERSION,
            "biais_nominal_consigne": biais_nominal,
            "indetermine": bool(c["ecart_aller_retour"] > SEUIL_DISPERSION)}
    return par_contraste


def i_q4(runs: list[dict], pente_pct_par_s: float | None) -> dict:
    """Le produit télescope : `r(64→128)·r(128→32)·r(32→64)` se réduit à
    `med(64_{i+1})/med(64_i)` — un test de DÉRIVE CYCLE-À-CYCLE.

    §5 (amendement `ab954f6`) : la première version lisait le MAXIMUM des
    produits contre la prédiction moyenne — 99–100 % d'INDÉTERMINÉ sur
    instrument sain. La lecture gravée est
    `|médiane(produits) − (1 + pente·T_cycle)| > max(IQR/2 des produits,
    borne de biais de §3 sur un cycle)` : le premier terme couvre le bruit,
    le second est le plancher DÉRIVÉ de la décision. Le FP n'est pas annoncé
    — il dépend de σ et se calcule APRÈS COUP (§5)."""
    produits = []
    for run in runs:
        blocs = run["blocs"]
        for i in range(0, len(blocs) - 3, 3):
            trio = blocs[i:i + 4]
            p = 1.0
            for a, b in zip(trio, trio[1:]):
                p *= b["mediane_ms"] / a["mediane_ms"]
            produits.append({"run": run["nom"], "produit": p,
                             "duree_cycle_s": trio[-1]["t_mur_fin"] - trio[0]["t_mur_debut"]})
    if not produits:
        return {"produits": [], "indetermine": True,
                "note": "aucun cycle complet — sortie sûre (garde 8)"}
    valeurs = np.array([p["produit"] for p in produits], dtype=float)
    t_cycle = float(np.median([p["duree_cycle_s"] for p in produits]))
    if pente_pct_par_s is None or math.isnan(pente_pct_par_s):
        return {"produits": produits, "indetermine": True,
                "note": ("pente de M-d non lisible : la prédiction n'existe "
                         "pas -> sortie sûre (garde 8)")}
    derive_predite = abs(pente_pct_par_s) / 100.0 * t_cycle
    mediane = float(np.median(valeurs))
    ecart = abs(mediane - (1.0 + derive_predite))
    seuil = max(_iqr_demi(valeurs), derive_predite)
    return {"produits": produits, "n_produits": int(valeurs.size),
            "mediane_produits": mediane, "t_cycle_median_s": t_cycle,
            "derive_predite": derive_predite,
            "ecart_a_la_prediction": ecart,
            "iqr_demi_produits": _iqr_demi(valeurs),
            "plancher_biais": derive_predite, "seuil_retenu": seuil,
            "indetermine": bool(ecart > seuil),
            "note": ("dérive cycle-à-cycle NON expliquée par la pente "
                     "caractérisée" if ecart > seuil else "expliquée")}


def i_q5(pente_pct_par_s: float | None, contrastes: dict) -> dict:
    """Le dimensionnement doit survivre à sa propre mesure, et la borne de
    §3 se recalcule sur les durées RÉELLES (les 12,5 s sont nominaux)."""
    plafond = FACTEUR_IQ5 * PRIOR_PENTE_PCT_PAR_S
    depasse = (pente_pct_par_s is not None
               and not math.isnan(pente_pct_par_s)
               and abs(pente_pct_par_s) > plafond)
    bornes = {}
    if pente_pct_par_s is not None and not math.isnan(pente_pct_par_s):
        for nom, c in contrastes.items():
            bornes[nom] = {
                "milieu_a_milieu_max_s": c["milieu_a_milieu_s"]["max"],
                "biais_reel_pct": abs(pente_pct_par_s) * c["milieu_a_milieu_s"]["max"]}
    return {"pente_mesuree_pct_par_s": pente_pct_par_s,
            "prior_pct_par_s": PRIOR_PENTE_PCT_PAR_S, "plafond": plafond,
            "cycles_non_consommes": bool(depasse),
            "bornes_recalculees": bornes,
            "note": ("pente > 2x le prior : les cycles de CE run ne sont PAS "
                     "consommés ; re-dimensionner et re-runner"
                     if depasse else "le dimensionnement survit à sa mesure")}


def i_q6(m_d: dict, sonde_retiree: bool) -> dict:
    """Throttle lu par TEST DE BIT sur la valeur entière (amendement
    `f9f57fc`), masque complet consigné — jamais par comparaison de chaîne.
    NON LISIBLE si la sonde a été retirée par `I-q1`."""
    if sonde_retiree:
        return {"lisible": False,
                "note": "sonde retirée par I-q1 -> NON LISIBLE (non mesuré, "
                        "pas indéterminé)"}
    masques, illisibles = [], 0
    for r in m_d["releves"]:
        v = _raisons_en_entier(r["raisons"])
        if v is None:
            illisibles += 1
        else:
            masques.append(v)
    if illisibles or not masques:
        return {"lisible": False, "releves_illisibles": illisibles,
                "note": "masque illisible : 0 signifierait « aucun throttle », "
                        "ce serait un PASS par défaut (garde 8)"}
    sw_power_cap = [m for m in masques if m & BIT_SW_POWER_CAP]
    return {"lisible": True, "n_releves": len(masques),
            "masques_distincts": sorted({hex(m) for m in masques}),
            "sw_power_cap_present": bool(sw_power_cap),
            "n_releves_sw_power_cap": len(sw_power_cap),
            "bit_teste": hex(BIT_SW_POWER_CAP)}


# ---------------------------------------------- les branches (§4)

def brancher(contrastes: dict, q3: dict, q4: dict) -> dict:
    """Q-A / Q-B / Q-C, PAR CONTRASTE — et seulement sur ceux qui ont survécu
    à §5. La priorité est celle que §4 grave : les indéterminations de §5
    s'évaluent AVANT les branches, parce que deux clauses pré-écrites ne
    peuvent pas conclure en sens opposés sur le même chiffre."""
    par_contraste = {}
    for nom, c in contrastes.items():
        forme = c["test_de_forme"]
        dispersion = c["dispersion_iqr_demi"]
        froid_chaud = c["ecart_froid_chaud"]

        # --- §5 d'abord. Aucune branche ne se prononce sur un contraste sorti.
        if forme["declenche"]:
            par_contraste[nom] = {
                "branche": "INDÉTERMINÉ", "cause": "§3 test de forme",
                "detail": forme,
                "blocs_extremes": _blocs_extremes(c)}
            continue
        if q3[nom]["indetermine"]:
            par_contraste[nom] = {
                "branche": "INDÉTERMINÉ", "cause": "I-q3 (dépendance au rang)",
                "detail": q3[nom]}
            continue
        if q4["indetermine"]:
            par_contraste[nom] = {
                "branche": "INDÉTERMINÉ", "cause": "I-q4 (dérive cycle-à-cycle)",
                "detail": {k: q4[k] for k in ("ecart_max", "derive_predite", "note")}}
            continue
        if froid_chaud is None:
            par_contraste[nom] = {
                "branche": "INDÉTERMINÉ",
                "cause": "accord froid/chaud non calculable — sortie sûre"}
            continue

        # --- puis les branches, sur l'interquartile SEUL (§3).
        pire = max(dispersion, froid_chaud)
        if pire > SEUIL_QC:
            branche, mot = "Q-C", ("NON QUALIFIABLE — l'appariement n'a rien "
                                   "acheté ; le problème est MATÉRIEL, et "
                                   "aucune cinquième tentative protocolaire "
                                   "n'est prévue")
        elif dispersion <= SEUIL_DISPERSION and froid_chaud <= SEUIL_DISPERSION:
            branche, mot = "Q-A", ("QUALIFIÉ pour les contrastes appariés "
                                   "intra-run ; les absolus inter-runs "
                                   "restent interdits")
        else:
            branche, mot = "Q-B", ("QUALIFIÉ RESTREINT — les deux bras dans "
                                   "le même run seulement")
        par_contraste[nom] = {
            "branche": branche, "verdict": mot,
            "dispersion_iqr_demi": dispersion,
            "ecart_froid_chaud": froid_chaud,
            "ecart_aller_retour": c["ecart_aller_retour"],
            "test_de_forme": forme}
    return par_contraste


def _blocs_extremes(c: dict) -> dict:
    """§3 : quand le test de forme déclenche, les blocs extrêmes sont NOMMÉS."""
    paires = c["paires"]
    if not paires:
        return {}
    bas = min(paires, key=lambda p: p["r"])
    haut = max(paires, key=lambda p: p["r"])
    return {"r_min": {"r": bas["r"], "amont": bas["bloc_amont"],
                      "aval": bas["bloc_aval"]},
            "r_max": {"r": haut["r"], "amont": haut["bloc_amont"],
                      "aval": haut["bloc_aval"]}}


def agreger(par_contraste: dict, q2: dict, q5: dict) -> dict:
    """§4 : le verdict global est la PIRE des trois branches, et tout
    contraste sorti par §5 rend le global INDÉTERMINÉ — ce protocole
    qualifie une MÉTHODE, pas trois paires de kernels.

    Deux états ferment §8 sans être Q-C : SUSPENDU (I-q2 : les chiffres sont
    sains, la démonstration est incomplète) et CYCLES NON CONSOMMÉS (I-q5)."""
    if q5["cycles_non_consommes"]:
        return {"global": "CYCLES NON CONSOMMÉS", "cause": "I-q5",
                "sigma_r": None, "section_8_ouverte": False,
                "verdict": ("la pente mesurée dépasse 2x le prior qui a "
                            "dimensionné les blocs : re-dimensionner et "
                            "re-runner ; ce run ne se lit pas")}
    if q2["derive_non_vue"]:
        return {"global": "SUSPENDU", "cause": "I-q2",
                "sigma_r": None, "section_8_ouverte": False,
                "verdict": ("la dérive n'a pas été VUE (écart max "
                            f"{q2['ecart_max']*100:.2f} % < 1 %) : la "
                            "répétabilité est consignée, la robustesse reste "
                            "NON ÉTABLIE, σ_r n'est pas gravée")}
    branches = [v["branche"] for v in par_contraste.values()]
    if "INDÉTERMINÉ" in branches:
        return {"global": "INDÉTERMINÉ", "cause": "au moins un contraste sorti par §5",
                "sigma_r": None, "section_8_ouverte": False,
                "branches": branches,
                "verdict": ("un échec inexpliqué sur l'un des trois spécimens "
                            "est un échec de la MÉTHODE")}
    pire = "Q-C" if "Q-C" in branches else ("Q-B" if "Q-B" in branches else "Q-A")
    sigma_r = max(v["dispersion_iqr_demi"] for v in par_contraste.values())
    return {
        "global": pire, "branches": branches,
        "sigma_r": sigma_r,
        "regle_d_usage": (f"toute lecture future distante de moins de "
                          f"{FACTEUR_USAGE_SIGMA}·σ_r = "
                          f"{FACTEUR_USAGE_SIGMA * sigma_r:.4f} d'un seuil est "
                          f"INDÉTERMINÉE (facteur {FACTEUR_USAGE_SIGMA} : "
                          f"CONVENTION déclarée au §4, achetée par "
                          f"l'endossement du prereg)"),
        "section_8_ouverte": pire in ("Q-A", "Q-B"),
        "absolus": "INTERDITS — Q-A ne réhabilite pas les mesures inter-runs",
    }


# ------------------------------------------------------------------ main

def main() -> int:
    _echo("== gardes ==")
    garde_artefact_vierge()
    endossement = garde_prereg_endosse()
    _echo(f"  prereg identique à {COMMIT_PREREG} "
          f"(sha256 {endossement['sha256_prereg'][:12]}…)")
    gardes = garde_kernels_intouches()
    _echo("  src/f1_gpu intouché")
    if _lire_smi() is None:
        raise MesureImpossible(
            "nvidia-smi indisponible : ni le critère « froid » (§2), ni "
            "I-q6, ni la pente contre la température ne seraient mesurables.")
    try:
        import cupy as cp
    except Exception as erreur:                          # pragma: no cover
        raise MesureImpossible(f"cupy indisponible : {erreur}")

    sonde = SondeHorloge()
    sonde.demarrer()
    refroidissements = []
    try:
        _echo("== pré-allocation (garde 4) ==")
        etats = preallouer(cp)
        for cote in COTES:
            _echo(f"  {cote}³ : {etats[cote]['octets']/2**20:7.1f} Mio × 3 tableaux")

        # Le pré-chrono PRÉCÈDE le premier refroidissement (§2, amendement).
        _echo("== dimensionnement par côté, une fois (§2) ==")
        plan = dimensionner(cp, etats, sonde)

        _echo("== M-s : la sonde perturbe-t-elle ? ==")
        refroidissements.append(refroidir(cp, sonde, "M-s"))
        m_s = executer_m_s(cp, etats, plan, sonde, "M-s")

        runs = []
        for nom, ordre, thermique in PLAN_CYCLES:
            if thermique == "froid":
                refroidissements.append(refroidir(cp, sonde, nom))
            _echo(f"== {nom} : {ordre}, départ {thermique} ==")
            run = executer_run_cycles(cp, etats, plan, sonde, nom, ordre,
                                      thermique)
            runs.append(run)
            _echo(f"  {len(run['blocs'])} blocs, écart de frontière max "
                  f"{max((f['ecart_mur_s'] for f in run['frontieres']), default=0):.3f} s")

        _echo("== M-d : la non-stationnarité elle-même ==")
        refroidissements.append(refroidir(cp, sonde, "M-d"))
        m_d = executer_m_d(cp, etats, plan, sonde)

        _echo("== M-s′ : la sonde, relue (départ FROID compris) ==")
        refroidissements.append(refroidir(cp, sonde, "M-s'"))
        m_s_prime = executer_m_s(cp, etats, plan, sonde, "M-s'")
    finally:
        sonde.suspendre("fin de session")

    # ------------------------------------------------------------ lecture
    _echo("== lecture ==")
    contrastes = lire_contrastes(runs)

    # La pente de M-d : contre le TEMPS (lisible même sonde retirée) et
    # contre la TEMPÉRATURE (qui, elle, meurt avec la sonde — I-q1).
    t_ms = axe_temps_ms(m_d)
    y = np.asarray(m_d["serie_ms"], dtype=float)
    n_tranches = max(2, int(m_d["duree_mur_s"]))
    bornes = np.linspace(0, y.size, n_tranches + 1, dtype=int)
    t_tranche, y_tranche = [], []
    for a, b in zip(bornes, bornes[1:]):
        if b > a:
            t_tranche.append(float(np.median(t_ms[a:b])) / 1000.0)
            y_tranche.append(float(np.median(y[a:b])))
    pente = _pente_theil_sen(np.asarray(t_tranche), np.asarray(y_tranche))

    q1 = i_q1(m_s, m_s_prime)
    q2 = i_q2(runs)
    biais_nominal = (abs(pente) / 100.0
                     * max(c["milieu_a_milieu_s"]["max"] for c in contrastes.values())
                     if not math.isnan(pente) else float("nan"))
    q3 = i_q3(contrastes, biais_nominal)
    q4 = i_q4(runs, pente)
    q5 = i_q5(pente, contrastes)
    q6 = i_q6(m_d, q1["retrait_applique"])

    par_contraste = brancher(contrastes, q3, q4)
    global_ = agreger(par_contraste, q2, q5)

    # ----------------------------------------------------------- artefact
    artefact = {
        "protocole": {
            "prereg": PREREG, "commit_prereg": COMMIT_PREREG,
            "nature": ("QUALIFICATION D'INSTRUMENT (règle 1 de CLAUDE.md, §7 "
                       "du prereg) — ne prononce RIEN sur V4, la cadence, le "
                       "côté ou le budget, et rien sur les absolus"),
            "question": ("un contraste apparié en rang est-il répétable "
                         "malgré la dérive, et avec quelle incertitude ?"),
            "chrono": (f"chrono.py INTOUCHÉ ; warmup {WARMUP_FRAMES} exclues, "
                       f"série ≥ {SERIE_FRAMES} ET ≥ {DUREE_MIN_BLOC_S} s, "
                       "dimensionnée PAR CÔTÉ une fois"),
            "axe_temps": ("somme cumulée des durées de frame, mise à "
                          "l'échelle du temps mur — répartit uniformément le "
                          "surcoût Python. Approximation DÉCLARÉE."),
            "estimateur_pente": ("Theil-Sen sur médianes de tranches d'~1 s "
                                 "— forme CHOISIE ici, le prereg exige un "
                                 "estimateur robuste sans la nommer"),
            "conventions_neuves": {
                "facteur_usage_sigma": FACTEUR_USAGE_SIGMA,
                "seuil_test_de_forme": SEUIL_FORME,
                "note": "déclarées au §4, achetées par l'endossement du prereg",
            },
            "git_head": _git(["rev-parse", "--short", "HEAD"]),
        },
        "gardes": {**gardes, "endossement": endossement},
        "plan_dimensionnement": plan,
        "refroidissements": refroidissements,
        "suspensions_de_sonde": sonde.suspensions,
        "marques": sonde.marques,
        "mesures": {"M-s": m_s, "M-s_prime": m_s_prime, "M-d": m_d,
                    "runs": runs},
        "contrastes": contrastes,
        "pente_m_d_pct_par_s": pente,
        "indeterminations": {"I-q1": q1, "I-q2": q2, "I-q3": q3,
                             "I-q4": q4, "I-q5": q5, "I-q6": q6},
        "branches_par_contraste": par_contraste,
        "verdict_global": global_,
    }
    SORTIE.parent.mkdir(parents=True, exist_ok=True)
    SORTIE.write_text(json.dumps(artefact, ensure_ascii=False, indent=1))

    for nom, c in contrastes.items():
        b = par_contraste[nom]
        _echo(f"  {nom:>10} : lecture {c['lecture_geometrique']:.4f} | "
              f"IQR/2 {c['dispersion_iqr_demi']*100:5.2f} % | forme "
              f"{(c['test_de_forme']['rapport'] or float('nan')):.2f} | "
              f"{b['branche']}")
    _echo(f"  I-q1 sonde : {q1['verdict']} (retrait : {q1['retrait_applique']})")
    _echo(f"  I-q2 dérive vue : {not q2['derive_non_vue']} "
          f"(max {q2['ecart_max']*100:.2f} %)")
    _echo(f"  I-q5 pente M-d : {pente:.4f} %/s (plafond {q5['plafond']:.4f})")
    _echo(f"  I-q6 : {q6}")
    _echo(f"  GLOBAL : {global_['global']} — §8 "
          f"{'OUVERT' if global_['section_8_ouverte'] else 'FERMÉ'}")
    _echo(f"  artefact : {SORTIE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
