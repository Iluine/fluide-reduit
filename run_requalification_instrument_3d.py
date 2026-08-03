#!/usr/bin/env python
"""DRIVER — RE-QUALIFICATION DE L'INSTRUMENT 3D.

Exécute le protocole de
`claude/prereg-requalification-instrument-3d-2026-08-04.md` (commit
`33b49be`, écrit AVANT ce fichier).

C'est de la qualification d'APPAREIL : par la règle 1 de `CLAUDE.md`, rien
ici ne prononce quoi que ce soit sur V4, la cadence, le côté ou le budget.

Ce que ce driver regarde au lieu de l'inférer : **l'horloge SM**, relevée
en continu par `nvidia-smi` pendant chaque série. `§A60` avait déduit une
montée en fréquence de temps de frame ; ici elle est observée.

`src/f1_gpu/chrono.py` n'est PAS modifié — il grave « AUCUN timestamp dans
les données retournées ». La trace temporelle est reconstruite HORS de lui,
par somme cumulée des durées qu'il rend déjà, mise à l'échelle du temps mur
mesuré autour de l'appel. L'approximation est déclarée : elle répartit
uniformément le surcoût Python entre frames.

Usage : `.venv/bin/python run_requalification_instrument_3d.py`
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

# ---------------------------------------------------------------- config
ARTEFACT_A60 = pathlib.Path("claude/lectures/paires-appariees-3d-2026-08-04.json")
SORTIE = pathlib.Path("claude/lectures/requalification-instrument-3d-2026-08-04.json")
PREREG = "claude/prereg-requalification-instrument-3d-2026-08-04.md"
COMMIT_PREREG = "33b49be"

N_SCALAIRES, N_STATIQUES = 3, 2
N_CHAMPS = 4 + N_SCALAIRES + N_STATIQUES
N_FENETRES = 3
GRAINE = 12345

IDLE_S = 10.0
DUREE_CIBLE_MS = 30_000.0
PERIODE_SONDE_S = 0.05
FENETRE_GLISSANTE = 100
BANDE_CONVERGENCE = 0.02      # T_conv : ±2 % du plateau, et le reste du temps
BANDE_PERTURBATION = 0.02     # I-w1
BANDE_COHERENCE_A60 = 0.15    # I-w4

# Les mesures, fixées ici, avant tout run (prereg §2).
PLAN: tuple[tuple[str, int, float, bool], ...] = (
    # (nom, côté, idle avant, sonde d'horloge active)
    ("m_i1", 64, IDLE_S, True),
    ("m_i2a", 64, 0.0, True),
    ("m_i2b", 64, 0.0, True),
    ("m_i3", 128, IDLE_S, True),
    ("m_i4", 32, IDLE_S, True),
    ("m_controle_sans_sonde", 64, IDLE_S, False),
)

_REQUETE_SMI = ("clocks.sm,clocks.max.sm,temperature.gpu,power.draw,"
                "clocks_event_reasons.active")


class MesureImpossible(RuntimeError):
    """Le run s'arrête AVANT de produire un chiffre."""


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


class SondeHorloge(threading.Thread):
    """Relève l'horloge SM en continu. Elle observe l'instrument ; I-w1
    vérifie qu'elle ne le perturbe pas."""

    def __init__(self):
        super().__init__(daemon=True)
        self.releves: list[dict] = []
        self._continuer = threading.Event()
        self._continuer.set()

    def run(self):
        while self._continuer.is_set():
            valeur = _lire_smi()
            if valeur is not None:
                self.releves.append({"t": time.perf_counter(), **valeur})
            time.sleep(PERIODE_SONDE_S)

    def arreter(self):
        self._continuer.clear()
        self.join(timeout=2.0)


# ------------------------------------------------------------- les gardes

def garde_kernels_intouches() -> dict:
    diff = _git(["diff", "--stat", "HEAD", "--", "src/f1_gpu/"])
    if diff:
        raise MesureImpossible(f"src/f1_gpu modifié :\n{diff}")
    return {"empreinte_a53": hashlib.sha256(
        fusionne_3d._SOURCE_3D.encode("utf-8")).hexdigest(),
        "empreinte_param_3_2": empreinte_param(N_SCALAIRES, N_STATIQUES),
        "git_diff_src": "vide"}


def lire_ancres() -> dict:
    """Le phénomène à expliquer, relu dans l'artefact de §A60."""
    if not ARTEFACT_A60.exists():
        raise MesureImpossible(f"artefact absent : {ARTEFACT_A60}")
    a60 = json.loads(ARTEFACT_A60.read_text())
    m64 = a60["mesures"]["64"]
    return {"a60_64_demi_1_ms": m64["mediane_demi_1_ms"],
            "a60_64_demi_2_ms": m64["mediane_demi_2_ms"],
            "a60_64_mediane_ms": m64["mediane_ms"],
            "a60_ecart_demi_series": m64["ecart_demi_series"]}


# ------------------------------------------------------------ les mesures

def mesurer(cp, nom: str, cote: int, idle_s: float, sonde_active: bool) -> dict:
    q = cp.asarray(etat_initial_jetable_3d(N_FENETRES, 1, cote, GRAINE,
                                           n_champs=N_CHAMPS))
    tampon = cp.array(q, copy=True)

    def frame():
        pas_f_fusionne_3d_param(q, cp, sortie=q, tampon_etage=tampon,
                                n_statiques=N_STATIQUES)

    # Pré-chrono (hors série) pour dimensionner, PUIS l'idle : l'idle doit
    # être la dernière chose avant la série, sinon il ne mesure rien.
    pre = chronometrer_frames(cp, frame, warmup=5, serie=20)
    duree = pre["stats"]["mediane_ms"]
    serie = max(SERIE_FRAMES, math.ceil(DUREE_CIBLE_MS / duree))

    horloge_avant_idle = _lire_smi()
    if idle_s > 0:
        cp.cuda.Device().synchronize()
        time.sleep(idle_s)
    horloge_apres_idle = _lire_smi()

    sonde = SondeHorloge() if sonde_active else None
    if sonde:
        sonde.start()
    t0 = time.perf_counter()
    # Warmup au PLANCHER gravé : on veut VOIR le transitoire, pas le
    # consommer. Sa trace est conservée et concaténée à la série.
    resultat = chronometrer_frames(cp, frame, warmup=WARMUP_FRAMES, serie=serie)
    t1 = time.perf_counter()
    if sonde:
        sonde.arreter()

    trace = list(resultat["warmup_ms"]) + list(resultat["serie_ms"])
    del q, tampon
    cp.get_default_memory_pool().free_all_blocks()
    return {
        "nom": nom, "cote": cote, "idle_s": idle_s, "sonde_active": sonde_active,
        "serie": serie, "warmup": WARMUP_FRAMES,
        "duree_estimee_ms": duree,
        "t_mur_debut": t0, "t_mur_fin": t1,
        "horloge_avant_idle": horloge_avant_idle,
        "horloge_apres_idle": horloge_apres_idle,
        "trace_ms": trace,
        "stats_serie": resultat["stats"],
        "releves_horloge": sonde.releves if sonde else [],
    }


# ------------------------------------------------------------- la lecture

def _mediane_glissante(x: list[float], fenetre: int) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.size < fenetre:
        return a.copy()
    return np.array([np.median(a[max(0, i - fenetre + 1):i + 1])
                     for i in range(a.size)])


def analyser(m: dict) -> dict:
    """T_conv : le premier instant à partir duquel la médiane glissante
    reste à moins de 2 % du plateau, ET LE RESTE jusqu'à la fin. Un
    critère sans permanence se satisferait d'un passage."""
    trace = m["trace_ms"]
    a = np.asarray(trace, dtype=float)
    n = a.size
    plateau = float(np.median(a[int(0.75 * n):]))
    glissante = _mediane_glissante(trace, FENETRE_GLISSANTE)
    dans_bande = np.abs(glissante - plateau) / plateau <= BANDE_CONVERGENCE
    # dernier indice hors bande ; la convergence est juste après
    hors = np.flatnonzero(~dans_bande)
    indice = int(hors[-1] + 1) if hors.size else 0
    converge = indice < n

    # Axe temps : somme cumulée mise à l'échelle du temps mur mesuré.
    cumul = np.cumsum(a)
    duree_mur = (m["t_mur_fin"] - m["t_mur_debut"]) * 1000.0
    echelle = duree_mur / cumul[-1] if cumul[-1] > 0 else 1.0
    t_ms = cumul * echelle
    t_conv_ms = float(t_ms[indice]) if converge and indice < n else float("nan")

    premier_quart = float(np.median(a[:max(1, n // 4)]))
    releves = m["releves_horloge"]
    horloges = [r["sm_mhz"] for r in releves]
    raisons = sorted({r["raisons"] for r in releves})
    temperatures = [r["temperature_c"] for r in releves]
    # Horloge du dernier quart des relevés — le « plateau » d'horloge.
    plateau_horloge = (float(np.median(horloges[int(0.75 * len(horloges)):]))
                       if horloges else None)
    return {
        "n_frames": n, "duree_mur_ms": duree_mur,
        "plateau_ms": plateau, "premier_quart_ms": premier_quart,
        "rapport_premier_quart_sur_plateau": premier_quart / plateau,
        "indice_convergence": indice, "converge": converge,
        "t_conv_ms": t_conv_ms,
        "frames_avant_convergence": indice,
        "horloge": {
            "n_releves": len(releves),
            "min_mhz": min(horloges) if horloges else None,
            "max_mhz": max(horloges) if horloges else None,
            "plateau_mhz": plateau_horloge,
            "premier_releve_mhz": horloges[0] if horloges else None,
            "raisons_observees": raisons,
            "temperature_min_c": min(temperatures) if temperatures else None,
            "temperature_max_c": max(temperatures) if temperatures else None,
        },
    }


def lire_le_resultat(ancres: dict, mesures: dict, analyses: dict) -> dict:
    a1, a2a, a2b = analyses["m_i1"], analyses["m_i2a"], analyses["m_i2b"]
    a3, a4 = analyses["m_i3"], analyses["m_i4"]
    ctrl = analyses["m_controle_sans_sonde"]

    # I-w1 : la sonde perturbe-t-elle ?
    perturbation = abs(a1["plateau_ms"] - ctrl["plateau_ms"]) / ctrl["plateau_ms"]
    sonde_neutre = perturbation <= BANDE_PERTURBATION

    # I-w2 : le plateau en est-il un ? (raisons de throttle autres qu'idle)
    raisons_suspectes = {}
    for nom, a in analyses.items():
        mauvaises = [r for r in a["horloge"]["raisons_observees"]
                     if r not in ("0x0000000000000000", "0x0000000000000001")]
        if mauvaises:
            raisons_suspectes[nom] = mauvaises
    plateau_propre = not raisons_suspectes

    # I-w3 : l'absence de transitoire ne se prononce pas sur un essai.
    sans_idle_sans_transitoire = [
        a["frames_avant_convergence"] <= WARMUP_FRAMES for a in (a2a, a2b)]
    reproduit_sans_idle = len(set(sans_idle_sans_transitoire)) == 1

    # I-w4 : le run reproduit-il le phénomène qu'il explique ?
    ecart_lent = (abs(a1["premier_quart_ms"] / N_FENETRES * 3
                      - ancres["a60_64_demi_1_ms"])
                  / ancres["a60_64_demi_1_ms"])
    ecart_plateau = (abs(a1["plateau_ms"] - ancres["a60_64_demi_2_ms"])
                     / ancres["a60_64_demi_2_ms"])
    reproduit_a60 = (ecart_lent <= BANDE_COHERENCE_A60
                     and ecart_plateau <= BANDE_COHERENCE_A60)

    # Le transitoire est-il un boost d'horloge, et le même en secondes ?
    t_conv = {n: analyses[n]["t_conv_ms"] for n in ("m_i1", "m_i3", "m_i4")}
    valides = [v for v in t_conv.values() if not math.isnan(v)]
    meme_duree = (bool(valides) and len(valides) == 3
                  and (max(valides) - min(valides)) / max(min(valides), 1e-9)
                  <= 0.5)
    boost_observe = all(
        analyses[n]["horloge"]["premier_releve_mhz"] is not None
        and analyses[n]["horloge"]["plateau_mhz"] is not None
        and analyses[n]["horloge"]["plateau_mhz"]
        > 1.2 * analyses[n]["horloge"]["premier_releve_mhz"]
        for n in ("m_i1", "m_i3", "m_i4"))
    depend_de_l_idle = (a1["frames_avant_convergence"]
                        > 2 * max(a2a["frames_avant_convergence"],
                                  a2b["frames_avant_convergence"], 1))

    # Branches PRÉ-ÉCRITES. Aucun `else` ne prononce par défaut.
    if not sonde_neutre:
        branche, verdict = (
            "I-w1", "INDÉTERMINÉ — la sonde d'horloge perturbe la mesure ; "
                    "seule la trace des temps est lisible")
    elif not plateau_propre:
        branche, verdict = (
            "I-w2", "INDÉTERMINÉ — throttle thermique ou de puissance : le "
                    "plateau n'en est pas un")
    elif not reproduit_a60:
        branche, verdict = (
            "I-w4", "INDÉTERMINÉ — le run ne reproduit pas le phénomène de "
                    "§A60 qu'il prétend expliquer")
    elif not reproduit_sans_idle:
        branche, verdict = (
            "I-w3", "INDÉTERMINÉ — les deux essais sans idle ne concluent "
                    "pas de même")
    elif not boost_observe:
        branche, verdict = (
            "W-C", "aucun transitoire d'HORLOGE observé alors que les temps "
                   "dérivent — le boost n'est pas la cause ; aucune règle de "
                   "warmup n'est adoptée")
    elif depend_de_l_idle or not meme_duree:
        branche, verdict = (
            "W-B", "le transitoire dépend de l'IDLE PRÉALABLE ou du côté — "
                   "un driver ne doit JAMAIS laisser le GPU au repos entre "
                   "deux points ; le warmup se prend au pire cas")
    else:
        branche, verdict = (
            "W-A", "le transitoire EST le boost d'horloge, fonction du temps "
                   "sous charge — le warmup est une DURÉE")

    return {
        "branche": branche, "verdict": verdict,
        "t_conv_ms": t_conv,
        "frames_avant_convergence": {
            n: analyses[n]["frames_avant_convergence"] for n in analyses},
        "plateaux_ms": {n: analyses[n]["plateau_ms"] for n in analyses},
        "premier_quart_sur_plateau": {
            n: analyses[n]["rapport_premier_quart_sur_plateau"] for n in analyses},
        "horloges": {n: analyses[n]["horloge"] for n in analyses},
        "i_w1_perturbation": {"ecart_relatif": perturbation,
                              "bande": BANDE_PERTURBATION,
                              "sonde_neutre": sonde_neutre},
        "i_w2_plateau": {"raisons_suspectes": raisons_suspectes,
                         "plateau_propre": plateau_propre},
        "i_w3_reproduction_sans_idle": {
            "essais": sans_idle_sans_transitoire,
            "reproduit": reproduit_sans_idle,
            "note": ("l'absence de transitoire est le résultat COMMODE — il "
                     "suffirait d'enchaîner les mesures ; il n'est retenu "
                     "que reproduit.")},
        "i_w4_coherence_a60": {
            "ecart_regime_lent": ecart_lent, "ecart_plateau": ecart_plateau,
            "a60_demi_1_ms": ancres["a60_64_demi_1_ms"],
            "a60_demi_2_ms": ancres["a60_64_demi_2_ms"],
            "reproduit": reproduit_a60},
        "regle_de_warmup_proposee": {
            "forme": ("le warmup se termine quand la médiane glissante sur "
                      f"{FENETRE_GLISSANTE} frames reste à moins de "
                      f"{BANDE_CONVERGENCE*100:.0f} % du plateau et LE RESTE"),
            "valeur_par_defaut_ms": (max(valides) if valides else None),
            "note": ("valeur mesurée au PIRE des côtés testés ; le plancher "
                     "gravé (30 frames de warmup, 300 de série) n'est jamais "
                     "abaissé — seulement dépassé."),
        },
    }


# ------------------------------------------------------------------ main

def main() -> int:
    if SORTIE.exists():
        raise MesureImpossible(f"{SORTIE} existe déjà — on n'écrase pas.")
    if _lire_smi() is None:
        raise MesureImpossible("nvidia-smi indisponible : le transitoire ne "
                               "serait qu'inféré, ce que §A60 a déjà fait.")
    try:
        import cupy as cp
    except Exception as erreur:                          # pragma: no cover
        raise MesureImpossible(f"cupy indisponible : {erreur}")

    _echo("== gardes ==")
    gardes = garde_kernels_intouches()
    _echo("  src/f1_gpu intouché")
    ancres = lire_ancres()
    _echo(f"  §A60 à 64³ : demi-séries {ancres['a60_64_demi_1_ms']:.4f} -> "
          f"{ancres['a60_64_demi_2_ms']:.4f} ms — à reproduire")

    _echo(f"== plan ({len(PLAN)} mesures, série ≥ {DUREE_CIBLE_MS/1000:.0f} s) ==")
    mesures, analyses = {}, {}
    for nom, cote, idle, sonde in PLAN:
        _echo(f"  {nom} : {cote}³, idle {idle:.0f} s, sonde "
              f"{'ON ' if sonde else 'OFF'} …")
        m = mesurer(cp, nom, cote, idle, sonde)
        a = analyser(m)
        mesures[nom], analyses[nom] = m, a
        h = a["horloge"]
        _echo(f"      plateau {a['plateau_ms']:8.4f} ms | 1er quart "
              f"{a['premier_quart_ms']:8.4f} ({a['rapport_premier_quart_sur_plateau']:.3f}×)"
              f" | T_conv {a['t_conv_ms']:8.1f} ms sur {a['n_frames']} frames"
              + (f" | horloge {h['premier_releve_mhz']:.0f} -> "
                 f"{h['plateau_mhz']:.0f} MHz" if h["plateau_mhz"] else ""))

    lecture = lire_le_resultat(ancres, mesures, analyses)

    # Les traces complètes sont conservées : un transitoire est une FORME.
    artefact = {
        "protocole": {
            "prereg": PREREG, "commit_prereg": COMMIT_PREREG,
            "nature": ("QUALIFICATION D'INSTRUMENT (règle 1 de CLAUDE.md) — "
                       "ne prononce RIEN sur V4, la cadence, le côté ou le "
                       "budget"),
            "question": ("quel est le transitoire, et quel warmup le dilue ?"),
            "chrono": (f"chrono.py INTOUCHÉ ; warmup au plancher gravé "
                       f"{WARMUP_FRAMES} (on veut VOIR le transitoire, pas le "
                       f"consommer) ; série ≥ {SERIE_FRAMES} ET "
                       f"≥ {DUREE_CIBLE_MS:.0f} ms"),
            "axe_temps": ("somme cumulée des durées de frame, mise à "
                          "l'échelle du temps mur mesuré autour de l'appel — "
                          "répartit uniformément le surcoût Python entre "
                          "frames. Approximation DÉCLARÉE."),
            "git_head": _git(["rev-parse", "--short", "HEAD"]),
        },
        "gardes": gardes,
        "ancres_lues": ancres,
        "mesures": {n: {k: v for k, v in m.items()} for n, m in mesures.items()},
        "analyses": analyses,
        "lecture": lecture,
    }
    SORTIE.parent.mkdir(parents=True, exist_ok=True)
    SORTIE.write_text(json.dumps(artefact, ensure_ascii=False, indent=1))

    _echo("== lecture ==")
    _echo(f"  I-w1 sonde neutre : {lecture['i_w1_perturbation']['sonde_neutre']} "
          f"({lecture['i_w1_perturbation']['ecart_relatif']*100:.2f} %)")
    _echo(f"  I-w2 plateau propre : {lecture['i_w2_plateau']['plateau_propre']}")
    _echo(f"  I-w4 reproduit §A60 : {lecture['i_w4_coherence_a60']['reproduit']}")
    _echo(f"  T_conv : {lecture['t_conv_ms']}")
    regle = lecture["regle_de_warmup_proposee"]
    _echo(f"  warmup proposé : {regle['valeur_par_defaut_ms']} ms")
    _echo(f"  branche {lecture['branche']} — {lecture['verdict']}")
    _echo(f"  artefact : {SORTIE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
