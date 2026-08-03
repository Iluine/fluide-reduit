#!/usr/bin/env python
"""DRIVER — LE MULTIPLICATEUR `c` EN 3D.

Exécute le protocole de `claude/prereg-multiplicateur-c-3d-2026-08-04.md`
(commit `c298a8b`, écrit AVANT le kernel paramétré `f368de9`).

GRANDEUR DE VERDICT : **`C` = coût d'une fenêtre 64³ portant le
vocabulaire réel** (3 scalaires advectés + 2 champs statiques lus), d'où
le cap fermé `n = (budget − non-F) / C` aux deux cadences.

Le driver REFUSE de mesurer si l'identité bit-pour-bit à (1,0) échoue, si
l'équivalence de motif échoue à une configuration, si les lectures des
champs statiques se révèlent éliminées par le compilateur, si le kernel de
§A53 a bougé, ou si les seuils recalculés depuis les artefacts ne
retombent pas sur ceux épinglés au prereg.

Usage : `.venv/bin/python run_multiplicateur_c_3d.py`
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys

import numpy as np

from src.f1_gpu import substrat_fusionne_3d as fusionne_3d
from src.f1_gpu.chrono import SERIE_FRAMES, WARMUP_FRAMES, chronometrer_frames
from src.f1_gpu.substrat_fusionne_3d import (
    TOL_EQUIVALENCE_ATOL,
    TOL_EQUIVALENCE_RTOL,
    pas_f_fusionne_3d,
)
from src.f1_gpu.substrat_fusionne_3d_param import (
    attributs_param,
    comptes_statiques_param,
    empreinte_param,
    pas_f_fusionne_3d_param,
)
from src.f1_gpu.substrat_jetable_3d import (
    CHAMPS_HYPERBOLIQUES,
    etat_initial_jetable_3d,
    pas_f_jetable_3d,
)

# ---------------------------------------------------------------- config
ARTEFACT_A53 = pathlib.Path("claude/lectures/cout-f-3d-2026-08-03.json")
SORTIE = pathlib.Path("claude/lectures/multiplicateur-c-3d-2026-08-04.json")
PREREG = "claude/prereg-multiplicateur-c-3d-2026-08-04.md"
COMMIT_PREREG = "c298a8b"

N_FENETRES = 3
N_3D = 64
GRAINE = 12345
N_SLOTS_V4 = 11        # §A51 : la dégressivité s'est effondrée en 3D

# Les six points, dans l'ordre du prereg §4.
CONFIGURATIONS: tuple[tuple[str, int, int], ...] = (
    ("m_c1", 1, 0), ("m_c2", 2, 0), ("m_c3", 3, 0), ("m_c4", 4, 0),
    ("m_c5", 3, 2), ("m_c6", 1, 2),
)
POINTS_PENTE = ("m_c1", "m_c2", "m_c3", "m_c4")

# Seuils ÉPINGLÉS au prereg §5 — recalculés ci-dessous depuis l'artefact ;
# un désaccord arrête le run. Aucun n'est déplaçable.
BORNE_HAUTE_RHO_C = 1.50          # 7,5 / 5 — champs facturés comme systèmes
C_MAX_60HZ_EPINGLE = 1.3514       # (16,7 − non-F) / 11
C_MAX_30HZ_EPINGLE = 2.8635       # (33,333 − non-F) / 11
TOLERANCE_SEUIL = 5e-4

BANDE_REPRODUCTION = 0.10         # I-c1
SEUIL_ELIMINATION = 0.02          # I-c3 : δ < 2 % de M-c1
BANDE_AFFINITE = 0.05             # I-c4
BANDE_COHERENCE = 0.05            # I-c5

BUDGET_60HZ = 16.7
BUDGET_30HZ = 1000.0 / 30.0


class MesureImpossible(RuntimeError):
    """Le run s'arrête AVANT de produire un chiffre."""


def _echo(m: str) -> None:
    print(m, flush=True)


def _git(commande: list[str]) -> str:
    return subprocess.run(["git", *commande], capture_output=True,
                          text=True).stdout.strip()


def _machine(cp) -> dict:
    libre, total = cp.cuda.runtime.memGetInfo()
    props = cp.cuda.runtime.getDeviceProperties(0)
    nom = props["name"]
    return {"gpu": nom.decode() if isinstance(nom, bytes) else str(nom),
            "vram_totale_mo": total / 1024 ** 2,
            "vram_libre_avant_mo": libre / 1024 ** 2,
            "cupy": cp.__version__,
            "python": ".".join(str(v) for v in sys.version_info[:3])}


def _etat_severe(n_fenetres: int, n_systemes: int, n: int, n_champs: int):
    """État qui allume TOUTES les branches (leçon de §A53 : l'état jetable
    ordinaire laisse le transport tangentiel sous `atol`)."""
    rng = np.random.default_rng(4242)
    forme = (n_fenetres, n_systemes, n, n, n)
    q = np.zeros((n_fenetres, n_systemes, n_champs, n, n, n), dtype=np.float32)
    h = 0.6 + 1.2 * rng.random(forme)
    sec = rng.random(forme) < 0.12
    h = np.where(sec, 0.0, h)
    q[:, :, 0] = h
    for champ in (1, 2, 3):
        q[:, :, champ] = np.where(sec, 0.0, h * (rng.random(forme) - 0.5))
    for champ in range(4, n_champs):
        q[:, :, champ] = np.where(sec, 0.0, h * 0.3 * rng.random(forme))
    return q


# ------------------------------------------------------------- les gardes

def garde_kernel_a53_intouche() -> dict:
    diff = _git(["diff", "--stat", "HEAD", "--",
                 "src/f1_gpu/substrat_fusionne_3d.py"])
    if diff:
        raise MesureImpossible(
            f"le kernel 3D de §A53 a un diff non vide :\n{diff}")
    return {"empreinte_a53": hashlib.sha256(
        fusionne_3d._SOURCE_3D.encode("utf-8")).hexdigest(),
        "git_diff_a53": "vide",
        "empreintes_param": {f"{ns}_{nst}": empreinte_param(ns, nst)
                             for _, ns, nst in CONFIGURATIONS}}


def garde_identite_bit_pour_bit(cp) -> dict:
    """LE VERROU D'ENTRÉE. À (1, 0) le kernel paramétré doit rendre
    EXACTEMENT la sortie de §A53 — il en est le dénominateur."""
    ecarts = {}
    for n_systemes in (1, 2):
        q0 = _etat_severe(2, n_systemes, 16, 5)
        attendu, _ = pas_f_fusionne_3d(cp.asarray(q0), cp)
        obtenu, _ = pas_f_fusionne_3d_param(cp.asarray(q0), cp)
        ecart = float(cp.abs(attendu - obtenu).max())
        ecarts[f"S{n_systemes}"] = ecart
        if ecart != 0.0:
            raise MesureImpossible(
                f"identité bit-pour-bit ÉCHOUÉE à (1,0), S={n_systemes} : "
                f"écart {ecart:.3e}. La paramétrisation a déplacé le motif ; "
                "le rapport ne mesurerait plus le vocabulaire mais la "
                "réécriture.")
    return {"ecarts": ecarts, "verdict": "identique bit pour bit"}


def garde_equivalence(cp) -> dict:
    """I-c2, à CHAQUE configuration, sur l'état sévère."""
    resultats = {}
    for nom, ns, nst in CONFIGURATIONS:
        n_champs = CHAMPS_HYPERBOLIQUES + ns + nst
        q0 = _etat_severe(2, 1, 16, n_champs)
        attendu, _ = pas_f_jetable_3d(q0.copy(), np, n_statiques=nst)
        obtenu, _ = pas_f_fusionne_3d_param(cp.asarray(q0), cp,
                                            n_statiques=nst)
        obtenu = cp.asnumpy(obtenu)
        ecart = float(np.abs(obtenu - attendu).max())
        resultats[nom] = ecart
        if not np.allclose(obtenu, attendu, rtol=TOL_EQUIVALENCE_RTOL,
                           atol=TOL_EQUIVALENCE_ATOL):
            raise MesureImpossible(
                f"I-c2 : équivalence de motif ÉCHOUÉE à {nom} "
                f"(ns={ns}, nst={nst}), écart {ecart:.3e}.")
    return {"ecarts_max": resultats, "rtol": TOL_EQUIVALENCE_RTOL,
            "atol": TOL_EQUIVALENCE_ATOL, "etat": "sévère"}


def garde_statiques_lus(cp) -> dict:
    """I-c3 APPLIQUÉE AUX STATIQUES — ajout de la session au protocole.

    Le prereg scope I-c3 aux scalaires ; le risque est identique, et pire,
    pour les statiques : rien ne dépend d'eux, donc `nvcc` les supprime et
    le chronomètre mesure un champ qui n'existe pas. Une garde AJOUTÉE ne
    relâche rien — elle ne peut que rendre la lecture plus stricte."""
    q0 = _etat_severe(1, 1, 16, 7)
    neutre, _ = pas_f_fusionne_3d_param(cp.asarray(q0), cp, n_statiques=2,
                                        poids_statique=0.0)
    charge, _ = pas_f_fusionne_3d_param(cp.asarray(q0), cp, n_statiques=2,
                                        poids_statique=1.0)
    ecart = float(cp.abs(neutre - charge).max())
    if ecart <= 1e-6:
        raise MesureImpossible(
            "les lectures des champs STATIQUES ont été éliminées par le "
            "compilateur (poids 0 et poids 1 donnent la même sortie) — tout "
            "coût mesuré pour eux serait fabriqué.")
    sans, _ = pas_f_fusionne_3d_param(
        cp.asarray(np.ascontiguousarray(q0[:, :, :5])), cp, n_statiques=0)
    neutralite = bool((cp.asnumpy(neutre)[:, :, :5] == cp.asnumpy(sans)).all())
    if not neutralite:
        raise MesureImpossible(
            "le repli à poids nul n'est PAS neutre : le mécanisme qui rend "
            "les lectures inéliminables change le résultat.")
    return {"ecart_poids_0_vs_1": ecart, "repli_neutre_bit_pour_bit": True}


def lire_ancres() -> dict:
    """§A41 : les faits du verdict sont LUS PAR UNE MACHINE dans un
    artefact, et les seuils RECALCULÉS puis confrontés au prereg."""
    if not ARTEFACT_A53.exists():
        raise MesureImpossible(f"artefact absent : {ARTEFACT_A53}")
    a53 = json.loads(ARTEFACT_A53.read_text())
    bloc_a53 = a53["m3_3d_s1"]["ms_par_bloc"]
    non_f = a53["ancres_lues"]["non_f_ms"]
    seuil = a53["ancres_lues"]["seuil_mort_ms"]
    if abs(seuil - BUDGET_60HZ) > 1e-9:
        raise MesureImpossible(
            f"le seuil de l'artefact ({seuil}) n'est pas le budget 60 Hz "
            f"({BUDGET_60HZ}).")
    c_max_60 = (BUDGET_60HZ - non_f) / N_SLOTS_V4
    c_max_30 = (BUDGET_30HZ - non_f) / N_SLOTS_V4
    for nom, calcule, epingle in (("C_max 60 Hz", c_max_60, C_MAX_60HZ_EPINGLE),
                                  ("C_max 30 Hz", c_max_30, C_MAX_30HZ_EPINGLE)):
        if abs(calcule - epingle) > TOLERANCE_SEUIL:
            raise MesureImpossible(
                f"{nom} recalculé {calcule:.6f} != épinglé au prereg "
                f"{epingle} — les artefacts ne sont plus ceux du "
                "pré-enregistrement. Aucun seuil n'est déplacé.")
    return {"bloc_a53_ms": bloc_a53, "non_f_ms": non_f, "seuil_ms": seuil,
            "rho_a53": a53["lecture"]["rho"], "n_slots_v4": N_SLOTS_V4,
            "c_max_60hz": c_max_60, "c_max_30hz": c_max_30,
            "budget_30hz_ms": BUDGET_30HZ}


# ------------------------------------------------------------ les mesures

def mesurer(cp, ns: int, nst: int) -> dict:
    n_champs = CHAMPS_HYPERBOLIQUES + ns + nst
    q = cp.asarray(etat_initial_jetable_3d(N_FENETRES, 1, N_3D, GRAINE,
                                           n_champs=n_champs))
    tampon = cp.array(q, copy=True)
    resultat = chronometrer_frames(
        cp, lambda: pas_f_fusionne_3d_param(q, cp, sortie=q,
                                            tampon_etage=tampon,
                                            n_statiques=nst))
    stats = resultat["stats"]
    return {"n_scalaires": ns, "n_statiques": nst, "n_champs": n_champs,
            "cote": N_3D, "n_fenetres": N_FENETRES, **stats,
            "ms_par_bloc": stats["mediane_ms"] / N_FENETRES,
            "comptes_statiques": comptes_statiques_param(ns, nst),
            "attributs": attributs_param(cp, ns, nst)}


def _ajustement_affine(xs, ys):
    """Pente et ordonnée par moindres carrés, plus le résidu maximal —
    le résidu est ce qui autorise (ou non) à parler de « pente »."""
    pente, ordonnee = np.polyfit(np.asarray(xs, dtype=float),
                                 np.asarray(ys, dtype=float), 1)
    residus = [y - (pente * x + ordonnee) for x, y in zip(xs, ys)]
    return float(pente), float(ordonnee), float(max(abs(r) for r in residus))


# ------------------------------------------------------------- la lecture

def lire_le_resultat(ancres: dict, mesures: dict) -> dict:
    m_c1 = mesures["m_c1"]["ms_par_bloc"]
    xs = [mesures[nom]["n_scalaires"] for nom in POINTS_PENTE]
    ys = [mesures[nom]["ms_par_bloc"] for nom in POINTS_PENTE]
    delta_scalaire, ordonnee, residu_max = _ajustement_affine(xs, ys)
    delta_statique = (mesures["m_c6"]["ms_par_bloc"] - m_c1) / 2.0
    C = mesures["m_c5"]["ms_par_bloc"]
    rho_c = C / m_c1

    # I-c1 : reproduction du point de §A53.
    ecart_a53 = abs(m_c1 - ancres["bloc_a53_ms"]) / ancres["bloc_a53_ms"]
    reproduit = ecart_a53 <= BANDE_REPRODUCTION

    # I-c3 : élimination par le compilateur (garde du côté FAVORABLE).
    elimination_possible = delta_scalaire < SEUIL_ELIMINATION * m_c1
    elimination_statique = abs(delta_statique) < SEUIL_ELIMINATION * m_c1

    # I-c4 : affinité en k.
    affine = residu_max <= BANDE_AFFINITE * m_c1

    # I-c5 : cohérence croisée du découpage advecté / statique.
    attendu_c5 = mesures["m_c3"]["ms_par_bloc"] + 2.0 * delta_statique
    ecart_coherence = abs(C - attendu_c5) / C
    coherent = ecart_coherence <= BANDE_COHERENCE

    # I-c6 : débordement de registres.
    spills = {nom: m["attributs"]["memoire_locale_octets"]
              for nom, m in mesures.items()
              if m["attributs"]["memoire_locale_octets"] > 0}

    n_60 = (ancres["seuil_ms"] - ancres["non_f_ms"]) / C
    n_30 = (BUDGET_30HZ - ancres["non_f_ms"]) / C

    # Branches PRÉ-ÉCRITES. Aucun `else` ne prononce par défaut.
    if not reproduit:
        branche, verdict = "I-c1", "INDÉTERMINÉ — harnais non reproduit"
    elif elimination_possible:
        branche, verdict = (
            "I-c3", "INDÉTERMINÉ — δ sous le seuil d'élimination")
    elif not affine:
        branche, verdict = (
            "I-c4", "PENTE NON REPORTÉE — points mesurés seuls")
    elif rho_c > BORNE_HAUTE_RHO_C:
        branche, verdict = (
            "C-C", "la borne haute ×1,50 est FAUSSE — chiffre non consommé "
                   "avant que la cause soit nommée")
    elif C <= ancres["c_max_30hz"]:
        branche, verdict = (
            "C-A", "la porte 33,3 tient V4 (11 slots) à 30 Hz")
    else:
        branche, verdict = (
            "C-B", "même 30 Hz ne tient pas V4 — l'arbitrage change de nature")

    return {
        "C_vocabulaire_reel_ms": C,
        "rho_c": rho_c,
        "branche": branche,
        "verdict": verdict,
        "delta_scalaire_ms": delta_scalaire,
        "delta_statique_ms": delta_statique,
        "ordonnee_ms": ordonnee,
        "residu_max_ms": residu_max,
        "seuils": {"borne_haute_rho_c": BORNE_HAUTE_RHO_C,
                   "c_max_60hz": ancres["c_max_60hz"],
                   "c_max_30hz": ancres["c_max_30hz"]},
        "cap_ferme": {
            "n_slots_60hz": n_60, "n_slots_60hz_entiers": int(n_60),
            "n_slots_30hz": n_30, "n_slots_30hz_entiers": int(n_30),
            "v4_demande": N_SLOTS_V4,
            "cap_a53_a_1_scalaire_60hz": (
                (ancres["seuil_ms"] - ancres["non_f_ms"])
                / ancres["bloc_a53_ms"]),
            "note": ("le non-F est RETENU CONSTANT (halos ~2 % -> ~20 %) : "
                     "le cap reste un MAJORANT. Cette mesure le resserre, "
                     "elle ne le transforme pas en plancher.")},
        "i_c1_reproduction": {"m_c1_ms": m_c1,
                              "bloc_a53_ms": ancres["bloc_a53_ms"],
                              "ecart_relatif": ecart_a53,
                              "bande": BANDE_REPRODUCTION,
                              "reproduit": reproduit},
        "i_c3_elimination": {
            "seuil_relatif": SEUIL_ELIMINATION,
            "seuil_absolu_ms": SEUIL_ELIMINATION * m_c1,
            "delta_scalaire_sous_seuil": elimination_possible,
            "delta_statique_sous_seuil": elimination_statique,
            "note": ("un δ nul et une élimination par le compilateur sont "
                     "indistinguables au chronomètre ; ce critère ne se "
                     "déclenche QUE du côté favorable, et c'est sa raison "
                     "d'être. Le volet STATIQUE est un ajout de la session "
                     "au protocole — une garde ajoutée ne relâche rien.")},
        "i_c4_affinite": {"residu_max_ms": residu_max,
                          "bande_ms": BANDE_AFFINITE * m_c1,
                          "affine": affine,
                          "points": dict(zip(map(str, xs), ys))},
        "i_c5_coherence": {"C_mesure_ms": C, "C_attendu_ms": attendu_c5,
                           "ecart_relatif": ecart_coherence,
                           "bande": BANDE_COHERENCE, "coherent": coherent},
        "i_c6_registres": {"debordements": spills,
                           "registres": {nom: m["attributs"]
                                         ["registres_par_thread"]
                                         for nom, m in mesures.items()}},
    }


# ------------------------------------------------------------------ main

def main() -> int:
    if SORTIE.exists():
        raise MesureImpossible(
            f"{SORTIE} existe déjà — un artefact verdict-grade ne s'écrase "
            "pas. Renommer l'ancien avant de relancer.")
    try:
        import cupy as cp
    except Exception as erreur:                          # pragma: no cover
        raise MesureImpossible(f"cupy indisponible : {erreur}")

    _echo("== gardes ==")
    gardes = {"kernel_a53": garde_kernel_a53_intouche()}
    _echo("  kernel de §A53 intouché")
    gardes["identite_bit_pour_bit"] = garde_identite_bit_pour_bit(cp)
    _echo("  identité bit-pour-bit à (1,0) : écart exactement 0")
    gardes["equivalence"] = garde_equivalence(cp)
    _echo("  I-c2 équivalence de motif OK aux six configurations")
    gardes["statiques_lus"] = garde_statiques_lus(cp)
    _echo(f"  statiques RÉELLEMENT lus (poids 0 vs 1 : "
          f"{gardes['statiques_lus']['ecart_poids_0_vs_1']:.3e})")
    ancres = lire_ancres()
    _echo(f"  ancres relues : bloc §A53 {ancres['bloc_a53_ms']:.4f} ms ; "
          f"C_max 30 Hz {ancres['c_max_30hz']:.4f} ms")

    _echo("== mesures ==")
    mesures = {}
    for nom, ns, nst in CONFIGURATIONS:
        mesures[nom] = mesurer(cp, ns, nst)
        _echo(f"  {nom}  ns={ns} nst={nst} nc={mesures[nom]['n_champs']} : "
              f"{mesures[nom]['ms_par_bloc']:.4f} ms/bloc "
              f"(p99 {mesures[nom]['p99_ms']:.3f})")

    lecture = lire_le_resultat(ancres, mesures)

    artefact = {
        "protocole": {
            "prereg": PREREG, "commit_prereg": COMMIT_PREREG,
            "question": ("coût marginal d'un scalaire advecté et d'un champ "
                         "statique lu, en 3D ; d'où C, coût d'une fenêtre "
                         "64³ à vocabulaire réel de §A51"),
            "decision_qui_en_depend": (
                "l'ordre des gestes de §A54-5 : fermer la fourchette du cap "
                "AVANT d'arbitrer la cadence"),
            "chrono": (f"B6 cuda-Events, warmup {WARMUP_FRAMES} EXCLU, "
                       f"série {SERIE_FRAMES}, médiane ET p99"),
            "git_head": _git(["rev-parse", "--short", "HEAD"]),
            "portee": ("F JOUET. Ne prononce RIEN sur le schéma eau 3D, le "
                       "rendu, le non-F 3D, ni f16. Statiques lus dans le "
                       "HALO (fidèle pour b0, MAJORANT pour id-matériau) et "
                       "mesurés en f32 alors que §A51 range e_th et ρ_s en "
                       "f16 : majorant de coût, donc minorant du cap."),
        },
        "machine": _machine(cp),
        "gardes": gardes,
        "ancres_lues": ancres,
        "mesures": mesures,
        "lecture": lecture,
    }
    SORTIE.parent.mkdir(parents=True, exist_ok=True)
    SORTIE.write_text(json.dumps(artefact, ensure_ascii=False, indent=1))

    _echo("== lecture ==")
    _echo(f"  δ scalaire = {lecture['delta_scalaire_ms']:+.4f} ms  |  "
          f"δ statique = {lecture['delta_statique_ms']:+.4f} ms")
    _echo(f"  C = {lecture['C_vocabulaire_reel_ms']:.4f} ms   "
          f"(ρ_c = {lecture['rho_c']:.4f}, borne haute "
          f"{BORNE_HAUTE_RHO_C})")
    cap = lecture["cap_ferme"]
    _echo(f"  cap fermé : {cap['n_slots_60hz']:.2f} fenêtres à 60 Hz · "
          f"{cap['n_slots_30hz']:.2f} à 30 Hz  (V4 en demande "
          f"{cap['v4_demande']})")
    _echo(f"  branche {lecture['branche']} — {lecture['verdict']}")
    _echo(f"  artefact : {SORTIE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
