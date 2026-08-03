#!/usr/bin/env python
"""DRIVER — LE COÛT DE F EN 3D.

Exécute le protocole de `claude/prereg-cout-f-3d-2026-08-03.md` (commit
`661237a`, écrit AVANT ce fichier et AVANT le jouet 3D `d4148d1`).

GRANDEUR DE VERDICT : **ρ = M-3 / M-1**, le coût d'un bloc quand le monde
gagne une dimension — mesuré INTRA-RUN, sur la même machine, sous le même
chronomètre. L'ancre archivée du 19/07 ne sert PAS de dénominateur : elle
sert de témoin de reproduction (I-1).

Ce driver REFUSE de mesurer si l'équivalence de motif échoue (I-2), si le
kernel 2D a bougé (garde 2), ou si les seuils recalculés depuis les
artefacts ne retombent pas sur ceux épinglés au prereg. Aucun seuil n'est
saisi à la main dans un chemin de verdict ; ils sont relus et revérifiés.

Usage : `.venv/bin/python run_cout_f_3d.py`
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys

import numpy as np

from src.f1_gpu import substrat_fusionne as fusionne_2d
from src.f1_gpu.chrono import SERIE_FRAMES, WARMUP_FRAMES, chronometrer_frames
from src.f1_gpu.substrat_fusionne import pas_f_fusionne
from src.f1_gpu.substrat_fusionne_3d import (
    TOL_EQUIVALENCE_ATOL,
    TOL_EQUIVALENCE_RTOL,
    attributs_kernel_3d,
    empreinte_source_3d,
    pas_f_fusionne_3d,
)
from src.f1_gpu.substrat_jetable import etat_initial_jetable
from src.f1_gpu.substrat_jetable_3d import (
    etat_initial_jetable_3d,
    pas_f_jetable_3d,
)

# ---------------------------------------------------------------- config
# Épinglée par le prereg §5, jamais réglée après coup.
ARTEFACT_ANCRE = pathlib.Path("outputs/f1/ma_prime.json")
ARTEFACT_BUDGET = pathlib.Path("outputs/f1/ma_quater.json")
SORTIE = pathlib.Path("claude/lectures/cout-f-3d-2026-08-03.json")
PREREG = "claude/prereg-cout-f-3d-2026-08-03.md"
COMMIT_PREREG = "661237a"

N_FENETRES = 3          # M-a′ : « n_fenetres: 3 » — le batch de l'ancre
N_2D = 512              # V4 : n_fov = 512
N_3D = 64               # §A51 : 512² = 64³ = 262 144 cellules
CELLULES_PAR_SLOT = 262_144

TAILLES_TEMOIN_2D = (256, 512, 1024)
TAILLES_TEMOIN_3D = (32, 64, 128)

# Seuils ÉPINGLÉS au prereg §6 — recalculés ci-dessous depuis les
# artefacts ; un désaccord fait échouer le run (ils ne doivent jamais
# diverger, et si l'un bouge c'est un fait, pas un réglage).
RHO_MORT_A_EPINGLE = 1.3196     # F seul >= 16,7 ms
RHO_MORT_B_EPINGLE = 1.1746     # frame complète, non-F 2D retenu
TOLERANCE_SEUIL = 5e-4

# I-1 : bande de reproduction de l'ancre (prereg §7).
BANDE_REPRODUCTION = 0.25
# I-4 : stabilité du coût par cellule à travers les tailles.
BANDE_STABILITE = 0.25

GRAINE = 12345

# ------------------------------------------------- M-6, comptes statiques
#
# Comptes EXACTS, lisibles dans les deux sources CUDA, servant de témoin
# d'attribution (prereg §7, I-3). Ils ne sont pas un modèle de coût : ils
# donnent un ENCADREMENT de ce que la seule arithmétique prédit. Un ρ
# mesuré hors de cet encadrement n'est pas faux — mais il ne s'attribue
# plus à « la dimension » sans nommer ce qui le porte.
#
# Par cellule et par étage RK2, chaque axe exécute UN `divergence_axe`,
# qui appelle 3 `pentes_axe` et 2 `flux_1d`, et lit 3 voisins directs.
#   - minmods         : n_axes × 3 pentes_axe × n_pentes
#   - solveurs HLL    : n_axes × 2 flux_1d
#   - upwinds         : n_axes × 2 flux_1d × (n_tangentielles + 1 scalaire)
#   - lectures `lire` : n_axes × (3 directes + 3 pentes_axe × 3)
COMPTES_STATIQUES = {
    "2D": {"axes": 2, "pentes": 4, "tangentielles": 1, "champs": 4},
    "3D": {"axes": 3, "pentes": 5, "tangentielles": 2, "champs": 5},
}


def _comptes(dim: str) -> dict:
    c = COMPTES_STATIQUES[dim]
    return {
        "minmods": c["axes"] * 3 * c["pentes"],
        "solveurs_hll": c["axes"] * 2,
        "upwinds_tangentiels": c["axes"] * 2 * (c["tangentielles"] + 1),
        "lectures_voisins": c["axes"] * (3 + 3 * 3),
        "octets_lus_par_lecture": c["champs"] * 4,
    }


def _encadrement_statique() -> dict:
    """Rapports 3D/2D de chaque famille comptée — l'encadrement de I-3."""
    a, b = _comptes("2D"), _comptes("3D")
    rapports = {
        "minmods": b["minmods"] / a["minmods"],
        "solveurs_hll": b["solveurs_hll"] / a["solveurs_hll"],
        "upwinds_tangentiels": (b["upwinds_tangentiels"]
                                / a["upwinds_tangentiels"]),
        "lectures_voisins": b["lectures_voisins"] / a["lectures_voisins"],
        "octets_lus": ((b["lectures_voisins"] * b["octets_lus_par_lecture"])
                       / (a["lectures_voisins"] * a["octets_lus_par_lecture"])),
    }
    valeurs = sorted(rapports.values())
    return {"rapports": rapports, "borne_basse": valeurs[0],
            "borne_haute": valeurs[-1],
            "central": float(np.median(list(rapports.values())))}


# ------------------------------------------------------------- utilitaires

class MesureImpossible(RuntimeError):
    """Le run s'arrête AVANT de produire un chiffre. Jamais un `assert`
    (qui disparaît sous `python -O`) dans un chemin de verdict."""


def _echo(message: str) -> None:
    print(message, flush=True)


def _machine(cp) -> dict:
    libre, total = cp.cuda.runtime.memGetInfo()
    props = cp.cuda.runtime.getDeviceProperties(0)
    return {
        "gpu": props["name"].decode() if isinstance(props["name"], bytes)
        else str(props["name"]),
        "vram_totale_mo": total / 1024 ** 2,
        "vram_libre_avant_mo": libre / 1024 ** 2,
        "cupy": cp.__version__,
        "python": ".".join(str(v) for v in sys.version_info[:3]),
    }


def _git(commande: list[str]) -> str:
    return subprocess.run(["git", *commande], capture_output=True,
                          text=True).stdout.strip()


# ------------------------------------------------------------ les gardes

def garde_kernel_2d_intouche() -> dict:
    """Le dénominateur de ρ. `git diff` non vide sur le fichier 2D ⇒ refus
    de mesurer : un rapport contre une référence modifiée n'est pas
    comparable au budget gravé."""
    diff = _git(["diff", "--stat", "HEAD", "--",
                 "src/f1_gpu/substrat_fusionne.py",
                 "src/f1_gpu/substrat_jetable.py"])
    if diff:
        raise MesureImpossible(
            f"les substrats 2D ont un diff non vide :\n{diff}")
    return {
        "empreinte_source_2d": hashlib.sha256(
            fusionne_2d._SOURCE.encode("utf-8")).hexdigest(),
        "empreinte_source_3d": empreinte_source_3d(),
        "git_diff_2d": "vide",
    }


def garde_equivalence(cp) -> dict:
    """I-2 : le fusionné 3D calcule ce qu'il déclare. Vérifiée ICI, dans le
    driver, et pas seulement dans pytest — un test sauté n'est pas un test
    réussi. État SÉVÈRE (vitesses d'ordre 1, cellules sèches) : l'état
    jetable ordinaire ne réveille pas le transport tangentiel, qui y est
    d'ordre u² et tombe sous `atol` (trou trouvé par mutation avant le
    run, cf. `tests/test_f3d_substrat.py`)."""
    rng = np.random.default_rng(4242)
    forme = (2, 2, 16, 16, 16)
    q = np.zeros((2, 2, 5, 16, 16, 16), dtype=np.float32)
    h = 0.6 + 1.2 * rng.random(forme)
    sec = rng.random(forme) < 0.12
    h = np.where(sec, 0.0, h)
    q[:, :, 0] = h
    for champ in (1, 2, 3):
        q[:, :, champ] = np.where(sec, 0.0, h * (rng.random(forme) - 0.5))
    q[:, :, 4] = np.where(sec, 0.0, h * 0.3 * rng.random(forme))

    attendu, _ = pas_f_jetable_3d(q.copy(), np)
    obtenu, _ = pas_f_fusionne_3d(cp.asarray(q), cp)
    obtenu = cp.asnumpy(obtenu)
    ecart = float(np.abs(obtenu - attendu).max())
    if not np.allclose(obtenu, attendu, rtol=TOL_EQUIVALENCE_RTOL,
                       atol=TOL_EQUIVALENCE_ATOL):
        raise MesureImpossible(
            f"I-2 : équivalence de motif 3D ÉCHOUÉE (écart max {ecart:.3e}, "
            f"rtol {TOL_EQUIVALENCE_RTOL}, atol {TOL_EQUIVALENCE_ATOL}). "
            "Le kernel ne calcule pas ce qu'il déclare — aucun chiffre "
            "n'est produit.")
    return {"ecart_max": ecart, "rtol": TOL_EQUIVALENCE_RTOL,
            "atol": TOL_EQUIVALENCE_ATOL, "etat": "sévère (sec + u ~ 1)"}


def lire_ancres() -> dict:
    """§A41 : chaque fait consommé par le verdict est LU PAR UNE MACHINE
    dans un artefact. Les seuils sont RECALCULÉS ici puis confrontés à
    ceux épinglés au prereg — un désaccord arrête le run."""
    for chemin in (ARTEFACT_ANCRE, ARTEFACT_BUDGET):
        if not chemin.exists():
            raise MesureImpossible(f"artefact absent : {chemin}")
    ancre = json.loads(ARTEFACT_ANCRE.read_text())
    budget = json.loads(ARTEFACT_BUDGET.read_text())

    mediane_ancre = ancre["frame_time"]["mediane_ms"]
    n_fenetres = ancre["meta"]["cellule"]["n_fenetres"]
    if n_fenetres != N_FENETRES:
        raise MesureImpossible(
            f"l'ancre M-a′ porte {n_fenetres} fenêtres, le driver en "
            f"mesure {N_FENETRES} : les deux ne sont pas comparables.")
    ancre_slot = mediane_ancre / n_fenetres
    ancre_bloc = ancre_slot / 2

    seuil = budget["lecture_mecanique"]["mort"]["seuil_ms"]
    f_2d = budget["lecture_mecanique"]["parts"]["f_batche_ms"]
    mediane_frame = budget["frame_time"]["mediane_ms"]
    non_f = mediane_frame - f_2d

    rho_a = seuil / f_2d
    rho_b = (seuil - non_f) / f_2d
    for nom, calcule, epingle in (("ρ_A", rho_a, RHO_MORT_A_EPINGLE),
                                  ("ρ_B", rho_b, RHO_MORT_B_EPINGLE)):
        if abs(calcule - epingle) > TOLERANCE_SEUIL:
            raise MesureImpossible(
                f"{nom} recalculé {calcule:.6f} != épinglé au prereg "
                f"{epingle} — les artefacts ne sont plus ceux du "
                "pré-enregistrement. Aucun seuil n'est déplacé : le run "
                "s'arrête.")
    return {
        "ancre_slot_c8_ms": ancre_slot, "ancre_bloc_c4_ms": ancre_bloc,
        "ancre_mediane_brute_ms": mediane_ancre, "ancre_n_fenetres": n_fenetres,
        "seuil_mort_ms": seuil, "f_2d_ms": f_2d,
        "mediane_frame_v4_ms": mediane_frame, "non_f_ms": non_f,
        "n_blocs_v4": budget["meta"]["candidat"]["n_blocs"],
        "n_slots_v4": budget["meta"]["candidat"]["n_slots"],
        "rho_mort_a": rho_a, "rho_mort_b": rho_b,
        "marge_2d_ms": seuil - mediane_frame,
    }


# ------------------------------------------------------------ les mesures

def mesurer_2d(cp, n: int, n_systemes: int) -> dict:
    q = cp.asarray(etat_initial_jetable(N_FENETRES, n, GRAINE))
    if n_systemes == 1:
        q = cp.ascontiguousarray(q[:, :1])
    tampon = cp.empty_like(q)
    resultat = chronometrer_frames(
        cp, lambda: pas_f_fusionne(q, cp, sortie=q, tampon_etage=tampon))
    stats = resultat["stats"]
    cellules = N_FENETRES * n * n
    return _lecture_mesure(stats, cellules, n_systemes, n, "2D")


def mesurer_3d(cp, n: int, n_systemes: int) -> dict:
    q = cp.asarray(etat_initial_jetable_3d(N_FENETRES, n_systemes, n, GRAINE))
    tampon = cp.empty_like(q)
    resultat = chronometrer_frames(
        cp, lambda: pas_f_fusionne_3d(q, cp, sortie=q, tampon_etage=tampon))
    stats = resultat["stats"]
    cellules = N_FENETRES * n * n * n
    return _lecture_mesure(stats, cellules, n_systemes, n, "3D")


def _lecture_mesure(stats: dict, cellules: int, n_systemes: int, n: int,
                    dim: str) -> dict:
    """Une mesure = une frame de `N_FENETRES` fenêtres. Le coût par BLOC
    (une fenêtre, un système) est la grandeur du modèle gravé ; le coût
    par cellule sert au témoin de taille."""
    blocs = N_FENETRES * n_systemes
    # `cote` et NON `n` : `stats_ms` retourne déjà un `n` (le nombre
    # d'échantillons de la série). Une première version écrivait `"n": n`
    # avant `**stats` — la clé était donc écrasée par 300, et le témoin de
    # taille se repliait sur un seul point au lieu de trois. Le premier
    # artefact du run porte le défaut ; il est archivé sous
    # `…-AVANT-CORRECTIF-CLE-N.json`, jamais écrasé.
    return {
        "dimension": dim, "cote": n, "n_systemes": n_systemes,
        "n_fenetres": N_FENETRES, "cellules_par_fenetre": cellules // N_FENETRES,
        **stats,
        "ms_par_slot": stats["mediane_ms"] / N_FENETRES,
        "ms_par_bloc": stats["mediane_ms"] / blocs,
        "ns_par_cellule_systeme": (stats["mediane_ms"] * 1e6
                                   / (cellules * n_systemes)),
    }


def temoin_taille(cp) -> dict:
    """I-4 : le coût par cellule est-il stable en taille ? Si non,
    l'hypothèse « le slot garde sa taille » (§A51 : 512² = 64³) ne se
    transporte pas au budget, et la lecture est INDÉTERMINÉE — quel que
    soit le SENS de l'instabilité (une aubaine de taille encaissée comme
    « dimension » serait la même faute à l'envers)."""
    mesures = {"2D": [], "3D": []}
    for n in TAILLES_TEMOIN_2D:
        _echo(f"    témoin 2D n={n}")
        mesures["2D"].append(mesurer_2d(cp, n, 1))
    for n in TAILLES_TEMOIN_3D:
        _echo(f"    témoin 3D n={n}")
        mesures["3D"].append(mesurer_3d(cp, n, 1))

    lecture = {}
    for dim, serie in mesures.items():
        couts = [m["ns_par_cellule_systeme"] for m in serie]
        etendue = (max(couts) - min(couts)) / min(couts)
        lecture[dim] = {
            "ns_par_cellule": {str(m["cote"]): c for m, c in zip(serie, couts)},
            "etendue_relative": etendue,
            "stable": etendue <= BANDE_STABILITE,
        }
        if len(lecture[dim]["ns_par_cellule"]) != len(serie):
            raise MesureImpossible(
                f"témoin de taille {dim} : {len(serie)} mesures repliées sur "
                f"{len(lecture[dim]['ns_par_cellule'])} entrées — collision "
                "de clé, le témoin ne rend pas ce qu'il a mesuré.")
    return {"mesures": mesures, "lecture": lecture,
            "bande": BANDE_STABILITE,
            "stable_des_deux_cotes": all(v["stable"] for v in lecture.values())}


# ------------------------------------------------------------- la lecture

def lire_le_resultat(ancres: dict, m1: dict, m2: dict, m3: dict, m4: dict,
                     encadrement: dict, taille: dict) -> dict:
    rho = m3["ms_par_bloc"] / m1["ms_par_bloc"]

    # I-1 : reproduction de l'ancre (le chemin 2D, intouché).
    ecart_ancre = abs(m2["ms_par_slot"] - ancres["ancre_slot_c8_ms"]) \
        / ancres["ancre_slot_c8_ms"]
    reproduit = ecart_ancre <= BANDE_REPRODUCTION

    # I-3 : attribution — ρ hors encadrement statique ?
    basse, haute = encadrement["borne_basse"], encadrement["borne_haute"]
    dans_encadrement = 0.5 * basse <= rho <= 2.0 * haute

    # Branches PRÉ-ÉCRITES (prereg §6). L'ordre est celui du document ;
    # aucun `else` ne prononce de PASS ou de FAIL par défaut.
    if not reproduit:
        branche, verdict = "I-1", "INDÉTERMINÉ"
    elif not taille["stable_des_deux_cotes"]:
        branche, verdict = "I-4", "INDÉTERMINÉ"
    elif rho < ancres["rho_mort_b"]:
        branche, verdict = "R-1", "V4 SURVIT au passage 3D (stencil seul)"
    elif rho < ancres["rho_mort_a"]:
        branche, verdict = "R-2", "MORT CONDITIONNELLE de V4 en 3D"
    else:
        branche, verdict = "R-3", "MORT INCONDITIONNELLE de V4 en 3D"

    f_3d = ancres["n_blocs_v4"] * m3["ms_par_bloc"]
    f_3d_transpose = ancres["f_2d_ms"] * rho
    frame_3d = f_3d_transpose + ancres["non_f_ms"]

    return {
        "rho": rho,
        "branche": branche,
        "verdict": verdict,
        "seuils": {"rho_mort_b_conditionnelle": ancres["rho_mort_b"],
                   "rho_mort_a_inconditionnelle": ancres["rho_mort_a"]},
        "i1_reproduction_ancre": {
            "mesure_slot_c8_ms": m2["ms_par_slot"],
            "ancre_slot_c8_ms": ancres["ancre_slot_c8_ms"],
            "ecart_relatif": ecart_ancre, "bande": BANDE_REPRODUCTION,
            "reproduit": reproduit},
        "i3_attribution": {
            "encadrement_statique": encadrement,
            "rho_dans_encadrement_elargi": dans_encadrement,
            "note": ("ρ hors [0.5·borne_basse, 2·borne_haute] ⇒ l'écart "
                     "n'est pas l'arithmétique ; registres et occupancy "
                     "sont à lire, et le résultat se prononce AVEC réserve "
                     "d'implémentation nommée.")},
        "i4_stabilite_taille": taille["lecture"],
        "linearite_systemes": {
            "2D_S2_sur_S1": m2["mediane_ms"] / m1["mediane_ms"],
            "3D_S2_sur_S1": m4["mediane_ms"] / m3["mediane_ms"],
            "note": ("l'ancre gravée pose 1.685 / 0.843 ≈ 2.00 en 2D. Le "
                     "modèle du budget compte des BLOCS : si la linéarité "
                     "tombe en 3D, le compte en blocs de V4 ne s'y "
                     "transporte pas tel quel.")},
        "budget_3d_transpose": {
            "f_3d_ms_par_blocs_mesures": f_3d,
            "f_3d_ms_par_rho_sur_ancre": f_3d_transpose,
            "non_f_2d_retenu_ms": ancres["non_f_ms"],
            "frame_3d_estimee_ms": frame_3d,
            "seuil_ms": ancres["seuil_mort_ms"],
            "marge_ms": ancres["seuil_mort_ms"] - frame_3d,
            "note": ("le non-F est RETENU CONSTANT — hypothèse neutre sur "
                     "le nombre de cellules (identique), mais les halos "
                     "passent de ~2 % (512²) à ~20 % (64³) des cellules : "
                     "le non-F 3D est vraisemblablement SUPÉRIEUR, donc "
                     "cette estimation est GÉNÉREUSE envers V4.")},
    }


# ------------------------------------------------------------------ main

def main() -> int:
    try:
        import cupy as cp
    except Exception as erreur:                          # pragma: no cover
        raise MesureImpossible(
            f"cupy indisponible — aucune mesure valide sans GPU : {erreur}")

    if SORTIE.exists():
        raise MesureImpossible(
            f"{SORTIE} existe déjà — un artefact verdict-grade ne s'écrase "
            "pas, et une lecture ne se relance pas par-dessus son archive. "
            "Renommer l'ancien avant de relancer.")

    _echo("== gardes ==")
    gardes = garde_kernel_2d_intouche()
    _echo("  kernel 2D intouché (git diff vide)")
    equivalence = garde_equivalence(cp)
    _echo(f"  I-2 équivalence de motif 3D OK (écart "
          f"{equivalence['ecart_max']:.3e})")
    ancres = lire_ancres()
    _echo(f"  ancres relues : bloc 2D {ancres['ancre_bloc_c4_ms']:.5f} ms ; "
          f"seuils ρ_B {ancres['rho_mort_b']:.4f} / "
          f"ρ_A {ancres['rho_mort_a']:.4f}")

    _echo("== mesures ==")
    _echo("  M-1  2D S=1 512²")
    m1 = mesurer_2d(cp, N_2D, 1)
    _echo(f"       {m1['ms_par_bloc']:.4f} ms/bloc")
    _echo("  M-2  2D S=2 512²")
    m2 = mesurer_2d(cp, N_2D, 2)
    _echo(f"       {m2['ms_par_slot']:.4f} ms/slot")
    _echo("  M-3  3D S=1 64³")
    m3 = mesurer_3d(cp, N_3D, 1)
    _echo(f"       {m3['ms_par_bloc']:.4f} ms/bloc")
    _echo("  M-4  3D S=2 64³")
    m4 = mesurer_3d(cp, N_3D, 2)
    _echo(f"       {m4['ms_par_slot']:.4f} ms/slot")
    _echo("  M-5  témoin de taille")
    taille = temoin_taille(cp)
    _echo("  M-6  attributs des kernels")
    attributs_3d = attributs_kernel_3d(cp)
    kernel_2d = fusionne_2d._obtenir_kernel(cp)
    attributs_2d = {"registres_par_thread": int(kernel_2d.num_regs),
                    "memoire_locale_octets": int(kernel_2d.local_size_bytes),
                    "max_threads_par_bloc": int(kernel_2d.max_threads_per_block)}
    encadrement = _encadrement_statique()

    lecture = lire_le_resultat(ancres, m1, m2, m3, m4, encadrement, taille)

    artefact = {
        "protocole": {
            "prereg": PREREG, "commit_prereg": COMMIT_PREREG,
            "question": ("ρ = coût d'un bloc en 3D / coût d'un bloc en 2D, "
                         "mesuré INTRA-RUN entre deux kernels fusionnés de "
                         "la même famille"),
            "decision_qui_en_depend": ("V4 survit-il au passage 3D, ou la "
                                       "porte 33,3 devient-elle LA décision ?"),
            "chrono": (f"B6 cuda-Events, warmup {WARMUP_FRAMES} EXCLU, "
                       f"série {SERIE_FRAMES}, médiane ET p99"),
            "slot": (f"{CELLULES_PAR_SLOT} cellules — {N_2D}² en 2D, "
                     f"{N_3D}³ en 3D (§A51, transposition exacte)"),
            "git_head": _git(["rev-parse", "--short", "HEAD"]),
            "portee": ("F JOUET des deux côtés : coût-représentatif, pas "
                       "physique-jeu. Ne prononce RIEN sur le schéma eau "
                       "3D, sur le c 3D, sur le rendu, ni sur la VRAM."),
            "run_anterieur_conserve": (
                "claude/lectures/cout-f-3d-2026-08-03-AVANT-CORRECTIF-CLE-N"
                ".json — même protocole, même code de mesure ; défaut de "
                "REPORTING seul (clé `n` du témoin de taille écrasée par le "
                "nombre d'échantillons, trois points repliés sur un). Les "
                "temps, ρ et la branche y sont valides et lisibles ; il est "
                "conservé pour que les deux ρ soient confrontables."),
        },
        "machine": _machine(cp),
        "gardes": {**gardes, "equivalence_de_motif": equivalence},
        "ancres_lues": ancres,
        "m1_2d_s1": m1, "m2_2d_s2": m2, "m3_3d_s1": m3, "m4_3d_s2": m4,
        "m5_temoin_taille": taille,
        "m6_attributs": {"2D": attributs_2d, "3D": attributs_3d,
                         "comptes_statiques": {"2D": _comptes("2D"),
                                               "3D": _comptes("3D")}},
        "lecture": lecture,
    }
    SORTIE.parent.mkdir(parents=True, exist_ok=True)
    if SORTIE.exists():
        raise MesureImpossible(
            f"{SORTIE} existe déjà — un artefact verdict-grade ne "
            "s'écrase pas. Renommer l'ancien avant de relancer.")
    SORTIE.write_text(json.dumps(artefact, ensure_ascii=False, indent=1))

    _echo("== lecture ==")
    _echo(f"  ρ = {lecture['rho']:.4f}   (seuils : ρ_B "
          f"{ancres['rho_mort_b']:.4f} · ρ_A {ancres['rho_mort_a']:.4f})")
    _echo(f"  branche {lecture['branche']} — {lecture['verdict']}")
    _echo(f"  artefact : {SORTIE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
