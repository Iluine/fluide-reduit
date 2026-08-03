#!/usr/bin/env python
"""DRIVER — PAIRES APPARIÉES : l'alignement de warps pèse-t-il ?

Exécute le protocole de `claude/prereg-paires-appariees-3d-2026-08-04.md`
(commit `2acf0c5`, écrit AVANT ce fichier).

GRANDEUR DE VERDICT : la **différence seconde**

    Δ(s) = c(s) − [ c(s−2) + c(s+2) ] / 2

sur quatre triplets centrés sur un multiple de 32. Elle annihile toute
tendance LISSE en taille et ne laisse que ce qui est DISCRET en `s`.

Deux choses que ce driver fait et que le précédent ne faisait pas :
la série se dimensionne sur la DURÉE de frame (plancher gravé jamais
abaissé, seulement dépassé), et **chaque point mesure son propre bruit**
par l'écart entre ses deux demi-séries — ce qui rend la PUISSANCE
mesurable, donc un résultat nul prononçable.

Usage : `.venv/bin/python run_paires_appariees_3d.py`
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import subprocess
import sys

import numpy as np

from src.f1_gpu import substrat_fusionne_3d as fusionne_3d
from src.f1_gpu.chrono import (
    SERIE_FRAMES,
    WARMUP_FRAMES,
    chronometrer_frames,
    stats_ms,
)
from src.f1_gpu.substrat_fusionne_3d_param import (
    attributs_param,
    empreinte_param,
    pas_f_fusionne_3d_param,
)
from src.f1_gpu.substrat_jetable_3d import etat_initial_jetable_3d

# ---------------------------------------------------------------- config
ARTEFACT_A55 = pathlib.Path("claude/lectures/multiplicateur-c-3d-2026-08-04.json")
SORTIE = pathlib.Path("claude/lectures/paires-appariees-3d-2026-08-04.json")
PREREG = "claude/prereg-paires-appariees-3d-2026-08-04.md"
COMMIT_PREREG = "2acf0c5"

N_SCALAIRES, N_STATIQUES = 3, 2
N_CHAMPS = 4 + N_SCALAIRES + N_STATIQUES        # 9 — le vocabulaire réel
N_FENETRES = 3
GRAINE = 12345

# Les quatre triplets, FIXÉS ICI, avant tout run (prereg §1). Ajouter ou
# retirer un côté après lecture fabriquerait le contraste — faute exacte
# de §A59.
TRIPLETS: tuple[tuple[int, int, int], ...] = (
    (30, 32, 34), (62, 64, 66), (94, 96, 98), (126, 128, 130))
COTES: tuple[int, ...] = tuple(sorted({s for t in TRIPLETS for s in t}))

# Dimensionnement de la série (prereg §2) — le plancher gravé
# `src/f1_gpu/chrono.py:3-6` n'est jamais abaissé, seulement dépassé.
DUREE_CIBLE_MS = 3000.0
FRAMES_PRE_CHRONO = 20

BANDE_REPRODUCTION = 0.10     # I-p1
BANDE_DEMI_SERIES = 0.03      # I-p2
EFFET_MINIMAL_PERTINENT = 0.10  # I-p3 — dérivé, non choisi (prereg §4)
BANDE_STABILITE = 0.10        # S-A / S-B


class MesureImpossible(RuntimeError):
    """Le run s'arrête AVANT de produire un chiffre."""


def _echo(m: str) -> None:
    print(m, flush=True)


def _git(c: list[str]) -> str:
    return subprocess.run(["git", *c], capture_output=True, text=True).stdout.strip()


def _machine(cp) -> dict:
    libre, total = cp.cuda.runtime.memGetInfo()
    props = cp.cuda.runtime.getDeviceProperties(0)
    nom = props["name"]
    return {"gpu": nom.decode() if isinstance(nom, bytes) else str(nom),
            "vram_totale_mo": total / 1024 ** 2,
            "vram_libre_avant_mo": libre / 1024 ** 2,
            "cupy": cp.__version__,
            "python": ".".join(str(v) for v in sys.version_info[:3])}


# ------------------------------------------------------------- les gardes

def garde_kernels_intouches() -> dict:
    diff = _git(["diff", "--stat", "HEAD", "--",
                 "src/f1_gpu/substrat_fusionne_3d.py",
                 "src/f1_gpu/substrat_fusionne_3d_param.py",
                 "src/f1_gpu/substrat_jetable_3d.py"])
    if diff:
        raise MesureImpossible(f"substrats 3D modifiés :\n{diff}")
    return {"empreinte_a53": hashlib.sha256(
        fusionne_3d._SOURCE_3D.encode("utf-8")).hexdigest(),
        "empreinte_param_3_2": empreinte_param(N_SCALAIRES, N_STATIQUES),
        "git_diff": "vide"}


def lire_ancres() -> dict:
    if not ARTEFACT_A55.exists():
        raise MesureImpossible(f"artefact absent : {ARTEFACT_A55}")
    a55 = json.loads(ARTEFACT_A55.read_text())
    m = a55["mesures"]["m_c5"]
    if (m["n_scalaires"], m["n_statiques"], m["cote"]) != (
            N_SCALAIRES, N_STATIQUES, 64):
        raise MesureImpossible("M-c5 de §A55 ne porte pas (3, 2) à 64³.")
    return {"temoin_64_ns_par_cellule": m["ms_par_bloc"] * 1e6 / 64 ** 3,
            "non_f_ms": a55["ancres_lues"]["non_f_ms"]}


# ------------------------------------------------------------ les mesures

def _etat(cp, cote: int):
    return cp.asarray(etat_initial_jetable_3d(N_FENETRES, 1, cote, GRAINE,
                                              n_champs=N_CHAMPS))


def mesurer(cp, cote: int) -> dict:
    """Un point : pré-chrono pour dimensionner, puis la série.

    Le bruit du point est l'écart entre les médianes de ses DEUX
    DEMI-SÉRIES. C'est lui qui rend la puissance mesurable — et donc un
    résultat nul prononçable (prereg §4)."""
    q = _etat(cp, cote)
    tampon = cp.array(q, copy=True)

    def frame():
        pas_f_fusionne_3d_param(q, cp, sortie=q, tampon_etage=tampon,
                                n_statiques=N_STATIQUES)

    pre = chronometrer_frames(cp, frame, warmup=5, serie=FRAMES_PRE_CHRONO)
    duree_estimee = pre["stats"]["mediane_ms"]
    serie = max(SERIE_FRAMES, math.ceil(DUREE_CIBLE_MS / duree_estimee))
    warmup = max(WARMUP_FRAMES, serie // 10)

    resultat = chronometrer_frames(cp, frame, warmup=warmup, serie=serie)
    echantillons = resultat["serie_ms"]
    moitie = len(echantillons) // 2
    s1 = stats_ms(echantillons[:moitie])
    s2 = stats_ms(echantillons[moitie:])
    stats = resultat["stats"]
    ecart_demi = abs(s1["mediane_ms"] - s2["mediane_ms"]) / stats["mediane_ms"]

    cellules = N_FENETRES * cote ** 3
    del q, tampon
    cp.get_default_memory_pool().free_all_blocks()
    return {
        "cote": cote, "groupe": "aligne" if cote % 32 == 0 else "chevauchant",
        "cellules_par_fenetre": cote ** 3,
        "fraction_cellules_de_bord": 1.0 - ((cote - 2) / cote) ** 3,
        "duree_estimee_ms": duree_estimee, "serie": serie, "warmup": warmup,
        **stats,
        "mediane_demi_1_ms": s1["mediane_ms"], "mediane_demi_2_ms": s2["mediane_ms"],
        "ecart_demi_series": ecart_demi,
        "serie_stable": ecart_demi <= BANDE_DEMI_SERIES,
        "ratio_mediane_sur_min": stats["mediane_ms"] / stats["min_ms"],
        "ns_par_cellule": stats["mediane_ms"] * 1e6 / cellules,
        "bruit_ns_par_cellule": ecart_demi * stats["mediane_ms"] * 1e6 / cellules,
        "attributs": attributs_param(cp, N_SCALAIRES, N_STATIQUES),
    }


# ------------------------------------------------------------- la lecture

def lire_le_resultat(ancres: dict, mesures: dict) -> dict:
    par_cote = {m["cote"]: m for m in mesures.values()}

    triplets = {}
    for bas, centre, haut in TRIPLETS:
        points = [par_cote[bas], par_cote[centre], par_cote[haut]]
        ecartes = [p["cote"] for p in points if not p["serie_stable"]]
        if ecartes:
            triplets[f"T{centre}"] = {
                "retenu": False, "cotes_ecartes": ecartes,
                "motif": ("I-p2 : demi-séries en désaccord — un triplet "
                          "amputé ne rend pas de Δ")}
            continue
        c_b, c_c, c_h = (p["ns_par_cellule"] for p in points)
        delta = c_c - (c_b + c_h) / 2.0
        # Bruit propagé : chaque point porte le sien, mesuré.
        u_b, u_c, u_h = (p["bruit_ns_par_cellule"] for p in points)
        bruit = math.sqrt(u_c ** 2 + (u_b / 2) ** 2 + (u_h / 2) ** 2)
        triplets[f"T{centre}"] = {
            "retenu": True, "centre_aligne": centre,
            "ns_par_cellule": {str(bas): c_b, str(centre): c_c, str(haut): c_h},
            "delta_ns": delta, "bruit_propage_ns": bruit,
            "delta_relatif": delta / c_c,
            "significatif": abs(delta) > bruit,
            "signe": (0 if abs(delta) <= bruit else (1 if delta > 0 else -1)),
            "effet_detectable_relatif": bruit / c_c,
            "puissance_suffisante": (bruit / c_c) < EFFET_MINIMAL_PERTINENT,
        }

    retenus = [t for t in triplets.values() if t["retenu"]]
    if not retenus:
        return {"branche": "I-p2", "verdict": "INDÉTERMINÉ — aucun triplet "
                "n'a de série stable", "triplets": triplets}

    signes = {t["signe"] for t in retenus if t["significatif"]}
    coherent = len(signes) <= 1
    puissance = all(t["puissance_suffisante"] for t in retenus)
    aucun_significatif = not any(t["significatif"] for t in retenus)

    # I-p1 : reproduction du témoin, RE-MESURÉ.
    c64 = par_cote[64]["ns_par_cellule"]
    ecart_temoin = (abs(c64 - ancres["temoin_64_ns_par_cellule"])
                    / ancres["temoin_64_ns_par_cellule"])
    reproduit = ecart_temoin <= BANDE_REPRODUCTION

    # I-p5 : registres constants.
    registres = {m["cote"]: m["attributs"]["registres_par_thread"]
                 for m in mesures.values()}
    registres_constants = len(set(registres.values())) == 1

    # Q2 — stabilité en taille sur les points RETENUS.
    couts = [m["ns_par_cellule"] for m in mesures.values() if m["serie_stable"]]
    etendue = ((max(couts) - min(couts)) / min(couts)) if couts else float("nan")

    # Branches PRÉ-ÉCRITES. Aucun `else` ne prononce par défaut.
    if not reproduit:
        branche, verdict = "I-p1", "INDÉTERMINÉ — témoin 64³ non reproduit"
    elif not registres_constants:
        branche, verdict = "I-p5", "INDÉTERMINÉ — registres non constants"
    elif len(retenus) < len(TRIPLETS):
        branche, verdict = (
            "I-p2", f"INDÉTERMINÉ — {len(TRIPLETS)-len(retenus)} triplet(s) "
                    "écarté(s) pour série instable")
    elif not coherent:
        branche, verdict = (
            "I-p4", "INDÉTERMINÉ — deux triplets significatifs de signes "
                    "OPPOSÉS : le dispositif ne mesure pas ce qu'il croit")
    elif aucun_significatif and not puissance:
        branche, verdict = (
            "I-p3", "INDÉTERMINÉ PAR MANQUE DE PUISSANCE — le nul n'est pas "
                    "prononçable : le dispositif n'aurait pas vu l'effet qui "
                    "compte")
    elif aucun_significatif:
        branche, verdict = (
            "P-A", "l'alignement de warps NE PÈSE PAS dans ce kernel "
                   "(avec puissance suffisante)")
    elif signes == {-1}:
        branche, verdict = (
            "P-B", "l'alignement PAIE — les candidats se réduisent aux "
                   "multiples de 32")
    else:
        branche, verdict = (
            "P-C", "les côtés ALIGNÉS sont plus chers — contredit le "
                   "mécanisme ; chiffre NON consommé avant nomination")

    return {
        "branche": branche, "verdict": verdict,
        "triplets": triplets,
        "q2_stabilite_taille": {
            "etendue_relative": etendue, "bande": BANDE_STABILITE,
            "branche": ("S-A" if etendue <= BANDE_STABILITE else "S-B"),
            "lecture": ("le coût par cellule est STABLE — l'inversion "
                        "s_max(budget) redevient valide"
                        if etendue <= BANDE_STABILITE else
                        "le coût dépend de la TAILLE — s_max reste un POINT "
                        "FIXE, jamais une constante"),
            "ns_par_cellule": {str(m["cote"]): m["ns_par_cellule"]
                               for m in mesures.values()}},
        "i_p1_temoin": {"mesure_ns": c64,
                        "a55_ns": ancres["temoin_64_ns_par_cellule"],
                        "ecart_relatif": ecart_temoin, "reproduit": reproduit},
        "i_p2_demi_series": {str(m["cote"]): {
            "ecart": m["ecart_demi_series"], "serie": m["serie"],
            "stable": m["serie_stable"]} for m in mesures.values()},
        "i_p3_puissance": {
            "effet_minimal_pertinent": EFFET_MINIMAL_PERTINENT,
            "effet_detectable_par_triplet": {
                nom: t.get("effet_detectable_relatif")
                for nom, t in triplets.items() if t["retenu"]},
            "puissance_suffisante_partout": puissance,
            "note": ("l'effet minimal est DÉRIVÉ : s_max ∝ c^(−1/3), donc un "
                     "surcoût de x % déplace s_max de x/3 % ; passer de 52 à "
                     "48 demande x ≈ 20 %, d'où 10 % comme moitié de marge. "
                     "Sans puissance, « aucun effet » et « aucun pouvoir de "
                     "détection » sont le même nombre.")},
        "i_p4_coherence_de_signe": {"signes_significatifs": sorted(signes),
                                    "coherent": coherent},
        "i_p5_registres": {str(k): v for k, v in registres.items()},
        "diagnostic_mediane_sur_min": {
            str(m["cote"]): m["ratio_mediane_sur_min"] for m in mesures.values()},
        "note_instrument": (
            "La lecture est sur la MÉDIANE. Le rapport médiane/minimum est "
            "reporté en DIAGNOSTIC : §A59 a montré que le minimum donne une "
            "image plus flatteuse, et basculer d'instrument après avoir vu un "
            "résultat est la faute que ce journal surveille."),
    }


# ------------------------------------------------------------------ main

def main() -> int:
    if SORTIE.exists():
        raise MesureImpossible(f"{SORTIE} existe déjà — on n'écrase pas.")
    try:
        import cupy as cp
    except Exception as erreur:                          # pragma: no cover
        raise MesureImpossible(f"cupy indisponible : {erreur}")

    _echo("== gardes ==")
    gardes = garde_kernels_intouches()
    _echo("  substrats 3D intouchés")
    ancres = lire_ancres()
    _echo(f"  témoin §A55 : {ancres['temoin_64_ns_par_cellule']:.4f} ns/cellule "
          "à 64³ — sera RE-MESURÉ")

    _echo(f"== balayage ({N_CHAMPS} champs, série dimensionnée sur "
          f"{DUREE_CIBLE_MS:.0f} ms, plancher {SERIE_FRAMES}) ==")
    mesures = {}
    for cote in COTES:
        m = mesurer(cp, cote)
        mesures[str(cote)] = m
        _echo(f"  s={cote:3d} [{m['groupe']:11s}] série {m['serie']:5d} : "
              f"{m['ns_par_cellule']:.4f} ns/cellule  "
              f"(demi-séries {m['ecart_demi_series']*100:5.2f} %"
              f"{'' if m['serie_stable'] else '  INSTABLE'}, "
              f"méd/min {m['ratio_mediane_sur_min']:.3f})")

    lecture = lire_le_resultat(ancres, mesures)

    artefact = {
        "protocole": {
            "prereg": PREREG, "commit_prereg": COMMIT_PREREG,
            "question": ("Q1 l'alignement de warps coûte-t-il quelque chose à "
                         "TAILLE ÉGALE ? Q2 le coût par cellule est-il stable "
                         "en taille, série correctement dimensionnée ?"),
            "grandeur_de_verdict": "Δ(s) = c(s) − [c(s−2) + c(s+2)] / 2",
            "triplets": [list(t) for t in TRIPLETS],
            "chrono": (f"B6 cuda-Events, warmup ≥ {WARMUP_FRAMES}, série "
                       f"≥ {SERIE_FRAMES} ET ≥ {DUREE_CIBLE_MS:.0f} ms, "
                       "médiane ET p99"),
            "git_head": _git(["rev-parse", "--short", "HEAD"]),
            "portee": ("Ne prononce RIEN sur la cadence, rien sur une "
                       "constante de design (la gravure reste s_max(budget)), "
                       "rien sur le rendu (dû de :4142), rien sur la monnaie "
                       "du slot (porte 3). Sépare le DISCRET du LISSE ; "
                       "n'attribue pas le lisse."),
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
    for nom, t in lecture["triplets"].items():
        if not t["retenu"]:
            _echo(f"  {nom} : ÉCARTÉ — {t['motif']}")
            continue
        _echo(f"  {nom} : Δ = {t['delta_ns']:+.4f} ns "
              f"({t['delta_relatif']*100:+.2f} %) | bruit "
              f"{t['bruit_propage_ns']:.4f} | "
              f"{'SIGNIFICATIF' if t['significatif'] else 'non significatif'}"
              f" | détectable {t['effet_detectable_relatif']*100:.2f} %")
    q2 = lecture["q2_stabilite_taille"]
    _echo(f"  Q2 : étendue {q2['etendue_relative']*100:.1f} % -> {q2['branche']}")
    _echo(f"  branche {lecture['branche']} — {lecture['verdict']}")
    _echo(f"  artefact : {SORTIE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
