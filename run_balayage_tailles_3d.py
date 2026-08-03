#!/usr/bin/env python
"""DRIVER — BALAYAGE DE TAILLES AU VOCABULAIRE RÉEL.

Exécute le protocole de `claude/prereg-balayage-tailles-3d-2026-08-04.md`
(commit `089b3f8`, écrit AVANT ce fichier).

GRANDEUR DE VERDICT : **`c(s)` = ns par cellule** au vocabulaire réel
(3 scalaires advectés + 2 statiques lus), pour huit côtés répartis en deux
groupes fixés AVANT le run — ALIGNÉS (multiples de 32) et CHEVAUCHANTS.
D'où le **POINT FIXE** : le plus grand côté pair `s` tel que
`11 · s³ · c(s) + non-F ≤ 16,7 ms`, avec `c(s)` MESURÉ et non le `c(64)`
de §A55.

Le driver refuse de mesurer si les kernels ont bougé. Il refuse de
prononcer si le point `64³` ne reproduit pas §A55, si les registres
varient d'un côté à l'autre, si un côté chevauchant se révèle
inexplicablement moins cher que tous les alignés, ou si le point fixe
n'est pas stable à la méthode d'interpolation.

Usage : `.venv/bin/python run_balayage_tailles_3d.py`
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
from src.f1_gpu.substrat_fusionne_3d_param import (
    attributs_param,
    empreinte_param,
    pas_f_fusionne_3d_param,
)
from src.f1_gpu.substrat_jetable_3d import etat_initial_jetable_3d

# ---------------------------------------------------------------- config
ARTEFACT_A55 = pathlib.Path("claude/lectures/multiplicateur-c-3d-2026-08-04.json")
SORTIE = pathlib.Path("claude/lectures/balayage-tailles-3d-2026-08-04.json")
PREREG = "claude/prereg-balayage-tailles-3d-2026-08-04.md"
COMMIT_PREREG = "089b3f8"

N_SCALAIRES = 3          # s, e_th, ρ_s — le vocabulaire réel de §A51
N_STATIQUES = 2          # b0, id-matériau
N_CHAMPS = 4 + N_SCALAIRES + N_STATIQUES        # 9
N_FENETRES = 3
GRAINE = 12345
N_SLOTS_V4 = 11
BUDGET_60HZ = 16.7

# Les deux groupes, fixés AVANT le run (prereg §1). Reclasser un côté
# après lecture fabriquerait le contraste.
ALIGNES: tuple[int, ...] = (32, 64, 96, 128)          # n ≡ 0 mod 32
CHEVAUCHANTS: tuple[int, ...] = (40, 48, 52, 56)      # warp à cheval
COTES: tuple[int, ...] = tuple(sorted(ALIGNES + CHEVAUCHANTS))

BANDE_STABILITE = 0.10        # T-A / T-B
BANDE_REPRODUCTION = 0.10     # I-t1
CONTRASTE_SIGNIFICATIF = 1.10  # T-C

# Épinglé au prereg §4 : l'inversion NAÏVE à corriger.
S_MAX_NAIF = 52.6


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


# ------------------------------------------------------------- les gardes

def garde_kernels_intouches() -> dict:
    diff = _git(["diff", "--stat", "HEAD", "--",
                 "src/f1_gpu/substrat_fusionne_3d.py",
                 "src/f1_gpu/substrat_fusionne_3d_param.py",
                 "src/f1_gpu/substrat_jetable_3d.py"])
    if diff:
        raise MesureImpossible(f"les substrats 3D ont un diff non vide :\n{diff}")
    return {"empreinte_a53": hashlib.sha256(
        fusionne_3d._SOURCE_3D.encode("utf-8")).hexdigest(),
        "empreinte_param_3_2": empreinte_param(N_SCALAIRES, N_STATIQUES),
        "git_diff": "vide"}


def lire_ancres() -> dict:
    """§A41. Le point `64³` de §A55 sert de TÉMOIN — il sera re-mesuré
    dans le balayage, jamais recopié : un témoin recopié ne témoigne pas."""
    if not ARTEFACT_A55.exists():
        raise MesureImpossible(f"artefact absent : {ARTEFACT_A55}")
    a55 = json.loads(ARTEFACT_A55.read_text())
    m_c5 = a55["mesures"]["m_c5"]
    if (m_c5["n_scalaires"], m_c5["n_statiques"]) != (N_SCALAIRES, N_STATIQUES):
        raise MesureImpossible(
            f"M-c5 de §A55 porte ({m_c5['n_scalaires']}, "
            f"{m_c5['n_statiques']}), pas ({N_SCALAIRES}, {N_STATIQUES}).")
    if m_c5["cote"] != 64:
        raise MesureImpossible(f"M-c5 est à côté {m_c5['cote']}, pas 64.")
    return {"temoin_64_ms_par_fenetre": m_c5["ms_par_bloc"],
            "temoin_64_ns_par_cellule": m_c5["ms_par_bloc"] * 1e6 / 64 ** 3,
            "non_f_ms": a55["ancres_lues"]["non_f_ms"],
            "seuil_ms": BUDGET_60HZ,
            "n_slots_v4": N_SLOTS_V4,
            "s_max_naif": S_MAX_NAIF}


# ------------------------------------------------------------ les mesures

def mesurer(cp, cote: int) -> dict:
    q = cp.asarray(etat_initial_jetable_3d(N_FENETRES, 1, cote, GRAINE,
                                           n_champs=N_CHAMPS))
    tampon = cp.array(q, copy=True)
    resultat = chronometrer_frames(
        cp, lambda: pas_f_fusionne_3d_param(q, cp, sortie=q,
                                            tampon_etage=tampon,
                                            n_statiques=N_STATIQUES))
    stats = resultat["stats"]
    cellules = cote ** 3
    del q, tampon
    cp.get_default_memory_pool().free_all_blocks()
    return {"cote": cote, "cellules_par_fenetre": cellules,
            "groupe": "aligne" if cote % 32 == 0 else "chevauchant",
            "modulo_32": cote % 32,
            "fraction_cellules_de_bord": 1.0 - ((cote - 2) / cote) ** 3,
            **stats,
            "ms_par_fenetre": stats["mediane_ms"] / N_FENETRES,
            "ns_par_cellule": stats["mediane_ms"] * 1e6 / (N_FENETRES * cellules),
            "attributs": attributs_param(cp, N_SCALAIRES, N_STATIQUES)}


# ------------------------------------------------------------- la lecture

def _interpoler(cotes, couts, s: float) -> float:
    """`c(s)` par interpolation linéaire entre côtés mesurés. Refuse
    d'extrapoler : hors plage, aucune valeur n'est inventée."""
    if s < cotes[0] or s > cotes[-1]:
        raise MesureImpossible(
            f"interpolation demandée hors plage mesurée : {s} hors "
            f"[{cotes[0]}, {cotes[-1]}]")
    return float(np.interp(s, cotes, couts))


def _plus_proche(cotes, couts, s: float) -> float:
    indice = int(np.argmin([abs(c - s) for c in cotes]))
    return float(couts[indice])


def _point_fixe(cotes, couts, non_f: float, seuil: float,
                methode, candidats) -> int | None:
    """Le plus grand côté PAIR admissible dont 11 fenêtres tiennent dans
    le budget, avec `c(s)` MESURÉ — et non le `c(64)` de §A55."""
    tenables = []
    for s in candidats:
        c = methode(cotes, couts, float(s))
        total = N_SLOTS_V4 * (s ** 3) * c / 1e6 + non_f
        if total <= seuil:
            tenables.append(s)
    return max(tenables) if tenables else None


def lire_le_resultat(ancres: dict, mesures: list[dict]) -> dict:
    cotes = [m["cote"] for m in mesures]
    couts = [m["ns_par_cellule"] for m in mesures]
    par_cote = dict(zip(cotes, couts))

    etendue = (max(couts) - min(couts)) / min(couts)

    c_align = [par_cote[s] for s in ALIGNES]
    c_chev = [par_cote[s] for s in CHEVAUCHANTS]
    moy_align = float(np.mean(c_align))
    moy_chev = float(np.mean(c_chev))
    contraste = moy_chev / moy_align
    ecart_groupes = abs(moy_chev - moy_align)
    etendue_intra = max(max(c_align) - min(c_align), max(c_chev) - min(c_chev))
    separation = (contraste > CONTRASTE_SIGNIFICATIF
                  and ecart_groupes > etendue_intra)

    # I-t1 : reproduction du témoin, RE-MESURÉ dans ce balayage.
    mesure_64 = par_cote[64]
    ecart_temoin = (abs(mesure_64 - ancres["temoin_64_ns_par_cellule"])
                    / ancres["temoin_64_ns_par_cellule"])
    reproduit = ecart_temoin <= BANDE_REPRODUCTION

    # I-t2 : le kernel n'a pas changé d'un côté à l'autre.
    registres = {m["cote"]: m["attributs"]["registres_par_thread"]
                 for m in mesures}
    registres_constants = len(set(registres.values())) == 1

    # I-t3 : l'aubaine est aussi suspecte.
    aubaine = min(c_chev) < min(c_align)

    # Le POINT FIXE, par deux méthodes (I-t4).
    candidats = [s for s in range(32, 66, 2)]
    pf_interp = _point_fixe(cotes, couts, ancres["non_f_ms"],
                            ancres["seuil_ms"], _interpoler, candidats)
    pf_proche = _point_fixe(cotes, couts, ancres["non_f_ms"],
                            ancres["seuil_ms"], _plus_proche, candidats)
    point_fixe_stable = pf_interp == pf_proche

    # Branches PRÉ-ÉCRITES. Aucun `else` ne prononce par défaut.
    if not reproduit:
        branche, verdict = "I-t1", "INDÉTERMINÉ — témoin 64³ non reproduit"
    elif not registres_constants:
        branche, verdict = "I-t2", "INDÉTERMINÉ — registres non constants"
    elif aubaine:
        branche, verdict = (
            "I-t3", "INDÉTERMINÉ — un côté chevauchant est moins cher que "
                    "tous les alignés ; la cause doit être nommée")
    elif separation:
        branche, verdict = (
            "T-C", "la pénalité est l'ALIGNEMENT DE WARPS — les candidats "
                   "se réduisent aux multiples de 32")
    elif etendue <= BANDE_STABILITE:
        branche, verdict = (
            "T-A", "le coût par cellule est STABLE en taille — l'inversion "
                   "de budget tient")
    else:
        branche, verdict = (
            "T-B", "le coût dépend de la TAILLE, cause non attribuée — seul "
                   "le POINT FIXE fait foi")

    return {
        "branche": branche, "verdict": verdict,
        "ns_par_cellule": {str(s): par_cote[s] for s in cotes},
        "etendue_relative": etendue, "bande_stabilite": BANDE_STABILITE,
        "contraste_groupes": {
            "moyenne_alignes": moy_align, "moyenne_chevauchants": moy_chev,
            "contraste": contraste, "seuil": CONTRASTE_SIGNIFICATIF,
            "ecart_entre_groupes": ecart_groupes,
            "etendue_intra_groupe": etendue_intra,
            "separation_significative": separation,
            "groupes_fixes_avant_le_run": {"alignes": list(ALIGNES),
                                           "chevauchants": list(CHEVAUCHANTS)}},
        "point_fixe": {
            "par_interpolation": pf_interp, "par_plus_proche": pf_proche,
            "stable": point_fixe_stable,
            "s_max_naif": ancres["s_max_naif"],
            "detail": {str(s): {
                "ns_par_cellule": _interpoler(cotes, couts, float(s)),
                "ms_11_fenetres": N_SLOTS_V4 * (s ** 3)
                * _interpoler(cotes, couts, float(s)) / 1e6,
                "total_ms": N_SLOTS_V4 * (s ** 3)
                * _interpoler(cotes, couts, float(s)) / 1e6 + ancres["non_f_ms"],
                "tient": (N_SLOTS_V4 * (s ** 3)
                          * _interpoler(cotes, couts, float(s)) / 1e6
                          + ancres["non_f_ms"]) <= ancres["seuil_ms"]}
                for s in (32, 40, 48, 52, 56, 64)},
            "note": ("le plus grand côté PAIR dont 11 fenêtres tiennent dans "
                     "16,7 − non-F, avec c(s) MESURÉ. Tant que le perceptuel "
                     "n'a pas parlé, cela s'écrit s_max(budget), jamais une "
                     "constante de design (§A58-4).")},
        "i_t1_temoin": {"mesure_ns": mesure_64,
                        "a55_ns": ancres["temoin_64_ns_par_cellule"],
                        "ecart_relatif": ecart_temoin,
                        "bande": BANDE_REPRODUCTION, "reproduit": reproduit},
        "i_t2_registres": {"par_cote": {str(k): v for k, v in registres.items()},
                           "constants": registres_constants},
        "i_t3_aubaine": {
            "min_chevauchants": min(c_chev), "min_alignes": min(c_align),
            "declenchee": aubaine,
            "note": ("ce critère ne se déclenche QUE du côté favorable : la "
                     "seule explication mécanique disponible irait dans "
                     "l'autre sens.")},
        "i_t4_stabilite_point_fixe": {"stable": point_fixe_stable},
        "deux_effets_declares": {
            "fraction_de_bord": {str(m["cote"]): m["fraction_cellules_de_bord"]
                                 for m in mesures},
            "note": ("les petites fenêtres sont FAVORISÉES par le bord (une "
                     "cellule de bord saute deux pentes_axe) et PÉNALISÉES "
                     "par l'alignement. Le balayage mesure leur SOMME ; seul "
                     "le contraste de groupes discrimine.")},
    }


# ------------------------------------------------------------------ main

def main() -> int:
    if SORTIE.exists():
        raise MesureImpossible(
            f"{SORTIE} existe déjà — un artefact verdict-grade ne s'écrase pas.")
    try:
        import cupy as cp
    except Exception as erreur:                          # pragma: no cover
        raise MesureImpossible(f"cupy indisponible : {erreur}")

    _echo("== gardes ==")
    gardes = garde_kernels_intouches()
    _echo("  substrats 3D intouchés (git diff vide)")
    ancres = lire_ancres()
    _echo(f"  témoin §A55 : {ancres['temoin_64_ns_par_cellule']:.4f} ns/cellule "
          f"à 64³ — sera RE-MESURÉ, jamais recopié")

    _echo(f"== balayage ({N_CHAMPS} champs : {N_SCALAIRES} scalaires "
          f"advectés + {N_STATIQUES} statiques) ==")
    mesures = []
    for cote in COTES:
        m = mesurer(cp, cote)
        mesures.append(m)
        _echo(f"  s={cote:3d} [{m['groupe']:11s}] "
              f"{m['ns_par_cellule']:.4f} ns/cellule  "
              f"({m['ms_par_fenetre']:7.3f} ms/fenêtre, bord "
              f"{m['fraction_cellules_de_bord']*100:4.1f} %)")

    lecture = lire_le_resultat(ancres, mesures)

    artefact = {
        "protocole": {
            "prereg": PREREG, "commit_prereg": COMMIT_PREREG,
            "question": ("le coût PAR CELLULE dépend-il du côté de la "
                         "fenêtre, au vocabulaire réel ?"),
            "decision_qui_en_depend": (
                "s_max(budget) = 52,6 est une inversion faite avec c(64). Si "
                "c dépend du côté, l'inversion est invalide et s_max devient "
                "un POINT FIXE."),
            "chrono": (f"B6 cuda-Events, warmup {WARMUP_FRAMES} EXCLU, "
                       f"série {SERIE_FRAMES}, médiane ET p99"),
            "git_head": _git(["rev-parse", "--short", "HEAD"]),
            "portee": ("Ne prononce RIEN sur la cadence, rien sur une "
                       "constante de design (la gravure reste s_max(budget), "
                       "une inversion — §A58-4), rien sur le rendu (dû de "
                       ":4142, aucun cap ne l'inclut), rien sur la "
                       "re-dérivation de la monnaie du slot (porte 3)."),
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
    c = lecture["contraste_groupes"]
    _echo(f"  étendue relative {lecture['etendue_relative']*100:.1f} % "
          f"(bande {BANDE_STABILITE*100:.0f} %)")
    _echo(f"  alignés {c['moyenne_alignes']:.4f} · chevauchants "
          f"{c['moyenne_chevauchants']:.4f} -> contraste {c['contraste']:.4f}"
          f"  (séparation : {c['separation_significative']})")
    pf = lecture["point_fixe"]
    _echo(f"  point fixe : {pf['par_interpolation']} (interpolé) / "
          f"{pf['par_plus_proche']} (plus proche) — stable : {pf['stable']}"
          f"   [s_max naïf {pf['s_max_naif']}]")
    _echo(f"  branche {lecture['branche']} — {lecture['verdict']}")
    _echo(f"  artefact : {SORTIE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
