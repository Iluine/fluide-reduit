#!/usr/bin/env python
"""LECTEUR — OÙ VIT LE RENDU.

Exécute le protocole de `claude/prereg-ou-vit-le-rendu-2026-08-04.md`
(commit `e2eba15`, écrit AVANT ce fichier).

**CE N'EST PAS UN DRIVER : IL NE MESURE RIEN.** Aucun `cupy`, aucun
chronomètre, aucun kernel. Tout nombre de physique vient d'un artefact
JSON existant ou est EXTRAIT du texte d'une ancre du journal. `§A61` —
« aucune mesure 3D absolue avant re-qualification » — est donc respecté
par construction : ce fichier n'a pas les moyens de mesurer.

Il place le rendu dans le budget de frame et en tire l'arithmétique :
le cap et le côté `s_max` comme FONCTIONS du coût de rendu `R` et du coût
d'interpolation `I`, et l'arbitrage de cadence `Δ` = travail(30 Hz) −
travail(60 Hz).

Il refuse de prononcer si une clé manque, si un fragment d'ancre a
disparu, si un nombre extrait d'un texte ne s'y retrouve pas, ou si une
branche devait reposer sur une distinction que le biais d'instrument de
`§A60` peut effacer.

**SECONDE PASSE (relecture adverse du 04/08).** La première lecture
(artefact `ou-vit-le-rendu-2026-08-04.json`, commit `bbabe02`) laissait
`I-r5` **promise et non mécanisée** : elle ne calculait que les ms/s au
lieu de « calculer des deux façons et reporter l'écart » (prereg §7). Le
paquet entier est désormais produit par UNE fonction appelée deux fois —
budgets gravés (16,7 / 33,333) et 1000/f — et la branche devient
INDÉTERMINÉE si elle diffère entre les deux. S'y ajoutent le volet de
`I-r4` qui manquait (biais de `§A61` sur `R_plancher` lui-même), les
côtés 30 Hz **étiquetés par `I`**, et un témoin de non-régression contre
la première lecture, qui n'est jamais écrasée.

Usage : `.venv/bin/python lire_ou_vit_le_rendu.py`
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess

# --------------------------------------------------------------------------
# CHEMINS
# --------------------------------------------------------------------------
ICI = pathlib.Path(__file__).resolve().parent
JOURNAL = ICI.parent / "pocCascade2phys" / "PREREGISTRATION.md"

ARTEFACT_C = pathlib.Path("claude/lectures/multiplicateur-c-3d-2026-08-04.json")
ARTEFACT_GATHER = pathlib.Path("claude/lectures/tranche-cout-rendu-2026-08-03.json")
ARTEFACT_P2 = pathlib.Path("claude/lectures/p2_chiffrage_rendu.lecture.json")
# PREMIÈRE LECTURE — verdict-grade, JAMAIS écrasée. Elle sert ici de témoin
# de non-régression : les grandeurs sous budget gravé doivent y être retrouvées.
ARTEFACT_PREMIERE = pathlib.Path("claude/lectures/ou-vit-le-rendu-2026-08-04.json")
SORTIE = pathlib.Path("claude/lectures/ou-vit-le-rendu-ir5-2026-08-04.json")

# --------------------------------------------------------------------------
# LES SEULES CONSTANTES ÉCRITES ICI, ET ELLES SONT DÉCLARÉES AU PREREG §8
# --------------------------------------------------------------------------
MS_PAR_SECONDE = 1000.0
IMAGES_PAR_SECONDE = 60.0          # (M-1) le rendu tourne à 60 fps partout
COTE_MESURE = 64                   # la fenêtre de §A55 est un 64³
CELLULES_MESUREES = COTE_MESURE ** 3
N_FENETRES_V4 = 11                 # V4 = 11 emplacements (§A16, cap dur)
BIAIS_A60 = 0.15                   # §A60 : les absolus 3D sont pessimistes de ~15 %
BANDE_ANNULATION = 0.005           # W-R1 : « indépendant de R » à 0,5 % près


class LectureImpossible(RuntimeError):
    """Fail-loud. Jamais un `assert` : il disparaît sous `python -O`."""


def _exiger(condition: bool, message: str) -> None:
    if not condition:
        raise LectureImpossible(message)


def _cle(dico: dict, chemin: str, source: str) -> float | str:
    """Descend un chemin pointé et MEURT si une marche manque."""
    courant = dico
    for marche in chemin.split("."):
        _exiger(isinstance(courant, dict) and marche in courant,
                f"clé absente : « {chemin} » dans {source} (bloqué à « {marche} »)")
        courant = courant[marche]
    _exiger(isinstance(courant, (int, float, str)),
            f"clé « {chemin} » de {source} n'est pas un scalaire : {type(courant)}")
    return courant


def _charger(chemin: pathlib.Path) -> dict:
    _exiger(chemin.exists(), f"artefact absent : {chemin}")
    return json.loads(chemin.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# LECTURE D'ANCRE — §A50 appliqué à un NOMBRE
# --------------------------------------------------------------------------
def _normaliser(texte: str) -> str:
    """Même normalisation que `verifier_ancres.py` : on aplatit les blancs
    ET les marqueurs de citation, sinon un fragment replié sur deux lignes
    d'un bloc `>` est introuvable."""
    sans_citation = re.sub(r"(?m)^[ \t]*>+[ \t]?", "", texte)
    sans_emphase = re.sub(r"[*_`]", "", sans_citation)
    return re.sub(r"\s+", " ", sans_emphase).strip()


def _lire_ancre(lignes: list[str], ligne: int, fragment: str,
                fenetre: int = 3) -> str:
    """Retourne le texte normalisé autour d'une ancre, APRÈS avoir vérifié
    que le fragment s'y trouve. Si le fragment a disparu, c'est un
    changement de fond : on meurt (§A50), on ne renumérote pas."""
    _exiger(1 <= ligne <= len(lignes),
            f"ancre `PREREGISTRATION.md:{ligne}` hors du journal "
            f"({len(lignes)} lignes)")
    bas = max(0, ligne - 1 - fenetre)
    haut = min(len(lignes), ligne + fenetre)
    contexte = _normaliser("\n".join(lignes[bas:haut]))
    attendu = _normaliser(fragment)
    _exiger(attendu in contexte,
            f"fragment DISPARU de `PREREGISTRATION.md:{ligne}` : « {fragment} ». "
            "Changement de fond — jamais une renumérotation silencieuse (§A50).")
    return contexte


def _nombre_de_l_ancre(contexte: str, motif: str, fragment: str) -> float:
    """Extrait un nombre DU TEXTE de l'ancre. Le nombre n'est écrit nulle
    part dans ce fichier : s'il change au journal, la lecture change."""
    trouve = re.search(motif, contexte)
    _exiger(trouve is not None,
            f"nombre introuvable dans l'ancre « {fragment} » (motif {motif!r})")
    return float(trouve.group(1).replace(",", "."))


# --------------------------------------------------------------------------
# LE MODÈLE (§2 et §3 du prereg)
# --------------------------------------------------------------------------
def n_interpolations(f: float) -> float:
    """(P-b) À 30 Hz de physique, 30 des 60 images tombent ENTRE deux pas et
    doivent être interpolées depuis le readout. À 60 Hz, aucune."""
    _exiger(f in (30.0, 60.0), f"cadence non traitée par le prereg : {f}")
    return IMAGES_PAR_SECONDE - f


def travail_par_seconde(R: float, I: float, f: float, budget_frame: float,
                        nonF: float, C: float) -> float:
    """Fenêtres·Hz réellement disponibles par seconde. (M-2) les temps
    s'ADDITIONNENT : majorant du temps ⇒ MINORANT du cap."""
    disponible = (f * budget_frame
                  - IMAGES_PAR_SECONDE * R
                  - n_interpolations(f) * I
                  - f * nonF)
    return disponible / C


def ms_par_fenetre(R: float, I: float, f: float, budget_frame: float,
                   nonF: float) -> float:
    """Millisecondes allouées à UNE des 11 fenêtres de V4, à un pas de
    physique donné."""
    par_pas = (budget_frame
               - (IMAGES_PAR_SECONDE * R + n_interpolations(f) * I) / f
               - nonF)
    return par_pas / N_FENETRES_V4


def cote_max(R: float, I: float, f: float, budget_frame: float,
             nonF: float, C: float) -> float | None:
    """`s_max(budget)` — une INVERSION, jamais une constante de design.
    Le geste inverse est gravé comme faute : `PREREGISTRATION.md:1033`
    « Choisir `r_fovea` pour que le budget passe = fabriquer le verdict »."""
    ms = ms_par_fenetre(R, I, f, budget_frame, nonF)
    if ms <= 0.0:
        return None                    # le budget est épuisé : pas de côté
    return (ms / C * CELLULES_MESUREES) ** (1.0 / 3.0)


def main() -> int:
    lignes = JOURNAL.read_text(encoding="utf-8").splitlines()

    # ---------------------------------------------------------------- ancres
    ancre_porte = _lire_ancre(
        lignes, 4218, "cohabitation GPU dans les ~13 ms restants")
    reserve_porte_ms = _nombre_de_l_ancre(
        ancre_porte, r"~(\d+(?:[.,]\d+)?)\s*ms restants",
        "cohabitation GPU dans les ~13 ms restants")

    ancre_du = _lire_ancre(
        lignes, 4142, "où vit le rendu — à traiter explicitement si ouverte")
    ancre_non_f = _lire_ancre(
        lignes, 9559, "non-F 2D à l'échelle de l'instrument, pas un rendu 3D à 1920")

    # ------------------------------------------------------------- artefacts
    art_c = _charger(ARTEFACT_C)
    art_gather = _charger(ARTEFACT_GATHER)
    art_p2 = _charger(ARTEFACT_P2)

    C = float(_cle(art_c, "lecture.C_vocabulaire_reel_ms", str(ARTEFACT_C)))
    nonF = float(_cle(art_c, "ancres_lues.non_f_ms", str(ARTEFACT_C)))
    budget = {
        60.0: float(_cle(art_c, "ancres_lues.seuil_ms", str(ARTEFACT_C))),
        30.0: float(_cle(art_c, "ancres_lues.budget_30hz_ms", str(ARTEFACT_C))),
    }
    _exiger(C > 0.0 and nonF > 0.0, "C ou non-F non strictement positif")

    gather_plancher = float(_cle(art_gather, "m2b_loi.deduction_1920x1080_ms",
                                 str(ARTEFACT_GATHER)))
    encodage_plancher = float(_cle(art_p2, "cellule_2b.mediane_ms",
                                   str(ARTEFACT_P2)))
    # (I-r1) L'INTERDICTION EST RECOPIÉE PAR LA MACHINE, PAS PAR LA SESSION.
    interdiction_p2 = str(_cle(art_p2, "cellule_2b.lecture.texte",
                               str(ARTEFACT_P2)))
    _exiger("INTERDICTION PRE-ECRITE" in interdiction_p2,
            "l'interdiction pré-écrite de P2 a disparu de son artefact — "
            "la garde héritée n'existe plus, on ne prononce pas")

    R_plancher = gather_plancher + encodage_plancher
    # La réserve gravée vaut ~13 ms par fenêtre de 33,3 ms, où 2 images sont
    # rendues : le coût PAR IMAGE qu'elle suppose est donc la moitié.
    images_par_fenetre_30hz = IMAGES_PAR_SECONDE / 30.0
    R_porte = reserve_porte_ms / images_par_fenetre_30hz

    # ------------------------------------------------------------------------
    # (I-r5) EXÉCUTÉE. Le prereg prescrivait « le lecteur calcule DES DEUX
    # FAÇONS et reporte l'écart » (P:202). La première lecture ne calculait que
    # les ms/s : la clause était PROMISE, pas mécanisée — exactement le défaut
    # que le corpus reproche partout ailleurs. Tout le paquet de grandeurs est
    # donc produit par UNE fonction, appelée deux fois, sur deux jeux de budgets.
    # ------------------------------------------------------------------------
    def evaluer(bud: dict[float, float]) -> dict:
        def delta(R: float, I: float) -> float:
            return (travail_par_seconde(R, I, 30.0, bud[30.0], nonF, C)
                    - travail_par_seconde(R, I, 60.0, bud[60.0], nonF, C))

        # CE QUE L'ANNULATION COÛTE, ET CE QUI LA CASSE. Sans ce contraste,
        # « R s'annule » ne serait qu'une vérification de mon algèbre par mon
        # code. L'annulation vient de (M-1) — 60 images par seconde PARTOUT.
        # Placement alternatif, explicitement HORS porte : le rendu suit la
        # cadence physique (30 fps à 30 Hz, pas d'interpolation). R n'y
        # disparaît plus.
        def delta_rendu_a_la_cadence(R: float) -> float:
            return ((30.0 * bud[30.0] - 30.0 * R - 30.0 * nonF) / C
                    - (60.0 * bud[60.0] - 60.0 * R - 60.0 * nonF) / C)

        d_plancher, d_porte = delta(R_plancher, 0.0), delta(R_porte, 0.0)
        ref = max(abs(d_plancher), abs(d_porte), 1e-12)
        sensi = abs(d_porte - d_plancher) / ref
        a_plancher, a_porte = (delta_rendu_a_la_cadence(R_plancher),
                               delta_rendu_a_la_cadence(R_porte))
        ref_alt = max(abs(a_plancher), abs(a_porte), 1e-12)

        # Point mort : Δ = 0 ⇔ I = nonF − (60·B60 − 30·B30)/30. Le second
        # terme est NUL si et seulement si les budgets valent 1000/f.
        point_mort = nonF - (60.0 * bud[60.0] - 30.0 * bud[30.0]) / 30.0

        cotes_ = {}
        for etiquette, R in (("R_plancher", R_plancher), ("R_porte", R_porte)):
            cotes_[etiquette] = {
                "R_ms": R,
                "s_max_60hz": cote_max(R, 0.0, 60.0, bud[60.0], nonF, C),
                "ms_par_fenetre_60hz": ms_par_fenetre(R, 0.0, 60.0, bud[60.0], nonF),
                "cap_60hz": travail_par_seconde(R, 0.0, 60.0, bud[60.0], nonF, C) / 60.0,
                "cap_30hz_I_nul": travail_par_seconde(R, 0.0, 30.0, bud[30.0], nonF, C) / 30.0,
                # ÉTIQUETTE OBLIGATOIRE : à 30 Hz le côté DÉPEND de I, qui est
                # inconnu et gravé « non gratuit ». Un `s_max` 30 Hz sans son
                # `I` est un chiffre du coin favorable.
                "s_max_30hz_selon_I": {
                    f"I={I:.4f}": cote_max(R, I, 30.0, bud[30.0], nonF, C)
                    for I in (0.0, nonF / 2.0, nonF)
                },
            }
        haut = cotes_["R_plancher"]["s_max_60hz"]
        bas = cotes_["R_porte"]["s_max_60hz"]
        _exiger(haut is not None and bas is not None,
                "l'encadrement du côté est vide : un des deux R épuise le budget")
        return {
            "budgets_ms": dict(bud),
            "ms_par_seconde": {f: f * bud[f] for f in (60.0, 30.0)},
            "delta_au_plancher": d_plancher,
            "delta_a_la_porte": d_porte,
            "sensibilite_relative_a_R": sensi,
            "R_s_annule": sensi < BANDE_ANNULATION,
            "contraste_rendu_a_la_cadence": {
                "delta_au_plancher": a_plancher, "delta_a_la_porte": a_porte,
                "sensibilite_relative_a_R": abs(a_porte - a_plancher) / ref_alt,
            },
            "point_mort_I_ms": point_mort,
            "delta_selon_I": {f"I={I:.4f}": delta(R_plancher, I)
                              for I in (0.0, nonF / 2.0, nonF, 2.0 * nonF)},
            "cotes": cotes_,
            "s_haut": haut, "s_bas": bas,
            "etendue_cote": (haut - bas) / haut,
        }

    budget_mille = {60.0: MS_PAR_SECONDE / 60.0, 30.0: MS_PAR_SECONDE / 30.0}
    lectures = {"budget_grave": evaluer(budget), "budget_1000_sur_f": evaluer(budget_mille)}

    def _ecart(chemin) -> float:
        a, b = chemin(lectures["budget_grave"]), chemin(lectures["budget_1000_sur_f"])
        return abs(a - b) / max(abs(a), abs(b), 1e-12)

    ecarts_i_r5 = {
        "point_mort_I": _ecart(lambda d: d["point_mort_I_ms"]),
        "delta_a_I_nul": _ecart(lambda d: d["delta_au_plancher"]),
        "s_max_60hz_plancher": _ecart(lambda d: d["s_haut"]),
        "cap_60hz_plancher": _ecart(lambda d: d["cotes"]["R_plancher"]["cap_60hz"]),
        "etendue_cote": _ecart(lambda d: d["etendue_cote"]),
    }
    # LA GARDE DE I-r5 : un verdict qui différerait entre les deux façons de
    # compter serait un verdict porté par un arrondi ⇒ INDÉTERMINÉ.
    branche_stable = (lectures["budget_grave"]["R_s_annule"]
                      == lectures["budget_1000_sur_f"]["R_s_annule"])

    grave = lectures["budget_grave"]
    delta_au_plancher = grave["delta_au_plancher"]
    delta_a_la_porte = grave["delta_a_la_porte"]
    sensibilite_a_R = grave["sensibilite_relative_a_R"]
    R_s_annule = grave["R_s_annule"]
    alt_plancher = grave["contraste_rendu_a_la_cadence"]["delta_au_plancher"]
    alt_porte = grave["contraste_rendu_a_la_cadence"]["delta_a_la_porte"]
    sensibilite_alt = grave["contraste_rendu_a_la_cadence"]["sensibilite_relative_a_R"]
    point_mort_I = grave["point_mort_I_ms"]
    arbitrage_par_I = grave["delta_selon_I"]
    cotes = grave["cotes"]
    s_haut, s_bas = grave["s_haut"], grave["s_bas"]
    etendue_cote = grave["etendue_cote"]
    par_seconde_grave = grave["ms_par_seconde"]
    ecart_arrondi = abs(par_seconde_grave[60.0] - par_seconde_grave[30.0])
    ecart_arrondi_rel = ecart_arrondi / MS_PAR_SECONDE

    # ------- I-r4, VOLET MANQUANT : le biais de §A61 sur R_plancher LUI-MÊME.
    # La première lecture perturbait C et non-F, jamais R — or R_plancher est
    # mesuré sur des séries d'environ 0,25 s, très en deçà du plus court T_conv
    # observé (2,5 s à 64³). Il peut vivre ENTIER dans le transitoire.
    sensibilite_s_max_a_R = {}
    for nom, facteur in (("R_plancher+15%", 1.0 + BIAIS_A60),
                         ("R_plancher-15%", 1.0 - BIAIS_A60)):
        perturbe = cote_max(R_plancher * facteur, 0.0, 60.0, budget[60.0], nonF, C)
        _exiger(perturbe is not None, f"{nom} épuise le budget")
        sensibilite_s_max_a_R[nom] = {
            "s_max_60hz": perturbe,
            "ecart_relatif": abs(perturbe - s_haut) / s_haut,
        }

    # ------- (I-r4) l'étendue survit-elle au biais d'instrument de §A60 ?
    etendues_perturbees = {}
    for nom, (fC, fN) in {
        "C+15%": (1.0 + BIAIS_A60, 1.0), "C-15%": (1.0 - BIAIS_A60, 1.0),
        "nonF+15%": (1.0, 1.0 + BIAIS_A60), "nonF-15%": (1.0, 1.0 - BIAIS_A60),
    }.items():
        haut = cote_max(R_plancher, 0.0, 60.0, budget[60.0], nonF * fN, C * fC)
        bas = cote_max(R_porte, 0.0, 60.0, budget[60.0], nonF * fN, C * fC)
        etendues_perturbees[nom] = (
            None if (haut is None or bas is None or haut <= 0.0)
            else (haut - bas) / haut)
    valides = [v for v in etendues_perturbees.values() if v is not None]
    _exiger(len(valides) == len(etendues_perturbees),
            "une perturbation de ±15 % épuise le budget : l'encadrement du "
            "côté n'est pas lisible sous le biais de §A60 ⇒ INDÉTERMINÉ")
    etendue_min = min(valides)
    cote_distinguable = etendue_min > BIAIS_A60

    # ----------------------------------------------------------- LA BRANCHE
    decidables = all(v == v and abs(v) != float("inf")          # NaN / inf
                     for v in (sensibilite_a_R, etendue_min, etendue_cote))
    if not decidables:
        branche, texte = "INDÉTERMINÉ", (
            "une grandeur de décision n'est pas finie : sortie sûre, jamais "
            "un verdict par défaut.")
    elif not branche_stable:
        branche, texte = "INDÉTERMINÉ", (
            "I-r5 : la branche diffère selon que l'on compte avec les budgets "
            "gravés (16,7 / 33,333) ou avec 1000/f. Un verdict porté par un "
            "arrondi n'est pas un verdict.")
    elif not R_s_annule:
        branche, texte = "W-R2", (
            "Δ dépend de R : le rendu entre dans l'arbitrage de cadence. "
            "Le dû de :4142 est BLOQUANT pour la cadence, et le chiffrage de "
            "P2 (composition + optique R2+) passe devant.")
    elif not cote_distinguable:
        branche, texte = "W-R3", (
            "Δ est indépendant de R, mais l'encadrement [R_plancher, R_porte] "
            "déplace s_max de moins que le biais d'instrument de §A60 : la "
            "question est sous le bruit. Aucune lecture de côté n'est "
            "prononcée — INDÉTERMINÉ, et le dire est le résultat.")
    elif R_s_annule and cote_distinguable:
        branche, texte = "W-R1", (
            "Δ est indépendant de R : où vit le rendu ne décide PAS la "
            "cadence, il décide le CÔTÉ. Le dû de :4142 est levé quant à la "
            "cadence et reporté intact sur le côté de fenêtre. Ce que la "
            "cadence attend n'est plus le rendu mais le NON-F 3D, l'un des "
            "deux multiplicateurs dus de §A53.")
    else:
        branche, texte = "INDÉTERMINÉ", (
            "combinaison non prévue par le prereg : sortie sûre, jamais un "
            "verdict par défaut.")

    # ------- TÉMOIN DE NON-RÉGRESSION contre la première lecture (bbabe02).
    # Ce qui change ici doit être ce que I-r5 AJOUTE, jamais ce qu'elle déplace.
    premiere = _charger(ARTEFACT_PREMIERE)
    non_regression = {}
    for nom, avant, apres in (
        ("delta_au_plancher",
         float(_cle(premiere, "annulation_de_R.delta_fenetres_hz_au_plancher",
                    str(ARTEFACT_PREMIERE))), delta_au_plancher),
        ("s_max_60hz_plancher",
         float(_cle(premiere, "cote.encadrement.R_plancher.s_max_60hz",
                    str(ARTEFACT_PREMIERE))), s_haut),
        ("etendue_cote",
         float(_cle(premiere, "cote.etendue_relative",
                    str(ARTEFACT_PREMIERE))), etendue_cote),
        ("point_mort_I",
         float(_cle(premiere, "arbitrage_de_cadence.point_mort_I_ms",
                    str(ARTEFACT_PREMIERE))), point_mort_I),
    ):
        ecart = abs(avant - apres) / max(abs(avant), 1e-12)
        _exiger(ecart < 1e-9,
                f"RÉGRESSION sur « {nom} » : {avant} → {apres} (écart {ecart:.3e}). "
                "La clause I-r5 devait AJOUTER une lecture, pas en déplacer une.")
        non_regression[nom] = {"premiere_lecture": avant, "ici": apres,
                               "ecart_relatif": ecart}

    # ---------------------------------------------------------------- sortie
    tete = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    artefact = {
        "protocole": "claude/prereg-ou-vit-le-rendu-2026-08-04.md (commit e2eba15)",
        "lecteur": {"fichier": __file__.split("/")[-1], "tete_git": tete},
        "nature": ("LECTURE, PAS MESURE — aucun GPU, aucun cupy, aucun "
                   "chronomètre. §A61 respecté par construction."),
        "ancres_lues": {
            "PREREGISTRATION.md:4142": ancre_du[:240],
            "PREREGISTRATION.md:4218": ancre_porte[:240],
            "PREREGISTRATION.md:9559": ancre_non_f[:240],
            "reserve_porte_ms_EXTRAITE_DU_TEXTE": reserve_porte_ms,
        },
        "entrees": {
            "C_fenetre_64_ms": {"valeur": C, "source": str(ARTEFACT_C)},
            "non_f_ms": {
                "valeur": nonF, "source": str(ARTEFACT_C),
                "PROVENANCE": ("non-F 2D à l'échelle de l'instrument, PAS un "
                               "non-F 3D à 1920 — PREREGISTRATION.md:9559"),
            },
            "budget_frame_ms": budget,
            "R_plancher_ms": {
                "valeur": R_plancher,
                "gather_plancher_ms": gather_plancher,
                "encodage_fusionne_ms": encodage_plancher,
                "sources": [str(ARTEFACT_GATHER), str(ARTEFACT_P2)],
            },
            "R_porte_ms": {
                "valeur": R_porte,
                "reserve_par_fenetre_33ms": reserve_porte_ms,
                "images_par_fenetre_30hz": images_par_fenetre_30hz,
            },
        },
        "interdiction_heritee_de_P2": interdiction_p2,
        "temoin_de_non_regression": {
            "contre": str(ARTEFACT_PREMIERE),
            "grandeurs": non_regression,
        },
        "modele": {
            "M-1": "60 images par seconde dans TOUS les placements",
            "M-2": ("les temps s'ADDITIONNENT — majorant du temps, donc "
                    "MINORANT du cap : tout cap lu ici est PESSIMISTE"),
            "M-3": "un seul GPU ⇒ le placement « hors budget » (P-d) n'existe pas",
            "n_interpolations": {"60 Hz": n_interpolations(60.0),
                                 "30 Hz": n_interpolations(30.0)},
        },
        "annulation_de_R": {
            "delta_fenetres_hz_au_plancher": delta_au_plancher,
            "delta_fenetres_hz_a_la_porte": delta_a_la_porte,
            "sensibilite_relative_a_R": sensibilite_a_R,
            "bande": BANDE_ANNULATION,
            "R_s_annule": R_s_annule,
            "d_ou_vient_l_annulation": {
                "cause": ("(M-1) 60 images par seconde dans les DEUX "
                          "placements ⇒ le terme 60·R est identique des deux "
                          "côtés et disparaît de la différence. L'annulation "
                          "est une conséquence du modèle, et le modèle est le "
                          "texte de la porte (:4218)."),
                "contraste_placement_rendu_a_la_cadence": {
                    "note": ("HORS porte : le rendu suit la cadence physique "
                             "(30 fps à 30 Hz, aucune interpolation). R n'y "
                             "disparaît plus — c'est ce que l'annulation coûte."),
                    "delta_au_plancher": alt_plancher,
                    "delta_a_la_porte": alt_porte,
                    "sensibilite_relative_a_R": sensibilite_alt,
                },
            },
        },
        "arbitrage_de_cadence": {
            "note": ("I-r2 : Δ n'est JAMAIS rendu comme un scalaire unique. "
                     "I (coût d'une interpolation de readout) est INCONNU et "
                     "explicitement non gratuit (:4219)."),
            "delta_selon_I": arbitrage_par_I,
            "point_mort_I_ms": point_mort_I,
            "formule": "Δ = 30·(nonF − I)/C, au terme d'arrondi près",
        },
        "cote": {
            "encadrement": cotes,
            "etendue_relative": etendue_cote,
            "etendues_sous_biais_A60": etendues_perturbees,
            "etendue_minimale": etendue_min,
            "biais_A60": BIAIS_A60,
            "distinguable": cote_distinguable,
        },
        "I_r5_les_deux_facons": {
            "note": ("CLAUSE EXÉCUTÉE. La première lecture (artefact "
                     "ou-vit-le-rendu-2026-08-04.json, lecteur bbabe02) ne "
                     "calculait que les ms/s : I-r5 y était PROMISE, pas "
                     "mécanisée. Tout le paquet est ici produit deux fois."),
            "ms_par_seconde_grave": par_seconde_grave,
            "ecart_ms": ecart_arrondi,
            "ecart_relatif": ecart_arrondi_rel,
            "ecarts_entre_les_deux_facons": ecarts_i_r5,
            "branche_stable_entre_les_deux": branche_stable,
            "ce_que_l_ecart_touche": (
                "Les grandeurs PAR FRAME (s_max, cap, étendue) sont "
                "insensibles : l'arrondi se simplifie. Le POINT MORT et Δ, "
                "eux, vivent par seconde et portent l'écart en entier — d'où "
                "un point mort à 1,768 sous les budgets gravés et exactement "
                "non-F sous 1000/f."),
            "lectures": lectures,
        },
        "sensibilite_de_s_max_au_biais_sur_R": {
            "note": ("VOLET MANQUANT DE I-r4 : la première lecture perturbait "
                     "C et non-F, jamais R_plancher, qui est mesuré sur des "
                     "séries d'environ 0,25 s — très en deçà du plus court "
                     "T_conv de §A61 (2,5 s à 64³). Il peut vivre ENTIER dans "
                     "le transitoire."),
            "s_max_60hz_de_reference": s_haut,
            "perturbations": sensibilite_s_max_a_R,
        },
        "branche": branche,
        "texte": texte,
        "portee": ("PLACE le rendu dans le budget — ne le CHIFFRE pas. La "
                   "dette P2 (composition + optique R2+) reste ouverte. Ni la "
                   "cadence, ni le côté, ni le placement ne sont tranchés."),
    }
    SORTIE.write_text(json.dumps(artefact, ensure_ascii=False, indent=1),
                      encoding="utf-8")
    print(f"branche = {branche}")
    print(f"  Δ(R_plancher) = {delta_au_plancher:.4f} fenêtres·Hz")
    print(f"  Δ(R_porte)    = {delta_a_la_porte:.4f} fenêtres·Hz")
    print(f"  sensibilité à R = {sensibilite_a_R:.3e} (bande {BANDE_ANNULATION})")
    mille = lectures["budget_1000_sur_f"]
    print(f"  point mort I = {point_mort_I:.4f} ms (budgets gravés) "
          f"| {mille['point_mort_I_ms']:.4f} ms (1000/f) — non-F = {nonF:.4f}")
    print(f"  I-r5 écarts entre les deux façons : "
          + ", ".join(f"{k} {v:.2%}" for k, v in ecarts_i_r5.items()))
    print(f"  s_max 60 Hz : {s_haut:.2f} (R_plancher) → {s_bas:.2f} (R_porte)")
    print("  s_max 30 Hz selon I (R_plancher) : "
          + ", ".join(f"{k}→{v:.2f}" for k, v in
                      cotes["R_plancher"]["s_max_30hz_selon_I"].items()))
    print(f"  étendue = {etendue_cote:.1%}, minimale sous ±15 % = {etendue_min:.1%}")
    print("  s_max vs biais ±15 % sur R_plancher : "
          + ", ".join(f"{k} {v['ecart_relatif']:.2%}"
                      for k, v in sensibilite_s_max_a_R.items()))
    print(f"→ {SORTIE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
