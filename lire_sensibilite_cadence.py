#!/usr/bin/env python
"""SENSIBILITÉ DE LA CADENCE — lecture, PAS mesure.

Protocole : `claude/analyse-sensibilite-cadence-2026-08-23.md`, commité SEUL
avant ce fichier (§A52). Question unique du §1, critère `T-1`/`T-2` du §4,
branches (a)/(b) du §5 — tous gravés avant que cette ligne existe.

**Aucun GPU, aucun `cupy`, aucun chronomètre.** `§A61` est respecté par
CONSTRUCTION : ce fichier n'a pas les moyens de mesurer. Tout nombre entre par
un artefact existant ; rien n'est recopié depuis un texte de journal.

Le modèle est celui de `lire_ou_vit_le_rendu.py` (§2/§3 du prereg
`claude/prereg-ou-vit-le-rendu-2026-08-04.md`), et il est VÉRIFIÉ contre
l'artefact `I-r5` avant tout balayage : si les quatre grandeurs du témoin de
non-régression ne se reproduisent pas, ce fichier MEURT.

Usage :
    .venv/bin/python lire_sensibilite_cadence.py
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

RACINE = pathlib.Path(__file__).resolve().parent
ART_RENDU = RACINE / "claude/lectures/ou-vit-le-rendu-ir5-2026-08-04.json"
ART_QUALIF = RACINE / "claude/lectures/requalification-instrument-3d-v2-2026-08-22.json"
SORTIE = RACINE / "claude/lectures/sensibilite-cadence-2026-08-23.json"

# --- constantes du MODÈLE, identiques à `lire_ou_vit_le_rendu.py` -----------
IMAGES_PAR_SECONDE = 60.0          # (M-1) le rendu tourne à 60 fps partout
COTE_MESURE = 64                   # la fenêtre de §A55 est un 64³
CELLULES_MESUREES = float(COTE_MESURE ** 3)
N_FENETRES_V4 = 11                 # V4 = 11 emplacements (§A16, cap dur)

# --- le domaine du §3, fixé AVANT calcul ------------------------------------
BIAIS_A60 = 0.15                   # §A60, ancre :9751
# §A63, borne EMPRUNTÉE — σ_r n'est PAS gravée. Le §3 écrivait « ±2,3 % » ;
# la garde ci-dessous a montré que l'artefact porte 2,3348 % (I-q3, 64→128).
# La borne est donc PRISE DANS L'ARTEFACT — ce qui ÉLARGIT le domaine balayé.
# Arrondir 2,3348 vers 2,3 aurait RÉTRÉCI l'incertitude après l'avoir lue.
BIAIS_RANG_ECRIT_AU_PREREG = 0.023
PAS_BALAYAGE_I_MS = 0.0005         # finesse du balayage de I sur [0, non-F]
TOLERANCE_TEMOIN = 1e-12           # le témoin doit se reproduire à l'identique


class LectureImpossible(RuntimeError):
    """Fail-loud. Jamais un `assert` : il disparaît sous `python -O`."""


def _exiger(condition: bool, message: str) -> None:
    if not condition:
        raise LectureImpossible(message)


def _cle(dico, chemin: str, source: str):
    """Descend un chemin pointé et MEURT si une marche manque."""
    courant = dico
    for marche in chemin.split("."):
        _exiger(isinstance(courant, dict) and marche in courant,
                f"clé absente : « {chemin} » dans {source} (bloqué à « {marche} »)")
        courant = courant[marche]
    return courant


def _charger(chemin: pathlib.Path) -> dict:
    _exiger(chemin.exists(), f"artefact absent : {chemin}")
    return json.loads(chemin.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# LE MODÈLE — recopié de `lire_ou_vit_le_rendu.py`, vectorisé
# --------------------------------------------------------------------------
def n_interpolations(f: float) -> float:
    """(P-b) À 30 Hz, 30 des 60 images tombent ENTRE deux pas de physique et
    doivent être interpolées depuis le readout. À 60 Hz, aucune."""
    _exiger(f in (30.0, 60.0), f"cadence non traitée par le prereg : {f}")
    return IMAGES_PAR_SECONDE - f


def travail_par_seconde(R, I, f, budget_frame, nonF, C):
    """Fenêtres·Hz disponibles par seconde. (M-2) les temps s'ADDITIONNENT :
    majorant du temps ⇒ MINORANT du cap ; tout cap lu ici est PESSIMISTE."""
    disponible = (f * budget_frame
                  - IMAGES_PAR_SECONDE * R
                  - n_interpolations(f) * I
                  - f * nonF)
    return disponible / C


def ms_par_fenetre(R, I, f, budget_frame, nonF):
    """Millisecondes allouées à UNE des 11 fenêtres de V4, à un pas donné."""
    par_pas = (budget_frame
               - (IMAGES_PAR_SECONDE * R + n_interpolations(f) * I) / f
               - nonF)
    return par_pas / N_FENETRES_V4


def cote_max(R, I, f, budget_frame, nonF, C):
    """`s_max(budget)` — une INVERSION, jamais une constante de design.
    Rend `nan` là où le budget est épuisé : pas de côté, pas de racine."""
    ms = np.asarray(ms_par_fenetre(R, I, f, budget_frame, nonF), dtype=float)
    sain = ms > 0.0
    sortie = np.full(ms.shape, np.nan)
    np.divide(ms, C, out=sortie, where=sain)
    return np.where(sain, (sortie * CELLULES_MESUREES) ** (1.0 / 3.0), np.nan)


# --------------------------------------------------------------------------
# GARDE — le modèle doit reproduire l'artefact AVANT tout balayage
# --------------------------------------------------------------------------
def garde_temoin(e: dict, budgets: dict, art: dict) -> dict:
    """Si ce modèle ne rend pas les quatre grandeurs de l'artefact `I-r5` à
    l'identique, il n'est pas le modèle de §A62 et rien de ce qui suit ne
    porte. MEURT plutôt que de balayer un modèle inventé."""
    R, nonF, C = e["R_plancher"], e["nonF"], e["C"]
    attendu = _cle(art, "temoin_de_non_regression.grandeurs", str(ART_RENDU))

    delta = (travail_par_seconde(R, 0.0, 30.0, budgets[30.0], nonF, C)
             - travail_par_seconde(R, 0.0, 60.0, budgets[60.0], nonF, C))
    s60 = float(cote_max(R, 0.0, 60.0, budgets[60.0], nonF, C))
    s_porte = float(cote_max(6.5, 0.0, 60.0, budgets[60.0], nonF, C))
    etendue = (s60 - s_porte) / s60          # rapportée au côté HAUT
    pt_mort = nonF + (30.0 * budgets[30.0] - 60.0 * budgets[60.0]) / 30.0

    obtenu = {"delta_au_plancher": float(delta), "s_max_60hz_plancher": s60,
              "etendue_cote": float(etendue), "point_mort_I": float(pt_mort)}
    ecarts = {}
    for cle, val in obtenu.items():
        ref = float(_cle(attendu, f"{cle}.ici", str(ART_RENDU)))
        ecart = abs(val - ref) / abs(ref) if ref else abs(val)
        _exiger(ecart <= TOLERANCE_TEMOIN,
                f"TÉMOIN CASSÉ sur « {cle} » : modèle {val!r} contre artefact "
                f"{ref!r} (écart relatif {ecart:.3e}). Le modèle recopié n'est "
                f"pas celui de §A62 — rien ne peut être balayé.")
        ecarts[cle] = {"modele": val, "artefact": ref, "ecart_relatif": ecart}
    return ecarts


# --------------------------------------------------------------------------
# LE BALAYAGE — §3, exhaustif sur les coins
# --------------------------------------------------------------------------
def facteurs_de_biais(biais_rang: float) -> dict[str, float]:
    """Les sept facteurs du §3, nommés. §A60 (±15 %) et la borne de rang
    empruntée à §A63 sont traités séparément ET combinés — un coin qu'on
    n'énumère pas reste disponible comme échappatoire tacite (§A62-2)."""
    a, r = BIAIS_A60, biais_rang
    return {
        "-15%·-2,3%": (1 - a) * (1 - r), "-15%": 1 - a, "-2,3%": 1 - r,
        "nominal": 1.0,
        "+2,3%": 1 + r, "+15%": 1 + a, "+15%·+2,3%": (1 + a) * (1 + r),
    }


def balayer(e: dict, budgets: dict, etiquette: str, biais_rang: float) -> dict:
    """Balaie `I` sur [0, non-F GRAVÉ] × tous les coins de (non-F, C, R), et
    évalue `T-1` (signe de Δ) et `T-2` (ordre des deux s_max)."""
    facteurs = facteurs_de_biais(biais_rang)
    nonF_grave = e["nonF"]                       # borne haute du balayage, E-6
    I = np.arange(0.0, nonF_grave + PAS_BALAYAGE_I_MS, PAS_BALAYAGE_I_MS)
    I = I[I <= nonF_grave]

    coins, delta_min, delta_max = [], np.inf, -np.inf
    t2_bascule = False
    ratio_min, ratio_max = np.inf, -np.inf
    croisements = []

    for n_nom, n_f in facteurs.items():
        for c_nom, c_f in facteurs.items():
            for r_nom, r_f in facteurs.items():
                nonF, C, R = e["nonF"] * n_f, e["C"] * c_f, e["R_plancher"] * r_f
                d = (travail_par_seconde(R, I, 30.0, budgets[30.0], nonF, C)
                     - travail_par_seconde(R, I, 60.0, budgets[60.0], nonF, C))
                s30 = cote_max(R, I, 30.0, budgets[30.0], nonF, C)
                s60 = cote_max(R, I, 60.0, budgets[60.0], nonF, C)

                # T-1 — le LIEU du croisement, analytique puis vérifié
                i_zero = nonF + (30.0 * budgets[30.0] - 60.0 * budgets[60.0]) / 30.0
                dans_le_domaine = bool(0.0 <= i_zero <= nonF_grave)

                # T-2 — l'ordre des deux côtés ; nan = budget épuisé, écarté
                ok = np.isfinite(s30) & np.isfinite(s60)
                _exiger(bool(ok.any()),
                        f"aucun point exploitable au coin {n_nom}/{c_nom}/{r_nom}")
                ratio = s30[ok] / s60[ok]
                inverse = bool((ratio <= 1.0).any())
                t2_bascule = t2_bascule or inverse

                delta_min, delta_max = min(delta_min, d.min()), max(delta_max, d.max())
                ratio_min, ratio_max = min(ratio_min, ratio.min()), max(ratio_max, ratio.max())
                if dans_le_domaine:
                    croisements.append(i_zero)
                coins.append({
                    "biais": {"non_F": n_nom, "C": c_nom, "R": r_nom},
                    "delta_a_I_nul": float(d[0]), "delta_a_I_max": float(d[-1]),
                    "I_ou_delta_s_annule_ms": float(i_zero),
                    "croisement_dans_le_domaine": dans_le_domaine,
                    "signe_bascule_dans_ce_coin": bool((d > 0).any() and (d < 0).any()),
                    "s30_sur_s60_min": float(ratio.min()),
                    "T2_inverse_dans_ce_coin": inverse,
                })

    t1_bascule = bool(delta_min < 0.0 < delta_max)
    return {
        "etiquette_budgets": etiquette, "budgets_ms": budgets,
        "n_coins": len(coins), "n_points_I": int(I.size),
        "I_balaye_ms": {"min": float(I.min()), "max": float(I.max()),
                        "pas": PAS_BALAYAGE_I_MS,
                        "borne_haute": "non-F GRAVÉ, étiqueté « 2D à l'échelle "
                                       "de l'instrument » (PREREGISTRATION.md:9559)"},
        "T1_signe_de_delta": {
            "delta_min": float(delta_min), "delta_max": float(delta_max),
            "bascule": t1_bascule,
            "lieu_du_croisement_ms": {
                "min": float(min(croisements)) if croisements else None,
                "max": float(max(croisements)) if croisements else None,
                "n_coins_ou_il_tombe_dans_le_domaine": len(croisements),
            },
        },
        "T2_ordre_des_cotes": {
            "s30_sur_s60_min": float(ratio_min), "s30_sur_s60_max": float(ratio_max),
            "bascule": t2_bascule,
        },
        "coins": coins,
    }


def contexte_hors_critere(e: dict, budgets: dict) -> dict:
    """§6 — reporté, ne décide RIEN. Écrit au prereg pour que ces chiffres ne
    puissent pas devenir des critères après avoir été vus."""
    R, nonF, C = e["R_plancher"], e["nonF"], e["C"]
    corps = {}
    for nom, r in (("R_plancher", R), ("R_porte", e["R_porte"])):
        corps[nom] = {
            "cap_60hz": float(travail_par_seconde(r, 0.0, 60.0, budgets[60.0], nonF, C) / 60.0),
            "cap_30hz_a_I_nul": float(travail_par_seconde(r, 0.0, 30.0, budgets[30.0], nonF, C) / 30.0),
            "cap_30hz_au_point_mort": float(travail_par_seconde(r, nonF, 30.0, budgets[30.0], nonF, C) / 30.0),
            "s_max_60hz_invariant_a_I": float(cote_max(r, 0.0, 60.0, budgets[60.0], nonF, C)),
            "s_max_30hz_a_I_nul": float(cote_max(r, 0.0, 30.0, budgets[30.0], nonF, C)),
            "s_max_30hz_au_point_mort": float(cote_max(r, nonF, 30.0, budgets[30.0], nonF, C)),
        }
    corps["fenetres_demandees_par_V4"] = N_FENETRES_V4
    return corps


def garde_C_et_R(e: dict, budgets: dict, biais_rang: float) -> dict:
    """MÉCANISE deux énoncés du résultat, au lieu de les affirmer.

    (1) `C` ne touche NI `T-1` NI `T-2` : diviseur positif de Δ, et il s'annule
        du rapport `s30/s60` puisque les deux côtés le portent (E-4, §A62-3).
    (2) `R` s'annule EXACTEMENT de Δ — le terme `60·R` est le même aux deux
        cadences (M-1). Il touche en revanche le rapport `s30/s60`, et ce fait
        est reporté plutôt que tu.

    Un énoncé affirmé et non vérifié est la faute `§A62-bis-2`. MEURT si l'un
    des deux est faux."""
    f = facteurs_de_biais(biais_rang)
    I = np.linspace(0.0, e["nonF"], 401)
    nonF = e["nonF"]

    def d(R, C):
        return (travail_par_seconde(R, I, 30.0, budgets[30.0], nonF, C)
                - travail_par_seconde(R, I, 60.0, budgets[60.0], nonF, C))

    def ratio(R, C):
        return (cote_max(R, I, 30.0, budgets[30.0], nonF, C)
                / cote_max(R, I, 60.0, budgets[60.0], nonF, C))

    ref_d, ref_ratio = d(e["R_plancher"], e["C"]), ratio(e["R_plancher"], e["C"])

    # (1) C : signe de Δ invariant, rapport s30/s60 invariant AU BIT PRÈS
    ecarts_C = []
    for c_f in f.values():
        C = e["C"] * c_f
        _exiger(bool(np.array_equal(np.sign(d(e["R_plancher"], C)), np.sign(ref_d))),
                f"C change le SIGNE de Δ au facteur {c_f} — l'énoncé est faux")
        ecarts_C.append(float(np.nanmax(np.abs(ratio(e["R_plancher"], C) - ref_ratio))))
    _exiger(max(ecarts_C) < 1e-12,
            f"C touche le rapport s30/s60 (écart {max(ecarts_C):.3e}) — E-4 est faux ici")

    # (2) R : Δ invariant AU BIT PRÈS ; le rapport, lui, bouge — on le CHIFFRE
    ecarts_R_delta, ecarts_R_ratio = [], []
    for r_f in f.values():
        R = e["R_plancher"] * r_f
        ecarts_R_delta.append(float(np.max(np.abs(d(R, e["C"]) - ref_d))))
        ecarts_R_ratio.append(float(np.nanmax(np.abs(ratio(R, e["C"]) - ref_ratio))))
    _exiger(max(ecarts_R_delta) < 1e-12,
            f"R ne s'annule PAS de Δ (écart {max(ecarts_R_delta):.3e}) — "
            f"l'annulation de §A62-3 ne tiendrait pas")

    return {
        "C_change_le_signe_de_delta": False,
        "C_ecart_max_sur_s30_sur_s60": max(ecarts_C),
        "R_ecart_max_sur_delta": max(ecarts_R_delta),
        "R_ecart_max_sur_s30_sur_s60": max(ecarts_R_ratio),
        "note": "C est exactement neutre sur T-1 ET T-2. R est exactement neutre "
                "sur T-1 ; il déplace le rapport s30/s60, sans jamais l'amener "
                "près de 1 (T-2 ne bascule nulle part).",
    }


def precision_derivee_de_la_decision(e: dict, jeux: dict, biais_rang: float) -> dict:
    """§5 branche (b) EXIGE ceci : nommer la grandeur, la précision requise,
    et DÉRIVER cette précision de la décision (§A60) — jamais la choisir.

    `Δ ∝ (non-F − I)` : son signe ne dépend NI de `C` (diviseur positif) NI de
    `R` (le terme `60·R` est identique des deux côtés et disparaît). Décider la
    cadence sur `T-1`, c'est décider le signe de `non-F − I`, et rien d'autre.
    """
    f = facteurs_de_biais(biais_rang)
    f_bas, f_haut = min(f.values()), max(f.values())
    lectures = {}
    for nom, b in jeux.items():
        decalage = (30.0 * b[30.0] - 60.0 * b[60.0]) / 30.0
        lectures[nom] = {
            "decalage_d_arrondi_ms": decalage,
            "I_sous_lequel_le_signe_est_DECIDE_positif_ms":
                e["nonF"] * f_bas + decalage,
            "I_au_dessus_duquel_le_signe_est_DECIDE_negatif_ms":
                e["nonF"] * f_haut + decalage,
            "largeur_de_la_zone_indecidable_ms":
                e["nonF"] * (f_haut - f_bas),
        }
    return {
        "grandeur_nommee": "I — le coût d'une interpolation de readout, "
                           "rapporté au non-F (PREREGISTRATION.md:9559, étiqueté "
                           "« 2D à l'échelle de l'instrument »)",
        "ce_qui_ne_compte_PAS": {
            "C": "diviseur POSITIF de Δ ; s'annule aussi du rapport s30/s60 — "
                 "aucun biais sur C ne peut faire basculer T-1 ni T-2. VÉRIFIÉ "
                 "par garde_C_et_R, pas affirmé.",
            "R": "le terme 60·R est identique aux deux cadences (M-1) et "
                 "disparaît EXACTEMENT de Δ — c'est l'annulation de §A62-3. Il "
                 "déplace en revanche le rapport s30/s60 ; chiffré, jamais tu.",
        },
        "lecture_en_ABSOLU": {
            "note": "si I et le non-F sont connus séparément, chacun portant le "
                    "biais ±15 % de §A60 combiné à la borne de rang de §A63",
            "par_facon_de_compter": lectures,
        },
        "lecture_en_RAPPORT": {
            "note": "si I et le non-F sont mesurés ADJACENTS DANS LE TEMPS, un "
                    "biais multiplicatif commun s'annule du rapport — c'est la "
                    "classe que §A61 laisse « probablement saine, À ÉTABLIR », et "
                    "que §A63 a précisément ÉCHOUÉ à qualifier.",
            "grandeur_suffisante": "le RAPPORT I/non-F, pas deux absolus",
            "precision_requise_derivee": biais_rang,
            "regle": "le signe est DÉCIDÉ si |I/non-F − 1| > la borne de rang ; "
                     "INDÉCIDABLE en deçà.",
            "zone_indecidable_du_rapport": [1.0 - biais_rang, 1.0 + biais_rang],
        },
        "ce_que_la_precision_requise_COUTE": {
            "en_absolu": "une mesure 3D ABSOLUE — INTERDITE par §A61 tant que "
                         "l'instrument n'est pas qualifié",
            "en_rapport": "un contraste apparié intra-run — la classe que la "
                          "re-qualification devait qualifier et n'a PAS qualifiée "
                          "(§A63, INDÉTERMINÉ global, σ_r non gravée)",
            "consequence": "dans les DEUX cas la re-qualification revient sur le "
                           "chemin critique de la cadence. Ce document NE CHOISIT "
                           "PAS entre matériel et estimateur : §A63-1 a laissé "
                           "l'attribution OUVERTE.",
        },
    }


def main() -> int:
    art = _charger(ART_RENDU)
    qualif = _charger(ART_QUALIF)

    e = {
        "C": float(_cle(art, "entrees.C_fenetre_64_ms.valeur", str(ART_RENDU))),
        "nonF": float(_cle(art, "entrees.non_f_ms.valeur", str(ART_RENDU))),
        "R_plancher": float(_cle(art, "entrees.R_plancher_ms.valeur", str(ART_RENDU))),
        "R_porte": float(_cle(art, "entrees.R_porte_ms.valeur", str(ART_RENDU))),
    }
    etiquettes = {
        "non_F": _cle(art, "entrees.non_f_ms.PROVENANCE", str(ART_RENDU)),
        "C": _cle(art, "entrees.C_fenetre_64_ms.source", str(ART_RENDU)),
        "R_plancher": _cle(art, "entrees.R_plancher_ms.sources", str(ART_RENDU)),
    }
    # §A63 — les bornes EMPRUNTÉES, lues et non recopiées
    rang = {c: float(_cle(qualif, f"indeterminations.I-q3.{c}.ecart_aller_retour",
                          str(ART_QUALIF))) for c in ("64->128", "128->32", "32->64")}
    disp = {c: float(_cle(qualif, f"contrastes.{c}.dispersion_iqr_demi", str(ART_QUALIF)))
            for c in ("64->128", "128->32", "32->64")}
    biais_rang = max(list(rang.values()) + list(disp.values()))
    _exiger(biais_rang >= BIAIS_RANG_ECRIT_AU_PREREG,
            f"la borne lue {biais_rang} est PLUS ÉTROITE que le {BIAIS_RANG_ECRIT_AU_PREREG} "
            f"du §3 : rétrécir le domaine après lecture est la fabrication que §A52 interdit")

    jeux = {
        "budget_grave": {60.0: 16.7, 30.0: 100.0 / 3.0},
        "budget_1000_sur_f": {60.0: 1000.0 / 60.0, 30.0: 1000.0 / 30.0},
    }
    # les budgets gravés viennent de l'artefact, jamais d'ici
    lus = _cle(art, "entrees.budget_frame_ms", str(ART_RENDU))
    for f, v in jeux["budget_grave"].items():
        _exiger(abs(float(lus[str(f)]) - v) < 1e-9,
                f"budget gravé {f} Hz : {v} contre {lus[str(f)]} dans l'artefact")

    temoin = garde_temoin(e, jeux["budget_grave"], art)
    neutralite = garde_C_et_R(e, jeux["budget_grave"], biais_rang)
    lectures = {nom: balayer(e, b, nom, biais_rang) for nom, b in jeux.items()}

    branches = {nom: ("(b) SENSIBLE" if (l["T1_signe_de_delta"]["bascule"]
                                         or l["T2_ordre_des_cotes"]["bascule"])
                      else "(a) INSENSIBLE") for nom, l in lectures.items()}
    accord = len(set(branches.values())) == 1

    sortie = {
        "protocole": "claude/analyse-sensibilite-cadence-2026-08-23.md",
        "lecteur": "lire_sensibilite_cadence.py",
        "nature": "LECTURE, PAS MESURE — aucun GPU, aucun cupy, aucun chronomètre. "
                  "§A61 respecté par construction.",
        "entrees": e, "etiquettes_des_entrees": etiquettes,
        "bornes_empruntees_a_A63": {
            "dependance_au_rang_I_q3": rang, "dispersions": disp,
            "borne_ecrite_au_prereg": BIAIS_RANG_ECRIT_AU_PREREG,
            "borne_retenue": biais_rang,
            "ECART_AU_PREREG": "le §3 écrivait ±2,3 % ; l'artefact porte 2,3348 % "
                               "(I-q3, 64→128). La borne LUE est retenue — le "
                               "domaine est ÉLARGI, jamais rétréci.",
            "AVERTISSEMENT": "σ_r n'est PAS gravée (§A63, §8 fermé) : ces bornes "
                             "sont EMPRUNTÉES, pas une incertitude qualifiée.",
        },
        "temoin_de_non_regression_contre_I_r5": temoin,
        "garde_neutralite_de_C_et_R": neutralite,
        "domaine_balaye": {"biais": facteurs_de_biais(biais_rang),
                           "I_ms": [0.0, e["nonF"]], "pas_ms": PAS_BALAYAGE_I_MS},
        "lectures": lectures,
        "I_r5_les_deux_facons": {
            "branches": branches, "accord": accord,
            "note": "La branche est INDÉTERMINÉE si elle diffère entre les deux "
                    "façons de compter (clause du §2 E-2, mécanisée).",
            "ecarts": {
                "point_mort_I": abs(
                    lectures["budget_grave"]["T1_signe_de_delta"]["lieu_du_croisement_ms"]["max"]
                    - lectures["budget_1000_sur_f"]["T1_signe_de_delta"]["lieu_du_croisement_ms"]["max"])
                / lectures["budget_1000_sur_f"]["T1_signe_de_delta"]["lieu_du_croisement_ms"]["max"],
            },
        },
        "precision_derivee_de_la_decision":
            precision_derivee_de_la_decision(e, jeux, biais_rang),
        "part_relative_de_delta": {
            "note": "Δ rapporté au travail total du côté 60 Hz — le « 6,2 % » de "
                    "§A62-3 en est la valeur nominale à I = 0. CONTEXTE, pas critère.",
            "travail_60hz_nominal_fenetres_hz": float(travail_par_seconde(
                e["R_plancher"], 0.0, 60.0, jeux["budget_grave"][60.0],
                e["nonF"], e["C"])),
            "delta_sur_travail_60hz": {
                nom: {"min": l["T1_signe_de_delta"]["delta_min"] / float(travail_par_seconde(
                          e["R_plancher"], 0.0, 60.0, jeux["budget_grave"][60.0],
                          e["nonF"], e["C"])),
                      "max": l["T1_signe_de_delta"]["delta_max"] / float(travail_par_seconde(
                          e["R_plancher"], 0.0, 60.0, jeux["budget_grave"][60.0],
                          e["nonF"], e["C"]))}
                for nom, l in lectures.items()},
        },
        "contexte_hors_critere_A6": contexte_hors_critere(e, jeux["budget_grave"]),
        "branche_tiree": (branches["budget_grave"] if accord else "INDÉTERMINÉE"),
    }
    SORTIE.write_text(json.dumps(sortie, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"témoin I-r5 : reproduit, écart max "
          f"{max(v['ecart_relatif'] for v in temoin.values()):.3e}")
    for nom, l in lectures.items():
        t1, t2 = l["T1_signe_de_delta"], l["T2_ordre_des_cotes"]
        print(f"\n[{nom}] {l['n_coins']} coins × {l['n_points_I']} valeurs de I")
        print(f"  T-1 signe de Δ   : bascule={t1['bascule']}  "
              f"Δ ∈ [{t1['delta_min']:+.4f}, {t1['delta_max']:+.4f}] fenêtres·Hz")
        print(f"      croisement I  : [{t1['lieu_du_croisement_ms']['min']}, "
              f"{t1['lieu_du_croisement_ms']['max']}] ms, dans le domaine sur "
              f"{t1['lieu_du_croisement_ms']['n_coins_ou_il_tombe_dans_le_domaine']}/{l['n_coins']} coins")
        print(f"  T-2 ordre côtés  : bascule={t2['bascule']}  "
              f"s30/s60 ∈ [{t2['s30_sur_s60_min']:.4f}, {t2['s30_sur_s60_max']:.4f}]")
        print(f"  ⇒ branche {branches[nom]}")
    pr = sortie["precision_derivee_de_la_decision"]
    print("\n--- §5(b) : la grandeur nommée et sa précision, DÉRIVÉE ---")
    print(f"  garde C/R (mécanisée) : C sur s30/s60 = "
          f"{neutralite['C_ecart_max_sur_s30_sur_s60']:.1e} ; R sur Δ = "
          f"{neutralite['R_ecart_max_sur_delta']:.1e} ; "
          f"R sur s30/s60 = {neutralite['R_ecart_max_sur_s30_sur_s60']:.4f}")
    for nom, v in pr["lecture_en_ABSOLU"]["par_facon_de_compter"].items():
        print(f"  [{nom}] signe DÉCIDÉ si I < "
              f"{v['I_sous_lequel_le_signe_est_DECIDE_positif_ms']:.4f} ms "
              f"ou I > {v['I_au_dessus_duquel_le_signe_est_DECIDE_negatif_ms']:.4f} ms "
              f"(zone indécidable large de "
              f"{v['largeur_de_la_zone_indecidable_ms']:.4f} ms)")
    z = pr["lecture_en_RAPPORT"]["zone_indecidable_du_rapport"]
    print(f"  en RAPPORT : décidé si |I/non-F − 1| > "
          f"{pr['lecture_en_RAPPORT']['precision_requise_derivee']:.5f} "
          f"(indécidable dans [{z[0]:.4f}, {z[1]:.4f}])")
    pd_ = sortie["part_relative_de_delta"]["delta_sur_travail_60hz"]["budget_grave"]
    print(f"  part relative de Δ (contexte, §6) : "
          f"[{pd_['min']:+.2%}, {pd_['max']:+.2%}] du travail 60 Hz")
    print(f"\nACCORD entre les deux façons : {accord}  ⇒  {sortie['branche_tiree']}")
    print(f"artefact : {SORTIE.relative_to(RACINE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
