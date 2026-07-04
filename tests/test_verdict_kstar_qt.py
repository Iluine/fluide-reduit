"""Tests unitaires PERSISTÉS de `k_star_seed_qt`/`k_star_groupe_qt`/
`gate_controles_qt` (scripts/run_arcA_verdict_qt.py, famille 2 quadtree).

Motif calqué sur `tests/test_verdict_kstar.py` (famille 1) -- MÊME esprit,
axe budgets (32, 64, 128, 256, 400, 1024, 2048) au lieu de tailles
(81, 273, 1041) : `k_star_seed`/`k_star_groupe` (famille 1) sont câblés en
dur sur l'axe à 3 niveaux (`TAILLE_PAR_NIVEAU`/`NIVEAUX_ORDRE_ASC`/`LEVELS`),
donc PAS génériques sur l'axe des tailles -- ce fichier ferme, pour les
équivalents `_qt` : (i) l'ordre croissant 32->...->2048, (ii) la clause
d'escalade 2xJND RÉELLEMENT déclenchée, (iii) « aucun budget ne satisfait ->
∞ », (iv) l'équivalence gate n=1 (`k_star_groupe_qt` sur groupe singleton ==
`k_star_seed_qt`), (v) le gate câblé sur l'attendu famille 2 (ferm=32), (vi) le
gate REQUALIFIÉ shuf (amendement de portée, PREREGISTRATION.md pocCascade2phys,
commit `fa54d20`, 2026-07-04 : shuf CONFORME ssi k* > CAP_FLOATS=409.6, VIOLATION
ssi k* <= 400 ; ferm inchangé), (vii) l'étoile Arc C (`_non_discriminant`, même
commit : budgets médians {1024, 2048} -> drapeau permanent).

Les briques GÉNÉRIQUES (`verdict_A3_jnd`, `capacite_insuffisante_jnd`, etc.,
importées telles quelles par `run_arcA_verdict_qt.py`) restent testées dans
`tests/test_verdict_kstar.py` (famille 1) -- pas de duplication ici.

Motif d'import verbatim, déjà utilisé entre scripts arcA (cf.
tests/test_verdict_kstar.py) : `sys.path.insert(ROOT)` puis
`from scripts.run_arcA_verdict_qt import ...`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import math

from scripts.run_arcA_measure import JND_LIST, L_LIST, L_LIST_CONTROL, SEEDS
from scripts.run_arcA_measure_qt import BUDGETS
from scripts.run_arcA_verdict_qt import (ATTENDU_FERM_QT, ATTENDU_SHUF_QT,
                                         CAP_FLOATS, _non_discriminant,
                                         gate_controles_qt, k_star_groupe_qt,
                                         k_star_seed_qt)

JND = 0.03  # JND de test, dans la plage réelle {2,3,4,5}% -- valeur non spéciale.


def _row(budgets_sous_jnd: tuple[int, ...]) -> np.ndarray:
    """Ligne bool alignée sur BUDGETS : True aux budgets listés (les autres
    False) -- format attendu par `k_star_seed_qt`/`k_star_groupe_qt`."""
    row = np.zeros(len(BUDGETS), dtype=bool)
    for budget in budgets_sous_jnd:
        row[BUDGETS.index(budget)] = True
    return row


# --- (i) ordre croissant 32 -> ... -> 2048 -----------------------------------


def test_k_star_seed_qt_ordre_croissant():
    """`k_star_seed_qt` retient le PLUS PETIT budget sous-JND dans l'ordre
    croissant de BUDGETS -- jamais un ordre arbitraire."""
    assert k_star_seed_qt(_row((32,))) == 32.0
    assert k_star_seed_qt(_row((256,))) == 256.0
    assert k_star_seed_qt(_row((2048,))) == 2048.0
    # 32 ET 2048 sous-JND -> doit retenir 32, pas 2048.
    assert k_star_seed_qt(_row((32, 2048))) == 32.0
    # 128 ET 1024 sous-JND, PAS 32 -> doit retenir 128, pas 1024.
    assert k_star_seed_qt(_row((128, 1024))) == 128.0


def test_k_star_seed_qt_aucun_budget_sous_jnd_est_infini():
    assert k_star_seed_qt(_row(())) == float("inf")


def test_k_star_groupe_qt_mediane_deja_infinie_ne_declenche_aucune_escalade():
    """Si la médiane des k*(seed) du groupe est déjà ∞ (majorité jamais
    sous-JND), `k_star_groupe_qt` retourne ∞ SANS entrer dans la boucle
    d'escalade (`audit["escalades"]` reste vide)."""
    n = 5
    sous_jnd = np.zeros((n, len(BUDGETS)), dtype=bool)   # personne jamais sous-JND
    ma1 = np.zeros((n, len(BUDGETS)))
    ma2 = np.zeros((n, len(BUDGETS)))
    k, audit = k_star_groupe_qt(sous_jnd, ma1, ma2, JND)
    assert k == float("inf")
    assert audit["mediane_initiale"] == float("inf")
    assert audit["escalades"] == []
    assert audit["k_final"] == float("inf")


# --- (ii) clause d'escalade 2xJND, RÉELLEMENT déclenchée ---------------------


def test_k_star_groupe_qt_escalade_declenchee_par_une_seed_polluante():
    """5 seeds sous-JND à budget=32 (médiane=32) ; UNE seed (la première) a
    max(M-A1,M-A2) = 2.5*JND > 2*JND À CE BUDGET -> la clause déclenche
    l'escalade vers le budget supérieur (64) ; aucune seed ne dépasse 2xJND à
    64 -> k* s'arrête à 64 (escalade d'UN SEUL budget, pas jusqu'à ∞)."""
    n = 5
    sous_jnd = np.zeros((n, len(BUDGETS)), dtype=bool)
    ma1 = np.zeros((n, len(BUDGETS)))
    ma2 = np.zeros((n, len(BUDGETS)))
    i32, i64 = BUDGETS.index(32), BUDGETS.index(64)
    for i in range(n):
        sous_jnd[i, i32] = True   # toutes sous-JND à 32 -> médiane initiale = 32
        sous_jnd[i, i64] = True   # et à 64 (pour que k* puisse s'y arrêter)
        ma1[i, i32] = 0.5 * JND   # sous JND ET sous 2xJND à 32...
        ma1[i, i64] = 0.1 * JND   # ...et bien sous JND à 64
    ma1[0, i32] = 2.5 * JND       # seed polluante : dépasse 2xJND à 32 -> déclenche

    k, audit = k_star_groupe_qt(sous_jnd, ma1, ma2, JND)

    assert audit["mediane_initiale"] == 32.0
    assert k == 64.0
    assert audit["k_final"] == 64.0
    assert len(audit["escalades"]) == 1
    assert audit["escalades"][0]["niveau_teste"] == 32.0
    assert audit["escalades"][0]["n_seeds_offenders"] == 1


def test_k_star_groupe_qt_escalade_totale_si_toutes_les_seeds_polluent_partout():
    """(iii) via `k_star_groupe_qt`, cas dégénéré : la médiane initiale est
    finie (32) mais CHAQUE budget testé (les 7) est en violation de la clause
    2xJND pour toutes les seeds -> la boucle épuise les 7 budgets et retourne
    ∞, sans exception ni boucle infinie."""
    n = 5
    sous_jnd = np.ones((n, len(BUDGETS)), dtype=bool)
    ma1 = np.full((n, len(BUDGETS)), 2.5 * JND)   # viole 2xJND À TOUS les budgets
    ma2 = np.zeros((n, len(BUDGETS)))
    k, audit = k_star_groupe_qt(sous_jnd, ma1, ma2, JND)
    assert k == float("inf")
    assert audit["mediane_initiale"] == 32.0
    assert audit["k_final"] == float("inf")
    assert [e["niveau_teste"] for e in audit["escalades"]] == [float(b) for b in BUDGETS]


# --- (iv) équivalence gate n=1 : k_star_groupe_qt(singleton) == k_star_seed_qt --


def test_equivalence_groupe_singleton_egale_k_star_seed_qt_cas_simples():
    """Preuve du docstring de `k_star_groupe_qt` VÉRIFIÉE PAR EXÉCUTION : pour
    un groupe réduit à 1 seed, `k_star_groupe_qt` == `k_star_seed_qt` de cette
    même seed, sur plusieurs cas (quelques budgets, et ∞) -- valeurs sous-JND
    respectant l'invariant réel (max < JND, jamais proche de 2xJND)."""
    for budgets_true in [(32,), (256,), (2048,), (32, 2048), (128, 1024), ()]:
        row = _row(budgets_true)
        attendu = k_star_seed_qt(row)
        ma1_grp = np.full((1, len(BUDGETS)), 0.1 * JND)
        ma2_grp = np.zeros((1, len(BUDGETS)))
        obtenu, _audit = k_star_groupe_qt(row[np.newaxis, :], ma1_grp, ma2_grp, JND)
        assert obtenu == attendu, f"budgets_true={budgets_true}"


def test_equivalence_groupe_singleton_egale_k_star_seed_qt_cas_proche_de_2xjnd():
    """Cas critique (miroir famille 1) : une seed sous-JND à son PROPRE
    budget (32) avec une valeur brute PROCHE de JND (0.9*JND, donc
    mécaniquement < 2*JND -- l'invariant qui rend la clause un no-op),
    réduite à un groupe singleton, NE DOIT JAMAIS s'auto-déclencher."""
    row = _row((32, 64))
    ma1_grp = np.zeros((1, len(BUDGETS)))
    ma1_grp[0, BUDGETS.index(64)] = 0.1 * JND
    ma1_grp[0, BUDGETS.index(32)] = 0.9 * JND
    ma2_grp = np.zeros((1, len(BUDGETS)))
    attendu = k_star_seed_qt(row)
    assert attendu == 32.0
    obtenu, audit = k_star_groupe_qt(row[np.newaxis, :], ma1_grp, ma2_grp, JND)
    assert obtenu == attendu
    assert audit["escalades"] == []   # no-op mécanique : jamais d'escalade à n=1


# --- (v) gate câblé sur l'attendu famille 2 (ferm=32) via k_star_groupe_qt --


def test_gate_controles_qt_conforme_sur_donnees_synthetiques_coherentes():
    """`gate_controles_qt` doit être câblé sur `k_star_groupe_qt` (n=1), sur
    l'attendu famille 2 (ferm=32, PAS 81) : jeu synthétique minimal aux axes
    attendus (nL=len(L_LIST), nSeed=5, nBudget=7, nJND=4) -- ferm sous-JND à
    budget=32 partout (k*=32=attendu) ; shuf jamais sous-JND (k*=∞=attendu)
    -> CONFORME, zéro violation."""
    nL, nSeed, nBudget, nJND = len(L_LIST), len(SEEDS), len(BUDGETS), len(JND_LIST)
    i32 = BUDGETS.index(32)

    sous_jnd_ferm = np.zeros((nL, nSeed, nBudget, nJND), dtype=bool)
    ma1_ferm = np.zeros((nL, nSeed, nBudget))
    ma2_ferm = np.zeros((nL, nSeed, nBudget))
    for L in L_LIST_CONTROL:
        iL = L_LIST.index(L)
        sous_jnd_ferm[iL, :, i32, :] = True
        ma1_ferm[iL, :, i32] = 0.1 * min(JND_LIST)

    sous_jnd_shuf = np.zeros((nL, nSeed, nBudget, nJND), dtype=bool)   # jamais sous-JND
    ma1_shuf = np.zeros((nL, nSeed, nBudget))
    ma2_shuf = np.zeros((nL, nSeed, nBudget))

    statut, violations = gate_controles_qt(sous_jnd_ferm, ma1_ferm, ma2_ferm,
                                           sous_jnd_shuf, ma1_shuf, ma2_shuf)
    assert statut == "CONFORME"
    assert violations == []


def test_gate_controles_qt_detecte_une_violation_ponctuelle():
    """Une seule cellule ferm cassée (seed 0, L=10, premier JND, plus jamais
    sous-JND nulle part) -> VIOLATION détectée EXACTEMENT sur cette cellule,
    avec les bons champs et `k_star_attendu == ATTENDU_FERM_QT` (32.0)."""
    nL, nSeed, nBudget, nJND = len(L_LIST), len(SEEDS), len(BUDGETS), len(JND_LIST)
    i32 = BUDGETS.index(32)

    sous_jnd_ferm = np.zeros((nL, nSeed, nBudget, nJND), dtype=bool)
    ma1_ferm = np.zeros((nL, nSeed, nBudget))
    ma2_ferm = np.zeros((nL, nSeed, nBudget))
    for L in L_LIST_CONTROL:
        iL = L_LIST.index(L)
        sous_jnd_ferm[iL, :, i32, :] = True
        ma1_ferm[iL, :, i32] = 0.1 * min(JND_LIST)

    sous_jnd_shuf = np.zeros((nL, nSeed, nBudget, nJND), dtype=bool)
    ma1_shuf = np.zeros((nL, nSeed, nBudget))
    ma2_shuf = np.zeros((nL, nSeed, nBudget))

    iL10 = L_LIST.index(10)
    sous_jnd_ferm[iL10, 0, :, 0] = False   # seed 0 : plus jamais sous-JND (k*=∞)

    statut, violations = gate_controles_qt(sous_jnd_ferm, ma1_ferm, ma2_ferm,
                                           sous_jnd_shuf, ma1_shuf, ma2_shuf)
    assert statut == "VIOLATION"
    assert len(violations) == 1
    v = violations[0]
    assert v["bras"] == "ferm"
    assert v["L"] == 10
    assert v["seed"] == SEEDS[0]
    assert v["jnd"] == JND_LIST[0]
    assert v["k_star_obtenu"] == float("inf")
    assert v["k_star_attendu"] == ATTENDU_FERM_QT
    assert ATTENDU_FERM_QT == 32.0
    assert ATTENDU_SHUF_QT == float("inf")


# --- (vi) gate REQUALIFIÉ (amendement de portée, PREREGISTRATION.md pocCascade2phys,
# commit fa54d20, 2026-07-04) : shuf CONFORME ssi k* > CAP_FLOATS (409.6), VIOLATION
# ssi k* <= 400 -- ferm INCHANGÉ (égalité stricte à 32). ---------------------------


def _gate_synthetic_shuf(k_star_vise: float) -> tuple[str, list[dict]]:
    """Jeu synthétique ferm+shuf minimal (mêmes axes que les tests (v) ci-dessus) où
    ferm ferme partout à 32 (CONFORME, inchangé) et shuf ferme EXACTEMENT à
    `k_star_vise` (un budget fini de BUDGETS, ou float("inf")) sur toute la grille de
    contrôle -- isole la clause requalifiée shuf de `gate_controles_qt` sans retoucher
    ferm ni la clause 2xJND."""
    nL, nSeed, nBudget, nJND = len(L_LIST), len(SEEDS), len(BUDGETS), len(JND_LIST)
    i32 = BUDGETS.index(32)

    sous_jnd_ferm = np.zeros((nL, nSeed, nBudget, nJND), dtype=bool)
    ma1_ferm = np.zeros((nL, nSeed, nBudget))
    ma2_ferm = np.zeros((nL, nSeed, nBudget))
    for L in L_LIST_CONTROL:
        iL = L_LIST.index(L)
        sous_jnd_ferm[iL, :, i32, :] = True
        ma1_ferm[iL, :, i32] = 0.1 * min(JND_LIST)

    sous_jnd_shuf = np.zeros((nL, nSeed, nBudget, nJND), dtype=bool)
    ma1_shuf = np.zeros((nL, nSeed, nBudget))
    ma2_shuf = np.zeros((nL, nSeed, nBudget))
    if math.isfinite(k_star_vise):
        i_vise = BUDGETS.index(int(k_star_vise))
        for L in L_LIST_CONTROL:
            iL = L_LIST.index(L)
            sous_jnd_shuf[iL, :, i_vise, :] = True
            ma1_shuf[iL, :, i_vise] = 0.1 * min(JND_LIST)
    # sinon (k_star_vise = inf) : sous_jnd_shuf reste tout à False -> k*=∞ partout.

    return gate_controles_qt(sous_jnd_ferm, ma1_ferm, ma2_ferm,
                             sous_jnd_shuf, ma1_shuf, ma2_shuf)


def test_gate_controles_qt_shuf_kstar_2048_est_conforme_sous_le_cap():
    """Amendement fa54d20 : shuf fermant EXACTEMENT à budget=2048 (> CAP_FLOATS =
    409.6) est désormais CONFORME. Avant l'amendement (égalité stricte à ∞), ceci
    aurait été une VIOLATION -- c'est très exactement le comportement observé sur
    `measures_qt.npz` (7 cellules shuf fermant à 2048, cf. rapport)."""
    statut, violations = _gate_synthetic_shuf(2048.0)
    assert statut == "CONFORME"
    assert violations == []


def test_gate_controles_qt_shuf_kstar_infini_reste_conforme():
    """k*=∞ (aucun budget jamais sous-JND) reste CONFORME après l'amendement --
    la requalification ÉLARGIT la zone de conformité, elle ne retire rien à ce qui
    l'était déjà avant."""
    statut, violations = _gate_synthetic_shuf(float("inf"))
    assert statut == "CONFORME"
    assert violations == []


def test_gate_controles_qt_shuf_kstar_400_est_en_violation():
    """k*=400 (<= CAP_FLOATS=409.6, donc SOUS le cap) reste une VIOLATION -- la
    requalification ne dispense QUE les budgets qui dépassent le cap (1024, 2048),
    pas les budgets sous-cap."""
    assert CAP_FLOATS == 409.6 and 400.0 <= CAP_FLOATS   # 400 est bien SOUS le cap
    statut, violations = _gate_synthetic_shuf(400.0)
    assert statut == "VIOLATION"
    assert len(violations) == len(L_LIST_CONTROL) * len(SEEDS) * len(JND_LIST)
    assert all(v["bras"] == "shuf" and v["k_star_obtenu"] == 400.0 for v in violations)
    assert all(v["k_star_attendu"] == ATTENDU_SHUF_QT for v in violations)


def test_gate_controles_qt_ferm_inchange_sous_amendement():
    """Ferm reste INCHANGÉ par l'amendement (portée shuf uniquement) : égalité
    stricte à 32 -- un ferm fermant à 64 (budget immédiatement supérieur, plausible
    si l'amendement avait été mal câblé sur les deux bras) reste une VIOLATION, PAS
    une conformité par "proximité" du cap ou d'un budget voisin."""
    nL, nSeed, nBudget, nJND = len(L_LIST), len(SEEDS), len(BUDGETS), len(JND_LIST)
    i64 = BUDGETS.index(64)

    sous_jnd_ferm = np.zeros((nL, nSeed, nBudget, nJND), dtype=bool)
    ma1_ferm = np.zeros((nL, nSeed, nBudget))
    ma2_ferm = np.zeros((nL, nSeed, nBudget))
    for L in L_LIST_CONTROL:
        iL = L_LIST.index(L)
        sous_jnd_ferm[iL, :, i64, :] = True   # ferme à 64, PAS 32
        ma1_ferm[iL, :, i64] = 0.1 * min(JND_LIST)

    sous_jnd_shuf = np.zeros((nL, nSeed, nBudget, nJND), dtype=bool)   # k*=∞, conforme
    ma1_shuf = np.zeros((nL, nSeed, nBudget))
    ma2_shuf = np.zeros((nL, nSeed, nBudget))

    statut, violations = gate_controles_qt(sous_jnd_ferm, ma1_ferm, ma2_ferm,
                                           sous_jnd_shuf, ma1_shuf, ma2_shuf)
    assert statut == "VIOLATION"
    assert len(violations) == len(L_LIST_CONTROL) * len(SEEDS) * len(JND_LIST)
    assert all(v["bras"] == "ferm" for v in violations)
    assert all(v["k_star_obtenu"] == 64.0 and v["k_star_attendu"] == 32.0
              for v in violations)


# --- (vii) étoile Arc C (clause de forme 2, amendement fa54d20) : `_non_discriminant` --


def test_non_discriminant_vrai_pour_budgets_1024_et_2048():
    """Les deux budgets diagnostiques qui ont motivé la requalification de l'attendu
    shuf (>= moitié de la résolution complète du domaine) portent le drapeau
    permanent « non-discriminant vs bruit »."""
    assert _non_discriminant(1024.0) is True
    assert _non_discriminant(2048.0) is True


def test_non_discriminant_faux_pour_budget_400_et_infini():
    """Un budget sous-cap (400, donc déjà discriminant par construction du cap) ou
    ∞ (jamais fermé) ne porte PAS le drapeau -- la clause de forme 2 ne s'applique
    qu'aux deux budgets diagnostiques {1024, 2048}."""
    assert _non_discriminant(400.0) is False
    assert _non_discriminant(float("inf")) is False
