"""Arc A / manche 1, famille 2 (quadtree) -- k*(L), gate des contrôles,
verdict §A3 mécanique, mapping des trois issues pré-écrites.

Contrat : « Arbitrage fork famille-vs-manche-2 » (PREREGISTRATION.md,
pocCascade2phys, commit `a8ed659`) + « AMENDEMENT FAMILLE 2 » du plan
(`docs/superpowers/plans/arc-a-manche1.md`, commit `78fbe0c`), reproduits
verbatim dans `.superpowers/sdd/refond-task9-harnais-qt-brief.md`. Ce script
est le POINT FINAL du pipeline famille 2 : il lit `outputs/arcA/
measures_qt.npz` (`run_arcA_measure_qt.py`) exclusivement et applique
MÉCANIQUEMENT les règles gravées -- il ne décide rien.

MÊME sémantique que `scripts/run_arcA_verdict.py` (famille 1, INTOUCHÉ) --
réutilisé par IMPORT tout ce qui est GÉNÉRIQUE sur l'axe des tailles
(`verdict_A3_jnd`, `verdict_A3_global`, `capacite_insuffisante_jnd`,
`verdict_A3_jnd_avec_capacite`, `verdict_A3_global_avec_capacite`,
`MESSAGE_INDETERMINE_CAPACITE`, `_to_jsonable`, palettes `_COULEURS_JND`/
`_COULEURS_BRAS` -- ces fonctions n'inspectent que k40/k80/IC/booléens
génériques, jamais un niveau ou une taille codée en dur, cf. leurs
docstrings). `k_star_seed`/`k_star_groupe`/`pente_bootstrap`/`cellule_A0`,
EUX, sont câblés en dur sur l'axe à 3 niveaux (1041->273->81, via
`TAILLE_PAR_NIVEAU`/`NIVEAUX_ORDRE_ASC`/`LEVELS`) -- PAS génériques sur l'axe
des tailles : ce module en fournit des équivalents `_qt`, à SÉMANTIQUE
IDENTIQUE mais paramétrés sur `BUDGETS` (7 valeurs), testés dans
`tests/test_verdict_kstar_qt.py` (même esprit que `tests/test_verdict_kstar.py`).

Extrait verbatim (brief, volet 2) :

    k*(L, seed ; JND) = plus petit budget de la GRILLE COMPLÈTE (7 valeurs,
    ordre croissant) sous-JND ; ∞ sinon ; clause 2×JND inchangée ; médiane
    5 seeds inchangée.
    Gate des contrôles EN PREMIER, par cellule : ferm attendu k* = 32 partout
    (2L × 5 seeds × 4 JND) ; shuf attendu k* = ∞ partout. Violation -> verdict
    v2 scellé (même mécanique que famille 1).
    Pente §A3 sur L ∈ {40, 80} + IC bootstrap 10 000 (default_rng(20260704)) ;
    PASS ssi IC ∋ 0 ET k*(80) ≤ 409.6 ; FAIL ssi pente > 0 hors IC sans
    plateau ; INDÉTERMINÉ sinon ; INDÉTERMINÉ-CAPACITÉ si k* = ∞ à tous les L
    (message gravé inchangé). Global : règle inchangée + global
    INDÉTERMINÉ-CAPACITÉ ssi 4/4.
    Mapping des trois issues (clause gravée, imprimé avec le verdict) :
      - global PASS -> « cellule 1 : court-circuit de la manche 2 sur ce
        substrat » ;
      - global FAIL -> « cellule 2 : FAIL-mur -- le ledger passe d'hypothèse
        à obligation » ;
      - global INDÉTERMINÉ-CAPACITÉ -> « NON-DÉMONTRÉ (règle de dernière
        famille) -- la dépense suivante est la manche 2 » ;
      - sinon -> « global INDÉTERMINÉ -- surface k*(L, JND) portée à l'Arc C »
        (lecture pré-écrite clause 4).

Ordre d'exécution (mécanique, non négociable, IDENTIQUE famille 1) :
  1. GATE DES CONTRÔLES (ferm/shuf), calculé PAR CELLULE (bras, L, seed, JND)
     -- `gate_controles_qt` APPELLE LITTÉRALEMENT `k_star_groupe_qt` avec un
     groupe réduit à une seule seed (n_seed=1) : MÊME code, mêmes budgets
     croissants, même clause 2xJND que le bras v2 -- la clause y est un
     no-op mécanique (même preuve que famille 1, cf. docstring de
     `k_star_groupe_qt`), vérifié par exécution
     (`tests/test_verdict_kstar_qt.py`). gate_controles_qt = "CONFORME" ssi
     zéro violation sur les 80 cellules (2 bras x {10,80} x 5 seeds x 4 JND),
     sinon "VIOLATION" -- imprimé et sérialisé EN PREMIER.
  2. k*(L) bras v2 (médiane + clause 2xJND), pente L∈{40,80} + IC bootstrap,
     verdict §A3 par JND + global -- CALCULÉS ET STOCKÉS dans tous les cas
     (audit), mais IMPRIMÉS comme verdict de la manche SSI
     gate_controles_qt = CONFORME. Si VIOLATION : la clé `verdict_v2_scelle`
     porte la mention explicite « NON LISIBLE » -- aucun PASS/FAIL/issue
     n'est imprimé comme verdict de la manche.
  3. Mapping des trois issues + surface k*(L, JND) + figures (PAS de GIF --
     YAGNI, le diagnostic escalier famille 1 suffit, cf. brief).

AUCUNE interprétation au-delà de ce mapping : ce script calcule, imprime,
sérialise -- il ne diagnostique rien de plus, ne recommande rien.

Usage :
  .venv/bin/python scripts/run_arcA_verdict_qt.py

Sortie : `outputs/arcA/verdict_qt.json` + `outputs/arcA/arcA_kstar_vs_L_qt.png`
+ `outputs/arcA/arcA_jnd_sensitivity_qt.png` +
`outputs/arcA/arcA_kstar_surface_qt.png`. Déterministe (bootstrap seedé
`default_rng(20260704)`, aucune physique rejouée ici -- lit `measures_qt.npz`
exclusivement) -- un double run produit un `verdict_qt.json` identique hors
`elapsed_seconds`.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.run_arcA_measure import (ARMS, CAP_FLOATS_10PCT, JND_LIST,
                                      L_LIST, L_LIST_CONTROL, SEEDS,
                                      _sanity_prefixe_tirage)
from scripts.run_arcA_measure_qt import BUDGETS, BUDGET_FERM, MEASURES_PATH
from scripts.run_arcA_verdict import (MESSAGE_INDETERMINE_CAPACITE,
                                      _COULEURS_BRAS, _COULEURS_JND,
                                      _to_jsonable,
                                      verdict_A3_global_avec_capacite,
                                      verdict_A3_jnd_avec_capacite)

OUT_DIR = ROOT / "outputs" / "arcA"
VERDICT_PATH = OUT_DIR / "verdict_qt.json"
FIG_KSTAR_PATH = OUT_DIR / "arcA_kstar_vs_L_qt.png"
FIG_SENSITIVITY_PATH = OUT_DIR / "arcA_jnd_sensitivity_qt.png"
FIG_SURFACE_PATH = OUT_DIR / "arcA_kstar_surface_qt.png"

# --- Constantes k*(L) (axe budgets, famille 2) -------------------------------
ORDRE_BUDGETS_ASC: tuple[float, ...] = tuple(float(b) for b in BUDGETS) + (float("inf"),)
CAP_FLOATS: float = CAP_FLOATS_10PCT                         # 409.6, INCHANGÉ

L_SLOPE: tuple[int, int] = (40, 80)                          # moitié haute de L
N_BOOTSTRAP: int = 10_000
SEED_BOOTSTRAP: int = 20260704                               # figée, fraîche PAR JND (idem famille 1)

ATTENDU_FERM_QT: float = float(BUDGET_FERM)   # amendement famille 2 : 32.0
ATTENDU_SHUF_QT: float = float("inf")         # inchangé vs famille 1


# --- k*(L, seed ; JND) et k*(L ; JND) (médiane + clause 2xJND), axe budgets --


def _idx_budget(budget: int) -> int:
    """Index du `budget` dans l'axe BUDGETS des tableaux measures_qt.npz --
    équivalent de `_idx_niveau` (famille 1), sur l'axe budgets (déjà en ordre
    croissant -- pas de renversement nécessaire, contrairement à LEVELS=(1,2,3)
    qui code l'ordre DÉCROISSANT de taille)."""
    return BUDGETS.index(budget)


def k_star_seed_qt(sous_jnd_row: np.ndarray) -> float:
    """k*(L, seed ; JND), famille 2 : plus petit budget dans l'ordre croissant
    de `BUDGETS` (32 -> ... -> 2048) dont la cellule est sous-JND (`sous_jnd_row`,
    bool aligné sur BUDGETS) ; ∞ (np.inf) si aucun -- équivalent exact de
    `k_star_seed` (famille 1), paramétré sur l'axe budgets."""
    for budget in BUDGETS:
        if bool(sous_jnd_row[_idx_budget(budget)]):
            return float(budget)
    return float("inf")


def k_star_groupe_qt(sous_jnd_grp: np.ndarray, ma1_grp: np.ndarray, ma2_grp: np.ndarray,
                     jnd: float) -> tuple[float, dict]:
    """k*(L ; JND) = médiane des k*(L, seed ; JND) du groupe, avec la clause
    « aucune seed > 2×JND au budget retenu par la médiane » -- ÉQUIVALENT
    EXACT de `k_star_groupe` (famille 1), paramétré sur l'axe budgets :
    si UNE seed du groupe a max(M-A1,M-A2) > 2·JND au budget retenu, on passe
    au budget supérieur (32->64->...->2048->∞) et on réitère ; si aucun budget
    ne satisfait, k* = ∞.

    `sous_jnd_grp`/`ma1_grp`/`ma2_grp` : shape (n_seed, n_budget), alignées
    sur BUDGETS. Fonction générique : utilisée pour le bras v2 (n_seed=5,
    médiane réelle) ET pour le GATE DES CONTRÔLES (n_seed=1 -- même preuve de
    no-op mécanique que `k_star_groupe` : si k*(seed) = budget B (fini), alors
    par construction max(M-A1,M-A2) < JND < 2·JND à B pour CETTE seed -- la
    seule du groupe -- donc aucune escalade n'est jamais déclenchée). Retourne
    (k*, audit) où `audit` documente la médiane initiale, les éventuelles
    escalades, et le budget final -- pour `verdict_qt.json`."""
    n_seed = sous_jnd_grp.shape[0]
    k_par_seed = np.array([k_star_seed_qt(sous_jnd_grp[i]) for i in range(n_seed)])
    mediane = float(np.median(k_par_seed))
    audit: dict = dict(k_par_seed=k_par_seed.tolist(), mediane_initiale=mediane,
                       escalades=[])
    if not math.isfinite(mediane):
        audit["k_final"] = float("inf")
        return float("inf"), audit

    idx = ORDRE_BUDGETS_ASC.index(mediane)
    while math.isfinite(ORDRE_BUDGETS_ASC[idx]):
        budget = ORDRE_BUDGETS_ASC[idx]
        b_idx = _idx_budget(int(budget))
        max_vals = np.maximum(ma1_grp[:, b_idx], ma2_grp[:, b_idx])
        offenders = max_vals > 2.0 * jnd
        if np.any(offenders):
            audit["escalades"].append(dict(niveau_teste=budget,
                                           n_seeds_offenders=int(offenders.sum())))
            idx += 1
            continue
        audit["k_final"] = budget
        return budget, audit

    audit["k_final"] = float("inf")
    return float("inf"), audit


def calculer_k_star_arm_qt(sous_jnd_arm: np.ndarray, ma1_arm: np.ndarray,
                           ma2_arm: np.ndarray, l_list_arm: tuple[int, ...]) -> dict:
    """k* complets d'un bras : par seed ET médiane+clause, pour chaque L de
    `l_list_arm` (grille propre au bras) x chaque JND de JND_LIST. Retourne
    {L: {jnd: {"par_seed": {seed: k}, "mediane": k, "audit": {...}}}} --
    équivalent exact de `calculer_k_star_arm` (famille 1), axe budgets."""
    out: dict = {}
    for L in l_list_arm:
        iL = L_LIST.index(L)
        out[L] = {}
        for jnd in JND_LIST:
            iJ = JND_LIST.index(jnd)
            k_par_seed = {seed: k_star_seed_qt(sous_jnd_arm[iL, iS, :, iJ])
                         for iS, seed in enumerate(SEEDS)}
            k_med, audit = k_star_groupe_qt(sous_jnd_arm[iL, :, :, iJ], ma1_arm[iL],
                                            ma2_arm[iL], jnd)
            out[L][jnd] = dict(par_seed=k_par_seed, mediane=k_med, audit=audit)
    return out


# --- Gate des contrôles (amendement famille 2 : ferm attendu 32) ------------


def gate_controles_qt(sous_jnd_ferm: np.ndarray, ma1_ferm: np.ndarray, ma2_ferm: np.ndarray,
                      sous_jnd_shuf: np.ndarray, ma1_shuf: np.ndarray, ma2_shuf: np.ndarray
                      ) -> tuple[str, list[dict]]:
    """Gate d'instrument (amendement famille 2), calculé PAR CELLULE (bras, L,
    seed, JND) -- PAS de médiane : appelle LITTÉRALEMENT `k_star_groupe_qt`
    avec un groupe réduit à CETTE SEULE seed (n_seed=1) -- même fonction,
    mêmes budgets croissants, même clause 2xJND que le bras v2. Équivalent
    exact de `gate_controles` (famille 1), attendus mis à jour pour la
    famille 2 (`ATTENDU_FERM_QT`=32.0, `ATTENDU_SHUF_QT`=∞). Comparaison par
    égalité EXACTE (mêmes raisons que famille 1 : valeurs discrètes dans
    {32.0, 64.0, ..., 2048.0, inf}, jamais un résultat d'arithmétique).

    Retourne (statut, violations) ; statut = "CONFORME" ssi zéro violation
    sur les 80 cellules (2 bras x {10,80} x 5 seeds x 4 JND), sinon
    "VIOLATION"."""
    violations: list[dict] = []
    specs = (
        ("ferm", sous_jnd_ferm, ma1_ferm, ma2_ferm, ATTENDU_FERM_QT),
        ("shuf", sous_jnd_shuf, ma1_shuf, ma2_shuf, ATTENDU_SHUF_QT),
    )
    for arm_name, sous_jnd_arm, ma1_arm, ma2_arm, attendu in specs:
        for L in L_LIST_CONTROL:
            iL = L_LIST.index(L)
            for jnd in JND_LIST:
                iJ = JND_LIST.index(jnd)
                for iS, seed in enumerate(SEEDS):
                    sous_jnd_grp = sous_jnd_arm[iL, iS:iS + 1, :, iJ]
                    ma1_grp = ma1_arm[iL, iS:iS + 1, :]
                    ma2_grp = ma2_arm[iL, iS:iS + 1, :]
                    k, _audit = k_star_groupe_qt(sous_jnd_grp, ma1_grp, ma2_grp, jnd)
                    conforme = (k == attendu)
                    if not conforme:
                        violations.append(dict(bras=arm_name, L=int(L), seed=int(seed),
                                               jnd=float(jnd), k_star_obtenu=k,
                                               k_star_attendu=attendu))
    statut = "CONFORME" if not violations else "VIOLATION"
    return statut, violations


# --- Pente k*(L) sur L∈{40,80} + IC 95% bootstrap inter-seeds (bras v2) ------


def pente_bootstrap_qt(sous_jnd_v2: np.ndarray, ma1_v2: np.ndarray, ma2_v2: np.ndarray,
                       jnd: float, k40_pt: float, k80_pt: float) -> dict:
    """Équivalent exact de `pente_bootstrap` (famille 1), câblé sur
    `k_star_groupe_qt` (axe budgets) -- pente ponctuelle (k*(80)-k*(40))/40
    (déjà calculée en amont, passée en argument) + IC 95 % par bootstrap
    inter-seeds : 10 000 tirages, resample des 5 seeds AVEC remise -- MÊME
    jeu d'indices tiré pour L=40 et L=80 à chaque tirage, recalcul
    médiane+clause+pente PAR TIRAGE, percentiles 2.5/97.5. `default_rng(
    20260704)` figée, FRAÎCHE pour cette cellule JND.

    Cas ∞ : identique famille 1 -- un tirage avec ∞ des DEUX côtés donne une
    pente indéfinie (exclue des percentiles, comptée dans
    `n_bootstrap_indefinis`) ; un seul côté infini donne ±∞, conservé."""
    iL40, iL80 = L_LIST.index(L_SLOPE[0]), L_LIST.index(L_SLOPE[1])
    jnd_idx = JND_LIST.index(jnd)
    n_seed = sous_jnd_v2.shape[1]
    rng = np.random.default_rng(SEED_BOOTSTRAP)

    def _k_pour(iL: int, idx_seeds: np.ndarray) -> float:
        k, _audit = k_star_groupe_qt(sous_jnd_v2[iL, idx_seeds, :, jnd_idx],
                                     ma1_v2[iL, idx_seeds, :], ma2_v2[iL, idx_seeds, :], jnd)
        return k

    pente_pt = ((k80_pt - k40_pt) / 40.0
               if math.isfinite(k40_pt) and math.isfinite(k80_pt) else float("nan"))

    pentes = np.empty(N_BOOTSTRAP, dtype=np.float64)
    for b in range(N_BOOTSTRAP):
        idx_tirage = rng.integers(0, n_seed, size=n_seed)
        k40_b = _k_pour(iL40, idx_tirage)
        k80_b = _k_pour(iL80, idx_tirage)
        pentes[b] = (k80_b - k40_b) / 40.0

    n_nan = int(np.isnan(pentes).sum())
    valides = pentes[~np.isnan(pentes)]
    if valides.size == 0:
        ic_lo, ic_hi = float("nan"), float("nan")
    else:
        ic_lo, ic_hi = (float(v) for v in np.percentile(valides, [2.5, 97.5]))

    return dict(k_star_L40=k40_pt, k_star_L80=k80_pt, pente=pente_pt,
               ic95=[ic_lo, ic_hi], n_bootstrap=N_BOOTSTRAP,
               n_bootstrap_indefinis=n_nan, seed_bootstrap=SEED_BOOTSTRAP)


# --- Mapping des trois issues pré-écrites (clause gravée, brief volet 2) ----

MESSAGE_ISSUE_PASS: str = "cellule 1 : court-circuit de la manche 2 sur ce substrat"
MESSAGE_ISSUE_FAIL: str = "cellule 2 : FAIL-mur — le ledger passe d'hypothèse à obligation"
MESSAGE_ISSUE_CAPACITE: str = ("NON-DÉMONTRÉ (règle de dernière famille) — la dépense "
                               "suivante est la manche 2")
MESSAGE_ISSUE_INDETERMINE: str = "global INDÉTERMINÉ — surface k*(L, JND) portée à l'Arc C"
MESSAGE_NON_LISIBLE: str = ("NON LISIBLE : gate d'instrument en violation (amendement "
                            "famille 2) — remonter avant toute lecture d'issue.")


def issue_mappee(gate_statut: str, verdict_global: str) -> str:
    """Mapping MÉCANIQUE des trois issues pré-écrites (clause gravée, brief
    volet 2 -- verbatim) sur le verdict global §A3 (+ clause capacité) du
    bras v2 :
      - gate VIOLATION -> non lisible, AUCUNE issue n'est mappée (même
        mécanique que `cellule_A0`, famille 1) ;
      - PASS -> cellule 1 (court-circuit manche 2 sur ce substrat) ;
      - FAIL -> cellule 2 (FAIL-mur, le ledger passe d'hypothèse à obligation) ;
      - INDETERMINE_CAPACITE -> NON-DÉMONTRÉ (règle de dernière famille,
        dépense suivante = manche 2) ;
      - sinon (INDETERMINE générique) -> global INDÉTERMINÉ, surface portée
        à l'Arc C (lecture pré-écrite clause 4)."""
    if gate_statut != "CONFORME":
        return MESSAGE_NON_LISIBLE
    if verdict_global == "PASS":
        return MESSAGE_ISSUE_PASS
    if verdict_global == "FAIL":
        return MESSAGE_ISSUE_FAIL
    if verdict_global == "INDETERMINE_CAPACITE":
        return MESSAGE_ISSUE_CAPACITE
    return MESSAGE_ISSUE_INDETERMINE


# --- Figures ------------------------------------------------------------------


def _y_categoriel_budgets() -> tuple[list[float], float, float]:
    """Positions y (log10) des 7 budgets, position catégorielle de ∞ (un pas
    de décade au-dessus du dernier budget, PAS à l'échelle -- même convention
    que famille 1), et y de la ligne de coupure."""
    y_budgets = [math.log10(b) for b in BUDGETS]
    pas_decade = y_budgets[-1] - y_budgets[-2]
    y_inf = y_budgets[-1] + pas_decade
    y_coupure = y_budgets[-1] + 0.35 * pas_decade
    return y_budgets, y_inf, y_coupure


def _position_y(k: float, y_inf: float) -> float:
    return y_inf if not math.isfinite(k) else math.log10(k)


def figure_kstar_vs_L_qt(k_table_v2: dict, path: Path) -> None:
    """`arcA_kstar_vs_L_qt.png` : équivalent exact de `figure_kstar_vs_L`
    (famille 1), axe y catégoriel sur les 7 budgets + ∞ (au lieu de 3
    tailles) ; mêmes conventions (dispersion des seeds, ligne de cap,
    ligne de coupure ∞)."""
    y_budgets, y_inf, y_coupure = _y_categoriel_budgets()
    y_cap = math.log10(CAP_FLOATS)

    x_pos = {L: math.log2(L) for L in L_LIST}
    jitter = {jnd: (i - 1.5) * 0.11 for i, jnd in enumerate(JND_LIST)}

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    for jnd in JND_LIST:
        couleur = _COULEURS_JND[jnd]
        xs_med, ys_med = [], []
        for L in L_LIST:
            cell = k_table_v2[L][jnd]
            xs_med.append(x_pos[L])
            ys_med.append(_position_y(cell["mediane"], y_inf))
            for k_seed in cell["par_seed"].values():
                ax.scatter(x_pos[L] + jitter[jnd], _position_y(k_seed, y_inf),
                          color=couleur, alpha=0.35, s=18, zorder=2,
                          edgecolors="none")
        ax.plot(xs_med, ys_med, "-o", color=couleur, linewidth=1.8, markersize=6,
               label=f"JND={jnd * 100:.0f} %", zorder=3)

    ax.axhline(y_cap, color="#555555", linestyle="--", linewidth=1.2,
              label=f"cap anti-trivialité = {CAP_FLOATS:.1f} floats (10 % de 4096)")
    ax.axhline(y_coupure, color="#999999", linestyle=":", linewidth=1.0)

    ax.set_xticks([x_pos[L] for L in L_LIST])
    ax.set_xticklabels([str(L) for L in L_LIST])
    ax.set_yticks(y_budgets + [y_inf])
    ax.set_yticklabels([str(b) for b in BUDGETS] + ["∞"])
    ax.set_xlabel("L (longueur d'histoire, épisodes)")
    ax.set_ylabel("k*(L) -- budget du résumé (floats-éq)")
    ax.set_title("Arc A -- k*(L) bras v2, famille 2 quadtree "
                "(médiane inter-seeds, dispersion des seeds)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def figure_jnd_sensitivity_qt(k80_par_bras: dict[str, dict[float, float]], path: Path) -> None:
    """`arcA_jnd_sensitivity_qt.png` : équivalent exact de
    `figure_jnd_sensitivity` (famille 1) -- k*(80) médian vs JND, les 3 bras,
    lignes de référence pré-enregistrées aux attendus famille 2 (ferm=32 au
    lieu de 81 ; shuf=∞ inchangé)."""
    y_budgets, y_inf, y_coupure = _y_categoriel_budgets()
    y_ferm = math.log10(ATTENDU_FERM_QT)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    x = [jnd * 100.0 for jnd in JND_LIST]
    for arm in ARMS:
        ys = [_position_y(k80_par_bras[arm][jnd], y_inf) for jnd in JND_LIST]
        ax.plot(x, ys, "-o", color=_COULEURS_BRAS[arm], linewidth=1.8, markersize=6,
               label=f"bras {arm}")

    ax.axhline(y_ferm, color=_COULEURS_BRAS["ferm"], linestyle="--", linewidth=1.0,
              alpha=0.7, label=f"attendu ferm = {ATTENDU_FERM_QT:.0f} (amendement famille 2)")
    ax.axhline(y_inf, color=_COULEURS_BRAS["shuf"], linestyle="--", linewidth=1.0,
              alpha=0.7, label="attendu shuf = ∞ (inchangé)")
    ax.axhline(y_coupure, color="#999999", linestyle=":", linewidth=1.0)

    ax.set_yticks(y_budgets + [y_inf])
    ax.set_yticklabels([str(b) for b in BUDGETS] + ["∞"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{jnd * 100:.0f} %" for jnd in JND_LIST])
    ax.set_xlabel("JND (%)")
    ax.set_ylabel("k*(L=80) -- budget du résumé (floats-éq)")
    ax.set_title("Arc A -- sensibilité JND de k*(L=80), famille 2 quadtree "
                "(3 bras : mesure + contrôles)")
    ax.legend(fontsize=8, loc="center right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def figure_kstar_surface_qt(k_table_v2: dict, path: Path) -> None:
    """`arcA_kstar_surface_qt.png` : surface k*(L, JND) (bras v2, médianes)
    rapportée TELLE QUELLE en heatmap catégoriel 4L x 4JND -- chaque cellule
    annotée du budget médian (ou "∞"), couleur = index catégoriel dans
    BUDGETS+∞ (colormap séquentielle) ; cellules dont le budget dépasse le
    cap anti-trivialité (409.6) marquées "(>cap)". Vue COMPLÉMENTAIRE des
    courbes k*(L) par JND (`figure_kstar_vs_L_qt`) -- même donnée, lecture en
    grille plutôt qu'en courbes."""
    ordre = list(BUDGETS) + [float("inf")]
    n = len(ordre)
    grille = np.zeros((len(L_LIST), len(JND_LIST)))
    for iL, L in enumerate(L_LIST):
        for iJ, jnd in enumerate(JND_LIST):
            k = k_table_v2[L][jnd]["mediane"]
            grille[iL, iJ] = ordre.index(k) if math.isfinite(k) else n - 1

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(grille, cmap="viridis", vmin=0, vmax=n - 1, aspect="auto",
                   origin="upper")
    for iL, L in enumerate(L_LIST):
        for iJ, jnd in enumerate(JND_LIST):
            k = k_table_v2[L][jnd]["mediane"]
            etiquette = "∞" if not math.isfinite(k) else str(int(k))
            if math.isfinite(k) and k > CAP_FLOATS:
                etiquette += "\n(>cap)"
            couleur_texte = "black" if grille[iL, iJ] >= (n - 1) * 0.6 else "white"
            ax.text(iJ, iL, etiquette, ha="center", va="center",
                   color=couleur_texte, fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(JND_LIST)))
    ax.set_xticklabels([f"{jnd * 100:.0f} %" for jnd in JND_LIST])
    ax.set_yticks(range(len(L_LIST)))
    ax.set_yticklabels([str(L) for L in L_LIST])
    ax.set_xlabel("JND (%)")
    ax.set_ylabel("L (longueur d'histoire, épisodes)")
    ax.set_title("Arc A -- surface k*(L, JND) bras v2, famille 2 quadtree (médianes)")
    cbar = fig.colorbar(im, ax=ax, ticks=range(n))
    cbar.ax.set_yticklabels([str(b) for b in BUDGETS] + ["∞"])
    cbar.set_label("k*(L, JND) -- budget (floats-éq)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --- Main ----------------------------------------------------------------------


def main() -> None:
    t0 = time.time()
    if not MEASURES_PATH.exists():
        raise RuntimeError(
            f"{MEASURES_PATH} introuvable -- run_arcA_measure_qt.py --assemble "
            "doit être exécuté avant ce script.")
    _sanity_prefixe_tirage()  # gratuite, importée de run_arcA_measure

    with np.load(MEASURES_PATH, allow_pickle=False) as d:
        meta = json.loads(str(d["meta_json"]))
        data = {k: d[k] for k in d.files if k != "meta_json"}

    l_par_bras = {arm: tuple(meta["L_par_bras"][arm]) for arm in ARMS}
    k_table: dict[str, dict] = {
        arm: calculer_k_star_arm_qt(data[f"sous_jnd_{arm}"], data[f"ma1_{arm}"],
                                    data[f"ma2_{arm}"], l_par_bras[arm])
        for arm in ARMS
    }

    # --- 1. GATE DES CONTRÔLES (amendement famille 2) -- EN PREMIER -----------
    gate_statut, gate_violations = gate_controles_qt(
        data["sous_jnd_ferm"], data["ma1_ferm"], data["ma2_ferm"],
        data["sous_jnd_shuf"], data["ma1_shuf"], data["ma2_shuf"])
    n_cells_par_bras = {"ferm": len(L_LIST_CONTROL) * len(SEEDS) * len(JND_LIST),
                        "shuf": len(L_LIST_CONTROL) * len(SEEDS) * len(JND_LIST)}
    n_violations_par_bras = {
        arm: sum(1 for v in gate_violations if v["bras"] == arm) for arm in ("ferm", "shuf")
    }

    print("=" * 78)
    print("ARC A / FAMILLE 2 (QUADTREE) -- GATE DES CONTRÔLES -- EN PREMIER")
    print("=" * 78)
    print(f"Attendus : ferm -> k*={ATTENDU_FERM_QT:.0f} partout ; shuf -> k*=∞ partout "
         f"(L∈{L_LIST_CONTROL}, seeds={SEEDS}, JND={[f'{j:.2f}' for j in JND_LIST]}).")
    for arm in ("ferm", "shuf"):
        print(f"-- bras {arm} : {n_violations_par_bras[arm]}/{n_cells_par_bras[arm]} "
             "cellules en violation --")
        for v in gate_violations:
            if v["bras"] != arm:
                continue
            print(f"  VIOLATION bras={v['bras']} L={v['L']:3d} seed={v['seed']} "
                 f"jnd={v['jnd']:.2f} : k*_obtenu={v['k_star_obtenu']} "
                 f"(attendu {v['k_star_attendu']})")
    print("-" * 78)
    print(f"gate_controles = {gate_statut}")
    if gate_statut == "VIOLATION":
        print("STOP : gate d'instrument en VIOLATION -- remonter. Le verdict §A3 du "
             "bras v2 est calculé et stocké pour audit (clé `verdict_A3_v2_audit` de "
             "verdict_qt.json), mais NON LISIBLE comme verdict de la manche.")
    print("=" * 78)

    # --- 2. k*(L) bras v2, pente + IC, verdict §A3 (calculé/stocké TOUJOURS) --
    print("ARC A / FAMILLE 2 -- k*(L) bras v2 (mesure) -- médiane inter-seeds + clause 2xJND")
    for L in L_LIST:
        for jnd in JND_LIST:
            cell = k_table["v2"][L][jnd]
            par_seed_fmt = {s: v for s, v in cell["par_seed"].items()}
            print(f"  L={L:3d} jnd={jnd:.2f} : k*={cell['mediane']}  "
                 f"par_seed={par_seed_fmt}")

    pentes_v2: dict[float, dict] = {}
    verdicts_par_jnd: dict[float, str] = {}
    for jnd in JND_LIST:
        k40 = k_table["v2"][L_SLOPE[0]][jnd]["mediane"]
        k80 = k_table["v2"][L_SLOPE[1]][jnd]["mediane"]
        res = pente_bootstrap_qt(data["sous_jnd_v2"], data["ma1_v2"], data["ma2_v2"],
                                 jnd, k40, k80)
        pentes_v2[jnd] = res
        k_par_L = {L: k_table["v2"][L][jnd]["mediane"] for L in L_LIST}
        verdicts_par_jnd[jnd] = verdict_A3_jnd_avec_capacite(
            k_par_L, k40, k80, res["ic95"][0], res["ic95"][1])

    verdict_global = verdict_A3_global_avec_capacite(verdicts_par_jnd)
    messages_par_jnd = {jnd: MESSAGE_INDETERMINE_CAPACITE
                        for jnd, v in verdicts_par_jnd.items()
                        if v == "INDETERMINE_CAPACITE"}
    message_global = (MESSAGE_INDETERMINE_CAPACITE
                      if verdict_global == "INDETERMINE_CAPACITE" else None)
    issue = issue_mappee(gate_statut, verdict_global)

    print("-" * 78)
    print(f"Pente k*(L) sur L∈{L_SLOPE} (bras v2) + IC95% bootstrap "
         f"({N_BOOTSTRAP} tirages, seed={SEED_BOOTSTRAP}, fraîche par JND) :")
    for jnd in JND_LIST:
        r = pentes_v2[jnd]
        print(f"  jnd={jnd:.2f} : k*(40)={r['k_star_L40']}  k*(80)={r['k_star_L80']}  "
             f"pente={r['pente']}  IC95=[{r['ic95'][0]}, {r['ic95'][1]}]  "
             f"(bootstrap indéfinis={r['n_bootstrap_indefinis']}/{N_BOOTSTRAP})")

    print("-" * 78)
    if gate_statut == "CONFORME":
        print("VERDICT DE LA MANCHE (§A3 + clause capacité, bras v2, mécanique) :")
        for jnd in JND_LIST:
            suffixe = (f"  [{MESSAGE_INDETERMINE_CAPACITE}]"
                       if jnd in messages_par_jnd else "")
            print(f"  jnd={jnd:.2f} : {verdicts_par_jnd[jnd]}{suffixe}")
        print(f"  GLOBAL (stable sur toute la plage JND ssi identique ci-dessus ; "
             f"clause capacité si 4/4 INDETERMINE_CAPACITE) : {verdict_global}")
        if message_global is not None:
            print(f"  [{message_global}]")
    else:
        print("verdict_v2_scelle : NON LISIBLE : gate d'instrument en violation "
             "(amendement famille 2) — remonter.")
    print(f"ISSUE MAPPÉE (clause gravée, brief volet 2) : {issue}")
    print("=" * 78)

    # --- 3. Surface k*(L, JND) (bras v2, médianes) -- rapportée TELLE QUELLE --
    surface_kstar_v2 = {str(L): {str(jnd): k_table["v2"][L][jnd]["mediane"]
                                 for jnd in JND_LIST}
                        for L in L_LIST}
    print("SURFACE k*(L, JND) -- bras v2, médianes (4L x 4JND) :")
    header = "  L\\JND  " + "  ".join(f"{jnd * 100:5.0f}%" for jnd in JND_LIST)
    print(header)
    for L in L_LIST:
        ligne = "  ".join(
            f"{k_table['v2'][L][jnd]['mediane']:7.1f}" if math.isfinite(k_table['v2'][L][jnd]['mediane'])
            else "    inf" for jnd in JND_LIST)
        print(f"  L={L:3d}   {ligne}")
    print("=" * 78)

    # --- 4. Sérialisation verdict_qt.json --------------------------------------
    k80_par_bras = {arm: {jnd: k_table[arm][L_SLOPE[1]][jnd]["mediane"] for jnd in JND_LIST}
                   for arm in ARMS}

    verdict_v2_audit = dict(par_jnd=verdicts_par_jnd, global_=verdict_global,
                            messages_capacite_par_jnd=messages_par_jnd,
                            message_capacite_global=message_global,
                            issue_mappee=issue)
    if gate_statut == "CONFORME":
        verdict_v2_scelle = dict(lisible=True, par_jnd=verdicts_par_jnd,
                                 global_=verdict_global,
                                 messages_capacite_par_jnd=messages_par_jnd,
                                 message_capacite_global=message_global,
                                 issue_mappee=issue)
    else:
        verdict_v2_scelle = dict(lisible=False, message=MESSAGE_NON_LISIBLE,
                                 issue_mappee=issue)

    # --- 5. Figures -------------------------------------------------------------
    figure_kstar_vs_L_qt(k_table["v2"], FIG_KSTAR_PATH)
    figure_jnd_sensitivity_qt(k80_par_bras, FIG_SENSITIVITY_PATH)
    figure_kstar_surface_qt(k_table["v2"], FIG_SURFACE_PATH)

    elapsed = time.time() - t0

    resultat = dict(
        meta=dict(
            source_measures=str(MEASURES_PATH.relative_to(ROOT)),
            L=list(L_LIST), L_par_bras=l_par_bras, seeds=list(SEEDS),
            budgets=list(BUDGETS), jnd=list(JND_LIST),
            cap_floats_10pct=CAP_FLOATS, budget_ferm=BUDGET_FERM,
            attendu_ferm_qt=ATTENDU_FERM_QT, attendu_shuf_qt="inf",
            L_slope=list(L_SLOPE), n_bootstrap=N_BOOTSTRAP,
            seed_bootstrap=SEED_BOOTSTRAP,
            convention_serialisation="float infini -> chaîne 'inf'/'-inf' ; NaN -> 'nan'",
            elapsed_seconds=elapsed,
        ),
        k_star=k_table,
        surface_kstar_v2=surface_kstar_v2,
        gate_controles=dict(statut=gate_statut, violations=gate_violations,
                            n_cellules_par_bras=n_cells_par_bras,
                            n_violations_par_bras=n_violations_par_bras),
        pentes_v2=pentes_v2,
        verdict_A3_v2_audit=verdict_v2_audit,
        verdict_v2_scelle=verdict_v2_scelle,
        issue_mappee=issue,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VERDICT_PATH.write_text(json.dumps(_to_jsonable(resultat), indent=2, ensure_ascii=False))
    print(f"[REPORT] -> {VERDICT_PATH}")
    print(f"[REPORT] -> {FIG_KSTAR_PATH}")
    print(f"[REPORT] -> {FIG_SENSITIVITY_PATH}")
    print(f"[REPORT] -> {FIG_SURFACE_PATH}")
    print(f"Temps de calcul total : {elapsed:.1f}s")


if __name__ == "__main__":
    main()
