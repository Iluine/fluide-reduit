"""Arc A / Task 6 -- k*(L), gate des contrôles, verdict §A3 mécanique
(manche 1, relance sur v2).

Contrat : plan `docs/superpowers/plans/arc-a-manche1.md` Task 6 + section
AMENDEMENTS (amendement (ii), gate des contrôles AVANT lecture du verdict v2),
reproduits verbatim dans `.superpowers/sdd/refond-task6-verdict-brief.md`. Ce
script est le POINT FINAL du pipeline : il lit `outputs/arcA/measures.npz`
(Task 5) exclusivement (sauf pour le GIF, seule physique de cette tâche) et
applique MÉCANIQUEMENT les règles gravées -- il ne décide rien.

Extrait verbatim (plan, Task 6) :

    k*(L, seed ; JND) = plus petite taille (81 -> 273 -> 1041) sous-JND ; ∞ si
    aucune. k*(L ; JND) = médiane des seeds, avec la clause « aucune seed >
    2×JND » (si une seed dépasse 2×JND au niveau retenu par la médiane ->
    k*(L) passe au niveau supérieur ; si aucun niveau ne satisfait -> ∞).
    Pente de k*(L) sur la moitié haute (L ∈ {40, 80}) + IC 95 % bootstrap
    inter-seeds (10 000 tirages).
    Verdict §A3 appliqué MÉCANIQUEMENT, par JND de la plage {2,3,4,5} % : PASS
    ssi IC de pente ∋ 0 ET k*(80) ≤ 409.6 floats (10 % de 4096) ; FAIL ssi
    pente > 0 hors IC sur toute la plage sans plateau ; sinon INDÉTERMINÉ.
    Verdict global retenu ssi stable sur TOUTE la plage JND, sinon INDÉTERMINÉ
    (sensibilité §A3).

Et l'amendement (ii), qui S'APPLIQUE AVANT toute lecture du verdict v2 :

    contrôle-fermable : attendu k* = 81 partout ; contrôle-infermable : attendu
    k* = ∞ partout. Toute cellule de contrôle qui viole son attendu -> STOP,
    remonter (gate d'instrument type G0). Les contrôles ne participent PAS au
    verdict §A3 ; ils conditionnent le droit de le lire.

Ordre d'exécution (mécanique, non négociable) :
  1. GATE DES CONTRÔLES (ferm/shuf), calculé PAR CELLULE (bras, L, seed, JND) --
     MÊME définition de k* que le bras v2 (mêmes niveaux 81->273->1041, même
     clause 2xJND -- la clause y est un no-op mécanique prouvé au docstring de
     `k_star_groupe`, PAS un raccourci pris ici). gate_controles = "CONFORME"
     ssi zéro violation sur les 80 cellules (2 bras x {10,80} x 5 seeds x 4
     JND), sinon "VIOLATION" -- imprimé et sérialisé EN PREMIER.
  2. k*(L) bras v2 (médiane + clause 2xJND), pente L∈{40,80} + IC bootstrap,
     verdict §A3 par JND + global -- CALCULÉS ET STOCKÉS dans tous les cas
     (audit), mais IMPRIMÉS comme verdict de la manche SSI gate_controles =
     CONFORME. Si VIOLATION : la clé `verdict_v2_scelle` porte la mention
     explicite « NON LISIBLE » -- aucun PASS/FAIL n'est imprimé comme verdict
     de la manche (le calcul brut reste dans `verdict_A3_v2_audit`, pour
     mémoire, jamais affiché comme un verdict).
  3. Figures + GIF (pire cellule M-A2 du bras v2, seule physique rejouée ici).

AUCUNE interprétation au-delà de §A5 : ce script calcule, imprime, sérialise --
il ne diagnostique rien, ne recommande rien. L'entrée de journal est écrite par
le contrôleur, pas par ce script.

Usage :
  .venv/bin/python scripts/run_arcA_verdict.py

Sortie : `outputs/arcA/verdict.json` + `outputs/arcA/arcA_kstar_vs_L.png` +
`outputs/arcA/arcA_jnd_sensitivity.png` + `outputs/arcA/arcA_worst_case.gif`.
Déterministe (bootstrap seedé `default_rng(20260704)`, physique du GIF pure/
déterministe comme Task 5) -- un double run produit un verdict.json identique
hors `elapsed_seconds` et hors le GIF (ré-encodage PNG/GIF non garanti
bit-identique par Pillow, la physique sous-jacente l'est).
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
from matplotlib.animation import FuncAnimation, PillowWriter

from scripts.run_arcA_measure import (ARMS, CAP_FLOATS_10PCT, JND_LIST, LEVELS,
                                      L_LIST, L_LIST_CONTROL, SEEDS,
                                      _center_rollout, _field_true,
                                      _load_history, _max_carrier,
                                      _sanity_prefixe_tirage)
from scripts.run_arcA_revalidate import B0, PARAMS, S_HALF_OP
from src.albedo import albedo
from src.sediment import run_episode_trajectoire
from src.summary import regenerate, summarize

OUT_DIR = ROOT / "outputs" / "arcA"
MEASURES_PATH = OUT_DIR / "measures.npz"
VERDICT_PATH = OUT_DIR / "verdict.json"
FIG_KSTAR_PATH = OUT_DIR / "arcA_kstar_vs_L.png"
FIG_SENSITIVITY_PATH = OUT_DIR / "arcA_jnd_sensitivity.png"
GIF_WORST_PATH = OUT_DIR / "arcA_worst_case.gif"

# --- Constantes k*(L) (spec Task 6 + amendement (ii)) ------------------------
TAILLE_PAR_NIVEAU: dict[int, int] = {1: 1041, 2: 273, 3: 81}
NIVEAU_PAR_TAILLE: dict[int, int] = {1041: 1, 273: 2, 81: 3}
NIVEAUX_ORDRE_ASC: tuple[int, ...] = (3, 2, 1)              # tailles 81->273->1041
ORDRE_TAILLES_ASC: tuple[float, ...] = (81.0, 273.0, 1041.0, float("inf"))
CAP_FLOATS: float = CAP_FLOATS_10PCT                        # 409.6, source Task 5

L_SLOPE: tuple[int, int] = (40, 80)                         # moitié haute de L
N_BOOTSTRAP: int = 10_000
SEED_BOOTSTRAP: int = 20260704                              # figée, fraîche PAR JND
                                                             # (reproductible indépendamment
                                                             # par cellule JND, sans dépendre
                                                             # de l'ordre d'itération).

ATTENDU_FERM: float = 81.0            # amendement (ii) : contrôle-fermable
ATTENDU_SHUF: float = float("inf")    # amendement (ii) : contrôle-infermable


# --- k*(L, seed ; JND) et k*(L ; JND) (médiane + clause 2xJND) ---------------


def _idx_niveau(level: int) -> int:
    """Index du niveau `level` dans l'axe LEVELS=(1,2,3) des tableaux mesures.npz."""
    return LEVELS.index(level)


def k_star_seed(sous_jnd_row: np.ndarray) -> float:
    """k*(L, seed ; JND) : plus petite taille dans l'ordre 81 -> 273 -> 1041 dont
    la cellule est sous-JND (`sous_jnd_row`, bool aligné sur LEVELS=(1,2,3)) ;
    ∞ (np.inf) si aucune."""
    for level in NIVEAUX_ORDRE_ASC:
        if bool(sous_jnd_row[_idx_niveau(level)]):
            return float(TAILLE_PAR_NIVEAU[level])
    return float("inf")


def k_star_groupe(sous_jnd_grp: np.ndarray, ma1_grp: np.ndarray, ma2_grp: np.ndarray,
                  jnd: float) -> tuple[float, dict]:
    """k*(L ; JND) = médiane des k*(L, seed ; JND) du groupe, avec la clause
    « aucune seed > 2×JND au niveau retenu par la médiane » (spec Task 6) :
    si UNE seed du groupe a max(M-A1,M-A2) > 2·JND au niveau retenu, on passe
    au niveau supérieur (81->273->1041->∞) et on réitère la vérification ; si
    aucun niveau ne satisfait, k* = ∞.

    `sous_jnd_grp`/`ma1_grp`/`ma2_grp` : shape (n_seed, n_level), alignées sur
    LEVELS=(1,2,3). Fonction générique : utilisée pour le bras v2 (n_seed=5,
    médiane réelle) ET pour le GATE DES CONTRÔLES (n_seed=1 -- la « médiane »
    d'une seule seed est trivialement sa propre valeur, et la clause y est un
    NO-OP MÉCANIQUE PROUVÉ : si k*(seed) = taille T (T fini), alors par
    construction max(M-A1,M-A2) < JND < 2·JND au niveau T pour CETTE seed --
    la seule du groupe -- donc aucune escalade n'est jamais déclenchée. Le
    gate réutilise donc EXACTEMENT la même définition et la même clause que le
    bras v2, sans branche spéciale ; le calcul dégénère proprement.

    Retourne (k*, audit) où `audit` documente la médiane initiale, les
    éventuelles escalades, et le niveau final -- pour `verdict.json`."""
    n_seed = sous_jnd_grp.shape[0]
    k_par_seed = np.array([k_star_seed(sous_jnd_grp[i]) for i in range(n_seed)])
    mediane = float(np.median(k_par_seed))
    audit: dict = dict(k_par_seed=k_par_seed.tolist(), mediane_initiale=mediane,
                       escalades=[])
    if not math.isfinite(mediane):
        audit["k_final"] = float("inf")
        return float("inf"), audit

    idx = ORDRE_TAILLES_ASC.index(mediane)
    while math.isfinite(ORDRE_TAILLES_ASC[idx]):
        taille = ORDRE_TAILLES_ASC[idx]
        level = NIVEAU_PAR_TAILLE[int(taille)]
        lvl_idx = _idx_niveau(level)
        max_vals = np.maximum(ma1_grp[:, lvl_idx], ma2_grp[:, lvl_idx])
        offenders = max_vals > 2.0 * jnd
        if np.any(offenders):
            audit["escalades"].append(dict(niveau_teste=taille,
                                           n_seeds_offenders=int(offenders.sum())))
            idx += 1
            continue
        audit["k_final"] = taille
        return taille, audit

    audit["k_final"] = float("inf")
    return float("inf"), audit


def calculer_k_star_arm(sous_jnd_arm: np.ndarray, ma1_arm: np.ndarray,
                        ma2_arm: np.ndarray, l_list_arm: tuple[int, ...]) -> dict:
    """k* complets d'un bras : par seed ET médiane+clause, pour chaque L de
    `l_list_arm` (grille propre au bras) x chaque JND de JND_LIST. Retourne
    {L: {jnd: {"par_seed": {seed: k}, "mediane": k, "audit": {...}}}}."""
    out: dict = {}
    for L in l_list_arm:
        iL = L_LIST.index(L)
        out[L] = {}
        for jnd in JND_LIST:
            iJ = JND_LIST.index(jnd)
            k_par_seed = {seed: k_star_seed(sous_jnd_arm[iL, iS, :, iJ])
                         for iS, seed in enumerate(SEEDS)}
            k_med, audit = k_star_groupe(sous_jnd_arm[iL, :, :, iJ], ma1_arm[iL],
                                         ma2_arm[iL], jnd)
            out[L][jnd] = dict(par_seed=k_par_seed, mediane=k_med, audit=audit)
    return out


# --- Gate des contrôles (amendement ii) --------------------------------------


def gate_controles(k_table_ferm: dict, k_table_shuf: dict) -> tuple[str, list[dict]]:
    """Gate d'instrument (amendement ii), calculé PAR CELLULE (bras, L, seed,
    JND) -- PAS de médiane : réutilise directement `par_seed` de
    `calculer_k_star_arm` (= `k_star_seed` par cellule, cf. preuve de non-effet
    de la clause 2xJND à n_seed=1 dans `k_star_groupe`). Comparaison par égalité
    EXACTE (pas de tolérance flottante) : toutes les valeurs en jeu sont des
    membres discrets de {81.0, 273.0, 1041.0, inf}, jamais le résultat d'une
    arithmétique -- l'égalité exacte est donc le bon test.

    Retourne (statut, violations) ; statut = "CONFORME" ssi zéro violation sur
    les 80 cellules (2 bras x {10,80} x 5 seeds x 4 JND), sinon "VIOLATION"."""
    violations: list[dict] = []
    specs = (("ferm", k_table_ferm, ATTENDU_FERM), ("shuf", k_table_shuf, ATTENDU_SHUF))
    for arm_name, table, attendu in specs:
        for L in L_LIST_CONTROL:
            for jnd in JND_LIST:
                for seed in SEEDS:
                    k = table[L][jnd]["par_seed"][seed]
                    conforme = (k == attendu)
                    if not conforme:
                        violations.append(dict(bras=arm_name, L=int(L), seed=int(seed),
                                               jnd=float(jnd), k_star_obtenu=k,
                                               k_star_attendu=attendu))
    statut = "CONFORME" if not violations else "VIOLATION"
    return statut, violations


# --- Pente k*(L) sur L∈{40,80} + IC 95% bootstrap inter-seeds (bras v2) ------


def pente_bootstrap(sous_jnd_v2: np.ndarray, ma1_v2: np.ndarray, ma2_v2: np.ndarray,
                    jnd: float, k40_pt: float, k80_pt: float) -> dict:
    """Pente ponctuelle (k*(80)-k*(40))/40 (déjà calculée en amont, passée en
    argument -- pas recalculée ici, source unique = `calculer_k_star_arm`) +
    IC 95 % par bootstrap inter-seeds : 10 000 tirages, resample des 5 seeds
    AVEC remise -- MÊME jeu d'indices tiré pour L=40 et L=80 à chaque tirage
    (préserve l'appariement seed<->seed entre les deux longueurs d'histoire),
    recalcul médiane+clause+pente PAR TIRAGE, percentiles 2.5/97.5.
    `default_rng(20260704)` figée, FRAÎCHE pour cette cellule JND (chaque JND
    est ainsi indépendamment reproductible sans dépendre de l'ordre d'itération
    des 4 JND).

    Cas ∞ : si un tirage a k*(40) ou k*(80) = ∞ des DEUX côtés à la fois, la
    pente de ce tirage est indéfinie (∞-∞) -- exclue du calcul des percentiles
    (`n_bootstrap_indefinis` compte ces tirages, reporté pour audit ; PAS
    d'arithmétique inventée sur l'infini). Un seul côté infini donne une pente
    ±∞, valeur réelle étendue valide, CONSERVÉE dans le calcul des percentiles."""
    iL40, iL80 = L_LIST.index(L_SLOPE[0]), L_LIST.index(L_SLOPE[1])
    jnd_idx = JND_LIST.index(jnd)
    n_seed = sous_jnd_v2.shape[1]
    rng = np.random.default_rng(SEED_BOOTSTRAP)

    def _k_pour(iL: int, idx_seeds: np.ndarray) -> float:
        k, _audit = k_star_groupe(sous_jnd_v2[iL, idx_seeds, :, jnd_idx],
                                  ma1_v2[iL, idx_seeds, :], ma2_v2[iL, idx_seeds, :], jnd)
        return k

    pente_pt = ((k80_pt - k40_pt) / 40.0
               if math.isfinite(k40_pt) and math.isfinite(k80_pt) else float("nan"))

    pentes = np.empty(N_BOOTSTRAP, dtype=np.float64)
    for b in range(N_BOOTSTRAP):
        idx_tirage = rng.integers(0, n_seed, size=n_seed)
        k40_b = _k_pour(iL40, idx_tirage)
        k80_b = _k_pour(iL80, idx_tirage)
        pentes[b] = (k80_b - k40_b) / 40.0   # inf/-inf/nan propagés tels quels

    n_nan = int(np.isnan(pentes).sum())
    valides = pentes[~np.isnan(pentes)]
    if valides.size == 0:
        ic_lo, ic_hi = float("nan"), float("nan")
    else:
        ic_lo, ic_hi = (float(v) for v in np.percentile(valides, [2.5, 97.5]))

    return dict(k_star_L40=k40_pt, k_star_L80=k80_pt, pente=pente_pt,
               ic95=[ic_lo, ic_hi], n_bootstrap=N_BOOTSTRAP,
               n_bootstrap_indefinis=n_nan, seed_bootstrap=SEED_BOOTSTRAP)


# --- Verdict §A3 (mécanique) --------------------------------------------------


def verdict_A3_jnd(k40: float, k80: float, ic_lo: float, ic_hi: float) -> str:
    """Verdict §A3 pour UNE cellule JND (spec Task 6, lecture mécanique) :
    - Cas ∞ (PRIORITAIRE) : si k*(40) ou k*(80) est infini, OU si l'IC n'est
      pas défini (bootstrap entièrement indéfini), la pente est indéfinie ->
      ni PASS ni FAIL ne peut être établi -> INDÉTERMINÉ (lecture mécanique du
      « sinon » du plan, aucune arithmétique inventée sur l'infini).
    - PASS ssi l'IC de pente contient 0 ET k*(80) <= CAP_FLOATS (409.6).
    - FAIL ssi pente > 0 ET 0 hors IC (pente positive, hors du bruit
      inter-seeds -- pas de plateau possible dès lors que la pente est
      strictement positive sur les 2 points L∈{40,80}).
    - INDÉTERMINÉ sinon (ex. IC ∋ 0 mais k*(80) > cap ; ou IC hors 0 sans
      pente > 0)."""
    if not (math.isfinite(k40) and math.isfinite(k80)):
        return "INDETERMINE"
    if not (math.isfinite(ic_lo) and math.isfinite(ic_hi)):
        return "INDETERMINE"
    ic_contient_zero = ic_lo <= 0.0 <= ic_hi
    if ic_contient_zero and k80 <= CAP_FLOATS:
        return "PASS"
    pente = (k80 - k40) / 40.0
    if pente > 0.0 and not ic_contient_zero:
        return "FAIL"
    return "INDETERMINE"


def verdict_A3_global(verdicts_par_jnd: dict[float, str]) -> str:
    """Verdict global §A3 : retenu SSI stable (identique) sur TOUTE la plage
    JND {2,3,4,5} %, sinon INDÉTERMINÉ (sensibilité §A3, obligatoire)."""
    valeurs = set(verdicts_par_jnd.values())
    if len(valeurs) == 1:
        return next(iter(valeurs))
    return "INDETERMINE"


def cellule_A0(gate_statut: str, verdict_global: str) -> str:
    """Citation MÉCANIQUE de la cellule §A0 sélectionnée par le verdict de la
    manche (grille pré-interprétée §A0, PREREGISTRATION.md pocCascade2phys) --
    aucune interprétation ajoutée au-delà de la grille déjà gravée."""
    if gate_statut != "CONFORME":
        return ("NON DÉTERMINÉE : verdict v2 non lisible (gate d'instrument en "
               "violation, amendement ii) -- remonter avant toute lecture §A0.")
    if verdict_global == "PASS":
        return ("Cellule 1 (état-complet PASS) -- fermeture dans z : architecture "
               "invariants-régénérables ; manche 2 court-circuitée par implication, "
               "sur CE substrat uniquement (§A0).")
    if verdict_global == "FAIL":
        return ("Cellule 2-ou-3 (à départager par la manche 2, §A0) -- un FAIL de "
               "la manche 1 est NON-EXISTENTIEL, pré-étiqueté « il ne tue rien ».")
    return ("INDÉTERMINÉ : §A0 non tranché -- options pré-écrites §A3 (étendre à "
           "16*L0 une seule fois, ou porter l'incertitude et ouvrir la manche 2), "
           "décision du contrôleur, hors du périmètre de ce script.")


# --- Sérialisation JSON (convention inf/nan -> chaîne, documentée ICI) -------


def _to_jsonable(obj):
    """Convertit récursivement une structure Python/NumPy en JSON-safe. Convention
    (documentée ICI, appliquée UNE SEULE fois, à la sérialisation) : un float
    infini devient la chaîne "inf"/"-inf", un NaN devient "nan" -- JSON natif ne
    porte pas ces valeurs ; aucune perte d'information (le calcul interne reste
    en float IEEE-754 partout, seule la sérialisation est affectée)."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        xf = float(obj)
        if math.isinf(xf):
            return "inf" if xf > 0 else "-inf"
        if math.isnan(xf):
            return "nan"
        return xf
    return obj


# --- Figures ------------------------------------------------------------------

_COULEURS_JND: dict[float, str] = {0.02: "#0072B2", 0.03: "#D55E00",
                                   0.04: "#009E73", 0.05: "#CC79A7"}  # Okabe-Ito
_COULEURS_BRAS: dict[str, str] = {"v2": "#0072B2", "ferm": "#009E73", "shuf": "#D55E00"}


def _y_inf(y_1041: float, pas_decade: float) -> float:
    """Position catégorielle de « ∞ » : un niveau au-dessus de 1041, PAS à
    l'échelle (cf. spec figure), séparé par un pas comparable aux paliers
    81->273->1041 en échelle log10."""
    return y_1041 + pas_decade


def _position_y(k: float, y_inf: float) -> float:
    return y_inf if not math.isfinite(k) else math.log10(k)


def figure_kstar_vs_L(k_table_v2: dict, path: Path) -> None:
    """`arcA_kstar_vs_L.png` : k* médian vs L, une courbe par JND, points
    individuels des seeds en dispersion (jitter horizontal) ; axe y catégoriel
    {81, 273, 1041, ∞} (échelle log10 pour les tailles finies, ∞ = niveau
    catégoriel au-dessus de 1041, séparé par une ligne de coupure) ; ligne
    horizontale au cap 409.6 (10 % de 4096)."""
    y_81, y_273, y_1041 = math.log10(81.0), math.log10(273.0), math.log10(1041.0)
    pas_decade = y_1041 - y_273
    y_inf = _y_inf(y_1041, pas_decade)
    y_coupure = y_1041 + 0.35 * pas_decade
    y_cap = math.log10(CAP_FLOATS)

    x_pos = {L: math.log2(L) for L in L_LIST}
    jitter = {jnd: (i - 1.5) * 0.11 for i, jnd in enumerate(JND_LIST)}

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
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
    ax.set_yticks([y_81, y_273, y_1041, y_inf])
    ax.set_yticklabels(["81", "273", "1041", "∞"])
    ax.set_xlabel("L (longueur d'histoire, épisodes)")
    ax.set_ylabel("k*(L) -- taille du résumé (floats)")
    ax.set_title("Arc A -- k*(L) bras v2 (médiane inter-seeds, dispersion des seeds)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def figure_jnd_sensitivity(k80_par_bras: dict[str, dict[float, float]], path: Path) -> None:
    """`arcA_jnd_sensitivity.png` : k*(80) médian vs JND, les 3 bras (v2/ferm/
    shuf) -- même axe y catégoriel que `figure_kstar_vs_L`, + lignes de
    référence pointillées aux attendus pré-enregistrés (ferm=81, shuf=∞) pour
    lire l'écart d'un coup d'œil."""
    y_81, y_273, y_1041 = math.log10(81.0), math.log10(273.0), math.log10(1041.0)
    pas_decade = y_1041 - y_273
    y_inf = _y_inf(y_1041, pas_decade)
    y_coupure = y_1041 + 0.35 * pas_decade

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    x = [jnd * 100.0 for jnd in JND_LIST]
    for arm in ARMS:
        ys = [_position_y(k80_par_bras[arm][jnd], y_inf) for jnd in JND_LIST]
        ax.plot(x, ys, "-o", color=_COULEURS_BRAS[arm], linewidth=1.8, markersize=6,
               label=f"bras {arm}")

    ax.axhline(y_81, color=_COULEURS_BRAS["ferm"], linestyle="--", linewidth=1.0,
              alpha=0.7, label="attendu ferm = 81 (amendement ii)")
    ax.axhline(y_inf, color=_COULEURS_BRAS["shuf"], linestyle="--", linewidth=1.0,
              alpha=0.7, label="attendu shuf = ∞ (amendement ii)")
    ax.axhline(y_coupure, color="#999999", linestyle=":", linewidth=1.0)

    ax.set_yticks([y_81, y_273, y_1041, y_inf])
    ax.set_yticklabels(["81", "273", "1041", "∞"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{jnd * 100:.0f} %" for jnd in JND_LIST])
    ax.set_xlabel("JND (%)")
    ax.set_ylabel("k*(L=80) -- taille du résumé (floats)")
    ax.set_title("Arc A -- sensibilité JND de k*(L=80), 3 bras (mesure + contrôles)")
    ax.legend(fontsize=8, loc="center right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --- GIF pire cellule M-A2 (bras v2) -- seule physique de cette tâche --------


def _pire_cellule_v2(ma2_v2: np.ndarray) -> tuple[int, int, int]:
    """Argmax de ma2_v2 (nL, nSeed, nLevel) -> (L, seed, level) de la pire
    cellule M-A2 du bras v2 (cf. spec figure)."""
    iL, iS, iLv = np.unravel_index(int(np.argmax(ma2_v2)), ma2_v2.shape)
    return int(L_LIST[iL]), int(SEEDS[iS]), int(LEVELS[iLv])


def generer_gif_pire_cas(ma2_v2: np.ndarray, path: Path) -> dict:
    """Rejoue UNIQUEMENT la pire cellule M-A2 du bras v2 (max de ma2_v2) --
    2 rollouts (`run_episode_trajectoire`), même protocole que Task 5 : centre
    = épisode L+1, motif de tirage verbatim de `run_history` (réutilisé via
    `scripts.run_arcA_measure._center_rollout`, PAS réimplémenté). Frames
    côte-à-côte albedo(vrai) / albedo(régénéré) au fil des snapshots
    (sous-échantillonnées à ~30 frames), titre avec Δχ du snapshot (max_carrier,
    même instrument que Task 5). Retourne les métadonnées de la cellule + une
    vérification bit-à-bit contre `ma2_v2` stocké (même physique, recalculée)."""
    L, seed, level = _pire_cellule_v2(ma2_v2)
    hist = _load_history(seed)
    field_true = _field_true("v2", L, seed, hist[f"s_L{L}"])
    s_regen = regenerate(summarize(field_true, level))
    center = _center_rollout(seed, L)

    traj_true = run_episode_trajectoire(field_true, B0, center, PARAMS)
    traj_regen = run_episode_trajectoire(s_regen, B0, center, PARAMS)
    if len(traj_true) != len(traj_regen):
        raise RuntimeError(
            f"GIF pire cas : longueurs de trajectoire différentes (vrai="
            f"{len(traj_true)}, regen={len(traj_regen)}) -- L={L}, seed={seed}, "
            f"level={level}. Incohérence de rollout, STOP.")

    a_true = [albedo(s, S_HALF_OP) for s in traj_true]
    a_regen = [albedo(s, S_HALF_OP) for s in traj_regen]
    dchi = [_max_carrier(ar, at) for ar, at in zip(a_regen, a_true)]

    iL, iS, iLv = L_LIST.index(L), SEEDS.index(seed), LEVELS.index(level)
    valeur_stockee = float(ma2_v2[iL, iS, iLv])
    valeur_recalculee = float(max(dchi))
    ecart_rel = (abs(valeur_recalculee - valeur_stockee) / valeur_stockee
                if valeur_stockee != 0.0 else abs(valeur_recalculee - valeur_stockee))
    if ecart_rel > 1e-9:
        raise RuntimeError(
            f"GIF pire cas : M-A2 recalculé ({valeur_recalculee}) diverge de "
            f"measures.npz ({valeur_stockee}) -- écart relatif {ecart_rel} > 1e-9. "
            "Physique non bit-identique à Task 5, STOP.")

    n_snap = len(a_true)
    n_frames = min(30, n_snap)
    idx = np.unique(np.linspace(0, n_snap - 1, n_frames).astype(int))
    frames_true = np.stack([a_true[i] for i in idx])
    frames_regen = np.stack([a_regen[i] for i in idx])
    dchi_sub = [dchi[i] for i in idx]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.4))
    im_true = axes[0].imshow(frames_true[0], cmap="viridis", vmin=0.0, vmax=1.0,
                             origin="lower")
    im_regen = axes[1].imshow(frames_regen[0], cmap="viridis", vmin=0.0, vmax=1.0,
                              origin="lower")
    axes[0].set_title("albedo(vrai)")
    axes[1].set_title("albedo(régénéré)")
    for ax in axes:
        ax.axis("off")
    fig.colorbar(im_regen, ax=list(axes), shrink=0.75)
    titre = fig.suptitle("")

    def update(t: int):
        im_true.set_data(frames_true[t])
        im_regen.set_data(frames_regen[t])
        titre.set_text(f"Arc A -- pire cellule M-A2 (v2) : L={L} seed={seed} ℓ={level} -- "
                       f"snapshot {idx[t]}/{n_snap - 1} -- Δχ_max_carrier={dchi_sub[t]:.4f}")
        return (im_true, im_regen, titre)

    anim = FuncAnimation(fig, update, frames=len(idx), blit=False)
    anim.save(path, writer=PillowWriter(fps=4))
    plt.close(fig)

    return dict(L=L, seed=seed, level=level, centre_rollout=list(center),
               ma2_stocke=valeur_stockee, ma2_recalcule=valeur_recalculee,
               n_snapshots=n_snap, n_frames_gif=int(len(idx)))


# --- Main ----------------------------------------------------------------------


def main() -> None:
    t0 = time.time()
    if not MEASURES_PATH.exists():
        raise RuntimeError(
            f"{MEASURES_PATH} introuvable -- Task 5 (scripts/run_arcA_measure.py "
            "--assemble) doit être exécutée avant Task 6.")
    _sanity_prefixe_tirage()  # gratuite, cf. Task 5 -- avant tout calcul

    with np.load(MEASURES_PATH, allow_pickle=False) as d:
        meta = json.loads(str(d["meta_json"]))
        data = {k: d[k] for k in d.files if k != "meta_json"}

    l_par_bras = {arm: tuple(meta["L_par_bras"][arm]) for arm in ARMS}
    k_table: dict[str, dict] = {
        arm: calculer_k_star_arm(data[f"sous_jnd_{arm}"], data[f"ma1_{arm}"],
                                 data[f"ma2_{arm}"], l_par_bras[arm])
        for arm in ARMS
    }

    # --- 1. GATE DES CONTRÔLES (amendement ii) -- EN PREMIER ------------------
    gate_statut, gate_violations = gate_controles(k_table["ferm"], k_table["shuf"])
    n_cells_par_bras = {"ferm": len(L_LIST_CONTROL) * len(SEEDS) * len(JND_LIST),
                        "shuf": len(L_LIST_CONTROL) * len(SEEDS) * len(JND_LIST)}
    n_violations_par_bras = {
        arm: sum(1 for v in gate_violations if v["bras"] == arm) for arm in ("ferm", "shuf")
    }

    print("=" * 78)
    print("ARC A / TASK 6 -- GATE DES CONTRÔLES (amendement ii) -- EN PREMIER")
    print("=" * 78)
    print(f"Attendus : ferm -> k*=81 partout ; shuf -> k*=∞ partout "
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
        print("STOP amendement (ii) : gate d'instrument en VIOLATION -- remonter. Le "
             "verdict §A3 du bras v2 est calculé et stocké pour audit (clé "
             "`verdict_A3_v2_audit` de verdict.json), mais NON LISIBLE comme "
             "verdict de la manche : aucun PASS/FAIL n'est imprimé ci-dessous.")
    print("=" * 78)

    # --- 2. k*(L) bras v2, pente + IC, verdict §A3 (calculé/stocké TOUJOURS) --
    print("ARC A / TASK 6 -- k*(L) bras v2 (mesure) -- médiane inter-seeds + clause 2xJND")
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
        res = pente_bootstrap(data["sous_jnd_v2"], data["ma1_v2"], data["ma2_v2"],
                              jnd, k40, k80)
        pentes_v2[jnd] = res
        verdicts_par_jnd[jnd] = verdict_A3_jnd(k40, k80, res["ic95"][0], res["ic95"][1])

    verdict_global = verdict_A3_global(verdicts_par_jnd)

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
        print("VERDICT DE LA MANCHE (§A3, bras v2, mécanique) :")
        for jnd in JND_LIST:
            print(f"  jnd={jnd:.2f} : {verdicts_par_jnd[jnd]}")
        print(f"  GLOBAL (stable sur toute la plage JND ssi identique ci-dessus) : "
             f"{verdict_global}")
        print(f"§A0 -- cellule sélectionnée : {cellule_A0(gate_statut, verdict_global)}")
    else:
        print("verdict_v2_scelle : NON LISIBLE : gate d'instrument en violation "
             "(amendement ii) — remonter.")
        print(f"§A0 -- cellule sélectionnée : {cellule_A0(gate_statut, verdict_global)}")
    print("=" * 78)

    # --- 3. Sérialisation verdict.json ----------------------------------------
    k80_par_bras = {arm: {jnd: k_table[arm][L_SLOPE[1]][jnd]["mediane"] for jnd in JND_LIST}
                   for arm in ARMS}

    verdict_v2_audit = dict(par_jnd=verdicts_par_jnd, global_=verdict_global)
    if gate_statut == "CONFORME":
        verdict_v2_scelle = dict(lisible=True, par_jnd=verdicts_par_jnd,
                                 global_=verdict_global,
                                 cellule_A0=cellule_A0(gate_statut, verdict_global))
    else:
        verdict_v2_scelle = dict(
            lisible=False,
            message="NON LISIBLE : gate d'instrument en violation (amendement ii) — remonter.",
            cellule_A0=cellule_A0(gate_statut, verdict_global))

    # --- 4. Figures -------------------------------------------------------------
    figure_kstar_vs_L(k_table["v2"], FIG_KSTAR_PATH)
    figure_jnd_sensitivity(k80_par_bras, FIG_SENSITIVITY_PATH)

    # --- 5. GIF pire cas (seule physique de cette tâche) -------------------------
    gif_meta = generer_gif_pire_cas(data["ma2_v2"], GIF_WORST_PATH)
    print(f"[GIF] pire cellule M-A2 (v2) : L={gif_meta['L']} seed={gif_meta['seed']} "
         f"ℓ={gif_meta['level']}  ma2_stocke={gif_meta['ma2_stocke']:.6f}  "
         f"ma2_recalcule={gif_meta['ma2_recalcule']:.6f}  -> {GIF_WORST_PATH}")

    elapsed = time.time() - t0

    resultat = dict(
        meta=dict(
            source_measures=str(MEASURES_PATH.relative_to(ROOT)),
            L=list(L_LIST), L_par_bras=l_par_bras, seeds=list(SEEDS),
            niveaux=list(LEVELS), jnd=list(JND_LIST),
            taille_par_niveau=TAILLE_PAR_NIVEAU, cap_floats_10pct=CAP_FLOATS,
            L_slope=list(L_SLOPE), n_bootstrap=N_BOOTSTRAP,
            seed_bootstrap=SEED_BOOTSTRAP,
            convention_serialisation="float infini -> chaîne 'inf'/'-inf' ; NaN -> 'nan'",
            elapsed_seconds=elapsed,
        ),
        k_star=k_table,
        gate_controles=dict(statut=gate_statut, violations=gate_violations,
                            n_cellules_par_bras=n_cells_par_bras,
                            n_violations_par_bras=n_violations_par_bras),
        pentes_v2=pentes_v2,
        verdict_A3_v2_audit=verdict_v2_audit,
        verdict_v2_scelle=verdict_v2_scelle,
        gif_pire_cas=gif_meta,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VERDICT_PATH.write_text(json.dumps(_to_jsonable(resultat), indent=2, ensure_ascii=False))
    print(f"[REPORT] -> {VERDICT_PATH}")
    print(f"[REPORT] -> {FIG_KSTAR_PATH}")
    print(f"[REPORT] -> {FIG_SENSITIVITY_PATH}")
    print(f"[REPORT] -> {GIF_WORST_PATH}")
    print(f"Temps de calcul total : {elapsed:.1f}s")


if __name__ == "__main__":
    main()
