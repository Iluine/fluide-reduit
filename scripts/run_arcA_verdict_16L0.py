"""Arc A / extension 16L₀ — Task 3 : VERDICT §A12 (gate → surface k*(L) aux
pins → pente moitié-haute {40,80,160} → prononcé mécanique), reproduit
verbatim dans `.superpowers/sdd/16L0-task3-brief.md`. Contrat :
`PREREGISTRATION.md` (pocCascade2phys) §A12, commit `3fd72aa` — appliqué
MÉCANIQUEMENT. Aucun seuil déplacé, aucune lecture « à la main » : le
verdict SORT du calcul (combinateur §A12, fonction pure, testée sur 3 jeux
synthétiques — une entrée par branche).

**Agrafage des deux couches** (aucune n'est mutée) :
`outputs/arcA/measures_qt.npz` (manche 1, GELÉ, L∈{10,20,40,80}) +
`outputs/arcA/measures_160.npz` (Task 2, L=160 bras v2 ; L∈{10,160} bras
ferm/shuf) → `L_LIST_FULL = (10,20,40,80,160)`, `ma1_full`/`ma2_full` shape
(5,5,7) = 4 lignes de `measures_qt.npz` + 1 ligne (L=160) de
`measures_160.npz` (`agrafe_L_axis`). `L_LIST` (`scripts.run_arcA_measure`,
(10,20,40,80)) reste INTACT — importé, jamais muté ni réécrit.

**Réutilisation OBLIGATOIRE (import, zéro réimplémentation)** :
`k_star_seed_qt`/`k_star_groupe_qt`/`CAP_FLOATS`/`N_BOOTSTRAP`/
`SEED_BOOTSTRAP`/`ATTENDU_FERM_QT`/`ATTENDU_SHUF_QT`/
`BUDGETS_NON_DISCRIMINANTS`/`_non_discriminant` (`scripts.run_arcA_verdict_qt`,
famille 2 quadtree, GELÉ) ; `sous_jnd_a_jnd`/`charge_pins`/`REGIMES`/`ROLES`
(`scripts.run_arcC_surface_kstar_pins`) ; `JND_LIST`/`L_LIST`/`SEEDS`
(`scripts.run_arcA_measure`) ; `BUDGETS` (`scripts.run_arcA_measure_qt`) ;
`L_V2`/`L_CONTROL`/`MEASURES_160_PATH`/`_load_history_160`
(`scripts.run_arcA_measure_16L0`, Task 2) ; `_to_jsonable`
(`scripts.run_arcA_verdict`).

**Gate §A11 reconduit à L∈{10,160}** (`gate_controles_16L0`) : ÉCRIT ICI en
boucle LOCALE (pas un appel direct à `gate_controles_qt`) — choix documenté
au rapport Task 3 : `gate_controles_qt` indexe ses arrays ferm/shuf via
`L_LIST.index(L)` (elles sont alignées sur les 4 lignes de `L_LIST`,
NaN-paddées aux L absents du bras) ; les arrays ferm/shuf de
`measures_160.npz`, elles, sont alignées DIRECTEMENT sur `L_CONTROL=(10,160)`
(2 lignes, PAS de padding) — `L_LIST.index(160)` lèverait `ValueError`. La
boucle ci-dessous itère donc `L_CONTROL` par POSITION (`enumerate`), mais
appelle LITTÉRALEMENT `k_star_groupe_qt` (n_seed=1, même fonction que le
gate famille 2) et réutilise les MÊMES constantes de comparaison
(`ATTENDU_FERM_QT` égalité exacte, `CAP_FLOATS` pour shuf, amendement de
portée `fa54d20`) — zéro réimplémentation de la RÈGLE de comparaison,
seule l'indexation d'axe diffère.

**Pente moitié-haute {40,80,160}** : OLS (moindres carrés) de k*(L) vs L —
généralisation fidèle de la pente 2-points `(k*(80)-k*(40))/40` de §A3
(dégénère EXACTEMENT dessus à 2 points, cf. `test_pente_ols_3pts`). Méthode
NOMMÉE par le contrôleur AVANT la mesure (ledger `.superpowers/sdd/
progress.md`, entrée « DÉCISION MÉTHODE Task3 », 2026-07-05) — cf.
`METHODE_PENTE_16L0`. Bootstrap IDENTIQUE à §A3 : `default_rng(
SEED_BOOTSTRAP)` fraîche par cellule JND, `N_BOOTSTRAP` tirages, MÊMES
indices seeds appliqués aux trois L, percentiles [2.5, 97.5] ; tout k* non
fini (∞ d'un côté) rend la pente du tirage NaN (exclue des percentiles,
comptée dans `n_bootstrap_indefinis`) — même mécanique que
`pente_bootstrap_qt`, seule la formule de pente change (2 points → OLS
3 points).

**Verdict §A12** (`verdict_A12_combinateur`) : fonction PURE, lue au pin
sévère IC ENTIER (3 colonnes ic_bas/pin/ic_haut). Priorité stricte :
CELLULE_1_PRONONCEE (les 3 colonnes « fermées ») > FAIL_MUR (colonne pin
centrale saut + pente positive hors IC) > INDETERMINE_JND (défaut — couvre
la divergence entre colonnes nommée par §A12 ET tout résidu qui n'est ni
l'une ni l'autre). Si le gate est en VIOLATION, le combinateur N'EST PAS
APPELÉ (`construit_verdict_complet` court-circuite AVANT — instrument cassé
= verdict non lisible, §A12).

Usage (run réel — nécessite `measures_160.npz`, produit par Task 2 en
arrière-plan) :
  .venv/bin/python scripts/run_arcA_verdict_16L0.py

Sorties : `outputs/arcA/verdict_16L0.json`, `outputs/arcA/arcA_kstar_16L0.png`,
`outputs/arcA/arcA_kstar_16L0_pirecas.png` (SECONDAIRE — PNG multi-panneaux
au lieu d'un GIF animé : choix YAGNI documenté, ne bloque pas le verdict,
cf. `figure_pire_cas_16L0`). Déterministe (bootstrap seedé) — un double run
produit un JSON identique (`test_determinisme`).

Point d'arrêt final : ce script REMONTE le verdict, il n'enchaîne NI sur la
manche 2 NI sur une 3ᵉ famille (§A12, clause gravée)."""
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

from scripts.run_arcA_measure import JND_LIST, L_LIST, SEEDS, _center_rollout
from scripts.run_arcA_measure_16L0 import L_CONTROL as L_CONTROL_160
from scripts.run_arcA_measure_16L0 import L_V2 as L_V2_160
from scripts.run_arcA_measure_16L0 import MEASURES_160_PATH, _load_history_160
from scripts.run_arcA_measure_qt import BUDGETS, _field_true_qt, _max_carrier
from scripts.run_arcA_measure_qt import MEASURES_PATH as MEASURES_QT_PATH
from scripts.run_arcA_revalidate import B0, PARAMS, S_HALF_OP
from scripts.run_arcA_verdict import _to_jsonable
from scripts.run_arcA_verdict_qt import (ATTENDU_FERM_QT, ATTENDU_SHUF_QT,
                                         BUDGETS_NON_DISCRIMINANTS, CAP_FLOATS,
                                         N_BOOTSTRAP, SEED_BOOTSTRAP,
                                         _non_discriminant, k_star_groupe_qt,
                                         k_star_seed_qt)
from scripts.run_arcC_surface_kstar_pins import (PINS_PATH, REGIMES, ROLES,
                                                 charge_pins, sous_jnd_a_jnd)
from src.albedo import albedo
from src.sediment import run_episode_trajectoire
from src.summary_quadtree import regenerate_qt, summarize_qt

OUT_DIR = ROOT / "outputs" / "arcA"
VERDICT_16L0_PATH = OUT_DIR / "verdict_16L0.json"
FIG_KSTAR_16L0_PATH = OUT_DIR / "arcA_kstar_16L0.png"
# PNG (pas GIF) : substitution YAGNI documentée (brief, secondaire, ne bloque
# pas le verdict) -- cf. `figure_pire_cas_16L0`.
FIG_PIRECAS_16L0_PATH = OUT_DIR / "arcA_kstar_16L0_pirecas.png"

# --- Grille agrafée (L_LIST importé INTACT, jamais muté) --------------------
L_LIST_FULL: tuple[int, ...] = tuple(L_LIST) + (160,)
L_SLOPE_16L0: tuple[int, int, int] = (40, 80, 160)  # moitié haute, §A12

PORTEE_16L0: str = "substrat v2, pin sévère n=1 Romain, famille qt"
MESSAGE_NON_LISIBLE_16L0: str = (
    "NON LISIBLE : gate d'instrument §A11 en violation à L∈{10,160} -- le "
    "combinateur §A12 n'est PAS évalué (instrument cassé = verdict non lisible).")
METHODE_PENTE_16L0: str = (
    "pente OLS (moindres carrés) de k*(L) vs L sur L∈{40,80,160} -- généralisation "
    "fidèle de la pente 2-points (k*(80)-k*(40))/40 de §A3 (dégénère EXACTEMENT "
    "dessus à 2 points, cf. test_pente_ols_3pts) ; méthode NOMMÉE par le contrôleur "
    "AVANT la mesure (ledger .superpowers/sdd/progress.md, entrée « DÉCISION "
    "MÉTHODE Task3 », 2026-07-05).")


# --- Chargement des deux couches (lecture seule) -----------------------------


def charge_measures_qt() -> dict[str, np.ndarray]:
    """Charge `measures_qt.npz` (manche 1, GELÉ) -- lecture seule, jamais
    réécrit."""
    with np.load(MEASURES_QT_PATH, allow_pickle=False) as d:
        return {k: np.asarray(d[k]) for k in d.files if k != "meta_json"}


def charge_measures_160() -> dict[str, np.ndarray]:
    """Charge `measures_160.npz` (Task 2) -- lecture seule. Lève RuntimeError
    clair si absent (rien n'est fabriqué : BLOCKED, le contrôleur doit avoir
    lancé Task 2 en arrière-plan avant ce verdict)."""
    if not MEASURES_160_PATH.exists():
        raise RuntimeError(
            f"{MEASURES_160_PATH} introuvable -- scripts/run_arcA_measure_16L0.py "
            "(Task 2) doit avoir tourné avant ce verdict. Rien n'est fabriqué : BLOCKED.")
    with np.load(MEASURES_160_PATH, allow_pickle=False) as d:
        return {k: np.asarray(d[k]) for k in d.files if k != "meta_json"}


# --- Agrafage de l'axe L (lecture pure, aucune mutation des entrées) --------


def agrafe_L_axis(ma_qt: np.ndarray, ma_160: np.ndarray) -> np.ndarray:
    """Concatène l'axe L (axe 0) : `ma_qt` (4,5,7)[L∈L_LIST] + `ma_160`
    (1,5,7)[L∈L_V2_160=(160,)] -> (5,5,7)[L∈L_LIST_FULL]. `measures_qt.npz`
    et `L_LIST` restent INTACTS -- cette fonction lit, ne mute rien."""
    if ma_qt.shape[0] != len(L_LIST):
        raise ValueError(f"ma_qt doit avoir {len(L_LIST)} lignes (L_LIST) ; shape={ma_qt.shape}")
    if ma_160.shape[0] != len(L_V2_160):
        raise ValueError(
            f"ma_160 doit avoir {len(L_V2_160)} ligne(s) (L_V2_160) ; shape={ma_160.shape}")
    return np.concatenate([ma_qt, ma_160], axis=0)


# --- Gate des contrôles §A11 reconduit à L∈{10,160} (cf. docstring module) --


def gate_controles_16L0(sous_jnd_ferm: np.ndarray, ma1_ferm: np.ndarray, ma2_ferm: np.ndarray,
                        sous_jnd_shuf: np.ndarray, ma1_shuf: np.ndarray, ma2_shuf: np.ndarray
                        ) -> tuple[str, list[dict]]:
    """Gate §A11 reconduit à L∈{10,160} (`measures_160.npz`) -- PAR CELLULE
    (bras, L, seed, JND), `k_star_groupe_qt` appelé avec un groupe réduit à
    n_seed=1 (même fonction, mêmes budgets croissants, même clause 2×JND que
    le bras v2 -- même preuve de no-op mécanique que `gate_controles_qt`).
    Itère `L_CONTROL_160` PAR POSITION (les arrays de `measures_160.npz` sont
    alignées directement dessus, PAS sur `L_LIST` -- cf. docstring module
    pour la justification de ce choix vs un appel direct à
    `gate_controles_qt`). Attendus INCHANGÉS (amendement famille 2, portée
    `fa54d20`) : ferm == `ATTENDU_FERM_QT` (32.0, égalité exacte) ; shuf
    conforme ssi k* > `CAP_FLOATS` (∞ SOUS LE CAP). Retourne (statut,
    violations) ; statut = "CONFORME" ssi zéro violation sur les 80 cellules
    (2 bras x 2 L x 5 seeds x 4 JND)."""
    violations: list[dict] = []
    specs = (
        ("ferm", sous_jnd_ferm, ma1_ferm, ma2_ferm),
        ("shuf", sous_jnd_shuf, ma1_shuf, ma2_shuf),
    )
    for arm_name, sous_jnd_arm, ma1_arm, ma2_arm in specs:
        for iL, L in enumerate(L_CONTROL_160):
            for iJ, jnd in enumerate(JND_LIST):
                for iS, seed in enumerate(SEEDS):
                    sous_jnd_grp = sous_jnd_arm[iL, iS:iS + 1, :, iJ]
                    ma1_grp = ma1_arm[iL, iS:iS + 1, :]
                    ma2_grp = ma2_arm[iL, iS:iS + 1, :]
                    k, _audit = k_star_groupe_qt(sous_jnd_grp, ma1_grp, ma2_grp, jnd)
                    if arm_name == "ferm":
                        conforme = (k == ATTENDU_FERM_QT)
                        attendu = ATTENDU_FERM_QT
                    else:
                        conforme = (k > CAP_FLOATS)
                        attendu = ATTENDU_SHUF_QT
                    if not conforme:
                        violations.append(dict(bras=arm_name, L=int(L), seed=int(seed),
                                               jnd=float(jnd), k_star_obtenu=k,
                                               k_star_attendu=attendu))
    statut = "CONFORME" if not violations else "VIOLATION"
    return statut, violations


# --- Surface k*(L, JND) sur une grille L arbitraire -------------------------


def surface_par_L(ma1: np.ndarray, ma2: np.ndarray, L_list: tuple[int, ...],
                  jnd: float) -> dict[int, dict]:
    """k*(L) pour un JND donné, sur `L_list` -- généralise `kstar_par_L`
    (`scripts.run_arcC_surface_kstar_pins`, câblée en dur sur `L_LIST` à 4
    éléments) à une grille ARBITRAIRE (ici `L_LIST_FULL` à 5 éléments, mais
    aussi testable directement sur `L_LIST` à 4 éléments pour la
    non-régression C-1a, SANS agrafage ni `measures_160.npz`). RÉUTILISE
    (jamais réimplémente) `k_star_seed_qt`/`k_star_groupe_qt`/
    `_non_discriminant`/`CAP_FLOATS`/`sous_jnd_a_jnd`."""
    sj = sous_jnd_a_jnd(ma1, ma2, jnd)
    out: dict[int, dict] = {}
    for iL, L in enumerate(L_list):
        par_seed = {seed: k_star_seed_qt(sj[iL, iS]) for iS, seed in enumerate(SEEDS)}
        k_med, audit = k_star_groupe_qt(sj[iL], ma1[iL], ma2[iL], jnd)
        out[L] = dict(
            mediane=k_med, par_seed=par_seed,
            non_discriminant=_non_discriminant(k_med),
            sup_cap=bool(math.isfinite(k_med) and k_med > CAP_FLOATS),
            audit=audit,
        )
    return out


def toutes_jnd_union(pins: dict[str, dict[str, float]]) -> list[float]:
    """JND_GRILLE ∪ JND_PINS : la grille {2,3,4,5} % (`JND_LIST`) suivie des
    6 JND réels des pins (sévère puis laxiste, ic_bas/pin/ic_haut) --
    dédoublonnée (aucune coïncidence connue entre grille et pins)."""
    vues: list[float] = list(JND_LIST)
    vues_set = set(vues)
    for regime in REGIMES:
        for role in ROLES:
            v = pins[regime][role]
            if v not in vues_set:
                vues.append(v)
                vues_set.add(v)
    return vues


def construit_surface_full(ma1_full: np.ndarray, ma2_full: np.ndarray,
                           jnds: list[float]) -> dict[float, dict]:
    """Surface complète {jnd: {"k_star_par_L": {L: {...}}}} sur `L_LIST_FULL`,
    pour chaque JND de `jnds` (grille ∪ 6 pins)."""
    return {jnd: dict(k_star_par_L=surface_par_L(ma1_full, ma2_full, L_LIST_FULL, jnd))
           for jnd in jnds}


# --- Pente OLS (généralisation n-points de la pente 2-points §A3) -----------


def pente_ols(Ls: tuple[float, ...], ks: tuple[float, ...]) -> float:
    """Pente OLS (moindres carrés) de k*(L) vs L -- généralisation fidèle à
    n points de la pente 2-points `(k*(80)-k*(40))/40` de §A3 : dégénère
    EXACTEMENT dessus à n=2 (identité algébrique, cf. `test_pente_ols_3pts`).
    Convention non-fini (§A12/Task3, cf. docstring module) : si UN SEUL des
    `ks` est non fini (∞), la pente est NaN -- jamais ±∞."""
    ks_arr = np.asarray(ks, dtype=np.float64)
    if not np.all(np.isfinite(ks_arr)):
        return float("nan")
    Ls_arr = np.asarray(Ls, dtype=np.float64)
    Lbar, kbar = Ls_arr.mean(), ks_arr.mean()
    num = float(np.sum((Ls_arr - Lbar) * (ks_arr - kbar)))
    den = float(np.sum((Ls_arr - Lbar) ** 2))
    return num / den


def pente_bootstrap_16L0(ma1_full: np.ndarray, ma2_full: np.ndarray, jnd: float,
                         k40_pt: float, k80_pt: float, k160_pt: float) -> dict:
    """Généralisation OLS 3-points de `pente_bootstrap_qt` : pente ponctuelle
    (`pente_ols`, déjà calculée à partir des médianes L=40/80/160 passées en
    argument) + IC 95 % bootstrap inter-seeds -- MÊME mécanique
    (`N_BOOTSTRAP` tirages, `default_rng(SEED_BOOTSTRAP)` fraîche, MÊMES
    indices seeds appliqués aux trois L, percentiles [2.5, 97.5], NaN
    exclue+comptée) que `pente_bootstrap_qt` (l.319-341) -- seule la formule
    de pente change (2 points -> OLS 3 points, `pente_ols`). `sous_jnd`
    recalculé à la volée via `sous_jnd_a_jnd` (le JND peut être un pin réel,
    hors grille `JND_LIST`, donc pas de tableau `sous_jnd` précalculé à
    réutiliser ici -- MÊME approche que `surface_par_L`)."""
    idx_L = [L_LIST_FULL.index(L) for L in L_SLOPE_16L0]
    n_seed = ma1_full.shape[1]
    rng = np.random.default_rng(SEED_BOOTSTRAP)

    def _k_pour(iL: int, idx_seeds: np.ndarray) -> float:
        ma1_g = ma1_full[iL, idx_seeds, :]
        ma2_g = ma2_full[iL, idx_seeds, :]
        sj_g = sous_jnd_a_jnd(ma1_g, ma2_g, jnd)
        k, _audit = k_star_groupe_qt(sj_g, ma1_g, ma2_g, jnd)
        return k

    pente_pt = pente_ols(L_SLOPE_16L0, (k40_pt, k80_pt, k160_pt))

    pentes = np.empty(N_BOOTSTRAP, dtype=np.float64)
    for b in range(N_BOOTSTRAP):
        idx_tirage = rng.integers(0, n_seed, size=n_seed)
        ks_b = tuple(_k_pour(iL, idx_tirage) for iL in idx_L)
        pentes[b] = pente_ols(L_SLOPE_16L0, ks_b)

    n_nan = int(np.isnan(pentes).sum())
    valides = pentes[~np.isnan(pentes)]
    if valides.size == 0:
        ic_lo, ic_hi = float("nan"), float("nan")
    else:
        ic_lo, ic_hi = (float(v) for v in np.percentile(valides, [2.5, 97.5]))

    return dict(k_star_L40=k40_pt, k_star_L80=k80_pt, k_star_L160=k160_pt,
               pente=pente_pt, ic95=[ic_lo, ic_hi], n_bootstrap=N_BOOTSTRAP,
               n_bootstrap_indefinis=n_nan, seed_bootstrap=SEED_BOOTSTRAP,
               methode_pente=METHODE_PENTE_16L0)


# --- Colonne de lecture (k80, k160, plat, saut, cap, IC) --------------------


def construit_colonne(k80: float, k160: float, ic95: list[float]) -> dict:
    """Les 5 booléens dérivés d'une colonne de lecture §A12 (rôle ic_bas /
    pin / ic_haut) : `plat`/`saut` comparent k*(160) à k*(80) ; `k160_sous_cap`
    compare k*(160) au cap importé ; `ic_contient_0`/`ic_positif` lisent l'IC
    bootstrap de la pente. Comparaisons Python natives : ∞==∞ -> True (plat),
    NaN <= 0 -> False (jamais de branche spéciale nécessaire, cf. IEEE-754)."""
    ic_lo, ic_hi = ic95
    return dict(
        k80=k80, k160=k160,
        plat=bool(k160 == k80),
        saut=bool(k160 > k80),
        k160_sous_cap=bool(k160 <= CAP_FLOATS),
        pente_ic=[ic_lo, ic_hi],
        ic_contient_0=bool(ic_lo <= 0.0 <= ic_hi),
        ic_positif=bool(ic_lo > 0.0),
    )


# --- Combinateur §A12 (fonction PURE -- cœur anti-fabrication) --------------


def verdict_A12_combinateur(colonnes: dict[str, dict]) -> dict:
    """Prononcé mécanique §A12, fonction PURE : `colonnes` =
    {"ic_bas": {...}, "pin": {...}, "ic_haut": {...}}, chacune produite par
    `construit_colonne`. Priorité STRICTE, exactement 3 branches (verbatim
    §A12-verdicts) :
      1. CELLULE_1_PRONONCEE ssi les TROIS colonnes sont « fermées »
         (plat ET ic_contient_0 ET k160_sous_cap).
      2. Sinon FAIL_MUR ssi la colonne "pin" (centrale) est saut ET
         ic_positif (pente positive hors IC).
      3. Sinon INDETERMINE_JND (défaut -- couvre la divergence nommée par
         §A12 entre colonnes ET tout résidu qui n'est ni une fermeture nette
         ni un mur net).
    Applique les deux gardes §A12 (traçabilité, pas de sur-lecture) dans
    `gardes` : échelle 3-valeurs du budget médian, et le soupçon pré-nommé
    (ic_bas monte pendant que le pin central reste plat)."""
    def _fermee(col: dict) -> bool:
        return bool(col["plat"] and col["ic_contient_0"] and col["k160_sous_cap"])

    fermees = {role: _fermee(colonnes[role]) for role in ROLES}
    col_centrale = colonnes["pin"]
    fail_mur_central = bool(col_centrale["saut"] and col_centrale["ic_positif"])
    diverge = len({fermees[role] for role in ROLES}) > 1

    if all(fermees.values()):
        prononce = "CELLULE_1_PRONONCEE"
    elif fail_mur_central:
        prononce = "FAIL_MUR"
    else:
        prononce = "INDETERMINE_JND"

    ic_bas_monte = bool(colonnes["ic_bas"]["saut"])
    pin_reste_plat = bool(colonnes["pin"]["plat"])
    gardes = dict(
        echelle_3_valeurs=(
            "« plat » vit sur l'échelle 3-valeurs {128,256,400} du budget médian -- "
            "la grille fine de budgets rend le saut lisible s'il existe ; ne pas "
            "sur-lire la platitude comme fermeture forte, c'est la MOITIÉ HAUTE "
            "{40,80,160} qui porte le verdict, pas les petits L."),
        ic_bas_arme=dict(
            ic_bas_monte=ic_bas_monte, pin_reste_plat=pin_reste_plat,
            soupcon_confirme=bool(ic_bas_monte and pin_reste_plat),
            note=("soupçon pré-nommé §C12/§A12 : si ic_bas monte pendant que le pin "
                 "central reste plat, l'œil est armé mais le verdict reste mécanique "
                 "(-> INDETERMINE_JND dans ce cas via la priorité ci-dessus, jamais "
                 "lu à la main)."),
        ),
        colonnes_fermees=fermees,
        diverge_entre_colonnes=diverge,
        fail_mur_colonne_centrale=fail_mur_central,
    )
    return dict(prononce=prononce, gardes=gardes)


def construit_verdict_complet(gate_statut: str,
                              colonnes_severe: dict[str, dict] | None) -> dict:
    """Assemble `verdict_16L0` : si le gate n'est PAS CONFORME, le
    combinateur §A12 N'EST PAS APPELÉ (instrument cassé = verdict non
    lisible, §A12) -- `colonnes_severe` peut alors être `None` sans jamais
    être déréférencé. Sinon, appelle `verdict_A12_combinateur` et rapporte
    prononce + colonnes (par_colonne) + gardes + portée gravée."""
    if gate_statut != "CONFORME":
        return dict(prononce="NON_LISIBLE", par_colonne=None, gardes=None,
                   portee=PORTEE_16L0, message=MESSAGE_NON_LISIBLE_16L0)
    combi = verdict_A12_combinateur(colonnes_severe)
    return dict(prononce=combi["prononce"], par_colonne=colonnes_severe,
               gardes=combi["gardes"], portee=PORTEE_16L0)


# --- Assemblage complet (pur, séparé de main() pour le test de déterminisme) -


def construit_resultat_complet() -> dict:
    """Assemble le JSON complet `verdict_16L0` depuis les npz gelés
    (`measures_qt.npz` + `measures_160.npz`) et les pins réels
    (`pins_spatial.json`). Fonction séparée de `main()` (pas d'I/O de sortie,
    pas d'impression) -- appelée deux fois par `test_determinisme`."""
    qt = charge_measures_qt()
    m160 = charge_measures_160()

    ma1_full = agrafe_L_axis(qt["ma1_v2"], m160["ma1_v2"])
    ma2_full = agrafe_L_axis(qt["ma2_v2"], m160["ma2_v2"])

    gate_statut, gate_violations = gate_controles_16L0(
        m160["sous_jnd_ferm"], m160["ma1_ferm"], m160["ma2_ferm"],
        m160["sous_jnd_shuf"], m160["ma1_shuf"], m160["ma2_shuf"])

    pins = charge_pins(PINS_PATH)
    jnds = toutes_jnd_union(pins)
    surface_kstar = construit_surface_full(ma1_full, ma2_full, jnds)

    pentes: dict[float, dict] = {}
    for jnd in jnds:
        k_par_L = surface_kstar[jnd]["k_star_par_L"]
        k40, k80, k160 = (k_par_L[L]["mediane"] for L in L_SLOPE_16L0)
        pentes[jnd] = pente_bootstrap_16L0(ma1_full, ma2_full, jnd, k40, k80, k160)

    jnd_severe = pins["severe"]
    colonnes_severe = {
        role: construit_colonne(
            surface_kstar[jnd_severe[role]]["k_star_par_L"][80]["mediane"],
            surface_kstar[jnd_severe[role]]["k_star_par_L"][160]["mediane"],
            pentes[jnd_severe[role]]["ic95"],
        )
        for role in ROLES
    }
    verdict_16L0 = construit_verdict_complet(gate_statut, colonnes_severe)

    meta = dict(
        source_measures_qt=str(MEASURES_QT_PATH.relative_to(ROOT)),
        source_measures_160=str(MEASURES_160_PATH.relative_to(ROOT)),
        source_pins=str(PINS_PATH.relative_to(ROOT)),
        contrat="PREREGISTRATION.md (pocCascade2phys) §A12, commit 3fd72aa",
        L=list(L_LIST_FULL), seeds=list(SEEDS), budgets=list(BUDGETS),
        jnd_grille=list(JND_LIST), pins_utilises=pins,
        cap_floats=CAP_FLOATS, budgets_non_discriminants=list(BUDGETS_NON_DISCRIMINANTS),
        L_slope=list(L_SLOPE_16L0), n_bootstrap=N_BOOTSTRAP, seed_bootstrap=SEED_BOOTSTRAP,
        methode_pente=METHODE_PENTE_16L0,
        convention_serialisation="float infini -> chaîne 'inf' ; NaN -> 'nan'",
    )

    return dict(
        meta=meta,
        gate_controles=dict(statut=gate_statut, violations=gate_violations),
        surface_kstar=surface_kstar,
        pentes=pentes,
        verdict_16L0=verdict_16L0,
    )


# --- Figures ------------------------------------------------------------------


def figure_kstar_16L0(resultat: dict, path: Path) -> None:
    """`arcA_kstar_16L0.png` : (gauche) k*(L) sur `L_LIST_FULL` aux 3 JND du
    pin sévère (conventions catégorielles identiques à `figure_kstar_vs_L_qt`
    famille 2) ; (droite) heatmap de la surface k*(L,JND) sur la grille union
    (grille + 6 pins), mêmes conventions que `figure_kstar_surface_qt` --
    repère visuel du plateau/saut à L=160."""
    surface = resultat["surface_kstar"]
    pins_severe = resultat["meta"]["pins_utilises"]["severe"]
    jnds_severe = [pins_severe["ic_bas"], pins_severe["pin"], pins_severe["ic_haut"]]
    labels = ["ic_bas", "pin", "ic_haut"]
    couleurs = ["#0072B2", "#D55E00", "#009E73"]  # Okabe-Ito

    y_budgets = [math.log10(b) for b in BUDGETS]
    pas_decade = y_budgets[-1] - y_budgets[-2]
    y_inf = y_budgets[-1] + pas_decade

    def _y(k: float) -> float:
        return y_inf if not math.isfinite(k) else math.log10(k)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.5))

    x_pos = {L: math.log2(L) for L in L_LIST_FULL}
    for jnd, label, couleur in zip(jnds_severe, labels, couleurs):
        k_par_L = surface[jnd]["k_star_par_L"]
        xs = [x_pos[L] for L in L_LIST_FULL]
        ys = [_y(k_par_L[L]["mediane"]) for L in L_LIST_FULL]
        ax1.plot(xs, ys, "-o", color=couleur, linewidth=1.8, markersize=6,
                 label=f"pin sévère {label}")
    ax1.axhline(math.log10(CAP_FLOATS), color="#555555", linestyle="--", linewidth=1.2,
               label="cap anti-trivialité (CAP_FLOATS)")
    ax1.axvline(math.log2(160), color="#999999", linestyle=":", linewidth=1.2,
               label="extension 16L₀")
    ax1.set_xticks([x_pos[L] for L in L_LIST_FULL])
    ax1.set_xticklabels([str(L) for L in L_LIST_FULL])
    ax1.set_yticks(y_budgets + [y_inf])
    ax1.set_yticklabels([str(b) for b in BUDGETS] + ["∞"])
    ax1.set_xlabel("L (longueur d'histoire, épisodes)")
    ax1.set_ylabel("k*(L) -- budget du résumé (floats-éq)")
    ax1.set_title("k*(L), 3 colonnes du pin sévère")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(True, alpha=0.25)

    jnds_tries = sorted(surface.keys())
    ordre = list(BUDGETS) + [float("inf")]
    grille = np.zeros((len(jnds_tries), len(L_LIST_FULL)))
    for iJ, jnd in enumerate(jnds_tries):
        for iL, L in enumerate(L_LIST_FULL):
            k = surface[jnd]["k_star_par_L"][L]["mediane"]
            grille[iJ, iL] = ordre.index(k) if math.isfinite(k) else len(ordre) - 1

    im = ax2.imshow(grille, cmap="viridis", vmin=0, vmax=len(ordre) - 1, aspect="auto",
                    origin="upper")
    for iJ, jnd in enumerate(jnds_tries):
        for iL, L in enumerate(L_LIST_FULL):
            k = surface[jnd]["k_star_par_L"][L]["mediane"]
            etiquette = "∞" if not math.isfinite(k) else str(int(k))
            couleur_texte = "black" if grille[iJ, iL] >= (len(ordre) - 1) * 0.6 else "white"
            ax2.text(iL, iJ, etiquette, ha="center", va="center", color=couleur_texte,
                     fontsize=7, fontweight="bold")
    ax2.set_xticks(range(len(L_LIST_FULL)))
    ax2.set_xticklabels([str(L) for L in L_LIST_FULL])
    ax2.set_yticks(range(len(jnds_tries)))
    ax2.set_yticklabels([f"{jnd * 100:.2f}%" for jnd in jnds_tries], fontsize=7)
    ax2.set_xlabel("L")
    ax2.set_ylabel("JND (grille ∪ 6 pins)")
    ax2.set_title("surface k*(L, JND) -- union grille+pins")
    fig.colorbar(im, ax=ax2, ticks=range(len(ordre)), label="k*(L,JND) (floats-éq)")

    fig.suptitle("Arc A / extension 16L₀ -- k*(L) et surface (famille 2 quadtree)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _pire_seed_L160(ma2_full: np.ndarray, budget_kstar: float) -> int:
    """Index (pas le seed lui-même) de la seed la plus discriminante à
    L=160, AU BUDGET k* retenu par le pin sévère central -- argmax de M-A2
    parmi les 5 seeds à ce budget précis (pas le max sur tous les budgets :
    le pire cas illustre PRÉCISÉMENT le budget que le verdict retient).
    `ma2_full` : (nL, nSeed, nBudget) aligné sur `L_LIST_FULL`."""
    iL160 = L_LIST_FULL.index(160)
    if math.isfinite(budget_kstar):
        iB = BUDGETS.index(int(budget_kstar))
        colonne = ma2_full[iL160, :, iB]
    else:
        colonne = ma2_full[iL160, :, -1]  # ∞ : le budget max de la grille sert de repère
    return int(np.argmax(colonne))


def figure_pire_cas_16L0(colonnes_severe: dict[str, dict], ma2_full: np.ndarray,
                         path: Path) -> dict:
    """`arcA_kstar_16L0_pirecas.png` -- SECONDAIRE (le verdict n'en dépend
    PAS, cf. docstring module) : PNG multi-panneaux (PAS un GIF animé --
    substitution YAGNI documentée : coûteux/complexe à produire ET à
    vérifier sans donnée réelle disponible au moment de l'écriture de ce
    module). Pire seed à L=160, au budget k* retenu par le pin sévère
    CENTRAL (`_pire_seed_L160`), albedo(vrai) vs albedo(régénéré-qt) à
    quelques snapshots du rollout (`run_episode_trajectoire`/`albedo`,
    réutilisés tels quels -- AUCUNE réimplémentation). Nécessite les 5
    `h160_s{seed}.npz` (Task 1) via `_load_history_160` -- lève RuntimeError
    clair si absents (rien n'est fabriqué)."""
    budget_kstar = colonnes_severe["pin"]["k160"]
    idx_seed = _pire_seed_L160(ma2_full, budget_kstar)
    seed = SEEDS[idx_seed]

    hist = _load_history_160(seed)
    field_true = _field_true_qt("v2", 160, seed, hist["s_L160"])
    budget = int(budget_kstar) if math.isfinite(budget_kstar) else BUDGETS[-1]
    s_regen = regenerate_qt(summarize_qt(field_true, budget))
    center = _center_rollout(seed, 160)

    traj_true = run_episode_trajectoire(field_true, B0, center, PARAMS)
    traj_regen = run_episode_trajectoire(s_regen, B0, center, PARAMS)
    if len(traj_true) != len(traj_regen):
        raise RuntimeError(
            f"pire-cas 16L0 : longueurs de trajectoire différentes -- seed={seed}, "
            f"budget={budget}. Incohérence de rollout, STOP.")

    a_true = [albedo(s, S_HALF_OP) for s in traj_true]
    a_regen = [albedo(s, S_HALF_OP) for s in traj_regen]
    dchi = [_max_carrier(ar, at) for ar, at in zip(a_regen, a_true)]

    n_snap = len(a_true)
    n_panneaux = min(6, n_snap)
    idx_snap = np.unique(np.linspace(0, n_snap - 1, n_panneaux).astype(int))

    fig, axes = plt.subplots(2, len(idx_snap), figsize=(2.6 * len(idx_snap), 5.4))
    if len(idx_snap) == 1:
        axes = axes.reshape(2, 1)
    for j, i in enumerate(idx_snap):
        axes[0, j].imshow(a_true[i], cmap="viridis", vmin=0.0, vmax=1.0, origin="lower")
        axes[1, j].imshow(a_regen[i], cmap="viridis", vmin=0.0, vmax=1.0, origin="lower")
        axes[0, j].set_title(f"t={i}\nΔχ={dchi[i]:.4f}", fontsize=8)
        for row in (0, 1):
            axes[row, j].axis("off")
    fig.text(0.01, 0.72, "vrai", rotation=90, va="center", fontsize=9)
    fig.text(0.01, 0.28, "régénéré-qt", rotation=90, va="center", fontsize=9)
    fig.suptitle(f"Arc A / 16L₀ -- pire cas L=160 seed={seed} budget={budget}")
    fig.tight_layout(rect=(0.01, 0.0, 1.0, 1.0))
    fig.savefig(path, dpi=130)
    plt.close(fig)

    return dict(L=160, seed=seed, budget=budget, centre_rollout=list(center),
               n_panneaux=int(len(idx_snap)), dchi_max=float(max(dchi)))


# --- Main ----------------------------------------------------------------------


def main() -> None:
    t0 = time.time()
    resultat = construit_resultat_complet()

    gate_statut = resultat["gate_controles"]["statut"]
    gate_violations = resultat["gate_controles"]["violations"]
    print("=" * 78)
    print("ARC A / EXTENSION 16L0 -- GATE DES CONTRÔLES (§A11 reconduit, L∈{10,160})")
    print("=" * 78)
    print(f"gate_controles = {gate_statut}  ({len(gate_violations)} violation(s) / 80 cellules)")
    for v in gate_violations:
        print(f"  VIOLATION bras={v['bras']} L={v['L']:3d} seed={v['seed']} "
             f"jnd={v['jnd']:.2f} : k*_obtenu={v['k_star_obtenu']} "
             f"(attendu {v['k_star_attendu']})")
    if gate_statut != "CONFORME":
        print("STOP : gate d'instrument en VIOLATION -- verdict §A12 NON ÉVALUÉ (NON_LISIBLE).")
    print("=" * 78)

    verdict = resultat["verdict_16L0"]
    print(f"VERDICT §A12 (extension 16L0) : {verdict['prononce']}")
    print(f"portée : {verdict['portee']}")
    print("=" * 78)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VERDICT_16L0_PATH.write_text(json.dumps(_to_jsonable(resultat), indent=2, ensure_ascii=False))
    print(f"[REPORT] -> {VERDICT_16L0_PATH}")

    figure_kstar_16L0(resultat, FIG_KSTAR_16L0_PATH)
    print(f"[REPORT] -> {FIG_KSTAR_16L0_PATH}")

    if verdict["prononce"] != "NON_LISIBLE":
        qt = charge_measures_qt()
        m160 = charge_measures_160()
        ma2_full = agrafe_L_axis(qt["ma2_v2"], m160["ma2_v2"])
        try:
            meta_pire = figure_pire_cas_16L0(verdict["par_colonne"], ma2_full,
                                            FIG_PIRECAS_16L0_PATH)
            print(f"[REPORT] -> {FIG_PIRECAS_16L0_PATH} (pire cas : {meta_pire})")
        except Exception as exc:  # SECONDAIRE -- ne bloque JAMAIS le verdict (brief)
            print(f"[REPORT] figure pire-cas NON produite (secondaire, ne bloque pas "
                 f"le verdict §A12 ci-dessus) : {type(exc).__name__}: {exc}")

    elapsed = time.time() - t0
    print(f"Temps de calcul total : {elapsed:.1f}s")


if __name__ == "__main__":
    main()
