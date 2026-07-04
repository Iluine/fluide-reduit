"""Arc C / Task 1 — générateur de stimuli spatiaux `stim(t)` + instrument de
mesure Δχ(t) (R1), qui alimenteront le harnais ABX (Task 2) et les sessions
de mesure (Task 3).

Contrat : addendum §C3 (« C-2 : harnais ABX sur stimuli réels ») de
PREREGISTRATION.md (pocCascade2phys, commits `c494d50` + `5d13bc4`), reproduit
verbatim dans `.superpowers/sdd/arcC-task1-stimuli-brief.md`. Spec :

    Stimuli spatiaux : paires (vrai, dégradé) issues des npz commités de la
    manche 1 -- aucun stimulus synthétique, aucun re-run de simulation. Axe
    de stimulation continu : mélange linéaire `stim(t) = (1-t)*vrai +
    t*régénéré(budget)`, t ∈ [0, 1] ; Δχ(t) est MESURÉ par l'instrument R1
    sur chaque stimulus présenté (pas supposé linéaire).

Résolution d'ambiguïté du contrôleur (brief, appliquée telle quelle) : le
mélange se fait en espace SÉDIMENT, pas en espace albedo -- `vrai` = `s_L`
(champ de sédiment settled du npz d'histoire), `régénéré(budget)` =
`regenerate_qt(summarize_qt(s_L, budget))` (champ de sédiment), `stim(t)` =
combinaison convexe des deux en espace sédiment. Le Δχ mesuré est
`delta_chi(albedo(stim(t), S_HALF_OP), albedo(s_L, S_HALF_OP))["max_carrier"]`
-- référence = albedo du champ VRAI, donc les bandes porteuses (cf.
`src.albedo.delta_chi`) sont fixées par le vrai champ, constantes le long de
t ; Δχ(t=0) = 0.0 exact par construction (stim(0) == s_true bit-exact).

Fonctions PURES numpy, AUCUNE RNG dans la génération (déterminisme strict) --
la seule source d'aléa possible (les 20 champs sources) vient des npz commités
de la manche 1, pas d'un tirage fait ici. JAMAIS de comparaison L2
point-à-point : le seul juge de fidélité est R1 (`delta_chi(...)
["max_carrier"]`, espace perceptuel, cf. discipline épistémique du dépôt).

Instrument et point d'opération IMPORTÉS tels quels (jamais réimplémentés,
cf. brief) : `S_HALF_OP` (= 0.05*RELIEF, gravé manche 1) de
`scripts.run_arcA_revalidate` ; `albedo`/`delta_chi` de `src.albedo` ;
`summarize_qt`/`regenerate_qt` (famille 2, quadtree) de
`src.summary_quadtree`."""
from __future__ import annotations

import numpy as np

from scripts.run_arcA_revalidate import S_HALF_OP
from src.albedo import albedo, delta_chi
from src.summary_quadtree import regenerate_qt, summarize_qt


def regenere_budget(s_true: np.ndarray, budget: int) -> np.ndarray:
    """régénéré(budget) = `regenerate_qt(summarize_qt(s_true, budget))` : champ
    de sédiment reconstruit par le compresseur quadtree famille 2 (manche 1)
    au budget donné (floats-équivalents, cf. `size_floats_qt`). Aucune
    réimplémentation -- délègue entièrement à `src.summary_quadtree`."""
    return regenerate_qt(summarize_qt(s_true, budget))


def melange(s_true: np.ndarray, s_regen: np.ndarray, t: float) -> np.ndarray:
    """Combinaison convexe `stim(t) = (1-t)*s_true + t*s_regen`, en espace
    SÉDIMENT (résolution d'ambiguïté du contrôleur, cf. docstring module) :
    un champ de sédiment est >= 0, et une combinaison convexe de deux champs
    >= 0 reste >= 0 -- borne physique préservée automatiquement, pas besoin de
    clip. `t=0` renvoie `s_true` bit-exact et `t=1` renvoie `s_regen`
    bit-exact (cas particuliers explicites, pour ne dépendre d'aucune
    propriété d'arrondi IEEE-754 de l'arithmétique générale)."""
    s_true = np.asarray(s_true, dtype=np.float64)
    s_regen = np.asarray(s_regen, dtype=np.float64)
    if s_true.shape != s_regen.shape:
        raise ValueError(
            f"melange : shapes incompatibles s_true={s_true.shape} != "
            f"s_regen={s_regen.shape}.")
    if not (0.0 <= t <= 1.0):
        raise ValueError(f"melange : t doit être dans [0, 1] (reçu {t!r}).")
    if t == 0.0:
        return s_true.copy()
    if t == 1.0:
        return s_regen.copy()
    return (1.0 - t) * s_true + t * s_regen


def delta_chi_stim(s_true: np.ndarray, s_stim: np.ndarray) -> float:
    """R1 mesuré sur le stimulus `s_stim` par rapport au champ VRAI, au point
    d'opération `S_HALF_OP` gravé (manche 1) :
    `delta_chi(albedo(s_stim, S_HALF_OP), albedo(s_true, S_HALF_OP))
    ["max_carrier"]` -- les bandes porteuses (cf. `src.albedo.delta_chi`) sont
    celles de la RÉFÉRENCE `s_true`, donc constantes le long d'un axe `t`
    (même `s_true` pour toute la courbe). `delta_chi_stim(s, s) == 0.0` par
    construction (mêmes readouts albedo)."""
    a_ref = albedo(np.asarray(s_true, dtype=np.float64), S_HALF_OP)
    a_stim = albedo(np.asarray(s_stim, dtype=np.float64), S_HALF_OP)
    return float(delta_chi(a_stim, a_ref)["max_carrier"])


def courbe_delta_chi(s_true: np.ndarray, budget: int,
                     ts: np.ndarray | None = None) -> np.ndarray:
    """Δχ(t) MESURÉ (pas supposé) pour chaque t de `ts` : régénère UNE FOIS
    (`régénéré(budget)`), mélange par t (`melange`), puis mesure
    (`delta_chi_stim`) -- une passe R1 par t, aucune interpolation. `ts` par
    défaut : `linspace(0, 1, 11)`.

    Δχ(0) = 0.0 exact (`melange(..., 0)` == `s_true` bit-exact) ; Δχ(1) =
    `delta_chi_stim(s_true, régénéré(budget))` = le M-A1 manche 1 au même
    (source, budget) -- même instrument, même régénérateur.

    La courbe est RAPPORTÉE TELLE QUELLE : AUCUNE hypothèse de monotonie ni de
    linéarité n'est faite ici (c'est une propriété à OBSERVER par l'appelant,
    pas à imposer -- spec §C3 : « Δχ(t) est MESURÉ [...] pas supposé
    linéaire »). Fonction pure, déterministe (aucune RNG)."""
    if ts is None:
        ts = np.linspace(0.0, 1.0, 11)
    ts = np.asarray(ts, dtype=np.float64)
    s_true = np.asarray(s_true, dtype=np.float64)
    s_regen = regenere_budget(s_true, budget)
    return np.array(
        [delta_chi_stim(s_true, melange(s_true, s_regen, float(t))) for t in ts],
        dtype=np.float64)
