"""Arc A / Task 1 — Substrat sédiment persistant (dépôt/érosion type Exner) sur b0 FIGÉ.

Enveloppe `simulate_wetdry_o2` (src/solver_wetdry.py, NON modifié — pas de nouveau
paramètre, limiteur `minmod` et `dry_eps` par défaut du module). Un « épisode » =
un pulse d'eau gaussien lâché sur le terrain effectif `b_eff = b0 + s` (s GELÉ
pendant tout l'épisode) -> relaxation -> loi de dépôt Exner intégrée séquentiellement
sur les snapshots sous-échantillonnés -> assèchement (h, hu, hv redeviennent 0 ;
seul `s` persiste, mis à jour entre épisodes : b_eff = b0 + s_nouveau).

Indexation array[y, x] (axe 0 = y, axe 1 = x) comme le reste du dépôt.

Note d'implémentation — N_settle vs t_end (lire avant de modifier `_relax_episode`) :
`simulate_wetdry_o2` n'expose que `t_end` (temps physique) ; le pas interne `dt` est
CFL-adaptatif et n'est PAS pilotable en nombre de pas. Pour obtenir exactement
`N_settle` pas solveur avec le VRAI dt interne (pas un dt forcé artificiellement
petit, ce qui reviendrait à changer la physique), `_relax_episode` appelle le
solveur une seule fois avec un budget de temps physique généreux (`_T_END_RELAX`,
choisi empiriquement — cf. commentaire sur la constante — pour que N_settle=600 pas
soient TOUJOURS atteints, quel que soit le centre du pulse dans la boîte
[0.15, 0.85]²), puis tronque la trajectoire retournée aux `N_settle` premiers pas
acceptés. Le reliquat (pas au-delà de N_settle, calculé mais inutilisé) est ignoré :
c'est le prix de piloter un solveur t_end-only par un budget de pas sans y toucher.

Constat empirique important (propriété du solveur figé, PAS un bug de ce module —
cf. task-1-report.md pour la mesure complète et le diagnostic, section « Correctif
post-revue » pour l'attribution corrigée du mécanisme) : le schéma bien équilibré
`_rhs_o2` (reconstruction MUSCL de la surface libre) n'est conservatif en masse QUE
tant qu'aucun front mouillé/sec n'atteint un mur réfléchissant. Au mur, la pente de
SURFACE (η=h+b) reste exactement nulle (padding Neumann/« edge » sur h ET b dans
`_pad_reflective` : la cellule fantôme est une copie exacte de la cellule réelle
adjacente, donc la différence arrière minmod est 0 et la pente η y est clippée à 0,
pas de gradient de surface non nul en jeu). La fuite vient d'ailleurs : le moment
normal `hu` est padé de façon ANTISYMÉTRIQUE au mur (`hup[:, 0] = -hup[:, 1]`, pour
imposer une vitesse normale nulle réfléchie) — contrairement à η, ce padding NE
donne PAS une différence arrière nulle pour la vitesse reconstruite ; la pente MUSCL
de vitesse à la cellule adjacente au mur n'est donc PAS clippée à 0 par minmod,
alors que celle du fantôme l'est (1er ordre au mur par construction). Cette
asymétrie de pente entre état gauche (fantôme) et état droit (cellule réelle) de
l'interface-mur casse la symétrie miroir du flux HLL (qui garantit Fh_mur=0 pour un
schéma 1er ordre / sur repos), laissant un flux de masse résiduel au mur (mesuré :
Fh_x ~ 1e-3 par interface une fois l'onde arrivée au mur, contre 0 exactement avant
contact). Sur une relaxation complète de N_settle=600 pas, la dérive de masse
mesurée (avant assèchement) va de ~3 % à ~13 % selon le centre du pulse — trois
ordres de grandeur au-dessus de la tolérance 1e-8 visée a priori. `run_episode` ne
peut pas corriger cela sans éditer `solver_wetdry.py` (interdit par le contrat) ; le
test de conservation de masse de ce module utilise donc une tolérance mesurée,
documentée, et non 1e-8.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import GridConfig
from src.solver_wetdry import simulate_wetdry_o2

# --- Terrain b0 FIGÉ (identique pour tous les runs, cf. plan Arc A manche 1) ---

_SLOPE_DROP: float = 0.5   # chute totale du plan incliné, de x=0 à x=W-1
_BUMP_CENTERS: tuple[tuple[float, float], ...] = ((0.3, 0.35), (0.6, 0.6), (0.45, 0.75))
_BUMP_AMPLITUDES: tuple[float, ...] = (0.25, 0.30, 0.20)
_BUMP_SIGMA: float = 6.0   # cellules


def default_terrain(grid: GridConfig) -> np.ndarray:
    """b0 FIGÉ : plan incliné le long de x (chute totale 0.5 de x=0 à x=W−1) + 3
    bosses gaussiennes fixes (centres en fractions (x,y), σ=6 cellules, amplitudes
    0.25/0.30/0.20). Terrain statique, identique pour tous les runs (asset connu).
    `relief ≡ max(b0) − min(b0)` (calculé par l'appelant, pas stocké ici)."""
    H, W = grid.H, grid.W
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    b = _SLOPE_DROP * (1.0 - xx / (W - 1))
    for (cxf, cyf), amp in zip(_BUMP_CENTERS, _BUMP_AMPLITUDES):
        cx, cy = cxf * (W - 1), cyf * (H - 1)
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2
        b = b + amp * np.exp(-r2 / (2.0 * _BUMP_SIGMA ** 2))
    return b


_GRID = GridConfig()                                    # 64x64, dx=dy=1 (défaut FIGÉ)
_B0_FIGE = default_terrain(_GRID)
_RELIEF: float = float(_B0_FIGE.max() - _B0_FIGE.min())  # relief du b0 FIGÉ
_DRY_EPS: float = 1e-4     # dry_eps par défaut de simulate_wetdry_o2 (non redéfini ici)

# Calibration k_d (cf. RÉSOLUTION D'AMBIGUÏTÉ / calibrate_kd ci-dessous). Reproduire :
#   .venv/bin/python -c "
#   from config import GridConfig
#   from src.sediment import default_terrain, calibrate_kd
#   b0 = default_terrain(GridConfig())
#   print(calibrate_kd(b0))"
# Bissection sur log10(k_d) dans [-6, 1], cible max(s)/relief dans [0.18, 0.22],
# seed de calibration=12345, 10 épisodes, 10 itérations max. GELÉ après calibration :
# plus aucune retouche.
# RE-GELÉ après le correctif « cellules mouillées » (revue Task 1 : dépôt/érosion
# restreints au masque wet dans _exner_step) et AVANT toute re-validation :
# calibrate_kd relancée UNE fois avec les mêmes défauts (seed 12345, cible
# [0.18, 0.22], 10 épisodes) sur le code corrigé -> valeur bit-à-bit IDENTIQUE
# (le correctif ne touche que la bande 1e-4 < h <= 1e-3 ; le chemin de bissection
# est inchangé et retourne le même point médian dyadique sur log10(k_d)).
KD_CALIBRE: float = 0.0019109529749704406


@dataclass(frozen=True)
class SedimentParams:
    """Paramètres FIGÉS de la dynamique sédiment (Exner) + du pulse d'un épisode.

    `h_p` est une valeur numérique figée (= 0.6·relief(b0) au point d'opération),
    PAS une fraction recalculée à la volée : le terrain b0 étant lui-même figé,
    c'est un unique nombre, gelé ici comme les autres champs de cette dataclass."""
    k_d: float = KD_CALIBRE
    k_e: float = KD_CALIBRE / 5.0     # FIGÉ a priori : k_e = k_d/5
    theta_c: float = 0.5              # FIGÉ a priori
    sigma_pulse: float = 5.0          # cellules
    h_p: float = 0.6 * _RELIEF        # hauteur crête du pulse = 0.6·relief(b0)
    N_settle: int = 600               # pas solveur de relaxation par épisode
    save_every: int = 2               # sous-échantillonnage des snapshots pour Exner


# Budget de temps physique (t_end) passé à simulate_wetdry_o2 pour garantir au moins
# N_settle=600 pas solveur acceptés, quel que soit le centre du pulse dans la boîte
# [0.15, 0.85]². Choix empirique (cf. task-1-report.md) : balayage de ~20 centres
# aléatoires + les 4 coins de la boîte avec le b0/pulse FIGÉS -> le nombre de pas
# atteint à t_end=130 va de 1160 (pire cas observé) à 2664, toujours >= 600 avec
# une marge >= 1.9x. Choisi au plus juste (pas de marge supplémentaire au-delà de
# 130) pour limiter le coût de calcul (chaque appel solveur tourne jusqu'à t_end,
# indépendamment de N_settle -- le budget calibration/tests en dépend directement).
_T_END_RELAX: float = 130.0


def _initial_pulse(grid: GridConfig, center_frac: tuple[float, float],
                   params: SedimentParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Monticule d'eau gaussien ajouté à h (u=v=0 initialement) : centre en fractions
    (x_frac, y_frac), σ=`params.sigma_pulse` cellules, hauteur crête `params.h_p`."""
    H, W = grid.H, grid.W
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    cxf, cyf = center_frac
    cx, cy = cxf * (W - 1), cyf * (H - 1)
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    h0 = params.h_p * np.exp(-r2 / (2.0 * params.sigma_pulse ** 2))
    z = np.zeros((H, W), dtype=np.float64)
    return h0, z.copy(), z.copy()


def _relax_episode(b_eff: np.ndarray, center_frac: tuple[float, float],
                   params: SedimentParams
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pulse + relaxation `simulate_wetdry_o2` (limiteur minmod, dry_eps par défaut
    du module), tronquée aux `params.N_settle` premiers pas acceptés. Retourne
    (times, h_seq, hu_seq, hv_seq) de longueur N_settle+1 (le snapshot 0 = CI)."""
    grid = GridConfig(H=b_eff.shape[0], W=b_eff.shape[1])
    h0, hu0, hv0 = _initial_pulse(grid, center_frac, params)
    times, hs, hus, hvs = simulate_wetdry_o2(h0, hu0, hv0, b_eff, grid, t_end=_T_END_RELAX)
    n_available = len(times) - 1
    if n_available < params.N_settle:
        raise RuntimeError(
            f"_T_END_RELAX={_T_END_RELAX} insuffisant : seulement {n_available} pas "
            f"solveur acceptés pour centre={center_frac}, N_settle={params.N_settle} requis. "
            "Augmenter _T_END_RELAX (garde-fou, ne devrait pas se produire dans la "
            "boîte [0.15, 0.85]² d'après le balayage empirique documenté).")
    n = params.N_settle + 1
    return times[:n], hs[:n], hus[:n], hvs[:n]


def _exner_step(s: np.ndarray, h: np.ndarray, hu: np.ndarray, hv: np.ndarray,
                dt: float, params: SedimentParams) -> np.ndarray:
    """Un pas d'Euler explicite de la loi de dépôt/érosion type Exner sur le
    snapshot (h, hu, hv) : θ=u²+v² sur les cellules mouillées au sens Exner
    (h > 10·dry_eps — seuil plus conservateur que le dry_eps du solveur, pour éviter
    les vitesses bruitées d'une désingularisation près du seuil sec du solveur) ;
    ds/dt = 1[wet]·(k_d·h·1[θ<θc]·(1−θ/θc) − k_e·s·1[θ>θc]·(θ/θc−1)) ; s ≥ 0 clampé.
    Le dépôt ET l'érosion sont restreints aux cellules mouillées (`wet`) : une
    cellule 0 < h ≤ 10·dry_eps est sèche au sens Exner et ne reçoit ni dépôt ni
    érosion, même si θ=0 y satisferait trivialement θ<θc (correctif post-revue)."""
    wet = h > 10.0 * _DRY_EPS
    u = np.zeros_like(h)
    v = np.zeros_like(h)
    u[wet] = hu[wet] / h[wet]
    v[wet] = hv[wet] / h[wet]
    theta = u ** 2 + v ** 2
    depot = wet * params.k_d * h * (theta < params.theta_c) * (1.0 - theta / params.theta_c)
    erosion = wet * params.k_e * s * (theta > params.theta_c) * (theta / params.theta_c - 1.0)
    s_new = s + dt * (depot - erosion)
    return np.maximum(s_new, 0.0)


def _integrate_exner(s0: np.ndarray, times: np.ndarray, hs: np.ndarray,
                     hus: np.ndarray, hvs: np.ndarray,
                     params: SedimentParams) -> np.ndarray:
    """Intègre la loi Exner séquentiellement sur les snapshots sous-échantillonnés
    (tous les `save_every`), dt = temps écoulé réel depuis le dernier snapshot
    utilisé (identique à « dt du snapshot × save_every » quand le dt CFL est
    localement ~constant, exact sinon — cf. RÉSOLUTION D'AMBIGUÏTÉ)."""
    s = s0.copy()
    n = len(times)
    for i in range(params.save_every, n, params.save_every):
        dt = float(times[i] - times[i - params.save_every])
        s = _exner_step(s, hs[i], hus[i], hvs[i], dt, params)
    return s


def run_episode(s: np.ndarray, b0: np.ndarray, center_frac: tuple[float, float],
                params: SedimentParams = SedimentParams()) -> np.ndarray:
    """Un épisode complet : pulse -> relaxation `simulate_wetdry_o2` sur
    `b_eff = b0 + s` (gelé pendant l'épisode) -> intégration Exner séquentielle sur
    les snapshots -> assèchement implicite (h, hu, hv ne sont pas retournés ; seul
    le nouveau `s` persiste). Fonction PURE : ne mute ni `s` ni `b0`."""
    b_eff = b0 + s
    times, hs, hus, hvs = _relax_episode(b_eff, center_frac, params)
    return _integrate_exner(s, times, hs, hus, hvs, params)


def run_history(seed: int, n_episodes: int, b0: np.ndarray,
                params: SedimentParams = SedimentParams(),
                checkpoints: set[int] | None = None) -> dict[int, np.ndarray]:
    """Séquence de `n_episodes` épisodes, centres tirés de `default_rng(seed)`
    uniformément dans la boîte [0.15, 0.85]² (fractions). Retourne
    {n: s_après_n_épisodes} aux `checkpoints` demandés (par défaut : {n_episodes}
    seulement)."""
    rng = np.random.default_rng(seed)
    if checkpoints is None:
        checkpoints = {n_episodes}
    s = np.zeros_like(b0, dtype=np.float64)
    out: dict[int, np.ndarray] = {}
    for ep in range(1, n_episodes + 1):
        center = (float(rng.uniform(0.15, 0.85)), float(rng.uniform(0.15, 0.85)))
        s = run_episode(s, b0, center, params)
        if ep in checkpoints:
            out[ep] = s.copy()
    return out


def calibrate_kd(b0: np.ndarray, target: float = 0.2, seed: int = 12345,
                 lo_log: float = -6.0, hi_log: float = 1.0,
                 max_iters: int = 10, n_episodes: int = 10,
                 band: float = 0.02) -> float:
    """Bissection sur log10(k_d) ∈ [lo_log, hi_log] pour max(s)/relief ≈ `target`
    (±`band`) après `n_episodes` (défaut 10, seed=12345 — jamais réutilisée pour
    les mesures ultérieures). Suppose max(s)/relief croissant en k_d (déposition
    plus forte -> plus de sédiment). Max `max_iters` itérations ; lève RuntimeError
    (BLOCKED) si la plage ne bracket pas la cible ou si la bissection ne converge
    pas — ne force RIEN, cf. contrat épistémique du projet."""
    relief = float(b0.max() - b0.min())

    def _max_s_ratio(k_d: float) -> float:
        params = SedimentParams(k_d=k_d, k_e=k_d / 5.0)
        hist = run_history(seed, n_episodes, b0, params, checkpoints={n_episodes})
        return float(hist[n_episodes].max()) / relief

    lo, hi = lo_log, hi_log
    ratio_lo, ratio_hi = _max_s_ratio(10.0 ** lo), _max_s_ratio(10.0 ** hi)
    if not (ratio_lo <= target <= ratio_hi):
        raise RuntimeError(
            f"calibrate_kd : la cible {target} n'est pas bracketée par la plage "
            f"log10(k_d) ∈ [{lo_log}, {hi_log}] -> ratios [{ratio_lo}, {ratio_hi}]. "
            "BLOCKED : ne pas forcer, élargir la plage nécessiterait une décision "
            "du contrôleur.")

    for _ in range(max_iters):
        mid = 0.5 * (lo + hi)
        k_d = 10.0 ** mid
        ratio = _max_s_ratio(k_d)
        if abs(ratio - target) <= band:
            return k_d
        if ratio < target:
            lo = mid
        else:
            hi = mid
    raise RuntimeError(
        f"calibrate_kd n'a pas convergé en {max_iters} itérations : dernier "
        f"k_d={k_d} (log10={mid}), ratio={ratio}, cible={target}±{band}. "
        "BLOCKED : ne pas forcer la constante.")


if __name__ == "__main__":
    # Reproduction de KD_CALIBRE (cf. commentaire ci-dessus). Ne modifie PAS la
    # constante gelée : affiche seulement la valeur pour audit / re-exécution.
    _b0 = default_terrain(_GRID)
    _k_d = calibrate_kd(_b0)
    print(f"k_d calibré = {_k_d!r} (KD_CALIBRE gelé = {KD_CALIBRE!r})")
