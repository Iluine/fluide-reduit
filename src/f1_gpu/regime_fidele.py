"""F1 — M-b tranche-1 / C4 : vérification de RÉGIME sur la série (§A29-C1).

Caveat gravé (§A28) : un chrono sur un domaine tout sec, ou tout mouillé
uniforme, ne vaut RIEN — le wetdry ne fait rien, la CFL n'est pas
sollicitée. L'état synthétisé à l'échelle V4 doit EXERCER le régime :
fronts wet/dry présents ET CFL sollicitée à **4–12× de marge** (mipmap
gravé §A28 : dt_frame sous-CFL au niveau fin, marge 4–12×).

AMENDEMENT B (§A29) : la marge est vérifiée SUR LA SÉRIE, pas seulement à
l'init, et FAIL-LOUD si elle s'effondre. Une C-property cassée en f32 ferait
monter smax, mangerait la marge, et le chrono mesurerait une DIVERGENCE, pas
le F fidèle. Câblage habituel « pas de lecture sans son PASS » (B9).

Ce module fournit : un état synthétisé qui sollicite le régime, la mesure de
régime par frame, et le contrôle PASS/FAIL sur la série. Il ne mesure aucun
temps (c'est C5)."""
from __future__ import annotations

import numpy as np

from src.f1_gpu.substrat_fidele import reduction_cfl_fidele
from src.f1_gpu.substrat_jetable import DRY_EPS

DT_FRAME: float = 1.0 / 60.0            # 16,667 ms, temps réel (K=1)
MARGE_CFL_BASSE: float = 4.0            # sous : CFL non sollicitée / effondrée
MARGE_CFL_HAUTE: float = 12.0           # au-dessus : état trop lent (repos)
FRACTION_WET_BASSE: float = 0.2         # sous : trop sec
FRACTION_WET_HAUTE: float = 0.8         # au-dessus : pas de vrai front


def etat_synthetise(n: int, cp, graine: int = 101):
    """État (q=(h,hu,hv), b) synthétisé qui EXERCE le régime : bathymétrie à
    relief (plan incliné + bosses, comme `default_terrain`), lac partiel
    (fronts wet/dry) et champ de vitesse le long de la pente pour solliciter
    la CFL dans la bande 4–12×. Ce n'est PAS une trajectoire physique — c'est
    un état représentatif pour le CHRONO (pas pour la fidélité, qui est
    tranche-2)."""
    rng = np.random.default_rng(graine)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    b = 0.5 * (1.0 - xx / (n - 1))                       # plan incliné en x
    for (cxf, cyf), amp in (((0.3, 0.35), 0.25), ((0.6, 0.6), 0.30),
                            ((0.45, 0.75), 0.20)):
        cx, cy = cxf * (n - 1), cyf * (n - 1)
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2
        b = b + amp * np.exp(-r2 / (2.0 * (n / 10.0) ** 2))
    relief = float(b.max() - b.min())
    eta = float(b.min() + 0.5 * relief)                 # lac ~ mi-relief
    h = np.maximum(eta - b, 0.0)
    wet = h > 1e-3
    # écoulement le long de −x (descente de pente), amplitude calibrée pour
    # smax = |u|+c dans [2, 6] (marge CFL 4–12× à dt_frame). c ~ sqrt(g·h).
    u = np.where(wet, 1.5 + 0.3 * rng.standard_normal((n, n)), 0.0)
    q = np.zeros((1, 1, 3, n, n), np.float32)
    q[0, 0, 0] = h.astype(np.float32)
    q[0, 0, 1] = (h * u).astype(np.float32)             # hu
    return cp.asarray(q), cp.asarray(b.reshape(1, 1, n, n).astype(np.float32))


def mesurer_regime(q, b, cp, dt_frame: float = DT_FRAME) -> dict:
    """Mesure de régime d'UNE frame : fraction mouillée, dt_CFL, marge, et
    présence d'un front wet/dry."""
    h = cp.asnumpy(q[0, 0, 0])
    wet = h > float(10.0 * DRY_EPS)
    fraction_wet = float(wet.mean())
    dt_cfl = reduction_cfl_fidele(q, cp)
    marge = dt_cfl / dt_frame
    # front présent : au moins une frontière wet/dry (voisinage mixte)
    bord = (wet[:-1, :] != wet[1:, :]).any() or (wet[:, :-1] != wet[:, 1:]).any()
    return {
        "fraction_wet": fraction_wet,
        "dt_cfl": float(dt_cfl),
        "marge_cfl": float(marge),
        "front_present": bool(bord),
    }


def verifier_regime_serie(serie: list[dict]) -> dict:
    """PASS/FAIL du régime sur la SÉRIE (amendement B). PASS ssi, sur TOUTES
    les frames : fronts présents, fraction mouillée dans [0,2 ; 0,8], et
    marge CFL dans [4 ; 12]. Un effondrement de la marge (C-property cassée,
    smax qui monte) fait FAIL — le chrono mesurerait une divergence."""
    if not serie:
        return {"pass": False, "motif": "série vide — rien à vérifier"}
    marges = [m["marge_cfl"] for m in serie]
    fractions = [m["fraction_wet"] for m in serie]
    fronts = all(m["front_present"] for m in serie)
    marge_min, marge_max = min(marges), max(marges)
    frac_min, frac_max = min(fractions), max(fractions)
    marge_ok = MARGE_CFL_BASSE <= marge_min and marge_max <= MARGE_CFL_HAUTE
    frac_ok = FRACTION_WET_BASSE <= frac_min and frac_max <= FRACTION_WET_HAUTE
    ok = bool(fronts and marge_ok and frac_ok)
    motif = "régime exercé sur toute la série" if ok else "; ".join(
        m for m, c in (
            ("pas de front sur au moins une frame", not fronts),
            (f"marge CFL hors [4,12] : min={marge_min:.2f} max={marge_max:.2f}",
             not marge_ok),
            (f"fraction wet hors [0.2,0.8] : min={frac_min:.3f} "
             f"max={frac_max:.3f}", not frac_ok)) if c)
    return {
        "pass": ok,
        "n_frames": len(serie),
        "marge_cfl_min": marge_min,
        "marge_cfl_max": marge_max,
        "marge_cfl_effondree": bool(marge_min < MARGE_CFL_BASSE),
        "fraction_wet_min": frac_min,
        "fraction_wet_max": frac_max,
        "fronts_partout": fronts,
        "bornes": {"marge": [MARGE_CFL_BASSE, MARGE_CFL_HAUTE],
                   "fraction_wet": [FRACTION_WET_BASSE, FRACTION_WET_HAUTE]},
        "motif": motif,
    }


def exiger_regime_pass(verification: dict | None) -> None:
    """Câblage B9 : AUCUNE lecture T1 sans le PASS du régime. Un chrono hors
    régime (sec, ou marge effondrée) est un chiffre dont on ne sait pas s'il
    mesure le F fidèle ou le vide/une divergence."""
    if not verification or not verification.get("pass"):
        motif = verification.get("motif") if verification else "non vérifié"
        raise RuntimeError(
            "RÉGIME NON EXERCÉ (§A28-A29, caveat/amendement B) : la "
            f"vérification de régime a ÉCHOUÉ — {motif}. Aucune lecture T1 "
            "n'est produite : un chrono hors régime ne vaut rien.")
