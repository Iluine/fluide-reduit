"""W1 — Solveur shallow-water MOUILLÉ/SEC (well-balanced + positivity-preserving).

Reconstruction hydrostatique (Audusse et al. 2004, SIAM J. Sci. Comput. 25(6):2050)
+ flux HLL + gestion mouillé/sec + SSP-RK2 (1er ordre en espace). NOUVEAU schéma —
n'altère PAS src/solver.py (Rusanov submergé, tag v1). Validé contre solutions
analytiques (cf. scripts/run_w1_validate.py / Task 6), pas par la seule masse.

Variables conservatives q=(h, hu, hv). Indexation array[y, x] : axe 0 = y (H lignes),
axe 1 = x (W colonnes). u = vitesse selon x, v = vitesse selon y. b = bathymétrie (z).

Schéma well-balanced (C-property) — Audusse et al. 2004 :
  - À chaque interface, reconstruction hydrostatique des hauteurs :
        z* = max(z_L, z_R) ;  h*_L = max(0, η_L − z*) ;  h*_R = max(0, η_R − z*).
  - Flux HLL sur les états reconstruits (h*_K, h*_K·u_K).
  - Correction de pression well-balanced ajoutée à la quantité de mouvement, côté K :
        (g/2)(h_K² − h*_K²).
    Au repos (η = const, u = 0) le flux HLL de qdm vaut (g/2)h*², et la correction
    le ramène exactement à (g/2)h_K² des deux côtés de chaque cellule → divergence
    nulle, vitesse parasite ≈ 0 (C-property). C'est le terme recoupé contre la
    référence ; la C-property en est le validateur.
"""
from __future__ import annotations

import numpy as np

from config import GRAVITY, GridConfig


def desingularize_velocity(h: np.ndarray, hu: np.ndarray, dry_eps: float) -> np.ndarray:
    """u = hu/h, mais 0 si h <= dry_eps (jamais de division explosive près du sec).

    Désingularisation à seuil : sous le seuil sec ``dry_eps`` la hauteur n'est pas
    fiable pour porter une vitesse, donc u est mis à 0 (positivité + pas de blow-up)."""
    h = np.asarray(h, dtype=np.float64)
    hu = np.asarray(hu, dtype=np.float64)
    u = np.zeros_like(h)
    wet = h > dry_eps
    u[wet] = hu[wet] / h[wet]
    return u


def _hll_flux(hL: np.ndarray, huL: np.ndarray, hR: np.ndarray, huR: np.ndarray,
              dry_eps: float) -> tuple[np.ndarray, np.ndarray]:
    """Flux HLL 1D (masse + qdm normale) sur des états DÉJÀ reconstruits.

    Vitesses d'onde de Davis, dry-aware : côté sec → vitesse fixée par la
    raréfaction du côté mouillé. Retourne (F_h, F_hu)."""
    g = GRAVITY
    uL = desingularize_velocity(hL, huL, dry_eps)
    uR = desingularize_velocity(hR, huR, dry_eps)
    cL = np.sqrt(g * np.maximum(hL, 0.0))
    cR = np.sqrt(g * np.maximum(hR, 0.0))
    # vitesses d'onde dry-aware : côté sec -> raréfaction du côté mouillé
    sL = np.where(hL > dry_eps, np.minimum(uL - cL, uR - cR), uR - 2.0 * cR)
    sR = np.where(hR > dry_eps, np.maximum(uL + cL, uR + cR), uL + 2.0 * cL)
    # flux physiques
    FhL, FhR = huL, huR
    FhuL = huL * uL + 0.5 * g * hL ** 2
    FhuR = huR * uR + 0.5 * g * hR ** 2

    def hll(FL, FR, UL, UR):
        return np.where(
            sL >= 0, FL,
            np.where(sR <= 0, FR,
                     (sR * FL - sL * FR + sL * sR * (UR - UL)) / (sR - sL + 1e-300)))

    return hll(FhL, FhR, hL, hR), hll(FhuL, FhuR, huL, huR)


def _pad_reflective(h: np.ndarray, hu: np.ndarray, hv: np.ndarray, b: np.ndarray):
    """Cellules fantômes pour parois réfléchissantes (miroir de src/solver.py).

    h, b : Neumann (edge) ; hu : composante normale négée aux bords x (gauche/droite) ;
    hv : composante normale négée aux bords y (haut/bas). Padding sur b aussi de sorte
    que la surface η = h + b reste continue au mur → reconstruction hydrostatique
    cohérente (le mur ne crée pas de marche de bathymétrie parasite).
    Retourne (hp, hup, hvp, bp), chacun (H+2, W+2)."""
    hp = np.pad(h, 1, mode="edge")
    hup = np.pad(hu, 1, mode="edge")
    hvp = np.pad(hv, 1, mode="edge")
    bp = np.pad(b, 1, mode="edge")
    hup[:, 0] = -hup[:, 1]      # bord gauche (x) : négation de h*u
    hup[:, -1] = -hup[:, -2]    # bord droit (x)
    hvp[0, :] = -hvp[1, :]      # bord haut (y) : négation de h*v
    hvp[-1, :] = -hvp[-2, :]    # bord bas (y)
    return hp, hup, hvp, bp


def _rhs(h: np.ndarray, hu: np.ndarray, hv: np.ndarray, b: np.ndarray,
         grid: GridConfig, dry_eps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Opérateur spatial L(U) : −div(flux) + source bathy (via reconstruction).

    Le terme source de bathymétrie est INTÉGRÉ dans la reconstruction hydrostatique
    (correction de pression d'Audusse), PAS sous forme d'un −g h ∇b séparé. Retourne
    (Lh, Lhu, Lhv) sur les cellules réelles (H, W)."""
    g = GRAVITY
    dx, dy = grid.dx, grid.dy
    hp, hup, hvp, bp = _pad_reflective(h, hu, hv, b)
    etap = hp + bp  # surface libre étendue (H+2, W+2)

    # ---- Interfaces selon x : entre colonne j (gauche) et j+1 (droite) -> (H+2, W+1)
    hL, hR = hp[:, :-1], hp[:, 1:]
    huL, huR = hup[:, :-1], hup[:, 1:]
    etaL, etaR = etap[:, :-1], etap[:, 1:]
    zstar = np.maximum(bp[:, :-1], bp[:, 1:])
    hstarL = np.maximum(0.0, etaL - zstar)
    hstarR = np.maximum(0.0, etaR - zstar)
    # moment normal reconstruit = h* * u (u inchangé via desingularisation sur l'état réel)
    uL = desingularize_velocity(hL, huL, dry_eps)
    uR = desingularize_velocity(hR, huR, dry_eps)
    Fh_x, Fhun_x = _hll_flux(hstarL, hstarL * uL, hstarR, hstarR * uR, dry_eps)
    # transport de la qdm TANGENTIELLE (hv) par le flux de masse (upwind sur le signe du flux)
    vL = desingularize_velocity(hL, hvp[:, :-1], dry_eps)
    vR = desingularize_velocity(hR, hvp[:, 1:], dry_eps)
    Fhvt_x = np.where(Fh_x >= 0.0, Fh_x * vL, Fh_x * vR)

    # ---- Interfaces selon y : entre ligne i (bas/L) et i+1 (haut/R) -> (H+1, W+2)
    hB, hT = hp[:-1, :], hp[1:, :]
    hvB, hvT = hvp[:-1, :], hvp[1:, :]
    etaB, etaT = etap[:-1, :], etap[1:, :]
    zstar_y = np.maximum(bp[:-1, :], bp[1:, :])
    hstarB = np.maximum(0.0, etaB - zstar_y)
    hstarT = np.maximum(0.0, etaT - zstar_y)
    vB = desingularize_velocity(hB, hvB, dry_eps)
    vT = desingularize_velocity(hT, hvT, dry_eps)
    Gh_y, Ghvn_y = _hll_flux(hstarB, hstarB * vB, hstarT, hstarT * vT, dry_eps)
    uB = desingularize_velocity(hB, hup[:-1, :], dry_eps)
    uT = desingularize_velocity(hT, hup[1:, :], dry_eps)
    Ghut_y = np.where(Gh_y >= 0.0, Gh_y * uB, Gh_y * uT)

    # ---- Correction de pression well-balanced (Audusse) sur la qdm NORMALE.
    # Au flux d'interface, ajouter côté K : (g/2)(h_K^2 - h*_K^2).
    # Pour la cellule i, son interface DROITE (i+1/2) utilise son état GAUCHE (L),
    # son interface GAUCHE (i-1/2) utilise son état DROIT (R).
    # x : flux corrigé "côté gauche" (vu de la cellule de gauche j) et "côté droit".
    h_jL = hp[:, :-1]   # hauteur réelle de la cellule à gauche de l'interface
    h_jR = hp[:, 1:]    # hauteur réelle de la cellule à droite de l'interface
    Sx_L = 0.5 * g * (h_jL ** 2 - hstarL ** 2)   # correction pour la cellule de gauche
    Sx_R = 0.5 * g * (h_jR ** 2 - hstarR ** 2)   # correction pour la cellule de droite
    # flux de qdm normale vu par la cellule de gauche / de droite de chaque interface x
    Fhun_x_left = Fhun_x + Sx_L
    Fhun_x_right = Fhun_x + Sx_R

    h_iB = hp[:-1, :]
    h_iT = hp[1:, :]
    Sy_B = 0.5 * g * (h_iB ** 2 - hstarB ** 2)
    Sy_T = 0.5 * g * (h_iT ** 2 - hstarT ** 2)
    Ghvn_y_bot = Ghvn_y + Sy_B
    Ghvn_y_top = Ghvn_y + Sy_T

    # ---- Divergence sur les cellules réelles (H, W).
    # Pour la cellule (réelle) j en x : flux sortant à droite (vu côté L) moins flux
    # entrant à gauche (vu côté R). Les indices x réels sont [1:-1] en y, colonnes [.].
    # Interfaces x réelles bordant les cellules réelles : index W-1 internes + 2 murs.
    def divx(Fmass, Fnorm_left, Fnorm_right, Ftang):
        # interfaces x : 0..W ; cellule réelle j (0..W-1) -> interface gauche j, droite j+1
        # restreindre aux lignes réelles [1:-1]
        m = Fmass[1:-1, :]
        nl = Fnorm_left[1:-1, :]
        nr = Fnorm_right[1:-1, :]
        tg = Ftang[1:-1, :]
        dh = (m[:, 1:] - m[:, :-1]) / dx
        # qdm normale : interface droite (j+1) vue côté gauche (L) - interface gauche (j) vue côté droit (R)
        dhun = (nl[:, 1:] - nr[:, :-1]) / dx
        dhvt = (tg[:, 1:] - tg[:, :-1]) / dx
        return dh, dhun, dhvt

    def divy(Gmass, Gnorm_bot, Gnorm_top, Gtang):
        m = Gmass[:, 1:-1]
        nb = Gnorm_bot[:, 1:-1]
        nt = Gnorm_top[:, 1:-1]
        tg = Gtang[:, 1:-1]
        dh = (m[1:, :] - m[:-1, :]) / dy
        # qdm normale (v) : interface haute (i+1) vue côté bas (B) - interface basse (i) vue côté haut (T)
        dhvn = (nb[1:, :] - nt[:-1, :]) / dy
        dhut = (tg[1:, :] - tg[:-1, :]) / dy
        return dh, dhvn, dhut

    dxh, dxhun, dxhvt = divx(Fh_x, Fhun_x_left, Fhun_x_right, Fhvt_x)
    dyh, dyhvn, dyhut = divy(Gh_y, Ghvn_y_bot, Ghvn_y_top, Ghut_y)

    Lh = -(dxh + dyh)
    Lhu = -(dxhun + dyhut)   # qdm en x : flux normal en x + flux tangentiel en y
    Lhv = -(dxhvt + dyhvn)   # qdm en y : flux tangentiel en x + flux normal en y
    return Lh, Lhu, Lhv


def _floor_dry(h: np.ndarray, hu: np.ndarray, hv: np.ndarray, dry_eps: float):
    """Plancher sec : cellules h <= dry_eps mises à (0, 0, 0). Positivité ailleurs."""
    h = np.maximum(h, 0.0)
    dry = h <= dry_eps
    h = np.where(dry, 0.0, h)
    hu = np.where(dry, 0.0, hu)
    hv = np.where(dry, 0.0, hv)
    return h, hu, hv


def _cfl_dt(h: np.ndarray, hu: np.ndarray, hv: np.ndarray, grid: GridConfig,
            cfl: float, dry_eps: float) -> float:
    """Pas de temps positivity-preserving : dt = cfl·min(dx,dy)/max(|u|+c, |v|+c)."""
    c = np.sqrt(GRAVITY * np.maximum(h, 0.0))
    u = desingularize_velocity(h, hu, dry_eps)
    v = desingularize_velocity(h, hv, dry_eps)
    smax = max(float((np.abs(u) + c).max()), float((np.abs(v) + c).max()))
    return cfl * min(grid.dx, grid.dy) / max(smax, 1e-12)


def _minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Limiteur minmod (élément par élément) : 0 si signes opposés, sinon le plus petit
    module en valeur signée. TVD, le plus diffusif (le plus sûr près du sec)."""
    return np.where(a * b <= 0.0, 0.0,
                    np.sign(a) * np.minimum(np.abs(a), np.abs(b)))


def _mc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Limiteur MC (monotonized central) : sign·min(2|a|, 2|b|, |a+b|/2), 0 si signes
    opposés. TVD aussi, mais PLUS NET que minmod (pente centrale écrêtée à 2× le plus
    petit côté) → front plus raide (~1 cellule)."""
    return np.where(a * b <= 0.0, 0.0,
                    np.sign(a) * np.minimum(np.minimum(2.0 * np.abs(a), 2.0 * np.abs(b)),
                                            0.5 * np.abs(a + b)))


_LIMITERS = {"minmod": _minmod, "mc": _mc}


def _slopes_x(arr: np.ndarray, lim) -> np.ndarray:
    """Pente x limitée par cellule sur un tableau padé (H+2,W+2), via le limiteur ``lim``.
    Colonnes fantômes (0, W+1) : pente 0 (reconstruction 1er ordre au mur)."""
    s = np.zeros_like(arr)
    bw = arr[:, 1:-1] - arr[:, :-2]   # différence arrière
    fw = arr[:, 2:] - arr[:, 1:-1]    # différence avant
    s[:, 1:-1] = lim(bw, fw)
    return s


def _slopes_y(arr: np.ndarray, lim) -> np.ndarray:
    """Pente y limitée par cellule sur un tableau padé (H+2,W+2), via ``lim``."""
    s = np.zeros_like(arr)
    bw = arr[1:-1, :] - arr[:-2, :]
    fw = arr[2:, :] - arr[1:-1, :]
    s[1:-1, :] = lim(bw, fw)
    return s


def _rhs_o2(h: np.ndarray, hu: np.ndarray, hv: np.ndarray, b: np.ndarray,
            grid: GridConfig, dry_eps: float, limiter: str = "minmod"
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Opérateur spatial L(U) au 2e ordre (MUSCL surface-gradient + hydrostatique).

    Reconstruit la SURFACE LIBRE η = h + b (PAS h) par pentes limitées (surface gradient
    method, Zhou et al. 2001 ; limiteur ``minmod`` par défaut, ``mc`` plus net) — au repos
    η=const → pente nulle → se réduit EXACTEMENT au 1er ordre `_rhs` → C-property héritée.
    Le lit b N'EST PAS reconstruit (z* = max des lits cellules, comme Audusse), ce qui
    garantit cette réduction. Réconciliation η/positivité : si une face d'une cellule donne
    h reconstruit < 0, on annule la pente de cette cellule (retour 1er ordre) → toutes les
    faces ont h ≥ 0. Puis reconstruction hydrostatique (h* = max(0, η_face − z*)) + HLL +
    correction de pression, identiques au 1er ordre mais sur les états reconstruits.
    Retourne (Lh, Lhu, Lhv) sur (H, W)."""
    g = GRAVITY
    dx, dy = grid.dx, grid.dy
    lim = _LIMITERS[limiter]
    hp, hup, hvp, bp = _pad_reflective(h, hu, hv, b)
    etap = hp + bp
    up = desingularize_velocity(hp, hup, dry_eps)
    vp = desingularize_velocity(hp, hvp, dry_eps)

    # ----- Pentes limitées + réconciliation positivité (par cellule, axe x) -----
    seta_x, su_x, sv_x = _slopes_x(etap, lim), _slopes_x(up, lim), _slopes_x(vp, lim)
    hLf_x = (etap - 0.5 * seta_x) - bp          # h reconstruit, face gauche de chaque cellule
    hRf_x = (etap + 0.5 * seta_x) - bp          # face droite
    bad_x = (hLf_x < 0.0) | (hRf_x < 0.0)       # surface reconstruite sous le lit
    seta_x = np.where(bad_x, 0.0, seta_x)
    su_x = np.where(bad_x, 0.0, su_x)
    sv_x = np.where(bad_x, 0.0, sv_x)

    # ----- Interfaces x : état L = face droite cellule j ; R = face gauche cellule j+1 -----
    etaL = etap[:, :-1] + 0.5 * seta_x[:, :-1]
    etaR = etap[:, 1:] - 0.5 * seta_x[:, 1:]
    uL = up[:, :-1] + 0.5 * su_x[:, :-1]
    uR = up[:, 1:] - 0.5 * su_x[:, 1:]
    vL = vp[:, :-1] + 0.5 * sv_x[:, :-1]
    vR = vp[:, 1:] - 0.5 * sv_x[:, 1:]
    bL_x, bR_x = bp[:, :-1], bp[:, 1:]          # lit cellule (NON reconstruit)
    hL_x, hR_x = etaL - bL_x, etaR - bR_x       # profondeurs reconstruites aux faces
    zstar_x = np.maximum(bL_x, bR_x)
    hstarL_x = np.maximum(0.0, etaL - zstar_x)
    hstarR_x = np.maximum(0.0, etaR - zstar_x)
    Fh_x, Fhun_x = _hll_flux(hstarL_x, hstarL_x * uL, hstarR_x, hstarR_x * uR, dry_eps)
    Fhvt_x = np.where(Fh_x >= 0.0, Fh_x * vL, Fh_x * vR)
    Sx_L = 0.5 * g * (hL_x ** 2 - hstarL_x ** 2)
    Sx_R = 0.5 * g * (hR_x ** 2 - hstarR_x ** 2)
    Fhun_x_left = Fhun_x + Sx_L
    Fhun_x_right = Fhun_x + Sx_R

    # ----- Pentes + réconciliation (axe y) -----
    seta_y, su_y, sv_y = _slopes_y(etap, lim), _slopes_y(up, lim), _slopes_y(vp, lim)
    hBf_y = (etap - 0.5 * seta_y) - bp
    hTf_y = (etap + 0.5 * seta_y) - bp
    bad_y = (hBf_y < 0.0) | (hTf_y < 0.0)
    seta_y = np.where(bad_y, 0.0, seta_y)
    su_y = np.where(bad_y, 0.0, su_y)
    sv_y = np.where(bad_y, 0.0, sv_y)

    # ----- Interfaces y : état B = face haute cellule i ; T = face basse cellule i+1 -----
    etaB = etap[:-1, :] + 0.5 * seta_y[:-1, :]
    etaT = etap[1:, :] - 0.5 * seta_y[1:, :]
    vB = vp[:-1, :] + 0.5 * sv_y[:-1, :]
    vT = vp[1:, :] - 0.5 * sv_y[1:, :]
    uB = up[:-1, :] + 0.5 * su_y[:-1, :]
    uT = up[1:, :] - 0.5 * su_y[1:, :]
    bB_y, bT_y = bp[:-1, :], bp[1:, :]
    hB_y, hT_y = etaB - bB_y, etaT - bT_y
    zstar_y = np.maximum(bB_y, bT_y)
    hstarB = np.maximum(0.0, etaB - zstar_y)
    hstarT = np.maximum(0.0, etaT - zstar_y)
    Gh_y, Ghvn_y = _hll_flux(hstarB, hstarB * vB, hstarT, hstarT * vT, dry_eps)
    Ghut_y = np.where(Gh_y >= 0.0, Gh_y * uB, Gh_y * uT)
    Sy_B = 0.5 * g * (hB_y ** 2 - hstarB ** 2)
    Sy_T = 0.5 * g * (hT_y ** 2 - hstarT ** 2)
    Ghvn_y_bot = Ghvn_y + Sy_B
    Ghvn_y_top = Ghvn_y + Sy_T

    # ----- Divergence (identique au 1er ordre) -----
    def divx(Fmass, Fnorm_left, Fnorm_right, Ftang):
        m, nl, nr, tg = (Fmass[1:-1, :], Fnorm_left[1:-1, :],
                         Fnorm_right[1:-1, :], Ftang[1:-1, :])
        dh = (m[:, 1:] - m[:, :-1]) / dx
        dhun = (nl[:, 1:] - nr[:, :-1]) / dx
        dhvt = (tg[:, 1:] - tg[:, :-1]) / dx
        return dh, dhun, dhvt

    def divy(Gmass, Gnorm_bot, Gnorm_top, Gtang):
        m, nb, nt, tg = (Gmass[:, 1:-1], Gnorm_bot[:, 1:-1],
                         Gnorm_top[:, 1:-1], Gtang[:, 1:-1])
        dh = (m[1:, :] - m[:-1, :]) / dy
        dhvn = (nb[1:, :] - nt[:-1, :]) / dy
        dhut = (tg[1:, :] - tg[:-1, :]) / dy
        return dh, dhvn, dhut

    dxh, dxhun, dxhvt = divx(Fh_x, Fhun_x_left, Fhun_x_right, Fhvt_x)
    dyh, dyhvn, dyhut = divy(Gh_y, Ghvn_y_bot, Ghvn_y_top, Ghut_y)
    return -(dxh + dyh), -(dxhun + dyhut), -(dxhvt + dyhvn)


def simulate_wetdry_o2(h0: np.ndarray, hu0: np.ndarray, hv0: np.ndarray, b: np.ndarray,
                       grid: GridConfig, cfl: float = 0.4, t_end: float = 2.0,
                       dry_eps: float = 1e-4, limiter: str = "minmod"
                       ) -> tuple[list[float], np.ndarray, np.ndarray, np.ndarray]:
    """Version 2e ordre (MUSCL surface-gradient) de simulate_wetdry. Même boucle SSP-RK2 /
    dt CFL / plancher sec, mais opérateur spatial `_rhs_o2`. ``limiter`` : ``minmod``
    (défaut, le plus sûr) ou ``mc`` (plus net, ~1 cellule). Le 1er ordre `simulate_wetdry`
    reste inchangé (chemin séparé)."""
    cfl = min(cfl, 0.4)
    h, hu, hv = (np.asarray(h0, np.float64).copy(),
                 np.asarray(hu0, np.float64).copy(),
                 np.asarray(hv0, np.float64).copy())
    b = np.asarray(b, np.float64)
    h, hu, hv = _floor_dry(h, hu, hv, dry_eps)

    times = [0.0]
    hs, hus, hvs = [h.copy()], [hu.copy()], [hv.copy()]
    t = 0.0
    max_steps = 1_000_000
    step = 0
    while t < t_end - 1e-12 and step < max_steps:
        step += 1
        dt = _cfl_dt(h, hu, hv, grid, cfl, dry_eps)
        if t + dt > t_end:
            dt = t_end - t
        L1h, L1hu, L1hv = _rhs_o2(h, hu, hv, b, grid, dry_eps, limiter)
        h1 = h + dt * L1h
        hu1 = hu + dt * L1hu
        hv1 = hv + dt * L1hv
        h1, hu1, hv1 = _floor_dry(h1, hu1, hv1, dry_eps)
        L2h, L2hu, L2hv = _rhs_o2(h1, hu1, hv1, b, grid, dry_eps, limiter)
        hn = 0.5 * h + 0.5 * (h1 + dt * L2h)
        hun = 0.5 * hu + 0.5 * (hu1 + dt * L2hu)
        hvn = 0.5 * hv + 0.5 * (hv1 + dt * L2hv)
        h, hu, hv = _floor_dry(hn, hun, hvn, dry_eps)
        t += dt
        times.append(t)
        hs.append(h.copy())
        hus.append(hu.copy())
        hvs.append(hv.copy())

    return times, np.stack(hs), np.stack(hus), np.stack(hvs)


def simulate_wetdry(h0: np.ndarray, hu0: np.ndarray, hv0: np.ndarray, b: np.ndarray,
                    grid: GridConfig, cfl: float = 0.4, t_end: float = 2.0,
                    dry_eps: float = 1e-4
                    ) -> tuple[list[float], np.ndarray, np.ndarray, np.ndarray]:
    """Intègre le système mouillé/sec jusqu'à ``t_end`` (SSP-RK2, dt adaptatif CFL).

    Variables conservatives en interne : q = (h, hu, hv). Sauvegarde un snapshot à
    chaque pas accepté (le premier snapshot est la condition initiale).

    Args:
        h0, hu0, hv0: hauteur et quantités de mouvement initiales (H, W).
        b: bathymétrie fixe (H, W).
        grid: grille régulière.
        cfl: nombre de Courant (positivity-preserving exige cfl <= 0.4).
        t_end: temps physique final.
        dry_eps: seuil de cellule sèche.

    Returns:
        (times, h_seq, hu_seq, hv_seq) avec les séquences empilées (T, H, W) et
        ``times`` la liste des temps physiques correspondants.
    """
    cfl = min(cfl, 0.4)  # positivity-preserving : borne dure
    h, hu, hv = (np.asarray(h0, np.float64).copy(),
                 np.asarray(hu0, np.float64).copy(),
                 np.asarray(hv0, np.float64).copy())
    b = np.asarray(b, np.float64)
    h, hu, hv = _floor_dry(h, hu, hv, dry_eps)

    times = [0.0]
    hs, hus, hvs = [h.copy()], [hu.copy()], [hv.copy()]
    t = 0.0
    max_steps = 1_000_000
    step = 0
    while t < t_end - 1e-12 and step < max_steps:
        step += 1
        dt = _cfl_dt(h, hu, hv, grid, cfl, dry_eps)
        if t + dt > t_end:
            dt = t_end - t

        # SSP-RK2 (Heun). Plancher sec après chaque étage.
        L1h, L1hu, L1hv = _rhs(h, hu, hv, b, grid, dry_eps)
        h1 = h + dt * L1h
        hu1 = hu + dt * L1hu
        hv1 = hv + dt * L1hv
        h1, hu1, hv1 = _floor_dry(h1, hu1, hv1, dry_eps)

        L2h, L2hu, L2hv = _rhs(h1, hu1, hv1, b, grid, dry_eps)
        hn = 0.5 * h + 0.5 * (h1 + dt * L2h)
        hun = 0.5 * hu + 0.5 * (hu1 + dt * L2hu)
        hvn = 0.5 * hv + 0.5 * (hv1 + dt * L2hv)
        h, hu, hv = _floor_dry(hn, hun, hvn, dry_eps)

        t += dt
        times.append(t)
        hs.append(h.copy())
        hus.append(hu.copy())
        hvs.append(hv.copy())

    return times, np.stack(hs), np.stack(hus), np.stack(hvs)
