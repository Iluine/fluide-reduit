"""F1 — M-b tranche-2 : fovéation à DEUX niveaux au 64² + halo-parent
rafraîchi ENTRE les étages RK2 (§A31).

Structure endossée (§A31) : au 64² du rederive, la pyramide de V4 dégénère,
et le halo-parent qu'exige la fidélité implique une fovéa à bord INTÉRIEUR.
D'où deux niveaux :
  - GROSSIER : le 64² décimé ×2 (32² à dx=2), murs réfléchissants au VRAI
    bord du domaine (physique, = le rederive) ;
  - FIN : une fovéa (sous-fenêtre) pleine résolution (dx=1), bord rempli par
    HALO-PARENT (grossier upsamplé, motif voisin ×2).

LE REFRESH ENTRE ÉTAGES, ma trouvaille de C1 : `pas_f_fidele` fait deux
étages RK2 ; un halo figé se PÉRIME au second (le 0,186 de C1 en était la
signature). Ici les deux niveaux avancent EN LOCKSTEP par étage, et le halo
de la fovéa est RECONSTRUIT depuis l'état intermédiaire du grossier avant
l'étage 2. Fait ICI (couche t2), JAMAIS dans `pas_f_fidele` — l'invariant
liant l'interdit. L'opérateur intérieur reste le kernel figé de C1
(`_obtenir_kernel_fidele`), appelé étage par étage.

Aucun kernel neuf : ce module ORCHESTRE le kernel de C1 (empreinte
aef7237d, re-gravée le 2026-07-25 pour §A33-CORRECTION — le PAS D'ESPACE
est devenu un ARGUMENT du kernel, ce qui est exactement l'invariant liant :
un mode de données, pas un fork de code)."""
from __future__ import annotations

import numpy as np

from src.f1_gpu.substrat_fidele import (
    MODE_HALO,
    MODE_REFLECHISSANT,
    _TAILLE_BLOC,
    _obtenir_kernel_fidele,
)

DECIMATION: int = 2                     # grossier = fin ÷ 2 (mipmap §A28)


def decimer(champ_fin, cp):
    """Décimation ×2 par moyenne 2×2 (grossier = fin décimé). Entrée
    (B, S, C, n, n) -> (B, S, C, n/2, n/2)."""
    b, s, c, n, _ = champ_fin.shape
    return champ_fin.reshape(b, s, c, n // 2, 2, n // 2, 2).mean(axis=(4, 6))


def construire_halo(q_fine, b_fine, q_coarse, b_coarse, oy: int, ox: int,
                    cp):
    """Halo padé de 2 de la fovéa : intérieur = q_fine ; anneau ±2 =
    grossier UPSAMPLÉ ×2 (motif voisin, nearest) à la position (oy, ox) de
    la fovéa dans le domaine fin. Retourne (halo_q (B,S,3,n+4,n+4),
    halo_b (B,S,n+4,n+4)).

    Le grossier est upsamplé une fois puis fenêtré autour de la fovéa avec
    padding 'edge' (le vrai bord du domaine, où le grossier réfléchit) —
    ainsi le halo reste défini même quand la fovéa touche le bord."""
    n = int(q_fine.shape[-1])
    nh = n + 4
    # grossier -> fin par répétition ×2 (nearest), puis padding edge de 2
    up_q = cp.repeat(cp.repeat(q_coarse, DECIMATION, axis=-1),
                     DECIMATION, axis=-2)
    up_b = cp.repeat(cp.repeat(b_coarse, DECIMATION, axis=-1),
                     DECIMATION, axis=-2)
    up_q = cp.pad(up_q, ((0, 0), (0, 0), (0, 0), (2, 2), (2, 2)), mode="edge")
    up_b = cp.pad(up_b, ((0, 0), (0, 0), (2, 2), (2, 2)), mode="edge")
    # fenêtre padée autour de la fovéa : [oy-2 : oy+n+2] en coords fin,
    # décalée de +2 par le padding.
    y0, x0 = oy, ox
    halo_q = up_q[:, :, :, y0:y0 + nh, x0:x0 + nh].copy()
    halo_b = up_b[:, :, y0:y0 + nh, x0:x0 + nh].copy()
    # intérieur = la fovéa elle-même (le grossier n'y sert que d'anneau)
    halo_q[:, :, :, 2:n + 2, 2:n + 2] = q_fine
    halo_b[:, :, 2:n + 2, 2:n + 2] = cp.ascontiguousarray(b_fine)
    return cp.ascontiguousarray(halo_q), cp.ascontiguousarray(halo_b)


def _etage(kernel, q_in, q_base, sortie, b, halo_q, halo_b, dt, w_base,
           mode, cp, dx_maille: float = 1.0):
    """UN étage du kernel figé de C1 (une passe). L'opérateur intérieur est
    inchangé ; seuls le mode, le halo et le PAS D'ESPACE varient.

    `dx_maille` : la maille physique du niveau évolué (§A33-CORRECTION) —
    DECIMATION pour le grossier, 1.0 pour la fovéa."""
    n = int(q_in.shape[-1])
    total = int(q_in.shape[0] * q_in.shape[1] * n * n)
    grille = ((total + _TAILLE_BLOC - 1) // _TAILLE_BLOC,)
    kernel(grille, (_TAILLE_BLOC,),
           (q_in, q_base, sortie, b, halo_q, halo_b, np.float32(dt),
            np.float32(1.0 / dx_maille), np.float32(w_base), np.int32(n),
            np.int64(total), np.int32(mode)))


def pas_deux_niveaux(q_coarse, b_coarse, q_fine, b_fine, oy: int, ox: int,
                     cp, dt: float):
    """Un pas SSP-RK2 des DEUX niveaux en lockstep, avec le halo de la fovéa
    RAFRAÎCHI entre les étages depuis l'intermédiaire du grossier.

    Grossier : bord réfléchissant (vrai bord). Fovéa : halo-parent, refresh
    inter-étage. Retourne (q_coarse_suivant, q_fine_suivant).

    PAS D'ESPACE PAR NIVEAU (§A33-CORRECTION) : le grossier est évolué à
    Δx = DECIMATION, la fovéa à Δx = 1. C'est ce qui manquait — les deux
    niveaux tournaient à Δx = 1, donc le grossier avançait à ~2× sa vitesse
    physique alors même que le dt est partagé en lockstep."""
    kernel = _obtenir_kernel_fidele(cp)
    dx_c = float(DECIMATION)
    c_tampon, c_out = cp.empty_like(q_coarse), cp.empty_like(q_coarse)
    f_tampon, f_out = cp.empty_like(q_fine), cp.empty_like(q_fine)

    # ── étage 1 (w_base = 0) ──
    _etage(kernel, q_coarse, q_coarse, c_tampon, b_coarse, q_coarse,
           b_coarse, dt, 0.0, MODE_REFLECHISSANT, cp, dx_maille=dx_c)
    halo_q, halo_b = construire_halo(q_fine, b_fine, q_coarse, b_coarse,
                                     oy, ox, cp)
    _etage(kernel, q_fine, q_fine, f_tampon, b_fine, halo_q, halo_b, dt,
           0.0, MODE_HALO, cp, dx_maille=1.0)

    # ── REFRESH : halo de la fovéa depuis l'intermédiaire du grossier ──
    halo_q2, halo_b2 = construire_halo(f_tampon, b_fine, c_tampon, b_coarse,
                                       oy, ox, cp)

    # ── étage 2 (w_base = 0.5) ──
    _etage(kernel, c_tampon, q_coarse, c_out, b_coarse, c_tampon, b_coarse,
           dt, 0.5, MODE_REFLECHISSANT, cp, dx_maille=dx_c)
    _etage(kernel, f_tampon, q_fine, f_out, b_fine, halo_q2, halo_b2, dt,
           0.5, MODE_HALO, cp, dx_maille=1.0)
    return c_out, f_out
