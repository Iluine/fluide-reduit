"""F1 — M-a′ : F FUSIONNÉ (RawKernel CUDA) — borne d'implémentation du
MÊME motif E4a (§A15-lecture-M-a, gravé 2026-07-19, commit 6e9ee46).

Protocole gravé : « RawKernel CUDA fusionné du MÊME motif E4a (2 systèmes
(h, hu, hv, s), stencil wetdry O2 complet, 2 étages SSP-RK2, réduction CFL
payée non consommée, dt figé, bathymétrie plate). » Objectif : minimiser
les PASSES MÉMOIRE (le naïf CuPy matérialise des dizaines de temporaires
pleins par étage), PAS changer le motif de coût — toute omission de calcul
serait un harnais complaisant, interdit.

Ce que le fusionné change / ne change pas, NOMMÉ :
  - CHANGE : un seul kernel par étage RK2 ; chaque thread traite UNE
    cellule (ses 4 champs) et recalcule localement les pentes/états
    d'interface dont il a besoin (lecture d'un halo ±2, RE-CALCUL des flux
    partagés entre cellules voisines). Le fusionné RÉORDONNE et RE-CALCULE
    — il n'OMET rien : padding réfléchissant (miroir + négation de la qdm
    normale), désingularisation, pentes minmod x/y sur (η, u, v, c_s),
    réconciliation positivité, reconstruction MUSCL, flux HLL dry-aware,
    transport tangentiel upwind (v, u ET c_s), corrections de pression
    hydrostatiques (calculées — nulles à bathymétrie plate, comme dans le
    jetable), divergence, plancher sec par étage. Mêmes constantes
    (GRAVITE, DRY_EPS, DT_JETABLE, CFL) importées/reconduites du jetable.
  - NE CHANGE PAS : la réduction CFL par frame est PAYÉE (le même
    `reduction_cfl` du jetable, réutilisé — jamais réimplémenté) et non
    consommée ; dt figé ; 2 systèmes indépendants sous stencil complet
    (E4a) ; f32 de bout en bout.

ÉQUIVALENCE DE MOTIF (testée, tolérance f32 nommée ci-dessous) : sur une
même fenêtre, `pas_f_fusionne` et `pas_f_jetable` produisent le même état
à TOL_EQUIVALENCE près — l'écart résiduel est le réordonnancement f32
(associativité), pas une physique différente.

CAP GRAVÉ (§A15-lecture-M-a) : DERNIÈRE escalade d'implémentation — aucun
CUDA Graphs, aucun exotique au-delà de ce fichier."""
from __future__ import annotations

import numpy as np

from src.f1_gpu.substrat_jetable import (
    DRY_EPS,
    DT_JETABLE,
    GRAVITE,
    reduction_cfl,
)

# Tolérance d'équivalence de motif (nommée) : réordonnancement f32 pur —
# quelques ULP composés sur ~10² opérations par cellule et par étage.
TOL_EQUIVALENCE_RTOL: float = 1e-4
TOL_EQUIVALENCE_ATOL: float = 1e-5

_NOM_KERNEL: str = "etage_ssp_wetdry_o2"

# Le kernel : UN étage q_out = w_base·q_base + (1−w_base)·(q_in + dt·L(q_in))
# + plancher sec — w_base = 0 (étage 1), 0.5 (étage 2). Layout contigu
# (B, 2, 4, n, n) f32 ; un thread = une cellule d'un bloc (fenêtre, système).
_SOURCE: str = r"""
#define G """ + f"{GRAVITE}f" + r"""
#define DRY_EPS """ + f"{DRY_EPS}f" + r"""

struct Etat { float h, hu, hv, s; };
struct Pentes { float eta, u, v, cs; };

__device__ __forceinline__ float desing(float h, float hq) {
    return (h > DRY_EPS) ? hq / fmaxf(h, DRY_EPS) : 0.0f;
}

__device__ __forceinline__ float minmod(float a, float b) {
    if (a * b <= 0.0f) return 0.0f;
    float signe = (a > 0.0f) ? 1.0f : -1.0f;
    return signe * fminf(fabsf(a), fabsf(b));
}

/* Lecture avec cellules fantômes réfléchissantes : miroir bord (edge) +
   négation de la composante NORMALE de la qdm (motif _pad_reflexif). */
__device__ Etat lire(const float* q, long base, int n, int y, int x) {
    float sx = 1.0f, sy = 1.0f;
    if (x < 0)  { x = 0;     sx = -1.0f; }
    if (x >= n) { x = n - 1; sx = -1.0f; }
    if (y < 0)  { y = 0;     sy = -1.0f; }
    if (y >= n) { y = n - 1; sy = -1.0f; }
    long nn = (long)n * n;
    long idx = base + (long)y * n + x;
    Etat e;
    e.h  = q[idx];
    e.hu = sx * q[idx + nn];
    e.hv = sy * q[idx + 2 * nn];
    e.s  = q[idx + 3 * nn];
    return e;
}

/* Pente limitée de la cellule RÉELLE (y, x) le long de l'axe (dy, dx),
   sur (η, u, v, c_s), avec réconciliation positivité (b = 0 : la face
   h = η ∓ 0.5·s_η doit rester >= 0, sinon pentes nulles — retour O1). */
__device__ Pentes pentes_axe(const float* q, long base, int n,
                             int y, int x, int dy, int dx) {
    Etat c = lire(q, base, n, y, x);
    Etat m = lire(q, base, n, y - dy, x - dx);
    Etat p = lire(q, base, n, y + dy, x + dx);
    float uc = desing(c.h, c.hu), um = desing(m.h, m.hu), up = desing(p.h, p.hu);
    float vc = desing(c.h, c.hv), vm = desing(m.h, m.hv), vp = desing(p.h, p.hv);
    float cc = desing(c.h, c.s),  cm = desing(m.h, m.s),  cp_ = desing(p.h, p.s);
    Pentes s;
    s.eta = minmod(c.h - m.h, p.h - c.h);          /* η = h (b = 0) */
    s.u   = minmod(uc - um, up - uc);
    s.v   = minmod(vc - vm, vp - vc);
    s.cs  = minmod(cc - cm, cp_ - cc);
    if ((c.h - 0.5f * s.eta) < 0.0f || (c.h + 0.5f * s.eta) < 0.0f) {
        s.eta = 0.0f; s.u = 0.0f; s.v = 0.0f; s.cs = 0.0f;
    }
    return s;
}

__device__ Pentes pentes_nulles() {
    Pentes s; s.eta = 0.0f; s.u = 0.0f; s.v = 0.0f; s.cs = 0.0f; return s;
}

/* Flux d'interface 1D complet (motif _flux_hll + corrections + upwind) :
   états de face reconstruits (η, u_n, u_t, c_s) des deux côtés ; b = 0.
   Sorties : F_h, F_n vu du côté L (+S_L) et R (+S_R), F_t, F_s. */
__device__ void flux_1d(float etaL, float unL, float utL, float csL,
                        float etaR, float unR, float utR, float csR,
                        float* Fh, float* FnL_corr, float* FnR_corr,
                        float* Ft, float* Fs) {
    float hL = etaL, hR = etaR;              /* profondeur de face (b = 0) */
    float hsL = fmaxf(0.0f, etaL);           /* h* = max(0, η − z*), z* = 0 */
    float hsR = fmaxf(0.0f, etaR);
    float hqL = hsL * unL, hqR = hsR * unR;
    float uL = desing(hsL, hqL), uR = desing(hsR, hqR);
    float cL = sqrtf(G * fmaxf(hsL, 0.0f));
    float cR = sqrtf(G * fmaxf(hsR, 0.0f));
    float sL = (hsL > DRY_EPS) ? fminf(uL - cL, uR - cR) : uR - 2.0f * cR;
    float sR = (hsR > DRY_EPS) ? fmaxf(uL + cL, uR + cR) : uL + 2.0f * cL;
    float FnLp = hqL * uL + 0.5f * G * hsL * hsL;
    float FnRp = hqR * uR + 0.5f * G * hsR * hsR;
    float fh, fn;
    if (sL >= 0.0f)      { fh = hqL; fn = FnLp; }
    else if (sR <= 0.0f) { fh = hqR; fn = FnRp; }
    else {
        float den = sR - sL + 1e-30f;
        fh = (sR * hqL - sL * hqR + sL * sR * (hsR - hsL)) / den;
        fn = (sR * FnLp - sL * FnRp + sL * sR * (hqR - hqL)) / den;
    }
    *Fh = fh;
    /* Corrections de pression well-balanced — CALCULÉES (nulles à b = 0,
       comme dans le jetable : le motif paie l'arithmétique). */
    *FnL_corr = fn + 0.5f * G * (hL * hL - hsL * hsL);
    *FnR_corr = fn + 0.5f * G * (hR * hR - hsR * hsR);
    *Ft = (fh >= 0.0f) ? fh * utL : fh * utR;
    *Fs = (fh >= 0.0f) ? fh * csL : fh * csR;
}

/* Divergence 1D d'un axe pour la cellule courante : calcule les DEUX
   interfaces (gauche/bas et droite/haut) et rend les 4 contributions.
   normale = composante hu (axe x) ou hv (axe y) ; tangentielle = l'autre. */
__device__ void divergence_axe(const float* q, long base, int n,
                               int y, int x, int dy, int dx, int axe_x,
                               float* dh, float* dn, float* dt_, float* ds) {
    Etat c = lire(q, base, n, y, x);
    Etat m = lire(q, base, n, y - dy, x - dx);
    Etat p = lire(q, base, n, y + dy, x + dx);
    int m_reel = (axe_x ? (x - dx) : (y - dy)) >= 0;
    int p_reel = (axe_x ? (x + dx) : (y + dy)) < n;
    Pentes pc = pentes_axe(q, base, n, y, x, dy, dx);
    Pentes pm = m_reel ? pentes_axe(q, base, n, y - dy, x - dx, dy, dx)
                       : pentes_nulles();
    Pentes pp = p_reel ? pentes_axe(q, base, n, y + dy, x + dx, dy, dx)
                       : pentes_nulles();
    /* variables de face par cellule : normale/tangentielle selon l'axe */
    float un_c = axe_x ? desing(c.h, c.hu) : desing(c.h, c.hv);
    float ut_c = axe_x ? desing(c.h, c.hv) : desing(c.h, c.hu);
    float un_m = axe_x ? desing(m.h, m.hu) : desing(m.h, m.hv);
    float ut_m = axe_x ? desing(m.h, m.hv) : desing(m.h, m.hu);
    float un_p = axe_x ? desing(p.h, p.hu) : desing(p.h, p.hv);
    float ut_p = axe_x ? desing(p.h, p.hv) : desing(p.h, p.hu);
    float sn_c = axe_x ? pc.u : pc.v, st_c = axe_x ? pc.v : pc.u;
    float sn_m = axe_x ? pm.u : pm.v, st_m = axe_x ? pm.v : pm.u;
    float sn_p = axe_x ? pp.u : pp.v, st_p = axe_x ? pp.v : pp.u;
    float cs_c = desing(c.h, c.s), cs_m = desing(m.h, m.s), cs_p = desing(p.h, p.s);

    float Fh_d, FnL_d, FnR_d, Ft_d, Fs_d;     /* interface droite/haute (c|p) */
    flux_1d(c.h + 0.5f * pc.eta, un_c + 0.5f * sn_c,
            ut_c + 0.5f * st_c, cs_c + 0.5f * pc.cs,
            p.h - 0.5f * pp.eta, un_p - 0.5f * sn_p,
            ut_p - 0.5f * st_p, cs_p - 0.5f * pp.cs,
            &Fh_d, &FnL_d, &FnR_d, &Ft_d, &Fs_d);
    float Fh_g, FnL_g, FnR_g, Ft_g, Fs_g;     /* interface gauche/basse (m|c) */
    flux_1d(m.h + 0.5f * pm.eta, un_m + 0.5f * sn_m,
            ut_m + 0.5f * st_m, cs_m + 0.5f * pm.cs,
            c.h - 0.5f * pc.eta, un_c - 0.5f * sn_c,
            ut_c - 0.5f * st_c, cs_c - 0.5f * pc.cs,
            &Fh_g, &FnL_g, &FnR_g, &Ft_g, &Fs_g);
    /* motif divx/divy : normale = (F+S_L)@droite − (F+S_R)@gauche */
    *dh  = Fh_d - Fh_g;
    *dn  = FnL_d - FnR_g;
    *dt_ = Ft_d - Ft_g;
    *ds  = Fs_d - Fs_g;
}

extern "C" __global__ void """ + _NOM_KERNEL + r"""(
        const float* q_in, const float* q_base, float* q_out,
        const float dt, const float w_base, const int n, const long total) {
    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) return;
    long nn = (long)n * n;
    long bloc_ws = t / nn;               /* index (fenêtre, système) aplati */
    long base = bloc_ws * 4 * nn;        /* offset du canal 0 du bloc */
    int y = (int)((t % nn) / n);
    int x = (int)(t % n);

    float dhx, dnx, dtx, dsx, dhy, dny, dty, dsy;
    divergence_axe(q_in, base, n, y, x, 0, 1, 1, &dhx, &dnx, &dtx, &dsx);
    divergence_axe(q_in, base, n, y, x, 1, 0, 0, &dhy, &dny, &dty, &dsy);
    /* L : qdm x = normale-en-x + tangentielle-en-y ; qdm y symétrique */
    float Lh  = -(dhx + dhy);
    float Lhu = -(dnx + dty);
    float Lhv = -(dtx + dny);
    float Ls  = -(dsx + dsy);

    long idx = base + (long)y * n + x;
    Etat c;
    c.h  = q_in[idx];
    c.hu = q_in[idx + nn];
    c.hv = q_in[idx + 2 * nn];
    c.s  = q_in[idx + 3 * nn];
    float un = w_base * q_base[idx]          + (1.0f - w_base) * (c.h  + dt * Lh);
    float uu = w_base * q_base[idx + nn]     + (1.0f - w_base) * (c.hu + dt * Lhu);
    float uv = w_base * q_base[idx + 2 * nn] + (1.0f - w_base) * (c.hv + dt * Lhv);
    float us = w_base * q_base[idx + 3 * nn] + (1.0f - w_base) * (c.s  + dt * Ls);
    /* plancher sec (motif _plancher_sec) */
    un = fmaxf(un, 0.0f);
    if (un <= DRY_EPS) { un = 0.0f; uu = 0.0f; uv = 0.0f; us = 0.0f; }
    q_out[idx]          = un;
    q_out[idx + nn]     = uu;
    q_out[idx + 2 * nn] = uv;
    q_out[idx + 3 * nn] = us;
}
"""

_TAILLE_BLOC: int = 256
_cache_kernel = None


def _obtenir_kernel(cp):
    """Compile (une fois, NVRTC — au warmup du chrono, jamais en série)
    et met en cache le RawKernel."""
    global _cache_kernel
    if _cache_kernel is None:
        _cache_kernel = cp.RawKernel(_SOURCE, _NOM_KERNEL)
    return _cache_kernel


def pas_f_fusionne(q, cp, sortie=None, tampon_etage=None) -> tuple:
    """Un pas SSP-RK2 complet du F FUSIONNÉ sur (B, 2, 4, n, n) f32 :
    réduction CFL du jetable (payée, non consommée — RÉUTILISÉE, jamais
    réimplémentée) + 2 lancements du kernel fusionné + plancher sec
    in-kernel. `sortie is q` autorisé (le kernel ne lit `q_base` qu'à sa
    propre cellule) ; `tampon_etage` préallouable par l'appelant (B2).

    Retourne (q_suivant, dt_cfl_diagnostic) — signature du jetable."""
    if q.dtype != cp.float32:
        raise ValueError(f"pas_f_fusionne : dtype {q.dtype} != float32.")
    if q.ndim != 5 or q.shape[2] != 4 or q.shape[1] != 2:
        raise ValueError(
            f"pas_f_fusionne : shape {q.shape} != (B, 2, 4, n, n) (E4a).")
    q = cp.ascontiguousarray(q)
    dt_cfl = reduction_cfl(q, cp)

    n = int(q.shape[-1])
    total = int(q.shape[0] * q.shape[1] * n * n)
    if tampon_etage is None:
        tampon_etage = cp.empty_like(q)
    if sortie is None:
        sortie = cp.empty_like(q)
    kernel = _obtenir_kernel(cp)
    grille = ((total + _TAILLE_BLOC - 1) // _TAILLE_BLOC,)
    arguments_communs = (np.float32(DT_JETABLE), np.int32(n), np.int64(total))
    kernel(grille, (_TAILLE_BLOC,),
           (q, q, tampon_etage, arguments_communs[0], np.float32(0.0),
            arguments_communs[1], arguments_communs[2]))
    kernel(grille, (_TAILLE_BLOC,),
           (tampon_etage, q, sortie, arguments_communs[0], np.float32(0.5),
            arguments_communs[1], arguments_communs[2]))
    return sortie, dt_cfl
