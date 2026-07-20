"""F1 — M-b tranche-1, composant C1 : F FIDÈLE b-aware (_SOURCE_FIDELE).

Ce que ce module EST, et ce qu'il n'est PAS. C'est le portage GPU de
l'opérateur bien équilibré d'Audusse `_rhs_o2` (src/solver_wetdry.py) à
BATHYMÉTRIE RÉELLE `b_eff = b0 + s` — le MOTEUR de M-b, pas un motif de
coût. Le kernel figé (`substrat_fusionne._SOURCE`) est plat (b = 0, η = h,
z* = 0, corrections hydrostatiques structurellement nulles) et sa signature
ne prend AUCUNE bathymétrie ; il ne peut donc pas porter la physique réelle.
C'est écrit AVANT toute mesure : l'ancre 12,655 est un PROXY (§A28).

CE QUI EST RÉUTILISÉ, ET CE QUI EST NEUF. Les helpers device
B-AGNOSTIQUES du préfixe figé — `desing`, `minmod`, les `#define`, les
structs — sont repris TELS QUELS (`_PREFIXE_BASE`, extrait de `_SOURCE`,
comme L3 a repris son préfixe). Le kernel figé n'est pas touché (diff vide
prouvable). Sont NEUFS, parce que b-aware : la lecture (`lire_b`), les
pentes sur η = h + b (`pentes_axe_b`), le flux d'Audusse (`flux_1d_b` :
z* = max(b_L, b_R), h* = max(0, η − z*), pression (g/2)(h² − h*²)), la
divergence (`divergence_axe_b`), et l'étage global.

SÉMANTIQUE DES CHAMPS, différente du jetable (§A28) : l'état évolué est
(h, hu, hv) — 3 champs ; `s` N'EST PAS advecté ici (c'est le sédiment, mis
à jour par Exner en C2). `b_eff = b0 + s` est une entrée CONSTANTE du pas.

BORD PLUGGABLE dès C1 (§A28, invariant liant) : le remplissage de halo est
un MODE de données, pas un fork de code. `mode = 0` réfléchissant (mur, t1) ;
`mode = 1` halo-parent (t2, lu depuis un tableau padé). L'opérateur
INTÉRIEUR — pentes, flux, divergence — est le MÊME objet dans les deux
modes ; seule `lire_b` choisit sa source. Prouvé par test : mode 1 avec un
halo rempli à l'identique du réfléchissant reproduit mode 0 bit à bit.

dt = temps de frame, un pas/frame, M = 1 (mipmap gravé §A28) : le pas est
sous-CFL au niveau fin, pas de sous-cyclage. La CFL est CALCULÉE
(`reduction_cfl_fidele`, pour la vérification de régime de C4) et NON
consommée pour sous-pas.

VERROU D'EMPREINTE dès le premier commit (§A29-C) : on ne refait pas le
coup de L3, resté sans verrou alors qu'il portait `reference += d`.
`_SOURCE_FIDELE` a son sha256 et sa longueur figés ci-dessous ; un test
échoue si le CUDA bouge."""
from __future__ import annotations

import hashlib

import numpy as np

from src.f1_gpu.substrat_fusionne import _SOURCE as _SOURCE_FUSIONNE
from src.f1_gpu.substrat_jetable import DRY_EPS, GRAVITE

# Réutilisation STRUCTURELLE du préfixe b-agnostique du figé : #define
# (G, DRY_EPS), struct Etat/Pentes, `desing`, `minmod` — tout ce qui
# précède la première fonction b=0 (`lire`). Le figé n'est pas modifié.
_MARQUEUR_B0: str = "/* Lecture avec cellules"
_PREFIXE_BASE: str = _SOURCE_FUSIONNE[: _SOURCE_FUSIONNE.index(_MARQUEUR_B0)]

_NOM_KERNEL_FIDELE: str = "etage_ssp_wetdry_o2_fidele"

# Corps b-aware NEUF : bord pluggable (mode 0 réfléchissant / mode 1 halo),
# pentes sur η = h + b, flux d'Audusse bien équilibré. Halo padé de 2
# cellules (stencil ±2) : (B, S, 3|1, n+4, n+4).
_CORPS_FIDELE: str = r"""
struct EtatB { float h, hu, hv, b; };
struct Pentes3 { float eta, u, v; };

__device__ __forceinline__ Pentes3 pentes3_nulles() {
    Pentes3 s; s.eta = 0.0f; s.u = 0.0f; s.v = 0.0f; return s;
}

/* Lecture (h, hu, hv, b) à (y, x). mode 0 : cellules fantômes
   RÉFLÉCHISSANTES (miroir edge + négation de la qdm NORMALE pour h,hu,hv ;
   copie edge pour b — un lit n'a pas de signe). mode 1 : lecture depuis un
   HALO padé de 2 (aucune réflexion — le bord est une continuation vers le
   grossier, pas un mur). L'invariant liant tient : l'intérieur ne change
   pas, seule la SOURCE change. */
__device__ EtatB lire_b(const float* q, const float* bb,
                        const float* halo_q, const float* halo_b,
                        long bloc, int n, int y, int x, int mode) {
    long nn = (long)n * n;
    if (mode == 0) {
        float sx = 1.0f, sy = 1.0f;
        if (x < 0)  { x = 0;     sx = -1.0f; }
        if (x >= n) { x = n - 1; sx = -1.0f; }
        if (y < 0)  { y = 0;     sy = -1.0f; }
        if (y >= n) { y = n - 1; sy = -1.0f; }
        long baseq = bloc * 3 * nn;
        long idx = baseq + (long)y * n + x;
        EtatB e;
        e.h  = q[idx];
        e.hu = sx * q[idx + nn];
        e.hv = sy * q[idx + 2 * nn];
        e.b  = bb[bloc * nn + (long)y * n + x];
        return e;
    }
    /* mode 1 : halo padé de 2, indexé (y+2, x+2), sans réflexion */
    int nh = n + 4;
    long nhh = (long)nh * nh;
    long baseq = bloc * 3 * nhh;
    long idx = baseq + (long)(y + 2) * nh + (x + 2);
    EtatB e;
    e.h  = halo_q[idx];
    e.hu = halo_q[idx + nhh];
    e.hv = halo_q[idx + 2 * nhh];
    e.b  = halo_b[bloc * nhh + (long)(y + 2) * nh + (x + 2)];
    return e;
}

/* Pentes limitées sur η = h + b (surface-gradient), (u, v) désingularisés,
   + réconciliation positivité : si une face de la cellule donne
   h reconstruit = (η ± 0.5 s_η) − b < 0, on annule la pente (retour O1).
   b au CENTRE de la cellule (non reconstruit, comme Audusse/_rhs_o2). */
__device__ Pentes3 pentes_axe_b(const float* q, const float* bb,
                                const float* halo_q, const float* halo_b,
                                long bloc, int n, int y, int x,
                                int dy, int dx, int mode) {
    EtatB c = lire_b(q, bb, halo_q, halo_b, bloc, n, y, x, mode);
    EtatB m = lire_b(q, bb, halo_q, halo_b, bloc, n, y - dy, x - dx, mode);
    EtatB p = lire_b(q, bb, halo_q, halo_b, bloc, n, y + dy, x + dx, mode);
    float ec = c.h + c.b, em = m.h + m.b, ep = p.h + p.b;
    float uc = desing(c.h, c.hu), um = desing(m.h, m.hu), up = desing(p.h, p.hu);
    float vc = desing(c.h, c.hv), vm = desing(m.h, m.hv), vp = desing(p.h, p.hv);
    Pentes3 s;
    s.eta = minmod(ec - em, ep - ec);
    s.u   = minmod(uc - um, up - uc);
    s.v   = minmod(vc - vm, vp - vc);
    float hLf = (ec - 0.5f * s.eta) - c.b;
    float hRf = (ec + 0.5f * s.eta) - c.b;
    if (hLf < 0.0f || hRf < 0.0f) { s.eta = 0.0f; s.u = 0.0f; s.v = 0.0f; }
    return s;
}

/* Flux d'interface 1D d'Audusse bien équilibré (états de face η, u_n, u_t
   + lits b_L, b_R des DEUX cellules). z* = max(b_L, b_R) ; h* = max(0, η −
   z*) ; profondeur réelle h = η − b ; HLL sur (h*, h*·u_n) ; correction de
   pression (g/2)(h² − h*²) versée séparément côté L et côté R (c'est elle
   qui équilibre le flux hydrostatique au repos — la C-property). */
__device__ void flux_1d_b(float etaL, float unL, float utL, float bL,
                          float etaR, float unR, float utR, float bR,
                          float* Fh, float* FnL_corr, float* FnR_corr,
                          float* Ft) {
    float zstar = fmaxf(bL, bR);
    float hL = etaL - bL, hR = etaR - bR;
    float hsL = fmaxf(0.0f, etaL - zstar);
    float hsR = fmaxf(0.0f, etaR - zstar);
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
    *FnL_corr = fn + 0.5f * G * (hL * hL - hsL * hsL);
    *FnR_corr = fn + 0.5f * G * (hR * hR - hsR * hsR);
    *Ft = (fh >= 0.0f) ? fh * utL : fh * utR;
}

/* Divergence 1D b-aware d'un axe : deux interfaces (gauche/bas, droite/haut).
   normale = hu (axe x) ou hv (axe y) ; tangentielle = l'autre. */
__device__ void divergence_axe_b(const float* q, const float* bb,
                                 const float* halo_q, const float* halo_b,
                                 long bloc, int n, int y, int x,
                                 int dy, int dx, int axe_x, int mode,
                                 float* dh, float* dn, float* dt_) {
    EtatB c = lire_b(q, bb, halo_q, halo_b, bloc, n, y, x, mode);
    EtatB m = lire_b(q, bb, halo_q, halo_b, bloc, n, y - dy, x - dx, mode);
    EtatB p = lire_b(q, bb, halo_q, halo_b, bloc, n, y + dy, x + dx, mode);
    int m_reel = (axe_x ? (x - dx) : (y - dy)) >= 0;
    int p_reel = (axe_x ? (x + dx) : (y + dy)) < n;
    Pentes3 pc = pentes_axe_b(q, bb, halo_q, halo_b, bloc, n, y, x, dy, dx, mode);
    Pentes3 pm = m_reel ? pentes_axe_b(q, bb, halo_q, halo_b, bloc, n,
                                       y - dy, x - dx, dy, dx, mode)
                        : pentes3_nulles();
    Pentes3 pp = p_reel ? pentes_axe_b(q, bb, halo_q, halo_b, bloc, n,
                                       y + dy, x + dx, dy, dx, mode)
                        : pentes3_nulles();
    float ec = c.h + c.b, em = m.h + m.b, ep = p.h + p.b;
    float un_c = axe_x ? desing(c.h, c.hu) : desing(c.h, c.hv);
    float ut_c = axe_x ? desing(c.h, c.hv) : desing(c.h, c.hu);
    float un_m = axe_x ? desing(m.h, m.hu) : desing(m.h, m.hv);
    float ut_m = axe_x ? desing(m.h, m.hv) : desing(m.h, m.hu);
    float un_p = axe_x ? desing(p.h, p.hu) : desing(p.h, p.hv);
    float ut_p = axe_x ? desing(p.h, p.hv) : desing(p.h, p.hu);
    float sn_c = axe_x ? pc.u : pc.v, st_c = axe_x ? pc.v : pc.u;
    float sn_m = axe_x ? pm.u : pm.v, st_m = axe_x ? pm.v : pm.u;
    float sn_p = axe_x ? pp.u : pp.v, st_p = axe_x ? pp.v : pp.u;

    float Fh_d, FnL_d, FnR_d, Ft_d;      /* interface droite/haute (c|p) */
    flux_1d_b(ec + 0.5f * pc.eta, un_c + 0.5f * sn_c, ut_c + 0.5f * st_c, c.b,
              ep - 0.5f * pp.eta, un_p - 0.5f * sn_p, ut_p - 0.5f * st_p, p.b,
              &Fh_d, &FnL_d, &FnR_d, &Ft_d);
    float Fh_g, FnL_g, FnR_g, Ft_g;      /* interface gauche/basse (m|c) */
    flux_1d_b(em + 0.5f * pm.eta, un_m + 0.5f * sn_m, ut_m + 0.5f * st_m, m.b,
              ec - 0.5f * pc.eta, un_c - 0.5f * sn_c, ut_c - 0.5f * st_c, c.b,
              &Fh_g, &FnL_g, &FnR_g, &Ft_g);
    *dh  = Fh_d - Fh_g;
    *dn  = FnL_d - FnR_g;
    *dt_ = Ft_d - Ft_g;
}

extern "C" __global__ void """ + _NOM_KERNEL_FIDELE + r"""(
        const float* q_in, const float* q_base, float* q_out,
        const float* b, const float* halo_q, const float* halo_b,
        const float dt, const float w_base, const int n, const long total,
        const int mode) {
    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) return;
    long nn = (long)n * n;
    long bloc = t / nn;                 /* index (fenêtre, système) aplati */
    long baseq = bloc * 3 * nn;
    int y = (int)((t % nn) / n);
    int x = (int)(t % n);

    float dhx, dnx, dtx, dhy, dny, dty;
    divergence_axe_b(q_in, b, halo_q, halo_b, bloc, n, y, x, 0, 1, 1, mode,
                     &dhx, &dnx, &dtx);
    divergence_axe_b(q_in, b, halo_q, halo_b, bloc, n, y, x, 1, 0, 0, mode,
                     &dhy, &dny, &dty);
    float Lh  = -(dhx + dhy);
    float Lhu = -(dnx + dty);           /* qdm x : normale-x + tangentielle-y */
    float Lhv = -(dtx + dny);           /* qdm y symétrique */

    long idx = baseq + (long)y * n + x;
    float h  = q_in[idx];
    float hu = q_in[idx + nn];
    float hv = q_in[idx + 2 * nn];
    float un = w_base * q_base[idx]          + (1.0f - w_base) * (h  + dt * Lh);
    float uu = w_base * q_base[idx + nn]     + (1.0f - w_base) * (hu + dt * Lhu);
    float uv = w_base * q_base[idx + 2 * nn] + (1.0f - w_base) * (hv + dt * Lhv);
    un = fmaxf(un, 0.0f);
    if (un <= DRY_EPS) { un = 0.0f; uu = 0.0f; uv = 0.0f; }
    q_out[idx]          = un;
    q_out[idx + nn]     = uu;
    q_out[idx + 2 * nn] = uv;
}
"""

_SOURCE_FIDELE: str = _PREFIXE_BASE + _CORPS_FIDELE

# ── VERROU D'EMPREINTE (§A29-C) — dès le premier commit ──────────────────
# sha256 de la CHAÎNE EXTRAITE, longueur figée à côté (recalculable). Le
# CUDA ne bouge pas sans que ce verrou parle.
LONGUEUR_SOURCE_FIDELE: int = len(_SOURCE_FIDELE)
SHA256_SOURCE_FIDELE: str = hashlib.sha256(
    _SOURCE_FIDELE.encode()).hexdigest()

MODE_REFLECHISSANT: int = 0        # mur (tranche-1)
MODE_HALO: int = 1                 # halo-parent (tranche-2)

_TAILLE_BLOC: int = 256
_cache_kernel_fidele = None


def _obtenir_kernel_fidele(cp):
    """Compile (une fois, NVRTC — au warmup, jamais en série) et met en
    cache le RawKernel fidèle."""
    global _cache_kernel_fidele
    if _cache_kernel_fidele is None:
        _cache_kernel_fidele = cp.RawKernel(_SOURCE_FIDELE, _NOM_KERNEL_FIDELE)
    return _cache_kernel_fidele


def reduction_cfl_fidele(q, cp) -> float:
    """CFL du substrat fidèle : dt = 0.4·min(dx,dy)/max(|u|+c, |v|+c),
    dx=dy=1, c=sqrt(g·h). CALCULÉE, non consommée pour sous-pas (M=1). Sert
    la vérification de régime (C4, marge 4–12×)."""
    h = cp.maximum(q[:, :, 0], 0.0)
    hu, hv = q[:, :, 1], q[:, :, 2]
    c = cp.sqrt(cp.float32(GRAVITE) * h)
    sec = h <= cp.float32(DRY_EPS)
    u = cp.where(sec, cp.float32(0.0), hu / cp.maximum(h, cp.float32(DRY_EPS)))
    v = cp.where(sec, cp.float32(0.0), hv / cp.maximum(h, cp.float32(DRY_EPS)))
    smax = float(cp.maximum(cp.abs(u) + c, cp.abs(v) + c).max())
    return 0.4 / max(smax, 1e-12)


def _valider(q, b, cp) -> None:
    if q.dtype != cp.float32 or b.dtype != cp.float32:
        raise ValueError("pas_f_fidele : q et b doivent être float32.")
    if q.ndim != 5 or q.shape[2] != 3:
        raise ValueError(
            f"pas_f_fidele : q shape {q.shape} != (B, S, 3, n, n) — l'état "
            "évolué est (h, hu, hv), s n'est PAS advecté (§A28).")
    n = q.shape[-1]
    if b.shape != (q.shape[0], q.shape[1], n, n):
        raise ValueError(
            f"pas_f_fidele : b shape {b.shape} != {(q.shape[0], q.shape[1], n, n)} "
            "— b_eff = b0 + s, un lit par (fenêtre, système).")


def pas_f_fidele(q, b, cp, dt: float, sortie=None, tampon_etage=None,
                 mode: int = MODE_REFLECHISSANT, halo_q=None, halo_b=None
                 ) -> tuple:
    """Un pas SSP-RK2 du F FIDÈLE b-aware sur (B, S, 3, n, n) f32, lit
    `b_eff` (B, S, n, n) CONSTANT. Retourne (sortie, dt_cfl).

    `mode` : MODE_REFLECHISSANT (mur, t1) ou MODE_HALO (halo-parent, t2).
    En MODE_HALO, `halo_q`/`halo_b` sont les tableaux padés de 2 cellules
    ((B, S, 3|1, n+4, n+4)) ; l'opérateur intérieur est le MÊME (prouvé
    bit-exact sur un étage). En MODE_REFLECHISSANT ils sont ignorés (des
    vues de q/b suffisent).

    CE QUE C1 IMPOSE À TRANCHE-2, NOMMÉ : ce wrapper fait DEUX étages RK2
    avec le MÊME halo. En MODE_REFLECHISSANT c'est correct — la réflexion
    est recalculée à chaque étage depuis l'état courant (`lire_b` reflète
    q_in/tampon). En MODE_HALO le halo passé ici serait PÉRIMÉ au second
    étage (il décrit l'état d'entrée, pas l'intermédiaire). Tranche-2 doit
    donc RAFRAÎCHIR le halo-parent entre les deux étages — la couture est
    là, pas dans l'opérateur intérieur. C'est la charge que le bord de
    tranche-1 (réfléchissant, self-consistant) transfère à tranche-2."""
    _valider(q, b, cp)
    q = cp.ascontiguousarray(q)
    b = cp.ascontiguousarray(b)
    dt_cfl = reduction_cfl_fidele(q, cp)

    n = int(q.shape[-1])
    total = int(q.shape[0] * q.shape[1] * n * n)
    if tampon_etage is None:
        tampon_etage = cp.empty_like(q)
    if sortie is None:
        sortie = cp.empty_like(q)
    if mode == MODE_REFLECHISSANT:
        # halo inutilisé — on passe q/b (jamais lus dans ce mode).
        halo_q, halo_b = q, b
    elif halo_q is None or halo_b is None:
        raise ValueError("pas_f_fidele : MODE_HALO exige halo_q et halo_b.")
    halo_q = cp.ascontiguousarray(halo_q)
    halo_b = cp.ascontiguousarray(halo_b)

    kernel = _obtenir_kernel_fidele(cp)
    grille = ((total + _TAILLE_BLOC - 1) // _TAILLE_BLOC,)
    dt_arg, n_arg, tot_arg = np.float32(dt), np.int32(n), np.int64(total)
    mode_arg = np.int32(mode)
    kernel(grille, (_TAILLE_BLOC,),
           (q, q, tampon_etage, b, halo_q, halo_b, dt_arg, np.float32(0.0),
            n_arg, tot_arg, mode_arg))
    kernel(grille, (_TAILLE_BLOC,),
           (tampon_etage, q, sortie, b, halo_q, halo_b, dt_arg,
            np.float32(0.5), n_arg, tot_arg, mode_arg))
    return sortie, dt_cfl
