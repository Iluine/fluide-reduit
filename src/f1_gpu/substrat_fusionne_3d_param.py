"""F 3D — M-c : F FUSIONNÉ 3D **PARAMÉTRÉ** en scalaires advectés et en
champs statiques (pré-enregistrement
`claude/prereg-multiplicateur-c-3d-2026-08-04.md`, commit `c298a8b`,
écrit AVANT ce fichier).

CE QU'IL MESURE : le coût marginal d'un scalaire advecté et celui d'un
champ statique lu, en 3D — les deux écarts entre le jouet à cinq champs
de §A53 et le vocabulaire réel de §A51 (7,5 éq-f32).

LAYOUT (B, S, C, n, n, n) f32, `C = 4 + NS + NST` :
  champs 0..3          hyperboliques (h, hu, hv, hw) — le noyau, invariant
  champs 4 .. 4+NS-1   scalaires ADVECTÉS (s, e_th, ρ_s…)
  champs 4+NS ..       champs STATIQUES (b0, id-matériau) — LUS, jamais avancés

**LE KERNEL DE §A53 N'EST PAS TOUCHÉ.** Il est le dénominateur de ρ_c ;
son empreinte est figée dans les tests. Ce module est un fichier neuf, et
son verrou d'entrée est plus fort qu'une reproduction statistique : à
`(NS=1, NST=0)` il doit produire une sortie **BIT POUR BIT IDENTIQUE** à
`pas_f_fusionne_3d`. Si la paramétrisation a changé quoi que ce soit au
motif, ce test tombe et aucun chronomètre ne part.

DEUX CHOIX D'IMPLÉMENTATION, DÉCLARÉS PARCE QU'ILS DÉPLACENT LE CHIFFRE :

**(1) Les statiques sont lus dans le HALO**, comme tous les autres champs
— 36 lectures par cellule et par étage. C'est **fidèle pour `b0`**, qui
est une grandeur de FACE (la bathymétrie entre dans la reconstruction des
états d'interface) ; c'est un **MAJORANT pour `id-matériau`**, qu'un
kernel de production lirait une seule fois au centre. Majorant de coût ⇒
minorant du cap : la réduction est nommée, non mesurée.

**(2) Les statiques sont repliés dans la sortie avec un POIDS PASSÉ EN
ARGUMENT, valant 0.0f.** Sans cela, `nvcc` supprimerait purement et
simplement des lectures dont rien ne dépend, et le chronomètre mesurerait
un champ qui n'existe pas. Le compilateur ne peut pas savoir que
l'argument vaut zéro ; la lecture est donc réelle. Et `0.0f * x + y == y`
exactement pour tout `x` fini : **le résultat numérique est inchangé**.
Le surcoût ajouté est d'un FMA par statique et par cellule — déclaré,
minuscule devant une lecture, et il va dans le sens du majorant.

C'est l'application directe de la garde I-c3 du prereg : *un δ nul et une
élimination par le compilateur sont indistinguables au chronomètre.*

Le cap d'implémentation de M-a′ est reconduit : aucun CUDA Graphs, aucun
exotique, aucune optimisation que le kernel de §A53 n'ait reçue.
`substrat_fusionne.py:8-9` s'applique sans changement de mot : « toute
omission de calcul serait un harnais complaisant, interdit »."""
from __future__ import annotations

import hashlib

import numpy as np

from src.f1_gpu.substrat_jetable import DRY_EPS, DT_JETABLE, GRAVITE
from src.f1_gpu.substrat_jetable_3d import (
    CHAMPS_HYPERBOLIQUES,
    decompte_champs,
    reduction_cfl_3d,
)

__all__ = ["pas_f_fusionne_3d_param", "source_param", "empreinte_param",
           "attributs_param", "comptes_statiques_param"]

_TAILLE_BLOC: int = 256          # identique à M-a′ et à §A53
_cache: dict[tuple[int, int], object] = {}


def _nom_kernel(ns: int, nst: int) -> str:
    return f"etage_ssp_wetdry_o2_3d_ns{ns}_nst{nst}"


def source_param(ns: int, nst: int) -> str:
    """Source CUDA pour `ns` scalaires advectés et `nst` champs statiques.

    Le corps est celui de `substrat_fusionne_3d._SOURCE_3D`, avec les
    scalaires en boucles `#pragma unroll` de borne connue à la
    compilation : mêmes opérations, même ordre, donc même résultat
    flottant à `ns = 1`."""
    if ns < 1 or nst < 0:
        raise ValueError(f"source_param : ns={ns} < 1 ou nst={nst} < 0.")
    nc = CHAMPS_HYPERBOLIQUES + ns + nst
    nom = _nom_kernel(ns, nst)
    champ_st = f"    float st[{nst}];\n" if nst else ""
    lire_st = "".join(
        f"    e.st[{i}] = q[idx + {CHAMPS_HYPERBOLIQUES + ns + i} * nnn];\n"
        for i in range(nst))
    somme_st = ("    float somme_st = 0.0f;\n"
                + "".join(f"    somme_st += c.st[{i}];\n" for i in range(nst))
                ) if nst else ""
    repli_st = ""
    if nst:
        repli_st = (
            "    /* Repli des STATIQUES : poids passé en argument, valant\n"
            "       0.0f. Le compilateur ne peut pas le savoir, donc il ne\n"
            "       supprime pas les lectures ; et 0*x + y == y exactement\n"
            "       pour x fini, donc le résultat est inchangé. */\n"
            "    un += poids_statique * somme_st;\n")
    return r"""
#define G """ + f"{GRAVITE}f" + r"""
#define DRY_EPS """ + f"{DRY_EPS}f" + r"""
#define NS """ + str(ns) + r"""
#define NC """ + str(nc) + r"""

struct Etat { float h, hu, hv, hw; float sc[NS];
""" + champ_st + r"""};
struct Pentes { float eta, u, v, w; float cs[NS]; };

__device__ __forceinline__ float desing(float h, float hq) {
    return (h > DRY_EPS) ? hq / fmaxf(h, DRY_EPS) : 0.0f;
}

__device__ __forceinline__ float minmod(float a, float b) {
    if (a * b <= 0.0f) return 0.0f;
    float signe = (a > 0.0f) ? 1.0f : -1.0f;
    return signe * fminf(fabsf(a), fabsf(b));
}

__device__ Etat lire(const float* q, long base, int n, int z, int y, int x) {
    float sx = 1.0f, sy = 1.0f, sz = 1.0f;
    if (x < 0)  { x = 0;     sx = -1.0f; }
    if (x >= n) { x = n - 1; sx = -1.0f; }
    if (y < 0)  { y = 0;     sy = -1.0f; }
    if (y >= n) { y = n - 1; sy = -1.0f; }
    if (z < 0)  { z = 0;     sz = -1.0f; }
    if (z >= n) { z = n - 1; sz = -1.0f; }
    long nnn = (long)n * n * n;
    long idx = base + ((long)z * n + y) * n + x;
    Etat e;
    e.h  = q[idx];
    e.hu = sx * q[idx + nnn];
    e.hv = sy * q[idx + 2 * nnn];
    e.hw = sz * q[idx + 3 * nnn];
#pragma unroll
    for (int i = 0; i < NS; ++i) e.sc[i] = q[idx + (4 + i) * nnn];
""" + lire_st + r"""    return e;
}

__device__ Pentes pentes_axe(const float* q, long base, int n,
                             int z, int y, int x, int dz, int dy, int dx) {
    Etat c = lire(q, base, n, z, y, x);
    Etat m = lire(q, base, n, z - dz, y - dy, x - dx);
    Etat p = lire(q, base, n, z + dz, y + dy, x + dx);
    float uc = desing(c.h, c.hu), um = desing(m.h, m.hu), up = desing(p.h, p.hu);
    float vc = desing(c.h, c.hv), vm = desing(m.h, m.hv), vp = desing(p.h, p.hv);
    float wc = desing(c.h, c.hw), wm = desing(m.h, m.hw), wp = desing(p.h, p.hw);
    Pentes s;
    s.eta = minmod(c.h - m.h, p.h - c.h);
    s.u   = minmod(uc - um, up - uc);
    s.v   = minmod(vc - vm, vp - vc);
    s.w   = minmod(wc - wm, wp - wc);
#pragma unroll
    for (int i = 0; i < NS; ++i) {
        float cc = desing(c.h, c.sc[i]);
        float cm = desing(m.h, m.sc[i]);
        float cp_ = desing(p.h, p.sc[i]);
        s.cs[i] = minmod(cc - cm, cp_ - cc);
    }
    if ((c.h - 0.5f * s.eta) < 0.0f || (c.h + 0.5f * s.eta) < 0.0f) {
        s.eta = 0.0f; s.u = 0.0f; s.v = 0.0f; s.w = 0.0f;
#pragma unroll
        for (int i = 0; i < NS; ++i) s.cs[i] = 0.0f;
    }
    return s;
}

__device__ Pentes pentes_nulles() {
    Pentes s; s.eta = 0.0f; s.u = 0.0f; s.v = 0.0f; s.w = 0.0f;
#pragma unroll
    for (int i = 0; i < NS; ++i) s.cs[i] = 0.0f;
    return s;
}

__device__ void flux_1d(float etaL, float unL, float ut1L, float ut2L,
                        const float* csL,
                        float etaR, float unR, float ut1R, float ut2R,
                        const float* csR,
                        float* Fh, float* FnL_corr, float* FnR_corr,
                        float* Ft1, float* Ft2, float* Fs) {
    float hL = etaL, hR = etaR;
    float hsL = fmaxf(0.0f, etaL);
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
    *FnL_corr = fn + 0.5f * G * (hL * hL - hsL * hsL);
    *FnR_corr = fn + 0.5f * G * (hR * hR - hsR * hsR);
    *Ft1 = (fh >= 0.0f) ? fh * ut1L : fh * ut1R;
    *Ft2 = (fh >= 0.0f) ? fh * ut2L : fh * ut2R;
#pragma unroll
    for (int i = 0; i < NS; ++i)
        Fs[i] = (fh >= 0.0f) ? fh * csL[i] : fh * csR[i];
}

__device__ void divergence_axe(const float* q, long base, int n,
                               int z, int y, int x,
                               int dz, int dy, int dx, int axe,
                               float* dh, float* dn, float* dt1, float* dt2,
                               float* ds) {
    Etat c = lire(q, base, n, z, y, x);
    Etat m = lire(q, base, n, z - dz, y - dy, x - dx);
    Etat p = lire(q, base, n, z + dz, y + dy, x + dx);
    int coord = (axe == 0) ? x : ((axe == 1) ? y : z);
    int m_reel = (coord - 1) >= 0;
    int p_reel = (coord + 1) < n;
    Pentes pc = pentes_axe(q, base, n, z, y, x, dz, dy, dx);
    Pentes pm = m_reel ? pentes_axe(q, base, n, z - dz, y - dy, x - dx,
                                    dz, dy, dx) : pentes_nulles();
    Pentes pp = p_reel ? pentes_axe(q, base, n, z + dz, y + dy, x + dx,
                                    dz, dy, dx) : pentes_nulles();

    float un_c, ut1_c, ut2_c, un_m, ut1_m, ut2_m, un_p, ut1_p, ut2_p;
    float sn_c, st1_c, st2_c, sn_m, st1_m, st2_m, sn_p, st1_p, st2_p;
    if (axe == 0) {
        un_c = desing(c.h, c.hu); ut1_c = desing(c.h, c.hv); ut2_c = desing(c.h, c.hw);
        un_m = desing(m.h, m.hu); ut1_m = desing(m.h, m.hv); ut2_m = desing(m.h, m.hw);
        un_p = desing(p.h, p.hu); ut1_p = desing(p.h, p.hv); ut2_p = desing(p.h, p.hw);
        sn_c = pc.u; st1_c = pc.v; st2_c = pc.w;
        sn_m = pm.u; st1_m = pm.v; st2_m = pm.w;
        sn_p = pp.u; st1_p = pp.v; st2_p = pp.w;
    } else if (axe == 1) {
        un_c = desing(c.h, c.hv); ut1_c = desing(c.h, c.hu); ut2_c = desing(c.h, c.hw);
        un_m = desing(m.h, m.hv); ut1_m = desing(m.h, m.hu); ut2_m = desing(m.h, m.hw);
        un_p = desing(p.h, p.hv); ut1_p = desing(p.h, p.hu); ut2_p = desing(p.h, p.hw);
        sn_c = pc.v; st1_c = pc.u; st2_c = pc.w;
        sn_m = pm.v; st1_m = pm.u; st2_m = pm.w;
        sn_p = pp.v; st1_p = pp.u; st2_p = pp.w;
    } else {
        un_c = desing(c.h, c.hw); ut1_c = desing(c.h, c.hu); ut2_c = desing(c.h, c.hv);
        un_m = desing(m.h, m.hw); ut1_m = desing(m.h, m.hu); ut2_m = desing(m.h, m.hv);
        un_p = desing(p.h, p.hw); ut1_p = desing(p.h, p.hu); ut2_p = desing(p.h, p.hv);
        sn_c = pc.w; st1_c = pc.u; st2_c = pc.v;
        sn_m = pm.w; st1_m = pm.u; st2_m = pm.v;
        sn_p = pp.w; st1_p = pp.u; st2_p = pp.v;
    }
    float cs_c[NS], cs_m[NS], cs_p[NS];
#pragma unroll
    for (int i = 0; i < NS; ++i) {
        cs_c[i] = desing(c.h, c.sc[i]);
        cs_m[i] = desing(m.h, m.sc[i]);
        cs_p[i] = desing(p.h, p.sc[i]);
    }

    float faceL[NS], faceR[NS];
    float Fh_d, FnL_d, FnR_d, Ft1_d, Ft2_d, Fs_d[NS];
#pragma unroll
    for (int i = 0; i < NS; ++i) {
        faceL[i] = cs_c[i] + 0.5f * pc.cs[i];
        faceR[i] = cs_p[i] - 0.5f * pp.cs[i];
    }
    flux_1d(c.h + 0.5f * pc.eta, un_c + 0.5f * sn_c,
            ut1_c + 0.5f * st1_c, ut2_c + 0.5f * st2_c, faceL,
            p.h - 0.5f * pp.eta, un_p - 0.5f * sn_p,
            ut1_p - 0.5f * st1_p, ut2_p - 0.5f * st2_p, faceR,
            &Fh_d, &FnL_d, &FnR_d, &Ft1_d, &Ft2_d, Fs_d);
    float Fh_g, FnL_g, FnR_g, Ft1_g, Ft2_g, Fs_g[NS];
#pragma unroll
    for (int i = 0; i < NS; ++i) {
        faceL[i] = cs_m[i] + 0.5f * pm.cs[i];
        faceR[i] = cs_c[i] - 0.5f * pc.cs[i];
    }
    flux_1d(m.h + 0.5f * pm.eta, un_m + 0.5f * sn_m,
            ut1_m + 0.5f * st1_m, ut2_m + 0.5f * st2_m, faceL,
            c.h - 0.5f * pc.eta, un_c - 0.5f * sn_c,
            ut1_c - 0.5f * st1_c, ut2_c - 0.5f * st2_c, faceR,
            &Fh_g, &FnL_g, &FnR_g, &Ft1_g, &Ft2_g, Fs_g);
    *dh  = Fh_d - Fh_g;
    *dn  = FnL_d - FnR_g;
    *dt1 = Ft1_d - Ft1_g;
    *dt2 = Ft2_d - Ft2_g;
#pragma unroll
    for (int i = 0; i < NS; ++i) ds[i] = Fs_d[i] - Fs_g[i];
}

extern "C" __global__ void """ + nom + r"""(
        const float* q_in, const float* q_base, float* q_out,
        const float dt, const float w_base, const int n, const long total,
        const float poids_statique) {
    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) return;
    long nn = (long)n * n;
    long nnn = nn * n;
    long bloc_ws = t / nnn;
    long base = bloc_ws * NC * nnn;
    long r = t % nnn;
    int z = (int)(r / nn);
    int y = (int)((r / n) % n);
    int x = (int)(r % n);

    float dhx, dnx, dt1x, dt2x, dsx[NS];
    float dhy, dny, dt1y, dt2y, dsy[NS];
    float dhz, dnz, dt1z, dt2z, dsz[NS];
    divergence_axe(q_in, base, n, z, y, x, 0, 0, 1, 0,
                   &dhx, &dnx, &dt1x, &dt2x, dsx);
    divergence_axe(q_in, base, n, z, y, x, 0, 1, 0, 1,
                   &dhy, &dny, &dt1y, &dt2y, dsy);
    divergence_axe(q_in, base, n, z, y, x, 1, 0, 0, 2,
                   &dhz, &dnz, &dt1z, &dt2z, dsz);
    float Lh  = -(dhx + dhy + dhz);
    float Lhu = -(dnx + dt1y + dt1z);
    float Lhv = -(dt1x + dny + dt2z);
    float Lhw = -(dt2x + dt2y + dnz);
    float Ls[NS];
#pragma unroll
    for (int i = 0; i < NS; ++i) Ls[i] = -(dsx[i] + dsy[i] + dsz[i]);

    long idx = base + ((long)z * n + y) * n + x;
    Etat c = lire(q_in, base, n, z, y, x);
""" + somme_st + r"""    float un = w_base * q_base[idx]           + (1.0f - w_base) * (c.h  + dt * Lh);
    float uu = w_base * q_base[idx + nnn]     + (1.0f - w_base) * (c.hu + dt * Lhu);
    float uv = w_base * q_base[idx + 2 * nnn] + (1.0f - w_base) * (c.hv + dt * Lhv);
    float uw = w_base * q_base[idx + 3 * nnn] + (1.0f - w_base) * (c.hw + dt * Lhw);
    float us[NS];
#pragma unroll
    for (int i = 0; i < NS; ++i)
        us[i] = w_base * q_base[idx + (4 + i) * nnn]
              + (1.0f - w_base) * (c.sc[i] + dt * Ls[i]);
""" + repli_st + r"""    un = fmaxf(un, 0.0f);
    if (un <= DRY_EPS) {
        un = 0.0f; uu = 0.0f; uv = 0.0f; uw = 0.0f;
#pragma unroll
        for (int i = 0; i < NS; ++i) us[i] = 0.0f;
    }
    q_out[idx]           = un;
    q_out[idx + nnn]     = uu;
    q_out[idx + 2 * nnn] = uv;
    q_out[idx + 3 * nnn] = uw;
#pragma unroll
    for (int i = 0; i < NS; ++i) q_out[idx + (4 + i) * nnn] = us[i];
}
"""


def empreinte_param(ns: int, nst: int) -> str:
    return hashlib.sha256(source_param(ns, nst).encode("utf-8")).hexdigest()


def comptes_statiques_param(ns: int, nst: int) -> dict:
    """Comptes d'opérations EXACTS, par cellule et par étage RK2 — le
    témoin d'attribution. Trois axes ; par axe un `divergence_axe`, qui
    appelle 3 `pentes_axe`, 2 `flux_1d` et lit 3 voisins directs."""
    axes, lectures_par_axe = 3, 3 + 3 * 3
    return {
        "n_scalaires": ns, "n_statiques": nst,
        "n_champs": CHAMPS_HYPERBOLIQUES + ns + nst,
        "minmods": axes * 3 * (4 + ns),
        "solveurs_hll": axes * 2,
        "upwinds_tangentiels": axes * 2 * (2 + ns),
        "lectures_voisins": axes * lectures_par_axe,
        "octets_lus": (axes * lectures_par_axe
                       * (CHAMPS_HYPERBOLIQUES + ns + nst) * 4),
    }


def _obtenir(cp, ns: int, nst: int):
    cle = (ns, nst)
    if cle not in _cache:
        _cache[cle] = cp.RawKernel(source_param(ns, nst), _nom_kernel(ns, nst))
    return _cache[cle]


def attributs_param(cp, ns: int, nst: int) -> dict:
    """Registres et mémoire locale — sans eux, un δ élevé ne se distingue
    pas d'un kernel qui déborde, et un δ NUL ne se distingue pas d'une
    élimination par le compilateur (garde I-c3 du prereg)."""
    kernel = _obtenir(cp, ns, nst)
    return {"registres_par_thread": int(kernel.num_regs),
            "memoire_locale_octets": int(kernel.local_size_bytes),
            "memoire_partagee_octets": int(kernel.shared_size_bytes),
            "max_threads_par_bloc": int(kernel.max_threads_per_block),
            "taille_bloc": _TAILLE_BLOC}


def pas_f_fusionne_3d_param(q, cp, sortie=None, tampon_etage=None,
                            n_statiques: int = 0,
                            poids_statique: float = 0.0) -> tuple:
    """Un pas SSP-RK2 complet sur (B, S, C, n, n, n) f32, S ∈ {1, 2}.

    Les `n_statiques` derniers champs sont LUS par le stencil et jamais
    écrits : ils traversent le pas inchangés (le tampon les porte).

    `poids_statique` vaut **0.0 pour toute mesure** — le repli est alors
    exactement neutre (`0*x + y == y`). Il n'existe que pour deux raisons,
    toutes deux load-bearing : empêcher `nvcc` de supprimer des lectures
    dont rien ne dépend, et **PROUVER qu'il ne l'a pas fait** — à poids
    non nul la sortie DOIT changer. Un poids non nul dans un chemin de
    mesure serait une autre physique ; le driver ne le passe jamais.

    Retourne (q_suivant, dt_cfl_diagnostic)."""
    if q.dtype != cp.float32:
        raise ValueError(f"pas_f_fusionne_3d_param : dtype {q.dtype} != f32.")
    if q.ndim != 6 or q.shape[1] not in (1, 2):
        raise ValueError(
            f"pas_f_fusionne_3d_param : shape {q.shape} — attendu "
            "(B, S, C, n, n, n) avec S ∈ {1, 2}.")
    if not (q.shape[-1] == q.shape[-2] == q.shape[-3]):
        raise ValueError(
            f"pas_f_fusionne_3d_param : fenêtre non cubique {q.shape[-3:]} — "
            "le slot 3D est un cube (§A51 : 512² = 64³ = 262 144 cellules).")
    ns, nst = decompte_champs(int(q.shape[2]), n_statiques)
    q = cp.ascontiguousarray(q)
    dt_cfl = reduction_cfl_3d(q, cp)

    n = int(q.shape[-1])
    total = int(q.shape[0] * q.shape[1] * n * n * n)
    if tampon_etage is None:
        tampon_etage = cp.array(q, copy=True)   # les statiques traversent
    elif nst:
        tampon_etage[...] = q
    if sortie is None:
        sortie = cp.array(q, copy=True)
    elif nst:
        sortie[...] = q
    kernel = _obtenir(cp, ns, nst)
    grille = ((total + _TAILLE_BLOC - 1) // _TAILLE_BLOC,)
    poids = np.float32(poids_statique)
    communs = (np.float32(DT_JETABLE), np.int32(n), np.int64(total))
    kernel(grille, (_TAILLE_BLOC,),
           (q, q, tampon_etage, communs[0], np.float32(0.0),
            communs[1], communs[2], poids))
    kernel(grille, (_TAILLE_BLOC,),
           (tampon_etage, q, sortie, communs[0], np.float32(0.5),
            communs[1], communs[2], poids))
    return sortie, dt_cfl
