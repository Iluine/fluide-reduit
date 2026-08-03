"""F 3D — M-3D : F FUSIONNÉ 3D (RawKernel CUDA), transposition d'un axe du
fusionné 2D (pré-enregistrement `claude/prereg-cout-f-3d-2026-08-03.md`,
commit `661237a`, écrit AVANT ce fichier).

C'EST CE MODULE QUI PORTE LE CHIFFRE. Le rapport de verdict ρ se lit entre
`pas_f_fusionne` (2D) et `pas_f_fusionne_3d` (3D) — deux kernels de la MÊME
famille d'implémentation : un thread par cellule, halo relu, flux
RE-CALCULÉS entre voisins, un lancement par étage RK2. Comparer un fusionné
à un naïf reproduirait à l'envers la faute du run 1 de la tranche : mesurer
le code de session au lieu de la structure. Le jetable 3D
(`substrat_jetable_3d.py`) n'existe que pour le test d'équivalence.

CE QUE LE FUSIONNÉ 3D CHANGE / NE CHANGE PAS, NOMMÉ — la même liste que son
aîné 2D, mot pour mot, parce que l'écart entre les deux doit être la
DIMENSION et rien d'autre :
  - CHANGE : rien dans le motif. Trois axes au lieu de deux, cinq champs
    par système au lieu de quatre, deux tangentielles par axe au lieu
    d'une — les trois seules choses, et elles SONT la question.
  - NE CHANGE PAS : padding réfléchissant (miroir + négation de la qdm
    normale, sur trois paires de faces), désingularisation, pentes minmod,
    réconciliation positivité, reconstruction MUSCL, flux HLL dry-aware,
    transport tangentiel upwind, corrections de pression hydrostatiques
    (calculées, nulles à b ≡ 0), divergence, plancher sec par étage,
    2 étages SSP-RK2, réduction CFL PAYÉE non consommée, dt figé, f32.

`substrat_fusionne.py:8-9` s'applique ici sans changement de mot : « PAS
changer le motif de coût — toute omission de calcul serait un harnais
complaisant, interdit ». La minoration tentante en 3D — quatre champs, pas
de `hw` — est interdite d'avance et par écrit (prereg §3).

CONVENTION NORMALE/TANGENTIELLES : identique à
`substrat_jetable_3d.NORMALE_PAR_AXE` / `TANGENTIELLES_PAR_AXE`. Un
désaccord entre les deux serait une physique différente et non une
implémentation différente — le test d'équivalence le détecte, et le test
d'isotropie détecte un axe mal câblé même quand les deux sont d'accord.

CAP D'IMPLÉMENTATION : celui de M-a′ est reconduit tel quel — aucun CUDA
Graphs, aucun exotique. Le kernel 3D ne reçoit aucune optimisation que son
aîné 2D n'ait reçue ; en donner une fabriquerait un ρ favorable, en refuser
une qu'il a fabriquerait un ρ défavorable. Les deux sont interdites."""
from __future__ import annotations

import numpy as np

from src.f1_gpu.substrat_jetable import DRY_EPS, DT_JETABLE, GRAVITE
from src.f1_gpu.substrat_jetable_3d import (
    CHAMPS_PAR_SYSTEME_3D,
    reduction_cfl_3d,
)

# Tolérances d'équivalence : celles de M-a′, RECONDUITES (jamais
# redéfinies — un seuil qu'on redéfinit est un seuil qu'on déplace).
from src.f1_gpu.substrat_fusionne import (      # noqa: E402  (ordre voulu)
    TOL_EQUIVALENCE_ATOL,
    TOL_EQUIVALENCE_RTOL,
)

__all__ = ["pas_f_fusionne_3d", "empreinte_source_3d",
           "TOL_EQUIVALENCE_RTOL", "TOL_EQUIVALENCE_ATOL"]

_NOM_KERNEL_3D: str = "etage_ssp_wetdry_o2_3d"

# Layout contigu (B, S, 5, n, n, n) f32, S ∈ {1, 2} ; un thread = une
# cellule d'un bloc (fenêtre, système). Comme en 2D, le kernel ne lit
# JAMAIS S : il itère `total` blocs aplatis.
_SOURCE_3D: str = r"""
#define G """ + f"{GRAVITE}f" + r"""
#define DRY_EPS """ + f"{DRY_EPS}f" + r"""

struct Etat { float h, hu, hv, hw, s; };
struct Pentes { float eta, u, v, w, cs; };

__device__ __forceinline__ float desing(float h, float hq) {
    return (h > DRY_EPS) ? hq / fmaxf(h, DRY_EPS) : 0.0f;
}

__device__ __forceinline__ float minmod(float a, float b) {
    if (a * b <= 0.0f) return 0.0f;
    float signe = (a > 0.0f) ? 1.0f : -1.0f;
    return signe * fminf(fabsf(a), fabsf(b));
}

/* Lecture avec cellules fantômes réfléchissantes : miroir bord (edge) +
   négation de la composante NORMALE de la qdm — TROIS paires de murs
   (motif _pad_reflexif_3d). */
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
    e.s  = q[idx + 4 * nnn];
    return e;
}

/* Pente limitée de la cellule RÉELLE (z, y, x) le long de l'axe
   (dz, dy, dx), sur (η, u, v, w, c_s), avec réconciliation positivité
   (b = 0 : la face h = η ∓ 0.5·s_η doit rester >= 0, sinon pentes nulles
   — retour O1). */
__device__ Pentes pentes_axe(const float* q, long base, int n,
                             int z, int y, int x, int dz, int dy, int dx) {
    Etat c = lire(q, base, n, z, y, x);
    Etat m = lire(q, base, n, z - dz, y - dy, x - dx);
    Etat p = lire(q, base, n, z + dz, y + dy, x + dx);
    float uc = desing(c.h, c.hu), um = desing(m.h, m.hu), up = desing(p.h, p.hu);
    float vc = desing(c.h, c.hv), vm = desing(m.h, m.hv), vp = desing(p.h, p.hv);
    float wc = desing(c.h, c.hw), wm = desing(m.h, m.hw), wp = desing(p.h, p.hw);
    float cc = desing(c.h, c.s),  cm = desing(m.h, m.s),  cp_ = desing(p.h, p.s);
    Pentes s;
    s.eta = minmod(c.h - m.h, p.h - c.h);          /* η = h (b = 0) */
    s.u   = minmod(uc - um, up - uc);
    s.v   = minmod(vc - vm, vp - vc);
    s.w   = minmod(wc - wm, wp - wc);
    s.cs  = minmod(cc - cm, cp_ - cc);
    if ((c.h - 0.5f * s.eta) < 0.0f || (c.h + 0.5f * s.eta) < 0.0f) {
        s.eta = 0.0f; s.u = 0.0f; s.v = 0.0f; s.w = 0.0f; s.cs = 0.0f;
    }
    return s;
}

__device__ Pentes pentes_nulles() {
    Pentes s; s.eta = 0.0f; s.u = 0.0f; s.v = 0.0f; s.w = 0.0f; s.cs = 0.0f;
    return s;
}

/* Flux d'interface 1D complet : états de face reconstruits
   (η, u_n, u_t1, u_t2, c_s) des deux côtés ; b = 0. DEUX tangentielles —
   c'est la conséquence mécanique du troisième axe. */
__device__ void flux_1d(float etaL, float unL, float ut1L, float ut2L, float csL,
                        float etaR, float unR, float ut1R, float ut2R, float csR,
                        float* Fh, float* FnL_corr, float* FnR_corr,
                        float* Ft1, float* Ft2, float* Fs) {
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
    *Ft1 = (fh >= 0.0f) ? fh * ut1L : fh * ut1R;
    *Ft2 = (fh >= 0.0f) ? fh * ut2L : fh * ut2R;
    *Fs  = (fh >= 0.0f) ? fh * csL  : fh * csR;
}

/* Divergence 1D d'un axe pour la cellule courante : calcule les DEUX
   interfaces et rend les 5 contributions.
   axe : 0 = x (normale hu, tangentielles hv, hw)
         1 = y (normale hv, tangentielles hu, hw)
         2 = z (normale hw, tangentielles hu, hv)
   — convention IDENTIQUE à substrat_jetable_3d.TANGENTIELLES_PAR_AXE. */
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

    /* Sélection normale / tangentielles selon l'axe. */
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
    float cs_c = desing(c.h, c.s), cs_m = desing(m.h, m.s), cs_p = desing(p.h, p.s);

    float Fh_d, FnL_d, FnR_d, Ft1_d, Ft2_d, Fs_d;   /* interface « droite » (c|p) */
    flux_1d(c.h + 0.5f * pc.eta, un_c + 0.5f * sn_c,
            ut1_c + 0.5f * st1_c, ut2_c + 0.5f * st2_c, cs_c + 0.5f * pc.cs,
            p.h - 0.5f * pp.eta, un_p - 0.5f * sn_p,
            ut1_p - 0.5f * st1_p, ut2_p - 0.5f * st2_p, cs_p - 0.5f * pp.cs,
            &Fh_d, &FnL_d, &FnR_d, &Ft1_d, &Ft2_d, &Fs_d);
    float Fh_g, FnL_g, FnR_g, Ft1_g, Ft2_g, Fs_g;   /* interface « gauche » (m|c) */
    flux_1d(m.h + 0.5f * pm.eta, un_m + 0.5f * sn_m,
            ut1_m + 0.5f * st1_m, ut2_m + 0.5f * st2_m, cs_m + 0.5f * pm.cs,
            c.h - 0.5f * pc.eta, un_c - 0.5f * sn_c,
            ut1_c - 0.5f * st1_c, ut2_c - 0.5f * st2_c, cs_c - 0.5f * pc.cs,
            &Fh_g, &FnL_g, &FnR_g, &Ft1_g, &Ft2_g, &Fs_g);
    /* motif div : normale = (F+S_L)@droite − (F+S_R)@gauche */
    *dh  = Fh_d - Fh_g;
    *dn  = FnL_d - FnR_g;
    *dt1 = Ft1_d - Ft1_g;
    *dt2 = Ft2_d - Ft2_g;
    *ds  = Fs_d - Fs_g;
}

extern "C" __global__ void """ + _NOM_KERNEL_3D + r"""(
        const float* q_in, const float* q_base, float* q_out,
        const float dt, const float w_base, const int n, const long total) {
    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) return;
    long nn = (long)n * n;
    long nnn = nn * n;
    long bloc_ws = t / nnn;              /* index (fenêtre, système) aplati */
    long base = bloc_ws * 5 * nnn;       /* offset du canal 0 du bloc */
    long r = t % nnn;
    int z = (int)(r / nn);
    int y = (int)((r / n) % n);
    int x = (int)(r % n);

    float dhx, dnx, dt1x, dt2x, dsx;
    float dhy, dny, dt1y, dt2y, dsy;
    float dhz, dnz, dt1z, dt2z, dsz;
    divergence_axe(q_in, base, n, z, y, x, 0, 0, 1, 0,
                   &dhx, &dnx, &dt1x, &dt2x, &dsx);
    divergence_axe(q_in, base, n, z, y, x, 0, 1, 0, 1,
                   &dhy, &dny, &dt1y, &dt2y, &dsy);
    divergence_axe(q_in, base, n, z, y, x, 1, 0, 0, 2,
                   &dhz, &dnz, &dt1z, &dt2z, &dsz);
    /* qdm x = normale-en-x + tangentielle-1-en-y + tangentielle-1-en-z ;
       qdm y = tangentielle-1-en-x + normale-en-y + tangentielle-2-en-z ;
       qdm z = tangentielle-2-en-x + tangentielle-2-en-y + normale-en-z. */
    float Lh  = -(dhx + dhy + dhz);
    float Lhu = -(dnx + dt1y + dt1z);
    float Lhv = -(dt1x + dny + dt2z);
    float Lhw = -(dt2x + dt2y + dnz);
    float Ls  = -(dsx + dsy + dsz);

    long idx = base + ((long)z * n + y) * n + x;
    Etat c;
    c.h  = q_in[idx];
    c.hu = q_in[idx + nnn];
    c.hv = q_in[idx + 2 * nnn];
    c.hw = q_in[idx + 3 * nnn];
    c.s  = q_in[idx + 4 * nnn];
    float un = w_base * q_base[idx]           + (1.0f - w_base) * (c.h  + dt * Lh);
    float uu = w_base * q_base[idx + nnn]     + (1.0f - w_base) * (c.hu + dt * Lhu);
    float uv = w_base * q_base[idx + 2 * nnn] + (1.0f - w_base) * (c.hv + dt * Lhv);
    float uw = w_base * q_base[idx + 3 * nnn] + (1.0f - w_base) * (c.hw + dt * Lhw);
    float us = w_base * q_base[idx + 4 * nnn] + (1.0f - w_base) * (c.s  + dt * Ls);
    /* plancher sec (motif _plancher_sec_3d) */
    un = fmaxf(un, 0.0f);
    if (un <= DRY_EPS) { un = 0.0f; uu = 0.0f; uv = 0.0f; uw = 0.0f; us = 0.0f; }
    q_out[idx]           = un;
    q_out[idx + nnn]     = uu;
    q_out[idx + 2 * nnn] = uv;
    q_out[idx + 3 * nnn] = uw;
    q_out[idx + 4 * nnn] = us;
}
"""

_TAILLE_BLOC: int = 256          # identique à M-a′ — jamais réglé pour la 3D
_cache_kernel_3d = None


def empreinte_source_3d() -> str:
    """Empreinte sha256 du source CUDA — gravée dans l'artefact, à côté de
    celle du kernel 2D. Motif de M-a-quater (« kernels F et L3 INTOUCHÉS
    (git diff + empreintes `_SOURCE` / `_SOURCE_L3`) »)."""
    import hashlib
    return hashlib.sha256(_SOURCE_3D.encode("utf-8")).hexdigest()


def _obtenir_kernel_3d(cp):
    """Compile (une fois, NVRTC — au warmup du chrono, jamais en série) et
    met en cache le RawKernel."""
    global _cache_kernel_3d
    if _cache_kernel_3d is None:
        _cache_kernel_3d = cp.RawKernel(_SOURCE_3D, _NOM_KERNEL_3D)
    return _cache_kernel_3d


def attributs_kernel_3d(cp) -> dict:
    """Registres, mémoire locale et occupancy théorique — reportés par le
    driver (témoin M-6 du prereg). Sans eux, un ρ élevé ne se distingue pas
    d'un kernel qui déborde ses registres : « la 3D coûte ça » et « mon
    kernel est mauvais » se ressemblent dans un chronomètre."""
    kernel = _obtenir_kernel_3d(cp)
    return {"registres_par_thread": int(kernel.num_regs),
            "memoire_locale_octets": int(kernel.local_size_bytes),
            "memoire_partagee_octets": int(kernel.shared_size_bytes),
            "max_threads_par_bloc": int(kernel.max_threads_per_block),
            "taille_bloc": _TAILLE_BLOC}


def pas_f_fusionne_3d(q, cp, sortie=None, tampon_etage=None) -> tuple:
    """Un pas SSP-RK2 complet du F FUSIONNÉ 3D sur (B, S, 5, n, n, n) f32 :
    réduction CFL 3D (payée, non consommée — RÉUTILISÉE du jetable 3D,
    jamais réimplémentée) + 2 lancements du kernel + plancher sec
    in-kernel. `sortie is q` autorisé (le kernel ne lit `q_base` qu'à sa
    propre cellule) ; `tampon_etage` préallouable par l'appelant (B2).

    Retourne (q_suivant, dt_cfl_diagnostic) — signature de l'aîné 2D."""
    if q.dtype != cp.float32:
        raise ValueError(f"pas_f_fusionne_3d : dtype {q.dtype} != float32.")
    if (q.ndim != 6 or q.shape[2] != CHAMPS_PAR_SYSTEME_3D
            or q.shape[1] not in (1, 2)):
        raise ValueError(
            f"pas_f_fusionne_3d : shape {q.shape} != (B, S, "
            f"{CHAMPS_PAR_SYSTEME_3D}, n, n, n) avec S ∈ {{1, 2}} — les "
            "cinq champs (h, hu, hv, hw, s) sont load-bearing (prereg §3).")
    if not (q.shape[-1] == q.shape[-2] == q.shape[-3]):
        raise ValueError(
            f"pas_f_fusionne_3d : fenêtre non cubique {q.shape[-3:]} — le "
            "slot 3D est un cube (§A51 : 512² = 64³ = 262 144 cellules).")
    q = cp.ascontiguousarray(q)
    dt_cfl = reduction_cfl_3d(q, cp)

    n = int(q.shape[-1])
    total = int(q.shape[0] * q.shape[1] * n * n * n)
    if tampon_etage is None:
        tampon_etage = cp.empty_like(q)
    if sortie is None:
        sortie = cp.empty_like(q)
    kernel = _obtenir_kernel_3d(cp)
    grille = ((total + _TAILLE_BLOC - 1) // _TAILLE_BLOC,)
    communs = (np.float32(DT_JETABLE), np.int32(n), np.int64(total))
    kernel(grille, (_TAILLE_BLOC,),
           (q, q, tampon_etage, communs[0], np.float32(0.0),
            communs[1], communs[2]))
    kernel(grille, (_TAILLE_BLOC,),
           (tampon_etage, q, sortie, communs[0], np.float32(0.5),
            communs[1], communs[2]))
    return sortie, dt_cfl
