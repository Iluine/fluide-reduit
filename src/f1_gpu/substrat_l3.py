"""F1 — KERNEL L3 : F fusionné + ÉMISSION des détails en épilogue
(achat A, §A18-complément / -2, pocCascade2phys d720da6).

MESURE JETABLE, ET C'EST DIT ICI. Ce module est une BORNE du coût de la
remontée, comme `substrat_fusionne` fut une borne de F. Il n'est pas un
composant de moteur, il ne prétend pas l'être, et le portage fidèle reste
non acheté.

CAP GRAVÉ : L3 est la DERNIÈRE escalade sur la remontée. Aucune escalade
ultérieure — pas de CUDA Graphs, pas d'exotique, pas de « version mieux
optimisée » — n'est autorisée après ce build.

LE KERNEL M-a′ EST INTOUCHÉ : `substrat_fusionne.py` n'est pas modifié,
son bloc `_SOURCE` garde l'empreinte
e8fcaad47db0af010728d5c5d878e1629191bfef4e0382fa80aa3cefd9f27f04, et la
borne M-a′ (1.685 ms/slot) est citée telle quelle, jamais re-mesurée.

ÉQUIVALENCE DE MOTIF, ACQUISE PAR CONSTRUCTION : les fonctions device du
fusionné (`desing`, `minmod`, `lire`, `pentes_axe`, `flux_1d`,
`divergence_axe`) ne sont PAS recopiées — le source L3 est bâti en
REPRENANT le préfixe de `substrat_fusionne._SOURCE` jusqu'au premier
`extern "C" __global__`, puis en y ajoutant un kernel global neuf. Le
stencil, les constantes et l'arithmétique sont donc littéralement les
mêmes objets texte ; seul l'épilogue est neuf. Un test vérifie que le
préfixe employé est bien celui du fusionné.

CHOIX D'IMPLÉMENTATION FIGÉS AU PROTOCOLE (prereg kernel L3, arbitrés
§A18-complément-2) — un chiffre décevant se lira « ce design, compacté
ainsi, coûte tant », jamais « il aurait fallu mieux l'implémenter » :

  1. ÉMISSION PAR ÉLÉMENT. Le schéma B4 émet par élément (un champ d'une
     cellule d'un système d'une fenêtre) : l'épilogue traite donc les 4
     champs de chaque thread, jusqu'à 4 émissions. Ce n'est pas un choix
     libre — c'est ce que l'équivalence avec l'extraction CuPy impose.
  2. LECTURE DE RÉFÉRENCE + ÉCRITURE PRÉDIQUÉE. Le thread lit
     `reference[p]`, calcule `d`, et n'écrit que si `|d| >= eps`. La
     passe de lecture est PAYÉE dans le kernel — le schéma CuPy la paie
     aussi. La mise à jour incrémentale `reference[p] += d` est faite
     dans le même épilogue, prédiquée : sans elle le schéma ne serait
     plus incrémental et l'équivalence tomberait.
  3. COMPACTION PAR AGRÉGATION DE WARP (`__ballot_sync` + `__popc`, UN
     atomique par warp et par champ, rangs par `__popc` du masque des
     lanes précédentes, base diffusée par `__shfl_sync`). Arbitrage
     Romain §A18-complément-2 contre l'atomicAdd naïf : la borne doit
     mesurer un design compétent, pas une file d'attente de 17.8 M
     atomiques sérialisés.
  4. PAS D'EARLY RETURN. Le fusionné sort par `if (t >= total) return;` ;
     ici les threads de queue restent dans le warp (drapeau `actif`,
     écritures prédiquées), car `__ballot_sync` exige le warp entier.
     Sans effet sur le coût en régime : `total` = n_blocs·n² est un
     multiple de la taille de bloc dès que n² l'est (512² l'est), donc
     AUCUN thread de queue n'existe à la taille mesurée ; il n'y en a que
     dans les tests à petite taille.
  5. DIMENSIONNEMENT AU PIRE CAS. Tout peut être émis : les buffers de
     sortie valent l'état entier (142.6 Mo à la config V2 — gate 1.35 Go,
     V2 mesurée 0.311 Go). Dimensionner en dessous exigerait de savoir
     d'avance combien sera émis.
  6. COMPTEUR LU À RETARD D'UNE FRAME (`CompacteurL3`) — voir sa
     docstring : aucune synchronisation par frame, la maladie s2 ne
     rouvre pas.

L'ORDRE D'ÉMISSION N'EST PAS GARANTI (compaction atomique). L'ensemble
émis, lui, l'est. Les tests comparent donc des ensembles TRIÉS — une
dérogation à la comparaison exacte du dépôt, isolée à ce chemin — et
vérifient d'abord l'UNICITÉ DES INDICES (consigne §A18-complément-2 : un
tri masquerait une double émission ; chaque élément n'émet qu'une fois
par frame par construction, donc le test est gratuit)."""
from __future__ import annotations

import numpy as np

from src.f1_gpu.substrat_fusionne import _SOURCE as _SOURCE_FUSIONNE
from src.f1_gpu.substrat_fusionne import _TAILLE_BLOC
from src.f1_gpu.substrat_jetable import DT_JETABLE, reduction_cfl

_NOM_KERNEL_L3: str = "etage_ssp_wetdry_o2_l3"
_MARQUEUR_GLOBAL: str = 'extern "C" __global__'

# Le préfixe du fusionné : constantes, structures et TOUTES les fonctions
# device, repris tels quels — l'équivalence de motif est structurelle.
_PREFIXE_FUSIONNE: str = _SOURCE_FUSIONNE[
    : _SOURCE_FUSIONNE.index(_MARQUEUR_GLOBAL)]

# Le kernel neuf : même corps que l'étage du fusionné (mêmes appels aux
# mêmes fonctions device, même arithmétique, même ordre) + l'épilogue
# d'émission. `emettre` est uniforme sur la grille (aucune divergence) :
# seul l'étage 2 émet, l'état final étant ce qui se compare à la référence.
_EPILOGUE: str = r"""
extern "C" __global__ void """ + _NOM_KERNEL_L3 + r"""(
        const float* q_in, const float* q_base, float* q_out,
        float* reference, float* valeurs_out, unsigned int* indices_out,
        unsigned int* compteur, const float dt, const float w_base,
        const int n, const long total, const int emettre, const float eps) {
    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    /* Pas d'early return : __ballot_sync exige le warp entier (choix 4). */
    int actif = (t < total);
    long t_eff = actif ? t : 0;
    long nn = (long)n * n;
    long bloc_ws = t_eff / nn;
    long base = bloc_ws * 4 * nn;
    int y = (int)((t_eff % nn) / n);
    int x = (int)(t_eff % n);

    float dhx, dnx, dtx, dsx, dhy, dny, dty, dsy;
    divergence_axe(q_in, base, n, y, x, 0, 1, 1, &dhx, &dnx, &dtx, &dsx);
    divergence_axe(q_in, base, n, y, x, 1, 0, 0, &dhy, &dny, &dty, &dsy);
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
    un = fmaxf(un, 0.0f);
    if (un <= DRY_EPS) { un = 0.0f; uu = 0.0f; uv = 0.0f; us = 0.0f; }
    if (actif) {
        q_out[idx]          = un;
        q_out[idx + nn]     = uu;
        q_out[idx + 2 * nn] = uv;
        q_out[idx + 3 * nn] = us;
    }

    /* ----- épilogue L3 : émission des détails |d| >= eps ----- */
    if (emettre) {
        float nouveaux[4];
        nouveaux[0] = un; nouveaux[1] = uu; nouveaux[2] = uv; nouveaux[3] = us;
        int lane = (int)(threadIdx.x & 31u);
        for (int champ = 0; champ < 4; ++champ) {
            long p = idx + (long)champ * nn;
            float d = actif ? (nouveaux[champ] - reference[p]) : 0.0f;
            int retenu = actif && (fabsf(d) >= eps);
            /* Agrégation de warp (choix 3) : un seul atomique par warp. */
            unsigned int masque = __ballot_sync(0xffffffffu, retenu);
            if (masque != 0u) {
                int chef = __ffs(masque) - 1;
                unsigned int depart = 0u;
                if (lane == chef) {
                    depart = atomicAdd(compteur, (unsigned int)__popc(masque));
                }
                depart = __shfl_sync(0xffffffffu, depart, chef);
                if (retenu) {
                    unsigned int rang = (unsigned int)__popc(
                        masque & ((1u << lane) - 1u));
                    valeurs_out[depart + rang] = d;
                    indices_out[depart + rang] = (unsigned int)p;
                    reference[p] += d;      /* schéma incrémental (choix 2) */
                }
            }
        }
    }
}
"""

_SOURCE_L3: str = _PREFIXE_FUSIONNE + _EPILOGUE

_cache_kernel_l3 = None


def _obtenir_kernel_l3(cp):
    """Compile (une fois, NVRTC — au warmup du chrono) et met en cache."""
    global _cache_kernel_l3
    if _cache_kernel_l3 is None:
        _cache_kernel_l3 = cp.RawKernel(_SOURCE_L3, _NOM_KERNEL_L3)
    return _cache_kernel_l3


def taille_sortie_max(q) -> int:
    """Nombre d'éléments émissibles au pire cas : tout l'état (choix 5)."""
    return int(q.shape[0] * q.shape[1] * 4 * q.shape[-1] * q.shape[-2])


class CompacteurL3:
    """Buffers d'émission + compteur LU À RETARD D'UNE FRAME (choix 6,
    endossé §A18-complément).

    Pourquoi ce retard : connaître la taille émise exige de lire le
    compteur, donc de rapatrier un scalaire, donc de SYNCHRONISER — la
    maladie que s2 a guérie sur la CFL. Ici le compteur est copié
    ASYNCHRONEMENT vers un tampon hôte *pinned*, et la frame courante
    dimensionne son transfert sur la valeur arrivée à la frame
    PRÉCÉDENTE. Aucun `synchronize()`, aucun `int(tableau_device)` par
    frame ; la lecture porte sur de la mémoire hôte, elle ne bloque pas.

    Conséquence NOMMÉE, reportée et jamais tue : si l'émission d'une frame
    dépasse celle de la précédente, une part des coefficients n'est pas
    transférée ce tour-là. Le schéma B4 étant incrémental (les résidus
    s'accumulent et repassent le seuil), rien n'est perdu définitivement —
    `ecart_emis_transferes()` rend l'écart lisible.

    B2 : tout est préalloué à la construction ; la boucle de frame
    n'alloue rien."""

    def __init__(self, cp, taille_max: int):
        self.cp = cp
        self.taille_max = int(taille_max)
        self.valeurs = cp.empty(self.taille_max, dtype=cp.float32)
        self.indices = cp.empty(self.taille_max, dtype=cp.uint32)
        self.compteur = cp.zeros(1, dtype=cp.uint32)
        # Tampon hôte page-locked : la copie asynchrone y atterrit.
        self._memoire_pinned = cp.cuda.alloc_pinned_memory(4)
        self.hote = np.frombuffer(self._memoire_pinned, dtype=np.uint32,
                                  count=1)
        self.hote[0] = 0          # frame 0 : aucun précédent (warmup, B6)
        self.tailles_vues: list[int] = []
        self.compteur_final = 0

    def taille_precedente(self) -> int:
        """Compteur de la frame précédente — lecture HÔTE, sans blocage."""
        return int(min(int(self.hote[0]), self.taille_max))

    def noter_taille(self, taille: int) -> None:
        """Enregistre la taille retenue pour cette frame (diagnostic du
        retard, agrégé hors chrono — une liste Python, pas de device)."""
        self.tailles_vues.append(int(taille))

    def cloturer_frame(self) -> None:
        """Copie asynchrone du compteur vers l'hôte (pour la frame
        suivante), puis remise à zéro. Les deux ordres sont soumis au
        stream courant, donc exécutés APRÈS le kernel de cette frame."""
        cp = self.cp
        cp.cuda.runtime.memcpyAsync(
            self._memoire_pinned.ptr, self.compteur.data.ptr, 4,
            cp.cuda.runtime.memcpyDeviceToHost,
            cp.cuda.Stream.null.ptr)
        self.compteur.fill(0)

    def vues_a_transferer(self, taille: int):
        """Tranches contiguës à remonter (valeurs f32 + indices u32)."""
        return self.valeurs[:taille], self.indices[:taille]

    def diagnostic_retard(self) -> dict:
        """Ce que le retard d'une frame a coûté, rendu lisible.

        Le transfert de la frame *n* est dimensionné sur le compteur de
        *n−1*. Si les tailles émises sont STABLES (min == max), le retard
        est exactement neutre : chaque frame transfère ce que la
        précédente a émis, et seule la dernière reste en attente. C'est
        la variabilité — pas un cumul, qui serait tautologique — qui dit
        si le retard mord. Agrégé HORS chrono."""
        tailles = [t for t in self.tailles_vues if t > 0]
        stable = bool(tailles) and min(tailles) == max(tailles)
        return {
            "frames_observees": len(self.tailles_vues),
            "taille_min": min(tailles) if tailles else 0,
            "taille_max": max(tailles) if tailles else 0,
            "taille_mediane": (int(np.percentile(tailles, 50))
                               if tailles else 0),
            "tailles_stables": stable,
            "compteur_final_non_transfere": self.compteur_final,
            "note": ("retard d'une frame (choix 6). Tailles stables => le "
                     "retard est neutre, seule la dernière frame reste en "
                     "attente. Sinon l'écart frame à frame est reporté "
                     "par min/max ; le schéma B4 étant incrémental, les "
                     "résidus repassent le seuil — rien n'est perdu."),
        }


def pas_f_l3(q, cp, reference, compacteur: CompacteurL3, sortie=None,
             tampon_etage=None, eps: float = 1e-4, reduction=reduction_cfl):
    """Un pas SSP-RK2 du kernel L3 : même motif que `pas_f_fusionne` (mêmes
    fonctions device, mêmes deux étages, même dt figé) + émission des
    détails à l'étage 2.

    `reduction` est injectable : le driver de borne passe la réduction
    CFL ON-DEVICE de s2 pour que les deux bras soient comparables (le
    défaut, celui du jetable, synchronise).

    Retourne (sortie, dt_cfl). Le compteur d'émission vit dans
    `compacteur` et n'est PAS lu ici (choix 6)."""
    if q.dtype != cp.float32:
        raise ValueError(f"pas_f_l3 : dtype {q.dtype} != float32.")
    if q.ndim != 5 or q.shape[2] != 4 or q.shape[1] not in (1, 2):
        raise ValueError(
            f"pas_f_l3 : shape {q.shape} != (B, S, 4, n, n) avec "
            "S ∈ {1, 2} systèmes (E4a).")
    if reference.shape != q.shape:
        raise ValueError(
            f"pas_f_l3 : référence {reference.shape} != état {q.shape}.")
    if compacteur.taille_max < taille_sortie_max(q):
        raise ValueError(
            f"pas_f_l3 : buffers d'émission trop petits "
            f"({compacteur.taille_max} < {taille_sortie_max(q)}) — le "
            "dimensionnement est au PIRE CAS (choix 5).")
    dt_cfl = reduction(q, cp)

    n = int(q.shape[-1])
    total = int(q.shape[0] * q.shape[1] * n * n)
    if tampon_etage is None:
        tampon_etage = cp.empty_like(q)
    if sortie is None:
        sortie = cp.empty_like(q)
    kernel = _obtenir_kernel_l3(cp)
    grille = ((total + _TAILLE_BLOC - 1) // _TAILLE_BLOC,)
    dt = np.float32(DT_JETABLE)
    n_arg, total_arg = np.int32(n), np.int64(total)
    eps_arg = np.float32(eps)
    # Étage 1 : aucune émission (l'état intermédiaire n'est pas le final).
    kernel(grille, (_TAILLE_BLOC,),
           (q, q, tampon_etage, reference, compacteur.valeurs,
            compacteur.indices, compacteur.compteur, dt, np.float32(0.0),
            n_arg, total_arg, np.int32(0), eps_arg))
    # Étage 2 : émission de l'état final contre la référence.
    kernel(grille, (_TAILLE_BLOC,),
           (tampon_etage, q, sortie, reference, compacteur.valeurs,
            compacteur.indices, compacteur.compteur, dt, np.float32(0.5),
            n_arg, total_arg, np.int32(1), eps_arg))
    return sortie, dt_cfl


def extraction_cupy_reference(fenetre, reference, xp, eps: float = 1e-4):
    """L'extraction B4 telle que la pyramide la fait aujourd'hui — la
    RÉFÉRENCE de l'équivalence, reproduite ici pour être comparable sans
    dépendre de la pyramide. Retourne (valeurs, indices) NON triés, et
    met à jour `reference` comme le schéma incrémental."""
    d = fenetre - reference
    masque = xp.abs(d) >= eps
    valeurs = d[masque]
    indices = xp.flatnonzero(masque).astype(xp.uint32)
    reference += xp.where(masque, d, xp.float32(0.0))
    return valeurs, indices
