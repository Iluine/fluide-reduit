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
et la borne M-a′ (1.685 ms/slot) est citée telle quelle, jamais
re-mesurée. L'empreinte qui le verrouille — la SEULE citée désormais,
re-calculable par qui l'exige (§A21) — porte sur la CHAÎNE CUDA EXTRAITE
et non sur le fichier :
    sha256(substrat_fusionne._SOURCE.encode()) = e18015f5...f30b4
Le kernel L3 de ce module a désormais le SIEN, de même forme
(`_SOURCE_L3`, chaîne extraite) : c'est lui qui porte `reference += d`,
il ne pouvait pas rester sans verrou. Les deux sont imposés par des
tests, et le code CUDA reste intouché même quand le Python autour bouge.

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
     d'avance combien sera émis. DEPUIS §A21 ces buffers sont en
     PING-PONG, donc DOUBLÉS ; `CompacteurL3.octets_buffers()` reporte le
     total pour que la résidence le montre.
  6. COMPTEUR LU À RETARD D'UNE FRAME (`CompacteurL3`) — voir sa
     docstring : aucune synchronisation par frame, la maladie s2 ne
     rouvre pas. Le ping-pong de §A21 le PRÉSERVE : il ne change que le
     jeu de buffers transféré, pas la façon de lire le compteur.

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
    """Buffers d'émission en PING-PONG — implémentation de l'INVARIANT
    gravé §A21 (pocCascade2phys 299adfc) :

        **la référence du device n'avance QUE sur ce qui a été
        effectivement TRANSFÉRÉ, jamais sur ce qui a été ÉMIS.**

    Le kernel, lui, est INTOUCHÉ : il fait `reference[p] += d` à
    l'émission, et il n'a aucun moyen de savoir ce qui sera transféré.
    L'invariant ne peut donc pas être tenu dans le kernel — il est tenu
    ICI, en garantissant que TOUT ce qui est émis finit transféré, une
    fois et une seule. La bijection émis ↔ reçu vaut l'invariant.

    ────────────────────────────────────────────────────────────────────
    CE QU'IL A FALLU RÉPARER (§A19-CORRECTION, deux défauts EXHIBÉS)
    ────────────────────────────────────────────────────────────────────
    L'ancien schéma faisait venir la TAILLE d'une frame et les DONNÉES
    d'une autre : `taille` était le compteur de n−1, le tampon contenait
    l'émission de n. Il en découlait deux pertes symétriques :
      (α) frame émettant PLUS que la précédente : les couples au-delà de
          `taille` n'étaient jamais transférés, alors que le kernel avait
          déjà fait `reference[p] += d` pour eux. Perte DÉFINITIVE — le
          détail est absorbé dans la référence et ne repasse plus le
          seuil. Le nota « le schéma étant incrémental, rien n'est perdu »
          était FAUX ; il est RETIRÉ (§A21), pas nuancé ;
      (β) frame émettant MOINS : la remise à zéro du compteur ne réécrit
          pas les buffers, et les positions [émis, taille) portaient
          encore les couples de la frame précédente — retransférés et
          RÉAPPLIQUÉS.

    ────────────────────────────────────────────────────────────────────
    LE PING-PONG, ET POURQUOI IL SUFFIT
    ────────────────────────────────────────────────────────────────────
    Deux jeux de buffers. La frame n écrit dans le jeu `n mod 2` ; le
    transfert de la frame n porte sur le jeu de n−1, dimensionné par le
    compteur de n−1. **Taille et données du même tour** : la cause commune
    des deux défauts disparaît, et avec elle les deux défauts.

    Le compteur, lui, n'avait rien de faux : `hote[0]` porte déjà le
    compte de n−1. C'est la DONNÉE qui était de la mauvaise frame. Le
    mécanisme du choix 6 est donc conservé tel quel.

    CHOIX 6 PRÉSERVÉ, ET C'EST VÉRIFIABLE : le correctif n'ajoute qu'une
    allocation à la construction (B2) et un basculement d'indice Python.
    Aucun `synchronize()` de stream ou de device, aucun
    `int(tableau_device)`, aucune copie bloquante par frame — un test
    l'impose en interdisant ces points d'entrée pendant une série de
    frames. SEULE attente autorisée (M8) : l'évènement du memcpy du
    compteur (4 octets, soumis une frame entière plus tôt — attente quasi
    nulle en régime), parce que sans lui l'invariant « taille et données
    du même tour » ne tenait que par les synchronisations incidentes des
    voisins (chrono B6, observateur, `remonter`).

    COÛT : les buffers d'émission doublent. Ils étaient dimensionnés au
    PIRE CAS (choix 5) ; `octets_buffers()` reporte le total réel pour que
    la résidence le montre.

    B2 : tout est préalloué à la construction ; la boucle de frame
    n'alloue rien."""

    N_JEUX: int = 2               # ping-pong : n mod 2

    def __init__(self, cp, taille_max: int):
        self.cp = cp
        self.taille_max = int(taille_max)
        # Les DEUX jeux, préalloués (B2). `_jeu` désigne celui où le
        # kernel de la frame courante écrit ; l'autre porte la frame
        # précédente, intégralement, et n'attend que son transfert.
        self._valeurs = [cp.empty(self.taille_max, dtype=cp.float32)
                         for _ in range(self.N_JEUX)]
        self._indices = [cp.empty(self.taille_max, dtype=cp.uint32)
                         for _ in range(self.N_JEUX)]
        self._jeu = 0
        # L'offset d'indices de chaque jeu. Les indices émis sont
        # RELATIFS au tableau passé au kernel (la tranche du tour) : un
        # consommateur qui reconstruit doit y ajouter l'offset du tour
        # ÉMETTEUR. Il doit donc voyager AVEC les données, sous le même
        # invariant qu'elles — sinon le ping-pong livre les données d'un
        # tour indexées par l'offset d'un autre.
        self._offsets = [0 for _ in range(self.N_JEUX)]
        self.compteur = cp.zeros(1, dtype=cp.uint32)
        # Tampon hôte page-locked : la copie asynchrone y atterrit.
        self._memoire_pinned = cp.cuda.alloc_pinned_memory(4)
        self.hote = np.frombuffer(self._memoire_pinned, dtype=np.uint32,
                                  count=1)
        self.hote[0] = 0          # frame 0 : aucun précédent (warmup, B6)
        # Évènement enregistré juste APRÈS le memcpyAsync du compteur
        # (review 28/07, M8) : sans lui, `taille_precedente()` lisait
        # `hote[0]` en espérant que la copie de la frame n−1 avait atterri
        # — vrai seulement grâce aux synchronisations INCIDENTES d'autres
        # composants (chrono B6, observateur, `remonter`). L'évènement rend
        # l'invariant vrai PAR CONSTRUCTION ; l'attente est quasi nulle en
        # régime (4 octets déjà partis depuis une frame entière). Préalloué
        # ici (B2), réutilisé à chaque frame ; jamais enregistré = attente
        # nulle (sémantique CUDA), ce qui couvre la frame 0.
        self._evenement_compteur = cp.cuda.Event(block=False,
                                                 disable_timing=True)
        self.tailles_vues: list[int] = []
        self.compteur_final = 0

    # ----- le jeu COURANT : c'est lui que le kernel reçoit -----

    @property
    def valeurs(self):
        """Buffer de valeurs de la frame COURANTE — celui que `pas_f_l3`
        passe au kernel."""
        return self._valeurs[self._jeu]

    @property
    def indices(self):
        """Buffer d'indices de la frame COURANTE."""
        return self._indices[self._jeu]

    def taille_precedente(self) -> int:
        """Compteur de la frame précédente — lecture HÔTE, garantie par
        l'évènement du memcpy (M8), pas par des syncs incidentes.

        C'est la taille du jeu PRÉCÉDENT, celui que `vues_a_transferer`
        rend : les deux viennent du même tour. Sans l'attente d'évènement,
        une lecture qui précéderait l'atterrissage du memcpy découperait le
        jeu de n−1 avec la taille de n−2 — exactement les défauts (α)/(β)
        que le ping-pong prétend éliminer par construction."""
        self._evenement_compteur.synchronize()
        return int(min(int(self.hote[0]), self.taille_max))

    def noter_taille(self, taille: int) -> None:
        """Enregistre la taille retenue pour cette frame (diagnostic du
        retard, agrégé hors chrono — une liste Python, pas de device)."""
        self.tailles_vues.append(int(taille))

    def cloturer_frame(self) -> None:
        """Copie asynchrone du compteur vers l'hôte (pour la frame
        suivante), remise à zéro, puis BASCULE du ping-pong. Les deux
        premiers ordres sont soumis au stream courant, donc exécutés
        APRÈS le kernel de cette frame ; la bascule est un entier Python.

        Après la bascule, le jeu que le kernel vient de remplir devient le
        jeu « précédent » — c'est lui que la frame suivante transférera,
        avec le compteur qui l'accompagne."""
        cp = self.cp
        cp.cuda.runtime.memcpyAsync(
            self._memoire_pinned.ptr, self.compteur.data.ptr, 4,
            cp.cuda.runtime.memcpyDeviceToHost,
            cp.cuda.Stream.null.ptr)
        # L'évènement marque l'atterrissage du memcpy ci-dessus (M8) :
        # `taille_precedente()` attendra LUI, pas le stream entier.
        self._evenement_compteur.record(cp.cuda.Stream.null)
        self.compteur.fill(0)
        self._jeu = (self._jeu + 1) % self.N_JEUX

    def noter_offset(self, offset: int) -> None:
        """Enregistre, pour le jeu COURANT, l'offset d'indices de
        l'émission de cette frame.

        Les indices émis sont relatifs au tableau passé au kernel — la
        tranche du tour, pas le groupe entier. L'offset qui les remet en
        place appartient donc au tour ÉMETTEUR, et doit voyager avec les
        données sous le MÊME invariant : taille, données ET offset du même
        tour. Le lui refuser livrerait les couples d'un tour indexés par
        l'offset d'un autre — écriture silencieuse dans le mauvais slot."""
        self._offsets[self._jeu] = int(offset)

    def offset_precedent(self) -> int:
        """Offset du jeu que `vues_a_transferer` rend — celui du tour qui
        a réellement émis ces couples."""
        return self._offsets[(self._jeu + 1) % self.N_JEUX]

    def vues_a_transferer(self, taille: int):
        """Tranches contiguës à remonter, prises dans le jeu PRÉCÉDENT —
        celui que le compteur `taille` décrit, et que `offset_precedent`
        indexe. Taille, données et offset du même tour : c'est là que
        l'invariant se tient."""
        precedent = (self._jeu + 1) % self.N_JEUX
        return (self._valeurs[precedent][:taille],
                self._indices[precedent][:taille])

    def octets_buffers(self) -> int:
        """Empreinte VRAM des buffers d'émission — les DEUX jeux. À
        reporter tel quel dans la résidence : le ping-pong double le
        dimensionnement au pire cas (choix 5), et cela doit se voir."""
        return int(sum(b.nbytes for b in self._valeurs)
                   + sum(b.nbytes for b in self._indices))

    def diagnostic_retard(self) -> dict:
        """Ce que le retard d'une frame coûte, rendu lisible.

        Le transfert de la frame *n* porte sur les DONNÉES et le COMPTEUR
        de *n−1* : la variabilité des tailles n'entame plus la complétude
        du transfert, elle ne dit plus que la variabilité de l'émission.
        Seule la dernière frame reste en attente à la fin de la série.
        Agrégé HORS chrono."""
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
            "octets_buffers": self.octets_buffers(),
            "note": ("retard d'une frame (choix 6), buffers en PING-PONG "
                     "(§A21) : données ET compteur viennent du même tour, "
                     "donc tout ce qui est émis est transféré une fois et "
                     "une seule. `tailles_stables` ne conditionne plus la "
                     "complétude — il ne décrit plus que la variabilité de "
                     "l'émission. Seule la dernière frame de la série "
                     "reste en attente."),
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
