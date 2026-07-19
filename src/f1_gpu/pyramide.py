"""F1 tranche-1 — composant (a2/a3) : pyramide de fenêtres actives + fovéa
mobile + schéma diff (choix B2, B3, B4 ; décisions gravées E4b, E4c, E4d).

Protocole gravé (§A15 M-a) : « Pyramide active à l'enveloppe §6 épinglée
(c=8, f32, n_fov=512, N_niv=10, γ₂≈3). F représentatif appliqué chaque
frame sur les fenêtres actives, fenêtre fovéa MOBILE (translation continue
-> re-prédiction Harten à chaque déplacement, le coût structurel que la
fovéa fixe ne paie pas). » Topologie §5 (spec) : niveau 0 CPU (vivant),
fenêtres actives GPU, descente = prédiction, remontée = coefficients de
détail — « l'interface CPU<->GPU est la transformée elle-même ».

RE-ÉPINGLAGE §A16 (pocCascade2phys 38a2739) — CONFIGURATION PAR SLOTS :
l'enveloppe DENSE (γ₂ fixe par niveau) est remplacée par un CAP DUR
D'EMPLACEMENTS — le budget est une LISTE DE SLOTS (niveau, c), somme des
coûts <= 16.7 ms (T2 inchangé). Un slot ≡ une fenêtre n_fov² à un niveau ;
c=8 <=> 2 systèmes (h, hu, hv, s), c=4 <=> 1 système. Le γ₂=3 fixe reste
le DÉFAUT (rétro-compatibilité stricte de M-a/M-c : `slots` non fourni =>
γ₂ slots c=8 par niveau GPU, géométrie tranche-1 au bit près). La fovéa
mobile E4c et les offsets B3 sont INCHANGÉS : les slots d'un niveau se
placent aux offsets B3 dans l'ordre (fovéal d'abord, puis énergie), et
translatent en LOCKSTEP (géométrie 4e0e719).

B3 — géométrie (rétablie §A15-complément-3, décision Romain : « les 3
fenêtres bougent ») : niveau 0 CPU de côté N0 = 256 (matérialisé numpy) ;
niveau j ∈ [1, N_niv−1] : monde virtuel de côté N0·2^j (JAMAIS
matérialisé), les fenêtres n_fov² du niveau — la fenêtre fovéale du
niveau + les fenêtres d'énergie à offsets y RELATIFS figés ±n_fov (offsets
de POSITION, pas fenêtres statiques), clampées au monde (chevauchement
possible aux niveaux grossiers : coût de calcul identique, jetable).
Les fenêtres d'un niveau translatent en LOCKSTEP avec le balayage E4c
(x commun) — majorant honnête de la descente, cohérent E4a (pour un
critère de mort, on mesure la borne haute). PORTÉE gravée
(§A15-complément-3) : le trafic de déplacement mesuré est un MAJORANT DE
CADENCE (lockstep avec le regard) ; la géométrie de production (fenêtres
d'énergie pilotées par l'énergie, cadence irrégulière) n'est PAS mesurée.
Alignement dyadique : l'origine x au niveau j suit ⌊centre_fin/2^(J−j)⌋
− n_fov/2 (clampée) — elle avance d'1 cellule toutes les 2^(J−j) frames
quand la fovéa balaie à 1 cellule fine/frame (E4c). B3 ne définit que
TROIS offsets (0, +n_fov, −n_fov) : un niveau porte donc au plus 3 slots
(1 fovéal + 2 énergie) — au-delà, fail-loud (inventer un 4e offset serait
une géométrie NEUVE, pas mesurée).

B4 — schéma diff (E4d) : à chaque déplacement d'un niveau, les cellules
ENTRANTES seulement sont prédites côté CPU depuis le niveau 0 (prédiction
VOISIN — le motif `upsample` de `src/multiresolution.py`, prototype M6 du
dépôt ; prédiction directe depuis le niveau 0 : les octets descendus sont
identiques à une chaîne niveau-à-niveau, seul le calcul CPU trivial
diffère — nommé) et descendues en UNE H2D batchée par niveau. La remontée,
chaque frame et pour chaque niveau touché par F : coefficients de détail
d = fenêtre − référence, SEUILLÉS à |d| >= EPS_DETAIL (valeurs f32 +
indices u32, deux D2H par niveau), puis référence += coefficients remontés
(schéma incrémental : les résidus sous-seuil S'ACCUMULENT dans d et
finissent par franchir le seuil — rien n'est perdu silencieusement, le
CPU « sait » exactement ce qui est remonté). EPS_DETAIL est un paramètre
d'INSTRUMENT figé avant run, reporté dans chaque JSON — pas un seuil de
verdict.

B2 (E4b) — tous les buffers d'état (fenêtres, références, monde 0) sont
préalloués ici à shapes fixes ; la boucle de frame n'alloue de l'état
JAMAIS (les temporaires du stencil et du `roll` passent par le mempool)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.f1_gpu.substrat_jetable import etat_initial_jetable, pas_f_jetable
from src.f1_gpu.transferts import TransfertComptable

N0_DEFAUT: int = 256        # côté du niveau 0 CPU (B3)
GAMMA2: int = 3             # fenêtres par niveau (γ₂ ≈ 3, note VRAM / §6)
EPS_DETAIL: float = 1e-4    # seuil du diff (B4) — paramètre d'instrument figé
GRAINE_MONDE: int = 12345   # contenu jetable seedé (reproductible, non mesuré)

CHAMPS_PAR_SYSTEME: int = 4          # (h, hu, hv, s) — E4a
SYSTEMES_PAR_C: dict[int, int] = {4: 1, 8: 2}   # c (champs) -> systèmes
C_DEFAUT: int = 8                    # enveloppe tranche-1 (§6 avant §A16)
ROLES_SLOT: tuple[str, ...] = ("fovea", "energie")
# Offsets y des fenêtres d'un niveau, en multiples de n_fov (B3) : le slot
# d'indice 0 est la fenêtre FOVÉALE (offset 0), les suivants l'énergie.
OFFSETS_B3: tuple[int, ...] = (0, 1, -1)


@dataclass(frozen=True)
class Slot:
    """Un EMPLACEMENT du cap dur §A16 : une fenêtre n_fov² au niveau
    `niveau`, portant `c` champs (c=8 <=> 2 systèmes, c=4 <=> 1 système).
    `role` est du REPORTING (§A16 budgète séparément fovéale et énergie) ;
    la géométrie, elle, ne lit que l'ORDRE des slots d'un niveau (indice 0
    = offset B3 nul = fenêtre fovéale)."""
    niveau: int
    c: int = C_DEFAUT
    role: str = "fovea"

    def __post_init__(self) -> None:
        if self.c not in SYSTEMES_PAR_C:
            raise ValueError(
                f"Slot : c={self.c} hors des valeurs gravées "
                f"{sorted(SYSTEMES_PAR_C)} (§A16 : c=8 => 2 systèmes, "
                "c=4 => 1 système).")
        if self.role not in ROLES_SLOT:
            raise ValueError(
                f"Slot : role={self.role!r} inconnu (attendu {ROLES_SLOT}).")

    @property
    def n_systemes(self) -> int:
        return SYSTEMES_PAR_C[self.c]


def slots_uniformes(n_niv: int, gamma2: int = GAMMA2,
                    c: int = C_DEFAUT) -> tuple[Slot, ...]:
    """Liste de slots de l'enveloppe DENSE tranche-1 : `gamma2` slots à `c`
    champs sur chaque niveau GPU (le slot 0 fovéal, les suivants énergie).
    C'est le défaut de `GeometriePyramide` — M-a/M-c inchangés."""
    slots: list[Slot] = []
    for j in range(1, n_niv):
        for k in range(gamma2):
            slots.append(Slot(niveau=j, c=c,
                              role="fovea" if k == 0 else "energie"))
    return tuple(slots)


@dataclass(frozen=True)
class GeometriePyramide:
    """Géométrie B3 (pure, testable CPU) : niveaux, mondes virtuels,
    origines des fenêtres en fonction du centre fovéal fin — configurée
    par une LISTE DE SLOTS (§A16). `slots` vide => enveloppe dense
    `slots_uniformes(n_niv, gamma2)` (comportement tranche-1)."""
    n_fov: int
    n_niv: int
    n0: int = N0_DEFAUT
    gamma2: int = GAMMA2
    slots: tuple[Slot, ...] = field(default=())

    def __post_init__(self):
        if self.n_niv < 2:
            raise ValueError(
                f"GeometriePyramide : n_niv={self.n_niv} < 2 (il faut au "
                "moins le niveau 0 CPU + un niveau GPU, B3).")
        if not self.slots:
            object.__setattr__(
                self, "slots", slots_uniformes(self.n_niv, self.gamma2))
        else:
            object.__setattr__(self, "slots", tuple(self.slots))
        self._valider_slots()

    # ----- validation de la liste de slots (fail-loud, §A16) -----

    def _valider_slots(self) -> None:
        niveaux_gpu = set(self.niveaux_gpu)
        for slot in self.slots:
            if slot.niveau not in niveaux_gpu:
                raise ValueError(
                    f"GeometriePyramide : slot au niveau {slot.niveau} hors "
                    f"des niveaux GPU {sorted(niveaux_gpu)} (le niveau 0 est "
                    "CPU, §5).")
        for j in sorted(niveaux_gpu):
            du_niveau = self.slots_du_niveau(j)
            if not du_niveau:
                raise ValueError(
                    f"GeometriePyramide : niveau GPU {j} sans aucun slot — "
                    "une pyramide trouée n'est pas mesurable (fail-loud).")
            if len(du_niveau) > len(OFFSETS_B3):
                raise ValueError(
                    f"GeometriePyramide : {len(du_niveau)} slots au niveau "
                    f"{j} > {len(OFFSETS_B3)} offsets B3 disponibles "
                    "(0, +n_fov, -n_fov) — une géométrie à plus d'offsets "
                    "serait NEUVE, non mesurée.")
            if len({s.c for s in du_niveau}) > 1:
                raise ValueError(
                    f"GeometriePyramide : c hétérogène au niveau {j} "
                    f"({sorted(s.c for s in du_niveau)}) — les fenêtres d'un "
                    "niveau sont batchées dans UN buffer, c doit y être "
                    "uniforme.")
            if du_niveau[0].role != "fovea":
                raise ValueError(
                    f"GeometriePyramide : le slot d'indice 0 du niveau {j} "
                    f"a le rôle {du_niveau[0].role!r} — l'offset B3 nul est "
                    "la fenêtre FOVÉALE (ordre load-bearing).")

    # ----- niveaux -----

    @property
    def niveau_fin(self) -> int:
        return self.n_niv - 1

    @property
    def niveaux_gpu(self) -> range:
        """Niveaux à fenêtres GPU : j = 1..N_niv−1 (le niveau 0 est CPU,
        §5). Le compte de cellules actives GPU vaut donc
        Σ_j n_slots(j)·n_fov² — l'enveloppe γ₂·n_fov²·N_niv de la note VRAM
        compte aussi le niveau 0 ; l'écart est reporté, jamais masqué."""
        return range(1, self.n_niv)

    # ----- slots -----

    def slots_du_niveau(self, j: int) -> tuple[Slot, ...]:
        """Slots du niveau `j`, DANS L'ORDRE de déclaration (l'indice 0
        prend l'offset B3 nul : c'est la fenêtre fovéale)."""
        return tuple(s for s in self.slots if s.niveau == j)

    def n_slots(self, j: int) -> int:
        return len(self.slots_du_niveau(j))

    def n_systemes_du_niveau(self, j: int) -> int:
        """Systèmes par fenêtre au niveau `j` (uniforme par construction,
        validé) : 2 si c=8, 1 si c=4."""
        return self.slots_du_niveau(j)[0].n_systemes

    def c_du_niveau(self, j: int) -> int:
        return self.slots_du_niveau(j)[0].c

    @property
    def n_slots_total(self) -> int:
        return len(self.slots)

    @property
    def blocs_actifs_gpu(self) -> int:
        """Nombre de BLOCS (fenêtre, système) — l'unité du modèle de coût
        §A16 (1 bloc ≡ 1 slot à c=4 ≡ 0.843 ms mesuré ; un slot c=8 en
        vaut 2). C'est ce compte, pas le compte de slots, qui est linéaire
        dans le coût prédit."""
        return sum(s.n_systemes for s in self.slots)

    @property
    def cellules_actives_gpu(self) -> int:
        """Cellules actives GPU = Σ slots · n_fov² (indépendant de c : le
        c multiplie les CHAMPS par cellule, pas les cellules — §A16 :
        « 12 slots ≈ 3.15 M cellules »)."""
        return self.n_slots_total * self.n_fov * self.n_fov

    # ----- mondes virtuels et origines (B3, INCHANGÉ) -----

    def cote_monde(self, j: int) -> int:
        return self.n0 * (2 ** j)

    def centre_fin_initial(self) -> int:
        return self.cote_monde(self.niveau_fin) // 2

    def _clamp_origine(self, o: int, j: int) -> int:
        return max(0, min(o, max(self.cote_monde(j) - self.n_fov, 0)))

    def origines(self, j: int, centre_fin: int) -> list[tuple[int, int]]:
        """Origines (oy, ox) des fenêtres du niveau j pour un centre fovéal
        fin donné — UNE par slot du niveau : slot 0 = fovéale (offset B3
        nul), slots suivants = énergie, offsets y RELATIFS = ±n_fov (B3).
        Le x est COMMUN aux fenêtres — elles translatent en LOCKSTEP avec
        le balayage (géométrie rétablie §A15-complément-3 : majorant de
        cadence)."""
        centre_j = centre_fin // (2 ** (self.niveau_fin - j))
        ox = self._clamp_origine(centre_j - self.n_fov // 2, j)
        oy_base = self._clamp_origine(
            self.cote_monde(j) // 2 - self.n_fov // 2, j)
        origines = []
        for multiple in OFFSETS_B3[: self.n_slots(j)]:
            oy = oy_base + multiple * self.n_fov
            origines.append((self._clamp_origine(oy, j), ox))
        return origines

    # ----- compte rendu de la configuration (JSON des drivers) -----

    def resume_slots(self) -> dict:
        """Description auditable de la config (le driver la grave dans son
        JSON à côté de la config pré-enregistrée)."""
        par_niveau = {
            str(j): {"n_slots": self.n_slots(j), "c": self.c_du_niveau(j),
                     "n_systemes": self.n_systemes_du_niveau(j),
                     "roles": [s.role for s in self.slots_du_niveau(j)]}
            for j in self.niveaux_gpu}
        par_role = {
            role: {"n_slots": sum(1 for s in self.slots if s.role == role),
                   "n_blocs": sum(s.n_systemes for s in self.slots
                                  if s.role == role)}
            for role in ROLES_SLOT}
        return {"n_slots_total": self.n_slots_total,
                "blocs_actifs_gpu": self.blocs_actifs_gpu,
                "cellules_actives_gpu": self.cellules_actives_gpu,
                "par_niveau": par_niveau, "par_role": par_role}


def appliquer_jetable(fenetres, xp, sortie):
    """Applicateur de F par DÉFAUT : le F jetable (a1, B5) — accepte 1 ou
    2 systèmes tel quel (aucune contrainte de parité)."""
    return pas_f_jetable(fenetres, xp, sortie=sortie)


class PyramideFovea:
    """Pyramide de fenêtres actives GPU + fovéa mobile + schéma diff —
    l'objet que les drivers M-a, M-c et M-a-ter font tourner (une frame =
    `frame()` ; le chrono et les compteurs vivent DEHORS : chrono (f),
    `TransfertComptable` (c)).

    Buffers préalloués à la construction (B2) : `fenetres[j]` et
    `references[j]` de shape (n_slots(j), n_systemes(j), 4, n_fov, n_fov)
    f32 par niveau GPU ; `monde0` (2, 4, N0, N0) f32 CPU. La descente
    initiale (prédiction pleine de chaque fenêtre) passe par `transferts`
    À LA CONSTRUCTION — le driver clôt ce bilan d'initialisation par
    `frame_suivante()` et l'EXCLUT du régime établi (B4 : payée
    hors-série).

    `pas_f` (§A16) : l'applicateur de F, injectable — défaut
    `appliquer_jetable` (M-a/M-c, inchangés). M-a-ter injecte
    l'applicateur du kernel FUSIONNÉ ; la signature est
    `pas_f(fenetres, xp, sortie) -> (etat, dt_cfl)`.

    `remontee_active` (§A16-lecture-M-a-ter, sonde d'attribution) : met
    l'étage de REMONTÉE B4 hors service pour isoler le coût de F. CE QUE
    L'OFF RETIRE — du TRAVAIL, uniquement : la soustraction
    `fen − ref`, le seuillage `|d| >= eps`, l'indexation booléenne et le
    `flatnonzero` (compaction), les DEUX D2H, et la mise à jour
    incrémentale de la référence. CE QUE L'OFF NE RETIRE PAS — la
    STRUCTURE : `references[j]` reste PRÉALLOUÉ à sa shape pleine (B2
    identique, même résidence, même pression sur le mempool), la boucle
    de niveaux est la même, F s'applique aux mêmes buffers aux mêmes
    shapes. On mesure une pyramide amputée de son travail de remontée,
    pas une pyramide plus petite. `False` n'est JAMAIS un défaut : M-a,
    M-c et M-a-ter gardent la remontée."""

    def __init__(self, xp, geo: GeometriePyramide,
                 transferts: TransfertComptable,
                 graine: int = GRAINE_MONDE, eps_detail: float = EPS_DETAIL,
                 pas_f=appliquer_jetable, remontee_active: bool = True):
        self.xp = xp
        self.geo = geo
        self.transferts = transferts
        self.eps_detail = float(eps_detail)
        self.pas_f = pas_f
        self.remontee_active = bool(remontee_active)
        self.centre_fin = geo.centre_fin_initial()
        self.monde0 = etat_initial_jetable(1, geo.n0, graine)[0]  # CPU f32
        self.fenetres: dict[int, object] = {}
        self.references: dict[int, object] = {}
        self._origines: dict[int, list[tuple[int, int]]] = {}
        for j in geo.niveaux_gpu:
            origines = geo.origines(j, self.centre_fin)
            bloc = self._predire_niveau_cpu(j, origines, 0, geo.n_fov)
            fenetre = transferts.descendre(bloc)
            self.fenetres[j] = fenetre
            self.references[j] = xp.array(fenetre, copy=True)
            self._origines[j] = origines

    # ----- prédiction CPU depuis le niveau 0 (B4) -----

    def _predire_fenetre_cpu(self, j: int, oy: int, ox: int,
                             x_debut: int, x_fin: int,
                             n_systemes: int) -> np.ndarray:
        """Prédiction voisin (motif M6) du bloc [oy:oy+n_fov) x
        [ox+x_debut:ox+x_fin) du niveau j depuis `monde0` — indices
        sources clippés au monde 0 (fenêtres clampées, B3). Les
        `n_systemes` premiers systèmes du monde 0 sont pris (c=4 => 1
        système : le contenu jetable n'est pas l'objet de la mesure)."""
        geo = self.geo
        facteur = 2 ** j
        iy = np.clip((oy + np.arange(geo.n_fov)) // facteur, 0, geo.n0 - 1)
        ix = np.clip((ox + np.arange(x_debut, x_fin)) // facteur,
                     0, geo.n0 - 1)
        return self.monde0[:n_systemes, :, iy[:, None], ix[None, :]]

    def _predire_niveau_cpu(self, j: int, origines: list[tuple[int, int]],
                            x_debut: int, x_fin: int) -> np.ndarray:
        """Bloc batché (n_slots(j), n_systemes(j), 4, n_fov, x_fin−x_debut)
        f32 — UNE descente par niveau (B4)."""
        n_systemes = self.geo.n_systemes_du_niveau(j)
        return np.stack([
            self._predire_fenetre_cpu(j, oy, ox, x_debut, x_fin, n_systemes)
            for oy, ox in origines])

    # ----- la frame -----

    def frame(self, delta_x: int = 1) -> dict:
        """Une frame complète (E4c : balayage `delta_x` = 1 cellule
        fine/frame par défaut ; `delta_x` >= n_fov = « saut de fenêtre »,
        diagnostic NON-verdictal E4c — re-prédiction pleine) :
          1. déplacer la fovéa ; par niveau déplacé, décaler les fenêtres
             et DESCENDRE les cellules entrantes prédites (B4, H2D) ;
          2. F sur les fenêtres actives de chaque niveau (E4a), via
             `pas_f` (jetable par défaut, fusionné pour M-a-ter) ;
          3. REMONTER les coefficients de détail seuillés de chaque niveau
             touché (B4, D2H), référence += coefficients remontés.

        Retourne un diagnostic léger {"niveaux_deplaces", "nnz_par_niveau",
        "dt_cfl_dernier"} — les octets/temps vivent dans `transferts`, le
        chrono dans le driver."""
        xp = self.xp
        geo = self.geo
        if delta_x < 0:
            raise ValueError(f"frame : delta_x={delta_x} < 0 (balayage +x).")
        self.centre_fin += delta_x

        niveaux_deplaces: list[int] = []
        nnz_par_niveau: dict[str, int] = {}
        dt_cfl = 0.0
        for j in geo.niveaux_gpu:
            fen = self.fenetres[j]
            ref = self.references[j]
            nouvelles = geo.origines(j, self.centre_fin)
            dx = nouvelles[0][1] - self._origines[j][0][1]
            if dx < 0:
                raise RuntimeError(
                    f"frame : recul d'origine au niveau {j} (dx={dx}) — "
                    "balayage E4c strictement +x.")
            if dx > 0:
                # Géométrie §A15-complément-3 : les fenêtres du niveau
                # translatent en LOCKSTEP (majorant de cadence) — la
                # descente entrante est batchée pour le niveau entier (B4).
                niveaux_deplaces.append(j)
                if dx >= geo.n_fov:
                    # Saut de fenêtre (diagnostic E4c) : re-prédiction pleine.
                    bloc = self._predire_niveau_cpu(j, nouvelles, 0, geo.n_fov)
                    entrant = self.transferts.descendre(bloc)
                    fen[...] = entrant
                    ref[...] = entrant
                else:
                    fen[...] = xp.roll(fen, -dx, axis=-1)
                    ref[...] = xp.roll(ref, -dx, axis=-1)
                    bloc = self._predire_niveau_cpu(
                        j, nouvelles, geo.n_fov - dx, geo.n_fov)
                    entrant = self.transferts.descendre(bloc)
                    fen[..., geo.n_fov - dx:] = entrant
                    ref[..., geo.n_fov - dx:] = entrant
            self._origines[j] = nouvelles

            # 2. F (batch des fenêtres du niveau, E4a/B5).
            _, dt_cfl = self.pas_f(fen, xp, fen)

            # 3. Remontée : coefficients de détail seuillés (B4/E4d).
            # `remontee_active=False` (sonde d'attribution) saute le
            # TRAVAIL de cet étage ; `ref` reste alloué et intact.
            if self.remontee_active:
                d = fen - ref
                masque = xp.abs(d) >= self.eps_detail
                valeurs = d[masque]
                indices = xp.flatnonzero(masque).astype(xp.uint32)
                self.transferts.remonter(valeurs)
                self.transferts.remonter(indices)
                ref += xp.where(masque, d, xp.float32(0.0))
                nnz_par_niveau[str(j)] = int(valeurs.size)

        return {"niveaux_deplaces": niveaux_deplaces,
                "nnz_par_niveau": nnz_par_niveau,
                "remontee_active": self.remontee_active,
                "dt_cfl_dernier": float(dt_cfl)}

    # ----- comptes rendus -----

    def octets_etat(self) -> int:
        """Octets des buffers d'ÉTAT GPU préalloués (fenêtres + références)
        — la part déterministe de la résidence (le mempool ajoute les
        temporaires recyclés, vérif #1 et gate §6 observent le tout)."""
        total = 0
        for j in self.geo.niveaux_gpu:
            total += int(self.fenetres[j].nbytes)
            total += int(self.references[j].nbytes)
        return total

    def octets_dense_equivalent_par_frame(self) -> int:
        """Équivalent DENSE de la remontée (si tous les coefficients de
        toutes les fenêtres remontaient chaque frame) — le majorant que le
        schéma diff doit battre, reporté par M-c à côté du mesuré (B4)."""
        geo = self.geo
        return (geo.blocs_actifs_gpu * CHAMPS_PAR_SYSTEME
                * geo.n_fov * geo.n_fov * 4)
