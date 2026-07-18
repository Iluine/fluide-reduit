"""F1 tranche-1 — composant (a2/a3) : pyramide de fenêtres actives + fovéa
mobile + schéma diff (choix B2, B3, B4 ; décisions gravées E4b, E4c, E4d).

Protocole gravé (§A15 M-a) : « Pyramide active à l'enveloppe §6 épinglée
(c=8, f32, n_fov=512, N_niv=10, γ₂≈3). F représentatif appliqué chaque
frame sur les fenêtres actives, fenêtre fovéa MOBILE (translation continue
-> re-prédiction Harten à chaque déplacement, le coût structurel que la
fovéa fixe ne paie pas). » Topologie §5 (spec) : niveau 0 CPU (vivant),
fenêtres actives GPU, descente = prédiction, remontée = coefficients de
détail — « l'interface CPU<->GPU est la transformée elle-même ».

B3 — géométrie : niveau 0 CPU de côté N0 = 256 (matérialisé numpy) ;
niveau j ∈ [1, N_niv−1] : monde virtuel de côté N0·2^j (JAMAIS
matérialisé), γ₂ = 3 fenêtres n_fov² par niveau — la fenêtre fovéale du
niveau + 2 fenêtres d'énergie à offsets y figés ±n_fov, clampées au monde
(chevauchement possible aux niveaux grossiers : coût de calcul identique,
jetable). Alignement dyadique : l'origine x au niveau j suit
⌊centre_fin/2^(J−j)⌋ − n_fov/2 (clampée) — elle avance d'1 cellule toutes
les 2^(J−j) frames quand la fovéa balaie à 1 cellule fine/frame (E4c).

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

from dataclasses import dataclass

import numpy as np

from src.f1_gpu.substrat_jetable import etat_initial_jetable, pas_f_jetable
from src.f1_gpu.transferts import TransfertComptable

N0_DEFAUT: int = 256        # côté du niveau 0 CPU (B3)
GAMMA2: int = 3             # fenêtres par niveau (γ₂ ≈ 3, note VRAM / §6)
EPS_DETAIL: float = 1e-4    # seuil du diff (B4) — paramètre d'instrument figé
GRAINE_MONDE: int = 12345   # contenu jetable seedé (reproductible, non mesuré)


@dataclass(frozen=True)
class GeometriePyramide:
    """Géométrie B3 (pure, testable CPU) : niveaux, mondes virtuels,
    origines des fenêtres en fonction du centre fovéal fin."""
    n_fov: int
    n_niv: int
    n0: int = N0_DEFAUT
    gamma2: int = GAMMA2

    def __post_init__(self):
        if self.n_niv < 2:
            raise ValueError(
                f"GeometriePyramide : n_niv={self.n_niv} < 2 (il faut au "
                "moins le niveau 0 CPU + un niveau GPU, B3).")

    @property
    def niveau_fin(self) -> int:
        return self.n_niv - 1

    @property
    def niveaux_gpu(self) -> range:
        """Niveaux à fenêtres GPU : j = 1..N_niv−1 (le niveau 0 est CPU,
        §5). Le compte de cellules actives GPU vaut donc
        γ₂·n_fov²·(N_niv−1) — l'enveloppe γ₂·n_fov²·N_niv de la note VRAM
        compte aussi le niveau 0 ; l'écart est reporté, jamais masqué."""
        return range(1, self.n_niv)

    @property
    def cellules_actives_gpu(self) -> int:
        return self.gamma2 * self.n_fov * self.n_fov * (self.n_niv - 1)

    def cote_monde(self, j: int) -> int:
        return self.n0 * (2 ** j)

    def centre_fin_initial(self) -> int:
        return self.cote_monde(self.niveau_fin) // 2

    def _clamp_origine(self, o: int, j: int) -> int:
        return max(0, min(o, max(self.cote_monde(j) - self.n_fov, 0)))

    def origines(self, j: int, centre_fin: int) -> list[tuple[int, int]]:
        """Origines (oy, ox) des γ₂ fenêtres du niveau j pour un centre
        fovéal fin donné : fenêtre 0 = fovéale du niveau, fenêtres 1/2 =
        énergie, offsets y = ±n_fov (B3). Le x est COMMUN aux γ₂ fenêtres
        (elles suivent le balayage ensemble)."""
        centre_j = centre_fin // (2 ** (self.niveau_fin - j))
        ox = self._clamp_origine(centre_j - self.n_fov // 2, j)
        oy_base = self._clamp_origine(
            self.cote_monde(j) // 2 - self.n_fov // 2, j)
        origines = []
        for decalage in (0, self.n_fov, -self.n_fov):
            origines.append((self._clamp_origine(oy_base + decalage, j), ox))
        return origines[: self.gamma2]


class PyramideFovea:
    """Pyramide de fenêtres actives GPU + fovéa mobile + schéma diff —
    l'objet que les drivers M-a et M-c font tourner (une frame =
    `frame()` ; le chrono et les compteurs vivent DEHORS : chrono (f),
    `TransfertComptable` (c)).

    Buffers préalloués à la construction (B2) : `fenetres[j]` et
    `references[j]` de shape (γ₂, 2, 4, n_fov, n_fov) f32 par niveau GPU ;
    `monde0` (2, 4, N0, N0) f32 CPU. La descente initiale (prédiction
    pleine de chaque fenêtre) passe par `transferts` À LA CONSTRUCTION —
    le driver clôt ce bilan d'initialisation par `frame_suivante()` et
    l'EXCLUT du régime établi (B4 : payée hors-série)."""

    def __init__(self, xp, geo: GeometriePyramide,
                 transferts: TransfertComptable,
                 graine: int = GRAINE_MONDE, eps_detail: float = EPS_DETAIL):
        self.xp = xp
        self.geo = geo
        self.transferts = transferts
        self.eps_detail = float(eps_detail)
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
                             x_debut: int, x_fin: int) -> np.ndarray:
        """Prédiction voisin (motif M6) du bloc [oy:oy+n_fov) x
        [ox+x_debut:ox+x_fin) du niveau j depuis `monde0` — indices
        sources clippés au monde 0 (fenêtres clampées, B3)."""
        geo = self.geo
        facteur = 2 ** j
        iy = np.clip((oy + np.arange(geo.n_fov)) // facteur, 0, geo.n0 - 1)
        ix = np.clip((ox + np.arange(x_debut, x_fin)) // facteur,
                     0, geo.n0 - 1)
        return self.monde0[:, :, iy[:, None], ix[None, :]]

    def _predire_niveau_cpu(self, j: int, origines: list[tuple[int, int]],
                            x_debut: int, x_fin: int) -> np.ndarray:
        """Bloc batché (γ₂, 2, 4, n_fov, x_fin−x_debut) f32 — UNE descente
        par niveau (B4)."""
        return np.stack([
            self._predire_fenetre_cpu(j, oy, ox, x_debut, x_fin)
            for oy, ox in origines])

    # ----- la frame -----

    def frame(self, delta_x: int = 1) -> dict:
        """Une frame complète (E4c : balayage `delta_x` = 1 cellule
        fine/frame par défaut ; `delta_x` >= n_fov = « saut de fenêtre »,
        diagnostic NON-verdictal E4c — re-prédiction pleine) :
          1. déplacer la fovéa ; par niveau déplacé, décaler les fenêtres
             et DESCENDRE les cellules entrantes prédites (B4, H2D) ;
          2. F jetable sur les fenêtres actives de chaque niveau (E4a) ;
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

            # 2. F jetable (batch γ₂ fenêtres du niveau, E4a/B5).
            _, dt_cfl = pas_f_jetable(fen, xp, sortie=fen)

            # 3. Remontée : coefficients de détail seuillés (B4/E4d).
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
        return (self.geo.gamma2 * 2 * 4 * geo.n_fov * geo.n_fov * 4
                * (geo.n_niv - 1))
