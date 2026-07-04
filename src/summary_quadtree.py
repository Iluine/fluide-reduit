"""Arc A / manche 1, famille 2 (spec GRAVÉE : PREREGISTRATION.md,
pocCascade2phys, commit `a8ed659`, section « Spec opérationnelle du
quadtree » ; reportée par `docs/superpowers/plans/arc-a-manche1.md`, commit
`78fbe0c`) -- raffinement adaptatif par blocs, l'adaptativité de Harten au
sens canonique : blocs LARGES là où le champ est ~nul, FINS sur les dépôts.

Une responsabilité : résumer un champ (64, 64) en un `SummaryQT` (arbre
quadtree dyadique, moyennes de feuilles) et le régénérer en un champ
(64, 64) -- sans jamais garder de référence vivante vers le champ d'origine
(anti-fuite, cf. `regenerate_qt` -- signature à UN paramètre).

Domaine FIXE 64x64, profondeur max 6 (feuille 1x1 insécable, 2**6 == 64).
Construction gloutonne déterministe (`summarize_qt`) : gain d'un split =
SSE expliquée par ses 4 enfants ; on splitte le gain max d'abord ; tie-break
lexicographique (gain, puis y, puis x) ; un nœud à gain nul n'est JAMAIS
splitté (condition de la projection -- même si le budget reste) ; arrêt
quand le budget est atteint ou qu'il ne reste plus aucun gain > 0.

Comptabilité (clause 1 gravée) : `size_floats_qt` = n_feuilles +
ceil(n_nœuds / 32) -- 1 bit de topologie par nœud, préordre. Un split coûte
TOUJOURS +3 feuilles / +4 nœuds, quel que soit le nœud choisi : le budget
est donc vérifié AVANT de committer un split, et un refus est définitif
(aucun autre candidat ne coûterait moins cher -- le coût est structurel, pas
lié à la position du nœud).

`regenerate_qt` = upsampling CONSTANT-PAR-BLOCS ADAPTATIF (chaque feuille
peinte à sa moyenne) : sur un champ positif, la moyenne d'un bloc est
elle-même positive -- PAS de clip, S∘R = id exact (le motif gravé de
l'itération instrument, remonté d'un niveau à la géométrie adaptative).

Indexation array[y, x] comme le reste du dépôt. Note numérique (leçon
cascade, gravée) : toutes les moyennes de blocs sont calculées par CASCADE
2x2 ITÉRÉE (pyramide de moyennes -- jamais de réduction plate numpy sur
> 4 éléments) : c'est ce qui garantit que la moyenne d'un bloc peint
constant redonne la constante bit-exacte, donc des gains exactement 0.0 sur
le champ repeint, donc S∘R = id y compris l'ARBRE."""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

_TAILLE_DOMAINE: int = 64
_PROFONDEUR_MAX: int = 6  # 2**6 == 64 : feuille 1x1 à la profondeur max

# Ordre FIXE des 4 enfants d'un nœud dyadique, utilisé PARTOUT (construction,
# préordre topologie/moyennes, décodage) -- (drow, dcol) : top-left,
# top-right, bottom-left, bottom-right. Changer cet ordre casserait S∘R = id
# (la topologie encodée ne correspondrait plus au décodage).
_ORDRE_ENFANTS: tuple[tuple[int, int], ...] = ((0, 0), (0, 1), (1, 0), (1, 1))


@dataclass(frozen=True, eq=False)
class SummaryQT:
    """Résumé quadtree gelé d'un champ (64, 64) :
    - `topology` : bits préordre (np.ndarray uint8, 0/1) -- 1 = nœud INTERNE
      (a exactement 4 enfants), 0 = FEUILLE. n_nœuds = `topology.size`
      (internes + feuilles).
    - `means` : moyennes des feuilles (float64), dans le MÊME ordre préordre
      que les bits à 0 de `topology`. n_feuilles = `means.size`.
    - `shape` : (64, 64), domaine fixe d'origine.

    RIEN d'autre n'est stocké : position/taille de chaque feuille sont
    DÉRIVABLES de sa place dans le parcours préordre (cf. tests) -- pas de
    champ `depth`/`positions` redondant à maintenir en cohérence.

    `eq=False` (leçon revue M-0bis sur `Summary` de `src/summary.py`) : une
    dataclass dont un champ est un np.ndarray ne doit PAS générer
    `__eq__`/`__hash__` par défaut -- `==` déléguerait à `ndarray.__eq__`
    (élément par élément, valeur de vérité ambiguë)."""
    topology: np.ndarray
    means: np.ndarray
    shape: tuple[int, int]


def _valider_champ(field: np.ndarray) -> np.ndarray:
    """Valide la shape (domaine FIXE 64x64, cf. spec gravée) et renvoie une
    COPIE float64 (jamais une vue -- anti-fuite : `summarize_qt` ne doit
    conserver aucune référence vivante vers le tableau de l'appelant)."""
    field = np.asarray(field)
    if field.shape != (_TAILLE_DOMAINE, _TAILLE_DOMAINE):
        raise ValueError(
            f"summarize_qt : domaine fixe {_TAILLE_DOMAINE}x{_TAILLE_DOMAINE} "
            f"(reçu shape={field.shape}).")
    return np.array(field, dtype=np.float64, copy=True)


def _moyenne_blocs_2x2(bloc: np.ndarray) -> np.ndarray:
    """Une étape de la cascade dyadique (autonome -- famille 2 est un module
    indépendant, ne dépend pas de `src/summary.py`) : moyenne par blocs 2x2,
    somme des 4 valeurs ((a+b)+(c+d)) puis division par 4.0 (exacte en
    IEEE-754 : division par une puissance de 2)."""
    H, W = bloc.shape
    return bloc.reshape(H // 2, 2, W // 2, 2).sum(axis=(1, 3)) / 4.0


def _pyramide_moyennes(field: np.ndarray) -> list[np.ndarray]:
    """Pyramide de moyennes indexée par PROFONDEUR d ∈ [0, 6] :
    `niveaux[6]` = `field` (blocs 1x1, profondeur maximale), `niveaux[0]` =
    moyenne globale (1x1, racine). Chaque niveau plus grossier = UNE étape de
    cascade 2x2 du niveau plus fin (leçon cascade gravée -- jamais de
    réduction plate numpy sur > 4 éléments) : la moyenne d'un bloc peint
    constant redonne la constante bit-exacte."""
    niveaux: list[np.ndarray] = [field] * (_PROFONDEUR_MAX + 1)
    for d in range(_PROFONDEUR_MAX - 1, -1, -1):
        niveaux[d] = _moyenne_blocs_2x2(niveaux[d + 1])
    return niveaux


def _gains_par_profondeur(niveaux: list[np.ndarray]) -> list[np.ndarray]:
    """gains[d][row, col] = gain SSE du split du nœud (profondeur d, position
    (row, col)) en ses 4 enfants (profondeur d+1) : SSE expliquée =
    Σ_enfants n_c·(moyenne_enfant − moyenne_parent)², n_c = aire (pixels
    fins) d'UN enfant -- identique pour les 4 (partition dyadique).
    Vectorisé sur toute la grille de profondeur d en une passe : arithmétique
    DÉRIVÉE de moyennes déjà exactes (la contrainte « cascade 2x2 » porte sur
    le calcul des MOYENNES, cf. `_pyramide_moyennes` -- pas sur le gain).
    Propriété clef pour S∘R = id arbre inclus : sur le champ repeint, les
    moyennes sous-feuille sont bit-identiques à la moyenne de la feuille
    (cascade sur bloc constant), donc enfant − parent == 0.0 exact, donc
    gain == 0.0 exact -- jamais candidat."""
    gains: list[np.ndarray] = []
    for d in range(_PROFONDEUR_MAX):
        parent = niveaux[d]
        enfants = niveaux[d + 1]
        c00 = enfants[0::2, 0::2]
        c01 = enfants[0::2, 1::2]
        c10 = enfants[1::2, 0::2]
        c11 = enfants[1::2, 1::2]
        cote_enfant = 2 ** (_PROFONDEUR_MAX - (d + 1))
        n_c = float(cote_enfant * cote_enfant)
        gains.append(n_c * ((c00 - parent) ** 2 + (c01 - parent) ** 2
                            + (c10 - parent) ** 2 + (c11 - parent) ** 2))
    return gains


def summarize_qt(field: np.ndarray, budget: int) -> SummaryQT:
    """Construction gloutonne déterministe du quadtree (spec gravée) :
    - candidats = feuilles courantes de profondeur < 6 (1x1 insécable) à gain
      STRICTEMENT > 0 (un nœud à gain nul n'est JAMAIS splitté, même si le
      budget reste -- condition de la projection ; gain nul = 0.0 exact) ;
    - on splitte le gain max d'abord ; tie-break lexicographique
      (gain, puis y, puis x) : à gain égal, plus petit y d'abord puis plus
      petit x (y, x = coin haut-gauche du bloc en coordonnées fines) --
      réalisé par un tas min sur (−gain, y, x) ;
    - AVANT de committer un split, on vérifie la taille APRÈS
      (+3 feuilles / +4 nœuds, cf. `size_floats_qt`) contre `budget` : si
      elle dépasse, le split est REFUSÉ et la construction s'arrête -- le
      coût d'un split est identique quel que soit le nœud (structurel), donc
      aucun autre candidat ne serait passé non plus ;
    - arrêt : budget atteint ou plus aucun gain > 0.

    Fonction PURE : ne mute pas `field`, n'en conserve aucune référence
    vivante (copie systématique, cf. `_valider_champ`)."""
    field = _valider_champ(field)
    if budget < 2:
        raise ValueError(
            f"summarize_qt : budget minimal 2 (racine seule = 1 feuille + "
            f"ceil(1/32) = 2 floats-éq) -- reçu {budget!r}.")
    niveaux = _pyramide_moyennes(field)
    gains = _gains_par_profondeur(niveaux)

    internes: set[tuple[int, int, int]] = set()
    n_feuilles = 1
    n_noeuds = 1
    # Tas min sur (−gain, y, x) : pop = gain max, puis plus petit y, puis
    # plus petit x -- exactement le tie-break gravé. (depth, row, col) en
    # queue de tuple pour retrouver le nœud ((y, x) seuls ne suffisent pas :
    # des blocs de tailles différentes partagent le même coin haut-gauche.
    # Ils ne peuvent toutefois pas être candidats SIMULTANÉMENT -- seul un
    # des deux est feuille courante -- donc (−gain, y, x) ne produit jamais
    # d'ex æquo ambigu dans le tas ; depth complète l'ordre par sûreté.)
    tas: list[tuple[float, int, int, int, int, int]] = []

    def _pousser(depth: int, row: int, col: int) -> None:
        """Enfile la feuille (depth, row, col) si elle est candidate au
        split : profondeur < 6 ET gain strictement positif."""
        if depth >= _PROFONDEUR_MAX:
            return  # feuille 1x1 insécable
        gain = float(gains[depth][row, col])
        if gain <= 0.0:
            return  # gain nul : jamais splittée (condition de la projection)
        cote = 2 ** (_PROFONDEUR_MAX - depth)
        heapq.heappush(tas, (-gain, row * cote, col * cote, depth, row, col))

    _pousser(0, 0, 0)

    while tas:
        futurs_feuilles = n_feuilles + 3
        futurs_noeuds = n_noeuds + 4
        if futurs_feuilles + -(-futurs_noeuds // 32) > budget:
            break  # refus définitif (coût structurel identique pour tous)
        _neg_gain, _y, _x, depth, row, col = heapq.heappop(tas)
        internes.add((depth, row, col))
        n_feuilles, n_noeuds = futurs_feuilles, futurs_noeuds
        for dr, dc in _ORDRE_ENFANTS:
            _pousser(depth + 1, row * 2 + dr, col * 2 + dc)

    # Sérialisation préordre (racine, puis enfants dans _ORDRE_ENFANTS) :
    # 1 bit par nœud, 1 moyenne par feuille, mêmes parcours au décodage.
    topo_bits: list[int] = []
    moyennes: list[float] = []

    def _preordre(depth: int, row: int, col: int) -> None:
        est_interne = (depth, row, col) in internes
        topo_bits.append(1 if est_interne else 0)
        if est_interne:
            for dr, dc in _ORDRE_ENFANTS:
                _preordre(depth + 1, row * 2 + dr, col * 2 + dc)
        else:
            moyennes.append(float(niveaux[depth][row, col]))

    _preordre(0, 0, 0)

    return SummaryQT(topology=np.array(topo_bits, dtype=np.uint8),
                     means=np.array(moyennes, dtype=np.float64),
                     shape=(_TAILLE_DOMAINE, _TAILLE_DOMAINE))


def size_floats_qt(summary: SummaryQT) -> int:
    """Taille du résumé en floats-équivalents (clause 1 gravée) :
    n_feuilles + ceil(n_nœuds / 32) -- 1 bit de topologie par nœud
    (préordre), 32 bits = 1 float-équivalent. n_nœuds = `topology.size`
    (internes + feuilles) ; n_feuilles = `means.size`."""
    n_feuilles = int(summary.means.size)
    n_noeuds = int(summary.topology.size)
    return n_feuilles + -(-n_noeuds // 32)


def _decoder_preordre(topo: np.ndarray, means: np.ndarray, idx_topo: int,
                      idx_mean: int, depth: int, row: int, col: int,
                      fine: np.ndarray) -> tuple[int, int]:
    """Décodage préordre récursif : consomme un bit de `topo` (1 = interne →
    récurse sur les 4 enfants dans `_ORDRE_ENFANTS`, 0 = feuille → consomme
    une moyenne de `means` et la peint sur le bloc dyadique (row, col) de
    profondeur `depth` dans `fine`). Renvoie les index APRÈS consommation --
    `regenerate_qt` vérifie qu'il ne reste ni bit ni moyenne en excès
    (topologie bien formée)."""
    if idx_topo >= topo.size:
        raise ValueError(
            "regenerate_qt : topologie tronquée (bit manquant en préordre).")
    est_interne = bool(topo[idx_topo])
    idx_topo += 1
    if est_interne:
        if depth >= _PROFONDEUR_MAX:
            raise ValueError(
                "regenerate_qt : bit interne à la profondeur maximale "
                "(feuille 1x1 insécable, ne peut pas avoir d'enfants).")
        for dr, dc in _ORDRE_ENFANTS:
            idx_topo, idx_mean = _decoder_preordre(
                topo, means, idx_topo, idx_mean,
                depth + 1, row * 2 + dr, col * 2 + dc, fine)
        return idx_topo, idx_mean

    if idx_mean >= means.size:
        raise ValueError(
            "regenerate_qt : moyennes insuffisantes pour la topologie déclarée.")
    cote = 2 ** (_PROFONDEUR_MAX - depth)
    y, x = row * cote, col * cote
    fine[y:y + cote, x:x + cote] = means[idx_mean]
    return idx_topo, idx_mean + 1


def regenerate_qt(summary: SummaryQT) -> np.ndarray:
    """Régénérateur du claim, UN SEUL paramètre (anti-fuite) : chaque feuille
    peinte à sa moyenne (constant-par-blocs ADAPTATIF, tailles dyadiques
    variables). PAS de clip : les moyennes d'un champ ≥ 0 sont elles-mêmes
    ≥ 0 -- une vérification de non-négativité (AssertionError) tient lieu
    d'assert de cohérence, jamais une correction.

    Vérification de cohérence interne minimale, par `raise` (jamais un
    `assert` nu, qui disparaît sous `python -O`) :
    - shape du résumé == domaine fixe 64x64 (ValueError) ;
    - topologie bien formée : le décodage préordre consomme EXACTEMENT tous
      les bits de `topology` et toutes les moyennes de `means`, ni plus ni
      moins, et aucun bit interne à profondeur 6 (ValueError) ;
    - régénéré non-négatif (AssertionError, cf. ci-dessus).

    Aucun global d'état, aucune lecture fichier, aucune vue du champ vrai ni
    de l'historique -- tout provient de `summary`."""
    H, W = summary.shape
    if (H, W) != (_TAILLE_DOMAINE, _TAILLE_DOMAINE):
        raise ValueError(
            f"regenerate_qt : domaine fixe {_TAILLE_DOMAINE}x{_TAILLE_DOMAINE} "
            f"(le résumé porte shape={(H, W)}).")

    topo = np.asarray(summary.topology)
    means = np.asarray(summary.means)
    fine = np.empty((H, W), dtype=np.float64)
    idx_topo, idx_mean = _decoder_preordre(topo, means, 0, 0, 0, 0, 0, fine)

    if idx_topo != topo.size:
        raise ValueError(
            f"regenerate_qt : topologie mal formée -- {topo.size - idx_topo} "
            "bit(s) en excès non consommé(s) par le préordre.")
    if idx_mean != means.size:
        raise ValueError(
            f"regenerate_qt : topologie mal formée -- {means.size - idx_mean} "
            "moyenne(s) en excès non consommée(s).")
    if np.any(fine < 0.0):
        raise AssertionError(
            "regenerate_qt : régénéré négatif -- incohérent avec l'hypothèse "
            "d'un champ d'origine ≥ 0 (les moyennes de blocs doivent être ≥ 0).")
    return fine
