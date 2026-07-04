"""Arc A / Task 4 (spec GRAVÉE, `docs/superpowers/plans/arc-a-manche1.md`,
AMENDÉE par l'arbitrage instrument `9bcb09a` du 2026-07-04) — compresseur/
régénérateur du champ sédiment `s`.

Une responsabilité : résumer un champ (H, W) en un `Summary` grossier
(moyenne par blocs dyadiques 2^ℓ + invariants de masse) et le régénérer en un
champ (H, W) — sans jamais garder de référence vivante vers le champ d'origine
(anti-fuite, cf. `regenerate` — signature à UN paramètre).

`regenerate` = upsampling CONSTANT-PAR-BLOCS (Harten ordre 0) : seule
projection EXACTE de la famille sur un champ positif sparse avec clip (motif
gravé, arbitrage instrument). L'ancien régénérateur bilinéaire est conservé
sous `regenerer_bilineaire` (AUDIT seulement — reproductibilité de M-0bis et
du run v1 — n'est plus le régénérateur du claim, plus jamais utilisé par les
mesures).

Indexation array[y, x] comme le reste du dépôt.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Sous-domaines de restitution des invariants de masse : grille FIXE 4x4 (16
# blocs), indépendante du niveau de résumé ℓ (cf. spec gravée : « masses par
# sous-domaine 4×4 (16 blocs de 16×16) » — 16x16 est la valeur concrète pour la
# grille 64x64 de ce projet ; le code reste générique en H//4, W//4).
_N_SOUS_DOMAINES: int = 4

_NIVEAUX_VALIDES: tuple[int, ...] = (1, 2, 3)


@dataclass(frozen=True)
class Summary:
    """Résumé gelé d'un champ (H, W) au niveau `level` :
    - `coarse` : moyennes par blocs dyadiques de côté 2^level, shape
      (H // 2**level, W // 2**level).
    - `invariants` : (1 + N_SOUS_DOMAINES²,) floats — [masse_totale,
      masse_sous_domaine_00, masse_sous_domaine_01, ..., masse_sous_domaine_33]
      (ordre ligne-major sur la grille 4x4 de sous-domaines, indices (i, j) avec
      i = bloc en y, j = bloc en x).
    - `level` : ℓ ∈ {1, 2, 3}.
    - `shape` : (H, W) du champ fin d'origine."""
    coarse: np.ndarray
    invariants: np.ndarray
    level: int
    shape: tuple[int, int]


def _valider_niveau(level: int) -> None:
    if level not in _NIVEAUX_VALIDES:
        raise ValueError(f"level doit être dans {_NIVEAUX_VALIDES} (reçu {level!r}).")


def _sous_domaines_bounds(H: int, W: int) -> tuple[int, int]:
    """Dimensions (bh, bw) d'un sous-domaine de la grille 4x4 -- lève ValueError
    si (H, W) n'est pas divisible par `_N_SOUS_DOMAINES` (garde-fou, comme
    `src.multiresolution.downsample`)."""
    if H % _N_SOUS_DOMAINES or W % _N_SOUS_DOMAINES:
        raise ValueError(
            f"shape ({H}, {W}) non divisible par {_N_SOUS_DOMAINES} sous-domaines.")
    return H // _N_SOUS_DOMAINES, W // _N_SOUS_DOMAINES


def _moyenne_blocs_2x2(field: np.ndarray) -> np.ndarray:
    """Une étape de la cascade dyadique : moyenne par blocs 2x2 -- somme des 4
    valeurs de chaque bloc ((a+b)+(c+d)) puis division par 4 (exacte en
    IEEE-754 : division par une puissance de 2)."""
    H, W = field.shape
    return field.reshape(H // 2, 2, W // 2, 2).sum(axis=(1, 3)) / 4.0


def summarize(field: np.ndarray, level: int) -> Summary:
    """Moyenne par blocs dyadiques 2^level (Harten cell-average, définition
    RÉCURSIVE : cascade 2x2 itérée `level` fois -- cf. `_moyenne_blocs_2x2`) +
    invariants scalaires (masse totale + 16 masses de sous-domaines 4x4,
    calculés sur le CHAMP FIN, inchangé). Fonction PURE : ne mute pas `field`,
    et ne conserve aucune référence vivante vers lui (tout est recopié via la
    cascade/`.sum()` avant le retour).

    Note numérique (décision contrôleur, itération instrument 2026-07-04) :
    cascade 2x2 itérée = définition récursive de Harten, sémantiquement
    identique à « moyenne par blocs dyadiques » ; sur champs généraux elle
    peut différer de l'ancienne réduction plate `.mean(axis=(1, 3))` d'ulps
    (ordre de sommation -- aucune valeur gravée au journal n'est citée
    au-delà de ~5 décimales). Choisie pour rendre S∘R = id EXACT en flottant
    à tous les niveaux sur un champ constant-par-blocs : chaque étape calcule
    (v+v)+(v+v) = 4v exact puis /4 exact -- alors que la réduction plate
    numpy sur 64 éléments (ℓ=3) emprunte un chemin non-binaire (sommes
    intermédiaires 3v, 5v, 7v... arrondies, ~1 ulp sur ~30-50 % des blocs ;
    cause isolée et documentée au rapport d'itération)."""
    _valider_niveau(level)
    field = np.asarray(field, dtype=np.float64)
    H, W = field.shape
    bs = 2 ** level
    if H % bs or W % bs:
        raise ValueError(f"shape ({H}, {W}) non divisible par le bloc dyadique {bs} "
                         f"(level={level}).")
    coarse = field
    for _ in range(level):
        coarse = _moyenne_blocs_2x2(coarse)

    bh, bw = _sous_domaines_bounds(H, W)
    n = _N_SOUS_DOMAINES
    invariants = np.empty(1 + n * n, dtype=np.float64)
    invariants[0] = float(field.sum())
    idx = 1
    for i in range(n):
        for j in range(n):
            bloc = field[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw]
            invariants[idx] = float(bloc.sum())
            idx += 1

    return Summary(coarse=coarse, invariants=invariants, level=level, shape=(H, W))


def size_floats(summary: Summary) -> int:
    """Taille du résumé en nombre de floats (coarse + invariants) : ℓ=1 ->
    1024+17=1041 ; ℓ=2 -> 256+17=273 ; ℓ=3 -> 64+17=81 (grille 64x64)."""
    return int(summary.coarse.size) + int(summary.invariants.size)


def _verifier_masse_restituee(valeur: float, attendu: float, label: str) -> None:
    """Vérifie qu'une masse régénérée (`valeur`) restitue la masse stockée
    dans le résumé (`attendu`) à 1e-12 RELATIF -- SANS rescale multiplicatif
    (motif gravé, arbitrage instrument 2026-07-04 : les moyennes de blocs sont
    exactes par construction dans le constant-par-blocs ; multiplier par un
    facteur 1±ulp -- bruit d'ordre de sommation -- détruirait l'identité
    bit-à-bit sans gagner d'exactitude). Cas `attendu == 0.0` (sous-domaine
    entièrement nul, champ sparse) : la tolérance relative est indéfinie (0/0)
    -- on retombe alors sur une tolérance ABSOLUE équivalente (1e-12).
    Lève `AssertionError` (jamais un `assert` nu) si la tolérance est violée."""
    if attendu == 0.0:
        if abs(valeur) > 1e-12:
            raise AssertionError(
                f"regenerate : {label} attendue nulle, obtenue {valeur} "
                "(tolérance absolue 1e-12 dépassée).")
        return
    ecart_relatif = abs(valeur - attendu) / abs(attendu)
    if ecart_relatif > 1e-12:
        raise AssertionError(
            f"regenerate : {label} régénérée ({valeur}) incohérente avec "
            f"l'invariant stocké ({attendu}) -- écart relatif {ecart_relatif} "
            "> 1e-12.")


def regenerate(summary: Summary) -> np.ndarray:
    """Régénérateur du claim (spec AMENDÉE, arbitrage instrument constant-par-
    blocs, gravé 2026-07-04 -- remplace l'ancien bilinéaire, cf.
    `regenerer_bilineaire` ci-dessous) : upsampling CONSTANT-PAR-BLOCS -- chaque
    bloc 2^level x 2^level est rempli de sa valeur `coarse` (Harten ordre 0,
    `np.repeat` sur les deux axes) --, clip >= 0 conservé (no-op sur un champ
    déjà positif : la moyenne d'un bloc >= 0 est elle-même >= 0), puis
    VÉRIFICATION (SANS rescale multiplicatif, cf. `_verifier_masse_restituee`)
    des 16 masses de sous-domaines 4x4 ET de la masse totale, tolérance 1e-12
    relatif -- `raise AssertionError` si violée.

    Sur un champ positif et sparse avec clip, le constant-par-blocs est la
    SEULE projection exacte de la famille (argument gravé, arbitrage
    9bcb09a) : les moyennes de blocs sont exactes par construction, ce qui
    entraîne en cascade des masses de sous-domaines exactes puis un rescale à
    1.0 -- d'où une simple vérification, jamais une correction (cf. tests
    `test_summarize_regenerate_summarize_identite_bit_exacte` : S∘R = id
    bit-à-bit sur le coarse).

    UN SEUL paramètre (anti-fuite) : aucun global d'état, aucune lecture
    fichier, aucune vue du champ vrai ni de l'historique -- tout provient de
    `summary`."""
    H, W = summary.shape
    bs = 2 ** summary.level
    fine = np.repeat(np.repeat(summary.coarse, bs, axis=0), bs, axis=1)
    fine = np.maximum(fine, 0.0)

    bh, bw = _sous_domaines_bounds(H, W)
    n = _N_SOUS_DOMAINES
    idx = 1
    for i in range(n):
        for j in range(n):
            sl = (slice(i * bh, (i + 1) * bh), slice(j * bw, (j + 1) * bw))
            masse = float(fine[sl].sum())
            _verifier_masse_restituee(masse, float(summary.invariants[idx]),
                                      f"masse du sous-domaine ({i},{j})")
            idx += 1

    total = float(fine.sum())
    _verifier_masse_restituee(total, float(summary.invariants[0]), "masse totale")
    return fine


def _bilinear_upsample(coarse: np.ndarray, shape: tuple[int, int], block: int) -> np.ndarray:
    """Upsampling bilinéaire séparable de `coarse` (moyennes par blocs de côté
    `block`) vers `shape`. Centre de bloc (i, j) placé à l'indice pixel fin
    `i*block + (block-1)/2` (cohérent avec la convention indice=coordonnée du
    reste du dépôt). Interpolation bilinéaire = deux interpolations linéaires
    1D successives (x puis y) sur une grille régulière -- résultat exact,
    équivalent au bilinéaire 2D complet. `np.interp` clampe hors-plage (pas
    d'extrapolation ni de repliement au bord). AUDIT SEULEMENT -- utilisé
    uniquement par `regenerer_bilineaire` ci-dessous."""
    Hc, Wc = coarse.shape
    H, W = shape
    cy = np.arange(Hc, dtype=np.float64) * block + (block - 1) / 2.0
    cx = np.arange(Wc, dtype=np.float64) * block + (block - 1) / 2.0
    yy = np.arange(H, dtype=np.float64)
    xx = np.arange(W, dtype=np.float64)

    inter_x = np.empty((Hc, W), dtype=np.float64)
    for i in range(Hc):
        inter_x[i, :] = np.interp(xx, cx, coarse[i, :])
    fine = np.empty((H, W), dtype=np.float64)
    for j in range(W):
        fine[:, j] = np.interp(yy, cy, inter_x[:, j])
    return fine


def regenerer_bilineaire(summary: Summary) -> np.ndarray:
    """AUDIT SEULEMENT -- reproductibilité de M-0bis et du run v1 ; N'EST PLUS
    le régénérateur du claim (remplacé par `regenerate`, constant-par-blocs,
    arbitrage instrument gravé 2026-07-04). Corps INCHANGÉ depuis
    l'implémentation d'origine : upsampling bilinéaire du coarse, clip >= 0,
    puis rescale multiplicatif par sous-domaine 4x4 pour restituer EXACTEMENT
    les 16 masses stockées (division protégée : sous-domaine de masse nulle
    après upsampling -> reste nul, pas de division par zéro) ; vérification
    (assertion) de cohérence interne de la masse totale. UN SEUL paramètre
    (même contrat d'anti-fuite que `regenerate`) : aucun global d'état,
    aucune lecture fichier, aucune vue du champ vrai ni de l'historique --
    tout provient de `summary`."""
    H, W = summary.shape
    bs = 2 ** summary.level
    fine = _bilinear_upsample(summary.coarse, (H, W), bs)
    fine = np.maximum(fine, 0.0)

    bh, bw = _sous_domaines_bounds(H, W)
    n = _N_SOUS_DOMAINES
    idx = 1
    for i in range(n):
        for j in range(n):
            sl = (slice(i * bh, (i + 1) * bh), slice(j * bw, (j + 1) * bw))
            bloc = fine[sl]
            bloc_sum = float(bloc.sum())
            cible = float(summary.invariants[idx])
            if bloc_sum == 0.0:
                fine[sl] = 0.0
            else:
                fine[sl] = bloc * (cible / bloc_sum)
            idx += 1

    total = float(fine.sum())
    total_attendu = float(summary.invariants[0])
    if not np.isclose(total, total_attendu, rtol=1e-9, atol=1e-9):
        raise AssertionError(
            f"regenerer_bilineaire : masse totale régénérée ({total}) incohérente "
            f"avec la masse totale du résumé ({total_attendu}) -- écart "
            f"{abs(total - total_attendu)}.")
    return fine
