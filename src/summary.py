"""Arc A / Task 4 (spec GRAVÉE, `docs/superpowers/plans/arc-a-manche1.md`) —
compresseur/régénérateur du champ sédiment `s`.

Une responsabilité : résumer un champ (H, W) en un `Summary` grossier
(moyenne par blocs dyadiques 2^ℓ + invariants de masse) et le régénérer en un
champ (H, W) — sans jamais garder de référence vivante vers le champ d'origine
(anti-fuite, cf. `regenerate` — signature à UN paramètre).

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


def summarize(field: np.ndarray, level: int) -> Summary:
    """Moyenne par blocs dyadiques 2^level (Harten cell-average) + invariants
    scalaires (masse totale + 16 masses de sous-domaines 4x4). Fonction PURE :
    ne mute pas `field`, et ne conserve aucune référence vivante vers lui (tout
    est recopié via `.mean()`/`.sum()` avant le retour)."""
    _valider_niveau(level)
    field = np.asarray(field, dtype=np.float64)
    H, W = field.shape
    bs = 2 ** level
    if H % bs or W % bs:
        raise ValueError(f"shape ({H}, {W}) non divisible par le bloc dyadique {bs} "
                         f"(level={level}).")
    Hc, Wc = H // bs, W // bs
    coarse = field.reshape(Hc, bs, Wc, bs).mean(axis=(1, 3))

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


def _bilinear_upsample(coarse: np.ndarray, shape: tuple[int, int], block: int) -> np.ndarray:
    """Upsampling bilinéaire séparable de `coarse` (moyennes par blocs de côté
    `block`) vers `shape`. Centre de bloc (i, j) placé à l'indice pixel fin
    `i*block + (block-1)/2` (cohérent avec la convention indice=coordonnée du
    reste du dépôt). Interpolation bilinéaire = deux interpolations linéaires
    1D successives (x puis y) sur une grille régulière -- résultat exact,
    équivalent au bilinéaire 2D complet. `np.interp` clampe hors-plage (pas
    d'extrapolation ni de repliement au bord)."""
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


def regenerate(summary: Summary) -> np.ndarray:
    """Upsampling bilinéaire du coarse, clip >= 0, puis rescale multiplicatif
    par sous-domaine 4x4 pour restituer EXACTEMENT les 16 masses stockées
    (division protégée : sous-domaine de masse nulle après upsampling -> reste
    nul, pas de division par zéro) ; vérification (assertion) de cohérence
    interne de la masse totale. UN SEUL paramètre (anti-fuite) : aucun global
    d'état, aucune lecture fichier, aucune vue du champ vrai ni de l'historique
    -- tout provient de `summary`."""
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
            f"regenerate : masse totale régénérée ({total}) incohérente avec la "
            f"masse totale du résumé ({total_attendu}) -- écart "
            f"{abs(total - total_attendu)}.")
    return fine
