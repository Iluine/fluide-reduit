"""Arc C / P1 — R1, le NOYAU DE RENDU-INSTRUMENT : albédo -> luminance affichée.

Gravé §A37 de `pocCascade2phys/PREREGISTRATION.md` (spec ENDOSSÉE
`claude/spec-p1-rendu-instrument-2026-07-25.md`, bloc REVUE inclus) :

  - **D-P1-1, OPTION B** : pleine résolution d'abord, NOYAU SEUL. L'interface
    est PÉRIMÈTRE-NEUTRE -- « un champ albédo pleine résolution -> une
    luminance affichée calibrée ». Le compositeur (fovéa + grossier + couture)
    est un incrément POST-P3 : le jour venu il sera un PRODUCTEUR DE CHAMP
    ALBÉDO de plus, zéro re-travail ici.
  - **D-P1-3 (a)** : chemin lumineux IDENTITÉ, `Y = A`. Une nouveauté à la
    fois -- P3 mesure le transport du pin sur la SEULE variable changée
    (pseudo-couleur -> luminance). Le lambertien (`src.albedo.relief_shaded`)
    est l'incrément R2, post-P3, avec sa PROPRE mesure de transport.
  - **D-P1-4 (a)** : inverse-EOTF sRGB, gravé EN FORMULE ici même. Portée sans
    surclame : linéarité SUPPOSÉE sRGB, JAMAIS mesurée au photomètre -- ce que
    l'ABX exige n'est pas la linéarité absolue mais le DÉTERMINISME et la
    STABILITÉ (les deux stimuli d'un essai traversent le MÊME encodage : biais
    commun).
  - **D-P1-2 (a)** : rééchantillonnage BILINÉAIRE, étage de R1, testé -- plus
    jamais une option de bibliothèque d'affichage. C'est le fait de code que P1
    remplace : le pin gravé jnd_sev 7.33 % a été mesuré à travers
    `imshow(cmap="viridis", vmin=0, vmax=1)` et son interpolation PAR DÉFAUT,
    non gravée, sans aucune gestion d'EOTF.

ORDRE DES ÉTAGES — AMENDÉ puis TRANCHÉ (Romain, 2026-07-25 ; correction
explicite au bloc REVUE de la spec, dont le tableau §2.3 ordonnait
encodage -> quantification -> rééchantillonnage) :

    A (64x64, [0,1])  ->  rééchantillonnage bilinéaire  (espace LINÉAIRE = A,
                                                         puisque Y = A)
                      ->  encodage sRGB                 (float)
                      ->  quantification uint8          (UNE SEULE FOIS, en bout)

Motif : ne jamais interpoler des valeurs QUANTIFIÉES, ni interpoler en espace
GAMMA ; une moyenne bilinéaire de luminances est une luminance (c'est
l'intention de D-P1-4). Épistémique consignée telle quelle : décision SANS a
priori, re-jugeable AU RÉSULTAT -- l'ordre est LOGGÉ dans la provenance de
chaque session (`scripts/run_arcC_session.py`), donc ATTRIBUABLE si P3 rend un
transport ≠ 1.

CE QUE CE MODULE N'EST PAS. Pas un rendu de production : pas de solve
d'équilibre, pas d'ombres portées, pas de GI (spec §2.2.3 -- dégradation
DÉLIBÉRÉE et nommée, R1 est un mapping LOCAL par pixel). Conséquence de portée
gravée d'avance : **P3 date son verdict sur R1** ; le pin transporté
s'étiquettera « à travers R1 », jamais « à travers le rendu ». C'est pour cela
que `tests/test_arcC_rendu.py` VERROUILLE l'empreinte sha256 de ce fichier
(tradition des kernels F1) : un verdict daté exige un objet daté.

PURETÉ (invariant §A20-1, ici PLUS DUR que §A20-3 : l'état éphémère de readout
y est *toléré*, dans l'instrument il est INTERDIT tout court). Aucune variable
de module mutable, aucun cache, aucune RNG, aucune horloge : R1 est fonction de
`a` et de `taille_px`, de rien d'autre. « `z` stable => image stable » n'est
pas argumenté ici, il est TESTÉ (déterminisme bit à bit). Un instrument avec
mémoire n'est pas re-jouable.

FAIL-LOUD, jamais un clip silencieux : l'albédo vit dans [0,1) par construction
(`albedo(s, s_half) = 1 - exp(-s/s_half)`, `src/albedo.py`) ; une valeur hors
bande, un NaN ou une mauvaise forme sont un BUG AMONT, pas une donnée à
rattraper en douce.

CE QUE R1 NE TOUCHE PAS : l'observable Δχ et le pin gravé restent dans l'espace
ALBÉDO, inchangés (`src/albedo.py`). R1 est un chemin d'AFFICHAGE ; aucun
seuil, aucune bande, aucun combinateur n'est modifié ici."""
from __future__ import annotations

import numpy as np

from src.arcC_calibration import N_CELLULES_DOMAINE

# --- Constantes gravées ------------------------------------------------------

# Inverse-EOTF sRGB standard (IEC 61966-2-1), écrite en toutes lettres plutôt
# qu'importée d'une bibliothèque de couleur : D-P1-4 exige que la formule soit
# LISIBLE dans le code, parce que c'est elle que la lecture de P3 interroge si
# le transport du pin s'écarte de 1.
SEUIL_LINEAIRE_SRGB: float = 0.0031308   # bascule branche linéaire / branche puissance
PENTE_LINEAIRE_SRGB: float = 12.92       # pente de la branche linéaire (près du noir)
GAIN_SRGB: float = 1.055                 # gain de la branche puissance
DECALAGE_SRGB: float = 0.055             # décalage de la branche puissance
GAMMA_SRGB: float = 2.4                  # exposant (l'encodage applique 1/GAMMA)

# Quantification : 256 niveaux, valeur maximale 255 (uint8 plein).
NIVEAU_MAX_UINT8: int = 255

# Description LITTÉRALE de la chaîne, destinée à la PROVENANCE de session
# (§A37 : l'ordre des étages est une pièce attribuable si P3 rend ≠ 1). Elle
# vit ici, à côté du code qu'elle décrit -- pas dans le script d'affichage,
# pour qu'un « mensonge prose/code » ne puisse pas s'installer entre les deux
# (leçon §A33-CORRECTION : c'est un tel écart qui avait caché le défaut Δx).
ORDRE_ETAGES: str = (
    "A[0,1] float64 -> reechantillonnage bilineaire (espace LINEAIRE, Y=A, "
    "centres de pixels) -> encodage sRGB (float) -> quantification uint8 (unique, np.rint)")

# Les DEUX chemins d'affichage que le sélecteur obligatoire `--rendu` propose
# (§A37). `"r1"` est celui que ce module implémente ; `"viridis"` est le chemin
# HISTORIQUE -- celui du pin gravé -- qui vit dans la coquille de session
# (`scripts/run_arcC_session.py`) et n'a rien à faire ici. La liste vit tout de
# même dans ce module PUR parce que la coquille et l'orchestrateur de campagne
# en ont tous deux besoin et s'importent circulairement : une source unique
# ailleurs qu'entre eux deux est la seule qui ne dérive pas.
CHEMINS_RENDU: tuple[str, ...] = ("viridis", "r1")


# --- Gardes FAIL-LOUD --------------------------------------------------------


def _valide_bande_unite(v: np.ndarray, nom: str) -> np.ndarray:
    """Garde commune : `v` est un tableau float64, FINI, entièrement dans
    [0, 1]. Lève `ValueError` sinon -- JAMAIS un clip silencieux.

    Le float64 est EXIGÉ (pas seulement accepté) : la re-jouabilité bit à bit
    des sessions P3 est un invariant testé, et un champ arrivé en float32
    rendrait une image différente sans que rien ne le signale."""
    arr = np.asarray(v)
    if arr.dtype != np.float64:
        raise ValueError(
            f"arcC_rendu : {nom} doit être float64 (reçu dtype={arr.dtype!r}) -- la "
            "re-jouabilité bit à bit de R1 ne tolère pas de conversion implicite.")
    if arr.size == 0:
        raise ValueError(f"arcC_rendu : {nom} est vide (aucun pixel à rendre).")
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            f"arcC_rendu : {nom} contient des valeurs non finies (NaN/inf) -- bug AMONT, "
            "jamais une donnée à rattraper.")
    minimum = float(arr.min())
    maximum = float(arr.max())
    if minimum < 0.0 or maximum > 1.0:
        raise ValueError(
            f"arcC_rendu : {nom} doit être dans [0, 1] (reçu min={minimum!r}, "
            f"max={maximum!r}) -- l'albédo est dans [0,1) par construction ; une valeur "
            "hors bande est un BUG AMONT, pas un dépassement à clipper.")
    return arr


def _valide_taille_px(taille_px: int) -> int:
    """`taille_px` : côté de l'image AFFICHÉE, en pixels d'écran -- entier
    strictement positif (il sort de `arcC_calibration.taille_domaine_px`, qui
    arrondit déjà à l'entier)."""
    if isinstance(taille_px, bool) or not isinstance(taille_px, (int, np.integer)):
        raise ValueError(
            f"arcC_rendu : taille_px doit être un entier (reçu {taille_px!r}).")
    if taille_px <= 0:
        raise ValueError(
            f"arcC_rendu : taille_px doit être > 0 (reçu {taille_px!r}).")
    return int(taille_px)


# --- Étage 5 : rééchantillonnage bilinéaire (D-P1-2 (a)) ---------------------


def _poids_bilineaires(n_entree: int, n_sortie: int) -> tuple[np.ndarray, np.ndarray,
                                                              np.ndarray]:
    """Indices bas/haut et poids d'UN axe, convention **CENTRES DE PIXELS**
    (demi-pixel), écrite en toutes lettres :

      le pixel de sortie `j` a son CENTRE à `(j + 0.5) / n_sortie` du domaine ;
      ramené en coordonnée d'ENTRÉE (dont le pixel `i` a son centre à
      `(i + 0.5) / n_entree`), cela donne
          `x(j) = (j + 0.5) * n_entree / n_sortie - 0.5`.

    Le `-0.5` EST la convention : sans lui on alignerait les BORDS de pixels,
    ce qui décale l'image d'un demi-pixel d'entrée et biaiserait la géométrie
    de la porteuse (§C7). `x` est borné à [0, n_entree - 1] : aux deux bords,
    la sortie sur-échantillonnée déborde du dernier centre d'entrée, et le
    bornage revient à PROLONGER le pixel de bord (extension par réplication) --
    choix nommé, pas un artefact.

    Retourne `(i0, i1, w)` avec `x = i0 + w`, `i1 = min(i0 + 1, n_entree - 1)`,
    `w` dans [0, 1[."""
    j = np.arange(n_sortie, dtype=np.float64)
    x = (j + 0.5) * (n_entree / n_sortie) - 0.5
    x = np.clip(x, 0.0, float(n_entree - 1))
    i0 = np.floor(x).astype(np.intp)
    i1 = np.minimum(i0 + 1, n_entree - 1)
    return i0, i1, x - i0


def _reechantillonne_general(a: np.ndarray, taille_px: int) -> np.ndarray:
    """Le calcul bilinéaire SÉPARABLE, sans court-circuit : axe 0 (lignes)
    PUIS axe 1 (colonnes) -- l'ordre est gravé (il n'a aucune conséquence
    mathématique, il en a une sur les derniers bits, et R1 doit être
    bit-reproductible).

    Séparé de `reechantillonne_bilineaire` pour que le court-circuit d'identité
    de celle-ci soit PROUVABLE plutôt qu'affirmé : un test compare les deux
    chemins à `taille_px == 64` (cf. `tests/test_arcC_rendu.py`)."""
    i0, i1, w = _poids_bilineaires(a.shape[0], taille_px)
    inter = a[i0, :] * (1.0 - w)[:, None] + a[i1, :] * w[:, None]
    return inter[:, i0] * (1.0 - w)[None, :] + inter[:, i1] * w[None, :]


def reechantillonne_bilineaire(a: np.ndarray, taille_px: int) -> np.ndarray:
    """Rééchantillonne le champ carré `a` (H x H) vers `taille_px x taille_px`,
    bilinéairement, en convention CENTRES DE PIXELS (cf. `_poids_bilineaires`).

    ÉTAGE DE R1, en espace LINÉAIRE (= albédo, puisque `Y = A`) : une moyenne
    de luminances est une luminance. Il s'exécute AVANT l'encodage sRGB et
    AVANT toute quantification (ordre tranché, cf. docstring module).

    COURT-CIRCUIT EXPLICITE : `taille_px == H` retourne une COPIE bit-exacte de
    `a`. Le chemin général la retournerait déjà (`x(j) = j` exactement, donc
    `w = 0`), mais l'identité de calibration est un falsifieur d'acceptation --
    elle est court-circuitée pour être exacte PAR CONSTRUCTION, et l'accord des
    deux chemins est testé plutôt que supposé."""
    arr = _valide_bande_unite(a, "a")
    taille = _valide_taille_px(taille_px)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(
            f"arcC_rendu : a doit être un champ CARRÉ 2D (reçu shape={arr.shape!r}).")
    if taille == arr.shape[0]:
        return arr.copy()
    return _reechantillonne_general(arr, taille)


# --- Étage 3 : encodage écran, inverse-EOTF sRGB (D-P1-4 (a)) ----------------


def srgb_encode(y: np.ndarray) -> np.ndarray:
    """Inverse-EOTF sRGB standard, formule GRAVÉE (D-P1-4 (a)) :

        v = 12.92 * y                          si y <= 0.0031308
        v = 1.055 * y**(1/2.4) - 0.055         sinon

    `y` est la luminance relative dans [0, 1] (ici `Y = A`, D-P1-3) ; `v` est
    la valeur d'écran non quantifiée, dans [0, 1]. Monotone croissante : c'est
    ce qui permet à la monotonie de la chaîne d'être un invariant testable.

    Portée, redite sans lissage : cette formule SUPPOSE un écran sRGB, elle ne
    le MESURE pas. Aucun photomètre n'a été passé sur l'écran de session. Ce
    que l'ABX exige est que les deux stimuli d'un essai traversent le MÊME
    encodage (biais commun), pas que l'écran soit fidèle à la norme."""
    arr = _valide_bande_unite(y, "y")
    # `np.where` évalue les DEUX branches : la branche puissance est donc
    # calculée aussi en y = 0. `y**(1/2.4)` y vaut 0.0 (pas de NaN), le
    # `np.maximum` n'est là que pour interdire toute puissance d'un négatif si
    # la garde amont venait à être contournée -- ceinture, pas remède.
    branche_puissance = GAIN_SRGB * np.power(np.maximum(arr, 0.0), 1.0 / GAMMA_SRGB) \
        - DECALAGE_SRGB
    return np.where(arr <= SEUIL_LINEAIRE_SRGB, PENTE_LINEAIRE_SRGB * arr, branche_puissance)


# --- Étage 4 : quantification uint8, UNE SEULE FOIS --------------------------


def quantifie_uint8(v: np.ndarray) -> np.ndarray:
    """Quantifie `v` (valeurs d'écran dans [0, 1]) en uint8 sur 256 niveaux,
    `np.rint` -- arrondi au plus proche, égalités arrondies au PAIR (règle de
    `np.rint`, nommée ici pour qu'elle ne surprenne personne).

    UNE SEULE quantification, en BOUT de chaîne (ordre tranché) : c'est
    exactement ce que l'ordre initial de la spec §2.3 ne garantissait pas, en
    plaçant le rééchantillonnage APRÈS, ce qui aurait interpolé des valeurs
    déjà quantifiées.

    Garde fail-loud sur l'entrée, comme partout ici : aucun clip. La chaîne de
    R1 ne peut pas sortir de [0, 1] (le bilinéaire est une combinaison convexe,
    `srgb_encode` est croissante de [0,1] sur [0,1]) -- si une valeur hors
    bande arrive, c'est un bug, et il doit parler."""
    arr = _valide_bande_unite(v, "v")
    return np.rint(arr * float(NIVEAU_MAX_UINT8)).astype(np.uint8)


# --- R1 : la composition des étages ------------------------------------------


def rendu_r1(a: np.ndarray, taille_px: int) -> np.ndarray:
    """**R1** : champ albédo 64x64 float64 dans [0,1] -> image uint8
    `taille_px x taille_px`, prête à être affichée en NIVEAUX DE GRIS.

    Fonction PURE, sans RNG, sans état : deux appels sur le même `a` rendent
    les mêmes octets (testé, pas supposé -- invariant anti-PERSIST §A20-1).

    La forme 64x64 est EXIGÉE : le domaine du projet est fixe
    (`arcC_calibration.N_CELLULES_DOMAINE`), et un champ d'une autre taille
    signifierait qu'un producteur non prévu (compositeur ? niveau grossier ?)
    est arrivé jusqu'ici sans décision. Ce n'est pas une limite technique du
    rééchantillonneur -- c'est un garde de PÉRIMÈTRE."""
    arr = _valide_bande_unite(a, "a")
    taille = _valide_taille_px(taille_px)
    attendu = (N_CELLULES_DOMAINE, N_CELLULES_DOMAINE)
    if arr.shape != attendu:
        raise ValueError(
            f"arcC_rendu : a doit être de forme {attendu!r} (reçu {arr.shape!r}) -- "
            "R1 rend le DOMAINE du projet, pas un champ quelconque.")

    # Étage 2 -- CHEMIN LUMINEUX : Y = A, IDENTITÉ (D-P1-3 (a)).
    # C'est CET étage, et lui seul, que l'incrément R2 post-P3 remplacerait par
    # `Y = A * irradiance_lambertienne(b0 + s)` -- avec sa PROPRE mesure de
    # transport. Il est écrit comme une ligne à part, jamais fondu dans les
    # étages voisins, pour que le point de substitution soit visible au grep.
    y = arr

    # Étage 5 -- rééchantillonnage, en espace LINÉAIRE (avant l'encodage).
    y_affiche = reechantillonne_bilineaire(y, taille)
    # Étage 3 -- encodage écran (sRGB), puis étage 4 -- quantification unique.
    return quantifie_uint8(srgb_encode(y_affiche))
