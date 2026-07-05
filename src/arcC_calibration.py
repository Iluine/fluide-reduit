"""Arc C / Task 2 — Auto-calibration d'écran portable (§C7 de PREREGISTRATION.md,
pocCascade2phys, commits `c494d50`/`5d13bc4`/`f39b402`, reproduit verbatim dans
`.superpowers/sdd/arcC-task2-abx-brief.md`, section « Auto-calibration »).

Trois fonctions PURES, déterministes, sans RNG :
  - `pixels_par_degre`      : ppd depuis une longueur de référence mesurée à
    l'écran (px vs mm physique) + une distance d'observation mesurée.
  - `taille_domaine_px`     : taille d'affichage (px) du domaine 64x64 pour
    poser la bande porteuse au pic CSF (§C7).
  - `plafond_texture`       : le plafond du régime texture (§C7 pièce 3,
    COLMATAGE MONOTONIE) -- taille angulaire max du domaine avant qu'une
    cellule (1/64ᵉ du domaine) n'atteigne le seuil d'acuité.

Une quatrième fonction, `observation_cellule_pic_csf`, COMPOSE les trois
primitives ci-dessus pour produire le REPORT OBLIGATOIRE (§C7 pièce 3) : la
taille angulaire d'une cellule à la géométrie pic-CSF, comparée au plafond
d'acuité -- PAS une fonction du contrat listée nommément, mais un assemblage
direct des trois pour satisfaire l'exigence de report (aucune décision prise
ici, seulement les chiffres)."""
from __future__ import annotations

import math

# Domaine FIXE 64x64 (cf. config.GridConfig, cascade/summary_quadtree.py) --
# grille de cellules dont on mesure la taille angulaire.
N_CELLULES_DOMAINE: int = 64

# Défauts nommés (§C7, brief) : bande porteuse 4-7 cyc/domaine, milieu = 5.5 ;
# pic de la CSF humaine (contraste-sensibilité) ~3 cycles/degré.
PORTEUSE_CYC_PAR_DOMAINE_DEFAUT: float = 5.5
C_DEG_CIBLE_DEFAUT: float = 3.0
# Plafond d'acuité par défaut (§C7 pièce 3) : ~1 arcmin, choix nommé à exposer.
SEUIL_ACUITE_ARCMIN_DEFAUT: float = 1.0


def pixels_par_degre(long_ref_px: float, long_ref_mm: float, distance_mm: float) -> float:
    """ppd = px_par_mm * mm_par_degre, avec :
      - `px_par_mm` = `long_ref_px / long_ref_mm` (densité de pixels de l'écran,
        mesurée via une longueur de référence affichée) ;
      - `mm_par_degre` = 2*distance*tan(0.5°) (formule standard : un degré
        d'angle visuel, à `distance_mm`, couvre `2*distance*tan(0.5°)` mm --
        c'est la longueur de l'arc sous-tendu par 1° vue de `distance_mm`,
        linéarisée en corde pour un demi-angle de 0.5°).

    Pure, déterministe -- aucune RNG, aucun effet de bord."""
    if long_ref_mm <= 0.0:
        raise ValueError(f"pixels_par_degre : long_ref_mm doit être > 0 (reçu {long_ref_mm!r}).")
    if distance_mm <= 0.0:
        raise ValueError(f"pixels_par_degre : distance_mm doit être > 0 (reçu {distance_mm!r}).")
    px_par_mm = long_ref_px / long_ref_mm
    mm_par_degre = 2.0 * distance_mm * math.tan(math.radians(0.5))
    return px_par_mm * mm_par_degre


def taille_domaine_px(ppd: float, porteuse_cyc_par_domaine: float, c_deg_cible: float) -> int:
    """Taille d'affichage (px) du domaine 64x64 telle que la bande porteuse
    (`porteuse_cyc_par_domaine` cycles/domaine) tombe à `c_deg_cible`
    cycles/degré à l'écran, à la géométrie `ppd` donnée.

    Dérivation : cycles/degré = porteuse / domaine_deg, et domaine_deg =
    domaine_px / ppd => domaine_px = porteuse * ppd / c_deg_cible. Arrondi à
    l'entier le plus proche (taille d'affichage physique, en pixels) --
    l'arrondi introduit un écart résiduel sur `c_deg_cible` RÉALISÉ, mesuré
    par le test d'inverse-cohérence (cf. tests/test_arcC_calibration.py)."""
    if c_deg_cible <= 0.0:
        raise ValueError(f"taille_domaine_px : c_deg_cible doit être > 0 (reçu {c_deg_cible!r}).")
    domaine_px = porteuse_cyc_par_domaine * ppd / c_deg_cible
    return int(round(domaine_px))


def plafond_texture(ppd: float, seuil_acuite_arcmin: float, *,
                    n_cellules: int = N_CELLULES_DOMAINE) -> dict:
    """Plafond du régime texture (§C7 pièce 3, COLMATAGE MONOTONIE) : au-delà
    de ce plafond, une cellule du domaine (1/`n_cellules`ᵉ, ici 1/64ᵉ) devient
    individuellement résolvable par l'œil (angle >= `seuil_acuite_arcmin`) --
    artefact de blocs visible, pas seulement un flou de sur-échantillonnage.

    Une cellule = `1/n_cellules` du domaine ; sa taille angulaire est donc
    exactement `domaine_deg / n_cellules`, où `domaine_deg` est la taille
    angulaire du DOMAINE ENTIER affiché. Poser cellule_deg == seuil (le point
    où le plafond est atteint) donne directement :
      domaine_deg_max = seuil_acuite_arcmin/60 * n_cellules
      domaine_px_max  = domaine_deg_max * ppd   (conversion en pixels d'écran,
                                                  POUR la géométrie `ppd` donnée)

    Retourne un dict {ppd, seuil_acuite_arcmin, n_cellules,
    cellule_arcmin_au_plafond (== seuil_acuite_arcmin, par construction),
    domaine_deg_max, domaine_px_max}. Aucune décision ici -- seulement les
    chiffres (cf. `observation_cellule_pic_csf` pour la comparaison)."""
    if ppd <= 0.0:
        raise ValueError(f"plafond_texture : ppd doit être > 0 (reçu {ppd!r}).")
    if seuil_acuite_arcmin <= 0.0:
        raise ValueError(
            f"plafond_texture : seuil_acuite_arcmin doit être > 0 (reçu {seuil_acuite_arcmin!r}).")
    domaine_deg_max = seuil_acuite_arcmin / 60.0 * n_cellules
    domaine_px_max = domaine_deg_max * ppd
    return dict(
        ppd=ppd, seuil_acuite_arcmin=seuil_acuite_arcmin, n_cellules=n_cellules,
        cellule_arcmin_au_plafond=seuil_acuite_arcmin,
        domaine_deg_max=domaine_deg_max, domaine_px_max=domaine_px_max)


def observation_cellule_pic_csf(
        ppd: float, *,
        porteuse_cyc_par_domaine: float = PORTEUSE_CYC_PAR_DOMAINE_DEFAUT,
        c_deg_cible: float = C_DEG_CIBLE_DEFAUT,
        n_cellules: int = N_CELLULES_DOMAINE,
        seuil_acuite_arcmin: float = SEUIL_ACUITE_ARCMIN_DEFAUT) -> dict:
    """REPORT OBLIGATOIRE (§C7 pièce 3, ne pas masquer) : à la géométrie
    pic-CSF choisie (`taille_domaine_px` avec les défauts porteuse=5.5,
    c_deg_cible=3), calcule la taille angulaire RÉALISÉE d'une cellule
    (arcmin, après arrondi de `taille_domaine_px` à l'entier) et la compare
    au plafond d'acuité `seuil_acuite_arcmin` (`plafond_texture`).

    Retourne les chiffres bruts (`domaine_px_pic_csf`, `domaine_deg_pic_csf`,
    `cellule_arcmin_pic_csf`, `seuil_acuite_arcmin`,
    `ratio_cellule_sur_seuil`) -- AUCUNE décision n'est prise ici (pas de
    booléen "dépasse"/"ok") : §C7 pièce 3 veut le chiffre surfacé, pas résolu
    en douce. `ratio_cellule_sur_seuil >= 1` signifie que la cellule au
    pic-CSF est DÉJÀ au-dessus (ou au) plafond d'acuité -- tension attendue
    sur un contenu grossier 64x64 avec porteuse 4-7 (cf. docstring module)."""
    domaine_px_pic_csf = taille_domaine_px(ppd, porteuse_cyc_par_domaine, c_deg_cible)
    domaine_deg_pic_csf = domaine_px_pic_csf / ppd
    cellule_arcmin_pic_csf = domaine_deg_pic_csf * 60.0 / n_cellules
    plafond = plafond_texture(ppd, seuil_acuite_arcmin, n_cellules=n_cellules)
    return dict(
        ppd=ppd, porteuse_cyc_par_domaine=porteuse_cyc_par_domaine, c_deg_cible=c_deg_cible,
        n_cellules=n_cellules, domaine_px_pic_csf=domaine_px_pic_csf,
        domaine_deg_pic_csf=domaine_deg_pic_csf, cellule_arcmin_pic_csf=cellule_arcmin_pic_csf,
        seuil_acuite_arcmin=seuil_acuite_arcmin,
        domaine_deg_max_plafond=plafond["domaine_deg_max"],
        ratio_cellule_sur_seuil=cellule_arcmin_pic_csf / seuil_acuite_arcmin)
