"""Arc C / Task 2 — tests de l'auto-calibration d'écran (`src/arcC_calibration.py`,
§C7 de PREREGISTRATION.md, reproduit verbatim dans
`.superpowers/sdd/arcC-task2-abx-brief.md`).

Trois familles :
  1. `pixels_par_degre` sur un cas connu, formule calculée indépendamment (à
     la main, via `math.tan`) à 1e-9.
  2. `taille_domaine_px` : inverse-cohérence -- la géométrie RENDUE (après
     arrondi entier) redonne bien la porteuse à `c_deg_cible`, à une
     tolérance dictée par l'arrondi (pas une tautologie : le test recalcule
     `c_deg_realise` depuis `domaine_px` arrondi, indépendamment de la
     formule interne).
  3. `plafond_texture` + `observation_cellule_pic_csf` (REPORT OBLIGATOIRE
     §C7 pièce 3) : cellule-arcmin et plafond sur cas connus, PUIS le report
     lui-même (aucun assert de décision -- seulement les chiffres, imprimés
     comme observation)."""
from __future__ import annotations

import math

import pytest

from src.arcC_calibration import (N_CELLULES_DOMAINE, observation_cellule_pic_csf,
                                  pixels_par_degre, plafond_texture, taille_domaine_px)

# --- Famille 1 : pixels_par_degre, cas connu ---------------------------------


def test_pixels_par_degre_cas_connu():
    """distance=600mm, long_ref_px=37.8, long_ref_mm=10.0 (=> px_par_mm=3.78,
    cf. brief). ppd attendu calculé INDÉPENDAMMENT ici via la formule
    standard (math.tan), pas par un second appel à l'implémentation."""
    distance_mm = 600.0
    long_ref_px, long_ref_mm = 37.8, 10.0
    px_par_mm = long_ref_px / long_ref_mm
    assert px_par_mm == pytest.approx(3.78, abs=1e-12)
    mm_par_degre_attendu = 2.0 * distance_mm * math.tan(math.radians(0.5))
    ppd_attendu = px_par_mm * mm_par_degre_attendu

    ppd = pixels_par_degre(long_ref_px, long_ref_mm, distance_mm)
    assert ppd == pytest.approx(ppd_attendu, abs=1e-9)
    # Valeur numérique gravée (calculée hors-ligne) -- sanity anti-régression.
    assert ppd == pytest.approx(39.58507229888187, abs=1e-9)


def test_pixels_par_degre_double_la_distance_double_le_ppd():
    """Sanity physique : mm_par_degre est linéaire en distance (petit angle),
    donc ppd double si la distance double, à px_par_mm fixé."""
    ppd_1 = pixels_par_degre(37.8, 10.0, 300.0)
    ppd_2 = pixels_par_degre(37.8, 10.0, 600.0)
    assert ppd_2 == pytest.approx(2.0 * ppd_1, rel=1e-12)


def test_pixels_par_degre_rejette_longueur_ref_non_positive():
    with pytest.raises(ValueError):
        pixels_par_degre(37.8, 0.0, 600.0)


def test_pixels_par_degre_rejette_distance_non_positive():
    with pytest.raises(ValueError):
        pixels_par_degre(37.8, 10.0, 0.0)


# --- Famille 2 : taille_domaine_px, inverse-cohérence ------------------------


@pytest.mark.parametrize("ppd", [20.0, 39.58507229888187, 60.0, 150.0])
@pytest.mark.parametrize("porteuse,c_deg_cible", [(5.5, 3.0), (4.0, 3.0), (7.0, 3.0), (5.5, 6.0)])
def test_taille_domaine_px_inverse_coherente(ppd, porteuse, c_deg_cible):
    """La géométrie RENDUE (domaine_px, entier) redonne c_deg_cible à une
    tolérance dictée par l'arrondi (+/- 0.5 px sur domaine_px) -- calcule
    c_deg_realise = porteuse / (domaine_px / ppd) INDÉPENDAMMENT et vérifie
    qu'il reste proche de la cible demandée."""
    domaine_px = taille_domaine_px(ppd, porteuse, c_deg_cible)
    assert isinstance(domaine_px, int)
    assert domaine_px > 0

    domaine_deg_realise = domaine_px / ppd
    c_deg_realise = porteuse / domaine_deg_realise
    # Tolérance : l'arrondi à +/-0.5px sur domaine_px change domaine_deg de
    # +/-0.5/ppd, donc c_deg_realise dévie au plus d'environ
    # c_deg_cible * (0.5/ppd) / domaine_deg_ideal en relatif -- borne large
    # (facteur 3) pour ne pas coupler le test à l'algèbre exacte de l'implém.
    domaine_deg_ideal = porteuse / c_deg_cible
    tol_rel = 3.0 * (0.5 / ppd) / domaine_deg_ideal
    assert c_deg_realise == pytest.approx(c_deg_cible, rel=max(tol_rel, 1e-6))


def test_taille_domaine_px_cas_connu_sans_arrondi():
    """Cas construit pour tomber pile sur un entier (ppd=30, porteuse=6,
    cible=3 => domaine_px = 6*30/3 = 60 exact, aucun résidu d'arrondi)."""
    assert taille_domaine_px(30.0, 6.0, 3.0) == 60


def test_taille_domaine_px_rejette_cible_non_positive():
    with pytest.raises(ValueError):
        taille_domaine_px(30.0, 5.5, 0.0)


# --- Famille 3 : plafond_texture + report pic-CSF (§C7 pièce 3) -------------


def test_plafond_texture_cas_connu():
    """seuil=1.0 arcmin, n_cellules=64, ppd=39.58507... (cas de la famille 1)
    => domaine_deg_max = 1/60*64 = 1.0666...deg (calcul indépendant),
    domaine_px_max = domaine_deg_max*ppd."""
    ppd = 39.58507229888187
    seuil = 1.0
    plafond = plafond_texture(ppd, seuil, n_cellules=64)

    domaine_deg_max_attendu = seuil / 60.0 * 64
    assert domaine_deg_max_attendu == pytest.approx(1.0666666666666667, abs=1e-12)
    assert plafond["domaine_deg_max"] == pytest.approx(domaine_deg_max_attendu, abs=1e-12)
    assert plafond["domaine_px_max"] == pytest.approx(domaine_deg_max_attendu * ppd, abs=1e-9)
    assert plafond["cellule_arcmin_au_plafond"] == pytest.approx(seuil, abs=1e-12)
    assert plafond["n_cellules"] == 64


def test_plafond_texture_domaine_px_max_est_lineaire_en_ppd():
    p1 = plafond_texture(40.0, 1.0)
    p2 = plafond_texture(80.0, 1.0)
    assert p2["domaine_px_max"] == pytest.approx(2.0 * p1["domaine_px_max"], rel=1e-12)
    # domaine_deg_max est indépendant de ppd (géométrie angulaire pure).
    assert p2["domaine_deg_max"] == pytest.approx(p1["domaine_deg_max"], rel=1e-12)


def test_plafond_texture_rejette_seuil_non_positif():
    with pytest.raises(ValueError):
        plafond_texture(40.0, 0.0)


def test_plafond_texture_rejette_ppd_non_positif():
    with pytest.raises(ValueError):
        plafond_texture(0.0, 1.0)


def test_observation_cellule_pic_csf_reproduit_l_exemple_du_brief():
    """Cas gravé du brief : « une cellule au pic-CSF ~1.8° fait ~1.7 arcmin »
    (domaine_deg_ideal = 5.5/3 = 1.8333, cellule_arcmin_ideal =
    1.8333*60/64 = 1.71875). Le calcul RÉALISÉ (après arrondi entier de
    `taille_domaine_px`) doit rester proche de ces valeurs -- l'écart mesure
    exactement le résidu d'arrondi, quel que soit ppd (testé sur 3 valeurs)."""
    domaine_deg_ideal = 5.5 / 3.0
    cellule_arcmin_ideal = domaine_deg_ideal * 60.0 / 64
    assert cellule_arcmin_ideal == pytest.approx(1.71875, abs=1e-12)

    for ppd in (25.0, 39.58507229888187, 80.0, 200.0):
        obs = observation_cellule_pic_csf(ppd)
        assert obs["domaine_deg_pic_csf"] == pytest.approx(domaine_deg_ideal, rel=0.05)
        assert obs["cellule_arcmin_pic_csf"] == pytest.approx(cellule_arcmin_ideal, rel=0.05)
        assert obs["n_cellules"] == N_CELLULES_DOMAINE


def test_observation_cellule_pic_csf_report_imprime_le_chiffre_grave():
    """REPORT OBLIGATOIRE (§C7 pièce 3, ne pas masquer) : imprime la taille
    angulaire d'une cellule au pic-CSF en arcmin et la compare au plafond
    d'acuité par défaut (1.0 arcmin) -- AUCUN assert de décision (la tension
    proche/au-dessus du plafond est une OBSERVATION, pas un critère de test :
    cf. discipline épistémique du dépôt, jamais de seuil retouché pour faire
    passer un gate)."""
    ppd = pixels_par_degre(37.8, 10.0, 600.0)  # géométrie exemple (cf. famille 1)
    obs = observation_cellule_pic_csf(ppd)
    print(
        f"[observation §C7 pièce 3] géométrie pic-CSF (porteuse="
        f"{obs['porteuse_cyc_par_domaine']} cyc/domaine @ {obs['c_deg_cible']} c/deg, "
        f"ppd={ppd:.4f}) : domaine = {obs['domaine_deg_pic_csf']:.4f} deg "
        f"({obs['domaine_px_pic_csf']} px), cellule = "
        f"{obs['cellule_arcmin_pic_csf']:.4f} arcmin ; plafond d'acuité = "
        f"{obs['seuil_acuite_arcmin']:.4f} arcmin (ratio cellule/plafond = "
        f"{obs['ratio_cellule_sur_seuil']:.4f}). Aucune décision : chiffre gravé "
        "pour observation (§C7 pièce 3, COLMATAGE MONOTONIE)."
    )
    # Sanity de forme seulement -- pas un critère de gate.
    assert obs["cellule_arcmin_pic_csf"] > 0.0
    assert obs["ratio_cellule_sur_seuil"] > 0.0
