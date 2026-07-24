"""Tests F1 — M-b : décomposition SPATIALE de Δχ, sans toucher l'observable
(§A32).

CE QUI EST ILLÉGITIME, ET POURQUOI ON NE LE FAIT PAS. Restreindre
`delta_chi` à une sous-fenêtre casse deux fois :
  1. le COMPLÉMENT de la fovéa est une couronne en L — `chi_bands` fait une
     `fft2` et exige un tableau rectangulaire (carré, même : `_radial_bins`
     suppose H = W). Zéro-remplir changerait `mean_a`, qui DIVISE χ ;
  2. même pour la fovéa seule, qui est rectangulaire : à 32² la dernière
     bande `(16, ∞)` s'arrête au Nyquist 22,6 au lieu de 45,25 — ce ne sont
     plus les mêmes bandes, et les PORTEUSES seraient redéfinies sur un
     autre domaine. C'est le piège §A14 : une sonde muette par construction.

CE QU'ON FAIT À LA PLACE : on masque LA DIFFÉRENCE, pas le domaine. Le
candidat hybride vaut la production dans la région étudiée et le rederive
ailleurs, sur le domaine PLEIN. L'observable n'est pas touché ; surtout, la
RÉFÉRENCE est la même dans les trois évaluations, donc les bandes PORTEUSES
— définies sur la seule référence — sont IDENTIQUES. C'est précisément ce
que la restriction détruisait.

Ce que ces tests protègent :
  1. l'hybride est bien production-dedans / rederive-dehors ;
  2. masque plein ⇒ on retombe EXACTEMENT sur le Δχ mesuré (non-régression
     de l'observable : la décomposition ne crée pas un autre instrument) ;
  3. les PORTEUSES sont identiques entre plein, fovéa et périphérie ;
  4. la partition d'énergie est EXACTEMENT additive (corroboration sans
     artefact spectral)."""
import numpy as np

from src.f1_gpu.cellule_mb import dchi
from src.f1_gpu.decomposition_spatiale import (
    champ_hybride,
    decomposer,
    fidelite_structurelle,
    masque_fenetre,
    partition_energie,
)

N = 64
COTE = 32


def _champs(graine=0):
    rng = np.random.default_rng(graine)
    ref = 0.2 + 0.3 * rng.random((N, N))
    cand = ref + 0.15 * rng.random((N, N))
    return cand, ref


# ----- 1. l'hybride -----

def test_masque_fenetre_geometrie():
    m = masque_fenetre(N, oy=16, ox=8, cote=COTE)
    assert m.shape == (N, N) and m.dtype == bool
    assert m.sum() == COTE * COTE
    assert m[16, 8] and m[47, 39]
    assert not m[15, 8] and not m[16, 7]


def test_hybride_production_dedans_rederive_dehors():
    cand, ref = _champs()
    m = masque_fenetre(N, 16, 8, COTE)
    h = champ_hybride(cand, ref, m)
    np.testing.assert_array_equal(h[m], cand[m])
    np.testing.assert_array_equal(h[~m], ref[~m])


def test_hybride_complementaire():
    """Les deux hybrides se recomposent : dedans l'un, dehors l'autre."""
    cand, ref = _champs()
    m = masque_fenetre(N, 16, 8, COTE)
    a, b = champ_hybride(cand, ref, m), champ_hybride(cand, ref, ~m)
    np.testing.assert_allclose(a + b - ref, cand, rtol=0, atol=1e-12)


# ----- 2. non-régression de l'observable -----

def test_masque_plein_redonne_exactement_le_dchi_mesure():
    """Le verrou : avec un masque plein, la décomposition retombe sur le Δχ
    déjà mesuré, au bit près. La décomposition n'est pas un autre
    instrument."""
    cand, ref = _champs()
    plein = np.ones((N, N), dtype=bool)
    assert dchi(champ_hybride(cand, ref, plein), ref) == dchi(cand, ref)


def test_masque_vide_donne_dchi_nul():
    cand, ref = _champs()
    vide = np.zeros((N, N), dtype=bool)
    assert dchi(champ_hybride(cand, ref, vide), ref) == 0.0


# ----- 3. les PORTEUSES sont identiques : le coeur de la légitimité -----

def test_porteuses_identiques_entre_plein_fovea_peripherie():
    """Les porteuses ne dépendent QUE de la référence, inchangée dans les
    trois évaluations. C'est ce qui rend les trois nombres comparables — et
    ce qu'une restriction de domaine aurait détruit."""
    cand, ref = _champs()
    d = decomposer(cand, ref, masque_fenetre(N, 16, 8, COTE))
    assert d["porteuses"] == d["porteuses_fovea"] == d["porteuses_peripherie"]


def test_decomposer_reporte_la_bande_dominante():
    """Quelle BANDE porte le max est directement diagnostique : la bande la
    plus fine est la signature d'une décimation 2×."""
    cand, ref = _champs()
    d = decomposer(cand, ref, masque_fenetre(N, 16, 8, COTE))
    assert d["bande_dominante"] in ("1", "2-3", "4-7", "8-15", "16-31")
    assert d["porteuses"][d["bande_dominante"]] is True


def test_decomposer_porte_le_plein_mesure():
    cand, ref = _champs()
    d = decomposer(cand, ref, masque_fenetre(N, 16, 8, COTE))
    assert d["dchi_plein"] == dchi(cand, ref)


# ----- 4. partition d'énergie : exacte, sans artefact spectral -----

def test_partition_energie_exactement_additive():
    """Corroboration sans artefact : ‖d‖²_fovéa + ‖d‖²_périphérie = ‖d‖²
    EXACTEMENT. Le masquage spectral, lui, fuit au bord — d'où ce second
    regard, additif par construction."""
    cand, ref = _champs()
    p = partition_energie(cand, ref, masque_fenetre(N, 16, 8, COTE))
    assert p["energie_fovea"] + p["energie_peripherie"] == p["energie_totale"]


def test_partition_energie_par_cellule_normalise_l_aire():
    """La fovéa fait 25 % de l'aire : comparer des énergies BRUTES
    favoriserait mécaniquement la périphérie. On normalise par cellule."""
    cand, ref = _champs()
    m = masque_fenetre(N, 16, 8, COTE)
    p = partition_energie(cand, ref, m)
    assert p["aire_fovea"] == COTE * COTE
    assert p["aire_peripherie"] == N * N - COTE * COTE
    attendu = p["energie_fovea"] / p["aire_fovea"]
    assert p["energie_par_cellule_fovea"] == attendu


# ----- 5. le discriminant « corruption » -----

def test_fidelite_structurelle_champ_identique():
    _, ref = _champs()
    f = fidelite_structurelle(ref, ref)
    assert f["correlation"] == 1.0
    assert f["erreur_rms_relative"] == 0.0


def test_fidelite_structurelle_distingue_melange_et_amplitude():
    """Le coeur du discriminant : un champ MÉLANGÉ (corruption délibérée)
    est décorrélé ; un champ mal AMPLIFIÉ reste parfaitement corrélé. Δχ
    peut être grand dans les deux cas — la corrélation les sépare."""
    rng = np.random.default_rng(3)
    ref = 0.2 + 0.3 * rng.random((N, N))
    melange = rng.permutation(ref.ravel()).reshape(N, N)
    amplifie = ref * 1.5
    assert abs(fidelite_structurelle(melange, ref)["correlation"]) < 0.2
    assert fidelite_structurelle(amplifie, ref)["correlation"] > 0.999


def test_partition_detecte_une_degradation_localisee():
    """Si l'écart est CONFINÉ à la périphérie, la partition doit le dire."""
    rng = np.random.default_rng(1)
    ref = 0.2 + 0.3 * rng.random((N, N))
    m = masque_fenetre(N, 16, 8, COTE)
    cand = ref.copy()
    cand[~m] += 0.2                       # dégradation hors fovéa seulement
    p = partition_energie(cand, ref, m)
    assert p["energie_fovea"] == 0.0
    assert p["energie_peripherie"] > 0.0
    assert p["fraction_fovea"] == 0.0
