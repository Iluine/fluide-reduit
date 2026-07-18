"""Tests Task S1 -- sonde fenêtrage-fovéa §A14
(`scripts/run_arcA_sonde_fenetre.py`, texte gravé : PREREGISTRATION.md §A14,
pocCascade2phys f08c340 -- le texte gravé fait foi).

Budget de calcul : les tests mécaniques (fenêtre, gardes, instrument)
n'exécutent AUCUNE physique (champs synthétiques + artefacts GELÉS
h160_s101.npz) ; seuls les tests de composition/rederivation simulent, au
MINIMUM (dt=1, n_emissions=2, vérité partagée scope module -- ~6 épisodes,
motif de test_arcA_m2_registre.py), pour rester sous la limite infra VM
45 s/processus. Le quadrant du mini-test physique est (0,1) : les centres
des épisodes 1-2 du seed 101 ((0.81, 0.40) et (0.70, 0.56)) déposent en
(0,1) et (1,1) -- le commit fenêtré en (0,1) est donc réellement avec perte
dès la première émission (delta_chi_fen[1] > 0 non trivial)."""
from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import numpy as np
import pytest

from scripts.run_arcA_revalidate import S_HALF_OP
from scripts.run_arcA_sonde_fenetre import (
    COTE_FENETRE,
    DT_SONDE,
    K_FEN,
    K_OUT_ESCALADE,
    N_EMISSIONS_SONDE,
    OUT_JSON_PATH,
    QUADRANTS_ORDRE,
    SEED_SONDE,
    chaine_fenetre,
    choisir_fenetre,
    commettre_fenetre,
    extraire_fenetre,
    fenetre_slices,
    plonger_fenetre,
    rederiver_emissions_fenetre,
    verifier_instrument_fenetre,
)
from scripts.run_arcA_m2_registre import chaine_verite
from src.summary_quadtree import size_floats_qt, summarize_qt

ROOT = Path(__file__).resolve().parents[1]
_H160_S101_PATH = ROOT / "outputs" / "arcA" / "histories" / "h160_s101.npz"

# JND sévère gravé (outputs/arcC/pins_spatial.json -> regimes.severe.jnd),
# même littéral que test_arcA_m2_sonde.py.
_JND_SEV = 0.07334065957385666


# --- 1. Constantes gravées §A14 ----------------------------------------------


def test_constantes_gravees_a14():
    """Les paramètres FIGÉS par le texte gravé §A14 : cellule reconduite
    §A13-3 (seed 101, Δt=4, 6 émissions), fenêtre fixe 32x32, k_fen=64
    (aire-proportionnel, choix (i)), k_out=64 (escalade PRÉ-ÉCRITE, figée,
    non implémentée dans le bras nu), ordre FIXE des quadrants."""
    assert SEED_SONDE == 101
    assert DT_SONDE == 4
    assert N_EMISSIONS_SONDE == 6
    assert COTE_FENETRE == 32
    assert K_FEN == 64
    assert K_OUT_ESCALADE == 64
    assert QUADRANTS_ORDRE == ((0, 0), (0, 1), (1, 0), (1, 1))


# --- 2. Mécanique de fenêtre --------------------------------------------------


def test_fenetre_slices_quadrants():
    """Quadrant (qy, qx) -> lignes [qy*32, (qy+1)*32), colonnes
    [qx*32, (qx+1)*32) (indexation array[y, x] du dépôt) ; quadrant hors de
    l'ordre fixe -> ValueError (fail-loud)."""
    attendus = {
        (0, 0): (slice(0, 32), slice(0, 32)),
        (0, 1): (slice(0, 32), slice(32, 64)),
        (1, 0): (slice(32, 64), slice(0, 32)),
        (1, 1): (slice(32, 64), slice(32, 64)),
    }
    for quadrant, (sy, sx) in attendus.items():
        assert fenetre_slices(quadrant) == (sy, sx)
    with pytest.raises(ValueError):
        fenetre_slices((2, 0))


def test_plonger_fenetre_zero_dehors():
    """Le plongement (choix (v)) : champ 32x32 -> champ 64x64 EXACTEMENT nul
    hors fenêtre, valeurs bit-identiques dedans ; shape != 32x32 ->
    ValueError (fail-loud)."""
    rng = np.random.default_rng(7)
    champ_fen = rng.uniform(0.0, 1.0, size=(32, 32))
    plein = plonger_fenetre(champ_fen, (1, 0))
    assert plein.shape == (64, 64)
    assert np.array_equal(plein[32:64, 0:32], champ_fen)
    dehors = plein.copy()
    dehors[32:64, 0:32] = 0.0
    assert np.all(dehors == 0.0)
    with pytest.raises(ValueError):
        plonger_fenetre(champ_fen[:16], (0, 0))


def test_commettre_fenetre_budget_dehors_nul_et_determinisme():
    """Commit fenêtré sur champ synthétique : taille <= k_fen=64 (garde
    budget) ; la reconstruction est EXACTEMENT nulle hors fenêtre (le commit
    ne porte AUCUNE information du dehors -- c'est ce qui rend le plongement
    licite) ; `extraire_fenetre` restitue la fenêtre de la reconstruction ;
    deux appels identiques -> commits bit-identiques (déterminisme)."""
    rng = np.random.default_rng(11)
    champ_fen = rng.uniform(0.0, 1.0, size=(32, 32))
    commit = commettre_fenetre(champ_fen, (0, 1), K_FEN)
    assert size_floats_qt(commit) <= K_FEN

    from src.summary_quadtree import regenerate_qt
    recon = regenerate_qt(commit)
    dehors = recon.copy()
    dehors[0:32, 32:64] = 0.0
    assert np.all(dehors == 0.0)
    assert np.array_equal(extraire_fenetre(commit, (0, 1)), recon[0:32, 32:64])

    commit2 = commettre_fenetre(champ_fen, (0, 1), K_FEN)
    assert np.array_equal(commit2.topology, commit.topology)
    assert np.array_equal(commit2.means, commit.means)


def test_commettre_fenetre_projection():
    """Propriété de projection S∘R = id sur le chemin fenêtré : recommettre
    l'extraction d'un commit redonne le commit bit-identique (topologie ET
    moyennes) -- les invariants commis dans la fenêtre sont préservés
    EXACTEMENT (reconduit du test É1-projection Task 0)."""
    rng = np.random.default_rng(13)
    champ_fen = rng.uniform(0.0, 1.0, size=(32, 32))
    commit = commettre_fenetre(champ_fen, (0, 0), K_FEN)
    commit2 = commettre_fenetre(extraire_fenetre(commit, (0, 0)), (0, 0), K_FEN)
    assert np.array_equal(commit2.topology, commit.topology)
    assert np.array_equal(commit2.means, commit.means)


def test_extraire_fenetre_fail_loud_si_dehors_non_nul():
    """Garde fail-loud : un commit dont la reconstruction est non nulle HORS
    fenêtre (ici : summarize_qt d'un champ plein, hors du chemin
    `commettre_fenetre`) est REJETÉ par `extraire_fenetre` (RuntimeError) --
    jamais de fenêtre extraite d'un commit contaminé par le dehors."""
    rng = np.random.default_rng(17)
    champ_plein = rng.uniform(0.5, 1.0, size=(64, 64))
    commit_plein = summarize_qt(champ_plein, K_FEN)
    with pytest.raises(RuntimeError):
        extraire_fenetre(commit_plein, (0, 0))


# --- 3. Garde anti-vacuité (mécanique, ordre fixe) ---------------------------


def _champ_signal(quadrant: tuple[int, int]) -> np.ndarray:
    """Champ 64x64 nul partout sauf un signal sinusoïdal (bande 4, supra-JND
    au readout albedo) dans la fenêtre `quadrant` -- fabrique des vérités
    synthétiques pour la garde, AUCUNE physique simulée."""
    champ = np.zeros((64, 64), dtype=np.float64)
    sy, sx = fenetre_slices(quadrant)
    xx = np.arange(32, dtype=np.float64)[None, :]
    champ[sy, sx] = S_HALF_OP * (2.0 + np.sin(2.0 * np.pi * 4.0 * xx / 32.0))
    return champ


def test_choisir_fenetre_aucune_vivante_fail_loud():
    """Vérité EXACTEMENT nulle dans les 4 quadrants (fenêtre morte partout =
    sonde triviale) -> RuntimeError (remonter, jamais mesurer)."""
    verite = {4: np.zeros((64, 64), dtype=np.float64)}
    with pytest.raises(RuntimeError):
        choisir_fenetre(verite, [4], _JND_SEV)


def test_choisir_fenetre_retient_le_quadrant_vivant():
    """Signal supra-JND uniquement en (1, 0) : (0,0) et (0,1) sont morts,
    essayés puis écartés ; (1,0) est le premier vivant -> retenu."""
    verite = {4: _champ_signal((1, 0))}
    resultat = choisir_fenetre(verite, [4], _JND_SEV)
    assert resultat["quadrant"] == (1, 0)


def test_choisir_fenetre_ordre_fixe_premier_vivant():
    """Signal en (0,1) ET (1,1) : la règle gravée est mécanique -- quadrants
    essayés dans l'ordre FIXE (0,0)->(0,1)->(1,0)->(1,1), PREMIER vivant
    retenu -> (0,1), zéro choix après lecture."""
    verite = {4: _champ_signal((0, 1)) + _champ_signal((1, 1))}
    resultat = choisir_fenetre(verite, [4], _JND_SEV)
    assert resultat["quadrant"] == (0, 1)


# --- 4. Vérification d'instrument (§A14, DUE en build, AVANT run) ------------


@pytest.mark.skipif(not _H160_S101_PATH.exists(),
                    reason="histoire gelée h160_s101.npz absente")
def test_verifier_instrument_fenetre_rapport():
    """La vérification gravée §A14 (« les bandes porteuses de max_carrier
    sont-elles définies sur 32x32 ? ») tourne sur les artefacts GELÉS
    (h160_s101.npz, s_L10/s_L20 -- épisodes 10 et 20, dans la plage sonde
    t ∈ [4, 24]), AUCUNE simulation. Vérifie : (a) les 5 bandes radiales ont
    des bins sur une grille 32x32 (fait structurel -- si FAUX, l'instrument
    est muet et ce test échoue = remontée) ; (b) le rapport est cohérent
    (muette <-> raisons non vides) ; (c) les porteuses fenêtre sont
    rapportées pour chaque (instantané, quadrant)."""
    rapport = verifier_instrument_fenetre()
    assert set(rapport["bandes_definies_32"]) == {"1", "2-3", "4-7", "8-15", "16-31"}
    assert all(rapport["bandes_definies_32"].values())
    assert isinstance(rapport["muette"], bool)
    assert rapport["muette"] == bool(rapport["raisons"])
    for nom in ("s_L10", "s_L20"):
        assert nom in rapport["porteuses_plein_domaine"]
        assert set(rapport["porteuses_fenetre"][nom]) == {"0,0", "0,1", "1,0", "1,1"}


# --- 5. Anti-fuite signature --------------------------------------------------


def test_anti_fuite_signature_rederiver_emissions_fenetre():
    """Signature STRICTE reconduite (motif Task 0) : exactement (registre,
    seed, dt, n_emissions, quadrant) -- pas de paramètre « champ vrai », pas
    d'accès aux états vivants d'un appel de `chaine_fenetre`."""
    sig = inspect.signature(rederiver_emissions_fenetre)
    assert list(sig.parameters.keys()) == [
        "registre", "seed", "dt", "n_emissions", "quadrant"]


# --- 6. Mini-chaîne physique (dt=1, n=2, quadrant (0,1)) ---------------------


@pytest.fixture(scope="module")
def verite_2() -> dict[int, np.ndarray]:
    """Histoire vérité seed=101, 2 épisodes (~2 x 3.8 s) -- partagée
    (cf. motif test_arcA_m2_registre.py)."""
    return chaine_verite(101, 2)


@pytest.fixture(scope="module")
def chaine_fen_2em(verite_2) -> dict:
    """chaine_fenetre(101, dt=1, n_emissions=2, k_fen=64, quadrant=(0,1)),
    vérité partagée -- ~2 x 3.8 s (moteur seul). Consommée en LECTURE SEULE
    sauf par le test anti-fuite décoy, qui mute `etats_vrais` (champ que ce
    module ne relit jamais ailleurs)."""
    return chaine_fenetre(101, dt=1, n_emissions=2, k_fen=64, quadrant=(0, 1),
                          verite=verite_2)


def test_composition_fenetre_sanity(chaine_fen_2em):
    """delta_chi_fen[0] == 0.0 EXACT (aucun commit avant la première
    émission : moteur et vérité bit-identiques, dedans comme dehors --
    choix (iv) : Δχ_fen,1 ≡ 0 structurel, exclu de la lecture) ET
    delta_chi_fen[1] > 0.0 (le commit fenêtré quantifie réellement -- si 0,
    le harnais ne mesure rien). Le diagnostic plein-domaine partage la
    structure ([0] == 0.0 exact) et les commits tiennent dans k_fen."""
    res = chaine_fen_2em
    assert res["delta_chi_fen"][0] == 0.0
    assert res["delta_chi_fen"][1] > 0.0
    assert res["delta_chi_plein"][0] == 0.0
    assert all(t <= res["k_fen"] for t in res["taille_commits"])
    assert res["episodes_emissions"] == [1, 2]


def test_rederivation_fenetre_bit_exacte_et_anti_fuite(chaine_fen_2em):
    """É2/É3 reconduits : les readouts-moteur se rederivent de (registre,
    seed) SEULS, bit-identiques -- la mutation PRÉALABLE de `etats_vrais`
    (décoy 999.0) prouve dans le même geste qu'aucun canal ne fuit du monde
    vrai vers la rederivation. Auto-cohérence : recommettre la fenêtre de
    chaque readout rederivé redonne le commit du registre bit-identique."""
    res = chaine_fen_2em
    for etat in res["etats_vrais"]:
        etat[:] = 999.0
    rederive = rederiver_emissions_fenetre(res["registre"], 101, 1, 2, (0, 1))
    assert len(rederive) == 2
    sy, sx = fenetre_slices((0, 1))
    for i in range(2):
        assert np.array_equal(rederive[i], res["readouts_moteur"][i])
        recommit = commettre_fenetre(rederive[i][sy, sx], (0, 1), 64)
        assert np.array_equal(recommit.topology, res["registre"][i].topology)
        assert np.array_equal(recommit.means, res["registre"][i].means)


# --- 7. Aucun timestamp calendaire dans le JSON de sortie --------------------

_MOTIF_DATE_CALENDAIRE = re.compile(r"\b\d{4}-\d{2}-\d{2}")


@pytest.mark.skipif(not OUT_JSON_PATH.exists(),
                    reason="m2_sonde_fenetre.json absent (sonde non exécutée)")
def test_json_sortie_sans_timestamp_calendaire():
    """`outputs/arcA/m2_sonde_fenetre.json` : AUCUN timestamp calendaire
    (motif reconduit de test_arcA_m2_sonde.py) -- seuls les chronos `*_s`
    (durées) sont autorisés."""
    texte = OUT_JSON_PATH.read_text(encoding="utf-8")
    assert _MOTIF_DATE_CALENDAIRE.search(texte) is None
    document = json.loads(texte)
    for cle in ("date", "date_session", "timestamp", "horodatage"):
        assert cle not in document
        assert cle not in document.get("meta", {})
