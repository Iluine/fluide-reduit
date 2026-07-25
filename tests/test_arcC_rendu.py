"""Arc C / P1 — falsifieurs d'acceptation du noyau R1 (`src/arcC_rendu.py`,
§A37 de `pocCascade2phys/PREREGISTRATION.md`, spec §5).

Chaque test NOMME son consommateur : c'est la règle D17 du projet (aucun
diagnostic sans une décision nommée qui en dépend). Les falsifieurs de P1
GATENT l'achat de P2 -- on chiffre un instrument VALIDÉ, jamais l'inverse.

Familles :
  1. déterminisme / test à blanc  -> re-jouabilité des sessions P3, anti-PERSIST
  2. identité de calibration      -> le rééchantillonneur s'efface bit-exact
  3. préservation de la porteuse  -> la LECTURE de P3 (§C7, pic CSF)
  4. monotonie de chaîne          -> ce que viridis garantissait par conception
  5. valeurs connues sRGB         -> la formule gravée fait ce qu'elle dit
  6. gardes fail-loud             -> jamais un clip silencieux
  7. empreinte-verrou             -> « P3 date son verdict sur R1 »

Tous purs, sans GPU, headless : la VM Cowork suffit à les développer, le
verdict P3 restera iluin-tworings3 natif."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from src import arcC_rendu
from src.arcC_calibration import N_CELLULES_DOMAINE
from src.arcC_rendu import (_reechantillonne_general, quantifie_uint8, reechantillonne_bilineaire,
                            rendu_r1, srgb_encode)

# Tailles d'affichage d'épreuve : 64 (identité de calibration), puis des
# facteurs NON ENTIERS -- c'est le cas général (§2.4 D-P1-2 : « un facteur non
# entier en général »), pas un cas limite exotique.
TAILLES_EPREUVE: tuple[int, ...] = (64, 137, 203, 412)


def _champ_seede(seed: int) -> np.ndarray:
    """Champ albédo d'épreuve 64x64 dans [0,1], seedé -- pas un albédo issu du
    solveur (les tests de R1 ne dépendent d'aucun état physique : R1 est une
    fonction de son entrée, et de rien d'autre)."""
    return np.random.default_rng(seed).uniform(0.0, 1.0, size=(N_CELLULES_DOMAINE,
                                                               N_CELLULES_DOMAINE))


# --- Famille 1 : déterminisme et test à blanc --------------------------------


@pytest.mark.parametrize("taille_px", TAILLES_EPREUVE)
def test_deterministe_memes_octets(taille_px):
    """Falsifieur 1 (spec §5) — deux appels de R1 sur le MÊME champ rendent des
    images bit-identiques. Consommateur : la RE-JOUABILITÉ des sessions P3 (un
    log rejoué doit reproduire les stimuli présentés) et l'invariant §A20-1
    « `z` stable => image stable » -- ici TESTÉ, pas argumenté, ce qui est toute
    la différence avec un shader neuronal qui dérive à état constant.

    Cinq appels, pas deux : un état de readout accidentel (cache, compteur,
    accumulateur) se trahit rarement au deuxième."""
    a = _champ_seede(11)
    reference = rendu_r1(a, taille_px)
    for _ in range(4):
        assert rendu_r1(a, taille_px).tobytes() == reference.tobytes()


@pytest.mark.parametrize("taille_px", TAILLES_EPREUVE)
def test_a_blanc_champs_egaux_donnent_images_identiques(taille_px):
    """Falsifieur 3 (spec §5), le TEST À BLANC — `A_cand ≡ A_ref` (deux
    tableaux ÉGAUX mais DISTINCTS en mémoire) => stimuli présentés identiques,
    donc l'ABX est au hasard PAR CONSTRUCTION. Consommateur : la validité même
    du protocole P3 -- si deux champs égaux rendaient deux images différentes,
    le sujet pourrait discriminer un artefact de rendu au lieu de Δχ.

    C'est l'analogue, côté instrument, du « readout propre, Δχ = 0 exact » de
    l'état des lieux §1.5."""
    a_ref = _champ_seede(12)
    a_cand = a_ref.copy()
    assert a_cand is not a_ref
    assert rendu_r1(a_cand, taille_px).tobytes() == rendu_r1(a_ref, taille_px).tobytes()


# --- Famille 2 : identité de calibration -------------------------------------


def test_identite_calibration_taille_64_efface_le_reechantillonneur():
    """Falsifieur « identité de calibration » (ordre de mission) — à
    `taille_px == 64`, la sortie est EXACTEMENT la quantification de l'entrée
    encodée : le rééchantillonneur s'efface, bit-exact.

    Consommateur : le contrôle A/A. Une session à géométrie 1:1 doit pouvoir
    être comparée à la chaîne sans rééchantillonnage sans qu'un demi-pixel de
    décalage ou un flou parasite s'y glisse."""
    a = _champ_seede(13)
    attendu = quantifie_uint8(srgb_encode(a))
    assert rendu_r1(a, N_CELLULES_DOMAINE).tobytes() == attendu.tobytes()


def test_court_circuit_identite_coincide_avec_le_chemin_general():
    """Le court-circuit de `reechantillonne_bilineaire` (`taille_px == H`
    renvoie une copie) n'est pas un RACCOURCI qui masquerait un désaccord : le
    chemin général, exécuté explicitement à la même taille, rend les mêmes
    octets. `x(j) = (j + 0.5)*1 - 0.5 = j` exactement en float64, donc `w = 0`
    et le poids du voisin est un zéro EXACT.

    Consommateur : le test d'identité de calibration ci-dessus -- sans cette
    vérification, il prouverait seulement que le court-circuit existe, pas que
    la convention de centres de pixels est cohérente avec lui."""
    a = _champ_seede(14)
    assert (reechantillonne_bilineaire(a, N_CELLULES_DOMAINE).tobytes()
            == _reechantillonne_general(a, N_CELLULES_DOMAINE).tobytes())


# --- Famille 3 : préservation de la porteuse ---------------------------------


# Atténuation d'amplitude TOLÉRÉE par le rééchantillonnage bilinéaire, fixée
# AVANT la mesure et depuis la théorie, jamais ajustée au résultat : la réponse
# du noyau triangulaire à la fréquence normalisée f = k/64 cycles/échantillon
# vaut ~sinc²(f), soit ~0.98 à k = 5 et ~0.96 à k = 7 (bande porteuse §C7
# 4-7 cyc/domaine). On borne à 0.90, avec la marge assumée : ce test dit « le
# bilinéaire ATTÉNUE », il ne prétend pas mesurer de combien.
ATTENUATION_MIN_PORTEUSE: float = 0.90


@pytest.mark.parametrize("k", (4, 5, 7))
@pytest.mark.parametrize("taille_px", (137, 203, 412))
def test_porteuse_reste_a_k_cycles_par_domaine(k, taille_px):
    """Falsifieur 4 (spec §5), versant R1 — une sinusoïde à `k` cycles/domaine
    rééchantillonnée reste à `k` cycles/domaine : le pic FFT ne BOUGE PAS en
    fréquence NORMALISÉE (bin `k` du spectre à `taille_px` points), seule son
    amplitude s'atténue.

    Consommateur : la LECTURE de P3. Le contrat §C7 pose la porteuse au pic de
    la CSF (~3 c/deg) ; si l'étage de rééchantillonnage DÉPLAÇAIT la porteuse,
    la géométrie de session ne viserait plus sa cible et un transport ≠ 1
    serait inattribuable (pin ? ou chemin ?).

    La mesure porte sur le RÉÉCHANTILLONNEUR SEUL, en espace linéaire : passer
    par `srgb_encode` d'abord fabriquerait des harmoniques (la formule est non
    linéaire) et l'on mesurerait la distorsion de l'EOTF, pas le déplacement de
    la porteuse."""
    ligne = 0.5 + 0.4 * np.sin(2.0 * np.pi * k * np.arange(N_CELLULES_DOMAINE) /
                               N_CELLULES_DOMAINE)
    a = np.repeat(ligne[None, :], N_CELLULES_DOMAINE, axis=0)

    spectre = np.abs(np.fft.rfft(reechantillonne_bilineaire(a, taille_px)[0])) / taille_px * 2.0
    pic = int(np.argmax(spectre[1:])) + 1
    assert pic == k, (f"la porteuse a MIGRÉ : pic au bin {pic}, attendu {k} -- le "
                      "rééchantillonnage déplace la fréquence, §C7 perd sa cible.")
    assert spectre[pic] / 0.4 >= ATTENUATION_MIN_PORTEUSE, (
        f"atténuation {spectre[pic] / 0.4:.4f} sous la borne théorique "
        f"{ATTENUATION_MIN_PORTEUSE} -- le bilinéaire ne fait pas ce que sa réponse "
        "fréquentielle prédit.")


# --- Famille 4 : monotonie de la chaîne --------------------------------------


@pytest.mark.parametrize("seed", (0, 1, 2, 3, 4))
@pytest.mark.parametrize("taille_px", TAILLES_EPREUVE)
def test_monotonie_de_chaine(seed, taille_px):
    """Falsifieur 5 (spec §5) — `A₁ ≤ A₂` partout => `rendu(A₁) ≤ rendu(A₂)`
    partout. Consommateur : la validité de l'axe de mesure. Viridis garantissait
    la monotonie de clarté PAR CONCEPTION (c'est sa raison d'être) ; R1 doit la
    garantir PAR TEST, sinon une différence d'albédo pourrait s'inverser à
    l'écran et l'ABX jugerait autre chose que ce que Δχ mesure.

    Chaque étage la préserve (le bilinéaire est une combinaison convexe à poids
    positifs, `srgb_encode` est croissante, `np.rint` est non décroissante) --
    ce test vérifie que la COMPOSITION la préserve aussi, y compris à travers
    la quantification qui écrase des écarts sans jamais les retourner."""
    rng = np.random.default_rng(seed)
    a1 = rng.uniform(0.0, 1.0, size=(N_CELLULES_DOMAINE, N_CELLULES_DOMAINE))
    a2 = np.minimum(a1 + rng.uniform(0.0, 0.3, size=a1.shape), 1.0)
    assert np.all(a1 <= a2)
    assert np.all(rendu_r1(a1, taille_px) <= rendu_r1(a2, taille_px))


# --- Famille 5 : valeurs connues sRGB ----------------------------------------


def test_valeurs_connues_srgb():
    """Falsifieur « valeurs connues » (ordre de mission) — la formule gravée
    fait ce qu'elle dit, aux deux bouts et sur deux points intermédiaires
    CALCULÉS depuis la formule puis codés en dur (donc invariants à toute
    refonte du code : si la valeur change, c'est la CHAÎNE qui a changé).

      y = 0.0        -> 12.92*0        = 0.0
      y = 0.002      -> 12.92*0.002    = 0.02584            (branche linéaire)
      y = 0.5        -> 1.055*0.5^(1/2.4) - 0.055 = 0.7353569830524495
      y = 1.0        -> 1.055 - 0.055  = 0.9999999999999999 (float64, cf. infra)

    Le dernier n'est PAS 1.0 exactement en float64 -- `1.055 - 0.055` vaut
    0.9999999999999999. La quantification retombe malgré tout sur 255
    (254.99999999999997 -> `np.rint` -> 255) : c'est dit ici plutôt que masqué,
    parce que c'est le genre de résidu qui fait un jour douter d'un blanc.

    Consommateur : D-P1-4 (EOTF gravé) -- sans ces points, « inverse-EOTF sRGB »
    ne serait qu'une intention de docstring."""
    assert srgb_encode(np.array([0.0])) == pytest.approx(0.0, abs=0.0)
    assert float(srgb_encode(np.array([0.002]))[0]) == pytest.approx(0.02584, abs=1e-15)
    assert float(srgb_encode(np.array([0.5]))[0]) == pytest.approx(0.7353569830524495, abs=1e-15)
    assert float(srgb_encode(np.array([1.0]))[0]) == pytest.approx(0.9999999999999999, abs=1e-15)

    noir = rendu_r1(np.zeros((N_CELLULES_DOMAINE, N_CELLULES_DOMAINE)), N_CELLULES_DOMAINE)
    blanc = rendu_r1(np.ones((N_CELLULES_DOMAINE, N_CELLULES_DOMAINE)), N_CELLULES_DOMAINE)
    assert np.all(noir == 0)
    assert np.all(blanc == 255)


# --- Famille 6 : gardes fail-loud --------------------------------------------


def test_garde_forme_non_64x64():
    """Garde de PÉRIMÈTRE : R1 rend le domaine du projet (64x64), pas un champ
    quelconque. Un champ d'une autre taille signifierait qu'un producteur non
    prévu (compositeur ? niveau grossier ?) est arrivé jusqu'ici sans décision
    -- Option A est différée post-P3, elle ne doit pas entrer par la fenêtre."""
    with pytest.raises(ValueError, match="forme"):
        rendu_r1(np.zeros((32, 32)), 64)
    with pytest.raises(ValueError, match="forme"):
        rendu_r1(np.zeros((64, 63)), 64)


def test_garde_valeurs_hors_bande_et_non_finies():
    """Fail-loud, JAMAIS un clip silencieux : l'albédo est dans [0,1) par
    construction (`1 - exp(-s/s_half)`) ; une valeur hors bande ou non finie
    est un BUG AMONT. Un clip la ferait disparaître dans une image plausible,
    et la session mesurerait un stimulus qui n'existe pas."""
    for valeur in (1.5, -0.1, np.nan, np.inf):
        champ = np.full((N_CELLULES_DOMAINE, N_CELLULES_DOMAINE), valeur)
        with pytest.raises(ValueError):
            rendu_r1(champ, 64)


def test_garde_dtype_et_taille_px():
    """Le float64 est EXIGÉ (une entrée float32 rendrait une autre image sans
    rien signaler, et la re-jouabilité bit à bit est un invariant testé) ; et
    `taille_px` doit être un entier > 0."""
    with pytest.raises(ValueError, match="float64"):
        rendu_r1(np.zeros((N_CELLULES_DOMAINE, N_CELLULES_DOMAINE), dtype=np.float32), 64)
    for mauvaise in (0, -1):
        with pytest.raises(ValueError, match="taille_px"):
            rendu_r1(_champ_seede(15), mauvaise)
    with pytest.raises(ValueError, match="entier"):
        rendu_r1(_champ_seede(15), 64.0)


# --- Famille 7 : EMPREINTE-VERROU --------------------------------------------


# GRAVÉE le 2026-07-25 (P1, §A37 endossé ; ordre de mission
# `claude/mission-p1-rendu-instrument.md`). Empreinte sha256 de la SOURCE de
# `src/arcC_rendu.py`, fichier lu en OCTETS -- tradition des verrous kernels F1
# (`tests/test_f1_substrat_fidele.py`).
#
# POURQUOI un verrou sur du Python pur, qui n'a pas de compilation à protéger :
# §A37 grave que « P3 date son verdict sur R1 ». Le pin transporté s'étiquettera
# « à travers R1 » -- cette étiquette n'a de sens que si R1 est un objet DATÉ.
# Ce test ne protège pas la stabilité, il protège l'ATTRIBUABILITÉ.
#
# Que faire quand il casse : il a fait son travail. Recalculer, re-graver ICI
# avec un commentaire datant la rupture et nommant sa provenance, et METTRE À
# JOUR toute citation de l'ancienne empreinte -- jamais l'inverse.
SHA256_ATTENDU: str = (
    "c5ac875798d7cdb770702e9d612683ab3448ec8556fa2e81b3d902d426e4fe10")
LONGUEUR_ATTENDUE: int = 15770


def test_empreinte_source_r1():
    """VERROU d'attribution (§A37) : la source de `src/arcC_rendu.py` n'a pas
    bougé sans qu'on le dise. Le fichier est lu en octets et haché ici même,
    indépendamment du module -- un verrou qui déléguerait son calcul à l'objet
    verrouillé ne verrouillerait rien."""
    source = Path(arcC_rendu.__file__).read_bytes()
    empreinte = hashlib.sha256(source).hexdigest()
    assert len(source) == LONGUEUR_ATTENDUE, (
        f"longueur de src/arcC_rendu.py = {len(source)}, attendue {LONGUEUR_ATTENDUE} -- "
        "R1 a changé. Si c'est voulu : re-graver ici, dater, et mettre à jour toute "
        "citation de l'empreinte.")
    assert empreinte == SHA256_ATTENDU, (
        f"empreinte de src/arcC_rendu.py = {empreinte}, attendue {SHA256_ATTENDU} -- "
        "R1 a changé. « P3 date son verdict sur R1 » : un verdict daté exige un objet "
        "daté.")
