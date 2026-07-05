"""Tests Manche 2 (§A13) Task 0 -- cœur du harnais registre-commis
(`scripts/run_arcA_m2_registre.py`, brief `.superpowers/sdd/m2-task0-brief.md`).

Dix familles de tests exigées par le brief (numérotées ci-dessous, items 1-10)
+ deux tests bonus bon marché (wrapper `centres_episodes`, grille de constantes).
Paramètres MINUSCULES imposés par le brief : seed=101, dt=1, n_emissions=2-3,
k=128 (sauf ferm : k=4096). Chaque épisode simulé coûte ~3.7-3.9 s CPU (mesuré
sur cette machine) -- les histoires vérité et les chaînes-registre sont
PARTAGÉES entre tests via des fixtures `scope="module"` partout où le brief le
permet (« état vrai quelconque », « optimisation autorisée, physique
bit-identique » du motif déjà en usage dans `scripts/run_arcA_measure.py`),
pour rester proche du budget de calcul visé par le brief malgré les items qui
imposent des appels réellement indépendants (item 2 : replay contre l'artefact
gelé h_s101.npz, n=10 non réductible ; item 5 : deux VRAIS process séparés)."""
from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.run_arcA_m2_registre import (
    centres_episodes,
    chaine_registre,
    chaine_verite,
    rederiver_emissions,
)
from scripts.run_arcA_measure import _draw_rollout_pairs
from src.summary_quadtree import regenerate_qt, summarize_qt

ROOT = Path(__file__).resolve().parents[1]
_H_S101_PATH = ROOT / "outputs" / "arcA" / "histories" / "h_s101.npz"


# --- Fixtures partagées (module) ---------------------------------------------


@pytest.fixture(scope="module")
def verite_3() -> dict[int, np.ndarray]:
    """Histoire vérité seed=101, 3 épisodes (~3 x 3.7 s) -- partagée par les
    tests qui n'ont besoin que d'un « état vrai quelconque » (item 1) ou d'une
    vérité déjà connue pour n_emissions=3 (items 3, 6, 8, 9, 10 via
    `chaine_v2_3em` ci-dessous), évitant de la resimuler à chaque fois."""
    return chaine_verite(101, 3)


@pytest.fixture(scope="module")
def chaine_v2_3em(verite_3) -> dict:
    """chaine_registre(101, dt=1, n_emissions=3, k=128, arm="v2"), vérité
    partagée (`verite_3`) -- ~3 x 3.7 s (moteur seul, vérité déjà connue).
    Consommée en LECTURE SEULE par items 6, 9, 10 ; item 8 mute
    `["etats_vrais"]` de CET objet -- sans risque pour les autres tests du
    module, aucun d'eux ne lit ce champ (cf. docstring du test 8)."""
    return chaine_registre(101, dt=1, n_emissions=3, k=128, arm="v2", verite=verite_3)


@pytest.fixture(scope="module")
def rederive_v2_3em(chaine_v2_3em) -> list[np.ndarray]:
    """rederiver_emissions(chaine_v2_3em["registre"], 101, 1, 3) -- ~3 x 3.7 s,
    calculé UNE fois et partagé par les tests qui n'ont pas besoin d'un second
    appel indépendant (item 6). L'item 8 (anti-fuite décoy) refait SON PROPRE
    appel après mutation -- ne réutilise pas cette fixture (cf. son docstring)."""
    return rederiver_emissions(chaine_v2_3em["registre"], 101, 1, 3, arm="v2")


# --- Bonus (gratuit, pas de physique) -----------------------------------------


def test_centres_episodes_wrapper_correspond_a_draw_rollout_pairs():
    attendu = tuple(_draw_rollout_pairs(101, 5))
    obtenu = centres_episodes(101, 5)
    assert obtenu == attendu
    assert isinstance(obtenu, tuple)
    assert all(isinstance(c, tuple) for c in obtenu)


# --- 1. É1-projection (S∘R = id sur un état vrai quelconque) -----------------


def test_e1_projection_sr_identite(verite_3):
    """Pour un état vrai QUELCONQUE (le brief illustre avec
    `chaine_verite(101, 2)[2]` ; « quelconque » n'impose pas n=2 -- on
    réutilise ici `verite_3[3]`, déjà calculé par la fixture module, pour
    éviter une histoire vérité redondante) : S = summarize_qt(etat, 128),
    R = regenerate_qt(S), S2 = summarize_qt(R, 128) -- S2 doit être
    bit-identique à S (topologie ET moyennes), propriété de projection : les
    invariants commis sont préservés EXACTEMENT."""
    etat = verite_3[3]
    S = summarize_qt(etat, 128)
    R = regenerate_qt(S)
    S2 = summarize_qt(R, 128)
    assert np.array_equal(S2.means, S.means)
    assert np.array_equal(S2.topology, S.topology)


# --- 2. É1-replay-ancrage (vs histoire gelée h_s101.npz) ---------------------


@pytest.mark.skipif(not _H_S101_PATH.exists(), reason="histoire gelée h_s101.npz absente")
def test_e1_replay_ancrage_vs_histoire_gelee():
    """`chaine_verite(101, 10)[10]` doit être bit-identique (np.array_equal)
    au checkpoint `s_L10` de `h_s101.npz` -- le replay des pulses depuis le
    seed (mêmes ancres B0/PARAMS que `scripts/run_arcA_histories.py`)
    reproduit l'histoire gelée committée. n=10 non réductible : c'est
    précisément la longueur du checkpoint gelé contre lequel on compare."""
    with np.load(_H_S101_PATH, allow_pickle=False) as data:
        s_l10_fige = np.array(data["s_L10"], dtype=np.float64, copy=True)
    verite = chaine_verite(101, 10)
    assert np.array_equal(verite[10], s_l10_fige)


# --- 3. ferm-chaîne (LE test d'intégration, contrôle §A13-4-1) ---------------


def test_ferm_chaine_controle_integration(verite_3):
    """Commit = copie exacte du champ complet (sans perte) -> delta_chi ==
    [0.0, 0.0, 0.0] EXACT (pas de tolérance) ET chaque readout moteur
    bit-identique à l'état vrai correspondant -- le contrôle-fermable ne doit
    JAMAIS diverger, quel que soit `k` (ici 4096, comme demandé par le brief ;
    `taille_commits` est indépendant de `k` pour ce bras)."""
    res = chaine_registre(101, dt=1, n_emissions=3, k=4096, arm="ferm", verite=verite_3)
    assert res["delta_chi"] == [0.0, 0.0, 0.0]
    assert res["taille_commits"] == [4096, 4096, 4096]
    assert len(res["readouts_moteur"]) == 3
    for i in range(3):
        assert np.array_equal(res["readouts_moteur"][i], res["etats_vrais"][i])


# --- 4. É3-déterminisme in-process -------------------------------------------


def test_e3_determinisme_in_process():
    """Deux appels IDENTIQUES de chaine_registre(101, dt=1, n_emissions=2,
    k=128, arm="v2") -> delta_chi identiques bit-à-bit, readouts identiques.
    `verite` précalculée UNE fois et partagée entre les deux appels
    (optimisation autorisée, physique bit-identique, motif déjà en usage dans
    `scripts/run_arcA_measure.py`) : n'affecte pas la propriété testée
    (identité stricte de DEUX appels à entrées identiques)."""
    verite_2 = chaine_verite(101, 2)
    res1 = chaine_registre(101, dt=1, n_emissions=2, k=128, arm="v2", verite=verite_2)
    res2 = chaine_registre(101, dt=1, n_emissions=2, k=128, arm="v2", verite=verite_2)
    assert res1["delta_chi"] == res2["delta_chi"]
    assert len(res1["readouts_moteur"]) == len(res2["readouts_moteur"]) == 2
    for i in range(2):
        assert np.array_equal(res1["readouts_moteur"][i], res2["readouts_moteur"][i])


# --- 5. É3-double-reconstruction cross-process -------------------------------

_SUBPROCESS_SCRIPT_REGISTRE = """
import numpy as np
from scripts.run_arcA_m2_registre import chaine_registre

res = chaine_registre(101, dt=1, n_emissions=2, k=128, arm="v2")
readouts = np.stack(res["readouts_moteur"])
np.save({out_path!r}, readouts)
"""


def test_e3_double_reconstruction_cross_process(tmp_path):
    """Deux process Python séparés exécutent chaine_registre(101, dt=1,
    n_emissions=2, k=128) et écrivent les readouts-moteur en .npy -- les deux
    fichiers doivent être bit-identiques (déterminisme réel entre process
    indépendants, pas un artefact de cache in-process)."""
    out1 = tmp_path / "readouts1.npy"
    out2 = tmp_path / "readouts2.npy"
    for out_path in (out1, out2):
        script = _SUBPROCESS_SCRIPT_REGISTRE.format(out_path=str(out_path))
        result = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    arr1 = np.load(out1)
    arr2 = np.load(out2)
    assert np.array_equal(arr1, arr2)


# --- 6. É3-rederivation -------------------------------------------------------


def test_e3_rederivation_bit_exacte_et_autocoherente(chaine_v2_3em, rederive_v2_3em):
    """rederiver_emissions(res["registre"], 101, 1, 3) -> chaque champ
    bit-identique à res["readouts_moteur"][i] ET summarize_qt(rederive[i],
    128) redonne res["registre"][i] (topologie + moyennes bit-identiques) --
    le registre est auto-cohérent."""
    res = chaine_v2_3em
    rederive = rederive_v2_3em
    assert len(rederive) == 3
    for i in range(3):
        assert np.array_equal(rederive[i], res["readouts_moteur"][i])
        resum = summarize_qt(rederive[i], 128)
        assert np.array_equal(resum.topology, res["registre"][i].topology)
        assert np.array_equal(resum.means, res["registre"][i].means)


# --- 7. Anti-fuite signature ---------------------------------------------------


def test_anti_fuite_signature_rederiver_emissions():
    sig = inspect.signature(rederiver_emissions)
    assert list(sig.parameters.keys()) == ["registre", "seed", "dt", "n_emissions", "arm"]


# --- 8. Anti-fuite décoy -------------------------------------------------------


def test_anti_fuite_decoy_mutation_etats_vrais(chaine_v2_3em):
    """Anti-fuite décoy : muter brutalement les états vrais retournés par
    `chaine_registre` (res["etats_vrais"][i][:] = 999.0), ré-appeler
    `rederiver_emissions` -> résultat inchangé bit-à-bit. Ce test refait SON
    PROPRE appel de `rederiver_emissions` (ne réutilise pas `rederive_v2_3em`)
    pour que la propriété soit vérifiée APRÈS la mutation, pas avant --
    `rederiver_emissions` n'a de toute façon AUCUN canal vers `etats_vrais`
    (cf. signature stricte, item 7) : la mutation ci-dessous du fixture
    `chaine_v2_3em` est sans risque pour les AUTRES tests du module, aucun
    d'eux ne lit `chaine_v2_3em["etats_vrais"]`."""
    res = chaine_v2_3em
    for etat in res["etats_vrais"]:
        etat[:] = 999.0
    rederive = rederiver_emissions(res["registre"], 101, 1, 3, arm="v2")
    assert len(rederive) == 3
    for i in range(3):
        assert np.array_equal(rederive[i], res["readouts_moteur"][i])


# --- 9. Composition possible (sanité, pas un verdict) ------------------------


def test_composition_erreur_possible_sanity(chaine_v2_3em):
    """delta_chi[0] == 0.0 exact (moteur et vérité bit-identiques jusqu'au
    premier commit) ET delta_chi[1] > 0.0 (le commit quantifie réellement --
    si 0, le harnais ne mesure rien)."""
    res = chaine_v2_3em
    assert res["delta_chi"][0] == 0.0
    assert res["delta_chi"][1] > 0.0


# --- 10. Garde budget ----------------------------------------------------------


def test_garde_budget_taille_commits_sous_k(chaine_v2_3em):
    res = chaine_v2_3em
    assert all(t <= res["k"] for t in res["taille_commits"])
