"""Tests unitaires PERSISTÉS de `k_star_seed`/`k_star_groupe`/`gate_controles`
(scripts/run_arcA_verdict.py, Task 6).

Correctif post-revue (finding Important) : le rapport Task 6 affirmait un
« test synthétique ad hoc » vérifiant la clause d'escalade 2xJND et
l'équivalence n=1 du gate -- ce test n'existait nulle part dans le dépôt
(preuve en prose seulement), et `gate_controles` lisait `par_seed` au lieu
d'appeler `k_star_groupe`. Ce fichier ferme les deux volets : (i) l'ordre
81->273->1041->∞, (ii) la clause d'escalade 2xJND RÉELLEMENT déclenchée,
(iii) « aucun niveau ne satisfait -> ∞ », (iv) l'équivalence gate n=1
(`k_star_groupe` sur groupe singleton == `k_star_seed`), y compris pour une
seed qui, dans un groupe plus large, causerait une escalade chez d'autres.

Motif d'import verbatim, déjà utilisé entre scripts arcA (cf. tests/test_run_v2.py) :
`sys.path.insert(ROOT)` puis `from scripts.run_arcA_verdict import ...`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_arcA_verdict import (ATTENDU_FERM, ATTENDU_SHUF, JND_LIST,
                                      LEVELS, L_LIST, L_LIST_CONTROL,
                                      MESSAGE_INDETERMINE_CAPACITE, SEEDS,
                                      gate_controles, k_star_groupe, k_star_seed,
                                      verdict_A3_global_avec_capacite,
                                      verdict_A3_jnd_avec_capacite)

JND = 0.03  # JND de test, dans la plage réelle {2,3,4,5}% -- valeur non spéciale.


def _row(niveaux_sous_jnd: tuple[int, ...]) -> np.ndarray:
    """Ligne bool alignée sur LEVELS=(1,2,3) : True aux niveaux listés (les
    autres False) -- format attendu par `k_star_seed`/`k_star_groupe`."""
    row = np.zeros(len(LEVELS), dtype=bool)
    for level in niveaux_sous_jnd:
        row[LEVELS.index(level)] = True
    return row


# --- (i) ordre 81 -> 273 -> 1041 (jamais un ordre arbitraire de LEVELS) ------


def test_k_star_seed_ordre_croissant_81_273_1041():
    """`k_star_seed` retient la PLUS PETITE taille sous-JND dans l'ordre
    81->273->1041, jamais la première True dans l'ordre naturel de LEVELS=(1,2,3)
    (qui listerait 1041 avant 81)."""
    assert k_star_seed(_row((3,))) == 81.0
    assert k_star_seed(_row((2,))) == 273.0
    assert k_star_seed(_row((1,))) == 1041.0
    # ℓ=3 (81) ET ℓ=1 (1041) sous-JND -> doit retenir 81, pas 1041.
    assert k_star_seed(_row((3, 1))) == 81.0
    # ℓ=2 (273) ET ℓ=1 (1041) sous-JND, PAS ℓ=3 -> doit retenir 273, pas 1041.
    assert k_star_seed(_row((2, 1))) == 273.0


# --- (iii) aucun niveau ne satisfait -> ∞ ------------------------------------


def test_k_star_seed_aucun_niveau_sous_jnd_est_infini():
    assert k_star_seed(_row(())) == float("inf")


def test_k_star_groupe_mediane_deja_infinie_ne_declenche_aucune_escalade():
    """Si la médiane des k*(seed) du groupe est déjà ∞ (majorité jamais
    sous-JND), `k_star_groupe` retourne ∞ SANS entrer dans la boucle
    d'escalade (`audit["escalades"]` reste vide -- pas de calcul inventé)."""
    n = 5
    sous_jnd = np.zeros((n, len(LEVELS)), dtype=bool)   # personne jamais sous-JND
    ma1 = np.zeros((n, len(LEVELS)))
    ma2 = np.zeros((n, len(LEVELS)))
    k, audit = k_star_groupe(sous_jnd, ma1, ma2, JND)
    assert k == float("inf")
    assert audit["mediane_initiale"] == float("inf")
    assert audit["escalades"] == []
    assert audit["k_final"] == float("inf")


# --- (ii) clause d'escalade 2xJND, RÉELLEMENT déclenchée ---------------------


def test_k_star_groupe_escalade_declenchee_par_une_seed_polluante():
    """5 seeds sous-JND à ℓ=3 (médiane=81) ; UNE seed (la première) a
    max(M-A1,M-A2) = 2.5*JND > 2*JND À CE NIVEAU -> la clause déclenche
    l'escalade vers le niveau supérieur (273) ; aucune seed ne dépasse 2xJND à
    273 -> k* s'arrête à 273 (escalade d'UN SEUL niveau, pas jusqu'à ∞)."""
    n = 5
    sous_jnd = np.zeros((n, len(LEVELS)), dtype=bool)
    ma1 = np.zeros((n, len(LEVELS)))
    ma2 = np.zeros((n, len(LEVELS)))
    i3, i2 = LEVELS.index(3), LEVELS.index(2)
    for i in range(n):
        sous_jnd[i, i3] = True   # toutes sous-JND à 81 -> médiane initiale = 81
        sous_jnd[i, i2] = True   # et à 273 (pour que k* puisse s'y arrêter)
        ma1[i, i3] = 0.5 * JND   # valeurs propres, sous JND ET sous 2xJND à 81...
        ma1[i, i2] = 0.1 * JND   # ...et bien sous JND à 273
    ma1[0, i3] = 2.5 * JND       # seed polluante : dépasse 2xJND à 81 -> déclenche

    k, audit = k_star_groupe(sous_jnd, ma1, ma2, JND)

    assert audit["mediane_initiale"] == 81.0
    assert k == 273.0
    assert audit["k_final"] == 273.0
    assert len(audit["escalades"]) == 1
    assert audit["escalades"][0]["niveau_teste"] == 81.0
    assert audit["escalades"][0]["n_seeds_offenders"] == 1


def test_k_star_groupe_escalade_totale_si_toutes_les_seeds_polluent_partout():
    """(iii) via `k_star_groupe`, cas dégénéré : la médiane initiale est finie
    (81) mais CHAQUE niveau testé (81, 273, 1041) est en violation de la
    clause 2xJND pour toutes les seeds -> la boucle épuise les 3 niveaux et
    retourne ∞, sans exception ni boucle infinie."""
    n = 5
    sous_jnd = np.ones((n, len(LEVELS)), dtype=bool)
    ma1 = np.full((n, len(LEVELS)), 2.5 * JND)   # viole 2xJND À TOUS les niveaux
    ma2 = np.zeros((n, len(LEVELS)))
    k, audit = k_star_groupe(sous_jnd, ma1, ma2, JND)
    assert k == float("inf")
    assert audit["mediane_initiale"] == 81.0
    assert audit["k_final"] == float("inf")
    assert [e["niveau_teste"] for e in audit["escalades"]] == [81.0, 273.0, 1041.0]


# --- (iv) équivalence gate n=1 : k_star_groupe(singleton) == k_star_seed ----


def test_equivalence_groupe_singleton_egale_k_star_seed_cas_simples():
    """Preuve du docstring de `k_star_groupe` VÉRIFIÉE PAR EXÉCUTION (pas
    seulement en prose) : pour un groupe réduit à 1 seed, `k_star_groupe` ==
    `k_star_seed` de cette même seed, sur plusieurs cas (chaque niveau, et
    ∞) -- valeurs sous-JND respectant l'invariant réel (max < JND, jamais
    proche de 2xJND par construction)."""
    for niveaux_true in [(3,), (2,), (1,), (3, 1), (2, 1), ()]:
        row = _row(niveaux_true)
        attendu = k_star_seed(row)
        ma1_grp = np.full((1, len(LEVELS)), 0.1 * JND)
        ma2_grp = np.zeros((1, len(LEVELS)))
        obtenu, _audit = k_star_groupe(row[np.newaxis, :], ma1_grp, ma2_grp, JND)
        assert obtenu == attendu, f"niveaux_true={niveaux_true}"


def test_equivalence_groupe_singleton_egale_k_star_seed_cas_proche_de_2xjnd():
    """Cas critique visé par le finding de revue : une seed sous-JND À SON
    PROPRE niveau (ℓ=3, 81) avec une valeur brute PROCHE de JND (0.9*JND,
    donc mécaniquement < 2*JND -- l'invariant qui rend la clause un no-op),
    réduite à un groupe singleton, NE DOIT JAMAIS s'auto-déclencher : c'est
    exactement la seed dont la valeur, dans un groupe à 5 seeds, resterait
    sous 2xJND et ne polluerait donc personne (contraste avec le test
    d'escalade ci-dessus, où la seed polluante viole délibérément
    l'invariant à 2.5*JND pour exercer la clause)."""
    row = _row((3, 2))
    ma1_grp = np.array([[0.0, 0.1 * JND, 0.9 * JND]])   # ordre LEVELS=(1,2,3)
    ma2_grp = np.zeros((1, len(LEVELS)))
    attendu = k_star_seed(row)
    assert attendu == 81.0
    obtenu, audit = k_star_groupe(row[np.newaxis, :], ma1_grp, ma2_grp, JND)
    assert obtenu == attendu
    assert audit["escalades"] == []   # no-op mécanique : jamais d'escalade à n=1


# --- Régression du finding : gate_controles appelle k_star_groupe LITTÉRALEMENT --


def test_gate_controles_conforme_sur_donnees_synthetiques_coherentes():
    """`gate_controles` doit être câblé sur `k_star_groupe` (n=1), pas sur une
    lecture directe de `par_seed` -- vérifié en montant un jeu synthétique
    minimal aux axes attendus (nL=len(L_LIST), nSeed=5, nLevel=3, nJND=4) :
    ferm sous-JND à ℓ=3 partout (k*=81=attendu) ; shuf jamais sous-JND
    (k*=∞=attendu) -> CONFORME, zéro violation."""
    nL, nSeed, nLevel, nJND = len(L_LIST), len(SEEDS), len(LEVELS), len(JND_LIST)
    i3 = LEVELS.index(3)

    sous_jnd_ferm = np.zeros((nL, nSeed, nLevel, nJND), dtype=bool)
    ma1_ferm = np.zeros((nL, nSeed, nLevel))
    ma2_ferm = np.zeros((nL, nSeed, nLevel))
    for L in L_LIST_CONTROL:
        iL = L_LIST.index(L)
        sous_jnd_ferm[iL, :, i3, :] = True
        ma1_ferm[iL, :, i3] = 0.1 * min(JND_LIST)

    sous_jnd_shuf = np.zeros((nL, nSeed, nLevel, nJND), dtype=bool)   # jamais sous-JND
    ma1_shuf = np.zeros((nL, nSeed, nLevel))
    ma2_shuf = np.zeros((nL, nSeed, nLevel))

    statut, violations = gate_controles(sous_jnd_ferm, ma1_ferm, ma2_ferm,
                                        sous_jnd_shuf, ma1_shuf, ma2_shuf)
    assert statut == "CONFORME"
    assert violations == []


def test_gate_controles_detecte_une_violation_ponctuelle():
    """Une seule cellule ferm cassée (seed 0, L=10, premier JND, plus jamais
    sous-JND nulle part) -> VIOLATION détectée EXACTEMENT sur cette cellule,
    avec les bons champs (bras/L/seed/jnd/k_star_obtenu/k_star_attendu)."""
    nL, nSeed, nLevel, nJND = len(L_LIST), len(SEEDS), len(LEVELS), len(JND_LIST)
    i3 = LEVELS.index(3)

    sous_jnd_ferm = np.zeros((nL, nSeed, nLevel, nJND), dtype=bool)
    ma1_ferm = np.zeros((nL, nSeed, nLevel))
    ma2_ferm = np.zeros((nL, nSeed, nLevel))
    for L in L_LIST_CONTROL:
        iL = L_LIST.index(L)
        sous_jnd_ferm[iL, :, i3, :] = True
        ma1_ferm[iL, :, i3] = 0.1 * min(JND_LIST)

    sous_jnd_shuf = np.zeros((nL, nSeed, nLevel, nJND), dtype=bool)
    ma1_shuf = np.zeros((nL, nSeed, nLevel))
    ma2_shuf = np.zeros((nL, nSeed, nLevel))

    iL10 = L_LIST.index(10)
    sous_jnd_ferm[iL10, 0, :, 0] = False   # seed 0 : plus jamais sous-JND (k*=∞)

    statut, violations = gate_controles(sous_jnd_ferm, ma1_ferm, ma2_ferm,
                                        sous_jnd_shuf, ma1_shuf, ma2_shuf)
    assert statut == "VIOLATION"
    assert len(violations) == 1
    v = violations[0]
    assert v["bras"] == "ferm"
    assert v["L"] == 10
    assert v["seed"] == SEEDS[0]
    assert v["jnd"] == JND_LIST[0]
    assert v["k_star_obtenu"] == float("inf")
    assert v["k_star_attendu"] == ATTENDU_FERM
    assert ATTENDU_SHUF == float("inf")   # sanity de la constante utilisée par le bras shuf


# --- Clause 1 (gravée 9bcb09a) : label INDETERMINE_CAPACITE ------------------

INF = float("inf")
# IC contenant 0 (compatible PASS si k fini sous cap) -- pour vérifier que la
# clause capacité est PRIORITAIRE sur la lecture pente/IC quand k*=∞ partout.
_IC_LO, _IC_HI = -1.0, 1.0


def test_verdict_jnd_indetermine_capacite_si_k_infini_a_tous_les_L():
    """Clause 1 : k*(L ; JND) = ∞ pour TOUS les L de la grille -> le verdict de
    cette cellule JND est INDETERMINE_CAPACITE (pas l'INDETERMINE générique)."""
    k_par_L = {L: INF for L in L_LIST}
    verdict = verdict_A3_jnd_avec_capacite(k_par_L, INF, INF, _IC_LO, _IC_HI)
    assert verdict == "INDETERMINE_CAPACITE"


def test_verdict_jnd_pas_capacite_si_un_L_est_fini():
    """Un seul L fini (ici L=10 -> 81) suffit à NE PAS déclencher la clause
    capacité : le verdict retombe sur la lecture mécanique §A3 existante
    (ici k*(40)=∞ -> INDETERMINE générique, comportement INCHANGÉ)."""
    k_par_L = {L: INF for L in L_LIST}
    k_par_L[10] = 81.0
    verdict = verdict_A3_jnd_avec_capacite(k_par_L, INF, INF, _IC_LO, _IC_HI)
    assert verdict == "INDETERMINE"


def test_verdict_jnd_k_finis_partout_semantique_inchangee():
    """AUCUN autre changement de sémantique : avec des k* finis partout, la
    cellule JND rend exactement ce que rendait la lecture §A3 (ici PASS :
    pente nulle, IC ∋ 0, k*(80)=81 <= cap)."""
    k_par_L = {L: 81.0 for L in L_LIST}
    verdict = verdict_A3_jnd_avec_capacite(k_par_L, 81.0, 81.0, _IC_LO, _IC_HI)
    assert verdict == "PASS"


def test_verdict_global_indetermine_capacite_si_4_sur_4():
    """Clause 1, verdict global : les 4 JND INDETERMINE_CAPACITE -> global
    INDETERMINE_CAPACITE (même message mécanique)."""
    verdicts = {jnd: "INDETERMINE_CAPACITE" for jnd in JND_LIST}
    assert verdict_A3_global_avec_capacite(verdicts) == "INDETERMINE_CAPACITE"


def test_verdict_global_cas_mixte_ne_produit_pas_le_label():
    """Cas mixte exigé par la spec : ∞ à un JND (INDETERMINE_CAPACITE), fini à
    un autre (ici PASS) -> le global n'est PAS INDETERMINE_CAPACITE ; la règle
    globale INCHANGÉE s'applique (non stable sur la plage -> INDETERMINE)."""
    verdicts = {jnd: "INDETERMINE_CAPACITE" for jnd in JND_LIST}
    verdicts[JND_LIST[-1]] = "PASS"
    verdict = verdict_A3_global_avec_capacite(verdicts)
    assert verdict != "INDETERMINE_CAPACITE"
    assert verdict == "INDETERMINE"


def test_verdict_global_regle_inchangee_hors_capacite():
    """Sans aucun INDETERMINE_CAPACITE, la règle globale est INCHANGÉE :
    stable sur toute la plage -> la valeur ; sinon INDETERMINE."""
    assert verdict_A3_global_avec_capacite(
        {jnd: "PASS" for jnd in JND_LIST}) == "PASS"
    assert verdict_A3_global_avec_capacite(
        {jnd: ("PASS" if i else "FAIL") for i, jnd in enumerate(JND_LIST)}
    ) == "INDETERMINE"


def test_message_indetermine_capacite_verbatim_grave():
    """Le message mécanique doit être VERBATIM celui gravé (clause 1,
    arbitrage 9bcb09a / brief volet 2)."""
    assert MESSAGE_INDETERMINE_CAPACITE == (
        "famille insuffisante à ce JND ; N'EST PAS le mur (le mur = croissance "
        "avec l'histoire) ; ne sélectionne pas la cellule 2/3 de §A0 ; fork "
        "famille-vs-manche-2 à remonter")
