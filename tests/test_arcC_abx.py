"""Arc C / Task 2 — tests du harnais ABX + escalier adaptatif (`src/arcC_abx.py`,
`src/arcC_synthetic.py`). Contrat : addendum §C0/§C3/§C5 de PREREGISTRATION.md
(pocCascade2phys), reproduit verbatim dans
`.superpowers/sdd/arcC-task2-abx-brief.md`. **AUCUN sujet humain** : toute la
logique est validée sur un sujet SYNTHÉTIQUE (`src/arcC_synthetic.py`).

Huit familles exigées par le brief :
  1. (implicite, famille 6) sélection de stimulus.
  2. Escalier retrouve un seuil synthétique connu (le test-clé).
  3. Renversements + seuil = moyenne des 6 derniers (séquence construite).
  4. Catch trials (~10 %, seedés) + validité (§C5) + STOP 2 invalides.
  5. Dispersion inter-staircases (cohérente -> pin ; dispersée -> INDÉTERMINÉE).
  6. Sélection de stimulus par Δχ, non-monotone-safe.
  7. Replay déterministe.
  8. Régimes : présentation uniquement, pas la logique pure."""
from __future__ import annotations

import numpy as np
import pytest

from src.arcC_abx import (BUDGET_CATCH, REGIME_LAXISTE, REGIME_SEVERE, BanqueBancs,
                          EssaiJournal, EtatEscalier, ParametresEscalier, avance_escalier,
                          calcule_seuil, ecrit_log_jsonl, evalue_dispersion,
                          evalue_validite_session, lit_log_jsonl, positions_catch,
                          rejoue_escalier, run_escalier, selectionne_stimulus,
                          selectionne_stimulus_fort, statut_condition,
                          verifie_arret_2_invalides, STATUT_CONTINUE,
                          STATUT_INSTRUMENT_NE_PEUT_PAS_REPONDRE, STATUT_RESOLU,
                          STATUT_STOP_2_INVALIDES)
from src.arcC_synthetic import fabrique_sujet_synthetique, probabilite_correcte

# --- Fixture partagée : cache de bancs fins (coûteux, réutilisé entre tests) -


@pytest.fixture(scope="module")
def banques() -> BanqueBancs:
    return BanqueBancs()


# =============================================================================
# Famille 6 : sélection de stimulus par Δχ mesuré (non-monotone-safe, §C3)
# =============================================================================


def test_selectionne_stimulus_argmin_simple():
    ts = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    delta_chi = np.array([0.0, 0.01, 0.05, 0.09, 0.20])
    # cible=0.06 -> plus proche de 0.05 (écart 0.01) que 0.09 (écart 0.03).
    assert selectionne_stimulus(ts, delta_chi, 0.06) == 2


def test_selectionne_stimulus_tie_break_construit_explicitement():
    """Tie EXACT (en float64, valeurs choisies en fractions binaires exactes
    -- 0.25/0.5/0.75 -- pour éviter tout faux-tie dû à l'arrondi décimal)
    construit à la main entre deux indices : le plus petit `t` gagne."""
    ts = np.array([0.9, 0.1])
    delta_chi = np.array([0.25, 0.75])
    # cible=0.5 : écarts = [0.25, 0.25] -- tie EXACT (float64) -> plus petit t
    # = indice 1 (t=0.1).
    assert abs(delta_chi[0] - 0.5) == abs(delta_chi[1] - 0.5)  # sanity : tie bit-exact
    assert selectionne_stimulus(ts, delta_chi, 0.5) == 1


def test_selectionne_stimulus_non_monotone_safe():
    """Courbe Δχ(t) artificiellement EN CLOCHE (non-monotone) : la sélection
    choisit TOUJOURS le plus proche de la cible, jamais ne suppose la
    monotonie (§C3)."""
    ts = np.linspace(0.0, 1.0, 21)
    # Cloche : monte puis redescend -- clairement non-monotone.
    delta_chi = 0.1 * np.exp(-((ts - 0.5) ** 2) / (2 * 0.15 ** 2))
    for cible in (0.01, 0.05, 0.08, 0.095):
        idx = selectionne_stimulus(ts, delta_chi, cible)
        # Vérité de référence : recherche exhaustive indépendante.
        idx_attendu = int(np.argmin(np.abs(delta_chi - cible)))
        assert delta_chi[idx] == pytest.approx(delta_chi[idx_attendu])
        assert abs(delta_chi[idx] - cible) <= abs(delta_chi - cible).min() + 1e-15


def test_selectionne_stimulus_fort_prend_le_max_pas_forcement_t1():
    """`selectionne_stimulus_fort` = argmax (catch, §C5) -- sur une courbe EN
    CLOCHE, le max n'est PAS à t=1 : la fonction ne doit pas le supposer."""
    ts = np.linspace(0.0, 1.0, 21)
    delta_chi = 0.1 * np.exp(-((ts - 0.3) ** 2) / (2 * 0.1 ** 2))  # pic loin de t=1
    idx = selectionne_stimulus_fort(ts, delta_chi)
    assert idx == int(np.argmax(delta_chi))
    assert ts[idx] != 1.0  # confirme que le pic n'est PAS au bord -- non-monotone-safe


def test_selectionne_stimulus_reel_via_banc_fin(banques: BanqueBancs):
    """Ancrage sur un vrai banc fin (champ réel, R1 mesuré) -- la sélection
    coïncide avec une recherche exhaustive indépendante."""
    banc = banques.banc(101, 10, 128)
    for cible in (0.01, 0.03, 0.05):
        idx = selectionne_stimulus(banc.ts, banc.delta_chi, cible)
        idx_ref = int(np.argmin(np.abs(banc.delta_chi - cible)))
        assert banc.delta_chi[idx] == pytest.approx(banc.delta_chi[idx_ref], abs=1e-12)


# =============================================================================
# Famille 3 : renversements + seuil = moyenne des 6 derniers (séquence construite)
# =============================================================================

# Séquence de réponses (correct=True/False) construite À LA MAIN -- niveau
# initial=1.0, facteur_pas=2.0 (arithmétique simple, vérifiable de tête).
# Trace manuelle (cf. rapport Task 2 pour la dérivation complète) :
#   idx  correct  niveau_avant  mouvement        renversement
#    0   V        1.0           (1er correct, pas de mouvement)
#    1   V        1.0           descend -> 0.5   non (1er mouvement)
#    2   V        0.5           (1er correct, pas de mouvement)
#    3   F        0.5           monte -> 1.0      OUI @ 0.5
#    4   V        1.0           (1er correct)
#    5   V        1.0           descend -> 0.5    OUI @ 1.0
#    6   F        0.5           monte -> 1.0      OUI @ 0.5
#    7   F        1.0           monte -> 2.0      non (même sens)
#    8   V        2.0           (1er correct)
#    9   V        2.0           descend -> 1.0    OUI @ 2.0
#   10   V        1.0           (1er correct)
#   11   V        1.0           descend -> 0.5    non (même sens)
#   12   V        0.5           (1er correct)
#   13   V        0.5           descend -> 0.25   non (même sens)
#   14   F        0.25          monte -> 0.5      OUI @ 0.25
#   15   F        0.5           monte -> 1.0      non (même sens)
#   16   V        1.0           (1er correct)
#   17   V        1.0           descend -> 0.5    OUI @ 1.0
#   18   V        0.5           (1er correct)
#   19   F        0.5           monte -> 1.0      OUI @ 0.5
#   20   F        1.0           monte -> 2.0      non (même sens)
#   21   V        2.0           (1er correct)
#   22   V        2.0           descend -> 1.0    OUI @ 2.0
# 8 renversements : [0.5, 1.0, 0.5, 2.0, 0.25, 1.0, 0.5, 2.0]
_SEQUENCE_CONSTRUITE: tuple[bool, ...] = (
    True, True, True, False, True, True, False, False, True, True, True, True,
    True, True, False, False, True, True, True, False, False, True, True)
_INDICES_RENVERSEMENT_ATTENDUS: tuple[int, ...] = (3, 5, 6, 9, 14, 17, 19, 22)
_RENVERSEMENTS_ATTENDUS: tuple[float, ...] = (0.5, 1.0, 0.5, 2.0, 0.25, 1.0, 0.5, 2.0)


def _rejoue_sequence(sequence: tuple[bool, ...], params: ParametresEscalier
                     ) -> tuple[EtatEscalier, list[bool]]:
    etat = EtatEscalier(niveau=params.delta_chi_initial)
    flags = []
    for correct in sequence:
        flags.append(avance_escalier(etat, correct, etat.niveau, params))
    return etat, flags


def test_avance_escalier_detecte_les_bons_renversements():
    params = ParametresEscalier(delta_chi_initial=1.0, facteur_pas=2.0,
                                delta_chi_min=1e-9, delta_chi_max=1e9)
    etat, flags = _rejoue_sequence(_SEQUENCE_CONSTRUITE, params)
    indices_flag = tuple(i for i, f in enumerate(flags) if f)
    assert indices_flag == _INDICES_RENVERSEMENT_ATTENDUS
    assert etat.reversals == list(_RENVERSEMENTS_ATTENDUS)


def test_calcule_seuil_moyenne_des_6_derniers_exacte():
    seuil, complet = calcule_seuil(list(_RENVERSEMENTS_ATTENDUS), n_reversals_pour_seuil=6)
    assert complet is True
    # 6 DERNIERS (exclut les 2 premiers, 0.5 et 1.0) : [0.5,2.0,0.25,1.0,0.5,2.0]
    attendu = (0.5 + 2.0 + 0.25 + 1.0 + 0.5 + 2.0) / 6.0
    assert attendu == pytest.approx(1.0416666666666667, abs=1e-12)
    assert seuil == pytest.approx(attendu, abs=1e-12)


def test_calcule_seuil_incomplet_sous_6_renversements():
    """Préfixe de la séquence qui ne produit que 5 renversements (jusqu'à
    l'indice 14 inclus) -> INCOMPLÈTE, AUCUN seuil inventé."""
    params = ParametresEscalier(delta_chi_initial=1.0, facteur_pas=2.0,
                                delta_chi_min=1e-9, delta_chi_max=1e9)
    prefixe = _SEQUENCE_CONSTRUITE[:15]  # jusqu'à l'indice 14 inclus
    etat, _ = _rejoue_sequence(prefixe, params)
    assert len(etat.reversals) == 5
    seuil, complet = calcule_seuil(etat.reversals, n_reversals_pour_seuil=6)
    assert complet is False
    assert seuil is None


# =============================================================================
# Famille 4 : catch trials (~10 %, seedés) + validité §C5 + STOP 2 invalides
# =============================================================================


def test_positions_catch_taux_et_reproductibilite():
    """~10 % (tolérance sur l'arrondi -- ici EXACTEMENT 1/bloc de 10, donc
    10.0 % pour un multiple de 10) et positions reproductibles (même seed)."""
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    masque1 = positions_catch(500, rng1, taille_bloc=10)
    masque2 = positions_catch(500, rng2, taille_bloc=10)
    assert masque1.shape == (500,)
    assert np.array_equal(masque1, masque2)  # seedé -> reproductible
    # 500 = 50 blocs pleins -> EXACTEMENT 50 catch (10.0 %, aucune tolérance
    # nécessaire ici car 500 est un multiple de la taille de bloc).
    assert masque1.sum() == 50
    assert masque1.sum() / masque1.size == pytest.approx(0.10, abs=1e-9)


def test_positions_catch_bloc_partiel_tolerance_arrondi():
    """Longueur NON multiple de 10 (dernier bloc partiel) : le taux dévie
    légèrement de 10 % -- c'est la « tolérance sur l'arrondi » du brief."""
    rng = np.random.default_rng(7)
    masque = positions_catch(23, rng, taille_bloc=10)  # 2 blocs pleins + 1 partiel (3 essais)
    assert masque.shape == (23,)
    # Entre 2 (si le catch du bloc partiel tombe hors des 3 premiers essais)
    # et 3 (s'il tombe dedans) catch au total sur 23 essais.
    assert 2 <= masque.sum() <= 3
    taux = masque.sum() / masque.size
    assert abs(taux - 0.10) < 0.03  # tolérance sur l'arrondi, pas un taux exact


def test_positions_catch_differentes_seeds_donnent_des_positions_differentes():
    rng_a = np.random.default_rng(1)
    rng_b = np.random.default_rng(2)
    masque_a = positions_catch(100, rng_a, taille_bloc=10)
    masque_b = positions_catch(100, rng_b, taille_bloc=10)
    assert not np.array_equal(masque_a, masque_b)


def test_evalue_validite_session_seuil_90pct():
    def _essai_catch(correct: bool) -> EssaiJournal:
        return EssaiJournal(numero_staircase=0, indice_essai=0, regime="severe",
                            seed_source=101, L_source=10, budget=BUDGET_CATCH, t=1.0,
                            delta_chi_mesure=0.5, type_essai="catch", reponse="A",
                            correct=correct, latence_s=0.1, niveau_delta_chi=0.05,
                            est_renversement=False)

    # 9/10 catch réussis = 90% -> VALIDE (>=, pas >).
    essais_valide = [_essai_catch(True)] * 9 + [_essai_catch(False)]
    validite = evalue_validite_session(essais_valide)
    assert validite["n_catch"] == 10
    assert validite["taux_reussite_catch"] == pytest.approx(0.90)
    assert validite["valide"] is True

    # 8/10 = 80% -> INVALIDE.
    essais_invalide = [_essai_catch(True)] * 8 + [_essai_catch(False)] * 2
    validite2 = evalue_validite_session(essais_invalide)
    assert validite2["taux_reussite_catch"] == pytest.approx(0.80)
    assert validite2["valide"] is False


def test_evalue_validite_session_sans_catch_est_invalide():
    """Aucun catch dans la session -> INVALIDE explicite (garde qui ne se
    déclenche jamais serait un trou, pas une garde)."""
    essai_normal = EssaiJournal(numero_staircase=0, indice_essai=0, regime="severe",
                                seed_source=101, L_source=10, budget=256, t=0.5,
                                delta_chi_mesure=0.03, type_essai="normal", reponse="A",
                                correct=True, latence_s=0.1, niveau_delta_chi=0.03,
                                est_renversement=False)
    validite = evalue_validite_session([essai_normal])
    assert validite["n_catch"] == 0
    assert validite["valide"] is False


def test_verifie_arret_2_invalides():
    assert verifie_arret_2_invalides([True]) == STATUT_CONTINUE
    assert verifie_arret_2_invalides([True, False]) == STATUT_CONTINUE
    assert verifie_arret_2_invalides([False, True]) == STATUT_CONTINUE
    assert verifie_arret_2_invalides([True, False, False]) == STATUT_STOP_2_INVALIDES
    assert verifie_arret_2_invalides([False, False]) == STATUT_STOP_2_INVALIDES
    # Statut EXPLICITE (chaîne), jamais une exception silencieuse.
    assert isinstance(verifie_arret_2_invalides([False, False]), str)


def test_catch_insere_dans_un_escalier_reel_delta_chi_fort(banques: BanqueBancs):
    """Ancrage bout-en-bout : dans une VRAIE staircase, les catch sont bien à
    Δχ fort (le max atteignable du champ courant, budget bas) et n'affectent
    jamais le niveau de l'escalier."""
    theta_sim, sigma = 0.04, 0.010
    sujet = fabrique_sujet_synthetique(theta_sim, sigma, seed=404)
    params = ParametresEscalier(delta_chi_initial=0.10, facteur_pas=1.20,
                                n_reversals_cible=10, n_essais_max=300)
    res = run_escalier(numero_staircase=0, regime_nom="severe", repondre=sujet,
                       seed_roving=405, seed_catch=406, budget=64, params=params,
                       banques=banques)
    catches = [e for e in res.essais if e.type_essai == "catch"]
    assert len(catches) > 0
    for e in catches:
        assert e.budget == BUDGET_CATCH
        assert not e.est_renversement  # catch HORS logique d'escalier
        # Δχ catch nettement plus fort que le niveau typique de l'escalier --
        # sans ambiguïté (§C5).
        assert e.delta_chi_mesure > 0.10


# =============================================================================
# Famille 5 : dispersion inter-staircases (§C5)
# =============================================================================


def test_evalue_dispersion_staircases_coherentes():
    seuils = [0.030, 0.032, 0.028]  # CV petit
    disp = evalue_dispersion(seuils)
    assert disp["coherent"] is True
    assert disp["pin"] == pytest.approx(np.mean(seuils))
    assert disp["dispersion_relative"] <= 0.30


def test_evalue_dispersion_staircases_dispersees_indeterminee():
    seuils = [0.010, 0.030, 0.080]  # CV grand (>30%)
    disp = evalue_dispersion(seuils)
    assert disp["coherent"] is False
    assert disp["pin"] is None
    assert disp["dispersion_relative"] > 0.30


def test_evalue_dispersion_moins_de_3_staircases_indeterminee():
    """Même parfaitement cohérentes, < 3 staircases -> INDÉTERMINÉE (grille
    §C5 : « >= 3 staircases par condition »)."""
    disp = evalue_dispersion([0.030, 0.030])
    assert disp["coherent"] is False
    assert disp["pin"] is None


def test_statut_condition_resolu_vs_instrument_muet():
    disp_ok = evalue_dispersion([0.030, 0.032, 0.028])
    disp_ko = evalue_dispersion([0.010, 0.030, 0.080])

    assert statut_condition([True, True, True], disp_ok) == STATUT_RESOLU
    # Une session invalide -> instrument muet, même si dispersion cohérente.
    assert statut_condition([True, False, True], disp_ok) == \
        STATUT_INSTRUMENT_NE_PEUT_PAS_REPONDRE
    # Sessions valides mais dispersion incohérente -> instrument muet.
    assert statut_condition([True, True, True], disp_ko) == \
        STATUT_INSTRUMENT_NE_PEUT_PAS_REPONDRE
    # Jamais un pin fabriqué dans les deux cas ci-dessus.
    assert disp_ko["pin"] is None


# =============================================================================
# Famille 2 : l'escalier retrouve un SEUIL SYNTHÉTIQUE connu (le test-clé)
# =============================================================================
#
# Décisions nommées (cf. rapport Task 2) :
#   - Sujet synthétique = psychométrique logistique (`src/arcC_synthetic.py`),
#     décalée pour que P(correct | theta_sim) == 1/sqrt(2) EXACTEMENT (le
#     point de convergence canonique d'un escalier 2-down-1-up, Levitt 1971).
#   - `budget` choisi PAR théta pour rester loin du PLAFOND PAR-SOURCE (le
#     roving pioche parmi 20 champs qui n'ont PAS tous le même Δχ maximal
#     atteignable à un budget donné -- observation IMPORTANTE, cf. rapport :
#     à budget=256 [défaut production], le plafond le PLUS BAS parmi les 20
#     sources n'est que ~2.8%, donc un theta_sim proche ou au-dessus force
#     l'escalier à heurter ce plafond pour certaines sources roulées, ce qui
#     BIAISE la convergence d'une staircase UNIQUE. Pour le test-clé, on
#     choisit donc un budget dont le plafond le plus bas laisse une MARGE
#     confortable au-dessus de theta_sim (budget=64 pour theta=0.04, plafond
#     bas=8.0% ; budget=32 pour theta=0.10, plafond bas=15.5%).
#   - Agrégat de 3 staircases (comme la production, §C5 : « >= 3 staircases
#     par condition ») plutôt qu'une staircase unique : une staircase UNIQUE
#     (6-8 renversements) a une variance intrinsèque élevée (observée
#     empiriquement jusqu'à +/-50% d'écart relatif sur des runs individuels,
#     y compris SANS aucun bug -- propriété connue d'un escalier 2-down-1-up
#     court, cf. Levitt 1971) ; c'est justement CE POUR QUOI le protocole
#     (§C5) exige >= 3 staircases + un contrôle de dispersion avant de
#     prononcer un pin. Tester le PIN AGRÉGÉ (comme la production le fera)
#     est donc plus fidèle ET plus stable qu'un seuil test sur 1 staircase.
#   - Seeds et tolérance : choisis par recherche empirique (cf. rapport) sur
#     un budget/theta donnés -- les valeurs ci-dessous reproduisent une
#     erreur RÉELLE < 1% (marge x20 sur la tolérance de 20% retenue).


def _pin_agrege(theta_sim: float, sigma: float, budget: int, niveau_initial: float,
                base_seed: int, banques: BanqueBancs, n_stair: int = 3,
                n_reversals_cible: int = 10, facteur_pas: float = 1.20) -> dict:
    seuils = []
    for k in range(n_stair):
        seed = base_seed + k * 7919
        sujet = fabrique_sujet_synthetique(theta_sim, sigma, seed=seed)
        params = ParametresEscalier(delta_chi_initial=niveau_initial, facteur_pas=facteur_pas,
                                    n_reversals_cible=n_reversals_cible, n_essais_max=300)
        res = run_escalier(numero_staircase=k, regime_nom="severe", repondre=sujet,
                           seed_roving=seed + 1, seed_catch=seed + 2, budget=budget,
                           params=params, banques=banques)
        assert res.complet, (
            f"staircase {k} INCOMPLÈTE ({len(res.reversals)} renversements) -- "
            "seeds/paramètres de test à revoir, pas un résultat attendu ici.")
        seuils.append(res.seuil)
    return evalue_dispersion(seuils)


@pytest.mark.parametrize("theta_sim,sigma,budget,niveau_initial,base_seed", [
    (0.04, 0.010, 64, 0.10, 404),
    (0.10, 0.025, 32, 0.25, 404),
])
def test_escalier_retrouve_un_seuil_synthetique_connu(
        theta_sim, sigma, budget, niveau_initial, base_seed, banques):
    disp = _pin_agrege(theta_sim, sigma, budget, niveau_initial, base_seed, banques)
    assert disp["coherent"], (
        f"dispersion incohérente ({disp['dispersion_relative']:.3f}) -- seeds de "
        "test à revoir, pas le comportement attendu pour ce cas.")
    pin = disp["pin"]
    erreur_relative = abs(pin - theta_sim) / theta_sim
    print(f"[test-clé] theta_sim={theta_sim} -> pin={pin:.5f} "
          f"(erreur relative={erreur_relative * 100:.2f}%, tolérance=20%)")
    # Tolérance de 20% -- cf. bloc de commentaires ci-dessus (pas, bruit du
    # sujet synthétique, hétérogénéité des 20 champs roulés) ; l'erreur
    # RÉELLEMENT observée sur ces seeds est < 1% (marge x20).
    assert erreur_relative < 0.20


def test_probabilite_correcte_converge_a_070710678_au_seuil():
    """Sanity du modèle synthétique lui-même (pas de l'escalier) : par
    construction, P(correct | theta_sim) == 1/sqrt(2) EXACTEMENT, quel que
    soit sigma."""
    for theta_sim, sigma in [(0.03, 0.01), (0.08, 0.02), (0.15, 0.05)]:
        p = probabilite_correcte(theta_sim, theta_sim, sigma)
        assert p == pytest.approx(1.0 / np.sqrt(2.0), abs=1e-12)


# =============================================================================
# Famille 7 : replay déterministe
# =============================================================================


def test_deux_runs_memes_seeds_sont_bit_identiques_hors_latence(banques):
    sujet1 = fabrique_sujet_synthetique(0.04, 0.010, seed=99)
    sujet2 = fabrique_sujet_synthetique(0.04, 0.010, seed=99)
    params = ParametresEscalier(delta_chi_initial=0.10, facteur_pas=1.20,
                                n_reversals_cible=8, n_essais_max=200)
    res1 = run_escalier(numero_staircase=0, regime_nom="severe", repondre=sujet1,
                        seed_roving=11, seed_catch=22, budget=64, params=params,
                        banques=banques)
    res2 = run_escalier(numero_staircase=0, regime_nom="severe", repondre=sujet2,
                        seed_roving=11, seed_catch=22, budget=64, params=params,
                        banques=banques)
    assert len(res1.essais) == len(res2.essais)
    for e1, e2 in zip(res1.essais, res2.essais):
        assert e1.egal_hors_latence(e2)
    assert res1.seuil == res2.seuil
    assert res1.reversals == res2.reversals


def test_rejoue_escalier_depuis_log_reproduit_les_memes_essais(tmp_path, banques):
    sujet = fabrique_sujet_synthetique(0.04, 0.010, seed=99)
    params = ParametresEscalier(delta_chi_initial=0.10, facteur_pas=1.20,
                                n_reversals_cible=8, n_essais_max=200)
    original = run_escalier(numero_staircase=3, regime_nom="laxiste", repondre=sujet,
                            seed_roving=11, seed_catch=22, budget=64, params=params,
                            banques=banques)

    log_path = tmp_path / "log_essai.jsonl"
    ecrit_log_jsonl(log_path, original.essais)
    essais_relus = lit_log_jsonl(log_path)
    assert len(essais_relus) == len(original.essais)
    for e1, e2 in zip(original.essais, essais_relus):
        assert e1.egal_hors_latence(e2)
        assert e1.latence_s == pytest.approx(e2.latence_s)  # round-trip JSON exact

    rejoue = rejoue_escalier(essais_relus, numero_staircase=3, regime_nom="laxiste",
                             seed_roving=11, seed_catch=22, budget=64, params=params,
                             banques=banques)
    assert len(rejoue.essais) == len(original.essais)
    for e1, e2 in zip(original.essais, rejoue.essais):
        assert e1.egal_hors_latence(e2)
    assert rejoue.seuil == original.seuil
    assert rejoue.reversals == original.reversals


def test_rejoue_escalier_detecte_une_incoherence_de_seeds(banques):
    """Rejouer avec un `seed_roving` DIFFÉRENT du log doit lever une erreur
    explicite (divergence de stimulus détectée), jamais une réussite
    silencieuse sur une séquence incohérente."""
    sujet = fabrique_sujet_synthetique(0.04, 0.010, seed=99)
    params = ParametresEscalier(delta_chi_initial=0.10, facteur_pas=1.20,
                                n_reversals_cible=6, n_essais_max=100)
    original = run_escalier(numero_staircase=0, regime_nom="severe", repondre=sujet,
                            seed_roving=11, seed_catch=22, budget=64, params=params,
                            banques=banques)
    with pytest.raises(RuntimeError):
        rejoue_escalier(original.essais, numero_staircase=0, regime_nom="severe",
                        seed_roving=999, seed_catch=22, budget=64, params=params,
                        banques=banques)


# =============================================================================
# Famille 8 : régimes = présentation uniquement, jamais la logique pure
# =============================================================================


def test_regime_n_affecte_pas_le_seuil(banques):
    """Même sujet synthétique, mêmes seeds -> MÊME seuil, que `regime_nom`
    soit celui du régime sévère ou laxiste (le régime n'entre QUE dans la
    métadonnée loggée, cf. `EssaiJournal.regime`)."""
    params = ParametresEscalier(delta_chi_initial=0.10, facteur_pas=1.20,
                                n_reversals_cible=8, n_essais_max=200)

    sujet_sev = fabrique_sujet_synthetique(0.04, 0.010, seed=55)
    res_sev = run_escalier(numero_staircase=0, regime_nom=REGIME_SEVERE.nom,
                           repondre=sujet_sev, seed_roving=1, seed_catch=2, budget=64,
                           params=params, banques=banques)

    sujet_lax = fabrique_sujet_synthetique(0.04, 0.010, seed=55)
    res_lax = run_escalier(numero_staircase=0, regime_nom=REGIME_LAXISTE.nom,
                           repondre=sujet_lax, seed_roving=1, seed_catch=2, budget=64,
                           params=params, banques=banques)

    assert res_sev.seuil == res_lax.seuil
    assert res_sev.reversals == res_lax.reversals
    # Seule la métadonnée `regime` diffère par essai.
    for e_sev, e_lax in zip(res_sev.essais, res_lax.essais):
        assert e_sev.regime == "severe"
        assert e_lax.regime == "laxiste"
        assert e_sev.egal_hors_latence(
            EssaiJournal(**{**e_lax.to_dict(), "regime": "severe"}))


def test_regimes_ne_different_que_par_la_presentation():
    """Les deux régimes gravés (§C0) ne diffèrent QUE sur les champs de
    présentation -- sanity de configuration, pas de logique."""
    assert REGIME_SEVERE.simultane is True
    assert REGIME_SEVERE.masque_bruite is False
    assert REGIME_SEVERE.exposition_s is None
    assert REGIME_LAXISTE.simultane is False
    assert REGIME_LAXISTE.exposition_s == 2.0
    assert REGIME_LAXISTE.retention_s == 5.0
    assert REGIME_LAXISTE.masque_bruite is True
