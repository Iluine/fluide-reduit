"""Arc C / Task 3 — tests de l'orchestrateur de campagne
(`scripts/run_arcC_orchestration.py`), addendum §C8 de PREREGISTRATION.md
(pocCascade2phys, commits `c494d50`→`d7c95e3`), reproduit verbatim dans
`.superpowers/sdd/arcC-task3-runner-brief.md`. **AUCUN sujet humain** : tout
est validé sur sujet SYNTHÉTIQUE (`src/arcC_synthetic.py`), sur les VRAIS
champs sources (npz commités manche 1, via `src/arcC_abx.py`).

Familles de tests (brief, §Tests 1/2/3/5) :
  1. Exclusion nommée par-source (D-2) : ancre < 0.16 -> exclue et NOMMÉE
     (Δχ au manifeste) ; ancre >= 0.16 -> incluse ; >= 10 exclusions -> STOP
     (statut explicite, pas exception).
  2. Catch en réserve forte (D-3) : catch tire le bout fort (Δχ élevé),
     jamais near-threshold ; taux < 90% -> INVALIDE (pas de recalibration) ;
     `--catch-sources-fortes` restreint le pool (testé OFF par défaut).
  3. Orchestration : >= 3 staircases x 2 régimes, seeds distinctes
     consignées, 2 sessions invalides consécutives -> STOP.
  5. Contingence géométrie-plafond (D-4) : `--geometrie plafond` calcule la
     bonne taille d'affichage (`plafond_texture`) ET tourne avec la MÊME
     ancre/famille/sources que `pic-csf`.
  9. Provenance (§C10, correctif) : `sujet`/`commit_harnais`/`date_session`
     stampés au manifeste par le HARNAIS (jamais par le post-traitement) ;
     `obtient_commit_harnais` renvoie le SHA git réel, ou `"inconnu"` (loggué)
     hors dépôt."""
from __future__ import annotations

import subprocess
from datetime import datetime

import numpy as np
import pytest

from src.arcC_abx import (BUDGET_CATCH, PAIRES_SOURCES, REGIME_LAXISTE, REGIME_SEVERE,
                          STATUT_CONTINUE, STATUT_STOP_2_INVALIDES, BanqueBancs,
                          ParametresEscalier, run_escalier)
from src.arcC_synthetic import fabrique_sujet_synthetique
from scripts.run_arcC_orchestration import (ANCRE_BUDGET, FRACTION_SOURCES_FORTES,
                                            HAUT_JND_PLAUSIBLE, N_EXCLUSIONS_STOP,
                                            N_STAIRCASES, ROOT, SEUIL_EXCLUSION,
                                            STATUT_STOP_TROP_EXCLUES,
                                            calcule_taille_affichage_px,
                                            construit_manifeste_exclusions, derive_seeds,
                                            mesure_ancre_source, obtient_commit_harnais,
                                            orchestre_campagne, selectionne_sources_fortes)


# =============================================================================
# Paramètres gravés exposés -- sanity anti-magie-enfouie
# =============================================================================


def test_parametres_graves_exposes():
    assert ANCRE_BUDGET == 32
    assert HAUT_JND_PLAUSIBLE == 0.08
    assert SEUIL_EXCLUSION == pytest.approx(0.16)
    assert N_STAIRCASES == 3
    # collision D-3 RÉSOLUE nommément : même valeur numérique que BUDGET_CATCH
    # (la séparation catch/near-threshold vient de la SÉLECTION -- bout fort,
    # `selectionne_stimulus_fort` -- pas d'un budget distinct, cf. docstring
    # module de l'orchestrateur).
    assert BUDGET_CATCH == ANCRE_BUDGET


# =============================================================================
# Famille 1 : exclusion nommée par-source (D-2), avant présentation
# =============================================================================


def test_construit_manifeste_exclusions_sur_les_20_sources_reelles():
    """Sur les 20 (seed, L) réels (manche 1), à budget=32 (ANCRE_BUDGET) et
    seuil gravé 0.16 : EXACTEMENT 1 source exclue (seed=103, L=10, ancre
    connue ~15.5%, cf. rapport Task 2 §3.3 -- la pire des 20). Valeur
    RE-MESURÉE ici (pas relue d'un catalogue), donc ancrée sur le générateur
    réel (Task 1), pas sur un artefact figé."""
    manifeste = construit_manifeste_exclusions()
    assert manifeste["n_sources"] == 20
    assert manifeste["budget"] == ANCRE_BUDGET
    assert manifeste["seuil_exclusion"] == pytest.approx(SEUIL_EXCLUSION)
    assert manifeste["statut"] == STATUT_CONTINUE
    assert manifeste["n_exclues"] == 1
    (seed_exclu, L_exclu), = manifeste["sources_exclues"]
    assert (seed_exclu, L_exclu) == (103, 10)
    entree = next(e for e in manifeste["entrees"] if e["seed"] == 103 and e["L"] == 10)
    assert entree["exclu"] is True
    assert entree["delta_chi_ancre"] == pytest.approx(0.15511273731102393, abs=1e-9)
    # NOMMÉE au manifeste avec son Δχ -- jamais retirée silencieusement.
    assert "delta_chi_ancre" in entree and "seed" in entree and "L" in entree
    assert len(manifeste["sources_incluses"]) == 19
    assert (103, 10) not in manifeste["sources_incluses"]


def test_mesure_ancre_source_coherente_avec_le_generateur_task1():
    """`mesure_ancre_source` = Δχ(t=1) MESURÉ par le générateur Task 1
    (`courbe_delta_chi`), pas une réimplémentation -- vérifié en recalculant
    indépendamment via `courbe_delta_chi(..., ts=[1.0])`."""
    from scripts.run_arcA_measure import _load_history
    from src.arcC_stimuli import courbe_delta_chi

    seed, L = 101, 10
    ancre = mesure_ancre_source(seed, L, ANCRE_BUDGET)
    s_true = np.asarray(_load_history(seed)[f"s_L{L}"], dtype=np.float64)
    attendu = float(courbe_delta_chi(s_true, ANCRE_BUDGET, ts=np.array([1.0]))[0])
    assert ancre == pytest.approx(attendu, abs=1e-12)


def test_exclusion_construite_avec_seuil_artificiel_declenche_le_stop_10():
    """Cas construit (brief, test #1) : un seuil ARTIFICIELLEMENT élevé
    (0.5, PAS le seuil gravé 0.16 -- ceci ne teste QUE la mécanique du
    statut STOP, pas une nouvelle valeur de contrat) sur les 20 sources
    réelles fait tomber >= 10 sources sous le seuil -> statut STOP explicite
    (jamais une exception)."""
    manifeste = construit_manifeste_exclusions(seuil_exclusion=0.5)
    assert manifeste["n_exclues"] >= N_EXCLUSIONS_STOP
    assert manifeste["statut"] == STATUT_STOP_TROP_EXCLUES
    assert isinstance(manifeste["statut"], str)  # statut, pas une exception


def test_source_juste_au_seuil_est_incluse_pas_exclue():
    """`exclu` ssi ancre STRICTEMENT < seuil (§C8 D-2 : `< 2xHAUT_JND`) --
    une source pile au seuil est INCLUSE."""
    manifeste = construit_manifeste_exclusions(
        sources=((101, 10),), seuil_exclusion=0.0)  # seuil 0 -> jamais exclu
    assert manifeste["n_exclues"] == 0
    assert manifeste["sources_incluses"] == [(101, 10)]


# =============================================================================
# Famille 2 : catch en réserve forte (D-3) + validité §C5 inchangée
# =============================================================================


def test_catch_tire_le_bout_fort_jamais_near_threshold():
    """Ancrage bout-en-bout (comme Task 2, mais à l'ancre budget-32 D-1) :
    dans une VRAIE staircase à budget=ANCRE_BUDGET, les catch restent au
    bout fort du banc -- nettement au-dessus du niveau near-threshold de
    l'escalier -- même quand le staircase normal ET le catch partagent
    EXACTEMENT le même budget (collision D-3 résolue par la SÉLECTION, pas
    le budget)."""
    banques = BanqueBancs()
    theta_sim, sigma = 0.06, 0.015
    sujet = fabrique_sujet_synthetique(theta_sim, sigma, seed=707)
    params = ParametresEscalier(delta_chi_initial=0.10, facteur_pas=1.20,
                                n_reversals_cible=10, n_essais_max=300)
    manifeste = construit_manifeste_exclusions()
    sources_incluses = tuple(manifeste["sources_incluses"])
    res = run_escalier(numero_staircase=0, regime_nom="severe", repondre=sujet,
                       seed_roving=708, seed_catch=709, budget=ANCRE_BUDGET,
                       params=params, banques=banques, sources=sources_incluses)
    catches = [e for e in res.essais if e.type_essai == "catch"]
    assert len(catches) > 0
    for e in catches:
        assert e.budget == ANCRE_BUDGET
        assert not e.est_renversement
        # Bout fort, SANS ambiguïté : le catch est l'argmax du banc de sa
        # source -- toute source INCLUSE a une ancre (Δχ à t=1) >= 0.16
        # (SEUIL_EXCLUSION), et l'argmax sur toute la courbe est >= son
        # point t=1 -- donc chaque catch reste nettement au-dessus du niveau
        # near-JND (HAUT_JND_PLAUSIBLE=0.08), quel que soit le niveau réel de
        # l'escalier à ce moment (jamais un niveau balayé near-threshold).
        assert e.delta_chi_mesure >= SEUIL_EXCLUSION
        assert e.delta_chi_mesure > HAUT_JND_PLAUSIBLE


def test_validite_inchangee_taux_catch_sous_90pct_est_invalide():
    """§C5 INCHANGÉ (pas de recalibration) : un taux de réussite catch < 90%
    reste INVALIDE même à l'ancre budget-32."""
    from src.arcC_abx import EssaiJournal, evalue_validite_session

    def _catch(correct: bool) -> EssaiJournal:
        return EssaiJournal(numero_staircase=0, indice_essai=0, regime="severe",
                            seed_source=101, L_source=10, budget=ANCRE_BUDGET, t=1.0,
                            delta_chi_mesure=0.60, type_essai="catch", reponse="A",
                            correct=correct, latence_s=0.1, niveau_delta_chi=0.05,
                            est_renversement=False)

    essais = [_catch(True)] * 8 + [_catch(False)] * 2  # 80% < 90%
    validite = evalue_validite_session(essais)
    assert validite["valide"] is False
    assert validite["taux_reussite_catch"] == pytest.approx(0.80)


def test_selectionne_sources_fortes_restreint_au_top_de_l_ancre():
    """`selectionne_sources_fortes` (option D-3, OFF par défaut) : ne
    retient QUE les sources à ancre la plus forte (fraction nommée,
    `FRACTION_SOURCES_FORTES`) parmi les INCLUSES."""
    entrees = [
        dict(seed=1, L=10, delta_chi_ancre=0.20, exclu=False),
        dict(seed=2, L=10, delta_chi_ancre=0.60, exclu=False),
        dict(seed=3, L=10, delta_chi_ancre=0.10, exclu=True),   # exclue -- jamais retenue
        dict(seed=4, L=10, delta_chi_ancre=0.40, exclu=False),
        dict(seed=5, L=10, delta_chi_ancre=0.50, exclu=False),
    ]
    fortes = selectionne_sources_fortes(entrees, fraction=0.5)
    # 4 incluses -> moitié = 2 -> les 2 plus fortes : seed=2 (0.60), seed=5 (0.50).
    assert set(fortes) == {(2, 10), (5, 10)}
    assert (3, 10) not in fortes  # exclue, jamais dans le pool catch-fortes


def test_catch_sources_fortes_off_par_defaut_dans_lorchestration(tmp_path):
    """`--catch-sources-fortes` OFF par défaut : le pool catch == le pool
    normal (pas de restriction), vérifié via `orchestre_campagne`."""
    fabrique = lambda k, regime_nom, seed_sujet: (  # noqa: E731
        fabrique_sujet_synthetique(0.05, 0.015, seed=seed_sujet), lambda: None)
    manifeste = orchestre_campagne(
        base_seed=1000, n_staircases=1, ppd=40.0, fabrique_repondre=fabrique,
        out_dir=tmp_path / "logs")
    assert manifeste["parametres_graves"]["catch_sources_fortes"] is False
    assert manifeste["parametres_graves"]["fraction_sources_fortes"] is None


# =============================================================================
# Famille 3 : orchestration -- >= 3 staircases x 2 régimes, seeds distinctes
# =============================================================================


def test_derive_seeds_produit_des_triplets_distincts():
    vus = set()
    for regime_idx in (0, 1):
        for k in range(3):
            seeds = derive_seeds(base_seed=42, regime_idx=regime_idx, k=k)
            triplet = (seeds["seed_roving"], seeds["seed_catch"], seeds["seed_sujet"])
            assert triplet not in vus
            vus.add(triplet)
            assert len({seeds["seed_roving"], seeds["seed_catch"], seeds["seed_sujet"]}) == 3


def test_orchestre_campagne_lance_n_staircases_x_2_regimes(tmp_path):
    fabrique = lambda k, regime_nom, seed_sujet: (  # noqa: E731
        fabrique_sujet_synthetique(0.05, 0.015, seed=seed_sujet), lambda: None)
    manifeste = orchestre_campagne(
        base_seed=2024, n_staircases=3, ppd=40.0, fabrique_repondre=fabrique,
        out_dir=tmp_path / "logs")
    assert manifeste["statut_global"] == STATUT_CONTINUE
    assert set(manifeste["regimes"].keys()) == {REGIME_SEVERE.nom, REGIME_LAXISTE.nom}
    for regime_nom, regime_data in manifeste["regimes"].items():
        assert len(regime_data["sessions"]) == 3
        seeds_roving = {s["seed_roving"] for s in regime_data["sessions"]}
        assert len(seeds_roving) == 3  # seeds distinctes consignées
        for s in regime_data["sessions"]:
            assert s["regime"] == regime_nom


def test_orchestre_campagne_stop_2_sessions_invalides(tmp_path):
    """Sujet qui échoue systématiquement les catch (répond toujours faux) ->
    2 sessions invalides consécutives -> STOP, statut explicite, pas
    d'exception, et l'orchestration NE LANCE PAS de 3e staircase pour ce
    régime."""
    from src.arcC_abx import autre_reponse

    def repondre_toujours_faux(essai):
        return autre_reponse(essai.reponse_correcte)

    fabrique = lambda k, regime_nom, seed_sujet: (repondre_toujours_faux, lambda: None)  # noqa: E731
    params = ParametresEscalier(delta_chi_initial=0.10, facteur_pas=1.20,
                                n_reversals_cible=4, n_essais_max=40)
    manifeste = orchestre_campagne(
        base_seed=99, n_staircases=3, ppd=40.0, fabrique_repondre=fabrique, params=params,
        out_dir=tmp_path / "logs")
    for regime_nom, regime_data in manifeste["regimes"].items():
        assert regime_data["statut_orchestration"] == STATUT_STOP_2_INVALIDES
        # STOP après 2 invalides consécutives -> pas les 3 staircases lancées.
        assert len(regime_data["sessions"]) < 3
        assert all(not s["validite"]["valide"] for s in regime_data["sessions"][-2:])


# =============================================================================
# Famille 5 : contingence géométrie-plafond (D-4), même ancre/famille/sources
# =============================================================================


def test_calcule_taille_affichage_px_differe_selon_geometrie():
    ppd = 40.0
    taille_pic_csf = calcule_taille_affichage_px("pic-csf", ppd)
    taille_plafond = calcule_taille_affichage_px("plafond", ppd)
    assert taille_pic_csf != taille_plafond
    assert taille_pic_csf > 0 and taille_plafond > 0


def test_calcule_taille_affichage_px_rejette_geometrie_inconnue():
    with pytest.raises(ValueError):
        calcule_taille_affichage_px("inconnue", 40.0)


def test_geometrie_plafond_meme_ancre_famille_sources_que_pic_csf(tmp_path):
    """§C8 D-4 : seule la géométrie (taille d'affichage) change entre les
    deux modes -- l'ancre budget, la famille de sources incluses/exclues,
    les seeds dérivées sont IDENTIQUES (sinon l'écart mesuré entre les deux
    staircases serait contaminé par un écart d'ancre)."""
    fabrique = lambda k, regime_nom, seed_sujet: (  # noqa: E731
        fabrique_sujet_synthetique(0.05, 0.015, seed=seed_sujet), lambda: None)
    m_pic = orchestre_campagne(base_seed=55, n_staircases=1, ppd=40.0,
                              geometrie="pic-csf", fabrique_repondre=fabrique,
                              out_dir=tmp_path / "logs_pic")
    m_plafond = orchestre_campagne(base_seed=55, n_staircases=1, ppd=40.0,
                                   geometrie="plafond", fabrique_repondre=fabrique,
                                   out_dir=tmp_path / "logs_plafond")
    assert m_pic["parametres_graves"]["ancre_budget"] == m_plafond["parametres_graves"]["ancre_budget"]
    assert m_pic["exclusions"]["sources_incluses"] == m_plafond["exclusions"]["sources_incluses"]
    assert m_pic["exclusions"]["sources_exclues"] == m_plafond["exclusions"]["sources_exclues"]
    for regime_nom in (REGIME_SEVERE.nom, REGIME_LAXISTE.nom):
        seeds_pic = [(s["seed_roving"], s["seed_catch"]) for s in m_pic["regimes"][regime_nom]["sessions"]]
        seeds_plafond = [(s["seed_roving"], s["seed_catch"])
                        for s in m_plafond["regimes"][regime_nom]["sessions"]]
        assert seeds_pic == seeds_plafond
    # SEULE la config d'affichage diffère.
    assert m_pic["config_affichage"]["taille_domaine_px"] != \
        m_plafond["config_affichage"]["taille_domaine_px"]
    assert m_pic["config_affichage"]["geometrie"] == "pic-csf"
    assert m_plafond["config_affichage"]["geometrie"] == "plafond"


def test_geometrie_defaut_est_pic_csf(tmp_path):
    fabrique = lambda k, regime_nom, seed_sujet: (  # noqa: E731
        fabrique_sujet_synthetique(0.05, 0.015, seed=seed_sujet), lambda: None)
    manifeste = orchestre_campagne(base_seed=7, n_staircases=1, ppd=40.0,
                                   fabrique_repondre=fabrique, out_dir=tmp_path / "logs")
    assert manifeste["parametres_graves"]["geometrie"] == "pic-csf"


# =============================================================================
# Famille 6 : conditions de validité §C9 (CORRECTIF avant session, pièces 2 & 3)
# =============================================================================


def test_conditions_validite_luminosite_et_conditions_dans_le_manifeste(tmp_path):
    fabrique = lambda k, regime_nom, seed_sujet: (  # noqa: E731
        fabrique_sujet_synthetique(0.05, 0.015, seed=seed_sujet), lambda: None)
    manifeste = orchestre_campagne(
        base_seed=11, n_staircases=1, ppd=40.0, fabrique_repondre=fabrique,
        out_dir=tmp_path / "logs", luminosite="120 nits", conditions="bureau, jour, stable",
        long_ref_px=1920.0, long_ref_mm=310.0, distance_mm=600.0)
    cv = manifeste["conditions_validite"]
    assert cv["luminosite"] == "120 nits"
    assert cv["conditions"] == "bureau, jour, stable"
    assert cv["calibration"]["ppd"] == pytest.approx(40.0)
    assert cv["calibration"]["long_ref_px"] == pytest.approx(1920.0)
    assert cv["calibration"]["long_ref_mm"] == pytest.approx(310.0)
    assert cv["calibration"]["distance_mm"] == pytest.approx(600.0)


def test_conditions_validite_absentes_par_defaut_reste_none(tmp_path):
    """Non-régression : aucun appelant existant ne passe luminosite/conditions
    -- le manifeste reste construit, avec `None` explicite (jamais fabriqué)."""
    fabrique = lambda k, regime_nom, seed_sujet: (  # noqa: E731
        fabrique_sujet_synthetique(0.05, 0.015, seed=seed_sujet), lambda: None)
    manifeste = orchestre_campagne(base_seed=12, n_staircases=1, ppd=40.0,
                                   fabrique_repondre=fabrique, out_dir=tmp_path / "logs")
    cv = manifeste["conditions_validite"]
    assert cv["luminosite"] is None
    assert cv["conditions"] is None
    assert cv["calibration"]["long_ref_px"] is None


def test_conditions_validite_presente_meme_sur_statut_stop_trop_exclues(tmp_path):
    fabrique = lambda k, regime_nom, seed_sujet: (  # noqa: E731
        fabrique_sujet_synthetique(0.05, 0.015, seed=seed_sujet), lambda: None)
    manifeste = orchestre_campagne(
        base_seed=13, n_staircases=1, ppd=40.0, fabrique_repondre=fabrique,
        out_dir=tmp_path / "logs", seuil_exclusion=0.5,  # seuil artificiel -> STOP (famille 1)
        luminosite="OSD 75%", conditions="salon, soir, lampe stable")
    assert manifeste["statut_global"] == STATUT_STOP_TROP_EXCLUES
    assert manifeste["conditions_validite"]["luminosite"] == "OSD 75%"


# =============================================================================
# Famille 9 : provenance (§C10 correctif) -- sujet/commit_harnais/date_session
# stampés au manifeste par le HARNAIS, jamais par le post-traitement (Task 3).
# =============================================================================


def test_provenance_stampee_par_defaut_synthetique(tmp_path):
    fabrique = lambda k, regime_nom, seed_sujet: (  # noqa: E731
        fabrique_sujet_synthetique(0.05, 0.015, seed=seed_sujet), lambda: None)
    manifeste = orchestre_campagne(
        base_seed=21, n_staircases=1, ppd=40.0, fabrique_repondre=fabrique,
        out_dir=tmp_path / "logs")
    assert manifeste["sujet"] == "synthetique"  # défaut, non-régression
    datetime.fromisoformat(manifeste["date_session"])  # ISO 8601, ne lève pas
    assert isinstance(manifeste["commit_harnais"], str) and len(manifeste["commit_harnais"]) > 0


def test_provenance_sujet_humain_stampee(tmp_path):
    """`sujet` stampé reflète le paramètre du HARNAIS -- pas déduit d'un
    quelconque champ du `fabrique_repondre` fourni (ici un sujet synthétique
    substitué, comme partout ailleurs dans ce fichier, pour rester testable)."""
    fabrique = lambda k, regime_nom, seed_sujet: (  # noqa: E731
        fabrique_sujet_synthetique(0.05, 0.015, seed=seed_sujet), lambda: None)
    manifeste = orchestre_campagne(
        base_seed=22, n_staircases=1, ppd=40.0, fabrique_repondre=fabrique, sujet="humain",
        out_dir=tmp_path / "logs")
    assert manifeste["sujet"] == "humain"


def test_provenance_stampee_meme_sur_statut_stop_trop_exclues(tmp_path):
    fabrique = lambda k, regime_nom, seed_sujet: (  # noqa: E731
        fabrique_sujet_synthetique(0.05, 0.015, seed=seed_sujet), lambda: None)
    manifeste = orchestre_campagne(
        base_seed=23, n_staircases=1, ppd=40.0, fabrique_repondre=fabrique,
        out_dir=tmp_path / "logs", seuil_exclusion=0.5)  # -> STOP (famille 1)
    assert manifeste["statut_global"] == STATUT_STOP_TROP_EXCLUES
    assert manifeste["sujet"] == "synthetique"
    datetime.fromisoformat(manifeste["date_session"])


def test_obtient_commit_harnais_correspond_a_git_rev_parse_head():
    attendu = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.strip()
    assert obtient_commit_harnais(ROOT) == attendu


def test_obtient_commit_harnais_hors_depot_renvoie_inconnu_et_loggue(tmp_path, capsys):
    """Hors dépôt git -> `"inconnu"`, mais l'échec est LOGGUÉ (imprimé), pas
    avalé silencieusement (§C10, correctif provenance)."""
    resultat = obtient_commit_harnais(tmp_path)
    assert resultat == "inconnu"
    sortie = capsys.readouterr().out
    assert "AVERTISSEMENT" in sortie
    assert "commit" in sortie.lower()


def test_valide_conditions_requises_leve_erreur_si_humain_sans_luminosite_ni_conditions():
    from scripts.run_arcC_orchestration import valide_conditions_requises_si_humain

    class _MarqueurErreur(Exception):
        pass

    def _erreur(message: str) -> None:
        raise _MarqueurErreur(message)

    with pytest.raises(_MarqueurErreur):
        valide_conditions_requises_si_humain("humain", None, None, _erreur)
    with pytest.raises(_MarqueurErreur):
        valide_conditions_requises_si_humain("humain", "120 nits", None, _erreur)
    with pytest.raises(_MarqueurErreur):
        valide_conditions_requises_si_humain("humain", None, "bureau", _erreur)


def test_valide_conditions_requises_ok_si_humain_avec_les_deux():
    from scripts.run_arcC_orchestration import valide_conditions_requises_si_humain

    def _erreur_jamais_appelee(message: str) -> None:
        raise AssertionError(f"erreur() ne doit PAS être appelé ici : {message}")

    valide_conditions_requises_si_humain("humain", "120 nits", "bureau", _erreur_jamais_appelee)


def test_valide_conditions_requises_ok_si_synthetique_sans_rien():
    """§C9 : OPTIONNELLES pour --sujet synthetique (non pertinentes, aucun
    humain devant l'écran) -- `erreur` n'est JAMAIS appelé."""
    from scripts.run_arcC_orchestration import valide_conditions_requises_si_humain

    def _erreur_jamais_appelee(message: str) -> None:
        raise AssertionError(f"erreur() ne doit PAS être appelé ici : {message}")

    valide_conditions_requises_si_humain("synthetique", None, None, _erreur_jamais_appelee)


def test_main_leve_systemexit_si_sujet_humain_sans_luminosite_ni_conditions(monkeypatch):
    """Intégration CLI : `main()` échoue AVANT tout import matplotlib (le
    chemin --sujet humain n'importe matplotlib que dans
    `_fabrique_repondre_humain`, lazy, jamais atteint ici -- la validation
    est placée juste après `parser.parse_args()`)."""
    import sys

    from scripts.run_arcC_orchestration import main

    argv = ["run_arcC_orchestration.py",
           "--base-seed", "1", "--sujet", "humain",
           "--long-ref-px", "1920", "--long-ref-mm", "310", "--distance-mm", "600"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        main()


# =============================================================================
# Coquille interactive (`scripts/run_arcC_session.py`) -- garde-fou §C9
# pièces 2 & 3, jumeau du test ci-dessus. Placé ici (à côté de son jumeau
# orchestrateur, choix nommé) plutôt que dans un fichier dédié.
# =============================================================================


def test_run_arcC_session_leve_systemexit_si_session_humaine_sans_luminosite_ni_conditions(
        monkeypatch):
    """Intégration CLI (coquille interactive) : `main()` échoue AVANT toute
    création de figure matplotlib -- `parser.error` (§C9 pièces 2 & 3) est
    placé juste après `parser.parse_args()`, avant `_cree_figure`/
    `_construit_repondre_humain` (jamais atteints ici). Toute session SANS
    `--replay` est une session humaine (cette coquille n'a pas de mode
    synthétique, cf. docstring module) -- --luminosite/--conditions sont
    donc REQUISES. Ce garde-fou n'a AUCUNE régression jusqu'ici : `main()`
    importe `matplotlib.pyplot` au niveau module (headless-safe, aucun
    écran requis) mais ne crée de figure que plus bas, après ce garde."""
    import sys

    from scripts.run_arcC_session import main

    argv = ["run_arcC_session.py",
           "--regime", "severe",
           "--numero-staircase", "1",
           "--seed-roving", "1",
           "--seed-catch", "2",
           "--long-ref-px", "1920", "--long-ref-mm", "310", "--distance-mm", "600"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        main()


def test_run_arcC_session_replay_sans_luminosite_ni_conditions_ne_leve_pas_systemexit(
        monkeypatch, tmp_path):
    """Cas symétrique POSITIF : `--replay` (aucune capture humaine, rien à
    consigner) n'exige PAS --luminosite/--conditions -- le garde §C9 ne
    s'applique qu'en session live (`args.replay is None`). Ce test reste
    focalisé SUR CE GARDE : le chemin --replay échoue ensuite pour une
    AUTRE raison (log inexistant -- `FileNotFoundError` dans
    `lit_log_jsonl`), jamais atteinte si le garde luminosité/conditions
    avait (à tort) déclenché un SystemExit en premier."""
    import sys

    from scripts.run_arcC_session import main

    argv = ["run_arcC_session.py",
           "--regime", "severe",
           "--numero-staircase", "1",
           "--seed-roving", "1",
           "--seed-catch", "2",
           "--long-ref-px", "1920", "--long-ref-mm", "310", "--distance-mm", "600",
           "--replay", str(tmp_path / "inexistant.jsonl")]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(FileNotFoundError):
        main()
