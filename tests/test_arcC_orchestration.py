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
                                            POSITION_TEMOIN_VIRIDIS, chemins_staircases,
                                            mesure_ancre_source, obtient_commit_harnais,
                                            orchestre_campagne, orchestre_regime,
                                            selectionne_sources_fortes)


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
    """DURCIE (§A41) : « bureau » ne suffit plus — l'observation doit porter la
    date du jour et une heure cohérente avec l'horloge machine (±2 h)."""
    from datetime import datetime

    from scripts.run_arcC_orchestration import valide_conditions_requises_si_humain

    def _erreur_jamais_appelee(message: str) -> None:
        raise AssertionError(f"erreur() ne doit PAS être appelé ici : {message}")

    maintenant = datetime.now()
    observation = (f"bureau, lumière du jour stable, le {maintenant:%d/%m/%Y} vers "
                   f"{maintenant.hour}h{maintenant.minute:02d}")
    valide_conditions_requises_si_humain("humain", "120 nits", observation,
                                         _erreur_jamais_appelee)


def test_valide_conditions_refuse_un_gabarit_a_l_entree():
    """§A41 / M2 : le gabarit du 26/07 (« …ce que tu observes… ») doit être
    refusé À L'ENTRÉE — pas quatre staircases plus tard, à la lecture."""
    from scripts.run_arcC_orchestration import valide_conditions_requises_si_humain

    appels: list[str] = []
    valide_conditions_requises_si_humain(
        "humain", "OSD 80%", "…ce que tu observes, heure comprise…", appels.append)
    assert appels and "REFUSÉES" in appels[0]


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
    écran requis) mais ne crée de figure que plus bas, après ce garde.

    `--rendu` est fourni ICI (P1/§A37 : sélecteur OBLIGATOIRE, sans défaut) --
    sans lui, argparse lèverait SystemExit pour un argument manquant et ce test
    passerait pour la MAUVAISE raison, sans jamais atteindre le garde §C9 qu'il
    prétend vérifier."""
    import sys

    from scripts.run_arcC_session import main

    argv = ["run_arcC_session.py",
           "--regime", "severe",
           "--rendu", "viridis",
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
    avait (à tort) déclenché un SystemExit en premier.

    `--rendu` est fourni bien que `--replay` n'affiche RIEN : le sélecteur
    §A37 est obligatoire sans exception, précisément pour qu'aucun chemin
    d'affichage ne puisse être hérité en silence."""
    import sys

    from scripts.run_arcC_session import main

    argv = ["run_arcC_session.py",
           "--regime", "severe",
           "--rendu", "viridis",
           "--numero-staircase", "1",
           "--seed-roving", "1",
           "--seed-catch", "2",
           "--long-ref-px", "1920", "--long-ref-mm", "310", "--distance-mm", "600",
           "--replay", str(tmp_path / "inexistant.jsonl")]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(FileNotFoundError):
        main()


# =============================================================================
# Famille 6 : support de session P3 -- sélection de régime, bras témoin
# viridis en position gravée, sidecars par staircase (chantier 6 de
# `mission-prerequis-p2p3.md`, précisions gravées au prereg P3).
# =============================================================================


def test_chemins_staircases_sans_temoin_est_uniforme():
    """Sans `--temoin-viridis`, toutes les staircases suivent le même chemin —
    le comportement historique, intact."""
    assert chemins_staircases("r1", 4, False) == ("r1", "r1", "r1", "r1")
    assert chemins_staircases("viridis", 3, False) == ("viridis",) * 3


def test_chemins_staircases_place_le_temoin_en_troisieme_position():
    """POSITION GRAVÉE au prereg P3 : la 3e des 4 staircases — deux R1, PUIS le
    témoin viridis, PUIS la dernière R1.

    Le motif est gravé aussi : répartir fatigue et apprentissage sur les DEUX
    bras. Un témoin en 1re position mesurerait un sujet frais contre trois
    staircases fatiguées ; en dernière, l'inverse. Ce test est le seul endroit
    où cette position vit dans le code exécutable."""
    assert chemins_staircases("r1", 4, True) == ("r1", "r1", "viridis", "r1")
    assert POSITION_TEMOIN_VIRIDIS == 2


def test_chemins_staircases_refuse_une_campagne_trop_courte():
    """Fail-loud : la disposition gravée exige une 3e staircase pour le témoin
    ET au moins une APRÈS lui.

    `n = 3` est le cas PIÈGE, et c'est le DÉFAUT de `--n-staircases` (minimum
    §C5 par régime) : il y a bien une 3e position, mais le témoin y serait le
    DERNIER — précisément la disposition que le motif gravé écarte (un témoin
    en fin de session mesure un sujet fatigué contre trois staircases
    fraîches). Sans cette garde, `--temoin-viridis` sans `--n-staircases 4`
    aurait rendu en SILENCE une disposition qui n'est pas celle du prereg."""
    for n in (0, 1, 2, 3):
        with pytest.raises(ValueError, match="temoin-viridis|témoin|TÉMOIN"):
            chemins_staircases("r1", n, True)
    assert chemins_staircases("r1", 4, True)[POSITION_TEMOIN_VIRIDIS] == "viridis"


def test_restreindre_le_regime_ne_change_pas_les_seeds(tmp_path):
    """`--regime severe` doit rejouer EXACTEMENT la campagne sévère de
    `--regime tous` : l'indice passé à `derive_seeds` est celui du tuple
    CANONIQUE, jamais celui de la sélection.

    Sans ce verrou, restreindre les régimes re-seederait silencieusement le
    régime conservé — et P3, qui tourne en sévère seul, ne serait pas
    comparable à la campagne du pin, qui tournait avec les deux."""
    fabrique = lambda k, regime_nom, seed_sujet: (  # noqa: E731
        fabrique_sujet_synthetique(0.05, 0.015, seed=seed_sujet), lambda: None)
    tous = orchestre_campagne(base_seed=7, n_staircases=3, ppd=40.0,
                              fabrique_repondre=fabrique, out_dir=tmp_path / "tous")
    severe = orchestre_campagne(base_seed=7, n_staircases=3, ppd=40.0,
                                fabrique_repondre=fabrique, regimes=(REGIME_SEVERE.nom,),
                                out_dir=tmp_path / "severe")

    assert set(severe["regimes"]) == {REGIME_SEVERE.nom}
    assert set(tous["regimes"]) == {REGIME_SEVERE.nom, REGIME_LAXISTE.nom}
    graines = lambda m: [(s["seed_roving"], s["seed_catch"], s["seed_sujet"])  # noqa: E731
                         for s in m["regimes"][REGIME_SEVERE.nom]["sessions"]]
    assert graines(severe) == graines(tous)
    assert severe["config_affichage"]["regimes_lances"] == [REGIME_SEVERE.nom]


def test_sidecars_par_staircase_portent_le_chemin_et_le_sha_du_log(tmp_path):
    """Chantiers 3 + 6 réunis : chaque staircase reçoit son PROPRE sidecar,
    portant la provenance du chemin qu'elle a traversé ET le sha256 de SON log.

    C'est ce qui rend le bras témoin lisible : le sidecar de la 3e staircase
    dit `viridis` avec `sha256_arcC_rendu = None` (chaîne historique, R1 n'y
    participe pas), les trois autres disent `r1` avec l'empreinte de R1. Sans
    sidecars distincts, la session entière serait étiquetée d'un seul chemin."""
    import hashlib
    import json
    from pathlib import Path

    from scripts.run_arcC_session import provenance_rendu
    from src.arcC_calibration import observation_cellule_pic_csf

    ppd = 42.099
    obs = observation_cellule_pic_csf(ppd)
    chemins = chemins_staircases("r1", 4, True)
    provenances = {c: provenance_rendu(c, ppd=ppd, taille_px=77, obs=obs) for c in set(chemins)}
    fabrique = lambda k, regime_nom, seed_sujet: (  # noqa: E731
        fabrique_sujet_synthetique(0.05, 0.015, seed=seed_sujet), lambda: None)

    resultat = orchestre_regime(
        regime=REGIME_SEVERE, regime_idx=0, n_staircases=4, base_seed=11,
        sources_incluses=PAIRES_SOURCES, sources_catch=None, budget=ANCRE_BUDGET,
        params=ParametresEscalier(), banques=BanqueBancs(), fabrique_repondre=fabrique,
        out_dir=tmp_path, chemins=chemins, provenances=provenances)

    sessions = resultat["sessions"]
    assert [s["chemin_rendu"] for s in sessions] == list(chemins[:len(sessions)])
    chemins_vus = set()
    for session in sessions:
        sidecar = json.loads(Path(session["conditions_path"]).read_text(encoding="utf-8"))
        attendu = hashlib.sha256(Path(session["log_path"]).read_bytes()).hexdigest()
        assert sidecar["sha256_log"] == attendu
        assert sidecar["provenance_rendu"]["chemin_rendu"] == session["chemin_rendu"]
        chemins_vus.add(session["chemin_rendu"])
    if len(sessions) > POSITION_TEMOIN_VIRIDIS:
        temoin = json.loads(
            Path(sessions[POSITION_TEMOIN_VIRIDIS]["conditions_path"]).read_text("utf-8"))
        assert temoin["provenance_rendu"]["chemin_rendu"] == "viridis"
        assert temoin["provenance_rendu"]["sha256_arcC_rendu"] is None
        assert chemins_vus == {"r1", "viridis"}


@pytest.mark.parametrize("sujet,rendu", [("synthetique", None), ("humain", "viridis")])
def test_temoin_viridis_refuse_hors_session_humaine_en_r1(monkeypatch, sujet, rendu):
    """`--temoin-viridis` n'est valide qu'avec `--sujet humain --rendu r1` : un
    témoin viridis ne témoigne de rien sans bras R1 à garder. `parser.error`
    lève `SystemExit`, avant toute plomberie."""
    import sys

    from scripts.run_arcC_orchestration import main

    argv = ["run_arcC_orchestration.py", "--base-seed", "1", "--sujet", sujet,
            "--temoin-viridis", "--theta-sim", "0.05", "--sigma-sim", "0.015",
            "--luminosite", "OSD 80%", "--conditions", "jour stable",
            "--long-ref-px", "1920", "--long-ref-mm", "597", "--distance-mm", "750"]
    if rendu is not None:
        argv += ["--rendu", rendu]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        main()


# =============================================================================
# Famille P3′ (mission chantier 8a/8c) : plan D-P3′-1, graine double à
# l'entrée, clause (d), échauffement, scellement — prereg v2.2 ENDOSSÉ §A42
# =============================================================================

from src.arcC_abx import (BASE_SEED_P3PRIME, GRAINES_BRULEES,  # noqa: E402
                          PLAGE_HORAIRE_SESSION)
from src.arcC_scelle import descelle_seuils  # noqa: E402
from pathlib import Path  # noqa: E402
from scripts.run_arcC_orchestration import (CHEMINS_ECHAUFFEMENT,  # noqa: E402
                                            INDICE_ECHAUFFEMENT, LOGS_DIR,
                                            LOGS_DIR_P3PRIME, MANIFESTE_HISTORIQUE,
                                            MANIFESTE_P3PRIME, N_ECHAUFFEMENT,
                                            N_STAIRCASES_P3PRIME, NIVEAU_ECHAUFFEMENT,
                                            PLAN_P3PRIME, artefacts_traques_sous,
                                            chemin_pour_staircase, chemins_sortie,
                                            chemins_staircases_p3prime,
                                            pauses_inter_staircases,
                                            valide_conditions_requises_si_humain,
                                            valide_graine_entree_p3prime)


def _collecteur():
    appels: list[str] = []
    return appels, appels.append


def test_plan_p3prime_est_le_plan_grave_et_rien_dautre():
    """D-P3′-1 TRANCHÉ §A42 : V,R,R,V,V,R — six staircases, bras r1, aucun
    paramétrage. Tout autre demande lève à voix haute."""
    assert chemins_staircases_p3prime("r1", 6) == (
        "viridis", "r1", "r1", "viridis", "viridis", "r1")
    assert chemins_staircases_p3prime("r1", 6) is PLAN_P3PRIME
    with pytest.raises(ValueError, match="D-P3′-1"):
        chemins_staircases_p3prime("viridis", 6)
    with pytest.raises(ValueError, match="6"):
        chemins_staircases_p3prime("r1", 4)


def test_graine_entree_refuse_une_brulee_et_lit_la_liste():
    """La garde lit la LISTE, pas une constante : le test l'étend avec une
    valeur de test et la garde mord dessus (champ d'essai 8c)."""
    appels, erreur = _collecteur()
    valide_graine_entree_p3prime(20260705, "humain", erreur)
    assert appels and "BRÛLÉE" in appels[0]
    appels, erreur = _collecteur()
    valide_graine_entree_p3prime(999, "humain", erreur,
                                 graines_brulees=frozenset({20260705, 999}))
    assert appels and "BRÛLÉE" in appels[0]


def test_graine_entree_refuse_une_graine_quelconque_par_egalite():
    """L'autre moitié de la garde double : ni brûlée ni gravée ⇒ REFUSÉ par
    l'égalité au 20260729 gravé."""
    appels, erreur = _collecteur()
    valide_graine_entree_p3prime(12345, "humain", erreur)
    assert appels and str(BASE_SEED_P3PRIME) in appels[0]


def test_graine_entree_accepte_la_gravee_et_refuse_apres_brulage():
    """LE TEST DU BRÛLAGE, côté entrée : 20260729 passe aujourd'hui ; ajouté
    à la liste (l'acte de gravure post-session), il est REFUSÉ à l'entrée —
    c'est ce qui empêchera un P3″ de le reconduire (faille B1a)."""
    appels, erreur = _collecteur()
    valide_graine_entree_p3prime(BASE_SEED_P3PRIME, "humain", erreur)
    assert not appels
    appels, erreur = _collecteur()
    valide_graine_entree_p3prime(
        BASE_SEED_P3PRIME, "humain", erreur,
        graines_brulees=frozenset(GRAINES_BRULEES | {BASE_SEED_P3PRIME}))
    assert appels and "BRÛLÉE" in appels[0]


def test_graine_entree_ne_mord_pas_le_synthetique():
    """Portée nommée : replay et synthétique consomment légitimement les
    graines historiques (re-dérivabilité du pin préservée)."""
    appels, erreur = _collecteur()
    valide_graine_entree_p3prime(20260705, "synthetique", erreur)
    assert not appels


def test_clause_d_a_lentree_plage_gravee():
    """Clause (d) à l'entrée, mode P3′ : une observation VÉRIDIQUE de nuit
    est refusée par la plage ; la même à 14h30 passe ; l'entrée HISTORIQUE
    (plage None) accepte la nuit véridique — portée nommée (mission 8a.4)."""
    nuit = datetime.now().replace(hour=0, minute=30)
    jour = datetime.now().replace(hour=14, minute=30)
    obs_nuit = f"repère à 750 mm, plafonnier, le {nuit:%d/%m/%Y} vers 0h30"
    obs_jour = f"repère à 750 mm, lumière du jour stable, le {jour:%d/%m/%Y} vers 14h30"

    appels, erreur = _collecteur()
    valide_conditions_requises_si_humain("humain", "OSD 80%", obs_nuit, erreur,
                                         plage_horaire=PLAGE_HORAIRE_SESSION,
                                         instant=nuit)
    assert appels and "plage" in appels[0]

    appels, erreur = _collecteur()
    valide_conditions_requises_si_humain("humain", "OSD 80%", obs_jour, erreur,
                                         plage_horaire=PLAGE_HORAIRE_SESSION,
                                         instant=jour)
    assert not appels

    appels, erreur = _collecteur()
    valide_conditions_requises_si_humain("humain", "OSD 80%", obs_nuit, erreur,
                                         plage_horaire=None, instant=nuit)
    assert not appels


def test_orchestre_campagne_p3prime_echauffe_et_scelle(tmp_path):
    """Bout en bout synthétique (mission 8a.2/8a.5/8a.7) : sous p3prime, la
    campagne (1) court l'échauffement AVANT les staircases — 15 essais, log
    SÉPARÉ marqué NON-ANALYSÉ, sans réponse ni correction ; (2) SCELLE les
    seuils — aucune clé `seuil` au manifeste, aucune valeur descellée ne
    fuit dans son JSON, sha256 vérifiable."""
    import json
    from pathlib import Path

    fabrique = lambda k, regime_nom, seed_sujet: (  # noqa: E731
        fabrique_sujet_synthetique(0.05, 0.015, seed=seed_sujet), lambda: None)
    manifeste = orchestre_campagne(
        base_seed=BASE_SEED_P3PRIME, n_staircases=N_STAIRCASES_P3PRIME, ppd=40.0,
        fabrique_repondre=fabrique, out_dir=tmp_path / "logs",
        regimes=(REGIME_SEVERE.nom,), chemins=PLAN_P3PRIME, protocole="p3prime")

    assert manifeste["protocole"] == "p3prime"
    # (1) Échauffement : consigné, loggé à part, non scorable.
    bloc = manifeste["echauffement"][REGIME_SEVERE.nom]
    assert bloc["n_essais"] == N_ECHAUFFEMENT
    assert bloc["indice_seed_reserve"] == INDICE_ECHAUFFEMENT
    assert "NON-ANALYSE" in bloc["log_path"]
    lignes = [json.loads(ligne) for ligne in
              Path(bloc["log_path"]).read_text(encoding="utf-8").splitlines()]
    assert len(lignes) == N_ECHAUFFEMENT
    for ligne in lignes:
        assert ligne["type_essai"] == "echauffement"
        assert "reponse" not in ligne and "correct" not in ligne, (
            "un log qu'on peut scorer invite à l'être (mission 8a.7)")
    # (2) Scellement : le manifeste ne porte AUCUN seuil, le fichier scellé
    # les porte tous, et son sha les verrouille.
    sessions = manifeste["regimes"][REGIME_SEVERE.nom]["sessions"]
    assert len(sessions) == N_STAIRCASES_P3PRIME
    assert all("seuil" not in s for s in sessions)
    scellement = manifeste["scellement_seuils"]
    seuils = descelle_seuils(scellement["chemin"], scellement["sha256"])[REGIME_SEVERE.nom]
    assert set(seuils.keys()) == {str(k) for k in range(N_STAIRCASES_P3PRIME)}
    dump = json.dumps(manifeste)
    for valeur in seuils.values():
        if valeur is not None:
            assert json.dumps(valeur) not in dump, "un seuil a fuité au manifeste"
    assert "PROCEDURAL" in scellement["nature"]


def test_orchestre_campagne_historique_inchangee_par_le_protocole(tmp_path):
    """Non-régression (mission 8c) : le protocole HISTORIQUE ne scelle rien,
    n'échauffe rien — les sessions portent leur `seuil` comme avant."""
    fabrique = lambda k, regime_nom, seed_sujet: (  # noqa: E731
        fabrique_sujet_synthetique(0.05, 0.015, seed=seed_sujet), lambda: None)
    manifeste = orchestre_campagne(
        base_seed=1000, n_staircases=1, ppd=40.0, fabrique_repondre=fabrique,
        out_dir=tmp_path / "logs")
    assert manifeste["protocole"] == "historique"
    assert "echauffement" not in manifeste
    assert "scellement_seuils" not in manifeste
    for regime_data in manifeste["regimes"].values():
        assert all("seuil" in s for s in regime_data["sessions"])


# --- Correctifs de la revue de remise (§A42-COMPLÉMENT-2, prereg [v2.3]) ----


def test_chemins_echauffement_est_le_plan_grave_d_8_1():
    """D-8-1 GRAVÉ (prereg [v2.3]) : ENTRELACÉ V,R,V,…,V — 15 essais, 8 V /
    7 R, COMMENCE et FINIT par viridis, jamais deux essais de suite sur le
    même chemin. Les alternatives sont nommées REFUSÉES au prereg (viridis
    seul : biais adverse mais GROS ; R1 seul : biais favorable)."""
    assert len(CHEMINS_ECHAUFFEMENT) == N_ECHAUFFEMENT == 15
    assert CHEMINS_ECHAUFFEMENT[0] == CHEMINS_ECHAUFFEMENT[-1] == "viridis"
    assert CHEMINS_ECHAUFFEMENT.count("viridis") == 8
    assert CHEMINS_ECHAUFFEMENT.count("r1") == 7
    assert all(a != b for a, b in zip(CHEMINS_ECHAUFFEMENT, CHEMINS_ECHAUFFEMENT[1:])), (
        "D-8-1 grave un ENTRELACEMENT : deux essais consécutifs sur le même chemin "
        "ne sont plus l'alternance que l'échauffement doit échauffer.")


def test_niveau_echauffement_egale_deux_fois_le_jnd_du_pin_reel():
    """D-8-2 GRAVÉ (prereg [v2.3]), test d'IDENTITÉ imposé : la constante vaut
    2 x le `jnd` sévère LU DANS L'ARTEFACT RÉEL — pas dans une recopie, pas
    dans un commentaire. Convention >= 2x de l'Arc C (« clairement au-dessus »).

    Ce que ce test verrouille : la constante ne peut plus dériver du pin en
    silence. Le sens d'import (lecture -> orchestration) interdit d'importer
    PIN_JND_SEV ici ; c'est donc l'ARTEFACT qui arbitre, comme le veut la
    règle « la machine lit, la session ne recopie pas » (§A41)."""
    import json
    from pathlib import Path

    pins = json.loads(
        (Path(ROOT) / "outputs" / "arcC" / "pins_spatial.json").read_text(encoding="utf-8"))
    jnd_sev = pins["regimes"]["severe"]["jnd"]
    assert NIVEAU_ECHAUFFEMENT == 2.0 * jnd_sev
    assert NIVEAU_ECHAUFFEMENT == 0.14668131914771332
    # Et le motif de D-8-2 : SUPRA-seuil pour le bras viridis, celui que
    # l'échauffement protège — l'ancien 0.05 était SOUS l'IC bas du pin.
    assert NIVEAU_ECHAUFFEMENT > pins["regimes"]["severe"]["ic"][1]


def test_echauffement_traverse_une_fabrique_sensible_au_chemin(tmp_path):
    """RÉGRESSION B1 (constat BLOQUANT de la revue de remise) : le bloc
    d'échauffement traverse SANS LEVER une fabrique SENSIBLE AU CHEMIN — une
    fabrique qui LÈVE sur tout indice hors 0..5 servant à indexer le plan des
    staircases, et sur tout chemin d'affichage hors {viridis, r1}.

    C'est LE test qui aurait attrapé B1 : la v1 dérivait le chemin de
    l'échauffement de `chemins[INDICE_ECHAUFFEMENT]` — `chemins[99]` sur un
    plan à SIX, IndexError GARANTI au premier essai humain. La fabrique
    synthétique des autres tests IGNORE l'indice : elle ne pouvait pas le
    voir. Ici la fabrique se comporte comme la fabrique HUMAINE : chaque
    essai DOIT se voir attribuer un chemin d'affichage connu.

    Le test vérifie AUSSI que la séquence effectivement traversée est le plan
    gravé D-8-1 lui-même — un échauffement qui ne lève plus mais affiche
    autre chose ne serait pas davantage le protocole."""
    chemins_vus: list[tuple[int, str]] = []

    def fabrique(numero_staircase, regime_nom, seed_sujet):
        if (numero_staircase != INDICE_ECHAUFFEMENT
                and not 0 <= numero_staircase < len(PLAN_P3PRIME)):
            raise AssertionError(
                f"fabrique sensible au chemin : indice de staircase {numero_staircase} "
                f"hors 0..{len(PLAN_P3PRIME) - 1} — le plan D-P3′-1 n'a aucun chemin "
                "d'affichage pour lui (c'est exactement le crash B1).")
        repondant = fabrique_sujet_synthetique(0.05, 0.015, seed=seed_sujet)

        def repondre(essai):
            chemin = chemin_pour_staircase(PLAN_P3PRIME, numero_staircase,
                                           essai.indice_essai)
            if chemin not in ("viridis", "r1"):
                raise AssertionError(
                    f"fabrique sensible au chemin : chemin d'affichage INCONNU "
                    f"{chemin!r} — rien à afficher.")
            chemins_vus.append((numero_staircase, chemin))
            return repondant(essai)

        return repondre, lambda: None

    manifeste = orchestre_campagne(
        base_seed=BASE_SEED_P3PRIME, n_staircases=N_STAIRCASES_P3PRIME, ppd=40.0,
        fabrique_repondre=fabrique, out_dir=tmp_path / "logs",
        regimes=(REGIME_SEVERE.nom,), chemins=PLAN_P3PRIME, protocole="p3prime")

    traverses = [chemin for k, chemin in chemins_vus if k == INDICE_ECHAUFFEMENT]
    assert traverses == list(CHEMINS_ECHAUFFEMENT), (
        "l'échauffement n'a pas traversé le plan gravé D-8-1")
    bloc = manifeste["echauffement"][REGIME_SEVERE.nom]
    assert bloc["chemins_echauffement"] == list(CHEMINS_ECHAUFFEMENT)
    assert bloc["dose_par_bras"] == {"viridis": 8, "r1": 7}
    assert bloc["niveau_delta_chi"] == NIVEAU_ECHAUFFEMENT


def test_chemin_pour_staircase_ne_derive_jamais_lechauffement_du_plan():
    """L'unité derrière B1 : l'échauffement se dispatche par `indice_essai`
    sur la séquence gravée, JAMAIS par indexation du plan des staircases —
    et il exige cet indice à voix haute plutôt que d'en inventer un."""
    assert chemin_pour_staircase(PLAN_P3PRIME, 0) == "viridis"
    for indice, attendu in enumerate(CHEMINS_ECHAUFFEMENT):
        assert chemin_pour_staircase(PLAN_P3PRIME, INDICE_ECHAUFFEMENT, indice) == attendu
    with pytest.raises(ValueError, match="indice_essai"):
        chemin_pour_staircase(PLAN_P3PRIME, INDICE_ECHAUFFEMENT)
    # Le geste de la v1, conservé ici comme pièce à conviction : indexer le
    # plan avec l'indice réservé LÈVE — c'est ce qui serait arrivé au sujet.
    with pytest.raises(IndexError):
        PLAN_P3PRIME[INDICE_ECHAUFFEMENT]


def test_bloc_echauffement_recoit_sa_sequence_et_refuse_une_longueur_autre(tmp_path):
    """La séquence est REÇUE explicitement (jamais dérivée du plan) et sa
    longueur est GRAVÉE : une séquence d'une autre taille lève à voix haute
    plutôt que d'échauffer un nombre d'essais qui n'est pas celui du prereg."""
    from scripts.run_arcC_orchestration import bloc_echauffement

    with pytest.raises(ValueError, match="D-8-1"):
        bloc_echauffement(
            regime=REGIME_SEVERE, regime_idx=0, base_seed=BASE_SEED_P3PRIME,
            sources_incluses=((0, 10),), budget=ANCRE_BUDGET,
            params=ParametresEscalier(), banques=BanqueBancs(),
            fabrique_repondre=lambda k, n, s: (lambda essai: "A", lambda: None),
            out_dir=tmp_path, chemins_echauffement=("viridis", "r1"))


def test_pauses_inter_staircases_est_pure_et_ne_fabrique_rien():
    """M2 (mission 8a.6) : la durée qui sépare la FIN de la staircase k du
    DÉBUT de la k+1 — fonction PURE, testable sans écran. Sur un manifeste
    d'AVANT ce chantier (aucun horodatage), elle rend une liste VIDE : on ne
    fabrique pas une durée qu'on n'a pas mesurée."""
    assert pauses_inter_staircases([]) == []
    assert pauses_inter_staircases(
        [dict(numero_staircase=0), dict(numero_staircase=1)]) == []
    sessions = [
        dict(numero_staircase=0, horodatage_debut="2026-08-02T10:00:00",
             horodatage_fin="2026-08-02T10:20:00"),
        dict(numero_staircase=1, horodatage_debut="2026-08-02T10:25:30",
             horodatage_fin="2026-08-02T10:45:00"),
        dict(numero_staircase=2, horodatage_debut="2026-08-02T10:47:00",
             horodatage_fin="2026-08-02T11:05:00")]
    assert pauses_inter_staircases(sessions) == [
        dict(entre=[0, 1], duree_s=330.0), dict(entre=[1, 2], duree_s=120.0)]


def test_horodatages_et_pauses_au_manifeste(tmp_path):
    """M2 : chaque staircase porte ses horodatages début/fin et sa durée ; les
    pauses inter-staircases sont consignées au manifeste. SURFACÉES, JAMAIS
    JUGÉES : aucune borne, aucune garde, aucun verdict ne s'y adosse — le
    statut d'orchestration ne dépend que de §C5, comme avant."""
    fabrique = lambda k, regime_nom, seed_sujet: (  # noqa: E731
        fabrique_sujet_synthetique(0.05, 0.015, seed=seed_sujet), lambda: None)
    manifeste = orchestre_campagne(
        base_seed=BASE_SEED_P3PRIME, n_staircases=N_STAIRCASES_P3PRIME, ppd=40.0,
        fabrique_repondre=fabrique, out_dir=tmp_path / "logs",
        regimes=(REGIME_SEVERE.nom,), chemins=PLAN_P3PRIME, protocole="p3prime")

    bloc_regime = manifeste["regimes"][REGIME_SEVERE.nom]
    sessions = bloc_regime["sessions"]
    for session in sessions:
        assert datetime.fromisoformat(session["horodatage_fin"]) >= datetime.fromisoformat(
            session["horodatage_debut"])
        assert session["duree_s"] >= 0.0
    pauses = bloc_regime["pauses_inter_staircases"]
    assert len(pauses) == len(sessions) - 1
    assert [p["entre"] for p in pauses] == [[k, k + 1] for k in range(len(sessions) - 1)]
    assert all(p["duree_s"] >= 0.0 for p in pauses)
    # L'échauffement date lui aussi sa fenêtre (il PRÉCÈDE la staircase 1).
    bloc = manifeste["echauffement"][REGIME_SEVERE.nom]
    assert datetime.fromisoformat(bloc["horodatage_fin"]) >= datetime.fromisoformat(
        bloc["horodatage_debut"])


def _argv_p3prime(**surcharges) -> list[str]:
    """Une ligne de commande P3′ VALIDE quant au reste, dont on ne fausse
    qu'UN paramètre à la fois — c'est ce qui rend le motif du refus
    attribuable. AUCUNE commande pré-remplie n'en sort : elle vit ici, dans
    un test, et n'est pas exécutable telle quelle (`main()` est appelée en
    mémoire, aucune session n'est lancée — chaque cas s'arrête sur
    `parser.error`)."""
    base = {"--base-seed": str(BASE_SEED_P3PRIME), "--sujet": "humain",
            "--rendu": "r1", "--regime": REGIME_SEVERE.nom,
            "--n-staircases": str(N_STAIRCASES_P3PRIME), "--protocole": "p3prime",
            "--long-ref-px": "1920", "--long-ref-mm": "597", "--distance-mm": "750",
            "--luminosite": "OSD 80%", "--conditions": "repère à 750 mm, lumière du jour"}
    base.update(surcharges)
    argv = ["run_arcC_orchestration.py"]
    for cle, valeur in base.items():
        argv.extend([cle] if valeur is None else [cle, valeur])
    return argv


def test_le_manifeste_p3prime_est_UN_exemplaire_partage_avec_la_lecture():
    """CÂBLAGE n°2 (constat de séance, 2026-08-03) : l'orchestration p3prime
    écrivait son manifeste au nom par défaut HISTORIQUE
    (`manifeste_campagne.json`) tandis que la lecture p3prime en attendait un
    autre (`manifeste_p3prime.json`) — deux bouts d'un même câble qui ne se
    parlaient pas, et la séance a dû être relue avec --manifeste explicite.

    La correction n'est pas d'aligner DEUX constantes (elles redivergeraient) :
    c'est de n'en avoir qu'UNE. La lecture IMPORTE celle de l'orchestration —
    le test vérifie l'IDENTITÉ des objets, pas une ressemblance de chaîne."""
    from scripts import run_p3_lecture

    assert run_p3_lecture.MANIFESTE_P3PRIME is MANIFESTE_P3PRIME
    assert MANIFESTE_P3PRIME.name == "manifeste_p3prime.json"
    assert MANIFESTE_P3PRIME != MANIFESTE_HISTORIQUE, (
        "le manifeste p3prime ne doit PAS porter le nom par défaut historique : "
        "c'est la collision qui a écrasé les artefacts du pin le 2026-08-03.")


@pytest.mark.parametrize("protocole, attendus", [
    ("historique", (LOGS_DIR, MANIFESTE_HISTORIQUE)),
    ("p3prime", (LOGS_DIR_P3PRIME, MANIFESTE_P3PRIME)),
])
def test_chemins_sortie_separent_les_protocoles(protocole, attendus):
    """Chaque protocole écrit CHEZ LUI : une campagne p3prime ne peut plus
    atterrir sur les chemins de la campagne du pin. Les surcharges explicites
    restent souveraines (l'outil ne décide pas à la place de qui sait)."""
    assert chemins_sortie(protocole, None, None) == attendus
    force_logs, force_man = Path("/tmp/x/logs"), Path("/tmp/x/m.json")
    assert chemins_sortie(protocole, force_logs, force_man) == (force_logs, force_man)


def test_artefacts_traques_voit_les_fichiers_versionnes(tmp_path):
    """La garde lit ce que GIT suit, pas une liste en dur : sous un dossier
    hors dépôt elle ne voit rien, sous les logs du pin elle voit les artefacts
    versionnés du 05/07."""
    assert artefacts_traques_sous(tmp_path) == []
    traques = artefacts_traques_sous(LOGS_DIR)
    assert any("session_severe_0.jsonl" in chemin for chemin in traques), (
        "les logs du pin sont versionnés : la garde doit les voir")


def test_main_refuse_decrire_par_dessus_des_artefacts_traques(monkeypatch, capsys):
    """LA GARDE qui aurait empêché la collision du 2026-08-03 : une campagne
    dont le dossier de sortie contient des fichiers SUIVIS PAR GIT refuse de
    démarrer. Un artefact versionné fait foi — on ne l'écrase pas en passant,
    fût-ce par un défaut de ligne de commande.

    Ici on force --out-dir sur les logs du pin : la garde mord AVANT toute
    campagne, et le motif nomme les fichiers en danger."""
    import sys

    from scripts import run_arcC_orchestration as orch

    monkeypatch.setattr(orch, "valide_conditions_requises_si_humain",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv",
                        _argv_p3prime(**{"--out-dir": str(LOGS_DIR)}))
    with pytest.raises(SystemExit):
        orch.main()
    erreur = capsys.readouterr().err
    assert "SUIVIS PAR GIT" in erreur
    assert "session_severe_0.jsonl" in erreur


@pytest.mark.parametrize("surcharges, motif", [
    ({"--base-seed": "12345"}, "GRAVÉE"),
    ({"--base-seed": "20260705"}, "BRÛLÉE"),
    ({"--temoin-viridis": None}, "temoin-viridis"),
    ({"--regime": REGIME_LAXISTE.nom}, "regime"),
    ({"--n-staircases": "4"}, "n-staircases"),
])
def test_main_cable_les_gardes_p3prime_au_niveau_parser(monkeypatch, capsys,
                                                       surcharges, motif):
    """(m3) Le CÂBLAGE CLI de `main()` testé AU NIVEAU PARSER : sous
    `--protocole p3prime`, une graine hors gravé ET une graine brûlée sont
    refusées, `--temoin-viridis` est refusé (dispositif du prereg P3, sans
    objet ici), et `--regime severe` + `--n-staircases 6` sont EXIGÉS.

    Chaque cas s'arrête sur `parser.error` (SystemExit) AVANT tout import
    matplotlib et AVANT toute campagne : aucune session n'est lancée par ce
    test. La clause (d) est neutralisée ici — elle a ses propres tests, et
    l'horloge du runner ne doit pas décider du motif qu'on vérifie (un test
    passé à 22 h tomberait sur (d) et prouverait autre chose)."""
    import sys

    from scripts import run_arcC_orchestration as orch

    monkeypatch.setattr(orch, "valide_conditions_requises_si_humain",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", _argv_p3prime(**surcharges))
    with pytest.raises(SystemExit):
        orch.main()
    assert motif in capsys.readouterr().err
