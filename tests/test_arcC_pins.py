"""Arc C / Task 3 — tests du post-traitement → pins
(`scripts/run_arcC_pins.py`), addendum §C8 de PREREGISTRATION.md
(pocCascade2phys, commits `c494d50`→`d7c95e3`), reproduit verbatim dans
`.superpowers/sdd/arcC-task3-runner-brief.md`. **AUCUN sujet humain, aucun
verdict de manche imprimé** : ce module CALCULE le pin + IC depuis les logs
d'une campagne (sujet SYNTHÉTIQUE en test) -- la LECTURE §C4 est au
contrôleur (Task 5), pas ici.

Familles de tests (brief, §Tests 4/6/7) :
  4. Agrégation -> pin + IC : seuils synthétiques connus -> `evalue_dispersion`
     + `statut_condition` produisent le pin (moyenne) et l'IC attendu ;
     dispersion > 30% -> INSTRUMENT_NE_PEUT_PAS_REPONDRE, pin=None.
  6. Bout-en-bout synthétique (LE GATE) : mini-campagne (sujet synthétique à
     theta connu) -> pins_spatial.json dont le JND agrégé retrouve theta +/-
     tolérance ; déterminisme (replay bit-identique hors `latence_s`).
  7. Schéma `pins_spatial.json` stable, round-trip."""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.arcC_abx import (STATUT_CONTINUE, STATUT_INSTRUMENT_NE_PEUT_PAS_REPONDRE,
                          STATUT_RESOLU, ParametresEscalier, lit_log_jsonl,
                          rejoue_escalier)
from src.arcC_synthetic import fabrique_sujet_synthetique
from scripts.run_arcC_orchestration import (N_STAIRCASES, ecrit_manifeste_json,
                                            lit_manifeste_json, orchestre_campagne)
from scripts.run_arcC_pins import (METHODE_IC, agrege_regime, calcule_ic,
                                   construit_pins, ecrit_pins_json, lit_pins_json)


# =============================================================================
# Famille 4 : agrégation -> pin + IC (seuils synthétiques connus)
# =============================================================================


def _session(numero: int, regime: str, seuil: float | None, complet: bool = True,
            valide: bool = True) -> dict:
    return dict(numero_staircase=numero, regime=regime, seed_roving=1, seed_catch=2,
               seed_sujet=3, log_path="/inexistant.jsonl", n_essais=20, n_reversals=8,
               complet=complet, seuil=seuil,
               validite=dict(n_catch=10, n_catch_ok=10, taux_reussite_catch=1.0,
                             seuil_reussite_catch=0.90, valide=valide))


def test_calcule_ic_min_max_sur_seuils_connus():
    ic = calcule_ic([0.030, 0.032, 0.028])
    assert ic["methode"] == METHODE_IC
    assert ic["borne_inf"] == pytest.approx(0.028)
    assert ic["borne_sup"] == pytest.approx(0.032)


def test_calcule_ic_liste_vide_est_none():
    assert calcule_ic([]) is None


def test_agrege_regime_staircases_coherentes_resolu():
    sessions = [_session(0, "severe", 0.030), _session(1, "severe", 0.032),
               _session(2, "severe", 0.028)]
    resultat = agrege_regime(sessions)
    assert resultat["statut"] == STATUT_RESOLU
    assert resultat["jnd"] == pytest.approx(np.mean([0.030, 0.032, 0.028]))
    assert resultat["ic"]["methode"] == METHODE_IC
    assert resultat["ic"]["borne_inf"] == pytest.approx(0.028)
    assert resultat["ic"]["borne_sup"] == pytest.approx(0.032)
    assert resultat["n_staircases"] == 3


def test_agrege_regime_dispersion_trop_grande_instrument_muet():
    """CV > 30% -> INSTRUMENT_NE_PEUT_PAS_REPONDRE, `jnd=None`, JAMAIS un pin
    fabriqué."""
    sessions = [_session(0, "severe", 0.010), _session(1, "severe", 0.030),
               _session(2, "severe", 0.080)]
    resultat = agrege_regime(sessions)
    assert resultat["statut"] == STATUT_INSTRUMENT_NE_PEUT_PAS_REPONDRE
    assert resultat["jnd"] is None
    assert resultat["ic"] is None


def test_agrege_regime_session_invalide_instrument_muet_meme_coherent():
    """Une session INVALIDE (§C5) -> instrument muet, même si les seuils
    (des staircases complètes/valides) seraient cohérents entre eux."""
    sessions = [_session(0, "severe", 0.030), _session(1, "severe", 0.032, valide=False),
               _session(2, "severe", 0.028)]
    resultat = agrege_regime(sessions)
    assert resultat["statut"] == STATUT_INSTRUMENT_NE_PEUT_PAS_REPONDRE
    assert resultat["jnd"] is None


def test_agrege_regime_staircase_incomplete_exclue_du_calcul_de_seuil():
    """Une staircase INCOMPLÈTE (`complet=False`, `seuil=None`) n'entre PAS
    dans les seuils agrégés (moins de 3 seuils complets -> incohérent, comme
    `evalue_dispersion` l'exige déjà)."""
    sessions = [_session(0, "severe", 0.030), _session(1, "severe", 0.032),
               _session(2, "severe", None, complet=False)]
    resultat = agrege_regime(sessions)
    assert resultat["n_staircases_completes"] == 2
    assert resultat["statut"] == STATUT_INSTRUMENT_NE_PEUT_PAS_REPONDRE  # n<3 complets
    assert resultat["jnd"] is None


# =============================================================================
# Famille 7 : schéma pins_spatial.json -- round-trip
# =============================================================================


def test_construit_pins_schema_stable_et_round_trip(tmp_path):
    manifeste = dict(
        parametres_graves=dict(ancre_budget=32, haut_jnd_plausible=0.08,
                               seuil_exclusion=0.16, n_staircases=3, budget_catch=32,
                               geometrie="pic-csf", catch_sources_fortes=False,
                               fraction_sources_fortes=None),
        config_affichage=dict(geometrie="pic-csf", ppd=40.0, taille_domaine_px=73),
        exclusions=dict(entrees=[dict(seed=103, L=10, delta_chi_ancre=0.1551, exclu=True)],
                        sources_exclues=[[103, 10]], sources_incluses=[[101, 10]],
                        n_exclues=1, n_sources=2, budget=32, seuil_exclusion=0.16,
                        n_exclusions_stop=10, statut=STATUT_CONTINUE),
        base_seed=1,
        statut_global=STATUT_CONTINUE,
        regimes={
            "severe": dict(sessions=[_session(0, "severe", 0.030), _session(1, "severe", 0.032),
                                     _session(2, "severe", 0.028)],
                          statut_orchestration=STATUT_CONTINUE, n_staircases_lancees=3),
            "laxiste": dict(sessions=[_session(0, "laxiste", 0.040), _session(1, "laxiste", 0.041),
                                      _session(2, "laxiste", 0.039)],
                           statut_orchestration=STATUT_CONTINUE, n_staircases_lancees=3),
        })
    pins = construit_pins(manifeste)
    for cle in ("parametres_graves", "exclusions", "statut_global", "regimes"):
        assert cle in pins
    assert pins["parametres_graves"]["methode_ic"] == METHODE_IC
    for regime_nom in ("severe", "laxiste"):
        for cle in ("jnd", "ic", "statut", "n_staircases"):
            assert cle in pins["regimes"][regime_nom]

    path = tmp_path / "pins_spatial.json"
    ecrit_pins_json(path, pins)
    relu = lit_pins_json(path)
    assert relu == pins
    # Round-trip via json standard aussi (schéma = JSON pur, pas d'objet opaque).
    with path.open("r", encoding="utf-8") as f:
        assert json.load(f) == pins


def test_construit_pins_statut_stop_exclusions_propage_regimes_vides():
    manifeste = dict(
        parametres_graves=dict(ancre_budget=32, haut_jnd_plausible=0.08,
                               seuil_exclusion=0.5, n_staircases=3, budget_catch=32,
                               geometrie="pic-csf", catch_sources_fortes=False,
                               fraction_sources_fortes=None),
        config_affichage=dict(geometrie="pic-csf", ppd=40.0, taille_domaine_px=73),
        exclusions=dict(entrees=[], sources_exclues=[[1, 10]] * 13, sources_incluses=[],
                        n_exclues=13, n_sources=20, budget=32, seuil_exclusion=0.5,
                        n_exclusions_stop=10, statut="STOP_TROP_DE_SOURCES_EXCLUES"),
        base_seed=1, statut_global="STOP_TROP_DE_SOURCES_EXCLUES", regimes={})
    pins = construit_pins(manifeste)
    assert pins["statut_global"] == "STOP_TROP_DE_SOURCES_EXCLUES"
    assert pins["regimes"] == {}


# =============================================================================
# Famille 6 : bout-en-bout synthétique (LE GATE)
# =============================================================================


def test_bout_en_bout_mini_campagne_retrouve_theta_connu(tmp_path):
    """LE GATE (brief) : mini-campagne synthétique à theta CONNU ->
    pins_spatial.json dont le JND agrégé retrouve theta +/- tolérance, comme
    le test-clé Task 2 (>= 3 staircases agrégées, ici les DEUX régimes)."""
    theta_sim, sigma = 0.10, 0.025  # marge x1.55 sous le plafond bas (15.5%, D-1)
    params = ParametresEscalier(delta_chi_initial=0.25, facteur_pas=1.20,
                                n_reversals_cible=10, n_essais_max=300)
    manifeste = orchestre_campagne(
        base_seed=4, n_staircases=N_STAIRCASES, ppd=40.0,
        theta_sim=theta_sim, sigma_sim=sigma, params=params,
        out_dir=tmp_path / "logs")
    assert manifeste["statut_global"] == STATUT_CONTINUE

    manifeste_path = tmp_path / "manifeste_campagne.json"
    ecrit_manifeste_json(manifeste_path, manifeste)
    manifeste_relu = lit_manifeste_json(manifeste_path)
    # Round-trip via JSON : les tuples (sources incluses/exclues) redeviennent
    # des listes -- comparaison normalisée via un aller-retour JSON du côté
    # gauche aussi (le SCHÉMA doit être stable, pas le type Python opaque).
    assert manifeste_relu == json.loads(json.dumps(manifeste))

    pins = construit_pins(manifeste_relu)
    pins_path = tmp_path / "pins_spatial.json"
    ecrit_pins_json(pins_path, pins)

    for regime_nom in ("severe", "laxiste"):
        resultat = pins["regimes"][regime_nom]
        assert resultat["statut"] == STATUT_RESOLU, (
            f"régime {regime_nom} : statut={resultat['statut']} -- seeds/paramètres de "
            "test à revoir, pas le comportement attendu ici.")
        erreur_relative = abs(resultat["jnd"] - theta_sim) / theta_sim
        print(f"[bout-en-bout] régime={regime_nom} theta_sim={theta_sim} -> "
              f"jnd={resultat['jnd']:.5f} (erreur relative={erreur_relative * 100:.2f}%)")
        assert erreur_relative < 0.20, (
            f"régime {regime_nom} : erreur relative {erreur_relative:.3f} > tolérance 20%.")

    # Déterminisme : rejouer chaque staircase depuis son log reproduit les
    # mêmes essais (hors latence_s), même seuil -- comme le test-clé Task 2.
    # Le pool de roving DOIT être le même sous-ensemble (post-exclusion, D-2)
    # que celui utilisé par l'orchestration, sinon `rejoue_escalier` détecte
    # -- à juste titre -- une divergence de stimulus (cf. son garde-fou).
    sources_incluses = tuple(tuple(s) for s in manifeste["exclusions"]["sources_incluses"])
    for regime_nom, regime_data in manifeste["regimes"].items():
        for session in regime_data["sessions"]:
            essais_log = lit_log_jsonl(session["log_path"])
            rejoue = rejoue_escalier(
                essais_log, numero_staircase=session["numero_staircase"], regime_nom=regime_nom,
                seed_roving=session["seed_roving"], seed_catch=session["seed_catch"],
                budget=32, params=params, sources=sources_incluses)
            assert len(rejoue.essais) == len(essais_log)
            for e1, e2 in zip(rejoue.essais, essais_log):
                assert e1.egal_hors_latence(e2)
            assert rejoue.seuil == session["seuil"]


def test_bout_en_bout_geometrie_plafond_meme_pins_que_pic_csf_structure(tmp_path):
    """La géométrie NE change QUE l'affichage (D-4) -- schéma pins identique,
    seule `config_affichage` diffère au manifeste amont (déjà testé dans
    `test_arcC_orchestration.py` ; ici, on vérifie juste que le pipeline pins
    fonctionne aussi pour `--geometrie plafond`, sans dépendre de l'affichage)."""
    theta_sim, sigma = 0.10, 0.025
    params = ParametresEscalier(delta_chi_initial=0.25, facteur_pas=1.20,
                                n_reversals_cible=10, n_essais_max=300)
    manifeste = orchestre_campagne(
        base_seed=4, n_staircases=N_STAIRCASES, ppd=40.0, geometrie="plafond",
        theta_sim=theta_sim, sigma_sim=sigma, params=params,
        out_dir=tmp_path / "logs")
    pins = construit_pins(manifeste)
    assert pins["regimes"]["severe"]["statut"] == STATUT_RESOLU
    assert pins["parametres_graves"]["geometrie"] == "plafond"
