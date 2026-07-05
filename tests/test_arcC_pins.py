"""Arc C / Task 3 (conformance §C10) — tests du post-traitement → pins
CONFORMES au contrat de sortie gravé (`scripts/run_arcC_pins.py`,
`schemas/arcC-pins-spatial-v1.schema.json`), reproduit verbatim dans
`.superpowers/sdd/arcC-task3conform-schema-brief.md`. **AUCUN sujet humain,
aucun verdict de manche imprimé/écrit** : ce module CALCULE des MESURES
(pin + IC par régime, roll-up timing, contrôle directionnel) depuis les logs
d'une campagne (sujet SYNTHÉTIQUE en test) -- la LECTURE §C4 est au
contrôleur (Task 5), jamais ici (`additionalProperties:false` du schéma
l'interdit structurellement).

Familles de tests (brief) :
  1. Validation schéma : sortie synthétique bout-en-bout VALIDE ; cas
     construit NON conforme (champ hors schéma / statut nu sans motif) ->
     `ecrit_pins_json` LÈVE avant écriture (rien n'est écrit).
  2. Statut 3-voies + `motif_statut` : RESOLU / INDETERMINE (dispersion
     > 0.30, jnd NON null) / INVALIDE (catch < 0.90, jnd/ic null).
  3. Contrôle directionnel : IC laxiste entièrement sous sévère -> `true` ;
     IC chevauchants (même mal ordonnés) -> `false` ; régime non-RESOLU ->
     `false`.
  4. Exclusions nommées : source exclue apparaît avec son `chi_ancre`,
     `utilisees` correct, jamais silencieuse.
  5. `branche_combinee_active` : 3 staircases -> `false`, pas d'`ic_combine` ;
     6 staircases -> `true`, `ic_combine` = mean±2SEM exact.
  6. Copie du schéma pocPhysicator : draft-07 valide, se charge.
  7. Non-régression bout-en-bout : figures + absence de verdict §C4
     préservées."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft7Validator

from src.arcC_abx import STATUT_CONTINUE, STATUT_STOP_2_INVALIDES, ParametresEscalier
from scripts.run_arcC_orchestration import N_STAIRCASES, orchestre_campagne
from scripts.run_arcC_pins import (SCHEMA_PATH, STATUT_INDETERMINE, STATUT_INVALIDE,
                                   STATUT_RESOLU, calcule_branche_combinee_active,
                                   calcule_controle_directionnel, calcule_ic,
                                   calcule_ic_combine, charge_schema, construit_pins,
                                   construit_regime_pins, construit_sources,
                                   determine_statut_regime, ecrit_pins_json, lit_pins_json,
                                   valide_pins)

SCHEMA = charge_schema()


# =============================================================================
# Fixtures de test -- sessions/manifeste CONSTRUITS (pas de campagne réelle)
# =============================================================================


def _session(numero: int, regime: str, seuil: float | None, *, complet: bool = True,
            valide: bool = True, n_catch: int = 10, n_catch_ok: int | None = None,
            n_essais: int = 20, n_reversals: int = 8, log_path: str | None = None) -> dict:
    """Enregistrement `sessions[]` -- format `orchestre_regime` (§C8)."""
    if n_catch_ok is None:
        n_catch_ok = n_catch if valide else 0
    taux = (n_catch_ok / n_catch) if n_catch > 0 else float("nan")
    return dict(
        numero_staircase=numero, regime=regime, seed_roving=1, seed_catch=2, seed_sujet=3,
        log_path=(log_path if log_path is not None else f"/inexistant_{regime}_{numero}.jsonl"),
        n_essais=n_essais, n_reversals=n_reversals, complet=complet, seuil=seuil,
        validite=dict(n_catch=n_catch, n_catch_ok=n_catch_ok, taux_reussite_catch=taux,
                     seuil_reussite_catch=0.90, valide=valide))


def _manifeste(regimes: dict, *, sujet: str = "synthetique", ppd: float = 40.0,
              taille_domaine_px: int = 73, luminosite: str | None = None,
              conditions: str | None = None, seuil_exclusion: float = 0.16,
              entrees_exclusions: list[dict] | None = None, n_sources: int = 20) -> dict:
    """Manifeste minimal (format `orchestre_campagne`) suffisant pour
    `construit_pins` -- pas de campagne réelle lancée."""
    entrees = entrees_exclusions if entrees_exclusions is not None else []
    n_exclues = sum(1 for e in entrees if e["exclu"])
    return dict(
        sujet=sujet, commit_harnais="deadbeef" * 5, date_session="2026-07-05T12:00:00",
        base_seed=1,
        parametres_graves=dict(ancre_budget=32, haut_jnd_plausible=0.08,
                               seuil_exclusion=seuil_exclusion, n_staircases=3, budget_catch=32,
                               geometrie="pic-csf", catch_sources_fortes=False,
                               fraction_sources_fortes=None),
        config_affichage=dict(geometrie="pic-csf", ppd=ppd, taille_domaine_px=taille_domaine_px),
        conditions_validite=dict(
            luminosite=luminosite, conditions=conditions,
            calibration=dict(ppd=ppd, long_ref_px=1920.0, long_ref_mm=310.0, distance_mm=600.0)),
        exclusions=dict(entrees=entrees, sources_exclues=[], sources_incluses=[],
                        n_exclues=n_exclues, n_sources=n_sources, budget=32,
                        seuil_exclusion=seuil_exclusion, n_exclusions_stop=10,
                        statut=STATUT_CONTINUE),
        statut_global=STATUT_CONTINUE,
        regimes=regimes)


# =============================================================================
# Famille 6 : copie du schéma -- draft-07 valide, se charge
# =============================================================================


def test_copie_schema_pocphysicator_charge_et_est_draft07_valide():
    schema = charge_schema()
    assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"
    assert schema["$id"] == "arcC-pins-spatial-v1.schema.json"
    Draft7Validator.check_schema(schema)  # méta-validation -- ne lève pas si draft-07 correct


def test_copie_schema_chemin_attendu():
    assert SCHEMA_PATH == Path(__file__).resolve().parents[1] / "schemas" / \
        "arcC-pins-spatial-v1.schema.json"
    assert SCHEMA_PATH.exists()


# =============================================================================
# Famille 2 : statut 3-voies + motif_statut
# =============================================================================


def test_determine_statut_regime_resolu_dispersion_basse_catch_ok():
    sessions = [_session(0, "severe", 0.030), _session(1, "severe", 0.032),
               _session(2, "severe", 0.028)]
    statut, motif, dispersion = determine_statut_regime(sessions, STATUT_CONTINUE)
    assert statut == STATUT_RESOLU
    assert motif is None
    assert dispersion["coherent"] is True


def test_construit_regime_pins_resolu_jnd_ic_non_null_pas_de_motif():
    sessions = [_session(0, "severe", 0.030), _session(1, "severe", 0.032),
               _session(2, "severe", 0.028)]
    regime = construit_regime_pins("severe", sessions, STATUT_CONTINUE, False)
    assert regime["statut"] == STATUT_RESOLU
    assert regime["jnd"] == pytest.approx(np.mean([0.030, 0.032, 0.028]))
    assert regime["ic"] == [pytest.approx(0.028), pytest.approx(0.032)]
    assert "motif_statut" not in regime


def test_construit_regime_pins_indetermine_dispersion_trop_grande_jnd_non_null_motif():
    """CV > 30% -> INDETERMINE (PAS INVALIDE) -- `jnd`/`ic` NON null (valeur
    centrale rapportée mais non fiable, `null` réservé à INVALIDE, §C10)."""
    sessions = [_session(0, "severe", 0.010), _session(1, "severe", 0.030),
               _session(2, "severe", 0.080)]
    regime = construit_regime_pins("severe", sessions, STATUT_CONTINUE, False)
    assert regime["statut"] == STATUT_INDETERMINE
    assert regime["jnd"] is not None
    assert regime["ic"] is not None
    assert "motif_statut" in regime and regime["motif_statut"]


def test_construit_regime_pins_invalide_catch_sous_90pct_jnd_ic_null_motif():
    sessions = [_session(0, "severe", 0.030), _session(1, "severe", 0.032, valide=False,
                                                        n_catch=10, n_catch_ok=5),
               _session(2, "severe", 0.028)]
    regime = construit_regime_pins("severe", sessions, STATUT_CONTINUE, False)
    assert regime["statut"] == STATUT_INVALIDE
    assert regime["jnd"] is None
    assert regime["ic"] is None
    assert "motif_statut" in regime and regime["motif_statut"]


def test_construit_regime_pins_invalide_arret_2_invalides_jnd_ic_null_motif():
    sessions = [_session(0, "severe", 0.05, valide=False, n_catch_ok=2),
               _session(1, "severe", 0.05, valide=False, n_catch_ok=1)]
    regime = construit_regime_pins("severe", sessions, STATUT_STOP_2_INVALIDES, False)
    assert regime["statut"] == STATUT_INVALIDE
    assert regime["jnd"] is None
    assert regime["ic"] is None
    assert "arrêt" in regime["motif_statut"].lower()


def test_construit_regime_pins_staircases_brutes_toutes_incluses_meme_incompletes():
    sessions = [_session(0, "severe", 0.030), _session(1, "severe", 0.032),
               _session(2, "severe", None, complet=False)]
    regime = construit_regime_pins("severe", sessions, STATUT_CONTINUE, False)
    assert len(regime["staircases"]) == 3  # TOUTES, y compris l'incomplète
    assert regime["staircases"][2]["seuil"] is None
    assert all(s["renversements_moyennes"] == 6 for s in regime["staircases"])


def test_construit_regime_pins_taux_catch_global_pondere_sur_tous_les_essais():
    sessions = [_session(0, "severe", 0.030, n_catch=10, n_catch_ok=10),
               _session(1, "severe", 0.032, n_catch=10, n_catch_ok=8, valide=True),
               _session(2, "severe", 0.028, n_catch=10, n_catch_ok=10)]
    # Session 1 valide malgré 8/10 (>= seuil_reussite_catch construit à la main ici) --
    # le test cible taux_catch_global, pas le gate de validité individuel.
    regime = construit_regime_pins("severe", sessions, STATUT_CONTINUE, False)
    assert regime["taux_catch_global"] == pytest.approx((10 + 8 + 10) / 30)


# =============================================================================
# Famille 3 : contrôle directionnel
# =============================================================================


def test_controle_directionnel_ic_laxiste_entierement_sous_severe_true():
    sev = dict(statut=STATUT_RESOLU, ic=[0.03, 0.04])
    lax = dict(statut=STATUT_RESOLU, ic=[0.01, 0.02])  # entièrement SOUS severe -> mauvais ordre
    assert calcule_controle_directionnel(sev, lax) is True


def test_controle_directionnel_ic_chevauchants_meme_mal_ordonnes_false():
    sev = dict(statut=STATUT_RESOLU, ic=[0.03, 0.05])
    lax = dict(statut=STATUT_RESOLU, ic=[0.02, 0.04])  # chevauchent (0.03-0.04 commun)
    assert calcule_controle_directionnel(sev, lax) is False


def test_controle_directionnel_ic_bon_ordre_disjoints_false():
    sev = dict(statut=STATUT_RESOLU, ic=[0.01, 0.02])
    lax = dict(statut=STATUT_RESOLU, ic=[0.04, 0.05])  # BON ordre (lax > sev) -- attendu, pas violation
    assert calcule_controle_directionnel(sev, lax) is False


def test_controle_directionnel_regime_non_resolu_false():
    sev = dict(statut=STATUT_INDETERMINE, ic=[0.03, 0.04])
    lax = dict(statut=STATUT_RESOLU, ic=[0.01, 0.02])
    assert calcule_controle_directionnel(sev, lax) is False
    sev2 = dict(statut=STATUT_INVALIDE, ic=None)
    lax2 = dict(statut=STATUT_RESOLU, ic=[0.01, 0.02])
    assert calcule_controle_directionnel(sev2, lax2) is False


# =============================================================================
# Famille 4 : exclusions nommées -- jamais silencieuses
# =============================================================================


def test_sources_exclues_nommees_avec_chi_ancre_et_utilisees_correct():
    entrees = [dict(seed=103, L=10, delta_chi_ancre=0.1551, exclu=True)] + \
        [dict(seed=s, L=10, delta_chi_ancre=0.5, exclu=False) for s in range(1, 20)]
    manifeste = _manifeste({}, entrees_exclusions=entrees, n_sources=20)
    src = construit_sources(manifeste)
    assert src["utilisees"] == 19
    assert len(src["exclues"]) == 1
    assert src["exclues"][0]["id"] == "s103_L10"
    assert src["exclues"][0]["chi_ancre"] == pytest.approx(0.1551)


def test_sources_aucune_exclusion_liste_vide():
    entrees = [dict(seed=s, L=10, delta_chi_ancre=0.5, exclu=False) for s in range(1, 21)]
    manifeste = _manifeste({}, entrees_exclusions=entrees, n_sources=20)
    src = construit_sources(manifeste)
    assert src["utilisees"] == 20
    assert src["exclues"] == []


# =============================================================================
# Famille 5 : branche_combinee_active + ic_combine (mean±2SEM)
# =============================================================================


def test_calcule_branche_combinee_active_false_avec_3_staircases_par_regime():
    regimes = {"severe": dict(sessions=[_session(i, "severe", 0.03) for i in range(3)]),
              "laxiste": dict(sessions=[_session(i, "laxiste", 0.04) for i in range(3)])}
    assert calcule_branche_combinee_active(regimes) is False


def test_calcule_branche_combinee_active_true_avec_6_staircases_par_regime():
    regimes = {"severe": dict(sessions=[_session(i, "severe", 0.03) for i in range(6)]),
              "laxiste": dict(sessions=[_session(i, "laxiste", 0.04) for i in range(6)])}
    assert calcule_branche_combinee_active(regimes) is True


def test_calcule_branche_combinee_active_false_si_un_seul_regime_a_6():
    regimes = {"severe": dict(sessions=[_session(i, "severe", 0.03) for i in range(6)]),
              "laxiste": dict(sessions=[_session(i, "laxiste", 0.04) for i in range(3)])}
    assert calcule_branche_combinee_active(regimes) is False


def test_construit_regime_pins_3_staircases_pas_dic_combine():
    sessions = [_session(i, "severe", v) for i, v in enumerate([0.030, 0.032, 0.028])]
    regime = construit_regime_pins("severe", sessions, STATUT_CONTINUE, False)
    assert "ic_combine" not in regime


def test_construit_regime_pins_6_staircases_ic_combine_mean_2sem_exact():
    seuils = [0.030, 0.032, 0.028, 0.031, 0.029, 0.033]
    sessions = [_session(i, "severe", v) for i, v in enumerate(seuils)]
    regime = construit_regime_pins("severe", sessions, STATUT_CONTINUE, True)
    assert "ic_combine" in regime
    arr = np.asarray(seuils)
    moyenne, sem = arr.mean(), arr.std(ddof=1) / np.sqrt(6)
    assert regime["ic_combine"][0] == pytest.approx(moyenne - 2.0 * sem)
    assert regime["ic_combine"][1] == pytest.approx(moyenne + 2.0 * sem)


def test_calcule_ic_combine_moins_de_2_seuils_est_none():
    assert calcule_ic_combine([]) is None
    assert calcule_ic_combine([0.03]) is None


def test_calcule_ic_min_max_sur_seuils_connus():
    assert calcule_ic([0.030, 0.032, 0.028]) == [pytest.approx(0.028), pytest.approx(0.032)]


def test_calcule_ic_liste_vide_est_none():
    assert calcule_ic([]) is None


# =============================================================================
# Famille : calibration -- luminosite/conditions None (synthétique) -> string
# =============================================================================


def test_construit_pins_calibration_luminosite_none_devient_synthetique(tmp_path):
    sessions_sev = [_session(i, "severe", 0.030 + 0.001 * i) for i in range(3)]
    sessions_lax = [_session(i, "laxiste", 0.040 + 0.001 * i) for i in range(3)]
    manifeste = _manifeste(dict(
        severe=dict(sessions=sessions_sev, statut_orchestration=STATUT_CONTINUE),
        laxiste=dict(sessions=sessions_lax, statut_orchestration=STATUT_CONTINUE)),
        luminosite=None, conditions=None)
    pins = construit_pins(manifeste, manifeste_path=tmp_path / "manifeste.json")
    assert pins["calibration"]["luminosite"] == "(synthetique)"
    assert pins["calibration"]["conditions"] == "(synthetique)"


def test_construit_pins_calibration_luminosite_fournie_preservee(tmp_path):
    sessions_sev = [_session(i, "severe", 0.030 + 0.001 * i) for i in range(3)]
    sessions_lax = [_session(i, "laxiste", 0.040 + 0.001 * i) for i in range(3)]
    manifeste = _manifeste(dict(
        severe=dict(sessions=sessions_sev, statut_orchestration=STATUT_CONTINUE),
        laxiste=dict(sessions=sessions_lax, statut_orchestration=STATUT_CONTINUE)),
        luminosite="120 nits", conditions="bureau, jour, stable")
    pins = construit_pins(manifeste, manifeste_path=tmp_path / "manifeste.json")
    assert pins["calibration"]["luminosite"] == "120 nits"
    assert pins["calibration"]["conditions"] == "bureau, jour, stable"


def test_construit_pins_timing_laxiste_absent_sidecar_zero(tmp_path, capsys):
    """Sujet synthétique : aucun sidecar `*.timing.jsonl` -> `0.0`/`0.0`,
    NOTÉ (imprimé), conforme au schéma (informatif nul)."""
    sessions_sev = [_session(i, "severe", 0.030 + 0.001 * i) for i in range(3)]
    sessions_lax = [_session(i, "laxiste", 0.040 + 0.001 * i) for i in range(3)]
    manifeste = _manifeste(dict(
        severe=dict(sessions=sessions_sev, statut_orchestration=STATUT_CONTINUE),
        laxiste=dict(sessions=sessions_lax, statut_orchestration=STATUT_CONTINUE)))
    pins = construit_pins(manifeste, manifeste_path=tmp_path / "manifeste.json")
    assert pins["timing_laxiste"] == dict(exposition_moyenne_s=0.0, derive_max_pct=0.0)
    assert "aucun sidecar" in capsys.readouterr().out


def test_construit_pins_champs_racine_provenance(tmp_path):
    sessions_sev = [_session(i, "severe", 0.030 + 0.001 * i) for i in range(3)]
    sessions_lax = [_session(i, "laxiste", 0.040 + 0.001 * i) for i in range(3)]
    manifeste = _manifeste(dict(
        severe=dict(sessions=sessions_sev, statut_orchestration=STATUT_CONTINUE),
        laxiste=dict(sessions=sessions_lax, statut_orchestration=STATUT_CONTINUE)),
        sujet="humain")
    chemin_manifeste = tmp_path / "manifeste.json"
    pins = construit_pins(manifeste, manifeste_path=chemin_manifeste)
    assert pins["schema"] == "arcC-pins-spatial-v1"
    assert pins["sujet"] == "humain"
    assert pins["base_seed"] == 1
    assert pins["commit_harnais"] == "deadbeef" * 5
    assert pins["date_session"] == "2026-07-05T12:00:00"
    assert pins["manifeste"] == str(chemin_manifeste)


def test_construit_pins_refuse_campagne_stop_trop_exclues(tmp_path):
    """Le schéma n'a AUCUNE représentation pour une campagne arrêtée avant le
    lancement des régimes -- refus explicite, PAS de `regimes={}` bricolé."""
    manifeste = _manifeste({})
    manifeste["statut_global"] = "STOP_TROP_DE_SOURCES_EXCLUES"
    manifeste["regimes"] = {}
    with pytest.raises(RuntimeError):
        construit_pins(manifeste, manifeste_path=tmp_path / "manifeste.json")


def test_construit_pins_refuse_regime_stop_2_invalides_aiguillage_avant_jsonschema(tmp_path):
    """Le trou (revue) : un régime arrêté à 2 staircases par le garde
    2-invalides (§C5, `STATUT_STOP_2_INVALIDES`) ne doit JAMAIS atteindre la
    validation `jsonschema` (« ... is too short » cryptique, `staircases`
    minItems=3) -- `construit_pins` intercepte AVANT, avec un message qui
    porte le FAIT (N=2 staircases), le RÉGIME, le CHEMIN DU MANIFESTE et la
    CONDUITE (arbitrage humain, "remonter"). AUCUNE voie de complétion
    automatique suggérée -- le STOP reste un STOP. `ecrit_pins_json` n'est
    JAMAIS atteint : aucun fichier n'est écrit, même partiel."""
    sessions_severe_stop = [_session(0, "severe", 0.05, valide=False, n_catch_ok=2),
                           _session(1, "severe", 0.05, valide=False, n_catch_ok=1)]
    sessions_laxiste = [_session(i, "laxiste", 0.040 + 0.001 * i) for i in range(3)]
    manifeste_path = tmp_path / "manifeste_campagne.json"
    manifeste = _manifeste(dict(
        severe=dict(sessions=sessions_severe_stop, statut_orchestration=STATUT_STOP_2_INVALIDES),
        laxiste=dict(sessions=sessions_laxiste, statut_orchestration=STATUT_CONTINUE)))
    pins_path = tmp_path / "pins_spatial.json"

    with pytest.raises(RuntimeError) as exc_info:
        pins = construit_pins(manifeste, manifeste_path=manifeste_path)
        ecrit_pins_json(pins_path, pins)  # jamais atteint -- construit_pins lève avant

    message = str(exc_info.value)
    assert "2 staircases" in message  # le FAIT (N réel de sessions du régime)
    assert "2-invalides" in message and "§C5" in message
    assert "severe" in message  # le RÉGIME concerné
    assert str(manifeste_path) in message  # le CHEMIN DU MANIFESTE
    assert "remonter" in message.lower()  # la CONDUITE : arbitrage humain, remonter
    # AUCUNE voie de complétion automatique suggérée : le message doit DISCLAIMER
    # explicitement la 3e staircase de complaisance, jamais la suggérer sans réserve.
    assert "n'est pas une invite" in message.lower() or "pas de contournement" in message.lower()
    assert not pins_path.exists()  # rien n'est écrit, pas même un référent partiel


def test_construit_pins_indetermine_n3_dispersion_haute_pins_valide_contre_schema(tmp_path):
    """Ferme la 3ᵉ branche (les 3 issues : complétée-résolue / complétée mais
    INDETERMINE (dispersion trop haute) / arrêtée ont chacune leur test) :
    3 staircases COMPLÈTES et VALIDES (catch >= 90 %) mais seuils très
    dispersés (CV > 30 %, §C5-3) -> `statut == INDETERMINE`, `motif_statut`
    présent, `jnd`/`ic` NON null (valeur centrale rapportée, non fiable --
    `null` réservé à INVALIDE), ET le pins produit est VALIDE contre le
    schéma §C10 (n'est PAS un cas de refus -- `construit_pins` produit un
    document, `ecrit_pins_json` l'écrit sans lever)."""
    sessions_severe_disperse = [_session(0, "severe", 0.010), _session(1, "severe", 0.030),
                                _session(2, "severe", 0.080)]  # CV ~90% >> 30%
    sessions_laxiste = [_session(i, "laxiste", 0.040 + 0.001 * i) for i in range(3)]
    manifeste = _manifeste(dict(
        severe=dict(sessions=sessions_severe_disperse, statut_orchestration=STATUT_CONTINUE),
        laxiste=dict(sessions=sessions_laxiste, statut_orchestration=STATUT_CONTINUE)))

    pins = construit_pins(manifeste, manifeste_path=tmp_path / "manifeste.json")
    resultat_severe = pins["regimes"]["severe"]
    assert resultat_severe["statut"] == STATUT_INDETERMINE
    assert "motif_statut" in resultat_severe and resultat_severe["motif_statut"]
    assert resultat_severe["jnd"] is not None
    assert resultat_severe["ic"] is not None

    pins_path = tmp_path / "pins_spatial.json"
    ecrit_pins_json(pins_path, pins)  # valide §C10 avant écriture -- ne lève PAS
    relu = lit_pins_json(pins_path)
    Draft7Validator(SCHEMA).validate(relu)  # LE GATE : conforme malgré INDETERMINE


# =============================================================================
# Famille 1 : validation schéma -- conforme accepté, non conforme LÈVE
# =============================================================================


def _pins_valide_minimal(tmp_path) -> dict:
    sessions_sev = [_session(i, "severe", 0.030 + 0.001 * i) for i in range(3)]
    sessions_lax = [_session(i, "laxiste", 0.040 + 0.001 * i) for i in range(3)]
    manifeste = _manifeste(dict(
        severe=dict(sessions=sessions_sev, statut_orchestration=STATUT_CONTINUE),
        laxiste=dict(sessions=sessions_lax, statut_orchestration=STATUT_CONTINUE)),
        luminosite="120 nits", conditions="bureau")
    return construit_pins(manifeste, manifeste_path=tmp_path / "manifeste.json")


def test_valide_pins_accepte_document_conforme(tmp_path):
    pins = _pins_valide_minimal(tmp_path)
    valide_pins(pins, SCHEMA)  # ne lève pas


def test_valide_pins_leve_si_champ_hors_schema(tmp_path):
    pins = _pins_valide_minimal(tmp_path)
    pins["verdict_manche"] = "PASS"  # additionalProperties:false à la racine
    with pytest.raises(ValueError, match="NON CONFORME"):
        valide_pins(pins, SCHEMA)


def test_valide_pins_leve_si_statut_nu_sans_motif(tmp_path):
    pins = _pins_valide_minimal(tmp_path)
    pins["regimes"]["severe"]["statut"] = STATUT_INDETERMINE  # sans motif_statut -> non conforme
    with pytest.raises(ValueError, match="NON CONFORME"):
        valide_pins(pins, SCHEMA)


def test_ecrit_pins_json_leve_avant_ecriture_rien_nest_ecrit(tmp_path):
    pins = _pins_valide_minimal(tmp_path)
    pins["verdict_manche"] = "PASS"
    chemin = tmp_path / "pins_spatial.json"
    with pytest.raises(ValueError):
        ecrit_pins_json(chemin, pins)
    assert not chemin.exists()  # JAMAIS un référent malformé écrit


def test_ecrit_pins_json_document_conforme_ecrit_et_round_trip(tmp_path):
    pins = _pins_valide_minimal(tmp_path)
    chemin = tmp_path / "pins_spatial.json"
    ecrit_pins_json(chemin, pins)
    relu = lit_pins_json(chemin)
    assert relu == pins
    with chemin.open("r", encoding="utf-8") as f:
        assert json.load(f) == pins
    Draft7Validator(SCHEMA).validate(relu)  # re-validation post round-trip, ne lève pas


# =============================================================================
# Famille 7 : bout-en-bout synthétique RÉEL (LE GATE) -- valide contre schéma,
# figures préservées, AUCUN verdict.
# =============================================================================


def test_bout_en_bout_mini_campagne_valide_contre_schema_et_retrouve_theta(tmp_path):
    theta_sim, sigma = 0.10, 0.025  # marge x1.55 sous le plafond bas (15.5%, D-1)
    params = ParametresEscalier(delta_chi_initial=0.25, facteur_pas=1.20,
                                n_reversals_cible=10, n_essais_max=300)
    manifeste = orchestre_campagne(
        base_seed=4, n_staircases=N_STAIRCASES, ppd=40.0,
        theta_sim=theta_sim, sigma_sim=sigma, params=params, out_dir=tmp_path / "logs",
        long_ref_px=1920.0, long_ref_mm=310.0, distance_mm=600.0,
        luminosite=None, conditions=None)  # sujet synthétique -- (synthetique) attendu
    assert manifeste["statut_global"] == STATUT_CONTINUE

    manifeste_path = tmp_path / "manifeste_campagne.json"
    manifeste_path.write_text(json.dumps(manifeste, ensure_ascii=False), encoding="utf-8")
    pins = construit_pins(manifeste, manifeste_path=manifeste_path)

    # LE GATE : valide contre le schéma §C10, de bout en bout.
    valide_pins(pins, SCHEMA)

    pins_path = tmp_path / "pins_spatial.json"
    ecrit_pins_json(pins_path, pins)
    relu = lit_pins_json(pins_path)
    Draft7Validator(SCHEMA).validate(relu)

    for regime_nom in ("severe", "laxiste"):
        resultat = pins["regimes"][regime_nom]
        assert resultat["statut"] == STATUT_RESOLU, (
            f"régime {regime_nom} : statut={resultat['statut']} "
            f"motif={resultat.get('motif_statut')} -- seeds/paramètres à revoir.")
        erreur_relative = abs(resultat["jnd"] - theta_sim) / theta_sim
        assert erreur_relative < 0.20, (
            f"régime {regime_nom} : erreur relative {erreur_relative:.3f} > tolérance 20%.")

    assert pins["calibration"]["luminosite"] == "(synthetique)"
    assert pins["sujet"] == "synthetique"
    assert len(pins["artefacts"]["logs"]) >= 1
    # AUCUN verdict de manche -- le document est structurellement clos
    # (additionalProperties:false) : aucune clé "verdict"/"pass"/"jnd_seuil"
    # ne peut exister, déjà garanti par la validation ci-dessus.


def test_bout_en_bout_figures_generees_sans_verdict(tmp_path):
    """Non-régression (famille 7) : les figures psychométriques existantes
    sont préservées (une par régime), AUCUN verdict de manche tracé/écrit."""
    from scripts.run_arcC_pins import figure_psychometrique

    theta_sim, sigma = 0.10, 0.025
    params = ParametresEscalier(delta_chi_initial=0.25, facteur_pas=1.20,
                                n_reversals_cible=10, n_essais_max=300)
    manifeste = orchestre_campagne(
        base_seed=5, n_staircases=N_STAIRCASES, ppd=40.0,
        theta_sim=theta_sim, sigma_sim=sigma, params=params, out_dir=tmp_path / "logs",
        long_ref_px=1920.0, long_ref_mm=310.0, distance_mm=600.0)
    pins = construit_pins(manifeste, manifeste_path=tmp_path / "manifeste.json")
    for regime_nom, regime_data in manifeste["regimes"].items():
        path_fig = tmp_path / f"arcC_psychometrique_{regime_nom}.png"
        figure_psychometrique(regime_nom, regime_data["sessions"], pins["regimes"][regime_nom],
                              path_fig)
        assert path_fig.exists() and path_fig.stat().st_size > 0
