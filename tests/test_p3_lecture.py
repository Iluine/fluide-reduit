"""P3 — tests de la LECTURE MÉCANIQUE (`scripts/run_p3_lecture.py`).

Source qui fait foi : `claude/prereg-p3-transport-pin.md` (ENDOSSÉ, §A38),
chantier 7 de `mission-prerequis-p2p3.md`.

Ce que ces tests garantissent : **chaque garde sait ÉCHOUER**. Une garde qu'on
n'a vue que passer n'est pas une garde, c'est une décoration — et celle-ci
protège la seule mesure humaine du projet, non répétable à volonté.

Les manifestes sont SYNTHÉTIQUES et construits ici : aucun test ne dépend d'une
session réelle, sauf les deux qui rejouent les pré-vols du 25/07 (artefacts
locaux, `outputs/` étant gitignoré — ils se sautent proprement hors de cette
machine, et la garantie qu'ils illustrent est déjà couverte par les tests
synthétiques)."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from scripts import run_arcC_pins, run_p3_lecture
from scripts.run_arcC_orchestration import PLAN_P3PRIME
from scripts.run_p3_lecture import (CHEMINS_ATTENDUS, LAMBDA_EQUIV, PIN_IC, PIN_JND_SEV,
                                    REGIME_P3, VERDICT_COMPATIBLE, VERDICT_DIFFERENT,
                                    VERDICT_DIFFERENT_ET_BORNE, VERDICT_EQUIVALENT,
                                    VERDICT_INDETERMINEE, VERDICT_NON_DISTINGUE,
                                    garde_chemins, garde_conditions,
                                    garde_conditions_p3prime, garde_validite_c5,
                                    lecture_transport, prononce, prononce_p3prime)
from src.arcC_abx import BASE_SEED_P3PRIME
from src.arcC_scelle import scelle_seuils

# La garde des conditions est ANCRÉE sur `date_session` (§A41) : une
# observation valide porte la date du jour et une heure à ±2 h de l'horodatage
# machine. Les manifestes synthétiques dérivent donc TOUT du même instant.
_MAINTENANT = datetime.now()
CONDITIONS_OK = (f"repère à 750 mm, lumière du jour stable, le {_MAINTENANT:%d/%m/%Y} "
                 f"vers {_MAINTENANT.hour}h{_MAINTENANT.minute:02d}, volets ouverts")


def _fabrique_manifeste(tmp_path: Path, *, seuils=(0.070, 0.075, 0.073, 0.072),
                        conditions=CONDITIONS_OK, chemins=CHEMINS_ATTENDUS,
                        valides=(True, True, True, True), complets=(True, True, True, True),
                        casse_sha=False) -> Path:
    """Écrit un manifeste P3 COMPLET (logs + sidecars réels, sha cohérents) et
    renvoie son chemin. `seuils` est ordonné par staircase : le 3e est celui du
    TÉMOIN, disposition gravée."""
    sessions = []
    for k, (seuil, chemin, valide, complet) in enumerate(zip(seuils, chemins, valides,
                                                             complets)):
        log = tmp_path / f"session_{REGIME_P3}_{k}.jsonl"
        log.write_text(f'{{"indice_essai": {k}}}\n', encoding="utf-8")
        sha = hashlib.sha256(log.read_bytes()).hexdigest()
        if casse_sha and k == 0:
            sha = "0" * 64
        conditions_path = tmp_path / f"session_{REGIME_P3}_{k}.conditions.json"
        conditions_path.write_text(json.dumps(dict(
            numero_staircase=k, regime=REGIME_P3, log_path=str(log),
            provenance_rendu=dict(chemin_rendu=chemin), sha256_log=sha)), encoding="utf-8")
        sessions.append(dict(
            numero_staircase=k, regime=REGIME_P3, log_path=str(log),
            conditions_path=str(conditions_path), chemin_rendu=chemin, seuil=seuil,
            complet=complet, validite=dict(valide=valide)))

    chemin_manifeste = tmp_path / "manifeste_p3.json"
    chemin_manifeste.write_text(json.dumps(dict(
        sujet="humain", date_session=_MAINTENANT.isoformat(),
        conditions_validite=dict(luminosite="OSD 80%", conditions=conditions),
        config_affichage=dict(chemins_staircases=list(chemins)),
        regimes={REGIME_P3: dict(sessions=sessions)})), encoding="utf-8")
    return chemin_manifeste


def _prononce(chemin_manifeste: Path) -> dict:
    manifeste = json.loads(chemin_manifeste.read_text(encoding="utf-8"))
    return prononce(manifeste, chemin_manifeste.parent)


# --- Constantes gravées ------------------------------------------------------


def test_pin_recopie_correspond_au_texte_du_prereg():
    """Le pin est recopié en PLEINE PRÉCISION depuis l'artefact
    `outputs/arcC/pins_spatial.json` ; le prereg l'écrit arrondi (7.33 %,
    [6.03, 8.67]). Ce test lie les deux : si l'artefact et le texte divergeaient
    un jour, la divergence casserait ici plutôt que de vivre en silence."""
    assert round(100.0 * PIN_JND_SEV, 2) == 7.33
    assert [round(100.0 * b, 2) for b in PIN_IC] == [6.03, 8.67]


def test_methode_ic_est_importee_jamais_reimplementee():
    """« méthode gravée, IMPORTÉE, jamais réimplémentée » : ce test vérifie
    l'IDENTITÉ des objets, pas une ressemblance de formule."""
    assert run_p3_lecture.calcule_ic is run_arcC_pins.calcule_ic
    assert run_p3_lecture.calcule_ic_combine is run_arcC_pins.calcule_ic_combine


def test_le_decideur_est_min_max_la_methode_du_pin():
    """§A38-CORRECTION-4 : le DÉCIDEUR est `min/max`, la méthode que l'artefact
    du pin déclare — les deux côtés de la comparaison sont réduits par la MÊME
    méthode, et « même méthode d'IC » redevient vrai au sens littéral.

    Verrou de la bascule : sur trois seuils dont le min/max recouvre l'IC du pin
    mais dont le mean±2SEM, PLUS LARGE, le recouvrirait aussi, on ne saurait pas
    lequel a décidé. On prend donc des seuils où les deux méthodes DIVERGENT :
    min/max disjoint (donc branche 2) alors que mean±2SEM recouvrirait."""
    # Trois seuils dont le PLUS BAS passe juste au-dessus de la borne haute du
    # pin (8.674 %) — min/max donc disjoint —, mais assez dispersés pour que
    # mean±2SEM redescende sous cette borne et recouvre.
    seuils = [0.0868, 0.0900, 0.0932]
    assert run_arcC_pins.calcule_ic(seuils)[0] > PIN_IC[1], "min/max doit etre disjoint"
    assert run_arcC_pins.calcule_ic_combine(seuils)[0] < PIN_IC[1], "2SEM recouvrirait"

    lecture = lecture_transport(seuils)
    assert lecture["verdict"] == VERDICT_DIFFERENT, (
        "c'est le min/max qui doit decider -- si le mean±2SEM decidait encore, "
        "cette lecture rendrait « compatible avec 1 »")
    assert lecture["ic_minmax_decideur"] == run_arcC_pins.calcule_ic(seuils)
    assert lecture["ic_mean2sem_surface"] == run_arcC_pins.calcule_ic_combine(seuils)


# --- Les gardes savent échouer ----------------------------------------------


def _manifeste_conditions(conditions, luminosite="OSD 80%",
                          date_session=None) -> dict:
    return dict(date_session=(date_session or _MAINTENANT.isoformat()),
                conditions_validite=dict(conditions=conditions, luminosite=luminosite))


@pytest.mark.parametrize("conditions", [
    "<repère de distance + lumière du jour stable, tel qu'observé>",
    "<ce que tu OBSERVES : repère, éclairage, heure — sans chevrons>",
    "repère à 750 mm <à compléter>",
    "…ce que tu observes, heure comprise…",       # la chaîne EXACTE du 26/07 (§A40)
    "repère à 750 mm [à compléter], 14h",
    "conditions p. ex. lumière du jour, 14h",
])
def test_garde_conditions_refuse_un_gabarit(conditions):
    """Un GABARIT n'est pas une observation. Les trois premières chaînes sont
    celles des pré-vols réels du 25/07 ; la quatrième est celle qui a TRAVERSÉ
    la garde anti-chevrons le 26/07 (§A40) — les marqueurs sont désormais une
    liste fermée, et la garde porteuse est l'ancrage date/heure."""
    garde = garde_conditions(_manifeste_conditions(conditions))
    assert garde["ok"] is False
    assert "GABARIT" in garde["motif"]


def test_garde_conditions_accepte_une_observation():
    assert garde_conditions(_manifeste_conditions(CONDITIONS_OK))["ok"] is True


def test_garde_conditions_refuse_une_absence():
    """Absente vaut gabarit : §C9 exige des conditions consignées, et « rien »
    n'est pas une observation non plus. Un manifeste SANS `date_session` est
    refusé aussi : rien pour ancrer la garde."""
    assert garde_conditions(_manifeste_conditions(None))["ok"] is False
    assert garde_conditions({})["ok"] is False


def test_garde_conditions_refuse_une_heure_incoherente():
    """LE cas §A41 : session à 00h27 sous une chaîne qui dit « 14h » — la
    session de nuit consignée « lumière du jour » meurt ici, mécaniquement.
    C'est le falsificateur le moins cher de la review du 29/07."""
    nuit = _MAINTENANT.replace(hour=0, minute=27)
    conditions = f"lumière du jour stable, le {nuit:%d/%m/%Y} vers 14h00, volets ouverts"
    garde = garde_conditions(_manifeste_conditions(conditions,
                                                   date_session=nuit.isoformat()))
    assert garde["ok"] is False
    assert "moins de 2 h" in garde["motif"]


def test_garde_conditions_refuse_la_date_d_hier():
    """Le copié-collé d'une VRAIE observation d'hier (bien formée, heure
    plausible) meurt sur la date — c'est ce que la version morphologique ne
    pouvait pas voir."""
    hier = _MAINTENANT - timedelta(days=1)
    conditions = (f"repère à 750 mm, lumière du jour stable, le {hier:%d/%m/%Y} "
                  f"vers {_MAINTENANT.hour}h{_MAINTENANT.minute:02d}")
    garde = garde_conditions(_manifeste_conditions(conditions))
    assert garde["ok"] is False
    assert "DATE du jour" in garde["motif"]


def test_144hz_ne_fabrique_pas_une_heure_fantome():
    """« écran 144hz » n'est pas une heure (M2) : le motif est ancré des deux
    côtés, sinon `44h`/`4h` matcherait dans un mot technique."""
    conditions = f"écran 144hz, le {_MAINTENANT:%d/%m/%Y}, volets ouverts"
    garde = garde_conditions(_manifeste_conditions(conditions))
    assert garde["heures_declarees_minutes"] == []
    assert garde["ok"] is False


def test_garde_conditions_verifie_aussi_luminosite():
    """Le champ `luminosite` n'était JAMAIS vérifié (M2) : un gabarit peut s'y
    loger aussi."""
    garde = garde_conditions(_manifeste_conditions(CONDITIONS_OK,
                                                   luminosite="[à remplir]"))
    assert garde["ok"] is False
    assert "GABARIT" in garde["motif"]


@pytest.mark.parametrize("chemins", [
    ("r1", "r1", "r1", "viridis"),        # témoin en DERNIÈRE position
    ("viridis", "r1", "r1", "r1"),        # témoin en PREMIÈRE
    ("r1", "r1", "viridis"),              # pas de R1 après le témoin
    None,                                  # campagne sans chemins consignés
])
def test_garde_chemins_refuse_toute_autre_disposition(chemins):
    """La disposition est GRAVÉE (r1, r1, viridis, r1). Toute autre mesurerait
    autre chose sous le même nom — un témoin en fin de session confronte un
    sujet fatigué à trois staircases fraîches."""
    manifeste = dict(config_affichage=dict(
        chemins_staircases=(list(chemins) if chemins else None)))
    assert garde_chemins(manifeste)["ok"] is False


def test_garde_chemins_accepte_la_disposition_gravee():
    assert garde_chemins(dict(config_affichage=dict(
        chemins_staircases=list(CHEMINS_ATTENDUS))))["ok"] is True


def test_garde_validite_c5_refuse_une_staircase_invalide():
    """Une seule staircase invalide suffit : la moyenne d'un bras contenant une
    staircase invalide n'est pas un seuil, c'est un mélange."""
    sessions = [dict(numero_staircase=k, validite=dict(valide=(k != 2))) for k in range(4)]
    garde = garde_validite_c5(sessions)
    assert garde["ok"] is False and garde["staircases_invalides"] == [2]
    assert garde_validite_c5([])["ok"] is False


def test_garde_sidecar_refuse_un_sha_qui_ne_correspond_pas(tmp_path):
    """LA garde de la liaison sidecar↔log (chantier 3) : un log qui n'est pas
    celui pour lequel le sidecar a été écrit devient visible. Sans elle,
    l'appariement reposerait sur le NOM DE FICHIER."""
    manifeste = _fabrique_manifeste(tmp_path, casse_sha=True)
    rapport = _prononce(manifeste)
    liaison = next(g for g in rapport["gardes"]["gardes"] if g["nom"] == "liaison_sidecar_log")
    assert liaison["ok"] is False
    assert liaison["details"][0]["motif"].startswith("sha256_log")
    assert rapport["verdict"] == VERDICT_INDETERMINEE


def test_toutes_les_gardes_sont_evaluees_pas_court_circuitees(tmp_path):
    """Un rapport qui s'arrête à la première garde tombée cache les suivantes,
    et la session serait relancée UNE FOIS PAR GARDE — sur une mesure humaine
    non répétable à volonté, c'est le pire des comportements."""
    manifeste = _fabrique_manifeste(tmp_path, conditions="<gabarit>",
                                    chemins=("r1", "viridis", "r1", "r1"), casse_sha=True)
    rapport = _prononce(manifeste)
    assert set(rapport["gardes"]["echouees"]) == {
        "conditions_ancrees_date_session", "ordre_des_chemins", "liaison_sidecar_log"}
    assert len(rapport["gardes"]["gardes"]) == 4      # la 4e a bien été évaluée aussi


# --- Le témoin passe AVANT les seuils R1 ------------------------------------


def test_temoin_hors_ic_rend_indeterminee_SANS_lire_les_seuils_r1(tmp_path):
    """Le cœur du chantier : témoin hors IC ⇒ INDÉTERMINÉE, et les seuils R1
    ne sont ni imprimés NI ÉCRITS.

    Écrire les seuils dans le JSON en s'abstenant de les imprimer ne
    protégerait de rien : un fichier se lit. Quand l'instrument est invalide,
    le résultat ne doit être consultable NULLE PART — c'est la seule protection
    efficace contre soi-même."""
    manifeste = _fabrique_manifeste(tmp_path, seuils=(0.070, 0.075, 0.20, 0.072))
    rapport = _prononce(manifeste)
    assert rapport["gardes"]["toutes_ok"] is True
    assert rapport["verdict"] == VERDICT_INDETERMINEE
    assert rapport["temoin"]["ok"] is False
    assert "r1" not in rapport
    assert "0.07" not in json.dumps(rapport["temoin"])   # aucun seuil R1 par ricochet
    assert "0.075" not in json.dumps(rapport)


@pytest.mark.parametrize("seuil_temoin", [PIN_IC[0], PIN_IC[1]])
def test_temoin_aux_bornes_de_lic_est_valide(tmp_path, seuil_temoin):
    """Bornes INCLUSES : l'IC gravé est un intervalle fermé, et un témoin pile
    à la borne n'est pas une dérive."""
    manifeste = _fabrique_manifeste(tmp_path, seuils=(0.070, 0.075, seuil_temoin, 0.072))
    assert _prononce(manifeste)["temoin"]["ok"] is True


def test_temoin_sur_escalier_non_convergent_rend_indeterminee(tmp_path):
    """Prereg P3, lecture 3 : escalier non convergent ⇒ INDÉTERMINÉ. Le témoin
    incomplet n'a pas de seuil comparable, et rien n'est lu ensuite."""
    manifeste = _fabrique_manifeste(tmp_path, complets=(True, True, False, True))
    rapport = _prononce(manifeste)
    assert rapport["verdict"] == VERDICT_INDETERMINEE
    assert "r1" not in rapport


# --- Les deux branches, sur manifestes synthétiques -------------------------


def test_branche_1_transport_compatible_avec_1(tmp_path):
    """Recouvrement d'IC ⇒ branche 1, avec son ANTI-SURCLAME : la seconde
    condition de D14 (pondération plate) reste NON MESURÉE, et la scission ne
    se referme pas ici."""
    manifeste = _fabrique_manifeste(tmp_path, seuils=(0.070, 0.075, 0.073, 0.072))
    rapport = _prononce(manifeste)
    assert rapport["verdict"] == VERDICT_COMPATIBLE and rapport["branche"] == "1"
    assert "NON MESUREE" in rapport["r1"]["texte"]
    assert "r_fovea" in rapport["r1"]["texte"]


def test_branche_2_transport_different_de_1(tmp_path):
    """IC disjoints ⇒ branche 2, avec direction et facteur T = jnd_R1 / pin,
    IC par les bornes. Et l'anti-surclame inverse : AUCUN seuil d'état ne
    bouge, jnd_sev^R1 est un pin NOUVEAU."""
    seuils_r1 = (0.150, 0.152, 0.151)
    manifeste = _fabrique_manifeste(
        tmp_path, seuils=(seuils_r1[0], seuils_r1[1], 0.073, seuils_r1[2]))
    rapport = _prononce(manifeste)
    assert rapport["verdict"] == VERDICT_DIFFERENT and rapport["branche"] == "2"
    r1 = rapport["r1"]
    assert r1["transport_T"] == pytest.approx(r1["jnd_r1"] / PIN_JND_SEV)
    assert r1["transport_T"] > 1.0 and "PLUS HAUT" in r1["direction"]
    # « IC par les bornes » : celles de l'IC DÉCIDEUR (min/max), pas du diagnostic.
    assert r1["transport_T_ic"] == pytest.approx(
        [b / PIN_JND_SEV for b in r1["ic_minmax_decideur"]])
    assert "AUCUN seuil d'etat ne bouge" in r1["texte"]
    assert "pin NOUVEAU" in r1["texte"]


def test_bras_r1_disperse_rend_indeterminee():
    """GARDE DE DISPERSION (§C5, review 29/07 M3) : CV > 30 % ⇒ INDÉTERMINÉE.
    La méthodologie du pin (`evalue_dispersion`) aurait rendu un tel bras
    INDÉTERMINÉ — il ne se lit pas contre le pin. Les seuils sont EXACTEMENT
    ceux du bras R1 du 26/07 (CV ≈ 42.8 %) : si cette garde avait existé et que
    le témoin était passé, P3 aurait quand même refusé de prononcer."""
    lecture = lecture_transport([0.025914125248894065, 0.015443609037015341,
                                 0.03799628704481502])
    assert lecture["verdict"] == VERDICT_INDETERMINEE and lecture["branche"] == "3"
    assert lecture["dispersion"]["coherent"] is False
    assert lecture["dispersion"]["cv"] > 0.30
    assert "DISPERSION" in lecture["motif"]


def test_bras_r1_coherent_surface_sa_dispersion(tmp_path):
    """Sur un bras sain, la dispersion est SURFACÉE au rapport (cv, seuil,
    coherent=True) — un chiffre qu'on ne voit jamais n'est pas une garde."""
    manifeste = _fabrique_manifeste(tmp_path, seuils=(0.070, 0.075, 0.073, 0.072))
    r1 = _prononce(manifeste)["r1"]
    assert r1["dispersion"]["coherent"] is True
    assert r1["dispersion"]["cv"] <= 0.30


@pytest.mark.parametrize("seuils", [[], [0.07], [0.07, 0.075], [0.07, 0.072, 0.074, 0.076]])
def test_bras_r1_ampute_ou_surnumeraire_rend_indeterminee(seuils):
    """Le prereg lit « l'IC des TROIS staircases R1 ». Deux ne suffisent pas :
    un min/max sur un bras amputé est un intervalle sur autre chose, et un seul
    seuil rendrait l'intervalle dégénéré `[x, x]` — un IC ne se FABRIQUE pas.

    Le cas à 2 seuils est le PIÈGE de la bascule : `calcule_ic` aurait rendu un
    intervalle parfaitement bien formé, là où `calcule_ic_combine` refusait de
    lui-même (SEM incalculable). En passant au min/max, ce refus devait être
    porté explicitement — sans ce test, la bascule aurait silencieusement rendu
    lisible un bras à deux staircases."""
    lecture = lecture_transport(seuils)
    assert lecture["verdict"] == VERDICT_INDETERMINEE and lecture["branche"] == "3"
    assert lecture["ic"] is None


def test_ic_mean2sem_est_surface_mais_jamais_juge(tmp_path):
    """Le `mean±2SEM` reste au rapport en DIAGNOSTIC (§A38-CORRECTION-4) : il
    documente ce que l'autre méthode aurait donné et garde visible, dans chaque
    lecture, la trace de la lettre fautive du prereg.

    Ce test verrouille qu'il est présent, distinct du décideur, et accompagné
    de son texte — sinon la correction deviendrait invisible une fois passée."""
    manifeste = _fabrique_manifeste(tmp_path, seuils=(0.070, 0.075, 0.073, 0.072))
    r1 = _prononce(manifeste)["r1"]
    assert r1["ic_minmax_decideur"] == [0.070, 0.075]
    assert r1["ic_mean2sem_surface"] != r1["ic_minmax_decideur"]
    assert r1["methode_ic"].startswith("min_max_seuils")
    assert "DECIDEUR = min/max" in r1["diagnostic_ic"]
    assert "[5.81, 8.86]" in r1["diagnostic_ic"]


# --- Les deux pré-vols du 25/07, dans leur rôle posthume ---------------------


CHEMINS_PREVOLS = [
    Path("outputs/arcC/p3_prevol_2026-07-25/manifeste_campagne.json"),
    Path("outputs/arcC/p3_prevol2_2026-07-25/manifeste_p3.json"),
]


@pytest.mark.parametrize("chemin", CHEMINS_PREVOLS, ids=["prevol1", "prevol2"])
def test_les_prevols_du_25_07_sont_refuses_par_la_garde_des_conditions(chemin):
    """Rôle posthume des deux pré-vols : être les champs d'essai de la garde 1.
    Tous deux portent un GABARIT à `conditions` et doivent être REFUSÉS —
    verdict INDÉTERMINÉE, aucun seuil lu.

    Artefacts LOCAUX (`outputs/` est gitignoré) : le test se saute ailleurs.
    La garantie qu'il illustre ne dépend PAS de lui — elle est couverte par
    `test_garde_conditions_refuse_un_gabarit`, qui rejoue leurs deux chaînes
    exactes sur des manifestes synthétiques."""
    complet = Path(run_p3_lecture.ROOT) / chemin
    if not complet.exists():
        pytest.skip(f"artefact local absent (outputs/ gitignoré) : {chemin}")
    rapport = prononce(json.loads(complet.read_text(encoding="utf-8")), complet.parent)
    assert "conditions_ancrees_date_session" in rapport["gardes"]["echouees"]
    assert rapport["verdict"] == VERDICT_INDETERMINEE
    assert "r1" not in rapport and "temoin" not in rapport


# =============================================================================
# P3′ (mission chantier 8c) — champs d'essai IMPOSÉS par le prereg v2.2 (§A42)
# =============================================================================

# Instant de session ÉPINGLÉ en pleine plage [09:00, 19:00] : un test ne
# dépend pas de l'heure à laquelle la suite court (clause (d), §A42 [v2.1]).
_INSTANT_P3P = _MAINTENANT.replace(hour=14, minute=30)
CONDITIONS_P3P = (f"repère à 750 mm, lumière du jour stable, le {_INSTANT_P3P:%d/%m/%Y} "
                  f"vers {_INSTANT_P3P.hour}h{_INSTANT_P3P.minute:02d}, volets ouverts")

# Bras de référence : le cas 2-équiv GRAVÉ au prereg v2.2 (ligne « champs
# d'essai ») — IC disjoints, IC_T = [1.111, 1.186] ⊂ [1/1.5, 1.5].
SEUILS_V_2EQUIV = (0.070, 0.071, 0.072)
SEUILS_R_2EQUIV = (0.080, 0.081, 0.083)


def _fabrique_manifeste_p3prime(tmp_path: Path, *, seuils_v=SEUILS_V_2EQUIV,
                                seuils_r=SEUILS_R_2EQUIV,
                                base_seed=BASE_SEED_P3PRIME, chemins=PLAN_P3PRIME,
                                conditions=CONDITIONS_P3P, date_session=None,
                                complets=(True,) * 6, valides=(True,) * 6,
                                casse_sha=False, protocole="p3prime") -> Path:
    """Manifeste P3′ synthétique COMPLET : six staircases au plan D-P3′-1,
    logs + sidecars réels (sha cohérents), seuils SCELLÉS (jamais en clair au
    manifeste — le schéma du chantier 8a.5)."""
    instant = date_session or _INSTANT_P3P
    it_v, it_r = iter(seuils_v), iter(seuils_r)
    seuils_scelles: dict[str, float | None] = {}
    sessions = []
    for k, (chemin, valide, complet) in enumerate(zip(chemins, valides, complets)):
        seuil = next(it_v) if chemin == "viridis" else next(it_r)
        seuils_scelles[str(k)] = float(seuil) if complet else None
        log = tmp_path / f"session_{REGIME_P3}_{k}.jsonl"
        log.write_text(f'{{"indice_essai": {k}}}\n', encoding="utf-8")
        sha = hashlib.sha256(log.read_bytes()).hexdigest()
        if casse_sha and k == 0:
            sha = "0" * 64
        conditions_path = tmp_path / f"session_{REGIME_P3}_{k}.conditions.json"
        conditions_path.write_text(json.dumps(dict(
            numero_staircase=k, regime=REGIME_P3, log_path=str(log),
            provenance_rendu=dict(chemin_rendu=chemin), sha256_log=sha)), encoding="utf-8")
        sessions.append(dict(
            numero_staircase=k, regime=REGIME_P3, log_path=str(log),
            conditions_path=str(conditions_path), chemin_rendu=chemin,
            complet=complet, validite=dict(valide=valide)))

    chemin_scelle = tmp_path / "seuils_scelles_p3prime.json"
    sha_scelle = scelle_seuils(chemin_scelle, {REGIME_P3: seuils_scelles})
    chemin_manifeste = tmp_path / "manifeste_p3prime.json"
    chemin_manifeste.write_text(json.dumps(dict(
        sujet="humain", protocole=protocole, date_session=instant.isoformat(),
        base_seed=base_seed,
        conditions_validite=dict(luminosite="OSD 80%", conditions=conditions),
        config_affichage=dict(chemins_staircases=list(chemins)),
        scellement_seuils=dict(chemin=str(chemin_scelle), sha256=sha_scelle),
        regimes={REGIME_P3: dict(sessions=sessions)})), encoding="utf-8")
    return chemin_manifeste


def _prononce_p3p(chemin_manifeste: Path) -> dict:
    manifeste = json.loads(chemin_manifeste.read_text(encoding="utf-8"))
    return prononce_p3prime(manifeste, chemin_manifeste.parent)


# --- Les branches, toutes ATTEIGNABLES (partition close, bornes incluses) ----


def test_p3prime_branche_1b_recouvrement_et_inclusion(tmp_path):
    """1b : recouvrement ET IC_T ⊂ [1/λ, λ] ⇒ ÉQUIVALENT à 1 à λ près — la
    SEULE branche qui établit la condition D14."""
    rapport = _prononce_p3p(_fabrique_manifeste_p3prime(
        tmp_path, seuils_v=(0.070, 0.075, 0.072), seuils_r=(0.071, 0.074, 0.073)))
    assert rapport["verdict"] == VERDICT_EQUIVALENT and rapport["branche"] == "1b"
    assert rapport["transport"]["recouvrement"] is True
    assert rapport["transport"]["inclusion_lambda"] is True
    assert "ÉTABLIT" in rapport["texte"] and "anti-surclame" in rapport["texte"].lower()


def test_p3prime_branche_1a_recouvrement_sans_inclusion(tmp_path):
    """1a : recouvrement SANS inclusion ⇒ NON DISTINGUÉ de 1 — un non-rejet
    n'établit RIEN (le cas-frontière du prereg : jamais 1b)."""
    rapport = _prononce_p3p(_fabrique_manifeste_p3prime(
        tmp_path, seuils_v=(0.040, 0.052, 0.046), seuils_r=(0.050, 0.062, 0.056)))
    assert rapport["verdict"] == VERDICT_NON_DISTINGUE and rapport["branche"] == "1a"
    assert rapport["transport"]["recouvrement"] is True
    assert rapport["transport"]["inclusion_lambda"] is False
    assert "N'EST PAS" in rapport["texte"]


def test_p3prime_branche_2_disjoints_sans_inclusion(tmp_path):
    """2 : IC disjoints sans inclusion ⇒ TRANSPORT ≠ 1, direction et facteur."""
    rapport = _prononce_p3p(_fabrique_manifeste_p3prime(
        tmp_path, seuils_v=(0.030, 0.031, 0.032), seuils_r=(0.060, 0.061, 0.063)))
    assert rapport["verdict"] == VERDICT_DIFFERENT and rapport["branche"] == "2"
    assert rapport["transport_T"] == pytest.approx(
        (0.060 + 0.061 + 0.063) / 3 / ((0.030 + 0.031 + 0.032) / 3))
    assert "PLUS HAUT" in rapport["direction"]


def test_p3prime_branche_2equiv_le_cas_grave_du_prereg(tmp_path):
    """LE cas 2-équiv GRAVÉ (§A42-COMPLÉMENT, résolution β) : V = {0.070,
    0.071, 0.072}, R = {0.080, 0.081, 0.083} ⇒ EXACTEMENT le prononcé double,
    jamais 1b ni 2 seule."""
    rapport = _prononce_p3p(_fabrique_manifeste_p3prime(tmp_path))
    assert rapport["verdict"] == VERDICT_DIFFERENT_ET_BORNE
    assert rapport["branche"] == "2-equiv"
    assert rapport["transport"]["recouvrement"] is False
    assert rapport["transport"]["inclusion_lambda"] is True
    # Le prononcé double voyage ENTIER : les deux moitiés dans le texte.
    assert "≠ 1" in rapport["texte"] and "λ" in rapport["texte"]
    assert "PRONONCÉ DOUBLE" in rapport["texte"]


def test_p3prime_cas_frontiere_de_borne_ic_t_exactement_lambda(tmp_path):
    """CONVENTION DE BORNE gravée (mission 8b) : IC_T = [x, 1.5] EXACTEMENT ⇒
    cellule d'inclusion, bornes INCLUSES. Valeurs binaires exactes
    (0.09375/0.0625 == 1.5 exact en f64) pour que le test frappe la borne,
    pas son voisinage."""
    rapport = _prononce_p3p(_fabrique_manifeste_p3prime(
        tmp_path, seuils_v=(0.0625, 0.063, 0.0635), seuils_r=(0.092, 0.093, 0.09375)))
    assert rapport["transport"]["ic_T"][1] == LAMBDA_EQUIV  # borne frappée exactement
    assert rapport["transport"]["inclusion_lambda"] is True
    assert rapport["verdict"] == VERDICT_DIFFERENT_ET_BORNE  # disjoints ⇒ 2-équiv


def test_p3prime_branche_3_atteignable(tmp_path):
    """3 : une garde rouge ⇒ INDÉTERMINÉE (ici : bras amputé)."""
    rapport = _prononce_p3p(_fabrique_manifeste_p3prime(
        tmp_path, complets=(True, True, False, True, True, True)))
    assert rapport["verdict"] == VERDICT_INDETERMINEE and rapport["branche"] == "3"
    assert "bras_complets_3_3" in rapport["gardes"]["echouees"]


# --- Dispersion : CV <= 30 % PAR BRAS, les deux bras (M3 / §C5) --------------


@pytest.mark.parametrize("bras_disperse", ["viridis", "r1"])
def test_p3prime_cv_hors_clous_rend_indeterminee_sans_divulguer(tmp_path, bras_disperse):
    """CV > 30 % sur UN bras (l'un puis l'autre) ⇒ INDÉTERMINÉE, et les
    seuils ne sont NI imprimés NI écrits — seul le CV (un ratio) est surfacé.
    Les seuils dispersés sont ceux du bras R1 réel du 26/07 (CV ≈ 42.8 %)."""
    disperses = (0.0259, 0.0154, 0.0380)
    kwargs = {"seuils_v": disperses} if bras_disperse == "viridis" else {"seuils_r": disperses}
    rapport = _prononce_p3p(_fabrique_manifeste_p3prime(tmp_path, **kwargs))
    assert rapport["verdict"] == VERDICT_INDETERMINEE
    assert rapport["dispersion_par_bras"][bras_disperse]["coherent"] is False
    assert rapport["dispersion_par_bras"][bras_disperse]["cv"] > 0.30
    dump = json.dumps(rapport)
    for seuil in disperses:
        assert str(seuil) not in dump, "un seuil a fuité dans le rapport INDÉTERMINÉE"
    assert "bras" not in rapport and "transport" not in rapport


# --- Garde graine à la LECTURE : égalité SEULE (§A42-COMPLÉMENT) -------------


@pytest.mark.parametrize("graine", [12345, 20260705])
def test_p3prime_graine_ni_brulee_ni_gravee_refusee_par_egalite(tmp_path, graine):
    """12345 (quelconque) ET 20260705 (brûlée) tombent par la MÊME garde :
    l'égalité au gravé — la moitié « égalité » de la garde double."""
    rapport = _prononce_p3p(_fabrique_manifeste_p3prime(tmp_path, base_seed=graine))
    assert rapport["verdict"] == VERDICT_INDETERMINEE
    assert "graine_egalite_gravee" in rapport["gardes"]["echouees"]


def test_p3prime_archive_reste_lisible_apres_brulage(tmp_path, monkeypatch):
    """LE TEST DU BRÛLAGE (§A42-COMPLÉMENT) : une archive à 20260729 avec
    20260729 AJOUTÉ à la liste des brûlées reste LISIBLE en lecture — la
    liste ne mord qu'à l'entrée, sinon l'acte de brûlage rendrait l'archive
    P3′ illisible (la re-dérivabilité promise)."""
    import src.arcC_abx as abx
    monkeypatch.setattr(abx, "GRAINES_BRULEES",
                        frozenset({20260705, BASE_SEED_P3PRIME}))
    rapport = _prononce_p3p(_fabrique_manifeste_p3prime(tmp_path))
    assert "graine_egalite_gravee" not in rapport["gardes"]["echouees"]
    assert rapport["verdict"] == VERDICT_DIFFERENT_ET_BORNE  # la lecture va au bout


# --- Gardes : toutes évaluées ; conditions (a)-(d) ; protocole ---------------


def test_p3prime_toutes_les_gardes_sont_evaluees(tmp_path):
    """§A38-CORRECTION-4 (choix endossé) : un rapport qui s'arrête à la
    première garde tombée cache les suivantes — ici QUATRE tombent ensemble
    et toutes sont rapportées."""
    chemin = _fabrique_manifeste_p3prime(
        tmp_path, base_seed=12345, conditions="…gabarit…", casse_sha=True,
        chemins=("r1", "r1", "viridis", "r1", "viridis", "viridis"))
    rapport = _prononce_p3p(chemin)
    assert rapport["verdict"] == VERDICT_INDETERMINEE
    assert {"graine_egalite_gravee", "conditions_ancrees_a_d", "plan_d_p3prime_1",
            "liaison_sidecar_log"} <= set(rapport["gardes"]["echouees"])
    assert len(rapport["gardes"]["gardes"]) == 7  # toutes évaluées, aucune sautée


def test_p3prime_clause_d_seule_refuse_une_session_veridique_de_nuit(tmp_path):
    """LE test qui distingue la plage du simple contrôle d'honnêteté : une
    session nocturne VÉRIDIQUE (« 00h30, plafonnier », date et heure honnêtes
    et cohérentes) passe (a)-(c) INTÉGRALEMENT et meurt par (d) SEULE."""
    nuit = _MAINTENANT.replace(hour=0, minute=30)
    conditions = (f"repère à 750 mm, plafonnier, le {nuit:%d/%m/%Y} vers 0h30, "
                  "volets fermés")
    rapport = _prononce_p3p(_fabrique_manifeste_p3prime(
        tmp_path, conditions=conditions, date_session=nuit))
    garde = next(g for g in rapport["gardes"]["gardes"]
                 if g["nom"] == "conditions_ancrees_a_d")
    assert garde["ok"] is False
    assert "clause (d)" in garde["motif"] and "plage" in garde["motif"]
    assert garde["motif"].count(";") == 0, "la clause (d) doit être le SEUL motif"
    assert rapport["verdict"] == VERDICT_INDETERMINEE


def test_p3prime_refuse_un_manifeste_historique(tmp_path):
    """Un manifeste P3 HISTORIQUE (sans protocole, sans scellement) lu en
    mode P3′ meurt à la garde de protocole — à voix haute, pas par KeyError."""
    chemin = _fabrique_manifeste(tmp_path)  # le fabricateur P3 historique
    rapport = _prononce_p3p(chemin)
    assert rapport["verdict"] == VERDICT_INDETERMINEE
    assert "protocole_p3prime" in rapport["gardes"]["echouees"]


def test_p3prime_session_p3_reelle_refusee_par_trois_chemins():
    """La session P3 du 26/07, champ d'essai posthume (mission 8c) : REFUSÉE
    par TROIS chemins indépendants — (b) aucune heure lisible, (c) ellipses,
    (d) 00h27 hors plage — chacun suffisant seul, les trois présents au motif.

    Artefact LOCAL (outputs/ gitignoré) : le test se saute ailleurs ; la
    garantie est couverte par les tests synthétiques ci-dessus."""
    complet = Path(run_p3_lecture.ROOT) / "outputs/arcC/manifeste_p3.json"
    if not complet.exists():
        pytest.skip("artefact local absent (outputs/ gitignoré) : manifeste_p3.json")
    garde = garde_conditions_p3prime(json.loads(complet.read_text(encoding="utf-8")))
    assert garde["ok"] is False
    assert "HEURE" in garde["motif"]            # (b)
    assert "GABARIT" in garde["motif"]          # (c)
    assert "clause (d)" in garde["motif"]       # (d)


def test_p3prime_diagnostics_surfaces_jamais_juges(tmp_path):
    """Sur un prononcé abouti : dérive viridis-du-jour vs pin gravé et
    seuils-par-position sont SURFACÉS (diagnostic), l'ordre des positions
    est celui du plan D-P3′-1."""
    rapport = _prononce_p3p(_fabrique_manifeste_p3prime(tmp_path))
    diag = rapport["diagnostics"]
    assert diag["derive_viridis_vs_pin"]["pin_grave"] == PIN_JND_SEV
    assert "jamais un critère" in diag["derive_viridis_vs_pin"]["note"]
    positions = [d["chemin"] for d in diag["seuils_par_position"]]
    assert positions == list(PLAN_P3PRIME)


# --- Correctifs de la revue de remise (§A42-COMPLÉMENT-2, prereg [v2.3]) ----


def test_p3prime_surface_les_pauses_inter_staircases(tmp_path):
    """M2 (mission 8a.6) : la lecture SURFACE la fenêtre temporelle de la
    session — durée de chaque staircase et pauses entre elles — en
    DIAGNOSTIC. JAMAIS JUGÉE : ici les pauses valent 5 min et le verdict est
    exactement celui que le même manifeste rend sans elles (cas 2-équiv
    gravé). Aucune borne n'existe, aucune garde n'en dépend."""
    chemin = _fabrique_manifeste_p3prime(tmp_path)
    manifeste = json.loads(chemin.read_text(encoding="utf-8"))
    sessions = manifeste["regimes"][REGIME_P3]["sessions"]
    for k, session in enumerate(sessions):
        debut = _INSTANT_P3P + timedelta(minutes=20 * k)
        session["horodatage_debut"] = debut.isoformat()
        session["horodatage_fin"] = (debut + timedelta(minutes=15)).isoformat()
        session["duree_s"] = 900.0

    rapport = prononce_p3prime(manifeste, chemin.parent)
    temps = rapport["diagnostics"]["temps_de_session"]
    assert [d["duree_s"] for d in temps["duree_par_staircase"]] == [900.0] * 6
    assert [p["entre"] for p in temps["pauses_inter_staircases"]] == [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]
    assert [p["duree_s"] for p in temps["pauses_inter_staircases"]] == [300.0] * 5
    assert "jamais jugées" in temps["note"]
    assert rapport["verdict"] == VERDICT_DIFFERENT_ET_BORNE


def test_p3prime_sans_horodatage_ne_fabrique_aucune_duree(tmp_path):
    """Le pendant du précédent : sur un manifeste SANS horodatages (tout
    manifeste d'avant ce chantier), la lecture rend des durées VIDES plutôt
    que des durées inventées — et le prononcé est le même."""
    rapport = _prononce_p3p(_fabrique_manifeste_p3prime(tmp_path))
    temps = rapport["diagnostics"]["temps_de_session"]
    assert temps["pauses_inter_staircases"] == []
    assert all(d["duree_s"] is None for d in temps["duree_par_staircase"])
    assert rapport["verdict"] == VERDICT_DIFFERENT_ET_BORNE


def test_run_p3_lecture_ne_lie_pas_graines_brulees():
    """(m2) VERROU D'IMPORT, prouvé par INSPECTION — pas par monkeypatch.

    §A42-COMPLÉMENT grave que la liste des brûlées ne mord QU'À L'ENTRÉE :
    si elle mordait aussi à la lecture, l'acte de brûler 20260729 après la
    session rendrait l'archive P3′ ILLISIBLE, et la re-dérivabilité promise
    tomberait avec elle.

    Pourquoi une inspection et non un `monkeypatch.setattr` : sous
    `from src.arcC_abx import GRAINES_BRULEES`, la lecture tiendrait sa
    PROPRE référence — patcher l'attribut du module source ne la changerait
    pas, et le test passerait au vert en ne prouvant rien. On vérifie donc
    (1) qu'aucun nom `GRAINES_BRULEES` n'est LIÉ dans le module, (2) que le
    symbole n'apparaît nulle part dans sa source — ce qui couvre aussi un
    import tardif à l'intérieur d'une fonction, invisible au niveau module."""
    import inspect

    assert not hasattr(run_p3_lecture, "GRAINES_BRULEES"), (
        "run_p3_lecture LIE GRAINES_BRULEES : la liste mordrait à la lecture et "
        "l'archive P3′ deviendrait illisible dès le brûlage (§A42-COMPLÉMENT).")
    source = inspect.getsource(run_p3_lecture)
    assert "GRAINES_BRULEES" not in source, (
        "le symbole GRAINES_BRULEES apparaît dans la source de run_p3_lecture -- "
        "même dans un import tardif ou un commentaire, il n'a rien à y faire : la "
        "garde de lecture est l'ÉGALITÉ SEULE au base_seed gravé.")
    # Et le pendant positif : l'ENTRÉE, elle, la lie bel et bien.
    import scripts.run_arcC_orchestration as orchestration
    assert "GRAINES_BRULEES" in inspect.getsource(orchestration)
