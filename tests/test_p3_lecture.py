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
from scripts.run_p3_lecture import (CHEMINS_ATTENDUS, PIN_IC, PIN_JND_SEV, REGIME_P3,
                                    VERDICT_COMPATIBLE, VERDICT_DIFFERENT,
                                    VERDICT_INDETERMINEE, garde_chemins, garde_conditions,
                                    garde_validite_c5, lecture_transport, prononce)

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
