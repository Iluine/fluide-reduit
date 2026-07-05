"""Tests Manche 2 (§A13) Task 1 -- SONDE (`scripts/run_arcA_m2_sonde.py`,
brief `.superpowers/sdd/m2-task1-brief.md`).

Ces tests sont RAPIDES et n'exécutent JAMAIS la chaîne réelle
(`chaine_verite`/`chaine_registre`, ~3-4 min) : ils testent uniquement la
lecture pré-écrite mécanique `lecture_sonde` (fonction PURE) sur des séries
Δχ_i SYNTHÉTIQUES, plus un contrôle statique (pas de timestamp calendaire)
sur le JSON déjà produit par l'exécution réelle unique (skipif si absent)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from scripts.run_arcA_m2_sonde import (
    OUT_JSON_PATH,
    charge_pins_severe,
    lecture_sonde,
)

ROOT = Path(__file__).resolve().parents[1]

# JND sévère gravé (`outputs/arcC/pins_spatial.json` -> regimes.severe.jnd),
# réutilisé TEL QUEL par les séries synthétiques ci-dessous (mêmes valeurs
# que les exemples du brief §"Tests").
_JND_SEV = 0.07334065957385666


# --- 1. lecture_sonde -- trois branches sur séries synthétiques --------------


def test_lecture_sonde_branche_s1():
    """S1 : Δχ_i traverse jnd_sev dès i=2 (« ex. [0, 0.08, 0.09, ...] avec jnd
    0.0733 » du brief) -- composition massive, remontée dès les premières
    émissions."""
    delta_chi = [0.0, 0.08, 0.09, 0.02, 0.01, 0.005]
    resultat = lecture_sonde(delta_chi, _JND_SEV)
    assert resultat["lecture"] == "S1"
    assert resultat["s1"] is True
    assert resultat["traverse_jnd_sev"] == [False, True, True, False, False, False]


def test_lecture_sonde_branche_s2():
    """S2 : série plate SOUS jnd_sev, aucune émission (i>=2) ne dépasse le
    pin, et le plateau tardif (émissions 4-6) ne dépasse pas 1.5x le plateau
    précoce (émissions 2-3) -- borné/contractant, la grille s'achète."""
    delta_chi = [0.0, 0.010, 0.012, 0.011, 0.013, 0.012]
    resultat = lecture_sonde(delta_chi, _JND_SEV)
    assert resultat["lecture"] == "S2"
    assert resultat["s1"] is False
    assert resultat["s2"] is True
    assert all(t is False for t in resultat["traverse_jnd_sev"])


def test_lecture_sonde_branche_autre():
    """AUTRE : croissance sous JND (aucune émission ne traverse jnd_sev) mais
    violant le facteur de contraction 1.5 -- série verbatim du brief
    (« ex. [0, 0.01, 0.01, 0.05, 0.06, 0.07] »)."""
    delta_chi = [0.0, 0.01, 0.01, 0.05, 0.06, 0.07]
    resultat = lecture_sonde(delta_chi, _JND_SEV)
    assert resultat["lecture"] == "AUTRE"
    assert resultat["s1"] is False
    assert resultat["s2"] is False
    assert all(t is False for t in resultat["traverse_jnd_sev"])
    # Le ressaut émissions 4-6 (max=0.07) excède largement 1.5x émissions 2-3
    # (max=0.01 -> seuil 0.015) : c'est précisément ce qui exclut S2.
    assert resultat["max_emissions_4_6"] > resultat["max_emissions_2_3"] * 1.5


# --- 2. Δχ_1 n'influence pas la lecture --------------------------------------


@pytest.mark.parametrize("delta_chi_base,lecture_attendue", [
    ([0.0, 0.08, 0.09, 0.02, 0.01, 0.005], "S1"),
    ([0.0, 0.010, 0.012, 0.011, 0.013, 0.012], "S2"),
    ([0.0, 0.01, 0.01, 0.05, 0.06, 0.07], "AUTRE"),
])
def test_delta_chi_1_n_influence_pas_la_lecture(delta_chi_base, lecture_attendue):
    """Δχ_1 = 0 (structurel, cf. `chaine_registre`) n'influence pas la
    lecture : changer la valeur de Δχ_1 (0.0 -> une valeur arbitraire, y
    compris supérieure à jnd_sev) ne change ni S1, ni S2, ni AUTRE -- seules
    les émissions i>=2 entrent dans `lecture_sonde` (cf. sa docstring)."""
    resultat_zero = lecture_sonde(delta_chi_base, _JND_SEV)
    assert resultat_zero["lecture"] == lecture_attendue

    delta_chi_mute = [999.0] + list(delta_chi_base[1:])
    resultat_mute = lecture_sonde(delta_chi_mute, _JND_SEV)
    assert resultat_mute["lecture"] == lecture_attendue
    assert resultat_mute["lecture"] == resultat_zero["lecture"]
    assert resultat_mute["s1"] == resultat_zero["s1"]
    assert resultat_mute["s2"] == resultat_zero["s2"]


# --- 3. Aucun timestamp calendaire dans le JSON de sortie --------------------

_MOTIF_DATE_CALENDAIRE = re.compile(r"\b\d{4}-\d{2}-\d{2}")


@pytest.mark.skipif(not OUT_JSON_PATH.exists(), reason="m2_sonde.json absent (sonde non exécutée)")
def test_json_sortie_sans_timestamp_calendaire():
    """`outputs/arcA/m2_sonde.json` ne doit contenir AUCUN timestamp
    calendaire (ni champ de date, ni sous-chaîne au format ISO
    AAAA-MM-JJ) -- seuls les chronos `*_s` (durées, pas des dates) sont
    autorisés."""
    texte = OUT_JSON_PATH.read_text(encoding="utf-8")
    assert _MOTIF_DATE_CALENDAIRE.search(texte) is None

    document = json.loads(texte)
    for cle in ("date", "date_session", "timestamp", "horodatage"):
        assert cle not in document
        assert cle not in document.get("meta", {})
        assert cle not in document.get("chronometrage", {})


# --- Bonus (gratuit, pas de physique) -- charge_pins_severe ------------------


def test_charge_pins_severe_valeurs_attendues():
    """`charge_pins_severe` lit réellement `outputs/arcC/pins_spatial.json`
    (le fichier gravé existe sur cette machine) : vérifie que le point
    central ET l'IC chargés correspondent aux valeurs citées par le brief
    (aucune chaîne réelle exécutée, simple lecture JSON)."""
    pins = charge_pins_severe()
    assert pins["jnd"] == pytest.approx(0.07334065957385666, abs=1e-12)
    assert pins["ic"][0] == pytest.approx(0.06033229334102097, abs=1e-12)
    assert pins["ic"][1] == pytest.approx(0.08673568621862042, abs=1e-12)
