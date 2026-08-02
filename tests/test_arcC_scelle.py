"""Scellement procédural des seuils P3′ (`src/arcC_scelle.py`, mission 8a.5).

Le foyer unique testable : l'orchestration scelle, la lecture descelle — les
deux à travers CE module. Chaque garde sait ÉCHOUER : un fichier altéré,
substitué ou absent lève, jamais un silence."""
from __future__ import annotations

import pytest

from src.arcC_scelle import descelle_seuils, scelle_seuils

SEUILS = {"severe": {"0": 0.070, "1": 0.080, "2": 0.081, "3": 0.071,
                     "4": 0.072, "5": 0.083}}


def test_scelle_puis_descelle_rend_les_seuils_bit_identiques(tmp_path):
    chemin = tmp_path / "seuils_scelles_p3prime.json"
    sha = scelle_seuils(chemin, SEUILS)
    assert descelle_seuils(chemin, sha) == SEUILS


def test_descelle_refuse_un_fichier_altere(tmp_path):
    """L'ALTÉRATION est détectable (sha256) — c'est la moitié du scellement
    que la procédure garantit ; la consultation, elle, ne laisse aucune
    trace (nature nommée, mission 8a.5)."""
    chemin = tmp_path / "seuils_scelles_p3prime.json"
    sha = scelle_seuils(chemin, SEUILS)
    contenu = chemin.read_text(encoding="utf-8")
    chemin.write_text(contenu.replace("0.07,", "0.17,"), encoding="utf-8")
    assert chemin.read_text(encoding="utf-8") != contenu, "le test doit avoir altéré"
    with pytest.raises(RuntimeError, match="ALTÉRÉ"):
        descelle_seuils(chemin, sha)


def test_descelle_refuse_un_sha_qui_nest_pas_le_sien(tmp_path):
    chemin = tmp_path / "seuils_scelles_p3prime.json"
    scelle_seuils(chemin, SEUILS)
    with pytest.raises(RuntimeError, match="ALTÉRÉ"):
        descelle_seuils(chemin, "0" * 64)


def test_descelle_refuse_un_fichier_absent(tmp_path):
    with pytest.raises(RuntimeError, match="INTROUVABLE"):
        descelle_seuils(tmp_path / "absent.json", "0" * 64)
