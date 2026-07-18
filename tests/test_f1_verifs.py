"""Tests F1 tranche-1 — registre des vérifications d'instrument + câblage
« tranche muette » (B9, `src/f1_gpu/verifs.py`). Registres en tmp_path —
le registre réel (`outputs/f1/verifs.json`) n'est JAMAIS touché par les
tests."""
import pytest

from src.f1_gpu.verifs import (
    enregistrer_verif,
    exiger_pass,
    lire_registre,
    verifier_sensibilite_chrono,
)


def test_enregistrer_et_lire(tmp_path):
    chemin = tmp_path / "verifs.json"
    enregistrer_verif("gel_bit_exact", True, {"code_retour": 0},
                      chemin=chemin)
    enregistrer_verif("vram_concordance", False, {"ecart_octets": 10 ** 9},
                      chemin=chemin)
    registre = lire_registre(chemin)
    assert registre["gel_bit_exact"]["pass"] is True
    assert registre["vram_concordance"]["pass"] is False
    # Ré-enregistrement : remplace l'entrée, conserve les autres.
    enregistrer_verif("vram_concordance", True, {"ecart_octets": 0},
                      chemin=chemin)
    registre = lire_registre(chemin)
    assert registre["vram_concordance"]["pass"] is True
    assert registre["gel_bit_exact"]["pass"] is True


def test_nom_inconnu_refuse(tmp_path):
    with pytest.raises(ValueError):
        enregistrer_verif("verif_inventee", True, {},
                          chemin=tmp_path / "verifs.json")


def test_exiger_pass_ok(tmp_path):
    chemin = tmp_path / "verifs.json"
    enregistrer_verif("gel_bit_exact", True, {}, chemin=chemin)
    enregistrer_verif("vram_concordance", True, {}, chemin=chemin)
    exiger_pass(("gel_bit_exact", "vram_concordance"), chemin=chemin)


def test_exiger_pass_manquante_tranche_muette(tmp_path):
    chemin = tmp_path / "verifs.json"
    enregistrer_verif("gel_bit_exact", True, {}, chemin=chemin)
    with pytest.raises(RuntimeError, match="TRANCHE MUETTE"):
        exiger_pass(("gel_bit_exact", "bande_passante_a_vide"),
                    chemin=chemin)


def test_exiger_pass_echouee_tranche_muette(tmp_path):
    chemin = tmp_path / "verifs.json"
    enregistrer_verif("gel_bit_exact", False, {}, chemin=chemin)
    with pytest.raises(RuntimeError, match="TRANCHE MUETTE"):
        exiger_pass(("gel_bit_exact",), chemin=chemin)


def test_registre_absent_est_muet(tmp_path):
    assert lire_registre(tmp_path / "absent.json") == {}
    with pytest.raises(RuntimeError, match="TRANCHE MUETTE"):
        exiger_pass(("gel_bit_exact",), chemin=tmp_path / "absent.json")


def test_sensibilite_chrono_seuil_5_pourcent():
    """Vérif #2 (B9) : 256->512 doit bouger la médiane d'au moins 5 %."""
    passe, details = verifier_sensibilite_chrono(10.0, 11.0)
    assert passe and details["ecart_relatif"] == pytest.approx(0.10)
    passe, _ = verifier_sensibilite_chrono(10.0, 10.2)
    assert not passe
    # Chrono inversé (512 plus rapide que 256) : non crédible -> FAIL.
    passe, _ = verifier_sensibilite_chrono(10.0, 9.0)
    assert not passe
