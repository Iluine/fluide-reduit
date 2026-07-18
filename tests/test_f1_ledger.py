"""Tests F1 tranche-1 — ledger v1 (B7) et load E7 (B8,
`src/f1_gpu/ledger.py`). AUCUN `run_episode` ici (VM, kill 45 s) : les
commits sont synthétiques (`summarize_qt` sur champs aléatoires — rapide),
l'évoluteur du segment courant est trivial (injection B8)."""
import numpy as np
import pytest

from src.f1_gpu.ledger import EcrivainLedger, charger_ledger, parser_ledger
from src.summary_quadtree import regenerate_qt, summarize_qt


def _commit(graine: int):
    champ = np.random.default_rng(graine).random((64, 64))
    return summarize_qt(champ, 64)


def _ledger_minimal(chemin):
    """Ledger : événements t=1..3, commits t=1 et t=2, événement t=3 =
    segment courant (après le dernier commit)."""
    ecrivain = EcrivainLedger(chemin, creer=True)
    ecrivain.ajouter_evenement(1, 101, "pulse_sediment")
    ecrivain.ajouter_commit(1, _commit(1), 1, (0, 0, 32), 64)
    ecrivain.ajouter_evenement(2, 101, "pulse_sediment")
    ecrivain.ajouter_commit(2, _commit(2), 1, (0, 0, 32), 64)
    ecrivain.ajouter_evenement(3, 101, "pulse_sediment")
    return ecrivain


def test_roundtrip_types_et_commit_bit_exact(tmp_path):
    chemin = tmp_path / "ledger"
    ecrivain = _ledger_minimal(chemin)
    ecrivain.ajouter_snapshot(3, np.full((64, 64), 0.25))
    ecrivain.ajouter_entree_joueur(4, {"action": "reserve-v1.1"})

    document = parser_ledger(chemin)
    assert document["n_par_type"] == {
        "evenement_seede": 3, "commit_emission": 2, "snapshot_racine": 1,
        "entree_joueur": 1}
    commit_relu = [e for e in document["entrees"]
                   if e["type"] == "commit_emission"][-1]
    original = _commit(2)
    assert np.array_equal(commit_relu["tableaux"]["topologie"],
                          original.topology)
    assert np.array_equal(commit_relu["tableaux"]["moyennes"],
                          original.means)
    snapshot = [e for e in document["entrees"]
                if e["type"] == "snapshot_racine"][0]
    assert np.array_equal(snapshot["tableaux"]["champ"],
                          np.full((64, 64), 0.25))


def test_cle_strictement_croissante_a_l_append(tmp_path):
    ecrivain = _ledger_minimal(tmp_path / "ledger")
    with pytest.raises(RuntimeError):
        ecrivain.ajouter_evenement(2, 101, "pulse_sediment")  # t recule


def test_creer_sur_ledger_existant_interdit(tmp_path):
    _ledger_minimal(tmp_path / "ledger")
    with pytest.raises(RuntimeError):
        EcrivainLedger(tmp_path / "ledger", creer=True)


def test_reprise_conserve_la_derniere_cle(tmp_path):
    chemin = tmp_path / "ledger"
    _ledger_minimal(chemin)
    repris = EcrivainLedger(chemin, creer=False)
    assert repris.derniere_cle == (3, 4)
    repris.ajouter_evenement(4, 101, "pulse_sediment")
    assert len(parser_ledger(chemin)["entrees"]) == 6


def test_corruption_payload_detectee(tmp_path):
    chemin = tmp_path / "ledger"
    _ledger_minimal(chemin)
    donnees = bytearray((chemin / "data.bin").read_bytes())
    donnees[len(donnees) // 2] ^= 0xFF
    (chemin / "data.bin").write_bytes(bytes(donnees))
    with pytest.raises(ValueError, match="sha256"):
        parser_ledger(chemin)


def test_corruption_dtype_meme_taille_detectee(tmp_path):
    """Revue adversariale 2026-07-19 : un dtype corrompu de MÊME taille
    d'élément (float64 -> int64) laisse octets/offsets/shape intacts —
    l'empreinte doit couvrir l'INTERPRÉTATION, pas seulement le contenu."""
    import json

    chemin = tmp_path / "ledger"
    _ledger_minimal(chemin)
    lignes = (chemin / "index.jsonl").read_text(
        encoding="utf-8").splitlines()
    entree = json.loads(lignes[1])           # premier commit_emission
    entree["payloads"][1]["dtype"] = "int64"  # moyennes : float64 -> int64
    lignes[1] = json.dumps(entree, ensure_ascii=False)
    (chemin / "index.jsonl").write_text(
        "\n".join(lignes) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        parser_ledger(chemin)


def test_corruption_shape_meme_taille_detectee(tmp_path):
    import json

    chemin = tmp_path / "ledger"
    _ledger_minimal(chemin)
    ecrivain = EcrivainLedger(chemin, creer=False)
    ecrivain.ajouter_snapshot(4, np.zeros((64, 64)))
    lignes = (chemin / "index.jsonl").read_text(
        encoding="utf-8").splitlines()
    entree = json.loads(lignes[-1])
    entree["payloads"][0]["shape"] = [32, 128]  # même nombre d'éléments
    lignes[-1] = json.dumps(entree, ensure_ascii=False)
    (chemin / "index.jsonl").write_text(
        "\n".join(lignes) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        parser_ledger(chemin)


def test_index_ampute_detecte(tmp_path):
    chemin = tmp_path / "ledger"
    _ledger_minimal(chemin)
    lignes = (chemin / "index.jsonl").read_text(
        encoding="utf-8").splitlines()
    (chemin / "index.jsonl").write_text(
        "\n".join(lignes[1:]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        parser_ledger(chemin)


def test_charger_e7_regenerate_puis_segment_courant(tmp_path):
    """B8 : état = regenerate(DERNIER commit), puis l'évoluteur injecté
    rejoue le seul segment courant (l'événement t=3, postérieur au commit
    t=2)."""
    chemin = tmp_path / "ledger"
    _ledger_minimal(chemin)
    rejoues = []

    def evoluer(etat, entree):
        rejoues.append(entree["t_sim"])
        return etat + 1.0

    televerse = []
    resultat = charger_ledger(chemin, evoluer=evoluer,
                              televerser=televerse.append)
    attendu = regenerate_qt(_commit(2)) + 1.0
    assert np.array_equal(resultat["etat"], attendu)
    assert rejoues == [3]
    assert resultat["n_segment"] == 1
    assert resultat["t_dernier_commit"] == 2
    assert len(televerse) == 1 and np.array_equal(televerse[0], attendu)


def test_charger_segment_sans_evoluteur_fail_loud(tmp_path):
    chemin = tmp_path / "ledger"
    _ledger_minimal(chemin)
    with pytest.raises(RuntimeError, match="évoluteur"):
        charger_ledger(chemin)


def test_charger_sans_commit_fail_loud(tmp_path):
    chemin = tmp_path / "ledger"
    ecrivain = EcrivainLedger(chemin, creer=True)
    ecrivain.ajouter_evenement(1, 101, "pulse_sediment")
    with pytest.raises(RuntimeError, match="aucun commit"):
        charger_ledger(chemin)


def test_commit_hors_budget_refuse_a_l_ecriture(tmp_path):
    ecrivain = EcrivainLedger(tmp_path / "ledger", creer=True)
    commit = _commit(9)
    with pytest.raises(RuntimeError, match="budget"):
        ecrivain.ajouter_commit(1, commit, 1, (0, 0, 32), k=1)
