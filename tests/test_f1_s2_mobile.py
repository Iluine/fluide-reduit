"""Tests F1 — bras s2-mobile : la prédiction de déplacement sous
orchestration batchée (§A17-lecture-s1-s2 G2 amendé, pocCascade2phys
61641c2).

LE RISQUE PROPRE À CE BRAS : il ré-orchestre la MACHINERIE DE
DÉPLACEMENT (prédiction des colonnes entrantes, descentes H2D,
roll/écritures) pour la faire tourner sur des buffers groupés. Omettre
une part de ce travail rendrait la prédiction batchée artificiellement
basse — et ferait conclure « la prédiction était de l'orchestration » à
tort, exactement l'erreur que le pré-enregistrement veut éviter. Le
verrou central compare donc, FRAME PAR FRAME, le trafic de ce bras à
celui de `PyramideFovea` mobile : mêmes niveaux déplacés, mêmes octets
H2D, même nombre d'appels. Il tourne sous numpy (F injecté), donc sans
device."""
import json

import numpy as np
import pytest

from scripts.run_f1_ma_ter import B_FRAME_MS, slots_v2
from scripts.run_f1_s2_mobile import (
    DELTA_X_MOBILE,
    PREDICTION_PER_NIVEAU_MS,
    PyramideBatcheeMobile,
    cadence_deplacement,
    lecture_mecanique,
    lire_a_batche_s2,
    plan_groupes,
)
from src.f1_gpu.backend import cupy_disponible
from src.f1_gpu.pyramide import (
    GeometriePyramide,
    PyramideFovea,
    Slot,
)
from src.f1_gpu.transferts import TransfertComptable

gpu_requis = pytest.mark.skipif(
    not cupy_disponible(), reason="cupy/device CUDA indisponible")

# Géométrie réduite au motif de V2 (c dégressif, énergie sur les fins).
SLOTS_TEST = (
    Slot(niveau=1, c=4, role="fovea"),
    Slot(niveau=2, c=4, role="fovea"),
    Slot(niveau=3, c=8, role="fovea"),
    Slot(niveau=3, c=8, role="energie"),
)
GEO_TEST = GeometriePyramide(n_fov=8, n_niv=4, n0=16, slots=SLOTS_TEST)


def _pas_f_neutre(fenetres, xp, sortie, tampon):
    """F no-op : isole la machinerie de déplacement, seule comparée."""
    return sortie, 0.0


def _bras_mobile(geo=GEO_TEST):
    transferts = TransfertComptable(np)
    bras = PyramideBatcheeMobile(np, geo, transferts, pas_f=_pas_f_neutre)
    transferts.frame_suivante()
    return bras, transferts


def _pyramide_reference(geo=GEO_TEST):
    """`PyramideFovea` mobile, remontée OFF, F neutre — le bras B."""
    transferts = TransfertComptable(np)
    pyramide = PyramideFovea(
        np, geo, transferts, remontee_active=False,
        pas_f=lambda fen, xp, sortie: (sortie, 0.0))
    transferts.frame_suivante()
    return pyramide, transferts


# ----- LE verrou : aucun travail de déplacement omis -----

def test_trafic_identique_a_la_pyramide_frame_par_frame():
    """Le verrou central : sur 12 frames — assez pour que TOUS les
    niveaux aient bougé au moins une fois (le niveau j n'avance qu'une
    frame sur 2^(J−j)) — le bras batché descend exactement les mêmes
    octets, en autant d'appels, que la pyramide per-niveau."""
    bras, transferts_bras = _bras_mobile()
    pyramide, transferts_ref = _pyramide_reference()
    for _ in range(12):
        bras.frame(delta_x=DELTA_X_MOBILE)
        pyramide.frame(delta_x=DELTA_X_MOBILE)
        bilan_bras = transferts_bras.frame_suivante()
        bilan_ref = transferts_ref.frame_suivante()
        assert bilan_bras["h2d_octets"] == bilan_ref["h2d_octets"]
        assert bilan_bras["h2d_n"] == bilan_ref["h2d_n"]


def test_memes_niveaux_deplaces_que_la_pyramide():
    """L'alignement dyadique doit être identique : mêmes niveaux, mêmes
    frames. Un niveau oublié ne descendrait rien et ne coûterait rien."""
    bras, _ = _bras_mobile()
    pyramide, _ = _pyramide_reference()
    for _ in range(12):
        diag_bras = bras.frame(delta_x=DELTA_X_MOBILE)
        diag_ref = pyramide.frame(delta_x=DELTA_X_MOBILE)
        assert diag_bras["niveaux_deplaces"] == sorted(
            diag_ref["niveaux_deplaces"])


def test_tous_les_niveaux_bougent_sur_la_fenetre_testee():
    """Contre-épreuve : si aucun niveau grossier ne bougeait sur 12
    frames, le verrou ci-dessus serait vide de sens."""
    bras, _ = _bras_mobile()
    vus: set[int] = set()
    for _ in range(12):
        vus.update(bras.frame(delta_x=DELTA_X_MOBILE)["niveaux_deplaces"])
    assert vus == set(GEO_TEST.niveaux_gpu)


def test_le_deplacement_ecrit_aussi_les_references():
    """La pyramide fait le `roll` et l'écriture des colonnes entrantes
    sur la fenêtre ET sur la référence, HORS de l'étage de remontée. Les
    omettre serait sous-compter le travail — donc les références du bras
    batché doivent bouger elles aussi."""
    bras, _ = _bras_mobile()
    avant = [ref.copy() for ref in bras.references]
    for _ in range(4):
        bras.frame(delta_x=DELTA_X_MOBILE)
    change = [not np.array_equal(apres, initial)
              for apres, initial in zip(bras.references, avant)]
    assert any(change)


def test_les_colonnes_entrantes_sont_bien_ecrites():
    """Le contenu descendu atterrit dans les fenêtres : la dernière
    colonne d'un niveau déplacé change à la frame où il bouge."""
    bras, _ = _bras_mobile()
    avant = [fen.copy() for fen in bras.fenetres]
    bras.frame(delta_x=DELTA_X_MOBILE)
    change = [not np.array_equal(apres, initial)
              for apres, initial in zip(bras.fenetres, avant)]
    assert any(change)


def test_recul_et_delta_negatif_fail_loud():
    bras, _ = _bras_mobile()
    with pytest.raises(ValueError, match="delta_x"):
        bras.frame(delta_x=-1)


# ----- le rangement des groupes -----

def test_plan_groupes_range_les_slots_par_niveau():
    """Les slots d'un même niveau doivent occuper une tranche CONTIGUË —
    c'est ce qui permet UNE descente H2D par niveau, comme la pyramide."""
    geo = GeometriePyramide(n_fov=512, n_niv=10, slots=slots_v2())
    plans = plan_groupes(geo)
    assert [p["n_systemes"] for p in plans] == [2, 1]
    deux, un = plans
    assert deux["n_slots"] == 5 and deux["niveaux"] == [8, 9]
    assert deux["tranches"] == {8: (0, 2), 9: (2, 5)}
    assert un["n_slots"] == 7 and un["niveaux"] == list(range(1, 8))
    assert un["tranches"][1] == (0, 1) and un["tranches"][7] == (6, 7)


def test_plan_groupes_couvre_les_17_blocs_de_v2():
    """La config mesurée reste V2 : 12 slots, 17 blocs."""
    geo = GeometriePyramide(n_fov=512, n_niv=10, slots=slots_v2())
    plans = plan_groupes(geo)
    assert sum(p["n_slots"] for p in plans) == 12
    assert sum(p["n_slots"] * p["n_systemes"] for p in plans) == 17


def test_tranches_sans_recouvrement_ni_trou():
    """Chaque slot du groupe appartient à exactement un niveau."""
    geo = GeometriePyramide(n_fov=512, n_niv=10, slots=slots_v2())
    for plan in plan_groupes(geo):
        couvert: list[int] = []
        for debut, fin in plan["tranches"].values():
            couvert.extend(range(debut, fin))
        assert sorted(couvert) == list(range(plan["n_slots"]))


# ----- ce que la médiane capture (limite de lecture) -----

def test_cadence_de_deplacement_est_dyadique():
    """Le niveau j avance une frame sur 2^(J−j) : sur V2, le niveau 9
    bouge à chaque frame et le niveau 1 une fois sur 256. La médiane ne
    voit donc que les niveaux fins — nuance portée par le JSON."""
    geo = GeometriePyramide(n_fov=512, n_niv=10, slots=slots_v2())
    cadence = cadence_deplacement(geo)
    periodes = cadence["periode_par_niveau_frames"]
    assert periodes["9"] == 1
    assert periodes["8"] == 2
    assert periodes["1"] == 256
    assert cadence["niveaux_a_chaque_frame"] == [9]
    assert "frame MÉDIANE" in cadence["note"]


def test_cadence_signale_les_niveaux_rares_sur_la_serie():
    """Un niveau dont la période dépasse la série ne bougerait pas une
    seule fois : il faut que la lecture puisse le voir."""
    geo = GeometriePyramide(n_fov=512, n_niv=10, slots=slots_v2())
    cadence = cadence_deplacement(geo)
    assert cadence["niveaux_moins_dune_fois_sur_la_serie"] == []


# ----- dépendance au JSON de s2 -----

def test_refuse_de_demarrer_sans_le_json_de_s2(tmp_path):
    """La prédiction batchée est une DIFFÉRENCE : sans son terme de
    gauche, elle n'existe pas. Le message dit quoi lancer."""
    with pytest.raises(RuntimeError, match="run_f1_s2_batche"):
        lire_a_batche_s2(tmp_path / "absent.json")


def test_refuse_un_json_s2_incomplet(tmp_path):
    chemin = tmp_path / "s2.json"
    chemin.write_text(json.dumps({"lecture_mecanique": {}}),
                      encoding="utf-8")
    with pytest.raises(RuntimeError, match="illisible ou incomplet"):
        lire_a_batche_s2(chemin)


def test_refuse_un_a_batche_absurde(tmp_path):
    chemin = tmp_path / "s2.json"
    chemin.write_text(
        json.dumps({"lecture_mecanique": {"a_batche_ms": -1.0}}),
        encoding="utf-8")
    with pytest.raises(RuntimeError, match="non exploitable"):
        lire_a_batche_s2(chemin)


def test_lit_a_batche_sans_le_recopier(tmp_path):
    chemin = tmp_path / "s2.json"
    chemin.write_text(
        json.dumps({"lecture_mecanique": {"a_batche_ms": 14.432}}),
        encoding="utf-8")
    lu = lire_a_batche_s2(chemin)
    assert lu["a_batche_ms"] == 14.432
    assert lu["source"] == str(chemin)


# ----- lecture mécanique -----

def _a_batche(valeur: float = 14.432) -> dict:
    return {"a_batche_ms": valeur, "source": "s2.json"}


def test_prediction_batchee_est_la_difference_gravee():
    """prédiction_batchée = A_mobile − A_batché."""
    lecture = lecture_mecanique(15.9, _a_batche())
    assert lecture["prediction_batchee"]["valeur_ms"] == pytest.approx(1.468)
    assert lecture["prediction_batchee"]["formule"] == "A_mobile - A_batché"


def test_cible_l1_l3_est_le_reste_du_budget():
    """cible = 16.7 − A_mobile — équivalente à la forme gravée
    16.7 − A_batché − prédiction_batchée."""
    lecture = lecture_mecanique(15.9, _a_batche())
    cible = lecture["cible_l1_l3"]["valeur_ms"]
    assert cible == pytest.approx(B_FRAME_MS - 15.9)
    prediction = lecture["prediction_batchee"]["valeur_ms"]
    assert cible == pytest.approx(B_FRAME_MS - 14.432 - prediction)


def test_cible_negative_se_lit_telle_quelle():
    """Un A_mobile au-dessus du seuil donne une cible négative, reportée
    sans commentaire."""
    lecture = lecture_mecanique(18.0, _a_batche())
    assert lecture["cible_l1_l3"]["valeur_ms"] == pytest.approx(-1.3)
    assert "telle quelle" in lecture["cible_l1_l3"]["note"]


def test_reference_per_niveau_est_situee_pas_conclue():
    """La prédiction per-niveau (2.135) est reportée pour situer ; le
    driver ne compare ni ne commente."""
    lecture = lecture_mecanique(15.9, _a_batche())
    prediction = lecture["prediction_batchee"]
    assert prediction["reference_per_niveau_ms"] == PREDICTION_PER_NIVEAU_MS
    assert "PAS pour conclure" in prediction["note"]


def test_le_bras_ne_prononce_rien():
    """Aucun mot de verdict, et la note de cible RENVOIE explicitement la
    décision à Romain plutôt que de la prendre (le mot « jouable » y
    figure, mais dans la phrase qui dénie l'interprétation — c'est cette
    dénégation que le test exige)."""
    lecture = lecture_mecanique(15.9, _a_batche())
    texte = json.dumps(lecture, ensure_ascii=False)
    for interdit in ("MORT", "PASS", "VERDICT"):
        assert interdit not in texte
    note_cible = lecture["cible_l1_l3"]["note"]
    assert "REPORTÉE sans interprétation" in note_cible
    assert "décision de Romain" in note_cible
    assert "pas une sortie de ce driver" in note_cible


# ----- plomberie GPU : le bras tourne vraiment -----

@gpu_requis
def test_bras_mobile_tourne_sur_gpu_avec_kernel_fusionne():
    """Plomberie (taille minuscule, JAMAIS une mesure) : le bras complet
    — déplacement batché puis F fusionné en 4 lancements — traverse le
    device, et la CFL reste on-device puis auditable."""
    import cupy as cp

    transferts = TransfertComptable(cp)
    bras = PyramideBatcheeMobile(cp, GEO_TEST, transferts)
    transferts.frame_suivante()
    for _ in range(4):
        bras.frame(delta_x=DELTA_X_MOBILE)
        bilan = transferts.frame_suivante()
    assert bilan["d2h_octets"] == 0          # remontée OFF
    for dt in bras.dt_cfl:
        assert isinstance(dt, cp.ndarray) and dt.shape == ()
    audit = bras.audit_cfl()
    assert audit["toutes_finies_positives"] is True
    for fen in bras.fenetres:
        assert bool(cp.isfinite(fen).all())


@gpu_requis
def test_residence_compte_fenetres_et_references():
    """Structure conservée : les références sont préallouées comme au
    bras A/s2, donc la résidence reste comparable."""
    import cupy as cp

    bras = PyramideBatcheeMobile(cp, GEO_TEST, TransfertComptable(cp))
    attendu = 2 * sum(int(f.nbytes) for f in bras.fenetres)
    assert bras.octets_etat() == attendu
    assert bras.n_blocs() == 2 * 2 + 2 * 1     # niveau 3 (2 slots x2) + 1,2
