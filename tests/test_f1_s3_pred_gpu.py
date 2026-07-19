"""Tests F1 — bras s3 : prédiction des colonnes entrantes GPU-side
(§A17-lecture-s2-mobile, L4-design, pocCascade2phys 47e0586).

LE RISQUE PROPRE À CE BRAS : en déplaçant la prédiction du CPU vers le
GPU, on peut en OMETTRE une part — et une prédiction manquante ne coûte
rien, ce qui ferait conclure « le GPU-side est bien moins cher » à tort.
Les verrous exigés le ferment :
  (a) aucune omission : les colonnes prédites, frame par frame, sont
      exactement celles que la géométrie impose (référence calculée
      INDÉPENDAMMENT des deux implémentations), et les écritures touchent
      la fenêtre ET la référence ;
  (b) équivalence : sur un parent cohérent, la descente parent->enfant
      donne le même bloc que la prédiction directe depuis le monde 0 ;
  (c) le niveau 1 paie bien son chemin CPU + H2D.
S'y ajoute la couverture du plan — quels slots peuvent réellement passer
GPU-side, et pourquoi les autres non."""
import json

import numpy as np
import pytest

from scripts.run_f1_ma_ter import N_NIV, slots_v2
from scripts.run_f1_s2_mobile import DELTA_X_MOBILE
from scripts.run_f1_s3_pred_gpu import (
    RAISON_NIVEAU_1,
    RAISON_PAS_DE_PARENT,
    RAISON_SYSTEMES,
    TOL_PREDICTION_ATOL,
    TOL_PREDICTION_RTOL,
    PyramideBatcheePredGpu,
    lecture_mecanique,
    lire_a_batche,
    lire_a_mobile,
    plan_prediction,
    predire_colonnes_gpu,
    resume_couverture,
)
from src.f1_gpu.backend import cupy_disponible
from src.f1_gpu.pyramide import (
    GeometriePyramide,
    Slot,
    monde0_jetable,
    predire_bloc_cpu,
)
from src.f1_gpu.transferts import TransfertComptable

gpu_requis = pytest.mark.skipif(
    not cupy_disponible(), reason="cupy/device CUDA indisponible")

# Motif de V2 en réduction : c dégressif + une fenêtre d'énergie au fin.
GEO_MIXTE = GeometriePyramide(
    n_fov=8, n_niv=4, n0=16,
    slots=(Slot(niveau=1, c=4, role="fovea"),
           Slot(niveau=2, c=4, role="fovea"),
           Slot(niveau=3, c=8, role="fovea"),
           Slot(niveau=3, c=8, role="energie")))

# Pyramide homogène : tous les niveaux à c=8, fovéaux seuls — tout ce qui
# a un parent GPU doit y passer GPU-side.
GEO_HOMOGENE = GeometriePyramide(
    n_fov=8, n_niv=5, n0=16,
    slots=tuple(Slot(niveau=j, c=8, role="fovea") for j in range(1, 5)))


def _bras(geo, cp=np):
    transferts = TransfertComptable(cp)
    bras = PyramideBatcheePredGpu(
        cp, geo, transferts, pas_f=lambda f, x, s, t: (s, 0.0))
    transferts.frame_suivante()
    return bras, transferts


def _colonnes_attendues(geo, centre_avant: int, centre_apres: int) -> int:
    """Référence INDÉPENDANTE des implémentations : pour chaque niveau,
    le nombre de colonnes entrantes vaut n_slots(j) x dx (ou la fenêtre
    entière si dx >= n_fov)."""
    total = 0
    for j in geo.niveaux_gpu:
        avant = geo.origines(j, centre_avant)[0][1]
        apres = geo.origines(j, centre_apres)[0][1]
        dx = apres - avant
        if dx > 0:
            largeur = geo.n_fov if dx >= geo.n_fov else dx
            total += geo.n_slots(j) * largeur
    return total


# ----- (a) aucune omission -----

@pytest.mark.parametrize("geo", [GEO_MIXTE, GEO_HOMOGENE])
def test_colonnes_predites_egalent_la_reference_geometrique(geo):
    """LE verrou : sur 12 frames, le total prédit (GPU + CPU) est
    exactement celui que la géométrie impose. Une colonne oubliée ne
    coûterait rien et fausserait la lecture."""
    bras, _ = _bras(geo)
    for _ in range(12):
        centre_avant = bras.centre_fin
        diag = bras.frame(delta_x=DELTA_X_MOBILE)
        attendu = _colonnes_attendues(geo, centre_avant, bras.centre_fin)
        assert diag["colonnes_gpu"] + diag["colonnes_cpu"] == attendu


def test_tous_les_niveaux_bougent_sur_la_fenetre_testee():
    """Contre-épreuve : sans elle, le verrou ci-dessus pourrait ne jamais
    exercer les niveaux grossiers."""
    bras, _ = _bras(GEO_MIXTE)
    vus: set[int] = set()
    for _ in range(12):
        vus.update(bras.frame(delta_x=DELTA_X_MOBILE)["niveaux_deplaces"])
    assert vus == set(GEO_MIXTE.niveaux_gpu)


def test_les_deux_chemins_sont_reellement_exerces():
    """Sur le motif mixte, il doit y avoir DU GPU et DU CPU — sinon le
    test ne prouverait rien sur le chemin qu'il n'exerce pas."""
    bras, _ = _bras(GEO_MIXTE)
    gpu = cpu = 0
    for _ in range(12):
        diag = bras.frame(delta_x=DELTA_X_MOBILE)
        gpu += diag["colonnes_gpu"]
        cpu += diag["colonnes_cpu"]
    assert gpu > 0 and cpu > 0


@pytest.mark.parametrize("geo", [GEO_MIXTE, GEO_HOMOGENE])
def test_le_deplacement_ecrit_fenetre_ET_reference(geo):
    """Comme la pyramide, hors étage de remontée : les deux buffers
    reçoivent le roll et les colonnes entrantes."""
    bras, _ = _bras(geo)
    avant_f = [f.copy() for f in bras.fenetres]
    avant_r = [r.copy() for r in bras.references]
    for _ in range(4):
        bras.frame(delta_x=DELTA_X_MOBILE)
    assert any(not np.array_equal(a, b)
               for a, b in zip(bras.fenetres, avant_f))
    assert any(not np.array_equal(a, b)
               for a, b in zip(bras.references, avant_r))


# ----- (b) équivalence à la prédiction CPU -----

def test_descente_parent_equivaut_a_la_prediction_directe():
    """Sur un parent COHÉRENT (celui qu'on obtient en descendant depuis le
    monde 0), la descente parent->enfant donne le même bloc que la
    prédiction directe — la propriété que `pyramide` énonçait déjà : « les
    octets descendus sont identiques à une chaîne niveau-à-niveau ».
    La prédiction voisin étant une SÉLECTION (aucune arithmétique),
    l'égalité est exacte ; la tolérance nommée est vérifiée en plus."""
    geo = GEO_HOMOGENE
    monde0 = monde0_jetable(geo)
    centre = geo.centre_fin_initial()
    j = 3                                     # enfant, parent = niveau 2
    n_sys = geo.n_systemes_du_niveau(j)
    oy, ox = geo.origines(j, centre)[0]
    oy_p, ox_p = geo.origines(j - 1, centre)[0]

    # Parent cohérent : sa fenêtre pleine prédite depuis le monde 0.
    parent = predire_bloc_cpu(monde0, geo, j - 1, [(oy_p, ox_p)], 0,
                              geo.n_fov, n_sys)[0]
    obtenu = predire_colonnes_gpu(parent, np, geo, oy, ox, oy_p, ox_p,
                                  geo.n_fov - 3, geo.n_fov)
    attendu = predire_bloc_cpu(monde0, geo, j, [(oy, ox)],
                               geo.n_fov - 3, geo.n_fov, n_sys)[0]
    np.testing.assert_allclose(obtenu, attendu, rtol=TOL_PREDICTION_RTOL,
                               atol=TOL_PREDICTION_ATOL)
    assert np.array_equal(obtenu, attendu)     # exacte, en fait


def test_descente_parent_equivaut_sur_la_fenetre_entiere():
    """Même propriété sur la fenêtre pleine (cas du saut E4c)."""
    geo = GEO_HOMOGENE
    monde0 = monde0_jetable(geo)
    centre = geo.centre_fin_initial()
    for j in (2, 3, 4):
        n_sys = geo.n_systemes_du_niveau(j)
        oy, ox = geo.origines(j, centre)[0]
        oy_p, ox_p = geo.origines(j - 1, centre)[0]
        parent = predire_bloc_cpu(monde0, geo, j - 1, [(oy_p, ox_p)], 0,
                                  geo.n_fov, n_sys)[0]
        obtenu = predire_colonnes_gpu(parent, np, geo, oy, ox, oy_p, ox_p,
                                      0, geo.n_fov)
        attendu = predire_bloc_cpu(monde0, geo, j, [(oy, ox)], 0,
                                   geo.n_fov, n_sys)[0]
        assert np.array_equal(obtenu, attendu)


def test_prediction_gpu_fail_loud_hors_fenetre_parente():
    """Une région hors de la fenêtre parente doit lever : une indexation
    silencieusement repliée fabriquerait des valeurs fausses."""
    geo = GEO_HOMOGENE
    parent = np.zeros((2, 4, geo.n_fov, geo.n_fov), dtype=np.float32)
    with pytest.raises(RuntimeError, match="sort de la fenêtre parente"):
        predire_colonnes_gpu(parent, np, geo, 10_000, 10_000, 0, 0,
                             0, geo.n_fov)


# ----- (c) le niveau 1 paie son chemin CPU + H2D -----

def test_niveau_1_reste_cpu_et_paie_son_h2d():
    """Le niveau 1 n'a pas de parent GPU (son parent est le monde 0) : il
    doit rester CPU, et son trafic H2D doit apparaître à la frame où il
    bouge."""
    plan = plan_prediction(GEO_MIXTE)
    assert plan[1]["gpu"] == {}
    assert plan[1]["cpu"] == [0]
    assert plan[1]["raisons"][0] == RAISON_NIVEAU_1

    bras, transferts = _bras(GEO_MIXTE)
    octets_quand_niveau1_bouge = []
    for _ in range(12):
        diag = bras.frame(delta_x=DELTA_X_MOBILE)
        bilan = transferts.frame_suivante()
        if 1 in diag["niveaux_deplaces"]:
            octets_quand_niveau1_bouge.append(bilan["h2d_octets"])
    assert octets_quand_niveau1_bouge
    assert all(o > 0 for o in octets_quand_niveau1_bouge)


def test_les_slots_gpu_ne_paient_aucun_h2d():
    """Contrepartie : sur une pyramide homogène, une frame où seuls des
    niveaux GPU-side bougent ne descend RIEN."""
    bras, transferts = _bras(GEO_HOMOGENE)
    vu_frame_sans_h2d = False
    for _ in range(8):
        diag = bras.frame(delta_x=DELTA_X_MOBILE)
        bilan = transferts.frame_suivante()
        if 1 not in diag["niveaux_deplaces"] and diag["colonnes_gpu"] > 0:
            assert bilan["h2d_octets"] == 0
            vu_frame_sans_h2d = True
    assert vu_frame_sans_h2d


# ----- le plan de prédiction sur la vraie V2 -----

def test_plan_v2_sept_slots_gpu_cinq_cpu():
    """Fait géométrique vérifié au build : les 8 fovéaux des niveaux 2..9
    ont un parent couvrant, mais le fovéal du niveau 8 a un parent à 1
    système quand il en veut 2 ; les 3 slots d'énergie n'ont aucun parent
    couvrant. D'où 7 GPU / 5 CPU."""
    geo = GeometriePyramide(n_fov=512, n_niv=N_NIV, slots=slots_v2())
    couverture = resume_couverture(geo)
    assert couverture["n_slots_gpu"] == 7
    assert couverture["n_slots_cpu"] == 5
    gpu = {(s["niveau"], s["slot"]) for s in couverture["slots_gpu"]}
    assert gpu == {(2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (9, 0)}


def test_plan_v2_raisons_des_replis_cpu():
    """Chaque repli porte sa raison — aucune n'est muette."""
    geo = GeometriePyramide(n_fov=512, n_niv=N_NIV, slots=slots_v2())
    raisons = {(s["niveau"], s["slot"]): s["raison"]
               for s in resume_couverture(geo)["slots_cpu"]}
    assert raisons[(1, 0)] == RAISON_NIVEAU_1
    assert raisons[(8, 0)] == RAISON_SYSTEMES
    assert raisons[(8, 1)] == RAISON_PAS_DE_PARENT
    assert raisons[(9, 1)] == RAISON_PAS_DE_PARENT
    assert raisons[(9, 2)] == RAISON_PAS_DE_PARENT


def test_plan_v2_frame_mediane_reste_majoritairement_cpu():
    """Conséquence de lecture, verrouillée : sur une frame médiane (seul
    le niveau 9 bouge), 1 slot passe GPU-side et 2 restent CPU — la baisse
    attendue sur la médiane est partielle PAR CONSTRUCTION."""
    geo = GeometriePyramide(n_fov=512, n_niv=N_NIV, slots=slots_v2())
    mediane = resume_couverture(geo)["frame_mediane"]
    assert mediane["niveaux_deplaces"] == [9]
    assert mediane["slots_gpu"] == 1
    assert mediane["slots_cpu"] == 2
    assert "partielle par construction" in mediane["note"]


def test_plan_est_stable_le_long_du_balayage():
    """Le plan a été vérifié stable au build ; il doit le rester pour
    toute position du centre (sinon la garde runtime lèverait en série)."""
    geo = GeometriePyramide(n_fov=512, n_niv=N_NIV, slots=slots_v2())
    reference = plan_prediction(geo)
    centre = geo.centre_fin_initial()
    for decalage in (1, 7, 63, 255, 399):
        courant = plan_prediction(geo, centre + decalage)
        for j in geo.niveaux_gpu:
            assert courant[j]["gpu"] == reference[j]["gpu"]
            assert courant[j]["cpu"] == reference[j]["cpu"]


# ----- dépendances fail-loud -----

def test_refuse_sans_le_json_de_s2(tmp_path):
    with pytest.raises(RuntimeError, match="run_f1_s2_batche"):
        lire_a_batche(tmp_path / "absent.json")


def test_refuse_sans_le_json_de_s2_mobile(tmp_path):
    with pytest.raises(RuntimeError, match="run_f1_s2_mobile"):
        lire_a_mobile(tmp_path / "absent.json")


def test_refuse_un_json_incomplet(tmp_path):
    chemin = tmp_path / "s2.json"
    chemin.write_text(json.dumps({"lecture_mecanique": {}}),
                      encoding="utf-8")
    with pytest.raises(RuntimeError, match="illisible ou incomplet"):
        lire_a_batche(chemin)


def test_lit_les_mesures_sans_les_recopier(tmp_path):
    s2 = tmp_path / "s2.json"
    s2.write_text(json.dumps({"lecture_mecanique": {"a_batche_ms": 14.432}}),
                  encoding="utf-8")
    mobile = tmp_path / "s2m.json"
    mobile.write_text(
        json.dumps({"lecture_mecanique": {"a_mobile_ms": 17.251}}),
        encoding="utf-8")
    assert lire_a_batche(s2)["valeur_ms"] == 14.432
    assert lire_a_mobile(mobile)["valeur_ms"] == 17.251


# ----- lecture mécanique -----

def _mesure(valeur: float, source: str = "x.json") -> dict:
    return {"valeur_ms": valeur, "source": source}


def test_prediction_gpu_est_la_difference_gravee():
    geo = GeometriePyramide(n_fov=512, n_niv=N_NIV, slots=slots_v2())
    lecture = lecture_mecanique(16.0, _mesure(14.432), _mesure(17.251),
                                resume_couverture(geo))
    assert lecture["prediction_gpu"]["valeur_ms"] == pytest.approx(1.568)
    assert lecture["prediction_gpu"]["prediction_cpu_side_ms"] == (
        pytest.approx(2.819))
    assert lecture["cible_l1_l3"]["valeur_ms"] == pytest.approx(0.7)


def test_cible_negative_se_lit_telle_quelle():
    geo = GeometriePyramide(n_fov=512, n_niv=N_NIV, slots=slots_v2())
    lecture = lecture_mecanique(17.0, _mesure(14.432), _mesure(17.251),
                                resume_couverture(geo))
    assert lecture["cible_l1_l3"]["valeur_ms"] == pytest.approx(-0.3)
    assert "telle quelle" in lecture["cible_l1_l3"]["note"]


def test_la_lecture_porte_la_couverture():
    """Sans la couverture sous les yeux, la prédiction GPU serait
    sur-lue : une part des slots reste CPU."""
    geo = GeometriePyramide(n_fov=512, n_niv=N_NIV, slots=slots_v2())
    lecture = lecture_mecanique(16.0, _mesure(14.432), _mesure(17.251),
                                resume_couverture(geo))
    assert lecture["couverture_gpu"]["n_slots_cpu"] == 5
    assert "couverture_gpu" in lecture["prediction_gpu"]["note"]


def test_le_bras_ne_prononce_rien():
    geo = GeometriePyramide(n_fov=512, n_niv=N_NIV, slots=slots_v2())
    lecture = lecture_mecanique(16.0, _mesure(14.432), _mesure(17.251),
                                resume_couverture(geo))
    texte = json.dumps(lecture, ensure_ascii=False)
    for interdit in ("MORT", "PASS", "VERDICT"):
        assert interdit not in texte
    note = lecture["cible_l1_l3"]["note"]
    assert "décision de Romain" in note
    assert "pas une sortie de ce driver" in note


# ----- plomberie GPU -----

@gpu_requis
def test_bras_s3_tourne_sur_gpu_avec_kernel_fusionne():
    """Plomberie (taille minuscule, JAMAIS une mesure) : le bras complet
    — descente GPU-side, repli CPU+H2D, puis F fusionné — traverse le
    device, CFL on-device auditée."""
    import cupy as cp

    transferts = TransfertComptable(cp)
    bras = PyramideBatcheePredGpu(cp, GEO_MIXTE, transferts)
    transferts.frame_suivante()
    for _ in range(6):
        diag = bras.frame(delta_x=DELTA_X_MOBILE)
        bilan = transferts.frame_suivante()
    assert bilan["d2h_octets"] == 0
    assert diag["colonnes_gpu"] + diag["colonnes_cpu"] >= 0
    audit = bras.audit_cfl()
    assert audit["toutes_finies_positives"] is True
    for fen in bras.fenetres:
        assert bool(cp.isfinite(fen).all())
