"""Tests F1 — driver M-a-ter (§A16, pocCascade2phys 38a2739).

Ces tests vérifient que la config V2 codée reproduit EXACTEMENT les
chiffres gravés (12 slots, 17 blocs, 14.33 ms prédits = 9.271 fovéale +
5.055 énergie) — un écart ici signifierait que le driver mesure autre
chose que le pré-enregistré. Aucun GPU requis : la config et l'adaptateur
sont du calcul pur."""
import numpy as np
import pytest

from scripts.run_f1_ma_ter import (
    ApplicateurFusionne,
    B_FRAME_MS,
    BANDE_MODELE_MS,
    COUT_PREDIT_GRAVE_MS,
    GATE_RESIDENCE_OCTETS,
    MS_PAR_SLOT,
    N_NIV,
    _slots_energie,
    cout_predit_ms,
    lecture_mecanique,
    slots_v1,
    slots_v2,
)
from src.f1_gpu.backend import cupy_disponible
from src.f1_gpu.pyramide import GeometriePyramide, PyramideFovea
from src.f1_gpu.transferts import TransfertComptable

N_FOV = 512
NIVEAU_FIN = N_NIV - 1          # 9


def _geo(slots) -> GeometriePyramide:
    return GeometriePyramide(n_fov=N_FOV, n_niv=N_NIV, slots=slots)


# ----- conformité de V2 au gravé §A16 -----

def test_v2_reproduit_les_comptes_graves():
    """§A16 : « fovéale 9 slots (2 niveaux fins à c=8, 7 supérieurs à c=4)
    + 3 slots énergie c=8 fins » ; « 12 slots ≈ 3.15 M cellules »."""
    geo = _geo(slots_v2())
    resume = geo.resume_slots()
    assert resume["n_slots_total"] == 12
    assert resume["blocs_actifs_gpu"] == 17
    assert resume["cellules_actives_gpu"] == 12 * N_FOV * N_FOV
    assert 3.14e6 < resume["cellules_actives_gpu"] < 3.15e6
    assert resume["par_role"]["fovea"]["n_slots"] == 9
    assert resume["par_role"]["energie"]["n_slots"] == 3


def test_v2_c_degressif_par_niveau():
    """c=8 aux 2 niveaux les plus fins, c=4 aux 7 supérieurs (R1)."""
    geo = _geo(slots_v2())
    assert [geo.c_du_niveau(j) for j in geo.niveaux_gpu] == (
        [4] * 7 + [8, 8])
    assert geo.n_systemes_du_niveau(NIVEAU_FIN) == 2
    assert geo.n_systemes_du_niveau(1) == 1


def test_v2_repartition_energie_conforme_a_b3():
    """B3 n'offre que 2 offsets d'énergie par niveau : les 3 slots
    d'énergie « fins » se répartissent donc 2 au plus fin + 1 au second —
    la seule répartition sans géométrie neuve."""
    geo = _geo(slots_v2())
    roles = geo.resume_slots()["par_niveau"]
    assert roles[str(NIVEAU_FIN)]["roles"] == ["fovea", "energie", "energie"]
    assert roles[str(NIVEAU_FIN - 1)]["roles"] == ["fovea", "energie"]
    assert roles[str(NIVEAU_FIN - 2)]["roles"] == ["fovea"]


def test_v2_cout_predit_egale_le_grave():
    """14.33 ms gravés — le modèle est la somme des ancres par slot."""
    assert cout_predit_ms(_geo(slots_v2())) == pytest.approx(
        COUT_PREDIT_GRAVE_MS, abs=0.01)


def test_v2_decomposition_foveale_energie_gravee():
    """§A16 : « 9.271 fovéale + 5.055 énergie »."""
    slots = slots_v2()
    foveale = sum(MS_PAR_SLOT[s.c] for s in slots if s.role == "fovea")
    energie = sum(MS_PAR_SLOT[s.c] for s in slots if s.role == "energie")
    assert foveale == pytest.approx(9.271, abs=0.001)
    assert energie == pytest.approx(5.055, abs=0.001)


def test_v2_predit_sous_le_seuil_et_dans_la_bande():
    """Cohérence nommée R3 : la bande est ENTIÈREMENT sous 16.7 ms."""
    predit = cout_predit_ms(_geo(slots_v2()))
    bas, haut = BANDE_MODELE_MS
    assert bas <= predit <= haut
    assert haut < B_FRAME_MS


# ----- V1, diagnostic non-verdictal -----

def test_v1_quatorze_slots_c4_partout():
    """§A16 : « V1 (c=4 partout, 14 slots) mesurée aussi »."""
    geo = _geo(slots_v1())
    assert geo.n_slots_total == 14
    assert geo.blocs_actifs_gpu == 14          # 1 système par slot
    assert all(geo.c_du_niveau(j) == 4 for j in geo.niveaux_gpu)
    assert cout_predit_ms(geo) == pytest.approx(11.80, abs=0.01)


# ----- fabrique d'énergie : garde fail-loud -----

def test_slots_energie_ne_deborde_pas_les_niveaux():
    """Plus d'énergie que 2 par niveau GPU ne tient pas : fail-loud plutôt
    qu'un débordement silencieux vers le niveau 0 CPU."""
    with pytest.raises(ValueError, match="ne tiennent"):
        _slots_energie(n_niv=3, n_energie=5, c=8)


# ----- S1 : les blocs mono-système sont le DESIGN, pas un accident -----

def test_v2_porte_des_niveaux_mono_systeme():
    """S1 (§A16-complément) : les 7 niveaux fovéaux c=4 de V2 portent UN
    système chacun, et le total (17 blocs) est impair. C'est le DESIGN du
    c dégressif — la garde de `pas_f_fusionne`, élargie à {1, 2}, les
    exécute tels quels ; leur coût est un objet de la mesure. Aucun
    regroupement, aucun padding (tous deux gravés REJETÉS)."""
    geo = _geo(slots_v2())
    mono = [j for j in geo.niveaux_gpu if geo.n_systemes_du_niveau(j) == 1]
    assert mono == [1, 2, 3, 4, 5, 6, 7]
    assert geo.blocs_actifs_gpu % 2 == 1        # impair, et c'est permis


def test_les_shapes_de_v2_sont_acceptees_par_la_garde_elargie():
    """Contrat d'interface : toute shape de niveau produite par V2 et V1
    satisfait la garde élargie (S ∈ {1, 2}, 4 champs, ndim 5) — vérifié
    sans GPU sur les shapes elles-mêmes."""
    for fabrique in (slots_v2, slots_v1):
        geo = _geo(fabrique())
        for j in geo.niveaux_gpu:
            shape = (geo.n_slots(j), geo.n_systemes_du_niveau(j), 4,
                     N_FOV, N_FOV)
            assert len(shape) == 5
            assert shape[1] in (1, 2)
            assert shape[2] == 4


# ----- intégration : les configs gravées tournent réellement -----

@pytest.mark.parametrize("fabrique", [slots_v2, slots_v1])
def test_config_gravee_tourne_de_bout_en_bout(fabrique):
    """La config RÉELLE (9 niveaux GPU, c dégressif) traverse une frame
    complète — descente des colonnes entrantes, F, remontée seuillée — à
    n_fov réduit et sous le F jetable (backend numpy, B1). Vérifie que la
    géométrie à slots tient à l'échelle de la config, pas seulement sur la
    maquette à 3 niveaux."""
    geo = GeometriePyramide(n_fov=4, n_niv=N_NIV, n0=4,
                            slots=fabrique())
    transferts = TransfertComptable(np)
    pyramide = PyramideFovea(np, geo, transferts)
    transferts.frame_suivante()
    for _ in range(3):
        pyramide.frame(delta_x=1)
        transferts.frame_suivante()
    for j in geo.niveaux_gpu:
        assert pyramide.fenetres[j].shape == (
            geo.n_slots(j), geo.n_systemes_du_niveau(j), 4, 4, 4)
        assert np.isfinite(pyramide.fenetres[j]).all()


# ----- plomberie GPU : l'applicateur fusionné sur la pyramide -----

@pytest.mark.skipif(not cupy_disponible(),
                    reason="cupy/device CUDA indisponible")
def test_applicateur_fusionne_sur_pyramide_a_c_melange():
    """Plomberie (taille minuscule, JAMAIS une mesure) : le kernel
    fusionné tourne réellement sur une pyramide dont les niveaux mêlent
    1 et 2 systèmes — le chemin exact de M-a-ter. Sans ce test, le
    premier run serait le premier essai du chemin.

    Vérifie aussi B2 : le tampon d'étage est alloué une fois par shape
    (aucune réallocation d'une frame à l'autre)."""
    import cupy as cp

    from src.f1_gpu.pyramide import Slot
    slots = (Slot(niveau=1, c=4, role="fovea"),
             Slot(niveau=2, c=8, role="fovea"),
             Slot(niveau=3, c=8, role="fovea"),
             Slot(niveau=3, c=8, role="energie"))
    geo = GeometriePyramide(n_fov=8, n_niv=4, n0=16, slots=slots)
    applicateur = ApplicateurFusionne()
    transferts = TransfertComptable(cp)
    pyramide = PyramideFovea(cp, geo, transferts, pas_f=applicateur)
    transferts.frame_suivante()

    for _ in range(3):
        pyramide.frame(delta_x=1)
        transferts.frame_suivante()

    shapes_attendues = {tuple(pyramide.fenetres[j].shape)
                        for j in geo.niveaux_gpu}
    assert len(shapes_attendues) == 3          # (1,1,..), (1,2,..), (2,2,..)
    assert len(applicateur._tampons) == len(shapes_attendues)
    for j in geo.niveaux_gpu:
        fen = pyramide.fenetres[j]
        assert fen.shape[1] == geo.n_systemes_du_niveau(j)
        assert bool(cp.isfinite(fen).all())


# ----- lecture mécanique : les branches pré-écrites -----

def _v2_factice(mediane_frame_ms: float, transferts_ms: float = 1.3,
                residence_octets: int = 1024 ** 3) -> dict:
    return {
        "frame_time": {"mediane_ms": mediane_frame_ms,
                       "p99_ms": mediane_frame_ms},
        "transferts_regime": {
            "transfert_total_ms": {"mediane_ms": transferts_ms}},
        "residence": {"mempool_total_octets": residence_octets},
        "cout_predit_ms": COUT_PREDIT_GRAVE_MS}


def test_lecture_scinde_frame_transferts_et_calcul():
    """S2 : les deux grandeurs sont reportées EN ÉVIDENCE et le calcul est
    exactement frame − transferts."""
    lecture = lecture_mecanique(_v2_factice(15.6, transferts_ms=1.3))
    assert lecture["mediane_frame_complete_ms"] == 15.6
    assert lecture["mediane_transferts_ms"] == 1.3
    assert lecture["mediane_calcul_ms"] == pytest.approx(14.3)
    assert lecture["modele_de_cout"]["grandeur_testee"].startswith(
        "composante CALCUL")


def test_lecture_bande_porte_sur_le_calcul_pas_sur_la_frame():
    """Le cas que S2 existe pour trancher : frame = 15.6 ms (sous le
    seuil), calcul = 14.3 ms DANS la bande. Sans S2, on aurait testé la
    bande sur 15.6 et fabriqué un AUTRE prévu d'avance."""
    lecture = lecture_mecanique(_v2_factice(15.6, transferts_ms=1.3))
    assert lecture["mediane_depasse_seuil"] is False
    assert lecture["modele_de_cout"]["calcul_hors_bande"] is False


def test_lecture_mort_porte_sur_la_frame_complete():
    """MORT sur la frame COMPLÈTE > 16.7 ms (T2 = budget de frame,
    transferts compris) — même si le CALCUL, lui, reste dans la bande."""
    lecture = lecture_mecanique(_v2_factice(16.8, transferts_ms=1.5))
    assert lecture["mediane_depasse_seuil"] is True
    assert lecture["mediane_calcul_ms"] == pytest.approx(15.3)
    assert lecture["modele_de_cout"]["calcul_hors_bande"] is False


def test_lecture_calcul_hors_bande_est_un_autre_sans_mort():
    """Calcul sous 12.18 ms : le modèle linéaire en slots est faux
    (AUTRE), sans aucune mort."""
    lecture = lecture_mecanique(_v2_factice(10.0, transferts_ms=1.0))
    assert lecture["mediane_depasse_seuil"] is False
    assert lecture["modele_de_cout"]["calcul_hors_bande"] is True


def test_lecture_gate_residence():
    lecture = lecture_mecanique(
        _v2_factice(14.3, residence_octets=GATE_RESIDENCE_OCTETS + 1))
    assert lecture["gate_residence"]["declenche"] is True
    assert "PAS une mort" in lecture["gate_residence"]["note"]


# ----- les chiffres gravés ne bougent pas -----

def test_constantes_gravees():
    """Aucun seuil ne bouge (§A16) — verrou explicite."""
    assert B_FRAME_MS == 16.7
    assert BANDE_MODELE_MS == (12.18, 16.48)
    assert COUT_PREDIT_GRAVE_MS == 14.33
    assert GATE_RESIDENCE_OCTETS == int(1.35 * 1024 ** 3)
    assert MS_PAR_SLOT == {8: 1.685, 4: 0.843}
