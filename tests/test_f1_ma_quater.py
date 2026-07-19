"""Tests F1 — M-a-quater, pipeline complet V4 (achat B,
§A18-lecture-borne-L3, pocCascade2phys 7c4a4ca).

Trois choses doivent être PROUVÉES, pas supposées :
  1. la géométrie V4 est un ARBRE (propriété E) — c'est ce qui rend
     caducs les cinq replis de s3, donc ce qui autorise 10 slots
     GPU-side et le niveau 1 seul en CPU ;
  2. L1 est ÉTALÉE — aucune frame ne remonte tout, et un cycle couvre
     chaque slot exactement une fois. C'est une consigne d'INTÉGRITÉ :
     une rafale ferait passer la médiane en cachant un stutter ;
  3. les champs d'échelle sont prédits à ZÉRO exact, tandis que l'état du
     harnais reste MOUILLÉ (l'anti-minoration a mesuré −16.9 %).
Le reste couvre la config, la lecture et l'absence de verdict."""
import json

import numpy as np
import pytest

from scripts.run_f1_ma_quater import (
    ANCRES_MESUREES_MS,
    BANDE_MS,
    CADENCE_L1,
    F_BATCHE_V4_MS,
    REMONTEE_L3_MESUREE_MS,
    PipelineMaQuater,
    cout_predit_f_ms,
    diagnostic_round_robin,
    geometrie_v4,
    lecture_mecanique,
    parts_derivees,
    plan_prediction_v4,
    plan_round_robin,
    resume_prediction,
    slots_v4,
)
from scripts.run_f1_ma_ter import B_FRAME_MS, N_NIV
from scripts.run_f1_s2_mobile import plan_groupes
from src.f1_gpu.backend import cupy_disponible
from src.f1_gpu.pyramide import GeometriePyramide
from src.f1_gpu.transferts import TransfertComptable

gpu_requis = pytest.mark.skipif(
    not cupy_disponible(), reason="cupy/device CUDA indisponible")

# V4 réelle en réduction : mêmes 11 slots, n_fov minuscule.
GEO_PETITE = GeometriePyramide(n_fov=8, n_niv=N_NIV, n0=8,
                               slots=slots_v4(N_NIV), emboitee=True)


# ----- 1. la géométrie V4 est un ARBRE (propriété E) -----

def test_v4_est_un_arbre_la_ou_b3_ne_letait_pas():
    """LE point de l'achat B : les offsets ±n_fov/2 placent les fenêtres
    d'énergie DANS la couverture parente, là où les ±n_fov de B3 les
    laissaient dehors — c'est le trou surfacé par s3, refermé."""
    emboitee = geometrie_v4()
    plate = GeometriePyramide(n_fov=512, n_niv=N_NIV, slots=slots_v4(N_NIV))
    assert emboitee.slots_sans_parente() == []          # arbre
    assert plate.slots_sans_parente() != []             # B3 ne l'est pas


def test_propriete_e_tient_tout_le_long_du_balayage():
    """Un emboîtement qui ne tiendrait qu'à la position initiale ne
    servirait à rien : la fovéa bouge."""
    geo = geometrie_v4()
    centre = geo.centre_fin_initial()
    for decalage in (0, 1, 7, 63, 255, 399):
        assert geo.slots_sans_parente(centre + decalage) == []


def test_v4_a_onze_slots_quinze_blocs_et_le_cout_grave():
    """Composition §A18 P3, sur les ancres MESURÉES (1.685 / 0.845 — et
    non l'étiquette 0.843 d'avant s1) : c'est elle qui donne 12.655."""
    geo = geometrie_v4()
    assert geo.n_slots_total == 11
    assert geo.blocs_actifs_gpu == 15
    assert cout_predit_f_ms(geo) == pytest.approx(F_BATCHE_V4_MS, abs=0.001)
    assert ANCRES_MESUREES_MS == {8: 1.685, 4: 0.845}


def test_zero_tour_dancetres():
    """Les 2 énergie tiennent dans la parente fovéale existante : aucun
    ancêtre supplémentaire à compter au cap."""
    geo = geometrie_v4()
    fin = N_NIV - 1
    assert geo.n_slots(fin) == 3            # fovéal + 2 énergie
    assert all(geo.n_slots(j) == 1 for j in range(1, fin))


# ----- 2. la prédiction : 10 GPU-side, niveau 1 seul CPU -----

def test_dix_slots_gpu_side_et_le_niveau_1_seul_en_cpu():
    """La propriété E rend caducs les cinq replis de s3 — c'est la
    vérification exigée."""
    resume = resume_prediction(geometrie_v4())
    assert resume["n_slots_gpu"] == 10
    assert resume["n_slots_cpu"] == 1
    assert resume["slots_cpu"] == [{"niveau": 1, "slot": 0}]
    assert resume["propriete_e_satisfaite"] is True


def test_le_plan_refuse_une_geometrie_non_emboitee():
    """Sur une géométrie plate, la descente parent→enfant n'est pas
    définie : fail-loud plutôt qu'une prédiction inventée."""
    plate = GeometriePyramide(n_fov=512, n_niv=N_NIV, slots=slots_v4(N_NIV))
    with pytest.raises(RuntimeError, match="propriété E violée"):
        plan_prediction_v4(plate)


def test_un_seul_niveau_porte_des_champs_dechelle():
    """Seul le fovéal du niveau 8 (c=8) a un parent c=4."""
    resume = resume_prediction(geometrie_v4())
    assert resume["niveaux_a_champs_echelle"] == [N_NIV - 2]


# ----- 3. L1 ÉTALÉE : la consigne d'intégrité -----

def test_aucune_frame_ne_remonte_tout():
    """LE piège fermé : une rafale ferait passer la médiane en laissant
    un stutter invisible au critère."""
    diagnostic = diagnostic_round_robin(plan_groupes(geometrie_v4()),
                                        CADENCE_L1)
    assert diagnostic["aucune_frame_ne_remonte_tout"] is True
    assert max(diagnostic["slots_par_tour"]) < diagnostic["slots_total"]


def test_un_cycle_couvre_chaque_slot_exactement_une_fois():
    """Même amortissement qu'une rafale : sur k frames, tout est remonté
    une fois et une seule."""
    diagnostic = diagnostic_round_robin(plan_groupes(geometrie_v4()),
                                        CADENCE_L1)
    assert diagnostic["cycle_couvre_chaque_slot_une_fois"] is True
    assert sum(diagnostic["slots_par_tour"]) == diagnostic["slots_total"]
    assert diagnostic["slots_total"] == 11


def test_les_tours_sont_des_tranches_contigues_et_disjointes():
    """La contiguïté permet d'appeler L3 sur une vue sans réordonner les
    buffers ni toucher aux kernels."""
    plans = plan_groupes(geometrie_v4())
    tours = plan_round_robin(plans, CADENCE_L1)
    for indice, plan in enumerate(plans):
        couvert: list[int] = []
        for tranches in tours:
            debut, fin = tranches[indice]
            assert debut <= fin
            couvert.extend(range(debut, fin))
        assert sorted(couvert) == list(range(plan["n_slots"]))


@pytest.mark.parametrize("k", [2, 4, 8])
def test_letalement_tient_pour_dautres_cadences(k):
    diagnostic = diagnostic_round_robin(plan_groupes(geometrie_v4()), k)
    assert diagnostic["cycle_couvre_chaque_slot_une_fois"] is True
    assert diagnostic["aucune_frame_ne_remonte_tout"] is True


def test_cadence_invalide_fail_loud():
    with pytest.raises(ValueError, match="k=0"):
        plan_round_robin(plan_groupes(geometrie_v4()), 0)


# ----- 4. champs d'échelle : zéro exact, harnais mouillé -----

@gpu_requis
def test_letat_initial_garde_ses_deux_systemes_mouilles():
    """L'anti-minoration a mesuré −16.9 % : un second système sec coûte
    franchement moins. Le harnais reste donc mouillé."""
    import cupy as cp
    pipeline = PipelineMaQuater(cp, GEO_PETITE, TransfertComptable(cp))
    for bloc in pipeline.fenetres:
        if bloc.shape[1] == 2:
            assert float(bloc[:, 1, 0].max()) > 0.0     # h du système 1


@gpu_requis
def test_les_champs_dechelle_sont_predits_a_zero_exact():
    """Conformité P2 : les systèmes au-delà de c_parent arrivent à ZÉRO
    EXACT. Le test porte sur la PRÉDICTION seule (`_deplacer`), car F
    s'applique juste après et repropage depuis les voisins mouillés —
    voir `test_f_rehumidifie_la_colonne_predite`."""
    import cupy as cp
    pipeline = PipelineMaQuater(cp, GEO_PETITE, TransfertComptable(cp))
    niveau = N_NIV - 2
    for _ in range(6):
        if niveau in pipeline._deplacer(1)["niveaux_deplaces"]:
            break
    vue = pipeline._vue_slot(pipeline.fenetres, niveau, 0)
    assert float(cp.abs(vue[1, :, :, -1]).max()) == 0.0   # système 1 nul
    assert float(cp.abs(vue[0, :, :, -1]).max()) > 0.0    # système 0 vivant


@gpu_requis
def test_f_rehumidifie_la_colonne_predite():
    """Le chemin est VIVANT : après F, la colonne entrante n'est plus
    nulle — le stencil propage depuis les voisins mouillés. L'assèchement
    des champs d'échelle est donc ATTÉNUÉ, ce qui borne d'autant la
    minoration que l'anti-minoration (−16.9 %) mesurait sur un système
    ENTIÈREMENT sec. À garder en tête à la lecture."""
    import cupy as cp
    pipeline = PipelineMaQuater(cp, GEO_PETITE, TransfertComptable(cp))
    niveau = N_NIV - 2
    for _ in range(6):
        if niveau in pipeline.frame()["niveaux_deplaces"]:
            break
    vue = pipeline._vue_slot(pipeline.fenetres, niveau, 0)
    apres_f = float(cp.abs(vue[1, :, :, -1]).max())
    assert apres_f > 0.0                      # F a repropagé
    assert apres_f < float(cp.abs(vue[0, :, :, -1]).max())   # mais faible


@gpu_requis
def test_lassechement_est_mesure_et_reporte():
    """Le prix de la conformité P2 est chiffré, pas tu."""
    import cupy as cp
    pipeline = PipelineMaQuater(cp, GEO_PETITE, TransfertComptable(cp))
    for _ in range(8):
        pipeline.frame()
    assechement = pipeline.assechement_champs_echelle()
    assert assechement["colonnes_mises_a_zero"] > 0
    assert 0.0 <= assechement["fraction_asséchée"] <= 1.0
    assert assechement["ecart_sec_mouille_mesure"] == -0.169


# ----- 5. le pipeline tourne -----

@gpu_requis
def test_le_pipeline_complet_traverse_le_device():
    """Plomberie (taille minuscule, JAMAIS une mesure) : déplacement,
    prédiction GPU-side, F batché, remontée L3 étalée, transferts."""
    import cupy as cp
    transferts = TransfertComptable(cp)
    pipeline = PipelineMaQuater(cp, GEO_PETITE, transferts)
    transferts.frame_suivante()
    tours_vus = set()
    for _ in range(8):
        diagnostic = pipeline.frame()
        transferts.frame_suivante()
        tours_vus.add(diagnostic["tour_l1"])
    assert tours_vus == set(range(CADENCE_L1))      # le round-robin tourne
    for bloc in pipeline.fenetres:
        assert bool(cp.isfinite(bloc).all())


@gpu_requis
def test_le_nombre_de_lancements_est_reporte():
    """Scinder L3 / fusionné porte les lancements de 4 à 8-12 : le compte
    réel est reporté, pas estimé (choix d'implémentation remonté)."""
    import cupy as cp
    pipeline = PipelineMaQuater(cp, GEO_PETITE, TransfertComptable(cp))
    lancements = {pipeline.frame()["lancements"] for _ in range(8)}
    assert lancements
    assert all(4 <= n <= 12 for n in lancements)


@gpu_requis
def test_la_remontee_ne_transfere_pas_a_la_premiere_frame():
    """Retard d'une frame du compacteur L3, reconduit tel quel."""
    import cupy as cp
    transferts = TransfertComptable(cp)
    pipeline = PipelineMaQuater(cp, GEO_PETITE, transferts)
    pipeline.frame()
    assert transferts.frame_suivante()["d2h_octets"] == 0


# ----- 6. lecture mécanique -----

def _stats(mediane: float, p99: float | None = None) -> dict:
    return {"mediane_ms": mediane, "p99_ms": p99 if p99 else mediane + 1.0}


def _lecture(mediane: float, p99: float | None = None, transferts: float = 0.4):
    geo = geometrie_v4()
    return lecture_mecanique(
        _stats(mediane, p99), transferts, resume_prediction(geo),
        diagnostic_round_robin(plan_groupes(geo), CADENCE_L1),
        {"fraction_asséchée": 0.3, "ecart_sec_mouille_mesure": -0.169,
         "colonnes_mises_a_zero": 1, "colonnes_de_reference": 3,
         "note": "reporté"},
        CADENCE_L1)


def test_mort_sur_la_mediane_frame_complete():
    assert _lecture(16.8)["mort"]["declenchee"] is True
    assert _lecture(15.0)["mort"]["declenchee"] is False
    assert _lecture(16.8)["mort"]["seuil_ms"] == B_FRAME_MS


def test_bande_recalculee_avec_l1():
    lecture = _lecture(15.0)
    assert lecture["bande_modele"]["bande_ms"] == [14.2, 15.9]
    assert lecture["bande_modele"]["hors_bande"] is False
    assert _lecture(13.0)["bande_modele"]["hors_bande"] is True
    assert "L1 k=4" in lecture["bande_modele"]["origine"]


def test_p99_est_reportee_en_evidence():
    """C'est là que se lirait un stutter : l'étalement de L1 doit la
    garder proche de la médiane."""
    lecture = _lecture(15.0, p99=17.5)
    evidence = lecture["p99_en_evidence"]
    assert evidence["valeur_ms"] == 17.5
    assert evidence["ecart_a_la_mediane_ms"] == pytest.approx(2.5)
    assert "stutter" in evidence["note"]


def test_parts_sont_etiquetees_derive_et_bouclent():
    """F = ancre, remontée = borne amortie par k, transferts COMPTÉS,
    prédiction = SOLDE — la somme reconstitue la médiane."""
    parts = parts_derivees(15.0, 0.4, CADENCE_L1)
    assert parts["etiquette"] == "[DÉRIVÉ]"
    assert parts["f_batche_ms"] == F_BATCHE_V4_MS
    assert parts["remontee_l3_amortie_ms"] == pytest.approx(
        REMONTEE_L3_MESUREE_MS / CADENCE_L1)
    somme = (parts["f_batche_ms"] + parts["remontee_l3_amortie_ms"]
             + parts["transferts_ms"] + parts["prediction_solde_ms"])
    assert somme == pytest.approx(15.0)
    assert "MORT porte sur la médiane MESURÉE" in parts["note"]


def test_la_lecture_porte_lassechement_et_letalement():
    lecture = _lecture(15.0)
    assert lecture["assechement_champs_echelle"]["fraction_asséchée"] == 0.3
    assert lecture["l1_round_robin"]["aucune_frame_ne_remonte_tout"] is True


def test_le_driver_ne_prononce_rien():
    lecture = _lecture(15.0)
    texte = json.dumps(lecture, ensure_ascii=False)
    for interdit in ("PASS", "VERDICT"):
        assert interdit not in texte
    # « MORT » n'apparaît que comme NOM de critère et branche pré-écrite.
    assert lecture["mort"]["branche_preecrite"].startswith("V4 mort")
    assert "revient à Romain" in lecture["rappel_portee"]


def test_constantes_gravees():
    assert BANDE_MS == (14.2, 15.9)
    assert F_BATCHE_V4_MS == 12.655
    assert REMONTEE_L3_MESUREE_MS == 5.122
    assert CADENCE_L1 == 4
    assert B_FRAME_MS == 16.7


def test_les_slots_de_v4_sont_bien_places():
    """9 fovéale (2 fins c=8, 7 c=4) + 2 énergie c=8 au niveau le plus
    fin."""
    slots = slots_v4()
    foveaux = [s for s in slots if s.role == "fovea"]
    energie = [s for s in slots if s.role == "energie"]
    assert len(foveaux) == 9 and len(energie) == 2
    assert sum(1 for s in foveaux if s.c == 8) == 2
    assert sum(1 for s in foveaux if s.c == 4) == 7
    assert all(s.c == 8 and s.niveau == N_NIV - 1 for s in energie)
    assert np.isclose(cout_predit_f_ms(geometrie_v4()), 12.655, atol=1e-3)
