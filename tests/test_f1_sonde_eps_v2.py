"""Tests F1 — sonde EPS v2 : l'observable corrigé (§A21-complément,
pocCascade2phys f943e9a).

Ce que ces tests protègent :
  1. **LE VERROU** — la connaissance du CPU lue sur `reference` ne peut
     pas diverger du grand livre des coefficients reçus. C'est l'identité
     qui n'existait PAS en v1, et dont l'absence a produit le faux
     verdict ;
  2. les DEUX vérités sont déclarées avant le run, et c'est vérité(n) —
     celle que le joueur voit — qui porte la règle ;
  3. le point d'observation est le SEUL où la connaissance est dans les
     coordonnées de la frame courante ;
  4. rien ne tourne avant endossement du pré-enregistrement neuf."""
import numpy as np
import pytest

from scripts.run_f1_ma_quater import PipelineMaQuater, slots_v4
from scripts.run_f1_sonde_eps import (
    CHAMP_SEDIMENT,
    ENDOSSEMENT_PREREG_V2,
    EPS_EN_VIGUEUR,
    SEUIL_IC_BAS,
    VERITE_COURANTE,
    VERITE_TRANSPORTEE,
    ConnaissanceCpu,
    ReconstructeurNiveau0,
    _slot_observe,
    exiger_endossement,
    lecture_mecanique,
)
from src.f1_gpu.backend import cupy_disponible, vers_cpu
from src.f1_gpu.pyramide import GeometriePyramide
from src.f1_gpu.transferts import TransfertComptable

gpu_requis = pytest.mark.skipif(
    not cupy_disponible(), reason="cupy/device CUDA indisponible")

GEO_PETITE = GeometriePyramide(n_fov=64, n_niv=10, n0=8,
                               slots=slots_v4(10), emboitee=True)
CADENCE: int = 4


def _pipeline(cp, capturer: bool = False) -> PipelineMaQuater:
    return PipelineMaQuater(cp, GEO_PETITE, TransfertComptable(cp),
                            k=CADENCE, eps=EPS_EN_VIGUEUR,
                            capturer_coefficients=capturer)


# ----- 1. LE VERROU : la connaissance ne peut pas diverger du grand livre -----

@gpu_requis
def test_verrou_la_connaissance_lue_egale_le_grand_livre_applique():
    """L'IDENTITÉ QUI N'EXISTAIT PAS.

    Ce que la sonde LIT (`ConnaissanceCpu`, capturée avant l'émission de
    la frame n) doit être EXACTEMENT ce que le CPU obtiendrait en
    APPLIQUANT les coefficients qu'il a reçus. Toute divergence est le
    défaut de v1, et le test n'a pas d'autre raison d'être.

    La paire comparée tient compte du décalage d'une frame que le
    ping-pong garantit (§A21) : à la frame n le CPU reçoit l'émission de
    n−1, et `reference` avant l'émission de n porte exactement
    Σ(émissions ≤ n−1). Comparer la connaissance à `reference` APRÈS
    l'émission de n serait s'accorder une frame d'avance.

    Trajectoire IMMOBILE (delta_x = 0) : aucune maintenance de
    coordonnées n'intervient, donc l'identité est nue."""
    import cupy as cp
    pipeline = _pipeline(cp, capturer=True)
    indice, position = _slot_observe(pipeline)
    observateur = ConnaissanceCpu(indice, position)
    grand_livre = [vers_cpu(r).astype(np.float64)
                   for r in pipeline.references]

    for _ in range(3 * CADENCE):
        pipeline.frame(delta_x=0, observateur=observateur)
        for groupe, valeurs, indices in pipeline.coefficients_frame:
            if valeurs.size:
                np.add.at(grand_livre[groupe].reshape(-1), indices, valeurs)
        applique = grand_livre[indice][position, 0, CHAMP_SEDIMENT]
        # Le device tient sa référence en f32, le grand livre cumule en
        # f64 : l'identité est exacte à l'arrondi f32 près, jamais plus.
        np.testing.assert_allclose(applique, observateur.connaissance,
                                   rtol=0, atol=1e-6)


@gpu_requis
def test_verrou_le_verrou_mord_si_la_connaissance_derive():
    """Contre-épreuve : un verrou qui n'attrape rien ne prouverait rien.
    On injecte une dérive de l'ampleur exacte que v1 produisait, et le
    verrou doit la voir."""
    import cupy as cp
    pipeline = _pipeline(cp)
    indice, position = _slot_observe(pipeline)
    pipeline.frame(delta_x=0)
    lu = vers_cpu(
        pipeline.references[indice][position, 0, CHAMP_SEDIMENT]
    ).astype(np.float64)
    derive = lu.copy()
    derive[0, 0] += 0.082          # l'ampleur du plancher de v1
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(derive, lu, rtol=0, atol=1e-5)


@gpu_requis
def test_le_defaut_de_v1_est_exhibe_par_le_deplacement():
    """Ce que le VERROU ne doit PAS masquer : dès que la fovéa bouge, la
    reconstruction indépendante de v1 s'écarte de `reference`, parce
    qu'elle n'applique ni le roll ni les colonnes prédites.

    Ce test fixe le diagnostic — il dit POURQUOI l'option 1 était
    nécessaire, et il documente ce que l'option 1 CRÉDITE au CPU (la
    prédiction du device). Sans lui, on pourrait croire que v1 était
    seulement imprécise."""
    import cupy as cp
    pipeline = _pipeline(cp, capturer=True)
    indice, position = _slot_observe(pipeline)
    ancien = ReconstructeurNiveau0(pipeline)

    for _ in range(CADENCE):
        pipeline.frame(delta_x=1)          # fovéa MOBILE (E4c)
        ancien.appliquer(pipeline.coefficients_frame)

    lu = vers_cpu(
        pipeline.references[indice][position, 0, CHAMP_SEDIMENT]
    ).astype(np.float64)
    ecart = float(np.max(np.abs(ancien.champ(indice, position) - lu)))
    assert ecart > 1e-3, (
        "le décalage spatial de v1 doit rester visible : c'est lui qui a "
        "produit le plancher 0.082")


# ----- 2. les deux vérités, déclarées avant le run -----

@gpu_requis
def test_les_deux_verites_sont_capturees_dans_les_memes_coordonnees():
    """K(n) et S̃(n−1) sont pris au MÊME instant, donc dans les mêmes
    coordonnées : c'est ce qui rend leur comparaison licite."""
    import cupy as cp
    pipeline = _pipeline(cp)
    indice, position = _slot_observe(pipeline)
    observateur = ConnaissanceCpu(indice, position)
    pipeline.frame(delta_x=1, observateur=observateur)
    assert observateur.connaissance is not None
    assert observateur.verite_transportee is not None
    assert observateur.connaissance.shape == (GEO_PETITE.n_fov,
                                              GEO_PETITE.n_fov)
    assert observateur.connaissance.shape == (
        observateur.verite_transportee.shape)


@gpu_requis
def test_lobservateur_lit_avant_lemission_de_la_frame():
    """Le point d'arrêt est APRÈS le déplacement et AVANT l'émission : la
    connaissance capturée ne doit PAS contenir ce que cette frame vient
    d'émettre — sinon la sonde s'accorderait une frame d'avance."""
    import cupy as cp
    pipeline = _pipeline(cp)
    indice, position = _slot_observe(pipeline)
    observateur = ConnaissanceCpu(indice, position)
    ecarts = []
    # Un CYCLE entier : le slot observé n'émet qu'un tour sur k (L1
    # étalée), donc c'est sur le cycle que la différence doit apparaître.
    for _ in range(CADENCE):
        pipeline.frame(delta_x=0, observateur=observateur)
        apres = vers_cpu(
            pipeline.references[indice][position, 0, CHAMP_SEDIMENT]
        ).astype(np.float64)
        ecarts.append(float(np.max(np.abs(apres
                                          - observateur.connaissance))))
    # L'émission entre dans `reference` APRÈS la capture : au tour du
    # slot, la connaissance capturée ne la contient pas encore.
    assert max(ecarts) > 0.0


def test_la_verite_de_la_regle_est_celle_que_le_joueur_voit():
    """Déclaré AVANT le run : vérité(n) porte la règle, vérité(n−1) est
    un diagnostic. Les confondre refabriquerait un artefact."""
    assert VERITE_COURANTE == "verite_n"
    assert VERITE_TRANSPORTEE == "verite_n_moins_1"
    lecture = _lecture_factice()
    verites = lecture["verite_de_lecture"]
    assert verites["verite_de_la_regle"] == VERITE_COURANTE
    assert verites["verite_diagnostique"] == VERITE_TRANSPORTEE
    assert verites["declaree_avant_run"] is True
    assert "instant présent" in verites["motif"]
    assert "Jamais la règle" in verites["usage_du_diagnostic"]


def _mesure(eps: float, delta_chi_max: float,
            delta_chi_max_n1: float | None = None) -> dict:
    if delta_chi_max_n1 is None:
        delta_chi_max_n1 = delta_chi_max
    par_verite = {
        VERITE_COURANTE: {
            "delta_chi_max": delta_chi_max,
            "delta_chi_max_sans_decimation": delta_chi_max},
        VERITE_TRANSPORTEE: {
            "delta_chi_max": delta_chi_max_n1,
            "delta_chi_max_sans_decimation": delta_chi_max_n1},
    }
    return {
        "eps": eps,
        "chrono": {"octets_par_frame_median": 1000 * (1 + eps),
                   "temps_transfert_median_ms": 2.0,
                   "frame_time": {"mediane_ms": 16.0, "p99_ms": 17.0}},
        "delta_chi": {
            "delta_chi_max": delta_chi_max,
            "verite_de_la_regle": VERITE_COURANTE,
            "par_verite": par_verite,
            "prix_peremption_une_frame": delta_chi_max - delta_chi_max_n1},
    }


def _lecture_factice() -> dict:
    mesures = [_mesure(1e-5, 0.02), _mesure(1e-4, 0.03), _mesure(1e-2, 0.09)]
    verification = {"instrument_valide": True, "branche": "i",
                    "label": "INSTRUMENT VALIDE"}
    return lecture_mecanique(mesures, verification, plancher=0.0)


def test_la_regle_q2_est_reconduite_pour_lobservable_corrige():
    """Le PLUS GRAND EPS dont Δχ max reste sous 0.0603 — inchangée dans
    sa forme, réénoncée sur le nouvel observable."""
    lecture = _lecture_factice()
    assert lecture["seuil_ic_bas"] == SEUIL_IC_BAS
    assert lecture["eps_retenu"] == 1e-4          # 1e-2 dépasse le seuil
    assert lecture["eps_satisfaisants"] == [1e-5, 1e-4]
    assert lecture["autre_remonte"] is False
    assert VERITE_COURANTE in lecture["regle"]


def test_la_portee_option_1_est_dite_dans_la_lecture():
    """La sonde mesure le CANAL, pas la prédiction propre au CPU. Cette
    limite doit voyager AVEC le résultat, pas rester dans un docstring."""
    portee = _lecture_factice()["portee_option_1"]
    assert "CANAL" in portee
    assert "miroir CPU" in portee
    assert "le canal l'est" in portee


# ----- 3. l'ancien reconstructeur est marqué, pas effacé -----

def test_lancien_reconstructeur_est_marque_superseded():
    """On ne réécrit pas le passé : `ReconstructeurNiveau0` reste, parce
    que des lectures ACQUISES en dépendent — mais il porte son statut."""
    doc = " ".join(ReconstructeurNiveau0.__doc__.split())
    assert "SUPERSEDED" in doc
    assert "NE PAS EMPLOYER POUR UNE LECTURE NEUVE" in doc
    assert "DÉCALÉES" in doc
    assert "lectures ACQUISES en dépendent" in doc


def test_la_connaissance_declare_sa_limite():
    """L'option 1 crédite le CPU de la prédiction du device. C'est une
    limite déclarée, et elle doit être lisible là où elle s'applique."""
    doc = ConnaissanceCpu.__doc__
    assert "CRÉDITER" in doc
    assert "option 2" in doc
    assert "CANAL" in doc


# ----- 4. rien ne tourne avant endossement -----

def test_le_prereg_v2_est_endosse_et_le_verrou_reste_armé():
    """ENDOSSÉ au §A22-complément (a4e06f6) : le drapeau est levé, et le
    verrou reste EN PLACE — il redira non si quelqu'un le rabaisse. Un
    verrou qu'on retire après usage ne protège plus rien."""
    import scripts.run_f1_sonde_eps as sonde

    assert ENDOSSEMENT_PREREG_V2 is True
    exiger_endossement()                       # ne lève plus

    ancien = sonde.ENDOSSEMENT_PREREG_V2
    try:
        sonde.ENDOSSEMENT_PREREG_V2 = False
        with pytest.raises(RuntimeError, match="NON ENDOSSÉ"):
            sonde.exiger_endossement()
    finally:
        sonde.ENDOSSEMENT_PREREG_V2 = ancien


def test_le_verrou_dit_de_quelle_decision_il_procede():
    """Un drapeau qui ne dit pas d'où il vient n'est qu'un interrupteur :
    le commentaire qui l'accompagne doit citer la décision."""
    import inspect

    from scripts import run_f1_sonde_eps

    source = inspect.getsource(run_f1_sonde_eps)
    tete = source[:source.index("ENDOSSEMENT_PREREG_V2: bool")]
    contexte = tete[-700:]
    assert "a4e06f6" in contexte
    assert "§A22-complément" in contexte
    assert "PROPOSE" in contexte      # ce qu'il autorise, et rien de plus


def test_le_verrou_dendossement_precede_toute_allocation():
    """`main` appelle `exiger_endossement` AVANT `exiger_cupy` : un run
    non endossé ne doit pas même toucher au device."""
    import inspect

    from scripts import run_f1_sonde_eps

    source = inspect.getsource(run_f1_sonde_eps.main)
    assert source.index("exiger_endossement") < source.index("exiger_cupy")
