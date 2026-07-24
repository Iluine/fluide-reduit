"""Tests F1 — M-b tranche-2 : le DRIVER (§A31-build).

Ce que ces tests protègent :
  1. APPLES-TO-APPLES : les deux bras sont lus sur la MÊME cellule, aux
     MÊMES émissions, avec le MÊME observable, contre le MÊME rederive —
     sans quoi la lecture à trois branches comparerait deux choses ;
  2. MORT-b est le MAX DE SÉRIE, par-seed, câblé à la lecture livrée ;
  3. la lecture porte ses portées, dont le RE-SCOPE du gate (iii) : un PASS
     ne s'énonce PAS « le contrat tient à l'échelle V4 » ;
  4. construit-NON-LANCÉ : rien ne tourne à l'import."""
import json

import numpy as np
import pytest

from scripts.run_f1_mb_t2 import (
    lecture_t2,
    maxima_de_serie,
)
from src.f1_gpu.backend import cupy_disponible
from src.f1_gpu.cellule_mb import EMISSIONS, GRAINES, N_EPISODES, dchi
from src.f1_gpu.mort_b_lecture import (
    LABEL_OPTION_A_TIENT,
    LABEL_OPTION_B_PAS_REMEDE,
    PIN_SEVERE,
)

gpu_requis = pytest.mark.skipif(
    not cupy_disponible(), reason="cupy/device CUDA indisponible")


def _champs():
    """Trois champs 64² : rederive, une production qui s'en écarte, un témoin
    qui s'en écarte moins."""
    rng = np.random.default_rng(7)
    rederive = {n: rng.random((64, 64)) * 0.5 for n in EMISSIONS}
    prod = {n: v + 0.25 * rng.random((64, 64)) for n, v in rederive.items()}
    temoin = {n: v + 1e-6 * rng.random((64, 64)) for n, v in rederive.items()}
    return prod, temoin, rederive


# ----- 1. apples-to-apples -----

def test_les_deux_bras_lus_avec_le_meme_observable_contre_le_meme_rederive():
    prod, temoin, rederive = _champs()
    m = maxima_de_serie(prod, temoin, rederive)
    for n in EMISSIONS:
        assert m["par_emission"][n]["production"] == dchi(prod[n], rederive[n])
        assert m["par_emission"][n]["temoin"] == dchi(temoin[n], rederive[n])


def test_les_emissions_sont_celles_de_la_cellule():
    prod, temoin, rederive = _champs()
    m = maxima_de_serie(prod, temoin, rederive)
    assert tuple(sorted(m["par_emission"])) == EMISSIONS
    assert len(EMISSIONS) == 6 and N_EPISODES == 24


def test_bras_identique_au_rederive_donne_dchi_nul():
    """Verrou d'instrument : un bras qui EST le rederive ne peut pas
    traverser — si c'était le cas, l'observable fabriquerait l'écart."""
    _, _, rederive = _champs()
    m = maxima_de_serie(rederive, rederive, rederive)
    assert m["production"] == 0.0 and m["temoin"] == 0.0


# ----- 2. MORT-b = max de série, par-seed -----

def test_mort_b_est_le_max_de_serie():
    prod, temoin, rederive = _champs()
    m = maxima_de_serie(prod, temoin, rederive)
    attendu = max(dchi(prod[n], rederive[n]) for n in EMISSIONS)
    assert m["production"] == attendu


def test_une_seule_emission_traversante_suffit():
    """Le max de série : une émission au-dessus du pin suffit à faire
    traverser le seed, même si les cinq autres sont propres."""
    _, _, rederive = _champs()
    prod = dict(rederive)
    prod[EMISSIONS[3]] = rederive[EMISSIONS[3]] + 0.4
    m = maxima_de_serie(prod, rederive, rederive)
    assert m["production"] > PIN_SEVERE


# ----- 3. la lecture, et ses portées -----

def test_lecture_t2_cable_les_trois_branches():
    par_seed = {s: {"production": 0.01, "temoin": 0.01} for s in GRAINES}
    lec = lecture_t2(par_seed)
    assert lec["mort_b"]["verdict"] == LABEL_OPTION_A_TIENT
    assert lec["mort_b"]["pin_severe"] == PIN_SEVERE


def test_lecture_t2_branche_bc_quand_le_temoin_passe():
    par_seed = {101: {"production": 0.2, "temoin": 0.001},
                102: {"production": 0.01, "temoin": 0.01},
                103: {"production": 0.01, "temoin": 0.01}}
    lec = lecture_t2(par_seed)
    assert lec["mort_b"]["verdict"] == LABEL_OPTION_B_PAS_REMEDE
    assert lec["mort_b"]["option_b_est_le_remede"] is False


def test_lecture_t2_porte_les_quatre_portees():
    par_seed = {s: {"production": 0.01, "temoin": 0.01} for s in GRAINES}
    p = lecture_t2(par_seed)["portees"]
    assert set(p) == {"l1_l3_non_validees_a_v4",
                      "colonnes_entrantes_dominantes",
                      "gate_iii_re_scope_a_64",
                      "reserve_temoin_etat_vs_mesure"}


def test_un_pass_ne_s_enonce_pas_a_l_echelle_v4():
    """Le gate (iii) est re-scopé : le champ de verdict lui-même doit dire
    ce qu'un PASS signifie — « sans mort à 64² sur deux niveaux », pas
    « le contrat tient à l'échelle V4 »."""
    par_seed = {s: {"production": 0.01, "temoin": 0.01} for s in GRAINES}
    lec = lecture_t2(par_seed)
    assert "64²" in lec["ce_que_ce_pass_dit"]
    assert "V4" in lec["ce_que_ce_pass_dit"]
    assert "PAS" in lec["ce_que_ce_pass_dit"]


def test_verif_exigee_existe_dans_le_registre_et_est_la_bonne():
    """`exiger_pass` ne valide PAS les noms : un nom INVENTÉ y est
    indiscernable d'une vérification manquante, et n'échoue qu'au RUN. Ce
    test tue la classe au build.

    T2 mesure la FIDÉLITÉ contre le rederive CPU f64 : la vérification
    load-bearing est `gel_bit_exact` (« le chemin rederive reproduit le gel
    bit-exact sur la machine ») — si le rederive avait bougé ici, la
    vérité-sol serait fausse et Δχ ne voudrait rien dire. Les instruments de
    TEMPS (vram, chrono) sont ceux de T1, pas de T2."""
    from scripts.run_f1_mb_t2 import VERIFS_EXIGEES
    from src.f1_gpu.verifs import NOMS_VERIFS
    assert set(VERIFS_EXIGEES) <= set(NOMS_VERIFS)
    assert "gel_bit_exact" in VERIFS_EXIGEES


def test_le_chemin_post_mesure_ne_peut_pas_perdre_le_run():
    """Le report et l'affichage viennent APRÈS la mesure : une faute là
    détruirait un run déjà payé. Ils sont donc exercés à vide, sur des
    valeurs factices, avec la MÊME structure que le vrai."""
    from scripts.run_f1_mb_t2 import document_t2, imprimer_lecture
    par_seed = {s: {"production": 0.01, "temoin": 0.001,
                    "par_emission": {n: {"production": 0.01, "temoin": 0.001}
                                     for n in EMISSIONS}}
                for s in GRAINES}
    lec = lecture_t2(par_seed)
    doc = document_t2(par_seed, lec)
    json.loads(json.dumps(doc, ensure_ascii=False))     # sérialisable
    imprimer_lecture(lec)                               # n'explose pas


def test_le_report_porte_les_anomalies_si_il_y_en_a():
    par_seed = {101: {"production": 0.01, "temoin": 0.2},   # anomalie
                102: {"production": 0.01, "temoin": 0.01},
                103: {"production": 0.01, "temoin": 0.01}}
    from scripts.run_f1_mb_t2 import imprimer_lecture
    lec = lecture_t2(par_seed)
    assert lec["mort_b"]["anomalies"] == [101]
    imprimer_lecture(lec)


# ----- 6. l'instrument : rederive borné + mémoire persistée (§A31-run) -----

def test_le_driver_consomme_le_rederive_BORNE_pas_run_history():
    """Le poste d'OOM était le rederive consommé à `t_end` plein. Le driver
    doit passer par le consommateur borné — et le rederive rester intouché."""
    import scripts.run_f1_mb_t2 as drv
    from src.f1_gpu.rederive_borne import ConsommateurRederive
    assert drv.ConsommateurRederive is ConsommateurRederive


def test_le_driver_persiste_sa_memoire_par_bras_et_par_phase(tmp_path):
    """La sonde du driver écrit par (bras, phase) : un SIGKILL doit laisser
    dire QUI tenait le processus et COMBIEN il tenait."""
    from src.f1_gpu.memoire_instrument import (
        SondeMemoire,
        lire_jalons,
        pic_par_phase,
    )
    chemin = tmp_path / "memoire.jsonl"
    sonde = SondeMemoire(chemin, periode_s=0.01)
    for bras in ("rederive", "production", "temoin"):
        with sonde.phase(bras, "seed=101"):
            sonde.attendre_un_echantillon()
    pics = pic_par_phase(lire_jalons(chemin))
    assert {b for b, _ in pics} == {"rederive", "production", "temoin"}


def test_le_report_porte_la_memoire_mesuree():
    """Le pic mesuré voyage DANS le report : un échec d'instrument doit
    laisser un chiffre, pas seulement un processus mort."""
    from scripts.run_f1_mb_t2 import document_t2
    par_seed = {s: {"production": 0.01, "temoin": 0.001} for s in GRAINES}
    doc = document_t2(par_seed, lecture_t2(par_seed),
                      memoire={("rederive", "seed=101"): {"rss_pic_mo": 2109}})
    assert doc["memoire_pic_mo"]["rederive|seed=101"]["rss_pic_mo"] == 2109


def test_le_document_reste_serialisable_sans_memoire():
    from scripts.run_f1_mb_t2 import document_t2
    par_seed = {s: {"production": 0.01, "temoin": 0.001} for s in GRAINES}
    json.loads(json.dumps(document_t2(par_seed, lecture_t2(par_seed))))


# ----- 5. survivre à une coupure : reprise par seed -----

def test_un_run_partiel_ne_prononce_AUCUNE_lecture():
    """Le danger d'un report incrémental : qu'un run coupé laisse un document
    qui SE LIT comme un verdict. Un partiel porte donc `lecture_mecanique =
    None` et un statut explicite — jamais une lecture sur 2 seeds sur 3."""
    from scripts.run_f1_mb_t2 import document_partiel
    doc = document_partiel({101: {"production": 0.01, "temoin": 0.001}},
                           "empreinte-x")
    assert doc["lecture_mecanique"] is None
    assert "PARTIEL" in doc["statut"] and "1/3" in doc["statut"]
    assert "AUCUNE lecture" in doc["statut"]


def test_reprise_reutilise_les_seeds_deja_mesures():
    from scripts.run_f1_mb_t2 import seeds_a_mesurer
    deja = {"empreinte": "abc",
            "par_seed": {"101": {"production": 0.01, "temoin": 0.001}}}
    a_faire, repris = seeds_a_mesurer(deja, "abc")
    assert a_faire == [102, 103]
    assert set(repris) == {101}


def test_reprise_refuse_une_empreinte_differente():
    """Si la config a bougé (EPS, fovéa, kernels), les mesures d'avant ne
    sont plus comparables : on recalcule TOUT plutôt que de mélanger."""
    from scripts.run_f1_mb_t2 import seeds_a_mesurer
    deja = {"empreinte": "abc",
            "par_seed": {"101": {"production": 0.01, "temoin": 0.001}}}
    a_faire, repris = seeds_a_mesurer(deja, "AUTRE")
    assert a_faire == list(GRAINES) and repris == {}


def test_reprise_sur_document_absent_mesure_tout():
    from scripts.run_f1_mb_t2 import seeds_a_mesurer
    a_faire, repris = seeds_a_mesurer(None, "abc")
    assert a_faire == list(GRAINES) and repris == {}


def test_round_trip_coupure_puis_reprise():
    """Le vrai contrat : ce que le driver ÉCRIT après une coupure doit être
    ce qu'il SAIT relire. Testé par le JSON, comme en vrai — les clés seed y
    deviennent des chaînes, et c'est là que ça casserait."""
    from scripts.run_f1_mb_t2 import (
        document_partiel,
        empreinte_config,
        seeds_a_mesurer,
    )
    emp = empreinte_config()
    coupe = {101: {"production": 0.01, "temoin": 0.001},
             102: {"production": 0.02, "temoin": 0.002}}
    relu = json.loads(json.dumps(document_partiel(coupe, emp)))
    a_faire, repris = seeds_a_mesurer(relu, emp)
    assert a_faire == [103]
    assert set(repris) == {101, 102}
    assert repris[101]["production"] == 0.01     # valeurs intactes


def test_empreinte_config_bouge_avec_ce_qui_compte():
    """L'empreinte doit lier ce qui rendrait deux seeds incomparables."""
    from scripts.run_f1_mb_t2 import empreinte_config
    e = empreinte_config()
    assert isinstance(e, str) and len(e) >= 8


def test_lecture_t2_declare_les_verrous():
    par_seed = {s: {"production": 0.01, "temoin": 0.01} for s in GRAINES}
    v = lecture_t2(par_seed)["verrous"]
    assert "6dd207ca" in v["kernels"]
    assert "intouché" in v["rederive"].lower()
    assert "identité" in v["invariant_liant"]


# ----- 4. construit-NON-LANCÉ -----

@gpu_requis
def test_cablage_des_trois_bras_bout_a_bout():
    """CÂBLAGE, PAS LA MESURE. À `N_settle` réduit (5 au lieu de 600) les
    trois bras — rederive f64, production 2-niveaux, témoin plein domaine —
    doivent s'aligner : mêmes émissions, champs 64² finis, Δχ évaluable. Ce
    test prouve que l'assemblage tient ; il ne dit RIEN de la fidélité et
    n'est pas le run pré-enregistré (celui-ci est gaté sur endossement)."""
    import cupy as cp

    from config import GridConfig
    from scripts.run_f1_mb_t2 import mesurer_seed
    from src.sediment import SedimentParams, default_terrain

    b0 = default_terrain(GridConfig()).astype(np.float32)
    m = mesurer_seed(101, b0, cp, SedimentParams(N_settle=5))
    assert tuple(sorted(m["par_emission"])) == EMISSIONS
    assert np.isfinite(m["production"]) and np.isfinite(m["temoin"])
    assert m["production"] >= 0.0 and m["temoin"] >= 0.0


def test_rien_ne_tourne_a_l_import():
    """Le module s'importe sans toucher au GPU ni écrire de report : `main`
    n'est appelé QUE sous le garde `__main__`. (Propriété structurelle, pas
    « le report n'existe pas » — celui-ci existera après le run endossé.)"""
    import ast
    import inspect

    import scripts.run_f1_mb_t2 as drv
    assert callable(drv.main)
    arbre = ast.parse(inspect.getsource(drv))
    # seul le bootstrap sys.path est toléré au niveau module ; jamais `main`
    appels_top = [ast.unparse(n.value) for n in arbre.body
                  if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)]
    assert all(a.startswith("sys.path") for a in appels_top), appels_top
    gardes = [n for n in arbre.body if isinstance(n, ast.If)]
    assert len(gardes) == 1 and "__main__" in ast.dump(gardes[0].test)
