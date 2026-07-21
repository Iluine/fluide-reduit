"""Tests F1 — M-b tranche-2 : la lecture à trois branches (§A31).

Le bras témoin rend possible une décision que la production seule ne
permettait pas : distinguer si une traversée relève de (a) f32 — que le
remède gravé (Option B) guérit — ou de (b)+(c) EPS/structure — qu'il ne
guérit pas. Un test couvre les trois branches."""
from src.f1_gpu.mort_b_lecture import (
    LABEL_ANOMALIE,
    LABEL_OPTION_A_TIENT,
    LABEL_OPTION_B_PAS_REMEDE,
    LABEL_OPTION_B_REMEDE,
    PIN_SEVERE,
    branche_seed,
    lecture_mort_b,
    portees_de_la_mesure,
)

BAS = 0.05      # sous 0,0733 : ne traverse pas
HAUT = 0.09     # au-dessus : traverse


# ----- les trois branches, par-seed -----

def test_pin_severe_grave():
    assert PIN_SEVERE == 0.0733


def test_branche_aucun_ne_traverse_option_a_tient():
    b = branche_seed(production_dchi_max=BAS, temoin_dchi_max=BAS)
    assert b["production_traverse"] is False
    assert b["label"] == LABEL_OPTION_A_TIENT
    assert b["option_a_morte"] is False


def test_branche_les_deux_traversent_option_b_remede():
    """production ET témoin traversent ⇒ cause (a), Option B EST le remède."""
    b = branche_seed(production_dchi_max=HAUT, temoin_dchi_max=HAUT)
    assert b["production_traverse"] is True and b["temoin_traverse"] is True
    assert b["label"] == LABEL_OPTION_B_REMEDE
    assert b["option_a_morte"] is True
    assert b["option_b_est_le_remede"] is True


def test_branche_production_traverse_temoin_passe_pas_le_remede():
    """production traverse, témoin PASSE ⇒ cause (b)+(c), Option B N'EST PAS
    le remède — le cœur de §A31."""
    b = branche_seed(production_dchi_max=HAUT, temoin_dchi_max=BAS)
    assert b["production_traverse"] is True and b["temoin_traverse"] is False
    assert b["label"] == LABEL_OPTION_B_PAS_REMEDE
    assert b["option_a_morte"] is True
    assert b["option_b_est_le_remede"] is False


def test_branche_anomalie_production_passe_temoin_traverse():
    """Cas logiquement anormal (le témoin a MOINS de sources) : remonté, pas
    masqué."""
    b = branche_seed(production_dchi_max=BAS, temoin_dchi_max=HAUT)
    assert b["label"] == LABEL_ANOMALIE
    assert b["option_a_morte"] is False


def test_frontiere_pin_severe_stricte():
    """> 0,0733 traverse ; = 0,0733 ne traverse pas (borne exclue)."""
    assert branche_seed(PIN_SEVERE, BAS)["production_traverse"] is False
    assert branche_seed(PIN_SEVERE + 1e-9, BAS)["production_traverse"] is True


# ----- agrégation par-seed -----

def test_agrege_option_a_tient_si_aucun_seed_ne_traverse():
    lec = lecture_mort_b({s: {"production": BAS, "temoin": BAS}
                          for s in (101, 102, 103)})
    assert lec["option_a_morte"] is False
    assert lec["verdict"] == LABEL_OPTION_A_TIENT
    assert lec["seeds_traversants"] == []


def test_agrege_mort_b_par_seed_un_seul_suffit():
    """MORT-b PAR-SEED : un seul seed traversant tue Option A. Ici ce seed
    est cause (a) (témoin traverse aussi) ⇒ Option B est le remède."""
    lec = lecture_mort_b({
        101: {"production": BAS, "temoin": BAS},
        102: {"production": HAUT, "temoin": HAUT},      # tue A, cause (a)
        103: {"production": BAS, "temoin": BAS}})
    assert lec["option_a_morte"] is True
    assert lec["seeds_traversants"] == [102]
    assert lec["verdict"] == LABEL_OPTION_B_REMEDE
    assert lec["option_b_est_le_remede"] is True


def test_agrege_un_traversant_bc_rend_le_remede_inapplicable():
    """Un seul seed traversant par (b)+(c) rend le remède gravé inapplicable,
    même si un autre seed est cause (a)."""
    lec = lecture_mort_b({
        101: {"production": HAUT, "temoin": HAUT},      # cause (a)
        102: {"production": HAUT, "temoin": BAS},       # cause (b)+(c)
        103: {"production": BAS, "temoin": BAS}})
    assert lec["option_a_morte"] is True
    assert set(lec["seeds_traversants"]) == {101, 102}
    assert lec["verdict"] == LABEL_OPTION_B_PAS_REMEDE
    assert lec["option_b_est_le_remede"] is False
    assert "remède ailleurs" in lec["note_remede"]


def test_agrege_remonte_les_anomalies():
    lec = lecture_mort_b({
        101: {"production": BAS, "temoin": HAUT},       # anomalie
        102: {"production": BAS, "temoin": BAS}})
    assert lec["anomalies"] == [101]
    assert lec["option_a_morte"] is False


# ----- les deux portées, DANS la lecture -----

def test_portees_portees_dans_la_lecture():
    p = portees_de_la_mesure()
    assert "NE LES VALIDE PAS à l'échelle V4" in p["l1_l3_non_validees_a_v4"]
    assert "~24 cellules" in p["colonnes_entrantes_dominantes"]
    assert "un TIERS" in p["colonnes_entrantes_dominantes"]
    assert "colonnes ENTRANTES" in p["colonnes_entrantes_dominantes"]
