"""Tests F1 — M-b tranche-1 / C5 : driver T1 (§A29-C1).

Ce que ces tests protègent (l'ASSEMBLAGE ; le RUN est le POINT D'ARRÊT) :
  1. la bande T1 RE-GELÉE : Exner amorti par la cadence dérivée (÷2), haute
     ~18,3 ; composée AVANT le run, plus jamais recomposée (§A25) ;
  2. le DIAGNOSTIC PRINCIPAL EN TÊTE : F_fidèle/F_proxy (amendement A), la
     bande au second rang ;
  3. le verdict T1 sur la MÉDIANE de la frame complète, seuil 16,7 ;
  4. la réserve de HALO reportée à part, seuil glissant 15,7 (§A29-C1) ;
  5. le RÉGIME dû avant lecture — fail-loud sans PASS (B9)."""
import hashlib

import pytest

from scripts.run_f1_mb_t1 import (
    BANDE_T1_MS,
    CADENCE_EXNER,
    F_PROXY_MS,
    LABEL_MORT,
    LABEL_SANS_MORT,
    LABEL_SOUS_RESERVE,
    POSTE_EXNER_AMORTI,
    POSTE_F_FIDELE,
    RESERVE_HALO_MS,
    SEUIL_MORT_MS,
    SEUIL_RESERVE_MS,
    lecture_t1,
    verdict_t1,
)

REGIME_OK = {"pass": True, "n_frames": 330, "marge_cfl_min": 5.0,
             "marge_cfl_max": 5.5, "fraction_wet_min": 0.4}


def _lecture(f_fidele_med: float, frame_med: float, p99: float = 16.0,
             regime=None):
    return lecture_t1(
        {"mediane_ms": f_fidele_med, "p99_ms": p99},
        {"mediane_ms": frame_med, "p99_ms": p99},
        REGIME_OK if regime is None else regime)


# ----- 1. bande re-gelée -----

def test_bande_re_gelee_exner_amorti_par_la_cadence():
    """Exner amorti par la cadence DÉRIVÉE (÷2, §A29-C1) : le poste passe de
    [1,5;3,0] à [0,75;1,5], et la bande haute vers ~18,3."""
    assert POSTE_EXNER_AMORTI == (0.75, 1.5)          # [1,5;3,0]/2
    assert CADENCE_EXNER == 2
    assert POSTE_F_FIDELE == (10.8, 15.2)
    bas, haut = BANDE_T1_MS
    assert bas == pytest.approx(12.76, abs=0.01)
    assert haut == pytest.approx(18.27, abs=0.01)     # ~18,3


# ----- 2. F_fidèle/F_proxy EN TÊTE -----

def test_representativite_en_tete():
    """Le diagnostic principal (amendement A) est le RAPPORT, EN TÊTE ; la
    bande vient au second rang."""
    lec = _lecture(13.2, 14.4)
    cles = list(lec.keys())
    assert cles[0] == "representativite"              # EN TÊTE
    assert cles.index("representativite") < cles.index("bande_modele")
    rep = lec["representativite"]
    assert rep["f_proxy_ms"] == F_PROXY_MS == 12.655
    assert rep["rapport_f_fidele_sur_proxy"] == pytest.approx(13.2 / 12.655)
    assert "COÛT" in rep["note"]


def test_le_rapport_porte_sa_portee_ne_valide_pas_e4a():
    """§A29 : ≈1 dit qu'ils COÛTENT pareil, pas qu'ils FONT pareil. La
    portée voyage AVEC le rapport (proxy 8 champs fond plat ; fidèle 3 +
    b-aware) — un rapport proche de 1 NE VALIDE PAS E4a."""
    portee = _lecture(12.7, 14.4)["representativite"]["portee"]
    assert "COÛTENT pareil, PAS qu'ils FONT pareil" in portee
    assert "NE VALIDE PAS E4a" in portee
    assert "b-aware" in portee


# ----- 3. verdict T1 : trois valeurs pré-écrites DANS le champ -----

def test_verdict_trois_valeurs_dans_le_champ():
    """§A29 : le label vit DANS le verdict, pas à côté. Les trois branches
    gravées, couvertes : > 16,7 MORT ; [15,7;16,7] SOUS RÉSERVE ; < 15,7
    SANS MORT."""
    assert verdict_t1(16.8) == LABEL_MORT
    assert verdict_t1(16.71) == LABEL_MORT
    assert verdict_t1(16.7) == LABEL_SOUS_RESERVE          # borne incluse
    assert verdict_t1(16.2) == LABEL_SOUS_RESERVE
    assert verdict_t1(15.7) == LABEL_SOUS_RESERVE          # borne incluse
    assert verdict_t1(15.69) == LABEL_SANS_MORT
    assert verdict_t1(14.0) == LABEL_SANS_MORT


def test_t1_mediane_complete_et_verdict_porte_le_label():
    """médiane complète = mesurée (F+Exner) + remontée 1,28 + transferts 0,11,
    et le VERDICT (label) est DANS le champ t1, sur cette médiane."""
    lec = _lecture(13.0, 15.5)          # 15,5 + 1,39 = 16,89 > 16,7
    assert lec["t1"]["mediane_frame_complete_ms"] == pytest.approx(16.89)
    assert lec["t1"]["verdict"] == LABEL_MORT
    assert lec["t1"]["mort"] is True
    assert lec["t1"]["seuil_mort_ms"] == SEUIL_MORT_MS
    assert lec["t1"]["seuil_reserve_ms"] == SEUIL_RESERVE_MS
    assert set(lec["t1"]["branches_preecrites"]) == {
        LABEL_MORT, LABEL_SOUS_RESERVE, LABEL_SANS_MORT}
    lec_res = _lecture(13.2, 14.4)      # 14,4 + 1,39 = 15,79 ∈ [15,7;16,7]
    assert lec_res["t1"]["verdict"] == LABEL_SOUS_RESERVE
    assert lec_res["t1"]["mort"] is False
    lec_ok = _lecture(11.0, 13.0)       # 13 + 1,39 = 14,39 < 15,7
    assert lec_ok["t1"]["verdict"] == LABEL_SANS_MORT


# ----- 4. réserve de halo -----

def test_reserve_halo_reportee_a_part_seuil_glissant():
    """§A29-C1 : le bord réfléchissant sous-compte le halo-parent ; la
    réserve [0,4;1,0] fait glisser le seuil à 15,7."""
    assert RESERVE_HALO_MS == (0.4, 1.0)
    assert SEUIL_RESERVE_MS == pytest.approx(15.7)
    # médiane 15,79 (< 16,7 : vit) mais + réserve haute 1,0 -> 16,79 > 16,7
    lec = _lecture(13.2, 14.4)
    rh = lec["reserve_halo"]
    assert lec["t1"]["mort"] is False
    assert rh["mediane_plus_reserve_haute_ms"] == pytest.approx(16.79)
    assert rh["mort_sous_reserve"] is True            # tuée par la réserve
    # médiane basse : réserve ne suffit pas
    lec2 = _lecture(11.0, 13.0)         # 13+1,39=14,39 ; +1,0=15,39 < 16,7
    assert lec2["reserve_halo"]["mort_sous_reserve"] is False


# ----- 5. régime dû avant lecture -----

def test_lecture_refuse_sans_regime_pass():
    """B9 : aucune lecture T1 sans le PASS du régime."""
    with pytest.raises(RuntimeError, match="RÉGIME NON EXERCÉ"):
        _lecture(13.0, 14.0, regime={"pass": False, "motif": "marge CFL "
                                     "hors [4,12]"})
    with pytest.raises(RuntimeError, match="RÉGIME NON EXERCÉ"):
        _lecture(13.0, 14.0, regime=None if False else {"pass": False})


def test_p99_en_evidence_et_portee():
    lec = _lecture(13.2, 14.4, p99=17.2)
    assert lec["p99_en_evidence"]["valeur_ms"] == 17.2
    assert "stutter" in lec["p99_en_evidence"]["note"]
    assert "rederive n'est pas appelé" in lec["rappel_portee"]
    assert "MORT-b) est tranche-2" in lec["rappel_portee"]


def test_bande_au_second_rang_et_cadence_reportee():
    lec = _lecture(13.2, 14.4)
    bande = lec["bande_modele"]
    assert bande["bande_ms"] == [BANDE_T1_MS[0], BANDE_T1_MS[1]]
    assert "LARGE et peu falsifiable" in bande["note"]
    assert bande["cadence_exner"]["cadence_exner_sur_wetdry"] == 0.5


# ----- 6. C1/C2/C4/figés intouchés -----

def test_kernels_intouches():
    from src.f1_gpu.exner_gpu import _SOURCE_EXNER
    from src.f1_gpu.substrat_fidele import _SOURCE_FIDELE
    from src.f1_gpu.substrat_fusionne import _SOURCE
    from src.f1_gpu.substrat_l3 import _SOURCE_L3
    attendus = {
        _SOURCE_FIDELE: "6dd207cae0a9c23a6a042db825a3259051e8491c1f118d614"
                        "e52a4b4ebc6265d",
        _SOURCE_EXNER: "3533fd0bde3927856fa97b012a78fd04cfc6ab8c9ccf5fe17b1"
                       "3b95c7cd63aa6",
        _SOURCE: "e18015f57e14263414239b18c51d25cd335250f58dde2ccb892a6e2c1"
                 "d6f30b4",
        _SOURCE_L3: "9533a130b6624ce1b5b76d7d8e206aaae2e55fcc34e2a8d4eafa315"
                    "e8014781a",
    }
    for source, sha in attendus.items():
        assert hashlib.sha256(source.encode()).hexdigest() == sha
