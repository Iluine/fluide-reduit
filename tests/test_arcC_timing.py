"""Arc C / Task 3 — CORRECTIF avant session : tests de l'accumulateur PUR des
durées d'exposition RÉALISÉES (`src/arcC_timing.py`), §C9 pièce 1 de
PREREGISTRATION.md (pocCascade2phys, `7110e31`), reproduit verbatim dans
`.superpowers/sdd/arcC-task3fix-timing-conditions-brief.md`. **AUCUN sujet
humain** : durées CONSTRUITES à la main (aucune dépendance matplotlib, cf.
docstring module `src/arcC_timing.py`).

Familles de tests (brief, §Tests 1/2) :
  1. Accumulateur timing : `enregistre_phase` accumule les bonnes durées par
     (essai, phase) ; `resume_timing` calcule réalisé-vs-nominal ; seuil
     d'avertissement à 25% testé PILE en dessous (24%) et au-dessus (26%).
  2. Sidecar timing : round-trip JSONL, une ligne par (essai, phase), champs
     attendus."""
from __future__ import annotations

import pytest

from src.arcC_abx import REGIME_LAXISTE, REGIME_SEVERE
from src.arcC_timing import (SEUIL_AVERTISSEMENT_ECART, EnregistrementPhase, JournalTiming,
                             ecrit_timing_jsonl, enregistre_phase, formate_resume_timing,
                             formate_timing_essai, lit_timing_jsonl, resume_timing)

# =============================================================================
# Famille 1 : accumulateur timing (enregistre_phase, resume_timing)
# =============================================================================


def test_enregistre_phase_calcule_la_duree_et_accumule_en_place():
    journal = JournalTiming()
    enr = enregistre_phase(journal, indice_essai=0, regime_nom="laxiste",
                           nom_phase="exposition_X", t_debut=10.0, t_fin=12.0, nominal=2.0)
    assert enr.duree == pytest.approx(2.0)
    assert len(journal.enregistrements) == 1
    assert journal.enregistrements[0] is enr


def test_enregistre_phase_accumule_plusieurs_essais_et_phases_distincts():
    journal = JournalTiming()
    enregistre_phase(journal, indice_essai=0, regime_nom="laxiste", nom_phase="exposition_X",
                     t_debut=0.0, t_fin=2.1, nominal=2.0)
    enregistre_phase(journal, indice_essai=0, regime_nom="laxiste", nom_phase="retention_X",
                     t_debut=2.1, t_fin=7.0, nominal=5.0)
    enregistre_phase(journal, indice_essai=1, regime_nom="laxiste", nom_phase="exposition_X",
                     t_debut=100.0, t_fin=102.3, nominal=2.0)
    assert len(journal.enregistrements) == 3
    cles = {(e.indice_essai, e.phase) for e in journal.enregistrements}
    assert cles == {(0, "exposition_X"), (0, "retention_X"), (1, "exposition_X")}


def test_enregistre_phase_leve_erreur_si_horloge_incoherente():
    journal = JournalTiming()
    with pytest.raises(ValueError):
        enregistre_phase(journal, indice_essai=0, regime_nom="laxiste", nom_phase="exposition_X",
                         t_debut=5.0, t_fin=4.0, nominal=2.0)


def _journal_exposition_constante(realise_s: float, n: int = 6, nominal: float = 2.0
                                  ) -> JournalTiming:
    """Construit un journal avec `n` phases `exposition_*` toutes réalisées à
    `realise_s` (durée constante -- moyenne = `realise_s` par construction)."""
    journal = JournalTiming()
    for i in range(n):
        enregistre_phase(journal, indice_essai=i, regime_nom="laxiste",
                         nom_phase=f"exposition_{i}", t_debut=0.0, t_fin=realise_s,
                         nominal=nominal)
    return journal


def test_resume_timing_moyenne_realisee_vs_nominale_correcte():
    journal = JournalTiming()
    # Trois expositions réalisées à 2.0/2.2/1.8 -- moyenne exacte = 2.0 = nominal.
    for i, realise in enumerate((2.0, 2.2, 1.8)):
        enregistre_phase(journal, indice_essai=i, regime_nom="laxiste",
                         nom_phase=f"exposition_{i}", t_debut=0.0, t_fin=realise, nominal=2.0)
    resume = resume_timing(journal, REGIME_LAXISTE)
    cat = resume["categories"]["exposition"]
    assert cat["n"] == 3
    assert cat["moyenne_realisee_s"] == pytest.approx(2.0)
    assert cat["nominal_s"] == pytest.approx(2.0)
    assert cat["ecart_relatif"] == pytest.approx(0.0)
    assert cat["avertissement"] is False
    assert resume["avertissement_global"] is False


def test_resume_timing_categories_exposition_et_retention_separees():
    journal = JournalTiming()
    enregistre_phase(journal, indice_essai=0, regime_nom="laxiste", nom_phase="exposition_X",
                     t_debut=0.0, t_fin=2.0, nominal=2.0)
    enregistre_phase(journal, indice_essai=0, regime_nom="laxiste", nom_phase="retention_X",
                     t_debut=0.0, t_fin=6.0, nominal=5.0)
    resume = resume_timing(journal, REGIME_LAXISTE)
    assert resume["categories"]["exposition"]["nominal_s"] == pytest.approx(2.0)
    assert resume["categories"]["retention"]["nominal_s"] == pytest.approx(5.0)
    assert resume["categories"]["exposition"]["n"] == 1
    assert resume["categories"]["retention"]["n"] == 1


def test_resume_timing_phase_sans_cible_temporelle_severe_pas_d_avertissement():
    """Régime sévère : phase `"simultane"`, AUCUNE cible temporelle
    (`regime.exposition_s`/`retention_s` valent `None`) -- catégorisée
    `"autre"`, jamais comparée, jamais un avertissement fabriqué sur du
    vide."""
    journal = JournalTiming()
    enregistre_phase(journal, indice_essai=0, regime_nom="severe", nom_phase="simultane",
                     t_debut=0.0, t_fin=12.7, nominal=None)
    resume = resume_timing(journal, REGIME_SEVERE)
    assert resume["categories"]["autre"]["n"] == 1
    assert resume["categories"]["autre"]["nominal_s"] is None
    assert resume["categories"]["autre"]["avertissement"] is False
    assert resume["categories"]["exposition"]["n"] == 0
    assert resume["categories"]["retention"]["n"] == 0
    assert resume["avertissement_global"] is False


def test_resume_timing_categorie_vide_nominal_none_pas_d_avertissement():
    journal = JournalTiming()  # aucun enregistrement du tout
    resume = resume_timing(journal, REGIME_LAXISTE)
    for cat in resume["categories"].values():
        assert cat["n"] == 0
        assert cat["moyenne_realisee_s"] is None
        assert cat["avertissement"] is False
    assert resume["avertissement_global"] is False


# --- Seuil d'avertissement 25% (brief : 24% -> pas d'alerte, 26% -> alerte) --


def test_resume_timing_ecart_24_pourcent_ne_declenche_pas_avertissement():
    # nominal=2.0, réalisé=2.48 -> écart = +24% exactement.
    journal = _journal_exposition_constante(2.0 * 1.24, n=1)
    resume = resume_timing(journal, REGIME_LAXISTE)
    cat = resume["categories"]["exposition"]
    assert cat["ecart_relatif"] == pytest.approx(0.24, abs=1e-9)
    assert cat["avertissement"] is False
    assert resume["avertissement_global"] is False


def test_resume_timing_ecart_26_pourcent_declenche_avertissement():
    # nominal=2.0, réalisé=2.52 -> écart = +26% exactement.
    journal = _journal_exposition_constante(2.0 * 1.26, n=1)
    resume = resume_timing(journal, REGIME_LAXISTE)
    cat = resume["categories"]["exposition"]
    assert cat["ecart_relatif"] == pytest.approx(0.26, abs=1e-9)
    assert cat["avertissement"] is True
    assert resume["avertissement_global"] is True
    assert "exposition" in resume["categories_en_alerte"]


def test_resume_timing_ecart_negatif_aussi_declenche_avertissement():
    """L'écart est en VALEUR ABSOLUE -- une exposition trop COURTE (-30%)
    doit aussi CRIER, pas seulement une exposition trop longue."""
    journal = _journal_exposition_constante(2.0 * 0.70, n=1)  # -30%
    resume = resume_timing(journal, REGIME_LAXISTE)
    assert resume["categories"]["exposition"]["avertissement"] is True


def test_seuil_avertissement_expose_est_25_pourcent():
    assert SEUIL_AVERTISSEMENT_ECART == pytest.approx(0.25)


def test_formate_resume_timing_contient_avertissement_si_declenche():
    journal = _journal_exposition_constante(2.0 * 1.40, n=1)
    resume = resume_timing(journal, REGIME_LAXISTE)
    texte = formate_resume_timing(resume)
    assert "AVERTISSEMENT" in texte


def test_formate_resume_timing_pas_d_avertissement_si_dans_la_marge():
    journal = _journal_exposition_constante(2.0 * 1.05, n=1)
    resume = resume_timing(journal, REGIME_LAXISTE)
    texte = formate_resume_timing(resume)
    assert "AVERTISSEMENT" not in texte


# =============================================================================
# Famille 2 : sidecar timing (round-trip JSONL)
# =============================================================================


def test_ecrit_puis_lit_timing_jsonl_round_trip(tmp_path):
    journal = JournalTiming()
    enregistre_phase(journal, indice_essai=0, regime_nom="laxiste", nom_phase="exposition_X",
                     t_debut=1.0, t_fin=3.1, nominal=2.0)
    enregistre_phase(journal, indice_essai=0, regime_nom="laxiste", nom_phase="retention_X",
                     t_debut=3.1, t_fin=8.2, nominal=5.0)
    enregistre_phase(journal, indice_essai=1, regime_nom="laxiste", nom_phase="exposition_A",
                     t_debut=20.0, t_fin=22.0, nominal=2.0)
    path = tmp_path / "session_0_laxiste.timing.jsonl"
    ecrit_timing_jsonl(path, journal)

    relu = lit_timing_jsonl(path)
    assert len(relu.enregistrements) == 3
    for original, relu_enr in zip(journal.enregistrements, relu.enregistrements):
        assert relu_enr.to_dict() == original.to_dict()


def test_ecrit_timing_jsonl_une_ligne_par_essai_phase_avec_champs_attendus(tmp_path):
    journal = JournalTiming()
    enregistre_phase(journal, indice_essai=2, regime_nom="laxiste", nom_phase="exposition_B",
                     t_debut=0.0, t_fin=2.5, nominal=2.0)
    path = tmp_path / "timing.jsonl"
    ecrit_timing_jsonl(path, journal)

    import json
    lignes = path.read_text(encoding="utf-8").splitlines()
    assert len(lignes) == 1
    d = json.loads(lignes[0])
    for champ in ("indice_essai", "regime", "phase", "t_debut", "t_fin", "duree", "nominal"):
        assert champ in d
    assert d["indice_essai"] == 2
    assert d["regime"] == "laxiste"
    assert d["phase"] == "exposition_B"
    assert d["duree"] == pytest.approx(2.5)
    assert d["nominal"] == pytest.approx(2.0)


def test_ecrit_timing_jsonl_journal_vide_ecrit_fichier_vide(tmp_path):
    path = tmp_path / "vide.timing.jsonl"
    ecrit_timing_jsonl(path, JournalTiming())
    assert path.exists()
    relu = lit_timing_jsonl(path)
    assert relu.enregistrements == []


def test_ecrit_timing_jsonl_cree_les_dossiers_parents(tmp_path):
    path = tmp_path / "sous" / "dossier" / "timing.jsonl"
    journal = JournalTiming()
    enregistre_phase(journal, indice_essai=0, regime_nom="severe", nom_phase="simultane",
                     t_debut=0.0, t_fin=1.0, nominal=None)
    ecrit_timing_jsonl(path, journal)
    assert path.exists()


# =============================================================================
# Famille 4 : affichage LIVE par essai (formate_timing_essai, pré-vol B-bis)
# =============================================================================


def test_formate_timing_essai_moyenne_par_categorie_avec_nominal_et_ecart():
    """Un essai laxiste (X seul, pour rester lisible) : la ligne live moyenne
    exposition et rétention réalisées par catégorie, avec nominal et écart
    relatif SIGNÉ -- réalisé 2.4s vs nominal 2.0s => +20%, sous le seuil 25%
    (pas de marque)."""
    enrs = [
        EnregistrementPhase(0, "laxiste", "exposition_X", 0.0, 2.4, 2.4, 2.0),
        EnregistrementPhase(0, "laxiste", "retention_X", 2.4, 7.4, 5.0, 5.0),
    ]
    ligne = formate_timing_essai(0, enrs)
    assert "essai 0" in ligne
    assert "exposition≈2.400s" in ligne
    assert "+20%" in ligne
    assert "retention≈5.000s" in ligne
    assert ">25%" not in ligne  # +20% et +0% sont sous le seuil


def test_formate_timing_essai_marque_l_ecart_au_dela_du_seuil():
    """Réalisé 2.6s vs nominal 2.0s => +30% > 25% => la ligne CRIE (marque
    `>25%`), cohérent avec l'avertissement de `resume_timing` (§C9)."""
    enrs = [EnregistrementPhase(3, "laxiste", "exposition_X", 0.0, 2.6, 2.6, 2.0)]
    ligne = formate_timing_essai(3, enrs)
    assert "essai 3" in ligne
    assert ">25%" in ligne


def test_formate_timing_essai_phase_sans_cible_severe_pas_de_comparaison():
    """Essai sévère (phase `simultane`, `nominal=None`) : la ligne montre la
    durée d'inspection SANS écart ni marque (aucune cible temporelle)."""
    enrs = [EnregistrementPhase(0, "severe", "simultane", 0.0, 3.7, 3.7, None)]
    ligne = formate_timing_essai(0, enrs)
    assert "autre≈3.700s" in ligne
    assert "%" not in ligne
    assert ">25%" not in ligne
