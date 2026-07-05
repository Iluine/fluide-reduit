"""Arc A / extension 16L₀ — tests du Task 3 (VERDICT §A12), cf. brief
`.superpowers/sdd/16L0-task3-brief.md`. Les tests sur DONNÉES RÉELLES qui
dépendent de `measures_160.npz` (Task 2, mesure réelle L=160)
(`test_agrafage_L_axis`, `test_determinisme`) sont marqués
`@pytest.mark.skipif` TANT QUE ce fichier n'est pas produit par le
contrôleur en arrière-plan -- écrits ainsi car `measures_160.npz` n'existait
PAS ENCORE au début de ce Task 3 ; il est apparu (commit `8a2b38b`) PENDANT
cette intervention, et ces deux tests s'exécutent donc désormais RÉELLEMENT
(passants, cf. rapport) au lieu d'être sautés -- la garde `skipif` reste en
place pour tout futur checkout où le fichier serait absent (ex. CI propre).
Le CŒUR ANTI-FABRICATION -- gate sur sous_jnd synthétique, pente OLS
3-points, combinateur §A12 sur ses 3 branches, et la non-régression de la
surface C-1a à L≤80 (qui n'a besoin QUE de `measures_qt.npz`, déjà présent
dès le départ) -- est prouvé SANS skip, sur synthétique ou sur les données
gelées existantes, indépendamment de la disponibilité de `measures_160.npz`."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

import scripts.run_arcA_verdict_16L0 as v16
from scripts.run_arcA_measure import JND_LIST, L_LIST, SEEDS
from scripts.run_arcA_measure_16L0 import MEASURES_160_PATH
from scripts.run_arcA_measure_qt import BUDGETS
from scripts.run_arcA_measure_qt import MEASURES_PATH as MEASURES_QT_PATH
from scripts.run_arcA_verdict_qt import ATTENDU_FERM_QT, ATTENDU_SHUF_QT
from scripts.run_arcC_surface_kstar_pins import PINS_PATH, SURFACE_PATH, charge_pins

# =============================================================================
# Données archivées, chargées une seule fois (réutilisées par plusieurs tests)
# =============================================================================

with np.load(MEASURES_QT_PATH, allow_pickle=False) as _d:
    MA1_QT_V2 = np.asarray(_d["ma1_v2"])
    MA2_QT_V2 = np.asarray(_d["ma2_v2"])

PINS = charge_pins(PINS_PATH)

# Artefact C-1a INDÉPENDANT (`run_arcC_surface_kstar_pins.py`, Task 2) --
# ancre de vérité de la garde d'agrafage (`test_non_regression_surface_C1a`) :
# ce JSON n'est jamais recalculé ici, seulement chargé et comparé.
with open(SURFACE_PATH, encoding="utf-8") as _f:
    SURFACE_C1A = json.load(_f)["surface_kstar_aux_pins"]


def _valeur_depuis_jsonable(v: float | str) -> float:
    """Inverse de la convention de `_to_jsonable` (`scripts.run_arcA_verdict`) :
    un float sérialisé dans l'artefact JSON redevient un float Python, y
    compris les cas non-finis (`"inf"`/`"-inf"`/`"nan"` -- k* peut être
    infini si aucun budget ne ferme). Lecture pure, aucune tolérance."""
    if isinstance(v, str):
        return {"inf": math.inf, "-inf": -math.inf, "nan": math.nan}[v]
    return float(v)

_MEASURES_160_PRETES = MEASURES_160_PATH.exists()
_SKIP_REASON_160 = (
    "measures_160.npz (Task 2, mesure réelle L=160) pas encore produit par le "
    "contrôleur en arrière-plan -- ce test sera levé et confirmé une fois le "
    "fichier présent.")


# =============================================================================
# 1. L_LIST_FULL : construction, L_LIST importé INTACT (jamais muté)
# =============================================================================


def test_L_LIST_FULL_construit_sur_L_LIST_intact():
    assert v16.L_LIST_FULL == (10, 20, 40, 80, 160)
    assert L_LIST == (10, 20, 40, 80)  # inchangé par ce module
    assert v16.L_LIST_FULL[:4] == L_LIST


# =============================================================================
# 2. Agrafage de l'axe L -- DONNÉES RÉELLES, skipif measures_160.npz absent
# =============================================================================


@pytest.mark.skipif(not _MEASURES_160_PRETES, reason=_SKIP_REASON_160)
def test_agrafage_L_axis():
    with np.load(MEASURES_160_PATH, allow_pickle=False) as d:
        ma1_160 = np.asarray(d["ma1_v2"])
        ma2_160 = np.asarray(d["ma2_v2"])

    ma1_full = v16.agrafe_L_axis(MA1_QT_V2, ma1_160)
    ma2_full = v16.agrafe_L_axis(MA2_QT_V2, ma2_160)

    assert ma1_full.shape == (5, 5, 7)
    assert ma2_full.shape == (5, 5, 7)
    assert np.array_equal(ma1_full[:4], MA1_QT_V2)
    assert np.array_equal(ma2_full[:4], MA2_QT_V2)
    assert np.array_equal(ma1_full[4], ma1_160[0])
    assert np.array_equal(ma2_full[4], ma2_160[0])


def test_agrafe_L_axis_leve_erreur_si_forme_incorrecte():
    ma_qt_ok = np.zeros((4, 5, 7))
    ma_160_mauvaise_forme = np.zeros((2, 5, 7))  # devrait être (1,5,7)
    with pytest.raises(ValueError, match="L_V2_160"):
        v16.agrafe_L_axis(ma_qt_ok, ma_160_mauvaise_forme)

    ma_qt_mauvaise_forme = np.zeros((3, 5, 7))  # devrait être (4,5,7)
    ma_160_ok = np.zeros((1, 5, 7))
    with pytest.raises(ValueError, match="L_LIST"):
        v16.agrafe_L_axis(ma_qt_mauvaise_forme, ma_160_ok)


# =============================================================================
# 3. Non-régression LOAD-BEARING : surface à L∈{10,20,40,80} == C-1a
#    (surface_kstar_aux_pins.json), bit-à-bit -- n'a besoin QUE de
#    measures_qt.npz (déjà présent) : PAS de skip.
# =============================================================================


def test_non_regression_surface_C1a():
    """LA garde d'agrafage : prouve que l'agrafage measures_qt+measures_160
    n'a PAS corrompu la manche 1, en comparant la surface k*(L) recalculée
    par `v16.surface_par_L` à l'artefact INDÉPENDANT C-1a
    (`SURFACE_C1A`, chargé une seule fois depuis `surface_kstar_aux_pins.json`
    au niveau module -- JAMAIS un re-appel de `k_star_groupe_qt`, ce qui
    serait tautologique). Couvre les 6 pins (2 régimes x 3 rôles) x les 4 L
    de la manche 1 (`L_LIST` -- L=160 n'existe PAS dans cet artefact C-1a,
    jamais comparé ici) -- 24 cellules, `mediane` ET `par_seed`, égalité
    EXACTE. Si l'agrafage ou `surface_par_L` a corrompu une seule cellule,
    ce test ÉCHOUE."""
    for regime in ("severe", "laxiste"):
        for role in ("ic_bas", "pin", "ic_haut"):
            jnd = PINS[regime][role]
            surface = v16.surface_par_L(MA1_QT_V2, MA2_QT_V2, L_LIST, jnd)
            k_par_L_artefact = SURFACE_C1A[regime][role]["k_star_par_L"]
            for L in L_LIST:
                cellule_artefact = k_par_L_artefact[str(L)]

                mediane_attendue = _valeur_depuis_jsonable(cellule_artefact["mediane"])
                assert surface[L]["mediane"] == mediane_attendue, (
                    f"{regime}/{role} L={L} : mediane verdict={surface[L]['mediane']} "
                    f"!= artefact C-1a={mediane_attendue}")

                par_seed_attendu = {
                    int(seed_str): _valeur_depuis_jsonable(v)
                    for seed_str, v in cellule_artefact["par_seed"].items()
                }
                assert surface[L]["par_seed"] == par_seed_attendu, (
                    f"{regime}/{role} L={L} : par_seed verdict={surface[L]['par_seed']} "
                    f"!= artefact C-1a={par_seed_attendu}")

    # Valeurs figées C-1a (`surface_kstar_aux_pins.json`), citées verbatim au
    # brief -- sanity supplémentaire (déjà couvert par la boucle sur
    # `SURFACE_C1A` ci-dessus, conservé pour lisibilité directe des nombres).
    jnd_pin_severe = PINS["severe"]["pin"]
    surface_pin = v16.surface_par_L(MA1_QT_V2, MA2_QT_V2, L_LIST, jnd_pin_severe)
    assert [surface_pin[L]["mediane"] for L in L_LIST] == [256.0, 256.0, 256.0, 256.0]

    jnd_ic_bas_severe = PINS["severe"]["ic_bas"]
    surface_ic_bas = v16.surface_par_L(MA1_QT_V2, MA2_QT_V2, L_LIST, jnd_ic_bas_severe)
    assert [surface_ic_bas[L]["mediane"] for L in L_LIST] == [256.0, 256.0, 400.0, 400.0]


# =============================================================================
# 4. Gate §A11 reconduit à L∈{10,160} -- LOGIQUE PURE, synthétique, pas de skip
# =============================================================================


def _grille_gate_conforme() -> dict[str, np.ndarray]:
    """ferm ferme partout à 32 (CONFORME) ; shuf ne ferme jamais (k*=∞,
    CONFORME) -- baseline propre sur l'axe L_CONTROL_160=(10,160)."""
    nL, nSeed, nBudget, nJND = 2, len(SEEDS), len(BUDGETS), len(JND_LIST)
    i32 = BUDGETS.index(32)

    sous_jnd_ferm = np.zeros((nL, nSeed, nBudget, nJND), dtype=bool)
    ma1_ferm = np.zeros((nL, nSeed, nBudget))
    ma2_ferm = np.zeros((nL, nSeed, nBudget))
    sous_jnd_ferm[:, :, i32, :] = True
    ma1_ferm[:, :, i32] = 0.1 * min(JND_LIST)

    sous_jnd_shuf = np.zeros((nL, nSeed, nBudget, nJND), dtype=bool)  # jamais sous-JND
    ma1_shuf = np.zeros((nL, nSeed, nBudget))
    ma2_shuf = np.zeros((nL, nSeed, nBudget))

    return dict(sous_jnd_ferm=sous_jnd_ferm, ma1_ferm=ma1_ferm, ma2_ferm=ma2_ferm,
               sous_jnd_shuf=sous_jnd_shuf, ma1_shuf=ma1_shuf, ma2_shuf=ma2_shuf)


def test_gate_16L0_conforme_sur_grille_propre():
    g = _grille_gate_conforme()
    statut, violations = v16.gate_controles_16L0(
        g["sous_jnd_ferm"], g["ma1_ferm"], g["ma2_ferm"],
        g["sous_jnd_shuf"], g["ma1_shuf"], g["ma2_shuf"])
    assert statut == "CONFORME"
    assert violations == []


def test_gate_detecte_violation():
    """ferm cassé à L=160 (seed 0, jnd[0]) -- k* != 32 -> VIOLATION détectée
    EXACTEMENT sur cette cellule."""
    g = _grille_gate_conforme()
    g["sous_jnd_ferm"][1, 0, :, 0] = False  # L=160 (index 1), seed 0 : jamais sous-JND -> k*=inf

    statut, violations = v16.gate_controles_16L0(
        g["sous_jnd_ferm"], g["ma1_ferm"], g["ma2_ferm"],
        g["sous_jnd_shuf"], g["ma1_shuf"], g["ma2_shuf"])

    assert statut == "VIOLATION"
    assert len(violations) == 1
    v = violations[0]
    assert v["bras"] == "ferm"
    assert v["L"] == 160
    assert v["seed"] == SEEDS[0]
    assert v["jnd"] == JND_LIST[0]
    assert v["k_star_obtenu"] == float("inf")
    assert v["k_star_attendu"] == ATTENDU_FERM_QT == 32.0


def test_gate_detecte_violation_shuf_sous_le_cap():
    """shuf fermant à budget=400 (<= CAP_FLOATS) à L=10 -> VIOLATION (portée
    fa54d20 : conforme ssi k* > CAP_FLOATS)."""
    g = _grille_gate_conforme()
    i400 = BUDGETS.index(400)
    g["sous_jnd_shuf"][0, :, i400, :] = True  # L=10 (index 0), tous seeds/JND
    g["ma1_shuf"][0, :, i400] = 0.1 * min(JND_LIST)

    statut, violations = v16.gate_controles_16L0(
        g["sous_jnd_ferm"], g["ma1_ferm"], g["ma2_ferm"],
        g["sous_jnd_shuf"], g["ma1_shuf"], g["ma2_shuf"])

    assert statut == "VIOLATION"
    assert len(violations) == len(SEEDS) * len(JND_LIST)
    assert all(v["bras"] == "shuf" and v["L"] == 10 and v["k_star_obtenu"] == 400.0
              for v in violations)
    assert all(v["k_star_attendu"] == ATTENDU_SHUF_QT for v in violations)


# =============================================================================
# 5. Pente OLS 3-points -- LOGIQUE PURE, pas de skip (nom exact du brief)
# =============================================================================


def _ma_deterministe_par_L(jnd: float, budget_par_L: dict[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """`ma1_full`/`ma2_full` (5,5,7) synthétiques, IDENTIQUES sur les 5 seeds
    (tout ré-échantillonnage bootstrap donne donc le MÊME k*(L) -- isole la
    mécanique pente_ols/IC de la variance inter-seeds, déjà couverte sur
    données réelles ailleurs). Pour chaque L de `budget_par_L`, sous_jnd
    devient vraie à partir du budget indiqué (valeur jnd/10, sous-JND) et
    fausse avant (valeur jnd*3, au-dessus)."""
    nL, nSeed, nBudget = len(v16.L_LIST_FULL), len(SEEDS), len(BUDGETS)
    ma1 = np.empty((nL, nSeed, nBudget))
    ma2 = np.empty((nL, nSeed, nBudget))
    budget_defaut = next(iter(budget_par_L.values()))
    for iL, L in enumerate(v16.L_LIST_FULL):
        b_target = budget_par_L.get(L, budget_defaut)
        i_target = BUDGETS.index(b_target)
        row = np.where(np.arange(nBudget) >= i_target, jnd * 0.1, jnd * 3.0)
        ma1[iL, :, :] = row
        ma2[iL, :, :] = row
    return ma1, ma2


def test_pente_ols_3pts():
    """Brief (§Étape 3) : k*(L) synthétique PLAT -> pente 0, IC bootstrap ∋ 0 ;
    monotone croissant -> pente > 0 ; dégénérescence 2-points ==
    (k*(80)-k*(40))/40 (formule §A3 exacte)."""
    # --- dégénérescence 2-points (identité algébrique, indépendante des données) --
    for k40, k80 in ((256.0, 400.0), (32.0, 32.0), (128.0, 256.0)):
        attendu = (k80 - k40) / 40.0
        obtenu = v16.pente_ols((40.0, 80.0), (k40, k80))
        assert math.isclose(obtenu, attendu, rel_tol=1e-12, abs_tol=1e-12)

    # --- k*(L) synthétique PLAT sur {40,80,160} -> pente ponctuelle 0, IC ∋ 0 ---
    jnd = 0.03
    ma1_plat, ma2_plat = _ma_deterministe_par_L(jnd, {40: 256, 80: 256, 160: 256})
    surface_plat = v16.surface_par_L(ma1_plat, ma2_plat, v16.L_LIST_FULL, jnd)
    k40, k80, k160 = (surface_plat[L]["mediane"] for L in v16.L_SLOPE_16L0)
    assert (k40, k80, k160) == (256.0, 256.0, 256.0)
    res_plat = v16.pente_bootstrap_16L0(ma1_plat, ma2_plat, jnd, k40, k80, k160)
    assert res_plat["pente"] == 0.0
    assert res_plat["ic95"][0] <= 0.0 <= res_plat["ic95"][1]
    assert res_plat["n_bootstrap"] == v16.N_BOOTSTRAP

    # --- k*(L) synthétique MONOTONE croissant -> pente ponctuelle > 0 -----------
    ma1_mono, ma2_mono = _ma_deterministe_par_L(jnd, {40: 128, 80: 256, 160: 400})
    surface_mono = v16.surface_par_L(ma1_mono, ma2_mono, v16.L_LIST_FULL, jnd)
    k40m, k80m, k160m = (surface_mono[L]["mediane"] for L in v16.L_SLOPE_16L0)
    assert (k40m, k80m, k160m) == (128.0, 256.0, 400.0)
    res_mono = v16.pente_bootstrap_16L0(ma1_mono, ma2_mono, jnd, k40m, k80m, k160m)
    assert res_mono["pente"] > 0.0


def test_pente_ols_non_fini_donne_nan():
    """Bonus (au-delà du brief) : cas ∞ d'un côté -> pente NaN, jamais ±∞
    (convention §A12/Task3 explicitement gravée, cf. docstring module)."""
    assert math.isnan(v16.pente_ols((40.0, 80.0, 160.0), (256.0, 256.0, float("inf"))))
    assert math.isnan(v16.pente_ols((40.0, 80.0, 160.0),
                                    (float("inf"), float("inf"), float("inf"))))


# =============================================================================
# 6. Combinateur §A12 -- fonction PURE, 3 branches synthétiques (nom exact du brief)
# =============================================================================


def _colonne(*, k80: float, k160: float, ic95: list[float]) -> dict:
    return v16.construit_colonne(k80, k160, ic95)


def test_combinateur_verdict_3_branches():
    """Brief (§Étape 4) : le combinateur pur renvoie CELLULE_1_PRONONCEE /
    FAIL_MUR / INDETERMINE_JND sur trois jeux d'entrées synthétiques construits
    pour chaque branche -- c'est le cœur anti-fabrication."""
    # --- branche 1 : les 3 colonnes fermées (plat + IC∋0 + sous cap) -----------
    colonnes_cellule1 = {
        "ic_bas": _colonne(k80=256.0, k160=256.0, ic95=[-2.0, 2.0]),
        "pin": _colonne(k80=256.0, k160=256.0, ic95=[-1.5, 1.5]),
        "ic_haut": _colonne(k80=256.0, k160=256.0, ic95=[-1.0, 1.0]),
    }
    res1 = v16.verdict_A12_combinateur(colonnes_cellule1)
    assert res1["prononce"] == "CELLULE_1_PRONONCEE"
    assert res1["gardes"]["colonnes_fermees"] == {"ic_bas": True, "pin": True, "ic_haut": True}
    assert res1["gardes"]["diverge_entre_colonnes"] is False

    # --- branche 2 : saut + pente positive à la colonne centrale (pin) ---------
    colonnes_fail_mur = {
        "ic_bas": _colonne(k80=256.0, k160=400.0, ic95=[1.0, 5.0]),
        "pin": _colonne(k80=256.0, k160=400.0, ic95=[0.5, 4.0]),
        "ic_haut": _colonne(k80=256.0, k160=400.0, ic95=[0.2, 3.0]),
    }
    res2 = v16.verdict_A12_combinateur(colonnes_fail_mur)
    assert res2["prononce"] == "FAIL_MUR"
    assert res2["gardes"]["fail_mur_colonne_centrale"] is True

    # --- branche 3 : exemple canonique du brief -- plat/fermé à pin & ic_haut,
    # montant à ic_bas -> le verdict par colonne DIFFÈRE, ni fermeture nette ni
    # mur net.
    colonnes_indetermine = {
        "ic_bas": _colonne(k80=256.0, k160=400.0, ic95=[0.5, 4.0]),    # monte
        "pin": _colonne(k80=256.0, k160=256.0, ic95=[-1.0, 1.0]),      # plat, fermé
        "ic_haut": _colonne(k80=256.0, k160=256.0, ic95=[-0.5, 0.5]),  # plat, fermé
    }
    res3 = v16.verdict_A12_combinateur(colonnes_indetermine)
    assert res3["prononce"] == "INDETERMINE_JND"
    assert res3["gardes"]["diverge_entre_colonnes"] is True
    assert res3["gardes"]["ic_bas_arme"]["ic_bas_monte"] is True
    assert res3["gardes"]["ic_bas_arme"]["pin_reste_plat"] is True
    assert res3["gardes"]["ic_bas_arme"]["soupcon_confirme"] is True


def test_construit_colonne_k160_infini_sous_cap_faux_mais_plat_vrai():
    # inf == inf -> plat=True (aucun mouvement), mais inf n'est jamais sous le cap.
    col = v16.construit_colonne(k80=float("inf"), k160=float("inf"), ic95=[0.0, 0.0])
    assert col["plat"] is True
    assert col["k160_sous_cap"] is False
    assert col["saut"] is False


# =============================================================================
# 7. Gate VIOLATION -> verdict NON_LISIBLE, combinateur JAMAIS appelé
# =============================================================================


def test_construit_verdict_complet_non_lisible_ne_evalue_pas_combinateur():
    # colonnes_severe=None : si construit_verdict_complet essayait d'évaluer
    # le combinateur malgré la violation, ceci lèverait immédiatement (None
    # n'est pas indexable par rôle) -- la garde est donc PROUVÉE, pas juste
    # affirmée.
    resultat = v16.construit_verdict_complet("VIOLATION", None)
    assert resultat["prononce"] == "NON_LISIBLE"
    assert resultat["par_colonne"] is None
    assert resultat["gardes"] is None
    assert resultat["portee"] == v16.PORTEE_16L0


def test_construit_verdict_complet_conforme_appelle_le_combinateur():
    colonnes = {
        "ic_bas": _colonne(k80=256.0, k160=256.0, ic95=[-1.0, 1.0]),
        "pin": _colonne(k80=256.0, k160=256.0, ic95=[-1.0, 1.0]),
        "ic_haut": _colonne(k80=256.0, k160=256.0, ic95=[-1.0, 1.0]),
    }
    resultat = v16.construit_verdict_complet("CONFORME", colonnes)
    assert resultat["prononce"] == "CELLULE_1_PRONONCEE"
    assert resultat["par_colonne"] == colonnes


# =============================================================================
# 8. Déterminisme du run complet -- DONNÉES RÉELLES, skipif measures_160.npz absent
# =============================================================================


@pytest.mark.skipif(not _MEASURES_160_PRETES, reason=_SKIP_REASON_160)
def test_determinisme():
    r1 = v16._to_jsonable(v16.construit_resultat_complet())
    r2 = v16._to_jsonable(v16.construit_resultat_complet())
    assert r1 == r2


# =============================================================================
# 9. Garde anti-fabrication : aucun seuil (CAP_FLOATS, pins, JND) réécrit
#    en dur dans ce module -- tout vient des imports / de pins_spatial.json.
# =============================================================================


def test_pas_de_seuil_deplace():
    src = v16.__file__ and open(v16.__file__, encoding="utf-8").read()
    assert "409.6" not in src, "CAP_FLOATS doit venir de l'import, jamais recopié en dur"
    for jnd in JND_LIST:
        assert repr(jnd) not in src, f"JND de grille {jnd} recopié en dur"
    for regime in ("severe", "laxiste"):
        for role in ("ic_bas", "pin", "ic_haut"):
            valeur = PINS[regime][role]
            assert repr(valeur) not in src, f"pin {regime}/{role} recopié en dur"


# =============================================================================
# 10. Pire seed L=160 -- logique de sélection PURE, synthétique
# =============================================================================


def test_pire_seed_L160_argmax():
    ma2 = np.zeros((5, 5, 7))
    iL160 = v16.L_LIST_FULL.index(160)
    ma2[iL160, 3, BUDGETS.index(256)] = 0.09  # seed d'indice 3 la plus discriminante
    idx = v16._pire_seed_L160(ma2, 256.0)
    assert idx == 3


def test_pire_seed_L160_budget_infini_utilise_dernier_budget():
    ma2 = np.zeros((5, 5, 7))
    iL160 = v16.L_LIST_FULL.index(160)
    ma2[iL160, 2, -1] = 0.5
    idx = v16._pire_seed_L160(ma2, float("inf"))
    assert idx == 2


# =============================================================================
# 11. Smoke test de la figure principale (bonus, pas exigé par le brief) --
#     synthétique-mais-cohérent (dupliqué depuis measures_qt.npz pour L=160,
#     pas de vraie physique) : garantit juste que le code de figure ne
#     plante pas, sans dépendre de measures_160.npz.
# =============================================================================


def test_figure_kstar_16L0_smoke(tmp_path):
    ma1_fake_160 = MA1_QT_V2[-1:]  # réutilise la ligne L=80 comme proxy de L=160
    ma2_fake_160 = MA2_QT_V2[-1:]
    ma1_full = v16.agrafe_L_axis(MA1_QT_V2, ma1_fake_160)
    ma2_full = v16.agrafe_L_axis(MA2_QT_V2, ma2_fake_160)

    jnds = v16.toutes_jnd_union(PINS)
    surface = v16.construit_surface_full(ma1_full, ma2_full, jnds)
    resultat = dict(surface_kstar=surface, meta=dict(pins_utilises=PINS))

    path = tmp_path / "smoke_kstar_16L0.png"
    v16.figure_kstar_16L0(resultat, path)
    assert path.exists()
    assert path.stat().st_size > 0
