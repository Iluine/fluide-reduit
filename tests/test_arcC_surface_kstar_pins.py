"""Arc C / Task C-1a — Task 2 : tests du re-seuillage de la surface k*(L)
famille-2 quadtree aux JND RÉELS du pin humain (`scripts/
run_arcC_surface_kstar_pins.py`), reproduit verbatim dans `.superpowers/sdd/
arcC-c1a-task2-kstar-brief.md`. **Re-lecture mécanique de mesures déjà
archivées** (`outputs/arcA/measures_qt.npz`) — aucune simulation, aucun RNG,
aucune nouvelle donnée : ces tests ne font que vérifier que le re-seuillage
appelle CORRECTEMENT `k_star_seed_qt`/`k_star_groupe_qt` (famille 2, JAMAIS
réimplémentées) sur les JND réels de `outputs/arcC/pins_spatial.json`.

Familles de tests (brief) :
  1. Non-régression bit-à-bit : `sous_jnd_a_jnd` aux 4 JND de grille ==
     `sous_jnd_v2` stocké dans `measures_qt.npz`.
  2. Non-régression mandatée par le plan : k*(L=80) à 5 % == 400.0.
  3. Monotonie de k*(L) en JND (non-croissant quand JND augmente), sur la
     grille {2,3,4,5 %} ∪ les 6 pins réels.
  4. Cohérence des drapeaux `sup_cap`/`non_discriminant` avec
     `_non_discriminant`/`CAP_FLOATS` (réutilisés, jamais réimplémentés).
  5. INTERDIT structurel : aucune clé `verdict`/`pente`/`issue`/`cellule`
     dans le JSON produit (forme SÉRIALISÉE, ce qui est réellement écrit sur
     disque) — scan STRICT, sans aucune exception. La clé de provenance a été
     renommée `meta.provenance_kstar` (au lieu de `verdict_reutilise`) pour
     que le garde reste absolu : le module dont les fonctions de verdict
     quadtree sont réutilisées est NOMMÉ en VALEUR, jamais en clé.
  6. Déterminisme : deux constructions sur les mêmes entrées -> structures
     numériquement identiques (aucune RNG dans tout le pipeline)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.run_arcA_measure import JND_LIST, L_LIST, SEEDS
from scripts.run_arcA_measure_qt import MEASURES_PATH
from scripts.run_arcA_verdict import _to_jsonable
from scripts.run_arcA_verdict_qt import (BUDGETS_NON_DISCRIMINANTS, CAP_FLOATS,
                                         _non_discriminant, k_star_groupe_qt)
from scripts.run_arcC_surface_kstar_pins import (PINS_PATH, charge_pins,
                                                 construit_resultat, construit_surface,
                                                 sous_jnd_a_jnd)

# --- Données archivées, chargées une seule fois (réutilisées par tous les tests) --

with np.load(MEASURES_PATH, allow_pickle=False) as _d:
    MA1_V2 = _d["ma1_v2"]
    MA2_V2 = _d["ma2_v2"]
    SOUS_JND_V2_STOCKE = _d["sous_jnd_v2"]

PINS = charge_pins(PINS_PATH)

SOUS_CHAINES_INTERDITES = ("verdict", "pente", "issue", "cellule")


def _cles_recursives(obj):
    """Génère toutes les clés de dict, récursivement (à travers listes)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _cles_recursives(v)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _cles_recursives(item)


# =============================================================================
# 1. Non-régression bit-à-bit : sous_jnd_a_jnd == sous_jnd_v2 stocké
# =============================================================================


def test_recalcul_sous_jnd_reproduit_stocke():
    for iJ, jnd in enumerate(JND_LIST):
        recalcule = sous_jnd_a_jnd(MA1_V2, MA2_V2, jnd)
        stocke = SOUS_JND_V2_STOCKE[..., iJ]
        assert recalcule.dtype == np.bool_
        assert np.array_equal(recalcule, stocke), f"jnd={jnd} : désaccord avec sous_jnd_v2 stocké"


# =============================================================================
# 2. Non-régression mandatée par le plan : k*(L=80) à 5 % == 400.0
# =============================================================================


def test_kstar_L80_a_5pct_vaut_400():
    iL80 = L_LIST.index(80)
    jnd = 0.05
    sj = sous_jnd_a_jnd(MA1_V2, MA2_V2, jnd)
    k_med, _audit = k_star_groupe_qt(sj[iL80], MA1_V2[iL80], MA2_V2[iL80], jnd)
    assert k_med == 400.0


# =============================================================================
# 3. Monotonie de k*(L) en JND (non-croissant), grille ∪ 6 pins
# =============================================================================


def test_monotonie_kstar_en_jnd():
    pins_jnds = [PINS[regime][role] for regime in ("severe", "laxiste")
                for role in ("ic_bas", "pin", "ic_haut")]
    assert len(pins_jnds) == 6
    jnds_tries = sorted(set(JND_LIST) | set(pins_jnds))

    def _kstar(L, jnd):
        iL = L_LIST.index(L)
        sj = sous_jnd_a_jnd(MA1_V2, MA2_V2, jnd)
        k_med, _audit = k_star_groupe_qt(sj[iL], MA1_V2[iL], MA2_V2[iL], jnd)
        return k_med

    for L in L_LIST:
        k_precedent = _kstar(L, jnds_tries[0])
        for jnd in jnds_tries[1:]:
            k_courant = _kstar(L, jnd)
            # ∞ compte comme +∞ dans la comparaison (convention module) : un JND
            # plus lâche ne peut jamais AUGMENTER k*.
            assert k_courant <= k_precedent, (
                f"L={L} : k*({jnd})={k_courant} > k*(précédent)={k_precedent} "
                "-- monotonie violée")
            k_precedent = k_courant


# =============================================================================
# 4. Cohérence sup_cap / non_discriminant
# =============================================================================


def test_cap_et_non_discriminant_coherents():
    surface = construit_surface(MA1_V2, MA2_V2, PINS)
    n_cellules = 0
    for regime in ("severe", "laxiste"):
        for role in ("ic_bas", "pin", "ic_haut"):
            for L in L_LIST:
                cellule = surface[regime][role]["k_star_par_L"][L]
                k = cellule["mediane"]
                assert cellule["sup_cap"] == (math.isfinite(k) and k > CAP_FLOATS)
                assert cellule["non_discriminant"] == _non_discriminant(k)
                n_cellules += 1
    assert n_cellules == 2 * 3 * len(L_LIST)


# =============================================================================
# 5. INTERDIT structurel : aucune clé verdict/pente/issue/cellule (sauf provenance)
# =============================================================================


def test_pas_de_verdict_ni_pente():
    # Scan de la forme SÉRIALISÉE (exactement ce qui est écrit sur disque via
    # json.dumps) — scan STRICT, aucune exception : la clé de provenance est
    # nommée `provenance_kstar` (le module réutilisé est en VALEUR, pas en clé).
    resultat = _to_jsonable(construit_resultat())
    for cle in _cles_recursives(resultat):
        cle_min = cle.lower()
        for interdite in SOUS_CHAINES_INTERDITES:
            assert interdite not in cle_min, (
                f"clé '{cle}' contient la sous-chaîne interdite '{interdite}'")


# =============================================================================
# 6. Déterminisme
# =============================================================================


def test_determinisme():
    r1 = construit_resultat()
    r2 = construit_resultat()
    assert r1 == r2
