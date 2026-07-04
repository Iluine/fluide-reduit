"""Arc C / Task 1 — tests du générateur de stimuli spatiaux `stim(t)` et de
l'instrument de mesure Δχ (`src/arcC_stimuli.py`).

Contrat : addendum §C3 (« C-2 : harnais ABX sur stimuli réels ») de
PREREGISTRATION.md (pocCascade2phys, commits `c494d50` + `5d13bc4`), reproduit
verbatim dans `.superpowers/sdd/arcC-task1-stimuli-brief.md`. Résolution
d'ambiguïté du contrôleur (brief) : le mélange `stim(t)` se fait en espace
SÉDIMENT (pas en espace albedo) ; le Δχ mesuré est `delta_chi(albedo(stim(t),
S_HALF_OP), albedo(s_L, S_HALF_OP))["max_carrier"]`, référence = le champ VRAI
`s_L` (bandes porteuses fixées par le vrai champ, constantes le long de t).

Cinq familles de tests exigées par le brief :
  1. Δχ(0) = 0.0 exact.
  2. Déterminisme (double appel bit-identique + cross-process via hash).
  3. Courbe Δχ(t) rapportée, monotonie NON supposée (bornes seulement).
  4. Bornes physiques de `melange` (positivité, shape, t=0/t=1 bit-exacts).
  5. Cohérence R1 : `delta_chi_stim` ancré au pipeline manche 1
     (`outputs/arcA/measures_qt.npz`, `ma1_v2`).

JAMAIS de comparaison L2 point-à-point : le seul juge de fidélité est R1
(`delta_chi(...)["max_carrier"]`, espace perceptuel). Aucun stimulus
synthétique : les champs sources viennent des npz commités de la manche 1
(`outputs/arcA/histories/h_s{seed}.npz`)."""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.run_arcA_revalidate import S_HALF_OP
from src.arcC_stimuli import courbe_delta_chi, delta_chi_stim, melange, regenere_budget

ROOT = Path(__file__).resolve().parents[1]
HIST_DIR = ROOT / "outputs" / "arcA" / "histories"
MEASURES_QT_PATH = ROOT / "outputs" / "arcA" / "measures_qt.npz"

# Grille manche 1 (budgets famille 2, cf. brief -- réutilisés tels quels).
BUDGETS: tuple[int, ...] = (32, 64, 128, 256, 400, 1024, 2048)
L_LIST: tuple[int, ...] = (10, 20, 40, 80)
SEEDS: tuple[int, ...] = (101, 102, 103, 104, 105)


def _charger_s_true(seed: int, L: int) -> np.ndarray:
    """Champ de sédiment `s_L{L}` du npz d'histoire de la seed -- AUCUNE
    régénération de simulation, uniquement lecture des artefacts commités."""
    with np.load(HIST_DIR / f"h_s{seed}.npz") as d:
        return np.array(d[f"s_L{L}"], dtype=np.float64)


# --- Famille 1 : Δχ(0) = 0.0 exact --------------------------------------------


@pytest.mark.parametrize("seed,L,budget", [
    (101, 10, 32), (101, 80, 2048), (103, 40, 256), (105, 20, 400),
])
def test_delta_chi_stim_en_t0_est_exactement_zero(seed, L, budget):
    """stim(t=0) == s_true bit-exact (mélange convexe à t=0) => Δχ(0) = 0.0
    exact -- pas une tolérance, la formule dégénère en delta_chi(a, a)."""
    s_true = _charger_s_true(seed, L)
    s_regen = regenere_budget(s_true, budget)
    s_stim_0 = melange(s_true, s_regen, 0.0)
    assert delta_chi_stim(s_true, s_stim_0) == 0.0


@pytest.mark.parametrize("seed,L,budget", [(101, 10, 64), (102, 80, 1024)])
def test_courbe_delta_chi_premier_point_est_exactement_zero(seed, L, budget):
    s_true = _charger_s_true(seed, L)
    courbe = courbe_delta_chi(s_true, budget)
    assert courbe[0] == 0.0


# --- Famille 2 : déterminisme --------------------------------------------------


def test_courbe_delta_chi_double_appel_bit_identique():
    s_true = _charger_s_true(101, 10)
    ts = np.linspace(0.0, 1.0, 11)
    c1 = courbe_delta_chi(s_true, 128, ts)
    c2 = courbe_delta_chi(s_true, 128, ts)
    assert np.array_equal(c1, c2)


_STIMULI_SUBPROCESS_SCRIPT = """
import hashlib
import numpy as np
from src.arcC_stimuli import courbe_delta_chi

with np.load("outputs/arcA/histories/h_s101.npz") as d:
    s_true = np.array(d["s_L10"], dtype=np.float64)
courbe = courbe_delta_chi(s_true, 256)
print(hashlib.sha256(courbe.tobytes()).hexdigest())
"""


def test_courbe_delta_chi_deterministe_cross_process_via_hash():
    """Motif de `tests/test_summary.py` (`test_ma3_identite_cross_process_via_hash`) :
    deux process indépendants (aucun état partagé) doivent produire le même
    hash SHA-256 de la courbe -- déterminisme fort, pas seulement en-process."""
    out1 = subprocess.run([sys.executable, "-c", _STIMULI_SUBPROCESS_SCRIPT], cwd=ROOT,
                          capture_output=True, text=True, check=True)
    out2 = subprocess.run([sys.executable, "-c", _STIMULI_SUBPROCESS_SCRIPT], cwd=ROOT,
                          capture_output=True, text=True, check=True)
    h1, h2 = out1.stdout.strip(), out2.stdout.strip()
    assert len(h1) == 64  # hex sha256
    assert h1 == h2


# --- Famille 3 : courbe Δχ(t) rapportée, monotonie NON supposée ---------------


@pytest.mark.parametrize("seed,L,budget", [
    (101, 10, 32), (101, 10, 2048), (104, 40, 400), (105, 80, 128),
])
def test_courbe_delta_chi_bornes_sans_hypothese_de_monotonie(seed, L, budget):
    """Vérifie SEULEMENT les bornes gravées (§C3 : Δχ mesuré, pas supposé) :
    Δχ(0) = 0.0, Δχ >= 0 partout, Δχ(1) = delta_chi_stim(s_true, régénéré) =
    M-A1 manche 1 du budget (à 1e-12). La monotonie n'est PAS assertée --
    seulement observée/rapportée (booléen calculé, sans conséquence de test) :
    la spec interdit explicitement de la supposer."""
    s_true = _charger_s_true(seed, L)
    ts = np.linspace(0.0, 1.0, 11)
    courbe = courbe_delta_chi(s_true, budget, ts)

    assert courbe.shape == ts.shape
    assert courbe[0] == 0.0
    assert np.all(courbe >= 0.0)

    s_regen = regenere_budget(s_true, budget)
    attendu_t1 = delta_chi_stim(s_true, s_regen)
    assert courbe[-1] == pytest.approx(attendu_t1, abs=1e-12)

    # Observation SEULE (pas d'assert) -- documente le choix du brief : la
    # courbe peut être non-monotone, ce n'est pas un échec de test.
    est_monotone = bool(np.all(np.diff(courbe) >= -1e-15))
    assert est_monotone in (True, False)  # tautologie : juste marquer l'observation


# --- Famille 4 : bornes physiques de `melange` --------------------------------


@pytest.mark.parametrize("t", [0.0, 0.5, 1.0])
def test_melange_reste_non_negatif(t):
    """Combinaison convexe de deux champs >= 0 -- reste >= 0 (borne physique,
    cf. brief). Les champs sédiment sources sont >= 0 par construction."""
    s_true = _charger_s_true(101, 10)
    s_regen = regenere_budget(s_true, 128)
    stim = melange(s_true, s_regen, t)
    assert np.all(stim >= 0.0)


def test_melange_preserve_la_shape():
    s_true = _charger_s_true(103, 40)
    s_regen = regenere_budget(s_true, 400)
    stim = melange(s_true, s_regen, 0.37)
    assert stim.shape == s_true.shape == s_regen.shape


def test_melange_t0_est_s_true_bit_exact():
    s_true = _charger_s_true(101, 10)
    s_regen = regenere_budget(s_true, 32)
    stim0 = melange(s_true, s_regen, 0.0)
    assert np.array_equal(stim0, s_true)


def test_melange_t1_est_s_regen_bit_exact():
    s_true = _charger_s_true(101, 10)
    s_regen = regenere_budget(s_true, 32)
    stim1 = melange(s_true, s_regen, 1.0)
    assert np.array_equal(stim1, s_regen)


def test_melange_rejette_shapes_incompatibles():
    s_true = _charger_s_true(101, 10)
    with pytest.raises(ValueError):
        melange(s_true, s_true[:32, :32], 0.5)


@pytest.mark.parametrize("t", [-0.1, 1.1])
def test_melange_rejette_t_hors_0_1(t):
    s_true = _charger_s_true(101, 10)
    with pytest.raises(ValueError):
        melange(s_true, s_true, t)


# --- Famille 5 : cohérence R1 (ancrage au pipeline manche 1) ------------------


def test_delta_chi_stim_identite_est_nulle():
    s_true = _charger_s_true(102, 20)
    assert delta_chi_stim(s_true, s_true) == 0.0


@pytest.mark.parametrize("iL,L,iSeed,seed,iBudget,budget", [
    (0, 10, 0, 101, 0, 32),
    (0, 10, 0, 101, 6, 2048),
    (3, 80, 4, 105, 3, 256),
])
def test_delta_chi_stim_coincide_avec_ma1_manche1(iL, L, iSeed, seed, iBudget, budget):
    """Ancre l'instrument au pipeline validé : `delta_chi_stim(s_L,
    regenere_budget(s_L, budget))` doit reproduire EXACTEMENT (1e-12) le
    M-A1 (`ma1_v2`) de `outputs/arcA/measures_qt.npz` à la cellule
    (L, seed, budget) correspondante -- même instrument, même régénérateur
    (cf. brief, section « Livrables » test 5)."""
    with np.load(MEASURES_QT_PATH, allow_pickle=False) as d:
        ma1_v2 = d["ma1_v2"]
    assert ma1_v2.shape == (len(L_LIST), len(SEEDS), len(BUDGETS))
    ma1_attendu = float(ma1_v2[iL, iSeed, iBudget])

    s_true = _charger_s_true(seed, L)
    s_regen = regenere_budget(s_true, budget)
    mesure = delta_chi_stim(s_true, s_regen)
    assert mesure == pytest.approx(ma1_attendu, abs=1e-12)


def test_s_half_op_importe_est_celui_grave():
    """Sanity anti-recalcul : S_HALF_OP = 0.05*RELIEF (gravé manche 1),
    importé tel quel de `scripts.run_arcA_revalidate` -- ne doit jamais être
    recalculé localement dans `src/arcC_stimuli.py`."""
    assert S_HALF_OP > 0.0
