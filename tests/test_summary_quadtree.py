"""Tests Arc A manche 1 / famille 2 (spec GRAVÉE, PREREGISTRATION.md
pocCascade2phys commit `a8ed659` section « Spec opérationnelle du quadtree »,
reportée par `docs/superpowers/plans/arc-a-manche1.md` commit `78fbe0c` --
brief `.superpowers/sdd/refond-task8-quadtree-brief.md`) : quadtree de
moyennes `src/summary_quadtree.py`. Neuf familles de tests exigées par le
brief + tests dirigés sur les points délicats nommés (tie-break à gains
EXACTEMENT égaux, refus de split hors-budget) -- la géométrie ne s'improvise
pas, chaque clause de la spec est testée, pas seulement implémentée.
La famille 7 (invariants) suit la clause du CONTRAT (`a8ed659`) et documente
une déviation vs la lettre du brief -- cf. la note de portée en tête de la
section 7 et le contre-exemple exécutable qui la clôt.

Domaine fixe 64x64, profondeur max 6 (feuille 1x1 insécable). Grille de
budgets gravée : {32, 64, 128, 256, 400, 1024, 2048}."""
from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.summary_quadtree import (
    SummaryQT,
    regenerate_qt,
    size_floats_qt,
    summarize_qt,
)

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = [32, 64, 128, 256, 400, 1024, 2048]


# --- Champs de test (trois familles exigées par le brief) -------------------


def _champ_sparse_positif(seed: int, taux_occupation: float = 0.05) -> np.ndarray:
    """Champ (64, 64) positif SPARSE (~5 % d'occupation) -- dépôts épars sur
    fond nul, typique du substrat sédiment."""
    rng = np.random.default_rng(seed)
    field = np.zeros((64, 64), dtype=np.float64)
    mask = rng.random((64, 64)) < taux_occupation
    field[mask] = rng.random(int(mask.sum())) * 3.0 + 0.05
    return field


def _champ_aleatoire_positif(seed: int) -> np.ndarray:
    """Champ (64, 64) positif dense, aléatoire (pas sparse)."""
    return np.random.default_rng(seed).random((64, 64)) * 2.5 + 0.01


def _champ_bosses(seed: int) -> np.ndarray:
    """Champ (64, 64) positif « réel-like » : quelques bosses gaussiennes
    (dépôts) sur un fond quasi nul -- troisième famille, distincte du sparse
    ponctuel et de l'aléatoire dense : c'est le motif où l'adaptativité du
    quadtree (blocs larges sur le fond ~nul, fins sur les bosses) est censée
    apporter quelque chose."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:64, 0:64]
    field = np.full((64, 64), 1e-6, dtype=np.float64)
    for _ in range(4):
        cy, cx = rng.uniform(8, 56, size=2)
        largeur = rng.uniform(2.0, 6.0)
        amplitude = rng.uniform(1.0, 4.0)
        field += amplitude * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * largeur ** 2)))
    return field


_CHAMPS = {
    "sparse": _champ_sparse_positif(101),
    "aleatoire_positif": _champ_aleatoire_positif(202),
    "bosses": _champ_bosses(303),
}


def _champ_quatre_quadrants() -> np.ndarray:
    """Champ (64, 64) à 4 quadrants constants DISTINCTS -- cas construit pour
    la comptabilité (un seul split possible, gain nul partout ensuite)."""
    field = np.empty((64, 64), dtype=np.float64)
    field[0:32, 0:32] = 1.0
    field[0:32, 32:64] = 2.0
    field[32:64, 0:32] = 3.0
    field[32:64, 32:64] = 4.0
    return field


# --- 1. S∘R = id bit-à-bit, ARBRE INCLUS -------------------------------------


@pytest.mark.parametrize("budget", BUDGETS)
@pytest.mark.parametrize("nom_champ", list(_CHAMPS.keys()))
def test_sr_identite_bit_exacte_arbre_inclus(nom_champ, budget):
    field = _CHAMPS[nom_champ]
    s1 = summarize_qt(field, budget)
    regen = regenerate_qt(s1)
    s2 = summarize_qt(regen, budget)

    assert np.array_equal(s2.topology, s1.topology), (
        f"S∘R = id violé sur la TOPOLOGIE (champ={nom_champ}, budget={budget}).")
    assert np.array_equal(s2.means, s1.means), (
        f"S∘R = id violé sur les MOYENNES (champ={nom_champ}, budget={budget}).")


# --- 2. Idempotence stricte ---------------------------------------------------


@pytest.mark.parametrize("budget", BUDGETS)
@pytest.mark.parametrize("nom_champ", list(_CHAMPS.keys()))
def test_idempotence_stricte(nom_champ, budget):
    field = _CHAMPS[nom_champ]
    r1 = regenerate_qt(summarize_qt(field, budget))
    r2 = regenerate_qt(summarize_qt(r1, budget))
    assert np.array_equal(r1, r2), f"idempotence violée (champ={nom_champ}, budget={budget})."


# --- 3. Déterminisme cross-process (motif de tests/test_summary.py) ---------


_SUBPROCESS_SCRIPT = """
import hashlib
import numpy as np
from src.summary_quadtree import summarize_qt, regenerate_qt

rng = np.random.default_rng(101)
field = np.zeros((64, 64), dtype=np.float64)
mask = rng.random((64, 64)) < 0.05
field[mask] = rng.random(int(mask.sum())) * 3.0 + 0.05
summary = summarize_qt(field, 256)
result = regenerate_qt(summary)
print(hashlib.sha256(result.tobytes()).hexdigest())
"""


def test_determinisme_cross_process_via_hash():
    out1 = subprocess.run([sys.executable, "-c", _SUBPROCESS_SCRIPT], cwd=ROOT,
                          capture_output=True, text=True, check=True)
    out2 = subprocess.run([sys.executable, "-c", _SUBPROCESS_SCRIPT], cwd=ROOT,
                          capture_output=True, text=True, check=True)
    h1, h2 = out1.stdout.strip(), out2.stdout.strip()
    assert len(h1) == 64  # hex sha256
    assert h1 == h2


# --- 4. Anti-fuite -------------------------------------------------------------


def test_regenerate_qt_signature_a_un_seul_parametre():
    sig = inspect.signature(regenerate_qt)
    assert len(sig.parameters) == 1


def test_regenerate_qt_apres_mutation_decoy_est_inchange():
    field = _champ_sparse_positif(555)
    decoy = field.copy()
    summary = summarize_qt(decoy, 256)
    result1 = regenerate_qt(summary)
    decoy[:] = 0.0
    decoy[:, :] = 999.0
    result2 = regenerate_qt(summary)
    assert np.array_equal(result1, result2)


# --- 5. Comptabilité exacte sur cas construits -------------------------------


def test_comptabilite_champ_constant():
    field = np.full((64, 64), 3.0)
    summary = summarize_qt(field, budget=2048)
    assert summary.topology.size == 1
    assert summary.means.size == 1
    assert size_floats_qt(summary) == 2  # 1 + ceil(1/32)


def test_comptabilite_quatre_quadrants_un_seul_split():
    field = _champ_quatre_quadrants()
    summary = summarize_qt(field, budget=2048)
    assert summary.means.size == 4       # +3 feuilles (1 -> 4)
    assert summary.topology.size == 5    # +4 nœuds (1 -> 5)
    assert size_floats_qt(summary) == 5  # 4 + ceil(5/32)


@pytest.mark.parametrize("budget", BUDGETS)
@pytest.mark.parametrize("nom_champ", list(_CHAMPS.keys()))
def test_comptabilite_jamais_hors_budget(nom_champ, budget):
    field = _CHAMPS[nom_champ]
    summary = summarize_qt(field, budget)
    assert size_floats_qt(summary) <= budget


def test_comptabilite_refus_du_split_qui_depasserait():
    """budget=4 : la taille racine seule (2) tient, mais un premier split
    coûterait 5 (4 feuilles + ceil(5/32)) -- le split doit être REFUSÉ malgré
    un gain > 0 (4 quadrants distincts), pas juste absent faute de gain."""
    field = _champ_quatre_quadrants()
    summary = summarize_qt(field, budget=4)
    assert summary.topology.size == 1
    assert summary.means.size == 1
    assert size_floats_qt(summary) == 2
    assert size_floats_qt(summary) <= 4

    # sanity : le même champ SANS la contrainte de budget aurait bien splitté
    # (le gain existe donc bien -- le refus ci-dessus est un refus de BUDGET,
    # pas un gain nul déguisé).
    summary_large = summarize_qt(field, budget=2048)
    assert summary_large.means.size == 4


# --- 6. Gain nul jamais splitté ------------------------------------------------


def test_gain_nul_jamais_splitte_champ_constant():
    field = np.full((64, 64), 7.0)
    summary = summarize_qt(field, budget=2048)
    assert np.array_equal(summary.topology, np.array([0], dtype=np.uint8))
    assert summary.means.shape == (1,)
    assert summary.means[0] == pytest.approx(7.0)


# --- 7. Invariants dérivés exacts ---------------------------------------------
#
# Contrat gravé (`a8ed659`, verbatim) : « Invariants (masse totale, masses
# 4×4) : DÉRIVABLES exactement des feuilles (les feuilles dyadiques ne
# chevauchent pas les sous-domaines 16×16 ou les contiennent entièrement) —
# non stockés, pas comptés au budget, vérifiés en test. »
#
# NOTE DE PORTÉE (déviation documentée vs la LETTRE du brief, fidèle au
# contrat -- cf. rapport, préoccupation n° 1) : le brief demandait aussi
# « les 16 masses ... == celles du champ ORIGINAL à 1e-12 relatif », sans
# restriction. C'est IMPOSSIBLE pour TOUT résumé tronqué (propriété
# information-théorique, pas un défaut d'implémentation) : une feuille 32x32
# ou 64x64 couvrant plusieurs sous-domaines 16x16 redistribue leur masse
# UNIFORMÉMENT entre eux (sa moyenne ne retient que la somme du bloc, pas sa
# répartition). Seules la masse TOTALE et la masse PAR FEUILLE sont
# préservées vs l'original à tout budget. Contre-exemple exécutable :
# `test_masses_sous_domaines_non_preservees_sous_troncature` ci-dessous.
# On teste donc : (i) invariants dérivés du résumé == ceux du champ RÉGÉNÉRÉ
# EXACTEMENT (la clause du contrat) ; (ii) masse totale ET masse par feuille
# vs champ ORIGINAL à 1e-12 relatif (les invariants réellement préservés) ;
# (iii) masses 16x16 vs ORIGINAL à 1e-12 relatif LÀ OÙ le contrat les rend
# dérivables (toutes les feuilles du sous-domaine incluses dedans), avec
# garde anti-vacuité.


def _regenere_reference(summary: SummaryQT) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Reconstruction INDÉPENDANTE (test -- n'appelle PAS `regenerate_qt`) de
    (topologie, moyennes, shape) : même principe (peindre chaque feuille à sa
    moyenne) mais implémentation séparée -- un test croisé pixel-à-pixel
    plutôt qu'une confiance aveugle dans le code source. Renvoie aussi la
    géométrie des feuilles [(y, x, côté), ...] en ordre préordre, DÉRIVÉE de
    la topologie seule -- ce que le contrat entend par « dérivables des
    feuilles » (rien d'autre n'est stocké)."""
    H, W = summary.shape
    fine = np.empty((H, W), dtype=np.float64)
    means = summary.means
    topo = summary.topology
    idx_m = [0]
    feuilles: list[tuple[int, int, int]] = []

    def rec(depth: int, row: int, col: int, idx_t: int) -> int:
        bit = int(topo[idx_t])
        idx_t += 1
        if bit == 1:
            for dr, dc in ((0, 0), (0, 1), (1, 0), (1, 1)):
                idx_t = rec(depth + 1, row * 2 + dr, col * 2 + dc, idx_t)
            return idx_t
        cote = 2 ** (6 - depth)
        y, x = row * cote, col * cote
        fine[y:y + cote, x:x + cote] = means[idx_m[0]]
        feuilles.append((y, x, cote))
        idx_m[0] += 1
        return idx_t

    idx_final = rec(0, 0, 0, 0)
    assert idx_final == topo.size
    assert idx_m[0] == means.size
    return fine, feuilles


def _somme_cascade(bloc: np.ndarray) -> float:
    """Réduction en SOMME (pas moyenne) par cascade 2x2 itérée -- jamais de
    réduction plate numpy sur > 4 éléments. Vérifié empiriquement (script ad
    hoc) : pour un bloc constant-par-blocs dyadique, cette cascade redonne la
    somme EXACTE quelle que soit la façon dont on la recalcule par ailleurs
    (mean·aire par feuille) -- une réduction `.sum()` numpy plate NE l'est
    PAS (dans une expérience à 2000 tirages sur des quadrants 8x8 « sales »,
    54 % de désaccords vs la somme feuille-par-feuille contre 0 % pour la
    cascade) : c'est pourquoi la comparaison EXACTE des invariants (test 7)
    passe par cette fonction des deux côtés, jamais par `.sum()` direct."""
    b = bloc
    while b.size > 1:
        H, W = b.shape
        b = b.reshape(H // 2, 2, W // 2, 2).sum(axis=(1, 3))
    return float(b[0, 0])


@pytest.mark.parametrize("budget", BUDGETS)
@pytest.mark.parametrize("nom_champ", list(_CHAMPS.keys()))
def test_invariants_derives_exacts(nom_champ, budget):
    """(i) du contrat : masse totale et 16 masses 16x16 recalculées depuis
    (topologie, moyennes, tailles de feuilles) -- via la reconstruction
    INDÉPENDANTE `_regenere_reference` + cascade -- == celles du champ
    RÉGÉNÉRÉ par `regenerate_qt`, EXACTEMENT (== flottant strict). Les
    feuilles dyadiques ne chevauchent jamais les sous-domaines 16x16 (une
    feuille y est incluse ou en contient un nombre entier) : la dérivation
    est bien définie."""
    field = _CHAMPS[nom_champ]
    summary = summarize_qt(field, budget)
    ref, _ = _regenere_reference(summary)
    regen = regenerate_qt(summary)
    assert np.array_equal(ref, regen), (
        f"reconstruction indépendante != regenerate_qt (champ={nom_champ}, budget={budget}).")

    assert _somme_cascade(ref) == _somme_cascade(regen)  # masse totale, EXACT
    for i in range(4):
        for j in range(4):
            sl = (slice(i * 16, (i + 1) * 16), slice(j * 16, (j + 1) * 16))
            assert _somme_cascade(ref[sl]) == _somme_cascade(regen[sl]), (
                f"masse 16x16 ({i},{j}) dérivée != régénérée "
                f"(champ={nom_champ}, budget={budget}).")


@pytest.mark.parametrize("budget", BUDGETS)
@pytest.mark.parametrize("nom_champ", list(_CHAMPS.keys()))
def test_invariants_masse_totale_et_par_feuille_vs_original(nom_champ, budget):
    """(ii) : les invariants réellement préservés vs le champ ORIGINAL, à
    TOUT budget -- masse totale et masse PAR FEUILLE (moyenne stockée x aire
    == masse du bloc original), à 1e-12 relatif. La préservation par feuille
    entraîne la préservation totale ; les deux sont testées séparément (la
    seconde localise une violation)."""
    field = _CHAMPS[nom_champ]
    summary = summarize_qt(field, budget)
    _, feuilles = _regenere_reference(summary)

    total_champ = float(field.sum())
    total_derive = sum(float(m) * (c * c) for m, (_, _, c) in zip(summary.means, feuilles))
    assert abs(total_derive - total_champ) <= 1e-12 * max(abs(total_champ), 1.0)

    for moyenne, (y, x, cote) in zip(summary.means, feuilles):
        masse_bloc = float(field[y:y + cote, x:x + cote].sum())
        masse_derivee = float(moyenne) * (cote * cote)
        assert abs(masse_derivee - masse_bloc) <= 1e-12 * max(abs(masse_bloc), 1.0), (
            f"masse de feuille (y={y}, x={x}, côté={cote}) non préservée "
            f"(champ={nom_champ}, budget={budget}) : dérivée={masse_derivee}, "
            f"originale={masse_bloc}.")


@pytest.mark.parametrize("nom_champ", list(_CHAMPS.keys()))
def test_invariants_masses_16x16_vs_original_ou_derivables(nom_champ):
    """(iii) : masses 16x16 vs champ ORIGINAL à 1e-12 relatif, restreint aux
    sous-domaines dont TOUTES les feuilles sont incluses dedans (côté <= 16)
    -- la condition du contrat (« les feuilles ... ne chevauchent pas les
    sous-domaines 16x16 ou les contiennent entièrement ») sous laquelle la
    masse du sous-domaine est dérivable du résumé. Garde anti-vacuité : au
    budget 2048, AU MOINS un sous-domaine porteur de masse doit être couvert
    par le test (sinon il ne teste rien)."""
    field = _CHAMPS[nom_champ]
    couverts_porteurs = 0
    for budget in BUDGETS:
        summary = summarize_qt(field, budget)
        _, feuilles = _regenere_reference(summary)
        for i in range(4):
            for j in range(4):
                y0, x0 = i * 16, j * 16
                dedans = [(m, f) for m, f in zip(summary.means, feuilles)
                          if y0 <= f[0] < y0 + 16 and x0 <= f[1] < x0 + 16]
                if any(f[2] > 16 for _, f in dedans):
                    continue  # une feuille déborde : masse non dérivable
                aire = sum(f[2] * f[2] for _, f in dedans)
                if aire != 16 * 16:
                    continue  # sous-domaine traversé par une feuille englobante
                m_champ = float(field[y0:y0 + 16, x0:x0 + 16].sum())
                m_derive = sum(float(m) * (f[2] * f[2]) for m, f in dedans)
                assert abs(m_derive - m_champ) <= 1e-12 * max(abs(m_champ), 1.0), (
                    f"masse 16x16 ({i},{j}) dérivable mais non restituée "
                    f"(champ={nom_champ}, budget={budget}).")
                if budget == 2048 and m_champ > 0.0:
                    couverts_porteurs += 1
    assert couverts_porteurs >= 1, (
        f"test vacueux (champ={nom_champ}) : aucun sous-domaine porteur de "
        "masse entièrement dérivable au budget 2048.")


def test_masses_sous_domaines_non_preservees_sous_troncature():
    """Contre-exemple EXÉCUTABLE de la note de portée (déviation vs la lettre
    du brief) : masse concentrée dans le seul sous-domaine (0,0), budget 5 ->
    seul le split racine tient (feuilles 32x32, chacune couvrant 4
    sous-domaines 16x16). La feuille TL redistribue sa masse UNIFORMÉMENT
    entre ses 4 sous-domaines (64 chacun au lieu de 256/0/0/0) : les 16
    masses vs l'ORIGINAL ne sont PAS restituables sous troncature -- par
    aucun résumé qui ne retient que la moyenne du bloc. La masse TOTALE,
    elle, est bien préservée (ce que teste (ii))."""
    field = np.zeros((64, 64), dtype=np.float64)
    field[0:16, 0:16] = 1.0   # masse 256 dans le sous-domaine (0,0), 0 ailleurs
    summary = summarize_qt(field, budget=5)
    regen = regenerate_qt(summary)

    assert summary.means.size == 4  # racine splittée, 4 feuilles 32x32
    assert float(regen[0:16, 0:16].sum()) == 64.0    # au lieu de 256 : redistribué
    assert float(regen[0:16, 16:32].sum()) == 64.0   # au lieu de 0
    assert float(regen.sum()) == 256.0               # la masse TOTALE, elle, tient


# --- 8. Monotonie de qualité (sanity) -----------------------------------------


@pytest.mark.parametrize("nom_champ", list(_CHAMPS.keys()))
def test_monotonie_sse_non_croissante_en_budget(nom_champ):
    field = _CHAMPS[nom_champ]
    sse_precedent = None
    for budget in BUDGETS:
        regen = regenerate_qt(summarize_qt(field, budget))
        sse = float(np.sum((field - regen) ** 2))
        if sse_precedent is not None:
            assert sse <= sse_precedent + 1e-9, (
                f"SSE croissante en budget (champ={nom_champ}, budget={budget}) : "
                f"{sse} > {sse_precedent}.")
        sse_precedent = sse


# --- 9. Champ nul --------------------------------------------------------------


def test_champ_nul():
    field = np.zeros((64, 64), dtype=np.float64)
    summary = summarize_qt(field, budget=2048)
    assert np.array_equal(summary.topology, np.array([0], dtype=np.uint8))
    assert np.array_equal(summary.means, np.array([0.0]))
    regen = regenerate_qt(summary)
    assert np.array_equal(regen, np.zeros((64, 64), dtype=np.float64))


# --- Tie-break (point délicat nommé, en sus des 9 familles) ------------------


def test_tie_break_gain_egal_plus_petit_y_puis_x_dabord():
    """Construit un champ où deux splits À DIFFÉRENTES POSITIONS ont
    EXACTEMENT le même gain -- vérifie que le glouton splitte d'abord celui
    de plus petit y, puis (à y égal) de plus petit x, conformément au
    tie-break lexicographique (gain, y, x) gravé au contrat."""
    field = np.zeros((64, 64), dtype=np.float64)
    # Quatre quadrants 32x32 (TL, TR, BL, BR), moyennes DISTINCTES m_i (pour
    # que la racine ait un gain > 0 et splitte -- seule candidate initiale)
    # mais chacun coupé en 2 bandes horizontales m_i-d / m_i+d avec le MÊME
    # écart d : le gain de split d'un nœud depth1 ne dépend QUE de d (pas de
    # m_i, cf. calcul : gain = n_c·Σ(enfant-parent)² = n_c·4d², indépendant du
    # centre) -- les 4 nœuds depth1 ont donc un gain EXACTEMENT égal, seule
    # leur position (y, x) diffère : (0,0), (0,32), (32,0), (32,32).
    # d et m_i DYADIQUES (représentables exactement) : moyennes de cascade et
    # gains alors EXACTS en flottant -- l'égalité des 4 gains est bit-à-bit,
    # le tie-break est réellement exercé (avec d=0.1, non dyadique, les gains
    # diffèrent d'ulps et le test ne teste pas le tie-break -- vérifié).
    d = 0.5
    m = (1.0, 2.0, 3.0, 4.0)  # TL, TR, BL, BR
    field[0:16, 0:32] = m[0] - d
    field[16:32, 0:32] = m[0] + d
    field[0:16, 32:64] = m[1] - d
    field[16:32, 32:64] = m[1] + d
    field[32:48, 0:32] = m[2] - d
    field[48:64, 0:32] = m[2] + d
    field[32:48, 32:64] = m[3] - d
    field[48:64, 32:64] = m[3] + d

    # budget pour EXACTEMENT 1 split (racine -> 4 enfants depth1, taille 5) :
    # les 4 enfants depth1 ont ensuite tous le même gain (structure identique
    # dans chaque quadrant) -- le split suivant doit choisir le quadrant de
    # plus petit y puis plus petit x, c'est-à-dire (0, 0) : depth1 (row=0,col=0).
    summary = summarize_qt(field, budget=8)  # 1 split racine (5) + 1 split enfant (8)
    assert summary.means.size == 7  # 4 (après split racine) - 1 + 4 = 7
    assert summary.topology.size == 9  # 5 + 4

    # Le deuxième split doit avoir eu lieu sur l'enfant (depth=1,row=0,col=0)
    # -- topologie préordre : [1 (racine), 1 (enfant TL splitté), 0,0,0,0
    # (petits-enfants TL, feuilles), 0 (TR), 0 (BL), 0 (BR)].
    attendu = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)
    assert np.array_equal(summary.topology, attendu), (
        f"tie-break violé : topologie obtenue {summary.topology.tolist()} != "
        f"attendue {attendu.tolist()} (le split devait choisir le plus petit "
        "y puis le plus petit x, ici le quadrant top-left).")


# --- Bilan (sanity) : suite verte + budgets couverts -------------------------


def test_grille_budgets_couvre_bien_le_contrat():
    assert BUDGETS == [32, 64, 128, 256, 400, 1024, 2048]
