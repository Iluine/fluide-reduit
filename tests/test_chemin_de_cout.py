"""Tests — CHEMIN-DE-COÛT (`src/f1_gpu/chemin_de_cout.py`), les quatre gardes
de §A48 vérifiées MÉCANIQUEMENT.

Backend numpy (B1), tailles minuscules (VM) : on teste ici la LOGIQUE et les
GARDES, jamais un coût. Le chiffre se mesure ailleurs, sous cupy.

Ce fichier ne mesure pas la fidélité et n'en approche pas : le chemin-de-coût
est jugé au coût seul, et la garde 2 existe précisément pour que personne ne
puisse l'amener sous un ABX.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.f1_gpu.chemin_de_cout import (
    CheminDeCoutInterdit,
    albedo_ecran,
    gather_chemin_de_cout,
)
from src.f1_gpu.pyramide import GeometriePyramide, PyramideFovea, Slot
from src.f1_gpu.transferts import TransfertComptable

N_FOV = 8
N0 = 16
N_NIV = 4  # niveau 0 CPU + niveaux GPU 1, 2, 3 ; niveau fin = 3, monde = 128

SLOTS = (
    Slot(niveau=1, c=4, role="fovea"),
    Slot(niveau=2, c=8, role="fovea"),
    Slot(niveau=3, c=8, role="fovea"),
)


def _pyramide():
    geo = GeometriePyramide(n_fov=N_FOV, n_niv=N_NIV, n0=N0, slots=SLOTS)
    transferts = TransfertComptable(np)
    pyr = PyramideFovea(np, geo, transferts, eps_detail=1e30)
    transferts.frame_suivante()  # clôt la descente initiale (B4)
    return pyr


def _peindre_par_niveau(pyr) -> None:
    """Peint le champ `s` (indice 3) de chaque fenêtre avec le numéro de son
    niveau — rend l'origine de chaque pixel écran LISIBLE."""
    for j in pyr.geo.niveaux_gpu:
        pyr.fenetres[j][:, :, 3, :, :] = float(j)


def _appeler_depuis(nom_module: str, fonction, *args, **kwargs):
    """Exécute `fonction` depuis une frame dont `__name__` vaut `nom_module`.

    C'est la seule façon honnête de tester la garde 2 : elle inspecte les
    `f_globals['__name__']` de la pile, donc il faut une vraie frame portant le
    nom d'un module perceptuel — pas un monkeypatch de la garde elle-même."""
    globaux = {"__name__": nom_module, "_f": fonction,
               "_a": args, "_k": kwargs}
    exec(compile("_res = _f(*_a, **_k)", "<frame-de-test>", "exec"), globaux)
    return globaux["_res"]


# ----- GARDE 2 : la serrure -------------------------------------------------

@pytest.mark.parametrize("module_perceptuel", [
    "src.arcC_abx", "src.arcC_stimuli", "src.arcC_orchestration",
    "src.arcC_rendu", "src.arcC_scelle", "src.arcC_calibration",
])
def test_serrure_refuse_tout_chemin_perceptuel(module_perceptuel) -> None:
    """§A48 garde 2 : le chemin-de-coût ne passe JAMAIS sous une mesure de
    fidélité. L'appel depuis un module perceptuel lève, quelle que soit la
    profondeur de la pile."""
    pyr = _pyramide()
    with pytest.raises(CheminDeCoutInterdit, match="chemin-de-coût appelé"):
        _appeler_depuis(module_perceptuel, gather_chemin_de_cout, pyr, 32)


def test_serrure_detecte_meme_en_profondeur() -> None:
    """La garde remonte TOUTE la pile : un module perceptuel qui délègue à un
    intermédiaire neutre est quand même refusé. Sans cela, une seule fonction
    d'emballage suffirait à la contourner."""
    pyr = _pyramide()

    def intermediaire_neutre(p, n):
        return gather_chemin_de_cout(p, n)

    with pytest.raises(CheminDeCoutInterdit):
        _appeler_depuis("src.arcC_abx", intermediaire_neutre, pyr, 32)


def test_serrure_laisse_passer_un_appelant_neutre() -> None:
    """Contre-épreuve : sans module perceptuel dans la pile, l'appel passe.
    Une serrure qui refuse tout ne prouve rien."""
    pyr = _pyramide()
    ecran = _appeler_depuis("src.f1_gpu.un_driver_de_cout",
                            gather_chemin_de_cout, pyr, 32)
    assert ecran.shape == (32, 32)


def test_la_violation_laisse_une_trace_meme_avalee() -> None:
    """La faille trouvée PAR MUTATION le 2026-08-03, et sa réparation.

    `CheminDeCoutInterdit` hérite de `RuntimeError` (§A48 l'exige) et ce module
    lève aussi des `RuntimeError` pour les trous de couverture : un appelant qui
    écrit `except RuntimeError` — geste banal — avale la serrure en silence.

    On ne prétend pas l'empêcher : on exige que la trace SURVIVE à
    l'étouffement, pour qu'un driver qui prononce un chiffre puisse refuser de
    l'écrire. Même logique que le scellement (`src/arcC_scelle.py`) : la
    non-cécité se consigne, elle ne s'interdit pas."""
    from src.f1_gpu.chemin_de_cout import violations_consignees

    pyr = _pyramide()
    avant = len(violations_consignees())

    try:  # exactement l'étouffement bien intentionné
        _appeler_depuis("src.arcC_abx", gather_chemin_de_cout, pyr, 32)
    except RuntimeError:
        pass

    apres = violations_consignees()
    if len(apres) != avant + 1:
        raise AssertionError(
            "La violation n'a pas laissé de trace : un `except RuntimeError` "
            "suffit alors à neutraliser la garde 2 sans que rien ne le montre.")
    assert "arcC_abx" in apres[-1]


def test_la_serrure_survit_a_python_dash_O() -> None:
    """§A43 : un `assert` disparaît sous `python -O`. La garde 2 doit donc être
    une exception explicite, et ce module ne doit contenir AUCUN `assert` dans
    son chemin de garde.

    Vérifié sur le source lui-même : c'est mécanique, et ça résiste à une
    réécriture distraite du module."""
    from pathlib import Path

    import src.f1_gpu.chemin_de_cout as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    lignes_assert = [ligne.strip() for ligne in source.splitlines()
                     if ligne.strip().startswith("assert ")]
    if lignes_assert:
        raise AssertionError(
            "Le chemin-de-coût contient un `assert` en chemin d'exécution "
            f"({lignes_assert!r}) : il s'évapore sous `python -O` et la garde 2 "
            "redevient une promesse. Utiliser une exception explicite.")
    assert issubclass(CheminDeCoutInterdit, RuntimeError)


# ----- GARDE 3 : disqualification par le nom --------------------------------

def test_aucune_dependance_vers_un_module_perceptuel() -> None:
    """§A48 garde 3 : jetable par construction. Le module n'importe rien de la
    chaîne instrumentée — ni `arcC_*`, ni les upsamples denses que la garde 1
    interdit (`src/multiresolution.py`, `src/summary.py`)."""
    from pathlib import Path

    import src.f1_gpu.chemin_de_cout as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    for ligne in source.splitlines():
        depouillee = ligne.strip()
        if not (depouillee.startswith("import ")
                or depouillee.startswith("from ")):
            continue
        for interdit in ("arcC", "multiresolution", "src.summary"):
            if interdit in depouillee:
                raise AssertionError(
                    f"Import interdit dans le chemin-de-coût : {depouillee!r}. "
                    "Un gather depuis un grossier dense, ou une dépendance vers "
                    "l'instrument, violent les gardes 1 et 3 de §A48.")


# ----- GARDE 1 : plancher sur structure réelle -------------------------------

def test_ecran_entierement_couvert() -> None:
    """Aucun NaN ne survit : chaque pixel vient d'une fenêtre active réelle."""
    pyr = _pyramide()
    ecran = gather_chemin_de_cout(pyr, 32)
    assert ecran.shape == (32, 32)
    assert not np.isnan(ecran).any()


def test_le_fin_ecrase_le_grossier() -> None:
    """INSERTION DURE : dans l'ordre croissant des niveaux, le fin écrase. Le
    centre de l'écran — couvert par la fenêtre fovéale du niveau fin — doit
    porter la valeur du niveau fin, pas celle d'un grossier."""
    pyr = _pyramide()
    _peindre_par_niveau(pyr)
    ecran = gather_chemin_de_cout(pyr, 32)

    centre = ecran[16, 16]
    assert centre == float(pyr.geo.niveau_fin), (
        f"Le centre porte {centre}, pas le niveau fin "
        f"{pyr.geo.niveau_fin} : l'insertion dure n'écrase pas dans le bon "
        "ordre.")
    # Et le grossier subsiste là où le fin ne couvre pas : sinon le gather ne
    # ferait rien d'intéressant et le coût mesuré serait celui d'un seul niveau.
    assert len(np.unique(ecran)) > 1, (
        "Un seul niveau apparaît à l'écran : l'écran ne déborde pas la fenêtre "
        "fine, et le coût mesuré ne serait pas représentatif d'un gather.")


def test_plus_proche_voisin_aucune_valeur_fabriquee() -> None:
    """GARDE 1, le test qui mord : toute valeur de l'écran appartient à
    l'ensemble des valeurs des fenêtres actives.

    Une interpolation — bilinéaire, cohérente, quelle qu'elle soit — fabrique
    des valeurs INTERMÉDIAIRES qui n'existent dans aucune fenêtre. Ce test
    l'attrape sans avoir à lire l'implémentation, et c'est ce qui garantit que
    le chiffre reste un PLANCHER de la classe gather."""
    pyr = _pyramide()
    rng = np.random.default_rng(20260803)
    for j in pyr.geo.niveaux_gpu:
        forme = pyr.fenetres[j][:, :, 3, :, :].shape
        pyr.fenetres[j][:, :, 3, :, :] = rng.random(forme).astype(np.float32)

    ecran = gather_chemin_de_cout(pyr, 32)

    valeurs_sources = set()
    for j in pyr.geo.niveaux_gpu:
        valeurs_sources.update(
            np.unique(pyr.fenetres[j][:, :, 3, :, :]).tolist())

    etrangeres = set(np.unique(ecran).tolist()) - valeurs_sources
    if etrangeres:
        raise AssertionError(
            f"{len(etrangeres)} valeur(s) à l'écran n'existent dans AUCUNE "
            "fenêtre active : le chemin interpole au lieu de prendre le plus "
            "proche voisin. Le chiffre cesserait d'être un plancher (§A48 "
            "garde 1).")


def test_trou_de_couverture_fail_loud() -> None:
    """Un écran plus large que la couverture des fenêtres actives lève, au lieu
    de se remplir par défaut. Un remplissage silencieux fausserait le coût
    mesuré — on paierait un écran que la structure ne sait pas produire."""
    pyr = _pyramide()
    with pytest.raises(RuntimeError, match="AUCUNE fenêtre active"):
        gather_chemin_de_cout(pyr, 4096)


def test_cote_px_invalide() -> None:
    pyr = _pyramide()
    with pytest.raises(ValueError, match="cote_px"):
        gather_chemin_de_cout(pyr, 0)


# ----- KERNEL UNIQUE : équivalence stricte ----------------------------------

def _cupy_ou_skip():
    from src.f1_gpu.backend import cupy_disponible
    if not cupy_disponible():
        pytest.skip("chemin GPU indisponible — l'équivalence kernel/Python ne "
                    "peut pas être vérifiée ici (jamais supposée).")
    import cupy
    return cupy


@pytest.mark.parametrize("cote_px", [16, 24, 32])
def test_kernel_strictement_equivalent_a_la_voie_python(cote_px) -> None:
    """LE test du kernel : mêmes valeurs, BIT À BIT.

    Le kernel parcourt du FIN vers le GROSSIER avec arrêt au premier slot
    couvrant ; la voie Python insère du GROSSIER vers le FIN avec écrasement.
    Les deux sont censés être équivalents — c'est un raisonnement, donc il se
    teste. Si le kernel divergeait, on aurait changé l'ARITHMÉTIQUE en croyant
    ne changer que l'implémentation, et le chiffre cesserait d'être comparable
    au précédent."""
    cp = _cupy_ou_skip()
    from src.f1_gpu.chemin_de_cout import GatherKernel

    geo = GeometriePyramide(n_fov=N_FOV, n_niv=N_NIV, n0=N0, slots=SLOTS)
    transferts = TransfertComptable(cp)
    pyr = PyramideFovea(cp, geo, transferts, eps_detail=1e30)
    transferts.frame_suivante()

    rng = np.random.default_rng(20260803)
    for j in geo.niveaux_gpu:
        forme = pyr.fenetres[j][:, :, 3, :, :].shape
        pyr.fenetres[j][:, :, 3, :, :] = cp.asarray(
            rng.random(forme).astype(np.float32))

    par_python = cp.asnumpy(gather_chemin_de_cout(pyr, cote_px))
    par_kernel = cp.asnumpy(GatherKernel(pyr).gather(cote_px))

    np.testing.assert_array_equal(par_kernel, par_python)


def test_kernel_fail_loud_sur_trou_de_couverture() -> None:
    """Le kernel écrit des NaN pour un pixel non couvert et le contrôle lève —
    même comportement que la voie Python, pas un remplissage silencieux."""
    cp = _cupy_ou_skip()
    from src.f1_gpu.chemin_de_cout import GatherKernel

    geo = GeometriePyramide(n_fov=N_FOV, n_niv=N_NIV, n0=N0, slots=SLOTS)
    transferts = TransfertComptable(cp)
    pyr = PyramideFovea(cp, geo, transferts, eps_detail=1e30)
    transferts.frame_suivante()

    with pytest.raises(RuntimeError, match="AUCUNE fenêtre active"):
        GatherKernel(pyr).gather(4096)


def test_kernel_verrouille_aussi() -> None:
    """La garde 2 s'applique au kernel, à la construction ET à l'appel : une
    seconde voie d'accès serait une porte dérobée sur la serrure."""
    cp = _cupy_ou_skip()
    from src.f1_gpu.chemin_de_cout import GatherKernel

    geo = GeometriePyramide(n_fov=N_FOV, n_niv=N_NIV, n0=N0, slots=SLOTS)
    transferts = TransfertComptable(cp)
    pyr = PyramideFovea(cp, geo, transferts, eps_detail=1e30)
    transferts.frame_suivante()

    with pytest.raises(CheminDeCoutInterdit):
        _appeler_depuis("src.arcC_abx", GatherKernel, pyr)

    kernel = GatherKernel(pyr)  # construit hors chemin perceptuel...
    with pytest.raises(CheminDeCoutInterdit):  # ...mais l'appel reste gardé
        _appeler_depuis("src.arcC_abx", kernel.gather, 32)


# ----- readout -------------------------------------------------------------

def test_albedo_ecran_est_la_formule_de_la_maison() -> None:
    """`albedo_ecran` réplique `src/albedo.py:27-30` sans l'importer (garde 3) :
    l'égalité est donc TESTÉE, pas supposée."""
    from src.albedo import albedo

    champ = np.linspace(0.0, 5.0, 64, dtype=np.float64).reshape(8, 8)
    np.testing.assert_allclose(albedo_ecran(champ, 1.5), albedo(champ, 1.5),
                               rtol=0, atol=0)


def test_albedo_ecran_refuse_un_s_half_nul() -> None:
    with pytest.raises(ValueError, match="s_half"):
        albedo_ecran(np.zeros((4, 4)), 0.0)
