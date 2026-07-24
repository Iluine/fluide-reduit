"""Vérification du portage — la comparaison MANQUANTE depuis C1 (§6 de
claude/verif-echelle-grossier-2026-07-25.md).

CE QUE CE FICHIER EST : le test unitaire qui compare le kernel GPU fidèle
(`_SOURCE_FIDELE`) à l'opérateur de référence qu'il déclare porter
(`solver_wetdry._rhs_o2`) — comparaison absente depuis l'origine
(`grep _rhs_o2 tests/` ne retournait rien : le verrou d'empreinte protège
la stabilité, pas la justesse). Autorisé par Romain le 2026-07-25 (D19-a).
Ce n'est PAS un diagnostic ni une mesure : un pas, aucun chrono, aucun
épisode, aucun seuil déplacé. Il est utile quel que soit son résultat.

CE QU'IL DÉCIDE : le kernel ne recevant AUCUN pas d'espace, s'il reproduit
`_rhs_o2` à Δx = 1 sur un champ NON TRIVIAL (flux non nuls — sinon on
reproduit l'aveuglement des tests d'équilibre, §3 de la vérification),
alors son emploi au niveau GROSSIER de T2 (physiquement Δx = 2) est à
mauvaise échelle : le soupçon de §0 de la séance É2 est CONFIRMÉ par
exécution. S'il reproduit `_rhs_o2` à Δx = 2, le soupçon est INFIRMÉ.

PRÉDICTIONS PRÉ-ÉCRITES (verif §6) :
  - lecture tient  : L_gpu ≡ L_ref(Δx=1) (à l'arithmétique f32 près)
                     et L_ref(Δx=1) = 2·L_ref(Δx=2) EXACTEMENT ;
  - lecture fausse : L_gpu ≡ L_ref(Δx=2).
Les deux branches sont décidables ; le test n'ÉCHOUE que si AUCUNE ne
tient — ce qui serait un TROISIÈME fait, à remonter tel quel.

POINT D'ARRÊT : le résultat se REMONTE à Romain — les lectures
décisionnelles D19-c/d sont volontairement NON pré-enregistrées (décision
Romain, 2026-07-25). Lancer avec `pytest -s` pour voir la lecture
imprimée. Le test de portage exige le GPU (iluin-tworings3) ; le test
d'échelle de la référence est CPU pur et tourne partout.

CE QUI A CHANGÉ APRÈS LE CORRECTIF (D19-c endossé, §A33-CORRECTION). Le
résultat est tombé : err(Δx=1) = 3.297e-05 (bruit f32), err(Δx=2) =
5.000e-01 (la signature du facteur 2) — soupçon CONFIRMÉ. Le kernel a été
rendu Δx-conscient. Ce fichier porte donc désormais TROIS tests, et son
rôle a basculé du diagnostic à la garde permanente :
  1. la référence elle-même, L(1) = 2·L(2) — inchangé, CPU pur ;
  2. le kernel à `inv_dx = 1.0` ≡ `_rhs_o2(Δx=1)` — NON-RÉGRESSION du
     régime mono-niveau (témoin, T1, fovéa fine), qui ne devait rien voir ;
  3. le kernel à `inv_dx = 0.5` ≡ `_rhs_o2(Δx=2)` — LE test du correctif :
     le niveau grossier tourne enfin à sa maille."""
import numpy as np
import pytest

from config import GridConfig
from src.f1_gpu.backend import cupy_disponible
from src.f1_gpu.substrat_fidele import (
    _TAILLE_BLOC,
    MODE_REFLECHISSANT,
    _obtenir_kernel_fidele,
)
from src.f1_gpu.substrat_jetable import DRY_EPS
from src.sediment import default_terrain
from src.solver_wetdry import _rhs_o2

gpu_requis = pytest.mark.skipif(
    not cupy_disponible(), reason="cupy/device CUDA indisponible")

N: int = 32          # la taille du GROSSIER de T2 (64² décimé ×2)
DT: float = 0.005    # sous-CFL large ; dt·L ≫ bruit f32, h reste loin du sec


def _champ_non_trivial():
    """32² MOUILLÉ PARTOUT : terrain réel décimé 2×2 (comme le grossier),
    bosse de surface libre + cisaillement doux. Flux NON nuls (l'exigence
    de §6 : un champ à flux nuls reproduirait l'aveuglement de §3), aucun
    front sec (les clamps du kernel ne mordent pas). Aller-retour f32 :
    CPU et GPU reçoivent EXACTEMENT les mêmes valeurs."""
    b64 = default_terrain(GridConfig()).astype(np.float64)
    b = b64.reshape(N, 2, N, 2).mean(axis=(1, 3))
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float64)
    eta = (float(b.max()) + 0.6
           + 0.25 * np.exp(-((yy - 11.0) ** 2 + (xx - 19.0) ** 2)
                           / (2.0 * 5.0 ** 2)))
    h = eta - b
    hu = h * (0.08 * np.sin(2.0 * np.pi * yy / N))
    hv = h * (0.05 * np.cos(2.0 * np.pi * xx / N))
    h, hu, hv, b = (a.astype(np.float32).astype(np.float64)
                    for a in (h, hu, hv, b))
    assert float(h.min()) > 0.3, "champ censé être mouillé partout"
    return h, hu, hv, b


def _refs(h, hu, hv, b):
    """L_ref à Δx=Δy=1 et à Δx=Δy=2, empilés (3, N, N), f64."""
    L1 = np.stack(_rhs_o2(h, hu, hv, b,
                          GridConfig(H=N, W=N, dx=1.0, dy=1.0), DRY_EPS))
    L2 = np.stack(_rhs_o2(h, hu, hv, b,
                          GridConfig(H=N, W=N, dx=2.0, dy=2.0), DRY_EPS))
    return L1, L2


def _divergence_gpu(cp, h, hu, hv, b, inv_dx: float):
    """UN étage du kernel de C1 (w_base = 0, mode 0 réfléchissant, un seul
    dt), à l'inverse de maille demandé. L extrait par D = (q_out − q_in)/dt.

    Appel BRUT du kernel (pas via `pas_f_fidele`) : on veut UN étage, pas un
    RK2 complet — c'est L qu'on compare, pas un pas."""
    q = np.zeros((1, 1, 3, N, N), np.float32)
    q[0, 0, 0], q[0, 0, 1], q[0, 0, 2] = h, hu, hv
    bb = b.astype(np.float32).reshape(1, 1, N, N)
    q_c = cp.ascontiguousarray(cp.asarray(q))
    b_c = cp.ascontiguousarray(cp.asarray(bb))
    sortie = cp.empty_like(q_c)
    total = N * N
    kernel = _obtenir_kernel_fidele(cp)
    grille = ((total + _TAILLE_BLOC - 1) // _TAILLE_BLOC,)
    kernel(grille, (_TAILLE_BLOC,),
           (q_c, q_c, sortie, b_c, q_c, b_c, np.float32(DT),
            np.float32(inv_dx), np.float32(0.0), np.int32(N),
            np.int64(total), np.int32(MODE_REFLECHISSANT)))
    return (cp.asnumpy(sortie)[0, 0].astype(np.float64)
            - np.stack([h, hu, hv])) / DT


# ----- 1. la référence elle-même : L(Δx=1) = 2·L(Δx=2), EXACTEMENT -----

def test_reference_l_dx1_egal_2x_l_dx2():
    """Prédiction pré-écrite de §6, branche commune : Δx n'entre dans
    `_rhs_o2` que par la division de la divergence, donc L(1) = 2·L(2)
    bit à bit (division/multiplication par 2 exactes en IEEE). CPU pur."""
    h, hu, hv, b = _champ_non_trivial()
    L1, L2 = _refs(h, hu, hv, b)
    assert float(np.abs(L1).max()) > 1e-3, \
        "flux nuls : le champ ne stresse rien (aveuglement §3)"
    assert np.array_equal(L1, 2.0 * L2)


# ----- 2. le portage : L_gpu contre les DEUX références -----

@gpu_requis
def test_portage_kernel_contre_rhs_o2():
    """UN étage du kernel figé (w_base = 0, mode 0 réfléchissant, un seul
    dt) sur le même champ que la référence. L extrait par
    D = (q_out − q_in)/dt. Le test affirme la DÉCIDABILITÉ (une référence
    colle à l'arithmétique f32 près, l'autre est à un facteur ~2) et
    IMPRIME la lecture — il ne présuppose aucune branche.

    APRÈS LE CORRECTIF (§A33-CORRECTION) : le kernel prend désormais
    `inv_dx`, et ce test l'appelle à `inv_dx = 1.0`. Son rôle a changé —
    il n'établit plus le défaut (c'est fait, par lui, le 2026-07-25), il
    est le témoin de NON-RÉGRESSION du régime Δx = 1 : celui du témoin, de
    T1 et de la fovéa fine, qui ne devaient RIEN voir passer. Les assertions
    sont inchangées ; err(Δx=1) doit rester au bruit f32 (3.297e-05)."""
    import cupy as cp

    h, hu, hv, b = _champ_non_trivial()
    L1, L2 = _refs(h, hu, hv, b)
    echelle = float(np.abs(L1).max())
    assert echelle > 1e-3
    # garde : aucun clamp (positivité/sec) ne mord après un pas
    assert float((h + DT * L1[0]).min()) > 10.0 * DRY_EPS

    D = _divergence_gpu(cp, h, hu, hv, b, inv_dx=1.0)
    err1 = float(np.abs(D - L1).max()) / echelle
    err2 = float(np.abs(D - L2).max()) / echelle
    lecture = (
        "à inv_dx = 1.0, L_gpu ≡ L_ref(Δx=1) — c'est la LECTURE ATTENDUE "
        "après correctif : le régime Δx = 1 (témoin, T1, fovéa fine) est "
        "inchangé. C'est cette même mesure qui, AVANT correctif et sans "
        "paramètre de maille, avait CONFIRMÉ le soupçon §0."
        if err1 < err2 else
        "L_gpu ≡ L_ref(Δx=2) à inv_dx = 1.0 → le correctif a inversé le "
        "sens du paramètre. RÉGRESSION, à remonter tel quel.")
    print(f"\n[portage _rhs_o2, inv_dx=1.0] err(Δx=1) = {err1:.3e} ; "
          f"err(Δx=2) = {err2:.3e}\n[lecture] {lecture}")

    # DÉCIDABILITÉ : une branche colle, l'autre est loin (facteur ~2 sur L).
    assert min(err1, err2) < 5e-3, (
        f"AUCUNE référence ne colle (err1={err1:.3e}, err2={err2:.3e}) : "
        "troisième fait — le portage diverge de sa référence pour une "
        "autre raison. À remonter, ne rien conclure sur Δx.")
    assert max(err1, err2) > 0.15, (
        f"références indiscernables (err1={err1:.3e}, err2={err2:.3e}) : "
        "le champ ne discrimine pas, test à durcir (flux trop faibles).")


# ----- 3. le CORRECTIF : inv_dx = 0.5 ≡ Δx = 2 (§A33-CORRECTION) -----

@gpu_requis
def test_kernel_a_inv_dx_un_demi_reproduit_rhs_o2_dx2():
    """LE test du correctif D19-c : le kernel devenu Δx-conscient, appelé à
    `inv_dx = 0.5`, reproduit `_rhs_o2(dx = 2)` — la maille du niveau
    GROSSIER de T2. C'est la propriété qui manquait ; sans elle le grossier
    avançait à ~2× sa vitesse physique.

    L'erreur est normalisée par l'échelle de la référence VISÉE (|L2|max),
    pas par celle de L1 : sinon la tolérance se relâcherait d'un facteur 2
    exactement là où on la resserre. Mêmes seuils que le test existant."""
    import cupy as cp

    h, hu, hv, b = _champ_non_trivial()
    L1, L2 = _refs(h, hu, hv, b)
    echelle2 = float(np.abs(L2).max())
    assert echelle2 > 1e-3

    D = _divergence_gpu(cp, h, hu, hv, b, inv_dx=0.5)
    err2 = float(np.abs(D - L2).max()) / echelle2
    err1 = float(np.abs(D - L1).max()) / echelle2
    print(f"\n[correctif Δx, inv_dx=0.5] err(Δx=2) = {err2:.3e} ; "
          f"err(Δx=1) = {err1:.3e}")

    assert err2 < 5e-3, (
        f"le kernel à inv_dx=0.5 ne reproduit PAS _rhs_o2(dx=2) "
        f"(err={err2:.3e}) : le correctif ne fait pas ce qu'il dit.")
    assert err1 > 0.15, (
        f"inv_dx=0.5 est indiscernable de Δx=1 (err={err1:.3e}) : le "
        "paramètre n'est pas consommé — correctif inopérant.")
