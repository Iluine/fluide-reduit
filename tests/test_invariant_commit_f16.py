"""Tranche — INVARIANT n°1 : le résumé de commit se calcule HORS de la précision
de stockage.

Énoncé (annexe §12 n°1 de `claude/seance-table-entree-2026-08-03.md`, gravé au
journal §A51) :

    « f16 permis au stockage vivant ; le chemin de commit calcule son résumé
      HORS de la précision de stockage (f64 hôte) avant d'écrire au ledger. »

Test de propriété exigé : *résumé(champ stocké f16) ≡ moyennes f64 des valeurs
f16, exactement*.

POURQUOI CET INVARIANT MÉRITE UN TEST ET PAS UNE RELECTURE. Le motif est
aujourd'hui celui du code (`src/summary_quadtree.py:77-86`, « COPIE float64
(jamais une vue -- anti-fuite) »), mais il n'existe encore AUCUN chemin de
commit moteur pour en hériter : `_valider_champ` est câblé sur le domaine fixe
64x64 du harnais. Le jour où le commit passe GPU-side pour la performance est le
seul où ce découplage casserait EN SILENCE — une dérive d'accumulation ne lève
aucune exception, elle fait juste maigrir une rivière.

CE QUE LE TEST DISCRIMINE — et c'est le point. Un test qui ne peut pas échouer
ne teste rien : `test_temoin_discriminant` construit explicitement le résultat
qu'une implémentation FAUTIVE (accumulation en f16) produirait, et exige qu'il
DIFFÈRE. Si le témoin cessait de discriminer, ce fichier échouerait au lieu de
donner un feu vert vide.

DÉTECTION VÉRIFIÉE PAR MUTATION (2026-08-03), pas supposée. En rendant
`_valider_champ` fautif — copie sans conversion, le f16 reste f16 —, **5 des 8
tests échouent**. Les trois survivants sont explicables et le sont dans leur
propre docstring : `test_temoin_discriminant` ne passe pas par `summarize_qt`
(il garde le test lui-même) ; le cas `float64` est insensible par construction ;
`test_les_moyennes_sont_en_f64` garde le dtype ÉCRIT AU LEDGER et non le lieu du
calcul — la mutation l'a révélé, il ne gardait pas ce qu'on croyait.

Le champ témoin est déterministe (aucune RNG) : chaque bloc 2x2 porte
(2048, 1, 1, 1). En f16 l'espacement des représentables vaut 2 au voisinage de
2048, si bien qu'une partie des `+1` est absorbée par l'arrondi ; en f64 la
somme vaut exactement 2051 et la moyenne 512.75. La divergence est franche, pas
une histoire d'ULP.

La valeur fautive exacte DÉPEND de la stratégie de sommation et n'est donc pas
codée en dur : numpy somme PAR PAIRES, d'où `(2048+1)+(1+1)` — le 2049 tombe à
mi-chemin entre deux représentables et s'arrondit au pair vers 2048, puis
`2048+2 = 2050`, moyenne 512.5. Une accumulation séquentielle donnerait 512.0.
Le test exige un écart FRANC, pas une valeur : ce qui doit tenir à travers les
versions de numpy, c'est que l'invariant discrimine — pas la façon dont une
implémentation fautive se trompe.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.summary_quadtree import regenerate_qt, size_floats_qt, summarize_qt

_COTE = 64
_BUDGET = 512


def _champ_temoin_f16() -> np.ndarray:
    """Champ 64x64 stocké en f16, dont chaque bloc 2x2 vaut (2048, 1, 1, 1).

    Les deux valeurs sont EXACTEMENT représentables en f16 (2048 = 2**11, 1 =
    2**0) : la quantification du stockage n'introduit aucune erreur ici, et
    toute divergence observée vient donc de l'ACCUMULATION, jamais de
    l'encodage. C'est ce qui rend le témoin lisible."""
    champ = np.ones((_COTE, _COTE), dtype=np.float16)
    champ[0::2, 0::2] = np.float16(2048.0)
    return champ


def _moyennes_blocs(champ: np.ndarray) -> np.ndarray:
    """Une étape de cascade dyadique, dans la précision du tableau reçu.

    Réplique volontairement la forme de `_moyenne_blocs_2x2`
    (`src/summary_quadtree.py:89-96`) : c'est en passant un f16 à CETTE
    fonction qu'on obtient le résultat fautif de référence."""
    h, w = champ.shape
    return champ.reshape(h // 2, 2, w // 2, 2).sum(axis=(1, 3)) / 4.0


def test_temoin_discriminant() -> None:
    """GARDE DU TEST LUI-MÊME : accumuler en f16 doit donner un résultat
    DIFFÉRENT de l'accumulation en f64. Sans cela, les tests suivants
    passeraient aussi sur une implémentation fautive."""
    champ = _champ_temoin_f16()

    fautif = _moyennes_blocs(champ)                      # accumulé en f16
    correct = _moyennes_blocs(champ.astype(np.float64))  # accumulé en f64

    if fautif.dtype != np.float16:
        raise AssertionError(
            "Le calcul 'fautif' de référence n'accumule pas en f16 "
            f"(dtype={fautif.dtype}) : le témoin ne discrimine plus rien.")

    if np.array_equal(fautif.astype(np.float64), correct):
        raise AssertionError(
            "Le témoin ne discrimine plus : accumulation f16 et f64 coïncident "
            "sur ce champ. Le test de l'invariant n°1 donnerait un feu vert "
            "vide — reconstruire un champ témoin avant de faire confiance à ce "
            "fichier.")

    # La moyenne f64 est exacte et indépendante de la stratégie de sommation
    # (2051/4, division par une puissance de 2) : elle, se code en dur.
    assert float(correct[0, 0]) == 512.75

    # L'écart doit être FRANC — un invariant qui ne se distinguerait de sa
    # violation que d'un ULP ne protégerait rien en pratique. La valeur fautive
    # elle-même n'est pas fixée : elle dépend de la stratégie de sommation.
    ecart = abs(float(fautif.astype(np.float64)[0, 0]) - float(correct[0, 0]))
    if ecart < 0.1:
        raise AssertionError(
            f"Écart f16/f64 trop faible ({ecart:g}) pour que ce témoin protège "
            "quoi que ce soit : reconstruire un champ témoin.")


def test_resume_f16_identique_au_resume_f64() -> None:
    """L'INVARIANT : résumer un champ stocké f16 rend exactement le même objet
    que résumer sa conversion f64 explicite — donc la conversion a bien lieu
    AVANT tout calcul, et rien n'est accumulé dans la précision de stockage."""
    champ_f16 = _champ_temoin_f16()

    depuis_f16 = summarize_qt(champ_f16, _BUDGET)
    depuis_f64 = summarize_qt(champ_f16.astype(np.float64), _BUDGET)

    np.testing.assert_array_equal(depuis_f16.topology, depuis_f64.topology)
    # Égalité BIT À BIT, pas une tolérance : l'invariant dit « exactement ».
    np.testing.assert_array_equal(depuis_f16.means, depuis_f64.means)
    assert depuis_f16.shape == depuis_f64.shape == (_COTE, _COTE)
    assert size_floats_qt(depuis_f16) == size_floats_qt(depuis_f64)


def test_les_moyennes_sont_en_f64() -> None:
    """Le résumé écrit au ledger est en f64, quel que soit le dtype d'entrée.

    PORTÉE EXACTE, mesurée et non supposée : ce test NE garde PAS l'invariant
    n°1. Le dtype de sortie est forcé à l'emballage
    (`src/summary_quadtree.py:217` « `means=np.array(moyennes, dtype=np.float64)` »),
    donc EN AVAL du lieu où le calcul se fait — et le test survit à la mutation
    qui casse l'invariant. Ce qu'il garde réellement : qu'un résumé f16 ne soit
    jamais écrit au registre, ce qui rendrait le ledger lui-même porteur de la
    dérive. C'est utile, et c'est autre chose."""
    resume = summarize_qt(_champ_temoin_f16(), _BUDGET)
    assert resume.means.dtype == np.float64


def test_le_resume_temoigne_du_stocke_pas_de_l_original() -> None:
    """Le résumé témoigne des valeurs RÉELLEMENT stockées, il ne rattrape pas la
    quantification.

    C'est la clause 2 de P0-b (`PREREGISTRATION.md:6494` « le VIVANT est
    contracté au TÉMOIGNAGE ») : un vivant f16 ne trahit aucun référent qu'il
    n'a plus. Si le résumé retrouvait le f64 d'origine, il témoignerait d'un
    état que le moteur n'a plus — et le ledger cesserait d'être re-dérivable
    depuis ce que la machine porte vraiment."""
    rng = np.random.default_rng(20260803)
    original_f64 = rng.random((_COTE, _COTE)) * 1000.0
    stocke_f16 = original_f64.astype(np.float16)

    resume_du_stocke = summarize_qt(stocke_f16, _BUDGET)
    resume_de_l_original = summarize_qt(original_f64, _BUDGET)

    if np.array_equal(resume_du_stocke.means, resume_de_l_original.means):
        raise AssertionError(
            "Le résumé du champ f16 coïncide avec celui du f64 d'origine : soit "
            "la quantification est sans effet sur ce champ (témoin à refaire), "
            "soit un chemin rattrape le référent perdu.")

    attendu = summarize_qt(stocke_f16.astype(np.float64), _BUDGET)
    np.testing.assert_array_equal(resume_du_stocke.means, attendu.means)


def test_masse_preservee_sur_le_chemin_f16() -> None:
    """É1 sur le chemin f16 : la reconstruction préserve EXACTEMENT la masse des
    valeurs stockées (propriété de projection du quadtree,
    `PREREGISTRATION.md:3258` « préservés EXACTEMENT »).

    La cascade dyadique divise par 4.0 — exact en IEEE-754 — donc l'égalité est
    testée bit à bit, sans tolérance."""
    champ_f16 = _champ_temoin_f16()
    reference_f64 = champ_f16.astype(np.float64)

    regenere = regenerate_qt(summarize_qt(champ_f16, _BUDGET))

    assert regenere.shape == (_COTE, _COTE)
    assert float(regenere.mean()) == float(reference_f64.mean())


@pytest.mark.parametrize("dtype_stockage", [np.float16, np.float32, np.float64])
def test_invariant_sur_les_trois_precisions(dtype_stockage) -> None:
    """L'invariant ne vise pas f16 en particulier : le résumé se calcule hors de
    la précision de stockage, quelle qu'elle soit. Le jour où un champ vit en
    f32 au large et f16 en fovéa (table §3, ligne 4), les deux passent par le
    même chemin de commit."""
    rng = np.random.default_rng(20260729)
    champ = (rng.random((_COTE, _COTE)) * 1000.0).astype(dtype_stockage)

    resume = summarize_qt(champ, _BUDGET)
    attendu = summarize_qt(champ.astype(np.float64), _BUDGET)

    assert resume.means.dtype == np.float64
    np.testing.assert_array_equal(resume.means, attendu.means)
