"""Arc C / Task 2 — sujet SYNTHÉTIQUE (observateur simulé à seuil connu),
utilisé EXCLUSIVEMENT pour valider la logique pure de l'escalier ABX
(`src/arcC_abx.py`) en test. **AUCUN sujet humain ici** -- les vraies
sessions sont Task 3, où `repondre` est branché sur un humain via la
coquille interactive (`scripts/run_arcC_session.py`).

Modèle psychométrique logistique standard (choix binaire A/B, taux de
devinette `g=0.5`) :

    P(correct | Δχ) = g + (1-g) / (1 + exp(-(Δχ - x0)/σ))

`x0` est le CENTRE de la logistique -- PAS directement `theta_sim` -- décalé
de façon à ce que `P(correct | theta_sim) == 1/√2` (≈ 70.71 %) EXACTEMENT :
c'est le point de convergence CANONIQUE d'un escalier 2-down-1-up (Levitt,
1971 -- 2 descentes pour 1 montée convergent vers le niveau où P(correct) =
2^(-1/2)), PAS le point milieu à 75 % d'une logistique symétrique naïve
centrée en `theta_sim` (qui donnerait P(theta_sim)=0.75, incohérent avec ce
que l'escalier est censé retrouver). Décision nommée (cf. rapport Task 2) :
sans ce décalage, le test-clé « l'escalier retrouve `theta_sim` » testerait
une grandeur qui n'est PAS ce que 2-down-1-up estime réellement."""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from src.arcC_abx import EssaiPropose, Reponse, autre_reponse

# Point de convergence canonique d'un escalier 2-down-1-up (Levitt 1971) :
# 2 descentes consécutives pour 1 montée <=> convergence vers P(correct) =
# 2^(-1/2) ≈ 0.70710678.
CIBLE_CONVERGENCE_2DOWN1UP: float = 1.0 / math.sqrt(2.0)


def probabilite_correcte(delta_chi: float, theta_sim: float, sigma: float,
                         taux_devinette: float = 0.5,
                         cible_convergence: float = CIBLE_CONVERGENCE_2DOWN1UP) -> float:
    """P(correct | delta_chi) du sujet synthétique de seuil `theta_sim` (au
    sens : `probabilite_correcte(theta_sim, theta_sim, sigma) ==
    cible_convergence`, EXACTEMENT, par construction du décalage `x0` -- cf.
    docstring module) et de pente `sigma` (échelle de transition, en Δχ).

    Dérivation de `x0` (`P(theta_sim) = cible_convergence`) :
        (cible - g)/(1-g) = 1/(1+exp(-(theta_sim-x0)/sigma))
        exp(-(theta_sim-x0)/sigma) = (1-g)/(cible-g) - 1 = (1-cible)/(cible-g)
        x0 = theta_sim + sigma * ln((1-cible)/(cible-g))
    """
    if not (taux_devinette < cible_convergence < 1.0):
        raise ValueError(
            f"probabilite_correcte : cible_convergence={cible_convergence!r} doit être "
            f"strictement entre taux_devinette={taux_devinette!r} et 1.0.")
    if sigma <= 0.0:
        raise ValueError(f"probabilite_correcte : sigma doit être > 0 (reçu {sigma!r}).")
    ratio = (1.0 - cible_convergence) / (cible_convergence - taux_devinette)
    x0 = theta_sim + sigma * math.log(ratio)
    z = (delta_chi - x0) / sigma
    return taux_devinette + (1.0 - taux_devinette) / (1.0 + math.exp(-z))


def fabrique_sujet_synthetique(theta_sim: float, sigma: float, seed: int,
                               taux_devinette: float = 0.5,
                               cible_convergence: float = CIBLE_CONVERGENCE_2DOWN1UP
                               ) -> Callable[[EssaiPropose], Reponse]:
    """Fabrique un sujet synthétique `repondre(essai) -> "A"|"B"` : RNG SEEDÉE
    (`seed`), consommée dans l'ORDRE des appels (un tirage uniforme par essai)
    -- déterminisme strict. Répond CORRECTEMENT avec probabilité
    `probabilite_correcte(essai.delta_chi, theta_sim, sigma, ...)`, sinon
    répond l'autre position. Le sujet ne regarde QUE `essai.delta_chi` --
    ignore volontairement le reste de la métadonnée (numéro de staircase,
    champ source, etc.), comme un humain ne verrait que les deux stimuli à
    l'écran, jamais leur provenance."""
    rng = np.random.default_rng(seed)

    def repondre(essai: EssaiPropose) -> Reponse:
        p = probabilite_correcte(essai.delta_chi, theta_sim, sigma,
                                 taux_devinette=taux_devinette,
                                 cible_convergence=cible_convergence)
        u = float(rng.uniform())
        if u < p:
            return essai.reponse_correcte
        return autre_reponse(essai.reponse_correcte)

    return repondre
