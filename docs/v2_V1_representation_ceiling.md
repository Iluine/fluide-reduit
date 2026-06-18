# V1 — Plafond de représentation inter-terrain (hauteur)

Base POD hauteur seule (n_channels=1), seuil d'énergie 0.9999.

- **k = 32** (énergie cumulée 0.999904). Pour mémoire, le POC mono-terrain donnait k≈43 ; un k plus grand ici est attendu et mesure la complexité accrue de la famille.
- Erreur de reconstruction train (plancher in-sample) : 0.0004.

## Plafond par régime (erreur L2 relative de h)

| régime | err | err_max (par frame) |
|---|---|---|
| interp | 0.0058 | 0.0070 |
| extrap_obstacle | 0.0178 | 0.0201 |
| extrap_channel | 0.1637 | 0.1723 |

## Verdict

Plafond INTERMÉDIAIRE (10–30% en extrapolation) : la base tient l'interpolation mais peine en extrapolation -> évaluer V2, puis conditionner la représentation (V3a : b en canal) si nécessaire. Signal porteur : extrap_channel (topologie jamais vue) = 0.1637 ; ne pas conclure 'ça généralise' sur les seuls cas de réfraction douce (interp/extrap_obstacle).

## Portée du résultat (calibrage du ✅)

Tous les terrains sont SUBMERGÉS : la dépendance au terrain passe par la réfraction (contraste de célérité ~1.7×, réel mais modéré). Un V1 qui passe certifie la généralisation DANS le régime submergé/réfraction — pas sur tout terrain de jeu. Le sec / les îles (sillages, séparation) sont un régime distinct plus dur, reporté en v2.5 (solveur mouillé/sec positivity-preserving & well-balanced).

Figure : `outputs/v2/v1_representation_ceiling.png`.