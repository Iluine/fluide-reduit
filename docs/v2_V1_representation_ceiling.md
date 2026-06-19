# V1 — Plafond de représentation inter-terrain (hauteur)

Base POD hauteur seule (n_channels=1), seuil d'énergie 0.9999.

- **k = 32** (énergie cumulée 0.999904). Ce k n'est pas directement comparable au k≈43 du POC mono-terrain : le POC comptait les modes pour l'état à 3 canaux [h, u, v] (les vitesses ajoutent des modes), alors que cette base V1 est HAUTEUR SEULE (n_channels=1). k=32 sur 9 terrains est modeste et indique que le champ de hauteur dans le régime submergé lisse est intrinsèquement bas-dimensionnel — cohérent avec les faibles plafonds interp/extrap_obstacle (un k élevé aurait signalé la base statique en difficulté, ce qui n'est pas le cas pour ces régimes).
- Erreur de reconstruction train (plancher in-sample) : 0.0004.

## Plafond par régime (erreur L2 relative de h)

| régime | err | err_max (par frame) |
|---|---|---|
| interp | 0.0058 | 0.0070 |
| extrap_obstacle | 0.0178 | 0.0201 |
| extrap_channel | 0.1637 | 0.1723 |

## Verdict

Plafond INTERMÉDIAIRE (10–30% en extrapolation). ATTENTION au routage : le pire régime extrap (extrap_channel) est une topologie HOLDOUT-ONLY, jamais en train -> probablement une limite de COUVERTURE et non de capacité (l'obstacle, lui VU en train, s'extrapole à <2% malgré une géométrie hors plage). Test décisif avant tout conditionnement de représentation : ajouter la topologie au train et re-mesurer (fait — cf. docs/v2_v1b_coverage_vs_capacity.md : plafond canal 16.4%->6.0%, k stable 24-32 => COUVERTURE). La vraie question est la DYNAMIQUE (V2) ; si le rollout rate, V3b (opérateur conditionné terrain), PAS V3a. Signal porteur : extrap_channel (topologie jamais vue) = 0.1637 ; ne pas conclure 'ça généralise' sur les seuls cas de réfraction douce (interp/extrap_obstacle).

## Portée du résultat (calibrage du ✅)

Tous les terrains sont SUBMERGÉS : la dépendance au terrain passe par la réfraction (contraste de célérité ~1.7×, réel mais modéré). Un V1 qui passe certifie la généralisation DANS le régime submergé/réfraction — pas sur tout terrain de jeu. Le sec / les îles (sillages, séparation) sont un régime distinct plus dur, reporté en v2.5 (solveur mouillé/sec positivity-preserving & well-balanced).

## Suite — couverture vs capacité (v1b)

Le plafond extrap_channel ci-dessus est mesuré topologie HOLDOUT-ONLY. L'expérience v1b (`docs/v2_v1b_coverage_vs_capacity.md`) montre qu'il est dominé par la COUVERTURE (16.4 % → 6.0 % en ajoutant des canaux au train, `k` stable 24–32, sans plateau) : la base statique tient, la suite est la dynamique (V2, puis V3b si l'opérateur rate), pas V3a.

Figure : `outputs/v2/v1_representation_ceiling.png`.
