# V2 — Transfert naïf (opérateur DMD global) sur terrain nouveau

Base POD hauteur (k=32) + opérateur DMD global écrêté (ρ=1.0000) ajusté sur tous les terrains d'entraînement, déroulé sur les terrains holdout depuis leur CI vraie.

Décomposition : plancher = erreur de représentation (encode-décode du h vrai) ; rollout = erreur depuis la CI via l'opérateur ; gap = rollout − plancher = erreur imputable à l'OPÉRATEUR.

| terrain | plancher (repr.) | rollout | rollout_max | gap (opérateur) |
|---|---|---|---|---|
| train_ref (train_bump0/drop_center) | 0.0004 | 0.0242 | 0.0368 | 0.0239 |
| interp | 0.0058 | 0.0461 | 0.0611 | 0.0403 |
| extrap_obstacle | 0.0178 | 0.0883 | 0.1229 | 0.0706 |
| extrap_channel | 0.1637 | 1.5725 | 2.2869 | 1.4089 |

## Verdict

L'opérateur global ne transfère pas sur le pire holdout (extrap_channel rollout=1.573 vs plancher 0.164, gap 1.409 ; in-sample 0.024). MAIS ce cliff est sur une topologie HOLDOUT-ONLY (absente du fit DMD). Le test couverture-opérateur (v2b, docs/v2_v2b_operator_coverage.md) montre qu'il est de la COUVERTURE (157%->11.8% en mettant des canaux au fit, gap opérateur alors ~comparable à l'obstacle) -> un A GLOBAL suffit une fois la topologie vue, V3b NON requis ; le résidu est le plancher de représentation (n-width, v1b), pas l'opérateur.

## Signature transport (foresight v1b)

DMD est linéaire ; le transport est là où les opérateurs linéaires souffrent le plus. Si la dégradation est advective (fronts d'onde déplacés / floutés / déphasés), c'est la même limite n-width que le résidu canal de V1, mais dans la dynamique. Voir les animations `outputs/v2/v2_rollout_*.gif` (vérité | prédiction | |erreur|) : l'erreur se concentre-t-elle sur les fronts en mouvement ?

## Couverture-opérateur (v2b)

Le cliff canal ci-dessus est mesuré topologie HOLDOUT-ONLY (absente du fit DMD). v2b (`docs/v2_v2b_operator_coverage.md`) montre qu'il est dominé par la COUVERTURE : 157 % → 11.8 % en mettant des canaux au fit, gap opérateur alors comparable à l'obstacle → un A global suffit une fois la topologie vue, **V3b non requis**. Le résidu (~6 %) est le plancher de représentation (n-width, v1b), pas l'opérateur.

Figure : `outputs/v2/v2_error_growth.png`.
