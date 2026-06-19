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

L'opérateur global NE TRANSFÈRE PAS proprement : pire holdout = extrap_channel rollout=1.573 vs plancher repr. 0.164 (gap opérateur 1.409), référence in-sample 0.024. Le défaut est l'OPÉRATEUR (pas la base) -> V3b (opérateur conditionné terrain). Inspecter les animations : si l'erreur suit les fronts d'onde (features floutées / en retard de phase), c'est la signature TRANSPORT (limite n-width de l'opérateur linéaire), cf. foresight v1b.

## Signature transport (foresight v1b)

DMD est linéaire ; le transport est là où les opérateurs linéaires souffrent le plus. Si la dégradation est advective (fronts d'onde déplacés / floutés / déphasés), c'est la même limite n-width que le résidu canal de V1, mais dans la dynamique. Voir les animations `outputs/v2/v2_rollout_*.gif` (vérité | prédiction | |erreur|) : l'erreur se concentre-t-elle sur les fronts en mouvement ?

Figure : `outputs/v2/v2_error_growth.png`.
