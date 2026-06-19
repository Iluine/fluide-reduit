# v2b — Couverture-opérateur : un A global suffit-il une fois la topologie vue ?

## Question

V2 a montré un **cliff** : rollout du canal holdout à **157 %** avec un opérateur DMD
global ajusté sur bosses+obstacles seuls — le canal était holdout-only **dans le fit
DMD aussi**. v1b avait montré, côté représentation, que couvrir la topologie effondre le
plafond (16 → 6 %). Question analogue, côté **dynamique** : si on met des canaux dans le
fit DMD (et dans la base, sinon le plancher de représentation cape le rollout), le
rollout du canal chute-t-il, ou un seul opérateur global est-il fondamentalement
incapable de porter bosse *et* canal (→ besoin de conditionnement V3b) ?

Le canal holdout (wall=1.0, y0=0.5, hw=8, soft=2) garde des paramètres **différents**
des canaux d'entraînement (param-extrapolation dans la topologie canal).

Script : `scripts/exp_v2b_operator_coverage.py`.

## Résultat (balayage de couverture-opérateur)

| canaux dans le fit (×2 CI) | k | ρ | train_ref | interp | obstacle | **canal rollout** | canal floor |
|---|---|---|---|---|---|---|---|
| 0 | 32 | 1.0 | 2.4 % | 4.6 % | 8.8 % | **157 %** | 16.4 % |
| 1 | 24 | 1.0 | 2.5 % | 4.0 % | 7.1 % | **36.4 %** | 14.6 % |
| 2 | 24 | 1.0 | 2.5 % | 4.1 % | 7.1 % | **17.2 %** | 9.3 % |
| 4 | 25 | 1.0 | 2.4 % | 4.0 % | 8.6 % | **11.8 %** | 6.0 % |

## Décomposition (n=4) : gap opérateur = rollout − plancher

| terrain | rollout | plancher (repr.) | gap opérateur |
|---|---|---|---|
| train_ref (in-sample) | 2.4 % | 0.04 % | 2.4 % |
| interp | 4.0 % | 0.6 % | 3.4 % |
| **canal** | 11.8 % | **6.0 %** | **5.8 %** |
| obstacle | 8.6 % | 1.8 % | 6.8 % |

## Lecture

1. **L'opérateur est *coverage-limited*, pas incapable.** Le cliff 157 % → 11.8 %
   s'effondre dès qu'un seul A global voit de la dynamique de canal — exactement
   l'analogue dynamique de v1b. Pas de plateau-au-catastrophique.
2. **Le gap opérateur du canal (5.8 %) ≈ celui de l'obstacle (6.8 %).** L'opérateur ne
   traite pas la dynamique de *bande* plus mal que celle de *tache* une fois couverte ;
   il est quasi **topologie-agnostique**. Le rollout canal plus élevé (11.8 %) est
   **hérité de son plancher de représentation (6 %)** = le résidu transport/n-width de
   v1b, qui vit dans la **base**, pas dans l'opérateur.
3. **`k` reste modeste (24–32), ρ=1.0** : pas d'explosion, base + opérateur statiques
   stables. Et les topologies **vues ne se dégradent pas** quand on ajoute des canaux
   (interp 4.6 → 4.0 %, obstacle stable) : un seul A porte tout le vocabulaire couvert.

## Verdict (résout la question centrale de la v2)

**On reste sur base statique + UN opérateur DMD global.** Dès que le train **couvre le
vocabulaire de topologies** (représentation *et* dynamique), le surrogate réduit reste
**plausible et borné** sur terrain nouveau — gap opérateur ~2–7 %, comparable à travers
les topologies. Pour un livrable de jeu, on **contrôle** ce vocabulaire.

- **V3b (opérateur conditionné terrain) n'est PAS requis** pour le but visuel : un A
  global suffit une fois la topologie vue.
- **V3a non plus** (v1b : pas de problème de représentation à corriger).
- **V5 (encodeur appris) non plus** : `k` ne grossit pas.
- La foresight transport était juste sur le *lieu* : le résidu transport est dans la
  **représentation** (plancher ~6 %, n-width), **pas** un déficit d'opérateur. Il
  redeviendrait dominant dans un régime à **forte translation** (positions de features
  largement variables) — c'est CE régime, pas la nouveauté topologique, qui justifierait
  un encodeur non-linéaire. Pour un vocabulaire à positions bornées, on est en deçà.

## Frontière restante

Hors du régime **submergé / réfraction / vocabulaire couvert & borné en translation** :
le sec / les îles (sillages, séparation → solveur mouillé/sec, v2.5) et les régimes à
forte translation (où la n-width linéaire mordrait → encodeur appris). Tout cela reste
hors-scope, et désormais **caractérisé** plutôt que supposé.
