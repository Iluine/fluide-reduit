# v1b — Couverture vs capacité sur la topologie canal

## Question

V1 a mesuré un plafond de représentation de **16.4 %** sur le canal, contre < 2 %
en interpolation et sur l'obstacle géométriquement extrapolé. Mais **le canal était
holdout-only** (jamais en entraînement). Ce 16.4 % est-il une limite de
**couverture** (la base n'a aucun mode en forme de bande parce qu'aucun canal n'était
en train) ou de **capacité** (la POD linéaire ne *peut pas* représenter un canal) ?

La preuve de mise en garde est dans les données de V1 elles-mêmes : l'obstacle
d'extrapolation se reconstruit à **1.8 %** alors qu'il est géométriquement hors plage
(σ=3 < [4,7], position hors plage) — **parce que des obstacles étaient en train**.
Donc « topologie vue → extrapolation des paramètres quasi gratuite ». Le canal n'a pas
cette chance.

## Protocole

On reconstruit le **même** canal holdout (wall=1.0, y0=0.5, hw=8, soft=2) avec une
base POD hauteur à laquelle on ajoute progressivement des canaux d'**entraînement**
(paramètres *différents* du holdout, submergés, 2 CI chacun, pour égaler la couverture
de l'obstacle : 4 terrains × 2 CI). On surveille `k`.

Script : `scripts/exp_v1b_channel_coverage.py`.

## Résultat (balayage de couverture)

| canaux en train (× 2 CI) | k | interp | extrap_obstacle | **canal (holdout)** |
|---|---|---|---|---|
| 0 (V1 d'origine) | 32 | 0.6 % | 1.8 % | **16.4 %** |
| 1 | 24 | 0.8 % | 1.9 % | **14.6 %** |
| 2 | 24 | 0.9 % | 2.0 % | **9.3 %** |
| 4 | 25 | 0.9 % | 1.9 % | **6.0 %** |

## Lecture

1. **Dominé par la couverture.** Le plafond canal chute de façon **monotone**
   (16.4 → 6.0 %) avec la couverture, **sans plateau** : ce n'est pas une incapacité
   de la POD, la base apprend des modes-bande dès qu'on lui en montre.
2. **Pas « gratuit une fois couvert ».** À couverture **égale** à l'obstacle
   (4 × 2 CI), le canal reste à **6.0 %** vs **~1.9 %** pour l'obstacle — un facteur
   ~3 résiduel. La topologie *bande* coûte modestement plus cher à une base de modes
   localisés que la topologie *tache*. C'est une constante, pas un mur.
3. **`k` n'explose pas** — il reste 24–32 (il *baisse* même : la structure canal, à
   forte variance cohérente, concentre l'énergie). C'est le signal décisif du gate V5 :
   couvrir une nouvelle topologie n'explose pas la base linéaire.

   (Note : ajouter des canaux fait monter très légèrement les plafonds interp/obstacle,
   0.6 → 0.9 % et 1.8 → 1.9 % — compétition pour les modes à seuil d'énergie fixe.
   Négligeable ; toutes les topologies *vues* restent < 2 %.)

## Conséquences sur le routage v2

- **La représentation n'est pas le goulot** dès que le train couvre le vocabulaire de
  topologies (modulo une petite constante topologie-dépendante). Pour un livrable de
  jeu, on **contrôle** ce vocabulaire → la généralisation *dans* chaque topologie est
  bon marché, et la seule chose qu'une base statique ne sait pas faire (une topologie
  jamais anticipée) un jeu n'en a pas besoin.
- **V3a (mettre `b` en canal de la base) est le mauvais levier** : il corrigerait un
  problème de représentation qui n'existe pas. La vraie question ouverte de la v2 est
  la **dynamique** : même là où la représentation est gratuite (bosses, obstacles), un
  opérateur DMD **global** prédit-il le rollout sur un terrain nouveau, alors que la
  bathymétrie entre dans la dynamique comme **terme source** ? → **V2**. Si le rollout
  échoue, ce sera l'opérateur → **V3b (opérateur conditionné terrain)**, pas V3a.
- **Gate V5 (encodeur appris) reformulé** : il ne se justifie **pas** par « plafond
  canal = 16 % ». Il se justifie par « **`k` explose** quand on essaie de couvrir tout
  le vocabulaire de topologies avec une base linéaire ». Quantité à suivre = croissance
  de `k`, pas le plafond d'une seule topologie jamais vue. Ici `k` reste modeste → la
  base statique tient, pas d'encodeur appris pour le but visuel.

## Verdict

**On reste sur base statique.** V1 + v1b montrent que la représentation tient dès que
le train couvre le vocabulaire (k ne grossit pas). La substance réelle de la v2 est la
**dynamique** : prochain pas = **V2** (transfert DMD naïf sur terrain nouveau), pour
voir si le gap dynamique suit — ou non — le coût de représentation.
