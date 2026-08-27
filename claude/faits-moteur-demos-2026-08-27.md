# Quatre faits de moteur, sortis des démos v2→v4 (27/08/2026)

Fichier de FAITS, pas de clôture. Régime §A64-4 : rien ici n'est un verdict ni
une décision, le journal `pocCascade2phys/PREREGISTRATION.md` reste intouché.
Ce qui est écrit ici a été mesuré pour ~25 s de compute chacun, en construisant
`run_demo_2a1_v2/v3/v4.py` (commits `463f71c`, `373f86f`, `2e3a926`). Chaque
fait porte SON STATUT : mesuré, ou attribué et non établi.

## 1. `n_fov / 2` est le plafond de ce qu'un écran peut montrer — MESURÉ

L'écran du gather est une fenêtre en coordonnées du niveau FIN ; la fenêtre la
plus large est celle du niveau GPU le plus grossier, le niveau 1, qui couvre
`n_fov / 2` cellules de monde. **Aucun écran ne peut montrer davantage, quels
que soient `n0` et `cote_px`.** C'était la cause STRUCTURELLE de la petitesse
de la v2 (16 cellules sur 128), pas un défaut de composition de scène.

COROLLAIRE, ET IL A COÛTÉ UN RUN : `n_fov` plafonne aussi la DURÉE d'un run.
Les origines en `x` suivent le centre fovéal, celles en `y` sont FIXES ;
l'écran glisse hors des fenêtres à raison d'un pixel fin par tick. Rétrécir
l'écran n'achète qu'un délai linéaire et maigre — rupture aux ticks
128 / 144 / 160 / 176 / 192 pour `cote_px` 384 / 352 / 320 / 288 / 256, à
`n_fov = 256`. Le seul vrai levier est `n_fov` : 384 tient 230 ticks.

À retenir pour le dimensionnement : **`n_fov` règle DEUX plafonds à la fois**,
combien de monde on voit et combien de temps on peut le balayer.

## 2. `src/sediment.py` n'est pas conservatif — donc interdit de `z` en l'état

MESURÉ. `_exner_step` est une SOURCE-PUITS locale —
`ds/dt = 1[wet]·(k_d·h·1[θ<θc]·(1−θ/θc) − k_e·s·1[θ>θc]·(θ/θc−1))` — et non la
divergence d'un flux : rien n'est transporté d'une cellule à l'autre, la
matière érodée ne va NULLE PART. Sur la scène de la v4, `Σs` passe de 1 536 à
660 en 120 ticks : **−57 %, monotone.**

CONSÉQUENCE SUR LE VOCABULAIRE DU MOTEUR (Romain, ce jour) : « comptable sur
les quantités conservées » est un pilier non négociable, et c'est le seul des
trois que cette loi viole. Pour une démo, sans conséquence — le champ est
régénéré et regardé, jamais audité. **Le jour où le sédiment entre dans `z`
comme physique de `F`, il lui faudra un transport EN FORME DE FLUX.** C'est une
contrainte sur le vocabulaire, découverte pour le prix d'une démo.

DEUX AUTRES PROPRIÉTÉS DE CETTE LOI, MESURÉES, qui bornent ce qu'une écriture
persistante peut être avec elle :

- **Elle n'érode que ce qu'elle a déposé, jamais la roche** (terme d'érosion
  ∝ `s`). Depuis un lit nu : ZÉRO cellule creusée, à tous les ticks. Il faut
  poser un manteau meuble pour que le creusement existe.
- **L'épaisseur du manteau s'annule du motif.** `s` décroît exponentiellement,
  donc `s/s₀` ne dépend pas de `s₀` : à 0,06 / 0,12 / 0,20, la fraction raclée
  jusqu'à la roche vaut 35,1 / 34,4 / 38,1 %, et le creusement maximal vaut
  EXACTEMENT le manteau à chaque fois.

⇒ Le vocabulaire actuel des écritures persistantes est **« redistribution
bornée par le manteau posé »**. Pas de canyons : la roche est inentamable par
construction, et la profondeur creusée est réglée par ce qu'on a posé, pas par
l'écoulement.

## 3. Le peigne de stries appartient au front sec du solveur

ÉTABLI — origine. Des stries rectilignes alignées sur les axes naissent au
front mouillé/sec vers le tick 60. Deux causes candidates ont été testées et
ÉCARTÉES : la cadence du lit (intégrer Exner un tick sur 16 laisse la structure
en place, rugosité 0,0134 → 0,0077) et le couplage lui-même (un TÉMOIN à lit
GELÉ produit la même rugosité d'eau, 0,0463 contre 0,0538 au tick 120). Le
peigne est donc du SOLVEUR, sur pente faible et long horizon — **latent depuis
la v3**, où il n'avait que moins de temps pour se développer.

MESURÉ — période. Sur les pixels du rendu : **période 8 px = 2 cellules fines
exactement**, harmonique à 16. C'est la fréquence maximale que la grille du
solveur peut porter.

ATTRIBUÉ, NON ÉTABLI — mécanisme. La période 2 est la signature d'un
découplage PAIR/IMPAIR au seuil de séchage : sous une lame mince, le flux entre
cellules quasi sèches est coupé et des rangées alternées restent
mouillées/sèches. Cohérent avec tout ce qui précède, mais **aucune expérience
contrôlée ne l'a isolé** : la statistique d'alternance mesurée sur le champ
plafonne à 0,54 dans la zone striée là où un mode pair/impair PUR donnerait
1,00. Le mécanisme est donc une hypothèse de travail, pas un fait.

MESURÉ — et c'est le fait de moteur. **Le lit est plus strié que l'eau qui le
strie** : taux d'alternance 0,232 dans l'eau du témoin à lit gelé, 0,536 dans
l'eau couplée, **0,574 dans le lit**. Une écriture persistante INTÈGRE le bruit
transitoire du substrat : ce qui n'était qu'un scintillement régénéré et oublié
à chaque frame devient une cicatrice permanente.

⇒ **Les écritures persistantes élèvent l'exigence perceptuelle du substrat.**
Le régime le plus permissif de la doctrine (régénéré-et-oublié) cesse de
s'appliquer dès que le monde se souvient. Rien à réparer aujourd'hui : le
défaut est borné et compris, et une chirurgie de solveur n'a aucune décision
moteur au bout. Deux portes le jour venu, et le choix sera PERCEPTUEL, pas
numérique — désingularisation de lame mince au solveur, ou masquage/tramage au
readout.

## 4. Loi de rectitude (Romain, 27/08) — première brique du vocabulaire perceptuel

**La rectitude, dans l'eau, est une signature de CONTRAINTE, jamais
d'écoulement.** L'œil ne rejette pas les lignes droites : il exige qu'elles
aient un contraigneur VISIBLE. Un canal, un quai, une digue — la droite est
lue comme héritée de la géométrie, et acceptée. Les stries de la v4 choquent
parce qu'elles sont droites SANS cause visible : l'œil cherche le mur qui les
expliquerait, ne le trouve pas, et conclut à la faute. La grille du solveur est
un contraigneur réel mais invisible dans le monde — c'est la définition même
d'un artefact.

TROIS CONSÉQUENCES OPÉRATOIRES :

1. **Elle précise le geste de readout.** Il ne s'agit pas d'éliminer la
   rectitude, mais d'empêcher l'écoulement d'en produire là où rien de visible
   ne la contraint.
2. **Elle donne un détecteur quasi gratuit** : cohérence rectiligne dans une
   zone d'eau non adjacente à un élément de terrain = drapeau d'artefact. Un
   lint perceptuel sur les sorties de démo, sans harnais ABX, sans JND, sans
   mesure d'instrument — juste des pixels.
3. **Elle porte son garde-fou** : Cascade est un monde VOXEL, plein de droites
   légitimes, et l'eau qu'elles contraignent aura le droit d'être droite. Le
   critère n'est jamais « pas de lignes » — c'est **« pas de lignes
   orphelines »**. Un détecteur naïf accuserait chaque canal du jeu.

STATUT : loi perceptuelle énoncée par Romain à la lecture de la vidéo v4, sur
un jugement d'œil immédiat. Elle n'est pas mesurée et n'attend pas de l'être
pour servir de critère de conception ; elle attend le jour où quelque chose en
dépendra.
