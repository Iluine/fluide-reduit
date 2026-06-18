# Spécification v2 — Généralisation du surrogate réduit à des terrains/CI nouveaux

> Suite du POC `shallow-water-rom` (tag `v1` sur `main`).
> Commanditaire : dev senior qui **relit et audite** (ne produit pas le Python).
> Code idiomatique, lisible, contrats de données explicites, invariants vérifiables.
> **Discipline héritée du POC : un seul pas décisif, diagnostic bon marché AVANT de
> construire quoi que ce soit.** Ne pas packager le front entier.

---

## 1. Contexte — ce que le POC a établi, ce que la v2 teste

Le POC a montré que, **sur un seul terrain fixe**, un modèle d'ordre réduit
(POD + DMD écrêté) reproduit le champ de **hauteur** de façon plausible et stable
(erreur ~5–7 %), masse conservée (projection M5), couture multirésolution gérable.
Il a aussi mesuré un plafond de représentation des **vitesses** sur CI non vues
(~30 %) — d'où le report de M4.

La v2 attaque l'axe que le POC n'a **jamais testé** : la généralisation à des
**terrains et CI nouveaux**, en restant sur un **livrable visuel** (on rend h et
η = h + b ; les vitesses restent invisibles, donc hors sujet ici).

## 2. Hypothèse à valider (unique)

**Le surrogate réduit de hauteur reste plausible et borné sur un terrain qu'il n'a
jamais vu.**

Distinction essentielle à instrumenter dès le départ (sinon on sur-vend) :

- **Interpolation** : terrain nouveau *à l'intérieur* de la plage de paramètres
  d'entraînement. Attendu plus facile.
- **Extrapolation** : terrain *hors* de la plage. Attendu plus dur, possiblement
  un mur.

Le succès se juge en **plausibilité visuelle** sur terrain nouveau (pas seulement
en L2) : pour un jeu, le critère est « ça ressemble à de l'eau crédible sur ce
relief », pas « ça colle au solveur ».

## 3. Contraintes

- **Frugal** : laptop, minutes. Réutilise au maximum le POC (solver, POD,
  DMD écrêté, projection M5, métriques, rendu). N'ajoute que le strict nécessaire.
- **Dépendances** : numpy/scipy/matplotlib pour le cœur ; `pytorch` (GPU CUDA
  ~4 Go ou CPU) **seulement si** l'escalade vers un encodeur appris est déclenchée
  (V5, hors-scope par défaut). Demande avant d'ajouter autre chose.
- **Diagnostic d'abord** : faire V1 (plafond de représentation inter-terrain)
  AVANT de construire le moindre conditionnement. V1 dit si le problème est
  représentation ou dynamique, et donc quoi construire.
- **Pas de régression** : le cas mono-terrain du POC (tag `v1`) doit rester intact.

## 4. Architecture — réutilisé vs nouveau

Réutilisé tel quel : `src/solver.py`, `src/pod.py`, `src/dmd.py`
(`clip_eigenvalues`), `src/mass_projection.py`, `src/metrics.py`, `src/render.py`.

Nouveau :
- `src/terrains.py` : famille de terrains paramétrée + tirage train/holdout.
- `src/conditioning.py` : conditionnement minimal (V3) — bathymétrie en canal,
  puis opérateur conditionné terrain.
- scripts `run_v0..v4`.

## 5. Contrats de données

- Terrain : `b` de forme `(H, W)`, accompagné d'un **vecteur de paramètres**
  `theta` (ex. `[hauteur_bosse, x0, y0, largeur, pente, ...]`) documenté.
- Jeu de données : `data/v2/<terrain_id>__<ic_id>.npz` contenant `h,u,v,b,theta`
  + métadonnées (dx, dt, schéma, CFL). Split déclaré dans `data/v2/split.json`
  (train / holdout_interp / holdout_extrap).

## 6. Jalons

### V0 — Famille de terrains (données)
- **Quoi** : paramétrer une famille de terrains (ex. bosses gaussiennes de
  hauteur/position/largeur variables, pente d'ensemble, un canal, un obstacle).
  Générer la vérité via le solveur existant : N terrains d'entraînement × M CI.
- **Holdout** : (a) un terrain **dans** la plage de params (interpolation),
  (b) un terrain **hors** plage (extrapolation), chacun avec des CI nouvelles.
- **Validation** : masse conservée par le solveur (~2e-16) sur tous les terrains ;
  split sauvegardé et reproductible.

### V1 — Diagnostic décisif : plafond de représentation inter-terrain — **prioritaire**
- **Quoi** : base POD calculée sur les terrains d'entraînement ; encode-décode les
  champs de hauteur **vrais** des terrains holdout (interp + extrap). **Aucune
  dynamique.**
- **Sortie** : erreur de reconstruction de h par terrain holdout (interp vs extrap),
  vs le cas train comme référence.
- **Validation** : tranche la question. Plafond bas (<~10 %) → la base statique
  tient le terrain nouveau, il « suffit » de conditionner la dynamique. Plafond
  haut (>~30 %, surtout en extrap) → la base statique ne span pas le terrain
  nouveau → conditionnement de la représentation nécessaire (et possiblement
  encodeur appris, V5). *C'est l'analogue exact du diagnostic de plafond vitesse
  du POC.*

### V2 — Transfert naïf (baseline)
- **Quoi** : base statique + DMD écrêté global (fit sur tous les terrains
  d'entraînement) → rollout sur terrain holdout → erreur hauteur + plausibilité.
- **Pourquoi** : montrer si/à quel point l'approche POC transfère « telle quelle ».
  L'opérateur global ignore que la bathymétrie change la dynamique (terme source) ;
  attends-toi à une dégradation. Établit le gap à combler.

### V3 — Conditionnement minimal (escalade graduée, s'arrêter dès que le gap ferme)
- **V3a — terrain dans la base** : ajouter `b` comme canal dans la POD pour que la
  base « voie » le contexte de relief. Cheap. Réévaluer V2.
- **V3b — opérateur conditionné terrain** : faire dépendre l'opérateur latent de
  descripteurs de terrain (ex. `theta` en entrée d'un opérateur conditionné, ou
  quelques opérateurs DMD par cluster de terrain, blendés par proximité de
  `theta`). Réévaluer.
- **Réutilise** M5 (projection masse) et le rendu sur terrain nouveau.
- **Gate** : si V3a/b ramènent la hauteur à « plausible et bornée » sur terrain
  nouveau → objectif visuel atteint, stop. Sinon → V5 (hors-scope par défaut).

### V4 — Évaluation inter-terrain (interp vs extrap) — **prioritaire**
- **Quoi** : erreur hauteur + dérive masse + **plausibilité visuelle** sur terrain
  jamais vu, en distinguant interpolation et extrapolation.
- **Sortie** : rendus côte à côte vérité | prédiction sur terrain nouveau (h et η),
  courbes d'erreur interp vs extrap, + un jugement de plausibilité documenté.
- **Validation** : caractérise *jusqu'où* ça généralise — succès attendu en interp,
  extrapolation caractérisée (pas forcément réussie, et c'est un résultat honnête).

## 7. Critères de succès

- **V1** : plafond de représentation inter-terrain chiffré (interp vs extrap). ✅/❌.
- **Hauteur** sur terrain nouveau : bornée et **visuellement plausible** au moins en
  interpolation ; extrapolation caractérisée.
- **Masse** : projection M5 réappliquée, dérive ~0.
- **Non-régression** : cas mono-terrain du POC (tag `v1`) inchangé.

## 8. Discipline diagnostic-first (rappel)

Faire **V1 avant V3**. V1 dit si le défaut est *représentation* (la base ne span pas
le terrain nouveau → conditionner/encodeur) ou seulement *dynamique* (la base span
mais l'opérateur global rate → conditionner l'opérateur suffit). Ne construis le
conditionnement qu'en sachant lequel des deux tu attaques.

## 9. Decision gate — escalade vers un encodeur appris (V5, hors-scope par défaut)

Si le plafond V1 en extrapolation est élevé (>~X %, à fixer après V1) **ou** si
V3a/b ne ferment pas le gap visuel → un **encodeur conditionné appris**
(autoencodeur conditionné sur `theta`, base non-linéaire) devient justifié. C'est le
premier vrai pas vers les « bases apprises » de la vision plus large, et c'est là
que le GPU CUDA (~4 Go, AMP) sert. **À spécifier comme une v2.5 séparée, pas à
empiler ici.**

## 10. Hors scope v2 (ne pas implémenter)

- Vitesses / fidélité physique (toujours invisibles au livrable).
- 3D / voxel ; portage temps-réel Rust/Burn ; multi-physique / couplage.
- Encodeur conditionné appris (renvoyé à V5/v2.5, sauf si V1/V3 le forcent — et
  alors dans une spec dédiée).

## 11. Notes pour l'agent (Claude Code)

- Construis incrémentalement, montre la figure/chiffre de chaque jalon avant le
  suivant. **V1 d'abord** : c'est le pas décisif, il oriente tout le reste.
- Réutilise les modules du POC sans les réécrire ; n'ajoute que `terrains.py`,
  `conditioning.py` et les scripts `run_v*`.
- Distingue toujours interpolation et extrapolation dans les métriques et les
  figures — c'est ce qui garde l'histoire calibrée.
- Garde la surface de dépendances minimale ; pas de `pytorch` avant V5.

---

## Annexe A — Décisions de design V0/V1 (validées avant implémentation)

Concrétisation des « ex. » de la spec, arbitrée avant d'écrire le plan. Principe
directeur retenu : **la valeur d'un diagnostic est bornée par la difficulté de ce
qu'il teste** — une famille trop lisse rendrait V1 incapable d'échouer, donc
informativement vide (faux positif « ça généralise »).

### A.1 Famille de terrains (`src/terrains.py`)

Deux topologies (frugal : une structure de plus que la bosse, pas trois), chacune
paramétrée, toutes deux bâties sur la machinerie gaussienne déjà présente dans le
POC (`make_terrain`/`initial_condition_gaussian_drop`) :

- **(A) bosse + pente** — `b = amp·exp(−r²/2σ²) + slope·x/(W−1)`.
  Déformation douce ; le `"bump"` du POC (amp=0.4, centre (0.5,0.5), σ≈10.7,
  slope=0) en est un point → non-régression triviale.
- **(B) obstacle** — gaussienne **étroite et haute** (σ petit, amplitude grande) :
  l'eau doit le **contourner** → sillage/séparation, écoulement qualitativement
  hors de l'enveloppe linéaire de (A). C'est ce qui met réellement la base statique
  à l'épreuve.

Descripteur `theta` (vecteur, pour le conditionnement V3b ultérieur) — champ
`kind ∈ {bump, obstacle, channel}` + params continus `[amp, x0_frac, y0_frac,
sigma, slope]`. `b` est dérivé de `theta` de façon déterministe.

### A.2 Plages d'entraînement (définissent « dedans »)

| param | plage train |
|---|---|
| amp (bump) | [0.2, 0.5] |
| amp (obstacle) | [0.6, 1.0] |
| (x0, y0)_frac | [0.4, 0.6]² |
| sigma (bump) | [8, 13] |
| sigma (obstacle) | [4, 7] |
| slope | [0, 0.01] |

### A.3 Tirage & split

- **Train** : ~9 terrains répartis **sur les deux topologies** (Latin-hypercube /
  tirage déterministe par graine), pas 9 variantes de la même bosse douce.
  × 2 CI (`drop_center`, `drop_offset`) ≈ 18 trajectoires. Grille 64×64, solveur
  inchangé (n_steps=800 → 201 frames).
- **Holdout interp** : un terrain (bump ou obstacle) à `theta` strictement dans les
  plages mais non tiré, CI nouvelle.
- **Holdout extrap (deux régimes, pour que l'extrapolation soit un vrai stress)** :
  1. **obstacle extrême** — σ proche de l'échelle de grille / amplitude au-delà de
     la plage (famille vue, params hors plage) ;
  2. **canal** — topologie **absente du train** (`b = mur·1[|y−y0| > w]` ou
     équivalent), générée uniquement en holdout. Extrapolation la plus honnête :
     la base ne l'a littéralement jamais vue. Coût quasi nul (un seul terrain).
- Split figé et reproductible dans `data/v2/split.json`.

### A.4 V1 — objet et sortie

- POD sur tous les snapshots de **hauteur** des trajectoires train (h **seul** —
  vitesses hors-scope §3). `k` au seuil énergie **0.9999** (cohérent POC).
- Encode-décode le h **vrai** de chaque holdout. Sortie : erreur L2 relative par
  holdout, avec la reconstruction train comme référence.
- V1 départage **trois** régimes : `interp` | `extrap-mêmes-familles` (obstacle
  extrême) | `extrap-topologie-nouvelle` (canal). Figure barres + champ résiduel.

### A.5 Périmètre du premier plan

Le plan d'implémentation initial couvre **V0 + V1 uniquement** (la colonne
diagnostique). V2–V4 ne sont pas détaillés tant que le chiffre de V1 n'est pas
connu : le contenu de V3 (conditionner la *représentation* vs l'*opérateur*) dépend
directement du verdict de V1. Détailler V3 maintenant serait du placeholder. On
montre le chiffre de V1, puis on planifie la suite en connaissance de cause.
