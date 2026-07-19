# Veille SOTA — Cascade face à l'état de l'art (2026-07-19)

> **Statut : première passe, à confiance graduée.** Recherche web (US-only), lectures
> d'abstracts et d'une page projet complète (PERSIST). Ce document ne fait foi de rien :
> il dit ce qui existe, ce qui est *pris*, ce qui reste ouvert, et ce que le projet peut
> **emprunter** au lieu de le redériver. Une revue sérieuse exigerait les PDF lus.

---

## Verdict en une ligne

**Le différenciateur de Cascade n'est pas pris — mais son cadrage l'est.** Et le pilier
fovéa-z, lui, est **publié ET breveté**. Le projet a dépensé sa journée dans la moitié
la plus encombrée du domaine, alors que son actif différenciant est déjà en banque.

---

## 1. Le différenciateur — « histoire persistante, non contradictoire, à compute borné »

### Le concurrent direct : **PERSIST** (ICML 2026, Edinburgh + Microsoft Research)

*Beyond Pixel Histories: World Models with Persistent 3D State* — Garcin, Walker,
McDonagh, Pearce, Bilen, He, Wang, Bian. `arXiv:2603.03482`, code public, modèles HF.

C'est le plus proche de la thèse Cascade jamais publié, et il est **voxel** :
- monde **voxel latent** régénéré à chaque pas, caméra et renderer séparés ;
- mémoire spatiale **à coût fixe** — explicitement contre les fenêtres de contexte
  pixel et contre le key-frame retrieval (« sans archive toujours croissante ») ;
- stabilité sur **des milliers de frames** là où les comparables cassent après
  quelques centaines ; édition du monde **directement en 3D** en cours de génération ;
- dynamique non observée **émergente** (une grotte s'inonde hors du champ).

**Ce qu'il ne fait PAS — et c'est exactement la ligne de Cascade.** Leur propre épisode
vitrine de 2000 frames (83 s) documente : un tronc d'arbre **disparaît** de la
représentation 3D ; des blocs de bois **apparaissent** spontanément ; la texture du sol
**dérive** vers le jaune ; une collision est **mal prédite** et l'agent traverse un bloc.
Ils nomment la cause : *autoregressive drift*, non corrigée (post-training non fait).

> **Lecture stratégique.** PERSIST livre une persistance **statistique avec récupération
> gracieuse**. Cascade vise la **non-contradiction comme garantie dure** : É3 bit-exact,
> 0 violation sur les contrôles, corruption **détectée et localisée à 100 %**, Δχ = 0.0
> exact sur le bras `ferm`, replay identique. Ce sont deux natures de garantie
> différentes. Le mot « persistant » est désormais **encombré** ; le mot qui reste libre
> est **« non contradictoire, auditable, reconstructible »**.

Deux autres différences structurelles à garder : PERSIST **génère** (mêmes conditions,
graines différentes ⇒ mondes différents, il *outpaint* le non-vu) là où Cascade
**calcule** depuis (seed, registre) ; et PERSIST exige des **annotations 3D à
l'entraînement** (limitation qu'ils nomment).

### Le reste du champ

`Matrix-Game 3.0` (mémoire long-horizon, streaming temps réel), `WorldPlay` (cohérence
géométrique long terme), `WorldScape` (diffusion autorégressive, cohérence 3D), Genie 2.
Tous convergent vers le même mot d'ordre — *long-horizon spatiotemporal consistency* —
et tous butent, d'après la littérature de synthèse, sur : « les objets disparaissent,
la physique dérive, la logique de jeu est fragile sur les sessions longues ».

**Conclusion §1 : la case « histoire non contradictoire par construction » est LIBRE.
Le cadrage marketing « persistant / long-horizon », lui, est saturé.**

---

## 2. Le pilier fovéa-z — **PUBLIÉ, et BREVETÉ** ⚠️

C'est le résultat le plus inconfortable de cette veille.

| travail | ce qu'il fait |
|---|---|
| *Foveated Fluid Animation in VR* (SJTU, IEEE VR ~2024) | simulation de fluide pilotée par le regard, détail des éclaboussures au point de fixation |
| *Scene-Based Foveated Fluid Animation in VR* (IEEE TVCG, ~2026) | **modèle perceptuel dépendant de l'excentricité ET de la courbure**, allocation dynamique des degrés de liberté, stabilité spatio-temporelle |
| *Temporal Foveated Fluid Animation in VR* (IEEE) | **foveation TEMPORELLE** — exactement la question de cadencement portée en dette σ_ω |
| *A Perceptual Model for Eccentricity-dependent Spatio-temporal Flicker Fusion* (arXiv 2104.13514) | modèle perceptuel excentricité × fréquence temporelle, appliqué au foveated graphics |
| **US 10740951 / US 10339692** | **brevets** : « Foveal adaptation of **particles and simulation models** in a foveated rendering system » |

**Ce que ça change pour Cascade :**

1. **Le pilier n'est pas nouveau.** « La distance/excentricité fixe un plafond de LOD de
   la *simulation*, pas seulement du rendu » est publié et industriellement revendiqué.
   La nouveauté de Cascade doit donc être cherchée ailleurs : dans le **couplage**
   fovéa × registre committé (le budget d'emplacements *et* la colonne persistante),
   pas dans la fovéa seule.
2. **Les brevets sont à connaître, pas à découvrir plus tard.** Portée à vérifier
   sérieusement (revendications, juridictions, applicabilité à un moteur non-VR).
3. **Bonne nouvelle opérationnelle, et elle est grosse** : `r_fovea` — le pin gaté,
   load-bearing, jamais mesuré — a de la **littérature avec modèles perceptuels
   d'excentricité déjà ajustés**. La campagne humaine gatée pourrait être *ancrée sur
   littérature* avec un falsificateur maison, au lieu d'être reconstruite depuis zéro.
   Idem pour la question de **cadencement** (foveation temporelle publiée) — la dette
   σ_ω a un corpus.

---

## 3. Les autres pièces — toutes établies séparément

- **Multirésolution ondelettes/Harten sur GPU** : solveurs adaptatifs multi-GPU établis
  en CFD (TUM/Springer ; grilles adaptées par ondelettes multicore/multi-GPU), taux de
  compression > 95 % pour < 1 % d'erreur. La numérique de `F` sur niveaux n'est pas un
  terrain neuf — le régime **temps réel jeu** l'est davantage.
- **Critère perceptuel plutôt que L2** : posture établie en graphics (métriques 3D JND
  dès 2005 ; *PP-Motion* 2025 combine supervision physique **et** perceptuelle ; études
  de perception de la viscosité des fluides). L'instrument maison (Δχ readout + ABX
  staircase) est propre au projet ; le **principe** ne l'est pas.
- **Replay déterministe event-sourced** : standard en systèmes distribués, et rejoué en
  2026 côté agents (*The Log is the Agent*, `arXiv:2605.21997` : replay déterministe,
  fork sur préfixe partagé, audit, validation stricte). « Le fichier de sauvegarde EST
  le ledger » n'est pas neuf. Ce qui l'est : le commit **avec perte, à budget borné**,
  dont on a mesuré qu'il tient le readout **sous JND** à travers les replays.
- **Son depuis la simulation** : champ **mature** — synthèse modale temps réel
  (Microsoft Research), `SYMPHONY` (UNC GAMMA) pour grands environnements, `DiffSound`
  (2024, modal différentiable), `PyImpact`/ThreeDWorld. La moitié son du projet n'a pas
  à être inventée : elle a une littérature à emprunter.

---

## 4. Ce que cette veille change — proposé, non décidé

1. **Affûter la claim.** Cesser d'opposer Cascade aux « modèles vidéo génératifs » en
   général : le comparable est PERSIST, et la ligne de démarcation n'est pas *persistant*
   mais **non contradictoire, auditable, reconstructible bit-exact**. Les artefacts
   documentés de PERSIST (objets qui disparaissent, collisions mal prédites) sont la
   meilleure formulation du problème que Cascade prétend résoudre — à citer.
2. **Rechercher la nouveauté dans le couplage**, pas dans la fovéa : *budget
   d'emplacements borné par l'observation* × *colonne persistante committée à budget
   aire-proportionnel*. Aucun des travaux trouvés ne fait les deux.
3. **Emprunter au lieu de redériver** : `r_fovea` et le cadencement ont des modèles
   perceptuels publiés. Cela peut alléger une campagne gatée sur le chemin critique.
4. **Vérifier les brevets** (US 10740951 / 10339692) avant d'aller plus loin sur le
   pilier fovéa — portée réelle, pas panique.
5. **Le son a un champ** : quand il sera appelé, ce sera de l'intégration, pas de
   l'invention.

## 5. Limites de cette veille

Recherche US-only, une seule passe, un seul PDF/page lu intégralement (PERSIST). Les
travaux fovéa-fluide ne sont lus qu'en résumé : leurs **régimes** (VR, fluide seul,
échelles, matériel) ne sont pas vérifiés, donc leur comparabilité au budget 3050 Ti de
Cascade **n'est pas établie**. Aucun chiffre de cette veille ne doit entrer dans un
raisonnement de budget sans lecture des sources.
