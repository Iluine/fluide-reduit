# Veille SOTA — Cascade face à l'état de l'art (2026-07-19)

> **Statut : révisé 2026-07-19 après lecture INTÉGRALE du papier PERSIST (v2, arXiv
> 2603.03482, expériences + annexes + limitations).** Les autres travaux ne sont lus
> qu'en résumé. Ce document ne fait foi de rien.
>
> **ERREUR CORRIGÉE, consignée** : la première version de cette veille était fondée sur
> la seule **page projet** de PERSIST — un document promotionnel qui ne mentionne
> **jamais** la vitesse d'inférence. J'y avais surestimé la menace. Le papier donne les
> chiffres, et ils renversent la lecture. Leçon d'instrument, la même que partout
> ailleurs dans ce projet : **une page projet n'est pas un protocole.**

---

## Verdict en une ligne

**Le différenciateur de Cascade n'est pas pris, et le concurrent le plus proche tourne
à 0,37 image par seconde sur A100.** PERSIST ne prétend pas au temps réel, borne sa
mémoire à un cube de 48³ voxels autour du joueur et **jette** tout ce qui en sort, et
n'énonce aucune garantie de déterminisme, de rejouabilité ni de conservation. En
revanche le pilier fovéa-z, lui, est **publié** (et breveté aux US) — c'est là qu'est
le vrai encombrement.

---

## 1. Le différenciateur — « histoire persistante, non contradictoire, à compute borné »

### Le concurrent direct : **PERSIST** (ICML 2026, Edinburgh + Microsoft Research)

*Beyond Pixel Histories: World Models with Persistent 3D State* — Garcin, Walker,
McDonagh, Pearce, Bilen, He, Wang, Bian. `arXiv:2603.03482`, code public, modèles HF.

C'est le plus proche de la thèse Cascade jamais publié, et il est **voxel** : monde
voxel latent régénéré à chaque pas, caméra et renderer séparés, édition du monde
directement en 3D, dynamique non observée émergente.

### Les chiffres que la page projet ne donne pas (papier, Annexe D, Table 14)

> *« All tests were conducted on a single **A100** GPU with a batch size of 1 and 20
> denoising steps. »*

| méthode | ms / frame | FPS |
|---|---|---|
| **PERSIST-S** | **2672.4** | **0.37** |
| Oasis | 3007.4 | 0.33 |
| WorldMem | 6387.1 | 0.16 |

Config accélérée (2 et 4 pas de débruitage) : *« 886 ms (**1.13 FPS**), a 3× improvement »*.
Et l'aveu, en toutes lettres (§7) : *« our implementation was not optimised for
inference speed and **does not yet achieve real-time inference** »*.

> **Comparaison, avec sa réserve honnête.** Cascade V4 : **16.589 ms sur une 3050 Ti**.
> Soit ~53× à ~161× plus rapide *en brut*, sur un GPU dont la bande passante est ~8×
> inférieure — deux à trois ordres de grandeur une fois le matériel normalisé.
> **RÉSERVE QUI COMPTE : leurs 2672 ms produisent l'IMAGE ; nos 16.589 ms ne la
> produisent pas.** La comparaison flatte Cascade de toute la moitié projection qui
> n'existe pas encore. Même en leur concédant l'intégralité d'un budget de 33.3 ms,
> l'écart reste d'un facteur ~80.

### « Fixed-cost memory » : ce que c'est réellement

Un **cube de 48³ voxels centré sur l'agent** (latent 12³×48, patches 4³). Et ce qui en
sort est **jeté** — leur propre limitation, verbatim :

> *« PERSIST presently tracks a finite region of 3D space centred on the agent, causing
> information from distant locations to be **discarded** as the agent moves through the
> environment. To retain spatial information over arbitrarily large environments,
> **future work could consider** a conditioning strategy based on **loading spatial
> chunks from a 3D memory bank**. »*

Autrement dit : ils ont remplacé une borne **temporelle** par une borne **spatiale**, et
présentent cela comme la levée de la borne. Aucune expérience ne teste la ré-entrée
au-delà du cube. **Et le « 3D memory bank » qu'ils appellent de leurs vœux en travaux
futurs, c'est exactement le registre de Cascade — mesuré, lui.**

### Garanties : aucune

Comptage sur le papier complet : `deterministic` **0**, `reproducible` **0**, `replay`
**0**, `invariant` **0**, `conservation` **0**. `guarantee` : une occurrence, sur
l'alignement de w₀ et o₀. `physics` : jamais au sens propre — seul « physical » apparaît
comme **mode d'échec** (*« agent phasing through terrain or floating »*).

Aveu majeur sur leur propre instrument (§6.1) : *« bypassing the learned camera model
introduces **physical inconsistencies that are not captured by the FVD** »*. **Leur
métrique est aveugle à la cohérence physique** — précisément ce que l'instrument maison
(Δχ readout + pin JND) mesure.

### Qualité et coût

Étude utilisateur (28 participants, >800 vidéos, échelle 1–5) : meilleure config
**3.2/5** en fidélité visuelle, **3.0/5** overall — et *« users consistently prefer
ground-truth videos across all metrics, highlighting that even our strongest
configurations **remain below the quality of real trajectories** »*. FVD à 600 frames :
PERSIST 148, PERSIST+w₀ 104, Oasis 875 (WorldMem : **pas de valeur** à 400 ni 600).

Entraînement : ~**460 heures** de jeu, ~100K trajectoires, ~40M interactions ; VAEs
4 et 12 jours sur **8×A100** ; denoisers 3 à 10 jours sur **8×H100** — de l'ordre de
**7 000+ GPU-heures**, avec **vérité-terrain voxel dense + pose caméra** exigée à
l'entraînement (limitation qu'ils nomment en premier). Environnement : **Luanti**
(clone open-source de Minecraft), via Craftium.

### Dérive résiduelle, documentée par eux

Épisode de 2000 pas (83 s à 24 Hz) : tronc d'arbre qui **disparaît** (28 s), blocs
**faussement prédits** (37 s), texture du sol qui **dérive** (54 s), **collision mal
prédite**, agent qui traverse un bloc (69 s). Cause nommée : *exposure bias*, mitigée
par diffusion forcing + 10 % de bruit — **sans aucune ablation chiffrée** de cette
mitigation dans le papier.

> **Lecture stratégique révisée.** PERSIST et Cascade ne sont pas concurrents : ils
> résolvent des **moitiés complémentaires**. PERSIST = projection apprise (image
> générée) sur un état **faible** (borné, stochastique, sans garantie). Cascade = état
> **fort** (non borné, déterministe, auditable) **sans aucune projection**. Le mot
> « persistant » est encombré ; les mots libres restent **« non contradictoire,
> auditable, reconstructible, temps réel »** — et le dernier n'est plus une promesse,
> c'est un écart mesuré de deux ordres de grandeur.
>
> **Le corollaire qui presse** : des gens compétents ont identifié le manque exact que
> Cascade comble, et l'ont écrit noir sur blanc en travaux futurs. La fenêtre n'est pas
> infinie.

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
   mais **non contradictoire, auditable, reconstructible bit-exact — et temps réel**.
   Les artefacts documentés de PERSIST (objets qui disparaissent, collisions mal
   prédites) et leur propre aveu (*« physical inconsistencies not captured by the FVD »*)
   sont la meilleure formulation du problème que Cascade prétend résoudre — à citer.
2. **Rechercher la nouveauté dans le couplage**, pas dans la fovéa : *budget
   d'emplacements borné par l'observation* × *colonne persistante committée à budget
   aire-proportionnel*. Aucun des travaux trouvés ne fait les deux.
3. **Emprunter au lieu de redériver** : `r_fovea` et le cadencement ont des modèles
   perceptuels publiés. Cela peut alléger une campagne gatée sur le chemin critique.
4. **Brevets — point RÉDUIT (Romain, 2026-07-19)** : les deux références trouvées sont
   des brevets **US**, donc territoriaux — sans effet en France. Vérification résiduelle,
   dix minutes le jour où ça compte : la famille a-t-elle des membres **EP** ? Nuance à
   ne pas perdre : l'art. 52 CBE exclut les programmes d'ordinateur *« en tant que
   tels »*, mais l'OEB délivre des brevets d'inventions mises en œuvre par ordinateur
   dès qu'un effet technique est établi, et **G 1/19** (Grande Chambre, 2021, simulation
   de foule piétonne) a jugé que les **simulations par ordinateur** ne sont pas exclues
   par nature. Information, pas conseil juridique. **Ce point n'est pas le sujet** : le
   vrai enjeu de §2 est que le pilier est *publié* — ce qu'aucune juridiction n'efface,
   et qui déplace la nouveauté vers le **couplage**, pas vers la fovéa.
5. **Le son a un champ** : quand il sera appelé, ce sera de l'intégration, pas de
   l'invention.

## 5. Limites de cette veille

Recherche US-only, une seule passe, un seul PDF/page lu intégralement (PERSIST). Les
travaux fovéa-fluide ne sont lus qu'en résumé : leurs **régimes** (VR, fluide seul,
échelles, matériel) ne sont pas vérifiés, donc leur comparabilité au budget 3050 Ti de
Cascade **n'est pas établie**. Aucun chiffre de cette veille ne doit entrer dans un
raisonnement de budget sans lecture des sources.
