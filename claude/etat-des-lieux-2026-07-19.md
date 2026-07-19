# Cascade — état des lieux au 2026-07-19

> Document de situation, pas de décision. Objet : empêcher qu'une ligne de mesure
> devienne sa propre fin, en montrant d'un coup d'œil ce qui est MESURÉ, ce qui est
> GATÉ, ce qui est DETTE, et ce qui n'a **jamais été touché**. Les chiffres viennent
> du journal `pocCascade2phys/PREREGISTRATION.md` (source de vérité) ; ce document
> ne fait foi de rien, il rend visible.

---

## L'objectif, en trois phrases

Cascade est un moteur physique de monde voxel pour un jeu sandbox, dont la thèse est
qu'un **seul état du monde `z` est la source de vérité unique** — l'image *et le son*
n'en étant que des projections perceptuelles co-égales, jamais des sorties générées
indépendamment. Le différenciateur face aux modèles vidéo génératifs n'est pas la
qualité d'une frame : c'est **l'histoire persistante et non contradictoire à compute
borné** — la plausibilité par-frame est tractable, la cohérence à long horizon est le
problème non résolu. Le critère de succès est **perceptuel, jamais L2**.

Piliers : un opérateur `F` unique sur les niveaux de Harten ; la distance fixe un
plafond de LOD, l'énergie raffine sous ce plafond ; la magie comme transducteur
d'énergie conservée ; la frontière éphémère/persistant où **seul le détail réinjecté
paie la taxe du bit-exact** ; la **fovéa-z** qui borne `L_eff` par l'observation au
lieu de le laisser croître avec le monde.

---

## 1. MESURÉ — ce qui est acquis, avec sa portée

### 1.1 Le différenciateur lui-même — l'acquis le plus solide

| fait | chiffre | portée |
|---|---|---|
| Registre fermé (verdict §A13) | k\*_chaîne ≤ **256 floats** (6.25 % du champ fin), tous Δt ∈ {1,4,16} | v2-sédiment, pin n=1, cette famille de politique |
| Contrôles de manche 2 | **0 violation** — ferm Δχ = 0.0 exact ; shuf supra-JND ; corruption localisée 100 % ; É3 double-run bit-exact | idem |
| Ledger en production | **1.17 Ko/commit**, 4.2 Mo pour 3600 commits | chaîne fenêtrée (k_fen=64) |
| Rechargement | **0.063 s** pour 1 h de ledger (7200 entrées, tous sha256 relus) | segment courant nul ; pire cas calculé ~2.9 s |
| Compaction | écart avec/sans snapshot **0.011 s** — la compaction achète du disque, **pas du load** | claim E7 confirmée par son propre diagnostic |
| Fenêtrage (§A14) | k_fen=64 sur 32² tient comme k\*=256 sur 64² ; max 0.046 = **63 % du pin** | 1 seed, 1 fenêtre fixe |

**Ce que ça veut dire** : le fichier de sauvegarde *est* le ledger, le monde se recharge
par re-dérivation, et le coût par émission est indépendant de la longueur de
l'historique. C'est exactement ce qu'un modèle génératif ne sait pas faire — et c'est
mesuré, pas argumenté.

### 1.2 Le pin perceptuel

`jnd_sev` = **7.33 %**, IC [6.03, 8.67] ; laxiste 11.54 % [9.37, 13.87]. Ancré en
cycles/degré, budget compresseur 32, n=1 (Romain), calibration écran gravée.
Consommateurs servis : §A12, §A13, §A14, et tous les seuils F1.

**Précision qui porte tout le reste** : le pin est mesuré sur le **READOUT
D'INSTRUMENT** (albedo + delta_chi en bandes porteuses) — un appareil construit pour
COMPARER, pas un rendu destiné à être vu. Le transfert de ce pin vers l'image que le
joueur verra réellement est la dette **albédo→luminance** (§C4-3), aujourd'hui
infalsifiable faute de rendu. Tous les seuils du projet reposent dessus.
**Atténuation nommée (2026-07-19)** : le rendu visé étant une **fonction déterministe
de `z`** (optique appliquée à l'état, §4), readout d'instrument et rendu ne sont pas
deux mondes séparés — le pin pourra se transporter par CONSTRUCTION plutôt que par
hypothèse, la dette rétrécit d'autant. Elle ne disparaît qu'une fois le rendu écrit.

### 1.3 Le moteur (F1) — le modèle de coût, mesuré terme à terme

| terme | chiffre | note |
|---|---|---|
| Ancre F | **6.43 ns/cellule/frame** (c=8, fusionné) | slot 512² : 1.685 ms (2 sys) / **0.845** (1 sys, batché) |
| Linéarité en systèmes | vraie **en batché** (+0.2 %), fausse isolée (+14.8 %) | s1 — l'étiquette « structurelle » remplacée par deux mesures |
| Orchestration | batchée = celle du moteur ; les +6.11 ms du per-niveau étaient de la **plomberie** | s2, modèle re-calibré à **+0.6 %** |
| Prédiction GPU-side emboîtée | **≈ 0 ms — gratuite** | M-a-quater ; « effondrement des stalls » confirmé contre le modèle linéaire |
| Remontée (L3, kernel) | 5.122 ms brut ; **1.280 amortie k=4** (prédit 1.2805 — 4 chiffres) | ×1.66 sur le CuPy (8.523) |
| Transferts | **2.727 ms** ; schéma diff 12.74 Mo < plein 16.78 ; pas de fuite d'échelle (ratio 0.997) | 99.4 % du trafic = remontée seuillée |
| **Frame V4 complète** | **médiane 16.589 ms** (budget 16.7) — **p99 19.652** | marge **0.111 ms = 0.7 %**, *inférieure à la dérive machine du jour (1.1 %)* |
| Résidence | 0.311 Go (gate 1.35) | la note VRAM est corroborée |

### 1.4 Deux morts prononcées (des résultats, pas des échecs)

**MORT-a naïve** : 703 ms contre 16.7 (×42) — aucune cellule du scan ne tenait.
**MORT-a architecturale** : le fusionné à 5.056 ms/niveau contre 1.856 pré-écrit (×2.7),
malgré un gain ×15.6 sur le naïf. Le cap anti-tapis-roulant a été honoré : aucune
escalade au-delà. Ce qui est mort : **le quadruplet majorant** (c=8 dense, n_fov=512,
N_niv=10) sur 3050 Ti — pas la fovéa-z, pas Cascade.

### 1.5 Instrument (faits à ne pas re-découvrir)

Readout **propre** (test à blanc Δχ = 0 exact) ; machine-instrument = iluin-tworings3
natif seul (la VM Cowork est bit-identique au gel mais tue à 45 s ; elle voit le GPU,
dev/tests seulement) ; le cloud **n'est pas** l'instrument (AVX512 au-dessus du plafond
machine) ; CuPy 14.1.1 / numpy 2.4.6 épinglé ; empreinte kernel F stable sur 10 commits ;
dérive machine du jour **−1.1 %**.

---

## 2. GATÉ — mesurable, mais volontairement pas encore

| objet | condition de déblocage | état |
|---|---|---|
| **F0-cloud** | session cloud (~2 min) | **DÛ — gate (ii) de la spec, traîne depuis le 18** |
| **M-b / tranche-2** | tranche-1 + M-a-quater sans mort ✓ ; gate d'instrument dégagé ✓ | **achetable** ; chiffrage suspendu à la lecture de l'attribution (b) |
| **r_fovea** (JND d'excentricité) | verdict tranche-moteur sans mort | garde anti-tapis-roulant tenue — load-bearing (S_eff 1.8–3.9, règle D4) |
| v1.1 multi-vue | sonde deux-fenêtres concurrentes (F3) | non armée |
| v2 3D | sonde 3D minimale (F4) | non armée |
| τ_relax post-hoc | achat du cadencement-relaxation (F5) | non armée |

---

## 3. DETTES — nommées, avec condition de réveil

- **σ_ω (axe temporel)** : réveil quand la spec **décide le cadencement**, la question
  perceptuelle en main (branche a″ préférée). *Effleurée deux fois aujourd'hui* (choix
  de k pour L1 ; balayage de k dans la sonde EPS) — et le réveil a été **refusé
  explicitement** les deux fois. Aucun réveil silencieux à ce jour.
- **EPS_DETAIL = 1e-4** : `[NON-ANCRÉ perceptuel]`. Rouvert par la branche de D-1(b) ;
  nouveau pré-enregistrement dû, dont l'observable dépend de l'attribution (b) en cours.
  Enjeu chiffré : ~2.5 ms de transferts, quand V4 ne tient qu'à 0.111 ms.
- **W1 / 128-vs-256** : mort-par-défaut (décision (d)) ; réanimation = décision neuve.
- **Transfert albédo→luminance** : hypothèse héritée (§C4-3), dormante avec σ_ω.
- **p99 = 19.652 ms** : le stutter de *cadence* a été supprimé (L1 étalée), celui des
  **bursts d'émission** ne l'est pas. Non résolu, non minimisé.

---

## 4. JAMAIS TOUCHÉ — les pieds non mesurés

Aucun n'est crédité d'avance ; **tous ajoutent du coût, aucun n'en retire.**

**Les deux premiers sont la MOITIÉ « PROJECTION » de la thèse** — les readouts
co-égaux par lesquels `z` devient perceptible. Elle n'a jamais été construite.

**Nature de cette moitié, précisée (Romain, 2026-07-19)** : chez Cascade, image et son
sont des **conséquences déterministes de `z`** — optique et acoustique appliquées à
l'état, pas génération. C'est l'opposé de l'approche PERSIST, dont le shader neuronal
*« can learn arbitrary rendering functions »* et prédit *« information not provided by
3D latents (texture, lighting, particle effects…) »* — d'où leur dérive de texture
**alors que leur état 3D reste stable** (leur pas 1296). Une projection optique ne peut
pas dériver ainsi : `z` stable ⇒ image stable, par construction. Conséquences :
- **la faisabilité n'est pas la question** — c'est de l'ingénierie connue, pas de la
  recherche ; **le COÛT l'est entièrement, et il n'est mesuré NI pour l'optique NI pour
  l'auditif** ;
- l'image devient **falsifiable contre l'état** (« ce readout est-il la projection
  correcte de `z` ? ») — question que PERSIST ne peut structurellement pas poser,
  son état étant lui aussi généré ;
- la dette **albédo→luminance rétrécit** : si le rendu est une fonction déterministe de
  `z`, le pin se transporte par CONSTRUCTION au lieu de par hypothèse (§1.2).

1. **Le SON.** Co-égal à l'image dans la thèse fondatrice. Zéro mesure, zéro spec,
   zéro ligne. Rien ne l'a jamais gaté : il n'a simplement jamais été appelé. Son coût
   n'est pas plus mesuré que celui de l'image.
2. **L'IMAGE RENDUE.** *N'existe pas non plus.* Ce qui existe est un **readout
   d'instrument** (albedo + delta_chi), fait pour comparer — pas pour être vu. Le rendu
   réel doit projeter une pyramide de Harten creuse avec fovéa, lointain venu du
   grossier CPU : ce n'est pas une rastérisation ordinaire, et ça occupe *l'autre
   moitié* de la frame que le motif de T2 réserve. **V4 à 16.589 ms signifie donc
   « 30 fps avec ~16.7 ms pour un rendu jamais chiffré ».** « Le moteur tient » reste
   hors de portée tant que cette moitié est vide.
3. **Le substrat-jeu réel.** Tout le budget moteur est mesuré sur un **proxy**
   (v2-sédiment, `c=8` enveloppe de travail). Le contrôle T1 (devenu **verdictal**)
   existe précisément pour ça — mais si le v-jeu est plus lourd, tout se rejoue.
4. **Le niveau 0 CPU à 500k cellules** : le harnais en mesure 65k — ×7.6 non mesuré.
5. **Le calcul 3D** : la note VRAM borne la *mémoire* (~64³) ; le **calcul** n'a aucune
   ancre.
6. **Les fenêtres d'énergie comme ressource de gameplay** : budgétées dans V4 (2 slots),
   jamais exercées par un contenu réel.
7. **Le cadencement temporel** : « 1 pas de F par fenêtre par frame » est une hypothèse
   de design, jamais mesurée (CFL locale ⇒ sous-pas par niveau).
8. **Multi-observateur / réseau** (v1.1), **game design** (hors spec par construction).

---

## 5. Chemin critique, honnête

```
F0-cloud (2 min, gate ii)                        ← dû, le moins cher de tous
attribution (b) : remontée pleine non incr.      ← minutes, en cours de build
  └─ détermine l'observable du nouveau prereg EPS
nouveau prereg EPS + re-mesure                   ← ~0.5–1 séance, levier ~2.5 ms
M-b / tranche-2 (portage fidèle)                 ← 3–5 séances, risque ÉLEVÉ, gate (iii)
  └─ décide Option A (quarantaine du non-déterminisme) ou repli Option B
r_fovea                                          ← débloqué après
```

## 6. Ce que je surveille, en tant que critique

**Le rythme.** Douze mesures et sondes dans la journée. La discipline a tenu — caps
honorés, points d'arrêt respectés, garde anti-tapis-roulant exercée une fois, trois
concessions consignées (dont deux de la session critique). Mais la *ligne* s'allonge,
et chaque pas nouveau a été justifié par le pas précédent. C'est exactement la forme
que prend un tapis roulant au niveau macro.

**Le proxy.** V4 tient à 0,7 % sur un substrat qui n'est pas le jeu, avec p99 non résolu
et huit pieds non mesurés qui tous coûtent. Dire « le moteur tient » aujourd'hui serait
surclamer ; ce qui est établi est : *le moteur tient au seuil, sur un proxy, à cette
enveloppe, sur cette machine.*

**Le déséquilibre — corrigé le 2026-07-19 après remarque de Romain.** Le premier cadrage
(« image mesurée, son absent ») était FAUX : **l'image rendue n'existe pas davantage que
le son.** Ce qui existe est un readout d'INSTRUMENT, fait pour comparer, pas pour être vu.
La thèse tient `z` pour source unique et l'image *et* le son pour projections co-égales :
la moitié **ÉTAT** (z, F, registre, fovéa, budget) est mesurée, chiffrée, gatée,
disciplinée — la moitié **PROJECTION** n'a jamais été construite, ni pour l'image, ni pour
le son. Et c'est sur elle que reposent tous les seuils, via le pin (§1.2) et sa dette
albédo→luminance, aujourd'hui infalsifiable.

Conséquence sur ce qu'on a le droit de dire : « le moteur tient » est hors de portée tant
que la moitié affichage du budget T2 est vide. Ce qui est établi reste : *la physique
tient au seuil, sur un proxy, à cette enveloppe, sur cette machine, le rendu non compté.*
