# Chiffrage — M-b (gate iii), l'écart live↔rederive avec le F FIDÈLE

**Gate d'achat. Aucune ligne de code avant endossement.**
Source : pocCascade2phys PREREGISTRATION.md §A24 (commit d66ec0b). Gate (i)
satisfait (V4 sans mort, médiane 14,490 < 16,7). M-b est le gate (iii).

> **En une phrase** : M-b n'est pas une sonde de coût de plus — c'est le
> premier build de la cascade dont l'objet EST un **moteur** (le F fidèle),
> et non un motif de coût dont le design est fixé ailleurs. C'est de là que
> vient tout le risque, et c'est ce qui justifie une découpe interne.

---

## 1. Ce que M-b mesure, et pourquoi c'est le F fidèle qui est acheté

| | |
|---|---|
| **LIVE** | le pipeline V4 **complet** — géométrie emboîtée, prédiction GPU-side, L3, L1 k=4 étalée, `EPS_PRODUCTION` = 1e-2 — mais avec le **F FIDÈLE** au lieu du F jetable |
| **REDERIVE** | la physique CPU f64 **INTOUCHÉE** : `run_episode` (`src/sediment.py`) → pulse → `simulate_wetdry_o2` → intégration Exner |
| **OBSERVABLE** | Δχ readout (`albedo` + `delta_chi["max_carrier"]`, `src/albedo.py`), primitives EXISTANTES, réutilisées |
| **PROTOCOLE** | 3 seeds {101, 102, 103}, Δt = 4, **6 émissions** ; MORT-b = max de série Δχ **> 0,0733 PAR-SEED** |
| **CONTRÔLE T1** | **VERDICTAL** : frame complète avec F fidèle **> 16,7 ms ⇒ V4 mort** |

Le tout le reste de la cascade F1 a mesuré des **motifs de coût** dont le
design vivait ailleurs : le fusionné réordonne un stencil sans toucher à la
physique ; L3 émet des détails en épilogue ; la prédiction GPU-side déplace
un calcul spécifié en §5. Le F jetable, lui, est explicitement une borne :
son docstring dit **« bathymétrie plate, dt figé, réduction CFL payée non
consommée, contenu jetable seedé »**.

Le F fidèle est autre chose. Le rederive `run_episode` fait, par épisode :
un **pulse** gaussien, **600 pas** de `simulate_wetdry_o2` (wetdry O2
CFL-**adaptatif**, MUSCL surface-gradient bien équilibré, murs réfléchissants,
HLL dry-aware) sur `b_eff = b0 + s` avec **bathymétrie réelle** non plate,
puis **~300 pas Exner** (dépôt/érosion sur cellules mouillées). Porter cela
sur GPU dans la structure V4, et le rendre assez fidèle au rederive f64 pour
que Δχ reste sous le JND, **est un moteur** — pas une mesure.

---

## 2. Le budget T1, chiffré — et le point où il peut casser

Le jetable V4 à 1e-2 tient **14,490 ms** (médiane). T1 borne à 16,7. Le F
fidèle dispose donc de **2,199 ms de marge** pour ce qu'il ajoute au-dessus
du jetable, PAR FRAME :

| Ce que le fidèle ajoute au jetable | Ordre de grandeur | Commentaire |
|---|---|---|
| Bathymétrie réelle (corrections hydrostatiques non nulles) | faible | le kernel les CALCULE déjà — mais à b plat elles sont nulles ; à b réel elles exercent un chemin **non vérifié** |
| dt CFL-**adaptatif** (consommer la réduction déjà payée) | faible→moyen | la réduction est payée ; consommer le dt est bon marché, MAIS un dt physique < budget de frame force des **sous-pas** — c'est là que le pire cas gonfle |
| Exner (dépôt/érosion) aux snapshots | moyen | ne tourne pas à chaque frame (save_every) — les frames-snapshot sont plus lourdes |
| Pulse (init d'épisode) | faible | une frame par épisode |

**Le risque T1 est le pire cas, pas la médiane.** Rappel §A23 : la **p99 du
jetable à 1e-2 est déjà 16,853 ms** — au-dessus de 16,7. Si T1 se lit sur la
médiane (cohérent avec la mort T2), le fidèle a 2,199 ms de marge médiane et
passe plausiblement. Si T1 se lit sur la frame la PLUS chère (dt-adaptatif à
sous-pas, ou frame-Exner), la p99 jetable est déjà par-dessus et le fidèle la
pousse plus loin. **À trancher avant le build : T1 sur médiane ou sur pire
frame ?** Je le remonte parce que le verdict de V4-b en dépend directement, et
que le gravé dit « frame complète > 16,7 » sans lever l'ambiguïté médiane/p99.

---

## 3. Composants, LOC, séances

Références de calibration (fichiers existants) : driver M-a-quater 771,
borne L3 404, kernel fusionné 302 CUDA-équiv, kernel L3 439, solver wetdry
CPU 448, sediment CPU 562. Séances récentes : 800–1300 LOC.

### C1 — Portage du F fidèle (le moteur)

Le wetdry O2 fidèle à `_rhs_o2` sur GPU, à bathymétrie réelle et dt
adaptatif, dans la structure fenêtrée V4. Le kernel figé sert d'échafaud (on
reprend son préfixe de fonctions device, comme L3 l'a fait), mais trois
choses sont neuves et non triviales : (a) la bathymétrie non plate exerce les
corrections hydrostatiques jamais vérifiées ; (b) le dt adaptatif consomme la
réduction (le figé la paie sans la consommer) ; (c) les **murs réfléchissants
vs les halos de fenêtre** — voir §5.

| Poste | LOC | Remarque |
|---|---|---|
| Kernel fidèle (préfixe réutilisé + dt adaptatif + b réel) | ~300–400 | nouveau kernel, ne touche PAS les figés |
| Contrôle CFL-adaptatif (boucle de sous-pas, réduction on-device de s2) | ~150–250 | la réduction existe ; la boucle est neuve |
| Harnais de fidélité (Δ au rederive f64 borné, cellule par cellule) | ~150–250 | **c'est lui qui décide si le portage est « assez fidèle »** |
| **Total C1** | **~600–900** | **1,2 – 2,0 séances** |

### C2 — Exner sur GPU

Dépôt/érosion type Exner sur cellules mouillées, aux snapshots, sur les
fenêtres. `_exner_step` est simple (Euler explicite masqué), mais le porter
sur la pyramide multi-résolution et le cadencer aux snapshots l'est moins.

| Poste | LOC | |
|---|---|---|
| Kernel Exner + contrôle aux snapshots | ~180–280 | nouveau |
| Tests | ~120–150 | |
| **Total C2** | **~300–430** | **0,4 – 0,6 séance** |

### C3 — Pulse, bathymétrie réelle, structure d'épisode

Pulse gaussien sur device, `default_terrain` (b0 figé) comme champ device
dans la structure fenêtrée, et la **boucle d'épisode** (relaxation N_settle +
snapshots Exner) mappée sur les frames. Contient le mapping épisode↔frame,
part du risque de glissement (§5).

| Poste | LOC | |
|---|---|---|
| Pulse + b0 device + réconciliation d'échelle (§5) | ~150–250 | |
| Contrôle d'épisode → frames | ~150–200 | le mapping |
| Tests | ~120–150 | |
| **Total C3** | **~420–600** | **0,5 – 0,8 séance** |

### C4 — Harnais d'émissions

3 seeds {101, 102, 103}, Δt = 4, 6 émissions. Le rederive est
`run_history`/`run_episode` **appelé tel quel** (intouché). À chaque émission :
reconstruire le s vivant, lire Δχ (`albedo`/`delta_chi` réutilisés),
accumuler la série, MORT-b **par-seed**.

| Poste | LOC | |
|---|---|---|
| Boucle d'émissions live + rederive, reconstruction, Δχ | ~250–350 | |
| MORT-b par-seed + série | ~80–120 | |
| Tests | ~150–200 | |
| **Total C4** | **~480–670** | **0,6 – 0,9 séance** |

### C5 — Driver + contrôle T1 verdictal

Pipeline V4 complet à F fidèle, chrono B6 sur la frame complète, T1
(frame > 16,7 ⇒ mort), lecture mécanique. Patron éprouvé (M-a-quater), plus
le verdict T1.

| Poste | LOC | |
|---|---|---|
| Driver + T1 verdictal | ~350–450 | référence M-a-quater 771 (mais structure réutilisée) |
| Tests | ~200–280 | |
| **Total C5** | **~550–730** | **0,6 – 0,9 séance** |

### C6 — Verrous

Rederive INTOUCHÉ (diff vide prouvé) ; F fidèle « assez fidèle » (Δ au
rederive borné, nommé) ; MORT-b par-seed câblé ; T1 verdictal ; `EPS_PRODUCTION`
1e-2 et k=4 figés ; kernels FIGÉS intouchés (empreintes `_SOURCE` /
`_SOURCE_L3`).

| Poste | LOC | |
|---|---|---|
| Verrous + tests | ~250–350 | |
| **Total C6** | **~250–350** | **0,3 – 0,5 séance** |

---

## 4. Total, et la découpe interne (car > 3 séances)

| Composant | LOC | Séances |
|---|---|---|
| C1. Portage F fidèle (moteur) | ~600–900 | 1,2 – 2,0 |
| C2. Exner GPU | ~300–430 | 0,4 – 0,6 |
| C3. Pulse / b réel / épisode | ~420–600 | 0,5 – 0,8 |
| C4. Harnais d'émissions | ~480–670 | 0,6 – 0,9 |
| C5. Driver + T1 | ~550–730 | 0,6 – 0,9 |
| C6. Verrous | ~250–350 | 0,3 – 0,5 |
| **TOTAL** | **~2600–3680** | **3,6 – 5,7** |

**> 3 séances — je propose une scission interne, risqué d'abord (règle 3),
exactement comme la découpe A/B de M-a-quater :**

- **Tranche 1 — le GATE de coût T1 (~2,1 – 3,4 séances)** : C1 + C2 + C3 +
  le driver-chrono + T1 (part de C5) + verrous minimaux. **Livrable : le
  verdict T1** — une frame complète avec le F fidèle (bathy réelle, dt
  adaptatif, Exner) tient-elle sous 16,7 ? C'est **le moins cher qui peut
  tuer V4-b** : si la frame fidèle crève le budget, MORT-b est prononcée par
  T1 **avant** que le harnais d'émissions ne soit construit. Une tranche au
  lieu de deux, et le repli 33.3 devient la décision.

- **Tranche 2 — la MESURE de fidélité (~1,5 – 2,3 séances)** : C4 + le reste
  de C5 (Δχ, MORT-b par-seed) + C6 complet. **Livrable : MORT-b par-seed** —
  l'écart live↔rederive sur les 6 émissions, contre 0,0733.

La vertu n'est pas budgétaire : elle est que **la fidélité ne se mesure que si
le coût passe**. Construire le harnais d'émissions d'abord produirait une belle
mesure d'un moteur dont on ne saurait pas s'il tient le temps réel.

---

## 5. Le morceau le plus susceptible de glisser mesure → prototype

**C'est le F fidèle (C1), et le risque est PIRE que pour L3.**

L3 était un morceau de moteur greffé sur F (« émettre en épilogue ») ; on a pu
le contenir en figeant ses choix d'implémentation à l'avance et en le déclarant
dernière escalade. Le F fidèle **EST le moteur**. Trois points précis où il
glisse :

1. **Murs réfléchissants vs halos de fenêtre.** Le solveur CPU applique des
   murs réfléchissants au bord du domaine 64². Les fenêtres de la pyramide V4
   ont des bords **intérieurs** : la bonne condition n'y est pas « mur
   réfléchi » mais « halo échangé avec le voisin / prédit depuis le parent ».
   Le kernel figé pade en réfléchissant PAR FENÊTRE (correct pour une borne de
   coût, faux pour la physique). Réconcilier cela est le cœur de la fidélité —
   et c'est là qu'« assez fidèle » devient un jugement, pas une mesure.

2. **f32 GPU contre f64 CPU.** Le rederive est f64 et intouché ; le live est
   f32 de bout en bout. La fidélité **ne peut pas** être bit-à-bit — seulement
   « assez pour Δχ < JND ». Chaque milliseconde manquante au budget T1
   appellera une simplification de la physique qui élargira Δχ, et la
   frontière entre « instrument » et « moteur qu'on optimise » s'effacera —
   le tapis roulant que le cap anti-escalade craignait, ici sans cap possible
   puisque le moteur EST l'objet.

3. **Le mapping épisode↔frame (C3).** Le rederive pense en épisodes (pulse →
   600 pas → Exner). Le live pense en frames (une frame = un pas, budget
   16,7 ms). Collapser un épisode dans une frame en ferait une borne de coût,
   pas la physique ; l'étaler fidèlement sur des frames change la sémantique de
   T1 (coût par frame soutenu sur toute la relaxation). Le choix se prend au
   build, et il est load-bearing pour ce que T1 signifie.

**Ce que je recommande pour contenir C1**, à graver avant sa première ligne
(pré-enregistrement du F fidèle, dû et non chiffré ici) :
- (i) **figer le critère de fidélité À L'AVANCE** — la borne de Δ au rederive
  qui définit « assez fidèle » — de sorte qu'un Δχ décevant se lise « ce
  portage-là, à cette fidélité, diverge de tant », jamais « il faut mieux
  porter » ;
- (ii) **nommer explicitement la condition de mur intérieur** (halo/prédiction)
  comme partie du protocole, pas comme un détail à régler en route ;
- (iii) écrire le F fidèle **comme le portage d'une physique gravée** (le
  rederive), pas comme un moteur à améliorer — et le dire dans le docstring.

Les autres composants ne portent pas ce risque au même degré : Exner est un
Euler masqué simple, le harnais d'émissions suit le patron des sondes, le
driver suit M-a-quater.

---

## 6. Ce que EPS 1e-2 et la géométrie emboîtée changent à tranche-1
(1200–1800 LOC estimés à l'époque)

Deux effets, de sens opposés, et il faut les dire tous les deux.

**La structure est DÉJÀ BÂTIE — l'estimation de tranche-1 est en grande partie
déjà dépensée.** Géométrie emboîtée, L3, L1 k=4 étalée, prédiction GPU-side,
transferts comptés : tout cela a été construit par les achats A/B et tourne
(M-a-quater). M-b **réutilise** ces pièces, il ne les rebâtit pas. Les
~1200–1800 LOC que tranche-1 chiffrait pour *bâtir le pipeline* ne sont donc
plus à repayer ; le neuf de M-b, ce sont C1–C6 ci-dessus, **re-centrés sur le
F fidèle**. La magnitude totale reste du même ordre, mais son CONTENU a
basculé : de « construire la structure » vers « construire le moteur fidèle ».

**EPS 1e-2 DESSERRE le gate T1, sans réduire le build.** À 1e-4 le jetable
était déjà à 16,589 (marge 0,111) ; le F fidèle n'y aurait eu quasi aucune
place. À 1e-2 la médiane tombe à 14,490 — **2,199 ms de marge**, presque
entièrement gagnée sur les transferts (2,449 → 0,110, trafic ÷ ~22). C'est ce
qui rend T1 plausible. Mais EPS 1e-2 ne retire aucune ligne de code, et il a
un **coût côté fidélité** : une reconstruction plus grossière (seuil plus haut)
peut diverger DAVANTAGE du rederive — donc pousser MORT-b, pas T1. Et la marge
que la sonde a établie à 1e-2 est celle du **CANAL** (portée 2 de §A23), PAS
du lointain : rien ne garantit encore que le lointain reste fidèle à 1e-2. La
tranche 2 (émissions) porte donc, seule, le risque que le gain de coût de 1e-2
se paie en divergence perceptuelle.

En somme : 1e-2 + emboîtée **rendent T1 atteignable et déplacent l'estimation
de la structure vers le moteur** ; ils ne rendent PAS M-b moins cher, et ils
transfèrent une part du risque du coût (T1, tranche 1) vers la fidélité
(MORT-b, tranche 2).

---

## 7. Ce que ce chiffrage ne couvre pas

- Le **pré-enregistrement du F fidèle** lui-même (document, dû AVANT le code,
  avec le critère de fidélité et la condition de mur intérieur figés) — non
  chiffré en séances ici.
- La **décision T1 médiane vs pire frame** (§2) : à trancher avant le build.
- La **décision porte 33.3** (repli si MORT-b) : nommée en repli, non
  instruite, hors de ce build.
- Toute mesure : **aucun run n'est prévu par ce chiffrage**. L'achat porte sur
  le build.

**POINT D'ARRÊT.** Chiffrage remonté, rien d'autre. Aucune ligne de code du
harnais M-b écrite.
