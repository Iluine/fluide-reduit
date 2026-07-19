# Chiffrage du build M-a-quater — gate d'achat (§A18 P5, pocCascade2phys 7a85fbc)

> **VERDICT EN PREMIÈRE LIGNE : 2.5 à 3.5 séances — AU-DESSUS DU CAP DE 2.**
> Le gate §A18 exige donc une remontée avant achat. Une découpe en DEUX achats
> séparés est proposée au §7 ; elle a une vertu propre, pas seulement
> budgétaire : elle met le morceau qui peut échouer en premier.
>
> Aucune ligne de code n'a été écrite. Ce document est le seul livrable.

---

## 1. Ce que le build doit accomplir — l'arithmétique d'abord

Avant de chiffrer l'effort, il faut chiffrer la CIBLE, sans quoi « combien ça
coûte » ne se juge pas.

| Poste | Valeur | Origine |
|---|---|---|
| F batché V4 | **12.655 ms** | §A18 P3, recalculé ici depuis les ancres mesurées 1.685 / 0.845 — concordance exacte |
| Budget T2 | 16.7 ms | inchangé |
| **Reste pour prédiction + remontée + transferts** | **4.045 ms** | 16.7 − 12.655 |

Or les acquis mesurés de la cascade donnent : prédiction **≥ 2.5 ms** dans
*toute* configuration mesurée (2.819 CPU-side, 2.570 GPU-side partiel), et
remontée **8.523 ms** telle qu'elle est aujourd'hui.

**Conséquence chiffrée : pour que V4 échappe à la mort, L3 doit ramener la
remontée de 8.523 ms à moins de ~1.5 ms — un facteur ≈ 5.7.** (Détail :
12.655 + 2.5 + R > 16.7 ⟺ R > 1.545.) Si la prédiction emboîtée descend sous
2.5 — ce que V4 rend possible puisque tous les slots ≥ 2 auront un parent —
l'exigence sur L3 se desserre d'autant, mais elle reste d'un facteur ≥ 4.

C'est le chiffre que je recommande de regarder avant de décider l'achat : le
build est justifié si un facteur ~5 sur la remontée est jugé plausible. Le
papier ne le promet pas, et ce chiffrage ne le promet pas davantage.

Note de cohérence sur la bande : frame prédite ∈ [13.9, 17.2] avec F = 12.655
revient à prédire prédiction + remontée + transferts ∈ [1.245, 4.545]. La borne
basse suppose une remontée quasi nulle ; la bande chevauche le seuil, ce que
§A18 assume explicitement.

---

## 2. Composant 1 — Kernel-moteur L3 (émission des détails en épilogue de F)

**Ce que c'est.** Un kernel CUDA **NEUF** (fichier neuf), reprenant le motif du
fusionné et ajoutant un épilogue : chaque thread, après avoir écrit son état
sortant, compare aux références et émet les coefficients |d| ≥ EPS_DETAIL. Le
fusionné actuel — 180 lignes CUDA, empreinte
`e8fcaad47db0af010728d5c5d878e1629191bfef4e0382fa80aa3cefd9f27f04` — reste
**INTACT** : la borne M-a′ n'est pas re-mesurée, elle est citée telle quelle.

**Contenu.**

| Poste | LOC | Remarque |
|---|---|---|
| CUDA : reprise du motif | ~180 | copie conforme, c'est la base de l'équivalence |
| CUDA : épilogue d'émission + compaction atomique | ~45 | le vrai travail neuf |
| Wrapper Python + buffers de sortie préalloués | ~110 | dimensionnement au pire cas |
| Pré-enregistrement dédié (document) | — | ~1 page, à rédiger AVANT le code |
| Tests d'équivalence (état + ensemble émis) | ~260 | cf. `test_f1_substrat_fusionne_s1.py`, 226 LOC pour 3 verrous |
| **Total** | **~595** | dont ~225 CUDA |

**Effort : 0.8 – 1.2 séance.** C'est le poste incompressible : le CUDA s'écrit et
se débugge plus lentement que le harnais Python, et l'équivalence doit être
prouvée, pas supposée.

**Risques nommés.**

1. **Ordre d'émission non déterministe.** Une compaction par `atomicAdd` produit
   le bon ENSEMBLE de coefficients mais dans un ordre variable d'un run à
   l'autre. Le dépôt teste bit-à-bit ; les tests d'équivalence devront comparer
   des ensembles triés, pas des séquences. Piège classique, coût : ~40 LOC de
   test en plus, mais il doit être vu d'avance sinon il se paie en débogage.
2. **Le déterminisme du dépôt.** La vérif #3 (gel bit-exact) ne porte pas sur ce
   chemin, mais la discipline « comparaison exacte » du projet devra être
   explicitement assouplie ICI, et nulle part ailleurs — à écrire dans le
   pré-enregistrement du kernel.
3. **Dimensionnement des buffers de sortie.** Au pire cas, tout est émis : les
   buffers valent la fenêtre entière. Les préallouer au pire cas est la seule
   option honnête (B2), et c'est de la VRAM en plus — à vérifier contre le gate
   1.35 Go, qui a de la marge (0.311 Go mesuré en V2).

---

## 3. Composant 2 — Géométrie emboîtée V4 (propriété E)

**Ce que c'est.** Remplacer le placement B3 (±n_fov, du harnais) par un
placement d'ARBRE : toute fenêtre déclare une parente couvrant intégralement son
empreinte, l'empreinte valant un quart de la fenêtre parente.

**Placement concret proposé.** Une fenêtre au niveau *j* occupe, en coordonnées
du niveau *j−1*, un quart de la fenêtre parente ; une parente couvre donc jusqu'à
4 enfants. Le fovéal de niveau *j* prend le quart centré sur le regard, ce qui
laisse **3 quarts libres** dans la fenêtre du fovéal *j−1*. Les **2 slots
d'énergie de V4 se placent dans 2 de ces 3 quarts libres, au niveau le plus
fin** — leur parente est alors le fovéal du niveau immédiatement supérieur, qui
existe déjà dans le budget. Conséquences : emboîtement satisfait par
construction, **aucune tour d'ancêtres nécessaire**, et le budget d'arbre reste
à 11 slots — exactement la composition V4 gravée (vérifié : 2×1.685 + 7×0.845 +
2×1.685 = 12.655 ms, concordance exacte avec §A18 P3).

Le choix du quart, parmi les 3 libres, doit être déterministe et non arbitraire :
je propose l'ordre « quart adjacent en y d'abord, puis en x », à graver dans la
géométrie et à tester, pour qu'aucune mesure ne dépende d'un choix implicite.

**Contenu.**

| Poste | LOC | Remarque |
|---|---|---|
| Géométrie V4 **parallèle** à B3 | ~180 | voir risque 1 ci-dessous |
| Placement quart-d'empreinte + validation d'arbre | ~120 | la validation est ce qui rend la propriété E vérifiable |
| Tests (placement, emboîtement, non-régression) | ~250 | |
| **Total** | **~550** | |

**Effort : 0.5 – 0.7 séance.**

**Risques nommés.**

1. **Ne pas casser l'historique.** `origines()` porte aujourd'hui B3 et cinq
   fichiers de tests en dépendent (≈ 60 tests touchent origines/lockstep/offsets).
   La géométrie V4 doit être **parallèle**, pas un remplacement : M-a, M-c, M-a′,
   les sondes s1/s2/s2-mobile/s3 doivent rester reproductibles au bit près,
   sinon toutes les ancres mesurées deviennent incomparables. C'est le point où
   un raccourci coûterait le plus cher.
2. La validation d'arbre est à écrire comme une garde fail-loud (une fenêtre
   sans parente couvrante doit lever), pas comme un commentaire.

---

## 4. Composant 3 — Champs d'échelle (P2)

**Ce que c'est.** Les champs au-delà de `c_parent` sont à contenu grossier nul
par définition : leur prédiction parent→enfant est un remplissage-zéro, exact.
Le trou « c dégressif vs descente » disparaît, et **le repli c-mismatch du niveau
8 disparaît avec lui** — ce slot passe GPU-side.

**Contenu.**

| Poste | LOC |
|---|---|
| Prédiction zéro pour les systèmes/champs > c_parent | ~60 |
| Retrait du repli c-mismatch (`RAISON_SYSTEMES`) + plan | ~20 |
| Tests (zéro exact, plan sans repli c, non-régression) | ~140 |
| **Total** | **~220** |

**Effort : 0.2 – 0.3 séance.** C'est le composant le moins cher.

**Risque nommé — et il n'est pas cosmétique.** Un système entièrement à zéro est,
pour le substrat, une fenêtre **sèche** : `desing` rend 0, le plancher sec
s'applique, et surtout **toutes les branches du warp partent du même côté**. Une
fenêtre uniformément sèche peut donc coûter MOINS qu'une fenêtre mouillée, par
absence de divergence. Si les champs d'échelle sont massivement nuls, la mesure
M-a-quater **minorerait** le coût de F — un harnais complaisant, exactement le
défaut que la discipline du projet interdit. Le build doit inclure une
vérification explicite : comparer le coût d'un bloc à système nul et d'un bloc
mouillé, et reporter l'écart. Si l'écart est significatif, la lecture de
M-a-quater devra en tenir compte, ou le contenu jetable des champs d'échelle
devra être re-décidé (décision de protocole, pas d'implémentation). **Coût de
cette vérification : ~60 LOC, incluse ci-dessus.**

---

## 5. Composant 4 — Compaction et transfert de la remontée L3

**Ce que c'est.** Le kernel émet des paires (indice, valeur) compactées ; il faut
les transférer à l'hôte. Deux D2H par niveau aujourd'hui ; ici, un gather déjà
fait par le kernel, puis le transfert.

**Contenu.**

| Poste | LOC |
|---|---|
| Compteur d'émission + gather côté sortie kernel | ~90 |
| Transfert D2H + comptabilisation dans `TransfertComptable` | ~80 |
| Tests (comptage exact, équivalence avec l'extraction CuPy actuelle) | ~160 |
| **Total** | **~330** |

**Effort : 0.3 – 0.5 séance.**

**Risque nommé — le compteur rouvre la maladie déjà diagnostiquée.** Pour savoir
combien d'octets transférer, il faut LIRE le compteur d'émission, donc rapatrier
un scalaire, donc **synchroniser** — précisément l'artefact que s2 a supprimé sur
la CFL et qui valait plusieurs millisecondes. Trois options, à trancher AVANT le
code (aucune n'est gratuite) : (a) transférer le pire cas à taille fixe — pas de
sync, mais le trafic explose et la mesure perd son sens ; (b) accepter une sync
par frame — honnête, mais on réintroduit ce qu'on vient de retirer ; (c) lire le
compteur avec une frame de retard, le transfert de la frame *n* étant dimensionné
au compteur de *n−1* — pas de sync, au prix d'une complexité et d'une
approximation à nommer. **Je recommande (c) avec l'écart reporté**, mais c'est
une décision de protocole qui appartient au pré-enregistrement du kernel, pas à
moi.

---

## 6. Composant 5 — Driver `run_f1_ma_quater.py` + Composant 6 — réserve L1

**Driver.** Pipeline complet V4 (F batché + prédiction GPU-side emboîtée +
remontée L3 + transferts), fovéa mobile E4c, chrono B6, vérifs reconduites.
Lecture mécanique : mort > 16.7 sur la frame complète, bande [13.9, 17.2],
parts F / prédiction / remontée / transferts.

| Poste | LOC | Remarque |
|---|---|---|
| Driver | ~480 | référence : `run_f1_s3_pred_gpu.py` = 581 |
| Tests | ~320 | référence : son fichier de tests = 383 |
| **Total** | **~800** | |

**Effort : 0.5 – 0.7 séance.**

**Point à trancher, remonté ici.** §A18 demande les *parts* F / prédiction /
remontée / transferts tout en précisant que **la série de sondes n'est pas
rouverte**. Or les parts, dans la cascade, ont toujours été obtenues par
différence entre bras. Sans bras supplémentaires, il reste deux voies :
instrumenter le driver par compteurs internes (cuda-Events autour de chaque
étage — mesure intrusive, qui ajoute des barrières et fausse ce qu'elle mesure),
ou dériver les parts d'ancres déjà gravées (F = 12.655 prédit, transferts
comptés exactement par `TransfertComptable`, remontée et prédiction par
soustraction). **Je recommande la seconde**, la première contredisant l'acquis de
s2. À confirmer au moment de l'achat.

**Réserve L1 (cadence k).** Un flag `cadence_remontee=k` dans la pyramide,
**chiffré mais NON armé** : la remontée ne s'exécute qu'une frame sur k, le flag
vaut 1 par défaut et aucun run ne l'active. ~45 LOC + ~70 de tests = **~115 LOC,
0.1 séance.** L'armer serait une décision de cadencement touchant la condition de
réveil σ_ω (R4) — hors de ce build.

---

## 7. Total, et la découpe que je recommande

| Composant | LOC | Séances |
|---|---|---|
| 1. Kernel-moteur L3 | ~595 | 0.8 – 1.2 |
| 2. Géométrie emboîtée V4 | ~550 | 0.5 – 0.7 |
| 3. Champs d'échelle | ~220 | 0.2 – 0.3 |
| 4. Compaction + D2H | ~330 | 0.3 – 0.5 |
| 5. Driver M-a-quater | ~800 | 0.5 – 0.7 |
| 6. Réserve L1 (flag, non armé) | ~115 | 0.1 |
| **TOTAL** | **~2610** | **2.4 – 3.5** |

Référence de calibration : les séances récentes ont produit 800 à 1300 LOC
(M-a-ter 1316, s1+s2 1018, s2-mobile 809, s3 964). 2610 LOC de Python de harnais
seraient ~2.2 séances ; le CUDA et ses équivalences justifient la majoration.

**Découpe proposée — deux achats, et le risqué d'abord :**

- **Achat A (1.1 – 1.7 séance)** : composants 1 et 4 — le kernel-moteur L3, son
  pré-enregistrement propre, sa compaction et son transfert. Livrable : une
  **borne** du coût de la remontée L3, mesurée seule, comme M-a′ fut une borne de
  F. Rien d'autre n'est touché ; la géométrie reste V2.
- **Achat B (1.3 – 1.8 séance)** : composants 2, 3, 5 et 6 — géométrie emboîtée,
  champs d'échelle, driver complet, flag L1.

La vertu de cette découpe n'est pas budgétaire. **Si la borne de l'achat A montre
que la remontée L3 ne descend pas d'un facteur ~5, V4 est mort avant que l'achat
B ne soit dépensé** — et le repli 33.3 devient la décision, avec un achat au lieu
de trois. C'est la règle 3 du dépôt (« le moins cher qui peut échouer, d'abord »)
appliquée à l'ordre des achats. À l'inverse, construire la géométrie V4 en
premier produirait une belle géométrie dont on ne saurait pas si elle sert.

---

## 8. Le morceau le plus susceptible de glisser mesure → prototype

**C'est le kernel-moteur L3 (composant 1), et l'écart n'est pas mince.**

Le harnais F1 a toujours mesuré des motifs de COÛT dont le design était fixé
ailleurs : le fusionné réordonne le stencil sans rien changer à la physique, la
prédiction GPU-side déplace un calcul spécifié en §5. Le kernel L3, lui, est un
**morceau de moteur** : « émettre les détails en épilogue » ne dit ni comment
compacter, ni comment dimensionner les sorties, ni comment transférer. Ces choix
— atomiques contre scan, layout de sortie, gestion du compteur — pèsent
lourdement sur le résultat. Mesurer « le coût de L3 » reviendrait donc à mesurer
**mon implémentation** de L3, pas le design L3.

Ce risque a un nom précis dans ce dépôt : le tapis roulant que le cap
anti-escalade craignait. Une fois le kernel écrit, chaque milliseconde manquante
appellera une optimisation de plus, et la frontière entre « instrument » et
« moteur en développement » s'effacera.

**Ce que je recommande pour le contenir**, à graver dans le pré-enregistrement du
kernel avant sa première ligne : (i) figer les choix d'implémentation À L'AVANCE
et les nommer comme partie du protocole, de sorte qu'un chiffre décevant se lise
« ce design-là coûte tant », jamais « il faut mieux l'implémenter » ; (ii)
reconduire explicitement un cap — le kernel L3 est la **dernière** escalade
autorisée sur la remontée, comme M-a′ le fut sur F ; (iii) écrire la borne comme
une mesure jetable, non comme un composant de moteur, et le dire dans le
docstring.

Les autres composants ne portent pas ce risque : la géométrie et les champs
d'échelle sont entièrement spécifiés par §2-rev1, le driver suit un patron
éprouvé quatre fois, le flag L1 n'est pas armé.

---

## 9. Ce que ce chiffrage ne couvre pas

- Le **pré-enregistrement du kernel L3** lui-même (document, ~1 page) : il est
  dû AVANT le code et n'est pas chiffré en séances ici.
- **F0-cloud**, toujours dû, indépendant.
- La décision **porte 33.3** et son motif T2 (où vit le rendu) : nommée en repli
  pré-nommé, non instruite, hors de ce build.
- Toute mesure : **aucun run n'est prévu par ce chiffrage**, et l'achat porte sur
  le build seul.

**POINT D'ARRÊT.** Chiffrage remonté, rien d'autre. Aucune ligne de code écrite,
aucun fichier du harnais touché.
