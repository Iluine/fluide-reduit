# PRÉ-ENREGISTREMENT — PAIRES APPARIÉES : l'alignement de warps pèse-t-il ? (2026-08-04)

> **Statut : PRÉ-ENREGISTREMENT, écrit AVANT tout run.** Le journal
> `pocCascade2phys/PREREGISTRATION.md` fait foi. Ancres au format §A50,
> **contrôlées par `verifier_ancres.py` avant commit**.
> **Ce document se commit SEUL** (règle §A52, quatrième application).
> **L'endossement est le commit de Romain ; aucun run avant.**

---

## §0. Ce que le run précédent a laissé, et pourquoi il faut recommencer

`§A59` a rendu **INDÉTERMINÉ** avec deux causes, toutes deux fautes du
protocole : les groupes n'étaient **pas appariés en taille** (le contraste
mesurait la taille), et les **médianes étaient contaminées par la gigue** aux
petits côtés (36 % d'étendue sur la médiane contre 12 % sur le minimum).

Ce document répond aux deux, et **ne réutilise aucun chiffre** du run précédent
autrement que comme témoin.

**Deux questions, un seul run :**

> **(Q1) L'alignement de warps coûte-t-il quelque chose, à TAILLE ÉGALE ?**
> **(Q2) Le coût par cellule est-il stable en taille, une fois la série
> correctement dimensionnée ?**

**La décision qui en dépend** (D17) : `s_max(budget) = 52,6` reste invalidée tant
que Q2 n'a pas de réponse ; et si Q1 répond « oui », les candidats se réduisent
aux multiples de 32 — `PREREGISTRATION.md:1032-1033` interdisant par ailleurs de
transformer cette inversion en constante de design tant que le perceptuel n'a pas
parlé.

---

## §1. LE PLAN — des TRIPLETS, pas des paires

Une paire `(aligné, chevauchant)` laisse passer tout écart de taille résiduel.
Un **triplet centré** l'annule à l'ordre deux :

| triplet | chevauchant bas | **aligné** | chevauchant haut |
|---|---|---|---|
| T32 | 30 | **32** | 34 |
| T64 | 62 | **64** | 66 |
| T96 | 94 | **96** | 98 |
| T128 | 126 | **128** | 130 |

**Tous pairs** — seule contrainte de la machinerie, `§A58` : *l'identité
`src/f1_gpu/pyramide.py:90-91` « l'identité oy_j − 2·oy_parent = n_fov/2 étant
exacte à tous les niveaux »* n'exige que la parité.

**La grandeur de verdict est une DIFFÉRENCE SECONDE :**

> **`Δ(s) = c(s) − [ c(s−2) + c(s+2) ] / 2`**

Elle **annihile toute tendance lisse en taille** — fraction de bord, cache,
occupancy — et ne laisse que ce qui est **discret en `s`**, c'est-à-dire
l'alignement. C'est ce que le plan de `§A59` ne faisait pas, et c'est pourquoi il
ne pouvait rien départager.

`Δ < 0` ⇒ l'aligné est **moins cher** ⇒ l'alignement paie.
`Δ > 0` ⇒ l'aligné est **plus cher** ⇒ contredit le mécanisme.
`Δ ≈ 0` ⇒ **voir §4 : un zéro ne se prononce pas sans puissance.**

---

## §2. LA SÉRIE SE DIMENSIONNE SUR LA DURÉE, PAS SUR UN COMPTE

`src/f1_gpu/chrono.py:3-6` grave « warmup ≥ 30 frames EXCLU ; série ≥ 300
frames » — un **plancher**, jamais un plafond. `§A59` a montré qu'à `s = 32` la
frame dure 1,02 ms et que 330 frames ne stabilisent pas la médiane (rapport
médiane/minimum **1,35**, contre **1,04** à `s = 40`).

> **`série = max(300, ⌈3 000 ms / durée_de_frame_estimée⌉)`**, warmup
> `= max(30, 10 % de la série)`.

La durée estimée vient d'un **pré-chrono de 20 frames** hors série, reporté à
part. Aucun seuil n'en dépend : il ne fait que dimensionner.

**Et la série se contrôle elle-même** (I-p2) : la médiane de la **première
moitié** et celle de la **seconde** doivent s'accorder. Un compte de frames choisi
d'avance est une espérance ; un accord entre demi-séries est une **mesure**.

---

## §3. LA CONFIGURATION, ÉPINGLÉE

Identique à `§A55` M-c5 et à `§A59` — **le vocabulaire réel** :

| paramètre | valeur |
|---|---|
| scalaires advectés / statiques lus | **3 / 2** (9 champs) |
| systèmes · fenêtres | 1 · 3 |
| kernel | `substrat_fusionne_3d_param`, **INTOUCHÉ** (`f368de9`) |
| chrono | B6 cuda-Events, médiane **et** p99, série dimensionnée (§2) |

**Le kernel ne change pas d'une ligne** : `n` est un argument, il n'entre nulle
part dans sa source. **La seule chose qui varie est le côté.**

---

## §4. LA RÈGLE NEUVE : NE PAS FABRIQUER UN NUL

`Δ ≈ 0` est le résultat **favorable** à la thèse en cours — il laisse `52` dans
les candidats. Or *un chiffre défavorable n'est pas plus sûr qu'un chiffre
favorable* (`PREREGISTRATION.md:8748`), et un **zéro** est le plus facile à
fabriquer de tous : **il suffit de mesurer mal.**

> **Un résultat NUL ne se prononce qu'accompagné de sa PUISSANCE :** l'effet
> minimal que le dispositif aurait détecté. Sans elle, « aucun effet » et « aucun
> pouvoir de détection » sont le même nombre.

**L'effet qui compte est fixé AVANT, et il se dérive** : `s_max ∝ c^(−1/3)`, donc
un surcoût de `x` % sur les chevauchants déplace `s_max` de `x/3` %. Pour faire
passer le candidat de **52** à **48**, il faut `s_max` sous 50, soit
`x ≈ 20 %`. D'où :

> **EFFET MINIMAL PERTINENT = 10 %** de `c` — la moitié de ce qui changerait le
> candidat, pour garder de la marge.

**La puissance est MESURÉE, non supposée** : le bruit de chaque point est
l'écart entre ses deux demi-séries (§2). `Δ` n'est déclaré significatif que s'il
dépasse ce bruit propagé ; et **P-A ne peut être prononcé que si ce bruit est
lui-même sous 10 %** — sinon le dispositif n'aurait pas vu l'effet qui compte, et
le verdict est **INDÉTERMINÉ par manque de puissance**, jamais « pas d'effet ».

---

## §5. LES BRANCHES, ÉCRITES AVANT LA LECTURE

**Q1 — l'alignement :**

> **P-A — `|Δ|` sous le bruit sur les quatre triplets, ET bruit < 10 % :
> l'alignement de warps NE PÈSE PAS dans ce kernel.** Cause disponible et à
> reporter : le stencil relit un halo ±1 sur **trois** axes, donc les voisins en
> `y` et `z` sont à `n` et `n²` d'écart — **l'essentiel du trafic est déjà non
> coalescé par construction**, et l'alignement en `x` ne porte qu'une fraction.
> ⇒ tous les côtés pairs restent candidats.
>
> **P-B — `Δ < 0` significatif et de signe COHÉRENT sur les quatre triplets :
> l'alignement paie.** Les candidats se réduisent aux multiples de 32, dont le
> seul sous `s_max` est **32**.
>
> **P-C — `Δ > 0` significatif : les alignés sont PLUS chers.** Contredit le
> mécanisme ; le chiffre **n'est pas consommé** avant qu'une cause soit nommée.

**Q2 — la stabilité en taille**, sur les 12 points :

> **S-A — étendue relative ≤ 10 % : le coût par cellule est stable.**
> L'inversion `s_max(budget)` redevient valide et le point fixe doit coïncider
> avec elle.
>
> **S-B — étendue > 10 % : le coût dépend de la taille.** `s_max` reste un
> **point fixe**, calculé sur `c(s)` mesuré, et reporté comme **intervalle** si
> les deux méthodes d'interpolation divergent.

---

## §6. LES INDÉTERMINATIONS

**(I-p1) Reproduction.** Le point `64³` doit reproduire `§A55` (9,2669
ns/cellule) à **10 %**. Sinon **INDÉTERMINÉ** : le harnais n'est pas celui-là.
*Sur le cas favorable* : ne porte que sur `64`, ne peut punir aucun candidat. ✔

**(I-p2) Stabilité de série, par point.** Si
`|médiane(1ʳᵉ moitié) − médiane(2ᵈᵉ moitié)| / médiane > 3 %`, **le point est
écarté et son triplet entier avec lui**. Un triplet amputé ne rend pas de `Δ`.
*Sur le cas favorable* : un point anormalement RAPIDE mais instable est écarté
comme un lent. ✔

**(I-p3) Puissance.** Voir §4 : `P-A` exige un bruit mesuré **< 10 %**. Sinon
**INDÉTERMINÉ par manque de puissance**.
*C'est le critère braqué sur le cas favorable de ce protocole* — le nul est ici
le résultat commode. ✔

**(I-p4) Cohérence de signe.** Si deux triplets rendent des `Δ` significatifs de
**signes opposés**, le dispositif ne mesure pas ce qu'il croit ⇒ **INDÉTERMINÉ**,
quel que soit le total.

**(I-p5) Registres constants** sur tous les côtés (la source ne dépend pas de
`n`). Un écart signifie qu'autre chose a bougé ⇒ **INDÉTERMINÉ**.

---

## §7. GARDES ANTI-FABRICATION

1. **Aucun seuil ne bouge** ; le plancher `≥ 300` frames et `≥ 30` de warmup sont
   respectés, jamais abaissés — seulement **dépassés**.
2. **La lecture reste sur la MÉDIANE.** `§A59` a montré que le minimum donne une
   image plus flatteuse (12 % contre 36 %) ; **basculer d'instrument après avoir
   vu un résultat est la faute que ce journal surveille**. Le rapport
   médiane/minimum est reporté comme **diagnostic**, jamais comme lecture.
3. **Les triplets sont fixés ici**, avant tout run. Ajouter ou retirer un côté
   après lecture fabriquerait le contraste — c'est la faute exacte de `§A59`.
4. **Le point `64³` est re-mesuré**, jamais recopié : un témoin recopié ne
   témoigne pas.
5. **Kernels intouchés** (`git diff` vide, empreintes relues).
6. Pas d'`assert` nu ; fail-loud ; `else` de combinateur = sortie INDÉTERMINÉE.
7. **L'artefact est un fichier** versionné dans `claude/lectures/`.

---

## §8. CE QUE CE RUN NE PRONONCERA PAS

- **Rien sur la cadence** ni sur une **constante de design** : la gravure reste
  `s_max(budget)`, une inversion (`§A58-4`).
- **Rien sur le rendu** : le dû de `PREREGISTRATION.md:4142` « Portes 33.3
  (motif T2 : où vit le rendu — à traiter explicitement si ouverte) » reste le
  **verrou réel**, et aucun cap calculé jusqu'ici ne l'inclut.
- **Rien sur la monnaie du slot** : `PREREGISTRATION.md:8200` « Transposition 3D
  exacte : `512² = 64³ = 262 144` cellules » ne vaut plus hors de `64`, et sa
  re-dérivation est la **porte 3** de `§A58`, à faire à voix haute.
- **Rien sur les causes profondes** du coût par cellule. Ce run sépare
  **discret** et **lisse** ; il n'attribue pas le lisse.

---

## §9. CE QUI SERA DÛ APRÈS

Selon la branche : si **P-A + S-A**, l'inversion redevient valide et la porte 2
se ferme — restent la porte 3 et le rendu. Si **S-B**, le point fixe et son
intervalle remplacent `s_max` partout où il est écrit. Dans tous les cas, **le
rendu reste devant**.

---

*Document rédigé par la session Claude, AVANT tout run. L'endossement est le
commit de Romain.*
