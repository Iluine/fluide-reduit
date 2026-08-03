# PRÉ-ENREGISTREMENT — BALAYAGE DE TAILLES AU VOCABULAIRE RÉEL (2026-08-04)

> **Statut : PRÉ-ENREGISTREMENT, écrit AVANT tout run.** Le journal
> `pocCascade2phys/PREREGISTRATION.md` fait foi. Ancres au format §A50,
> **contrôlées par `verifier_ancres.py` avant commit**.
> **Ce document se commit SEUL** (règle §A52, troisième application).
> **L'endossement est le commit de Romain ; aucun run avant.**

---

## §0. La question, et la décision qui en dépend

C'est la **porte 2** de §A58, posée par Romain :

> *Toute la stabilité par cellule est mesurée à 32 / 64 / 128, et à 5 champs. Un
> côté 52 peut payer du coalescing et de l'alignement de warps qu'aucune
> puissance de deux ne révèle — et §A55 vient de montrer que les constantes de
> coût de ce kernel ont des falaises.*

> **Le coût PAR CELLULE dépend-il du côté de la fenêtre, au vocabulaire réel ?**

**La décision nommée qui en dépend** (D17) : `s_max(budget) = 52,6` est une
**inversion de budget** faite avec le coût par cellule de `64³`. Si ce coût
dépend du côté, **l'inversion est invalide** : `s_max` n'est plus un nombre mais
un **point fixe**, et la voie « 60 Hz à fenêtres plus petites » change de forme
ou meurt.

---

## §1. CE QUE LE BALAYAGE EST — un CONTRASTE conçu, pas une pêche

Le mécanisme nommé par Romain est vérifiable : le kernel met **un thread par
cellule**, avec `x = t % n`. Un warp de 32 threads consécutifs lit 32 `x`
consécutifs **si et seulement si `n` est multiple de 32** ; sinon le warp
**chevauche deux lignes**. Le balayage est donc construit en **deux groupes**,
choisis avant de mesurer :

| groupe | côtés | `n mod 32` |
|---|---|---|
| **ALIGNÉS** | 32, 64, 96, 128 | 0 |
| **CHEVAUCHANTS** | 40, 48, 52, 56 | 8, 16, 20, 24 |

Tous sont **pairs**, seule contrainte que la machinerie impose (§A58 : *« la
machinerie ne divise jamais `n_fov` plus d'une fois »*, l'identité
`src/f1_gpu/pyramide.py:90-91` « l'identité oy_j − 2·oy_parent = n_fov/2 étant
exacte à tous les niveaux » exigeant la seule parité).

**Si la pénalité est le coalescing, les deux groupes se séparent.** C'est une
prédiction falsifiable, écrite avant, et non une courbe qu'on interprétera après.

---

## §2. UN SECOND EFFET DE TAILLE, EN SENS INVERSE — nommé avant de mesurer

Le balayage ne mesurera **pas** le coalescing seul. Un second effet agit, et
dans l'autre sens : **la fraction de cellules de BORD**. Une cellule de bord
saute deux appels `pentes_axe` (les branches `m_reel` / `p_reel`), donc **coûte
moins cher**.

| côté | 32 | 40 | 48 | 52 | 56 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|---|---|
| cellules de bord | 17,6 % | 14,3 % | 12,0 % | 11,1 % | 10,3 % | 9,1 % | 6,1 % | 4,6 % |

> **Les petites fenêtres sont favorisées par le bord et pénalisées par
> l'alignement.** Le balayage mesure leur SOMME. **Les séparer n'est pas
> demandé** et ne sera pas prétendu : toute attribution à l'un des deux seuls
> devra passer par le contraste aligné/chevauchant du §1, qui est le seul
> discriminant conçu pour ça.

*(C'est aussi l'explication candidate de la non-monotonie déjà mesurée par §A53 —
5,98 ns à 32³, 6,47 à 64³, 6,08 à 128³ — jamais expliquée à l'époque.)*

---

## §3. LA CONFIGURATION, ÉPINGLÉE AVANT

Identique à **M-c5 de §A55** — la seule qui porte le vocabulaire réel — sauf le
côté :

| paramètre | valeur |
|---|---|
| scalaires advectés | **3** (`s`, `e_th`, `ρ_s`) |
| champs statiques lus | **2** (`b0`, `id-matériau`) |
| champs par système | **9** |
| systèmes | 1 |
| fenêtres par mesure | 3 |
| kernel | `substrat_fusionne_3d_param`, **INCHANGÉ** (`f368de9`) |
| chrono | B6, ≥ 300 frames, warmup exclu, médiane **et** p99 |

**Le kernel ne change pas d'une ligne.** `n` n'entre dans sa source à aucun
endroit — il est un argument. C'est ce qui rend ce balayage honnête : **la seule
chose qui varie est le côté.**

---

## §4. LES GRANDEURS

- `c(s)` = **ns par cellule** = `médiane / (3 · s³)` pour chaque côté ;
- **étendue relative** = `(max c − min c) / min c`, sur tout le balayage ;
- **contraste de groupes** = `moyenne(c sur CHEVAUCHANTS) / moyenne(c sur
  ALIGNÉS)` ;
- **le POINT FIXE** : le plus grand côté pair `s` tel que
  `11 · s³ · c(s) + 1,835 ms ≤ 16,7 ms`, avec `c(s)` **mesuré**, interpolé
  linéairement entre les côtés balayés — et non le `c(64)` de §A55.

**Pour mémoire, l'inversion NAÏVE à corriger** (calculée avec `c` constant =
9,2669 ns) : `11 × 52³` tient à 16,17 ms ; `11 × 56³` dépasse à 19,74 ms. D'où
`s_max = 52,6`, et le candidat `52`.

---

## §5. LES BRANCHES, ÉCRITES AVANT LA LECTURE

Seuils immuables :

| seuil | valeur |
|---|---|
| bande de stabilité (étendue relative) | **10 %** |
| bande de reproduction du point `64³` | **10 %** |
| séparation de groupes déclarée significative | contraste **> 1,10** ET écart entre groupes **> l'étendue intra-groupe** |

> **T-A — étendue ≤ 10 % : le coût par cellule est STABLE en taille.**
> L'inversion de budget est valide, `s_max(budget) = 52,6` tient, et le plus
> grand côté admis sous la barre reste **52**. Le point fixe est reporté et doit
> coïncider avec l'inversion naïve.
>
> **T-B — étendue > 10 %, sans séparation de groupes : le coût dépend de la
> TAILLE, cause non attribuée.** `s_max` cesse d'être un nombre : **seul le
> POINT FIXE fait foi**, et il est reporté comme tel, avec la mention explicite
> que la cause (bord, cache, occupancy) n'est pas séparée.
>
> **T-C — séparation de groupes significative : la pénalité est
> l'ALIGNEMENT DE WARPS.** Le contraste conçu au §1 a mordu. Conséquence dure :
> **la liste des candidats se réduit aux côtés multiples de 32**, dont le seul
> sous la barre est **32**. Le point fixe est alors calculé sur le seul groupe
> aligné. *(Ce n'est pas une catastrophe mais un changement d'échelle : à 32³ le
> budget porte bien plus de 11 fenêtres — le cap cesse d'être la contrainte et
> la granularité devient la question.)*

---

## §6. LES INDÉTERMINATIONS — vérifiées sur le cas FAVORABLE

`PREREGISTRATION.md:8536-8537` « critère d'indétermination se vérifie sur le cas
FAVORABLE autant que sur le cas défavorable, sinon il punit la qualité qu'il
prétend contrôler ».

**(I-t1) Reproduction.** Si le point `64³` s'écarte de plus de **10 %** de
`M-c5` de §A55 (2,4293 ms par fenêtre), le harnais n'est pas celui de §A55 :
**INDÉTERMINÉ**, aucune lecture.
*Sur le cas favorable* : ce critère ne porte que sur `64³` et ne peut pas punir
un côté candidat rapide. ✔

**(I-t2) Le kernel n'a pas changé.** Registres par thread **identiques à tous les
côtés** (la source ne dépend pas de `n`). Un écart signifie que quelque chose
d'autre a bougé ⇒ **INDÉTERMINÉ**.
*Sur le cas favorable* : un côté qui serait rapide PARCE QUE son kernel diffère
échoue ici. ✔

**(I-t3) L'AUBAINE EST AUSSI SUSPECTE.** Si un côté **chevauchant** se révèle
**moins cher par cellule** que tous les alignés, ⇒ **INDÉTERMINÉ**, et la cause
doit être nommée avant consommation. Motif : la seule explication mécanique
disponible irait dans l'autre sens ; une aubaine inexpliquée dans un balayage
qui décide d'une constante est exactement le chiffre qu'on encaisse à tort.
*Ce critère ne se déclenche QUE du côté favorable* — c'est sa raison d'être. ✔

**(I-t4) Le point fixe doit être STABLE à l'interpolation.** Si le côté rendu
par le point fixe change selon qu'on interpole `c(s)` linéairement ou qu'on
prend le `c` mesuré du côté le plus proche, ⇒ le balayage est **trop lâche** :
le point fixe est reporté comme un **intervalle**, jamais comme un côté.

---

## §7. GARDES ANTI-FABRICATION

1. **Aucun seuil ne bouge** ; 16,7 ms, ≥ 300 frames, tolérances : tous importés.
2. **Le kernel paramétré et celui de §A53 restent INTOUCHÉS** (`git diff` vide,
   empreintes relues).
3. **Le point `64³` est mesuré DANS LE BALAYAGE**, jamais repris de §A55 : c'est
   le témoin, et un témoin recopié ne témoigne pas.
4. **Le couple aligné/chevauchant est fixé AVANT le run** (§1), en deux groupes
   nommés. Reclasser un côté après lecture serait fabriquer le contraste.
5. **Deux effets de taille sont déclarés en sens inverse** (§2) ; aucune
   attribution à l'un seul ne sera écrite sans le contraste du §1.
6. Pas d'`assert` nu ; fail-loud ; `else` de combinateur = sortie INDÉTERMINÉE.
7. **L'artefact est un fichier** versionné dans `claude/lectures/`.

---

## §8. CE QUE CE BALAYAGE NE PRONONCERA PAS

- **Rien sur la cadence.** Il donne un côté admissible, pas un choix.
- **Rien sur `52³` comme constante de design.** `PREREGISTRATION.md:1032-1033`
  « Choisir `r_fovea` pour que le budget passe = fabriquer le verdict » : la
  gravure reste **`s_max(budget)`**, une inversion, tant que le perceptuel n'a
  pas parlé (exigence de forme de §A58-4).
- **Rien sur le rendu** — le verrou réel reste le dû de `PREREGISTRATION.md:4142`
  « Portes 33.3 (motif T2 : où vit le rendu — à traiter explicitement si
  ouverte) », et **aucun cap calculé jusqu'ici ne l'inclut**.
- **Rien sur la re-dérivation de la monnaie du slot** — porte 3 de §A58, qui
  suit et se fait à voix haute.

---

## §9. CE QUI SERA DÛ APRÈS

La **porte 3** : à côté ≠ 64, `512² = 64³ = 262 144` — `PREREGISTRATION.md:8200`
« Transposition 3D exacte : `512² = 64³ = 262 144` cellules » — n'existe plus, et
le cap d'emplacements, la lecture VRAM et les tables de §A51 se re-dérivent
explicitement. Puis **où vit le rendu**.

---

*Document rédigé par la session Claude, AVANT tout run. L'endossement est le
commit de Romain.*
