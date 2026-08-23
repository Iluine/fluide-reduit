# ANALYSE DE SENSIBILITÉ DE LA CADENCE — squelette pré-enregistré (2026-08-23)

> **CE DOCUMENT EST COMMITÉ SEUL, AVANT TOUT CALCUL** (`§A52` : *le
> pré-enregistrement se commit SEUL, AVANT le premier run*). Les résultats
> seront **appendus** au `§7`, jamais substitués au texte ci-dessous.
>
> **Aucune mesure. Aucun GPU. Aucun `cupy`.** Tout nombre vient d'un artefact
> existant ou est extrait du texte d'une ancre. `§A61` est respecté **par
> construction** : ce document n'a pas les moyens de mesurer.
>
> **Aucun cinquième protocole de qualification n'est ouvert ici**, et aucun
> seuil du corpus n'est déplacé.

---

## §1. LA QUESTION, UNE SEULE

> **Dans quel domaine d'incertitude sur `(non-F, I, C)` le choix de cadence —
> 60 Hz avec des fenêtres de ~52³ contre 30 Hz avec des fenêtres de 66,8³ à
> 65,5³ — CHANGE-T-IL ?**

Ce n'est **pas** la question « quelle cadence choisir ». La cadence appartient à
Romain (`§A62`). La question posée ici est **antérieure** : le corpus dit-il
quelque chose que l'incertitude actuelle rendrait faux, ou bien la décision
est-elle hors d'atteinte de cette incertitude ?

**Ce que ce document ne prononcera pas**, écrit avant de calculer :

- il ne choisit **pas** la cadence ;
- il ne prononce **pas** sur la Porte 3 de `§A58` ;
- il ne rouvre **pas** l'attribution d'`I-q3` (`§A63-1`) ;
- il ne produit **aucune** valeur mesurée neuve, et n'autorise aucun run.

---

## §2. LES ENTRÉES GRAVÉES

Chaque ligne porte son ancre textuelle : `` `fichier:NNN` « fragment exact » ``
— **le texte fait foi, le numéro est le chemin** (`§A50`). Les valeurs
numériques ne sont pas recopiées ici : elles sont **lues par la machine** dans
`claude/lectures/ou-vit-le-rendu-ir5-2026-08-04.json`, qui déclare lui-même sa
provenance. Ce document grave les **ÉNONCÉS** ; l'artefact porte les chiffres.

### E-1 — la forme de l'arbitrage

`PREREGISTRATION.md:10019`
« **(2) Tout l'arbitrage de cadence vaut `Δ = 30·(non-F − I)/C`.** »

### E-2 — l'écart à `I` = 0, et le point mort

`PREREGISTRATION.md:10028`
« **Point mort à `I` = 1,768 ms.** À `I` = 0, le 30 Hz gagne **6,2 %** du travail »

Le point mort **1,768** est un artefact d'arrondi (`§A62-bis-1`), et sa forme
propre est :

`PREREGISTRATION.md:10135`
« `1000/f`, le point mort tombe **exactement** sur non-F. »

⇒ Le balayage se fait **sous `1000/f`**, où le point mort vaut exactement le
non-F. La lecture « budgets gravés » (16,7 / 33,333) est produite **en regard**,
et l'écart entre les deux est reporté : c'est `I-r5` mécanisée (`§A62-bis-2`),
et **la branche est INDÉTERMINÉE si elle diffère entre les deux façons.**

### E-3 — l'asymétrie des deux côtés vis-à-vis de `I`

`PREREGISTRATION.md:10202`
« Le 60 Hz, lui, **ne dépend pas de `I`** : aucune image n'y est interpolée. Un »

Et le côté 30 Hz, lui, en dépend — étendue ~2 % sur la plage `[0, non-F]` :

`PREREGISTRATION.md:10198` « | 0 | 66,82 | »
`PREREGISTRATION.md:10200` « | **non-F** (point mort) | **65,45** | »

**Étiquettes obligatoires.** Tout `s_max` 30 Hz cité ici porte son `I` : un
`s_max` 30 Hz sans son `I` est un chiffre du **coin favorable** — c'est
exactement la faute `§A62-bis-4`, et **c'est la faute à ne pas refaire**.

### E-4 — l'invariance de l'étendue à un biais sur `C`

`PREREGISTRATION.md:10041`
« **Étendue 16,4 %** — et elle est **exactement invariante à un biais »

### E-5 — le biais d'instrument sur les absolus

`PREREGISTRATION.md:9751`
« systématiquement PESSIMISTES d'environ 15 %.** »

### E-6 — la provenance du non-F retenu, et sa dette

`PREREGISTRATION.md:9559`
« non-F **2D à l'échelle de l'instrument**, pas un rendu 3D à 1920. »

**Le non-F utilisé ici n'est pas le non-F 3D dû par `§A53`.** Il porte cette
étiquette dans tout le document, et le balayage de `I` va **de 0 au non-F ainsi
étiqueté**, pas à un non-F 3D qui n'existe pas.

### E-7 — ce que `§A63` ajoute à l'incertitude, hier

`PREREGISTRATION.md:10276` « **une dépendance au RANG malgré »

`PREREGISTRATION.md:10274`
« Les trois dispersions (interquartile/2, relatives) sont à **deux ordres de »

Les grandeurs chiffrées correspondantes — **2,2–2,3 %** de dépendance au rang et
**1,09 / 1,34 / 2,21 %** de dispersion — sont lues dans
`claude/lectures/requalification-instrument-3d-v2-2026-08-22.json`
(clés `indeterminations.I-q3` et `contrastes.*.dispersion_iqr_demi`).
**Ce sont des incertitudes de RAPPORT, pas d'absolu**, et elles sont portées ici
comme une perturbation **multiplicative de ±2,3 %** appliquée en plus du ±15 %.

> **Ce que `§A63` NE permet PAS de faire ici, dit maintenant** : il n'autorise
> aucune re-lecture des absolus, et son verdict est INDÉTERMINÉ. Les ±2,3 %
> sont donc utilisés comme une **borne empruntée**, non comme un `σ_r` gravé —
> `σ_r` n'est pas gravée et `§8` reste fermé.

---

## §3. LE DOMAINE D'INCERTITUDE BALAYÉ, FIXÉ AVANT CALCUL

| paramètre | domaine | origine |
|---|---|---|
| `I` | **`[0, non-F]`**, balayage fin | l'inconnue de `§A62-6` ; borne haute = le non-F étiqueté de E-6 |
| `non-F` | `× (1 ± 0,15)` puis `× (1 ± 0,023)` | E-5 puis E-7 |
| `C` | `× (1 ± 0,15)` puis `× (1 ± 0,023)` | E-5 puis E-7 |
| `R` | `× (1 ± 0,15)` puis `× (1 ± 0,023)` | E-5 puis E-7 ; volet ajouté par `§A62-bis-5` |
| budgets | les DEUX façons : gravés (16,7 / 33,333) **et** `1000/f` | E-2, `I-r5` |

**Les coins sont balayés exhaustivement** (tous les signes de biais
simultanément), **et** la borne combinée `× (1 ± 0,15)·(1 ± 0,023)` est traitée
comme un cas à part entière. Aucune combinaison n'est écartée parce qu'elle
serait « peu plausible » : un coin qu'on n'énumère pas reste disponible comme
échappatoire tacite (`§A62-2`, `P-d` écrit pour être tué).

---

## §4. LE CRITÈRE D'INSENSIBILITÉ, GRAVÉ AVANT CALCUL

> **La décision CHANGE si, à l'intérieur du domaine du `§3`, le SIGNE de `Δ` ou
> le CÔTÉ RETENU bascule.**

Les deux tests, écrits explicitement pour qu'aucun ne se précise après coup :

- **`T-1` — le signe de `Δ`.** `Δ = 30·(non-F − I)/C` est le travail que le
  30 Hz gagne, en fenêtres·Hz par seconde, sur le 60 Hz. `T-1` **bascule** s'il
  existe un point du domaine où `Δ < 0` et un autre où `Δ > 0`. Le **lieu** du
  basculement (la valeur de `I` où `Δ` s'annule) est reporté, et **le déplacement
  de ce lieu sous les biais est la mesure de la sensibilité à ces biais**.
- **`T-2` — le côté retenu.** L'énoncé de la décision est un **troc** :
  résolution temporelle contre résolution spatiale. `T-2` **bascule** si l'ordre
  des deux `s_max` s'inverse quelque part dans le domaine — c'est-à-dire si le
  60 Hz cesse d'être le côté PLUS PETIT. Là, il n'y aurait plus de troc du tout :
  une cadence dominerait sur les deux axes, et la décision serait tranchée par
  l'arithmétique, pas par le design.

**Aucun autre critère ne sera ajouté après avoir vu les chiffres.** En
particulier, ni les caps, ni la marge de V4, ni l'étendue du côté n'entrent dans
`T-1`/`T-2` — voir `§6`.

---

## §5. LES DEUX BRANCHES, ÉCRITES AVANT LE CALCUL

> **BRANCHE (a) — INSENSIBLE.** Ni `T-1` ni `T-2` ne basculent dans tout le
> domaine du `§3`.
> ⇒ **La cadence est une décision de DESIGN** — résolution temporelle contre
> résolution spatiale — et non un chiffre en attente.
> ⇒ **L'interdiction de `§A61` est RÉTRÉCIE, pas levée** : elle continue de
> bloquer les mesures 3D absolues, mais elle bloque alors des absolus dont
> **aucune décision en attente n'a besoin**. La re-qualification **sort du
> chemin critique** de la cadence.
> ⇒ Ce qui reste dû sur la cadence est une **décision de Romain**, formulée en
> termes perceptifs, pas une mesure.

> **BRANCHE (b) — SENSIBLE.** `T-1` ou `T-2` bascule quelque part dans le
> domaine.
> ⇒ **Nommer QUELLE grandeur** porte le basculement, **à quelle PRÉCISION** il
> faudrait la connaître pour que la décision cesse de basculer, et **dériver
> cette précision de la DÉCISION** — la règle de `§A60` : *l'effet qui compte se
> dérive de la décision, il ne se choisit pas*.
> ⇒ **Ce document NE CHOISIT PAS** entre « le problème est matériel » et « le
> problème est l'estimateur » : `§A63-1` a laissé cette attribution OUVERTE, et
> rien ici ne la referme.
> ⇒ Si la grandeur nommée exige une mesure 3D absolue, dire explicitement que
> **la re-qualification revient sur le chemin critique**, et s'arrêter là.

**Les deux branches finissent au même endroit : `ARRÊT`, remontée à Romain.**
Aucune ne s'enchaîne sur la Porte 3 de `§A58`, ni sur quoi que ce soit d'autre.

---

## §6. CE QUI N'EST PAS UN CRITÈRE, ET QUI SERA POURTANT REPORTÉ

Écrit ici pour que ces chiffres ne puissent pas devenir des critères après coup.
Ils sont **du contexte pour la remontée**, ils ne décident rien :

- les **caps** (fenêtres tenables) des deux côtés, et le fait que V4 en demande
  11 — `§A62` a déjà établi qu'**aucun cap ne l'atteint** au plancher mesuré du
  rendu ;
- l'**étendue du côté** due à l'hypothèse sur `R` (16,4 %), et son invariance à
  `C` (E-4) ;
- l'**écart entre les deux façons de compter** les budgets (`I-r5`).

---

## §7. RÉSULTATS

*(vide — à APPENDRE après le calcul, dans un commit distinct de celui-ci)*
