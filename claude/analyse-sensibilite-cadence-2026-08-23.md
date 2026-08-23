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

## §7. RÉSULTATS — **BRANCHE (b) : SENSIBLE**, et la sensibilité tient dans UNE grandeur

Lecteur : `lire_sensibilite_cadence.py`. Artefact :
`claude/lectures/sensibilite-cadence-2026-08-23.json`.
**Aucune mesure, aucun GPU, aucun `cupy`.** Balayage : **343 coins × 3 671
valeurs de `I` × 2 façons de compter les budgets.**

### §7-0 — DEUX GARDES ONT MORDU AVANT TOUT RÉSULTAT

**(1) LE §3 ÉCRIVAIT UNE BORNE TROP ÉTROITE.** Le `§3` posait `± 2,3 %` de
dépendance au rang ; l'artefact de `§A63` porte **2,3348 %** (`I-q3`, 64→128).
La borne est désormais **LUE dans l'artefact**, plus écrite à la main — le
domaine balayé est **ÉLARGI**, jamais rétréci. Arrondir 2,3348 vers 2,3 après
l'avoir lu aurait été rétrécir l'incertitude pour se faciliter la conclusion.
La garde qui l'a vu est un `_exiger`, pas une relecture.

> **Cette borne reste EMPRUNTÉE, et le document le redit ici** : `σ_r` n'est
> pas gravée, `§8` de la re-qualification reste FERMÉ (`§A63`). 2,3348 % est
> une dépendance au rang **INEXPLIQUÉE**, pas une incertitude qualifiée.

**(2) LE TÉMOIN DE NON-RÉGRESSION A ATTRAPÉ UNE FAUTE DE MA PROPRE ALGÈBRE.**
Le lecteur meurt si son modèle ne reproduit pas les quatre grandeurs de
l'artefact `I-r5`. Il est mort une fois — l'étendue du côté rapportée au côté
BAS au lieu du côté HAUT. Corrigé, les quatre grandeurs se reproduisent à
**écart relatif 0,000·10⁰ exactement** (`Δ`, `s_max` 60 Hz, étendue, point mort).
*Sans ce témoin, le balayage aurait tourné sur un modèle inventé.*

### §7-1 — `T-2` NE BASCULE NULLE PART : le troc EXISTE dans tout le domaine

| façon de compter | `s30/s60` sur tout le domaine | `T-2` bascule ? |
|---|---|---|
| budgets gravés (16,7 / 33,333) | **[1,2544 ; 1,2908]** | **non** |
| `1000/f` | **[1,2554 ; 1,2918]** | **non** |

Le 30 Hz achète **toujours** un côté **25 à 29 % plus grand**, dans les 343
coins et à toutes les valeurs de `I` balayées. **Le troc « résolution
temporelle contre résolution spatiale » n'est jamais annulé par
l'incertitude** : aucune cadence ne domine l'autre sur les deux axes.

### §7-2 — `T-1` BASCULE, ET C'EST CE QUI TIRE LA BRANCHE (b)

| façon de compter | `Δ` sur le domaine (fenêtres·Hz) | lieu du croisement `Δ = 0` | coins où il tombe dans `[0, non-F]` |
|---|---|---|---|
| budgets gravés | **[−5,6277 ; +31,1340]** | `I` ∈ [**1,4567** ; **1,8112**] ms | 245 / 343 |
| `1000/f` | **[−4,6360 ; +32,1257]** | `I` ∈ [**1,5234** ; **1,8350**] ms | 196 / 343 |

Le signe de `Δ` change à l'intérieur du domaine : il existe des points où le
30 Hz délivre plus de travail et d'autres où c'est le 60 Hz. **`T-1` bascule
⇒ branche (b).**

**Les DEUX façons de compter tirent la MÊME branche** ⇒ la clause `I-r5` du
`§2` E-2 **ne rend pas la lecture INDÉTERMINÉE**. L'arrondi 1002/1000 déplace
le croisement de **3,63 %** — exactement ce que `§A62-bis-2` avait établi pour
le point mort — sans toucher la branche.

### §7-3 — CE QUI NE COMPTE PAS, MÉCANISÉ PLUTÔT QU'AFFIRMÉ

| énoncé | vérification | résultat |
|---|---|---|
| `C` ne touche ni `T-1` ni `T-2` | signe de `Δ` invariant ; `s30/s60` invariant | écart **4,4·10⁻¹⁶** |
| `R` s'annule exactement de `Δ` | `Δ` invariant sous ±17,6 % sur `R` | écart **1,1·10⁻¹³** |
| `R` touche `s30/s60` | chiffré au lieu d'être tu | **2·10⁻⁴** — sans jamais approcher 1 |

`C` est **exactement neutre** : diviseur positif de `Δ`, et il s'annule du
rapport des deux côtés parce que les deux le portent (E-4). ⇒ **La maladie
d'instrument de `§A61` ne peut PAS, par `C`, faire basculer la décision de
cadence.** `R` s'annule de `Δ` — c'est l'annulation de `§A62-3`, ici
re-vérifiée sous biais et non plus seulement à valeur nominale.

> ⇒ **Sur les trois grandeurs de la question du `§1`, DEUX sont hors de cause.
> `Δ ∝ (non-F − I)` : décider la cadence sur `T-1`, c'est décider le signe de
> `non-F − I`, et rien d'autre.**

### §7-4 — LA GRANDEUR NOMMÉE ET SA PRÉCISION, **DÉRIVÉE DE LA DÉCISION**

Ce que le `§5` branche (b) exige, et la règle de `§A60` : *l'effet qui compte
se dérive de la décision, il ne se choisit pas.*

**Grandeur : `I`, le coût d'une interpolation de readout — rapporté au non-F**
qui porte l'étiquette de `PREREGISTRATION.md:9559`
« non-F **2D à l'échelle de l'instrument**, pas un rendu 3D à 1920. »

**Lecture en ABSOLU** — si `I` et le non-F sont connus séparément, chacun
portant `± 15 %` (`§A60`) combiné à la borne de rang :

| façon de compter | signe DÉCIDÉ si | zone INDÉCIDABLE |
|---|---|---|
| budgets gravés | `I` < **1,4567 ms** ou `I` > **2,0929 ms** | large de **0,6362 ms** |
| `1000/f` | `I` < **1,5234 ms** ou `I` > **2,1596 ms** | large de **0,6362 ms** |

**Lecture en RAPPORT** — si `I` et le non-F sont mesurés **adjacents dans le
temps**, un biais multiplicatif commun s'annule du rapport :

> **Le signe est DÉCIDÉ si `|I/non-F − 1| > 2,335 %` ; INDÉCIDABLE dans
> `[0,9767 ; 1,0233]`.**

**La décision n'a donc pas besoin de deux absolus : elle a besoin d'un
RAPPORT.** C'est exactement la classe que `§A61` laisse « probablement saine,
**à ÉTABLIR, pas à supposer** » — et que `§A63` a précisément **échoué** à
qualifier.

**Et la précision requise, 2,335 %, EST le chiffre que `§A63` a mesuré comme
dépendance au rang inexpliquée.** La décision de cadence se joue donc
**exactement au bord de ce que l'instrument sait aujourd'hui rendre**. Ce
n'est pas une coïncidence commode : c'est la même borne des deux côtés, parce
que c'est la même machine.

### §7-5 — CE QUE ÇA COÛTE, ET CE QUE ÇA NE TRANCHE PAS

- **En absolu** : une mesure 3D ABSOLUE — **INTERDITE par `§A61`** tant que
  l'instrument n'est pas qualifié.
- **En rapport** : un contraste apparié intra-run — la classe que la
  re-qualification devait qualifier et **n'a pas qualifiée** (`§A63`,
  INDÉTERMINÉ global, `σ_r` non gravée).

> ⇒ **Dans les deux cas, la re-qualification REVIENT sur le chemin critique de
> la cadence.** La branche (a) — « la re-qualification sort du chemin
> critique » — **n'est pas tirée.**

**Ce document NE TRANCHE PAS** entre « le problème est matériel » et « le
problème est l'estimateur » : `§A63-1` a laissé l'attribution **OUVERTE**, et
rien ici ne la referme. Il ne prononce pas non plus sur la Porte 3 de `§A58`.

### §7-6 — LE CONTEXTE DU `§6`, REPORTÉ ET NE DÉCIDANT RIEN

Ces chiffres étaient nommés au `§6` **avant** d'être vus, précisément pour
qu'ils ne puissent pas devenir des critères.

| | 60 Hz | 30 Hz à `I` = 0 | 30 Hz au point mort |
|---|---|---|---|
| `s_max` au plancher mesuré du rendu | **51,99** (invariant à `I`) | **66,82** | **65,45** |
| cap, plancher mesuré | **5,90** | 12,52 | 11,76 |
| `s_max` sous la réserve gravée | **43,46** | 56,62 | 54,68 |
| cap, réserve gravée | 3,44 | 7,61 | 6,86 |

**V4 en demande 11.** L'interdiction héritée de P2 continue de mordre : le
plancher de rendu est confortable et **ne sauve rien**.

**Part relative de `Δ`** : sur tout le domaine, `Δ` va de **−1,59 %** à
**+8,80 %** du travail du côté 60 Hz — le « 6,2 % » de `§A62-3` en est la
valeur **nominale à `I` = 0**, désormais encadrée. **Ce n'est pas un critère**
et ce document ne l'a pas transformé en seuil : `T-1` est un test de SIGNE,
et il a été écrit avant que ces bornes existent.

> **AMENDEMENT (2026-08-23, relecture adverse de Romain) — L'ÉTIQUETTE
> MANQUAIT À LA BORNE HAUTE, ET C'EST LA FAUTE `§A62-bis-4`, COMMISE DANS LE
> DOCUMENT QUI LA NOMME COMME CELLE À NE PAS REFAIRE.**
> Régime de correction à la CLAUSE (`§A47-PRÉCISION-2`) : la ligne ci-dessus
> n'est pas réécrite, elle est amendée ici, datée et visible. **Aucun chiffre
> ne change ; ce qui change est ce qu'un lecteur peut en conclure.**
>
> Le **+8,80 %** n'est pas une valeur du domaine parmi d'autres : il est
> atteint **exactement à `I` = 0**, au coin `non-F` = +15 %·+2,3 % et
> `C` = −15 %·−2,3 % (`R` y est indifférent, puisqu'il s'annule de `Δ`). Or
> `I` = 0 est **gravé impossible** par le corpus :
> `PREREGISTRATION.md:4219` « par fenêtre [design nommé, non décidé, non gratuit]. »
> Et `Δ` est **monotone décroissante en `I` sur les 343
> coins** (vérifié dans l'artefact) : **tout le haut de la fourchette est la
> limite `I → 0`**, donc inatteignable.
>
> ⇒ **La fourchette est ASYMÉTRIQUE, et son extrémité favorable est
> INACCESSIBLE.** La borne basse **−1,59 %**, elle, est atteinte à
> `I` = non-F gravé — une valeur que rien n'interdit. Écrire
> « `Δ` ∈ [−1,59 % ; +8,80 %] » sans cette étiquette **sur-vend le côté
> favorable**, exactement comme « ~67³ » sur-vendait le côté 30 Hz.
>
> **Ce que l'amendement NE fait PAS** : il ne nomme aucune cadence préférable
> et n'ajoute aucun critère. `T-1` reste un test de signe, et le `§6` reste
> une section qui ne décide rien. **La cadence appartient à Romain.**

---

## §8. ARRÊT

**Branche (b) tirée. STOP.** Rien ne s'enchaîne : ni la Porte 3 de `§A58`, ni
un cinquième protocole de qualification, ni une mesure de `I`. La cadence
appartient à Romain (`§A62`), et ce document ne fait que lui rendre la forme
exacte de ce qui manque.
