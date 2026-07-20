# Pré-enregistrement — SONDE EPS (N1), version 2

**Statut : ENDOSSÉ §A22 sous trois amendements, INTÉGRÉS ci-dessous.
REMONTÉ pour endossement du texte FINAL. Aucun run avant.**
Sources : pocCascade2phys PREREGISTRATION.md §A21-complément (f943e9a) et
**§A22 (a0a305a)** — amendements (A) critère de discrimination et plancher
de bruit, (B) branche (iii-bis) péremption, (C) la discrimination ne gate
qu'en cas d'échec.
Verrou mécanique : `ENDOSSEMENT_PREREG_V2 = False` dans
`scripts/run_f1_sonde_eps.py` — le driver refuse fail-loud avant même de
toucher au device. Le lever est la décision d'endossement.

---

## 1. Pourquoi un pré-enregistrement NEUF et non un amendement

L'ancien (§A19) est **superseded**. Il n'est pas corrigé sur les bords :
son observable était faux.

Le reconstructeur entretenait une copie CPU **indépendante**, alimentée par
les seuls coefficients reçus. Il n'appliquait ni le *roll* ni les colonnes
prédites que le device écrit dans `references` à chaque déplacement de
fovéa (E4c, 1 cellule fine/frame). La sonde comparait donc des régions
**spatialement décalées**, d'un décalage qui croissait avec le balayage.

C'est cela — et non une infidélité du grossier — qui produisait le plancher
0.082, insensible à EPS comme à k, resté inexpliqué pendant toute la série
D-1/D-2. **Aucune lecture de la v1 n'est reconduite** : ni le plancher, ni
la branche (ii), ni la comparaison restreinte.

---

## 2. Ce que la sonde compare désormais

| | |
|---|---|
| **VÉRITÉ** | niveau 0 par décimation exacte (moyennes Harten, `multiresolution.downsample`) de l'état fin courant |
| **CONNAISSANCE DU CPU** | **lue sur `reference`** (option 1, tranchée par Romain) |
| **OBSERVABLE** | Δχ readout — `albedo(S_HALF_OP)` puis `delta_chi(...)["max_carrier"]`, primitives existantes, espace instrument, jamais l'état (règle 2) |

Le schéma B4 pose que `reference` **est** le modèle de ce que le CPU sait :
le kernel émet `d = etat − reference`, transmet `d`, puis fait
`reference += d`. Depuis le ping-pong (§A21), tout ce qui est émis est
transféré une fois et une seule — l'identité est donc exacte, à une frame
près. **Une copie indépendante peut diverger du device ; une lecture directe
ne le peut pas.** C'est tout l'objet du correctif.

### L'instant de lecture est gravé

La connaissance est capturée **après le déplacement et avant l'émission** de
la frame. C'est le **seul** instant où `reference` porte la connaissance du
CPU *dans les coordonnées de la frame courante*. Lue plus tôt, elle est dans
les coordonnées de la frame précédente ; lue plus tard, elle contient déjà
ce que le CPU ne recevra qu'à la frame suivante.

C'est précisément la confusion de ces instants qui a coûté la journée.

---

## 3. Les deux vérités — déclarées AVANT le run

Consigne gravée : *à la frame n, le CPU connaît l'état de n−1.* Contre quelle
vérité comparer doit donc être dit d'avance. **Les deux sont déclarées et
toutes deux reportées** ; le surcoût est marginal (un champ capturé de plus,
une évaluation de Δχ de plus, sur la même trajectoire).

### vérité(n) — K(n) contre S(n) — **PORTE LA RÈGLE**

Ce que le joueur voit : canal **et** péremption acceptée.

**Justification.** La règle Q2 décide d'**innocuité perceptuelle**. Or le
joueur ne perçoit pas un canal — il perçoit un écart à l'instant présent.
Lire la règle sur la vérité transportée reviendrait à s'accorder gratuitement
la frame de retard que le joueur, lui, subit. Un seuil doit se juger sur ce
qui est vu.

### vérité(n−1) — K(n) contre S̃(n−1) — **DIAGNOSTIC**

L'état d'où la connaissance provient, transporté dans les mêmes coordonnées.
Isole la fidélité du **canal seul** (seuil, cadence, transfert), sans la frame
de retard. **Jamais la règle.**

### Leur écart

C'est le **prix de la frame de retard**, reporté comme tel
(`prix_peremption_une_frame`). En cas d'échec, il dit s'il faut incriminer le
**seuil** ou le **retard**. C'est ce discriminateur que l'amendement (B)
câble à la table, sous la branche **(iii-bis)** — et c'est par là que
l'attribution (a) se trouve **armée** (§ 5).

---

## 4. Règle de décision (Q2), reconduite

> **EPS retenu = le PLUS GRAND EPS dont le Δχ MAX de série, lu sur vérité(n)
> et à l'échelle DÉCIMÉE du niveau 0, reste < 0.0603 (ic_bas)** — lecture sur
> l'IC entier, discipline §A13.

Si aucun EPS du balayage ne satisfait, 1e-5 compris : **AUTRE remonté, aucun
EPS retenu par défaut.** Jamais de choix après courbe.

- **Échelle épinglée** : la règle lit le Δχ *décimé* — le grossier comparé à
  ce que le grossier devrait être, à sa propre résolution. Le non-décimé est
  reporté en **diagnostic seul** ; la relation n'est pas monotone (décimer
  déplace les bandes porteuses que `delta_chi` seul lit).
- **Logique du critère**, inchangée et toujours un fait de logique : le joueur
  ne voit jamais la référence. Δχ < 0.0603 ⇒ **innocuité ÉTABLIE** ;
  Δχ > 0.0603 ⇒ **innocuité NON ÉTABLIE** — et **jamais** « nocivité établie ».

---

## 4bis. (A) Critère de discrimination et plancher de bruit

Le critère entre au pré-enregistrement, et il a **deux conditions** :

| | |
|---|---|
| **FORME** | amplitude **relative** du Δχ max sur le balayage ≥ **5 %** |
| **ÉCHELLE** | amplitude **absolue** au-dessus d'un plancher de bruit **mesuré par réplicat** |

**Le plancher est mesuré, pas supposé.** Le même EPS — celui *en vigueur*,
donc pas un choix — est rejoué à l'identique : graine, géométrie, cadence,
série. Tout ce qui sépare les deux passes est du bruit par définition. Le
plancher retenu est le plus grand écart observé **sur les deux vérités**, la
règle lisant l'une et (iii-bis) l'autre. Coût : une passe Δχ sur douze.

**Pourquoi le seuil relatif de 5 % est conservé, et pourquoi c'est un choix.**
Une amplitude relative est **sans échelle** : elle ne devient pas fausse
quand l'observable passe de ~0.082 (artefact) à ~1e-4 (attendu). Ce qui la
rendait insuffisante, c'est qu'elle **statuait seule** — 5 % de 1e-4 valent
5e-6, et rien ne disait si 5e-6 était du signal ou du bruit. Le défaut
n'était pas sa valeur, c'était l'absence de plancher.

En choisir une autre aujourd'hui reviendrait à la choisir *en connaissant
l'échelle de l'observable* : un seuil formé avec la donnée en vue. On garde
donc le critère de forme tel quel, et on lui adjoint un critère d'échelle qui,
lui, est mesuré.

**Si le plancher ressort nul**, cela dit que la chaîne est déterministe, pas
qu'elle est infiniment précise : le critère relatif porte alors seul, et le
driver le dit plutôt que d'en profiter. **Sans plancher évalué**, la
discrimination reste *indéterminée* — jamais prononcée sur la forme seule,
qui fut la faute de la v1.

---

## 5. Table de branches et vérification d'instrument

Le même balayage est d'abord joué à **k=1**, et `lecture_mecanique` refuse de
produire la lecture k=4 sans son PASS/FAIL enregistré (câblage B9).

### (C) La discrimination ne gate qu'en cas d'échec

Si **au moins un EPS passe** sur vérité(n), l'instrument **suffit** pour cette
conclusion : un observable qui reste sous le seuil est innocent quelle que
soit sa pente. La discrimination devient alors un **diagnostic**. Elle ne
**gate** que lorsque rien ne passe — le seul cas où il faut savoir si l'on
mesure quelque chose, et c'est là que (ii), (iii) et (iii-bis) se séparent.

### La table, close sur quatre issues

| | condition | lecture |
|---|---|---|
| **(i)** | un EPS passe sur vérité(n) | **INSTRUMENT VALIDE** |
| **(iii-bis)** | aucun sur vérité(n), au moins un sur vérité(n−1) | **PÉREMPTION** |
| **(iii)** | aucun sur les deux, mais le balayage discrimine | **MÉCANISME DE REMONTÉE EN QUESTION** |
| **(ii)** | aucun sur les deux, pas de discrimination | **SONDE MUETTE sur EPS** |

Exhaustives et mutuellement exclusives ; un test de couverture exerce les
quatre.

### (B) La branche (iii-bis) — PÉREMPTION

Retirer la seule frame de retard suffirait à satisfaire la règle : **la cause
est le retard, pas le mécanisme.** Le mécanisme de remontée est **exonéré**,
et la décision passe à k — donc au **réveil σ_ω (R4)**, qui appartient à
Romain. Le driver le reporte et ne réveille rien.

Le discriminateur est `prix_peremption_une_frame`, déjà reporté par frame :
l'écart entre les deux vérités **est** le prix du retard.

**Ordre remonté comme choix.** Le gravé dit que (ii), (iii) et (iii-bis) se
séparent sous le gate, sans fixer leur ordre. (iii-bis) est évaluée **avant**
la discrimination, parce qu'elle repose sur une **mesure directe** — le prix
du retard — et non sur la pente du balayage. Une sonde peut être muette sur
EPS tout en mesurant parfaitement ce prix : les deux axes sont indépendants,
et « la cause est le retard » est plus informatif que « sonde muette ».

### L'attribution (a) est désormais ARMÉE

Son falsificateur gravé était « comparer contre la vérité **décalée d'une
frame** » — c'est exactement vérité(n−1), que la v2 reporte à chaque frame.
**La branche (iii-bis) est sa lecture pré-écrite** : (a) n'a plus besoin d'un
bras dédié, elle est tranchée par la table. L'attribution (b) reste **non
armée**, son bras restant gaté.

*Nota* : à k=1 subsiste la frame de retard du compteur (choix 6) — la
vérification teste sous péremption **minimale, pas nulle**. C'est précisément
ce qui rend (iii-bis) lisible à k=1 : la seule péremption restante **est**
cette frame de retard.

**Ce n'est pas un balayage de k** : deux valeurs seulement, k=4 (mesure, figée)
et k=1 (vérification). Toute décision sur k reste gatée σ_ω (R4).

---

## 6. Portée — ce que la sonde NE mesure PAS

Le device inscrit dans `reference` les colonnes qu'il a **prédites GPU-side**.
Les lire, c'est donc **créditer le CPU d'une prédiction identique à celle du
device**.

> La sonde mesure la fidélité du **CANAL** — seuil, cadence, transfert — et
> **non** celle de la prédiction propre au CPU.

C'est une limite déclarée, pas un détail. Mesurer la seconde exige le **miroir
CPU (option 2)**, nommé comme travail de **production** et délibérément non
construit. **Aucun résultat de cette sonde ne pourra s'énoncer « le lointain
est fidèle », seulement « le canal l'est ».** La limite est portée dans la
lecture elle-même (`portee_option_1`), pas seulement dans ce document — elle
doit voyager avec le résultat.

---

## 7. Le verrou, et ce qu'il a déjà trouvé

`test_verrou_la_connaissance_lue_egale_le_grand_livre_applique` échoue si la
connaissance lue s'écarte de ce que le CPU obtiendrait en appliquant les
coefficients reçus. **C'est l'identité qui n'existait pas en v1**, et dont
l'absence a produit le faux verdict.

Il a mordu immédiatement, sur le correctif ping-pong lui-même : le ping-pong
déplaçait les **données** d'un tour à l'autre sans déplacer leur **offset
d'indices**, si bien que les coefficients de n−1 étaient réindexés par
l'offset de n — écriture silencieuse dans le mauvais slot. L'offset relève du
même invariant que la taille et les données ; il est désormais confié au
compacteur (`noter_offset` / `offset_precedent`) et voyage avec elles.

Un second test exhibe le défaut de v1 lui-même (dès que la fovéa bouge, la
reconstruction indépendante s'écarte de `reference`) : sans lui, on pourrait
croire la v1 seulement imprécise.

---

## 8. Ce qui est figé tant que rien n'est endossé

- **EPS reste 1e-4, k reste 4, rien n'est réglé.**
- Kernels F et L3 **intouchés** dans leur code CUDA. Empreintes sur la chaîne
  extraite, re-calculables :
  - `sha256(substrat_fusionne._SOURCE)` = `e18015f5…f30b4` (8223 car.)
  - `sha256(substrat_l3._SOURCE_L3)` = `9533a130…781a` (9635 car.)
- `ReconstructeurNiveau0` est conservé mais marqué **SUPERSEDED** : des
  lectures acquises en dépendent, et on ne réécrit pas le passé.

---

## 9. Ce qui est demandé

**Endosser ce texte final**, ou le renvoyer amendé. Les trois amendements de
§A22 y sont intégrés (§ 4bis pour A, § 5 pour B et C). Sur endossement,
`ENDOSSEMENT_PREREG_V2` est levé — par Romain, pas par moi — et le balayage
peut tourner.

Trois conséquences à connaître avant de trancher, aucune traitée ici :

1. **`run_f1_attribution_b.py` reste gaté** — il est bâti sur l'observable v1
   et son verrou d'invariant refusera de prononcer. Le ré-armer sur
   l'observable corrigé est une décision distincte.
2. **L'attribution (a) n'a plus besoin de son bras** : (iii-bis) la tranche
   depuis la table. C'est un bras d'économisé, pas un bras d'ajouté.
3. **La lecture D-2 gagne en cohérence** : l'option 1 supprime la capture de
   coefficients dont la v1 avait besoin, ce qui va dans le sens des +0.822 ms
   attribués à l'appareil de sonde. Ce n'est pas une re-mesure, seulement une
   cohérence à noter.

### Un choix, et une chose que je n'ai pas su trancher seul

- **Choix remonté** : l'ordre d'évaluation de (iii-bis) avant la
  discrimination (§ 5). Le gravé fixe que les trois se séparent sous le gate,
  pas leur ordre.
- **Point à surveiller à la lecture** : le plancher mesuré par réplicat n'a
  qu'**un** réplicat. S'il ressort très petit devant l'amplitude du balayage,
  la question ne se pose pas. S'il en est du même ordre, un réplicat unique ne
  suffira pas à trancher, et il faudra en jouer plusieurs — décision qui
  appartiendra à Romain à la lecture, pas au driver.
