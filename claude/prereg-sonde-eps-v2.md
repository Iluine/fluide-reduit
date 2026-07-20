# Pré-enregistrement — SONDE EPS (N1), version 2

**Statut : REMONTÉ POUR ENDOSSEMENT. Aucun run avant.**
Source : pocCascade2phys PREREGISTRATION.md §A21-complément (commit f943e9a).
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
**seuil** ou le **retard** — distinction que l'attribution (a), nommée et non
armée, demanderait.

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

## 5. Vérification d'instrument et table de branches, reconduites

Le même balayage est d'abord joué à **k=1**, et `lecture_mecanique` refuse de
produire la lecture k=4 sans son PASS/FAIL enregistré (câblage B9). Une sonde
muette se détecte avant, pas après (§A14).

| | |
|---|---|
| **(i)** | k=1 discrimine **et** au moins un EPS passe ⇒ **INSTRUMENT VALIDE** |
| **(ii)** | k=1 ne discrimine pas ⇒ **SONDE MUETTE sur EPS** — ne rien régler, remonter |
| **(iii)** | k=1 discrimine mais **aucun** EPS ne passe ⇒ **MÉCANISME DE REMONTÉE EN QUESTION**, avec ses deux attributions nommées et non armées |

*Nota* : à k=1 subsiste la frame de retard du compteur (choix 6) — la
vérification teste la discrimination sous péremption **minimale, pas nulle**.

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

**Endosser ce pré-enregistrement**, ou le renvoyer amendé. Sur endossement,
`ENDOSSEMENT_PREREG_V2` est levé et le balayage peut tourner.

Deux conséquences à connaître avant de trancher, ni l'une ni l'autre traitée
ici :

1. **`run_f1_attribution_b.py` reste gaté** — il est bâti sur l'observable v1
   et son verrou d'invariant refusera de prononcer. Le ré-armer sur
   l'observable corrigé est une décision distincte.
2. **La lecture D-2 gagne en cohérence** : l'option 1 supprime la capture de
   coefficients dont la v1 avait besoin, ce qui va dans le sens des +0.822 ms
   attribués à l'appareil de sonde. Ce n'est pas une re-mesure, seulement une
   cohérence à noter.
