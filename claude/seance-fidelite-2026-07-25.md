# SÉANCE FIDÉLITÉ — ouverture de l'arc PROJECTION (2026-07-25)

> **Statut : PAPER GRADE, aucun run.** Ouverte sur décision de Romain (2026-07-25,
> après §A35) : arc unique gaté, papier d'abord. Le journal
> `pocCascade2phys/PREREGISTRATION.md` fait foi (fin de fichier : §A35) ; ce document
> raisonne sur lui et ne le remplace pas. La règle D17 s'applique à tout l'arc :
> *aucun diagnostic sans une décision nommée qui en dépend explicitement.*
>
> Étiquetage : `[MESURÉ]` / `[GRAVÉ]` / `[CALCUL]` / `[TRANSPOSITION-HYPOTHÈSE]` /
> `[PROPOSITION]` (position de la session critique, à endosser ou rejeter).

---

## 0. Les gates de l'arc — chacun gaté sur le précédent

| gate | objet | livrable | coût |
|---|---|---|---|
| **P0** | LE CONTRAT : contre quoi la fovéa doit-elle être fidèle | cette séance + décisions | papier |
| **P1** | spec du RENDU MINIMAL D'INSTRUMENTATION (albédo→luminance) | spec paper-grade | papier |
| **P2** | CHIFFRAGE du rendu — première ancre d'un coût jamais mesuré | mesure chrono | court |
| **P3** | TRANSPORT DU PIN à travers le rendu — falsifie l'atténuation de D14 | mesure ABX courte | session |

`[GRAVÉ]` Le budget V4 actuel signifie « ~16.7 ms pour un rendu jamais chiffré »
(état des lieux §4) — P2 est la première ancre de la moitié manquante du budget.
`[GRAVÉ]` Le gate (iii′) est né non mesurable (§A35) — P1–P3 construisent son
instrument. Chaque mesure de l'arc doit changer une décision (D17), et la décision
qu'elle change est nommée dans ce document AVANT qu'elle soit spécifiée.

---

## 1. La question, posée sur les faits gravés

**« Contre quoi la fovéa doit-elle être fidèle ? »** (question ouverte par §A33,
reprise en §A35). Ce qui est acquis au moment de la poser :

- `[MESURÉ, §A34]` Le contrat mesuré par T2 — identité-sous-JND à la vérité f64
  pleine résolution, pondération uniforme — est **ÉCHOUÉ** : production 1.47–2.42
  contre pin 0.0733 (20–33×), 3/3 seeds, instrument validé contre référence.
- `[MESURÉ, §A34]` **Le témoin traverse pour un seed sur trois** (101 : 0.07731) —
  un système SANS fovéa, mono-niveau, f32 propre, franchit le pin. Le contrat
  uniforme n'est pas seulement échoué par l'architecture fovéale : il est
  possiblement insatisfiable par toute implémentation GPU f32, à l'hypersensibilité
  mesurée de l'observable (`[GRAVÉ]` 6e-5 d'état ⇒ 0.077 de Δχ).
- `[CALCUL, séance É2 §2.2 — c dérivé du CFL, physique du substrat, robuste au
  correctif Δx ; ambiguïté d'horizon §3.1 notée et défavorable]` Les trois remèdes
  du halo abolissent l'architecture : recouvrement ×45–×570 d'aire fine (ou +9.8 ms
  = 4.5× la marge V4) ; « bord propre » = le même cas ; bord commis ≈ 9.9 Go/h
  contre 1.17 Ko/commit — le ledger deviendrait proportionnel au temps.
- `[GRAVÉ, §A35]` Les attributions fines (b vs c) sont SUSPENDUES et dormantes ;
  rien dans cette séance ne les consomme ni ne les rouvre.

**La question n'est donc pas « comment réparer la fovéa pour C-uni »** — cette voie
est arithmétiquement fermée — **mais quel contrat un monde fovéé peut honorer sans
cesser d'être une fovéa, et sans trahir la thèse existentielle** (historique
persistant non-contradictoire à compute borné, critère perceptuel jamais L2).

---

## 2. Les quatre contrats candidats

### C-uni — identité-sous-JND à la vérité pleine résolution, uniforme

Le contrat que T2 a mesuré. **ÉCHOUÉ et fermé** (§1). Ce qu'il reste : **l'étalon du
HARNAIS DU REGISTRE** — le rederive f64 reste le juge de la re-dérivabilité des
commits (REGISTRE_FERME, §A13, y tient : k*_chaîne ≤ 256 floats, tous Δt). C-uni
meurt comme contrat de la fovéa, pas comme instrument du ledger.

### C-obs — identité-sous-JND pondérée par l'observateur (= É2-projection, gate iii′)

La comparaison passe par la projection (albédo→luminance) avec pondération
d'excentricité et de distance — la transposition que `r_fovea` devait résoudre
(`[GRAVÉ, §A32-décomp]`), qui a désormais **un consommateur nommé** : sa garde
anti-tapis-roulant se lève le jour où P3 passe, pas avant. Exigences d'instrument :
rendu déterministe de `z`, calibration cycles/degré héritée d'Arc C, capacité à
rejouer les conditions ABX du pin. **Risque nommé, qui voyage (caveat D14)** : si le
transport du pin est ≈ 1 ET la pondération plate, C-obs ≡ C-uni et hérite de son
échec. P3 décide.

### C-ledger — fidélité au TÉMOIGNAGE (re-dérivabilité + non-contradiction)

Le référent n'est plus « le monde vrai » mais **le registre** : (i) tout readout émis
est re-dérivable sous JND depuis le ledger — la PREMIÈRE clause d'É2, **déjà tenue et
mesurée** (`[MESURÉ]` REGISTRE_FERME) ; (ii) rien de montré ne contredit ce qui a été
commis ou perçu. La vérité f64 cesse d'être l'étalon du VIVANT ; elle reste celui du
harnais. C'est la lecture « ledger des témoins » (séance d'idées du 25/07) : la
non-contradiction est relative à ce qui a été observé — ce qui n'a jamais été perçu
peut être re-dérivé librement sans contradiction possible.

**Ce que ça dissout** : la non-commutation `décim∘F ≠ F∘décim` cesse d'être une
faute — deux trajectoires divergentes mais toutes deux cohérentes avec le même
ledger sont ÉGALES au sens du contrat. Le halo n'a plus à reproduire la vérité.
**L'objection la plus forte** : le multi-observateur (v1.1) et le game design
semblent exiger un référent commun (« le barrage a cédé ou non ») — mais **le
registre EST ce référent commun** ; c'est sa raison d'être. À peser en P0-b.

### C-strat — fidélité STRATIFIÉE commis/vivant `[PROPOSITION]`

La synthèse que la session propose, alignée sur la frontière éphémère/persistant
(pilier antérieur à tout échec) :

1. **Le COMMIS est contracté à l'identité-sous-JND** (C-uni restreint au registre) —
   déjà tenu, mesuré, borné (1.17 Ko/commit) ;
2. **Le VIVANT est contracté à C-ledger** : plausible, et non-contradictoire avec le
   commis et le perçu ;
3. **Le juge de (2) est C-obs** : la plausibilité et la non-contradiction se
   constatent À TRAVERS la projection pondérée-observateur — jamais en espace
   d'état, jamais contre f64.

Ce que C-strat **dissout** : le plancher du halo (le vivant n'est plus jugé contre
la vérité pleine résolution). Ce qu'il **ne dissout PAS**, et il faut le dire avec
la même netteté : le vivant doit rester plausible EN VUE, le juge perceptuel
n'existe pas encore (P1–P3), et **un vivant qui diverge visiblement — un pop au
commit, une berge qui saute — reste un échec du contrat**. On échange un critère
échoué contre un critère plus dur à construire, pas contre une absence de critère.

---

## 3. Ce que chaque survivant exige de P1 (le rendu minimal)

Commun à C-obs et C-strat : un rendu **déterministe** albédo→luminance de `z`
(`[GRAVÉ]` « le rendu visé est une fonction déterministe de z » — état des lieux §4 ;
c'est l'anti-PERSIST : `z` stable ⇒ image stable par construction, l'image devient
falsifiable contre l'état). Calibré en cycles/degré (conversion écran d'Arc C,
gravée). **Décision de périmètre pour P1** : l'instrument doit-il projeter la
STRUCTURE RÉELLE (fovéa fine + grossier upsamplé, halo compris) — sans quoi il ne
peut pas juger le vivant fovéal — ou une image pleine résolution suffit-elle pour
P3 (transport du pin) ? Les deux périmètres n'ont pas le même coût ; P1 tranche.

---

## 4. Falsifieurs les moins chers, nommés AVANT toute spec

- **C-uni** : déjà falsifié (§A34). Rien à payer.
- **C-obs / atténuation D14** : **P3 — le transport du pin.** Rejouer les paires ABX
  d'Arc C à travers le rendu minimal ; mesurer si le seuil se transporte (≈ 1 ou
  non). **Les deux branches changent des décisions** : transport ≈ 1 sans
  pondération ⇒ la scission D14 se referme, C-obs hérite de l'échec de C-uni, et
  C-strat reste seul en course ; transport ≠ 1 ⇒ le gate (iii′) reçoit son échelle
  et `r_fovea` son consommateur.
- **C-strat (la strate vivante)** : **l'ABX de substitution** — « monde vécu » vs
  « monde re-dérivé après commit », à travers le rendu, à l'instant d'un commit. Un
  observateur qui détecte la substitution (pop) falsifie la strate. Testable après
  P1 sur le substrat existant, cellule courte. C'est le premier critère de tout
  l'arc qui mesure la thèse existentielle EXACTEMENT là où elle vit : « aucun
  observateur ne peut prendre la substitution en défaut » (§A13-0) — enfin dans
  l'espace où un observateur regarde.

---

## 5. Ce que cette séance NE dit PAS

- Elle **ne choisit pas** le contrat — c'est P0-b, à Romain.
- Elle **ne réhabilite pas** la fovéa telle que construite : sous tout contrat
  survivant, le vivant doit passer un juge perceptuel qui n'existe pas encore.
- Elle **ne prédit pas** le coût du rendu (P2 existe pour ça) et **ne rouvre pas**
  b-vs-c ni (γ) (dormants, règle D17).
- Elle **n'écrit pas** la spec fovéa-z v2 : le contrat d'abord, la spec ensuite.

---

## 6. Décisions pour Romain (P0) — la séance s'arrête ici

- **(P0-a)** Consigner C-uni : **mort comme contrat de fovéa, vivant comme étalon du
  harnais du registre** — en toutes lettres au journal ?
- **(P0-b)** Le contrat de la fovéa : **C-strat** (stratifié — recommandation de la
  session) / **C-obs pur** / **C-ledger pur** / **différer au vu de P3** ?
- **(P0-c)** Lancer **P1** (spec du rendu minimal d'instrumentation) comme prochaine
  étape, avec sa décision de périmètre (structure réelle vs pleine résolution) posée
  en tête de spec ?
- **(P0-d)** Graver l'ouverture de l'arc au journal (**§A36** : gates P0–P3, contrat
  choisi ou différé, falsifieurs nommés, règle D17 rappelée) ?

---

## 7. Contre-argumentaire contre moi-même

**L'objection la plus forte contre C-strat** : il ressemble à « affaiblir le critère
jusqu'à ce qu'il passe » — la manœuvre interdite. Ce qui l'en défend : (1) le pilier
« perceptuel, jamais L2 » et la frontière éphémère/persistant sont **antérieurs à
tout échec** — C-strat est une lecture du corpus, pas une invention post-verdict ;
(2) l'échec de C-uni est **gravé et n'est pas re-lu** ; (3) C-strat ne supprime pas
de juge, il en change : le juge perceptuel-en-vue est **plus dur à construire** que
le Δχ d'instrument, et l'échange « critère échoué contre critère non encore
mesurable » est dit en toutes lettres.

**Ce qui tuerait C-strat** : (i) la non-contradiction au ledger s'avère
non-opérationnalisable en clauses falsifiables (P1 le dira — si la strate vivante ne
produit pas d'ABX de substitution mesurable, elle n'est pas un contrat) ; (ii) le
multi-observateur exige un référent que le ledger ne peut pas fournir (v1.1, non
armé — à consigner comme risque, pas à instruire) ; (iii) P3 révèle que la
projection n'atténue RIEN (transport ≈ 1, pondération plate) ET que l'ABX de
substitution échoue au premier commit — alors le vivant fovéal est perceptuellement
infidèle même sous le contrat le plus favorable, et c'est l'architecture fovéale
elle-même — pas son contrat — qui est réfutée, cette fois sans appel.

Les trois sont falsifiables. Le premier est gratuit (papier), les deux autres
attendent leur gate.
