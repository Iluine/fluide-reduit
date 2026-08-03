# PRÉ-ENREGISTREMENT — RE-QUALIFICATION DE L'INSTRUMENT 3D (2026-08-04)

> **Statut : PRÉ-ENREGISTREMENT, écrit AVANT tout run.** Le journal
> `pocCascade2phys/PREREGISTRATION.md` fait foi. Ancres au format §A50,
> **contrôlées par `verifier_ancres.py` avant commit**.
> **Ce document se commit SEUL** (règle §A52, cinquième application).
> **L'endossement est le commit de Romain ; aucun run avant.**

---

## §0. Ce que ce document EST, et ce qu'il n'est pas

C'est une **qualification d'INSTRUMENT**, pas une mesure d'architecture. La
règle 1 de `CLAUDE.md` — *« G0/G0′ valident l'INSTRUMENT ; C1–C4 valident
l'ARCHITECTURE. Jamais l'inverse »* — s'applique : **rien de ce qui suit ne
prononcera quoi que ce soit sur V4, la cadence, le côté de fenêtre ou le
budget.** Il produit **un chiffre d'appareil**, et rien d'autre.

C'est le **dû neuf et bloquant** de `§A60`.

---

## §1. LE FAIT À EXPLIQUER

`§A60` a mesuré, sur douze points : les grandes fenêtres **accélèrent** pendant
leur série — seconde moitié **12 à 18 %** plus rapide que la première — tandis
que les petites, avec 2 200 à 2 900 frames, sont stables à **1 %**. Le plancher
gravé `src/f1_gpu/chrono.py:3-6` « warmup ≥ 30 frames EXCLU ; série ≥ 300
frames » ne dilue pas ce transitoire sur les kernels 3D.

**Ce qui rend ce protocole possible : le transitoire est OBSERVABLE.**
`nvidia-smi` rend l'horloge SM en direct — **210 MHz au repos, 2 100 MHz au
maximum**, un facteur **10**. Là où `§A60` a *inféré* une montée en fréquence à
partir de temps de frame, ce run la **regarde**.

**Contradiction que le protocole doit trancher, et qui interdit la conclusion
facile** : à `s = 128`, 300 frames durent **17,8 s** et les demi-séries divergent
de 17 % ; à `s = 32`, 2 923 frames durent **2,3 s** et elles s'accordent à
0,03 %. **Un transitoire purement fonction du TEMPS SOUS CHARGE prédirait
l'inverse.** L'hypothèse de la session est donc que ce qui compte est l'**IDLE
QUI PRÉCÈDE** — le driver de `§A60` génère l'état sur CPU entre chaque point
(un `kron` de 362 Mo en f64 au plus grand côté), laissant le GPU retomber à
210 MHz. **Cette hypothèse est écrite ici pour être testée, pas pour être
illustrée.**

---

## §2. LES MESURES

Configuration : celle du vocabulaire réel (`3` scalaires + `2` statiques,
9 champs, 3 fenêtres, 1 système), kernel `substrat_fusionne_3d_param`
**INTOUCHÉ**.

| # | idle préalable | côté | ce qu'elle isole |
|---|---|---|---|
| **M-i1** | **10 s, GPU au repos** | 64 | le transitoire depuis un état froid |
| **M-i2** | **aucun** (enchaîné à M-i1) | 64 | le transitoire disparaît-il si l'on n'idle pas ? |
| **M-i3** | 10 s | 128 | le temps de convergence dépend-il du côté ? |
| **M-i4** | 10 s | 32 | idem, à l'autre bout |

Chaque mesure : **série longue** (≥ 3 000 frames ou ≥ 30 s, le plus grand),
**trace complète des temps de frame** conservée, et **relevé de l'horloge SM**
en parallèle à ~20 Hz.

**Grandeur dérivée — le TEMPS DE CONVERGENCE `T_conv`** : le premier instant à
partir duquel la médiane glissante sur 100 frames reste **à moins de 2 %** de la
médiane du **dernier quart** de la série, et **le reste jusqu'à la fin**. Un
critère qui n'exige pas la permanence se satisferait d'un passage.

---

## §3. LE LIVRABLE : UN WARMUP QUI SE MESURE AU LIEU D'ÊTRE DÉCRÉTÉ

`§A60` a montré qu'un warmup **compté en frames** ne transporte pas d'un kernel
à l'autre. Le livrable n'est donc pas un nombre de frames mais une **règle de
convergence** :

> **Le warmup se termine quand il est MESURÉ terminé** — quand la médiane
> glissante cesse de dériver — exactement comme la série se dimensionne sur la
> durée et non sur un compte (`prereg des paires appariées §2`).

`T_conv` mesuré ici donne la **valeur par défaut** et le **plafond de sécurité**
de cette règle. Le plancher gravé (`≥ 30` frames de warmup, `≥ 300` de série)
n'est **jamais abaissé** : il est **dépassé**.

---

## §4. LES BRANCHES, ÉCRITES AVANT LA LECTURE

> **W-A — l'horloge monte puis plafonne, et `T_conv` est le même en SECONDES
> aux trois côtés.** Le transitoire est le boost, il est fonction du **temps
> sous charge**, et le warmup est une **durée**. La règle de §3 est adoptée avec
> `T_conv` mesuré.
>
> **W-B — `T_conv` diffère selon le côté, ou dépend de l'IDLE PRÉALABLE
> (M-i1 ≫ M-i2).** Le boost n'est pas seul en cause, ou il dépend de l'état
> antérieur. **Conséquence opératoire immédiate** : tout driver doit **ne jamais
> laisser le GPU au repos entre deux points** — ou re-warmer après chaque
> interruption. La règle de §3 tient toujours, mais son plafond doit être pris
> au **pire cas**.
>
> **W-C — aucun transitoire d'horloge observé, et les temps dérivent quand
> même.** Le boost n'est pas la cause. **INDÉTERMINÉ** : aucune règle de warmup
> n'est adoptée avant que la vraie cause soit nommée, et toute mesure 3D reste
> suspendue.

---

## §5. LES INDÉTERMINATIONS

**(I-w1) L'observateur ne doit pas perturber.** Une série est mesurée **avec**
le relevé d'horloge et une autre **sans**, à configuration identique. Si les
médianes diffèrent de plus de **2 %**, l'instrument de mesure de l'instrument
perturbe ce qu'il mesure ⇒ le relevé est **retiré de la chaîne** et seule la
trace des temps est lue.
*Sur le cas favorable* : si le relevé rendait la série plus **rapide**, le
critère mord pareil. ✔

**(I-w2) Le plateau doit être un plateau.** Si `clocks_event_reasons` signale un
throttle **thermique ou de puissance** pendant le dernier quart, le « plateau »
n'en est pas un et `T_conv` ne décrit rien de stable ⇒ **INDÉTERMINÉ**, et la
question devient celle d'une dérive longue, qu'aucune de nos séries n'a jamais
cherchée.

**(I-w3) L'absence de transitoire ne se prononce pas sur un seul essai.** Si
`M-i2` (sans idle) ne montre **aucun** transitoire, c'est le résultat
**commode** — il suffirait d'enchaîner les mesures. Il ne sera retenu que
**reproduit** : M-i2 est exécutée **deux fois**, et les deux doivent conclure de
même.
*C'est le critère braqué sur le cas favorable de ce protocole.* ✔

**(I-w4) Cohérence avec `§A60`.** La première moitié de `M-i1` doit retrouver
l'ordre de grandeur des demi-séries lentes de `§A60` (≈ 7,45 ms à 64³) et le
plateau celui des demi-séries rapides (≈ 6,12 ms). Si le run ne **reproduit pas
le phénomène qu'il vient expliquer**, il n'explique rien ⇒ **INDÉTERMINÉ**.

---

## §6. GARDES ANTI-FABRICATION

1. **`src/f1_gpu/chrono.py` n'est PAS modifié.** Il grave « AUCUN timestamp dans
   les données retournées » ; la trace temporelle est reconstruite **hors de
   lui**, par somme cumulée des durées de frame qu'il rend déjà. L'instrument
   verdict-grade reste intouché par une enquête sur lui-même.
2. **Aucun plancher n'est abaissé.** `≥ 30` warmup et `≥ 300` série tiennent ;
   ce document ne peut que les **relever**.
3. **Les kernels sont intouchés** (`git diff` vide, empreintes relues).
4. **Les traces complètes sont conservées dans l'artefact** — pas seulement les
   statistiques. Un transitoire est une **forme** ; le résumer en scalaire avant
   de l'avoir regardé serait perdre l'objet.
5. **L'hypothèse de la session (l'idle préalable) est TESTÉE par M-i2, pas
   illustrée.** Le protocole prévoit explicitement qu'elle soit fausse (W-A).
6. Pas d'`assert` nu ; fail-loud ; `else` de combinateur = INDÉTERMINÉ.

---

## §7. CE QUE CE RUN NE PRONONCERA PAS

- **Rien sur V4, la cadence, le côté, le budget.** C'est de l'appareil.
- **Rien sur la correction des mesures passées.** `§A60` a établi que les
  absolus 3D sont pessimistes d'environ 15 % et que les rapports **pourraient**
  l'être ; **ce run ne corrige rien rétroactivement** — il rend l'instrument
  utilisable pour la suite. Re-mesurer `ρ` sous le nouveau warmup serait un
  **autre protocole**, avec son propre pré-enregistrement.
- **Rien sur le rendu**, dû de `PREREGISTRATION.md:4142` « Portes 33.3 (motif
  T2 : où vit le rendu — à traiter explicitement si ouverte) », qui reste devant.

---

## §8. CE QUI SERA DÛ APRÈS

La question ouverte par `§A60` et **non refermée par ce run** : `ρ = 2,03`
est-il biaisé ? Elle demande une **re-mesure de §A53 sous l'instrument
re-qualifié**, en 2D **et** en 3D, avec son propre pré-enregistrement. Ce
document ne fait que rendre cette re-mesure **possible**.

---

*Document rédigé par la session Claude, AVANT tout run. L'endossement est le
commit de Romain.*
