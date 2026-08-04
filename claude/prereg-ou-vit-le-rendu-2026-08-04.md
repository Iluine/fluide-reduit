# PRÉ-ENREGISTREMENT — OÙ VIT LE RENDU (2026-08-04)

> **Statut : PRÉ-ENREGISTREMENT, écrit AVANT toute lecture.** Le journal
> `pocCascade2phys/PREREGISTRATION.md` fait foi. Ancres au format §A50,
> **contrôlées par `verifier_ancres.py` avant commit**.
> **Ce document se commit SEUL** (règle §A52, sixième application).
> **L'endossement est le commit de Romain ; aucune lecture avant.**

---

## §0. Ce que ce document EST, et ce qu'il n'est pas

C'est le **dû gravé de la porte** : `PREREGISTRATION.md:4142` « *Portes 33.3
(motif T2 : où vit le rendu — à traiter explicitement si ouverte)* ». La porte
est ouverte depuis `§A57` ; le dû est exigible depuis le 19/07.

**Ce n'est PAS le chiffrage du rendu.** Cette dette-là appartient à P2, elle est
ouverte, et la lecture pré-gravée de P2 l'a explicitement **déplacée** vers la
composition et l'optique au-delà de `Y = A` — elle n'a pas rétréci.

**Aucune mesure. Aucun GPU. Aucun `cupy`.** Tout nombre consommé vient d'un
**artefact JSON existant** ou d'une **ancre textuelle lue dans le journal**.
`§A61` — *« aucune mesure 3D absolue avant re-qualification »* — est donc
respecté **par construction**, pas par promesse : le lecteur ne peut pas mesurer,
il n'en a pas les moyens.

Ce que ce document produit : **une PLACE pour le rendu dans le budget, et
l'arithmétique qui en découle**. Placer n'est pas chiffrer, et instruire un dû
n'est pas le trancher.

---

## §1. LE FAIT QUI REND LE DÛ URGENT

`§A58-6` l'a consigné : « *Aucun des cinq caps calculés cette nuit n'inclut le
rendu, et le non-F retenu à 1,835 ms est un non-F 2D à l'échelle de l'instrument,
pas un rendu 3D à 1920.* »

Pire que « n'inclut pas » : **les deux cadences ne le comptent pas de la même
façon.** La table de `§A57-1` porte, dans la même colonne :

| lecture | budget physique | rendu compté |
|---|---|---|
| 60 Hz | 16,7 ms | **zéro** |
| 30 Hz sous la réserve de la porte | 20,33 ms | **~13 ms par fenêtre de 33,3** |

**Deux termes qui ne diffèrent pas seulement par la variable testée.** C'est la
huitième règle (`§A61-2`) appliquée à une comparaison qui n'est pas une mesure :
*l'appariement est la condition d'existence du contraste.* La quatrième
occurrence, si elle passe, serait la première dans de l'arithmétique.

---

## §2. LE MODÈLE, DÉCLARÉ AVANT TOUT NOMBRE

**(M-1) Le rendu tourne à 60 images par seconde dans TOUS les placements
considérés.** C'est le texte même de la porte (`:4218` « rendu 60 fps par
interpolation du readout ») et c'est ce qui rend les placements comparables. Un
placement qui rendrait à 30 fps est un autre objet, non traité ici.

**(M-2) Les temps s'ADDITIONNENT sur le GPU.** C'est un **majorant** du temps
total : `:4218` nomme une « cohabitation GPU », donc un recouvrement, qui ne peut
qu'améliorer. Un majorant sur le temps est un **minorant sur le cap** ⇒ **tout
cap lu ici est PESSIMISTE**, et c'est le sens sûr.

**(M-3) Il y a UN seul GPU** (RTX 3050 Ti Laptop, `tranche-cout-rendu…json`,
`machine.gpu`). Aucun placement « le rendu vit ailleurs » n'existe sur la machine
cible. Cela se dit, cela ne s'implique pas.

**(M-4) Notation.** `R` = coût d'**une image rendue** à 1920×1080 (ms),
**inconnu** ; `I` = coût d'**une interpolation de readout** (ms), **inconnu et
explicitement non gratuit** (`:4219` « design nommé, non décidé, non gratuit ») ;
`nonF` = coût non-F par pas de physique ; `C` = coût d'une fenêtre 64³ au
vocabulaire réel ; `f` = cadence physique.
`nonF` et `C` sont **lus dans les artefacts**, jamais recopiés.

---

## §3. LES PLACEMENTS À TRAITER EXPLICITEMENT

Le dû dit « à traiter explicitement ». Les voici, tous, y compris ceux qui
meurent :

| | placement | ce qu'il coûte par seconde |
|---|---|---|
| **P-a** | **sériel, même cadence** — physique et rendu au même `f`, en série | `f·nonF + 60·R` (à `f = 60`) |
| **P-b** | **porte 33,3** — physique 30 Hz, rendu 60 fps par interpolation du readout | `30·nonF + 60·R + 30·I` |
| **P-c** | **cohabitation concurrente** — même comptabilité que P-a/P-b, recouvrement | ≤ P-a / P-b (M-2) |
| **P-d** | **hors budget** — rendu sur un autre organe | **N'EXISTE PAS** (M-3) |

`P-b` porte `30·I` et non `60·I` : à 30 Hz de physique, **30 des 60 images**
tombent entre deux pas et doivent être interpolées ; les 30 autres tombent sur un
pas. À 60 Hz, aucune ne l'est.

---

## §4. CE QUE LE LECTEUR CALCULE, ET D'OÙ IL LIT

Toutes les entrées sont **lues par la machine** (`§A41`) :

| grandeur | source, lue et vérifiée |
|---|---|
| `C` = coût d'une fenêtre 64³ | `claude/lectures/multiplicateur-c-3d-2026-08-04.json`, `lecture.C_vocabulaire_reel_ms` |
| `nonF` | même artefact, `ancres_lues.non_f_ms` |
| budgets de frame gravés | même artefact, `ancres_lues.seuil_ms` et `budget_30hz_ms` |
| **plancher du gather** (composition) | `claude/lectures/tranche-cout-rendu-2026-08-03.json`, `m2b_loi.deduction_1920x1080_ms` |
| **plancher de l'encodage** (noyau fusionné) | `claude/lectures/p2_chiffrage_rendu.lecture.json`, `cellule_2b.mediane_ms` |
| **l'interdiction pré-écrite de P2** | même artefact, `cellule_2b.lecture.texte` — **recopiée par le LECTEUR, pas par la session** |
| **la réserve de la porte** | `PREREGISTRATION.md:4218`, fragment « cohabitation GPU dans les ~13 ms restants » — **le nombre `13` est EXTRAIT du texte de l'ancre**, non écrit dans le code |

Grandeurs produites :

- `travail(R, I, f)` = fenêtres·Hz réellement disponibles par seconde ;
- `cap(R, I, f)` = fenêtres par pas de physique ;
- `s_max(R, I, f)` à **11 fenêtres** — l'**inversion** de `§A57-3`, jamais une
  constante ; le geste inverse est gravé comme faute :
  `PREREGISTRATION.md:1033` « Choisir `r_fovea` pour que le budget passe =
  fabriquer le verdict » ;
- **`Δ(R, I)` = travail(30 Hz) − travail(60 Hz)** — l'arbitrage de la cadence ;
- ces grandeurs à **`R = R_plancher`** (lu) et à **`R = R_porte`** (extrait de
  l'ancre), pour **encadrer**, jamais pour choisir.

---

## §5. LA PRÉDICTION DE LA SESSION, CONSIGNÉE AVANT LE CHIFFRE

`§A53` a gravé la règle : *consigner la prédiction avant le chiffre, surtout
quand le résultat spectaculaire est l'attendu.* J'ai fait cette algèbre de tête
avant d'écrire le lecteur, et la voici, falsifiable :

> **(1) `R` disparaît de l'arbitrage de cadence.** 60 images sont rendues par
> seconde dans les deux placements, donc `60·R` est le même terme des deux côtés
> et **s'annule exactement** dans `Δ`. La conservation « cellules × cadence » de
> `§A57-3`, lue là-bas comme une quasi-coïncidence à 5,7 %, serait une
> **identité algébrique**, et l'écart résiduel serait exactement `nonF`.
>
> **(2) L'arbitrage entier vaut `Δ = 30·(nonF − I)/C`.** Le 30 Hz gagne les
> `nonF` de 30 pas de physique et paie 30 interpolations. **Point mort à
> `I = nonF`**, soit ≈ 1,835 ms — et `PREREGISTRATION.md:9558` avertit que c'est
> un « non-F **2D à l'échelle de l'instrument** » : le point mort lui-même n'est
> pas ancré en 3D.
>
> **(3) Le rendu ne décide donc PAS la cadence : il décide le CÔTÉ.** De
> `R_plancher` à `R_porte`, `s_max` à 11 fenêtres se déplacerait d'environ
> **52 à 43** — soit ~8 unités de côté, et rien d'autre.

**Si le lecteur contredit l'un des trois, c'est le lecteur qui a raison.**

---

## §6. LES BRANCHES, ÉCRITES AVANT LA LECTURE

> **W-R1 — `Δ` est indépendant de `R`** (annulation exacte, résidu sous 0,5 %).
> Alors **où vit le rendu ne décide pas la cadence** ; le dû de `:4142` est levé
> *quant à la cadence*, et **reporté intact sur le côté de fenêtre**. Ce que la
> cadence attend n'est plus le rendu mais **le non-F 3D** — l'un des deux
> multiplicateurs dus de `§A53`.
>
> **W-R2 — `Δ` dépend de `R`.** Un terme du modèle fait rentrer le rendu dans
> l'arbitrage. Alors **le dû est BLOQUANT pour la cadence**, et le chiffrage de
> P2 (composition + optique R2+) passe devant.
>
> **W-R3 — INDÉTERMINÉ.** Si l'encadrement `[R_plancher, R_porte]` déplace
> `s_max` de moins que la précision de l'instrument (`§A60` : les absolus 3D sont
> pessimistes d'environ 15 %), **aucune lecture de côté n'est prononcée** — la
> question serait sous le bruit, et le dire est le résultat.

---

## §7. LES INDÉTERMINATIONS

**(I-r1) LE PLANCHER N'EST PAS LE RENDU — interdiction héritée, contraignante.**
La lecture pré-gravée de P2 dit, dans l'artefact : *« le coût inconnu du rendu ne
vit PAS dans l'encodage ; il vit dans la COMPOSITION et dans l'optique au-delà de
Y = A (R2+). La dette se DÉPLACE, elle ne rétrécit pas. INTERDICTION PRÉ-ÉCRITE
d'en conclure "le rendu tient". »* Elle s'applique **mot pour mot** ici. Aucune
branche ne peut conclure que le rendu tient, ni qu'il est bon marché.
*Sur le cas favorable* : c'est précisément quand `s_max(R_plancher)` sortira
confortable que cette interdiction mordra. Elle est écrite pour ce moment-là. ✔

**(I-r2) L'APPARIEMENT DU MODÈLE.** Les deux cadences ne doivent différer que par
la cadence. Le lecteur **ne calcule jamais `Δ` à `I = 0` en silence** : `I` reste
symbolique et l'arbitrage est rendu comme **fonction de `I`**, avec son point
mort. Si le lecteur produisait un `Δ` scalaire unique, la quatrième occurrence de
la faute serait consommée ⇒ **INDÉTERMINÉ**.

**(I-r3) `nonF` EST UN CHIFFRE 2D.** L'arbitrage entier lui est proportionnel, et
`PREREGISTRATION.md:9559` le dit déjà : « non-F **2D à l'échelle de l'instrument**,
pas un rendu 3D à 1920 ». L'artefact doit porter cette provenance à côté du
nombre. Un `Δ` publié
sans son étiquette serait un chiffre ancré fabriqué ⇒ **INDÉTERMINÉ**.

**(I-r4) LES PLANCHERS DE RENDU HÉRITENT DE §A61.** Ils ont été mesurés les 26/07
et 03/08 sur **la même machine sans point de fonctionnement stable**. Ce sont des
planchers, et un plancher dont l'instrument dérive de ~3 % sur trois minutes reste
un plancher à cette précision — mais **aucune branche ne peut reposer sur une
distinction plus fine que 15 %** (`§A60`). Le lecteur reporte l'écart et échoue
bruyamment si une branche s'y appuie.

**(I-r5) LE BUDGET GRAVÉ N'EST PAS `1000/f`.** Le corpus grave 16,7 ms à 60 Hz et
33,333 à 30 Hz — soit 1002 et 1000 ms par seconde. L'écart, 0,2 %, est un
arrondi. Le lecteur calcule **des deux façons** et reporte l'écart ; si un verdict
en dépendait, il serait **INDÉTERMINÉ** par construction.

---

## §8. GARDES ANTI-FABRICATION

1. **Aucun nombre de physique n'est écrit dans le lecteur.** Tout vient d'un
   artefact ou d'une ancre. Le lecteur **échoue bruyamment** si une clé manque, si
   un fragment d'ancre a disparu, ou si un nombre extrait d'un texte ne s'y
   retrouve pas. Seules exceptions, déclarées : les constantes de temps
   (1000 ms/s, 60 images/s) et la géométrie 64³.
2. **Le nombre `13` de la porte est EXTRAIT du texte du journal**, pas recopié.
   S'il change au journal, la lecture change ; si le fragment disparaît, le
   lecteur meurt. C'est `§A50` appliqué à un nombre.
3. **L'interdiction de P2 est recopiée par la MACHINE depuis l'artefact** et
   figure telle quelle dans la sortie. Une interdiction que la session
   retranscrirait serait une interdiction que la session peut adoucir.
4. **Aucun seuil n'est posé, donc aucun ne peut être déplacé.** Ce document ne
   grave aucun critère sur la cadence ni sur le côté.
5. Pas d'`assert` nu ; fail-loud partout ; `else` de combinateur = sortie sûre
   (**INDÉTERMINÉ**), jamais un PASS/FAIL par défaut.
6. **`s_max` est rendu comme une INVERSION**, avec son argument, jamais comme une
   constante — l'exigence de forme de `§A58`.

---

## §9. CE QUE CE DOCUMENT NE PRONONCERA PAS

- **Le chiffrage du rendu.** Dette P2, ouverte, déplacée vers la composition et
  l'optique R2+. Ce document dit **où** le rendu vit dans le budget, jamais
  **combien** il coûte.
- **La cadence.** Elle appartient à Romain. Ce document lui rend l'arbitrage
  explicite ; il ne le tranche pas.
- **Le côté de fenêtre.** La porte 3 de `§A58` — re-dériver la monnaie du slot si
  le côté quitte 64 — **n'est pas levée** et ce document ne la lève pas.
- **Aucune rétractation.** `§A53`, `§A55`, `§A57`, `§A59`, `§A60`, `§A61` restent
  ce qu'elles sont.

---

## §10. CE QUI SERA DÛ APRÈS

Nommé d'avance, pour qu'aucun enchaînement ne se fasse en silence :

1. **`I`, le coût de l'interpolation du readout** — inconnu, explicitement non
   gratuit, et **point mort de l'arbitrage de cadence** si la prédiction §5 tient.
2. **Le non-F 3D** — l'un des deux multiplicateurs dus de `§A53`, et la seule
   grandeur qui déplace la cadence sous W-R1.
3. **Le chiffrage de la composition et de l'optique R2+** — la dette P2, que rien
   ici n'avance.
4. Les trois passent **derrière la re-qualification de l'instrument** (`§A61`),
   qui reste devant toute mesure.

---

*Document rédigé par la session Claude, AVANT toute lecture. L'endossement est le
commit de Romain.*
