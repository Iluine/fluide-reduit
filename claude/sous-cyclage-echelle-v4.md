# Paper-grade — sous-cyclage à l'échelle V4

**Paper-grade. Aucun run, aucune ligne de code.**
Source : pocCascade2phys PREREGISTRATION.md §A27 (commit 210a17b). T1 est
SÉPARABLE (mesure de temps, tranche-1 à l'échelle V4) ; ma conclusion « T1
vacant » du §A26 est corrigée. La question du sous-cyclage, prématurée d'un
cran au §A26, est maintenant décisive : tranche-1 tourne à l'échelle V4, où le
niveau fin est potentiellement sub-résolution.

> **La réponse en une phrase, et elle sauve le budget — sous condition
> gravée** : le multiplicateur sur F dépend d'UN paramètre,
> `n_9 = dt_frame / dt_CFL(fin)`. Si le niveau fin est à la résolution de la
> PHYSIQUE (dx=1, le substrat), alors `n_9 = 0,08–0,24 < 1` : **aucun
> sous-cyclage, M = 1, V4 survit**. Mon 100–256× était une erreur
> structurelle ; le ~5× de Romain est la bonne règle appliquée à une fovéa
> sur-résolue qui n'est pas justifiée. La seule question qui reste est de
> spec : le niveau fin descend-il SOUS la physique ? S'il le fait, V4 meurt.

---

## 1. La résolution physique de chaque niveau

### Ce que le code fixe

`cote_monde(j) = n0 · 2^j` (n0 = 256, niveau_fin = 9) fixe les COMPTES DE
CELLULES par niveau, et `n_fov = 512` la taille de fenêtre. Le kernel F est
**cellulaire** : stencil en unités de cellule, `dt` figé (DT_JETABLE),
réduction CFL payée non consommée. **Il est agnostique à l'échelle** — il
avance des cellules, il ne connaît pas de dx physique.

### Ce que le code NE fixe PAS

Le code ne fixe **aucun dx physique ni aucune taille de monde**. Rien n'y dit
à quelle résolution PHYSIQUE le niveau fin correspond, ni où, dans la
hiérarchie des dx (facteur 2⁸ = 256 du grossier au fin), siège la résolution
du substrat. C'est **la** question de spec ouverte, et elle décide tout.

### L'hypothèse retenue — NOMMÉE, pas supposée

**Le niveau le plus fin (j=9) est à la résolution de la PHYSIQUE : dx = 1, le
dx du substrat `run_episode`. La pyramide DÉCIME vers l'extérieur (champ
lointain grossier) ; elle ne raffine PAS sous la physique.**

Justification, pas commodité :
- `run_episode` est à dx = 1 sur 64². **Il n'existe aucune vérité sous la
  maille dx = 1.** Une fovéa qui raffinerait sous dx = 1 afficherait du détail
  que la physique ne calcule pas — de l'interpolation, pas de la simulation.
- M-b compare le live au rederive 64² (dx = 1) en Δχ. Il n'y a **rien de
  sub-cellulaire à comparer** : une fovéa sur-résolue mesurerait sa propre
  invention.
- Donc la pyramide est un **mipmap** (fin = pleine résolution physique,
  grossier = décimation pour le lointain), PAS un AMR qui raffine la physique
  sous la maille de base.

L'hypothèse ALTERNATIVE — la fovéa raffine sous dx = 1 (AMR vrai) — est
nommée au §3 comme le scénario où V4 meurt ; elle exigerait une physique
sub-cellulaire que le rederive intouché ne fournit pas.

---

## 2. dt_CFL par niveau, sous-pas, et le multiplicateur

### Le dt CFL par niveau

`dt_CFL(j) = 0,4 · dx_j / smax`, avec `dx_j = dx_9 · 2^(9−j)` (le fin a le
plus petit dx, donc le plus petit dt_CFL ; les grossiers ont dt_CFL jusqu'à
2⁸ = 256× plus grand). Le niveau FIN est le plus contraignant.

### Le nombre de sous-pas par niveau — la règle CORRECTE

Un niveau ne sous-cycle QUE là où le pas de frame viole sa CFL :

```
n_j = max(1, ⌈ dt_frame / dt_CFL(j) ⌉) = max(1, ⌈ n_9 / 2^(9−j) ⌉)
```

où `n_9 = dt_frame / dt_CFL(fin)`. **C'est le point structurel de Romain, et
il a raison : le sous-cyclage est plafonné à 1 partout où `dt_frame ≤
dt_CFL(j)`, pas un `2^(j−1)` en aveugle.**

### Le multiplicateur sur F, pondéré par les blocs de V4

Blocs par niveau (V4) : j=1..7 → 1 chacun, j=8 → 2, j=9 → 6 (fovéa c=8 = 2 +
2 énergie c=8 = 4). Total 15.

```
M = Σ_j blocs(j) · n_j  /  15
```

| n_9 | M | sous-pas n_j (j=9 … j=1) |
|---|---|---|
| **1** | **1,0** | 1 1 1 1 1 1 1 1 1 |
| 2 | 1,4 | 2 1 1 1 1 1 1 1 1 |
| 4 | 2,3 | 4 2 1 1 1 1 1 1 1 |
| 10 | 5,3 | 10 5 3 2 1 1 1 1 1 |
| 20 | 10,3 | 20 10 5 3 2 1 1 1 1 |
| 61 | 30,7 | 61 31 16 8 4 2 1 1 1 |
| 256 | 128 | 256 128 64 32 16 8 4 2 1 |

Pour n_9 grand, **M ≈ n_9 / 2** (le fin, 6 blocs, domine).

### Mon 100–256× contre le ~5× de Romain — lequel, et pourquoi

**Mon 100–256× était une erreur STRUCTURELLE, pas cosmétique.** J'ai appliqué
un `2^(j−1)` en aveugle (M = 128, indépendant de n_9) — le sous-cyclage
Berger-Oliger PLEIN, qui n'est correct QUE si le pas de frame est calé sur la
CFL du niveau LE PLUS GROSSIER (n_9 = 256). Cela suppose EN PLUS du fin
sur-résolu de 256× un pas de frame 256× plus grand que la CFL fine — soit
~12× le temps réel. Double hypothèse agressive, jamais dite.

**Le ~5× de Romain est la bonne RÈGLE** (`max(1, ⌈…⌉)`, sous-cyclage seulement
où nécessaire) **appliquée à n_9 ≈ 10**, ce qui correspond à une fovéa
sur-résolue de ~64× en temps réel. La règle est juste ; l'hypothèse
(sur-résolution ~64×) ne l'est pas davantage que la mienne.

**Le vrai résultat n'est ni 5 ni 128 : c'est que M est une FONCTION de n_9,
et que n_9 dépend de la résolution du niveau fin.** Chiffré, en temps réel
(dt_frame = 1/60 = 0,0167 s), smax = 1,9–5,7 :

| Niveau fin | dx_9 | n_9 | **M** |
|---|---|---|---|
| **= physique (dx=1)** | 1 | **0,08–0,24** | **1,0** |
| 64× sous la physique | 1/64 | 5–15 | 3–8 |
| 256× (= span pyramide) | 1/256 | 20–60 | 11–31 |

---

## 3. V4 survit-il ? Franchement.

### Sous l'hypothèse retenue (fin = physique) : OUI, M = 1

À dx_9 = 1, `dt_CFL(fin) = 0,4/smax = 0,07–0,21 s`, et `dt_frame = 0,0167 s`
siège **4 à 12× SOUS** cette CFL. Donc `n_9 = 0,08–0,24 < 1` : le pas de
frame est sous-CFL AU NIVEAU FIN, **aucun niveau ne sous-cycle, M = 1**. Les
niveaux grossiers sont sur-résolus en temps (stable, gratuit — ils font un
seul pas). **Le budget tranche-1 tient tel qu'il était : ~15 ms, marge
positive. V4 survit.**

La garde de Romain est satisfaite : l'état synthétisé à l'échelle V4 a des
fronts wet/dry et smax = 1,9–5,7, donc **la CFL est SOLLICITÉE** (elle est une
contrainte réelle que le pas de frame respecte avec une marge de 4–12×) —
sollicitée n'est pas violée, et le chrono vaut.

### C'est une PROPRIÉTÉ DE SPEC à graver, pas une commodité

M = 1 tient **si et seulement si** le niveau fin ne descend pas sous la
résolution de la physique. C'est exactement la « raison légitime » que la
tâche invitait, et elle est load-bearing :

> **À GRAVER : la pyramide est un MIPMAP — le niveau le plus fin est à la
> résolution de la physique du substrat ; les niveaux se DÉCIMENT vers
> l'extérieur, jamais ne raffinent sous la maille de base.** Sous cette
> propriété, un pas de frame sous-CFL à l'échelle de base est sous-CFL au
> niveau fin, et aucun niveau ne sous-cycle.

Si un jour quelqu'un laisse la fovéa raffiner sous dx = 1 (AMR vrai), le
sous-cyclage réapparaît par la table du §2, et **V4 meurt dès ~4–12× de
sur-résolution** (M franchit ~2–3, la marge de 1,7 ms disparaît) ; à span
plein (256×), M ≈ 11–31, V4 est catastrophiquement mort. La survie de V4
tient donc entièrement à cette ligne de spec.

### Si V4 meurt (fovéa sur-résolue exigée) — ce qui reste vivant

Si la spec EXIGE une fovéa sub-physique (détail sous la maille), le
sous-cyclage tue le budget de frame. Ce qui NE meurt PAS avec le budget :

- **Le registre** (la mécanique de mesure pré-enregistrée) — intact.
- **La géométrie emboîtée** (propriété E, arbre, offsets ±n_fov/2) — une
  géométrie correcte reste correcte quel que soit le pas de temps.
- **Le kernel L3** et sa remontée — la compaction/émission ne dépend pas du
  sous-cyclage.
- **La prédiction GPU-side gratuite** (parent→enfant) — indépendante du dt.
- **EPS 1e-2** — la fidélité du canal établie par la sonde tient.

Ce qui doit être ROUVERT :
- **L'intégration temporelle mono-dt** (un pas de frame pour tous les
  niveaux). Sous AMR vrai, il faut soit sous-cycler (coût ×M), soit un schéma
  qui évite la contrainte CFL fine (§ ci-dessous), soit renoncer au raffinement
  sous la physique.
- **La victoire de coût 14,490** : mesurée à M = 1, elle serait VOID sous
  sous-cyclage et devrait être re-mesurée avec le multiplicateur.
- **La question même** : la fovéa a-t-elle BESOIN de raffiner sous la
  physique ? Si la réponse est non (l'hypothèse retenue), rien de ceci ne se
  pose.

### La seule autre issue qui sauverait un fovéa sur-résolu

Si l'on VOULAIT une fovéa sub-physique sans payer le sous-cyclage, il
faudrait une propriété de spec forte, à graver et à prouver, PAS une
commodité :
- **schéma implicite** au niveau fin (inconditionnellement stable, pas de
  contrainte CFL) — mais le F actuel est explicite (SSP-RK2), ce serait un
  moteur neuf, hors cap ;
- **niveau fin non hyperbolique** (le détail fin est diffusif/relaxé, pas
  ondulatoire) — possible si le sédiment fin ne porte pas d'ondes de gravité,
  mais à démontrer, pas à supposer ;
- **borne d'énergie** garantissant la stabilité au pas grossier — non
  établie ici.

Aucune de ces trois n'est acquise. **En leur absence, un fovéa sur-résolu
IMPOSE le sous-cyclage, et le budget de V4 n'y survit pas.** L'issue propre
est donc l'hypothèse retenue : fin = physique, décimation vers l'extérieur.

---

## 4. Conclusion, sans l'habiller

- Le multiplicateur sur F est `M ≈ n_9/2`, `n_9 = dt_frame/dt_CFL(fin)` — une
  FONCTION, pas une constante. Mon 128 (blanket 2^(j−1)) était faux ; la règle
  de Romain (sous-cycler seulement où nécessaire) est la bonne.
- **Sous l'hypothèse retenue (fin = résolution physique, décimation vers
  l'extérieur), n_9 < 1, M = 1, aucun sous-cyclage, et V4 SURVIT** — le budget
  tranche-1 ~15 ms tient.
- **Cette survie est conditionnelle à UNE ligne de spec** (mipmap, pas AMR
  sous la physique), qui doit être GRAVÉE avant l'achat. Ce n'est pas une
  commodité : si la fovéa raffine sous la physique, le sous-cyclage revient et
  V4 meurt (M ≈ 11–31 à span plein).
- La décision de spec due, remontée : **la fovéa de V4 raffine-t-elle SOUS la
  résolution de la physique, oui ou non ?** Non ⇒ V4 vit, chiffrage tranche-1
  inchangé. Oui ⇒ V4 meurt au budget, et il faut soit une propriété de
  stabilité gravée (implicite / non-hyperbolique / borne d'énergie, aucune
  acquise), soit rouvrir l'intégration temporelle.

---

## 5. Portée

Kernels FIGÉS intouchés (`_SOURCE` = e18015f5…f30b4 / `_SOURCE_L3` =
9533a130…781a) ; k reste 4 ; `run_f1_attribution_b.py` gaté ; pas de miroir
CPU ; aucun achat. Analyse sur le papier ; ne modifie rien.

**POINT D'ARRÊT.** Document remonté, rien d'autre. Aucune ligne de code
écrite, aucun run.
