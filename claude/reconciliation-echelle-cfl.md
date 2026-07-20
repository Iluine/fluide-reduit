# Paper-grade — réconciliation d'échelle et CFL par niveau

**Paper-grade. Aucun run, aucune ligne de code.**
Source : pocCascade2phys PREREGISTRATION.md §A26 (commit 28f5111). Deux
questions de la réserve sur l'ancre (§A25), dans l'ordre, et elles seules.
Les trois autres points (K, p99, mur intérieur) attendent. §A26 a tranché
dt = temps de frame, un pas/frame sous-CFL — donc la burstiness du §A25 est
suspendue au régime, pas rejetée.

> **Les deux réponses en une phrase** : le substrat réel 64² est plus petit
> que le niveau LE PLUS GROSSIER de la pyramide (256²), donc M-b ne peut pas
> tourner à l'échelle de V4 sans un rederive qui n'existe pas ; à l'échelle
> qu'impose le rederire, l'ancre s'effondre d'un facteur ~960 (12,655 →
> ~0,013 ms) et T1 devient vacant. Le sous-cyclage, lui, est ÉVITÉ par le
> choix §A26 — mais seulement tant que le niveau fin n'est pas plus fin que
> le substrat, ce que la spec ne garantit pas.

---

## La géométrie, exactement

`cote_monde(j) = n0 · 2^j`, avec `n0 = 256` (N0_DEFAUT), `n_niv = 10`,
`niveau_fin = 9`, `n_fov = 512`.

| Niveau | j=0 (CPU) | j=1 | … | j=9 (fin) |
|---|---|---|---|---|
| `cote_monde` (cellules) | 256 | 512 | … | **131 072** |

- Le monde de V4, au niveau le plus fin, fait **131 072 × 131 072 cellules**.
- La fovéa (n_fov = 512) en couvre **0,39 %** de la largeur.
- Le dx varie d'un facteur **2⁸ = 256** sur les 9 niveaux GPU (2⁹ si l'on
  compte le niveau 0) — c'est bien le facteur que Romain a nommé.
- Coût F mesuré : 15 blocs × 512² = **3 932 160 block-cells → 12,655 ms**,
  soit ~3,2 ns/block-cell.

Le fait décisif, chiffré : **le substrat réel 64² est plus petit que
`cote_monde(0) = 256²`** — il ne remplit AUCUN niveau de la pyramide, pas
même le plus grossier.

---

## 1. Réconciliation d'échelle 64² ↔ n_fov

### À quelle échelle M-b doit tourner, et pourquoi

Le rederive est `run_episode` sur `GridConfig` **64²**, CPU f64, **INTOUCHÉ**.
C'est la vérité-sol. Pour que Δχ(live, rederive) ait un sens, le live et le
rederive doivent être **le même monde**. Or :

- **Vers le haut (échelle V4)** : plonger le 64² dans le monde 131 072² de
  V4 exigerait un rederive 131 072² — que la physique CPU f64 (64², intouchée)
  ne fournit pas, et que `run_episode` ne calcule pas. Impossible sans
  toucher au rederive (interdit) ou sans que live et rederive divergent
  d'échelle (ce qui casse la comparaison même de M-b).
- **Vers le bas (échelle 64²)** : la seule qui respecte le rederive intouché.
  Mais 64² < 256² : la pyramide n'a plus de niveau à remplir. La structure
  multi-niveaux **dégénère** — une seule fenêtre pleine résolution suffit,
  et sur 64² la pleine résolution est GRATUITE (4 096 cellules).

**Donc M-b DOIT tourner à 64², parce que c'est la seule échelle où le rederive
intouché est la vérité.** Il n'y a pas d'alternative qui garde à la fois le
rederive intouché et la comparaison live↔rederive.

### Ce que devient l'ancre F, sans l'atténuer

À 64² pleine résolution (un système, un « bloc ») : 4 096 block-cells contre
3 932 160 à n_fov=512.

```
ancre F(64²) = 12,655 × 4096 / 3 932 160 ≈ 0,013 ms   (facteur ~960)
```

**L'ancre s'effondre de ~960× : 12,655 ms → ~0,013 ms.** Même avec une petite
pyramide de quelques niveaux (n_fov ~ 32–64) au lieu d'une fenêtre pleine,
elle reste ~0,01–0,2 ms. Je ne l'atténue pas : c'est deux à trois ordres de
grandeur.

### Ce que V4 signifiait encore

Il faut le dire net : **la victoire de V4 (14,490 ms contre le repli 33.3)
est une victoire de GRAND monde.** Elle mesure le coût de fovéer un monde de
131 072² — là où la pleine résolution serait ingérable. Sur un monde 64², la
fovéation n'a aucun objet : la pleine résolution y est triviale. Donc :

- **La victoire de V4 tient — pour l'échelle où elle a été mesurée.** M-a-quater
  a établi qu'une pyramide fovéée anime un grand monde shallow-water sous
  16,7 ms. C'est réel, à 131 072².
- **Ce qui s'effondre, c'est la capacité de M-b à TESTER cette victoire.**
  M-b, arrimé à la physique réelle 64², ne peut pas exercer la fovéation à
  l'échelle où elle sert. À 64², M-b mesure la fidélité du **portage de la
  physique** (le F fidèle reproduit-il `run_episode` ?), PAS le coût ni la
  fovéation de V4.

### Ce que ça fait au verdict T1

**T1 devient vacant à 64².** L'ancre effondrée (~0,013 ms) plus Exner,
remontée et transferts (tous eux-mêmes à ~0 sur 4 096 cellules) donnent une
frame de l'ordre de **~0,1–0,5 ms** — deux ordres sous 16,7. T1 « frame
complète > 16,7 » ne peut PAS se déclencher à 64². Un gate qui ne peut pas
échouer n'est pas un gate. **Le contrôle T1 de M-b, à l'échelle qu'impose le
rederive, ne verdicte rien.**

C'est une **question de spec, pas un détail de build** : soit M-b vise la
fidélité du portage physique à 64² (et T1 n'est pas son gate — MORT-b l'est),
soit on veut tester le coût de V4 à grande échelle (et il faut un rederive à
grande échelle, qui n'existe pas). Les deux ne tiennent pas ensemble sous
« rederive intouché ».

---

## 2. CFL par niveau / sous-cyclage

### Le schéma sous-cycle-t-il ? Non — par construction §A26.

Le pipeline V4 batché avance TOUS les niveaux en UN pas de F par frame, avec
UN dt unique (le jetable : dt figé ; la réduction CFL est payée, non
consommée). §A26 a tranché : **dt = temps de frame, un pas/frame, sous-CFL.**
C'est le choix qui ÉVITE le sous-cyclage : on n'itère pas le niveau fin ; on
avance tout au dt de frame. **Il n'y a donc pas de somme de Berger-Oliger sur
les niveaux** — le coût reste un pas par frame, l'ancre telle que mesurée.

### Ce qui rend le pas grossier admissible au niveau fin

La condition de stabilité, explicite : le pas de frame doit satisfaire la CFL
**au niveau le plus fin** (le plus contraignant, plus petit dx).

```
dt_frame ≤ dt_CFL(fin) = 0,4 · dx_fin / smax
```

Le niveau fin a le plus petit dx, donc le plus petit dt_CFL. Les niveaux
grossiers ont dt_CFL 2⁸ = 256× plus grand — ils sont donc **sur-résolus en
temps** (ils avancent un dt bien plus petit que leur propre CFL). C'est
STABLE (jamais instable de sur-résoudre), inefficace (le grossier pourrait
faire de plus grands pas), mais **gratuit** : ils font quand même un seul pas.
C'est exactement ce que le pipeline batché single-dt fait déjà.

Est-ce que `dt_frame ≤ dt_CFL(fin)` tient ? À l'échelle 64² (le fin = dx du
substrat = 1) :

```
dt_CFL(fin) = 0,4 · 1 / smax ≈ 0,4/1,9 … 0,4/5,7 ≈ 0,21 … 0,07 s
dt_frame (temps réel)        = 0,0167 s
```

**dt_frame = 0,0167 ≤ 0,07–0,21 : sous-CFL par ~4 à 12×.** Admissible. À 64²,
le choix §A26 tient et il n'y a pas de sous-cyclage. Réponse cohérente avec
Q1 : à l'échelle où M-b doit tourner, pas de niveaux fins problématiques.

### La tension, nommée : et si la fovéa est VRAIMENT plus fine que le substrat ?

Voici où je dois être franc plutôt que rassurant. Le **but** d'une fovéa est
d'être PLUS FINE que le grossier — c'est sa raison d'être. Si le niveau fin
de V4 représente une résolution plus fine que le substrat (dx_fin < ~0,08),
alors :

```
dt_CFL(fin) = 0,4 · dx_fin / smax  <  dt_frame
```

**la CFL est VIOLÉE au niveau fin, le pas de frame y est INSTABLE, et §A26
« un pas/frame sous-CFL » ne tient plus.** On est alors FORCÉ de sous-cycler
(Berger-Oliger) : le niveau j fait 2^(j−1) sous-pas, et le coût fin se
multiplie. La somme sur les niveaux, si le grossier fait 1 pas :

```
coût ∝ Σ_j n_slots(j) · 2^(j−1)   — dominée par le fin (2⁸ = 256×)
```

Le coût F **explose de ~100 à 256×**, et T1 est mort à toute échelle où la
fovéa est réellement sub-résolution.

**Rien dans la spec ne fixe la résolution physique du niveau fin.** À 64²,
la question est MUETTE (le fin = le substrat, pas plus fin). Mais dès que la
fovéa prétend à du détail sous la maille du substrat — ce qui est sa
vocation — la CFL par niveau redevient contraignante et §A26 ne la couvre
plus. **C'est donc une question ouverte de la spec, pas un détail de M-b :
quelle résolution physique le niveau fin représente-t-il, et le pas de frame
y est-il sous-CFL ?** Tant qu'elle n'est pas tranchée, « pas de sous-cyclage »
n'est vrai qu'à l'échelle dégénérée 64².

---

## 3. Ce que ces deux réponses changent au re-chiffrage de tranche-1

Le re-chiffrage du §A25 (~15,0–16,1 ms, marge 0,6–1,7) supposait l'ancre
12,655 tenant à n_fov=512. Les deux réponses le déplacent radicalement :

| | À 64² (échelle imposée par le rederive) | À l'échelle V4 (131 072²) |
|---|---|---|
| Ancre F | ~0,013 ms (effondrée ×960) | 12,655 ms |
| Sous-cyclage | non (fin = substrat, sous-CFL) | non SI fin non sub-résolution ; sinon ×100–256 |
| Frame complète | **~0,1–0,5 ms** | 15–16 ms, ou explose si sous-cyclage |
| Rederive disponible | oui (64² intouché) | **NON** |
| T1 | **vacant** (ne peut pas atteindre 16,7) | pertinent, mais rien à comparer |

**Le re-chiffrage ~15–16 ms est FAUX à l'échelle où M-b peut tourner.** À 64²,
le budget réel est ~0,1–0,5 ms et T1 passe vacamment. Le gate T1 n'est donc
pas « plausible » ou « compromis » — il est **sans objet** à cette échelle.

Et je ne l'habille pas : le vrai résultat n'est pas sur la marge de T1. C'est
que **M-b, tel que spécifié, ne peut pas servir le rôle de T1.** T1 devait
verdicter si le F fidèle tient le budget de frame à l'échelle de V4 ; à
l'échelle du rederive il tient trivialement, et à l'échelle de V4 il n'y a pas
de rederive. **Découvert au papier, cela coûte une décision de spec ; découvert
au premier run, cela aurait coûté un build entier pour un gate qui ne peut pas
échouer.**

### Ce qui reste vrai, et ce qui doit être tranché

- **Reste vrai** : la victoire de coût de V4 (14,490 vs 33.3) à grande
  échelle, établie par M-a-quater. M-b n'y touche pas.
- **À trancher, avant tout achat de M-b** (question de spec, remontée) :
  1. M-b mesure-t-il la fidélité du portage physique **à 64²** — auquel cas
     T1 n'est pas son gate (MORT-b l'est), et T1 doit être retiré ou
     re-signifié ;
  2. ou veut-on tester le coût de V4 **à grande échelle** — auquel cas il
     faut un rederive à cette échelle, ce que « CPU f64 intouché » interdit,
     et la spec de M-b est à reprendre ;
  3. et, indépendamment : la **résolution physique du niveau fin** (§2), qui
     décide si le sous-cyclage réapparaît dès qu'on quitte 64².

Aucune de ces trois n'est une ligne de code de M-b — ce sont des décisions
gravées dues avant l'achat.

---

## 4. Portée

Kernels FIGÉS intouchés (`_SOURCE` = e18015f5…f30b4 / `_SOURCE_L3` =
9533a130…781a) ; k reste 4 ; `run_f1_attribution_b.py` gaté ; pas de miroir
CPU ; aucun achat. Ce document est une analyse sur le papier ; il ne modifie
rien.

**POINT D'ARRÊT.** Document remonté, rien d'autre. Aucune ligne de code
écrite, aucun run.
