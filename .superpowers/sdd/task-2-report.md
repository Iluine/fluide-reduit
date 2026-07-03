# Task 2 — Rapport du GATE de re-validation (BLOQUANT)

Branche : `arc-a-etat-complet`. Brief : `.superpowers/sdd/task-2-brief.md`.
Script : `scripts/run_arcA_revalidate.py`. Sortie machine :
`outputs/arcA/revalidation.json` (+ intermédiaires `step_a/b/c_overlap/c_separee.{json,npz}`).

## VERDICT GLOBAL : **FAIL**

Deux des trois critères sont manqués : **(a)** et **(c)/séparée**. Seul **(b)**
passe. Conformément au contrat épistémique du projet, **aucun seuil ni
paramètre n'a été retouché** — les valeurs ci-dessous sont les mesures brutes
du run réel (grille 64×64, `SedimentParams` par défaut, `KD_CALIBRE` figé),
n=10 épisodes/histoire.

## (a) Path-dependence — **FAIL**

Séquence A = 10 centres tirés de `default_rng(7)` dans [0.15,0.85]² ; séquence
B = la même liste inversée. `corr(s_A, s_B)` = coefficient de Pearson des
champs sédiment aplatis.

| mesuré | seuil | référence d'origine | verdict |
|---|---|---|---|
| corr(s_A, s_B) = **0.9929** | ≤ 0.7 | 0.39 | **FAIL** |

L'ordre des pulses affecte quasiment pas l'état final du substrat reconstruit :
les deux histoires (ordre direct / ordre inversé du même multiset de 10
centres) convergent vers un champ sédiment quasi identique. Le substrat
reconstruit est bien plus "mémoïse-partout" que le substrat d'origine
(corr 0.99 vs 0.39 référence) — cohérent avec le motif déjà observé dans le
projet frère `pocCascade2phys` (T1 : substrat (quasi-)périodique →
mémoïser-partout perceptuellement optimal).

## (b) Survie albedo (ordre vs simultané-même-masse) — **PASS**

`s_ordre` = s_A de (a). `s_sim` = un seul épisode dont le pulse initial est
la somme des 10 gaussiennes (mêmes centres/σ/h_p que la séquence ordonnée),
relaxé + intégré Exner à l'identique, puis rescalé multiplicativement pour
`Σs_sim = Σs_ordre` (facteur de rescale mesuré : 0.6004 — le pulse simultané
brut dépose ~1.66× plus de masse que la séquence ordonnée avant rescale,
`Σs_sim_brut=25.78` vs `Σs_ordre=15.48`).

| mesuré | seuil | référence d'origine | verdict |
|---|---|---|---|
| f_albedo(point d'op, JND=5%) = **0.8308** (83.1 %) | ≥ 2 % | 3.5–10 % | condition OK |
| f_relief(JND=5%) = **0.00195** (0.2 %) | — (référence) | 1.2 % | — |
| f_albedo > f_relief pour TOUT S_HALF balayé | — | — | **PASS** |

Détail du balayage S_HALF (fraction du relief, JND=5% fixe) :

| S_HALF (×relief) | f_albedo | > f_relief (0.00195) ? |
|---|---|---|
| 0.005 | 0.8132 | oui |
| 0.010 | 0.8486 | oui |
| 0.050 (point d'op) | 0.8308 | oui |
| 0.100 | 0.8267 | oui |
| 0.200 | 0.8245 | oui |

Le signal "ordre vs simultané" est **massivement plus fort** que la référence
d'origine (83 % vs 3.5–10 % attendu) : le readout albedo révèle un écart
énorme entre les deux protocoles de pulse, largement au-dessus du plancher
f_relief (0.2 %) — cohérent en soi avec (a) : (a) teste la RÉORDONNANCE d'une
même séquence de 10 pulses séquentiels (peu d'effet, champ hydrodynamique
quasi équivalent à chaque étape) tandis que (b) compare séquentiel vs
**simultané** (dynamique complètement différente — une seule vague massive
superposée vs 10 vagues indépendantes en série), donc une divergence bien
plus grande est physiquement attendue et n'est pas contradictoire avec (a).

## (c) Closure f(k) — **FAIL** (overlap PASS, séparée FAIL)

Pour k ∈ {1,4,7,10} et chaque géométrie : histoire X = préfixe (10−k
centres) + suffixe (k centres) tirés d'un seul `default_rng(1000+k)`
(overlap) / `default_rng(2000+k)` (séparée) ; histoire Y = même préfixe
inversé + même suffixe. `f(k) = f_fraction(albedo(s_X), albedo(s_Y),
S_HALF_op, jnd=5%)`. Tolérance "non-croissante" : +0.002 (bruit de grille,
constante `_NONINCREASING_TOL` dans le script).

### Géométrie overlap (centres y ∈ [0.4, 0.6]) — PASS

| k | f(k) |
|---|---|
| 1 | 0.3384 |
| 4 | 0.2141 |
| 7 | 0.1873 |
| 10 | 0.0000 |

Séquence strictement décroissante (dans la tolérance) : **non-croissante = True**.
f(10) = 0 < 0.05. → **PASS**.

### Géométrie séparée (boîte pleine [0.15,0.85]²) — FAIL

| k | f(k) |
|---|---|
| 1 | 0.3274 |
| 4 | **0.3630** |
| 7 | 0.1116 |
| 10 | 0.0000 |

Rupture de monotonie : f(4)=0.363 > f(1)+0.002=0.329 → **non-croissante = False**.
f(10) = 0 < 0.05 (cette partie du critère est respectée) mais la condition
conjointe échoue. → **FAIL**.

Chaque k est tiré d'un seed indépendant (1000+k / 2000+k), donc chaque point
f(k) provient d'une réalisation géométrique distincte (pas d'échantillons
emboîtés) : la géométrie "overlap" (centres contraints à une bande étroite en
y) produit une décroissance propre, tandis que la géométrie "séparée" (boîte
pleine, tirages plus dispersés) montre un rebond au k=4 — signe que la
propriété de closure n'est pas robuste sur tout l'espace des centres pour ce
substrat reconstruit.

## Temps de calcul

| phase | épisodes (run_episode) | temps mesuré |
|---|---|---|
| (a) | 20 (2 histoires × 10) | 57.6 s |
| (b) | 1 relaxation (pulse 10× superposé) | 2.3 s |
| (c) overlap | 70 (4 k × 2 histoires × 10, sauf k=10 : Y réutilise X bit-exact) | 222.1 s |
| (c) séparée | 70 (idem) | 200.2 s |
| **total (a+b+c)** | 161 | **482.1 s (~8 min)** |

(Un smoke test préalable avec n_episodes=3 a validé la plomberie du script en
~74 s avant de lancer le run réel à n=10 par phases séparées, chacune sous la
limite de 600 s par appel.)

## Conclusion

Le GATE de re-validation **échoue** : le substrat sédiment reconstruit
(Task 1) ne reproduit PAS fidèlement les propriétés (a) path-dependence et
(c) closure f(k)/séparée du substrat d'origine, bien qu'il reproduise
largement (b) survie albedo. **Conformément au contrat, aucune tâche
suivante du plan Arc A n'est lancée et aucun paramètre n'est retouché.** Le
verdict est rapporté tel quel au contrôleur pour décision.
