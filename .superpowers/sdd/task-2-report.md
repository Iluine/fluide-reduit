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

---

# Itération v2 : co-calibration (k_d, k_e) aux deux signatures gravées

Itération de conception NOMMÉE, pré-enregistrée au journal pocCascade2phys
(`PREREGISTRATION.md`, entrée 2026-07-04) AVANT implémentation, suite à
l'arbitrage du contrôleur (option 1 : diagnostic cheap puis UNE itération,
re-validée aux MÊMES seuils). Changement unique autorisé : remplacer l'a
priori `k_e = k_d/5` (jamais une donnée d'origine) par une co-calibration
aux DEUX signatures gravées du substrat d'origine — taux (max(s)/relief ∈
[0.18, 0.22]) ET resfrac (masse totale érodée / masse totale déposée sur
l'histoire, ∈ [0.60, 0.66] ; mesurée à 0.174 sur le substrat v1, ~4× sous
la plage gravée, cf. `outputs/arcA/diag_pathdep.json`). θ_c = 0.5, structure
d'épisode, terrain : INCHANGÉS. **Aucun seuil du gate déplacé.**

## Co-calibration — protocole et valeurs gelées

`src/sediment.py::cocalibrate_kd_ke` : bissections log10 ALTERNÉES (k_d →
taux à k_e fixé ; k_e → resfrac à k_d fixé, plage log10(k_e) ∈ [-6, 1]),
vérification CONJOINTE des deux cibles à chaque fin de cycle, max 4 cycles,
seed 12345 / 10 épisodes (protocole identique à la calibration V1). resfrac
est mesuré par `resfrac_history` (les deux termes de `_exner_step` agrégés
sur tous les snapshots et épisodes, via un paramètre `return_terms` optionnel
qui ne change pas le comportement par défaut).

Exécutée UNE fois (2026-07-04, 1126.7 s ≈ 19 min, trace :
`outputs/arcA/cocalib_v2_checkpoint.json` + `cocalib_v2_result.json`) :
**convergence au 1er cycle**, vérification conjointe tenue.

| constante | valeur gelée | note |
|---|---|---|
| `KD_CALIBRE_V2` | 0.0019109529749704406 | bit-à-bit identique à V1 (même chemin de bissection dyadique) |
| `KE_CALIBRE_V2` | 0.005232991146814947 | ≈ 13.7× le k_e V1 (3.822e-4) ; k_e/k_d ≈ 2.74 (V1 : 0.2) |

Signatures vérifiées aux constantes gelées (défauts `SedimentParams`,
seed 12345, 10 épisodes) — les DEUX cibles tenues simultanément :

| signature | mesuré | cible gravée |
|---|---|---|
| max(s)/relief | **0.18875904854649675** | [0.18, 0.22] ✓ |
| resfrac | **0.6425366558324981** | [0.60, 0.66] ✓ |

Constantes GELÉES AVANT la re-validation v2 (commit dédié) ; les V1 restent
en trace commentée. Suite de tests complète après gel : **105 passed**
(102 + 3 nouveaux tests `resfrac_history`).

## VERDICT du gate v2 (mêmes seuils, même script, `"version": "v2-cocalibration"`)

### VERDICT GLOBAL : **FAIL** (2/3 — mêmes critères manqués que v1)

### (a) Path-dependence — **FAIL**

| mesuré v2 | v1 | seuil | référence origine | verdict |
|---|---|---|---|---|
| corr(s_A, s_B) = **0.9446** | 0.9929 | ≤ 0.7 | 0.39 | **FAIL** |

La prédiction directionnelle pré-enregistrée (« le retravail fort re-pondère
le dépôt vers les pulses récents → corr doit BAISSER nettement ») est
partiellement confirmée dans le SENS (0.9929 → 0.9446, soit −0.048) mais
très loin de l'amplitude requise : le substrat reste quasi commutatif à
l'ordre des pulses, même avec resfrac dans la plage gravée (0.64). Le taux
de retravail n'était donc PAS le mécanisme dominant de la path-dependence
d'origine — la signature resfrac est nécessaire mais pas suffisante.

### (b) Survie albedo — **PASS**

| mesuré v2 | v1 | seuil | verdict |
|---|---|---|---|
| f_albedo(point d'op, JND=5%) = **0.8511** | 0.8308 | ≥ 0.02 | OK |
| f_relief = **0.0020** | 0.00195 | — (plancher) | — |

Balayage S_HALF (×relief) : 0.005 → 0.8777 ; 0.010 → 0.8650 ; 0.050 →
0.8511 ; 0.100 → 0.8491 ; 0.200 → 0.8479 — tous > f_relief. **PASS**.

### (c) Closure f(k) — **FAIL** (overlap PASS, séparée FAIL)

Overlap (bande y ∈ [0.4, 0.6]) — PASS :

| k | f(k) v2 | f(k) v1 |
|---|---|---|
| 1 | 0.4412 | 0.3384 |
| 4 | 0.3115 | 0.2141 |
| 7 | 0.2810 | 0.1873 |
| 10 | 0.0000 | 0.0000 |

Non-croissante (tol +0.002) = True ; f(10) = 0 < 0.05. → **PASS**.

Séparée (boîte pleine [0.15, 0.85]²) — FAIL :

| k | f(k) v2 | f(k) v1 |
|---|---|---|
| 1 | 0.6018 | 0.3274 |
| 4 | 0.3040 | 0.3630 |
| 7 | **0.3516** | 0.1116 |
| 10 | 0.0000 | 0.0000 |

Rupture de monotonie : f(7)=0.3516 > f(4)+0.002=0.3060 (v1 : rebond au k=4).
f(10) = 0 < 0.05 respecté, mais la condition conjointe échoue. → **FAIL**.

Temps de calcul cumulé du gate v2 (a+b+c) : 491.0 s. Sorties : nouveaux
`outputs/arcA/{revalidation,step_a,step_b,step_c_overlap,step_c_separee}.json`
(+ npz) écrasant les v1 (préservés dans l'historique git, commit 554c2a9) ;
champ `"version": "v2-cocalibration"` dans `revalidation.json`.

## Conclusion v2

L'itération autorisée est CONSOMMÉE et le gate **échoue à nouveau** sur les
mêmes deux critères (a) et (c)/séparée — avec une amélioration marginale de
(a) (0.9929 → 0.9446) qui confirme le sens de la prédiction directionnelle
mais pas son amplitude. Conformément au pré-enregistrement (« Un critère
manqué → STOP définitif de la voie "reconstruction" ») : **STOP définitif,
remonté au contrôleur — le fork suivant appartient à Romain.** Aucun seuil
ni paramètre n'a été retouché ; aucune tâche suivante du plan Arc A n'est
lancée.
