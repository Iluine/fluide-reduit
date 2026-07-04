# Rapport M-0bis — Sonde de testabilité de la manche 1 sur le substrat v2

Brief : `.superpowers/sdd/refond-task2-m0bis-brief.md`. Contrat : entrée « M-0bis
PRÉ-ENREGISTRÉE » de `PREREGISTRATION.md` (dépôt pocCascade2phys, commit `ae1c926`).
Branche : `arc-a-etat-complet`.

## Ce qui est implémenté

### Partie 1 — `src/summary.py` (spec GRAVÉE, plan manche 1 Task 4)

Module autonome, une responsabilité, interface exactement `summarize`, `regenerate`,
`Summary`, `size_floats` :

- `Summary` : dataclass gelée `(coarse, invariants, level, shape)`.
- `summarize(field, level)` : moyenne par blocs dyadiques `2**level` (`ℓ ∈ {1,2,3}`,
  reshape + `.mean(axis=(1,3))`) + invariants = `[masse_totale, 16 masses de
  sous-domaines 4×4]` (17 floats, sous-domaines FIXES 16×16 indépendants de `ℓ`).
- `size_floats(summary)` = `coarse.size + invariants.size` (générique, pas de table
  câblée) — vérifié : ℓ=1→1041, ℓ=2→273, ℓ=3→81 (champ fin 4096, grille 64×64).
- `regenerate(summary)` (UN paramètre) : upsampling bilinéaire séparable
  (`_bilinear_upsample`, deux passes `np.interp` 1D — x puis y — mathématiquement
  équivalentes au bilinéaire 2D complet sur grille régulière, extrapolation par
  plafonnement aux bords) → clip `≥ 0` → rescale multiplicatif par sous-domaine
  4×4 pour restituer exactement les 16 masses stockées (division protégée : bloc
  de masse nulle après upsampling → reste nul) → assertion de cohérence interne
  masse totale (`np.isclose`, `rtol=atol=1e-9`).
- Clause stochasticité (sha256) : **non implémentée**, conformément à la note du
  brief (« régénérateur purement déterministe → clause satisfaite trivialement,
  YAGNI ») ; le test M-A3 cross-process reste couvert.

### Partie 2 — `run_episode_trajectoire` additive (`src/sediment.py`)

Deux fonctions ajoutées EN FIN de fichier, sans toucher une seule ligne du code
existant :

- `_integrate_exner_trajectoire(s0, times, hs, hus, hvs, params)` : même boucle de
  contrôle que `_integrate_exner` (mêmes indices de snapshots, même `dt`), réutilise
  `_exner_step` **inchangée** (pas de duplication de la loi Exner), mais accumule
  et retourne la liste complète des `s` intermédiaires (`s0` en tête).
- `run_episode_trajectoire(s, b0, center_frac, params)` : même structure que
  `run_episode` (pulse → `_relax_episode` inchangée → intégration Exner), mais
  retourne la trajectoire complète au lieu du seul état final. Le DERNIER élément
  est bit-identique à `run_episode(...)` sur le même cas (prouvé par test).
- `_integrate_exner` et `run_episode` ne sont ni modifiées ni appelées par les
  nouvelles fonctions : zéro risque de régression sur leurs appelants existants.

### Partie 2 — Sonde `scripts/run_arcA_testability.py`

Script autonome, style aligné sur `run_arcA_m0.py` :

- Histoires directe/inversée : réutilise `BOX`/`N_EPISODES`/`SEED`/`_f_ordre`
  (`scripts.run_arcA_m0`) et `B0`/`GRID`/`PARAMS`/`RELIEF`/`S_HALF_OP`/
  `_draw_centers`/`_run_sequence` (`scripts.run_arcA_revalidate`) — aucun des deux
  fichiers n'est modifié ni ré-exécuté.
- 11ᵉ centre du rollout : tiré en CONTINUANT la même instance `rng` (seed 7) après
  les 10 premiers tirages — garantit par construction les 10 premiers centres
  bit-identiques à M-0 ; vérifié en plus par une assertion explicite (tirage de
  contrôle sur un second `rng` frais) avant tout calcul.
- M-A1 instantané et M-A2-mini (via `run_episode_trajectoire`, max le long du
  rollout + argmax rapporté pour audit) calculés pour les 2 histoires × 3 niveaux.
- `f_close` (diagnostic, réutilise `_f_ordre` telle quelle — formule symétrique
  identique à la spec) sur la paire instantanée (A_regen, A_vrai), JND ∈
  {2,3,4,5} %.
- Verdict mécanique : niveaux sous cap = {2,3} (81/273 floats), seuil FIGÉ 0.02,
  DEUX histoires — `T_triviale` ssi tout est sous seuil, `T_testable` sinon.
- Sortie unique `outputs/arcA/testability_v2.json`.

## Évidence TDD — `src/summary.py`

**RED** (module inexistant, avant implémentation) :

```
.venv/bin/pytest tests/test_summary.py -q
```
```
ImportError while importing test module '.../tests/test_summary.py'.
tests/test_summary.py:16: in <module>
    from src.summary import Summary, regenerate, size_floats, summarize
E   ModuleNotFoundError: No module named 'src.summary'
1 error in 0.14s
```

**GREEN intermédiaire** (implémentation initiale — round-trip constant `3.7`,
`np.array_equal` strict) :

```
.venv/bin/pytest tests/test_summary.py -q
```
```
.....................F.
FAILED tests/test_summary.py::test_round_trip_champ_constant_est_exact[3] - assert False
1 failed, 20 passed in 0.28s
```

Diagnostic : la moyenne dyadique ℓ=3 (64 éléments) d'une constante décimale
arbitraire (3.7, non représentable exactement en binaire) laisse ~1 ULP de bruit
d'arrondi (`4.44e-16` absolu) — propriété universelle de l'arithmétique flottante
IEEE754 (vérifié empiriquement : `4.0`/`2.5`/`0.125`/`1.0`, tous des dyadiques,
sont bit-exacts à tous les niveaux ; `3.7` seul dévie, uniquement à ℓ=3). Correction
du **test** (pas de l'implémentation) : constante dyadique `4.0` au lieu de `3.7`,
pour tester la propriété « round-trip exact » sans bruit de représentation
non pertinent — ce n'est pas un écart de spec, l'exactitude bit-à-bit d'un
round-trip sur `3.7` est mathématiquement impossible en IEEE754 quel que soit
l'algorithme.

**GREEN final** :

```
.venv/bin/pytest tests/test_summary.py -q
```
```
.....................
21 passed in 0.34s
```

## Tests et résultats

- `tests/test_summary.py` (21 tests, nouveaux) : tailles exactes (1041/273/81),
  forme du coarse, invariants (masse totale, sous-masses), M-A3 (double
  régénération + identité cross-process via `subprocess`+sha256), anti-fuite
  (signature 1 paramètre, mutation décoy après résumé sans effet), invariants
  restitués à 1e-10 relatif, round-trip constant exact, forme/positivité de sortie.
- `tests/test_sediment.py` (+4 tests nouveaux, 19 au total dans ce fichier) :
  endpoint de trajectoire bit-identique à `run_episode`, longueur/premier élément
  de la trajectoire, pureté (pas de mutation), déterminisme bit-à-bit.
  `.venv/bin/pytest tests/test_sediment.py -q` → **19 passed in 132.85s**.
- Suite complète : `.venv/bin/pytest -q` → **130 passed in 135.92s** (105
  existants + 21 `test_summary.py` + 4 nouveaux `test_sediment.py` — aucune
  régression, les 105 tests préexistants sont restés verts sans modification).
- Relancée une dernière fois juste avant commit (après le fix JSON de la sonde,
  qui ne touche que `scripts/run_arcA_testability.py`, hors périmètre des
  tests) : `.venv/bin/pytest -q` → **130 passed in 136.70s** — confirmé stable.

## Chiffres de la sonde

Exécution : `.venv/bin/python scripts/run_arcA_testability.py`, temps de calcul
90.5 s CPU. `relief = 0.6043843991164223`, `S_HALF_OP = 0.030219219955821115`
(= 0.05·relief). 11ᵉ centre du rollout (identique pour les deux histoires) :
`(0.3007160887649193, 0.2621484237004912)` — les 10 premiers centres tirés en
continuant la même `rng(seed=7)` sont vérifiés bit-à-bit identiques à ceux de
M-0 par l'assertion du script (aucun échec levé).

Tailles (floats) : ℓ=1 → 1041 (hors cap, indicatif), ℓ=2 → 273 (sous cap),
ℓ=3 → 81 (sous cap) ; champ fin = 4096.

**M-A1 (instantané) et M-A2-mini (max le long du rollout, 301 snapshots) —
`max_carrier` (Δχ, max sur bandes porteuses) :**

| Histoire | ℓ (floats) | M-A1 | M-A2-mini (argmax/301) | max(M-A1,M-A2-mini) | Sous cap ? | < 0.02 ? |
|---|---:|---:|---:|---:|:---:|:---:|
| direct  | 1 (1041) | 0.03173 | 0.03394 (k=86)  | 0.03394 | non (indicatif) | — |
| direct  | 2 (273)  | 0.14558 | 0.16653 (k=89)  | 0.16653 | **oui** | **NON** |
| direct  | 3 (81)   | 0.50599 | 0.54764 (k=85)  | 0.54764 | **oui** | **NON** |
| inverse | 1 (1041) | 0.02560 | 0.02560 (k=0)   | 0.02560 | non (indicatif) | — |
| inverse | 2 (273)  | 0.13103 | 0.15822 (k=98)  | 0.15822 | **oui** | **NON** |
| inverse | 3 (81)   | 0.55094 | 0.62804 (k=98)  | 0.62804 | **oui** | **NON** |

Bandes porteuses (≥10 % de la puissance AC de la référence) : identiques sur
TOUTES les cellules mesurées — `{"1": true, "2-3": true, "4-7": true, "8-15":
false, "16-31": false}` ; la bande `4-7` porte systématiquement le maximum
(cohérent avec la taille caractéristique des bosses du terrain b0, σ=6 cellules).

**f_close (diagnostic non porteur, convention M-0, comparaison instantanée
A_regen vs A_vrai, point d'opération) :**

| Histoire | ℓ | JND=2% | JND=3% | JND=4% | JND=5% |
|---|---:|---:|---:|---:|---:|
| direct  | 1 | 0.5798 | 0.5020 | 0.4434 | 0.3892 |
| direct  | 2 | 0.7603 | 0.6965 | 0.6445 | 0.6013 |
| direct  | 3 | 0.8882 | 0.8452 | 0.8110 | 0.7798 |
| inverse | 1 | 0.6370 | 0.5483 | 0.4749 | 0.4185 |
| inverse | 2 | 0.8135 | 0.7427 | 0.6804 | 0.6260 |
| inverse | 3 | 0.9448 | 0.9124 | 0.8804 | 0.8457 |

### Verdict mécanique

Niveaux sous cap = {ℓ=2 (273), ℓ=3 (81)}. Condition `max(M-A1, M-A2-mini) < 0.02`
pour LES QUATRE cellules (2 histoires × 2 niveaux sous cap) :
`{"direct": {"2": false, "3": false}, "inverse": {"2": false, "3": false}}`
— **aucune** des quatre n'est sous le seuil.

```
Verdict mécanique : T_testable
```

Appliqué mécaniquement depuis le seuil unique 0.02 (JND le plus sévère de
[2,5] %) : la condition « toutes les cellules sous cap, deux histoires, sous
0.02 » est fausse (elle échoue même très largement — les valeurs sous cap vont
de 0.13 à 0.63, un ordre de grandeur au-dessus du seuil) → `T_testable`, pas
`T_triviale`. La compression manche-1 (résumé grossier + régénération
bilinéaire + rescale de masse) sur le substrat v2 n'efface PAS le signal
albedo au sens du JND perceptuel : elle discrimine réellement — c'est le
résultat qui rend la question « le résumé grossier tient-il sous JND ? » non
triviale et donc digne d'être posée par une manche 1 complète (Tasks 5/6 du
plan), si le contrôleur choisit de la relancer malgré le STOP `lecture_2` de
M-0. Diagnostic cohérent : `f_close` (5 %) est très élevé partout (39–85 %),
confirmant que même la fraction du domaine en excès de JND est massive,
pas un artefact de la seule statistique `max_carrier`.

### Déterminisme

Double exécution (`.venv/bin/python scripts/run_arcA_testability.py` deux fois,
90.5 s puis 92.2 s CPU — les tableaux imprimés sont identiques valeur par
valeur) : comparaison programmatique des deux JSON après retrait du champ
`meta.elapsed_seconds` (seul artefact de mesure hors calcul) →
```
IDENTIQUE
```

## Fichiers modifiés / créés

- `src/summary.py` (nouveau).
- `tests/test_summary.py` (nouveau, 21 tests).
- `src/sediment.py` (modifié — additif seulement, fin de fichier : 2 fonctions,
  aucune ligne existante touchée).
- `tests/test_sediment.py` (modifié — import + 4 tests additifs en fin de fichier).
- `scripts/run_arcA_testability.py` (nouveau).
- `outputs/arcA/testability_v2.json` (nouveau, artefact load-bearing, commité
  malgré le `.gitignore` global `outputs/`, comme les autres JSON `outputs/arcA/*`
  déjà commités).
- `.superpowers/sdd/refond-task2-m0bis-report.md` (ce rapport).

## Self-review

- **Complétude spec `summary.py`** : tailles exactes 1041/273/81 testées et
  vérifiées ; suite complète M-A3 (double régénération + identité cross-process
  subprocess+sha256), anti-fuite (signature 1 paramètre + mutation décoy),
  invariants restitués (1e-10 relatif), round-trip constant (exact bit-à-bit,
  constante dyadique — cf. évidence TDD) toutes présentes et vertes.
- **Complétude sonde** : 2 histoires × 3 niveaux couverts pour M-A1 ET
  M-A2-mini ; `f_close` par histoire × ℓ × JND∈{2,3,4,5}% ; JSON contient
  M-A1/M-A2-mini (avec `delta_chi`/`carriers` complets, pas seulement
  `max_carrier`), `f_close`, toutes les constantes demandées (seed, 10 centres,
  11ᵉ centre, S_HALF_OP, tailles, relief, k_d/k_e V2), et le verdict.
- **Verdict appliqué mécaniquement** : seuil unique `0.02` (constante nommée
  `THRESH_TESTABILITE`), niveaux sous cap `{2,3}` (constante `CAP_LEVELS`),
  comparaison `max(M-A1,M-A2-mini) < seuil` pour chacune des 4 cellules
  (2 histoires × 2 niveaux), ET logique, aucune branche d'interprétation :
  le code ne contient aucune condition qui dépendrait du résultat pour choisir
  entre `T_triviale`/`T_testable` autrement que par cette règle unique.
- **Discipline** : `src/solver_wetdry.py`, `src/terrains.py`, `src/render.py`,
  `scripts/run_arcA_revalidate.py`, `scripts/run_arcA_m0.py` non touchés (vérifié
  — `git diff` ne les liste pas). `src/albedo.py` non touché non plus (la
  conversion de types numpy→natif se fait dans `run_arcA_testability.py`, à la
  frontière de sérialisation, pas dans `albedo.py`). Les 105 tests préexistants
  restent verts SANS modification de leur code — seul l'import en tête de
  `tests/test_sediment.py` a été étendu (ajout de `run_episode_trajectoire`) et
  4 tests ajoutés en fin de fichier ; aucune ligne existante de ce fichier ni de
  `src/sediment.py` n'a été modifiée. Pas de constante gelée retouchée
  (`KD_CALIBRE_V2`, `KE_CALIBRE_V2`, `SedimentParams()`, seuils `0.02`/{2,3}
  tous repris tels quels du brief). Suite complète relancée une dernière fois
  avant commit : `.venv/bin/pytest -q` → voir section Tests.

## Préoccupations

- **Bug auto-détecté et corrigé pendant le développement de la sonde** (pas de
  relecteur externe impliqué à ce stade) : le premier run de
  `scripts/run_arcA_testability.py` a crashé à l'écriture du JSON
  (`TypeError: Object of type bool is not JSON serializable`). Cause : sous
  numpy 2.4.6, `chi_bands`/`delta_chi` (`src/albedo.py`, non modifié) produisent
  des scalaires `numpy.float64` pour `max_carrier`/`delta_chi`, et
  `max(numpy.float64, numpy.float64) < seuil` retourne un `numpy.bool` (dont
  `__class__.__name__` est littéralement `"bool"` en numpy 2.x — d'où le
  message d'erreur trompeur) plutôt qu'un `bool` Python natif. Corrigé dans
  `_delta_chi_albedo` (`run_arcA_testability.py`) par une conversion explicite
  vers types Python natifs à la frontière de sérialisation — `src/albedo.py`
  n'a pas été touché (aucun de ses appelants existants n'est affecté par ce
  comportement, qui n'était simplement jamais exercé via `json.dumps`
  auparavant). Les deux runs post-correctif sont bit-identiques (vérifié).
- **Verdict `T_testable`** — résultat honnête, aucun seuil ajusté : les valeurs
  mesurées (0.13–0.63 sur les niveaux sous cap) sont loin au-dessus de 0.02,
  pas une frontière ambiguë — pas de tentation d'infléchir, pas d'ambiguïté
  d'application. C'est un résultat cohérent avec le STOP `lecture_2` déjà
  gravé pour M-0 (§A8) : le readout albedo amplifie fortement les écarts sur ce
  substrat, donc un résumé grossier (qui lisse ces écarts) les manque
  largement — la sonde documente QUANTITATIVEMENT cette même sensibilité, sous
  un angle différent (compression, pas ordre des pulses). Décision de relancer
  ou non la manche 1 laissée au contrôleur, comme demandé.
- `ruff` n'est pas installé dans le venv de ce dépôt (contrairement à
  pocCascade2phys) — pas de lint automatisé exécuté ; relecture manuelle
  effectuée (imports utilisés, pas de code mort, style aligné sur les fichiers
  existants du dépôt).
