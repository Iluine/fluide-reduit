# Rapport Task 1 — M-0 (§A8) : borne d'histoire-readout du substrat v2

Brief : `.superpowers/sdd/refond-task1-brief.md`. Plan : `docs/superpowers/plans/arc-a-refondation.md`
(Task 1, ligne 84-88). Contrat : addendum §A6–§A11 de `PREREGISTRATION.md` (dépôt pocCascade2phys,
commit `598fabd`).

## Ce qui a été implémenté

`scripts/run_arcA_m0.py` (nouveau, autonome) :

- Réutilise **sans modification** `scripts/run_arcA_revalidate.py` : import direct de
  `_draw_centers`, `_run_sequence`, `GRID`, `B0`, `PARAMS`, `RELIEF`, `S_HALF_FRACTIONS`,
  `S_HALF_OP`. Aucune ligne de `run_arcA_revalidate.py` n'a été touchée — ses sorties JSON
  restent bit-identiques par construction (fichier non modifié, non ré-exécuté ici).
- Génère les deux histoires du protocole (a) du gate v2 : `default_rng(7)`, 10 centres dans
  [0.15, 0.85]², séquence directe puis inversée, `SedimentParams()` par défaut (= constantes
  V2 gelées `KD_CALIBRE_V2`/`KE_CALIBRE_V2`).
- Calcule `A_direct = albedo(s_direct, s_half)`, `A_inversé = albedo(s_inversé, s_half)` pour
  chaque `S_HALF ∈ {0.005, 0.01, 0.05, 0.10, 0.20}·relief`, et
  `f_ordre(jnd, s_half) = fraction(|A_direct − A_inversé| / ⟨(A_direct+A_inversé)/2⟩ > jnd)`
  pour chaque `JND ∈ {2, 3, 4, 5} %` — balayage complet 5×4 = 20 cellules.
- Diagnostic non porteur : `corr(s_direct, s_inversé)` (Pearson) et `f_ordre` du même protocole
  appliqué au readout `relief_shaded` (plancher de canal, pas de `S_HALF`).
- Verdict mécanique : cellule porteuse = (JND=5 %, S_HALF=0.05·relief) ; `lecture_1` si
  `f_ordre ≤ 0.01`, `lecture_2` sinon. Aucune autre condition.
- Écrit `outputs/arcA/m0_v2_readout.json` (balayage complet + diagnostics + constantes utilisées
  + verdict) et imprime un tableau lisible.

### Écart de convention documenté (pas une ambiguïté)

`f_fraction` de `src/albedo.py` normalise par la moyenne d'un **unique** champ de référence
`⟨y⟩` (l'un des deux membres comparés est désigné comme référence). Le contrat §A8 demande
explicitement `⟨A⟩ = moyenne de (A_direct + A_inversé)/2` — direct et inversé jouent ici un rôle
**symétrique**, aucun des deux n'est privilégié. Ces deux conventions de dénominateur diffèrent
(champ unique vs moyenne des deux champs) ; `f_fraction` n'a donc pas été réutilisée telle
quelle. J'ai réimplémenté `_f_ordre` en miroir exact de `f_fraction` (même garde-fou du cas
dégénéré `⟨A⟩=0` : rapport nul là où l'écart brut est nul, `+inf` sinon) mais avec le
dénominateur symétrique prescrit par le contrat. Ce n'est pas un point bloquant : le contrat
donne la formule exacte à calculer, il n'y avait pas d'ambiguïté sur *quoi* calculer, seulement
sur la réutilisation littérale de la fonction existante.

## Tests

- Aucun test dédié ajouté : le script de mesure ne factorise aucun nouveau code partagé (il
  importe `scripts/run_arcA_revalidate.py` tel quel, sans le modifier), conformément à la
  consigne du contrôleur ("le script de mesure lui-même n'exige pas de test dédié").
- Suite complète exécutée deux fois autour de l'implémentation (aucune régression) :
  `.venv/bin/pytest -q` → **105 passed in 118.72s**.
- Ciblé au préalable : `.venv/bin/pytest -q tests/test_albedo.py tests/test_sediment.py` →
  **26 passed in 114.00s**.
- Déterminisme vérifié empiriquement : deux exécutions successives de
  `scripts/run_arcA_m0.py` produisent un JSON **bit-identique** (hors `elapsed_seconds`,
  seul artefact de mesure hors calcul) — comparaison programmatique `a == b` après retrait du
  champ de timing → `IDENTIQUE`.

## Chiffres de la mesure

Exécution : `.venv/bin/python scripts/run_arcA_m0.py`, temps de calcul ~57.6 s CPU.

`corr(s_direct, s_inversé) = 0.9445524777450136` (référence origine gravée au journal, dernière
itération v2 : 0.9446 — correspondance quasi exacte, confirme que `SedimentParams()` par défaut
sur cette branche est bien la V2 co-calibrée utilisée dans la dernière mesure du gate).

Tableau complet `f_ordre(JND, S_HALF)` (fraction du domaine, readout albedo) :

| S_HALF (·relief) | JND=2 % | JND=3 % | JND=4 % | JND=5 % |
|---:|---:|---:|---:|---:|
| 0.005 | 0.8889 | 0.8552 | 0.8252 | 0.7942 |
| 0.010 | 0.8870 | 0.8464 | 0.8147 | 0.7856 |
| **0.050** | 0.8621 | 0.8196 | 0.7871 | **0.7595** |
| 0.100 | 0.8540 | 0.8110 | 0.7778 | 0.7483 |
| 0.200 | 0.8513 | 0.8042 | 0.7720 | 0.7407 |

Diagnostic non porteur — même protocole sur le readout `relief_shaded` (plancher de canal, pas
de `S_HALF`) :

| JND | 2 % | 3 % | 4 % | 5 % |
|---:|---:|---:|---:|---:|
| f_ordre (relief) | 0.001465 | 0.000977 | 0.000732 | 0.000488 |

**Cellule porteuse** (JND=5 %, S_HALF=0.05·relief, prescrite par le contrat) :
`f_ordre = 0.759521484375`, seuil `≤ 0.01` → **`lecture_2`**.

### Verdict mécanique

```
LECTURE 2 : f_ordre(porteuse) = 0.7595 > 0.01
```

Appliqué mécaniquement depuis le seuil du brief, sans interprétation : `0.7595 > 0.01` donc
`lecture_2`. Selon le contrat §A8 et le plan (ligne 88, ligne 49-51) : **STOP, remonter** — la
hiérarchie corr↔readout est inversée (corr=0.9446 très élevé, suggérant une quasi-commutativité
de l'ordre des pulses ; mais le readout albedo révèle un écart perceptible sur ~76 % du domaine
à la cellule porteuse), le gate §A9 doit être repensé avant tout build. Tasks 2–4 du plan Arc A
re-fondation ne sont **pas** lancées (bloquant par construction du plan).

Le plancher de canal (relief ombré) confirme que ce n'est PAS un artefact générique de la
méthode : sur le même protocole direct/inversé, le readout relief ne voit presque rien
(f_ordre ≤ 0.15 %), alors que le readout albedo voit ~76-89 % — l'amplification est spécifique
au readout sédiment saturant, cohérent avec la lecture 2 (« le readout AMPLIFIE une histoire que
corr sous-estime »).

## Fichiers modifiés / créés

- `scripts/run_arcA_m0.py` (nouveau).
- `outputs/arcA/m0_v2_readout.json` (nouveau, artefact load-bearing — ajouté malgré le
  `.gitignore` global `outputs/`, comme les autres JSON `outputs/arcA/*` déjà commités,
  conformément au plan : « AUCUN artefact load-bearing hors git »).
- `.superpowers/sdd/refond-task1-report.md` (ce rapport, nouveau).

`.superpowers/sdd/progress.md` a une modification préexistante non commitée (entrée Task 0,
antérieure à cette tâche) : je ne l'ai pas touchée, elle n'est pas de mon fait et hors du
périmètre de cette tâche (le brief ne demande pas de mise à jour de ce ledger).

## Self-review

- **Complétude** : balayage complet 4 JND × 5 S_HALF (20 cellules) couvert ; diagnostics
  (corr + plancher relief) rapportés ; verdict mécanique appliqué depuis le seuil unique
  prescrit ; JSON contient balayage complet + diagnostics + verdict + constantes (grille,
  k_d/k_e V2, relief, seed, centres, S_HALF_op). Conforme au brief.
- **Qualité** : noms clairs (`_f_ordre`, `balayage`, `cellule_porteuse`), code en français
  (docstrings, commentaires, messages imprimés), style aligné sur `run_arcA_revalidate.py`
  (mêmes conventions de nommage, mêmes formats de clés JSON `f"{frac:.3f}"`).
- **Discipline (YAGNI)** : pas d'argparse/mode `--smoke` (pas demandé, la mesure n'a pas de
  sens en version dégradée — contrairement au gate v2 qui devait composer ~160 épisodes et
  gérer une reprise par phases, ici 20 épisodes tiennent dans un seul appel) ; pas de
  sauvegarde `.npz` des champs bruts (non demandée, la reproductibilité bit-à-bit vérifiée
  suffit à recalculer `s_direct`/`s_inversé` à la demande) ; aucune constante gelée touchée ;
  `src/solver_wetdry.py`, `src/terrains.py`, `src/render.py` non touchés.
- **Verdict** : appliqué mécaniquement (`f_ordre_porteur <= THRESH_F_ORDRE_PORTEUSE`, seuil
  `0.01` figé en constante nommée, comparaison unique, aucune branche d'interprétation
  supplémentaire dans le code).

## Préoccupations

- Le résultat est `lecture_2` (STOP) — c'est un résultat honnête, non un échec d'implémentation :
  aucun seuil n'a été ajusté, aucune tentative de faire pencher la mesure vers `lecture_1`. Le
  plan (ligne 49-51) anticipe explicitement ce point d'arrêt et prescrit de remonter plutôt que
  d'enchaîner sur Task 2 — je m'arrête donc ici, comme demandé par le contrôleur dans le message
  de tâche ("les deux issues sont également acceptables — n'infléchis RIEN pour obtenir l'une ou
  l'autre").
- L'écart de convention avec `f_fraction` (dénominateur symétrique vs. dénominateur
  référence-unique) est documenté ci-dessus et dans le docstring du script ; à faire confirmer
  par le contrôleur si une autre lecture de "même convention ⟨A⟩ que f_fraction" était attendue.
