# Task 1 — Rapport d'implémentation (substrat sédiment durable)

Branche : `arc-a-etat-complet`. Brief : `.superpowers/sdd/task-1-brief.md`.

## Ce qui a été fait

- `src/sediment.py` : `default_terrain`, `SedimentParams` (dataclass gelée),
  `run_episode`, `run_history`, `calibrate_kd`, + helpers privés `_relax_episode`
  (pulse + relaxation `simulate_wetdry_o2` tronquée à `N_settle` pas),
  `_exner_step`/`_integrate_exner` (loi de dépôt/érosion).
- `src/albedo.py` : `albedo`, `relief_shaded`, `f_fraction`, `chi_bands`, `delta_chi`.
- `tests/test_sediment.py` (11 tests), `tests/test_albedo.py` (11 tests).
- `simulate_wetdry_o2` et les autres points d'intégration (`config.py`,
  `src/terrains.py`, `src/render.py`) n'ont pas été modifiés.

## Décision d'implémentation importante : N_settle vs `t_end`

`simulate_wetdry_o2` n'expose que `t_end` (temps physique) ; `dt` est CFL-adaptatif
et il n'existe aucun paramètre « nombre de pas ». Pour obtenir exactement les
`N_settle=600` pas prescrits **avec le vrai `dt` interne** (pas un `dt` forcé
artificiellement petit, ce qui changerait la physique), `_relax_episode` :
1. appelle le solveur une seule fois avec un budget de temps physique généreux
   `_T_END_RELAX` (constante module, cf. commentaire dans `sediment.py`) ;
2. tronque la trajectoire retournée aux 601 premiers snapshots (indices 0..600).

`_T_END_RELAX` a été calibré empiriquement (balayage de 15 centres aléatoires +
les 4 coins de la boîte [0.15,0.85]², avec le b0/pulse FIGÉS) : à `t_end=130`, le
nombre de pas acceptés va de 1160 (pire cas observé) à 2664, toujours largement
au-dessus de 600. `_T_END_RELAX=130` est utilisé (garde-fou `RuntimeError` dans le
code si jamais un centre venait à violer cette marge). Le reliquat de trajectoire
au-delà du pas 600 est calculé puis jeté — c'est le prix de piloter un solveur
`t_end`-only par un budget de pas sans y toucher. **Preuve que cette troncature ne
biaise pas la physique** : `_cfl_dt` ne dépend que de l'état courant (jamais de
`t_end`), et le seul endroit où `t_end` influence une trajectoire est
l'écrêtage du tout dernier pas (`if t+dt>t_end: dt=t_end-t`) — qui ne peut se
produire qu'au voisinage de `t_end`, jamais avant le pas 600 tant que `t_end` est
assez grand. Confirmé empiriquement : la calibration `k_d` a été exécutée deux
fois, une fois avec `_T_END_RELAX=150` (première tentative) puis une fois avec
`_T_END_RELAX=130` (valeur finale) — **même seed, mêmes 12 évaluations de
bissection, résultat bit-à-bit identique** : `KD_CALIBRE = 0.0019109529749704406`
dans les deux cas.

## Constat majeur (à valider par le contrôleur) : conservation de masse

**`simulate_wetdry_o2` (schéma `_rhs_o2`, 2e ordre bien équilibré MUSCL) n'est PAS
exactement conservatif en masse une fois qu'un front mouillé/sec atteint un mur
réfléchissant avec un gradient de surface non nul.**

Diagnostic (reproductible, cf. mesures ci-dessous) : au mur, la pente MUSCL côté
cellule fantôme est gelée à 0 par construction (« 1er ordre au mur », documenté
dans le code du solveur), mais la cellule réelle adjacente garde sa propre pente
(non nulle dès qu'il y a un gradient de surface réel près du mur). Ceci casse
l'antisymétrie exacte du flux miroir qui garantit `Fh_mur=0` pour un schéma 1er
ordre ou au repos. Mesure directe du flux de masse à l'interface du mur une fois
l'onde arrivée : `Fh_x ≈ ±9e-4` (non nul), contre exactement `0.0` avant que
l'onde n'atteigne le mur.

Impact mesuré sur un épisode complet (N_settle=600, b0/pulse FIGÉS) : la dérive de
masse relative entre le 1er et le 600e snapshot va de **~3 % à ~13 %** selon le
centre du pulse (mesures sur ~9 centres, dont les 4 coins et le centre de la boîte
[0.15,0.85]²) — trois à quatre ordres de grandeur au-dessus de la tolérance 1e-8
prescrite dans la résolution d'ambiguïté (« le solveur est conservatif par flux »).

Vérification que ce n'est PAS un bug de `run_episode`/l'intégration Exner : le même
comportement apparaît en appelant `simulate_wetdry_o2` **seul**, sur un terrain
plat (b=0) avec juste un pulse sur fond sec — drift +10.8 % sur `t_end=30` d'un cas
minimal, alors qu'un cas **entièrement mouillé** (pas de front sec, même
pulse/terrain) donne un drift de -2.7e-5 (conforme à une tolérance de type 1e-8 à
1e-5, cohérent avec la résolution d'ambiguïté dans CE régime). L'instrumentation
du bilan de positivité (`_floor_dry`) montre que le "plancher de positivité" ne
peut pas expliquer la dérive (retrait cumulé mesuré ~0.44 sur un total ~57, très
inférieur à la dérive observée) — la fuite vient de la reconstruction MUSCL au
mur elle-même, pas d'un garde-fou de positivité.

**Conséquence pour ce Task 1** : le test de conservation de masse
(`test_mass_conserved_within_measured_tolerance`) utilise une tolérance mesurée
et documentée (`< 0.20`, marge sur le ~13.1 % mesuré pour le centre de test
`(0.5, 0.5)`), **PAS** 1e-8. Je n'ai pas déplacé un seuil scientifique de verdict
(aucun n'existe encore à ce stade — Task 1 est instrumental, pas un gate C1-C4) ;
j'ai corrigé une hypothèse d'implémentation qui s'est révélée fausse à la mesure,
et je le documente au lieu de le cacher. Le calcul du dépôt Exner (`k_d·h·...`)
reste bien défini indépendamment de cette dérive : il consomme le (h,u,v) réel du
solveur à chaque snapshot, quelle que soit la fidélité de la conservation
globale — mais **le champ `s` calibré ci-dessous est donc calibré contre un `h`
qui n'est pas un système fermé physiquement conservatif** une fois le mur touché.
Je n'ai PAS édité `solver_wetdry.py` (interdit par le contrat) donc je n'ai pas pu
corriger la cause racine.

## Calibration k_d

Commande de reproduction exacte (aussi inscrite en commentaire dans
`src/sediment.py` au-dessus de `KD_CALIBRE`) :

```bash
.venv/bin/python -c "
from config import GridConfig
from src.sediment import default_terrain, calibrate_kd
b0 = default_terrain(GridConfig())
print(calibrate_kd(b0))"
```

- Bissection sur `log10(k_d) ∈ [-6, 1]`, cible `max(s)/relief ∈ [0.18, 0.22]`
  (± 0.02 autour de 0.2), seed=12345, 10 épisodes, max 10 itérations.
- **Convergée** (pas de BLOCKED) : `KD_CALIBRE = 0.0019109529749704406`,
  `max(s)/relief` mesuré à convergence ≈ 0.2 (dans la bande [0.18, 0.22]).
- `k_e = k_d / 5 = 3.8219059...e-04` (dérivé, FIGÉ a priori).
- Exécutée deux fois (deux valeurs de `_T_END_RELAX` différentes, cf. section
  ci-dessus) : résultat bit-à-bit identique dans les deux cas → forte confiance
  dans la reproductibilité.
- Coût : ~15-17 minutes (12 évaluations de `run_history(seed,10,...)` × ~10
  épisodes chacune × ~3-4s/épisode, domaine 64×64, CPU).

## Résolution d'ambiguïté "min_depth" pour le seuil θ (loi Exner)

`solver_wetdry.py` n'expose pas de paramètre `min_depth` séparé (seulement
`dry_eps`, défaut `1e-4`) — `config.SolverConfig.min_depth` appartient à l'AUTRE
solveur (`src/solver.py`), pas à `solver_wetdry.py`. J'ai interprété
« h > 10·min_depth » de la spec comme `h > 10·dry_eps` (= `1e-3`), en utilisant le
`dry_eps` par défaut du module concerné. Documenté dans le docstring de
`_exner_step`.

## Tests — commandes et résultat

```bash
.venv/bin/python -m pytest -q
```

```
101 passed in 94.42s (0:01:34)
```

(79 tests existants + 11 `tests/test_sediment.py` + 11 `tests/test_albedo.py`,
tous verts, aucun warning.)

Détail des tests sédiment notables :
- `test_mass_conserved_within_measured_tolerance` : dérive mesurée < 0.20 (cf.
  constat ci-dessus) — PAS 1e-8.
- `test_s_nonnegative_after_episode`, `test_s_unchanged_on_never_wetted_cells`
  (560/4096 cellules jamais mouillées identifiées pour le centre de test
  `(0.5,0.5)`, N_settle=600) — passent avec la vraie valeur `KD_CALIBRE`
  (dépôt non trivial, donc le test n'est pas vide de sens).
- `test_run_episode_is_pure_no_mutation`, `test_run_episode_deterministic_bitwise`,
  `test_run_history_checkpoints_and_determinism` : bit-à-bit identiques.
- `test_path_dependence_two_seeds_differ`, `test_path_dependence_reversed_order_differs` :
  smoke, PAS la re-validation (Task 2).

## Budget de calcul (avertissement pour la suite)

Chaque appel à `run_episode`/`_relax_episode` coûte ~3-4 s (domaine 64×64, CPU,
`_T_END_RELAX=130` → ~1200-2600 pas solveur réellement calculés, dont seuls les
600 premiers sont utilisés). `run_history(seed, 10, ...)` coûte donc ~35-40 s.
Un run complet Arc A Task 3 (80 épisodes × 5 seeds = 400 épisodes) coûterait de
l'ordre de **20-25 minutes** avec les paramètres actuels — à anticiper pour la
suite (Task 2/3), pas bloquant pour Task 1.

**Avertissement d'environnement** : dans cette session, les process Python
lancés en arrière-plan via `nohup ... &` manuel (en dehors du mécanisme
`run_in_background` du harnais) ont été tués sans trace à deux reprises,
probablement à cause d'un teardown de sandbox entre appels d'outil. Les calculs
longs (>2 min) doivent passer par le paramètre `run_in_background` du tool Bash
(ou tenir dans le timeout foreground de 600 s), pas par `nohup` manuel.

## Doutes / points à trancher par le contrôleur

1. **Conservation de masse (voir constat ci-dessus)** — c'est le point le plus
   important de ce rapport. La résolution d'ambiguïté « tolérance 1e-8 » ne tient
   pas empiriquement dès qu'un front mouillé/sec touche un mur réfléchissant, ce
   qui arrive systématiquement dans la fenêtre N_settle=600 (l'onde met
   ~13-17 pas physiques à atteindre le mur, très en-deçà des ~600 pas de la
   relaxation). Options pour la suite : (a) accepter la dérive comme propriété
   connue du solveur figé et garder une tolérance mesurée (ce qui a été fait ici) ;
   (b) réduire N_settle pour rester avant le premier contact au mur (changerait
   la physique de l'épisode, contredit le N_settle=600 figé) ; (c) modifier le
   terrain/pulse pour amortir davantage avant contact (hors périmètre Task 1,
   b0/pulse sont figés) ; (d) accepter que le champ `s` résultant est calibré
   contre un système hydrologique non strictement fermé et le documenter comme
   limitation connue dans la suite Arc A (Task 2 GATE de re-validation devra en
   tenir compte s'il compare des quantités sensibles à la masse totale).
2. Interprétation N_settle=600 comme "pas solveur tronqués depuis un run à t_end
   généreux" (cf. section dédiée) — pas d'autre mécanisme disponible vu le
   contrat de `simulate_wetdry_o2` (t_end-only, pas de n_steps). Prouvé neutre
   sur le résultat par calibration bit-à-bit reproductible à deux `_T_END_RELAX`
   différents.
3. `min_depth` interprété comme `dry_eps` (cf. section dédiée) — `solver_wetdry.py`
   n'a pas de paramètre `min_depth` séparé.
4. `KD_CALIBRE` a été calculé et gelé UNE fois puis vérifié une seconde fois par
   reproductibilité (pas une deuxième calibration indépendante avec une graine
   différente) — conforme à « exécutée une fois ».
