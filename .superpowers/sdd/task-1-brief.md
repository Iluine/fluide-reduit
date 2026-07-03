## Global Constraints (toutes les tâches)

- **Ne pas éditer** `src/solver_wetdry.py`, `src/terrains.py`, `src/render.py` — les envelopper.
- **Jamais L2 sur l'état comme critère.** Toute figure d'erreur d'état = diagnostic, marquée
  « non-critère » dans son titre.
- **Anti-fuite** : `regenerate(summary)` — signature à UN paramètre, aucun global d'état, aucune
  lecture fichier, aucune vue du champ vrai ni de l'historique. Vérifié par test.
- **Déterminisme** : toute stochasticité du régénérateur seedée par SHA-256 du résumé sérialisé
  (si le régénérateur est purement déterministe, la clause est satisfaite trivialement mais le
  test M-A3 reste dû : double appel → identité relative ≤ 1e-12).
- **Durabilité** : tout artefact load-bearing (npz d'histoires, résultats, figures) sous
  `outputs/arcA/` et COMMITÉ. Rien en /tmp.
- Code en **français** (docstrings/commentaires), interfaces publiques typées, contrats
  formes/dtypes documentés (indexation `array[y, x]` comme le reste du dépôt), pas de cleverness.
- Reproductibilité : `numpy.random.default_rng(seed)` exclusivement, seeds explicites.

## Paramètres physiques FIGÉS (avant toute mesure — ne pas retoucher)

- Grille : `GridConfig` par défaut (64×64, dx=dy=1). Solveur : `simulate_wetdry_o2`
  (limiteur par défaut minmod), `min_depth`/`dry_eps` par défaut du module.
- **Terrain b0 (FIXE, identique pour tous les runs — asset statique connu)** : plan incliné le
  long de x (chute totale 0.5 de x=0 à x=W−1) + 3 bosses gaussiennes fixes
  (centres (0.3, 0.35), (0.6, 0.6), (0.45, 0.75) en fractions (x_frac, y_frac), σ = 6 cellules,
  amplitudes 0.25 / 0.30 / 0.20). **relief ≡ max(b0) − min(b0)**. Domaine initialement SEC (h=0).
- **Épisode (= 1 pulse)** : monticule d'eau gaussien ajouté à h : centre tiré uniformément dans la
  boîte [0.15, 0.85]² (fractions), σ_pulse = 5 cellules, hauteur pic h_p = 0.6·relief →
  relaxation `simulate_wetdry_o2` sur **N_settle = 600 pas** (dt CFL interne), `save_every = 2` →
  assèchement : h, hu, hv remis à 0 ; `s` persiste. Forçage d'un run = séquence des centres tirés
  de `default_rng(seed_forçage)`.
- **Loi de dépôt (type Exner, appliquée séquentiellement sur les snapshots de l'épisode,
  b_eff gelé pendant l'épisode, mis à jour ENTRE épisodes : b_eff = b0 + s)** :
  θ = u² + v² sur cellules mouillées (h > 10·min_depth), et
  `ds/dt = k_d · h · 1[θ<θ_c] · (1 − θ/θ_c) − k_e · s · 1[θ>θ_c] · (θ/θ_c − 1)`,
  intégrée en Euler explicite avec le dt de chaque snapshot (×save_every), s ≥ 0 clampé.
  **θ_c = 0.5 ; k_e = k_d/5 (FIGÉS a priori).** k_d : voir calibration unique (Task 1).
- **Point d'opération (gravé au journal d'origine)** : k_d calibré pour max(s) ≈ 20 % du relief
  après L₀ = 10 pulses (seed de calibration = 12345, jamais réutilisée ensuite).
- **Readout albedo** : `A = 1 − exp(−s / S_HALF)`, S_HALF = 0.05·relief au point d'op ;
  robustesse balayée sur S_HALF ∈ [0.005, 0.20]·relief.
- **Readout relief (référence de re-validation uniquement)** : ombrage lambertien rasant de
  b0 + s, source à 15° d'élévation, azimut 45°, normalisé [0,1].

## Instruments (gravés au journal pocCascade2phys 2026-07-03 — répliquer exactement)

- **f(X, Y ; JND)** (re-validation) : fraction du domaine où |readout(X) − readout(Y)| / ⟨readout(Y)⟩
  > JND. JND ∈ {2 %, 5 %}.
- **Δχ par bande (M-A1/M-A2)** : FFT 2D de A, puissance par octave radiale
  k ∈ {1, 2-3, 4-7, 8-15, 16-31} cycles/domaine ; χ_b = RMS(bande)/⟨A⟩ ; bandes PORTEUSES = ≥ 10 %
  de la puissance AC totale ; Δχ_b = |χ_b(candidat) − χ_b(vrai)| ; **critère = max sur bandes
  porteuses vs JND**, JND balayé sur {2, 3, 4, 5} %.

## Tasks
### - [ ] Task 1 — Substrat sédiment durable (`src/sediment.py`, `src/albedo.py`, tests)

`src/sediment.py` :
- `default_terrain(grid) -> np.ndarray` — le b0 FIGÉ ci-dessus.
- `run_episode(s, b0, center_frac, params) -> np.ndarray` — un épisode complet (pulse → relaxation
  via `simulate_wetdry_o2` avec b = b0 + s gelé → intégration Exner sur les snapshots → assèchement),
  retourne le nouveau `s`. Pur (pas de mutation d'argument).
- `run_history(seed, n_episodes, b0, params, checkpoints=None) -> dict[int, np.ndarray]` — séquence
  seedée d'épisodes, retourne {n: s_après_n_épisodes} aux checkpoints demandés.
- `SedimentParams` dataclass gelée (k_d, k_e, θ_c, σ_pulse, h_p, N_settle, save_every) avec les
  valeurs FIGÉES par défaut ; k_d par défaut = la valeur calibrée (voir ci-dessous).
- `calibrate_kd(b0, target=0.2, seed=12345) -> float` — recherche (bissection sur log k_d, ~8 iters)
  pour max(s)/relief ≈ target à 10 épisodes. Exécutée UNE fois ; la valeur obtenue est inscrite en
  constante module `KD_CALIBRE` avec un commentaire donnant la commande de reproduction. Après le
  gel, plus aucune retouche.

`src/albedo.py` :
- `albedo(s, s_half) -> np.ndarray`, `relief_shaded(b) -> np.ndarray` (specs ci-dessus),
- `f_fraction(x, y, jnd) -> float`, `chi_bands(a) -> (chi, puissance_par_bande)`,
  `delta_chi(a_cand, a_ref) -> dict` (Δχ par bande + masque porteuses + max porteuses).

Tests (`tests/test_sediment.py`, `tests/test_albedo.py`) :
- épisode : masse d'eau conservée pendant la relaxation (avant assèchement, tolérance solveur) ;
  s ≥ 0 ; s ne bouge pas sur cellules jamais mouillées ; déterminisme (même seed → même s, bit-à-bit).
- path-dependence sanity (2 seeds → s différents ; ordre inversé → s différents) — smoke, PAS la
  re-validation.
- albedo : monotone en s, sature vers 1 ; f_fraction(x,x)=0 ; chi_bands : puissances somment à la
  puissance AC ; delta_chi(a,a) = 0.
- La suite existante (79 tests) reste verte.

