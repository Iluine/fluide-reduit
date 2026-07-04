# Arc A — Manche 1 : fermeture état-complet du champ persistant (sédiment)

> **Contrat** : l'addendum §A0–§A5 + les entrées du 2026-07-03 de
> `~/project/python/pocCascade2phys/PREREGISTRATION.md` **font foi verbatim**. Un seuil manqué
> est un résultat, pas un bug. Aucun seuil n'est ajusté après la première mesure.
> Branche : `arc-a-etat-complet`. Tests : `.venv/bin/python -m pytest -q` (79 existants doivent
> rester verts). Stack : numpy/scipy/matplotlib, CPU. Aucune dépendance nouvelle.

## Contexte

Le substrat sédiment validé (journal pocCascade2phys, 2026-06-30 → 2026-07-01) a été perdu
(scratchpad /tmp purgé au reboot). Arbitrage Romain : **reconstruction DURABLE** ici (code commité),
re-validée contre les critères déjà gravés, **sans re-réglage** (calibration unique de k_d autorisée
AVANT la re-validation, puis tout gelé). But de la manche : mesurer si un résumé grossier
déterministe-régénérable du champ sédiment `s` tient le readout albedo sous JND — instantanément
(M-A1) ET sous la dynamique (M-A2) — avec une taille k\*(L) qui sature en L.

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

### - [ ] Task 2 — GATE de re-validation (`scripts/run_arcA_revalidate.py`) — BLOQUANT

Seuils FIGÉS ICI, avant exécution (référence = valeurs du journal d'origine) :
- **(a) path-dependence** : 2 runs L₀=10, même multiset de centres, ordre inversé (seed 7) →
  PASS ssi corr(s_A, s_B) ≤ 0.7 (référence d'origine : 0.39).
- **(b) survie albedo** : f_albedo(ordre vs simultané-même-masse ; JND=5 %, point d'op) ≥ 2 %
  ET f_albedo > f_relief pour TOUT S_HALF ∈ {0.005, 0.01, 0.05, 0.10, 0.20}·relief
  (référence : 3.5–10 % ≫ 1.2 %). Simultané-même-masse = tous les pulses en un épisode
  (superposition des monticules), s final rescalé à la masse de s_ordre.
- **(c) closure f(k)** : histoires jumelles divergentes avant les k derniers pulses (préfixe
  inversé, même multiset), k ∈ {1, 4, 7, 10}, aux DEUX géométries (overlap : centres dans la
  même bande y ∈ [0.4, 0.6] ; séparée : centres sur toute la boîte) → PASS ssi f(k) non-croissante
  et f(10) < JND=5 %.
- Verdict imprimé + écrit dans `outputs/arcA/revalidation.json`. **Un critère manqué = STOP :
  rapporter, ne PAS lancer les tâches suivantes, ne PAS retoucher les paramètres.**

### - [ ] Task 3 — Harnais d'histoires (`scripts/run_arcA_histories.py`)

- Seeds de forçage : {101, 102, 103, 104, 105}. Histoires NESTED : un run de 80 épisodes par seed,
  checkpoints s à L ∈ {10, 20, 40, 80} (préfixes = les histoires courtes).
- Sauvegarde `outputs/arcA/histories/h_s{seed}.npz` : s aux 4 checkpoints, b0, params (asdict),
  seed, versions. Committé.
- Sanity dans le script : deux seeds → f(readout) > 0 entre leurs s finaux (path-dependence au
  readout) ; relance d'un checkpoint → bit-à-bit identique (reproductibilité).

### - [ ] Task 4 — Compresseur + régénérateur (`src/summary.py`, tests)

- `summarize(field, level) -> Summary` : moyenne par blocs dyadiques 2^ℓ, ℓ ∈ {1, 2, 3}
  (Harten cell-average) + invariants scalaires : masse totale + masses par sous-domaine 4×4
  (16 blocs de 16×16). `Summary` = dataclass gelée (coarse, invariants, level, shape) ;
  `size_floats(summary) -> int` (ℓ=1 : 1024+17 = 1041 ; ℓ=2 : 273 ; ℓ=3 : 81 ; champ fin = 4096).
- `regenerate(summary) -> np.ndarray` — UN paramètre. Upsampling bilinéaire du coarse, clip ≥ 0,
  puis rescale multiplicatif par sous-domaine 4×4 pour restituer exactement les 16 masses
  (division protégée : bloc de masse nulle → bloc nul), vérification masse totale.
  Si une stochasticité s'avérait nécessaire : `default_rng(int(sha256(bytes_du_summary), 16) % 2**63)`.
- Tests (`tests/test_summary.py`) : M-A3 (double régénération → écart relatif ≤ 1e-12, ET
  identité cross-process via subprocess + hash) ; anti-fuite (signature à 1 paramètre via
  `inspect.signature` ; régénérer après mutation d'un champ vrai décoy → résultat inchangé) ;
  tailles exactes ; invariants restitués (masses 4×4 à 1e-10 relatif) ; round-trip sur champ
  constant = exact.

### - [ ] Task 5 — Harnais de mesure M-A1/M-A2 (`scripts/run_arcA_measure.py`)

Pour chaque (L ∈ {10,20,40,80}, seed ∈ {101..105}, ℓ ∈ {1,2,3}) :
- **M-A1** : Δχ(albedo(regenerate(summarize(s_L, ℓ))), albedo(s_L)) au point d'op S_HALF.
- **M-A2** : rollout T_fwd = 1 épisode — même centre de pulse (l'épisode L+1 de la même seed de
  forçage) — depuis s_regen ET depuis s_L (vrai) ; Δχ à chaque snapshot d'épisode entre les deux
  trajectoires d'albedo ; statistique = max le long du rollout.
- Critère par cellule : sous-JND ssi max(M-A1, M-A2) < JND, évalué pour JND ∈ {2, 3, 4, 5} %.
- Sortie `outputs/arcA/measures.npz` indexée (L, seed, ℓ, jnd) + méta. Committé.

### - [ ] Task 6 — k\*(L), verdict mécanique, figures (`scripts/run_arcA_verdict.py`)

- k\*(L, seed ; JND) = plus petite taille (81 → 273 → 1041) sous-JND ; ∞ si aucune. k\*(L ; JND) =
  médiane des seeds, avec la clause « aucune seed > 2×JND » (si une seed dépasse 2×JND au niveau
  retenu par la médiane → k\*(L) passe au niveau supérieur ; si aucun niveau ne satisfait → ∞).
- Pente de k\*(L) sur la moitié haute (L ∈ {40, 80}) + IC 95 % bootstrap inter-seeds (10 000 tirages).
- Verdict §A3 appliqué MÉCANIQUEMENT, par JND de la plage {2,3,4,5} % : PASS ssi IC de pente ∋ 0
  ET k\*(80) ≤ 409.6 floats (10 % de 4096) ; FAIL ssi pente > 0 hors IC sur toute la plage sans
  plateau ; sinon INDÉTERMINÉ. Verdict global retenu ssi stable sur TOUTE la plage JND, sinon
  INDÉTERMINÉ (sensibilité §A3).
- Figures (`outputs/arcA/`) : `arcA_kstar_vs_L.png` (une courbe par JND, dispersion inter-seeds),
  `arcA_jnd_sensitivity.png`, GIF `arcA_worst_case.gif` (albedo vrai vs régénéré, pire cellule
  M-A2). Impression du verdict citant la cellule §A0 sélectionnée. **Aucune interprétation
  au-delà de §A5** — l'entrée de journal est écrite par le contrôleur, pas par ce script.

---

## AMENDEMENTS — Relance sur v2 (2026-07-04, gravés au contrat `8e02dd6` AVANT lancement)

Historique : gate Task 2 FAIL (×2, reconstruction close) → M-0 lecture_2 → M-0bis T_testable →
arbitrage Romain : relance des Tasks 3/5/6 SUR v2. Task 4 (`src/summary.py`) : FAITE et revue
sous M-0bis (commits 32e9004..cc61056). Le reste du plan est INCHANGÉ, plus les amendements :

- **(i) Substrat = v2 gelé** (`SedimentParams()` par défaut). Assumé : pas l'original (gate
  d'identité FAIL) ; claims lus sur CE substrat (§A5). Testabilité établie par M-0bis.
- **(ii) Contrôles d'instrument synthétiques (remplacent §A11)**, pipeline complet aux cellules
  L ∈ {10, 80} × seeds {101..105}, JND {2..5} % :
  - contrôle-fermable : `s_ferm = regenerate(summarize(s_L, 3))` — attendu k\* = 81 partout ;
  - contrôle-infermable : `s_shuf` = permutation des 4096 cellules de s_L par
    `default_rng(9001 + 1000·L + seed)` — attendu k\* = ∞ partout.
  - Toute cellule de contrôle qui viole son attendu → STOP, remonter (gate d'instrument type G0).
  - Les contrôles ne participent PAS au verdict §A3 ; ils conditionnent le droit de le lire.
- Exécution : Task 3 (histoires) → Task 5 (mesures, étendue aux contrôles) → Task 6 (verdict).

## AMENDEMENT INSTRUMENT — Régénérateur constant-par-blocs (2026-07-04, gravé au contrat `9bcb09a`)

Suite au gate d'instrument en VIOLATION (ferm 40/40, plancher bilinéaire supra-JND à ℓ=3) :

- **Spec Task 4 amendée** : `regenerate(summary)` = **upsampling CONSTANT-PAR-BLOCS** (chaque
  bloc 2^ℓ rempli de sa valeur coarse — Harten ordre 0), clip ≥ 0 conservé (no-op sur champ ≥ 0),
  **rescale remplacé par une VÉRIFICATION** des 16 masses 4×4 (1e-12 relatif, `raise` si violée)
  sans multiplication. Propriétés à TESTER : S∘R = id **bit-à-bit** (summarize(regenerate(x),
  level) == coarse exact) ; idempotence R∘S∘R∘S = R∘S bit-à-bit ; M-A3/anti-fuite/tailles
  inchangés. Le bilinéaire est conservé sous `regenerer_bilineaire` (audit, reproductibilité
  M-0bis/run v1), plus jamais utilisé par les mesures.
- **Clause 1** : issue « k\* = ∞ uniforme (aucune fermeture sous cap à aucun L, JND donné) »
  pré-écrite = **INDETERMINE_CAPACITE** (famille insuffisante à ce JND ; PAS le mur ; pas de
  cellule §A0 2/3 ; fork famille-vs-manche-2 remonté). Task 6 amendée mécaniquement pour émettre
  ce label. Prédiction directionnelle gravée : M-A1/M-A2 v2 montent à chaque ℓ.
- **Clause 2** : diagnostic non-bloquant, cellule (L=10, seed=101, ℓ=3) : trajectoire complète
  Δχ(t) du rollout s_ferm vs s_L, sérialisée ; rapporter Δχ(0), max_t Δχ(t), rapport.
- **Clause 3** : contrôle-fermable requalifié en contrôle de PLOMBERIE (passe par construction) ;
  le contrôle discriminant est le shuf.
- Re-run Tasks 5–6 (mêmes scripts, mêmes seuils, nouvelle sémantique de `regenerate` par import),
  remontée du verdict dans tous les cas.
