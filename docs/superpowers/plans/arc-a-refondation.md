# Arc A — Re-fondation du substrat sédiment (suspension) — Plan de mission

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.
> Steps en cases `- [ ]`. Gravé le 2026-07-04, AVANT tout code de re-fondation.

**Goal :** (1) exécuter M-0 (borne d'histoire-readout du v2, ~2 min) ; (2) construire le
substrat re-fondé à transport en suspension (BC périodiques, conservation en assert) ;
(3) balayage Goldilocks de w_s ; (4) gate de conception (a')/(b')/(c'). S'arrêter là et
remonter — les mesures k\*(L) de la manche 1 ne se lancent qu'après validation du gate par
le relecteur.

**Contrat :** l'addendum **§A6–§A11** de `PREREGISTRATION.md` (dépôt pocCascade2phys, commit
`598fabd`) fait foi verbatim, en plus des §A0–§A5 inchangés (commit `a9cdb98`). Un seuil
manqué est un résultat. Aucun seuil n'est ajusté après première mesure. Toute itération de
conception = physiquement motivée, nommée, pré-annoncée, remontée.

**Dépôts :** code + tests + `outputs/arcA/` dans **pocPhysicator**, branche
`arc-a-etat-complet` (continuer). Contrat + journal dans pocCascade2phys. Chaque entrée de
journal cite ses commits. **AUCUN artefact load-bearing hors git** (leçon reboot 2026-07-03) :
toute archive `.npz` nécessaire à une relecture est commitée ou régénérable par script
commité + seed.

**Tech stack :** Python 3.12 (`.venv` existant : `.venv/bin/python`, `.venv/bin/pytest`),
numpy/scipy/matplotlib uniquement. CPU, minutes par run. Aucune dépendance nouvelle sans
demander.

## Global Constraints

- **Task 0 avant tout code** : addendum §A6–§A11 apposé verbatim (FAIT, commit `598fabd`
  côté contrat) + smoke de reproductibilité v2 (hash du champ `s` final consigné).
- **NE JAMAIS ÉDITER** : `src/solver_wetdry.py`, `src/terrains.py`, `src/render.py`
  (solveur figé, validé Thacker/Ritter/Stoker). Le mode périodique se fait par WRAPPER
  (padding périodique autour de l'appel), pas par modification du solveur. Si le wrapper
  est matériellement impossible sans toucher au solveur : STOP, remonter.
- **Jamais L2 comme critère.** corr(s_A, s_B) et resfrac sont des DIAGNOSTICS (rapportés,
  jamais bloquants). Le critère porteur vit au readout (§A9-a').
- **Sensibilité systématique** : tout chiffre porteur est rapporté sur JND ∈ [2, 5] % et
  S_HALF ∈ [0.005, 0.20]·relief ; verdict retenu seulement s'il est stable sur les plages.
- **Assert conservation bloquant** (§A10) actif dans TOUS les runs, y compris calibration :
  masse(s) + masse(c) − (érodé − déposé) à chaque pas, tolérance 1e-10 relatif par épisode.
  Un assert qui tire = bug, STOP.
- **Déterminisme par seed** : un run = f(seed) reproductible bit-à-bit ; testé.
- **Résultats append-only** : entrée datée au journal (pocCascade2phys), données brutes en
  `outputs/arcA/*.json`, figures nommées. Verdicts appliqués MÉCANIQUEMENT depuis §A8–§A10.
- **v2 = bras nul (§A11)** : `KD_CALIBRE` et `KE_CALIBRE_V2` dans `src/sediment.py` restent
  intacts et vivants ; le code suspension est ADDITIF (nouveau module ou nouveaux
  paramètres), il ne casse aucun des 105 tests existants.
- **Code, commentaires, docstrings, messages : en français.**
- Points d'arrêt obligatoires : après M-0 si lecture 2 (§A8) ; après le balayage w_s si
  BLOCKED ; après le gate (Task 4) DANS TOUS LES CAS — remonter le résultat, ne pas
  enchaîner sur les mesures k\*(L).

## Paramètres physiques gelés (hérités, inchangés sauf mention §A10)

- Grille 64×64, indexation array[y, x] ; terrain `default_terrain()` (plan incliné drop 0.5
  + 3 bosses gaussiennes, relief = 0.6044).
- Épisode : pulse d'eau gaussien seedé → relaxation 600 pas → assèchement (s persiste,
  et désormais : `c` intégralement déposé à l'assèchement, jamais détruit).
- Érosion/dépôt : θ = u²+v², θ_c = 0.5 (a priori ; re-calibrable UNE fois avec k_d, §A10) ;
  `b_eff = b0 + s` gelé pendant l'épisode, mis à jour entre épisodes.
- Readout albedo : A = 1 − exp(−s/S_HALF), point d'op S_HALF = 0.05·relief.
- Histoires : L₀ = 10 pulses, même multiset d'amplitudes, ordre direct vs inversé
  (protocole de `scripts/run_arcA_revalidate.py`, réutiliser la génération d'histoires).
- **NOUVEAU (§A10)** : BC périodiques (les deux axes) ; champ suspendu `c(x)` ; knob `w_s`.

## Interfaces existantes (pour les implémenteurs)

- `src/sediment.py` : `default_terrain()`, `SedimentParams` (frozen dataclass),
  `run_episode(s, b0, center_frac, params)`, `run_history(seed, ...)`, `resfrac_history`,
  constantes gelées `KD_CALIBRE = 0.0019109529749704406`,
  `KE_CALIBRE_V2 = 0.005232991146814947`. NE PAS MODIFIER les chemins de code v2.
- `src/albedo.py` : `albedo(s, s_half)`, `relief_shaded(b)`, `f_fraction(A1, A2, jnd)`,
  `chi_bands`, `delta_chi`.
- `src/solver_wetdry.py` : `simulate_wetdry_o2(...)` (FIGÉ). Reconstruction hydrostatique
  d'Audusse, HLL, murs réflexifs par padding — le wrapper périodique doit neutraliser ces
  murs par padding périodique de (h, hu, hv, b) et recadrage.

## Tasks

- [x] **Task 0 — Contrat.** Addendum §A6–§A11 apposé verbatim, commit dédié `598fabd`
  (pocCascade2phys). Plan gravé (ce fichier). Smoke v2 : un run `run_history` aux constantes
  gelées, hash SHA-256 du champ `s` final consigné au ledger — le bras nul doit rester vivant.

- [ ] **Task 1 — M-0 (§A8), AVANT tout build.** Sur v2 gelé : deux histoires même-multiset
  d'ordre inversé (protocole du gate v2), `f_ordre(v2)` sur les plages JND ∈ [2, 5] % et
  S_HALF ∈ [0.005, 0.20]·relief. Livrables : `scripts/run_arcA_m0.py`,
  `outputs/arcA/m0_v2_readout.json`, rapport avec lecture §A8 appliquée mécaniquement.
  **Si lecture 2 (f_ordre > 1 % du domaine à JND = 5 %, S_HALF op) : STOP, remonter.**

- [ ] **Task 2 — Substrat suspension (§A10).** Nouveau module `src/suspension.py` (additif) :
  champ `c(x)` : érosion → suspension, advection (schéma upwind sur le courant du solveur
  figé + diffusion faible), dépôt `w_s·c` sur cellules θ < θ_c, reprise depuis `s`,
  `b_eff = b0 + s`. **BC périodiques** par wrapper (padding périodique ; solveur non
  modifié). Assèchement inter-épisodes : `c` intégralement déposé, jamais détruit.
  Assert conservation (tol 1e-10 relatif/épisode) actif partout. Tests :
  conservation (assert + test dédié), déterminisme par seed, périodicité effective
  (translater terrain + centres de pulses d'un offset → champs translatés à l'identique),
  advection sanity (paquet de `c` dans un courant uniforme → déplacement du bon nombre de
  cellules). Les 105 tests existants restent verts.

- [ ] **Task 3 — Calibration unique + balayage Goldilocks w_s (§A10).**
  (i) Calibration pré-annoncée de k_d/θ_c vers s_max ≈ 20 % relief à L₀ = 10, à w_s médian ;
  constantes gelées, consignées. (ii) Balayage w_s : ≥ 5 valeurs log-espacées, ≥ 3 seeds par
  valeur ; par w_s : `f_ordre` (plages JND/S_HALF) + diagnostics (corr, resfrac, fraction
  redéposée à distance > 1 cellule). Livrables : `scripts/run_arcA_goldilocks.py`,
  `outputs/arcA/goldilocks_ws.json`, `outputs/arcA/arcA_f_ordre_vs_ws.png` (courbe + bandes
  de sensibilité). Sélection du point d'op sur PLATEAU selon §A10, consignée avec
  justification lisible sur la courbe. **Aucun w_s ne tient (a') sur la plage → BLOCKED,
  remonter sans forcer.**

- [ ] **Task 4 — Gate de conception (§A9), au point d'op gelé.** (a') `f_ordre ≥ 5 %` robuste
  sur JND ∈ [2, 5] % × S_HALF ∈ [0.005, 0.20]·relief ; (b') `f_ordre(albedo) ≥ 3 ×
  f_ordre(relief)` sur toute la plage S_HALF ; (c') `f(k)` moyenné sur ≥ 3 seeds, monotone
  décroissante (tol +0.002) aux deux géométries (overlap y∈[0.4,0.6] / séparée).
  Diagnostics rapportés : corr, resfrac, turbidité (intégrale de colonne de `c`).
  Livrables : `scripts/run_arcA_gate_conception.py`, `outputs/arcA/gate_conception.json`,
  verdict mécanique. **POINT D'ARRÊT OBLIGATOIRE : remonter le résultat — PASS, FAIL ou
  BLOCKED — avant toute mesure k\*(L).** Rappel §A9 : itération possible seulement nommée +
  pré-annoncée + mêmes seuils, après arbitrage.

- [ ] **Task 5 — (GATÉE derrière validation du gate par le relecteur.)** Reprise manche 1
  (compresseur/régénérateur/M-A1-A2-A3/k\*(L)) selon le plan gravé
  (`docs/superpowers/plans/arc-a-manche1.md`, Tasks 3–6), avec DEUX ajouts : le pipeline
  tourne en parallèle sur le **bras nul v2** (§A11, attendu : saturation triviale ;
  indiscernabilité re-fondé/v2 = instrument menteur → STOP) ; et la décision
  turbidité-dans-M-A (§A10) aura été gravée au point d'arrêt.
