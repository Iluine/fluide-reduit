# Plan d'exécution post-manche 2 — 2026-07-18 (pour Claude Code)

Contexte : §A13-résultat GRAVÉ (pocCascade2phys 5633986) — VERDICT **REGISTRE_FERME**,
contrôles 0 violation. Conséquence §A13-0 active : architecture ledger VIABLE. Arc V :
backup abandonné (décision Romain 2026-07-18) → fork temporel (a/a′/a″/d) DÛ.

## Règles d'exécution (héritées, MESURÉES — ne pas re-découvrir)

- Toute mesure verdict-grade ou sonde : machine iluin-tworings3, **terminal natif**
  (`.venv/bin/python`, Python 3.12.3 + numpy 2.4.6). La VM Cowork locale est bit-identique
  au gel (fait mesuré 2026-07-15, `test_replay_prefixe_bit_identique` PASS 30.6 s) mais son
  infra tue tout processus à 45 s → VM = dev, lecture, post-hoc léger UNIQUEMENT.
- Pré-enregistrement AVANT mesure ; revue adversariale AVANT implémentation ; choix
  d'implémentation nommés dans les docstrings et ENDOSSÉS par Romain avant run ; point
  d'arrêt à chaque lecture de sonde et chaque verdict ; ledger SDD append-only.
- Cœur Task 0 manche 2 (`run_arcA_m2_registre.py`, revu Approved) : INTOUCHÉ. Toute
  extension passe par de nouveaux scripts qui le consomment.

## Tasks S0+S1+S2 — FAITS (2026-07-18) : sonde exécutée, §A14-lecture GRAVÉE

S1 build revu (conforme §A14, 13 tests PASS vérifiés indépendamment en VM bit-identique) ;
choix (v)-(viii) ENDOSSÉS ; S2 run natif — lecture AUTRE, bornée loin sous le pin
(max 0.046 = 63 % du pin), escalade deux-étages NON déclenchée, trois faits gravés
(§A14-lecture, pocCascade2phys) : budget aire-proportionnel préserve la tenue ;
contamination dehors→dedans sous plancher ; ressaut-relaxation vu 2×.
PROCHAIN PAS : spec fovéa-z PAPER GRADE (gate ouvert), sections obligatoires (budget VRAM
épinglé, GPU-déterminisme), cette lecture comme premier appui du mariage registre×fovéa.

## Task S0 (détail archivé) — §A14 GRAVÉ (pocCascade2phys f08c340)

Décisions Romain consignées : fenêtre fixe 32×32 alignée quadtree ; BRAS NU UNIQUE (un
seul étage) ; deux-étages rétrogradé en escalade PRÉ-ÉCRITE (k_out=64, figé). S1 débloqué :
implémenter EXACTEMENT §A14 — la vérification d'instrument (bandes max_carrier définies
sur 32×32) est DUE avant tout run, remonter si muette.

## Task S0 (archive) — Pré-enregistrement sonde fenêtrage-fovéa [PAPER GRADE, avec Romain]

Objet : falsificateur le moins cher du mariage registre×fovéa (le cœur de la future spec
fovéa-z ; exclusion §A13-5 « fenêtrage spatial » attaquée de front, AVANT que la spec fige).

- Rédiger l'annexe de pré-enregistrement au journal pocCascade2phys (style §A13-3 sonde) :
  claim de la sonde, protocole (base cellule sonde manche 2 : 1 seed, Δt=4, k=256 ;
  émissions/commits FENÊTRÉS — politique de fenêtre à FIGER : taille, fixe vs mobile,
  ancrage), lecture Δχ dedans-fenêtre vs hors-fenêtre avec attendus ET portées gravés,
  critères mécaniques, AUCUN verdict (sonde = lecture remontée, jamais de branchement auto).
- Questions à trancher par Romain au moment de la rédaction (pas avant, pas après) :
  politique de fenêtre, référence de lecture hors-fenêtre (le hors-fenêtre est NON observé
  par construction — que promet-on ? rien, coarse-LOD, ou borne laxiste ?).
- GATE : endossement Romain du texte avant toute ligne de code.

## Task S1 — Build sonde [Claude Code]

- Nouveau script `scripts/run_arcA_sonde_fenetre.py` consommant `chaine_registre` /
  `rederiver_emissions` (signatures strictes anti-fuite reconduites) + tests dédiés
  (mécanique de fenêtre, gardes fail-loud, déterminisme). Ruff clean. Revue avant merge.

## Task S2 — Run natif + lecture [Romain exécute, lecture mécanique remontée]

- Run court (ordre de grandeur sonde manche 2 : minutes). Lecture pure mécanique,
  remontée à Romain. POINT D'ARRÊT.

## Front temporel — TRANCHÉ : (d), gravé au journal (2026-07-18)

- Décision (d) GRAVÉE (pocCascade2phys, section « Décision fork temporel ») : W1
  mort-par-défaut ; σ_ω = dette nommée avec condition de réveil écrite ; **gate fovéa-z
  OUVERT**. La spec commence PAPER GRADE (structure, invariants éphémère/persistant — le
  registre §A13 est la colonne persistante —, pins requis dont r_fovea, incertitude σ_ω
  portée explicitement). Aucun code de spec avant le paper grade.
- **Sections OBLIGATOIRES de la spec (ajout 2026-07-18)** : (1) budget mémoire — épingler
  (c, b, N_niv, n_fov^d) contre l'enveloppe de `claude/note-budget-vram-2026-07-18.md`
  (Harten×fovéa : monde en log, budget dominé par n_fov^d ; 3D/4 Go ⇒ fine-fovéa ~64³ aux
  hypothèses par défaut) ; (2) GPU-déterminisme — la bit-identité est non-défaut sur GPU ;
  la frontière éphémère/persistant doit dire qui a droit au non-déterminisme et où vivent
  les commits.

## Piste nommée — TRANCHE-MOTEUR (à chiffrer après S0, pas avant)

- Falsificateur le moins cher du mot « moteur » : F sur grille vivante + fenêtre fovéa
  mobile, sur la 3050 Ti, mesure frame-time vs L_eff dans l'enveloppe VRAM ci-dessus.
- Discipline : critères de mort PRÉ-ÉCRITS avant tout run (ex. « < N fps à L_eff minimal
  viable ⇒ la fenêtre §C3 est morte en pratique ») ; c'est une MESURE, pas un prototype ;
  chiffrage du build remonté à Romain avant achat.

## Interdits explicites

- Aucun enchaînement automatique après S2 ni après toute lecture.
- τ_relax post-hoc (registres m2_parts) : NON armé — attendre que la spec le demande.
- Aucune modification du cœur manche 2, aucune re-mesure de grille.

## En attente (hors Claude Code)

- Push par Romain : `pocPhysicator` (arc-a-manche2-registre) et `pocCascade2phys`
  (t1-spring-config) — commits locaux en avance.
