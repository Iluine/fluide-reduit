# Brief de reprise — projet Cascade (2026-07-18, après la session « manche 2 → spec fovéa-z »)

## Rôle et règles d'engagement

Critique senior adversarial, en français. Romain décide seul (il lit le code, n'écrit pas
le Python) ; Claude Code implémente ; la session lit, challenge, pré-enregistre, revoit.
JAMAIS d'enchaînement automatique après un verdict ou une lecture de sonde — point
d'arrêt, remonter, décision de Romain.

## Sources de vérité (dans cet ordre — elles portent TOUT l'état)

1. **poc-cascade-1** (GitHub) = `pocCascade2phys` (local), branche `t1-spring-config` :
   - `PREREGISTRATION.md` — le contrat + journal append-only. FIN DE FICHIER : §A13-résultat
     (VERDICT REGISTRE_FERME 2026-07-18), décision fork temporel (d) + dette σ_ω, §A14 +
     §A14-lecture (sonde fenêtrage), F0′-lecture (divergence inter-machines).
   - `SPEC-FOVEA-Z.md` — la spec fovéa-z, **8 sections TOUTES ENDOSSÉES** (décisions
     D1-D13 consignées in-situ). Elle FERA FOI après deux mesures : F0-cloud et F1
     (tranche-moteur). Lire « STATUT DU DOCUMENT » en fin de fichier.
2. **fluide-reduit** (GitHub) = `pocPhysicator` (local), branche `arc-a-manche2-registre` :
   - `claude/plan-apres-manche2-2026-07-18.md` — le plan d'exécution Claude Code.
   - `claude/note-budget-vram-2026-07-18.md` — l'enveloppe mémoire (Harten×fovéa).
   - `.superpowers/sdd/progress.md` — le ledger SDD append-only.

## État en une ligne

Manche 2 close (REGISTRE_FERME, contrôles 0 violation) ; gate fovéa-z OUVERT (fork (d)) ;
sonde fenêtrage lue (mariage registre×fovéa : premier appui) ; spec v0 complète et
endossée — il lui manque F0-cloud et F1 pour faire foi.

## Prochain pas exact, selon le type de session

- **Session CLOUD** : exécuter **F0-cloud** (~2 min) — `run_history(101, 10, B0, PARAMS,
  checkpoints={10})` vs `s_L10` de `outputs/arcA/histories/h_s101.npz` ;
  **PERSISTER l'état divergent** (leçon gravée : le FAIL originel ne l'avait pas été) ;
  calculer Δχ readout (albedo + delta_chi, max_carrier) vs jnd_sev 0.0733 ; remonter la
  lecture. Attendu : divergence bit (le cloud n'est pas l'instrument) — la question est
  SOUS-JND OU PAS.
- **Session LOCALE** (machine Romain) : **pré-enregistrement de la TRANCHE-MOTEUR**
  (spec §8-F1) — les critères de mort CHIFFRÉS avant toute ligne de code GPU (frame-time
  vs L_eff sur 3050 Ti, écart live↔rederive sous f32, débit PCIe du schéma diff, coût de
  load/replay), rédigé AVEC Romain, gravé au journal. C'est la prochaine vraie séance.

## Faits d'instrument (mesurés — ne pas re-découvrir)

- Mesures verdict-grade : machine iluin-tworings3 UNIQUEMENT, terminal natif
  (`.venv/bin/python`, Python 3.12.3 + numpy 2.4.6).
- La VM Cowork locale est bit-identique au gel (mesuré) MAIS son infra tue tout processus
  à 45 s → dev, lecture, post-hoc léger seulement. (Python 3.12.3 s'y reconstruit via
  debs noble + loader glibc, cf. sessions précédentes.)
- Le cloud N'EST PAS l'instrument (divergence probable par chemins SIMD AVX512 au-dessus
  du plafond machine, cf. F0′-lecture) — mais c'est le bon endroit pour F0-cloud.
- Push GitHub : Romain uniquement (clés hors VM). Commits :
  `git -c user.name="Claude (Cascade)" -c user.email="noreply@anthropic.com" commit`.

## Discipline (le projet vit d'elle — la mémoire du projet porte le reste)

Pré-enregistrer AVANT de mesurer ; le moins cher qui peut échouer, d'abord ; chiffres
inconfortables reportés en évidence ; portées nommées, jamais surclamées ; INDÉTERMINÉ
est un résultat ; étiquetage spec MESURÉ / TRANSPOSITION-HYPOTHÈSE / NON-ANCRÉ ; les
états divergents se PERSISTENT toujours ; aucun seuil ne se déplace après coup.
