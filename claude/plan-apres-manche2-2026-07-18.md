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

## Task S0 — Pré-enregistrement sonde fenêtrage-fovéa [PAPER GRADE, avec Romain]

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

## Front temporel — SUSPENDU à la décision du fork (a/a′/a″/d) [Romain]

- **Si (d)** : graver au journal (décision + W1 mort-par-défaut + σ_ω = dette nommée
  réveillable) → gate fovéa-z OUVERT → la spec commence PAPER GRADE (structure, invariants
  éphémère/persistant — le registre §A13 est la colonne persistante —, pins requis dont
  r_fovea, incertitude σ_ω portée explicitement). Aucun code de spec avant le paper grade.
- **Si (a″) ou (a′)** : Task T0 = passe de vérification in-repo (ce qui existe du solveur
  W / rendu luminance / harnais ABX Arc C réutilisable) → chiffrage honnête remonté AVANT
  achat ; puis T1 build harnais temporel, sessions Romain, pin σ_ω, décision W1.
- **Si (a)** : idem T0 d'abord — le coût de (a) a changé de catégorie depuis la perte
  d'Arc V ; aucun engagement sans re-chiffrage.

## Interdits explicites

- Aucun enchaînement automatique après S2 ni après toute lecture.
- τ_relax post-hoc (registres m2_parts) : NON armé — attendre que la spec le demande.
- Aucune modification du cœur manche 2, aucune re-mesure de grille.

## En attente (hors Claude Code)

- Push par Romain : `pocPhysicator` (arc-a-manche2-registre) et `pocCascade2phys`
  (t1-spring-config) — commits locaux en avance.
