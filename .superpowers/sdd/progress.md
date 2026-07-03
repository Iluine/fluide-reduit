# Arc A manche 1 — ledger SDD
Plan : docs/superpowers/plans/arc-a-manche1.md (branche arc-a-etat-complet)
Task 1: complete (commits e36dd23..be88024, revue + correctif + re-revue 4/4 clean)
  Minors reportés à la revue finale : docstring test seuil-wet surclame (érosion no-op structurel) ; calibrate_kd(max_iters=0) → NameError théorique.
  Note instrument (pour journal Task 2) : dérive de masse d'eau 3–13 %/épisode au mur wet/dry (propriété du solveur figé, mécanisme diagnostiqué : padding antisymétrique du moment normal → pente de vitesse asymétrique → flux HLL non nul au mur). KD_CALIBRE=0.0019109529749704406 gelée, max(s)/relief=0.2056.
Task 2 : GATE de re-validation exécuté (n=10 épisodes/histoire) → **VERDICT GLOBAL FAIL**.
  (a) path-dependence FAIL : corr(s_A,s_B)=0.9929 (seuil ≤0.7, référence origine 0.39) — le
  substrat reconstruit est quasi insensible à l'ordre des pulses (mémoïse-partout).
  (b) survie albedo PASS : f_albedo_op=0.8308 (seuil ≥0.02) ≫ f_relief=0.00195, dominance
  albedo>relief sur tout le balayage S_HALF.
  (c) closure f(k) : overlap PASS (f décroissante 0.338→0.214→0.187→0.000), séparée FAIL
  (rebond f(4)=0.363 > f(1)=0.327+tol, non-monotone). Aucun seuil ni paramètre retouché.
  Rapport : task-2-report.md. Prochaine tâche du plan Arc A NON lancée (gate bloquant).
Task 2: complete (commit 554c2a9) — GATE FAIL 2/3 : (a) corr=0.9929 FAIL, (b) PASS (mais 83 % vs réf 3.5–10 %), (c) FAIL géom. séparée (rebond k=4). Pipeline STOPPÉ : Tasks 3–6 NON lancées, aucun paramètre retouché. Remonté à Romain.
