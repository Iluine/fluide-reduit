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
Itération v2 (co-calibration, commits 486d637+ebdf32c) : signatures tenues (taux 0.18876, resfrac 0.64254) ; GATE v2 FAIL 2/3 — (a) corr=0.9446 (sens prédit confirmé, amplitude non), (c) séparée rebond f(7). STOP DÉFINITIF voie reconstruction (itération consommée). Arc A manche 1 suspendue, fork chez Romain.

# Arc A re-fondation (§A6–§A11) — ledger SDD
Plan : docs/superpowers/plans/arc-a-refondation.md (branche arc-a-etat-complet, commit b778924)
Contrat : addendum §A6–§A11 gravé dans pocCascade2phys (commit 598fabd, dédié, avant tout code).
Task 0: complete — addendum + plan gravés ; smoke v2 (bras nul §A11 vivant) : run_history(12345, 10)
  aux constantes gelées → max(s)/relief = 0.18875904854649675 (= gel 0.18876),
  sha256(s) = ab77208fddd1b86ea2ce0068fa2a704978e8687c18e8d9cbff97c3a83619f4c7.
Task 1 (M-0): complete (commit 09fca34, revue clean 1er passage) — VERDICT lecture_2 (§A8) :
  f_ordre(v2) = 0.7595 à la cellule porteuse (JND=5 %, S_HALF=0.05·relief), seuil ≤ 0.01.
  corr = 0.94455 (= gate v2). Plancher relief : f_ordre ≤ 0.0015. STOP pré-enregistré :
  Tasks 2–4 NON lancées, gate §A9 à repenser avant tout build. Remonté à Romain.
  Minors reportés à la revue finale : littéraux N_EPISODES/SEED/BOX dupliqués (phase_a ne les
  expose pas) ; pas de micro-test _f_ordre ; print dict brut du plancher relief.
M-0bis (sonde de testabilité): complete (commits 32e9004..cc61056, revue clean 1er passage) —
  VERDICT T_testable : max(M-A1, M-A2-mini) sous cap ∈ [0.13, 0.63] vs seuil 0.02, les 4 cellules
  échouent le test de trivialité. src/summary.py (spec gravée Task 4 manche 1) construit en TDD,
  21 tests ; run_episode_trajectoire additive (+4 tests) ; 130 verts. Remonté à Romain.
  Minors reportés à la revue finale : Summary.__eq__/__hash__ latent (suggérer eq=False) ;
  duplication boucle de contrôle _integrate_exner_trajectoire ; assert nu 11ᵉ centre (python -O) ;
  round-trip testé sur constante dyadique 4.0 ; pas de test level invalide.
Manche 1 relancée SUR v2 (arbitrage Romain, amendements gravés 8e02dd6 ; plan amendé 069b70e).
Task 3 (histoires): complete (commit 04ac96f, revue clean 1er passage) — 5 npz (seeds 101-105,
  L∈{10,20,40,80}), sanity PASS ×2 (f_ordre(101,102)=0.7136 ; replay bit-à-bit). 130 tests verts.
  Minors : constantes reconstruites localement vs importées ; _load_history_npz charge tout.
  Note pour Task 5/6 : max(s)/relief dérive au-delà de la fenêtre de calibration à L=80 (attendu,
  la calibration visait L₀=10).
Task 5 (mesures): complete (commits a64185b+4a572c5, revue clean 1er passage) — measures.npz
  3 bras. v2 : rien ne ferme sous cap (ℓ=2/3 ∈ [0.10, 0.85]) ; ℓ=1 (hors cap) ∈ [0.022, 0.117].
  Contrôle shuf CONFORME (1.36-2.28, jamais fermé). Contrôle ferm EN VIOLATION de l'attendu
  k*=81 : plancher de round-trip du régénérateur bilinéaire à ℓ=3 = 0.09-0.19 > JND max —
  observation brute, lecture formelle en Task 6 (gate d'instrument attendu : STOP).
  Minors : _versions() recalculé à l'assemble ; errstate redondant ; sanity préfixe re-exécutée.
Task 6 (verdict): complete (commits 96eaf70+cf6c131, correctif 12f0446, revue + re-revue Approved) —
  GATE DES CONTRÔLES = VIOLATION (ferm 40/40 : jamais k*=81 ; shuf 0/40 conforme).
  Verdict §A3 v2 SCELLÉ « NON LISIBLE » (audit : INDÉTERMINÉ ×4 JND, k*(L)=∞ partout).
  139 tests verts. Manche 1 au point d'arrêt obligatoire — remonté à Romain.
  Minors reportés à la revue finale : duplication axes catégoriels entre figures ; clé JSON
  "global_" ; main() ~150 lignes ; invariant sous_jnd violé sciemment dans le test d'escalade
  (documenté).
Itération ordre 0 : BLOCKED intermédiaire correct de l'implémenteur — S∘R bit-à-bit échouait
  d'1 ULP à ℓ=3 (numpy .mean non-binaire sur 64 élts/bloc). Décision contrôleur : summarize
  ré-implémentée en cascade 2×2 itérée (définition récursive de Harten, sémantique inchangée,
  exacte par construction sur blocs constants) — promesse gravée rendue vraie en flottant.
  Reproduction bit-exacte de M-0bis/run v1 = par checkout historique (consigné).
Itération ordre 0 : complete (commits c35e4dd/ab7de1e/04b8163/c518b11, revue Approved, vérif
  indépendante sur données) — 164 tests. Gate CONFORME (ferm 0/40 exact 0.0 ; shuf 0/40).
  PREMIER VERDICT LISIBLE : JND 2 % = INDETERMINE_CAPACITE ; 3/4/5 % = INDÉTERMINÉ ;
  global INDÉTERMINÉ. k* finis pour la première fois : 1041 aux petits L (hors cap),
  ∞ à L=80 (JND 4-5 %). Prédiction directionnelle FALSIFIÉE (Δχ baissent 120/120).
  Diag escalier : Δχ(0)=0.3265, max=0.3875, rapport 1.187. Remonté à Romain (fork clause 1).
  Minors : docstrings longs ; tolérance absolue 1e-12 non nommée dans _verifier_masse_restituee.
