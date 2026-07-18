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
Famille 2 quadtree (module) : commits 18e52b9+e0eae53, 286 tests (122 nouveaux), S∘R strict
  arbre inclus aux 7 budgets × 3 champs. ARBITRAGE IMPLÉMENTEUR VALIDÉ PAR LE CONTRÔLEUR :
  la clause brief « masses 16×16 vs original 1e-12 » était insatisfiable sous troncature
  (feuille englobante = redistribution uniforme, contre-exemple en test exécutable) ; lecture
  actée = conservation TOTALE 1e-12 + par-feuille 1e-12 + 16×16 exactes vs RÉGÉNÉRÉ +
  16×16 vs original là où dérivables (anti-vacuité). Fidélité spatiale jugée par le readout
  (design). Interruption limite API pendant le GREEN, reprise propre. Revue en cours.
Famille 2 (module) : complete (revue Approved, vérifications par exécution indépendante).
  Minors : boundary 400→398 non pinné en test ; formule taille dupliquée ; init _pyramide indirect.
Famille 2 (harnais + RUN) : complete (commits 2ee1120/0a81832/a86b929, revue Approved, violation
  vérifiée dans les données brutes) — 295 tests. GATE = VIOLATION : shuf 7/40, UNIQUEMENT au
  budget plafond 2048 (JND 3-5 %) ; sous cap : 0 violation (400 → 0.52-0.91, jamais fermé) ;
  ferm PARFAIT 0/40 (Δχ=0.0 exact, 7 budgets). Verdict scellé NON LISIBLE. Surface v2 (audit,
  sous scellé) : premières fermetures SOUS CAP de la manche (400 et 256 à JND 4-5 %, tous L).
  Remonté à Romain (portée de l'attendu shuf). Minors : duplication verdict_qt justifiée ;
  titre figure ; npz témoin gitignoré.
Re-lecture gate (amendement fa54d20) : commits 49189d1+11091a0, re-revue Approved — gate CONFORME
  0/80 ; k*/pentes bit-identiques à l'audit (déverrouillage = règle de lecture seule) ; 301 tests.
  VERDICT LISIBLE : 2%=INDÉT, 3%=INDÉT, 4%=PASS, 5%=PASS ; global INDÉTERMINÉ → surface → Arc C.
  6/16 cellules étoilées non-discriminantes. Minor : rupture de schéma surface_kstar_v2 (documentée).

## Arc C — Pin JND (branche arc-c-pin-jnd, plan docs/superpowers/plans/arc-c-pin-jnd.md)
Task 0 (partiel) : addendum §C0-§C5 gravé (pocCascade2phys c494d50, commit dédié) ; branche +
  plan commités ; inventaire : stimuli spatiaux OK (h_s101..105.npz : s_L{10,20,40,80} 64×64),
  rendu réel OK (render.py/io_utils : viridis, vmin=0, vmax=1, origin lower) ;
  BOUSSINESQ Ra=1e7 ABSENT de la machine (grep src/scripts/docs vide — code jamais porté ici,
  journal seul ; cf. mémoire substrat-sediment-introuvable). POINT D'ARRÊT remonté à Romain :
  validation C-0 + conditions d'affichage + arbitrage axe temporel sans générateur.

Task 0 (LEVÉ) : §C7 gravé (pocCascade2phys f39b402) — refonte géométrie cible-jeu VALIDÉE Romain :
  pin ancré c/deg + harnais auto-calibré (borne inter-écrans a fortiori) ; deux régimes bornés,
  plancher pixel-peep MESURÉ ; plafond texture calculé (colmatage monotonie) ; fork zoom ouvert
  avec prix (zoom pixel ⇒ plancher ⇒ manche 1 NON-DÉMONTRÉ à ce JND). C-0 validé, temporel (b)+(c)
  confirmé (spatial d'abord, backup Arc V cherché, fork a/a′/d dû avant Task 4 si (c) échoue).
Task 1 : DONE (commits 4f5c1c6, d98642f) — src/arcC_stimuli.py + scripts/build_arcC_stimuli.py +
  outputs/arcC/stimuli_catalogue.npz + tests. 327/327 verts. R1 recoupe ma1_v2 à l'EXACT 0.0 sur
  7 budgets. Catalogue bracket JND [~1-6%] : budget 256 (Δχ(1)∈[2.78,7.85]%), 400 (∈[1.77,4.18]%).
  140 courbes toutes monotones (observé, non imposé). EN REVUE (relecteur dispatché).
Task 1 : COMPLETE (commits 4f5c1c6..9bef82f, review clean). Re-revue sur pièces : 3 findings
  corrigés (assert tautologique→observation ; chargeur ré-importé avec garde-fou b0 ; test
  S_HALF_OP==0.05·RELIEF). Suite complète 327/327 verte, npz catalogue bit-identique.
Task 2 : dispatché (harnais ABX + staircase 2 régimes + auto-calibration §C7 ; sujet synthétique,
  aucun humain). Brief .superpowers/sdd/arcC-task2-abx-brief.md. BASE = 9bef82f.
Task 2 : IMPLÉMENTÉ (commits 1300399 calibration, 468d534 ABX). Implémenteur coupé par limite
  session API après rapport (263 l., complet) avant commit → contrôleur a committé + spot-vérifié +
  confirmé suite complète 382/382. Findings load-bearing : (1) §C7 pièce 3 cellule pic-CSF =
  1.73 arcmin (ratio 1.73 > acuité 1.0, quasi-invariant écran) — pin déjà 1.73× au-delà du plafond
  texture ; DÉCISION ROMAIN. (2) plafond Δχ par-source hétérogène (budget 256 → pire source 2.78%)
  → budget-par-régime à graver en Task 3. Caveats rapport : format ABX 3-stimuli à relire avant
  Task 3 ; n_essais_max=200 non testé à la limite. EN REVUE (relecteur dispatché a8e39fa).
Task 2 : COMPLETE (commits 1300399..468d534, review clean). Relecteur : Spec ✅ CONFORME +
  Qualité Approved ; vérif indépendante poussée (arithmétique escalier tracée main, §C7 + plafonds
  par-source recalculés, test adversarial règle inversée → tolérance 20% discriminante, zéro fuite
  ABX, latence_s seul champ hors replay bit-exact). 4 Minor (aucun Critical/Important, pas de fixeur) :
  (M1) clip dupliqué 2 lignes dans avance_escalier — cosmétique ; (M2) BUDGET_CATCH=32 fixe →
  CONTRAINTE TASK 3 : si un régime prend budget staircase=32, le catch perd sa marge de séparation
  → BUDGET_CATCH doit rester strictement plus bas/séparé du budget de staircase ; (M3) dispersion/
  statut/arret non appelés par la coquille = orchestration Task 3, pas un gap ; (M4) test report §C7
  faible par conception (§C7 interdit tout gate sur ce chiffre). Minors → triage revue finale de
  branche (différée : Task 3-4 ajoutent du code).
GATES TASK 3 (toutes = Romain) : (a) décision géométrie §C7 (reco contrôleur = (i) accepter la
  tension, ré-étiqueter « géométrie la plus sensible comme-prévu ») ; (b) 2 chiffres écran (longueur
  réf + distance mesurée, calibration par session) ; (c) budget-par-régime (marge sous plafond bas
  par-source) + contrainte M2 (BUDGET_CATCH séparé). Task 3 EXIGE Romain comme sujet → frontière SDD.
§C8 CLÔTURÉ (pocCascade2phys d7c95e3) : (β) ancre = quadtree budget 32 VALIDÉ Romain (81 floats
  = lapsus inter-famille ; décision 1 même-famille bat le chiffre) ; 3 conséquences opérationnelles
  → brief Task 3, ne rouvrent pas le contrat.
Task 3 (BUILD) : dispatché (a01d5d18). Runner orchestré (≥3 staircases × 2 régimes) + post-traitement
  → pins_spatial.json + IC + figures, câblant §C8 : ancre 32 (D-1), exclusion nommée par-source
  <0.16 ≥10=STOP (D-2), catch réserve forte t≈1 + validité sans recalibration + fallback OFF (D-3),
  mode géométrie-plafond même-ancre (D-4). HAUT_JND_PLAUSIBLE=0.08 → SEUIL_EXCLUSION=0.16 (~1 source).
  Validé SUJET SYNTHÉTIQUE (aucun humain). Brief arcC-task3-runner-brief.md. BASE = 2ce40a0.
  RESTE APRÈS CE BUILD : Romain sujet devant l'écran = seule étape que la machine ne peut pas faire.
Task 3 (BUILD) : DONE (commits 58acba3 abx+sources rétro-compat, 205e706 orchestration, f2b633a
  pins, fa2ae7c coquille). 408/408 verts. Catch collision résolue par argmax (preuve : catch ≥ 0.16
  car argmax ≥ Δχ(t=1)=ancre) ; IC = bornes min/max (bootstrap n=3 rejeté = fausse précision) ;
  gate e2e retrouve θ=0.10 à ~3% (2 régimes RÉSOLU). Concern : base_seed e2e re-cherché (pool 20→19
  change RNG). EN REVUE (relecteur a5128d07).
Task 3 (BUILD) : COMPLETE (commits 58acba3..fa2ae7c, review clean). Relecteur : Spec ✅ CONFORME +
  Qualité Approved. Vérif indépendante : ancre (103,10)=0.155 recalculée→exclue (19/20) ; preuve
  catch≥0.16 revérifiée math ; D-4 change SEULE la taille (73px vs 43px) ; aucun verdict manche ;
  ANTI-SEED-HACKING : balayage base_seed 0-29, ~7/30 dépassent tol 20% → discrimine, base_seed=4
  médian typique PAS outlier ; 408/408 relancés. 3 Minor (aucun Critical/Important, pas de fixeur) :
  (M1) test frontière dégénéré seuil=0.0 (frontière réelle vérifiée OK par relecteur, test trompeur) ;
  (M2) duplication ecrit/lit JSON manifeste vs pins ; (M3) derive_seeds collision si n-staircases≥1000
  (non atteignable). → triage revue finale de branche (Tasks 4-5 ajoutent du code).
FRONTIÈRE ATTEINTE : Tasks 0-3 (tout le buildable) closes, 408/408. Protocole de session remis à
  Romain. SEULE étape restante que la machine ne peut pas faire = Romain sujet devant l'écran.
§C9 gravé (pocCascade2phys 7110e31) : conditions de validité d'exécution Task 3 (Romain, avant
  données) — durées réalisées loggées par essai (validité laxiste + pré-vol B-bis) ; luminosité
  fixée+notée manifeste (choix C-1 n°3) ; repère distance + éclairage stable (anti-zoom §C7) ;
  amorçage θ=0.025 ; posture répondre-au-percept.
Task 3 CORRECTIF (avant session) : dispatché (a52ae008). (1) logging durées réalisées par essai
  (accumulateur pur testé + sidecar timing + résumé réalisé-vs-nominal + avertissement >25%),
  glue matplotlib partagée coquille/orchestrateur ; (2) --luminosite/--conditions REQUIS si humain
  → manifeste conditions_validite. Brief arcC-task3fix-timing-conditions-brief.md. BASE = 38a02e7.
Task 3 CORRECTIF : DONE (commits bcbdfd6 timing, c317f3b conditions). 432/432 verts (408+24 neufs :
  17 timing + 7 conditions). Nouveau src/arcC_timing.py (accumulateur pur) câblé dans le chemin
  humain partagé ; sidecar timing par staircase + résumé réalisé-vs-nominal + avert >25% (écart
  absolu). --luminosite/--conditions requis si humain → manifeste conditions_validite. EN REVUE
  (relecteur ac589bca) — dernier gate avant données irréversibles.
Task 3 CORRECTIF revue : Spec ✅ CONFORME + Qualité Needs-fixes. Relecteur a vérifié sur pièces :
  accumulateur RÉELLEMENT pur (0 matplotlib en sys.modules), seuil 25% recalculé (24→pas d'alerte,
  26→alerte), écart absolu capte trop-court ET trop-long (correct §C9), parser.error humain prouvé
  par test CLI réel, analogie coquille live⇒requiert cohérente. IMPORTANT : test manquant côté
  coquille (garde-fou conditions) esquivé pour raison INVALIDE (relecteur a prouvé matplotlib
  importable sans écran, parser.error avant figure) → fixeur dispatché (ab851b86) : ajoute le test
  CLI (monkeypatch argv + SystemExit) + dé-duplication ecrit_manifeste_json. 2 Minor structure
  laissés (recordés). BASE fix = 4a80731.
Task 3 : affichage LIVE durées par essai ajouté (commit ci-après) — formate_timing_essai (pur, 3
  tests) dans le chemin humain partagé → B-bis exécutable en quelques-essais+Ctrl-C (le sidecar
  batch n'existait qu'à complétion). 437/437. ARC C BUILD ENTIÈREMENT CLOS (Tasks 0-3 + correctifs
  timing/conditions + live-print, tout revu/vérifié). Prochaine action = Romain sujet (protocole
  A→B→B-bis→C→D remis). Puis lecture §C4 contrôleur sur pins_spatial.json (côté 3/4%, contingence).
§C10 gravé (pocCascade2phys 416a011) : contrat de sortie pins_spatial.json (JSON Schema draft-07,
  schemas/arcC-pins-spatial-v1.schema.json) AVANT données humaines. Nouveaux éléments : calibration
  embarquée + commit_harnais/date_session/sujet (provenance) ; timing_laxiste roll-up ; controle_
  directionnel JND_lax≥JND_sev (IC disjoints mauvais ordre → STOP) ; branche à-cheval ic_combine
  mean±2SEM ; statut RESOLU/INDETERMINE/INVALIDE + motif obligatoire ; seuils bruts par staircase.
  jsonschema 4.26.0 installé (pur Python, n'affecte pas JAX/warp).
Task CONFORMANCE : dispatché (ab78bad2). run_arcC_pins.py réécrit pour émettre + VALIDER contre le
  schéma (fail loud) ; orchestrateur stampe 3 champs provenance ; copie schéma pocPhysicator ; dep
  pyproject ; tests (validation schéma + statut 3-voies + directionnel + exclusions + à-cheval).
  Synthétique seul, jamais outputs/arcC/ commité. Brief arcC-task3conform-schema-brief.md. BASE=9adcfd7.
Task CONFORMANCE : DONE (commits c31729a schéma+dep, 7756882 provenance, 3f490f6 pins+validation,
  8d5f781 tests). 467/467 (437+30). jsonschema → requirements.txt (pas de pyproject ici). FRONTIÈRE
  CONTRAT signalée : campagne arrêtée (≥10 exclusions / régime à 2 staircases <minItems:3) → pins
  LÈVE (fail-loud), pas d'artefact malformé — le STOP vit au manifeste. À ratifier par Romain. EN
  REVUE (relecteur a6bda86b).
Task CONFORMANCE : COMPLETE (commits c31729a..8d5f781, review clean). Spec ✅ + Qualité Approved.
  Relecteur vérif indépendante : schémas identiques byte-pour-byte ; validation-avant-écriture testée ;
  statut 3-voies + directionnel + mean±2SEM + cellule_arcmin recalculés ; scénario 2-staircases
  reconstruit → fail-loud réel. 2 Minor (aucun Critical/Important) : (M1) pas de test e2e du chemin
  arrêt-2-invalides + message jsonschema cryptique ("is too short") ; (M2) pas de test explicite du
  statut INDETERMINE n<3-non-2-invalides. Les DEUX Minor sont EN AVAL du ⚠️ contrat (a vs b) → à
  résoudre avec le choix de Romain, pas avant.
⚠️ CONTRAT REMONTÉ À ROMAIN : le schéma (minItems:3) ne représente PAS un régime arrêté à 2
  staircases (STOP_2_INVALIDES §C5) → pins LÈVE, aucun artefact. Fork (a) garder fail-loud + message
  clair vs (b) assouplir le schéma pour un artefact minimal traçable. Reco contrôleur = (a)+message.
Invariant §C10 gravé (pocCascade2phys 448048f) : pins = campagne complétée par construction ;
  arrêt protocolaire = fait de manifeste jamais de référent ; fail-loud aiguillage (avant jsonschema,
  pas de complétion auto), 3 issues campagne testées e2e.
Task FIX fail-loud : dispatché (a9faff30). (1) détection STATUT_STOP_2_INVALIDES dans construit_pins
  AVANT jsonschema → RuntimeError clair (fait N staircases + régime + manifeste + conduite remonter,
  aucune voie de complétion) ; "is too short" cryptique rendu inatteignable. (2) message anti-3e-
  session. (3) deux tests e2e : arrêt-2-invalides → lève + aucun fichier ; INDETERMINE n≥3 disp>30%.
  Schéma gravé INTACT. BASE = 5cb2f5f.
Task FIX fail-loud : COMPLETE (commit e8f9460, vérifié contrôleur). 469/469. Message porte les 4
  éléments (fait+N, régime, manifeste, conduite remonter) + anti-3e-session ; tire AVANT jsonschema ;
  cryptique "is too short" INATTEIGNABLE (reproduit en indépendant : False) ; 2 tests e2e (arrêt-2
  → lève + not pins_path.exists() ; INDETERMINE n=3 disp>30% → statut/motif/jnd/ic + pins valide).
  Schéma gravé INTACT.

═══ ARC C — BUILD ENTIÈREMENT CLOS CÔTÉ MACHINE ═══
Contrat §C0-§C10 + schéma gravés. Pipeline complet, revu, testé : Task0 (harnais stimuli/R1) →
Task1 (générateur) → Task2 (ABX+staircase+calibration) → Task3 (orchestration+pins+timing+conformance
schéma+fail-loud). 469 tests. 3 issues de campagne fermées et testées (résolue/invalide/arrêtée).
Contrat de sortie validé-avant-écriture. SEULE CASE VIDE = Romain sujet (3 chiffres règle → A B Bbis C D).
Après ses logs : je lance run_arcC_pins.py (D) + lecture §C4 (côté 3/4%, contingence géométrie-plafond).
PRÉ-VOL B (Romain) a révélé 2 défauts de plomberie (mode d'échec prévu, mien à corriger) :
  (1) venv sans Qt → matplotlib Agg → pas de fenêtre, waitforbuttonpress boucle ; (2) chemin
  d'affichage ne montrait jamais la fenêtre (draw() sans ion/show). Correctif commit 21136f3 :
  src/arcC_backend.py (selectionne_backend_qt + assert_backend_interactif garde PUR 15 tests),
  QtAgg+garde+ion/show/flush dans les 2 chemins live, conftest force Agg (tests headless-safe),
  PySide6 dans requirements. jsonschema+PySide6 installés venv. Qt/xcb vérifié sur DISPLAY=:0.
  484/484. Affichage interactif (fenêtre visible + touches a/b) = re-run B de Romain (glue non
  testable headless). B prêt.
Re-run B (Romain) : fenêtre s'ouvre (backend OK) mais fermer ne coupe pas + rouvre blanche.
  2e défaut interactif corrigé (commit 52db4eb) : SessionInterrompue sur close_event/Échap,
  attrapée main()+campagne (arrêt propre, aucune donnée = cohérent §C10) ; boucle plt.pause(0.05)
  au lieu de waitforbuttonpress ; suptitle instructions. 484/484. Glue non testable headless →
  re-run B de Romain. Question ouverte : 1re fenêtre montrait-elle A/B/X, ou déjà blanche
  (défaut de rendu séparé) ?

═══ PIN SPATIAL MESURÉ (§C11, 1re donnée humaine) ═══
Campagne C complète (Romain n=1, base_seed 20260705, commit harnais 3d2a6ed). Catch 12/12=100%,
disp 0.18/0.196 ≤0.30, timing dérive 1.5%, 1/20 exclue (s103_L10). pins VALIDÉ §C10.
JND_sev=0.0733 [0.0603,0.0867] ; JND_lax=0.1154 [0.0937,0.1387] ; directionnel conforme.
LECTURE §C4 (mécanique) : consommateur 1 IC entier ≥4% (≥5%, PASS a fortiori) → CELLULE-1-CANDIDATE,
prononcé définitif gaté sur extension 16L₀ (L=160, NON lancée) ; contingence géométrie-plafond NE
s'arme PAS (IC≥4%) ; kx=1@1% moot (JND_sev≥2%) ; consommateurs 2&4 (W1, gate fovéa-z) en attente du
pin TEMPOREL non mesuré. Blemish : conditions=placeholder (mesure non invalidée). Contrat §C11=6c50345.
Forks Romain : 16L₀ ? axe temporel (a/a′/d) ? politique zoom §C7-4 ?

═══ C-1a — MICRO-MESURES POUR ENCADREMENT BIBLIO (Romain, plan C-1a) ═══
Base = commit 196fe75, branche arc-c-pin-jnd. Deux mesures d'analyse PURE (0 sim, 0 RNG, 0 donnée
humaine), entrées toutes vérifiées présentes (pré-vol falsifiabilité : histories/h_s*.npz + albedo.py
+ run_arcA_verdict_qt.py + measures_qt.npz + pins_spatial.json — RIEN de supposé). Point d'arrêt =
remonter les deux JSON (C-1 chat consomme le ratio ; arbitrage 16L₀ Romain consomme la surface).
Task 1: ratio chi_rms/Michelson bande 4-7 sur 19 sources incluses (vrai + régénéré-32) ; sanity sinus
  pur = 1/√2 ; garde d'honnêteté = dispersion IQR inter-sources. Sortie ratio_rms_michelson.json.
Task 2: re-seuillage k*(L) famille-2 quadtree aux 6 JND réels du pin (exacts pins_spatial.json ;
  RÉUTILISE k_star_groupe_qt) ; INTERDIT verdict/pente ; non-régression k*(80)@5%=400 + sous_jnd bit.
  Sortie surface_kstar_aux_pins.json + écart sévère↔laxiste en floats/L.
Task 1: complete (commits 19dfa4d..061885a, revue SPEC✅ + QUALITÉ approuvé, 9/9 + suite 493/493).
  Reviewer a vérifié indépendamment : déterminisme (MD5 JSON identique 2 runs), binning R1 réutilisé
  (pas de réimpl), aucun clip, 19 sources dynamiques (103,10 exclue), sanity 1/√2 <1%, Parseval <1e-9,
  structure JSON conforme. MESURE : ratio vrai médiane 0.4771 IQR 0.0931 [0.193,0.622] (dispersion x3
  réelle, héritée par C-1) ; régénéré-32 médiane 0.3856 IQR 0.0744. Débordements <0 signalés (non clippés).
  Minor pour revue finale : test_sources_incluses_triees faible (tri déjà interne) — sans impact mesure.
  ⚠️ CAVEAT MATÉRIEL POUR C-1 (à remonter à Romain au point d'arrêt) : c_michelson ∈ [1.90,4.82] méd 2.29
  pour 19/19 sources — Michelson >1 partout (hors régime valide [0,1]). Cause : mean_albedo ~0.04 (minuscule)
  + bande 4-7 y portant RMS comparable/sup (χ_rms>1 pour 14/19). La bande 4-7 EST porteuse (≥10% AC) 19/19.
  Le ratio natif mesuré (0.477) = ce que le plan a défini, MAIS convertir un seuil biblio petit-contraste à
  travers lui extrapole la saturation-pédestal (ratio natif r_nat = r0·(1+s/2mean), r0=2·RMS/pp le ratio
  scale-invariant petit-contraste). Mesure NON redéfinie (seuil pré-enregistré) ; décision C-1/Romain :
  garder natif, ou ajouter r0=2·RMS/(peak-to-peak) comme mesure compagne ? À nommer au point d'arrêt.
Task 2: complete (commits 98d30f0..27fdde7 impl + fix contrôleur, revue SPEC✅ + QUALITÉ approuvé, 6/6).
  Reviewer a vérifié indépendamment : 0 RNG, k* réutilisé (pas de réimpl), sous_jnd_a_jnd fidèle bit-à-bit,
  2 non-régressions OK (sous_jnd grille bit-exact + k*(80)@5%=400), JND lus runtime, monotonie sur grille∪pins,
  0 fuite verdict/pente sur 428 clés, double-run octet-identique. Fix contrôleur (renommage endossé par reviewer) :
  meta.verdict_reutilise → provenance_kstar (garde test STRICT sans exception, scan forme sérialisée) — incohérence
  de MON brief, pas de la pré-registration ; aucune valeur k* touchée. TABLE k* aux 6 JND réels (mécanique, AUCUN
  verdict lu — lecture = fork 16L₀ de Romain) : pin sévère 7.33% → k*=256 PLAT sur L∈{10,20,40,80} (< 400 du
  point-grille 5%, < cap 409.6) ; ic_bas sévère 6.03% → 256,256,400,400 ; pin laxiste 11.5% → 128,256,256,256 ;
  ic_haut laxiste 13.9% → 128 partout. Aucun k* ∞/>cap/non-discriminant sur ces 6 JND. Écart deux-étages (floats,
  au pin) : L=10 delta −128 (laxiste ferme à 128<256), L∈{20,40,80} delta 0. Sortie surface_kstar_aux_pins.json.
Task 3 (AJOUTÉE par décision Romain §C12) : ratio r0 = 2·RMS/(peak-to-peak) compagnon scale-invariant.
  complete (commits 6304426..5fb15d7, revue SPEC✅ + QUALITÉ approuvé, 9/9 + suite 508/508). Reviewer a
  vérifié : natif BYTE-IDENTIQUE (md5 8b541e47 avant/après, jamais ouvert en écriture), r0 sur _composante_bande
  RÉUTILISÉE (pas de reconstruction), sanity 1/√2, scale/pédestal-invariance 1e-9, cohérence-natif non-tautologique
  (FFT-Parseval vs std-spatial convergent), 19 sources, dispersion rapportée. MESURE : r0 vrai médiane 0.1663
  IQR 0.0145 [0.147,0.182] (6× plus resserré que natif — mean-indépendant) ; régénéré-32 médiane 0.1427.
  r0 ≪ 1/√2 = 0.707 ⟹ bande très spiky (crest factor ~6, cohérent « quasi-binaire »). Sortie ratio_r0_scale_invariant.json.
  Minors revue finale : ruff absent venv (lint non exécuté) — idem Task 1/2.
─── C-1a CLOS côté harnais : 3 mesures livrées+validées (ratio natif, k* re-seuillé, r0 compagnon). Suite 508/508.
  Revue whole-branch SDD non lancée : 3 mesures INDÉPENDANTES (sorties disjointes, r0 lit le natif en RO), chacune
  déjà revue spec+qualité, natif byte-identique vérifié, suite verte — aucune surface d'intégration à balayer.
  Point d'arrêt = 3 JSON remontés. Restent à Romain (non dus) : C-1 livraison (ordre de grandeur), 16L₀, axe temporel.

═══════════════════════════════════════════════════════════════════════════════
MISSION : Extension 16L₀ (L=160) — prononcé cellule 1 (§A12). Base = a5c2731 (arc-c-pin-jnd).
  Branche : arc-c-pin-jnd (descendante d'arc-a-etat-complet nommée au plan ; SEULE à réunir
  histoires/mesures Arc A + pin Arc C — déviation notée à Romain). Pin sévère IC ENTIER lu de
  pins_spatial.json : ic_bas 0.06033 / pin 0.07334 / ic_haut 0.08674.
Task 0 : contrat §A12 gravé (pocCascade2phys commit 3fd72aa, append-only après §C12). run_history
  GELÉ vérifié (blob 88e16e2d, md5 b564723a) — appelé n_episodes=160, non modifié. complete.
Task 1 (générateur h160_s{seed}.npz + vérif bit-identité préfixes) : DISPATCHÉ (implémenteur sonnet).
  Génération complète 5 seeds (~80 min) = job contrôleur en arrière-plan APRÈS validation script.
ARCHI 16L₀ (tranchée sur digest Explore) : couche ADDITIVE séparée. L_LIST=(10,20,40,80) gelé,
  importé par toute la chaîne (run_arcA_measure:107) — NON muté. h160_*.npz (Task1) + measures_160.npz
  (Task2, v2@160 + ferm/shuf@{10,160}, même layout (L,seed,budget)) séparés ; Task3 agrafe
  measures_qt.npz (L∈{10..80}) + measures_160.npz (L=160). measures_qt.npz / h_s*.npz / run_arcA_measure*
  INTACTS. Réutilise _measure_cell_qt/_field_true_qt/k_star_groupe_qt par import. Brief Task2 écrit.
DÉCISION MÉTHODE Task3 (nommée avant mesure, veto Romain possible) : pente moitié-haute {40,80,160}
  = OLS de k*(L) vs L (généralisation fidèle du 2-pts §A3 (k80-k40)/40), bootstrap comme §A3
  (10000, seed 20260704, mêmes indices seeds across L, IC percentile, ∞→NaN). Signal dominant
  k*(160) plat-256 vs saut-400 = comparaison directe budgets quantifiés, robuste à l'estimateur.
Task 1 : complete (commits a5c2731..9a2aeac, revue SPEC✅ + QUALITÉ approuvé). 2 fichiers nouveaux,
  aucun existant touché ; src/sediment.py blob 88e16e2d confirmé gelé. Relecteur a vérifié activement :
  bit-à-bit EXACT (np.array_equal, 0 allclose ; altération 1e-12 → BLOCKED/SystemExit), pas de clobber
  des originaux, 4 tests réels (replay ~29 s bit-identique). Suite 512/512. Minor non introduit
  (VERIFY_PATH figé au chargement, pattern miroir) → pas de correctif. Génération 5 seeds LANCÉE (arrière-plan).
Task 2 : complete (commits 9a2aeac..cfe4f29 + fix 1fc6ffe). Revue SPEC✅(réserve) + QUALITÉ approuvé ;
  finding Important corrigé (shuf@L=10 ajouté au test non-régression sous skipif — ferm+shuf couverts,
  np.array_equal strict, réutilise _mesurer_cellule) + Minor (ARMS importé). Relecteur a vérifié :
  _measure_cell_qt/_field_true_qt appelés aux bonnes signatures, sous_jnd délégué à sous_jnd_a_jnd,
  schéma measures_160.npz conforme (ma1/ma2_v2 (1,5,7) L=[160] ; ferm/shuf (2,5,7) L=[10,160]),
  aucun verdict/pente/k*, aucun fichier gelé touché, meta sans horodatage. Suite 517+1skip.
  RESTE (⚠ contrôleur) : lever skipif + confirmer non-régression L=10 bit-à-bit APRÈS mesure réelle.
Task 3 (verdict §A12) : DISPATCHÉ (implémenteur sonnet). Script + tests synthétiques/skipif ; run réel
  du verdict = contrôleur après measures_160.npz.
Task 1 npz commités (9c76d16) : 5× h160 + verify_16L0 PASS (préfixes bit-à-bit 5/5, path-dep L=160 f_ordre=0.730).
Task 2 mesure réelle : measures_160.npz commité (807 s). NON-RÉGRESSION L=10 CONFIRMÉE bit-à-bit (contrôleur,
  item ⚠ levé) : measures_160[ma1/ma2 × ferm/shuf] tranche L=10 == measures_qt, np.array_equal True 4/4.
  Aperçu (NON verdict) : ferm=0 exact partout ; shuf grand partout ; v2@160 ma1@256≈0.06-0.08 (straddle pin
  0.0733 — le verdict k*(160) plat-256 vs saut-400 se joue là, calcul mécanique Task 3, PAS d'œil).
Task 3 : complete (commits 8a2b38b..068a8f6 + fix Important b920e68 + verdict run). Revue opus SPEC✅ +
  QUALITÉ approuvé ; combinateur §A12 vérifié empiriquement (INDETERMINE sur cas réel, else=sortie sûre) ;
  Important corrigé (garde non-régression ancrée à surface_kstar_aux_pins.json, 6 pins×4 L bit-à-bit,
  non-tautologie prouvée) ; 3 Minor sans correctif (règle NaN moot ; clés JSON = pente/n_bootstrap_indefinis ;
  gate inline autorisé). Suite 536/0skip.
VERDICT §A12 = INDETERMINE_JND (gate CONFORME 0/80). Commit verdict = <à suivre>. k*(L) pin sévère :
  ic_bas 256,256,400,400,400 | pin 256,256,256,256,400 (saut 16×, pente IC∋0) | ic_haut 256 plat×5.
  Divergence colonnes ⇒ JND-dépendant DANS IC = surface k*(L,JND) pour Arc C. Soupçon pré-nommé NON réalisé
  (pin central saute, pas ic_bas). Re-dérivé main OK. RESTE : journal §A12 (contrat) + remontée verdict.
Revue whole-branch (opus) : PRÊT À GRAVER — instrument gelé préservé bit-à-bit end-to-end, appariement
  seeds prouvé, portée honnête (divergence tranchée par règle générale pas histoire anticipée), complétude OK.
  2 mineurs non-bloquants non exigés (assert appariement runtime ; scalaire bascule — surface déjà dans JSON).
Journal §A12-résultat gravé au contrat (pocCascade2phys, commit dédié). Verdict artefacts = c645558 (pocPhysicator).
═══ MISSION 16L₀ CLOSE. Verdict = INDETERMINE_JND (gate CONFORME). Cellule 1 NON prononcée définitivement :
  JND-dépendante DANS l'IC sévère = surface k*(L,JND) portée à l'Arc C. Règle dernière famille consommée.
  Point d'arrêt atteint : verdict remonté à Romain. AUCUN enchaînement manche 2 / 3e famille. ═══

═══════════════════════════════════════════════════════════════════════════════
MISSION : Manche 2 — registre-commis (§A13). Base = 6457301 (branche arc-a-manche2-registre,
  forkée d'arc-c-pin-jnd après clôture 16L₀). Contrat §A13 gravé : pocCascade2phys 811b11d.
  Pins lus de outputs/arcC/pins_spatial.json : sev 7.334 % IC [6.033, 8.674] ;
  lax 11.540 % IC [9.365, 13.874] — identiques à l'en-tête §A13. Substrat gelé vérifié :
  src/sediment.py blob 88e16e2d / md5 b564723a (== manche 1). Instruments présents :
  histoires h_s*/h160_s*, run_arcA_measure_qt, run_arcC_surface_kstar_pins.
  Points d'arrêt OBLIGATOIRES : après sonde (S1/S2), après gate d'achat si > 10 h, après verdict.
Task 0 (contrat+inventaire) : gravure ✓, inventaire ✓ ; cœur (commit/reconstruire/harnais chaîne
  + tests É1/É3/ferm-chaîne) → à dispatcher après carte Explore des API gelées.
Task 0 cœur : DISPATCHÉ (implémenteur sonnet, base 6457301). Brief m2-task0-brief.md. Décisions
  contrôleur gravées au brief : t0 = s=0 (chaîne-vérité == histoire gelée, ancrage h_s101), commit
  de l'ÉTAT-MOTEUR (composition possible = l'objet du claim), ordre strict mesurer→committer→
  ré-ancrer→continuer (Δχ_1 ≡ 0 structurel, documenté), rederiver_emissions(registre,seed,dt,n,arm)
  = foncteur É2/É3 signature stricte anti-fuite.
Task 0 : complete (commit 008e770, revue opus Approved — spec ✅ 10/10 tests + 4 fonctions,
  ordre §A13-1 strict vérifié, commit-du-moteur confirmé ligne à ligne, déviations #1/#2
  équivalence stricte). 11 tests dédiés + suite 547 passed, gels 3/3 inchangés.
  Minor (pour revue finale) : RELIEF importé non utilisé (noqa) ; rederiver arm="ferm" non
  couvert par test ; pas de garde verite trop court (KeyError).
Task 1 (sonde) : DISPATCHÉ (implémenteur sonnet, base 008e770). Brief m2-task1-brief.md.
  Lecture S1/S2/AUTRE pré-écrite mécanique ; point d'arrêt OBLIGATOIRE après (remonter la courbe).
── PAUSE SESSION (2026-07-06 soir) ──
Task 1 : IN-FLIGHT à la pause. Script scripts/run_arcA_m2_sonde.py + tests/test_arcA_m2_sonde.py
  écrits (NON commités, non revus) ; run sonde réelle probablement en cours ; outputs/arcA/
  m2_sonde.json PAS encore produit. REPRISE : vérifier si m2_sonde.json + commit existent ;
  sinon re-dispatcher Task 1 (brief m2-task1-brief.md intact) — les fichiers non commités
  peuvent servir de base mais doivent repasser par run réel + tests + revue.
  RAPPEL points d'arrêt : après sonde → remonter la courbe Δχ_i à Romain AVANT Task 2.
Task 1 : implémenteur DONE avant fermeture session (commit fceddf5 — sonde + tests + json + png).
  8 tests dédiés + suite 555 passed, gels inchangés. Série Δχ_i = [0, .0057, .0292, .0554, .0276,
  .0305], lecture=AUTRE (pic sous-JND à i=4 puis relaxation ; aucune traversée, même pas ic_bas).
  Chronos ~2.8 s/épisode → grille estimée ~2-2.5 h (largement sous gate W0).
  RESTE À FAIRE : revue Task 1 (non revue !) ; courbe REMONTÉE à Romain (fait, séance du 06/07
  soir) ; décision Romain sur l'achat de la grille ATTENDUE (lecture AUTRE = pas de branchement
  automatique) ; puis Task 2 gate d'achat.

── REPRISE SESSION (2026-07-15, Claude Cowork cloud) ──
Revue Task 1 (la dette « revue à faire ») : FAITE — sonde CONFORME §A13-3 (verbatim cité, lecture
  pure mécanique, chrono séparé vérité/moteur, aucun verdict). UN choix d'implémentation nommé :
  facteur 1.5 du critère S2 (pas dans le verbatim « borné/contractant ») — LOAD-BEARING (ratio
  observé 1.90 : avec facteur 2.0 la lecture aurait été S2→achat auto). Le protocole a bien remonté
  la forme au lieu de l'absorber. Minor Task 0 reconduits (rederiver ferm non couvert ; garde
  verite courte absente) — non bloquants.
DÉCISION ROMAIN (2026-07-15, point d'arrêt sonde levé) : GRILLE COMPLÈTE ACHETÉE (option
  pré-enregistrée §A13-3) ; exécution SUR LA MACHINE ROMAIN.
FAIT D'INSTRUMENT (mesuré, pas supposé) : le conteneur cloud N'EST PAS l'instrument gelé —
  run_history(101,10) vs h_s101.npz : bit-identité FAIL, écart max 2.7 % à 10 épisodes ;
  hypothèse logicielle ÉLIMINÉE (numpy 2.4.6 + python 3.12.3 épinglés = mêmes versions que le
  npz : FAIL identique) → cause CPU/BLAS (dispatch SIMD), amplifiée par la path-dependence.
  Conséquence gravée : toute mesure verdict-grade manche 2 tourne sur iluin-tworings3 ; le cloud
  sert de sandbox de dev (573 tests OK hors les 2 gels inter-machine, qui échouent PAR CONSTRUCTION
  ailleurs que sur l'instrument).
Tasks 2/3 BUILD (cloud, base b0e4723) : scripts/run_arcA_m2_grille.py (gate d'achat réel +
  phases résumables vérité/cellules/ferm/contrôles/assemble) + scripts/run_arcA_m2_verdict.py
  (combinateur §A13-2 pur, else=INDETERMINE sortie sûre) + 24 tests dédiés (test_arcA_m2_grille,
  test_arcA_m2_verdict) — suite locale PASS, ruff clean. SIX choix d'implémentation PRÉ-ENREGISTRÉS
  dans les docstrings ((i) vérité partagée préfixe-stable bit-exacte ; (ii) contrôles shuf/corruption
  POST-HOC via rederiver_emissions, cœur Task 0 INTOUCHÉ ; (iii) shuf = permutation des moyennes par
  commit, rng 9001+1000*dt+seed ; (iv) corruption = +10·S_HALF_OP sur la feuille de plus grande aire
  du commit émission 3, localisation mesurée au ré-ancrage ≥50 % ; (v) spot-check É2/É3 sur cellule
  (101,4,256) ; (vi) gate = benchmark réel cellule Δt=1 extrapolé comptabilité exacte 2640 épisodes).
  Comptabilité : vérité 480 + moteur 1890 + ferm 126 + rederive 24 + shuf 72 + corruption 48.
  Plomberie gate vérifiée cloud (7.6 s/ép → 5.58 h PASS) puis JSON SUPPRIMÉ (le vrai gate sortira
  de la machine d'exécution). Chrono sonde Romain (2.8 s/ép) → projection ~2 h.
RESTE À FAIRE : revue des choix (i)-(vi) par Romain AVANT exécution (comme le 1.5 de la sonde) ;
  transfert branche vers la machine Romain (pont desktop ou patch) ; --gate puis --tout sur
  iluin-tworings3 ; verdict = point d'arrêt OBLIGATOIRE, remonter, pas d'enchaînement.

── REPRISE SESSION (2026-07-15, tâche locale « sur votre ordinateur », VM Cowork sur iluin-tworings3) ──
TRANSFERT : les 2 commits appliqués par patch (git am) sur base b0e4723 — bit-exactitude PROUVÉE
  par les hashes de blobs (0fa5874 / 9055098 / e871af5 / 9348056 / c86b8c1 = index des patchs
  d'origine). Nouveaux SHA locaux f5be907 + c542548 (committer différent, contenu identique).
FAIT D'INSTRUMENT (mesuré, pas supposé) : la VM locale EST l'instrument gelé —
  test_replay_prefixe_bit_identique PASS (30.6 s) avec Python 3.12.3 exact (debs noble extraits
  + loader glibc, sans root) + site-packages de la .venv du dépôt (numpy 2.4.6, binaires natifs).
  Contraste cloud (FAIL 2×) → cause CPU/BLAS confirmée par un 3e point : même CPU ⇒ bit-identité.
  Les 24 tests grille+verdict PASS aussi (31.7 s).
LIMITE D'INFRA (mesurée) : chaque commande de la VM tourne dans un bwrap jetable
  (--die-with-parent, --unshare-pid), plafond DUR 45 s, aucun processus ne survit entre appels →
  aucune phase de grille exécutable ici (vérité ~4.5 min/seed, cellule Δt=16 ~13 min). Gate NON
  lancé dans la VM (choix (vi) : le gate se chronomètre sur LA machine qui exécute la grille,
  et cette VM ne peut pas l'exécuter) → repli chemin 1 : gate → --tout → verdict dans le
  terminal natif (.venv/bin/python, hors VM). Le PASS bit-identité de la VM reste porté : si
  l'infra lève un jour le plafond 45 s, la VM locale est un instrument valide.
RESTE À FAIRE : (natif, terminal Romain) pytest de vérification (~1 min) ; --gate (~40 s) ;
  --tout (~2 h, résumable) ; run_arcA_m2_verdict.py ; verdict = point d'arrêt OBLIGATOIRE —
  remonter à Romain, graver §A13-résultat au journal de pocCascade2phys AVEC lui, pousser les
  branches GitHub ensuite. Lecture d'innovation (ressaut-puis-relaxation) : APRÈS le verdict.

── CLÔTURE MANCHE 2 (2026-07-18) ──
EXÉCUTION NATIVE (terminal Romain) : 25 tests PASS (53.96 s) ; gate MESURÉ PASS (2.569 s/ép ×
  2640 = 1.88 h ≤ 10 h) ; grille lancée le 15, interrompue, REPRISE par parts le 18 (SKIPs
  corrects, design résumable exercé) ; anomalie chrono seed 103 (2×) = temps horloge seulement.
CONTRÔLES §A13-4 : 0 violation (ferm 0.0 exact ×3Δt ; shuf Δχ₂=1.028/0.752/0.693 ≫ JND ;
  corruption 0.3905 supra-JND, préfixe bit-identique, localisation 100 % ; É2/É3 bit-exacts).
VERDICT §A13-2 (combinateur mécanique) : **REGISTRE_FERME** — trois colonnes IC sévère fermées,
  k*_chaîne ≤ 256 floats tous Δt ; détail {ic_bas: 256/256/256, pin: 256/256/128,
  ic_haut: 128/256/128}. Chiffres inconfortables consignés au journal ((16,400,ic_bas) clause
  2·JND seed 102 ém.4 0.1369 non-monotone en k ; (4,128) médiane 0.0963 > pin).
POINT D'ARRÊT HONORÉ : verdict remonté, texte endossé par Romain, §A13-résultat GRAVÉ au journal
  de pocCascade2phys (commit 5633986, branche t1-spring-config). Lecture d'innovation portée
  (ressaut-puis-relaxation généralisé, k*(16)<k*(4) au pin) = décision neuve, PAS un enchaînement.
RESTE À FAIRE : push des deux branches par Romain (clés SSH hors VM) ; puis décisions neuves à
  ouvrir avec lui (consommateurs du verdict : gate fovéa-z ; cadencement-relaxation).
