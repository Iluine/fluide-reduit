# Chiffrage du build — harnais TRANCHE-MOTEUR (F1) — 2026-07-18

> **Statut : PROPOSITION soumise à Romain — c'est le gate d'achat (§A15, règle plan).**
> Aucune ligne de code de tranche avant endossement de ce chiffrage. Sources faisant
> foi : pocCascade2phys `PREREGISTRATION.md` §A15 (gravé 2026-07-18), `SPEC-FOVEA-Z.md`
> §5/§6/§8 ; règles d'exécution : `claude/plan-apres-manche2-2026-07-18.md`,
> enveloppe : `claude/note-budget-vram-2026-07-18.md`. Les chiffres §A15 (16.7 ms,
> 0.0733 par-seed, 4.2 ms, 30 s) sont gravés et ne sont pas rediscutés ici.
> Estimations ancrées sur le code réel du dépôt (`run_arcA_m2_registre.py` 228 L,
> `src/solver_wetdry.py` 448 L, `src/sediment.py` 562 L, sonde §A14 574 L + 13 tests).

**Unités.** « Séance » = une session Claude Code bornée (build + tests + revue
adversariale + cycle d'endossement — la cadence maison, ~S1 de la sonde §A14 = 1 séance).
LOC = ordre de grandeur **style maison** (docstrings denses en français + tests dédiés
inclus). Risque = probabilité que le composant glisse de « mesure » vers « prototype »
ou dépasse son chiffrage.

---

## 0. Lecture du périmètre (ce que le harnais est / n'est pas)

Code autorisé (§A15) : scripts consommant les primitives existantes + **chemin GPU
jetable**. Le cœur manche 2 (`run_arcA_m2_registre.py`) est INTOUCHÉ — toute extension
passe par de nouveaux scripts qui le consomment. Implantation proposée :
`src/f1_gpu/` (modules jetables du chemin GPU) + `scripts/run_f1_*.py` (un driver par
mesure M-a/M-b/M-c/M-d + vérifications d'instrument), tests sous `tests/test_f1_*.py`.
Le chemin rederive CPU f64 (numpy 2.4.6) n'est pas modifié d'une ligne.

Un point structurel découvert en chiffrant, qui commande tout le reste : **(a) et (b)
ne sont PAS le même F.** (a) exige un F *coût-représentatif* (stencil + c=8 champs,
« pas en physique-jeu », T1) — jetable, aucune exigence de fidélité numérique. (b)
exige que le chemin vivant GPU f32 joue **la même physique** que le rederive CPU f64
(`run_episode` : solveur wetdry O2 CFL-adaptatif + Exner + pulses), sinon l'écart Δχ
live↔rederive confond « non-déterminisme f32 GPU » (l'objet de la mesure, contrat D5)
et « physique différente » (un artefact de harnais). (b) est donc un portage
numériquement fidèle — c'est exactement le risque nommé §A15, et il est réel (cf. §3).

---

## 1. Découpage et chiffrage par composant

| # | Composant | Effort | LOC (~) | Risque |
|---|---|---|---|---|
| (a) | Chemin GPU F coût-représentatif + pyramide Harten + fovéa mobile | 2–3 séances | 1000–1500 | MOYEN |
| (b) | Portage fidèle commits/readout sur chemin vivant GPU f32 (M-b) | 3–5 séances | 1200–1800 | **ÉLEVÉ** |
| (c) | Compteurs PCIe + bande passante à vide | 0.5 séance | 250–350 | FAIBLE |
| (d) | Générateur ledger 1 h (3600 commits) + load | 1 séance | 300–450 | MOYEN |
| (e) | Les 4 vérifications d'instrument §A15 | 0.5 séance | 150–250 | FAIBLE |
| (f) | Chrono GPU + driver de scan M-a | 0.5–1 séance | 250–350 | FAIBLE |
| | **Total tranche complète** | **8–11 séances** | **~3200–4700** | |
| | **Total SANS (b)** (cf. scission, §3) | **~5–6 séances** | **~2000–2900** | |

### (a) Chemin GPU F — substrat v2 étendu à c=8, pyramide, fovéa mobile — sert M-a (+M-c)

Trois sous-blocs :

1. **F jetable c=8 f32** (~250–350 L) : transcription CuPy du *motif de coût* du stencil
   wetdry O2 (`_rhs`/HLL/reconstruction/padding — mêmes accès mémoire, mêmes passes),
   étendue à 8 champs f32. Choix d'implémentation à endosser **(E4a)** : les 8 champs =
   duplication du système (h, hu, hv, s) ×2, chacun sous le stencil complet — coût
   majorant honnête ; l'alternative « 3 physiques + 5 passifs advectés » minore le coût
   et fabriquerait un M-a complaisant.
2. **Pyramide Harten fenêtres actives** (~350–500 L) : structure par niveau j=0..9,
   γ₂≈3 fenêtres de n_fov² par niveau (~7.9 M cellules à l'enveloppe), extraction
   alignée-dyadique (le plongement §A14 (v) est le prototype de l'interface),
   prédiction Harten (descente) et coefficients de détail (remontée). **Préallocation à
   shapes fixes par niveau** (E4b) — pas de realloc par frame : condition à la fois du
   chrono propre et de la vérif VRAM #1.
3. **Fovéa mobile + re-prédiction** (~200–300 L) : translation continue de la fenêtre ;
   à chaque déplacement, les cellules entrantes sont re-prédites depuis le parent
   (le coût structurel que la fovéa fixe ne paie pas — c'est précisément lui que M-a
   doit voir). Choix à endosser **(E4c)** : trajectoire par défaut = balayage à
   1 cellule fine/frame en x (+ un cas « saut de fenêtre » reporté en diagnostic,
   non verdictal). **(E4d)** cadence de remontée des détails : par défaut, remontée
   chaque frame des coefficients des seules fenêtres touchées par F — c'est le
   « schéma diff » que M-c mesure ; remonter TOUT chaque frame reviendrait au plein
   que le diff doit battre.

**Risque MOYEN, nommé :** l'overhead de lancement kernel CuPy (des dizaines d'ops
numpy-style × ~10 fenêtres × niveaux, à quelques µs le lancement) peut peser à
l'échelle d'un budget 16.7 ms. Mitigations nommées d'avance, PAS achetées :
`@cupy.fuse`, `RawKernel` fusionné, CUDA Graphs. Si M-a meurt à cause de l'overhead
de lancement et non du calcul, c'est une lecture d'instrument à remonter, pas un
verdict silencieux — le scan N_niv×n_fov (f) permet de distinguer les deux (un coût
~constant en N_niv signe l'overhead).

### (b) Portage commits/readout sur le chemin vivant GPU f32 — sert M-b

Ce que M-b exige réellement (lecture du code, pas une opinion) : la chaîne d'émissions
fenêtrées (k_fen aire-proportionnel, cap 10 %, mécanique §A14 reconduite) tournée sur
un chemin vivant qui EST `run_episode` en f32 GPU. Sous-blocs :

1. **Portage fidèle du solveur** (~450–650 L) : `simulate_wetdry_o2` (reconstruction
   MUSCL/minmod, HLL, wet/dry, SSP-RK2, CFL adaptatif) en CuPy f32. La transcription
   est mécanique (CuPy mime numpy) mais la **fidélité de transcription est l'enjeu de
   la mesure** : une erreur de portage se lirait comme un écart Option A.
2. **Portage de l'épisode** (~200–300 L) : pulse gaussien, relaxation N_settle=600,
   intégration Exner séquentielle, assèchement. Les centres restent tirés CPU
   (`_draw_rollout_pairs`, motif VERBATIM — aucun aléa GPU).
3. **Chaîne d'émissions fenêtrées côté GPU + driver M-b** (~250–350 L) : aux émissions,
   descente D2H de la fenêtre (les commits vivent CPU-side, H2 : le diff GPU ne touche
   jamais le persistant), `summarize_qt`/plongement §A14 inchangés (CPU f64 sur le
   readout descendu), ré-ancrage, Δχ albedo+delta_chi/max_carrier — l'espace
   instrument, jamais l'état. Rederive : `f(registre, seeds)` CPU f64 existant,
   signatures strictes anti-fuite reconduites.
4. **Validation du port** (~300–500 L de tests) : mêmes gardes fail-loud, test de
   plausibilité du port (le f32 GPU sur UN épisode doit rester borné-proche du f64
   CPU — contrôle d'instrument du harnais, seuil mécanique, PAS un verdict).

Choix d'implémentation à endosser **(E5)** : le chemin vivant calcule **son propre dt
CFL en f32** (c'est son droit, Option A — la séquence de pas peut différer du f64).
Forcer la séquence dt du rederive f64 dans le vivant réduirait l'écart mesuré et
fabriquerait le verdict : interdit, nommé ici pour qu'il ne soit pas « découvert »
en cours de build.

**Risque ÉLEVÉ, le morceau-prototype :** c'est le seul composant où le harnais
implémente de la *physique fidèle* neuve (tout le reste consomme l'existant ou du
jetable). ~40 % du build total, la moitié du risque total. C'est lui qui porte la
question SCISSION (§3).

### (c) Compteurs PCIe + bande passante à vide — sert M-c + vérif #4

Classe `TransfertComptable` unique par laquelle passent TOUS les H2D/D2H du harnais
(octets + temps par cuda-Events, par direction, par frame) — CuPy rend chaque
transfert explicite (`cp.asarray`/`.get()`), rien ne traverse implicitement, ce qui
rend le comptage exact par construction (~120–180 L). Script bande passante à vide
(vérif #4) : transferts bruts aller/retour à tailles croissantes, pageable et pinned,
médiane+p99 (~100–150 L). Test de fuite d'échelle MORT-c(2) : N_niv 10→11 à fovéa
fixe (doubler le monde = un niveau de plus), trafic/frame ~constant attendu — un
paramètre du driver (f), pas un composant neuf. Risque FAIBLE.

### (d) Générateur de ledger 1 h + load par re-dérivation — sert M-d

Format ledger v1 (§4) : entrées (a) événement seedé / (c) commit d'émission /
(d) snapshot-racine, append-only, clé (t_sim, seq), sérialisation simple
(~150–200 L). Load (~100–150 L) : lecture + validation des 3600 entrées,
reconstruction de l'état à la dernière émission, reprise du vivant (upload GPU).
Driver M-d + diagnostic snapshot-racine au milieu (~80–120 L).

Deux points à trancher, nommés AVANT build :

- **(E6) Coût de génération — le coût caché du composant.** 3600 commits générés par
  la chaîne réelle = 3600×Δt épisodes de physique. Sur le chemin CPU f64 (épisode
  ~2–5 s), Δt=4 ⇒ ~8–20 h : infaisable en séance. Proposition : cadence de génération
  **Δt=1 épisode/commit** (la sémantique de chaîne est préservée ; l'étiquette t_sim
  « 1 commit/s » est un label de cadencement, pas une horloge physique) ⇒ ~2–5 h,
  run natif de nuit, NON-verdictal (MORT-d ne chronomètre que le LOAD, jamais la
  génération). Alternative si (b) est acheté d'abord : générer par le chemin GPU
  vivant (minutes) — mais cela ordonnerait (b) avant (d), cf. §3.
- **(E7) Interprétation de « re-dérivation jusqu'à la dernière émission ».** Par la
  mécanique même du ré-ancrage (`s_moteur = regenerate_qt(commit)`, cœur m2),
  l'état à la dernière émission est fonction du DERNIER commit seul + du segment de
  seeds courant — le load n'a PAS à rejouer les 3600 segments (ce qui coûterait ~la
  génération et tuerait MORT-d par construction, sans rien mesurer d'intéressant).
  Load = parse+validation intégrale du ledger + regenerate(dernier commit) + replay
  du segment courant + reprise. C'est une conséquence de §4, pas un adoucissement de
  MORT-d — mais je la nomme pour endossement explicite, car elle borne ce que 30 s
  mesure : le coût du LEDGER (parse, intégrité), pas celui de la physique.

Risque MOYEN (porté par E6/E7, pas par le code).

### (e) Les 4 vérifications d'instrument §A15 — dues avant toute lecture

1. **Résidence VRAM** : `cupy` mempool `used_bytes()`/`total_bytes()` vs
   `nvidia-smi --query-gpu=memory.used` — concordance à tolérance nommée (l'écart
   contexte CUDA est constant et mesurable à vide) (~60–100 L).
2. **Sensibilité du chrono** : n_fov 256→512 doit bouger la médiane — une assertion
   du driver (f) (~30–50 L).
3. **Gel bit-exact CPU f64** : reconduction du test existant
   (`test_replay_prefixe_bit_identique`), exécuté natif APRÈS installation du stack
   GPU — c'est aussi la garde de coexistence numpy 2.4.6 (cf. §2). ~0 L neuve.
4. **Bande passante à vide** : couverte par (c).

Risque FAIBLE. Câblage « tranche muette ⇒ remonter » : chaque driver refuse de
produire une lecture si sa vérification préalable n'a pas un PASS enregistré.

### (f) Chrono GPU + driver de scan — sert M-a (et porte c, e2)

Module chrono (cuda-Events, sync explicite avant/après, warmup 30 exclu, série 300,
médiane ET p99 reportées toutes deux) (~80–120 L) + driver de scan
N_niv ∈ {2,4,7,10} × n_fov ∈ {256,512} + cellule d'enveloppe, sorties JSON
sans timestamp dans les données de mesure, lecture mécanique remontée (~170–230 L).
Wall-clock de run : 8 cellules × 330 frames — minutes, négligeable. Risque FAIBLE.

---

## 2. Stack GPU (T6) — proposition : **CuPy** (choix nommé, PAS acheté)

**Choix nommé : CuPy 13.x (`cupy-cuda12x`), même venv, `numpy==2.4.6` épinglé.**

Justification contre les contraintes §A15 (f32, 4 Go, coexistence numpy 2.4.6 du
rederive CPU f64 intouché) et contre ce que la tranche MESURE :

- **M-c exige l'attribution exacte des octets PCIe.** CuPy ne transfère RIEN
  implicitement : chaque H2D/D2H est un appel explicite → le compteur (c) est exact
  par construction. C'est le disqualifiant de **JAX** : XLA gère les transferts et le
  placement implicitement — les octets/frame deviennent une inférence, pas un comptage.
- **Vérif #1 (résidence VRAM).** Mempool CuPy interrogeable et désactivable ;
  JAX préalloue par défaut ~75 % de la VRAM (la concordance allocateur/nvidia-smi
  devient un artefact de config) ; torch OK sur ce point.
- **Fovéa mobile = shapes qui bougent.** JAX recompile par shape (la re-prédiction au
  déplacement déclencherait des tempêtes de JIT ou imposerait un padding qui fausse
  le coût mesuré). CuPy/torch : exécution eager, aucun coût caché de compilation
  dans la frame.
- **(b) est une transcription de numpy.** CuPy mime l'API numpy : le portage fidèle
  de `solver_wetdry`/`sediment` est ligne-à-ligne — c'est la mitigation LA plus
  directe du risque « fidélité de transcription » de (b). torch imposerait une
  traduction d'API complète (risque de divergence sémantique silencieuse,
  ex. broadcasting, indexation avancée).
- **Chrono** : cuda-Events + sync explicite disponibles dans les trois — non
  discriminant.
- **Second choix : torch** (si CuPy rencontrait un mur d'install sur l'instrument) —
  tout y est faisable, au prix du risque de traduction ci-dessus. **JAX écarté** pour
  cette tranche malgré son usage prouvé dans pocCascade2phys (autre venv, autre
  usage : là-bas le graphe différentiable unique est le levier ; ici il n'achète rien
  et coûte l'attribution PCIe).

Conditions d'installation (font partie de e) : driver 570/CUDA 12.8 → wheel
`cupy-cuda12x` ; vérifier à l'install que la version CuPy retenue déclare le support
numpy 2.4.x (sinon épingler la plus récente qui le fait) ; **re-jouer la vérif #3
(gel bit-exact) après install** — c'est elle qui prouve que le chemin rederive n'a
pas bougé, quel que soit ce que pip a fait. VRAM : l'enveloppe 0.9 Go passe large
sous 4 Go (gate §6 reconduit : résidence > 1.35 Go ⇒ re-épinglage, décision).

---

## 3. Question SCISSION M-b (§A15) — réponse : **OUI, scinder**

Le chiffrage confirme le risque nommé §A15 : **(b) domine** — ~40 % de l'effort
(3–5 séances sur 8–11), le seul risque ÉLEVÉ, et le seul composant qui implémente de
la physique fidèle neuve (la définition même du glissement mesure→prototype).

| Option | Coût build | Ce qu'elle achète | Ce qu'elle coûte/retarde |
|---|---|---|---|
| **Tranche unique** (a–f d'un coup) | 8–11 séances avant toute lecture | Verdict F1 complet en une campagne ; spec fait foi dès le verdict (si sans mort) | Si MORT-a ou MORT-c tombe, les 3–5 séances de (b) sont dépensées pour rien — contraire à « le moins cher qui peut échouer, d'abord » |
| **Scission (recommandée)** : tranche-1 = M-a/M-c/M-d via (a)(c)(d)(e)(f) ; tranche-2 = M-b via (b) | Tranche-1 : 5–6 séances ; tranche-2 : 3.5–5 séances (achetée SEULEMENT si tranche-1 sans mort) | M-a est le falsificateur du mot « moteur » : il tue le plus, le plus tôt, le moins cher. (b) profite en plus de l'infrastructure débuggée de tranche-1 (chrono, compteurs, pyramide) | Le verdict F1 complet — donc « la spec fait foi » (gate iii §8) ET le déblocage r_fovea (garde §7) — attend la fin de tranche-2 : +3.5–5 séances de latence sur le chemin critique si tout survit. M-d/E6 : génération du ledger par CPU (nuit) au lieu du chemin GPU vivant |

Les cellules M-a/M-c/M-d et M-b sont indépendantes par construction (aucun chiffre
gravé ne lie leurs protocoles) : la scission ne modifie AUCUN critère de mort, aucun
protocole, aucune portée — seulement l'ordre d'achat. Elle ne demande donc pas de
re-gravure du §A15, seulement une consigne d'exécution endossée (à consigner au
journal comme décision d'ordre, style T7).

**Recommandation : scission, tranche-1 d'abord.** C'est la discipline maison
appliquée à son propre build. Si Romain préfère la tranche unique (pour compresser la
latence vers « la spec fait foi »), le chiffrage ci-dessus la couvre telle quelle.

---

## 4. Coûts de run (hors build, pour information — machine iluin-tworings3 natif)

- M-a : minutes (8 cellules × 330 frames + vérifs). M-c : minutes (à vide + scan).
- M-b : 3 seeds × 6 émissions × Δt=4 — GPU minutes + rederive CPU ~5–10 min.
- M-d : **génération ~2–5 h CPU natif (nuit, non-verdictal, cf. E6)** ; load : la
  mesure elle-même (< minutes).
- VM Cowork : dev/lecture uniquement (kill 45 s) ; aucune mesure n'y est valide.

## 5. Décisions demandées à Romain (le gate d'achat)

- **E1** — Endosser le chiffrage global (tableau §1) : ouvre le build, rien d'autre.
- **E2** — T6 : stack GPU = CuPy (`cupy-cuda12x`), même venv, numpy épinglé,
  vérif #3 re-jouée après install. (Second choix nommé : torch.)
- **E3** — SCISSION : tranche-1 (M-a/M-c/M-d) puis tranche-2 (M-b) — recommandée —
  ou tranche unique.
- **E4** — Choix nommés du composant (a) : (E4a) 8 champs = (h,hu,hv,s)×2 sous
  stencil complet ; (E4b) shapes fixes préalloués par niveau ; (E4c) trajectoire
  fovéa = balayage 1 cellule fine/frame ; (E4d) remontée = coefficients des fenêtres
  touchées, chaque frame.
- **E5** — (si (b) acheté) le vivant f32 calcule son propre dt CFL — jamais la
  séquence du f64.
- **E6** — Génération du ledger M-d : cadence Δt=1 épisode/commit, CPU natif de
  nuit, non-verdictal.
- **E7** — Interprétation du load M-d : parse intégral + regenerate(dernier commit)
  + segment courant (conséquence du ré-ancrage §4, nommée pour endossement).

---

**POINT D'ARRÊT.** Chiffrage remonté. Aucune ligne de code de tranche, aucune
installation, aucune mesure tant que E1–E7 (a minima E1/E2/E3) ne sont pas tranchées.
