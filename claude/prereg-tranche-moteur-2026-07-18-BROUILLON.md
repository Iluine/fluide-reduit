# Pré-enregistrement — TRANCHE-MOTEUR (F1) — ENDOSSÉ, GRAVÉ AU JOURNAL

> **Statut : ENDOSSÉ (Romain, 2026-07-18) et GRAVÉ — la version qui fait foi est
> pocCascade2phys `PREREGISTRATION.md` §A15.** Ce fichier est l'archive de travail.
> Aucune ligne de code de tranche avant chiffrage du build endossé (règle plan).

## Objet

F1 brûle 4 hypothèses d'un coup : (a) frame-time vs L_eff [le falsificateur du mot
« moteur »], (b) écart live↔rederive sous f32 [contrat D5/Option A], (c) débit PCIe du
schéma diff [topologie §5], (d) coût de load/replay du ledger [§4]. C'est une **MESURE,
pas un prototype** : le code autorisé est un harnais de tranche (scripts consommant les
primitives existantes + un chemin GPU jetable), PAS le moteur. Un critère de mort
déclenché = remontée + décision, jamais un contournement. Verdict SANS MORT requis pour :
la spec fait foi (gate iii), campagne r_fovea débloquée (garde anti-tapis-roulant §7).

**Ordre de dépense (D13) : F0-cloud d'abord.** F0 (~2 min, session cloud) n'est pas lu.
Ce pré-enregistrement peut être gravé avant, mais l'EXÉCUTION de F1 attend la lecture F0
`[PROPOSITION — T7 : Romain peut découpler si F0-cloud tarde, F0 ne gate que
l'hypothèse inter-machines §5, pas la tranche elle-même]`.

## Instrument

- Machine : **iluin-tworings3, terminal natif** (RTX 3050 Ti, 4 Go). La VM Cowork
  (45 s max, pas de GPU) = dev et lecture seulement `[MESURÉ : faits d'instrument]`.
- Stack GPU : **à nommer et endosser AVANT build** (T6) — contrainte : f32, 4 Go,
  Python 3.12.3 coexistant avec numpy 2.4.6 du chemin rederive. Candidats : CuPy /
  torch / JAX. `[NON-ANCRÉ — choix d'implémentation, docstring + endossement]`
- Chrono GPU : synchronisation explicite avant/après mesure ; warmup ≥ 30 frames EXCLU ;
  série ≥ 300 frames ; **médiane ET p99 reportées toutes deux** (chiffres inconfortables
  en évidence) `[PROPOSITION]`.
- **Vérification d'instrument DUE avant toute lecture (sinon remonter, tranche muette) :**
  1. Résidence VRAM mesurable et concordante (allocateur du stack vs nvidia-smi) ;
  2. Sensibilité du chrono : n_fov 256→512 doit bouger le frame-time (chrono non muet) ;
  3. Le chemin rederive CPU f64 reproduit le gel bit-exact sur la machine (le test de
     gel existant, reconduit) ;
  4. Bande passante PCIe soutenable MESURÉE à vide (transfert brut aller/retour) —
     c'est le dénominateur de M-c, mesuré AVANT de mesurer le schéma diff.

## Les 4 mesures et leurs critères de mort

### M-a — frame-time vs L_eff (le mot « moteur »)

**Protocole.** Pyramide active à l'enveloppe §6 épinglée (c=8, f32, n_fov=512, N_niv=10,
γ₂≈3 → ~7.9 M cellules actives, M_VRAM ~0.9 Go `[MESURÉ-par-calcul : note VRAM]`).
F représentatif appliqué chaque frame sur les fenêtres actives, **fenêtre fovéa MOBILE**
(translation continue → re-prédiction Harten à chaque déplacement, le coût structurel que
la fovéa fixe ne paie pas). Scan : N_niv actifs ∈ {2, 4, 7, 10} × n_fov ∈ {256, 512}
`[PROPOSITION]`. Le choix de F est T1 — il doit être représentatif en COÛT (stencil,
c=8 champs), pas en physique-jeu.

**MORT-a `[T2 TRANCHÉ Romain 2026-07-18]` :** médiane frame-time > **16.7 ms** à la
cellule d'enveloppe (n_fov=512, N_niv=10) — 60 fps plein, ou 30 fps avec la moitié de
frame réservée au rendu (le rendu n'est PAS mesuré ici). p99 : reportée, non verdictale
en v1.

**Gate §6 (reconduit, pas une mort) :** résidence mesurée > 1.5 × 0.9 Go ⇒ re-épinglage
du quadruplet — décision, jamais glissement.

### M-b — écart live↔rederive aux émissions (contrat D5/Option A)

**Protocole.** Chaîne d'émissions fenêtrées sur le chemin VIVANT GPU f32 (commits
k_fen aire-proportionnel, cap 10 % reconduit `[MESURÉ : §A13/§A14]`) ; rederive =
f(registre, seeds), CPU f64, disciplines du harnais. Δχ readout aux émissions
(albedo + delta_chi, max_carrier — l'espace instrument, jamais l'état). Cellule :
**3 seeds {101, 102, 103}, Δt=4, 6 émissions** `[PROPOSITION — M-b porte une mort, une
sonde 1-seed serait sous-dimensionnée ; Δt=4 est l'axe dur mesuré §A13-résultat]`.

**MORT-b `[T3 TRANCHÉ Romain 2026-07-18]` :** max de série Δχ live↔rederive > **0.0733**
(pin sévère), **par-seed** — un seul seed qui traverse suffit (le contrat protège une
production, un pic qui traverse EST l'événement). ⇒ Option A morte, repli Option B
(tout-déterministe) = **décision neuve** — son coût de frame n'est pas mesuré dans cette
tranche sans nouveau pré-enregistrement. `[MESURÉ : pin Arc C 7.33 %, IC [6.03, 8.67] %]`

### M-c — débit PCIe réel du schéma diff (topologie §5)

**Protocole.** Compter les octets/frame réels du trafic CPU↔GPU : descente (prédiction
Harten des fenêtres actives) + remontée (coefficients de détail). Enveloppe attendue :
échelle n_fov² — brut ~c·b·n_fov² ≈ 8.4 Mo/direction/frame plein-fovéa
`[MESURÉ-par-calcul : 512²×8×4]` ; le schéma diff doit faire MIEUX que le plein.

**MORT-c (deux branches) `[T4 TRANCHÉ Romain 2026-07-18]` :**
1. temps de transfert mesuré > **25 % × 16.7 ms = 4.2 ms** à la cellule d'enveloppe ;
2. **fuite d'échelle** : le trafic croît avec la taille du monde au lieu de n_fov²
   (doubler le monde à fovéa fixe doit laisser le trafic ~constant — sinon la topologie
   §5 est morte telle que dessinée).

### M-d — coût de load/replay du ledger (§4)

**Protocole.** Ledger de référence : **1 h simulée à 1 commit/s ≈ 3600 commits
≈ 11.5 Mo** `[MESURÉ-par-calcul : note VRAM]`, généré par la tranche elle-même.
load = re-dérivation chemin bit-exact jusqu'à la dernière émission + reprise du vivant.

**MORT-d `[T5 TRANCHÉ Romain 2026-07-18]` :** temps de load > **30 s** SANS
snapshot-racine. Diagnostic non-verdictal reporté : le même load AVEC un snapshot-racine
(format (d), §4) au milieu du ledger — chiffre ce que la compaction achète, sans qu'elle
porte le verdict.

## Ce que la tranche ne dit pas (portées, non surclamées)

Rien sur : r_fovea / excentricité (campagne gatée APRÈS verdict sans mort), 3D
(fine-fovéa 64³ = enveloppe calculée, pas mesurée), multi-vue/concurrence (v1.1),
qualité perceptuelle produit (pin n=1), coût du RENDU (hors tranche — physique seule),
coût de l'Option B (nouveau pré-enregistrement si repli).

## Chiffrage du build (DÛ avant achat — règle plan)

À remonter par Claude Code AVANT toute ligne : estimation du harnais de tranche
(chemin GPU F + fovéa mobile, portage commits/readout au vivant GPU, compteurs PCIe,
générateur de ledger 1 h). Gate : endossement Romain du chiffrage.

## Décisions Romain

**TRANCHÉES (2026-07-18) :** **T2** B_frame = 16.7 ms (p99 reportée non verdictale) ;
**T3** MORT-b = pin 0.0733, par-seed ; **T4** f_PCIe = 25 % (⇒ 4.2 ms) ;
**T5** T_load = 30 s (snapshot = diagnostic).

**TRANCHÉES (2026-07-18, « ok pour les 3 points ») :** **T1** substrat v2 étendu à
c=8 champs (coût-représentatif) ; **T6** stack GPU nommé au chiffrage, endossé AVANT
build ; **T7** exécution F1 DÉCOUPLÉE de F0-cloud (F0 ne gate que l'hypothèse
inter-machines §5 ; reste dû à la prochaine session cloud).
