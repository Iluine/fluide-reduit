"""F1 TRANCHE-1 — harnais de tranche-moteur, composants (a)(c)(d)(e)(f).

Gate d'achat TRANCHÉ (pocCascade2phys `PREREGISTRATION.md` §A15 + §A15-complément,
commit 8a26035, E1-E7 endossés Romain 2026-07-18 ; chiffrage endossé :
`claude/chiffrage-tranche-moteur.md`). Le composant (b) — portage fidèle M-b —
n'est PAS acheté : tranche-2, gatée sur verdict tranche-1 SANS MORT (E3).

Ce paquet est le CHEMIN GPU JETABLE autorisé par §A15 : un harnais de MESURE,
pas un moteur. Le cœur manche 2 (`scripts/run_arcA_m2_registre.py`) et le
chemin rederive CPU f64 (numpy 2.4.6) sont INTOUCHÉS — toute consommation
passe par import. Aucun seuil de verdict n'est défini ici : les chiffres
gravés (16.7 ms, 4.2 ms, 30 s, gate résidence 1.35 Go) ne sont consommés que
par les drivers `scripts/run_f1_*.py`, en LECTURE MÉCANIQUE remontée — jamais
un verdict prononcé, jamais un enchaînement automatique.

Modules :
  - `backend.py`    : sélection numpy/cupy (B1) ;
  - `chrono.py`     : chrono GPU cuda-Events, warmup 30 / série 300,
                      médiane ET p99 (B6) — composant (f) ;
  - `transferts.py` : `TransfertComptable` (octets + temps H2D/D2H par frame)
                      et bande passante à vide — composant (c), vérif #4 ;
  - `substrat_jetable.py` : F jetable c=8 f32, motif de coût du stencil
                      wetdry O2 (B5, E4a) — composant (a) ;
  - `pyramide.py`   : pyramide de fenêtres actives + fovéa mobile +
                      schéma diff descente/remontée (B3, B4, E4b-E4d) —
                      composant (a) ;
  - `ledger.py`     : format ledger v1 (§4 spec), écriture append-only,
                      parse+validation intégrale, load E7 (B7, B8) —
                      composant (d) ;
  - `verifs.py`     : registre des vérifications d'instrument + câblage
                      « tranche muette ⇒ remonter » (B9) — composant (e).

CHOIX D'IMPLÉMENTATION NOMMÉS PAR CE BUILD (cadence maison : à endosser par
Romain AVANT tout run de mesure — points que le gravé fixe en intention mais
pas en mécanique ; le détail vit dans la docstring du module porteur) :

  B1  backend injecté : chaque composant prend un module tableau `xp`
      (cupy pour la MESURE — les drivers l'exigent —, numpy uniquement pour
      tester la logique en VM : pas de GPU, kill 45 s). Le chemin numpy ne
      produit JAMAIS une lecture.
  B2  E4b opérationnalisé : les BUFFERS D'ÉTAT (fenêtres, tampons de
      prédiction, monde niveau 0) sont préalloués à shapes fixes à la
      construction — zéro allocation d'état par frame ; les TEMPORAIRES du
      stencil passent par le mempool CuPy (recyclage à shapes constantes,
      régime atteint pendant le warmup 30, exclu de la série).
  B3  géométrie pyramide : niveau 0 = CPU (§5), N0 = 256 par côté ; niveaux
      GPU j = 1..N_niv−1, γ₂ = 3 fenêtres n_fov² par niveau : la fovéale
      du niveau + 2 fenêtres d'énergie FIGÉES au centre initial (offsets y
      ±n_fov, clampées au monde — chevauchement possible aux niveaux
      grossiers, coût identique). SEULE la fovéale suit le balayage
      (consigne 2, §A15-complément-2 gravé) : le trafic de déplacement est
      un MINORANT nommé, porté dans la portée de la lecture M-c ;
      alignement dyadique : l'origine fovéale au niveau j suit
      ⌊centre_fin/2^(J−j)⌋ (elle bouge d'1 cellule toutes les 2^(J−j)
      frames). F s'applique chaque frame aux γ₂ fenêtres (E4a inchangé).
  B4  schéma diff opérationnalisé (E4d) : descente H2D = les seules cellules
      ENTRANTES des fenêtres déplacées cette frame, prédites CPU depuis le
      niveau 0 (prédiction voisin, motif M6 du dépôt), une descente batchée
      par niveau déplacé — la descente pleine initiale est payée hors-série ;
      remontée D2H = coefficients de détail SEUILLÉS (|d| ≥ EPS_DETAIL figé,
      valeurs f32 + indices u32) des fenêtres touchées par F cette frame,
      puis la référence du diff est remise à l'état remonté (schéma
      incrémental : le CPU « sait » ce qui est remonté). Remonter les
      coefficients DENSES de toutes les fenêtres ferait ×γ₂·(N_niv−1) le
      plein que le diff doit battre — ce ne serait pas le schéma §5.
      EPS_DETAIL est NON-ANCRÉ perceptuellement (consigne 1,
      §A15-complément-2) : hérité par le pré-enregistrement tranche-2
      comme SUSPECT NOMMÉ si M-b lit AUTRE ; la lecture M-c reporte le
      dense-équivalent EN ÉVIDENCE.
  B5  F jetable : motif de coût du stencil wetdry O2 RÉEL (padding
      réfléchissant, pentes minmod x/y, états d'interface, flux HLL,
      corrections de pression, divergence, 2 étages SSP-RK2, plancher sec,
      réduction CFL par frame — le moteur réel la paie), appliqué à
      2 systèmes (h, hu, hv, s) indépendants (E4a), vectorisés sur un axe
      système et batchés par niveau (3 fenêtres/lancement — le moteur réel
      batcherait des fenêtres identiques ; réduit l'overhead de lancement
      sans réduire le calcul). dt FIGÉ (la stabilité du jetable n'est pas
      l'objet ; la réduction CFL est calculée pour son coût, pas consommée).
      Bathymétrie PLATE (b = 0) : z*, h* et corrections calculés quand même
      — coût identique, stabilité du jetable assurée.
  B6  chrono : cuda-Events + synchronisation device avant série et par
      frame ; warmup 30 EXCLU ; série 300 ; médiane et p99 =
      numpy.percentile(50 | 99), interpolation linéaire ; AUCUN timestamp
      dans les données de mesure.
  B7  ledger v1 : répertoire {index.jsonl append-only + data.bin
      append-only} ; entrée = {type, t_sim, seq, meta, payloads
      (offset, taille, sha256, dtype, shape)} ; clé (t_sim, seq)
      strictement croissante, garantie à l'APPEND et re-vérifiée au PARSE ;
      les commits fenêtrés consomment les primitives §A14
      (`commettre_fenetre`/`extraire_fenetre` — cœur intouché).
  B8  load E7 : parse+validation INTÉGRALE (toutes les entrées, tous les
      payloads relus, tous les sha256) → `regenerate_qt(dernier commit)` →
      replay du segment courant (évoluteur INJECTÉ ; production =
      `run_episode`, tests = évoluteur trivial) → reprise du vivant = H2D de
      l'état (compté dans le chrono de load). Le DEHORS de la fenêtre n'est
      PAS reconstruit (éphémère, §3 — seul le persistant porte témoignage :
      lecture mécanique d'E7 endossé). Écart matériel avec/sans snapshot
      (consigne E7-2) : |Δ| > max(10 % du load sans snapshot, 0.5 s) —
      seuil d'INSTRUMENT figé avant run, pas un seuil de verdict.
  B9  câblage « tranche muette ⇒ remonter » : registre
      `outputs/f1/verifs.json` ; chaque driver appelle `exiger_pass(...)`
      avant toute lecture (RuntimeError sinon). Vérif #2 (sensibilité
      chrono) : à N_niv identique, médiane(n_fov=512) doit dépasser
      médiane(n_fov=256) d'au moins 5 % — seuil d'instrument nommé.

Mesures : machine iluin-tworings3, terminal natif UNIQUEMENT (VM = dev/tests).
POINT D'ARRÊT après chaque run : lecture remontée à Romain, rien d'autre."""
