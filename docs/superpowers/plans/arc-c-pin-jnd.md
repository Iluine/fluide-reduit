# Arc C — Harnais de pin JND (C-2) — Plan de mission

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development
> (recommandé) ou superpowers:executing-plans. Steps en cases `- [ ]`.

> Contrat : addendum §C0–§C5 de PREREGISTRATION.md (pocCascade2phys, commit `c494d50`) — fait
> foi verbatim. Ce plan est la Partie 2 du prompt de mission, apposée telle quelle.

**Goal :** construire le harnais ABX + staircase (axes spatial et temporel, régimes sévère
et laxiste selon §C0), générer les stimuli depuis les artefacts commités, faire tourner les
sessions de mesure (sujet : Romain), produire les quatre pins (valeur + IC) et appliquer
MÉCANIQUEMENT les lectures §C4. La partie C-1 (bibliographie) n'est PAS dans cette mission
(exécutant : session critique).

**Contrat :** l'addendum §C0–§C5 fait foi verbatim. Aucun seuil ni attendu retouché après
première donnée. Sessions invalides = résultats, consignés.

**Dépôts :** code + tests + `outputs/arcC/` dans pocPhysicator (branche `arc-c-pin-jnd`) ;
contrat + journal dans le dépôt du contrat. Aucun artefact load-bearing hors git ; les logs
bruts de CHAQUE essai (stimulus, Δχ mesuré, réponse, latence) sont commités.

**Tech stack :** Python 3.12, numpy/scipy/matplotlib uniquement (harnais interactif
matplotlib + clavier — pas de dépendance UI neuve sans demander). CPU.

## Global Constraints
- **Task 0 avant tout code** : apposer §C0–§C5 ; **POINT D'ARRÊT : validation Romain de
  C-0 (mapping étages→régimes) ET des conditions d'affichage** (distance, taille affichée,
  colormap `render.py`, luminance/gamma, durées) — consignées au journal AVANT le premier
  stimulus. Vérifier la présence des npz manche 1 (stimuli spatiaux, commités).
- **Axe spatial d'abord** (zéro régénération). Axe temporel : la régénération des films
  Boussinesq Ra=10⁷ (fin Nz=128, 4 graines, T=100) est plafonnée à **2 h CPU — STOP et
  remonter si dépassé** ; archives résultantes commitées (leçon reboot).
- **Δχ mesuré par stimulus** (instrument R1 existant), jamais interpolé depuis t.
- Staircase 2-down-1-up ; seuil = moyenne des 6 derniers renversements ; ≥ 3 staircases
  par condition ; catch trials ~10 % à Δχ fort ; validité §C5 appliquée mécaniquement.
- Randomisation seedée et consignée (sessions reproductibles au replay des logs).
- Points d'arrêt : après Task 0 (C-0) ; après les sessions spatiales (pins spatiaux
  remontés AVANT lancement du temporel) ; après Task 5 (lectures §C4) — TOUJOURS.

## Tasks
- [ ] **Task 0 — Contrat + conditions.** Addendum, validation C-0, conditions d'affichage
  gravées, inventaire stimuli. Point d'arrêt.
- [ ] **Task 1 — Stimuli spatiaux.** Générateur `stim(t)` (mélange vrai/régénéré par budget,
  champs albedo rendus via `render.py` aux conditions gravées) + Δχ(t) mesuré R1 par
  stimulus. Tests : déterminisme, Δχ(0)=0, courbe Δχ(t) rapportée (mesurée, pas supposée
  monotone), bornes physiques.
- [ ] **Task 2 — Harnais ABX + staircase.** Deux régimes (§C0) : sévère (simultané,
  inspection libre) / laxiste (séquentiel, D = 5 s, masque bruité, 2 s d'exposition).
  Staircase, catch trials, logs bruts par essai. Tests : logique staircase sur sujet
  synthétique (psychométrique simulée → seuil retrouvé ± tolérance), replay déterministe
  des logs, insertion catch conforme.
- [ ] **Task 3 — Sessions spatiales (sujet : Romain).** ≥ 3 staircases × {sévère, laxiste},
  validité §C5 par session. Sorties : JND_sev^spat, JND_lax^spat (+ IC),
  `outputs/arcC/pins_spatial.json`, figures psychométriques. **Point d'arrêt : pins
  spatiaux remontés.**
- [ ] **Task 4 — Stimuli + sessions temporels.** Régénération films (cap 2 h), filtre de
  réduction σ_ω(kx=1) paramétrique [−10, −60] % (réduction VÉRIFIÉE par re-mesure σ_ω sur
  chaque stimulus — jamais supposée), sessions ABX vidéo aux deux régimes. Sorties :
  JND_sev^temp, JND_lax^temp (+ IC), `outputs/arcC/pins_temporel.json`. Option consommateur
  3 : une staircase spatiale sur stimuli Boussinesq (coût marginal, remplace l'hypothèse de
  transfert par une mesure).
- [ ] **Task 5 — Lectures §C4, mécaniques.** Les quatre consommateurs, sur l'IC entier de
  chaque pin, verdicts pré-écrits appliqués tels quels (surface k\* : cellule-1-candidate →
  l'extension 16L₀ est une mission SÉPARÉE, gatée derrière ce résultat, non lancée ici ;
  étau W1 ; kx=1@1 % ; état du gate fovéa-z). Entrée de journal + point d'arrêt final.
