# ORDRE DE MISSION — pré-requis P2/P3 (endossés §A38 le 2026-07-25)

> **Pour : session Claude Code.** Développement possible sur la VM Cowork (la
> cellule 2 exige le GPU pour ses TESTS d'équivalence — la VM le voit ; le CHRONO
> verdict-grade n'est jamais à toi). **Sources qui font foi** :
> `claude/prereg-p2-chiffrage-rendu.md` et `claude/prereg-p3-transport-pin.md`
> (ENDOSSÉS — les protocoles, bandes et lectures y sont gravés, tu les
> IMPLÉMENTES, tu ne les interprètes pas), §A38, §A37.

## Trois chantiers — et rien d'autre

1. **`scripts/run_p2_chiffrage.py`** — le runner des deux cellules du prereg P2 :
   - cellule 1 : `rendu_r1` numpy, champ réel (h_s101, s_L10 — celui des images
     d'exemple), `taille_px` depuis les arguments de calibration §C7 ; médiane et
     p95 sur 300 appels après 30 de chauffe, chrono monotone hôte ;
   - cellule 2 : sonde CuPy des étages sRGB+quantification (Y = A) sur champ
     1920×1080 f32 — formules RECOPIÉES de `arcC_rendu.py` avec son sha en
     commentaire ; **l'ÉQUIVALENCE d'abord** : mêmes uint8 que le numpy sur un
     champ test, tolérance ZÉRO ; si écart, le runner s'arrête, imprime l'écart
     chiffré (combien de pixels, de combien) et rend verdict AUTRE — jamais une
     tolérance silencieuse ; puis médiane/p95 sur 300 appels après 30 de chauffe,
     `cp.cuda.Device().synchronize()` avant CHAQUE lecture d'horloge ;
   - les BANDES et les BRANCHES du prereg codées telles quelles, prononcé
     mécanique, JSON complet écrit sous `outputs/arcC/p2_chiffrage.json`
     (empreinte R1 recalculée dedans, machine, versions, date) — la lecture
     versionnée se recopie APRÈS le run de Romain, pas par toi.
2. **Garde de COMPARABILITÉ, bloquante** (`scripts/run_arcC_session.py`) —
   **ré-écrite après ta propre remontée (§A38-CORRECTION) : l'ancienne garde
   d'acuité était falsifiée, ppd s'annule à pic-CSF** : sous `--sujet humain`,
   si la géométrie de session ≠ `pic-csf` (celle de la campagne du pin,
   manifeste du 2026-07-05) ⇒ `RuntimeError` d'AIGUILLAGE (tradition
   `arcC_backend` : nommer le défaut, la CAUSE — la comparabilité à l'IC gravé
   exige les conditions du pin —, la correction exacte, acter qu'aucune donnée
   n'a été produite). Le report §C7 (`observation_cellule_pic_csf`) reste un
   CHIFFRE SURFACÉ au sidecar — jamais un booléen (pièce 3, refus délibéré,
   PRÉSERVÉ). Replay et synthétique : INCHANGÉS (rien ne s'affiche). Tests :
   `--geometrie plafond --sujet humain` ⇒ lève ; `--geometrie pic-csf
   --sujet humain` ⇒ passe (la campagne du pin serait passée, par identité) ;
   synthétique/replay à toute géométrie ⇒ inchangés.
3. **Liaison sidecar↔log** (`scripts/run_arcC_session.py`, à la CLÔTURE de
   session) : sha256 du fichier JSONL final écrit dans le sidecar de conditions
   (clé `sha256_log`), `arcC_abx.py` INTOUCHÉ (post-traitement de coquille).
   Test : le sha du sidecar égale celui du log relu ; un log altéré d'un octet ⇒
   désaccord détectable.

4. **[AJOUTÉ — §A38-CORRECTION-2] Critère de NATURE pour l'équivalence de la
   cellule 2** (`scripts/run_p2_chiffrage.py`) : remplace la tolérance zéro,
   falsifiée structurellement par ta propre remontée. Le runner prononce
   mécaniquement, dans cet ordre : (1) équivalence structurelle — rendue
   MÉCANIQUE par toi (sha vérifié à l'exécution), confirmée telle quelle ;
   (2)+(3) **FUSIONNÉES (§A38-CORRECTION-3 — ton ZÉRO d'isolation falsifié à
   son tour, décision Romain)** : critère de nature UNIFIÉ appliqué à CHAQUE
   bras — isolation (f64, entrée castée f32) ET f32 complet : tout désaccord =
   bascule pure, |Δniveau| == 1, sinon AUTRE ; l'attribution par cause redevient
   un DIAGNOSTIC — comptages surfacés au JSON par champ et par bras, jamais
   jugés ; (4) taux et `distance_max_a_la_bascule` surfacés, JAMAIS jugés —
   inchangé. Ton falsifieur (sonde à 2 niveaux) et l'agrégation sans moyenne :
   confirmés, à étendre au bras d'isolation.
   **Champs test GRAVÉS** : rampe `linspace(0,1)` + 3 uniformes seedés (tes
   seeds existants) ; le champ réel NE COMPTE PAS pour l'équivalence
   (≤ 4096 valeurs distinctes — consigné insuffisant) et reste le champ de la
   cellule 1. La règle des zones (branche 1 ≤ 0.5 ms ; branche 2 ≥ 2.199 ms ;
   entre : rien, ratio surfacé) est désormais GRAVÉE au prereg — ton
   implémentation du point n°1 est confirmée telle quelle. Tests à mettre à
   jour en conséquence ; toujours AUCUN chrono verdict-grade par toi.

5. **[AJOUTÉ — §A39-CORRECTION] Cellule 2b : le noyau FUSIONNÉ promu au
   runner** (`scripts/run_p2_chiffrage.py`) : ton kernel fusionné (un seul
   kernel, constantes en paramètres) devient l'objet d'une cellule 2b —
   MÊME bande [0.02, 0.5], MÊME critère de nature unifié via `compare_bras`,
   MÊMES zones, prononcé mécanique, dans le même JSON (clé `cellule_2b`,
   la 2a reste intacte avec son AUTRE). Tests : équivalence des quatre champs,
   falsifieur (sonde décalée ⇒ AUTRE), bandes verrouillées étendues à 2b.
   Le script de scratchpad meurt ; sa substance vit ici.
6. **[AJOUTÉ — prereg P3, précisions gravées] Support de session P3 dans
   l'orchestration** (`scripts/run_arcC_orchestration.py`) :
   - `--regime {severe,laxiste,tous}`, défaut `tous` (comportement historique
     intact) — P3 utilisera `severe` ;
   - `--temoin-viridis` : valide UNIQUEMENT avec `--sujet humain --rendu r1` ;
     insère UNE staircase témoin en chemin viridis **en 3e position sur 4**
     (R1, R1, témoin, R1 — position gravée au prereg P3), avec son propre
     sidecar (provenance viridis : sha=None, chaîne historique — la fonction
     existante fait déjà foi), seeds dérivées du même schéma déterministe ;
   - le manifeste consigne l'ordre effectif des staircases et leurs chemins.
   Tests : ordre des staircases vérifié ; `--temoin-viridis` sans humain/r1 ⇒
   erreur d'argparse ; sidecars distincts et liés à leurs logs (chantier 3).

## Garde-fous (ils sont le contrat)

- **`src/arcC_rendu.py` INTOUCHÉ** — l'empreinte c5ac8757… doit rester verte ;
  si un chantier semble l'exiger, ARRÊT et remontée.
- **INTOUCHÉS aussi** : `src/albedo.py`, `src/arcC_calibration.py`,
  `src/arcC_abx.py`, `src/arcC_stimuli.py`, tout `src/f1_gpu/`, tout seuil,
  toute bande (celles du prereg se COPIENT, ne s'ajustent pas).
- **AUCUNE MESURE VERDICT-GRADE** : tu vérifies que le runner TOURNE (un appel
  court suffit) ; les 300 appels officiels sont à Romain, terminal natif. La VM
  tue à 45 s — dimensionne tes vérifications en conséquence.
- **Cap 0.5 séance pour la cellule 2** (gravé au prereg) ; si le chiffrage
  total dépasse 1 séance, remonter avant de continuer.
- Lint : `pocCascade2phys/.venv/bin/ruff check --line-length 100 --select E,F,W`
  (le binaire N'EST PAS dans pocPhysicator/.venv — fait corrigé §A38).
- Commits : `git -c user.name="Claude (Cascade)" -c user.email="noreply@anthropic.com"
  commit` ; **push : Romain uniquement**.

## POINT D'ARRÊT — la mission s'arrête là

Tests verts (suite complète sur iluin-tworings3 avant remontée) ⇒ REMONTER avec :
la sortie de la suite, la confirmation d'empreinte R1, et le résultat de
l'équivalence cellule 2. **Le chrono P2 est LANCÉ PAR ROMAIN** ; la session
humaine P3 vient après P2 et sur sa décision explicite. Aucun enchaînement.
