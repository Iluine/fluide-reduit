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
2. **Garde d'acuité BLOQUANTE** (`scripts/run_arcC_session.py`) : sous
   `--sujet humain`, si `observation_cellule_pic_csf` rend cellule ≥ seuil
   d'acuité à la géométrie de session ⇒ `RuntimeError` d'AIGUILLAGE (tradition
   `arcC_backend` : nommer le défaut, la cause, la correction exacte, acter
   qu'aucune donnée n'a été produite). Replay et synthétique : INCHANGÉS (rien
   ne s'affiche). Test : géométrie au-dessus du plafond ⇒ lève ; géométrie de
   session nominale ⇒ passe.
3. **Liaison sidecar↔log** (`scripts/run_arcC_session.py`, à la CLÔTURE de
   session) : sha256 du fichier JSONL final écrit dans le sidecar de conditions
   (clé `sha256_log`), `arcC_abx.py` INTOUCHÉ (post-traitement de coquille).
   Test : le sha du sidecar égale celui du log relu ; un log altéré d'un octet ⇒
   désaccord détectable.

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
