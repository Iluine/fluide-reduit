# Séance papier post-cascade — emboîtement, c-dégressif, candidat V4, M-a-quater — ENDOSSÉ

> **Statut : ENDOSSÉ (Romain, 2026-07-19) — P1-P5 tranchés tels que recommandés.
> Version faisant foi : pocCascade2phys PREREGISTRATION.md §A18 + SPEC-FOVEA-Z
> §2-rev1. Ce fichier est l'archive de travail.** Entrées : §A17 complet (cascade close,
> 7779f29). Ancres mesurées utilisées partout : slot 2-sys batché **1.685 ms**,
> slot 1-sys batché **0.845 ms**, transferts diff **0.61 ms** à 12 slots,
> remontée CuPy non optimisée **8.523 ms**, prédiction mesurée **[2.570, 2.819]**
> selon chemin (ancre du ré-emboîtement AMBIGUË — traitée en bande à deux
> modèles, §4). Après endossement : gravure journal + addenda spec §2-rev.

## §1 — L'EMBOÎTEMENT, gravé comme propriété de structure (résout le trou n°1)

**Propriété E (à graver en §2 de la spec) :** l'ensemble actif est un ARBRE —
toute fenêtre active au niveau j ≥ 1 déclare une fenêtre parente au niveau j−1
qui couvre intégralement son empreinte. Fait géométrique utile : l'empreinte
d'une fenêtre n×n fine = (n/2)² au parent = **un quart de fenêtre parente** —
une parente peut couvrir jusqu'à 4 enfants disjoints (2×2).

**Règle de placement (remplace les offsets B3 ±n_fov, qui étaient du harnais) :**
une fenêtre d'énergie au niveau j ne peut naître QUE dans l'union des fenêtres
actives du niveau j−1. Si l'événement est hors couverture, le routeur monte une
TOUR d'ancêtres — comptée dans le cap de slots (le budget d'emplacements devient
un budget d'ARBRE, pas de liste plate). Conséquence chiffrée : grâce au
quart-d'empreinte, 2 fenêtres d'énergie fines adjacentes à la fovéale partagent
la couverture parente EXISTANTE — coût marginal nul si placées dans le cône,
1 slot/2 niveaux d'écart si hors cône. `[MESURABLE à M-a-quater]`

## §2 — c-dégressif vs descente : les champs d'échelle (résout le trou n°2)

Deux options honnêtes :
- **(a) Init locale nommée** : les champs absents du parent reçoivent, en
  colonnes entrantes, un opérateur local (zéro-information du dehors, motif
  §A14 (v)). `[NON-ANCRÉ perceptuel, falsificateur à nommer]`
- **(c) CHAMPS D'ÉCHELLE (recommandée)** : les champs au-delà de c_parent sont
  DÉFINIS à contenu grossier nul — nés de l'échelle fine, comme les coefficients
  de détail le sont chez Harten. Leur prédiction parent→enfant est le
  remplissage-zéro, EXACT PAR DÉFINITION (pas une approximation). Contrat
  induit pour le substrat-jeu : un champ qui a du contenu grossier doit exister
  À TOUS les niveaux ; un champ fin-seulement doit être structurellement
  d'échelle. C'est une contrainte de design DU SUBSTRAT, gravée en spec — elle
  élimine le trou au lieu de le boucher.

## §3 — Candidat V4 (re-épinglé sur ancres)

**Composition** : fovéale 9 slots (2 fins c=8 + 7 c=4) + **2 slots énergie c=8**
(V2 en avait 3 — un slot rendu au budget), placement EMBOÎTÉ (§1), champs
d'échelle (§2c). Scénario R2 inchangé (20°, n=512, J=9, monde 500k, 3050 Ti).

**Arithmétique (ancres mesurées)** :
- F batché = 2×1.685 + 7×0.845 + 2×1.685 = **12.655 ms** `[ancres MESURÉES]` ;
- Prédiction (tout GPU-side sauf niveau 1, emboîtement rendant les 5 replis s3
  caducs) : **bande à deux modèles** — effondrement des stalls (~0.3–0.6) vs
  résidu linéaire (~1.5–2.0). `[AMBIGUÏTÉ MESURÉE, M-a-quater tranche]` ;
- Remontée : cible = 16.7 − 12.655 − prédiction ∈ **[2.0, 3.7 ms]** — à financer
  par **L3** (émission des détails par le kernel : supprime les passes CuPy ;
  reste compaction + transfert ~0.6 mesuré + gather) ; **L1 en RÉSERVE
  pré-enregistrée** (cadence k=2 si L3 seul ne suffit pas — k choisi par budget,
  étiqueté [NON-ANCRÉ perceptuel], réveil σ_ω DIFFÉRÉ conformément à R4).

**Budget : 16.7 ms INCHANGÉ (T2).** La porte 33.3 reste le REPLI PRÉ-NOMMÉ si
M-a-quater meurt : elle se prendrait avec le motif T2 traité en face — physique
à 30 Hz + rendu 60 fps par interpolation du readout (les deux cadences
découplées), le coût GPU du rendu cohabitant dans les ~13 ms restants par
fenêtre de 33.3. `[Design nommé, non décidé, non gratuit]`

## §4 — M-a-quater : pré-enregistrement (chiffres figés ici)

- **Protocole** : pipeline COMPLET du candidat V4 — F batché + prédiction
  GPU-side emboîtée + remontée L3 (kernel-moteur émettant |d| ≥ EPS_DETAIL) +
  transferts. Fovéa mobile E4c, chrono B6, vérifs reconduites, natif.
- **MORT : médiane frame complète > 16.7 ms** ⇒ candidat V4 mort ⇒ le repli
  pré-nommé (porte 33.3, motif T2 traité) devient LA décision à prendre —
  jamais un enchaînement.
- **Bande de modèle (deux modèles, honnête)** : frame prédite ∈
  [12.655+0.3+1.0, 12.655+2.0+2.5] = **[13.9, 17.2 ms]** — la bande CHEVAUCHE
  le seuil : le papier ne promet pas le PASS, il le rend mesurable. Hors bande
  = AUTRE remonté. Diagnostics reportés : parts F / prédiction / remontée /
  transferts (attribution mécanique, série de sondes NON rouverte).
- **Gates de build (avant tout code)** : chiffrage Claude Code remonté —
  notamment le KERNEL-MOteur L3 : kernel NEUF avec son propre pré-enregistrement
  (équivalence de motif au fusionné + émission), la borne M-a′ restant intacte
  et citée telle quelle ; géométrie emboîtée ; champs d'échelle. Si chiffrage
  > 2 séances : remonter avant achat.
- **EPS_DETAIL** : reconduit à 1e-4, TOUJOURS [NON-ANCRÉ perceptuel], suspect
  nommé hérité par tranche-2 (consigne 1 inchangée).

## Décisions Romain — TRANCHÉES (2026-07-19)

**P1** propriété E + règle de placement GRAVÉES (§2-rev1) ; **P2** = (c) champs
d'échelle ; **P3** = candidat V4 endossé (16.7, porte 33.3 repli pré-nommé avec
motif T2 traité) ; **P4** = L1 réserve pré-enregistrée (k par budget,
[NON-ANCRÉ], σ_ω différé R4) ; **P5** = M-a-quater endossé (protocole, bande
[13.9, 17.2], gates de build — chiffrage kernel-moteur L3 avant achat).
