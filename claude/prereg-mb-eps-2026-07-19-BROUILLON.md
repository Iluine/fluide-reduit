# Pré-enregistrement — sonde EPS (N1) + M-b re-scopée V4 (N2) — ENDOSSÉ

> **Statut : ENDOSSÉ (Romain, 2026-07-19) — Q1 = N1 puis N2 ; Q2 = plus grand
> EPS sous jnd_sev sur l'IC ENTIER (Δχ max < 0.0603) ; Q3 = contrôle T1 devient
> CRITÈRE PRÉ-ÉCRIT. Version faisant foi : pocCascade2phys §A19.** Entrées : §A18-lecture-M-a-quater (5c4aeee — V4 AU
> SEUIL, transferts 2.727, p99 19.652, EPS_DETAIL suspect depuis
> §A15-complément consigne 1) ; §A15 (M-b, seuils T3 gravés) ; §A16/§A18 (M-b
> re-scopée à l'enveloppe re-épinglée). Gate (iii) de la spec = M-b sans mort.
> **CONCESSION consignée (session critique)** : la formule « balayage EPS gratuit
> dans le run M-b » était fausse sur l'OBSERVABLE — EPS gouverne la fidélité du
> niveau 0 vivant, pas le contrat live↔rederive ; il n'atteint le Δχ de M-b
> qu'au second ordre (colonnes entrantes). Observable corrigé en §3.

## §1 — Séquencement proposé : N1 (bon marché) avant N2 (cher)

- **N1 — sonde EPS** : ~0.5–1 séance (le harnais existe : pyramide emboîtée,
  kernel L3, readout, pin). Consommateurs : budget transferts (2.727 ms), p99
  bursts (19.652), et le suspect §A15 cesse d'être réglé par budget.
- **N2 — M-b re-scopée V4** : 3–5 séances, risque ÉLEVÉ (chiffrage tranche-1
  reconduit) — le portage fidèle de `run_episode` en GPU f32. C'est le plus
  gros achat du projet ; il porte le gate (iii).
« Le moins cher qui peut échouer, d'abord » : N1 peut changer l'enveloppe dans
laquelle N2 sera mesurée ; l'inverse n'est pas vrai.
**F0-cloud (gate ii, ~2 min) reste dû, indépendant — à la prochaine session cloud.**

## §2 — N2 : M-b re-scopée V4 (protocole ; SEUILS INCHANGÉS)

- **Chemin vivant** = pipeline V4 COMPLET tel que mesuré à M-a-quater
  (géométrie emboîtée, prédiction GPU-side, L3, L1 k=4 étalée, fovéa mobile),
  avec le **F FIDÈLE** (portage de `run_episode` : wetdry O2 CFL-adaptatif,
  Exner, pulses, bathymétrie réelle) — pas le jetable coût-représentatif.
- **Chemin rederive** = f(registre, seeds), CPU f64, INTOUCHÉ (le gel bit-exact
  reste le contrôle).
- **Cellule (gravée §A15, reconduite telle quelle)** : 3 seeds {101, 102, 103},
  Δt = 4, 6 émissions. Commits fenêtrés, k_fen aire-proportionnel, cap 10 %.
- **MORT-b (T3, gravé, NON déplacé)** : max de série Δχ live↔rederive
  **> 0.0733** (pin sévère), **PAR-SEED** ⇒ Option A morte, repli Option B
  (tout-déterministe) = décision neuve, coût de frame non mesuré ici.
- **CONTRÔLE DE REPRÉSENTATIVITÉ T1 (consigne gravée à la scission)** : la
  frame-time du F FIDÈLE est re-mesurée dans le même run et comparée aux
  12.655 ms du proxy jetable. **Point à trancher (Q3)** : ce contrôle était
  gravé NON-VERDICTAL ; V4 tenant le budget à 0.7 %, un F fidèle plus cher de
  >0.7 % suffirait à faire mourir V4 — le contrôle est devenu load-bearing.
- **Gate de chiffrage** : chiffrage du portage remonté AVANT achat (règle
  maison) ; scission interne possible si > 3 séances.

## §3 — N1 : sonde EPS — l'observable CORRIGÉ

**Ce qu'EPS_DETAIL gouverne réellement** : la fidélité du niveau 0 vivant
(alimenté par la remontée seuillée). Ses consommateurs perceptuels : le readout
du lointain, et les colonnes entrantes prédites au déplacement.

- **Vérité** : niveau 0 par décimation exacte (moyennes Harten) de l'état fin
  courant — le grossier parfait.
- **Vivant** : niveau 0 reconstruit par la remontée seuillée à EPS,
  incrémentale, **L1 k=4 étalée incluse** (la péremption d'amortissement fait
  partie du régime de production — honnête).
- **Observable** : Δχ readout (albedo + delta_chi, max_carrier) entre les deux —
  espace instrument, jamais l'état. Fovéa mobile E4c, série 300 frames.
- **Balayage** : EPS ∈ {1e-5, 1e-4 (gravé), 3e-4, 1e-3, 3e-3, 1e-2}.
- **Reporté par EPS** : Δχ max et médian ; octets/frame ; temps de transfert ;
  médiane et p99 frame (le lien avec les bursts).
- **RÈGLE DE DÉCISION à pré-enregistrer (Q2)** — sans elle, on choisirait EPS
  après avoir vu la courbe, c'est-à-dire on fabriquerait le verdict.

**EXCLUSION NOMMÉE — le balayage de k.** Balayer k perceptuellement reviendrait
à décider le CADENCEMENT avec la question perceptuelle en main : c'est
exactement la **condition de réveil de σ_ω (R4)**. k reste FIGÉ à 4
`[NON-ANCRÉ, choisi par budget]` dans cette sonde. Si le budget exige un jour de
bouger k, le réveil du fork est une décision explicite de Romain — jamais un
effet de bord de sonde.

## §4 — Portées (ce que ces mesures ne diront pas)

Rien sur : r_fovea / excentricité (toujours gaté) ; 3D ; multi-vue (v1.1) ;
niveau 0 CPU à 500k cellules (pied n°1, toujours non mesuré) ; qualité produit
(pin n=1) ; coût de l'Option B si M-b meurt (nouveau pré-enregistrement).

## Décisions Romain — TRANCHÉES (2026-07-19)

- **Q1 = N1 → N2** : sonde EPS d'abord, M-b ensuite (chiffrage avant achat).
- **Q2 = règle EPS pré-enregistrée** : EPS retenu = **le plus grand EPS dont le
  Δχ max de série reste < 0.0603 (ic_bas)** — lecture sur l'IC ENTIER, discipline
  §A13 reconduite. Si aucun EPS du balayage ne satisfait (y compris 1e-5) :
  AUTRE remonté, aucun EPS retenu par défaut.
- **Q3 = contrôle T1 VERDICTAL** : la frame complète mesurée avec le F FIDÈLE
  > 16.7 ms ⇒ **V4 MORT** (critère pré-écrit avant tout chiffre ; le laisser
  non-verdictal aurait permis de constater la mort sans la prononcer).
