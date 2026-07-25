# PRÉ-ENREGISTREMENT P2 — chiffrage du noyau de rendu (gravé AVANT toute mesure)

> **Statut : ENDOSSÉ (Romain, 2026-07-25) — gravé §A38.** Le build de la
> cellule 2 et le runner passent par l'ordre de mission `mission-prerequis-p2p3.md` ;
> **les cellules sont LANCÉES PAR ROMAIN, verdict-grade natif.**
> Gate P2 de l'arc projection (§A36) : « première
> ancre de la moitié manquante du budget V4 ». Sources : spec P1 endossée (§A37,
> bloc REVUE + correction d'étages), remontée mission P1 (commit 2bbcca2, R1
> verrouillé c5ac8757…). Règle D17 : chaque cellule nomme la décision qui la
> consomme. Verdict-grade : **iluin-tworings3, terminal natif**.

## Portée, dite d'abord — ce que P2 peut et ne peut pas ancrer

Par la décision D-P1-1 (Option B), **la COMPOSITION est exclue** : le
rééchantillonnage à l'échelle écran appartient au compositeur (le gather
pyramide→écran), différé post-P3. P2 chiffre donc le NOYAU (encodage sRGB +
quantification, Y = A) et l'instrument R1 tel qu'il servira en session. **P2 est
la première ancre de la moitié manquante, PAS la moitié entière** — quiconque lira
« le rendu coûte X » surclamera ; P2 dira « le noyau coûte X ».

## Cellule 1 — R1 tel quel (instrument de session)

- **Protocole** : `rendu_r1(a, taille_px)` numpy, champ albédo réel (h_s101,
  s_L10 — celui des images d'exemple), `taille_px` de la géométrie de session
  courante ; médiane et p95 sur **300 appels après 30 de chauffe**, chrono
  monotone hôte. Aucun GPU.
- **Bande pré-écrite (deux modèles, hors bande = AUTRE, remonté tel quel)** :
  médiane ∈ **[0.05, 5] ms/appel** (borne basse : flops purs ~10⁵ ; borne haute :
  overhead numpy sur petites matrices ×50).
- **Décision consommatrice (D17)** : l'intégrité des TIMINGS de présentation du
  régime sévère (§C0 — séquentiel, masque, durées). Lecture pré-écrite :
  médiane ≤ 5 ms ⇒ le rendu est invisible dans les timings de session P3 ;
  au-delà ⇒ le protocole P3 doit PRÉ-CALCULER les stimuli rendus (décision de
  protocole, prise AVANT la session humaine, jamais pendant).

## Cellule 2 — le noyau à l'échelle V4 (l'ancre budget)

- **Objet** : étages encodage sRGB (formule de R1, f32) + quantification uint8,
  Y = A, sur champ **1920×1080** (~2.07·10⁶ px), GPU (CuPy, 3050 Ti). PAS de
  rééchantillonnage (composition exclue, dit ci-dessus).
- **Build gaté** : sonde CuPy élémentaire (~30 lignes, formules RECOPIÉES de
  `arcC_rendu.py` avec référence à son sha), **cap 0.5 séance** — chiffrage
  Claude Code au-delà ⇒ remonter avant achat. La sonde vérifie d'abord
  l'ÉQUIVALENCE au numpy (mêmes uint8 sur un champ test, tolérance zéro —
  l'arrondi f32/f64 de la puissance 1/2.4 peut casser le bit-exact : si écart,
  le REMONTER chiffré, ne pas le tolérer en silence).
- **Protocole** : médiane et p95 sur 300 appels après 30 de chauffe,
  synchronisation device avant chaque lecture d'horloge (tradition B6).
- **Bande pré-écrite (deux modèles)** : médiane ∈ **[0.02, 0.5] ms** (modèle
  compute ~0.01 ms ; modèle bande passante ~10 Mo à 224 Go/s ≈ 0.05 ms ; marge
  lancements ×10). Hors bande = AUTRE — l'histoire du projet (MORT-a : ×42
  papier→mesure) est la raison d'être de cette cellule.
- **Décision consommatrice (D17)** : la ligne de budget « ~16.7 ms pour un rendu
  jamais chiffré » (état des lieux §4). Lectures pré-écrites, les DEUX branches :
  1. médiane ≪ marge V4 (2.199 ms) ⇒ **lecture gravée d'avance : le coût inconnu
     du rendu ne vit PAS dans l'encodage — il vit dans la COMPOSITION et dans
     l'optique au-delà de Y = A (R2+). La dette se DÉPLACE, elle ne rétrécit
     pas.** Interdiction pré-écrite d'en conclure « le rendu tient ».
  2. médiane comparable ou supérieure à la marge ⇒ le repli 33.3 ms (§A18, porte
     pré-nommée) passe de « nommé » à « à instruire » — décision neuve, jamais un
     enchaînement.

## Ce que P2 ne mesure PAS (refus explicites)

Pas de composition (D-P1-1) ; pas de rendu fovéé ; pas de session humaine ; pas
de Δχ ; pas de re-chrono T1 (dette datée 6dd207ca, réveil non atteint — P2 ne
consomme pas le coût du fidèle) ; aucune modification de R1 (empreinte
c5ac8757… doit rester verte).

## Livrables

Lecture versionnée `claude/lectures/p2_chiffrage_rendu.lecture.json` (cellules,
bandes, chiffres, branches prononcées mécaniquement) ; POINT D'ARRÊT après les
deux cellules — remonter, aucune suite sans décision.
