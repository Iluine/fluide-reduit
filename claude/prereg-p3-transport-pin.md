# PRÉ-ENREGISTREMENT P3 — transport du pin à travers R1 (gravé AVANT toute mesure)

> **Statut : ENDOSSÉ (Romain, 2026-07-25) — gravé §A38.** Les pré-requis de
> build passent par `mission-prerequis-p2p3.md` ; **aucune session humaine avant
> pré-requis verts, P2 couru, et décision explicite de Romain.** Gate P3 de
> l'arc projection (§A36) : le falsifieur de l'atténuation D14, celui qui donne
> son échelle au gate (iii′). Sources : §A37 (R1 verrouillé c5ac8757…), §C0–§C5
> (harnais ABX, escalier, catch trials), §C11–§C12 (le pin), spec P1 §4.
> Verdict-grade : **iluin-tworings3, terminal natif, session humaine (Romain,
> n=1)** — la seule machine ET le seul sujet du pin d'origine.

## La question, en une ligne

**Le pin jnd_sev = 7.33 % IC [6.03, 8.67]** `[GRAVÉ]`, mesuré à travers le chemin
viridis (pseudo-couleur, fait de code §A37), **se transporte-t-il à travers R1**
(luminance sRGB, ordre d'étages amendé) ? Les deux branches changent des
décisions nommées (§A35/D15, seance-fidelite §4) ; aucune n'est souhaitée.

## Pré-requis de build, gatés (AVANT la session, cap 0.5 séance, mission dédiée)

1. **Garde d'acuité BLOQUANTE** (recommandation session critique, endossée
   Romain 25/07) : sous `--sujet humain`, si `observation_cellule_pic_csf` rend
   cellule ≥ seuil d'acuité à la géométrie de session, la session **REFUSE de
   démarrer** (RuntimeError d'aiguillage, tradition `arcC_backend`) — un run
   au-dessus du plafond serait un instrument invalide, pas une donnée.
2. **Liaison sidecar↔log** (recommandation endossée) : le sidecar de conditions
   porte le **sha256 du log JSONL en fin de session** (écrit à la clôture ;
   `arcC_abx.py` INTOUCHÉ — c'est un post-traitement de la coquille). La lecture
   versionnée du verdict CITE ce sha : plus d'appariement par nom seul.
3. **Pré-calcul des stimuli si la cellule 1 de P2 l'exige** (lecture pré-écrite
   de P2) — décision de protocole prise avant la session, jamais pendant.

## Protocole (le harnais gravé, un seul changement à la fois)

- **Identique à la campagne du pin** : mêmes 20 sources (seed, L) de manche 1,
  même axe `stim(t)` en espace sédiment, même escalier 2-down-1-up, mêmes catch
  trials et gardes de validité §C5, même méthode d'IC (mean±2SEM combiné),
  **régime SÉVÈRE seul** (c'est jnd_sev que T2 consomme ; le laxiste n'est pas
  re-mesuré). Calibration §C7 re-mesurée en début de session (règle §A37-3),
  valeurs au sidecar.
- **Le seul changement : `--rendu r1`.** Provenance complète exigée au sidecar
  (sha256 R1, ordre d'étages, calibration) — c'est elle qui rend un transport
  ≠ 1 attribuable.
- **BRAS TÉMOIN, même session (tradition §A31 — exigé)** : UNE staircase
  complète en `--rendu viridis`, entrelacée ou immédiatement adjacente, mêmes
  gardes. **Garde de validité pré-écrite** : si le témoin viridis sort de l'IC
  gravé [6.03, 8.67], la session entière est **INDÉTERMINÉE** (le sujet ou
  l'écran a dérivé depuis Arc C) — aucune lecture de transport n'est prononcée.
  Sans ce bras, un écart R1-vs-pin serait inattribuable (chemin ? dérive ?).

## Lectures pré-écrites (prononcées mécaniquement, aucune à l'œil)

Sous témoin valide, avec IC(R1) = IC de jnd_sev^R1 (même méthode) :

1. **IC(R1) ∩ [6.03, 8.67] ≠ ∅ ⇒ TRANSPORT COMPATIBLE AVEC 1.** Conséquences
   gravées d'avance : la condition « transport ≈ 1 » du caveat D14 est ÉTABLIE ;
   la SECONDE condition (« pondération plate ») reste **NON MESURÉE** — P3
   présente un stimulus fovéal (~2°), il ne dit RIEN de l'excentricité ; la
   scission D14 ne se referme donc PAS ici (elle attendrait `r_fovea`, gaté).
   Le gate (iii′) se mesure avec le pin gravé tel quel à travers R1 — le juge se
   durcit, le contrat ne s'affaiblit pas (§A36 P0-b, caveat).
2. **IC disjoints ⇒ TRANSPORT ≠ 1, prononcé, avec direction et facteur**
   T = jnd_sev^R1 / 7.33 (IC par les bornes). Conséquences gravées : le gate
   (iii′) reçoit SON échelle (jnd_sev^R1) ; `r_fovea` reçoit son consommateur
   (§A32-décomp) ; **AUCUN seuil d'état ne bouge** — 0.0733 et tous les seuils
   T2/F1 restent les seuils de l'INSTRUMENT Δχ-albedo, gravés pour leur config
   exacte ; jnd_sev^R1 est un pin NOUVEAU (le pin-projection), il ne remplace
   rien rétroactivement.
3. **Escalier non convergent, catch trials échoués, ou dispersion §C5 hors
   garde ⇒ INDÉTERMINÉ** — un résultat, consigné tel quel, jamais re-couru en
   relâchant une garde.

## Ce que P3 ne mesure PAS (refus explicites, contre la surclame d'avance)

- **Pas la pondération d'excentricité** (stimulus fovéal unique) — c'est le
  juge (iii′) lui-même, pas son étalon ; `r_fovea` reste gaté.
- **Pas le vivant fovéal** (les stimuli sont mono-niveau pleine résolution,
  Option B) — l'ABX de substitution attend le compositeur, post-P3.
- **Pas de re-mesure du pin viridis** (le témoin est une garde de validité, pas
  une re-mesure : une staircase, pas la campagne).
- **Pas le son, pas R2** (lambertien) — R1 seul, daté par son empreinte ; un
  rendu plus riche re-mesurerait SON transport (§A37).

## Livrables

Sidecar(s) + logs JSONL (bruts, conservés) ; lecture versionnée
`claude/lectures/p3_transport_pin.lecture.json` (témoin, branches, chiffres,
prononcé mécanique) ; POINT D'ARRÊT : le verdict remonte AVANT toute suite —
le choix du contrat (C-strat vs la fermeture de D14) et l'achat du compositeur
se reposent dessus. Aucun enchaînement.
