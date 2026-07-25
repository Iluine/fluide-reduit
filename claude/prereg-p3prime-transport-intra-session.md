# PRÉ-ENREGISTREMENT P3′ — transport du pin, mesure INTRA-SESSION (gravé AVANT toute mesure)

> **Statut : BROUILLON soumis à revue Romain — rien ne se construit, rien ne se
> mesure avant endossement et gravure (§A41 à venir).** Ce document ne contient
> AUCUNE commande, à dessein : l'outillage (chantier 8) vient après la revue, et
> la commande finale sera tapée le jour de session, jamais fournie pré-remplie.
> Sources : §A40 (P3 INDÉTERMINÉE, décision α), prereg P3 + corrections (tout ce
> qui n'est pas contredit ici reste gravé), §A38-CORRECTION-4 (méthode d'IC).

## La question, re-posée sur le fait de §A40

P3 comparait le bras R1 d'aujourd'hui à un pin mesuré il y a trois semaines ;
le témoin a établi que ce pont est ILLISIBLE (5.06 % hors IC par le bas —
l'apprentissage ne se désapprend pas). **P3′ mesure le transport là où il vit :
T = jnd_R1 / jnd_viridis, même sujet, même session, même état.** Ni la dérive
inter-jours ni la pratique ne peuvent plus casser cette comparaison — les deux
bras la portent ensemble. **Le pin gravé [6.03, 8.67] reste le référent du
HARNAIS (T2, seuils d'état) — INTOUCHÉ ; il n'est plus le référent du
transport.** Atout consigné : le plateau de pratique du sujet (5 sessions),
coûteux pour P3, JOUE POUR P3′ — un sujet stabilisé dérive moins en cours de
session.

## Protocole

- **SIX staircases, deux bras entrelacés, ordre GRAVÉ** — **D-P3′-1, à trancher
  en revue** : `[recommandation]` **V, R, R, V, V, R** (contre-balancement
  ABBA-AB : positions viridis {1,4,5}, R1 {2,3,6}, sommes 10/11 — la dérive
  linéaire intra-session, fatigue ou chauffe, se répartit presque également ;
  l'alternance stricte V,R,V,R,V,R donnerait 9/12, penchée). L'ordre effectif
  est vérifié par une garde, comme pour P3 : un ordre déplacé en silence n'est
  plus le protocole pré-enregistré.
- **Chaque bras : TROIS staircases COMPLÈTES exigées** (clause du bras complet,
  §A38-CORRECTION-4 — étendue aux DEUX bras : un bras amputé ne se lit pas,
  verdict INDÉTERMINÉE).
- Reste IDENTIQUE au gravé de P3 : mêmes 20 sources, même axe `stim(t)`, même
  escalier 2-down-1-up, catch trials et validité §C5 par staircase, géométrie
  pic-CSF, calibration §C7 re-mesurée en début de session, provenance complète
  par staircase (chemin, sha R1 pour le bras R1, sha=None véridique pour
  viridis), liaison sidecar↔log.
- **`base_seed = 20260705`** reconduit (graine canonique, indices 0..5 dérivés
  par le schéma déterministe existant). L'appariement aux staircases du pin
  perd son rôle de garde (le référent est intra-session) — il reste une
  propriété de re-dérivabilité, surfacée, jamais jugée.
- **Conditions de session** : lumière du jour stable, OSD 80 %, sujet frais,
  **JOUR NEUF** (gravé §A40 : rien ne se re-mesure le 26). Pauses libres entre
  staircases, consignées au manifeste (durées surfacées, jamais jugées). Coût
  nommé : six staircases ≈ 150–250 essais plus catch — plus long que la
  campagne du pin ; c'est le prix de porter le référent dans la session.

## Gardes (toutes évaluées, jamais court-circuitées — l'acquis de P3 reconduit)

1. **Anti-gabarit DURCIE** (la fuite de §A40 est le cas d'école) : la chaîne
   `--conditions` est REFUSÉE si elle contient des chevrons, des points de
   suspension (`…` ou `...`), ou AUCUNE heure (motif `\d{1,2}h` ou `\d:\d\d`) —
   une observation datée contient son heure. Absente vaut gabarit, inchangé.
2. Ordre des chemins == l'ordre gravé D-P3′-1.
3. Validité §C5 par staircase ; bras complets (3+3).
4. Liaison sidecar↔log (sha256) pour les six.
**Plus de garde de témoin trans-sessions** — c'est le changement de design :
le bras viridis du jour EST le référent. Sa valeur face au pin gravé est
**SURFACÉE en diagnostic de dérive** (documenter le chemin du sujet depuis
Arc C), jamais un critère.

## Lectures pré-écrites (prononcées mécaniquement, IC min/max par bras — la méthode du pin, §A38-CORRECTION-4 ; le mean±2SEM surfacé, jamais décideur)

Sous gardes vertes, avec IC_V = min/max des trois seuils viridis et IC_R = idem
R1 :

1. **IC_R ∩ IC_V ≠ ∅ ⇒ TRANSPORT COMPATIBLE AVEC 1.** Conséquences gravées :
   la condition « transport ≈ 1 » du caveat D14 est ÉTABLIE — dans sa version
   intra-session, la plus forte disponible ; **l'anti-surclame VOYAGE
   inchangé** : la pondération d'excentricité n'est PAS mesurée (stimulus
   fovéal ~2°), la scission D14 ne se referme pas ici ; le gate (iii′) mesure
   avec le pin d'instrument via R1 — le juge se durcit, le contrat ne
   s'affaiblit pas (§A36 P0-b).
2. **IC disjoints ⇒ TRANSPORT ≠ 1, prononcé, avec direction et facteur** :
   T = jnd_R1/jnd_V (points), IC de T par les bornes croisées
   [min_R/max_V, max_R/min_V]. Le gate (iii′) reçoit son échelle RELATIVE ;
   `r_fovea` reçoit son consommateur ; **AUCUN seuil d'état ne bouge** — le pin
   gravé reste le pin du harnais, jnd du jour et T sont des grandeurs NOUVELLES.
3. **Gardes échouées, bras amputé, catch/dispersion §C5 hors clous ⇒
   INDÉTERMINÉE** — un résultat, consigné, jamais re-couru en relâchant.

## Ce que P3′ ne mesure PAS (refus explicites, inchangés)

Pas la pondération d'excentricité (le juge (iii′) attend son échelle, pas
l'inverse) ; pas le vivant fovéal (stimuli mono-niveau, Option B — l'ABX de
substitution attend le compositeur) ; pas de re-pin (le pin gravé est intouché,
la dérive du sujet est SURFACÉE, jamais re-jugée) ; pas le son, pas R2 — R1
seul, daté par son empreinte c5ac8757….

## Ce que chaque branche change (D17, en toutes lettres)

| branche | décision consommatrice |
|---|---|
| compatible avec 1 | le caveat D14 (condition transport ÉTABLIE) ; le gate (iii′) mesurera avec le pin d'instrument via R1 |
| transport ≠ 1 | l'échelle relative du gate (iii′) ; le consommateur de `r_fovea` ; la spec du compositeur héritera de T |
| INDÉTERMINÉE | la décision suivante appartient à Romain — aucune lecture par défaut |

## Outillage (chantier 8 — GATÉ sur l'endossement de ce prereg, rien avant)

Orchestration : plan d'entrelacement gravé (l'ordre D-P3′-1), les six
staircases, manifeste avec ordre effectif. Lecture : mode P3′ (gardes durcies,
IC min/max par bras, ratio et branches ci-dessus, diagnostic de dérive
surfacé). Champs d'essai imposés : les deux pré-vols du 25/07 ET la session P3
du 26/07 doivent être REFUSÉS par la garde anti-gabarit durcie (les ellipses du
26 en sont le cas d'école — rôle posthume). Chaque branche de lecture prouvée
falsifiable sur manifestes synthétiques, comme au chantier 7.

**POINT D'ARRÊT : revue de ce prereg par Romain (D-P3′-1 comprise) ; sur
endossement — gravure §A41, chantier 8, puis la session, un jour neuf, lumière
du jour. Aucun enchaînement.**
