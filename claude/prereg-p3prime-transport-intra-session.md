# PRÉ-ENREGISTREMENT P3′ — transport du pin, mesure INTRA-SESSION (gravé AVANT toute mesure)

> **Statut : ENDOSSÉ (Romain, 2026-07-29) — version faisant foi : §A42,
> complété §A42-COMPLÉMENT (`[v2.2]` : partition 1b/2/2-équiv sur la grille
> inclusion × recouvrement — résolution β, α refusée ; portée de la liste des
> graines : entrée seule, lecture par égalité, archive re-jouable).**
> La revue critique de la v2 (2026-07-29, soir) a rendu : ENDOSSABLE avec deux
> corrections nommées, gravées ici en `[v2.1]` À L'ENDOSSEMENT, avant toute
> construction : (1) plage horaire de session — la version OUTILLÉE de
> « lumière du jour », qui n'était qu'un surclame en v2 (la garde (b) vérifie
> l'honnêteté de l'heure, pas la lumière : une session nocturne VÉRIDIQUE
> passait) ; (2) graines BRÛLÉES — une graine humaine ne sert qu'une fois, la
> garde refuse la liste, pas une valeur nominative. **D-P3′-1 et D-P3′-2 :
> TRANCHÉS (§A42), voir en place.** Chantier 8 DÉBLOQUÉ — sur ordre de mission
> explicite, aucun enchaînement.
> Historique : la v1 a été REFUSÉE par la review en profondeur du 2026-07-29
> (deux bloquants : B1a graine = feuille de réponses, B1b branche 1 promouvant
> un non-rejet en « ÉTABLIE » ; plus M2/M3/M4/M5) ; la v2 intègre ces
> corrections, marquées `[v2]`. Ce document ne contient AUCUNE commande, à
> dessein : la commande finale sera tapée le jour de session, jamais fournie
> pré-remplie.
> Sources : §A40 (P3 INDÉTERMINÉE ; décision α/β/γ laissée OUVERTE — le chemin α
> est celui que CE prereg proposait, endossé en §A42), prereg P3 + corrections
> (tout ce qui n'est pas contredit ici reste gravé), §A38-CORRECTION-4 (méthode
> d'IC), §A41 + §A41-ERRATUM (faits de conditions et d'antidatage ; règle « la
> machine lit, la session ne recopie pas »).

## La question, re-posée sur le fait de §A40

P3 comparait le bras R1 d'aujourd'hui à un pin mesuré il y a trois semaines ;
le témoin a rendu ce pont ILLISIBLE (5.06 % hors IC par le bas). `[v2]`
Précision due (M1) : ce témoin exigeait qu'un tirage nouveau tombe dans le
min/max de 3 tirages antérieurs — pour 4 variables échangeables, il échoue
avec probabilité exactement 1/2 SOUS HYPOTHÈSE NULLE. L'INDÉTERMINÉE du 26/07
était donc largement pré-déterminée par construction ; « l'apprentissage des
pré-vols » reste la cause la plus parcimonieuse mais est SOUS-DÉTERMINÉE par
ce test. Le passage à l'intra-session est la bonne réponse pour DEUX raisons :
la dérive du sujet, et la puissance nulle du pont trans-sessions lui-même. **P3′ mesure le transport là où il vit :
T = jnd_R1 / jnd_viridis, même sujet, même session, même état.** Ni la dérive
inter-jours ni la pratique ne peuvent plus casser cette comparaison — les deux
bras la portent ensemble. **Le pin gravé [6.03, 8.67] reste le référent du
HARNAIS (T2, seuils d'état) — INTOUCHÉ ; il n'est plus le référent du
transport.** Atout consigné : le plateau de pratique du sujet (5 sessions),
coûteux pour P3, JOUE POUR P3′ — un sujet stabilisé dérive moins en cours de
session.

## Protocole

- **SIX staircases, deux bras entrelacés, ordre GRAVÉ** — **D-P3′-1 : TRANCHÉ
  (§A42)** : **V, R, R, V, V, R** (contre-balancement ABBA-AB : positions
  viridis {1,4,5}, R1 {2,3,6}, sommes 10/11 — la dérive linéaire
  intra-session, fatigue ou chauffe, se répartit presque également ;
  l'alternance stricte V,R,V,R,V,R donnerait 9/12, penchée). `[v2.1]` Raison
  du choix, adverse et nommée : après échauffement, le résidu dominant est la
  FATIGUE de fin de session, qui pèse sur le bras de plus grand Σk² — R
  ({2,3,6}, Σk² = 49 contre V = 42) — donc gonfle jnd_R, donc biaise T VERS 1,
  CONTRE la branche 2 anticipée (T ≈ 0.5, non-cécité ci-dessous). **Le miroir
  R,V,V,R,R,V est REFUSÉ explicitement** : il biaiserait dans le sens
  confortable. L'ordre effectif est vérifié par une garde, comme pour P3 : un
  ordre déplacé en silence n'est plus le protocole pré-enregistré.
- **Chaque bras : TROIS staircases COMPLÈTES exigées** (clause du bras complet,
  §A38-CORRECTION-4 — étendue aux DEUX bras : un bras amputé ne se lit pas,
  verdict INDÉTERMINÉE).
- Reste IDENTIQUE au gravé de P3 : mêmes 20 sources, même axe `stim(t)`, même
  escalier 2-down-1-up, catch trials et validité §C5 par staircase, géométrie
  pic-CSF, calibration §C7 re-mesurée en début de session, provenance complète
  par staircase (chemin, sha R1 pour le bras R1, sha=None véridique pour
  viridis), liaison sidecar↔log.
- **`[v2]` `base_seed = 20260729`, NEUF, gravé ici** (recopié au manifeste,
  vérifié par une garde ; indices 0..5 dérivés par le schéma déterministe
  existant — le mécanisme `derive_seeds` est sain, c'est la graine qui ne
  l'était pas). **La reconduction de `20260705` est REFUSÉE (B1a)** : dans
  `run_escalier` (`src/arcC_abx.py:493-494`), l'ordre des sources ET le côté
  correct A/B de chaque essai sont tirés de `rng_roving` dans un ordre fixe,
  indépendamment des réponses du sujet — la graine encode la FEUILLE DE
  RÉPONSES, que le sujet a déjà rejouée quatre fois (pin, deux pré-vols, P3).
  La re-dérivabilité est préservée par CONSIGNATION de la graine neuve, pas
  par sa reconduction. `[v2.1]` **GRAINES BRÛLÉES, le principe remplace la
  valeur** : une graine présentée à un sujet humain ne sert qu'UNE fois. La
  garde refuse toute graine de la liste des graines brûlées — aujourd'hui
  {20260705} ; après la session P3′, 20260729 y ENTRE. Refuser `20260705`
  nominativement (la v2) laissait un futur P3″ reconduire `20260729` avec
  exactement la faille B1a. `[v2.2]` **PORTÉE DE LA LISTE (§A42-COMPLÉMENT) :
  la liste ne mord qu'à L'ENTRÉE** (empêcher la réutilisation sur un sujet) ;
  à la LECTURE, l'ÉGALITÉ au `base_seed` gravé suffit et pinne tout — une
  graine quelconque est refusée par l'égalité, `20260705` aussi, et l'archive
  P3′ reste RE-JOUABLE à l'identique après le brûlage de `20260729` (la
  re-dérivabilité promise plus haut ; sans cette portée, la garde double de
  lecture rendait l'archive illisible dès l'acte de gravure).
- **`[v2]` Bloc d'échauffement NON ANALYSÉ, gravé d'avance** : N = 15 essais
  ABX non scorés (niveau fixe supra-seuil, mêmes sources, aucun seuil produit),
  AVANT la staircase 1. Raison chiffrée : la position 1 est la plus contaminée
  par l'échauffement et D-P3′-1 la place dans le bras V (référent) — sans ce
  bloc, T serait biaisé vers le bas, vers la branche « transport ≠ 1 ». Les
  **seuils-par-position sont SURFACÉS** en diagnostic (jamais jugés) pour
  documenter la dérive intra-session résiduelle.
- **Conditions de session** : lumière du jour stable, OSD 80 %, sujet frais,
  **JOUR NEUF** (gravé §A40 : rien ne se re-mesure le 26). `[v2]` Ces
  conditions ne sont plus DÉCLARÉES mais VÉRIFIÉES : garde ancrée sur
  `date_session` (garde 1 ci-dessous) — la leçon de §A41 (la session P3 a
  couru à 00h27 sous une gravure « lumière du jour »). Pauses libres entre
  staircases, consignées au manifeste (durées surfacées, jamais jugées). Coût
  nommé : six staircases ≈ 150–250 essais plus catch et échauffement — plus
  long que la campagne du pin ; c'est le prix de porter le référent dans la
  session.
- **`[v2]` Coût INDÉTERMINÉE chiffré d'avance** : avec la clause « un catch
  raté invalide sa staircase » (~20 catchs sur 6 staircases), un taux de lapse
  de 2 % donne P(session INDÉTERMINÉE par ce seul canal) ≈ 33 %. Coût nommé et
  ACCEPTÉ à l'endossement — pas découvert après ; une session INDÉTERMINÉE par
  catch se consigne et ne se re-court pas le même jour.

## Gardes (toutes évaluées, jamais court-circuitées — l'acquis de P3 reconduit)

1. **`[v2]` Anti-gabarit ANCRÉE — remplace la version morphologique de la v1
   (REFUSÉE : elle laissait passer crochets/accolades, « p. ex. 14h », le
   copié-collé d'une observation d'hier, et « 144hz » matchait `\d{1,2}h`).**
   La vérité de référence est `date_session` du manifeste (horodatage machine,
   jamais tapé par un humain). La chaîne `--conditions` est REFUSÉE sauf si :
   (a) elle contient LA DATE DU JOUR, cohérente avec `date_session` ;
   (b) elle contient une heure explicite (motif ancré par frontières de mot),
   cohérente avec `date_session` à ±2 h — c'est la garde qui aurait attrapé à
   la fois la fuite du gabarit, la session de nuit du 26/07 et le fait faux du
   journal (§A41) ;
   (c) elle ne contient AUCUN marqueur de gabarit : chevrons, ellipses (`…` ou
   `...`), crochets ou accolades, « p. ex. », « exemple », le mot nu
   « CONDITIONS » ;
   (d) `[v2.1]` **PLAGE HORAIRE DE SESSION : `date_session` ∈ [09:00, 19:00]
   locale**, sinon session REFUSÉE. C'est la version OUTILLÉE de « lumière du
   jour » : les clauses (a)-(b) vérifient que l'heure déclarée est HONNÊTE,
   pas qu'il fait jour — une session nocturne véridique (« 00h30,
   plafonnier », date du jour) passait (a)-(c) intégralement, et la v2
   surclamait « lumière du jour VÉRIFIÉE par la garde ». La condition du pin
   redevient mécanique : hors plage, la machine refuse, personne ne déclare.
   La garde s'applique aux DEUX champs `conditions` ET `luminosite` (jamais
   vérifié en v1). Absente vaut gabarit, inchangé. Ironie consignée : le
   manifeste du pin porte `conditions: "CONDITIONS"` — le référent ne
   satisferait pas cette garde ; elle ne s'applique qu'aux sessions NOUVELLES.
2. Ordre des chemins == l'ordre gravé D-P3′-1.
3. Validité §C5 par staircase ; bras complets (3+3).
4. **`[v2]` Dispersion §C5 : CV ≤ 30 % PAR BRAS** — la méthodologie du pin
   (`evalue_dispersion`, `determine_statut_regime`), jusqu'ici JAMAIS
   implémentée dans la lecture transport (M3). Fait de référence : le bras R1
   du 26/07 était à CV 42.7 % — le pin aurait été INDÉTERMINÉ à cette
   dispersion. Cette garde ferme aussi la moitié du mécanisme complaisant de
   B1b : une session bruitée ne peut plus glisser vers la branche confortable,
   elle devient INDÉTERMINÉE.
5. Liaison sidecar↔log (sha256) pour les six.
**Plus de garde de témoin trans-sessions** — c'est le changement de design :
le bras viridis du jour EST le référent. Sa valeur face au pin gravé est
**SURFACÉE en diagnostic de dérive** (documenter le chemin du sujet depuis
Arc C), jamais un critère.

## Lectures pré-écrites (prononcées mécaniquement, IC min/max par bras — la méthode du pin, §A38-CORRECTION-4 ; le mean±2SEM surfacé, jamais décideur)

**`[v2]` La puissance du test, chiffrée AVANT (l'omission qui a fait refuser la
v1, B1b).** Le recouvrement de deux min/max sur 3+3 tirages échangeables est un
test de T=1 au niveau α = 1 − 2·(3!·3!/6!) = 0.10 exactement (argument de
rang). Sa puissance, simulée aux dispersions réellement observées (CV 42.7 %
au bras R1 du 26/07) : un transport réel T = 0.5 resterait « recouvrant »
~49 % du temps ; T = 0.7, ~78 %. **Un non-rejet à cette puissance N'ÉTABLIT
RIEN** — d'où la scission de la branche 1 en 1a/1b ci-dessous. L'IC de T par
bornes croisées aura une largeur attendue d'un facteur ~2.5–3 : gravé pour que
personne ne sur-lise T.

Sous gardes vertes (dispersion comprise, garde 4), avec IC_V = min/max des
trois seuils viridis et IC_R = idem R1, T = jnd_R1/jnd_V (points),
IC_T = [min_R/max_V, max_R/min_V] :

`[v2.2]` **PARTITION GRAVÉE (§A42-COMPLÉMENT) — les branches vivent sur la
grille inclusion × recouvrement, CLOSE et EXCLUSIVE** : la revue du feu vert
(Romain, 2026-07-29) a montré que 1b et 2 telles qu'écrites pouvaient tirer
EN MÊME TEMPS (IC disjoints ET IC_T ⊆ [1/λ, λ] — le cas existe : V =
{0.070, 0.071, 0.072}, R = {0.080, 0.081, 0.083} donne IC disjoints et
IC_T = [1.11, 1.19] ⊂ [0.667, 1.5]). Résolution β gravée, α REFUSÉE :
sous α (1b exigerait le recouvrement, disjoint ⇒ branche 2 toujours), une
session strictement PLUS PRÉCISE perdrait l'établissement de D14 qu'une
session plus bruitée obtiendrait — dégrader la mesure jusqu'au recouvrement
achèterait la branche confortable : le mécanisme de B1b réincarné,
non-monotone en qualité d'instrument. β est monotone (moins de bruit
n'affaiblit jamais l'établissement) et statistiquement propre (TOST :
« ≠ 1 ET borné dans la marge » est cohérent et plus fort). **L'inclusion de
IC_T décide de D14-à-λ ; la disjonction décide du prononcé ≠ 1 ; les deux
vraies = la branche 2-équiv, prononcé DOUBLE.**

1a. **`[v2]` IC_R ∩ IC_V ≠ ∅ ET IC_T ⊄ [1/λ, λ] ⇒ TRANSPORT NON DISTINGUÉ
   DE 1 (α ≈ 0.10, puissance ci-dessus).** C'est un NON-REJET, pas une
   équivalence : **la condition « transport ≈ 1 » du caveat D14 N'EST PAS
   ÉTABLIE** par cette branche ; elle reste ouverte, consignée comme telle.
   Le gate (iii′) ne reçoit rien ; l'anti-surclame voyage inchangé.
1b. **`[v2]` IC_T ⊂ [1/λ, λ] `[v2.2]` ET IC_R ∩ IC_V ≠ ∅ ⇒ TRANSPORT
   ÉQUIVALENT À 1, à la marge λ gravée**
   — **D-P3′-2 : TRANCHÉ (§A42), λ = 1.5.** `[v2.1]` Raison chiffrée :
   [1/1.5, 1.5] est une fenêtre de largeur 2.25 face à un IC_T attendu de
   largeur ~2.5–3 — la branche n'est atteignable que par une session à faible
   dispersion centrée sur 1, exactement la sémantique que « ÉTABLIE » doit
   exiger ; λ = 1.25 la rendrait lettre morte, λ = 2 affaiblirait D14 sous
   l'utile. C'est la
   SEULE branche qui ÉTABLIT la condition du caveat D14 — dans sa version
   intra-session, la plus forte disponible, et bornée par λ en toutes lettres
   (« établie à un facteur ≤ 1.5 près »). Coût nommé : avec un IC_T large d'un
   facteur ~2.5–3, cette branche est EXIGEANTE — c'est voulu ; si elle n'est
   pas atteinte, la lecture honnête est 1a, jamais un surclassement. **
   L'anti-surclame VOYAGE inchangé** : la pondération d'excentricité n'est PAS
   mesurée (stimulus fovéal ~2°), la scission D14 ne se referme pas ici ; le
   gate (iii′) mesure avec le pin d'instrument via R1 — le juge se durcit, le
   contrat ne s'affaiblit pas (§A36 P0-b).
2. **IC disjoints `[v2.2]` ET IC_T ⊄ [1/λ, λ] ⇒ TRANSPORT ≠ 1, prononcé,
   avec direction et facteur** : T et IC_T comme ci-dessus. Le gate (iii′)
   reçoit son échelle RELATIVE ; `r_fovea` reçoit son consommateur ; **AUCUN
   seuil d'état ne bouge** — le pin gravé reste le pin du harnais, jnd du
   jour et T sont des grandeurs NOUVELLES.
2-équiv. **`[v2.2]` IC disjoints ET IC_T ⊆ [1/λ, λ] ⇒ PRONONCÉ DOUBLE** :
   TRANSPORT ≠ 1 (direction et facteur — l'échelle part au gate (iii′),
   `r_fovea` reçoit son consommateur) **ET équivalence à la marge λ ÉTABLIE
   (T ≠ 1, borné)** — la condition D14 est établie à λ près, au sens borné,
   jamais au sens d'identité. Les deux moitiés du prononcé voyagent
   ENSEMBLE ; en citer une sans l'autre est un surclame.
3. **Gardes échouées, bras amputé, catch/dispersion §C5 hors clous ⇒
   INDÉTERMINÉE** — un résultat, consigné, jamais re-couru en relâchant.

## Ce que P3′ ne mesure PAS (refus explicites, inchangés)

Pas la pondération d'excentricité (le juge (iii′) attend son échelle, pas
l'inverse) ; pas le vivant fovéal (stimuli mono-niveau, Option B — l'ABX de
substitution attend le compositeur) ; pas de re-pin (le pin gravé est intouché,
la dérive du sujet est SURFACÉE, jamais re-jugée) ; pas le son, pas R2 — R1
seul, daté par son empreinte c5ac8757….

## `[v2]` Non-cécité consignée (M4 — irrattrapable, donc dit en toutes lettres)

Les trois seuils R1 du 26/07 (0.0259 / 0.0154 / 0.0380) et le témoin (0.0506)
sont EN CLAIR dans `manifeste_p3.json`, écrit par l'orchestrateur AVANT toute
lecture — l'étanchéité « ni lus ni écrits » protégeait le RAPPORT, pas le
regard. **Ce prereg n'a donc PAS été écrit à l'aveugle** : quiconque a ouvert
le manifeste peut anticiper que P3′ donnera vraisemblablement la branche 2
avec T ≈ 0.5. Consigné ici pour que le verdict de P3′ soit lu avec cette
réserve. Correctif STRUCTUREL (pas incantatoire), dû au chantier 8+ : c'est
l'orchestrateur qu'il faut changer — les seuils des bras sont SCELLÉS
(sha256 dans le manifeste, valeurs dans un fichier à part non ouvert) jusqu'au
prononcé mécanique du verdict.

## Ce que chaque branche change (D17, en toutes lettres)

| branche | décision consommatrice |
|---|---|
| 1a — non distingué de 1 | `[v2]` RIEN ne se décide : la condition D14 reste OUVERTE (consignée « non distinguée, non établie ») ; la suite appartient à Romain |
| 1b — équivalent à 1 (marge λ) | le caveat D14 (condition transport ÉTABLIE à λ près) ; le gate (iii′) mesurera avec le pin d'instrument via R1 |
| 2 — transport ≠ 1 | l'échelle relative du gate (iii′) ; le consommateur de `r_fovea` ; la spec du compositeur héritera de T |
| 2-équiv — ≠ 1 ET borné à λ `[v2.2]` | les DEUX : D14 établie à λ près (au sens borné) ET l'échelle T au gate (iii′) / `r_fovea` / compositeur — le prononcé double voyage entier |
| INDÉTERMINÉE | la décision suivante appartient à Romain — aucune lecture par défaut |

## Outillage (chantier 8 — débloqué par §A42, lancé sur ordre de mission explicite)

Orchestration : plan d'entrelacement gravé (l'ordre D-P3′-1), bloc
d'échauffement non scoré, les six staircases, manifeste avec ordre effectif et
`base_seed = 20260729` vérifié par garde. Lecture : mode P3′ (gardes ancrées,
garde CV par bras, IC min/max par bras, ratio, branches 1a/1b/2/3 ci-dessus,
diagnostic de dérive et seuils-par-position surfacés). `[v2]` Champs d'essai
imposés : les deux pré-vols du 25/07 ET la session P3 du 26/07 doivent être
REFUSÉS — les pré-vols par les marqueurs, la session P3 par la garde (b)
heure↔`date_session` (session de nuit, 00h27) ET par les marqueurs (ellipses)
`[v2.1]` ET par la clause (d) (00h27 est hors plage [09:00, 19:00]) ;
un manifeste synthétique au CV > 30 % sur un bras doit rendre INDÉTERMINÉE ;
`[v2.2]` côté ENTRÉE : une graine de la LISTE DES GRAINES BRÛLÉES doit être
REFUSÉE (le test étend la liste — il prouve qu'elle lit la liste, pas une
constante) ; côté LECTURE : une graine ni brûlée ni gravée (ex. 12345) ET
`20260705` doivent être REFUSÉES par L'ÉGALITÉ seule, et une archive à
`20260729` doit rester LISIBLE même quand `20260729` figure dans la liste
(le test du brûlage : re-jouable en lecture, refusée à l'entrée) ; un
manifeste véridique à `date_session` hors plage (« 00h30, plafonnier », date
et heure honnêtes) doit être REFUSÉ par (d) seule ; `[v2.2]` le cas
2-équiv : V = {0.070, 0.071, 0.072}, R = {0.080, 0.081, 0.083} doit rendre
EXACTEMENT le prononcé double (≠ 1 ET borné à λ), jamais 1b ni 2 seule.
Chaque branche de lecture prouvée falsifiable sur manifestes synthétiques,
comme au chantier 7.

**POINT D'ARRÊT : v2.1 ENDOSSÉE, gravure §A42 FAITE (2026-07-29). Prochain
pas : ordre de mission chantier 8 (décision explicite), puis la session, un
jour neuf, `date_session` dans la plage gravée [09:00, 19:00], garde 1
(a)-(d) verte. Aucun enchaînement.**
