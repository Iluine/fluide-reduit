# ORDRE DE MISSION — chantier 8 : outillage P3′ (orchestration, lecture, champs d'essai)

> **Statut : BROUILLON v3 (2026-07-29, 23h45 — horloge lue) — v2 = revue
> Romain intégrée (cinq corrections + cinq décisions) + quatre affûtages ;
> v3 = les deux trous résiduels de la re-scrutée du feu vert, gravés
> §A42-COMPLÉMENT : précédence 1b/2 (résolution β, partition close avec
> branche 2-équiv) et portée de la garde-liste (entrée seule, lecture par
> égalité, archive re-jouable après brûlage). Le build ne démarre qu'après
> le feu vert explicite. Aucun enchaînement.**
> **Pour : session Claude Code.** Développement possible sur la VM Cowork
> (aucun GPU requis ; le conteneur n'est PAS l'instrument — aucune mesure
> verdict-grade, aucune session humaine ne t'appartient). **Sources qui font
> foi** : `claude/prereg-p3prime-transport-intra-session.md` (**ENDOSSÉ v2.1,
> §A42** — protocole, gardes, branches et constantes y sont GRAVÉS : tu les
> IMPLÉMENTES, tu ne les interprètes pas), §A41 + §A41-ERRATUM (règle « la
> machine lit, la session ne recopie pas »), §A42 (D-P3′-1 = V,R,R,V,V,R
> miroir refusé ; D-P3′-2 : λ = 1.5 ; plage horaire ; graines brûlées).

## Trois chantiers — et rien d'autre

### 8a — Orchestration P3′ (`scripts/run_arcC_orchestration.py`)

1. **Plan d'entrelacement** : généraliser `chemins_staircases` pour porter le
   plan gravé D-P3′-1 — six staircases `(viridis, r1, r1, viridis, viridis,
   r1)` — derrière un mode explicite (ex. `--protocole p3prime`). **Le
   comportement historique est INTACT par défaut** : P3 (r1,r1,viridis,r1) et
   les campagnes du pin rejouent à l'identique, leurs tests existants verts
   sans modification. L'ordre effectif s'écrit au manifeste ; une garde le
   compare au plan gravé (l'acquis de P3 : un ordre déplacé en silence n'est
   plus le protocole).
2. **Bloc d'échauffement** : N = 15 essais ABX NON SCORÉS avant la staircase 1
   (niveau fixe supra-seuil, mêmes 20 sources, aucun seuil produit, aucune
   écriture dans les logs de staircase). Consigné au manifeste : nombre,
   niveau, durée. Sa graine est dérivée du schéma existant avec un indice
   RÉSERVÉ (hors 0..5), surfacée au manifeste — les staircases gardent leurs
   indices canoniques.
3. **Graine — garde d'ENTRÉE, double** : sous `--sujet humain` en mode P3′,
   `base_seed` doit ÉGALER le `20260729` gravé ET être ABSENT de la liste
   `GRAINES_BRULEES = {20260705}` (UN exemplaire, `src/arcC_abx.py`, importée
   — jamais recopiée). **[v3] La LISTE ne mord qu'ICI, à l'entrée** — c'est
   elle qui empêchera un P3″ de reconduire `20260729` après son brûlage.
   Portée nommée : sessions HUMAINES nouvelles seulement — replay et
   synthétique consomment légitimement les graines historiques
   (re-dérivabilité du pin préservée).
4. **Garde 1 (a)-(d) à l'entrée** : la clause (d) — `date_session` ∈
   [09:00, 19:00] locale — s'ajoute à `verifie_observation_conditions`
   (UN exemplaire, paramètre de plage avec défaut gravé ; à l'entrée
   l'instant vient de l'horloge, à la lecture de `date_session`). Les clauses
   (a)-(c) existent déjà — tu ne les retouches pas. **[rev] La plage s'évalue
   sur `date_session` — le DÉBUT de session** : commencée 18h50, finie 20h10 =
   dans la plage (tranché ici, pas en séance). **Limite nommée** : la plage
   est un proxy SAISONNIER du jour — exacte en été (P3′ attendu), imparfaite
   en hiver ; la description de lumière continue de vivre dans l'observation
   (a)-(c), la plage n'en dispense pas.
5. **Scellement des seuils (correctif structurel M4, mode P3′ seulement)** :
   les seuils des bras ne s'écrivent PLUS en clair au manifeste — ils vont
   dans un fichier scellé à part, son sha256 au manifeste ; seule la lecture
   mécanique, gardes toutes vertes, le descelle. Le schéma du manifeste P3
   historique ne bouge pas. **[rev] NATURE NOMMÉE : scellement PROCÉDURAL,
   pas cryptographique** — il protège du regard ACCIDENTEL et rend toute
   ALTÉRATION détectable (sha256) ; la CONSULTATION, elle, ne laisse AUCUNE
   trace — un fichier se lit, c'est la leçon M4 elle-même. Si quelqu'un ouvre
   le fichier scellé avant le prononcé, la non-cécité se CONSIGNE au verdict,
   comme au prereg v2. Les helpers vivent dans un module DÉDIÉ
   `src/arcC_scelle.py` (UN exemplaire importé par l'orchestration ET la
   lecture, testé seul — la raison est le foyer unique testable, pas une
   « pureté » d'`arcC_abx.py` que `ecrit_log_jsonl` dément déjà).
6. Pauses libres entre staircases : durées consignées au manifeste, surfacées,
   jamais jugées.
7. **[rev] Échauffement : log SÉPARÉ, marqué NON-ANALYSÉ** (suffixe explicite
   dans le nom de fichier, référencé au manifeste) — il PROUVE que
   l'échauffement a eu lieu tel que gravé (15 essais, niveau, horodatages)
   sans inviter personne à le scorer ; aucun outil de lecture ne l'ouvre.

### 8b — Lecture P3′ (`scripts/run_p3_lecture.py`, mode P3′)

**[rev] TOUTES les gardes sont ÉVALUÉES et rapportées, aucune ne
court-circuite les autres** (§A38-CORRECTION-4, choix endossé : une mesure
humaine ne se relance pas une garde à la fois) ; l'ordre ci-dessous ne
gouverne que le PRONONCÉ du motif principal — et les seuils ne se DESCELLENT
que TOUT VERT. Les gardes : liaison sidecar↔log (sha256, les six) ; garde
conditions (a)-(d) ; **[v3] garde graine à la lecture : ÉGALITÉ SEULE au
`base_seed` gravé (20260729)** — elle pinne tout : une graine quelconque
(12345) ET `20260705` tombent par l'égalité. **La liste des brûlées ne mord
PAS ici** (§A42-COMPLÉMENT) : sinon l'acte de brûlage rendrait l'archive P3′
illisible — la re-dérivabilité promise exige qu'une archive à `20260729`
reste re-jouable après brûlage. Portée : mode P3′ seulement, le mode P3
historique lit légitimement son manifeste archivé à 20260705 ; garde chemins
(== plan
D-P3′-1) ; validité §C5 par staircase + bras complets 3+3 ; **dispersion
CV ≤ 30 % PAR BRAS — les deux bras, viridis compris** (`evalue_dispersion`
importée, jamais réimplémentée). Tant qu'une garde est rouge : INDÉTERMINÉE,
**aucun seuil imprimé ni écrit ni descellé**.

Puis, gardes vertes : IC min/max par bras (`calcule_ic` importée),
T = jnd_R1/jnd_V, IC_T = [min_R/max_V, max_R/min_V], branches prononcées
TELLES QUELLES : 1a (recouvrement sans IC_T ⊂ [1/λ, λ] — rien ne s'établit),
1b (IC_T ⊂ [1/λ, λ], **λ = 1.5 gravé §A42**, constante nommée avec sa source),
2 (IC disjoints sans inclusion — direction et facteur), **[v3] 2-équiv (IC
disjoints ET IC_T ⊆ [1/λ, λ] ⇒ PRONONCÉ DOUBLE : ≠ 1 avec direction/facteur
ET équivalence à λ établie au sens borné — §A42-COMPLÉMENT, résolution β ;
la partition vit sur la grille inclusion × recouvrement, close et exclusive,
1b exige désormais AUSSI le recouvrement)**, 3 (INDÉTERMINÉE). **[rev]
CONVENTION DE BORNE, gravée : ⊂ à BORNES INCLUSES** — IC_T = [x, 1.5]
exactement ⇒ la cellule d'inclusion (1b ou 2-équiv selon le recouvrement).
Ce qui tranche : la symétrie avec la convention du pin (« un témoin pile
à la borne n'est pas une dérive »). L'argument adverse (bornes strictes, 1b
plus dur) est nommé et ÉCARTÉ : sur des seuils mesurés, l'égalité exacte est
de mesure nulle — la convention pèse sur le déterminisme du prononcé et les
tests, pas sur la puissance. Le mean±2SEM surfacé, jamais décideur.
Diagnostics surfacés jamais jugés : dérive viridis-du-jour vs pin gravé ;
seuils-par-position.

### 8c — Champs d'essai (tests, tous IMPOSÉS par le prereg v2.1)

- Les TROIS manifestes historiques REFUSÉS : les deux pré-vols du 25/07 (par
  les marqueurs) ; la session P3 du 26/07 par TROIS chemins, chacun testé
  suffisant seul — (b) aucune heure lisible, (c) ellipses, (d) 00h27 hors
  plage.
- Un manifeste synthétique VÉRIDIQUE hors plage (« 00h30, plafonnier », date
  et heure honnêtes et cohérentes) REFUSÉ par (d) SEULE — c'est le test qui
  distingue la plage du simple contrôle d'honnêteté.
- CV > 30 % sur UN bras (l'un puis l'autre) ⇒ INDÉTERMINÉE, seuils non
  divulgués.
- **[v3] Graine, les deux côtés testés là où ils mordent** : à l'ENTRÉE,
  graine de la liste ⇒ REFUSÉ (le test étend la liste — il prouve qu'elle lit
  la LISTE, pas une constante) ; à la LECTURE, graine ni brûlée ni gravée
  (ex. 12345) ⇒ REFUSÉ par l'égalité, `20260705` ⇒ REFUSÉ par l'égalité, ET
  **le test du brûlage** : une archive à `20260729` avec `20260729` AJOUTÉ à
  la liste reste LISIBLE en lecture et REFUSÉE à l'entrée.
- Chaque branche 1a/1b/2/2-équiv/3 prouvée ATTEIGNABLE sur manifestes
  synthétiques (dont le cas-frontière : recouvrement des IC sans inclusion de
  IC_T ⇒ 1a, jamais 1b ; **[rev] le cas-frontière de BORNE : IC_T = [x, 1.5]
  exactement ⇒ cellule d'inclusion, bornes incluses ; [v3] et le cas 2-équiv
  GRAVÉ : V = {0.070, 0.071, 0.072}, R = {0.080, 0.081, 0.083} ⇒ EXACTEMENT
  le prononcé double, jamais 1b ni 2 seule**).
- Non-régression : mode P3 historique et campagnes du pin inchangés (leurs
  tests existants verts, empreintes de seeds identiques en replay).

## Garde-fous (le contrat de la mission)

- **Modules du pin INTOUCHÉS** : `arcC_rendu`, `arcC_stimuli`, le flux
  d'essais de `run_escalier`. Dans `src/arcC_abx.py`, SEULS ajouts autorisés :
  clause (d) paramétrée, `GRAINES_BRULEES` et le `base_seed` gravé en
  constantes — rien dans le tirage ni le scoring. **[rev] Le scellement vit
  dans `src/arcC_scelle.py`, module dédié** (cf. 8a.5), pas dans abx.
  Empreinte R1 `c5ac8757…` inchangée, vérifiée à la remise.
- **Toutes les constantes sont GRAVÉES ailleurs et recopiées avec leur
  source** (λ = 1.5, plage [09:00, 19:00], CV 30 %, N = 15, base_seed
  20260729, plan V,R,R,V,V,R) — tu n'inventes AUCUN seuil. Une tension avec
  le gravé se REMONTE, ne se résout pas en local (tradition §A38-*).
- Aucune commande pré-remplie dans les documents ; aucun chrono verdict-grade ;
  AUCUNE session humaine lancée par toi.
- **[rev] Rapport de remise : N passés / M SAUTÉS, les sauts LISTÉS** —
  « verte-avec-sauts » ne se dit pas « verte ». Sur la VM Cowork le GPU est
  absent : les sauts f1 sont ATTENDUS et acceptables pour CE chantier (aucun
  code GPU touché) ; la suite GPU appartient à la machine-instrument. Ruff
  via `pocCascade2phys/.venv` (fait d'instrument §A38).
- **[rev] Qui brûle `20260729`** : APRÈS la session P3′, son ajout à
  `GRAINES_BRULEES` est un acte de GRAVURE — entrée journal + commit dédié —
  jamais un réflexe de la session qui vient de mesurer.
- **Cap : 1 séance.** Dépassement ⇒ remontée, pas de poursuite silencieuse.
- Remise : rapport de build (`.superpowers/sdd/`), diff VIDE sur les
  intouchés, revue critique AVANT commit.

**POINT D'ARRÊT DE LA MISSION : à la remise, revue critique puis retour à
Romain. La session humaine P3′ vient APRÈS, un jour neuf, `date_session` dans
la plage, sur sa décision explicite — jamais dans la foulée du build.**
