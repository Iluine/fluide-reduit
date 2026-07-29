# ORDRE DE MISSION — chantier 8 : outillage P3′ (orchestration, lecture, champs d'essai)

> **Statut : BROUILLON soumis à revue Romain (2026-07-29, 23h — horloge lue).
> Le build ne démarre qu'après son feu vert explicite. Aucun enchaînement.**
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
3. **Graine** : `base_seed = 20260729` recopié au manifeste et vérifié.
   **GRAINES BRÛLÉES** : liste `GRAINES_BRULEES = {20260705}` en UN exemplaire
   (`src/arcC_abx.py`, importée partout — jamais recopiée). Refus à l'entrée
   sous `--sujet humain` ET à la lecture. **PORTÉE NOMMÉE : la garde ne mord
   que les sessions HUMAINES nouvelles** — replay et synthétique consomment
   légitimement les graines historiques (re-dérivabilité du pin préservée).
4. **Garde 1 (a)-(d) à l'entrée** : la clause (d) — `date_session` ∈
   [09:00, 19:00] locale — s'ajoute à `verifie_observation_conditions`
   (UN exemplaire, paramètre de plage avec défaut gravé ; à l'entrée
   l'instant vient de l'horloge, à la lecture de `date_session`). Les clauses
   (a)-(c) existent déjà — tu ne les retouches pas.
5. **Scellement des seuils (correctif structurel M4, mode P3′ seulement)** :
   les seuils des bras ne s'écrivent PLUS en clair au manifeste — ils vont
   dans un fichier scellé à part, son sha256 au manifeste ; seule la lecture
   mécanique, gardes toutes vertes, le descelle. Le schéma du manifeste P3
   historique ne bouge pas.
6. Pauses libres entre staircases : durées consignées au manifeste, surfacées,
   jamais jugées.

### 8b — Lecture P3′ (`scripts/run_p3_lecture.py`, mode P3′)

Gardes AVANT toute lecture de seuil, dans cet ordre : liaison sidecar↔log
(sha256, les six) ; garde conditions (a)-(d) ; garde graine (liste) ; garde
chemins (== plan D-P3′-1) ; validité §C5 par staircase + bras complets 3+3 ;
**dispersion CV ≤ 30 % PAR BRAS — les deux bras, viridis compris**
(`evalue_dispersion` importée, jamais réimplémentée). Tant qu'une garde est
rouge : INDÉTERMINÉE, **aucun seuil imprimé ni écrit** (le scellement de 8a.5
rend cette étanchéité structurelle, plus déclarative).

Puis, gardes vertes : IC min/max par bras (`calcule_ic` importée),
T = jnd_R1/jnd_V, IC_T = [min_R/max_V, max_R/min_V], branches prononcées
TELLES QUELLES : 1a (recouvrement sans IC_T ⊂ [1/λ, λ] — rien ne s'établit),
1b (IC_T ⊂ [1/λ, λ], **λ = 1.5 gravé §A42**, constante nommée avec sa source),
2 (IC disjoints — direction et facteur), 3 (INDÉTERMINÉE). Le mean±2SEM
surfacé, jamais décideur. Diagnostics surfacés jamais jugés : dérive
viridis-du-jour vs pin gravé ; seuils-par-position.

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
- Graine de la liste ⇒ REFUSÉ ; le test étend la liste avec une valeur de
  test — il prouve que la garde lit la LISTE, pas une constante.
- Chaque branche 1a/1b/2/3 prouvée ATTEIGNABLE sur manifestes synthétiques
  (dont le cas-frontière : recouvrement des IC sans inclusion de IC_T ⇒ 1a,
  jamais 1b).
- Non-régression : mode P3 historique et campagnes du pin inchangés (leurs
  tests existants verts, empreintes de seeds identiques en replay).

## Garde-fous (le contrat de la mission)

- **Modules du pin INTOUCHÉS** : `arcC_rendu`, `arcC_stimuli`, le flux
  d'essais de `run_escalier`. Dans `src/arcC_abx.py`, SEULS ajouts autorisés :
  clause (d) paramétrée, `GRAINES_BRULEES`, helpers de scellement — rien dans
  le tirage ni le scoring. Empreinte R1 `c5ac8757…` inchangée, vérifiée à la
  remise.
- **Toutes les constantes sont GRAVÉES ailleurs et recopiées avec leur
  source** (λ = 1.5, plage [09:00, 19:00], CV 30 %, N = 15, base_seed
  20260729, plan V,R,R,V,V,R) — tu n'inventes AUCUN seuil. Une tension avec
  le gravé se REMONTE, ne se résout pas en local (tradition §A38-*).
- Aucune commande pré-remplie dans les documents ; aucun chrono verdict-grade ;
  AUCUNE session humaine lancée par toi.
- Suite de tests complète verte avant remise ; ruff via
  `pocCascade2phys/.venv` (fait d'instrument §A38).
- **Cap : 1 séance.** Dépassement ⇒ remontée, pas de poursuite silencieuse.
- Remise : rapport de build (`.superpowers/sdd/`), diff VIDE sur les
  intouchés, revue critique AVANT commit.

**POINT D'ARRÊT DE LA MISSION : à la remise, revue critique puis retour à
Romain. La session humaine P3′ vient APRÈS, un jour neuf, `date_session` dans
la plage, sur sa décision explicite — jamais dans la foulée du build.**
