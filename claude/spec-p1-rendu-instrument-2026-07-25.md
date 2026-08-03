# SPEC P1 — RENDU MINIMAL D'INSTRUMENTATION (albédo→luminance) — 2026-07-25

> **Statut : PAPER GRADE — spec soumise à revue, RIEN ne se construit avant le
> verdict de Romain.** Gate P1 de l'arc projection (§A36, P0-c). Le journal
> `pocCascade2phys/PREREGISTRATION.md` fait foi (fin de fichier : §A36) ; la séance
> `seance-fidelite-2026-07-25.md` et la note `note-orientation-v2-2026-07-25.md`
> nourrissent. **Règle D17 sur toute la spec** : *aucun diagnostic sans une décision
> nommée qui en dépend explicitement.*
>
> Étiquetage : `[GRAVÉ]` / `[FAIT DE CODE]` (vérifiable au grep, cité file:line) /
> `[CALCUL]` (enveloppe papier, jamais mesurée) / `[ESTIMATION]` (effort de build,
> à confronter au chiffrage Claude Code) / `[DÉCISION D-P1-x]` (à trancher en revue) /
> `[PROPOSITION]` (recommandation de la session, Romain tranche).
>
> **À graver avec l'entrée journal de P1, après revue** (décisions Romain du 25/07) :
> les six pistes du §7 de la note d'orientation ; la lecture différée O-Voxel
> (exclu comme base de `z`, candidat couche readout de production, réexamen post-P3) ;
> la règle d'hygiène des faits d'instrument (« un fait d'instrument porte sa date ;
> à la reprise, re-vérifier les moins chers avant de les citer »).

---

## 0. LA DÉCISION DE PÉRIMÈTRE (D-P1-1) — elle commande tout le reste

**La question posée par §A36** : l'instrument projette-t-il la STRUCTURE RÉELLE
(fovéa fine + grossier upsamplé, halo compris) ou une image PLEINE RÉSOLUTION
d'abord ?

**Le fait qui décompose la question** `[PROPOSITION, structurante]` : le noyau de
rendu est **périmètre-neutre** si son interface est « un champ albédo pleine
résolution → une luminance affichée calibrée ». Projeter la structure réelle =
le MÊME noyau précédé d'un **COMPOSITEUR** (état 2-niveaux → champ albédo composite
pleine résolution : upsample du grossier + insertion de la fovéa + couture). Les
deux options ne sont donc pas deux rendus : c'est **un noyau, avec ou sans
compositeur**.

### Option B — PLEINE RÉSOLUTION d'abord (noyau seul)

- **Ce qu'elle sert (D17)** : **P3 exactement** — le transport du pin rejoue les
  paires ABX d'Arc C, qui sont des champs **mono-niveau pleine résolution**
  (`arcC_stimuli.py` : `stim(t)` = mélange en espace sédiment des npz de manche 1 ;
  aucune structure fovéale dedans). P3 est le prochain gate de l'arc et le
  falsifieur du caveat D14 : ses **deux branches changent des décisions gravées**
  (transport ≈ 1 sans pondération ⇒ la scission D14 se referme, C-obs hérite de
  l'échec de C-uni ; transport ≠ 1 ⇒ le gate (iii′) reçoit son échelle et `r_fovea`
  son consommateur — §A35, seance-fidelite §4). Sert aussi **P2** sur le terme
  dominant (le chemin de shading par pixel).
- **Coût build** `[ESTIMATION]` : **0.5–1 séance Claude Code.** Pièces existantes :
  `albedo()` importé tel quel, `arcC_calibration.py` (trois fonctions pures),
  harnais ABX en logique pure (`arcC_abx.py`, zéro dépendance affichage — la
  coquille `run_arcC_session.py` substitue le chemin d'affichage). Le neuf : le
  noyau (§2) + ses tests + le branchement session.
- **Ce qu'elle ne sert PAS** : juger le VIVANT fovéal (ABX de substitution,
  gate (iii′) en conditions réelles) — le compositeur n'existe pas.

### Option A — STRUCTURE RÉELLE (noyau + compositeur)

- **Ce qu'elle sert en plus** : l'ABX de substitution et le jugement du vivant
  fovéal — **dont aucun n'est mesurable avant P3** (le juge n'a pas d'échelle tant
  que le transport du pin n'est pas mesuré ; seance-fidelite §4 gate l'ABX de
  substitution « après P1 » mais sa LECTURE dépend du pin transporté).
- **Coût build** `[ESTIMATION]` : **+0.5–1 séance** par-dessus B (compositeur +
  tests de couture), ET une décision de plus à prendre à froid : l'opérateur de
  couture fovéa/grossier — qui est précisément la « frontière feuille/projection »
  du falsifieur unifié C-strat (§A36). La prendre MAINTENANT, sans pin transporté
  pour la juger, c'est spécifier la pièce la plus délicate de l'arc sans son
  instrument de mesure.
- **Verdict D17** : entre P1 et P3, **aucune décision nommée ne consomme le
  compositeur.** Il arrive à son heure : après P3, comme incrément séparé, avec le
  pin transporté pour le juger.
  *(**BORNÉ le 2026-08-03 — §A48**, clause normative amendée à la ligne par
  §A47-PRÉCISION-2. L'énoncé reste vrai : aucune décision sur la couture n'est
  consommée, et le compositeur **instrument** reste post-P3 — l'argument ci-dessus, qui
  le protège de son juge absent, tient intégralement. Mais son ÉNUMÉRATION est datée du
  25/07 : la correction de but a créé un consommateur qu'elle ne pouvait pas lister —
  le **coût de rendu**, dont dépend la lecture de l'enveloppe 3D, et que cette même
  section range plus bas dans la classe composition (`:74-80`). §A48 autorise en
  conséquence un **chemin-de-coût** : naïf dans l'opérateur, réel dans la structure
  creuse, jugé au chronomètre SEUL, interdit de substrat à toute mesure perceptuelle
  par une serrure `RuntimeError`, et superséé par le compositeur instrument post-P3.)*

### Ce que le choix de B laisse ouvert, dit sans lissage

**P2 sur Option B chiffre le noyau, PAS la composition.** À l'échelle V4, la
composition n'est pas négligeable `[CALCUL]` : construire la mosaïque albédo écran
(~2·10⁶ px à 1920×1080) depuis une pyramide creuse est un gather du même ordre que
le shading lui-même. **P2 sera donc « la première ancre de la moitié manquante »
(§A36), pas la moitié entière** — le coût de composition reste une dette nommée,
mesurable quand le compositeur existera. Quiconque lit P2 comme « le rendu coûte X »
surclame ; P2 dira « le NOYAU coûte X ».

**`[PROPOSITION]` : Option B.** L'interface du noyau (§2) est gravée
périmètre-neutre pour que le compositeur, le jour venu, soit un producteur de champ
albédo de plus — zéro re-travail du noyau. **Romain tranche.**

---

## 1. Ce que P1 remplace — le fait de code qui fonde la spec

`[FAIT DE CODE]` **Le pin a été mesuré en PSEUDO-COULEUR, pas en luminance.**
Le chemin d'affichage des sessions Arc C est
`ax.imshow(image, cmap="viridis", vmin=0.0, vmax=1.0, origin="lower")`
(`scripts/run_arcC_session.py:107`, doc module l.26). La dette « albédo→luminance »
(§C4-3, état des lieux §1.2) est donc concrète : jnd_sev 7.33 % IC [6.03, 8.67]
`[GRAVÉ]` a été mesuré à travers la rampe viridis de matplotlib, avec le
rééchantillonnage par défaut d'imshow, sans gestion d'EOTF.

Portée honnête, dans les deux sens : viridis est **par conception monotone et
quasi-linéaire en clarté** (c'est sa raison d'être) — le transport pourrait être
≈ 1. Mais « pourrait » est exactement ce que **P3 mesure** ; ce n'est une dispense
de rien. P1 construit le chemin remplaçant ; P3 dira ce que le pin devient à
travers lui.

**Ce que P1 ne touche PAS** : l'observable Δχ (`src/albedo.py:96-139` — χ_b =
RMS(bande)/⟨a⟩ par octaves radiales, bandes porteuses ≥ 10 % de l'AC de la
référence, critère = max_carrier) reste **inchangé, dans l'espace albédo**. Le pin
gravé reste le pin. P1 est un chemin d'AFFICHAGE ; aucun seuil, aucune bande,
aucun combinateur n'est modifié.

---

## 2. Le contrat du noyau R1 — albédo→luminance, déterministe, calibré

### 2.1 Interface (périmètre-neutre, gravée)

```
R1 : (A : champ albédo H×W, float64, valeurs [0,1]) → I : image affichable
     (quantifiée, taille calibrée), PURE, sans RNG, sans état.
```

- **Entrée** : un champ albédo pleine résolution — aujourd'hui produit par
  `albedo(s, S_HALF_OP)` (S_HALF_OP = 0.05·RELIEF, gravé manche 1, importé de
  `scripts.run_arcA_revalidate`, jamais réimplémenté) ; demain, le cas échéant,
  par le compositeur (Option A différée).
- **Sortie** : l'image présentée par la coquille de session, en remplacement
  drop-in du chemin viridis (même point d'insertion que
  `run_arcC_session.py:102-107`).

### 2.2 Les invariants hérités de §A20 `[GRAVÉ]`, appliqués à l'instrument

1. **Aucun état de readout load-bearing** — R1 est une fonction pure de `A`. Pour
   l'instrument minimal, PLUS DUR que §A20-3 : l'état éphémère de readout y est
   *toléré* (AA temporel, exposition) ; dans R1 il est **interdit tout court**.
   Un instrument avec mémoire n'est pas re-jouable.
2. **`z` stable ⇒ image stable, par construction** (anti-PERSIST) — testé, pas
   argumenté (§5).
3. **Pas de solve d'équilibre** (radiosité/LPV de §A20-3) : dégradation délibérée
   et NOMMÉE. R1 est un mapping local par pixel. Conséquence de portée, gravée
   d'avance : **P3 date son verdict sur R1** — un rendu de production plus riche
   (ombres portées, GI) est un AUTRE chemin, dont le transport se re-mesurerait.
   Le pin transporté sera étiqueté « à travers R1 », jamais « à travers le rendu ».

### 2.3 La chaîne, étage par étage

| # | étage | contenu | statut |
|---|---|---|---|
| 1 | albédo | `albedo(s, S_HALF_OP)` | importé tel quel `[GRAVÉ]` |
| 2 | luminance relative | `Y = A` (identité) | **D-P1-3** |
| 3 | encodage écran | inverse-EOTF sRGB de `Y`, formule standard gravée en toutes lettres dans le code | **D-P1-4** |
| 4 | quantification | uint8, arrondi au plus proche, gravé | neuf, trivial, testé |
| 5 | rééchantillonnage vers la taille calibrée | `taille_domaine_px(ppd, 5.5, 3.0)` px | **D-P1-2** |
| 6 | présentation | fenêtre session, fond gravé, régimes §C0 inchangés (sévère/laxiste = présentation seulement, `arcC_abx.py` l.20-26) | coquille existante |

### 2.4 Les trois décisions d'étage (options + recommandation, Romain tranche)

**(D-P1-3) Chemin lumineux : identité ou relief ?**
- **(a) `Y = A` — identité** `[PROPOSITION]` : le strict albédo→luminance, UNE
  nouveauté à la fois. P3 mesure alors le transport du pin sur la SEULE variable
  changée (pseudo-couleur → luminance). C'est la discipline du projet.
- (b) `Y = A · irradiance_lambertienne(b0+s)` : plus proche du jeu
  (`relief_shaded` existe, `src/albedo.py`), mais **confond deux transports**
  (colorimétrie ET géométrie lumineuse) dans une seule mesure P3 — inattribuable
  par construction. Consigné comme **incrément R2, post-P3**, avec sa propre
  mesure de transport s'il est acheté.

**(D-P1-4) EOTF : linéarisation sRGB ou valeurs brutes ?**
- **(a) inverse-EOTF sRGB gravé** `[PROPOSITION]` : la luminance affichée devient
  ~linéaire en `A` sur un écran standard. Portée dite sans surclame : linéarité
  **supposée sRGB, jamais mesurée au photomètre** — ce que l'ABX exige n'est pas
  la linéarité absolue mais le déterminisme et la stabilité (les deux stimuli d'un
  essai traversent le MÊME EOTF ; biais commun). Conditions de session gravées
  d'Arc C inchangées (OSD, lumière stable).
- (b) valeurs brutes (comme le chemin viridis aujourd'hui) : une hypothèse
  implicite de moins dans le code, une non-linéarité non documentée de plus à
  l'écran. Rejetée par la session : l'implicite est le seul vrai coût.

**(D-P1-2) Rééchantillonnage : la seule décision d'instrument délicate.**
Le domaine 64² s'affiche à `taille_domaine_px` — un facteur **non entier** en
général (l'arrondi est dans le contrat, `arcC_calibration.py`). Le chemin actuel
laisse ce choix à imshow (interpolation par défaut, non gravée). Options :
- **(a) bilinéaire, gravé dans R1** `[PROPOSITION]` : doux, déterministe, identique
  pour les deux stimuli de chaque essai (biais commun) ; le flou introduit est
  celui d'un rééchantillonnage nommé, pas d'un défaut caché.
- (b) nearest à facteur entier forcé : géométrie exacte mais la porteuse quitte le
  pic CSF (à ppd courant, arrondir 64²→N·64 px déplace la porteuse hors de
  ~3 c/deg — le contrat §C7 perd sa cible), et les blocs de cellules deviennent
  un artefact propre (garde `plafond_texture`).
- Dans tous les cas : le rééchantillonnage est un ÉTAGE DE R1 (numpy, testé,
  déterministe), **plus jamais une option de bibliothèque d'affichage**.

### 2.5 Calibration — héritée d'Arc C, valeurs re-mesurées

Les trois fonctions pures de `src/arcC_calibration.py` (§C7) sont le contrat :
`pixels_par_degre` (longueur de référence px/mm + distance), `taille_domaine_px`
(porteuse 5.5 cyc/domaine au pic CSF 3 c/deg), `plafond_texture` (acuité
~1 arcmin). **Les VALEURS (écran, distance) sont des entrées de session,
re-mesurées à chaque session de mesure** — application directe de la règle
d'hygiène des faits d'instrument. Le report `observation_cellule_pic_csf` reste
obligatoire.

---

## 3. Coûts au papier `[CALCUL — enveloppes, PAS des prédictions]`

- **Noyau par pixel** : étages 2–4 ≈ 5–10 flops/px (identité + pow sRGB +
  quantif) ; +rééchantillonnage bilinéaire ≈ 10 flops/px. À l'échelle instrument
  (64² → ~10⁴ px affichés) : **négligeable, microsecondes CPU** — le coût de P1
  n'est PAS un coût machine, c'est un coût de build et de décisions.
- **Enveloppe V4, pour situer P2** : 1920×1080 ≈ 2.07·10⁶ px × ~30 flops ≈
  60 MFLOP/frame ; borne bande passante ~35 Mo/frame ≈ 0.15 ms à 224 Go/s
  (3050 Ti). L'enveloppe dit UNE chose : **P2 est achetable** (une mesure courte
  suffira) — elle ne dit pas que le rendu est gratuit. L'histoire du projet
  (MORT-a : ×42 entre papier et mesure) interdit de croire une enveloppe ; P2
  existe pour ça. Et P2-sur-B exclut la composition (§0, dit deux fois à dessein).
- **Build** `[ESTIMATION, à confronter au chiffrage Claude Code]` : Option B
  0.5–1 séance ; Option A +0.5–1. Cap de la maison : chiffrage > 2 séances ⇒
  remonter avant achat.

---

## 4. Ce que P1 sert, nommément (D17, en toutes lettres)

| livrable P1 | décision nommée qui le consomme |
|---|---|
| noyau R1 validé (tests §5 verts) | **achat de P2** (chiffrage sur un instrument validé, jamais l'inverse) |
| chemin d'affichage drop-in dans la coquille ABX | **P3** : rejouer les paires (seed, L) × stim(t) d'Arc C à travers R1 — la mesure qui tranche le caveat D14 |
| rééchantillonnage et EOTF gravés | la LECTURE de P3 (sans eux, un transport ≠ 1 serait inattribuable : pin ou chemin ?) |
| interface périmètre-neutre | l'achat du compositeur post-P3 (Option A différée) sans re-travail |

Aucune autre mesure n'est ouverte par P1. b-vs-c, (γ), le profil radial : dormants,
rien ici ne les consomme.

---

## 5. Falsifieurs et critères d'acceptation (avant tout chrono, avant P2)

1. **Déterminisme** : deux appels de R1 sur le même `A` ⇒ images bit-identiques
   (même machine). Consommateur : la re-jouabilité des sessions P3.
2. **Stabilité** : `z` stable ⇒ suite d'images bit-identiques (anti-PERSIST,
   §A20-1 testé).
3. **Test à blanc** : `A_cand ≡ A_ref` ⇒ stimuli présentés identiques ⇒ l'ABX est
   au hasard par construction (l'analogue du « readout propre, Δχ = 0 exact » de
   l'état des lieux §1.5).
4. **Cohérence inverse de calibration** : la porteuse affichée retombe à
   `c_deg_cible` ± l'arrondi documenté (le test existant de
   `test_arcC_calibration.py`, étendu au chemin complet R1).
5. **Monotonie de la chaîne** : A₁ < A₂ (partout) ⇒ Y₁ ≤ Y₂ après quantification —
   la luminance ne peut pas inverser l'ordre de l'albédo (ce que viridis
   garantissait par conception, R1 doit le garantir par test).
6. **Garde d'acuité** : `plafond_texture` vérifié à la géométrie de session,
   report obligatoire — sinon l'artefact de blocs est un stimulus caché.

Chaque test est pur, sans GPU, exécutable partout ; la VM Cowork suffit pour les
développer, **le verdict P3 restera iluin-tworings3 natif** `[GRAVÉ]`.

---

## 6. Ce que P1 NE fait PAS (portées, refus explicites)

- **Pas de son** — co-égal en thèse, absent de l'arc, nommé (note v2 §8) ; « co-égal »
  se lit en JUGEMENT, pas en FALSIFIABILITÉ (**§A45**, 2026-08-03).
- **Pas de solve d'équilibre** ni d'ombres : R1 est l'instrument, pas le rendu de
  production (§2.2.3). O-Voxel/TRELLIS.2 : écarté pour l'instrument (une chaîne
  →mesh n'est ni minimale ni au format du harnais) ; reste la lecture différée à
  graver (candidat couche readout de production, réexamen post-P3).
- **Pas de pondération d'excentricité** : c'est le juge du gate (iii′), il attend
  l'échelle que P3 lui donnera ; la spécifier avant serait écrire le juge sans son
  étalon.
- **Pas de compositeur** si Option B est retenue (§0) — incrément post-P3, avec la
  décision de couture prise le pin transporté en main.
  *(**PRÉCISÉ le 2026-08-03 — §A48** : ce refus vise le compositeur **INSTRUMENT**,
  celui qui tranche la couture et se juge au pin transporté ; il reste entier. Il ne
  vise pas le **chemin-de-coût** de la tranche-moteur, jugé au coût seul et qui ne passe
  jamais sous un ABX — quatre gardes en §A48.)*
- **Pas de re-mesure du pin** : jnd_sev 7.33 % reste le pin gravé ; P3 mesure son
  TRANSPORT, pas sa valeur.
- **Pas de temps réel** : P1 est un instrument de session ABX ; le 60 fps, le
  GPU, l'interpolation de readout (porte 33.3, §A18) n'entrent pas ici.
- **Les six pistes de la note v2** : aucune n'apporte de solution à P1 (vérifié
  piste à piste — sauter/tourner, ledger-auteur, versionnage F, rembobinage,
  checklist, hystérésis relèvent de F, du ledger ou du post-P3). Elles se gravent
  avec l'entrée journal pour mémoire, elles ne modifient pas cette spec.

---

## 7. Séquence après revue (aucun enchaînement automatique)

1. Revue de cette spec par Romain — tranche D-P1-1 (périmètre), D-P1-2
   (rééchantillonnage), D-P1-3 (chemin lumineux), D-P1-4 (EOTF).
2. Gravure de l'entrée journal P1 (avec les trois gravures en attente du
   préambule) — par la session, commit local ; push Romain.
3. Ordre de mission Claude Code (`mission-p1-rendu-instrument.md`) — RIEN ne se
   code avant.
4. Tests §5 verts → remonter → décision d'achat P2 (pré-enregistrement séparé,
   chrono court, natif).

**POINT D'ARRÊT : cette spec s'arrête ici. Rien ne se construit avant revue.**

---

## REVUE 2026-07-25 — DÉCISIONS TRANCHÉES (Romain) ; la spec passe à ENDOSSÉE

- **D-P1-1 : OPTION B** — pleine résolution d'abord, noyau R1 seul ; interface
  périmètre-neutre gravée ; compositeur = incrément post-P3, décision de couture
  prise le pin transporté en main. Conséquence P2 (§0) confirmée telle quelle.
- **D-P1-3 : (a)** — chemin lumineux identité `Y = A` ; lambertien = incrément R2
  post-P3, avec sa propre mesure de transport s'il est acheté.
- **D-P1-4 : (a)** — inverse-EOTF sRGB gravé en formule ; linéarité supposée sRGB,
  jamais surclamée.
- **D-P1-2 : (a)** — rééchantillonnage bilinéaire gravé comme étage de R1, testé,
  plus jamais délégué à l'affichage.

Gravure journal : §A37 (pocCascade2phys). Prochaine étape de §7 : ordre de mission
Claude Code, **sur décision explicite** — rien ne se code avant.

**CORRECTION EXPLICITE (2026-07-25, après endossement — rien n'est réécrit)** :
l'ordre des étages 3-5 du tableau §2.3 (encodage → quantification →
rééchantillonnage) est **REMPLACÉ** par l'ordre amendé de l'ordre de mission,
tranché par Romain : **rééchantillonnage bilinéaire en espace LINÉAIRE (= A,
puisque Y = A) → encodage sRGB → quantification uint8 unique en bout de chaîne**.
Motif : pas d'interpolation de valeurs quantifiées, pas d'interpolation en espace
gamma ; une moyenne de luminances est une luminance (l'intention de D-P1-4).
Épistémique consignée : décision sans a priori, re-jugeable au résultat — l'ordre
des étages est loggé dans la provenance de chaque session (attribuable si P3
rend un transport ≠ 1).
