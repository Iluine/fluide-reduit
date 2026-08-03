# SÉANCE-TABLE — entrée : vocabulaire, budget (c, b) par (champ, niveau), frontières du moteur (2026-08-03)

> **Statut : PAPER GRADE, document d'ENTRÉE de séance. Aucun run, rien de commité.**
> Exécute les impositions **1** et **2** de `note-budget-vram-2026-07-18.md` (« Épingler
> (c, b, N_niv, n_fov^d cible) — le quadruplet EST le budget » ; section GPU-déterminisme
> obligatoire) — jamais exécutées à ce jour. L'imposition 3 (tranche-moteur dans
> l'enveloppe) l'a été : F1/V4, §A18.
> Le journal `pocCascade2phys/PREREGISTRATION.md` fait foi (fin de fichier : **§A47**).
> Ce document raisonne sur lui et ne le remplace pas.
> Rédigé par la session Claude (critique) sur demande de Romain (2026-08-03) ;
> **l'endossement de chaque [PROPOSITION] appartient à Romain.**
>
> Étiquetage : `[MESURÉ]` / `[GRAVÉ]` / `[LU-DANS-LE-CODE]` / `[CALCUL]` /
> `[TRANSPOSITION-HYPOTHÈSE]` / `[PROPOSITION]` /
> `[DÉCISION-ROMAIN-CHAT 2026-08-03 — à graver au journal avec l'endossement]`

---

## §0. Décisions de cadre du 2026-08-03 (chat, à graver)

1. **Machine cible : non rouverte.** « La 3050 Ti est le GPU minimal pour 1920, pas une
   machine 4K » `[GRAVÉ : annexe-enveloppe-jeu-2026-07-19.md:48-49]`. L'enveloppe se lit
   contre cette cible.
2. **La frontière de F.** F = les règles **physiques** du monde — jamais le comportement
   des entités, PNJ ou joueurs ; ceux-ci relèvent de « règles du jeu »
   `[DÉCISION-ROMAIN-CHAT]`. Conséquence : le pari « un seul F » porte sur la matière ;
   l'agentivité hors-champ (couche stratégique) est un système séparé dont les événements
   entrent au **même ledger**. Aucun échec futur de la couche agents n'est imputable au
   pari F.
3. **Multiplayer = multi-PC**, pas d'écran partagé : le calcul fin est porté par la
   machine de chaque observateur `[DÉCISION-ROMAIN-CHAT]` (conséquences : §8).

---

## §1. Le quadruplet devient une table

La séance ne produit pas un quadruplet scalaire : elle produit une **table
champs × niveaux**, dont c(niveau), b(champ, niveau) et le quadruplet sortent
mécaniquement. Justifications (session 2026-08-03, vérifiées sur artefacts) :

- **b s'indexe par (champ, niveau)**, pas par champ : la prédiction de Harten fait
  remonter le contenu grossier dans la reconstruction fine — un champ du bilan stocké
  f16 à un niveau grossier contaminerait sa reconstruction si la fenêtre s'y déplace.
- **Trois contraintes SÉPARÉES, à ne pas fusionner** (chaque fusion produit une
  pathologie : interdiction trop large ou trou) :
  - **(i) déterminisme d'ordre** — contrainte sur les *kernels du chemin de commit*,
    orthogonale à b. Déjà satisfaite par l'architecture : la re-dérivation est
    « CPU f64 pur, hors-frame par construction » `[GRAVÉ : SPEC-FOVEA-Z.md:288]`.
    Elle vit à la section commit, pas dans la table.
  - **(ii) précision d'accumulation** — contrainte sur le *bilan* : f16 est déterministe,
    son mal propre est la dérive d'accumulation. Réparable sans toucher au stockage
    (accumulateurs f32 sur stockage f16, sommation compensée).
  - **(iii) exactitude moyenne-nulle** — contrainte sur la *représentation* : des détails
    quantizés f16 acquièrent une moyenne résiduelle. Réparable **par construction**
    (quantifier-puis-corriger au store, ou layout type lifting). La propriété parent =
    moyenne(enfants) est structurelle dans le stockage coarse+détails (« la montagne ne
    peut pas flipper, par construction » `[GRAVÉ : note-orientation-v2 §3]`) ; l'objet
    fragile sous quantization est la moyenne nulle des détails, pas l'identité parent.
- **Colonnes de statut par (champ, niveau)** : *au bilan* (linéaire / non-linéaire /
  non) ; *moyenne-nulle* (schéma de store, pas interdiction) ; *réchauffable(τ_dec)* —
  dynamique, voir ci-dessous.
- **« Réchauffable » n'est pas statique.** Fovéa mobile ⇒ aucune région n'est
  géométriquement condamnée au froid. Ce qui rend un niveau définitivement
  non-réchauffable, c'est le ledger : région committée + phase fine au-delà de τ_dec ⇒
  la re-dérivation passe par (ledger + tirage semé), plus par le grossier stocké.
  Réchauffable = f(champ, région, temps depuis commit), gouverné par **τ_dec**.
  **Conséquence : le budget se lit en fonction de τ_dec, avec branches écrites — pas
  comme un chiffre unique.** La sonde τ_dec a désormais des consommateurs nommés
  (§A46 ; §6 et §8 ci-dessous) : règle D17 satisfaite.

---

## §2. La règle du commit découplé `[LU-DANS-LE-CODE — à graver comme invariant]`

> **f16 permis au stockage vivant ; le chemin de commit calcule son résumé HORS de la
> précision de stockage (f64 hôte) avant d'écrire au ledger.**

Vérification (2026-08-03) : l'état vivant est transféré hôte (`src/f1_gpu/backend.py:66`
`.get()`, tampons épinglés `src/f1_gpu/transferts.py`) ; `summarize_qt` **force une copie
float64 avant tout calcul** (`src/summary_quadtree.py:79-86`, « COPIE float64 (jamais une
vue — anti-fuite) »), moyennes par blocs en f64 (`:95`), stockées f64 au ledger
(`src/f1_gpu/ledger.py:164`). Aucun résumé
n'est calculé GPU-side dans la précision de stockage : la règle est le motif actuel du
code.

**Portée exacte de la vérification** : ce motif est celui du **harnais** —
`_valider_champ` est câblé sur un domaine FIXE 64×64 (`src/summary_quadtree.py`) — il
n'existe encore **aucun chemin de commit moteur** pour en hériter. « Lu dans le code »
est vrai et étroit. Raison de plus de graver la règle comme invariant : faute de quoi le
futur chemin moteur naîtra sans elle.

Appuis de contrat : É1 contraint la **reconstruction** (« préservés EXACTEMENT —
propriété de projection du quadtree » `[GRAVÉ : PREREGISTRATION.md:3258]`), qui est CPU
f64 de bout en bout `[GRAVÉ : SPEC:288]` ; « le VIVANT est contracté au TÉMOIGNAGE »
`[GRAVÉ : §A36 P0-b clause 2, :6494]` — un vivant f16 ne trahit aucun référent qu'il n'a
plus. Ce que f16 peut abîmer : la dérive **visible** du bilan vivant (une rivière qui
maigrit) — faute de plausibilité, jugée au perceptuel, pas par É1.

**Garde** : l'imposition 2 de la note VRAM pose « où vivent les commits (CPU-side ?) » —
le jour où le commit passe GPU-side pour la perf est le seul où ce découplage casserait
en silence. D'où l'invariant + son test de propriété (annexe §12, n°1) : *résumé d'un
champ stocké f16 ≡ moyennes f64 des valeurs f16, exactement.*

Conséquence pour la table : la colonne « au bilan » n'interdit aucun stockage — elle se
réduit à la dérive visible (perceptuel) + l'invariant de commit. **Le levier b s'ouvre
bien plus largement que l'enveloppe ne le supposait.**

---

## §3. Vocabulaire minimal `[PROPOSITION — à endosser, amender ou rayer champ par champ]`

Les classes τ_dec sont des **attentes** ; la sonde tranchera. La table est écrite en
vocabulaire 2.5D (shallow-water + Exner, le substrat existant) ; la migration 3D vraie
(dette « 3D ») peut changer la ligne eau — la réserve (§4) couvre aussi cela.

| # | Champ | Rôle | Bilan | Classe τ_dec | Niveaux | b | éq-f32 fin |
|---|-------|------|-------|--------------|---------|---|-----------|
| 1 | `b` lit/terrain | la montagne, le creusement (Exner) | masse, linéaire | mémoire longue (LE champ commis) | tous | f32 | 1 |
| 2 | `id-matériau` | catégoriel : roche/sable/eau/bois + flags (conducteur, inflammable) | aucun | quasi-infini (change par événements) | tous | u8 | 0,25 |
| 3 | `h` hauteur d'eau | surface libre | masse, linéaire | court (détails), long (niveau moyen) | tous | f32 fovéa, f16 large | 1 |
| 4 | `hu, hv` (+`hw` 3D) | impulsion | impulsion, non-linéaire | court | tous (le courant se lit de loin) | f32 fovéa, f16 large sous schéma moyenne-nulle | 2–3 |
| 5 | `e_th` énergie thermique | chaleur, feu — stockée en ÉNERGIE, pas en T (c'est le bus que la magie source) | énergie, non-linéaire | moyen (braises) | fovéa + présence grossière | f16 stockage, résumé f64 | 0,5 |
| 6 | `e_ch` combustible/chimique | ce qui peut brûler/réagir ; sa déplétion EST le char | masse/énergie, linéaire (converti) | long (persistant) | tous | f16 | 0,5 |
| 7 | `ρ_s` fumée | panache advecté | aucun (jamais commis) | **éphémère** — vivant, jamais au registre | **fovéa seulement** | f16 | 0,5 |
| 8 | `u_atm` vent/météo | forceur phénoménologique | aucun (prescrit) | long | **grossiers seulement** | f16 | 0 |

**Somme fovéa-fin, PAR DIMENSION** (u8 = 0,25 ; f16 = 0,5) : **2.5D = 5,75 éq-f32**
(table ci-dessus, ligne à ligne) ; **6,75 avec `hw`**. En 3D vraie, les lignes 3–4
n'existent pas telles quelles : il faut une fraction volumique ou ρ, trois composantes
de vitesse, et selon le schéma une pression — le vocabulaire de **base** 3D atterrit
vraisemblablement à **~7–9 éq-f32** `[NON-ANCRÉ : schéma eau 3D non choisi — dette
« 3D »]`. c_fov ≈ 8 de l'enveloppe reste l'ordre de grandeur que le vocabulaire minimal
donne, à condition que 7–8 soient scopés par niveau et que l'albédo reste côté β.

**Contre-liste — nommé mais HORS c :** albédo/χ (readout du sédiment/char — cache **β**,
le pin d'Arc C le mesure comme tel) ; vorticité, pression (dérivés, β) ; magie (terme
source sur le bus) ; vagues d'étrave, écume (éphémère de readout pur, régénéré par
frame). Électricité, gravité : voir cases, §5.

**Les deux lignes pour l'œil de Romain :** (5) stocker l'énergie plutôt que la
température — bilan linéaire en stockage, lecture plus complexe ; (7) fumée « vivante
mais jamais commise » — présume qu'un panache disparu hors-fovéa est acceptable :
question τ_dec/revisite qu'il peut trancher autrement.

---

## §4. Équivalents-f32, réserve, règle d'admission `[PROPOSITION]`

1. **La table compte en équivalents-f32 par niveau** : Σ b(champ)/4. C'est cette somme
   que l'enveloppe lit — c et b y fusionnent définitivement.
2. **Épingler minimum + réserve nommée : ~4 éq-f32 de réserve — mangée D'ABORD par la
   migration 3D du vocabulaire de base** (fraction/ρ, 3 composantes, pression : ce n'est
   pas de la physique nouvelle), le reste pour les milieux advectés futurs (lave, gaz
   lourd, fractions multi-phases…). Lecture de l'enveloppe 3D à ~11 éq-f32 : 64³ ≈
   **2,89 Go** `[CALCUL : formule note VRAM, linéarité en c·b]`, laissant **~1,1 Go**
   pour rendu + framework (contre ~1,9 Go à c=8) — et le rendu est **NON MESURÉ des
   deux côtés** (§A20-5 ; annexe:57). L'enveloppe ne dit pas « passe » : elle dit
   *passe si l'autre moitié tient dans 1,1 Go*, et personne ne le sait. Le nombre, pas
   le verdict — même régime que la dette p99. 96³ reste CASSE à ce compte.
3. **Règle d'admission** : *aucune physique n'entre sans nommer sa case d'abord ; seule
   la case chère (§5.8) débite la réserve.* Le budget ne se re-négocie pas à chaque
   idée ; les « indispensables pas encore nommés » ont une place bornée.

---

## §5. Les cases — taxonomie d'admission `[PROPOSITION]`

| Case | Mécanisme | Exemples | Coût fin |
|------|-----------|----------|----------|
| 1 | **Solve par état** (elliptique, quasi-instantané à l'échelle de F) | lumière `[GRAVÉ §A20-2]` ; électricité (Kirchhoff sur le sous-graphe conducteur de l'u8) ; tenue quasi-statique (JAX-FEM dans la stack) | 0 |
| 2 | **Terme source sur le bus d'énergie** | magie `[GRAVÉ]` ; dissipation électrique ; conversions | 0 |
| 3 | **Catégoriel** (u8 id-matériau + flags) | matériaux, états discrets, conducteur/inflammable | 0,25 partagé |
| 4 | **Forceur grossier-seulement** | u_atm ; gravité *variable* (zones) — le g constant est un paramètre de F, pas un champ | 0 au fin |
| 5 | **Cache matérialisé β** | albédo, vorticité, pression | côté β |
| 6 | **Événement + noyau à forme close semé** (pulses rejoués depuis seeds, §A13 — *déjà dans le code*) | explosions/blast, foudre, séismes, tout l'impulsif | ~0 (ledger) |
| 7 | **Entités lagrangiennes** (hors z, coût par-entité) | débris, troncs, charrettes, créatures, LE JOUEUR | hors budget cellulaire (§6) |
| 8 | **LA CASE CHÈRE : milieu continu advecté avec impulsion propre au niveau fin** | lave, gaz lourd, fractions multi-phases | **débite la réserve** |

**Chaînes de cases** : la rupture structurelle = solve (1) → seuil → événement (6) →
entités (7) → redépôt eulérien (§6). Une physique peut habiter une chaîne, pas une case.

---

## §6. Entités lagrangiennes — le cycle E→L→E `[PROPOSITION]`

- **Cycle** : rupture (événement : conversion masse z → entités) → vol → stabilisation
  (événement : conversion entités → z ; bosse sur `b`, voxels id=roche à l'échelle
  cellule, éboulis granulaire ou albédo sous-maille).
- **DEUX écritures comptables au ledger** encadrent le cycle : masse quittant z = masse
  des entités = masse redéposée. Famille É1. Sans elles : fuite de masse silencieuse.
- **Le vol n'est pas du témoignage.** Trajectoires de contact chaotiques ⇒ seul le
  *résultat* est commis, re-dérivable depuis (événement, graine). Interaction joueur en
  vol = événement (seul cas où le vol laisse trace).
- **Observateur absent : pas de vol simulé** — tirage semé du résultat, contraint par le
  ledger (le mécanisme de P0-b, appliqué aux débris).
- **LE TIRAGE PRODUIT UNE TRACE, pas seulement un état** : (état final, trace d'impacts
  datés/localisés/magnitudes, trajectoires en forme close balistique+rebonds). L'audio
  impulsif se synthétise depuis la trace — mot pour mot le re-scoping §A46 (« fonction
  déterministe de (événements du ledger, t) »). **Cohérence son↔débris par même graine,
  par construction** : le bloc entendu à gauche sera trouvé à gauche.
- **Deux prédicats d'absence** (conséquence §A45/§A46) : absent-des-yeux (cône) ≠
  absent-des-oreilles (sphère, occlusion, derrière la tête).
- **Trois grades de promotion**, testés trajectoire par trajectoire contre les régions
  d'observation sur [t_rupture, t_posé] :
  1. **tiré** — ne croise rien : état final seul ;
  2. **entendu** — croise la sphère auditive seule : son synthétisé *depuis la
     trajectoire tirée*, rien ne se simule ;
  3. **vivant** — croise le cône visuel ou la zone d'interaction : lagrangien réel
     (on doit pouvoir l'esquiver).
- **Une entité, une autorité** : à la promotion, le vivant supersède le tiré *pour cette
  entité seule* ; le tirage est révisable par-entité, pas monolithique ; la masse reste
  bornée par les deux écritures quel que soit le chemin.
- **La promotion est une couture** — le « transitoire en vol » nommé par §A46 : premier
  cas d'usage concret du **second consommateur de τ_dec** (tolérance ~ms).
- **Garde anti-sur-ingénierie** : les trajectoires tirées n'ont besoin que de
  plausibilité perceptuelle (F illusionniste sur les mécanismes, comptable sur les
  bilans) — le critère de douceur de couture est perceptuel, mesuré par la sonde, pas
  balistique.

---

## §7. Le bus d'énergie — forme « vie », clause d'admission des formes `[PROPOSITION]`

- **e_vie comme forme du bus** si manipulable par le joueur (drain druidique, sol épuisé,
  zone morte — la comptabilité de conversion rend tout cela non-contradictoire et
  re-dérivable gratuitement). Sinon : forceur case 4 (carte de biome). **Critère :
  manipulable par le joueur ⇒ bus.** `[DÉCISION-ROMAIN : confirmée dans son principe en
  chat (« énergie de vie qui pilote la vitesse de croissance ») ; la variante bus-vs-
  forceur reste à trancher.]`
- **CLAUSE D'ADMISSION DE TOUTE FORME DU BUS** : elle hérite de l'obligation de **forme
  close par morceaux** — la végétation est jumpable (croissance en forme close pendant
  l'absence) ; si sa vitesse devient f(e_vie(t)), le jump ne survit que si e_vie est
  constant par morceaux entre événements, grossier, lent. *Une forme finement fluctuante
  détruit la jumpabilité de tout ce qu'elle pilote.*
- **Variante météo-comme-événements** (alternative à la ligne 8 de la table) : fronts
  semés à trajectoire en forme close, évalués à la demande, zéro cellule stockée,
  interrogeables par les témoins (§8). Même choix bus-ou-forceur que e_vie, à Romain.

---

## §8. Multi-observateur, multi-PC `[PROPOSITION]`

- **Observateurs perceptuels vs testimoniaux.** Perceptuel = fovéa + sphère auditive,
  coûte n_fov^d : **les joueurs humains seulement**. Testimonial = PNJ : lecteurs
  d'événements du **ledger** — un PNJ « voit » l'effondrement comme un événement commis
  dans sa région, pas comme des pixels. Sa non-contradiction se maintient en
  espace-événements (ordres de grandeur moins cher), son témoignage cite la source qui
  fonde la re-dérivation : exact par construction. Le grade 3 (cher) ne se multiplie pas
  avec la population du monde.
- **Multi-PC** `[DÉCISION-ROMAIN-CHAT : pas d'écran splitté ; le fin est calculé par la
  machine de chaque observateur]` :
  - **Partagé = ledger + niveaux grossiers. Fin = élaboration locale**, licite par
    contrat (« le VIVANT est contracté au TÉMOIGNAGE », P0-b clause 2 ; au-delà de τ_dec
    la vérité fine est épistémiquement vide). Deux joueurs devant la même cascade :
    deux réalisations fines de la même vérité grossière.
  - **Le ledger est assez petit pour être le protocole réseau** : ~3,2 Ko/commit,
    ~11,5 Mo/h `[CALCUL : note VRAM §ledger]`. L'argument de boundedness change de
    métier sans changer d'un octet.
  - **Séquenceur unique** : clé (t_sim, seq) strictement croissante ⇒ un log append-only
    n'a qu'un écrivain — une autorité de séquencement (serveur, éventuellement la
    machine d'un joueur) tient le ledger canonique + le tick grossier. **Anti-triche
    structurel** : le fin est cosmétique par contrat ; pour mentir sur le monde il faut
    faire commettre un événement, et l'événement est séquencé.
  - **Budget : l'enveloppe reste vraie PAR CLIENT** (chaque PC paie son n_fov^d) — le
    multi-PC est le seul modèle multi compatible avec la table §4.
  - **É3 inter-machines acquiert un consommateur.** Amendement 2 : « É3 est
    MACHINE-LOCALE… inter-machines, sous-JND — NON MESURÉ », falsificateur marqué
    GRATUIT `[GRAVÉ : SPEC:292]`. Deux fovéas chevauchantes sur deux machines = l'écart
    inter-machines lu en espace readout : la mesure dormante a désormais sa décision
    (règle D17).
  - **Transfert d'autorité** (entité passant de la région de A à celle de B) = couture,
    troisième cas d'usage du second consommateur τ_dec.
  - **Nommés, non résolus** : latence des conditions aux limites grossières (F
    illusionniste tolère — « mieux » n'est pas un chiffre) ; chevauchement d'écritures
    persistantes (deux joueurs, même trou : résolu par le séquenceur, au prix de latence
    perçue).
- **Conséquence build** : PNJ, quêtes, filtre de promotion, synthèse audio — tout
  *interroge* l'historique en boucle de jeu ; le ledger actuel (jsonl+bin, parse
  intégral au load) est une archive de replay, pas une base de requête. **Un modèle de
  lecture (index, vues matérialisées) se prévoit dans la tranche, ne se retrofitte
  pas.**

---

## §9. Génération = tirage de naissance à t=0 `[PROPOSITION]`

Le générateur de monde n'est pas un système à part : c'est le **tirage de naissance**
(P0-b) évalué avant le premier observateur. La naissance du monde = **entrée n°0 du
ledger**, semée, re-dérivable comme le reste. Générer et raffiner s'unifient.

**Contrainte propre : plausibilité-sous-F du passé implicite.** Le monde de départ porte
une histoire fabriquée (vallées, deltas, éboulis) qui doit ressembler à ce que F aurait
produit — sinon la première érosion réelle contredit le passé, une contradiction avec un
témoin d'avant les témoins. L'illusionniste appliqué au passé : les mensonges de F
doivent être cohérents entre eux, y compris avant t=0. Critère perceptuel (checklist
sans référence), pas test automatique.

---

## §10. Le gouverneur — jumeau dynamique de l'enveloppe `[PROPOSITION]`

L'enveloppe borne le pire cas statiquement ; en jeu, le pire cas arrive (incendie
généralisé + rupture + douze entités promues dans la fovéa). Sans politique écrite, la
réponse sera un freeze — la pire au sens perceptuel.

- **Boutons, tous déjà chiffrés** : n_fov (cubique), J, seuil de promotion, scope fumée.
  Ancre : **6,43 ns/cellule/frame** `[MESURÉ : M-a′, §A15]`.
- **Boucle** : mesurer la frame → dégrader *avant* le débordement → ré-élargir après.
  Le dépassement de budget devient une baisse de LOD momentanée.
- **La hiérarchie de sacrifice est une décision de design perceptuel — À ROMAIN.**
  Proposition d'ordre : fumée → le large → la promotion (grade 3 → 2) → **jamais le
  bilan ni les invariants commis**.

---

## §11. Clause de contrat : versionnage de F `[PROPOSITION]`

Le contrat de re-dérivation épingle silencieusement la **version de F** : un patch qui
change F rend l'historique re-dérivable vers un monde différent de celui quitté. La
machinerie de sortie existe : `ajouter_snapshot` — « commit PLEIN sans perte »
`[LU-DANS-LE-CODE : src/f1_gpu/ledger.py:166-172]`. **Clause** : à chaque migration de version,
snapshot-racine puis coupure ; l'histoire d'avant devient préfixe gelé, rejouable sous
l'ancien F seulement. Une ligne au contrat du registre maintenant, pas une crise de
compatibilité plus tard.

---

## §12. ANNEXE — Registre des invariants nés de la session (chacun s'attache à SON build)

La tranche telle que cadrée (z + registre + F + rendu naïf) n'a ni entités, ni multi,
ni gouverneur : **elle ne prend que le n°1** (plus l'extension des n°10–11 existants).
Les autres s'activent le jour où leur build existe — le registre est là pour qu'ils ne
s'évaporent pas d'ici là, pas pour re-gonfler la phase pré-build.

| # | Invariant | Source | Test de propriété candidat | Statut |
|---|-----------|--------|---------------------------|--------|
| 1 | Le résumé de commit se calcule hors précision de stockage | §2 | résumé(champ f16) ≡ moyennes f64 des valeurs f16, exactement | code conforme `[LU]`, test à écrire |
| 2 | Moyenne-nulle des détails sous quantization | §1(iii) | store f16 → reconstruire → masse parent exacte | à écrire (schéma de store à choisir) |
| 3 | Deux écritures comptables encadrent E→L→E | §6 | masse(z perdue) = Σ masse(entités) = masse(z redéposée) | à écrire |
| 4 | Une entité, une autorité (jamais deux simultanées) | §6, §8 | promotion et transfert : l'ancien état est révoqué atomiquement | à écrire |
| 5 | Trace et état final sortent de la même graine | §6 | cohérence son↔débris sur re-tirage | à écrire |
| 6 | Toute forme du bus est en forme close par morceaux | §7 | jump(intervalle) ≡ intégration pas-à-pas, aux événements près | à écrire |
| 7 | Séquenceur unique : (t_sim, seq) strictement croissant multi-clients | §8 | déjà garanti à l'APPEND et re-vérifié au PARSE `[LU : f1_gpu/__init__.py B7]` — à étendre au multi-écrivain | partiel |
| 8 | Le gouverneur ne touche jamais les invariants commis | §10 | toute dégradation LOD préserve É1 | à écrire |
| 9 | Prédicat d'absence par readout (yeux ≠ oreilles) | §6 | aucune promotion manquée : trajectoire ∩ région ⇒ grade ≥ 2 | à écrire |
| 10 | Parents exacts par construction (« la montagne ne flippe pas ») | `[GRAVÉ : note-orientation §3]` | structurel — test H existant à étendre aux stores f16 | existant |
| 11 | Invariant de projection : aucun readout load-bearing | `[GRAVÉ : §A20-1]` | existant (§A20) — à étendre au canal auditif impulsif (§A46) | existant/partiel |
| 12 | Plausibilité-sous-F du monde généré | §9 | checklist sans référence — critère perceptuel, PAS automatisable | protocole à écrire |

---

## §13. Ce que la séance doit produire, et ce qui reste dû

**Livrables de la séance** (une session de design, zéro mesure) :
1. La table §3 endossée/amendée champ par champ ; le quadruplet s'en déduit.
2. La lecture de l'enveloppe 3D à (minimum + réserve), **branches écrites avant
   lecture** — en fonction de τ_dec (§1), contre la machine cible (§0.1).
3. Les décisions §0 gravées au journal ; les [PROPOSITION] endossées ou rejetées.

**Fait au moment de l'écriture** (vérifié disque, 2026-08-03) : les deux commits
d'hygiène sont **COMMITTÉS** — `pocCascade2phys` **ac30788** (§A45/§A46/§A47 +
pointeur SPEC §9), `pocPhysicator` **9be386e** (distinction ×3 + P0-b périmée). Ne pas
les refaire ; ne pas ré-appendre §A45/§A46 au journal, qui les porte déjà.

**Reste dû** : **P0-son** (décision Romain) ; **l'échange compositeur** ; **la
séance-table** (ce document en est l'entrée) ; **la tranche verticale** — avec ses
slots issus d'ici : test invariant n°1, FSI minimal (une entité, un couplage, deux
sens), esquisse du modèle de lecture ledger, squelette de gouverneur. La tranche reste
la seule source d'information que le papier ne peut pas produire : le coût du rendu,
l'autre moitié du chiffrage.
