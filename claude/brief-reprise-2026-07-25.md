# Brief de reprise — projet Cascade (2026-07-25, après la vérification d'échelle)

> **Ce brief remplace `brief-reprise-2026-07-19.md`, périmé.** Depuis : la séance É2
> a été tenue (`seance-e2-2026-07-24.md`, auditée, 11 écarts corrigés, amendée le 25),
> et le soupçon d'instrument de son §0 a été **CONFIRMÉ PAR LECTURE DE CODE**
> (`verif-echelle-grossier-2026-07-25.md`). Ce brief sert aussi de consolidation pour
> la mémoire du projet Claude (l'état du 15/07 y est périmé de dix jours).

## Rôle et règles d'engagement

Critique senior adversarial, en français. Romain décide seul (il lit le code, n'écrit
pas le Python) ; Claude Code implémente ; la session lit, challenge, pré-enregistre,
revoit. **JAMAIS d'enchaînement automatique après un verdict ou une lecture** — point
d'arrêt, remonter, décision de Romain.

## Sources de vérité (dans cet ordre)

1. **pocCascade2phys** (`t1-spring-config`) : `PREREGISTRATION.md` — contrat + journal
   append-only, **fin de fichier §A33** ; `SPEC-FOVEA-Z.md` (9 sections + rev1/rev2/§9).
2. **pocPhysicator** (`arc-a-manche2-registre`) : `claude/lectures/*.lecture.json`
   (les lectures versionnées — `outputs/` est gitignoré) ; `claude/etat-des-lieux-2026-07-19.md`
   (MESURÉ / GATÉ / DETTE / JAMAIS TOUCHÉ) ; les documents de séance.
3. Ce brief rend visible ; **il ne fait foi de rien**.

## État en une ligne

**Gate (i) SATISFAIT avec marge** (V4 à 14.490 ms sur 16.7, marge 2.199 ms — §A23-2b ;
les valeurs 16.589/0.111 sont PÉRIMÉES) ; **gate (ii) F0-cloud déféré et déprioritisé** ;
**gate (iii) : attributions de §A32/§A32-décomp/§A33 SUSPENDUES (§A33-CORRECTION) — le
défaut d'instrument est CONFIRMÉ PAR EXÉCUTION le 25/07** (err(Δx=1)=3.3e-05 vs
err(Δx=2)=0.500) : le grossier de T2 tournait à mauvaise échelle. **D19 est TRANCHÉ**
(a: test fait ; b: suspension gravée ; c ENDOSSÉ : kernel Δx-conscient, T2 re-couru).
**Le re-run T2 (25/07, instrument validé) a RE-PRONONCÉ MORT-b, PLUS LARGE (§A34)** :
production 1.47–2.42 (20–33× le pin, AU-DESSUS de la plage de corruption délibérée),
**témoin seed 101 TRAVERSE (0.07731 > 0.0733)**, Option B n'est pas le remède. Le FAIT
est solide ; les attributions fines (b vs c, « structurel », « par le halo ») restent
suspendues, non re-mesurées. **La séance D14–D18 est CLOSE (§A35)** : scission de É2
endossée (état/projection, caveat du transport-par-construction gravé) ; **gate (iii)
ÉCHOUÉ sur É2-état, prononcé** ; **gate (iii′) NÉ sur É2-projection** — la moitié
projection (image d'abord) est LE CHEMIN CRITIQUE de sa mesure ; la spec fovéa-z ne
fait pas foi en l'état ; non-transposabilité gravée sur 1.47–2.42 (Π ≫ 1) ; règle
gravée : aucun diagnostic sans décision qui en dépend. **P0 TRANCHÉE (§A36)** : C-uni mort comme
contrat de fovéa (vivant comme étalon du harnais) ; **le contrat de la fovéa est
C-STRAT version F-unique** (commis à l'identité-sous-JND, vivant au témoignage, juge =
projection pondérée-observateur ; un seul F scale-aware, tirage de naissance, falsifieur
unifié « aucune transition détectable ») ; principes v2 endossés gravés par référence
(`note-orientation-v2-2026-07-25.md`). **Spec P1 ENDOSSÉE (§A37, 25/07 après revue :
Option B pleine résolution, Y = A identité, inverse-EOTF sRGB, bilinéaire gravé ;
fait de code : le pin a été mesuré à travers viridis) — prochain pas : ordre de
mission Claude Code, sur décision explicite**, puis P2 chiffrage (du NOYAU seul,
portée pré-écrite), puis P3 transport du pin. (γ) et b-vs-c dormants sous la règle D17.

## LE FAIT DOMINANT — défaut d'instrument confirmé par lecture de code (25/07)

**Le niveau grossier de T2 est avancé au Δx du fin.** Chaîne de preuve
(`verif-echelle-grossier-2026-07-25.md`, tout re-vérifiable au grep) :

- la référence `_rhs_o2` (`solver_wetdry.py` l.328-338) **divise par Δx** ; le kernel
  GPU `_SOURCE_FIDELE` qui déclare la porter **ne reçoit aucune longueur** (ses `dx`/`dy`
  sont des décalages de stencil ±1) → portage fidèle **uniquement à Δx = 1** ;
- `reduction_cfl_fidele` calcule aussi le dt du grossier **à Δx = 1** ; le « choix 3 :
  le grossier à dx = 2 » de `production_fidele` existe **en prose, nulle part en code** ;
- `grep _rhs_o2 tests/` ne retourne **RIEN** : le portage n'a jamais été comparé à sa
  référence ; tous les tests existants sont des propriétés d'équilibre (flux nuls) ou
  d'identité — **aveugles par construction** à un facteur multiplicatif sur `L`.

**Conséquence si la lecture tient** : `L_grossier` = 2× le correct, dt partagé en
lockstep ⇒ **le grossier avance à ~2× la vitesse physique du fin** (désynchronisation
~48 unités sur la fenêtre retenue 48.44). Aucun réglage de dt ne compense (dt/2 romprait
le lockstep). Atténuation portée avec le chiffre : le substrat relaxe — **aucune
amplitude n'est prédite, rien n'annonce un PASS** (le témoin frôle déjà le pin à 0.077).

**Portée** — CONTAMINÉ si la lecture tient : tout ce qui descend de T2 — **§A32 (MORT-b),
§A32-décomposition, §A33 (plancher)**. INTACT : le pin (Arc C), §A13/§A14 (registre,
fenêtrage), le modèle de coût F1, M-a-quater/V4, **T1 et son ancre 0.924**, le diagnostic
OOM, la propriété E, le mipmap, L3, **le bras témoin de T2** (mono-niveau Δx = 1, correct
par construction, 0.0015–0.077).

**Concession consignée (§5 de la vérification)** : « MORT-b n'est pas rouvert » est
**RETIRÉ** (amendement du 25/07 dans la séance É2). Survit : **« Option B n'est pas le
remède »** (adossé au témoin, propre), **le lieu** (la périphérie porte 22×, jusqu'à
169×, d'énergie d'erreur par cellule), **la méthode** (masquer la différence, pas le
domaine). Ne survit pas : **« Option A est morte »**.

## Prochain pas exact : le CORRECTIF, puis le re-run T2 (D19 TRANCHÉ le 25/07)

État D19 : **(a)** test unitaire `tests/test_portage_rhs_o2.py` écrit et exécuté sur
iluin-tworings3 — **CONFIRMÉ**, lecture pré-écrite prononcée telle quelle
(err(Δx=1) = 3.297e-05 = bruit f32 ; err(Δx=2) = 0.500 = la signature du facteur 2) ;
**(b)** suspension gravée (§A33-CORRECTION + §A33-CORRECTION-résultat, append-only) ;
**(c)** ENDOSSÉ ; **(d)** sans objet.

1. **FAIT (commit 26d53e2, 2026-07-25)** : kernel Δx-conscient (`inv_dx` dans la
   signature, grossier à Δx=DECIMATION aux quatre étages), `reduction_cfl_fidele`
   INTOUCHÉE, empreinte re-gravée (aef7237d…, 9646), **1124 tests verts**, 3e test de
   portage : inv_dx=1.0 → err(Δx=1)=3.297e-05 (le chiffre exact d'avant correctif — le
   régime Δx=1 n'a rien vu passer, ET c'est mesuré bit-exact contre l'ancien kernel) ;
   inv_dx=0.5 → err(Δx=2)=6.331e-05. Kernels _SOURCE/L3/EXNER : diff vide vérifié.
2. **Romain lance lui-même le re-run T2**, inchangé par ailleurs (mêmes seuils, mêmes
   seeds, mêmes bandes, cellule §A15 intacte), verdict-grade sur sa machine.
3. **Le verdict du re-run remonte avant toute suite** : D14–D18 (scission É2, destin du
   gate (iii), non-transposabilité, profil radial, formule publique) se reposent
   dessus. Rappel de portée : **rien n'annonce un PASS** — le témoin propre frôle déjà
   le pin à 0.077.

## Décisions ouvertes derrière D19 (séance É2, §5)

- **(D14)** Scission É2-état (mesuré, plancher établi) / É2-projection (non mesurable,
  gatée sur le rendu). Objection à peser d'abord : si le pin se transporte **par
  construction** (rendu = fonction déterministe de `z`), la scission est vide et (α)
  est la seule lecture.
- **(D15)** Gate (iii) : **(α)** sur É2-état ⇒ ÉCHOUÉ ; **(β)** sur É2-projection ⇒ NON
  MESURABLE, la moitié PROJECTION devient **le chemin critique**. (La 3e branche —
  critère fovéa-conscient — est **fermée par la mesure** : périphérie parfaitement
  réparée ⇒ Δχ reste 0.109–0.572, « aucun remède unilatéral ».)
- **(D16)** Graver : la plage 0.056–0.664 est mesurée en régime saturé Π ≫ 1, **non
  transposable en magnitude** vers Π < 1.
- **(D17)** Entériner le refus du profil radial (aucune branche ne change une décision).
- **(D18)** Formule publique remplaçant « gate (iii) MORT » (contredit une ligne gravée
  de §A33 ⇒ **entrée de correction explicite**, jamais une réécriture).

## Conclusions de la séance É2 (24/07, auditée, sous la réserve de §0)

- **Q1** : É2 est **écrit en projections et mesuré en états** (l'albédo est un champ de
  `z` ; le pin vit sur un readout d'instrument « construit pour comparer, pas pour être
  vu ») ; la tension est **interne au contrat** §A13-0, qui quantifie sur un observateur
  ET se déclare mesurable. Scinde §A33 en **(P1) [MESURÉ]** plancher Δχ-instrument
  0.056–0.664 à seuil nul et budget plein (« resserrer EPS ne peut pas franchir ce
  plancher » ; non-monotone : 3/18 émissions où retirer le seuil AGGRAVE) et
  **(P2) [NON MESURABLE]** « donc É2 échoue ».
- **Q2** : la contamination du halo est **INHÉRENTE au couplage grossier→fin** au sens
  précis : les trois remèdes abolissent soit la fovéa (recouvrement : d ≥ 91–366 ⇒
  ×45–×570 à fovéa 32 ; ×1.84 à V4 = +9.8 ms = **4.5× la marge** ; bord propre = même
  cas), soit la bornitude du registre (bord commis : **9.9 Go/h** contre 1.17 Ko/commit —
  paie É2 avec le différenciateur). Le sujet de la spec : **décider CONTRE QUOI la fovéa
  doit être fidèle.** (Mécanisme non-commutation `décim∘F ≠ F∘décim` : hypothèse.)
- **Q3** : le caveat d'échelle est **falsifié comme motif d'espoir par sa propre
  arithmétique** (Π = c·T/(n_fov/2) : à la cadence d'émission, V4 est traversée
  entièrement, Π = 1.43 ; ambiguïté d'horizon 48.44/130 défavorable et signalée). Il
  survit **uniquement comme étiquette de portée**.

## Acquis MESURÉ (inchangé, avec portées)

- **Le différenciateur** : §A13 REGISTRE_FERME (gravé 18/07, endossé) — k*_chaîne ≤ 256
  floats (6.25 % du champ fin) pour tous Δt ∈ {1,4,16}, contrôles 0 violation ; ledger
  **1.17 Ko/commit**, load 63 ms/1 h, compaction = disque pas load ; fenêtrage §A14
  (k_fen=64 tient, max 63 % du pin — sonde, 1 seed). Portée : v2-sédiment, pin n=1,
  cette famille de politique.
- **Le pin** (Arc C) : jnd_sev **7.33 %** IC [6.03, 8.67] ; laxiste 11.54 % [9.37, 13.87] ;
  ancré en cycles/degré, budget compresseur 32. Dette albédo→luminance : infalsifiable
  faute de rendu ; atténuation nommée — transport **par construction** si le rendu est
  une fonction déterministe de `z`.
- **Le moteur F1** : ancre 6.43 ns/cellule/frame ; V4 **14.490 ms / 16.7** ; prédiction
  GPU-side emboîtée **gratuite** ; L3 amortie 1.280 ms (prédit à 4 chiffres) ;
  transferts 2.727 ms ; MORT-a naïve (×42) et architecturale (×2.7) prononcées — mort
  du quadruplet majorant sur 3050 Ti, **pas de la fovéa-z**. T1 : ancre proxy honnête
  **0.924**. Le tout **sur un proxy** (v2-sédiment, c=8 enveloppe ; le fidèle coûte
  ~2.5× le proxy PAR CHAMP), **rendu non compté**.

## Faits d'instrument (NE PAS re-découvrir)

- Verdict-grade : **iluin-tworings3, terminal natif** (`.venv/bin/python`, Python 3.12.3,
  numpy 2.4.6, CuPy 14.1.1). VM Cowork = dev/lecture (bit-identique au gel mais **kill
  45 s** ; voit le GPU, jamais une mesure). Le cloud n'est **pas** l'instrument (AVX512).
- **Quatre empreintes kernels** recalculables : fusionné e18015f5…, L3 9533a130…,
  **FIDELE aef7237d… (9646)** — re-gravée le 25/07 (correctif Δx, commit 26d53e2, onze
  citations à jour) —, EXNER 3533fd0b…. L'ancienne FIDELE 6dd207ca… (9248) reste
  **l'empreinte de datation** de l'ancre T1 0.924 et du coût 11.696 ms (chrono non
  re-daté : bit-exact à inv_dx=1 mais +1 multiplication/cellule/étage).
- `ruff` : le binaire vit dans **`pocCascade2phys/.venv/bin/ruff`** (0.15.20) —
  **ABSENT de `pocPhysicator/.venv`**, et AUCUNE config ruff dans les dépôts
  (« ruff clean » au sens d'un dépôt n'est pas prononçable ; lint effectif :
  `--line-length 100 --select E,F,W`). Corrigé le 25/07 au soir, première
  application payante de la règle §A37-3 : le « INSTALLÉ dans .venv » du matin
  était imprécis de DÉPÔT, et avait pourtant été « vérifié ». OOM T2 = **RAM hôte** (escalade de `t_end`
  depuis le bas, RLIMIT_AS sur VmSize). À EPS = 0 **le cap 10 % mord** (budget H·W).
  **Δχ non monotone en erreur d'état** (les bras ne s'additionnent pas). **Masquer la
  DIFFÉRENCE, jamais le domaine.** Hypersensibilité : 6e-5 d'état ⇒ 0.077 de Δχ.
- Push GitHub : **Romain uniquement**. Commits :
  `git -c user.name="Claude (Cascade)" -c user.email="noreply@anthropic.com" commit`.

## Dettes et risques nommés, NON armés

σ_ω (réveil = décision de cadencement) ; **chrono T1 daté sur 6dd207ca — réveil :
toute décision consommant le coût du fidèle à la marge (gravé §A33-CORRECTION-exécution)** ;
`r_fovea` gaté ; effondrement du dt en assèchement ; p99 non résolu ; **la moitié PROJECTION n'existe pas — ni image rendue ni
son** (tous deux conséquences déterministes de `z`, coût mesuré pour AUCUN ; « V4 tient »
signifie « ~16.7 ms pour un rendu jamais chiffré ») ; substrat-jeu non figé ; niveau 0
CPU 500k (×7.6 non mesuré) ; calcul 3D sans ancre ; cadencement « 1 pas/fenêtre/frame »
jamais mesuré ; emboîtement bidirectionnel : nommé, non armé, absence délibérée.

## Innovation portée (pour APRÈS les verdicts)

Ressaut-puis-relaxation vu 2× (deux protocoles) ; k*(16) < k*(4) au pin ⇒ **cadencement
des commits sur la relaxation du substrat** — décision neuve, jamais un enchaînement.

## Discipline (le projet vit d'elle)

Pré-enregistrer AVANT de mesurer ; le moins cher qui peut échouer, d'abord ; chiffres
inconfortables en évidence ; portées jamais surclamées ; INDÉTERMINÉ est un résultat ;
**un échec d'instrument n'est pas un résultat** ; corriger l'INSTRUMENT, jamais la
mesure ; ne jamais rétrécir la mesure pour l'instrument ; une bande se pré-enregistre
pour sa config exacte ; le journal est append-only — les contradictions passent par une
**entrée de CORRECTION explicite** ; **pas de quatrième diagnostic** (tenu : la
vérification du 25/07 est une lecture de code à coût nul, et le test §6 est le test
unitaire manquant, utile quel que soit son résultat) ; concéder proprement et
immédiatement — deux concessions de la session critique sont au dossier de cette
séquence, dont « MORT-b n'est pas rouvert », retirée. **Un fait d'instrument porte
sa date** — à la reprise, re-vérifier les moins chers (un `ls`, un `--version`)
avant de les citer (règle gravée §A37, née du résidu « ruff absent »).
