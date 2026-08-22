# PRÉ-ENREGISTREMENT — RE-QUALIFICATION DE L'INSTRUMENT 3D, v2 (2026-08-22)

> **Statut : PRÉ-ENREGISTREMENT, écrit AVANT tout run.** Le journal
> `pocCascade2phys/PREREGISTRATION.md` fait foi. Ancres au format §A50,
> **contrôlées par `verifier_ancres.py` avant commit**.
> **Ce document se commit SEUL** (règle §A52).
> **L'endossement est le commit de Romain ; aucun run avant.**

---

## §0. Ce que ce document EST, et ce qu'il n'est pas

C'est une **qualification d'INSTRUMENT**, pas une mesure d'architecture. La
règle 1 de `CLAUDE.md` s'applique : **rien de ce qui suit ne prononcera quoi
que ce soit sur V4, la cadence, le côté de fenêtre ou le budget.**

C'est le dû **(1)** de `CLAUDE.md:282` « NE PAS LANCER DE MESURE 3D
ABSOLUE », et la **quatrième tentative de qualification** — après `§A59`,
`§A60`, `§A61` : trois runs, trois fautes de plan, toutes de la même famille.
Ce document est construit comme la correction point par point de ces trois
fautes (`PREREGISTRATION.md:9933` « un dispositif où **le rang d'exécution est
apparié** (mesures alternées ou »), pas comme une v1 assagie.

**Le renversement qui distingue v2 de v1.** v1 cherchait *le warmup qui
atteint le plateau*. `§A61` a établi qu'**il n'y a pas de plateau**
(`PREREGISTRATION.md:9839` « la machine n'a pas de point de fonctionnement
stable »). v2 ne cherche donc plus à stabiliser l'instrument — elle demande :

> **Un contraste apparié en rang est-il répétable malgré la dérive, et avec
> quelle incertitude ?**

C'est exactement la conjecture que `§A61-4` laisse ouverte
(`PREREGISTRATION.md:9916` « rapports mesurés INTRA-RUN entre points adjacents
dans le ») : *à ÉTABLIR, pas à supposer*. Si elle tient, l'instrument est
qualifié **pour les contrastes**, jamais pour les absolus — et les deux
grandeurs que la cadence attend (`I`, non-F 3D) sont des contrastes.

---

## §1. LES ACQUIS SUR LESQUELS v2 S'APPUIE, ET LES PRIORS QU'ELLE RE-MESURE

- **La règle d'appariement**, payée trois fois en une nuit
  (`PREREGISTRATION.md:9875` « une comparaison ne vaut que si les deux termes
  ne diffèrent QUE par la ») : rang d'exécution, état thermique et périmètre
  de calcul sont des variables au même titre que celle qu'on teste.
- **La dérive a un ordre de grandeur prior** : ~3 % sur trois minutes
  (`§A61-4`), soit **~0,017 %/s**. Ce prior sert à DIMENSIONNER les blocs
  (§2) ; il est RE-MESURÉ par `M-d` et la garde `I-q5` invalide le
  dimensionnement s'il était faux.
- **La concordance côte à côte est possible sur cette machine** : `§A53`,
  deux runs appariés, **0,63 %** (`PREREGISTRATION.md:9918` « côte à côte »).
  C'est la preuve d'existence qui justifie le seuil de 2 % (≈ 3× cette
  concordance) — un chiffre hérité, pas choisi pour passer.
- **`T_conv` n'a pas de constante** : 2,5 s à 64³, 19,3 s à 128³, 22,1 s à
  32³ (`PREREGISTRATION.md:9899` « `T_conv` mesuré »). Aucun bloc ne consomme
  donc de « warmup convergé » : le warmup par bloc reste le plancher gravé
  (30 frames, exclues), et c'est l'APPARIEMENT, pas la convergence, qui porte
  la validité.

---

## §2. LES MESURES

Configuration : vocabulaire réel (3 scalaires + 2 statiques, 9 champs,
3 fenêtres), kernel `substrat_fusionne_3d_param` **INTOUCHÉ**.

**Le bloc**, unité de tout le protocole : warmup 30 frames (exclues) puis
`max(300 frames, 3 s)` de série, médiane intra-bloc. À 128³ le plancher de
300 frames domine (~18 s) ; à 64³ et 32³, les 3 s.

**Le dimensionnement est fixé PAR CÔTÉ, UNE FOIS, en tête de session — pas
par bloc.** Un pré-chrono d'un bloc plancher (30 + 300 frames) par côté,
consigné et NON consommé, convertit les 3 s en un compte
`serie[côté] = max(300, ⌈3 s / t̂_frame⌉)`, gravé dans l'artefact avant la
première mesure. Deux blocs du même côté ont ainsi un TRAVAIL identique ;
leur durée mur peut différer — c'est le signal, pas un confondu. Un
dimensionnement par bloc rendrait deux blocs 64³ inappariables : positions
différentes dans la dérive ET travail différent. Biais du pré-chrono, nommé :
il s'exécute dans le transitoire, donc `t̂` est LENT et les blocs peuvent
durer moins de 3 s au régime rapide — jamais sous 300 frames, le plancher
tient. Le pré-chrono précède le premier refroidissement.

**Le cycle** : blocs enchaînés `64³ → 128³ → 32³`, **sans jamais laisser le
GPU au repos entre blocs** (conséquence opératoire de W-B/v1, confirmée par
`§A61`). Chaque paire de blocs ADJACENTS dans le temps donne un rapport
apparié ; la paire (32³, 64³) se lit à la frontière entre cycles.

| # | contenu | départ thermique | ce qu'elle isole |
|---|---|---|---|
| **M-s** | 6 blocs 64³, sonde `ON/OFF` alternée par bloc | froid | la sonde perturbe-t-elle ? (`I-q1`) — version APPARIÉE du contrôle que `§A61-1` a mis en dernier |
| **M-c1** | 4 cycles, ordre ALLER (64→128→32) | froid | les rapports appariés, référence |
| **M-c2** | 4 cycles, ordre RETOUR (32→128→64) | chaud (enchaîné) | ordre × chaud |
| **M-c3** | 4 cycles, ordre RETOUR | froid | ordre × froid |
| **M-c4** | 4 cycles, ordre ALLER | chaud (enchaîné) | aller × chaud |
| **M-d** | série continue 64³, 180 s | froid | la NON-STATIONNARITÉ elle-même : pente contre temps et contre température |

Le plan `M-c1…M-c4` est un **2 × 2 complet** (ordre × état thermique de
départ), et ce qu'il achète doit être dit exactement. **L'ORDRE est équilibré
contre le rang** : ALLER aux rangs {1, 4}, RETOUR aux rangs {2, 3}. **L'état
thermique, lui, reste confondu avec le rang à +1** — structurel, pas
corrigeable par permutation : un « chaud » est par définition un état hérité,
il suit toujours un froid. La lecture froid/chaud est donc un contraste
« départ contrôlé contre état hérité », pas une variable orthogonale au
temps ; elle est atténuée (chaque `r_i` est déjà apparié intra-run), non
annulée. Le dispositif CORRIGE la faute de `§A61-1` — où le contrôle sans
sonde était le dernier (`PREREGISTRATION.md:9860` « La dérive est monotone,
et le contrôle sans sonde est le DERNIER. ») — pour l'ordre et la sonde ; il
ne fait que la BORNER pour le thermique.

**« Froid »** se définit par une mesure, pas un décret : idle jusqu'à
température GPU ≤ 55 °C, plafonné à 180 s ; la température atteinte est
consignée dans l'artefact. **« Chaud »** : enchaîné sans idle. La pente de
`M-d` est estimée par **Theil–Sen sur médianes de tranches** — l'estimateur
robuste promis, nommé ici parce que le banc a montré que le taire laissait
un choix de protocole au driver.

**Ordre d'exécution, gravé ici** : `pré-chrono (3 côtés) → M-s → M-c1 →
M-c2 → M-c3 → M-c4 → M-d → M-s′` — les « chauds » enchaînés, les « froids »
précédés du refroidissement mesuré. **`M-s′` reprend le plan de `M-s`, DÉPART FROID compris** — sinon un
désaccord thermique entre les deux se lirait comme un verdict sonde. **Cinq
refroidissements en tout** (`M-s`, `M-c1`, `M-c3`, `M-d`, `M-s′`) : le début
de session n'est pas supposé froid, il est rendu froid. **`M-d` vient APRÈS
les quatre runs de cycles, et ce choix a un prix nommé** : si `I-q5` mord,
les quatre runs (~7 min de GPU) sont perdus. L'alternative — `M-d` d'abord —
réchaufferait la machine et fausserait le « froid » de `M-c1` : on paie le
risque plutôt que le confondu. Le plan complet est fixé dans le driver avant
tout run.

Relevé `nvidia-smi` (~20 Hz) : horloge SM, température, puissance,
`clocks_event_reasons` — pendant toutes les mesures sauf les blocs `OFF` de
`M-s`. Traces complètes conservées. Budget : **≈ 12 min de GPU actif,
dérivé (pré-chrono compris) ; mur ≤ 27 min au plafond des cinq
refroidissements** (180 s chacun).
Seul le plafond est gravé — la valeur attendue dépend des refroidissements
réels, qui ne sont pas dérivables ici.

---

## §3. L'ESTIMATEUR, ÉCRIT AVANT LES DONNÉES

Pour un contraste `X→Y` : chaque paire de blocs adjacents `(X_i, Y_i)` donne
`r_i = med(Y_i)/med(X_i)`. **Lecture = médiane des `r_i`.** Dispersion :
**deux chiffres, tous deux rapportés**, aux rôles SÉPARÉS —

- **L'écart interquartile/2 DÉCIDE, SEUL.** Les branches de §4 se lisent sur
  lui et sur rien d'autre.
- **L'étendue/2 est un TEST DE FORME, pas un second juge** : si le rapport
  `(étendue/2)/(interquartile/2)` dépasse **5,5**, le verdict du contraste est
  **INDÉTERMINÉ** et les blocs extrêmes sont nommés dans l'artefact. Sous ce
  seuil, elle est consignée sans voix au chapitre.

**Toutes les dispersions de ce document s'entendent RELATIVES à la médiane
du contraste** — les trois contrastes vivent à trois ordres de grandeur
d'écart (r ≈ 8,4 contre r ≈ 0,017) et une dispersion absolue comparée au
seuil de 2 % aurait rendu Q-A/Q-C dépendants de l'échelle (fabrication
trouvée par le banc du driver, garde 9, corrigée). Cas dégénérés, gravés :
dispersion nulle des deux côtés = rien à juger, le test de forme ne tire
pas ; corps central nul avec extrême non nul = **INDÉTERMINÉ**, sortie sûre.
La dispersion d'un contraste se calcule sur l'**UNION des `r_i` des deux
ordres** — lecture conservatrice : le biais de paire signé (±0,43 % au
majorant) GONFLE cette union, direction Q-B, jamais Q-A.

**Le seuil 5,5, et pourquoi il a remplacé 4,4.** La première version gravait
4,4 = p90 **par contraste** ; le banc du driver a montré que l'agrégation de
§4 en changeait le sens — trois contrastes à 10 % chacun font
`1 − 0,9³ = 27 %` de faux positif GLOBAL. La convention vit au niveau où la
décision se prend (règle de `§A60`) : c'est le **FP global ~10 %** qui est
la convention, et le seuil par contraste s'en DÉRIVE — quantile `0,9^⅓` ≈
p96,5 du rapport sous le même modèle, soit 5,36 (n = 16) / 5,53 (n = 12),
**seuil unique 5,5** pris au plus grand. La
relecture adverse a attaqué l'hypothèse i.i.d. et l'attaque a échoué **dans
la bonne direction** : une dérive résiduelle linéaire raccourcit les queues
du rapport et fait BAISSER le faux positif (9,6 % → 4,2 % à pente 4× le
bruit) — le seuil est conservateur vis-à-vis de la non-stationnarité de
`§A61`. Faire juger les deux dispersions à la même barre aurait fabriqué de
l'INDÉTERMINÉ par mécanique pure : sur n = 12–16, l'étendue/2 vaut ~2,8×
l'interquartile/2 sans le moindre outlier.

Un bloc throttlé isolé ne peut pas fabriquer un Q-C — argument STRUCTUREL,
pas simulé : les quartiles ont un point de rupture à 25 % et un bloc pèse
6–8 % de l'échantillon. **Sa DÉTECTION par le test de forme, en revanche,
n'est pas annoncée ici** : la puissance du test dépend entièrement de la
dispersion vraie de l'instrument — précisément la grandeur que ce protocole
existe pour mesurer. Annoncer un chiffre de puissance maintenant exigerait de
supposer σ connue, et tout scalaire publié serait un chiffre de coin (faute
`§A62-bis-4`). **Gravé à la place : la puissance est calculée APRÈS COUP par
le lecteur, à partir de la dispersion interquartile LUE, et consignée dans
l'artefact** — la règle de `§A60` (un nul ne se prononce qu'accompagné de sa
puissance) appliquée dans le seul ordre honnête : la puissance après le
paramètre. Le modèle de ce calcul est gravé ici : un `r_i` contaminé, bruit
gaussien à la dispersion lue, n du contraste ; et **les amplitudes de
contamination sont DÉRIVÉES des excursions d'horloge observées dans les
traces du run** (rapport min/max de l'horloge SM — `§A61` a relevé
`PREREGISTRATION.md:9887` « L'horloge SM varie de **1 035 à », soit des
blocs décalés bien au-delà de tout +5 % décoratif), pas choisies.

**Et si `I-q1` retire la sonde, cette dérivation tombe avec elle** : sans
trace d'horloge, les amplitudes ne sont plus dérivables du run ⇒ la puissance
devient **NON LISIBLE**, au même titre que `I-q6` et pour la même raison — on
ne dérive pas d'une trace qu'on vient de juger perturbante. Le test de forme
garde alors son pouvoir de DÉCLENCHER ; c'est son SILENCE qui n'établit plus
rien, faute de puissance calculable.

C'est le témoin que `§A52` exige d'un chiffre défavorable comme d'un
favorable, sans punir la qualité qu'il contrôle. Compte des paires, dérivé
avant le run : 12 blocs
par run donnent 11 paires adjacentes — **16 paires pour 64↔128, 16 pour
128↔32, 12 pour 32↔64** (la paire 32↔64 vit aux frontières de cycles, il y en
a une de moins par run).

Pourquoi c'est un estimateur **robuste à la dérive**, et pas une médiane sur
population mouvante (la troisième faute de `§A61-5`) :

1. La dérive entre dans les DEUX membres d'une paire, décalés de la durée
   **MILIEU-À-MILIEU** de la paire (centre du bloc `X` au centre du bloc
   `Y`) : pire cas dérivé ≈ **12,5 s** (paires impliquant 128³ : 1,5 s +
   warmup ~2 s + 9 s), **majoré à 25 s**. Au prior de 0,017 %/s, le biais par
   paire est **≤ 0,43 % au majorant** (au pire cas dérivé : 12,5 s ×
   0,017 %/s = 0,2125 %, arrondi PAR EXCÈS à 0,22 %) — borne
   PRÉ-DÉRIVÉE, re-vérifiée contre la pente mesurée par `M-d` (`I-q5`).
   Les 12,5 s sont NOMINAUX : le lecteur recalcule la borne sur les durées
   de blocs RÉELLES et la pente de `M-d` — volet mécanisé d'`I-q5`.
2. Ce biais résiduel est signé par l'ordre du couple : il **change de signe
   entre ALLER et RETOUR**. **La lecture finale est la MOYENNE GÉOMÉTRIQUE
   des deux lectures orientées** — `√(r_ALLER · r_RETOUR)`, chacune étant la
   médiane des `r_i` de son ordre, le RETOUR réorienté dans le sens du
   contraste. Le premier ordre s'annule alors PAR CONSTRUCTION : si
   `r_ALLER = r(1+ε)` et `r_RETOUR = r(1−ε)`, la moyenne géométrique vaut
   `r√(1−ε²)` — pour ε = 3 %, un résidu de 4,5·10⁻⁴, du second ordre. La
   géométrique, et non l'arithmétique, parce qu'une lecture de RAPPORT doit
   s'inverser quand on inverse le contraste. **Une médiane sur l'UNION des
   `r_i` des deux ordres n'a PAS cette propriété** : elle dépend des
   effectifs et de la forme, et l'annulation y redeviendrait un espoir.

---

## §4. LES BRANCHES, ÉCRITES AVANT LA LECTURE

**Priorité, écrite maintenant : les indéterminations de §5 s'évaluent AVANT
les branches.** Q-A/Q-B/Q-C ne se prononcent que sur les contrastes ayant
survécu à §5 — deux clauses pré-écrites ne peuvent pas conclure en sens
opposés sur le même chiffre, et le `else` de combinateur ne couvre que
l'absence de branche, pas leur conflit.

> **Q-A — QUALIFIÉ POUR LES CONTRASTES APPARIÉS INTRA-RUN.** Les trois
> conditions tiennent : dispersion des `r_i` ≤ **2 %** par contraste ;
> accord aller/retour et froid/chaud ≤ **2 %** ; dérive cycle-à-cycle
> expliquée (`I-q4`). ⇒ L'incertitude d'instrument `σ_r` = la pire des trois
> dispersions interquartiles (agrégation ci-dessous) est **GRAVÉE**, avec la
> règle d'usage : *toute lecture future
> distante de moins de 3·σ_r d'un seuil est INDÉTERMINÉE*. **Les absolus
> inter-runs restent interdits** — Q-A ne les réhabilite pas.
>
> **Q-B — QUALIFIÉ RESTREINT.** Répétable intra-run mais l'accord
> froid/chaud casse le 2 % en restant sous 15 %, ou la dispersion vit entre
> 2 et 15 %. **Un désaccord ALLER/RETOUR n'a PAS de version restreinte** :
> c'est `I-q3`, INDÉTERMINÉ par la priorité ci-dessus — une dépendance au
> rang malgré l'appariement est une faillite de l'ESTIMATEUR, qu'aucune
> restriction d'usage ne contourne ; une dépendance thermique est une
> propriété de la MACHINE, contournable en mettant les deux bras dans le
> même run. ⇒ Contrastes valides SEULEMENT les deux
> bras dans le même run et l'incertitude gravée à la valeur observée ; tout
> seuil plus fin que 3× cette valeur est déclaré illisible sur cette machine.
>
> **Q-C — NON QUALIFIABLE, INDÉTERMINÉ.** Dispersion ou désaccord > **15 %**
> — l'ampleur même du biais qui a disqualifié les absolus
> (`PREREGISTRATION.md:9701` « les absolus 3D sont pessimistes d'environ
> 15 % ») : l'appariement n'a rien acheté.
> ⇒ Le problème est **MATÉRIEL** (plafond de puissance, refroidissement,
> machine), pas protocolaire. **Aucune cinquième tentative protocolaire** ;
> la suite est un changement matériel ou l'abandon des mesures 3D fines sur
> cette machine.

**Règle d'agrégation, écrite maintenant — les branches sont PAR CONTRASTE,
mais §8 consomme un verdict GLOBAL** : le verdict global est **la pire des
trois branches**, et **tout contraste sorti par §5 (INDÉTERMINÉ) rend le
global INDÉTERMINÉ**. Ce protocole prétend qualifier une MÉTHODE — les
contrastes appariés intra-run — pas trois paires de kernels : un échec
inexpliqué sur l'un des trois spécimens est un échec de la méthode, et rien
ne dit qu'il épargnerait les contrastes que `I` et le non-F demanderont.
`§8` ne s'ouvre que sur un global Q-A ou Q-B.

Le seuil de 2 % vient de §1 (3× la concordance de `§A53`) ; celui de 15 % de
`§A60`. Aucun des deux n'est nouveau. **Deux constantes de ce document, en
revanche, SONT NEUVES : le facteur 3 de la règle d'usage `3·σ_r`, et le
budget de faux positif GLOBAL ~10 % du test de forme** — dont le seuil par
contraste 5,5 est DÉRIVÉ (§3), pas choisi. Ni dérivées, ni héritées — des
**CONVENTIONS**, déclarées comme telles, que l'endossement de ce document
achète explicitement. **Budget d'INDÉTERMINÉ sur instrument sain,
nommé pour n'être découvert par personne.** Une seule de ses deux
composantes est calculable AVANT le run — celle du test de forme, parce que
le rapport étendue/interquartile est invariant d'échelle et ne dépend donc
PAS de σ : **9,25 % mesuré** (deux contrastes à n = 16, un à n = 12, seuil
5,5). Celle d'`I-q4` dépend de σ et se calcule après coup (§5). Le total ne
peut donc pas être gravé ici sans supposer connue la grandeur même que ce
run mesure : il est CONSIGNÉ dans l'artefact, et il vaut AU MOINS les
9,25 % du test de forme — de l'ordre d'un run sain sur dix à refaire,
~12 min de GPU, le prix du conservatisme.
Tout protocole consommateur devra soit les reprendre en les citant, soit
dériver de SA décision ses propres seuils — la règle de `§A60` : l'effet qui
compte se dérive de la décision, il ne se choisit pas.

---

## §5. LES INDÉTERMINATIONS, BRAQUÉES SUR LES CAS FAVORABLES

**(I-q1) La sonde, en version appariée.** `M-s` : rapports entre blocs
adjacents `ON/OFF`, **ORIENTÉS — `r_i = med(ON)/med(OFF)` pour chaque paire,
quel que soit l'ordre temporel des deux blocs.** Sans cette orientation la
garde serait AVEUGLE PAR CONSTRUCTION : les 5 paires adjacentes de 6 blocs
alternés changent de sens une fois sur deux, et une médiane de rapports
mutuellement inverses lit un signe décidé par la PARITÉ, pas par la sonde.
**Critère : `|médiane(r_i) − 1| > 2 %`** — l'écart de la MÉDIANE, jamais la
médiane des écarts, qui mord sur du bruit pur (sonde parfaitement neutre :
médiane de `|r_i − 1|` = 0,674 × la dispersion inter-blocs, donc un verdict
« la sonde perturbe » dès que celle-ci atteint 3 %). Si le critère mord, le
relevé perturbe ⇒
retiré de la chaîne des cycles (traces de temps seules). `M-s` est re-exécutée
en fin de session (`M-s′`, même plan) ; **si `M-s` et `M-s′` concluent
différemment**, le verdict sonde est INDÉTERMINÉ et le retrait s'applique par
défaut — la branche est mécanisée dans le driver, pas laissée à la relecture
(faute `§A62-bis-2`).
**Si le retrait s'applique, la circularité sonde ↔ `M-d` est tranchée ainsi** :
la pente de `M-d` CONTRE LE TEMPS reste lisible — elle ne vit que dans la
trace des durées, sans sonde — et `I-q5` avec elle ; la pente contre la
TEMPÉRATURE et `I-q6` deviennent **NON LISIBLES** (non mesurées, pas
indéterminées). On ne valide jamais des cycles sans sonde par une pente prise
sous une sonde jugée perturbante.
*Le critère mord dans les deux sens — accélération comme ralentissement.* ✔

**(I-q2) La dérive doit être VUE pour être annulée.** Si, dans les runs de
cycles, l'écart absolu entre le premier et le dernier bloc d'un même côté est
< **1 %**, alors la répétabilité constatée ne DÉMONTRE PAS la robustesse à la
dérive — c'est le résultat commode. La condition s'évalue **PAR CONTRASTE** :
`max(écart du côté X, écart du côté Y) < 1 %` ⇒ CE contraste est suspendu,
et le global avec lui (agrégation de §4). ⇒ La répétabilité est consignée, la
robustesse reste **NON ÉTABLIE**, et un re-run est dû dans les conditions où
`M-d` montre la dérive. **Articulation avec §4, écrite pour être codée, pas
relue : tant que la CONDITION d'`I-q2` est REMPLIE — écart premier→dernier
bloc < 1 %, le cas commode — AUCUNE branche de §4 n'est prononcée** — le
verdict est **SUSPENDU**
(pas INDÉTERMINÉ : les chiffres sont sains, la démonstration est incomplète),
`σ_r` n'est pas gravée et **§8 reste fermé** jusqu'au re-run. Q-A ne peut pas
s'ouvrir sur une robustesse non testée. *C'est le critère braqué sur le cas
favorable de ce protocole.* ✔

**(I-q3) L'ordre ne doit rien porter.** Si la lecture ALLER et la lecture
RETOUR d'un même contraste divergent de > 2 % (au-delà du biais pré-dérivé de
§3), le contraste dépend du rang MALGRÉ l'appariement ⇒ **INDÉTERMINÉ** sur ce
contraste — pas de moyenne qui enterre le désaccord.

**(I-q4) Le produit télescope — le mot « transitivité » est RETIRÉ.**
`r(64→128) · r(128→32) · r(32→64)` se réduit algébriquement à
`med(64_{i+1})/med(64_i)` : c'est un test de **dérive cycle-à-cycle**, pas une
cohérence des rapports entre eux. Il borne précisément l'hypothèse de §3 —
dérive petite à l'échelle d'un cycle — et n'importe toujours aucune valeur
externe. **Critère, corrigé par le banc du driver (garde 9)** : la première
version lisait le MAXIMUM des produits contre la prédiction moyenne —
99–100 % d'INDÉTERMINÉ sur instrument sain, mesuré deux fois (banc et
contre-simulation), la famille exacte de la faute d'échelle de §3. La
lecture gravée est : `|médiane(produits) − (1 + pente·T̂_cycle)| >
max(IQR/2 des produits, borne de biais de §3 recalculée sur un cycle)` ⇒
**INDÉTERMINÉ** — le premier terme couvre le bruit, le second est le
plancher DÉRIVÉ de la décision : c'est précisément l'hypothèse de §3 que ce
critère protège. **Son taux de faux positif n'est PAS annoncé ici** : il
dépend de la dispersion vraie de l'instrument — l'inconnue que ce protocole
existe pour mesurer — et le banc du driver le fait varier de 0 % à ~6 % sur
la plage plausible de `n` et de σ. Il se calcule **APRÈS COUP** contre la
dispersion LUE, par le même modèle que la puissance du §3, et se consigne
dans l'artefact ; un scalaire publié maintenant serait un chiffre de coin
(faute `§A62-bis-4`), et la règle est déjà celle du §3. **Quand `I-q2` mord (pas de dérive
visible), `I-q4` passe TRIVIALEMENT et ne compte pour rien** — les deux
critères ne se secourent pas.

**(I-q5) Le dimensionnement doit survivre à sa propre mesure.** Si la pente
mesurée par `M-d` dépasse **2× le prior** (0,017 %/s) qui a dimensionné les
blocs, la borne de §3 ne tient plus ⇒ les cycles de CE run ne sont **pas
consommés** ; re-dimensionner les blocs sur la pente mesurée et re-runner.

**(I-q6) Si la machine s'est mise à bien se porter** *(lisible seulement si
la sonde a survécu à `I-q1` — sinon NON LISIBLE, voir I-q1)*. Le throttle se
lit par **test de BIT sur la valeur entière** de `clocks_event_reasons`
(`0x4` = SW Power Cap, celui des cinq séries de `§A61`), masque complet
consigné — jamais par comparaison de chaîne. Si `M-d` ne montre NI
dérive NI throttle (`0x4` absent partout), l'état matériel a changé depuis
`§A61` (alimentation, pilote, profil de puissance — consignés dans
l'artefact). ⇒ Ce run **ne réhabilite pas les absolus pour autant** : cela
demanderait un protocole de stationnarité dédié, avec son prereg.

---

## §6. GARDES ANTI-FABRICATION

1. **`src/f1_gpu/chrono.py` n'est PAS modifié** ; trace temporelle
   reconstruite hors de lui, approximation déclarée (comme v1).
2. **Aucun plancher abaissé** : warmup ≥ 30, série ≥ 300 par bloc — dépassés,
   jamais réduits.
3. **Kernels intouchés** : `git diff` vide, empreintes relues dans l'artefact.
4. **Jamais de repos GPU non contrôlé entre blocs** ; tout idle est
   intentionnel, mesuré, consigné. Mécanisation exigée du driver : **sonde
   UNIQUE de session** (arrêtée seulement autour des blocs `OFF` de `M-s`,
   coût ≤ une période, consigné en idle intentionnel — la sonde reste
   l'instrument du critère « froid » même si `I-q1` la retire de la chaîne
   de verdict) ; **états PRÉ-ALLOUÉS par côté**, ré-initialisés
   device-to-device à chaque bloc, copie chronométrée ; **écart mur entre
   fin de bloc et début du suivant consigné à chaque frontière**.
5. **Traces complètes dans l'artefact** — les rapports appariés sont des
   formes avant d'être des scalaires.
6. **Environnement consigné** : version du pilote, plafond de puissance
   (`nvidia-smi -q -d POWER`), alimentation secteur — sans quoi `I-q6` ne
   peut pas conclure.
7. **Aucune valeur attendue externe** : ni `ρ`, ni `§A53`, ni `§A60` ne
   servent de cible aux rapports. La seule auto-référence autorisée est le
   produit télescopé de `I-q4`.
8. Pas d'`assert` nu ; fail-loud ; `else` de combinateur = **INDÉTERMINÉ**.
9. **Avant tout run, chaque branche de §5 est relue CONTRE LE CODE du
   driver** — pas contre son intention (faute `§A62-bis-2` : `I-r5` était
   écrite, datée, endossée, non codée). La relecture est consignée dans le
   message de commit du driver.

---

## §7. CE QUE CE RUN NE PRONONCERA PAS

- **Rien sur V4, la cadence, le côté, le budget.** C'est de l'appareil.
- **Rien sur les absolus** — même en Q-A, l'interdiction de `CLAUDE.md:282`
  « NE PAS LANCER DE MESURE 3D ABSOLUE » reste entière pour les mesures
  inter-runs.
- **Rien rétroactivement** sur `§A53`–`§A60`.
- **Il ne mesure ni `I` ni le non-F 3D.** Il établit seulement si un
  protocole futur POURRA les mesurer — comme contrastes appariés contre `C`,
  dans un même run, chacun avec son propre pré-enregistrement.

---

## §8. CE QUI SERA DÛ APRÈS

Si Q-A ou Q-B **GLOBAL** (règle d'agrégation de §4, verdict non suspendu par
`I-q2`) : les protocoles de **`I`** et du **non-F 3D** — les deux
grandeurs de `CLAUDE.md:291` « LES DEUX GRANDEURS QUE LA CADENCE ATTEND »,
consommées par `Δ = 30·(non-F − I)/C` où seul
le RAPPORT `(non-F − I)/C` compte — conçus dès l'origine comme des contrastes
intra-run appariés, à l'incertitude `σ_r` gravée ici. Puis la porte 3 de
`§A58`, derrière eux. Si Q-C : une décision matérielle de Romain, avant tout
protocole.

---

*Document rédigé par la session Claude, AVANT tout run. L'endossement est le
commit de Romain.*
