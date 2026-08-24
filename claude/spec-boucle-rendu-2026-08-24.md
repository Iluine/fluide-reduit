# SPEC — BOUCLE DE RENDU (porte 33,3, cadence LOGIQUE 2:1) — 2026-08-24

> **CE DOCUMENT EST UN SPEC DE CODE MOTEUR, PAS UN PRÉ-ENREGISTREMENT DE MESURE.**
> Il décrit un composant à construire. **Il ne mesure rien, n'autorise aucun run,
> ne chiffre ni `I` ni aucune cadence réelle.** `§A61` tient :
> `PREREGISTRATION.md:9927` « **Aucune nouvelle mesure 3D absolue ne devrait être produite avant que »
>
> Il se commit SEUL, avant toute ligne de code. **La revue de Romain est
> l'endossement.**
>
> ⚠ **L'ÉPINGLAGE DES DEUX CÔTÉS (`§2`) EST UNE DÉRIVATION DE CETTE SESSION, PAS
> UN FAIT ENDOSSÉ** — il tombe des constantes du module et du `§3` du spec readout,
> et n'a été prononcé nulle part ailleurs. C'est ce que l'ARRÊT du `§7` soumet.

**Le composant est un CONSOMMATEUR** : il branche `interpolation_readout` et
`chemin_de_cout` dans une boucle qui produit deux images écran par pas de
physique. **Il ne modifie ni l'un, ni l'autre, ni `pyramide.py`.**

---

## §1. PORTÉE

**DEDANS**, et rien d'autre : l'**orchestration 2:1** (un pas de physique sert DEUX
écrans) ; les **deux côtés paramétrables** du `§2`, par un seul chemin de code ; la
**sortie image offline** — des tableaux écran rendus à l'appelant, **aucune fenêtre,
aucun présentateur**. **DEHORS** : toute mesure ; tout chiffre de `I` ; toute cadence
réelle ; tout branchement 3D ; toute présentation ; le choix du côté (`§6`).

**LA CONVERSION `s` → IMAGE VIT DEHORS, DANS LE DRIVER.** Le tick rend des champs
de **sédiment** ; la lecture albédo qui en fait une image est
`src/f1_gpu/chemin_de_cout.py:344` « def albedo_ecran(ecran_s: np.ndarray, s_half: float) -> np.ndarray: »
et **`s_half` y est un paramètre d'APPELANT, sans aucune valeur endossée dans le
corpus**. Le graver ici serait choisir un paramètre hors de toute décision —
exactement le motif par lequel le `§5` du spec readout refuse son propre seuil :
`claude/spec-interpolation-readout-2026-08-23.md:239-240` « Le graver ici serait choisir un seuil hors de toute décision — exactement ce que `§A60` interdit. »
⇒ **La boucle ne convertit rien et n'importe pas `albedo_ecran`** ; l'appelant qui
veut une image applique la lecture lui-même, avec **son** `s_half` (`§6`).

**LA CADENCE EST LOGIQUE.** « 2:1 » est un RAPPORT DE COMPTE : deux écrans par
`frame()`, pas deux écrans par 33,3 ms. **Aucune horloge murale n'entre ici** — ni
`time`, ni `cuda.Event`, ni attente, ni régulation. Une cadence logique n'en a pas
besoin, et `§A61` en interdit l'usage tant que l'instrument n'est pas qualifié.

---

## §2. LES DEUX CÔTÉS, ÉPINGLÉS DEPUIS LES CONSTANTES DU MODULE

Les deux valeurs sont LUES dans le module, jamais recopiées :
`src/f1_gpu/interpolation_readout.py:102` « ALPHA_INTERPOLER: float = 0.5 »
`src/f1_gpu/interpolation_readout.py:103` « ALPHA_EXTRAPOLER: float = 1.5 »

### §2-1 — RULING 1 : `α = 1,0` CONTOURNE LE NOYAU — UN SEUL `melanger` PAR TICK

`claude/spec-interpolation-readout-2026-08-23.md:109` « **TRANCHÉ : les images exactes CONTOURNENT le noyau.** Elles lisent `s_cur` »

**Le motif est la comptabilité gravée, pas une commodité** :
`claude/prereg-ou-vit-le-rendu-2026-08-04.md:87` « | **P-b** | **porte 33,3** — physique 30 Hz, rendu 60 fps par interpolation du readout | `30·nonF + 60·R + 30·I` | »

Le terme est **`30·I`**. **Deux `melanger` par tick feraient payer le noyau à 60
images/s : la comptabilité deviendrait `60·I` et contredirait un document endossé.**
Le module le redit comme un **CONTRAT D'APPELANT, pas une option**.

> ⇒ **L'écran à `α = 1,0` se produit par `gather_chemin_de_cout(pyramide, cote_px)`
> DIRECT sur la pyramide** (`indice_champ=3` par défaut), **jamais par
> `melanger(1.0)`** — **un seul `melanger` par tick, dans les DEUX côtés.** C'est la
> lecture exacte de « `α = 1` gratuit » : gratuit veut dire **hors du noyau, donc
> hors de `I`**.

L'écran interpolé passe, lui, par la `VueInterpolee`, dont l'axe des champs est un
singleton —
`src/f1_gpu/interpolation_readout.py:300` « l'axe des champs est un SINGLETON, donc le gather s'appelle avec »
⇒ **`indice_champ=0` pour l'interpolé, `indice_champ=3` (défaut) pour l'exact.** Les
deux appellent le MÊME gather, non modifié.

### §2-2 — RULING 2 : LE COUPLE SORT EN `α` CROISSANT, ET LE CÔTÉ NE CHOISIT QUE LA PAIRE

Après le `frame()` qui produit l'état `n`, avec `s_prev = n−1` et `s_cur = n` :
**interpoler** = `α = 0,5` (`n − ½`) **puis** l'exact `α = 1,0` (`n`) ;
**extrapoler** = l'exact `α = 1,0` (`n`) **puis** `α = 1,5` (`n + ½`).

> **UNIFORMITÉ GRAVÉE** : dans les deux côtés, **exactement un des deux écrans vaut
> `α = 1,0`**, et le couple sort **en `α` croissant** — le côté ne choisit QUE la
> paire d'`α`. C'est elle qui rend le **tick unique** : un chemin de code, une
> structure de garde, aucun branchement par côté hors du couple.

### §2-3 — LES DEUX LATENCES — **ET UN ÉPINGLAGE DE MESURE, PAS UNE SIMPLE CITATION**

Le `§10` du spec readout sépare deux grandeurs, et les confondre « nomme mal ce
qui retarde » :
`claude/spec-interpolation-readout-2026-08-23.md:358` « | latence de l'**ÉTAT DU MONDE** | **un pas** (33,3 ms à 30 Hz) | **0** | »
`claude/spec-interpolation-readout-2026-08-23.md:360` « Le gather tourne au **centre fovéal COURANT à 60 Hz dans les deux modes** : la »

**La latence de VUE est NULLE des deux côtés** : les deux écrans d'un tick sont
gatherisés au centre fovéal COURANT (garde `§4-6`). Seul l'**ÉTAT DU MONDE** retarde,
et seulement en interpoler.

> ⚠ **CE PARAGRAPHE ÉPINGLE LA MESURE DE LA LIGNE D'ÉTAT DU MONDE CITÉE
> CI-DESSUS, IL NE FAIT PAS QUE LA CITER.** Endosser le `§2-3`, c'est endosser **la lecture** de ce chiffre, et pas
> seulement sa présence dans le corpus. **La mesure retenue : « un pas (33,3 ms à
> 30 Hz) » est l'ÉCART PAR IMAGE ENTRE LES DEUX CÔTÉS** — l'image interpolée
> d'interpoler montre `n − ½`, celle d'extrapoler `n + ½`, et l'écart vaut un pas
> plein. **C'est la seule lecture sous laquelle le chiffre est exact.**
>
> **LA LECTURE ÉCARTÉE, NOMMÉE POUR QU'ELLE NE RESTE PAS DISPONIBLE** : lu comme
> **latence ABSOLUE du mode interpoler**, le même chiffre serait **une
> demi-période, pas 33,3 ms** — l'image interpolée montre `n − ½` alors que l'état
> `n` existe déjà. Une lecture écartée sans être nommée revient par la porte de
> derrière ; celle-ci est écartée **et** écrite.

L'étiquette « endossée » que ce document porte sur cette ligne du `§10` ne couvrait
jusqu'ici qu'un demi-pas de relecture — la citation, pas sa mesure. **Le prononcé
ci-dessus ferme ce demi-pas.** Une troisième grandeur existe, du même objet et
**non épinglée** : au niveau du flux 2:1, l'écran `α = 1,0` étant COMMUN aux deux
côtés, l'écart moyen entre les deux couples tombe à une demi-période. C'est une
**dérivation de cette session**, elle n'est pas ce que la ligne du `§10` mesure, et
elle n'est employée nulle part dans ce document.

---

## §3. LE CONTRAT DU TICK

**MONTAGE** : construire la pyramide, puis `installer_capture_en_registre(pyramide)`
— **UNE FOIS, après la construction et AVANT le premier `frame()`** :
`src/f1_gpu/interpolation_readout.py:637` « la pyramide et avant le premier `frame()` : un `frame()` intercalé »
Une seconde installation lève `ReadoutCaptureImpossible` :
`src/f1_gpu/interpolation_readout.py:642` « une capture en registre est DÉJÀ installée sur cette pyramide. »
L'applicateur rendu est conservé ; son `.tampon` est ce que `melanger` consomme.

> **POURQUOI ELLE EST OBLIGATOIRE ICI ALORS QU'ELLE NE L'ÉTAIT PAS DANS LES BANCS.**
> Les bancs tournaient à **fovéa immobile** : douze verrous verts n'ont pas vu le bug
> de registre du `§12-1` parce qu'aucun n'appelait `frame()`. **Cette boucle appelle
> `frame(1)` — la fovéa BOUGE à chaque tick**
> (`src/f1_gpu/pyramide.py:447` « def frame(self, delta_x: int = 1) -> dict: »).
> Sans elle, `TamponReadout.capturer` serait nécessairement pré-roll et `melanger`
> lèverait `ReadoutHorsRegistre` à chaque tick. **La voie 2 n'est pas un raffinement
> de cette boucle : elle en est la condition d'existence.**
>
> **ET L'ORDRE DÉGÉNÉRÉ, QUI LUI NE LÈVE PAS.** `frame()` **puis**
> `TamponReadout.capturer` ne déclenche **aucune** garde : `_centre_capture` prend
> le centre COURANT, donc `ReadoutHorsRegistre` se tait — mais `s_prev == s_cur`,
> **tout écran interpolé devient identique à l'exact**, `clamps == 0`, et rien ne
> le signale. C'est l'asymétrie silencieuse que le `§4-6` chasse par ailleurs.
> ⇒ **La boucle n'appelle JAMAIS `TamponReadout.capturer` ; son seul point de
> capture est l'applicateur.**

**LE TICK, dans l'ordre, et il est le même des deux côtés** : (1) `pyramide.frame(1)`
— l'applicateur capture `s_prev` par niveau, post-roll et pré-pas de physique, l'état
passe de `n−1` à `n` ; (2) l'écran d'`α` le PLUS PETIT du couple ; (3) l'écran d'`α`
le plus grand. Selon le côté, l'un des deux est l'exact (gather direct, `indice_champ=3`) et l'autre
l'interpolé (`melanger(α)` puis gather sur la vue, `indice_champ=0`) — **un seul
`melanger` par tick** (`§2-1`). Le tick rend les deux écrans et les compteurs de
compte rendu de la vue (`clamps`, `non_finis`), **sans prononcer sur aucun d'eux** :
le seuil qui invaliderait une mesure appartient au prereg de la lecture
d'orientation, pas ici.

---

## §4. LES GARDES

**§4-1 — RIEN DU READOUT NE REMONTE DANS L'ÉTAT** (le `§4` du spec readout, inchangé)
`claude/spec-interpolation-readout-2026-08-23.md:196` « **Le chemin de `s_out` se termine dans le rendu. Aucun autre consommateur »
La boucle EST ce rendu : elle ne réinjecte ni `s_out`, ni `s_prev`, ni un écran, ni
rien qui en dérive, dans le tenseur que `F` consomme.

**§4-2 — LE SENS D'ÉCRITURE SOUS LA VOIE 2** (`§13-4`)
`claude/spec-interpolation-readout-2026-08-23.md:514` « **`frame()` écrit dans le tampon de readout ; le module, lui, ne fait que »
La boucle n'écrit **jamais** elle-même dans le tampon de readout — c'est `frame()`,
par l'applicateur, qui le fait. La boucle LIT.

**§4-3 — RULING 3 : `boucle_rendu` RESTE HORS DE `_MODULES_ETAT`.** Le critère
d'entrée gravé dans le module dit :
`src/f1_gpu/interpolation_readout.py:133` « # le module AVANCE l'état (un pas de temps, `F`, un pilote qui les enchaîne) ou »
Un lecteur l'appliquant **à la lettre** y mettrait `boucle_rendu`, puisque `tick`
appelle `frame()` — et `_verrouiller_consommateur` lèverait alors
`ReadoutInterditDansEtat` à **chaque** `melanger` : **la boucle deviendrait
impossible.** Ce qui tranche est le `§4` cité en `§4-1` : la boucle **EST** le
rendu, le consommateur légitime et nommé. Elle **appelle** l'état ; elle ne
l'avance pas elle-même, et rien d'elle n'écrit dans le tenseur que `F` consomme.

> **POURQUOI CECI EST ÉCRIT PLUTÔT QUE LAISSÉ AU BON SENS.** Cette liste s'est
> révélée fausse **deux fois en deux revues** — cinq modules manquants, puis trois :
> `src/f1_gpu/interpolation_readout.py:124` « # UNE SECONDE REVUE, à la clôture de branche, en a trouvé TROIS DE PLUS »
> La prochaine revue appliquera le critère à la lettre, y ajoutera `boucle_rendu`
> **et cassera la boucle sans voir pourquoi.** Le motif est gravé ici pour qu'elle le
> trouve avant d'agir. — Elle n'entre **pas davantage** dans `_MODULES_PERCEPTUELS`
> de `chemin_de_cout` : code moteur, aucune fidélité jugée, aucun stimulus produit.

> ⚠ **CE QUI MÉCANISE CETTE EXEMPTION — SANS LUI ELLE EST UNE PROMESSE.** Telle
> qu'elle est écrite ci-dessus, l'exemption de `_MODULES_ETAT` est une garde
> **promise dans un document**, c'est-à-dire, au mot du corpus repris en `§4-5`,
> une **garde absente**. Son contre-poids est nommé et il appartient à la tâche 1 :
> **le verrou (a) — un run de `N` ticks AVEC rendu et un run de `N` `frame()` NUS
> doivent laisser des `fenetres` BIT-IDENTIQUES.** C'est lui, et lui seul, qui
> porte mécaniquement ce que ce `§4-3` affirme : que la boucle **appelle** l'état
> sans l'avancer autrement, et que rien du readout n'y a fuité. **L'exemption
> n'est sûre que par ce verrou** ; s'il n'est pas écrit, le `§4-3` retombe au rang
> de promesse.

**§4-4 — UNE VUE N'EST PAS UN ÉCRAN.** Deux niveaux, tenus séparés. **La VUE aliase
le tampon interne** —
`src/f1_gpu/interpolation_readout.py:305` « ⚠ `fenetres` ALIASE le tampon interne `_s_out` du `TamponReadout` — ce n'est »
⇒ elle est valide **jusqu'au `melanger` suivant, et pas au-delà**. **L'ÉCRAN, lui,
est déjà matérialisé par le gather, qui ALLOUE sa sortie** —
`src/f1_gpu/chemin_de_cout.py:181` « ecran = xp.full((cote_px, cote_px), xp.nan, dtype=xp.float32) »
⇒ **il n'aliase rien et n'a PAS besoin d'être copié.**

> **NE PAS COPIER N'EST PAS UNE COMMODITÉ.** Le module refuse la copie défensive avec
> son motif —
> `src/f1_gpu/interpolation_readout.py:312` « copie de plus dans `I` à chaque image interpolée : le choix appartient au »
> — donc une copie d'écran serait **MORTE** : elle ne garderait rien et coûterait.
> ⇒ **La boucle ne copie aucun écran** et ne conserve aucune vue au-delà de son tick.

**§4-5 — EXIGENCE POSÉE POUR LA TÂCHE 1, PAS ÉCRITE ICI : LE VERROU PORTE SUR LA
VUE.** Un verrou formulé « les deux écrans d'un tick diffèrent, donc la copie a eu
lieu » serait **ÉTEINT PAR CONSTRUCTION** — il passerait avec ou sans copie,
puisque le gather alloue de toute façon. C'est
`PREREGISTRATION.md:10152` « **Une garde promise dans un document est une garde absente** — c'est ce que le »
et c'est la famille des verrous que le spec readout a trouvés **trois fois** plus
étroits que leur prose, **tous par mutation, aucun par relecture**.

> Le verrou qui garde réellement quelque chose porte sur la **VUE** : conserver la
> `VueInterpolee` d'un tick, faire le tick suivant, vérifier que `vue.fenetres` a
> **CHANGÉ** — tandis que l'écran déjà gatherisé, lui, est **INCHANGÉ**. Les deux
> moitiés comptent : la première montre l'alias, la seconde qu'il ne s'étend pas à
> l'écran.

**§4-6 — LE CENTRE FOVÉAL NE BOUGE QU'À L'INTÉRIEUR DE `frame()`.** Les deux
écrans d'un tick partagent **un seul** centre fovéal. Le déplacer entre eux serait
SILENCIEUX du côté de l'écran exact : le gather relit `geo.origines(j, centre_fin)`
alors que le CONTENU des fenêtres n'a pas été roulé, et rendrait un champ faux sans
lever. Du côté interpolé, `ReadoutHorsRegistre` mordrait — et c'est cette asymétrie
qui rend la garde nécessaire. ⇒ **Aucune écriture de `pyramide.centre_fin` hors de
`frame()`.**

**§4-7 — LE FAIL-LOUD DU GATHER NE SE RATTRAPE PAS.**
`src/f1_gpu/chemin_de_cout.py:164` « FAIL-LOUD : un pixel qu'aucune fenêtre active ne couvre reste `NaN` et fait »
Un trou de couverture est un **fait de structure** : la boucle ne le rattrape pas, ne
le remplit pas, ne le compte pas pour l'ignorer — elle laisse lever.

**§4-8 — AUCUN `except RuntimeError` LARGE, JAMAIS.** `CheminDeCoutInterdit`,
`ReadoutInterditDansEtat`, `ReadoutHorsRegistre`, `ReadoutSansCapture` et
`ReadoutCaptureImpossible` héritent toutes de `RuntimeError`, et le gather lève
`RuntimeError` pour ses trous. C'est la leçon de la mutation du 03/08 :
`src/f1_gpu/chemin_de_cout.py:95` « couverture. Un appelant qui écrit `except RuntimeError` autour du gather — »
⇒ La boucle n'écrit **aucun** `except RuntimeError`, ni `except Exception`. Un
rattrapage futur se ferait par **classe nommée**, motif écrit à côté.

**§4-9 — AUCUN `assert`, NULLE PART.** Le `§4` du spec readout le grave pour les
serrures qu'il pose, et la règle vaut ici à l'identique :
`claude/spec-interpolation-readout-2026-08-23.md:192-193` « **Jamais un `assert`** : il disparaît sous `python -O` (`§A43`). »
Toute vérification de la boucle — ordre du montage, appartenance d'un `α` au
couple, forme d'un écran — se fait par `if` et levée d'une **classe nommée**. **Un
`assert` disparaît sous `python -O`, donc une garde qui en dépend est absente
exactement dans le régime où l'on mesurera.**

---

## §5. QUESTION OUVERTE NOMMÉE — LE CENTRE FOVÉAL AVANCE UNE FOIS PAR TICK, LES ÉCRANS SORTENT PAR DEUX

Posée, **pas tranchée**, et posée ici plutôt qu'absorbée en silence. **Énoncée en
cadence LOGIQUE, la seule que ce document admette** (`§1`) : compter en images par
seconde ici contredirait sa propre portée.

Le `§10` écrit que le gather tourne au centre fovéal courant « à 60 Hz dans les deux
modes ». Sous cette boucle la lettre tient — **les DEUX gathers d'un tick lisent tous
deux le centre COURANT, la vue ne retarde jamais** — mais ce centre n'est **AVANCÉ
que par `frame()`**, donc **UNE FOIS PAR TICK, pour DEUX écrans** : les deux images
d'un tick partagent la même position de caméra.

Ce n'est **pas** une latence de vue, et ce document ne la renomme pas ainsi : c'est
un **pas de rafraîchissement de la caméra — un par tick et non un par écran** —
jamais nommé nulle part. Le porter à un par écran exigerait de déplacer la fovéa
entre les deux écrans d'un tick, donc de rouler les fenêtres hors de `frame()`, donc
de toucher `pyramide.py` : hors du périmètre de ce document, et interdit par `§4-6`
tant qu'il n'est pas amendé.
⇒ **DÛ, sans échéance forcée, soumis à l'ARRÊT du `§7`.** Le fond du dû est
inchangé ; seule son unité l'est.

---

## §6. CE QUE CE DOCUMENT NE TRANCHE PAS

- **LE CÔTÉ** — il appartient à la lecture d'orientation, qui n'est **pas ordonnée**
  et dont le prereg n'est **pas écrit** :
  `claude/spec-interpolation-readout-2026-08-23.md:548` « **La lecture d'orientation n'est PAS ordonnée**, et son prereg n'est pas »
  et l'ordre du programme le place après deux grandeurs qui ne sont pas lui :
  `PREREGISTRATION.md:10099` « et le côté ensemble, avec la porte 3 de `§A58` — re-dériver la monnaie du slot »
  **La boucle est paramétrable parce que le côté n'est pas choisi** : garde, pas souplesse.
- **TOUTE CADENCE RÉELLE** — le 2:1 est logique (`§1`) ; ce document ne dit ni que
  30 Hz et 60 fps sont atteignables, ni qu'ils ne le sont pas.
- **TOUT CHIFFRE DE `I`**, et toute part de `I`.
- **`s_half`**, et toute valeur de lecture albédo — la conversion `s` → image vit
  DEHORS (`§1`) et son paramètre appartient à l'appelant, pas à ce document.
- **LE PLACEMENT IMAGE du `§9`**, toujours non chiffré :
  `claude/spec-interpolation-readout-2026-08-23.md:537` « **Le `§9` reste NON CHIFFRÉ** : le placement IMAGE n'est toujours pas évalué, »

---

## §7. CE QUE CE DOCUMENT NE FAIT PAS — **ARRÊT**

- **AUCUNE MESURE.** `§A61` tient ; `§A63` n'a pas qualifié l'instrument et son dû
  reste **une décision de Romain, INTACTE** :
  `PREREGISTRATION.md:10409` « **Une décision de Romain, et rien d'autre.** Aucune branche ne s'enchaîne : »
- **AUCUN VERDICT DE CÔTÉ**, aucune comparaison, aucun jugement de perçu.
- **AUCUN BRANCHEMENT 3D.** La chaîne visée est la pyramide 2D, comme le readout.
- **LES TROIS DÛS OUVERTS DU SPEC READOUT SONT INCHANGÉS**, et ce document n'en
  ferme aucun : le **placement IMAGE** du `§9` (ancré au `§6`) ; la **pureté de la
  chaîne de mesure face à un composant à état** —
  `claude/spec-interpolation-readout-2026-08-23.md:219` « **DÛ NOMMÉ, POUR LE PREREG DE LA LECTURE D'ORIENTATION** : *chronométrer un »
  — et l'**élargissement R2+** du `§8` —
  `claude/spec-interpolation-readout-2026-08-23.md:320` « **`s` SEUL EST UN PLANCHER, PAS LE COMPTE FINAL.** Aujourd'hui la chaîne de »
- **RIEN NE S'ENCHAÎNE.** La tâche 1 attend l'endossement de ce document —
  **notamment sur l'épinglage des deux côtés du `§2`, qui est une dérivation de
  cette session et pas un fait endossé**, et sur la question ouverte du `§5`.

---

## §8. FAIT CONSIGNÉ — LA RELECTURE DU 2026-08-24

**Ce paragraphe consigne un fait de relecture. Il ne signe pas à la place de
Romain, et il n'affirme pas que ce document est endossé par le commit qui l'ajoute.**

Romain a relu ce document **en entier, sur pièces**, contrôlé les deux commits de la
série et vérifié à la main les ancres load-bearing. **Verdict : ENDOSSABLE.** Ce que
la relecture prononce :

1. **L'épinglage des deux côtés (`§2`, `§2-2`) est ENDOSSÉ** — les paires d'`α`,
   l'ordre croissant et l'uniformité « exactement un écran à `α = 1,0` » sont tenus
   pour une **dérivation propre** des constantes du module, du tranchage des images
   exactes du `§3` du spec readout et de la ligne `P-b` du prereg
   `où-vit-le-rendu` — les trois ancres du `§2-1` et du `§2`. C'est le point
   que le `§7` soumettait ; il cesse d'être une dérivation en attente.
2. **Le `§2-3` épingle la mesure du `§10`** — « un pas (33,3 ms) » se lit comme
   **écart PAR IMAGE entre les deux côtés**, et la lecture « latence absolue du mode
   interpoler » (qui donnerait une demi-période) est **écartée**. Voir le `§2-3`.
3. **Le `§5` est ENDOSSÉ COMME DÛ SANS ÉCHÉANCE**, garde `§4-6` comprise : la
   question reste ouverte, mais son statut de dû n'est plus en suspens.

Document rédigé par la session Claude ; **l'endossement est le commit de Romain.**
