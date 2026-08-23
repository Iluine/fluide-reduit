# SPEC — INTERPOLATION DE READOUT (porte 33,3) — 2026-08-23

> **CE DOCUMENT EST UN SPEC DE CODE MOTEUR, PAS UN PRÉ-ENREGISTREMENT DE MESURE.**
> Il décrit un composant à construire. **Il ne mesure rien, n'autorise aucun run,
> et ne chiffre pas `I`.** L'interdiction de `§A61` tient :
> `PREREGISTRATION.md:9927` « **Aucune nouvelle mesure 3D absolue ne devrait être produite avant que »
>
> Il se commit SEUL. **La revue de Romain est l'endossement.**

**Justification du composant, dans sa formulation exacte.** L'interpolation de
readout est **le seul chemin vers `I`**, et un composant permanent de **la
seule** architecture 30 Hz. Si la cadence tombe à 60 Hz, son écriture aura été
**le prix de la décision**, pas un composant. *Elle n'est pas « utile dans les
deux issues » — cette formulation était une sur-vente de portée, retirée.*

---

## §1. LE PLACEMENT, ET POURQUOI IL NE VIOLE PAS `§A48`

Nouveau module **`src/f1_gpu/interpolation_readout.py`**.

**L'interface est : le composant ÉCRIT un champ, le gather le LIT.**
`chemin_de_cout` reste **identique à l'octet** ; seule son *entrée* est permutée
— le champ `s` interpolé, passé par la même API publique de la pyramide. Rien
n'est remplacé.

La garde qu'il ne faut pas toucher :
`src/f1_gpu/chemin_de_cout.py:15` « la structure »). L'arithmétique est la plus bête permise : PLUS PROCHE VOISIN »

C'est une **garde gravée** (`§A48`, garde 1), **pas une dette**. Le plancher
« plus proche voisin » continue de décrire exactement ce que fait
`chemin_de_cout`, parce que ce module-ci ne touche pas à son arithmétique.

### §1-bis — L'ALTERNATIVE, ÉNUMÉRÉE POUR ÊTRE TUÉE

**« Fusionner l'interpolation DANS le gather »** — **MORTE par `§A48` garde 1.**
Elle rendrait l'arithmétique du gather plus chère que « la plus bête permise »,
donc son chiffre cesserait d'être un **plancher**, et le `0,4488 ms` de `§A52`
perdrait son étiquette. Écrite ici parce qu'un placement qu'on n'énumère pas
reste **disponible comme échappatoire tacite** (`§A62-2`, où `P-d` est écrit
pour être tué).

---

## §2. LE NOYAU — UN SEUL, AFFINE ; LES DEUX MODES EN SONT DEUX VALEURS DE `α`

```
s_out = (1 − α)·s_prev + α·s_cur

    mode INTERPOLER   α = ½     entre n et n+1     — deux états RÉELS
    mode EXTRAPOLER   α = 1½    depuis n−1 et n    — état qui n'a jamais existé
```

> **CONSÉQUENCE GRAVÉE : `I` EST INVARIANT AU MODE PAR CONSTRUCTION.** Même
> arithmétique, mêmes octets lus, mêmes octets écrits. Une mesure future ne
> pourra comparer que **la latence et le perçu** — **jamais un confondant
> d'arithmétique entre modes**. C'est une garde STRUCTURELLE : elle ne dépend
> d'aucune vigilance, elle tient parce qu'il n'y a qu'un noyau.

> **AMENDEMENT 3 (2026-08-23) — LA PREMIÈRE FORMULATION DU VERROU (b) ÉTAIT
> ÉTEINTE, ET C'EST UNE MUTATION QUI L'A DIT.** Le `§7` demandait la
> « **bit-identité entre les deux modes à `α` ÉGAL** ». L'énoncé est
> **incohérent** : les deux modes SONT deux valeurs de `α`, donc « les deux
> modes à `α` égal » ne décrit rien. Le test qui en est sorti comparait deux
> appels au **même** `α` — un test de DÉTERMINISME, que **tout** noyau branché
> sur `α` satisfait, puisqu'il reste déterministe à `α` fixé.
> **Un mutant le passait, et il passait TOUTE la suite d'alors** : un
> branchement `if α > 1` avec une autre formule.
>
> Ce que ce paragraphe affirme est qu'il n'existe **qu'un noyau**, et la
> conséquence testable est l'**AFFINITÉ** : `s_out` est linéaire en `α`, donc
> trois évaluations sont **colinéaires**. Un chemin propre à un mode la brise.
>
> **La faute a traversé trois étages** — spec, plan, implémentation — parce
> qu'aucun ne pouvait la voir depuis sa position, et **seule une MUTATION l'a
> montrée, pas une relecture**. C'est `§A53` au mot près, et c'est la même
> découverte que le 03/08 sur `chemin_de_cout`. Le verrou censé mécaniser ce
> paragraphe était lui-même **éteint**.

> **PORTÉE EXACTE DU VERROU RÉPARÉ — MESURÉE, PAS SUPPOSÉE.** Une première
> rédaction de cet amendement affirmait qu'un décalage constant de −0,001
> violait aussi la colinéarité. **C'est FAUX**, et c'est l'implémenteur qui
> l'a vu et signalé au lieu de corriger en silence. Vérifié en lançant les
> deux mutants :
>
> | mutant | résultat |
> |---|---|
> | décalage **conditionné au mode** (`if α > 1`) | **1 failed, 6 passed** — l'unique échec est la colinéarité |
> | décalage **inconditionnel** (`−0,001` partout) | **3 failed, 4 passed** — attrapé par les verrous de BORNES ; la colinéarité passe |
>
> Un décalage indépendant de `α` **s'annule** dans l'identité, puisque `s_out(0)`
> et `s_out(1)` le portent aussi. ⇒ **Les deux familles de verrous sont NON
> REDONDANTES : chacune est l'unique détecteur de sa classe.** Ce n'est pas une
> faiblesse — c'est la portée exacte, et l'écrire évite de croire que le verrou
> garde plus qu'il ne garde. **C'est `§A53` appliqué à la garde qui venait de
> réparer une violation de `§A53`.**

L'arithmétique affine est l'**hypothèse nulle** (null-first). Aucune
justification perceptuelle n'est revendiquée pour elle : si une mesure future la
condamne, c'est un incrément, pas une reprise.

---

## §3. LES FRAMES EXACTES CONTOURNENT LE NOYAU — ET C'EST CE QUI DÉFINIT `I`

À 60 images/s sur une physique à 30 Hz, **une image sur deux tombe sur un pas**
(`α ∈ {0, 1}`) et l'autre non.

> **TRANCHÉ : les images exactes CONTOURNENT le noyau.** Elles lisent `s_cur`
> directement, sans passer par le module.

**Le motif est la comptabilité gravée, pas une préférence** :
`claude/prereg-ou-vit-le-rendu-2026-08-04.md:87`
« | **P-b** | **porte 33,3** — physique 30 Hz, rendu 60 fps par interpolation du readout | `30·nonF + 60·R + 30·I` | »

Le terme est **`30·I`**, pas `60·I`. Si les images exactes traversaient le
noyau, elles le paieraient aussi et la comptabilité deviendrait `60·I` —
contredisant un document endossé. ⇒

> **DÉFINITION DE `I`, gravée ici** : `I` = **coût d'UNE image interpolée**,
> **30 par seconde**. C'est exactement le `30·` de `Δ = 30·(non-F − I)/C`.
> Sans ce tranchage, `I` porterait un facteur 2 caché.

### §3-bis — LA MISE À JOUR DE `s_prev` EST **DANS** `I` — petit frère du facteur 2

Le tampon `s_prev` doit être avancé d'un pas de physique au pas suivant. Ce
coût n'était pas nommé.

> **TRANCHÉ : il est DANS `I`.** Motif, et il est structurel : **cette
> maintenance n'existe QUE parce qu'on interpole** — à 60 Hz aucun `s_prev`
> n'est nécessaire. Elle survient **une fois par pas de physique**, soit
> **30/s à 30 Hz** — le même compte que `I`. La charger au **non-F** la ferait
> entrer dans une grandeur **gravée et mesurée SANS interpolation** : ce serait
> contaminer le **non-F gravé** (ancré au `§8`) avec un terme qui lui est étranger.
>
> ⇒ **`I` = noyau affine + maintenance de `s_prev`, par image interpolée.**
> L'arithmétique tombe juste : `30·(noyau + maintenance) = 30·I`.

**Deux régimes, et lequel s'applique est un DÛ, pas une supposition :**

| régime | condition | coût |
|---|---|---|
| **échange de pointeurs** | `F` écrit **hors place** | **0 octet** |
| **copie d2d** ← **C'EST CELUI-CI** | `F` écrit **EN PLACE** | `245 760 o` lus + écrits (n_fov = 64), 30/s |

> **DÛ FERMÉ (amendement 2) — c'est la COPIE, pas l'échange.** Sur la pyramide
> 2D, `F` écrit **EN PLACE** : `src/f1_gpu/pyramide.py:501` « _, dt_cfl = self.pas_f(fen, xp, fen) »
> — le tampon de sortie passé à `pas_f` **est** le tampon d'entrée `fen`. Un
> échange de pointeurs ne peut donc rien conserver : `s_prev` doit être une
> **COPIE RÉELLE, prise AVANT `pas_f`**.
>
> ⇒ **`I` porte `245 760 o` lus + écrits par image interpolée** (`n_fov` = 64),
> 30 fois par seconde. **Le terme n'est PAS nul.** Le dû se ferme du côté
> DÉFAVORABLE, et il se ferme **par lecture du code**, pas par une mesure.

*L'indication qui avait été donnée pointait dans l'autre sens et elle était
hors sujet* : le noyau **3D** écrit dans un `q_out` distinct
(`src/f1_gpu/substrat_fusionne_3d_param.py:336` « q_out[idx]           = un; »),
mais ce module vise la pyramide **2D**, et c'est elle qui décide. **C'est
exactement pourquoi elle avait été étiquetée « indication, pas un fait ».**

---

## §4. `s_out` EST ÉPHÉMÈRE — LA GARDE DE FOND

> **`s_out` est un pur produit de READOUT. Il n'est JAMAIS écrit dans l'état,
> JAMAIS lu par `F`.**

Le corpus distingue écriture éphémère et écriture persistante, et il est **plus
dur** encore sur le readout :
`PREREGISTRATION.md:6570`
« §A20-3 : l'état éphémère de readout est interdit tout court dans l'instrument), »

`claude/spec-p1-rendu-instrument-2026-07-25.md:141`
« 1. **Aucun état de readout load-bearing** — R1 est une fonction pure de `A`. Pour »

**Le risque exact, nommé** : si le champ interpolé fuyait un jour dans l'état,
il corromprait la **comptabilité de conservation** — un `s` qui n'a jamais été
produit par `F` entrerait dans le ledger. **Cette comptabilité n'est pas
négociable**, et une fuite ne se signalerait par aucun symptôme immédiat : c'est
la signature exacte de `§A53` (*un terme payé en arithmétique et nul en valeur
passe tous les verrous numériques*), retournée.

**Trois verrous, structurel d'abord** :

1. **STRUCTUREL** — `s_out` est écrit dans un **tampon de sortie séparé**, qui
   n'appartient pas à l'état de la pyramide. Le module **ne détient aucune
   référence en écriture** vers les tableaux d'état. Un verrou structurel est
   le seul que `§A53` ne sait pas contourner.
2. **DE PILE** — `_verrouiller_consommateur()` lève une `RuntimeError` explicite
   si l'appelant de `s_out` n'est pas sur le chemin de rendu. Même patron que
   `_verrouiller_appelant()` de `chemin_de_cout`. **Jamais un `assert`** : il
   disparaît sous `python -O` (`§A43`).
3. **DE TEST** — `§7`, verrou (c).

> **Le chemin de `s_out` se termine dans le rendu. Aucun autre consommateur
> n'existe, et aucun ne pourra exister sans AMENDEMENT de ce document.**

### §4-bis — CE MODULE ENFREINT À LA LETTRE LES DEUX GARDES QU'IL CITE, ET VOICI POURQUOI IL EN A LE DROIT

**Il faut le dire avant qu'un relecteur le trouve** : `§A20-3` et la spec P1
(toutes deux ancrées ci-dessous) posent qu'**aucun état de readout
load-bearing** n'existe — R1 est une fonction PURE. Or **`s_prev` EST un état de readout load-bearing : le
premier du code.** Citer ces gardes en appui sans nommer ce point, c'est
s'abriter derrière ce qu'on enfreint.

La défense, en deux jambes, **et elle ne vaut que parce qu'elle est écrite** :

1. **PÉRIMÈTRE.** Ces gardes sont scellées au périmètre de l'**INSTRUMENT** —
   la chaîne de mesure R1, l'ABX de session. Le texte le dit lui-même :
   `PREREGISTRATION.md:6570` « §A20-3 : l'état éphémère de readout est interdit tout court dans l'instrument), »
   Ce module est du **code moteur**, hors de ce périmètre. Il n'entre pas dans
   R1 et ne produit aucun stimulus d'essai.
2. **NATURE.** `s_prev` ne contient que des **COPIES de champs réellement
   produits par `F`**. C'est de l'**HISTOIRE**, pas de la **SYNTHÈSE**. La
   synthèse — `s_out` — reste, elle, **sans état** : fonction pure de
   `(s_prev, s_cur, α)`, vérifiée par le verrou (f) de `§7`.

> **DÛ NOMMÉ, POUR LE PREREG DE LA LECTURE D'ORIENTATION** : *chronométrer un
> composant À ÉTAT remet en cause l'hypothèse de PURETÉ de la chaîne de
> mesure.* Ce prereg devra **TRAITER** ce point — pas le découvrir en cours de
> route. Inscrit ici pour qu'il arrive à lui comme un dû, pas comme une
> surprise.


---

## §5. LE CLAMP — COMPTÉ ET REPORTÉ, JAMAIS FAIL-LOUD

`albedo` est **monotone bornée** (`A = 1 − exp(−s/s_half)`), donc un `s`
extrapolé hors domaine **se voit**. Le mode EXTRAPOLER peut produire `s < 0`.

> **TRANCHÉ : le clamp `s ≥ 0` COMPTE et REPORTE, il ne lève pas.** Ceci est du
> **code moteur** : il tourne. Le compteur de clamps est rendu avec le champ,
> par appel.

**Le seuil au-delà duquel un comptage INVALIDE une mesure n'appartient PAS à ce
document** : il appartient au **prereg de la lecture d'orientation**, qui n'est
pas écrit et n'est pas ordonné. Le graver ici serait choisir un seuil hors de
toute décision — exactement ce que `§A60` interdit.

*Ce point corrige une contradiction du design présenté en séance : « fail-loud
sur sortie de borne » et « compté, pas silencieux » sont deux comportements
incompatibles. Le second est retenu.*

---

## §6. LE TAMPON, CHIFFRÉ — ET SON ÉTIQUETTE

Un `s_prev` par fenêtre sert **les deux modes** (c'est `§2` : un seul noyau).

**Le tampon se dimensionne sur les PLANS `s`, pas sur les SLOTS.** Le tenseur
est `src/f1_gpu/pyramide.py:385` « `references[j]` de shape (n_slots(j), n_systemes(j), 4, n_fov, n_fov) »
— et un slot `c = 8` porte **DEUX systèmes** (`SYSTEMES_PAR_C`, `§A16`), donc
deux plans `s`. Pour V4 (2 fins `c=8` + 2 énergie `c=8` + 7 `c=4`) :
`4×2 + 7×1` = **15 plans `s` pour 11 slots.**

| `n_fov` | `15 × n_fov² × 4 o` | en Kio |
|---|---|---|
| 64 | **245 760 o** | **240 Kio** (exactement) |
| 52 | **162 240 o** | **158,44 Kio** |

Contre **3 781 Mo** de VRAM machine. Chiffré, non plaidé.

> ⚠ **CE QUE « +1 CHAMP » VEUT DIRE — ET CE QU'IL NE VEUT PAS.**
> L'empreinte est **ÉQUIVALENTE à +1 champ** sur les 4 de la pyramide
> (`CHAMPS_PAR_SYSTEME = 4`). **C'est une comparaison de TAILLE, et rien
> d'autre.**
>
> **`s_prev` ne vit PAS dans le tenseur que `F` consomme.** C'est un **tampon
> côté readout**, exactement comme `s_out`. La lecture inverse — `s_prev`
> stocké comme cinquième champ du tenseur — serait la fuite que le verrou 1 de
> `§4` déclare **structurellement impossible** : `F` balaie les champs, et un
> état de readout rangé parmi eux finirait consommé. L'ambiguïté est tuée ici
> plutôt que laissée à l'implémentation.

> ⚠ **ÉTIQUETTE : `11` slots — donc `15` plans — est la DEMANDE de V4, un
> MAJORANT de dimensionnement —
> pas un engagement.** V4 est mort en 3D (`§A53`, `R-3`) et **aucun cap ne
> l'atteint** (5,90 à 60 Hz au plancher mesuré du rendu). `11` est retenu parce
> qu'un majorant à 240 Kio ne coûte rien à assumer. Sans cette étiquette, ce
> serait un chiffre de coin de plus.

---

## §7. TESTS — TDD, `.venv/bin/pytest`, backend numpy

`src/f1_gpu/backend.py` grave la règle : `xp` est **cupy exclusivement pour la
MESURE** ; le backend **numpy n'existe que pour tester la LOGIQUE en VM**. Ces
tests sont de la LOGIQUE — ils ne mesurent rien.

| | verrou | ce qu'il attrape |
|---|---|---|
| **(a)** | `α = 0` rend exactement `s_prev` ; `α = 1` exactement `s_cur` | le noyau affine **aux bornes** — propriété du noyau, jamais un chemin de production (`§3` : les images exactes le CONTOURNENT) |
| **(b)** | **AFFINITÉ** : `s_out(α) = s_out(0) + α·(s_out(1) − s_out(0))`, vérifiée aux deux `α` de mode, hors zone de clamp | `§2` MÉCANISÉ — sans lui, l'invariance de `I` au mode est une promesse, donc une garde absente (`§A62-bis-2`). **Voir l'amendement 3 : la première formulation était éteinte.** |
| **(c)** | l'état de la pyramide est **bit-identique après appel** ; `_verrouiller_consommateur()` lève sur un appelant hors rendu | `§4` — la garde éphémère |
| **(d)** | le clamp **compte** et le compteur est rendu ; il ne lève pas | `§5` |
| **(e)** | le MÊME gather, non modifié, lit la vue **octet pour octet** à `α = 1`, sur un champ **NON UNIFORME** | la surface de `VueInterpolee` est suffisante ET propagée. **Voir l'amendement 4 : ce verrou ne prouve PAS la non-régression — c'est le `git diff` qui la porte.** |
| **(f)** | deux appels sur les mêmes entrées rendent les mêmes octets | fonction pure, anti-PERSIST |

---

## §8. LES ÉTIQUETTES, GRAVÉES À LA NAISSANCE DE `I`

**`I` sera « à l'échelle de l'instrument ».** La seule chaîne qui existe est la
pyramide 2D, et le gather 1920 est mesuré dessus. C'est **la même famille
d'étiquette** que le non-F retenu :
`PREREGISTRATION.md:9559`
« non-F **2D à l'échelle de l'instrument**, pas un rendu 3D à 1920. »

> C'est une **FORCE** pour le rapport `I/non-F` — deux grandeurs de **même
> échelle**, sur le **même instrument**, mesurables **adjacentes dans le temps**,
> donc un biais multiplicatif commun s'annule. Mais **non étiquetée, c'est
> `§A62-bis-4` qui recommence.** Le tampon est apposé ici, avant que la grandeur
> existe.

**DÛ NOMMÉ : le `I` 3D**, de la même famille que le non-F 3D de `§A53`
(`PREREGISTRATION.md:8826` « **DÛ, avec consommateur nommé** : le **non-F en 3D** (halos, remontée, »).

**`s` SEUL EST UN PLANCHER, PAS LE COMPTE FINAL.** Aujourd'hui la chaîne de
rendu ne consomme qu'un champ : `arcC_rendu` (R1, `§A37`) prend un champ
**albédo**, avec le chemin lumineux **identité `Y = A`**, et
`albedo(s, s_half)` ne lit que le **sédiment**. Mais l'interdiction héritée de
P2 mord ici, **du côté favorable, comme dans `§A62-5`** : *le coût inconnu du
rendu vit dans la COMPOSITION et dans l'optique au-delà de `Y = A` (R2+) ; la
dette se DÉPLACE, elle ne rétrécit pas.* Le lambertien `relief_shaded` (qui lit
`b0 + s`) est l'incrément **R2, post-P3**. ⇒ **La liste « un champ » est un
plancher étiqueté. Ne jamais en conclure « l'interpolation est bon marché ».**

---

## §9. LE PLACEMENT NON ÉNUMÉRÉ — DÛ, NON CHIFFRÉ

**« Interpoler l'IMAGE »** — 30 gathers complets + 30 mélanges écran —
**n'apparaît dans AUCUNE table de placements** : ni dans le `§3` du prereg
`où-vit-le-rendu` (`P-a`…`P-d`), ni dans `§A62-2`.

Ce que ça ferait, et pourquoi ça compte : sous ce placement le côté 30 Hz
paierait `30·R + 30·I_img` contre `60·R` à 60 Hz — donc **`R` ne s'annulerait
plus de `Δ`**, et l'annulation de `§A62-3` (sensibilité `1,3·10⁻¹⁵`), qui est le
résultat central de cette entrée, **suppose le placement ÉTAT**.

> **CONSIGNÉ, NON CHIFFRÉ.** Le chiffrer ici serait étendre une analyse hors de
> tout pré-enregistrement. **À traiter AVANT tout chiffrage de `I`.**
>
> Un argument, donné pour un argument et **pas pour un fait** : sous foveation
> mobile, deux champs écran consécutifs ne sont pas en registre (le centre
> fovéal a bougé, les fenêtres aussi), donc un mélange écran naïf fantômerait.
> **Non établi.**

---

## §10. LES DEUX LATENCES, NOMMÉES SÉPARÉMENT

| | mode INTERPOLER | mode EXTRAPOLER |
|---|---|---|
| latence de **VUE** (centre fovéal) | **0** | **0** |
| latence de l'**ÉTAT DU MONDE** | **un pas** (33,3 ms à 30 Hz) | **0** |

Le gather tourne au **centre fovéal COURANT à 60 Hz dans les deux modes** : la
vue ne retarde jamais. Seul l'état du monde retarde, et seulement en mode
interpoler.

*Confondre les deux — dire « +33,3 ms de latence d'affichage » — nomme mal ce
qui retarde. La faute a été commise en séance et corrigée avant ce document ; si
elle survivait, une mesure future trancherait sur un chiffre mal nommé.*

---

> **AMENDEMENT 4 (2026-08-24) — LE VERROU (e) PROMETTAIT PLUS QU'IL NE TIENT,
> ET SON CHAMP TÉMOIN AVAIT UN POINT AVEUGLE.** Deux constats d'une revue sur
> pièces, tous deux dans le texte de ce spec.
>
> **(1) Aucun test ne peut prouver « le gather rend les mêmes octets qu'AVANT
> l'existence de ce module »** — il y faudrait une référence historique. La
> première rédaction le demandait pourtant, et le test qui en est sorti
> comparait une constante repeinte **à elle-même** : n'importe quelle version
> du gather l'aurait satisfait. **Ce qui porte réellement cette garantie est
> `git diff --stat <base de branche> -- src/f1_gpu/chemin_de_cout.py`** — vide,
> et complété par `git log <base>..HEAD -- <fichier>` également vide : aucun
> commit de la série n'a jamais touché ce fichier, pas seulement l'état final
> qui coïncide. C'est `§A62-bis-2` appliqué à un test : *une garde promise est
> une garde absente*, et un test qui promet plus qu'il ne tient est de la même
> famille.
>
> **(2) LE CHAMP TÉMOIN ÉTAIT CONSTANT, ET LE GATHER Y EST INSENSIBLE À
> `centre_fin`.** Mesuré : en décalant `centre_fin` de +1 dans la vue, **les
> DOUZE verrous du fichier passaient**. Une régression silencieuse de la
> surface exposée n'était vue par rien. Le champ est désormais peint **non
> uniforme**, et le test **vérifie d'abord qu'il l'est** — une garde sur la
> garde, parce qu'un témoin qui redeviendrait constant rouvrirait le trou sans
> bruit.
>
> **Troisième fois de la série qu'un verrou s'avère plus étroit que sa prose**
> (après le `§7(b)` éteint et sa portée sur-vendue). Les trois ont été trouvés
> par MUTATION, aucun par relecture — `§A53` ne se lasse pas.

---

## §12. AMENDEMENT 5 — CE QUE LA REVUE FINALE A MIS EN DÉFAUT

Trois clauses de ce spec sont **fausses ou sur-vendues**. Elles sont amendées
ici, pas réécrites plus haut.

### §12-1 — LE BUG DE REGISTRE, ET L'ARGUMENT DU `§9` QUI TOMBE AVEC LUI

Les fenêtres sont **fovéa-relatives** (`src/f1_gpu/pyramide.py:264` « centre_j = centre_fin // (2 ** (self.niveau_fin - j)) »)
et `frame()` **translate le contenu** avant d'appliquer `F`, dans la même
boucle (`src/f1_gpu/pyramide.py:491` « fen[...] = xp.roll(fen, -dx, axis=-1) »
puis `src/f1_gpu/pyramide.py:501` « _, dt_cfl = self.pas_f(fen, xp, fen) »).
`s_prev`, lui, est indexé en coordonnées **locales** et n'est jamais roulé.

⇒ **Le mélange combinait la cellule monde `x` avec la cellule `x − dx`.** Sur
les colonnes entrantes, la valeur produite **n'a jamais existé** : mesuré à
**153,52** entre deux cellules qui n'ont jamais coexisté. Silencieux —
`clamps == 0`, aucune levée.

> ⚠ **LE `§9` DE CE SPEC EST FAUX SUR CE POINT.** Il donnait le fantômage sous
> fovéa mobile comme argument **pour** le placement ÉTAT, en le rangeant du côté
> du placement IMAGE. **Le placement ÉTAT le porte à l'identique, et pour la
> raison exacte que le `§9` nommait lui-même** (« les fenêtres aussi »).
> L'argument était étiqueté « donné pour un argument, pas pour un fait » — **il
> était faux, et l'étiquette est ce qui a permis de le dire sans rien rétracter
> d'autre.**

**Ce qui est fait, et ce qui ne l'est pas.** Le composant **REFUSE désormais de
produire** sous fovéa mobile : la corruption silencieuse est devenue un échec
bruyant. **Le design n'est PAS tranché.** Le point de capture correct est
post-roll / pré-`pas_f`, **par niveau**, à l'intérieur de `frame()`. Trois
voies, coûts différents, **toutes dans `I`** et **aucune neutre pour le
`§3-bis`** : un point d'extension dans `frame()`, un `s_prev` en coordonnées
monde, ou un `roll` miroir de `s_prev`. **Décision de Romain.**

### §12-2 — LE `§2` SUR-VEND : « MÊMES OCTETS ÉCRITS » EST FAUX DEPUIS LE `§5`

Le `§2` grave que `I` est invariant au mode « par construction — mêmes octets
lus, **mêmes octets écrits** ». Le clamp du `§5`, ajouté après, est une
**écriture par masque dont le volume dépend des DONNÉES** et n'est atteignable
qu'en EXTRAPOLER. Les deux modes exécutent donc des quantités de travail
différentes, et les tests le gravent eux-mêmes (`clamps > 0` en extrapoler,
`clamps == 0` en interpoler).

⇒ **L'invariance de `I` au mode tient sur l'ARITHMÉTIQUE AFFINE, pas sur le
clamp.** Le verrou (b) le dit d'ailleurs à sa façon : il choisit ses valeurs
« pour que le clamp ne morde jamais ». **Le prereg de la lecture d'orientation
devra traiter ce confondant**, pas le découvrir.

### §12-3 — LE `§4` DÉCRIT UNE LISTE BLANCHE, LE CODE EST UNE LISTE NOIRE

Le `§4` écrit que la serrure lève « si l'appelant de `s_out` **n'est pas sur le
chemin de rendu** ». Le code fait l'inverse : il lève si l'appelant **est
nommé** dans `_MODULES_ETAT`. Tout module non nommé passe.

Le commentaire du module est honnête et nomme cette limite ; **c'est ce spec qui
ne l'avait pas été.** Et la liste s'est révélée fausse **deux fois** en deux
rondes de revue — cinq modules manquants, puis trois. **Aucun test ne peut dire
ce qui manque à une liste** : c'est le **verrou STRUCTUREL** qui porte la
garantie, et lui seul.

Second point de portée, non corrigé : la serrure garde la **production** de
`s_out`, pas son **accès**. Un module d'état qui reçoit une `VueInterpolee`
produite ailleurs lit `vue.fenetres` sans rien déclencher.

---

## §11. CE QUE CE DOCUMENT NE FAIT PAS — **ARRÊT**

- **Aucune mesure de `I`.** `§A61` l'interdit toujours, et `§A63` a rendu
  INDÉTERMINÉ : `σ_r` n'est pas gravée, `§8` de la re-qualification reste FERMÉ.
- **La lecture d'orientation n'est PAS ordonnée**, et son prereg n'est pas
  écrit. Sa structure devra être **tripartite** : ce que la décision demande
  (une séparation non nulle), ce que l'instrument rend, et **l'écart entre les
  deux déclaré pour ce qu'il est** — la zone INDÉTERMINÉE.
- **Ce document ne prononce ni sur la cadence, ni sur le côté, ni sur la
  Porte 3 de `§A58`.** La cadence appartient à Romain.
- **Rien ne s'enchaîne.** Le signe de `Δ` attendra son ordre.
