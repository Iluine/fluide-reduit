# Clôture du chantier boucle de rendu — extraction du ledger SDD (2026-08-25)

> **POURQUOI CE DOCUMENT EXISTE.** Le workspace SDD du chantier
> (`.superpowers/sdd/plan-boucle-rendu/`, git-ignoré) va être supprimé — c'est le
> protocole une fois la branche propre et poussée. Mais il était le **seul lieu de
> vie** des findings parqués avec leurs motifs et des points remontés non tranchés.
> Un répertoire git-ignoré comme unique provenance est la faute `§A41-c` sous une
> autre forme : la provenance redevient un nom de fichier, invisible aux sessions
> cloud et à toute reprise. Ce document extrait le résidu porteur **avant**
> l'effacement. Les blocs cités sont **VERBATIM du ledger** (état au moment du
> ledger — les `:NNN` internes aux citations datent de là et ne sont pas des
> ancres vivantes). Les diffs de revue et les briefs meurent avec le workspace
> sans perte : ils sont reconstructibles depuis git.
>
> Rédigé par la session Cowork (Claude), ledger lu sur pièces ;
> **l'endossement est le commit de Romain.**
>
> ⚠ **CORRECTION DE RELECTURE (session Claude Code, 25/08).** Une première
> rédaction déclarait la numérotation 1–14 du tri **perdue** et en donnait deux
> lectures « vraisemblables ». **Elle n'est pas perdue** — le rapport de la revue
> finale de branche est dans le transcript de la session qui l'a commandée, et le
> tableau complet est restitué au `§3`. **Les deux lectures étaient FAUSSES**, et
> elles étaient vérifiables : le `n° 10` n'est pas le Minor « rend » vs attribut
> mais **M6 / le `§2-1`** ; le `n° 13` n'est pas l'entorse `np.array_equal` mais
> **la prose de la chasse à l'horloge**. Corrigé aux `§2-bis` et `§3`.
> *Une lecture honnêtement étiquetée « lecture » reste fausse si la source
> existait et n'a pas été ouverte* — c'est la leçon du chantier appliquée à son
> propre document de clôture.

---

## §1. ÉTAT FINAL — LES FAITS

- **14 commits**, `7f1808d..b35c8a3`, tous poussés :
  `origin/arc-a-manche2-registre` = `b35c8a3` = HEAD, arbre propre.
- **Suite complète : `1479 passed in 692.13s`, exit 0, au commit `b35c8a3`** — le
  hash figure en tête de la sortie. Progression au rythme des verrous :
  1477 (`2f4353f`) → 1478 (`18952fa`) → 1479 (`b35c8a3`).
- `src/f1_gpu/boucle_rendu.py` est en **gel de comportement depuis `18952fa`** :
  les deux commits suivants (`87baf71`, `b35c8a3`) ne touchent que docstrings et
  prose de tests.
- Spec `claude/spec-boucle-rendu-2026-08-24.md` **endossée** (relecture du
  2026-08-24 : ENDOSSABLE, amendements faits). Le côté n'est **pas choisi** ;
  `§A61` est **intact** — aucune mesure, aucun chiffre de `I`.

---

## §2. LES TROIS POINTS REMONTÉS À LA CLÔTURE — NON TRANCHÉS, POUR ROMAIN

Verbatim du ledger :

> 1. **`verifier_ancres.py` est structurellement aveugle à la classe de renvoi
>    qui a produit R1.** R1 a survécu à un renommage parce que c'est un renvoi
>    **par nom de test**, pas une ancre `fichier:ligne`. Le vérificateur balaie
>    les secondes ; les premières pourrissent **en silence** à chaque renommage,
>    et ce fichier en compte plusieurs dizaines — les docstrings de ce dépôt se
>    citent mutuellement, c'est leur qualité. Le contrôle serait bon marché (un
>    renvoi `test_...` doit correspondre à une fonction existante). **Ce n'est
>    pas propre à la tâche 3 : c'est un trou d'outillage qui croît avec la
>    discipline de citation elle-même.**
> 2. **Ce qui garde `§A61` ici n'est mécanisé que pour un seul `print`.** La
>    vague finale a établi qu'un chronomètre tourne par transitivité et que ce
>    qui tient `§A61` est qu'**aucun chiffre n'en sorte**. Cette garde n'est
>    mécanisée que par l'égalité exacte de `capsys.out` sur l'unique `print` du
>    driver. `TransfertComptable` accumule ses `h2d_ms`/`d2h_ms` dans `bilans`
>    pendant tout le run, et rien n'empêcherait un futur driver de la même
>    famille de les emporter dans un JSON — **sans intention, par simple
>    sérialisation d'un objet qui les contient**. Le garde-fou tient parce que la
>    surface de sortie est d'une ligne ; **il ne se transporte pas.** Dû nommé,
>    non ouvert : c'est au prereg de la lecture d'orientation de décider ce qu'un
>    driver a le droit d'écrire.
> 3. **La démo prouve que la chaîne tourne ; elle ne rend pas visible ce qu'elle
>    orchestre.** Déjà jugée « limite honnêtement nommée » — remontée pour une
>    raison de **séquence** : le jour où l'on voudra regarder le 2:1, et ce jour
>    viendra puisque le but est un moteur qui tourne, les trois voies (`pas_f`
>    injecté, plus de bits, cumul sur plusieurs ticks) appartiennent chacune à un
>    prereg qui n'est pas écrit. **Aucune n'est un réglage de driver.**

## §2-bis. LE QUATRIÈME POINT POUR ROMAIN, QUE LE RAPPORT DE CLÔTURE N'A PAS REDIT

**Ce point n'est PAS le `n° 10` du tri** — c'est un **Minor distinct** de la revue
finale de branche, qui ne faisait pas partie des 14 parqués et que la revue a
explicitement « laissé à Romain ». Le `n° 10`, lui, est **M6 / le `§2-1`**, repris
au `§3` sous T2-5. Verbatim du Minor :

> **Minor laissé à Romain :** le `§3` dit que le tick « rend » les compteurs ; le
> code les expose en attribut. Le choix est motivé et jugé **meilleur**, mais
> diverge de la lettre — une ligne au `§3` le rangerait si Romain amende un jour.

---

## §3. LES FINDINGS PARQUÉS, PAR PROVENANCE — « PEUVENT RESTER OUVERTS », AVEC LEURS MOTIFS

Tri de la revue finale, verbatim : « **Tri des 14 findings parqués : 12 “peuvent
rester ouverts”, 1 à corriger (n° 13), 1 à porter à Romain (n° 10).** »

**LA NUMÉROTATION, RESTITUÉE** — elle n'était pas perdue. Tableau du rapport de la
revue finale de branche, avec sa colonne de verdict :

| n° | finding | tâche | verdict |
|---|---|---|---|
| 1 | renvoi à sens unique épigraphe/`§7` → `§8` | T0 | peut rester ouvert |
| 2 | pouvoir mince du verrou (b) | T1 | peut rester ouvert |
| 3 | quasi-tautologie de `test_les_deux_cotes_couvrent_la_table_des_couples` | T1 | peut rester ouvert |
| 4 | deux points de forme | T1 | peut rester ouvert |
| 5 | `_origines` non photographié | T1 | peut rester ouvert |
| 6 | `monde0` comparé en totalité | T1 | peut rester ouvert |
| 7 | pas d'accesseur public sur `TransfertComptable` | T1 | peut rester ouvert |
| 8 | `clamps == 0` / `non_finis == 0` intuable | T2 | peut rester ouvert |
| 9 | la moitié 2 masque la moitié 3 | T2 | peut rester ouvert |
| **10** | **M6 — le `§2-1` ne tient que sur le verrou de compte** | T2 | **ni fix ni ouvert : à porter à l'auteur** |
| 11 | « lignes ≤ 79 » inexact | T2 | peut rester ouvert |
| 12 | `cote_px` sur le seul `COTE_INTERPOLER` | T3 | peut rester ouvert |
| **13** | **prose de la chasse à l'horloge plus large que sa portée** | T3 | **à corriger avant remise** |
| 14 | `argparse._actions`, API privée | T3 | peut rester ouvert |

⇒ Le **`n° 13`** est la **prose de la chasse à l'horloge**, corrigée dans la vague
finale par **I2** (et non l'entorse `np.array_equal`, qui était **I3** et un Minor
neuf de la revue, hors des 14). Le **`n° 10`** est **M6 / le `§2-1`**, ci-dessous
en T2-5 — c'est **lui**, le point que la revue portait à Romain, et c'est le plus
load-bearing du chantier.

Les parqués, tels que les rapports de tâche les portent :

**Tâche 1 :**

> 2. **Le verrou (c)** teste un mécanisme de `interpolation_readout`, que je n'ai
>    pas le droit de muter : son pouvoir de détection reste le seul des verrous
>    de ce fichier à n'être pas mesuré par mutation.
> 3. **Le montage ne garde pas « installer avant le premier `frame()` »**,
>    délibérément et par écrit. L'analyse tombe si un appelant futur appelle
>    `melanger` hors du tick.
> 4. **`ruff` n'est pas installé** dans le `.venv` de ce dépôt — aucune passe de
>    lint sur les deux fichiers.

(Le point 1 de T1 — le plan désaligné sur la spec — est mort avec le chantier :
la spec a fait autorité, comme ordonné.)

**Tâche 2 :**

> 1. **Deux assertions de (e) n'ont été tuées par aucun mutant : `clamps == 0` et
>    `non_finis == 0`.** Ce sont des gardes sur la **validité du témoin**, pas
>    sur le module [...] Je les rapporte comme non tuées plutôt que de laisser
>    croire qu'elles verrouillent le module.
> 2. **La moitié 2 de (d) masque la moitié 3 sous la plupart des mutants** [...]
>    **une revue future qui lirait le tableau sans l'attribution fine croirait
>    que la moitié 3 ne sert à rien**. C'est écrit ici pour qu'elle ne le croie
>    pas.
> 5. **(e) ne peut pas distinguer le gather direct d'un `melanger(1.0)`** (M6).
>    Ce n'est pas un trou, c'est la frontière que le `§2-1` dessine — mais elle
>    signifie que **le `§2-1` tient sur un verrou de COMPTE et sur lui seul**. Si
>    `test_un_seul_melanger_par_tick_et_jamais_a_alpha_exact` venait à être
>    affaibli, la comptabilité `30·I` ne serait plus gardée par rien, et aucun
>    verrou de pixels ne le signalerait. Nommé pour qu'il ne soit pas retiré
>    comme redondant.

**Tâche 3 :**

> 1. **La couverture du `cote_px` n'est constatée que sur `COTE_INTERPOLER`.**
>    Les deux côtés partagent le même gather, le même `geo`, le même `cote_px`
>    et le même `centre_fin` : leur prédicat de couverture est identique. Risque
>    faible, non corrigé.
> 3. **`construire_parseur()._actions` est une API privée d'`argparse`**, sans
>    alternative publique pour introspecter `required` et `choices`. Noté, non
>    corrigé.

(Le point 2 de T3 — la chasse à l'horloge locale au driver — est devenu **I1**,
traité dans la vague finale : jeu d'horloges partagé entre module et driver,
`Event`/`sleep`/`synchronize` inclus.)

Motifs notables du tri, verbatim :

> le verrou (b) a vu **sa promesse inter-tâches honorée** par (d) ; `_origines`
> est « correctement consigné plutôt qu'ajouté » ; `monde0` en totalité échouera
> « bruyamment avec sa clé sous les yeux » — « c'est la bonne façon » ;
> `clamps == 0` est « le régime honnête » ; le `cote_px` sur un seul côté est
> **vérifié sans effet** (les deux côtés partagent gather et centre).

---

## §4. PROVENANCE DE LA QUALITÉ DES VERROUS — LES CAMPAGNES DE MUTATION

Verbatim :

> **Bilan des campagnes de mutation : 34 mutants sur trois campagnes, 33 tués.**
> Le survivant est un mutant **équivalent en comportement** (niveau de
> compression `zlib`), nommé comme tel.

Et le fait le plus utile à une session future, verbatim :

> **7 mutants du module, 7 tués. Quatre sont invisibles à TOUT verrou de
> comportement** et ne meurent que par les contrôles de source : `sleep` et
> `cuda.Event` **en code jamais exécuté** ; l'écriture **indexée** du tampon ;
> l'écriture par **alias local**. Les deux derniers parce que le tampon n'est ni
> dans la photo d'état de (a) ni distinguable par le témoin de (e), qui relit le
> tampon corrompu et concorde avec lui.

⇒ **Affaiblir un contrôle de SOURCE de `tests/test_boucle_rendu.py` au motif
qu'« aucun comportement ne change » est exactement la faute que ces quatre
mutants documentent.** La portée exacte de ce que le contrôle d'écriture ne voit
pas (alias du tableau, `.fill()`, `np.copyto`) est écrite dans sa docstring — la
promesse et la portée coïncident, c'est un choix, pas un oubli.

---

## §5. CE QUI MEURT AVEC LE WORKSPACE, ET POURQUOI C'EST ACCEPTABLE

Les diffs de revue (`review-*.diff`) : reconstructibles par `git diff` sur les 14
commits. Les briefs et rapports de tâche : du processus, dont le porteur est
extrait ci-dessus. Le rapport de la revue finale de branche : résumé au ledger,
repris en `§3` — **et sa numérotation 1–14 est restituée là**, contre ce
qu'affirmait la première rédaction de ce document.

⇒ **Rien de porteur ne meurt.** La « seule perte réelle » qui était nommée ici
n'en était pas une : la source existait, elle n'avait pas été ouverte. Ce qui
disparaît est du processus — l'ordre des rounds, les paquets de diff, les briefs —
et git le porte déjà ou le rend reconstructible.

Après commit et push de ce document, le workspace peut être supprimé.
