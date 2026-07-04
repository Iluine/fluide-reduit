# Rapport — Famille 2 : quadtree de moyennes (`src/summary_quadtree.py`)

Brief : `.superpowers/sdd/refond-task8-quadtree-brief.md` (contrat `a8ed659` pocCascade2phys +
amendement plan `78fbe0c`). Branche `arc-a-etat-complet`.

## Implémentation

### Représentation de la topologie

`SummaryQT` = dataclass gelée `eq=False` (leçon revue M-0bis : pas d'`__eq__`/`__hash__` générés
sur des champs ndarray) avec EXACTEMENT trois champs :

- `topology` : `np.ndarray` uint8 de bits **préordre** — 1 = nœud interne (4 enfants), 0 = feuille.
  n_nœuds = `topology.size`.
- `means` : moyennes des feuilles (float64), dans le même ordre préordre que les bits à 0.
  n_feuilles = `means.size`.
- `shape` : (64, 64).

RIEN d'autre n'est stocké : position/profondeur/taille de chaque feuille sont dérivées du parcours
préordre lui-même (ordre des enfants FIXE et documenté : TL, TR, BL, BR = `_ORDRE_ENFANTS`,
utilisé identiquement à l'encodage et au décodage). Le décodage (`_decoder_preordre`) vérifie que
la topologie est bien formée : tous les bits ET toutes les moyennes consommés exactement, aucun
bit interne à la profondeur 6 (`ValueError` sinon) ; non-négativité du régénéré (`AssertionError`,
jamais un `assert` nu).

### Structure du glouton (`summarize_qt`)

1. **Pyramide de moyennes** (`_pyramide_moyennes`) : niveaux d ∈ [0, 6], `niveaux[6]` = champ,
   chaque niveau plus grossier = UNE étape de cascade 2×2 (`((a+b)+(c+d))/4`) du niveau plus fin —
   jamais de réduction plate numpy (leçon gravée). C'est ce qui rend les moyennes d'un bloc peint
   constant bit-exactes, donc les gains sur le champ repeint exactement 0.0, donc S∘R = id arbre
   inclus.
2. **Gains précalculés** (`_gains_par_profondeur`) : gain[d][r,c] = n_c·Σ_enfants(moy_e − moy_p)²,
   vectorisé par niveau — arithmétique dérivée des moyennes exactes (la contrainte cascade porte
   sur les moyennes, pas sur le gain).
3. **Tas min sur `(−gain, y, x)`** : pop = gain max d'abord, puis plus petit y, puis plus petit x
   — exactement le tie-break gravé. (y, x) = coin haut-gauche en coordonnées fines ; deux feuilles
   courantes ne partagent jamais un coin (partition), donc pas d'ex æquo ambigu ; (depth, row, col)
   en queue de tuple par sûreté et pour retrouver le nœud.
4. **Candidats** : profondeur < 6 (feuille 1×1 insécable) ET gain strictement > 0 (« un nœud à
   gain nul n'est JAMAIS splitté », même budget restant — condition de la projection).
5. **Budget vérifié AVANT de committer** : futur = (feuilles+3) + ceil((nœuds+4)/32) ; si
   futur > budget → refus et arrêt DÉFINITIF. Justifié : le coût d'un split est structurel
   (+3/+4 quel que soit le nœud), donc si le meilleur candidat ne rentre pas, aucun ne rentre.
6. Sérialisation préordre unique en fin de construction (bits + moyennes depuis la pyramide).

Anti-fuite : copie systématique du champ à l'entrée (`_valider_champ`), aucune référence vivante ;
`regenerate_qt` a UN paramètre, tout provient de `summary`.

### Argument S∘R = id arbre inclus (résumé)

Sur le champ repeint : (a) sous chaque feuille, blocs constants → cascade bit-exacte → gains
sous-feuille = 0.0 exact → jamais candidats ; (b) au niveau de chaque feuille, la moyenne cascade
du bloc repeint == la moyenne stockée (cascade d'une constante : ((v+v)+(v+v))/4 = v exact) ; (c)
par induction vers le haut, toute la pyramide aux positions feuille-ou-au-dessus est bit-identique
à celle du champ original → mêmes gains, même ordre de tas, mêmes tailles → même arbre, mêmes
moyennes. Vérifié par le test 1 sur 3 champs × 7 budgets (topologie ET moyennes en array_equal).

## Évidence TDD

- **RED** : `tests/test_summary_quadtree.py` écrit d'abord ; `ModuleNotFoundError:
  No module named 'src.summary_quadtree'` constaté à la collecte (avant toute implémentation).
- **GREEN** : implémentation de `src/summary_quadtree.py` ; premier run : 79/85 verts, 6 rouges —
  les 6 rouges étaient des DÉFAUTS DE TESTS, pas d'implémentation :
  1. *Tie-break* : mon champ utilisait d=0.1 (non dyadique) → les 4 gains « égaux » différaient
     d'ulps, le tie-break n'était pas réellement exercé. Corrigé avec d=0.5 et centres dyadiques :
     égalité des gains bit-à-bit, le test vérifie alors la topologie attendue exacte
     (split du quadrant top-left, plus petit y puis x). Passe.
  2. *Invariants vs champ ORIGINAL* : la clause du brief était mathématiquement insatisfiable
     (cf. « Déviation documentée » ci-dessous). Reformulée fidèle au contrat.
- Suite quadtree finale : **122 passed** (0.54 s). Suite complète : cf. section Tests.

## Déviation documentée (test 7, invariants) — la seule

**Lettre du brief** : « masse totale et les 16 masses de sous-domaines 16×16 recalculées depuis
(topologie, moyennes, tailles de feuilles) == celles du champ ORIGINAL à 1e-12 relatif, et celles
du champ régénéré EXACTEMENT ».

**Problème (prouvé, pas supposé)** : pour les 16 masses vs l'ORIGINAL, c'est impossible pour TOUT
résumé tronqué — une feuille 32×32 ou 64×64 couvrant plusieurs sous-domaines 16×16 ne retient que
la somme de son bloc et redistribue donc la masse UNIFORMÉMENT entre eux. Contre-exemple exécuté
avant de toucher au test : masse 256 concentrée dans le sous-domaine (0,0), budget 5 (seul le
split racine tient) → masses régénérées 64/64/64/64 au lieu de 256/0/0/0 (totale : 256 = 256,
préservée). C'est information-théorique : 16 masses indépendantes ne se reconstruisent pas depuis
moins de floats dans la région. Échec observé empiriquement d'abord (sparse-64 : 21.02 vs 17.17).

**Le contrat gravé (`a8ed659`, qui fait foi) ne demande PAS cela** — verbatim : « Invariants
(masse totale, masses 4×4) : DÉRIVABLES exactement des feuilles (les feuilles dyadiques ne
chevauchent pas les sous-domaines 16×16 ou les contiennent entièrement) — non stockés, pas comptés
au budget, vérifiés en test. » La clause du brief sur-dérive du contrat (le non-chevauchement
donne l'exactitude vs le champ RÉGÉNÉRÉ et la préservation de la masse TOTALE et PAR FEUILLE vs
l'original — pas la préservation par sous-domaine).

**Ce qui est testé à la place** (note de portée en tête de section 7 du fichier de tests) :
- (i) invariants dérivés du résumé (reconstruction indépendante + sommes en cascade) == ceux du
  champ régénéré **EXACTEMENT** (== flottant strict), 16 masses + totale, 3 champs × 7 budgets —
  la clause du contrat, telle quelle ;
- (ii) masse totale ET masse PAR FEUILLE vs champ ORIGINAL à 1e-12 relatif, 3 champs × 7 budgets —
  les invariants réellement préservés (par-feuille ⇒ totale ; testés séparément pour localiser) ;
- (iii) masses 16×16 vs ORIGINAL à 1e-12 restreint aux sous-domaines entièrement dérivables
  (toutes feuilles incluses), avec garde anti-vacuité (≥ 1 sous-domaine porteur de masse couvert
  au budget 2048 par champ) ;
- le contre-exemple lui-même, gravé en test exécutable
  (`test_masses_sous_domaines_non_preservees_sous_troncature`).

**Conséquence à remonter au fork** : contrairement à la famille 1 (qui STOCKAIT les 17 invariants,
17 floats comptés), la famille 2 ne préserve PAS les 16 masses de sous-domaines vs l'original sous
troncature. Si un harnais de mesure ou une lecture de verdict s'appuie sur cette préservation, il
doit le savoir. La masse totale, elle, est préservée à ~1e-16 relatif par construction.

## Tests + résultats

- `tests/test_summary_quadtree.py` : **122 tests**, tous verts (0.6 s).
  1. S∘R = id bit-à-bit ARBRE INCLUS : 3 champs (sparse ~5 %, aléatoire positif dense, bosses
     réel-like) × 7 budgets {32, 64, 128, 256, 400, 1024, 2048} — topologie ET moyennes en
     `array_equal` strict : **21/21**.
  2. Idempotence stricte R(S(R(S(x,B)),B)) == R(S(x,B)) : 21/21.
  3. Déterminisme cross-process (hash SHA-256 via subprocess, motif de `test_summary.py`) : 1/1.
  4. Anti-fuite (signature à 1 paramètre + mutation décoy) : 2/2.
  5. Comptabilité exacte : constante → 1 feuille/1 nœud/size 2 ; quatre-quadrants → 4/5/5 ;
     jamais hors-budget (21 cas) ; refus du split qui dépasserait (budget 4, gain > 0 prouvé
     par contraste) : 24/24.
  6. Gain nul jamais splitté (constante, budget 2048 → racine seule) : 1/1.
  7. Invariants dérivés : (i) exact vs régénéré 21/21 ; (ii) totale + par-feuille vs original
     21/21 ; (iii) 16×16 dérivables vs original + anti-vacuité 3/3 ; contre-exemple 1/1.
  8. Monotonie SSE non-croissante en budget sur la grille : 3/3.
  9. Champ nul → racine seule, régénéré zéros bit-exacts : 1/1.
  + Tie-break à gains EXACTEMENT égaux (champ dyadique construit, topologie attendue exacte) ;
    garde grille de budgets.
- Suite complète : **286 passed** (164 existants + 122 nouveaux), aucun fichier existant modifié.
- Comptabilité vérifiée à la main hors tests : budget 2048 sur champ dense → 655 splits, feuilles
  1966, nœuds 2621, size = 1966 + ceil(2621/32) = 2048 exactement ; split 656 ferait 2051 > 2048.
  Budget 400 → size 398, split suivant 402 > 400, refusé.
- Performance : `summarize_qt(64×64, budget=2048)` = **0.002 s** sur champ dense (pire cas,
  tous gains > 0) — contrainte < 1 s tenue avec marge.

## Fichiers

- `src/summary_quadtree.py` (nouveau) — `SummaryQT`, `summarize_qt`, `regenerate_qt`,
  `size_floats_qt` + helpers privés (`_pyramide_moyennes`, `_gains_par_profondeur`,
  `_decoder_preordre`).
- `tests/test_summary_quadtree.py` (nouveau) — 122 tests.
- Aucun fichier existant touché (vérifié : `git status` ne montre que les 2 nouveaux fichiers).

## Self-review

- **Chaque clause de la spec testée ?** Oui : partition dyadique/profondeur 6 (implicites dans
  S∘R + décodage qui refuse un interne à depth 6), gain SSE (tie-break exercé sur égalité exacte),
  gain max d'abord (monotonie + tie-break), gain nul jamais splitté (test 6 + argument S∘R), arrêt
  budget/gain (tests 5), comptabilité +3/+4 et ceil (tests 5, cas mains), budget respecté après
  chaque split (21 cas + refus), dataclass eq=False à 3 champs (rien d'autre stocké — les tests
  dérivent tout de topologie+moyennes+shape), regenerate 1 paramètre sans clip (signature +
  non-négativité par raise), cascade 2×2 (implémentation + conséquence testée : S∘R strict).
- **S∘R strict à TOUS les budgets sur les 3 champs ?** Oui — 21 combinaisons, topologie ET
  moyennes, array_equal.
- **Comptabilité exacte vérifiée à la main ?** Oui — cas construits (2 ; 5) + formule fermée
  (1+3k, 1+4k) revérifiée sur k=655 et k=127, refus aux frontières exactes.
- **Fichiers existants intouchés ?** Oui. **Interface propre ?** 4 symboles publics documentés,
  erreurs par `raise` typés, prêts pour `run_arcA_measure_qt.py`.
- Points faibles assumés : le tie-break n'est exercé qu'à profondeur 1 (construire une égalité
  exacte multi-profondeur sans perdre le contrôle de l'ordre est fragile) ; la récursion du
  préordre est bornée par depth ≤ 6 (pas de risque de pile) ; `budget < 2` lève ValueError
  (non spécifié au contrat, documenté — la grille gravée commence à 32).

## Préoccupations

1. **La déviation test 7** (ci-dessus) : reformulation fidèle au contrat `a8ed659`, mais c'est une
   ARBITRATION D'IMPLÉMENTEUR sur la lettre du brief — à valider par le contrôleur. Si la lecture
   voulue était bien « 16 masses vs original partout », alors la famille 2 telle que gravée ne
   peut pas la satisfaire (contre-exemple exécutable au fichier de tests) et il faut le trancher
   AVANT le harnais de mesure.
2. Conséquence architecture : la famille 2 ne restitue pas les masses de sous-domaines sous
   troncature (la famille 1 les stockait). À garder en tête pour la lecture des verdicts Δχ si
   la métrique en dépend.
3. Le tie-break (gain, y, x) est implémenté via tas sur (−gain, y, x) : correct tant que deux
   feuilles courantes ne partagent jamais leur coin haut-gauche (vrai pour une partition) —
   documenté dans le code.
