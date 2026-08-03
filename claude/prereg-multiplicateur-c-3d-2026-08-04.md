# PRÉ-ENREGISTREMENT — LE MULTIPLICATEUR `c` EN 3D (2026-08-04)

> **Statut : PRÉ-ENREGISTREMENT, écrit AVANT tout run et AVANT toute ligne de
> kernel paramétré.** Le journal `pocCascade2phys/PREREGISTRATION.md` fait foi.
> Ancres sortantes au format §A50 — `` `fichier:NNN` « fragment » `` — et
> **contrôlées par `verifier_ancres.py`** avant commit, ce qui est neuf : c'est
> le premier document du programme dont les ancres passent une machine.
> **Ce document se commit SEUL** (règle §A52, deuxième application).
> **L'endossement est le commit de Romain ; aucun run avant.**

---

## §0. La question, et la décision qui en dépend

§A53 a mesuré le coût d'un bloc 3D avec **un jouet à cinq champs** —
(h, hu, hv, hw, s) : un système hyperbolique et **un seul** scalaire advecté.
Le vocabulaire réel de §A51 en porte davantage. La question est donc :

> **Que coûte un scalaire advecté de plus, en 3D ?**

**La décision nommée qui en dépend** (D17) : elle est **l'ordre des gestes**,
posé par Romain et gravé en §A54-5 — `PREREGISTRATION.md:8972-8973`
« Le moins cher qui peut échouer, avant la décision qui ne peut plus être
défaite ». Le cap de §A53 est un **plafond** ; deux réductions nommées
l'attendent. Celle-ci est bon marché et **neutre au schéma eau 3D** : ce qu'un
schéma tranche est le traitement de l'impulsion et de la pression ; l'advection
de `s`, `e_th`, `ρ_s` est la même sous tous. **Mesurer ici ne décide rien.**

**Ce que la mesure change, exactement** : sans elle, le cap n'est pas « 8 » mais
« **entre 5 et 8** » — la borne haute `7,5/5 = ×1,50` est connue, la borne basse
ne l'est pas. Un intervalle de 5 à 8 fenêtres est trop lâche pour arbitrer une
cadence. **La mesure ferme la fourchette ; elle ne la déplace pas.**

---

## §1. L'UNITÉ, FIXÉE AVANT — et une vérification de robustesse de §A53

**§A53 a chargé V4 comme 15 BLOCS.** C'est l'unité gravée du modèle de coût
(V4 = 11 slots dont 4 à c=8, soit 15 systèmes). Mais en 3D **la dégressivité
s'est effondrée** — `PREREGISTRATION.md:8042` « c_fin 3D = 7,5 éq-f32 ;
c_grossier = 7,0. » : les 11 slots portent tous à peu près le même vocabulaire.
La transposition fidèle est donc **11 slots à vocabulaire plein**, non 15 blocs
à cinq champs.

**Arithmétique de contrôle sur des nombres gravés (aucun run) :** au coût
plancher mesuré de 1,7138 ms, `11 × 1,7138 = 18,85 ms > 16,7`. **La branche R-3
de §A53 tient sous les DEUX décompositions** — elle ne dépend pas du choix
d'unité. C'est une vérification de robustesse, pas une correction : §A53 avait
choisi l'unité gravée, ce qui était le geste juste.

**Ce que le choix d'unité déplace, c'est le CAP, et il faut le dire net.** Le
« 8 » de §A53 était mesuré à `S=1` : un bloc y coïncide avec **une fenêtre 64³
portant un système de cinq champs**. Le cap de §A53 est donc **déjà en
fenêtres** — il n'y a pas de contradiction à lever, seulement une unité à
nommer. Ce document mesure ce que devient cette fenêtre quand elle porte le
vocabulaire réel.

---

## §2. LE VOCABULAIRE RÉEL, LU DANS §A51 — et ce que le jouet paie déjà

| ligne §A51 | champ | statut sous F | payé par le jouet de §A53 ? |
|---|---|---|---|
| 3 | `h` fraction d'eau | hyperbolique | **oui** |
| 4 | `hu, hv, hw` | hyperbolique | **oui** |
| 1b | `s` couche déposée | **advecté** | **oui** (l'unique scalaire) |
| 5 | `e_th` — `PREREGISTRATION.md:8037` « `e_th` **enthalpie**, diffus seul » | **advecté** | **NON** |
| 7 | `ρ_s` — `PREREGISTRATION.md:8038` « `ρ_s` fumée » | **advecté** | **NON** |
| 1a | `b0` occupation de solide | statique, LU par le stencil | **arithmétique oui, bande passante NON** |
| 2 | `id-matériau` | statique, LU | **NON** |
| 6, 8 | `e_ch`, `u_atm` | dérivés / forme close | sans objet |

**Deux écarts, de natures différentes, à ne pas confondre :**
- **+2 scalaires ADVECTÉS** (`e_th`, `ρ_s`) : pente limitée + upwind par face et
  par axe. Coût d'arithmétique **et** de bande passante.
- **+2 champs STATIQUES lus** (`b0`, `id-matériau`) : lus dans le halo, jamais
  avancés. Coût de **bande passante seule**.

**Précision sur `b0`, parce qu'elle est facile à rater.** Le kernel de §A53
matérialise la bathymétrie à zéro : il **paie l'arithmétique** des corrections
de pression (elle est comptée dans l'inventaire structurel) mais **ne lit aucun
tableau**. Le trafic mémoire de `b0` n'est donc **pas** dans l'ancre. C'est
exactement le genre de poste qui se fond dans un chiffre si on ne l'isole pas —
`PREREGISTRATION.md:8534-8535` « tout poste découvert à l'implémentation
s'isole et se déclare, jamais il ne se fond dans le chiffre principal ».

**Choix de largeur, déclaré d'avance et dans le sens sûr.** §A51 range `e_th` et
`ρ_s` en **f16** (0,5 éq-f32). Ils seront mesurés en **f32** : c'est un
**majorant de coût**, donc un **minorant du cap**. La réduction f16 est
**nommée, non mesurée** ; elle ne peut qu'agrandir le cap, jamais le réduire.

---

## §3. CE QU'ON MESURE, ET CE QU'ON NE TOUCHE PAS

Un **kernel 3D paramétré** en `N_SCALAIRES` (scalaires advectés) et
`N_STATIQUES` (champs lus, non avancés), dans un **fichier neuf**.

**Le kernel de §A53 n'est PAS modifié.** C'est le dénominateur de tout ce qui
suit ; son empreinte est déjà figée dans `tests/test_f3d_substrat.py`. La garde
qui interdit d'alléger le nouveau est celle de son aîné —
`src/f1_gpu/substrat_fusionne.py:8-9` « toute omission de calcul serait un
harnais complaisant, interdit ».

**Verrou d'identité, plus fort qu'une reproduction statistique** : le kernel
paramétré à `(N_SCALAIRES=1, N_STATIQUES=0)` doit produire, sur le même état,
une sortie **BIT POUR BIT IDENTIQUE** à `pas_f_fusionne_3d`. Si la
paramétrisation a changé quoi que ce soit au motif, ce test tombe. C'est la
condition d'entrée : **sans elle, aucun chronomètre n'est lancé.**

---

## §4. LES GRANDEURS

Configuration identique à M-3 de §A53 — 3 fenêtres de 64³, un système, chrono
B6 (≥ 300 frames, warmup exclu, médiane **et** p99), état jetable standard.

| # | `N_SCALAIRES` | `N_STATIQUES` | rôle |
|---|---|---|---|
| **M-c1** | 1 | 0 | **reproduction de §A53 M-3** (attendu 1,7138 ms/bloc) |
| **M-c2** | 2 | 0 | pente |
| **M-c3** | 3 | 0 | pente — et le compte d'advectés du vocabulaire réel |
| **M-c4** | 4 | 0 | pente, quatrième point |
| **M-c5** | 3 | 2 | **le vocabulaire réel de §A51** — la mesure de verdict |
| **M-c6** | 1 | 2 | isole le coût des statiques **seuls** |

**Grandeurs dérivées :**

- `δ_scalaire` = pente de l'ajustement affine sur M-c1…M-c4 ;
- `δ_statique` = `(M-c6 − M-c1) / 2` ;
- **`C` = M-c5** — coût d'une fenêtre 64³ à vocabulaire réel ;
- **`ρ_c` = C / M-c1** — le multiplicateur, à confronter à la borne haute
  **1,50** posée avant ;
- **le cap fermé** : `n(60 Hz) = (16,7 − 1,835) / C` et
  `n(30 Hz) = (33,333 − 1,835) / C`.

Le `1,835` est le non-F de la frame V4, **retenu constant**, avec la même
réserve que §A53 : les halos passent de ~2 % des cellules à ~20 %, donc le
non-F 3D est vraisemblablement supérieur, **donc le cap produit reste un
majorant**. Cette mesure **resserre** le plafond, elle ne le transforme pas en
plancher.

---

## §5. LES BRANCHES, ÉCRITES AVANT LA LECTURE

**Seuils immuables, posés maintenant :**

| seuil | valeur | provenance |
|---|---|---|
| borne haute de `ρ_c` | **1,50** | `7,5 / 5` — champs facturés comme des systèmes |
| `C` max pour 11 slots à **60 Hz** | **1,3514 ms** | `(16,7 − 1,835) / 11` |
| `C` max pour 11 slots à **30 Hz** | **2,8635 ms** | `(33,333 − 1,835) / 11` |

> **C-A — `C` ≤ 2,8635 ms : la porte 33,3 tient V4 (11 slots) à 30 Hz.**
> Le cap se lit en clair aux deux cadences. L'arbitrage de cadence devient
> chiffré des deux côtés, et il appartient à Romain.
>
> **C-B — `C` > 2,8635 ms : même 30 Hz ne tient pas V4.** L'arbitrage change de
> nature : ce n'est plus une cadence à choisir mais un **nombre de fenêtres** à
> réduire, ou un vocabulaire à retailler. Aucun enchaînement automatique.
>
> **C-C — `ρ_c` > 1,50 : la borne haute de §A54-5 est FAUSSE.** Ce serait un
> **résultat**, pas un bug : un champ advecté coûterait alors **plus** qu'un
> champ de système, ce qui contredit l'argument même qui fonde la borne (une
> advection n'a pas de solveur de Riemann). Le chiffre ne serait **pas consommé**
> avant que la cause soit nommée (registres, bande passante, occupancy).

**CERTITUDE PRÉ-RUN, à écrire pour qu'elle ne passe pas pour une découverte** :
`C ≥ M-c1 = 1,7138 > 1,3514`. **Aucune valeur de `δ` ne peut faire tenir V4 à
60 Hz** — c'est déjà acquis depuis §A53 et cette mesure ne le rouvrira pas.

**PRÉDICTION CONSIGNÉE AVANT LE CHIFFRE** (§A53 en a fait une règle) : pour
franchir 2,8635 ms il faudrait `δ_scalaire > 0,575 ms`, soit **un tiers du coût
d'un système hyperbolique entier pour un simple scalaire advecté**. La session
attend donc **C-A**, et le dit avant. Le risque de cette attente est nommé au
§6 : elle rend le résultat favorable *attendu*, donc elle appelle une garde
contre le résultat favorable **fabriqué**.

---

## §6. LES INDÉTERMINATIONS — vérifiées sur le cas FAVORABLE

Règle appliquée : `PREREGISTRATION.md:8536-8537` « critère d'indétermination se
vérifie sur le cas FAVORABLE autant que sur le cas défavorable, sinon il punit
la qualité qu'il prétend contrôler ». Ici **le cas favorable est l'attendu** —
c'est donc de ce côté que la garde doit mordre le plus fort.

**(I-c1) Reproduction.** `|M-c1 − 1,7138| / 1,7138 > 10 %` ⇒ **INDÉTERMINÉ**.
Le harnais n'est pas celui de §A53.
*Sur le cas favorable* : un `δ` petit ne peut pas éteindre I-c1, qui ne porte
que sur le point `k=1`. ✔

**(I-c2) Équivalence de motif, à CHAQUE `k`.** Le kernel paramétré doit
reproduire un jetable paramétré à `rtol=1e-4, atol=1e-5`, sur l'**état sévère**
(vitesses d'ordre 1, cellules sèches, gradients raides) — l'état jetable
ordinaire n'allume pas le transport tangentiel, leçon de §A53. Échec ⇒ aucun
chronomètre.
*Sur le cas favorable* : un kernel rapide **parce qu'il omet** échoue ici. C'est
exactement l'effet voulu. ✔

**(I-c3) L'ÉLIMINATION PAR LE COMPILATEUR — la garde principale de ce
protocole.** Si `δ_scalaire < 2 % de M-c1`, ⇒ **INDÉTERMINÉ**.
Motif : un scalaire advecté calculé mais jamais consommé est **supprimé par
nvcc**, et le chronomètre affiche alors `δ ≈ 0` — un résultat magnifiquement
favorable, et faux. **Un `δ` nul et une élimination sont indistinguables au
chronomètre.** Lever l'indétermination exige de regarder ailleurs : registres
par thread à chaque `k` (ils doivent croître) et, si besoin, le PTX.
*Ce critère ne se déclenche QUE du côté favorable* — c'est sa raison d'être, et
le symétrique exact de la garde anti-mort de §A52. ✔

**(I-c4) Affinité en `k`.** Si le résidu maximal de l'ajustement affine sur
M-c1…M-c4 dépasse **5 %** du coût mesuré ⇒ la pente n'est pas reportée et
**seuls les points mesurés le sont**. Pas d'extrapolation sur un modèle qui ne
tient pas.
*Sur le cas favorable* : une non-linéarité **décroissante** (le 4ᵉ scalaire
moins cher que le 2ᵉ) déclenche I-c4 tout autant — sans quoi on encaisserait
une aubaine en la nommant « pente ». ✔

**(I-c5) Cohérence croisée.** `|M-c5 − (M-c3 + 2·δ_statique)| > 5 %` ⇒ le
découpage advecté/statique ne décrit pas ce qui est mesuré ; `C` est reporté
tel quel et la **décomposition** est déclarée non établie. C'est un test que le
protocole peut **échouer** : deux chemins indépendants doivent donner le même
nombre.

**(I-c6) Débordement de registres.** Si `local_size_bytes > 0` à un `k`
quelconque, le fait est reporté **en évidence** et le résultat prononcé **avec
réserve d'implémentation nommée** : un spill fait sauter `δ` pour une raison
d'implémentation, pas de structure. Ce n'est pas une indétermination — un coût
réel reste un coût réel — mais il ne s'attribue pas au vocabulaire sans le dire.

---

## §7. GARDES ANTI-FABRICATION

1. **Aucun seuil ne bouge** : 16,7 ms, `rtol/atol` de M-a′, ≥ 300 frames, tous
   importés.
2. **Le kernel 3D de §A53 reste INTOUCHÉ** (`git diff` vide, empreinte figée) —
   il est le dénominateur de `ρ_c`.
3. **Identité bit-pour-bit à `k=1`** (§3) : condition d'entrée du chronomètre.
4. **Les scalaires ajoutés doivent être RÉELLEMENT advectés** : le jetable de
   référence calcule leurs valeurs, et l'équivalence les compare champ par
   champ. Un scalaire écrit mais non transporté échoue I-c2 ; un scalaire ni
   écrit ni transporté échoue I-c3.
5. **Inventaire structurel** du motif à chaque `k` — les comptes d'opérations
   doivent croître comme prévu. C'est le verrou qui a attrapé, en §A53, la
   suppression d'un terme payé en arithmétique et nul en valeur.
6. **Pas d'`assert` nu**, fail-loud partout, `else` de combinateur = sortie
   INDÉTERMINÉE.
7. **L'artefact est un fichier** versionné dans `claude/lectures/` ; une sortie
   de console n'est pas un artefact. L'entrée de journal ne citera que des
   valeurs relues dans le JSON.
8. **Les ancres de ce document ont été passées à `verifier_ancres.py`** avant
   commit — et le seront à nouveau avant l'entrée de journal.

---

## §8. CE QUE CETTE MESURE NE PRONONCERA PAS

- **Rien sur le schéma eau 3D.** Aucune pression matérialisée, aucun solve de
  Poisson, aucun tampon de flux aux faces. C'est ce qui rend la mesure possible
  sans trancher : seule l'advection passive est en jeu.
- **Rien sur le non-F 3D** — l'autre réduction nommée, qui demande du build et
  qui ne peut que réduire encore le cap.
- **Rien sur la cadence.** Elle appartient à Romain, et C-A comme C-B ne font
  que lui donner les deux nombres.
- **Rien sur f16.** Mesuré en f32 = majorant ; la réduction est nommée, non
  mesurée.
- **Aucun verdict de gate**, aucune réouverture de §A53.

---

## §9. CE QUI SERA DÛ APRÈS

- Le **non-F en 3D** (halos, remontée, prédiction), dernière réduction nommée
  du cap.
- La **réduction f16** sur `e_th` et `ρ_s`, si elle vaut son geste.
- Une ligne au **registre de supersessions** si cette mesure supersède un
  chiffre de §A53 — la discipline de §A54-6, qui vaut pour la session comme
  pour quiconque.

---

*Document rédigé par la session Claude, AVANT tout run et AVANT toute ligne de
kernel paramétré. L'endossement est le commit de Romain.*
