# Pré-enregistrement — TRANCHE-2 de M-b : la fidélité (gate iii)

**Pré-enregistrement. POINT D'ARRÊT ICI, avant le build. Aucun run.**
Source : pocCascade2phys PREREGISTRATION.md §A30 (commit 375f7e3). T1 est
prononcé SANS MORT (F_fidèle/F_proxy = 0,924 — l'ancre était honnête). Achat
de tranche-2 (~1,5–2,3 séances). Un choix — la réconciliation d'échelle 64² —
mérite endossement AVANT le build ; ce document le pose.

> **Ce que ce document gèle** : la lecture MORT-b (per-seed, pin sévère
> 0,0733), la structure de comparaison live↔rederive, le bord halo-parent
> rafraîchi entre étages RK2, l'invariant liant par identité — ET remonte
> pour endossement la RÉCONCILIATION D'ÉCHELLE, qui n'est pas triviale : au
> 64² du rederive, la pyramide de V4 dégénère (§A26), et le halo-parent
> qu'exige §A30 implique une fovéation, pas un domaine plein.

---

## 1. Ce que tranche-2 mesure

| | |
|---|---|
| **MORT-b** | max de série Δχ **live↔rederive** aux émissions **> 0,0733** (pin sévère IC entier, 7,33 %), **PAR-SEED** |
| **Conséquence** | MORT-b ⇒ Option A morte ⇒ repli Option B = décision neuve, coût de frame NON mesuré ici |
| **LIVE** | le pipeline V4 (fidèle F C1 + Exner C2 + L3 + L1 k=4, EPS 1e-2), à l'échelle 64², bord HALO-PARENT |
| **REDERIVE** | `run_history` 64² CPU f64, **INTOUCHÉ** (§A27) — la vérité-sol |
| **OBSERVABLE** | Δχ readout (`albedo` + `delta_chi["max_carrier"]`), primitives existantes |

T1 était le TEMPS ; T2 est la FIDÉLITÉ. Le rederive, absent de T1, est ici le
juge. La comparaison porte sur ce que le CPU SAIT (sa connaissance reconstruite
via la remontée L3 seuillée à EPS 1e-2 — l'option 1 de la sonde v2) contre la
vérité f64 pleine du rederive.

---

## 2. La cellule §A15, reconduite

Gravée, reconduite sans changement :
- **3 seeds** {101, 102, 103} ;
- **Δt = 4**, **6 émissions** (checkpoints de comparaison le long de
  l'historique d'épisodes) ;
- **commits fenêtrés** : émission = `SummaryQT` (topologie u8 + moyennes f64)
  + fenêtre (niveau, rect aligné-dyadique), via `src/f1_gpu/ledger.py`
  INTOUCHÉ ;
- **k_fen aire-proportionnel** : le budget de floats d'une fenêtre est
  proportionnel à son aire ;
- **cap 10 %** : borne anti-trivialité (le commit ne peut pas porter plus de
  10 % de la pleine résolution).

Le ledger et `run_history` sont RÉUTILISÉS tels quels — ce ne sont pas des
objets neufs de tranche-2.

---

## 3. LA RÉCONCILIATION D'ÉCHELLE — le choix à endosser

C'est le point non trivial, et je le remonte plutôt que de le préempter.

**Le problème (rappel §A26).** Le rederive vit à 64². La pyramide de V4
(`n0 = 256`, 10 niveaux, `cote_monde(0) = 256²`) est plus grande que le 64² à
son niveau LE PLUS GROSSIER : le substrat ne remplit AUCUN niveau. À 64² la
fovéation dégénère.

**Ce que §A30 impose, et ce qu'il révèle.** Le bord DOIT être halo-parent, pas
réfléchissant (« physiquement faux pour la fidélité »). Or un halo-parent
n'existe QUE si la fovéa est une SOUS-fenêtre d'un monde plus grand — un bord
INTÉRIEUR. Donc, à 64², la structure vivante n'est PAS un domaine 64² plein
(qui aurait des murs réfléchissants PHYSIQUES au vrai bord) : c'est une
**fovéation à 2 niveaux sur le 64²** —
- **niveau grossier** : le 64² entier à résolution décimée, murs
  réfléchissants au VRAI bord (physique, correct, = le rederive) ;
- **niveau fin** : une fovéa (sous-fenêtre) à pleine résolution, bord
  INTÉRIEUR rempli par **halo-parent** (continuation depuis le grossier
  upsamplé).

C'est cette structure que je pré-enregistre, et voici pourquoi elle est
cohérente : le halo-parent de la fovéa est correct (le bord fin n'est pas un
mur), ET le mur réfléchissant du grossier est correct (c'est le vrai bord du
domaine 64², celui du rederive). Le mipmap gravé (§A28, fin = physique,
décimation vers l'extérieur) tient : le fin est à la résolution du substrat
(dx=1), le grossier décime.

**Ce qui reste à endosser** (les choix, nommés) :
1. **La structure 2-niveaux** ci-dessus — est-ce la bonne, ou faut-il plus de
   niveaux ? Je retiens 2 (fin + grossier) : c'est le minimum qui exerce le
   halo-parent tout en gardant un vrai mur au bord réel.
2. **La position et la taille de la fovéa** sur le 64² — quelle fraction, et
   suit-elle le gaze (comme E4c) ou est-elle fixe ? Je propose : fovéa mobile
   E4c (cohérent avec V4), taille telle que le grossier reste non trivial.
3. **La résolution du grossier** (facteur de décimation fin→grossier) — 2×
   (un seul niveau d'écart), cohérent avec le mipmap.

Si l'un de ces trois n'est pas ce que Romain veut, la lecture change ; d'où le
POINT D'ARRÊT ici.

---

## 4. La comparaison live↔rederive

À chaque émission (6 par seed, Δt=4 épisodes) :
- le **live** évolue le substrat sédiment par épisodes sur GPU (fidèle F +
  Exner, cadence save_every), REMONTE ses détails via L3 seuillé EPS 1e-2, et
  le CPU reconstruit sa connaissance (la `reference`/grand-livre, option 1 —
  jamais une copie qui puisse diverger, cf. sonde v2) ;
- le **rederive** évolue le même historique sur CPU f64 (`run_history`,
  intouché) ;
- **Δχ** = `delta_chi(albedo(connaissance_cpu), albedo(rederive))["max_carrier"]`
  au point d'opération S_HALF_OP, à l'échelle de lecture DÉCIMÉE (comme la
  sonde), espace instrument.
- **MORT-b PAR-SEED** : `max_émissions Δχ > 0,0733` ⇒ ce seed tue Option A.

Le grand-livre ≡ `reference` device est l'identité vérifiée par la sonde v2 ;
tranche-2 la RECONDUIT (verrou), avec la physique fidèle au lieu du jetable.

---

## 5. Le bord : halo-parent, rafraîchi ENTRE LES ÉTAGES RK2

Ma propre trouvaille de C1, load-bearing ici :
- **Halo-parent** : la fovéa lit son halo ±2 depuis le grossier upsamplé
  (mode 1 du kernel C1, déjà présent, prouvé bit-exact sur un étage).
- **RAFRAÎCHI ENTRE LES DEUX ÉTAGES RK2** : le wrapper `pas_f_fidele` fait
  deux étages ; en mode halo le halo passé se PÉRIME au second étage (il
  décrit l'état d'entrée, pas l'intermédiaire). Le **0,186** de C1 en était
  la signature exacte. Tranche-2 DOIT donc reconstruire le halo-parent APRÈS
  l'étage 1, depuis l'état intermédiaire du grossier, avant l'étage 2. Sans
  ce rafraîchissement, la fidélité mesurée serait un artefact d'étage.

Choix d'implémentation nommé : le rafraîchissement se fait au niveau du
DRIVER (t2), pas en modifiant `pas_f_fidele` — l'invariant liant l'interdit
(§6). Le driver appellera les DEUX étages séparément avec un halo reconstruit
entre eux, OU un wrapper t2 qui orchestre les deux étages + le refresh SANS
toucher l'opérateur intérieur. À trancher au build (mineur), pas un blocage.

---

## 6. L'invariant liant : identité, pas diff

`t1.pas_f_fidele is t2.pas_f_fidele` — le F fidèle est LE MÊME OBJET. Un test
l'assert par identité. Tranche-2 n'introduit AUCUNE variante du F ; elle
change le BORD (mode halo au lieu de réfléchissant) et le RAFRAÎCHISSEMENT
(donnée + orchestration), jamais l'opérateur intérieur. La couture est le
`mode` et le halo, comme C1 l'a gravé.

Cadence Exner : toujours lue de `save_every` (jamais choisie), comme C2.

---

## 7. Chiffrage du build

| Composant | LOC | Séances |
|---|---|---|
| T1. Structure 64² 2-niveaux (grossier + fovéa, halo-parent) | ~250–400 | 0,4 – 0,6 |
| T2. Refresh halo entre étages RK2 (orchestration driver) | ~120–200 | 0,2 – 0,3 |
| T3. Évolution live par épisodes + connaissance CPU (option 1) | ~250–350 | 0,4 – 0,6 |
| T4. Comparaison Δχ aux émissions + MORT-b par-seed + verrous | ~250–350 | 0,4 – 0,6 |
| T5. Driver + lecture (3 seeds, Δt=4, 6 émissions) + tests | ~250–350 | 0,3 – 0,5 |
| **TOTAL** | **~1120–1650** | **1,7 – 2,6** |

Bornes cohérentes avec l'estimation §A28 (~1,5–2,3). La fourchette haute
(2,6) dépasse un peu ; le poste qui peut glisser est **T2/le refresh halo** —
c'est un objet neuf (le halo-parent inter-étage n'a jamais tourné), et c'est
là qu'un débogage de fidélité peut coûter. Si T2 glisse au-delà de 0,3, POINT
D'ARRÊT et remontée avant le driver.

---

## 8. Ce que ce pré-enregistrement ne couvre pas

- Le **coût de frame du repli Option B** (si MORT-b) : décision neuve, non
  mesurée ici (§A30).
- La **cellule/le ledger** eux-mêmes : reconduits, intouchés.
- Les **quatre réserves gravées de T1** (frame composée, per-champ ~2,5×) :
  contexte, pas objet de tranche-2.
- Toute mesure : aucun run avant endossement.

---

## 9. Portée et interdits

Kernels FIGÉS et C1–C5 intouchés (quatre empreintes : `_SOURCE` e18015f5,
`_SOURCE_L3` 9533a130, `_SOURCE_FIDELE` 6dd207ca, `_SOURCE_EXNER` 3533fd0b) ;
k reste 4 ; `run_f1_attribution_b.py` gaté ; pas de miroir CPU ; le rederive
CPU f64 (`run_history`) reste INTOUCHÉ.

**POINT D'ARRÊT.** Pré-enregistrement remonté, avant le build. Le build de
tranche-2 attend l'endossement de la réconciliation d'échelle (§3, les trois
choix nommés) et du reste. Aucun run.
