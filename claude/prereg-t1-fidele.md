# Pré-enregistrement — TRANCHE-1 de M-b : le F fidèle à l'échelle V4, et le gate T1

**Pré-enregistrement. POINT D'ARRÊT ICI, avant le build.**
Source : pocCascade2phys PREREGISTRATION.md §A28 (commit 9edaa47) +
SPEC-FOVEA-Z §2-rev2. Propriété gravée : la pyramide est un MIPMAP (fin =
résolution de la physique, décimation vers l'extérieur) ⇒ M = 1, K ≤ ~4,2×
temps réel. Achat de tranche-1 autorisé (~2,1–3,4 séances) ; ce document
gèle les choix AVANT la première ligne, comme le kernel L3 le fut, parce que
c'est le morceau qui glisse le plus et qu'il n'a pas de cap.

> **Ce que ce document gèle** : les choix d'implémentation du F fidèle
> (nommés, pour qu'un chrono décevant se lise « ce design-là coûte tant »),
> la structure de l'invariant liant (même code des deux tranches, par
> construction), la vérification d'instrument du régime (PASS dû avant
> lecture), et **la bande de T1 pré-enregistrée pour CETTE config** — plus
> jamais recomposée après coup (§A25).

---

## 1. Le F fidèle — choix d'implémentation, gelés

`run_episode` (rederive, CPU f64, intouché) : pulse → `simulate_wetdry_o2`
(wetdry O2 bien équilibré Audusse, MUSCL surface-gradient, HLL dry-aware,
murs réfléchissants) sur `b_eff = b0 + s` → intégration Exner. Le portage GPU
à l'échelle V4, un pas par frame, dt = temps de frame, M = 1.

**Fait technique qui décide l'architecture** : le kernel figé
(`substrat_fusionne`) a la signature `(q_in, q_base, q_out, dt, w_base, n,
total)` — **aucune bathymétrie**. Ses corrections hydrostatiques sont
« calculées, nulles à bathymétrie plate » : le chemin b-aware n'existe pas.
Le F fidèle est donc un **kernel NEUF**, pas une réutilisation du figé.

Choix gelés (un chiffre décevant lira « ce design coûte tant », jamais « il
faut mieux l'implémenter ») :

1. **Kernel b-aware NEUF, kernels figés INTOUCHÉS.** Le F fidèle reprend du
   préfixe figé les fonctions device **b-agnostiques** réutilisables (minmod,
   désingularisation, HLL sur états donnés) — comme L3 a repris le préfixe —
   et ajoute la reconstruction bien équilibrée à b réel (η = h + b, z* =
   max(z_L, z_R), h* = max(0, η − z*), pression (g/2)(h² − h*²)). Le figé
   n'est pas touché ; c'est un `_SOURCE_FIDELE` distinct.
2. **Sémantique des champs, différente du jetable, nommée.** Le jetable
   transporte 4 champs (h, hu, hv, s) à bathymétrie plate. Le fidèle évolue
   **(h, hu, hv)** par le wetdry O2 sur `b_eff = b0 + s`, et **s** est mis à
   jour par un kernel Exner SÉPARÉ (dépôt/érosion sur cellules mouillées),
   PAS advecté. C'est la physique de `run_episode`, pas le motif de coût du
   jetable — donc l'ancre 12,655 n'est qu'un PROXY, pas la mesure (voir la
   bande §5).
3. **dt = temps de frame, un pas/frame, M = 1.** Sous la propriété mipmap
   gravée, dt_frame siège 4–12× sous la CFL fine : sous-CFL, pas de
   sous-cyclage. Le kernel ne consomme PAS la CFL pour sous-pas — il la
   CALCULE (pour la vérification d'instrument §4), il ne l'itère pas.
4. **Exner GPU NEUF.** `_exner_step` (Euler explicite masqué wet, ds/dt =
   1[wet](k_d·h·1[θ<θc](1−θ/θc) − k_e·s·1[θ>θc](θ/θc−1)), s ≥ 0) porté en un
   kernel pointwise, cadencé UNE FOIS PAR FRAME (pire cas pour T1 ; le
   save_every de `run_episode` est une décimation de snapshots, non un
   allègement de coût de frame).
5. **Remontée L3 + transferts RÉUTILISÉS tels quels** : kernel L3 figé, L1
   k=4 étalée, EPS_PRODUCTION 1e-2, `TransfertComptable`. Rien de neuf.
6. **Bords en tranche-1 : réfléchissants par fenêtre** (le padding du figé),
   comme le jetable. Le mur intérieur (halos parent) est de tranche-2 (§6).
7. **Confinement du moteur sans cap.** Le F fidèle est écrit comme le
   **portage d'une physique GRAVÉE** (`run_episode`), pas comme un moteur à
   optimiser : (i) ces choix sont gelés ici ; (ii) le critère de fidélité
   n'est PAS de tranche-1 — tranche-1 ne mesure que le TEMPS, la fidélité est
   le verdict de tranche-2 contre le rederive ; (iii) le docstring dira qu'un
   chrono hors bande est un résultat sur CE portage, pas un appel à mieux
   porter.

---

## 2. L'invariant liant — même code des deux tranches, par CONSTRUCTION

Le F de tranche-1 (échelle V4) et de tranche-2 (64², contre le rederive)
doivent être **le même code**, structurellement, pas surveillé par un diff.

**Construction** : une SEULE fonction `pas_f_fidele(etat, b_eff, cp, bords,
...)`, où l'échelle est un **paramètre de DONNÉES** (les shapes des tableaux),
jamais une branche de code. Tranche-1 l'appelle sur des fenêtres à l'échelle
V4 ; tranche-2 sur des fenêtres 64². Il n'y a pas deux fonctions à garder
synchrones — il y en a **une**, importée par les deux drivers.

- Le test de l'invariant n'est PAS un diff : c'est une **identité d'objet** —
  `run_f1_mb_t1` et `run_f1_mb_t2` importent le MÊME symbole
  `pas_f_fidele`, et un test l'assert (`t1.pas_f_fidele is t2.pas_f_fidele`).
  Une divergence exigerait de forker la fonction, ce que l'identité interdit.
- **La seule couture admise est le BORD** (§6) : le remplissage de halo est
  un paramètre `bords` (réfléchissant en t1, halo-parent en t2). L'opérateur
  INTÉRIEUR — stencil, reconstruction, flux, Exner — est le même objet. Le
  bord est de la donnée injectée, pas du code dupliqué.

---

## 3. dt et échelle — la propriété mipmap, reconduite

Le dt de frame est sous-CFL au niveau fin PARCE QUE le fin est à la
résolution de la physique (mipmap gravé §A28). Le kernel calcule la CFL par
frame (réduction on-device de s2, réutilisée) et la reporte en marge — non
pour sous-cycler, mais pour la vérification d'instrument (§4) et pour PROUVER
que M = 1 tient sur l'état synthétisé. Si la marge CFL tombe sous 1 (état trop
violent), la lecture est REFUSÉE : ce serait un état hors régime mipmap.

---

## 4. Caveat de régime — vérification d'instrument, PASS dû AVANT lecture

Un domaine tout sec, ou tout mouillé uniforme, donnerait un chrono sans
valeur (le wetdry ne fait rien, la CFL n'est pas sollicitée). L'état
synthétisé à l'échelle V4 doit **exercer le régime**. Vérification
d'instrument, câblage habituel « pas de lecture sans son PASS » (B9) :

`verifier_regime` PASS ssi, sur l'état synthétisé :
- **fronts wet/dry présents** : fraction mouillée dans une bande gravée
  (ni ~0, ni ~1 ; p. ex. [0,2 ; 0,8]) ET au moins un front (gradient de h à
  travers une frontière wet/dry) au-dessus d'un seuil ;
- **CFL sollicitée à 4–12× de marge** : `dt_CFL(fin) / dt_frame ∈ [4, 12]` —
  la borne basse garantit que la CFL n'est pas triviale (état trop lent), la
  haute que M = 1 tient (pas de sous-cyclage). Hors de [4, 12], AUTRE
  d'instrument, on remonte.

`lecture_t1` REFUSE fail-loud de produire un verdict sans ce PASS enregistré.
Un chrono sur un domaine sec serait un chiffre dont on ne saurait pas s'il
mesure le F fidèle ou le vide.

---

## 5. T1 — pré-enregistrement, bande GELÉE pour cette config

**T1 est VERDICTAL, sur la MÉDIANE de la frame complète, seuil 16,7 ms.**
p99 reportée EN ÉVIDENCE (le stutter, suspendu au régime §A25). Config exacte :
V4 emboîtée n_fov=512, 15 blocs, EPS_PRODUCTION 1e-2, L1 k=4, un pas/frame,
bords réfléchissants, Exner par frame.

### La bande, pré-enregistrée MAINTENANT (§A25 : plus jamais après coup)

Composée depuis les postes, chacun avec sa borne nommée AVANT tout run. Je ne
la resserre pas : le F fidèle b-aware + Exner n'a PAS d'ancre mesurée, donc
l'incertitude est réelle et la bande est large — c'est honnête, pas faible.

| Poste | Borne basse | Borne haute | Origine |
|---|---|---|---|
| F fidèle (wetdry O2 b-aware, 3 champs) | 10,8 | 15,2 | ancre jetable 12,655 × [0,85 ; 1,2] (3 champs vs 4, + reconstruction b-aware) |
| Exner (pointwise wet, par frame) | 1,5 | 3,0 | ~1/6–1/4 d'un F, 1 passe sans stencil |
| Remontée L3 amortie (k=4) | 1,28 | 1,28 | borne L3, fixe |
| Transferts (1e-2) | 0,11 | 0,11 | mesuré, fixe |
| Dérive machine | −0,18 | +0,18 | constante gravée |
| **BANDE T1 (bords réfléchissants)** | **13,5** | **19,8** | |

**La bande STRADDLE le seuil de mort 16,7** (centre ~16,4). C'est exactement
pourquoi T1 est verdictal et vaut d'être mesuré : sur le papier, la survie de
V4-b est sur le fil, et seule la mesure tranche.

- **Médiane hors [13,5 ; 19,8]** ⇒ **AUTRE** : le modèle de coût du F fidèle
  est faux, à remonter (distinct du verdict de mort).
- **Médiane dans la bande ET ≤ 16,7** ⇒ V4-b vit T1 ; **> 16,7** ⇒ **MORT-b
  par T1**, le repli 33.3 devient la décision.

### Réserve de halo, explicite (en attente de tranche-2)

La bande ci-dessus est à **bords réfléchissants** — elle SOUS-COMPTE le F
fidèle du coût des halos intérieurs que tranche-2 ajoutera : **+[0,2 ; 0,5]
ms** (GPU-side, sans transfert, §6). **Conséquence gravée pour le verdict** :
si la médiane T1 dépasse **16,2**, la réserve de halo peut à elle seule la
porter au-dessus de 16,7 en tranche-2. Le verdict T1 rapporte donc les DEUX :
la médiane réfléchissante, et la médiane + réserve de halo. Un « vit » à
16,3 est un « vit sous réserve halo ».

---

## 6. Ce que le traitement des bords en tranche-1 impose à tranche-2

Tranche-1 pade en **réfléchissant par fenêtre** — le plus bas coût, celui de
l'ancre, correct pour un chrono. Ce que cela impose à tranche-2 (le mur
intérieur, que je ne tranche PAS ici) :

1. **Un delta de coût, borné** : les halos parent que tranche-2 ajoutera
   coûtent +[0,2 ; 0,5] ms (GPU-side, sans H2D/D2H, la prédiction prolongée de
   2 cellules). La réserve du §5 le porte déjà.
2. **Une couture, pas une réécriture** : le bord est un paramètre `bords`
   (§2) — tranche-2 branche le remplissage halo-parent SANS toucher
   l'opérateur intérieur. L'invariant liant tient.
3. **Une INTERDICTION de réutiliser le bord de t1 pour la fidélité** : le
   padding réfléchissant est FAUX physiquement (le bord de fenêtre n'est pas
   un mur — c'est une continuation vers le grossier). Tranche-2 NE PEUT PAS
   réutiliser le bord de tranche-1 pour son Δχ contre le rederive ; elle DOIT
   le remplacer par le halo-parent. Tranche-1 fige donc le point d'injection
   du bord, pas son contenu.

Autrement dit : tranche-1 me lie à rendre le bord **pluggable** dès la
première ligne, faute de quoi tranche-2 devrait forker le kernel et
l'invariant liant tomberait.

---

## 7. Chiffrage du build — et où il peut dépasser 3,4 séances

| Composant | LOC | Séances |
|---|---|---|
| C1. Kernel F fidèle b-aware (reconstruction Audusse, bords pluggables) | ~450–650 | **1,0 – 1,8** |
| C2. Exner GPU (kernel + cadence par frame) | ~250–350 | 0,4 – 0,6 |
| C3. Bathymétrie réelle à l'échelle V4 (b0 → monde V4) + b_eff = b0+s | ~150–250 | 0,3 – 0,4 |
| C4. État synthétisé + vérification d'instrument (régime, CFL, fronts) | ~250–350 | 0,4 – 0,6 |
| C5. Driver T1 + chrono B6 + bande pré-enregistrée + p99 + verrous | ~350–500 | 0,5 – 0,7 |
| **TOTAL** | **~1450–2100** | **2,6 – 4,1** |

**Je dépasse potentiellement 3,4 séances** — la fourchette haute (4,1) est
au-dessus. Le poste qui la fait basculer est **C1, le kernel b-aware** : c'est
le moteur sans cap que j'ai nommé, et sa reconstruction bien équilibrée à b
réel (jamais exercée sur GPU — le figé est plat) est le débogage de fidélité
qui glisse. Je le remonte comme demandé.

**Recommandation, pour contenir le dépassement** : un **point de contrôle
après C1 seul** — le kernel b-aware compilé et vérifié bien équilibré (C-
property : au repos sur terrain réel, vitesse parasite ≈ 0, comme le valide
le CPU). Si C1 tient dans 1,0–1,3, le reste (C2–C5, patrons éprouvés) rentre
sous 3,4. Si C1 glisse au-delà de 1,8, on le sait AVANT d'avoir construit le
driver, et on décide de continuer ou d'arrêter — la règle 3, appliquée à
l'intérieur de la tranche.

---

## 8. Ce que ce pré-enregistrement ne couvre pas

- **Le mur intérieur** (halos parent) : tranche-2. Tranche-1 le rend
  pluggable et lui réserve son delta de coût (§6), rien de plus.
- **MORT-b** (fidélité live↔rederive) : tranche-2. Tranche-1 ne mesure que
  le TEMPS ; aucun Δχ, aucun rederive appelé.
- **Le critère de fidélité** du F fidèle : gelé en tranche-2, pas ici — c'est
  ce qui empêche le moteur sans cap de dériver en tranche-1.
- **Toute mesure** : ce document est un pré-enregistrement. Aucun run, aucune
  ligne de code écrite.

---

## 9. Portée

Kernels FIGÉS intouchés (`_SOURCE` = e18015f5…f30b4 / `_SOURCE_L3` =
9533a130…781a — le F fidèle est un `_SOURCE_FIDELE` NEUF) ; k reste 4 ;
`run_f1_attribution_b.py` gaté ; pas de miroir CPU ; le rederive CPU f64 reste
intouché (non appelé en tranche-1).

**POINT D'ARRÊT.** Pré-enregistrement de T1 remonté, avant le build. Aucune
ligne de code écrite, aucun run. Le build de tranche-1 attend l'endossement
de ces choix et de la bande, et le chiffrage C1 (dépassement possible de 3,4)
attend l'arbitrage.
