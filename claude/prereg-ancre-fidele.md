# Pré-enregistrement paper-grade — réserve sur l'ancre du F fidèle

**Paper-grade. Aucun run, aucune ligne de code M-b.**
Source : pocCascade2phys PREREGISTRATION.md §A25 (commit 179ba1d). Les deux
points de glissement nommés au chiffrage de M-b (§A24) sont hissés en
**réserve sur l'ancre** et pré-enregistrés ici, avant tout achat : le
mapping épisode↔frame, et les halos de fenêtre.

> **Résultat en une phrase, et il contredit l'attente que je porte :** dérivé
> du CFL réel, le rapport pas/frame en temps réel est **< 1** — l'ancre
> 12,655 ne se multiplie PAS en temps réel, elle est déjà mesurée à ~2–4×
> le temps réel. T1-médiane n'est donc pas compromis sur le papier ; il
> devient tendu au point de fonctionnement de M-a-quater, et il ne casse que
> si le jeu tourne en accéléré. Je ne l'habille pas dans l'autre sens pour
> coller à l'intuition « moteur lourd ».

---

## 1. Terme A — le mapping épisode↔frame, dérivé du CFL réel

### Le pas de temps réel du substrat

`_cfl_dt` (solver_wetdry.py) : `dt = 0,4 · min(dx,dy) / max(|u|+c)`, avec
`c = sqrt(g·h)`. Constantes réelles du substrat : `g = 9,81`, `dx = dy = 1`,
`h_p = 0,6·relief = 0,3626` (pulse), grille 64².

| Régime dans l'épisode | smax | dt_CFL |
|---|---|---|
| Pic du pulse (u ≈ 0) | c = 1,886 | **0,212 s/pas** |
| Effondrement (|u| ≈ c) | 2c | 0,106 s/pas |
| Front type Ritter (|u| ≈ 2c) | 3c | 0,071 s/pas |
| Moyenne plein épisode (t_end=130, 1160–2664 pas *documentés*) | — | 0,049 – 0,112 s/pas |

La phase active (les N_settle = 600 premiers pas, ceux de l'épisode) siège
du côté **court** de la fourchette — vitesses hautes, dt petit. Retenu :
**dt_CFL ≈ 0,05–0,07 s/pas** en phase active (borné en bas par le front
Ritter 0,071 et en haut par la moyenne 0,05, cohérent avec les comptes
documentés). Je n'invente pas : ces bornes viennent du CFL et du compte de
pas déjà gravé dans le docstring de `sediment.py`.

### Combien de pas une frame doit-elle avancer, en temps réel ?

Un pas avance `dt_CFL` de temps SIMULÉ. Une frame 60 fps dure 16,667 ms =
**0,01667 s** de temps mur. « Temps réel » = 1 s simulée par 1 s mur. Donc
la frame doit avancer 0,01667 s de temps simulé :

```
pas / frame (temps réel) = 0,01667 / dt_CFL = 0,01667 / (0,05 … 0,07)
                         = 0,24 … 0,33   (phase active)
                         = 0,08          (au pic, dt=0,212)
```

**Le rapport est < 1.** En temps réel, une frame n'avance MÊME PAS un pas de
F complet — le solveur sous-pase (un pas tous les ~3 à 4 frames). Dérivé du
CFL, pas d'une convention.

### Ce que cela dit de l'ancre 12,655

L'ancre a été mesurée à **1 pas de F par frame** (M-a-quater : « 1 cellule
fine/frame », un étage RK2 par frame). Un pas avançant 0,05–0,07 s de simulé
en 0,01667 s de mur, **1 pas/frame tourne déjà à**

```
facteur temps réel = dt_CFL / frame = (0,05 … 0,07) / 0,01667 ≈ 1,8 … 4,2×
```

**l'ancre 12,655 est donc mesurée à ~2–4× le temps réel, pas au temps réel.**
En clair, avec le chiffre demandé :
- **En temps réel, l'ancre ne se multiplie pas — elle se DIVISE** : la frame
  fait 0,24–0,33 pas, soit un coût de F amorti de ~3–4 ms/frame si l'on
  pouvait fractionner (on ne peut pas — voir burstiness ci-dessous).
- **L'ancre se multiplie SEULEMENT au-delà de ~1,8–4,2× le temps réel.** Le
  facteur de multiplication est linéaire : à K× le temps réel,
  `pas/frame = K · 0,01667 / dt_CFL = K / (1,8 … 4,2)`, et F par frame
  = `pas/frame · 12,655` dès que ce rapport ≥ 1.

### La conséquence qui n'est PAS bénigne : la burstiness

Le rapport < 1 en temps réel ne rend pas F gratuit — il le rend **bursty**.
On ne peut pas faire 0,3 pas de F : on en fait 1 toutes les ~3–4 frames. Les
frames-pas coûtent l'ancre pleine (~12,7 ms + le reste) ; les frames-sans-pas
coûtent ~le rendu. **La médiane devient une frame SANS pas (donc bon marché),
et tout le coût migre vers la p99.** Or L1 étalée étale la REMONTÉE, pas le pas
de F : c'est une **nouvelle source de stutter que L1 ne couvre pas**. §A25 a
gardé la question du stutter nommée à part — c'est exactement ici qu'elle mord.

---

## 2. Terme B — les halos de fenêtre

Le jetable pade en **réfléchissant par fenêtre** (docstring du fusionné :
« padding réfléchissant (miroir + négation de la qdm normale) »). C'est
correct pour une borne de coût, FAUX pour la physique : le bord d'une fenêtre
de la pyramide n'est pas un mur, c'est une continuation vers le monde plus
grossier. De vrais halos doivent remplir la bordure `±2` (stencil du schéma,
« lecture d'un halo ±2 ») depuis l'extérieur.

### Volume d'échange

Fenêtre `n_fov = 512`. Halo largeur 2, quatre côtés :

```
cellules de halo ≈ 4 · n_fov · 2 = 4096 par champ   (≈ 1,6 % de l'aire 512²)
```

~1,6 % de trafic mémoire en plus par fenêtre et par pas. Multiplié par les
15 blocs actifs, cela reste ~1,6 % du trafic de F.

### D'où vient le halo, et coûte-t-il un transfert ?

**Sur GPU, sans transfert.** Dans la géométrie emboîtée, la fovéa est UNE
fenêtre fine par niveau ; au-delà de son bord il n'existe pas de fine
voisine — seulement le **parent**, plus grossier, déjà résident. Le halo
`±2` fin = `±1` cellule parent, upsamplée : c'est **exactement la prédiction
GPU-side étendue de 2 cellules au-delà du bord**. Le mécanisme existe déjà
(la prédiction remplit les colonnes entrantes) ; le halo en est un
prolongement latéral. Aucun H2D/D2H : le parent est sur le device.

### Coût en lancements et en temps

- **Lancements** : soit le halo est fusionné dans le kernel F (il lit le
  buffer parent) — 0 lancement de plus ; soit un petit kernel de gather
  séparé — +1–2 lancements/frame, soit ~quelques µs (ordre établi pour L1).
- **Temps** : ~1,6 % de trafic sur un F memory-bound ⇒ **~0,2–0,5 ms/frame**.
  Petit.

Le coût des halos n'est donc PAS dans le runtime — il est dans le **LOC et la
fidélité** : gather correct du halo depuis le parent à la bonne résolution,
sur des bords emboîtés, et surtout la question de savoir si un halo
parent-grossier rend la physique de bord FIDÈLE au domaine 64² du rederive
(qui, lui, a de vrais murs réfléchissants au bord réel). C'est le glissement,
pas la milliseconde.

---

## 3. Re-chiffrage de la tranche-1 de M-b, les deux termes intégrés

### Budget par frame au point de fonctionnement 1 pas/frame

C'est le point où l'ancre a été mesurée (~2–4× temps réel). Table :

| Poste | ms | Origine |
|---|---|---|
| F fidèle (même motif E4a) | 12,655 | ancre V4 |
| Bathymétrie réelle (hydrostatique non nul) | +0,1 – 0,3 | chemin calculé mais jamais exercé |
| Halos de fenêtre | +0,2 – 0,5 | §2, GPU sans transfert |
| Exner (amorti, save_every=2) | +0,65 – 1,25 | passe pointwise ~0,1–0,2× F, 1 fois/2 pas |
| Remontée L3 amortie (k=4) | +1,28 | borne L3 |
| Transferts (à 1e-2) | +0,11 | mesuré |
| **Total frame-pas** | **~15,0 – 16,1 ms** | |
| Marge à 16,7 | **+0,6 … +1,7 ms** | |

### Verdict T1-médiane (§A25 : T1 sur la médiane), franc

**Pas déjà compromis sur le papier — mais plausible seulement sous condition,
et la condition est une décision de design, pas de physique.**

1. **En temps réel (pas/frame < 1)** : la médiane est une frame SANS pas de F,
   donc bon marché (~remontée + transferts + rendu ≈ 1,5 ms). T1-médiane passe
   trivialement. Tout le coût est en p99 (les frames-pas à ~15–16 ms) — la
   question du stutter, non couverte par L1.

2. **Au point de M-a-quater (1 pas/frame, ~2–4× temps réel)** : médiane
   ~15,0–16,1 ms, **marge 0,6–1,7 ms**. Positive mais MINCE — de l'ordre de
   quelques fois la dérive machine (0,18 ms). Une estimation d'Exner au haut
   de la fourchette, ou une frame-Exner + frame-pas coïncidentes, peut la
   croiser. « Plausible mais tendu », pas « confortable ».

3. **En accéléré au-delà de ~1,8–4,2× temps réel (pas/frame > 1)** : F se
   multiplie linéairement — `F = (K / 1,8…4,2) · 12,655`. À K = 6× temps réel,
   F ≈ 1,4–3,3 × 12,655 ≈ 18–42 ms. **T1 est alors compromis outright**, et
   aucun des autres postes n'y change rien.

**Le paramètre décisif est la vitesse de jeu K, que le CFL ne fixe pas.** La
structure même de M-b (historique de ~24 épisodes, 6 émissions à Δt=4) suppose
que la MESURE défile en accéléré massif ; mais ce que T1 protège, c'est le
JEU, et sa vitesse de rendu est le paramètre ouvert. À graver avant l'achat :
**à quelle vitesse le monde persistant se joue-t-il ?** À temps réel, T1 est
sûr (et le stutter devient la vraie question) ; à ≤ 2–4× il est tendu ; au-delà
il casse.

### Un terme qui peut TOUT déplacer, nommé et non résolu

L'ancre 12,655 est à `n_fov = 512`. Le substrat réel (`default_terrain`) est
**64²**. « Le pipeline V4 complet » impose n_fov = 512, donc le 64² doit être
plongé dans un monde 512 (pavé ou upscalé) — ou la pyramide de M-b tourne à un
n_fov réduit. Si M-b tourne à l'échelle 64, l'aire par fenêtre chute d'un
facteur ~64 et **l'ancre F s'effondre** (~0,2 ms), rendant T1 trivial ; si
elle tourne à 512, l'ancre tient. Ce n'est PAS tranché, et cela change le
verdict d'un ordre de grandeur. Réconciliation d'échelle substrat↔pyramide :
à graver avec le reste.

---

## 4. Ce qui doit être gravé avant l'achat de M-b (la réserve sur l'ancre)

Ce document EST le pré-enregistrement paper-grade des deux termes. Restent à
trancher, AVANT toute ligne de code, et par Romain :

1. **La vitesse de jeu K** (§3). Elle décide si l'ancre se multiplie. Sans
   elle, le budget T1 n'a pas de valeur — c'est le budget faux qu'on préfère
   découvrir au papier (une séance) qu'au premier run (trois).
2. **T1 médiane vs le stutter** : §A25 met T1 sur la médiane et garde le
   stutter à part. Mais le mapping montre que le coût de F MIGRE vers la p99
   en temps réel, et que L1 ne l'étale pas. La p99 doit être un critère
   NOMMÉ de M-b, pas un report — sinon T1-médiane passera en cachant le
   stutter qu'il était censé exclure.
3. **La condition de mur intérieur** (halo parent vs mur réfléchi) : le
   critère de fidélité de bord, figé à l'avance, faute de quoi « assez
   fidèle » se règle après la courbe.
4. **La réconciliation d'échelle** substrat 64² ↔ pyramide n_fov (§3), qui
   change le verdict d'un ordre de grandeur.

---

## 5. Portée

Kernels FIGÉS intouchés (empreintes `_SOURCE` = e18015f5…f30b4 / `_SOURCE_L3`
= 9533a130…781a) ; k reste 4 ; `run_f1_attribution_b.py` gaté ; pas de miroir
CPU. Aucune mesure : ce document est une analyse sur le papier. L'achat de M-b
reste gaté sur les quatre points du §4.

**POINT D'ARRÊT.** Document remonté, rien d'autre. Aucune ligne de code M-b
écrite, aucun run.
