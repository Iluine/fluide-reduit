# POC — Simulateur réduit appris de fluide 2D (shallow-water)

Modèle d'ordre réduit (POD + DMD) approximant un solveur shallow-water 2D, pour
trancher trois hypothèses : H1 (colonne vertébrale POD+DMD viable), H2 (rollout
long-horizon stable et borné), H3 (couture multirésolution mobile cohérente).
Le parcours de durcissement a isolé et résolu trois défauts orthogonaux —
représentation, opérateur, conservation — avant de conclure.

**v2 (en cours)** étend le POC à des **terrains nouveaux** (le POC tient sur un
seul terrain fixe — tag `v1`). La colonne diagnostique V0+V1 mesure si une base
POD statique de hauteur généralise hors du terrain d'entraînement. Verdict :
**oui dans le régime submergé/réfraction** (interpolation et obstacle extrapolé
< 2 %), mais une **topologie jamais vue (canal) coûte 16 %**. Détails dans la
section **v2 — Généralisation inter-terrain** plus bas.

---

## Installation

**Prérequis :** Python 3.12+, `uv` (gestionnaire de paquets).

```bash
# Créer l'environnement virtuel
uv venv .venv --python 3.12

# Installer les dépendances (CPU uniquement)
uv pip install --python .venv/bin/python -r requirements.txt
```

---

## Exécution du pipeline

Le pipeline s'exécute en cascade M0 → M1 → M2 → M3 → M5 → M6 → M7 (il n'y a
**pas** de script run_m4) :

| Script | Rôle |
|--------|------|
| `scripts/run_m0_generate.py` | Génère l'oracle : 4 CI × 800 pas → 201 frames chacune (603 snapshots d'entraînement) |
| `scripts/run_m1_pod.py` | Calcule la base POD k=43, sauvegarde modes et courbe d'énergie |
| `scripts/run_m2_dmd.py` | Ajuste l'opérateur DMD sur les coordonnées réduites, applique l'écrêtage des valeurs propres |
| `scripts/run_m3_eval_rollout.py` | Évalue le rollout long-horizon (H2) par canal : erreur HEIGHT, RMS vitesses, dérive masse |
| `scripts/run_m5_mass_projection.py` | Applique la projection de masse (offset additif uniforme) et mesure son effet sur la dérive |
| `scripts/run_m6_multiresolution.py` | Évalue la couture multirésolution mobile (H3) : saut dur vs fondu linéaire |
| `scripts/run_m7_render.py` | Génère les animations de la hauteur h (heatmap) et de la surface libre η=h+b (terrain) |

Chaque script sauvegarde ses sorties dans `./outputs/`.

---

## Tests

```bash
.venv/bin/python -m pytest -q   # 54/54 (35 POC + 19 v2)
```

Couverture POC : POD (encode/decode, énergie), DMD (fit, rollout, rayon spectral,
écrêtage), shallow-water (solver Rusanov, CFL 2D), métriques (relative_l2_error,
rms_growth, mass_drift, seam_jump), multiresolution (down/up, fenêtre mobile,
fondu linéaire), rendu (heatmap, surface η), mass_projection (offset, garde-fou).
Couverture v2 : terrains (famille paramétrée, CI submergée, résidu au repos /
well-balancedness), POD `n_channels` (hauteur seule, non-régression du chemin
3-canaux), génération V0 (contrat données + garde-fous), plafond V1.

---

## Architecture

| Module | Rôle |
|--------|------|
| `src/solver.py` | Solveur shallow-water 2D, flux de Rusanov (Lax-Friedrichs local), CFL 2D, parois réfléchissantes, masse conservée à ~2e-16 (M0) |
| `src/pod.py` | SVD réduit, standardisation par canal (`scale`), encode/decode (M1) ; `n_channels` pour une POD hauteur seule (v2, défaut 3 = chemin POC inchangé) |
| `src/dmd.py` | DMD — fit moindres carrés, rollout autorégressif, `clip_eigenvalues` (ρ≤1), rayon spectral (M2) |
| `src/metrics.py` | `relative_l2_error`, `error_growth`, `rms_growth`, `mass_series`, `seam_jump` (H2/H3) |
| `src/mass_projection.py` | Projection de masse — offset uniforme additif, garde-fou positivité (M5) |
| `src/multiresolution.py` | Downsampling/upsampling par bloc, fondu linéaire dans une fenêtre mobile (M6) |
| `src/render.py` | Heatmap hauteur h + surface libre η=h+b colormap terrain (M7) |
| `src/terrains.py` | **(v2)** Famille de terrains paramétrée (bosse/obstacle gaussiens + canal tanh), CI au repos submergée, `rest_residual` (well-balancedness), tirage déterministe train/holdout |
| `src/io_utils.py` | Chargement/sauvegarde GIF (PIL avec fallback matplotlib) |
| `config.py` | Constantes partagées (gravité, grille, CFL, seuil énergie POD) |

---

## Résultats — les trois hypothèses

### Fil rouge : trois défauts orthogonaux, trois niveaux

Le durcissement a révélé trois problèmes indépendants, chacun isolé par un
diagnostic ciblé avant toute action :

1. **Représentation** — une couture verticale apparaissait dans les rendus en
   phase calme. La reconstruction plein-rang étant exacte, ce n'était pas un bug
   de reshape : c'était un artefact de troncature k=16 (énergie POD insuffisante).
   Résolu en portant k à 43 (seuil 99.99 %).

2. **Opérateur** — passer à k=43 a introduit 2 modes de queue avec |λ|>1
   (rayon spectral brut ρ=1.010), provoquant une croissance non physique sur la
   CI de test non vue (erreur hauteur max 19.8 %). Résolu par écrêtage des
   valeurs propres (ρ→1.0), sans ajouter de modes. L'erreur hauteur max a été
   divisée par 2.6 (19.8 % → 7.7 %).

3. **Conservation** — une dérive de masse ~2 % croissante persiste,
   indépendamment de k, comme signal structurel de la dynamique tronquée.
   Résolu par la projection M5 (offset uniforme additif open-loop), ramenée à
   la précision machine (~1e-14).

L'ancienne valeur ~30 % d'erreur était un artefact de métrique : l'état empilé
[h,u,v] était dominé par les vitesses dont l'erreur relative explose quand
‖u‖→0 en phase calme. Rapportée par canal, l'erreur HEIGHT est 5–7 % (bornée).

---

### H1 — Colonne vertébrale POD+DMD ✅

- **POD (M1)** : k=43 modes à seuil énergie 0.9999 (99.99 % d'énergie
  cumulative). Compression d'état : **43 coordonnées latentes** pour un état complet de 3 canaux × 64 × 64 = **12 288 valeurs**, soit **≈ 0,35 % (réduction ≈ 286×)**. Les 43 modes sont retenus à partir de 603 snapshots d'entraînement — c'est le rang de la base, à ne pas confondre avec le taux de compression. Standardisation per-canal (`scale`).
  Contrat : `X ≈ scale·(Phi·z) + mean`.
- **DMD (M2)** : opérateur linéaire A (43×43) par moindres carrés. Rayon
  spectral brut ρ=1.010 (2 valeurs propres > 1) → écrêté à ρ=1.000.
  L'écrêtage est la baseline opérationnelle.

Figures : `m1_energy.png`, `m1_modes.png`, `m1_recon_error_vs_k.png`,
`m2_dmd_vs_truth.gif`.

---

### H2 — Rollout long-horizon stable et borné ✅

Évaluation sur 201 frames (800 pas, save_every=4). 4 CI :
drop_center, drop_offset, dam_break (entraînement) ; drop_test (test retenu).

| Métrique | drop_center (vue) | drop_test (non vu) |
|----------|-------------------|-------------------|
| h_rel_final | ~5.1–5.5 % | 6.7 % |
| h_rel_max | ~5.5 % | 7.7 % (avant écrêtage : 19.8 %) |
| u_rms_final | 0.250 m/s (74 % de la réf 0.335 m/s) | 0.261 m/s (86 % de la réf 0.304 m/s) |
| v_rms_final | 0.062 m/s (18 % de la réf 0.335 m/s) | 0.091 m/s (30 % de la réf 0.304 m/s) |
| Dérive masse finale | ~2 % (croissante) | ~1.5 % |
| Explosé ? | non | non |

Les vitesses sont rapportées en RMS absolu normalisé par la référence d'échelle
(max-sur-t du RMS par frame de la vérité), robuste aux phases calmes où ‖u‖→0.

Figures : `m3_longhorizon_drop_center.gif`, `m3_longhorizon_drop_test.gif`,
`m3_error_growth.png`, `m3_velocity_rms.png`, `m3_mass_drift.png`.

---

### M5 — Projection de masse ✅

Offset uniforme additif par frame (garde-fou de sortie open-loop). La masse
cible est la masse de la CI initiale — valide car les parois réfléchissantes
conservent la masse vraie à 2e-16 (précision machine).

| CI | Dérive OFF | Dérive ON | h_rel_final OFF | h_rel_final ON |
|----|-----------|-----------|-----------------|----------------|
| drop_center (vue) | +1.96 % | ~2.15e-14 % | 0.0548 | 0.0512 |
| drop_test (test) | +1.54 % | ~0 % | 0.0675 | 0.0657 |

L'erreur hauteur est inchangée ou légèrement améliorée (5.48 % → 5.12 % sur
drop_center) : la composante de dérive du niveau moyen est absorbée.

Figure : `m5_mass_drift.png`.

---

### H3 — Couture multirésolution mobile ✅

Fenêtre fine mobile sur fond grossier. Le fondu linéaire (feathering) réduit
le saut de couture de **−67 % en moyenne / −75 % au maximum** par rapport au
collage dur :

| Mode | Saut moyen | Saut max |
|------|-----------|---------|
| Collage dur | 0.029 | 0.081 |
| Fondu linéaire w=3 | 0.010 | 0.020 |

Figures : `m6_seam_jump.png`, `m6_window_moving.gif`.

---

## Sorties

### M0 — Oracle (solveur complet)

- `m0_control_drop_center.gif` : animation shallow-water CI drop_center (800 pas → 201 frames)

### M1 — POD (k=43)

- `m1_energy.png` : énergie cumulative vs nombre de modes (seuil 99.99 % à k=43)
- `m1_modes.png` : 4 premiers modes spatiaux (canal h)
- `m1_recon_error_vs_k.png` : erreur L2 de reconstruction vs k

### M2 — DMD

- `m2_dmd_vs_truth.gif` : rollout court-horizon DMD vs vérité (côte à côte)

### M3 — Évaluation H2

- `m3_longhorizon_drop_center.gif` : rollout 201 frames, CI vue (vérité | prédiction)
- `m3_longhorizon_drop_test.gif` : rollout 201 frames, CI test (généralisation)
- `m3_error_growth.png` : erreur L2 relative de la hauteur h vs temps (vue vs test)
- `m3_velocity_rms.png` : RMS absolu des vitesses u, v vs temps (vue vs test)
- `m3_mass_drift.png` : dérive de masse vs temps

### M5 — Projection de masse

- `m5_mass_drift.png` : dérive OFF (DMD brut, ~2 %) vs ON (projeté, ~précision machine)

### M6 — Couture multirésolution (H3)

- `m6_seam_jump.png` : saut de couture moyen/max — collage dur vs fondu w=3
- `m6_window_moving.gif` : fenêtre fine mobile sur fond grossier (illustration du fondu)

### M7 — Rendu

- `m7_height_heatmap.gif` : animation heatmap de la hauteur d'eau h prédite
- `m7_surface_eta.gif` : surface libre η = h + b (colormap terrain)

---

## M4 et M5 — Décision de portée

**M5 est implémenté** : la projection de masse ramène la dérive de ~2 % à la
précision machine sans dégradation de l'erreur hauteur. C'est un garde-fou de
sortie open-loop, non une dynamique conservative apprise.

**M4 (approximateur non-linéaire, correcteur MLP résiduel sur les coordonnées
réduites) est déféré** : sa cible principale — les vitesses — est invisible au
livrable visuel, et les vitesses sur la CI non vue butent sur le plafond de
représentation POD (~30 % de l'erreur sur unseen ICs). Sur la CI vue, les
vitesses sont dynamics-limited (représentation 0.9 % mais ~74–86 % de la
référence en rollout), ce qui signifie que même une correction parfaite
n'améliorerait pas le visuel. Les critères stricts d'activation H2 (erreur
HEIGHT < 50 %, pas d'explosion, dérive < 5 %) ne sont pas déclenchés.

Voir `docs/M4_M5_decision_gate.md` pour les critères d'activation formels et
les résultats mesurés complets.

---

## Limites caractérisées

**Portée de la validation** : ces résultats valent dans un cadre frugal — un seul terrain (bathymétrie statique), des régimes doux (gouttes gaussiennes, rupture de barrage modérée ; pas de choc transcritique ni de turbulence), 2D uniquement (ni 3D/voxel), hors-ligne (pas de temps-réel), et la généralisation n'est testée que sur **une** condition initiale jamais vue, sur le **même** terrain. Les trois hypothèses sont donc tranchées *dans ce périmètre*, pas prouvées en général.

- **Vitesses (plafond de représentation)** : sur les CI non vues, les vitesses
  atteignent ~30 % d'erreur relative. Ce plafond est structurel : augmenter k
  ou raffiner la dynamique DMD n'y changera pas grand chose sans plus de données
  d'entraînement diversifiées.
- **M5 est un garde-fou de sortie** : la projection de masse corrige la dérive
  en post-traitement open-loop ; ce n'est pas une dynamique conservative apprise.
  La variante par pénalité Lagrangienne sur le fit DMD reste une piste ouverte.

---

## v2 — Généralisation inter-terrain (V0+V1)

La v2 attaque la première limite ci-dessus — *un seul terrain* — en restant sur
le livrable visuel (hauteur ; les vitesses restent hors-scope). Elle réutilise le
solveur, la POD et les métriques du POC (tag `v1`) et n'ajoute que `src/terrains.py`
plus deux scripts. **Discipline diagnostic-first** : un seul pas décisif (V1) avant
de construire le moindre conditionnement de dynamique.

### V0 — Famille de terrains + oracle valide

Famille paramétrée : bosse/obstacle gaussiens (même générateur, plages
différentes) + un **canal à parois lissées (tanh)** réservé au holdout. Tirage
déterministe : **9 terrains d'entraînement** (5 bosses + 4 obstacles) × 2 CI, plus
3 holdout — 1 interpolation, 1 obstacle **extrapolé par la géométrie** (σ étroit
hors plage + position hors plage), 1 canal (**topologie jamais vue**).

CI au **repos submergé** (`h = η₀ − b`, η₀ = 1.5) : eau partout mouillée, la
dépendance au terrain passe par la **réfraction** (célérité √(g·h)). Le lit sec
est exclu volontairement — il mesurerait une défaillance du solveur (Rusanov non
well-balanced), pas un plafond de représentation.

Garde-fous d'oracle (la conservation de masse ~2e-16 est nécessaire mais **pas
suffisante**) : positivité avec marge (`min(h) = 0.424 > 0.1`), **résidu au repos /
well-balancedness** (max 0.106 < tol 0.15), sanity visuel des terrains extrap. Ce
garde-fou a effectivement tripé sur l'obstacle extrap le plus raide → amplitude
ramenée de 1.0 à 0.6 (résidu 0.188 → 0.106), σ géométrique conservé.

### V1 — Plafond de représentation inter-terrain (le pas décisif)

POD **hauteur seule** (k=32, seuil énergie 0.9999) ajustée sur les terrains
d'entraînement ; encode-décode du h **vrai** des terrains holdout. **Aucune
dynamique** — on mesure si la base statique *span* un terrain nouveau.

| régime | erreur L2 relative (hauteur) |
|--------|------------------------------|
| plancher train (in-sample) | 0.0004 |
| interpolation | 0.6 % |
| extrapolation obstacle (géométrie) | 1.8 % |
| extrapolation canal (topologie neuve) | **16.4 %** |

**Verdict : INTERMÉDIAIRE**, porté **entièrement par le canal**. Une base POD
statique de hauteur généralise presque trivialement dans le régime
submergé/réfraction (interpolation et obstacle extrapolé < 2 %) ; c'est la
topologie *jamais vue* qui la met à l'épreuve (16 %).

> `k = 32` (hauteur seule) n'est **pas** comparable au `k = 43` du POC (état à
> 3 canaux [h,u,v]) — les vitesses ajoutent des modes. La représentation de
> hauteur n'est donc pas le goulot pour les régimes doux.

### Couverture vs capacité (v1b) — le 16 % du canal est de la *couverture*

Le canal était **holdout-only** : le 16 % ne dit pas « la POD ne sait pas faire les
canaux », il dit « la base entraînée sur bosses+obstacles n'a aucun mode en forme de
bande ». La preuve est juste à côté — l'obstacle, **vu en train**, s'extrapole à 1.8 %
malgré une géométrie hors plage. Un balayage de couverture le confirme : on ajoute des
canaux (params différents) au train et on re-mesure le **même** canal holdout.

| canaux en train (×2 CI) | k | canal |
|--------------------------|----|-------|
| 0 | 32 | 16.4 % |
| 1 | 24 | 14.6 % |
| 2 | 24 | 9.3 % |
| 4 | 25 | **6.0 %** |

Chute **monotone, sans plateau** → **dominé par la couverture**. À couverture égale à
l'obstacle, le canal reste à ~6 % vs ~1.9 % (petit coût résiduel de la topologie
*bande*, pas un mur). Surtout, **`k` n'explose pas** (24–32) — c'est le signal qui
décide le gate V5 : couvrir une nouvelle topologie ne fait pas grossir la base linéaire
→ **on reste sur base statique**. Détail : `docs/v2_v1b_coverage_vs_capacity.md`.

### Portée (calibrage du ✅)

Tous les terrains sont **submergés** → la dépendance au terrain passe uniquement
par la réfraction (contraste de célérité ~1.7×, réel mais modéré). Un V1 qui passe
certifie la généralisation **dans le régime submergé/réfraction** — pas sur tout
terrain de jeu. Le sec / les îles (sillages, séparation) sont un régime distinct,
plus dur, **reporté en v2.5** (solveur mouillé/sec positivity-preserving &
well-balanced). Le plafond pourrait aussi être **data-limité** (9 terrains) : si un
plafond ressort haut, le check bon marché « ajouter des terrains fait-il baisser le
plafond ? » passe **avant** d'escalader vers un encodeur appris (GPU).

### Scripts v2

| Script | Rôle |
|--------|------|
| `scripts/run_v0_generate.py` | Génère le dataset (famille × CI) via le solveur, applique les garde-fous d'oracle, écrit `data/v2/<terrain>__<ci>.npz` + `data/v2/split.json` |
| `scripts/run_v1_representation.py` | Mesure le plafond de représentation inter-terrain (interp / extrap_obstacle / extrap_channel), écrit le verdict chiffré |

Spec : `docs/superpowers/specs/2026-06-19-v2-generalisation-terrains-design.md`.
Plan : `docs/superpowers/plans/2026-06-19-v2-generalisation-terrains.md`.
Verdict détaillé : `docs/v2_V1_representation_ceiling.md`.
Couverture vs capacité : `docs/v2_v1b_coverage_vs_capacity.md`.
Expérience : `scripts/exp_v1b_channel_coverage.py`.

### V2 + v2b — dynamique (résolu)

Un opérateur DMD **global** (fit sur les terrains d'entraînement) déroulé sur terrain
nouveau **transfère sur les topologies vues** (in-sample 2.4 %, interp 4.6 %,
obstacle 8.8 %) mais **cliffe sur le canal** holdout-only (157 %). Le test
couverture-opérateur (v2b) montre que ce cliff est de la **couverture**, pas de la
capacité :

| canaux dans le fit DMD | 0 | 1 | 2 | 4 |
|------------------------|----|----|----|----|
| rollout canal | 157 % | 36 % | 17 % | **11.8 %** |

Une fois le canal vu, le **gap opérateur** du canal (rollout − plancher = 5.8 %) est
**comparable à celui de l'obstacle** (6.8 %) — l'opérateur global est topologie-agnostique.
Le rollout canal résiduel (11.8 %) est **hérité de son plancher de représentation**
(6 %, le résidu n-width de v1b), pas un déficit d'opérateur. `k` reste 24–32, ρ=1.0.

→ **Pour le but visuel : ni V3a, ni V3b, ni V5.** Un seul A global + base statique
suffit, dès que le train **couvre le vocabulaire de topologies** (qu'un jeu contrôle).
*(Supporté, pas prouvé : effondrement de couverture net mais établi sur le point de
fonctionnement testé, ρ=1.0 marginal.)*
La foresight transport était juste sur le *lieu* : le résidu transport est dans la
**représentation**, pas l'opérateur. Détails : `docs/v2_V2_transfer.md`,
`docs/v2_v2b_operator_coverage.md` (scripts `run_v2_transfer.py`, `exp_v2b_operator_coverage.py`).

**Frontière restante (caractérisée, pas supposée)** : hors du régime
submergé / réfraction / vocabulaire couvert & borné en translation — le sec / les îles
(sillages, séparation → solveur mouillé/sec, **v2.5**) et les régimes à **forte
translation** (où la n-width linéaire mordrait → encodeur appris). Hors-scope.

---

## Notes techniques

- **Indexation** : `array[y, x]` (axe 0 = y = lignes, axe 1 = x = colonnes).
- **Solveur** : volumes finis, flux de Rusanov (LF local), CFL 2D, parois réfléchissantes.
- **POD** : SVD tronqué, seuil énergie 0.9999, standardisation per-canal (`scale`).
- **DMD** : moindres carrés + `clip_eigenvalues` (ρ≤1) pour la stabilité du rollout.
- **Métriques** : hauteur en erreur L2 relative, vitesses en RMS absolu normalisé par la référence d'échelle.
