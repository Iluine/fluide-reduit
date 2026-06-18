# POC — Simulateur réduit appris de fluide 2D (shallow-water)

Modèle d'ordre réduit (POD + DMD) approximant un solveur shallow-water 2D, pour
trancher trois hypothèses : H1 (colonne vertébrale POD+DMD), H2 (dérive du
rollout long-horizon), H3 (couture multirésolution mobile).

## Installation

**Prérequis :** Python 3.12+, `uv` (gestionnaire de paquets).

```bash
# Créer et activer l'environnement virtuel
uv venv .venv --python 3.12

# Installer les dépendances (CPU uniquement pour v1)
uv pip install --python .venv/bin/python -r requirements.txt
```

## Exécution complète du pipeline

Le pipeline complet (M0 → M1 → M2 → M3 → M6 → M7) s'exécute en cascade par :

```bash
.venv/bin/python scripts/run_m0_generate.py   # Oracle : 4 rollouts × 201 frames (603 snapshots d'entraînement)
.venv/bin/python scripts/run_m1_pod.py        # POD : réduction k=43 modes
.venv/bin/python scripts/run_m2_dmd.py        # DMD : apprise du système réduit
.venv/bin/python scripts/run_m3_eval_rollout.py  # H2 : évaluation long-horizon
.venv/bin/python scripts/run_m6_multiresolution.py # H3 : couture multirésolution
.venv/bin/python scripts/run_m7_render.py     # Rendu : heatmap + surface eta
```

Chaque script sauvegarde ses sorties dans `./outputs/` (PNG + GIF d'animation).

## Tests

```bash
.venv/bin/python -m pytest -q  # Suite complète (27 tests)
```

Couverture : POD (encode/decode, énergie), DMD (fit, rollout, rayon spectral),
shallow-water (solver, CFL), métrique (erreur L2, dérive masse, saut couture),
multiresolution (down/up, fendu mobile), rendu (heatmap, surface).

## Architecture

| Module | Rôle |
|--------|------|
| `src/solver.py` | Solveur shallow-water 2D, flux de Rusanov (Lax-Friedrichs local) conservatif (M0) |
| `src/pod.py` | SVD réduit + encode/decode (M1) |
| `src/dmd.py` | DMD (fit moindres carrés, rollout autorégressif) (M2) |
| `src/metrics.py` | Évaluations H2/H3 : erreur L2, dérive masse, saut couture |
| `src/multiresolution.py` | Downsampling/upsampling + fondu linéaire dans une fenêtre mobile (M6) |
| `src/render.py` | Heatmap de la hauteur h + surface libre η=h+b (M7) |
| `src/io_utils.py` | Charge/sauvegarde GIF (PIL fallback matplotlib) |
| `config.py` | Constantes partagées (gravité, grille, solver, POD) |

## Sorties

### M0 — Oracle (solveur complet)

- `m0_control_drop_center.gif` : Animation shallow-water (CI goutte centrée, 800 pas → 201 frames)

### M1 — POD (réduit k=43)

- `m1_energy.png` : Courbe d'énergie cumulée vs nombre de modes → seuil 99.99 % atteint en k=43
- `m1_modes.png` : Visualisation des 4 premiers modes spatiaux (canal h)
- `m1_recon_error_vs_k.png` : Erreur de reconstruction (L2 test) vs k

### M2 — DMD (apprentissage du système réduit)

- `m2_dmd_vs_truth.gif` : Comparaison côte à côte DMD rollout (court horizon) vs vérité

### M3 — Évaluation H2 (rollout long-horizon)

- `m3_longhorizon_drop_center.gif` : Rollout DMD sur CI vue (201 frames, vérité | prédiction côte à côte)
- `m3_longhorizon_drop_test.gif` : Rollout DMD sur CI test (généralisation)
- `m3_error_growth.png` : Erreur L2 relative de HAUTEUR (h) vs temps (vue vs test)
- `m3_velocity_rms.png` : RMS absolu des vitesses (u, v) vs temps (vue vs test)
- `m3_mass_drift.png` : Dérive de masse (différentiel hauteur intégré) vs temps

### M6 — Évaluation H3 (multirésolution + couture)

- `m6_seam_jump.png` : Saut de couture (saut moyen vs max, collage dur vs fondu w=3)
- `m6_window_moving.gif` : Fenêtre fine mobile sur fond grossier (illustration du fondu linéaire)

### M7 — Rendu

- `m7_height_heatmap.gif` : Animation heatmap de la hauteur d'eau h prédite
- `m7_surface_eta.gif` : Surface libre η = h + b (heatmap, cmap 'terrain')

## Interprétation des résultats

| Hypothèse | Métrique | Seuil | Mesuré (drop_center) | Mesuré (drop_test) | Verdict |
|-----------|----------|-------|----------------------|--------------------|---------|
| **H1 : POD+DMD** | Énergie POD @ k=43 | ≥99.99 % | 99.99 % | — | ✅ PASS |
| **H1 : POD+DMD** | Rayon spectral A (DMD) | <1.05 | 1.010 | — | ✅ PASS (stable) |
| **H2 : Hauteur** | h_rel_final (err L2 relative h) | <50 % | 5.1 % | 13.2 % | ✅ PASS (borné) |
| **H2 : Hauteur** | h_rel_max | <500 % | 5.1 % | 19.8 % | ✅ PASS |
| **H2 : Vitesses** | u_rms_final (RMS absolu) | non-exp. | 0.246 m/s | 0.347 m/s | ✅ PASS |
| **H2 : Vitesses** | v_rms_final (RMS absolu) | non-exp. | 0.050 m/s | 0.479 m/s | ✅ PASS |
| **H2 : Masse** | Dérive masse finale | <5 % | 2.0 % | 1.5 % | ✅ PASS |
| **H3 : Couture** | Collage dur (moyen) | - | 0.029 | — | ✅ PASS |
| **H3 : Couture** | Fondu w=3 (moyen) | −67 % vs collage | 0.010 | — | ✅ PASS |

> **Note :** l'ancienne figure ~30–37 % (erreur empilée [h,u,v]) était dominée par
> les vitesses dont l'erreur relative explose quand ‖u‖→0 (phases calmes).
> En rapportant h séparément (métrique relative) et u,v en RMS absolu, l'erreur
> HEIGHT est 5–13 % (bornée). La couture verticale visible avec k=16 est résolue
> en passant à k=43 (énergie 99.99 %).

## Résultats des hypothèses

- **H1 ✅** : POD reconstruit à 99.99 % d'énergie avec k=43 modes (sur 603 snapshots) ; DMD reproduit la dynamique à court horizon (rayon spectral 1.010, stable). Voir `m1_energy.png`, `m2_dmd_vs_truth.gif`.

- **H2 (caractérisé, par canal)** : Rollout stable/borné, non explosé.
  - HEIGHT (h) : erreur L2 relative finale 5.1 % (vue) / 13.2 % (test), max 5.1 % / 19.8 %. Bornée et faible.
  - VITESSES (u, v) : RMS absolu final 0.246 / 0.050 m/s (vue) et 0.347 / 0.479 m/s (test). Métrique non normalisée (robuste aux phases calmes où ‖u‖→0).
  - Dérive masse : 2.0 % (vue) / 1.5 % (test).
  - *L'ancienne valeur ~30 % était l'erreur sur l'état empilé [h,u,v], dominée par les vitesses ; la couture verticale était un artefact k=16 résolu par k=43.*
  - Voir `m3_error_growth.png`, `m3_velocity_rms.png`, `m3_mass_drift.png`.

- **H3 (caractérisé)** : Couture cohérente ; saut moyen collage dur 0.029 / max 0.081, fondu w=3 0.010 / max 0.020 (−67 %/−75 %). Voir `m6_seam_jump.png`.

## Décision M4/M5

La baseline DMD (M1–M3, k=43) satisfait aux critères stricts de l'H2 (erreur HEIGHT
bornée à 5–13 %, pas d'explosion, dérive masse ~2 %). Cependant, si une fidélité
long-horizon meilleure sur la CI de test (h_rel < 10 %) ou l'annulation de la dérive
de masse de ~2 % est souhaitée, M4 (dynamique non-linéaire) ou M5 (pénalité de
conservation) doivent être envisagés. **Ces modules restent déférés en v1** ; voir
`docs/M4_M5_decision_gate.md` pour le critère d'activation explicite.

## Notes

- **Indexation des tableaux** : `array[y, x]` (axe 0 = y = lignes = H, axe 1 = x = colonnes = W).
- **Shallow-water 2D** : volumes finis, flux de Rusanov (LF local), conservatif, CFL 2D = 0.45.
- **POD** : SVD + dimensionnement par canal (h, u, v).
- **DMD** : Moindres carrés + rayon spectral pour stabilité.
- **Multirésolution** : Downsampling par bloc, upsampling linéaire, fondu linéaire dans une fenêtre mobile pour
  lisser la couture entre résolutions.
