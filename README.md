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
.venv/bin/python scripts/run_m1_pod.py        # POD : réduction k=16 modes
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

### M1 — POD (réduit k=16)

- `m1_energy.png` : Courbe d'énergie cumulée vs nombre de modes → seuil 99 % atteint en k=16
- `m1_modes.png` : Visualisation des 4 premiers modes spatiaux (canal h)
- `m1_recon_error_vs_k.png` : Erreur de reconstruction (L2 test) vs k

### M2 — DMD (apprentissage du système réduit)

- `m2_dmd_vs_truth.gif` : Comparaison côte à côte DMD rollout (court horizon) vs vérité

### M3 — Évaluation H2 (rollout long-horizon)

- `m3_longhorizon_drop_center.gif` : Rollout DMD sur CI vue (201 frames, vérité | prédiction côte à côte)
- `m3_longhorizon_drop_test.gif` : Rollout DMD sur CI test (généralisation)
- `m3_error_growth.png` : Croissance d'erreur L2 relative vs temps (vue vs test)
- `m3_mass_drift.png` : Dérive de masse (différentiel hauteur intégré) vs temps

### M6 — Évaluation H3 (multirésolution + couture)

- `m6_seam_jump.png` : Saut de couture (saut moyen vs max, collage dur vs fondu w=3)
- `m6_window_moving.gif` : Fenêtre fine mobile sur fond grossier (illustration du fondu linéaire)

### M7 — Rendu

- `m7_height_heatmap.gif` : Animation heatmap de la hauteur d'eau h prédite
- `m7_surface_eta.gif` : Surface libre η = h + b (heatmap, cmap 'terrain')

## Interprétation des résultats

| Hypothèse | Métrique | Seuil | Mesuré | Verdict |
|-----------|----------|-------|--------|---------|
| **H1 : POD+DMD** | Énergie POD @ k=16 | ≥99 % | 99.0 % | ✅ PASS |
| **H1 : POD+DMD** | Rayon spectral A (DMD) | <1 | 0.999 | ✅ PASS (stable) |
| **H2 : Dérive H2** | Err final L2 | <50 % | ~30–37 % | ✅ PASS (borné) |
| **H2 : Dérive H2** | Pas d'explosion | - | NON | ✅ PASS |
| **H2 : Dérive H2** | Dérive masse | <5 % | ~2 % | ✅ PASS |
| **H3 : Couture** | Collage dur (moyen) | - | 0.029 | ✅ PASS |
| **H3 : Couture** | Fondu w=3 (moyen) | −67 % vs collage | 0.010 | ✅ PASS |

## Résultats des hypothèses

- **H1 ✅** : POD reconstruit à 99 % d'énergie avec k=16 modes (sur 603 snapshots) ; DMD reproduit la dynamique à court horizon. Voir `m1_energy.png`, `m2_dmd_vs_truth.gif`.

- **H2 (caractérisé)** : Rollout stable/borné, erreur ~30–37 % (erreur L2 calculée sur l'état empilé [h,u,v], dominée par le canal h), dérive masse ~2 %, généralise (test≈vue). Voir `m3_error_growth.png`, `m3_mass_drift.png`.

- **H3 (caractérisé)** : Couture cohérente ; saut moyen collage dur 0.029 / max 0.081, fondu w=3 0.010 / max 0.020 (−67 %/−75 %). Voir `m6_seam_jump.png`.

## Décision M4/M5

La baseline DMD (M1–M3) satisfait aux critères stricts de l'H2 (erreur bornée, pas
d'explosion, dérive masse modérée). Cependant, si une fidélité long-horizon
meilleure que ~30 % ou l'annulation de la dérive de masse de ~2 % est souhaitée,
M4 (dynamique non-linéaire) ou M5 (pénalité de conservation) doivent être
envisagés. **Ces modules restent déférés en v1** ; voir
`docs/M4_M5_decision_gate.md` pour le critère d'activation explicite.

## Notes

- **Indexation des tableaux** : `array[y, x]` (axe 0 = y = lignes = H, axe 1 = x = colonnes = W).
- **Shallow-water 2D** : volumes finis, flux de Rusanov (LF local), conservatif, CFL 2D = 0.45.
- **POD** : SVD + dimensionnement par canal (h, u, v).
- **DMD** : Moindres carrés + rayon spectral pour stabilité.
- **Multirésolution** : Downsampling par bloc, upsampling linéaire, fondu linéaire dans une fenêtre mobile pour
  lisser la couture entre résolutions.
