# POC — Simulateur réduit appris de fluide 2D (shallow-water)

Modèle d'ordre réduit (POD + DMD) approximant un solveur shallow-water 2D, pour
trancher trois hypothèses : H1 (colonne vertébrale POD+DMD viable), H2 (rollout
long-horizon stable et borné), H3 (couture multirésolution mobile cohérente).
Le parcours de durcissement a isolé et résolu trois défauts orthogonaux —
représentation, opérateur, conservation — avant de conclure.

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
.venv/bin/python -m pytest -q   # 35/35
```

Couverture : POD (encode/decode, énergie), DMD (fit, rollout, rayon spectral,
écrêtage), shallow-water (solver Rusanov, CFL 2D), métriques (relative_l2_error,
rms_growth, mass_drift, seam_jump), multiresolution (down/up, fenêtre mobile,
fondu linéaire), rendu (heatmap, surface η), mass_projection (offset, garde-fou).

---

## Architecture

| Module | Rôle |
|--------|------|
| `src/solver.py` | Solveur shallow-water 2D, flux de Rusanov (Lax-Friedrichs local), CFL 2D, parois réfléchissantes, masse conservée à ~2e-16 (M0) |
| `src/pod.py` | SVD réduit, standardisation par canal (`scale`), encode/decode (M1) |
| `src/dmd.py` | DMD — fit moindres carrés, rollout autorégressif, `clip_eigenvalues` (ρ≤1), rayon spectral (M2) |
| `src/metrics.py` | `relative_l2_error`, `error_growth`, `rms_growth`, `mass_series`, `seam_jump` (H2/H3) |
| `src/mass_projection.py` | Projection de masse — offset uniforme additif, garde-fou positivité (M5) |
| `src/multiresolution.py` | Downsampling/upsampling par bloc, fondu linéaire dans une fenêtre mobile (M6) |
| `src/render.py` | Heatmap hauteur h + surface libre η=h+b colormap terrain (M7) |
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
  cumulative). Compression ≈ 43/603 ≈ 7 %. Standardisation per-canal (`scale`).
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

- **Vitesses (plafond de représentation)** : sur les CI non vues, les vitesses
  atteignent ~30 % d'erreur relative. Ce plafond est structurel : augmenter k
  ou raffiner la dynamique DMD n'y changera pas grand chose sans plus de données
  d'entraînement diversifiées.
- **M5 est un garde-fou de sortie** : la projection de masse corrige la dérive
  en post-traitement open-loop ; ce n'est pas une dynamique conservative apprise.
  La variante par pénalité Lagrangienne sur le fit DMD reste une piste ouverte.

---

## Notes techniques

- **Indexation** : `array[y, x]` (axe 0 = y = lignes, axe 1 = x = colonnes).
- **Solveur** : volumes finis, flux de Rusanov (LF local), CFL 2D, parois réfléchissantes.
- **POD** : SVD tronqué, seuil énergie 0.9999, standardisation per-canal (`scale`).
- **DMD** : moindres carrés + `clip_eigenvalues` (ρ≤1) pour la stabilité du rollout.
- **Métriques** : hauteur en erreur L2 relative, vitesses en RMS absolu normalisé par la référence d'échelle.
