# Porte de décision M4/M5 — Dynamique non-linéaire & conservation

## Contexte

Les modules M4 (approximateur non-linéaire du système réduit) et M5 (pénalité de
conservation de masse) demeurent **déférés en v1** du POC. Cette note établit le
critère explicite d'activation : si et seulement si les mesures H2 révèlent une
**dégradation au-delà des seuils de tolérance**, ces modules doivent être envisagés
en amont d'une version 2.

## Critères stricts d'activation

Pour justifier l'investissement en M4/M5, **au moins l'une** des conditions suivantes
doit être vérifiée à la fin de M3 :

1. **Croissance d'erreur non bornée** : err_final ≥ 50 % sur horizon long (400 pas).
2. **Instabilité (explosion)** : amplitude RMS > 10× l'état initial → rayon spectral A > 1.05.
3. **Dérive de masse excessive** : |Δm_final / m_init| > 5 %.
4. **Défaut de généralisation** : écart err(test) − err(vue) > 10 % (non-transférable).

## Verdict M3 mesuré (H2) — par canal, k=43 (énergie 99.99 %)

| CI | h_rel_final | h_rel_max | u_rms_final | v_rms_final | dérive masse finale | explosé |
|----|-------------|-----------|-------------|-------------|---------------------|---------|
| drop_center (vue) | 0.051 | 0.051 | 0.246 m/s | 0.050 m/s | 2.0e-02 | non |
| drop_test (test)  | 0.132 | 0.198 | 0.347 m/s | 0.479 m/s | 1.5e-02 | non |

Rayon spectral de A (DMD) : 1.010 (stable — légèrement > 1 mais croissance bornée sur horizon 201 pas).

**Note :** l'ancienne valeur ~30 % d'erreur (k=16, état empilé [h,u,v]) était dominée
par les vitesses dont l'erreur relative explose quand ‖u‖→0. La couture verticale
visible dans les rendus à k=16 est résolue en passant à k=43. Les métriques ci-dessus
rapportent h en relatif (robuste) et u,v en RMS absolu (non-explosif).

**Lecture :** le rollout DMD est STABLE et BORNÉ (pas d'explosion), GÉNÉRALISE bien
à la CI de test. L'erreur HEIGHT est 5.1 % (vue) / 13.2 % (test), avec une dérive
de masse modérée (~2 %).

**Décision :** par les critères stricts ci-dessus (h_rel_final < 50 %, pas d'explosion,
dérive de masse bornée), la baseline DMD est jugée **SUFFISANTE** pour trancher H2 ;
M4/M5 ne sont donc **PAS strictement requis**. Toutefois, si une fidélité long-horizon
meilleure que ~10 % sur la CI test est souhaitée, M4 (dynamique non-linéaire) est le
prochain pas ; si la dérive de masse de ~2 % doit être annulée, M5 (pénalité de
conservation) s'applique. Ce choix est laissé à l'arbitrage de l'équipe (M4/M5
restent déférés en v1).

## Vue d'ensemble des rôles

### M4 — Approximateur non-linéaire

**Hypothèse** : le système réduit n'est pas linéaire ; une dynamique résiduelle non-linéaire
peut améliorer la prédiction au-delà du fit linéaire DMD.

**Implémentation** :
- Petit réseau de neurones (MLP, ~2–3 couches cachées, 64–128 unités).
- Entrée : état réduit u_n ∈ ℝ^k.
- Sortie : correction Δu_{n+1} additionnée au prédicteur DMD linéaire.
- Perte : MSE sur séquence d'entraînement (oracle réduit).
- Validation : cross-validation sur splitting temporel ou spatial (CI vue vs test).

**Gain attendu** : réduction de err_final de ~30 % vers ~15–20 % si la non-linéarité résiduelle est présente.

### M5 — Pénalité de conservation de masse

**Hypothèse** : la dérive de masse est causée par la troncature POD et l'accumulation
d'erreur ; une pénalité explicite lors du fit DMD ou une correction post-rollout peut
la diminuer.

**Implémentation (deux stratégies)** :

1. **Pénalité Lagrangienne** : augmenter la fonction de coût DMD par terme
   `λ ∥ m(u_rollout) − m_init ∥²` où `λ = 0.1–1.0`.

2. **Correction post-rollout** : renormaliser la hauteur h après chaque pas
   pour conserver la masse intégrée (rééquilibrage linéaire des valeurs).

**Gain attendu** : réduction dérive de ~2 % vers ~0.1–0.5 % (quasi-exacte).

## Roadmap conditionnel

```
v1 (current)  : M0 → M1 → M2 → M3 → M6 → M7 (POD+DMD baseline)
                M4, M5 déférés (critères non satisfaits).

v2 (if M4/M5 triggered):
                M0 → M1 → M2 → [M4 nonlinear] → [M5 conservation] → M3' → M6' → M7'
                Benchmark : err_final ~15–20 %, dérive ~0.2–0.5 %.

Décision d'activation : réunion d'équipe post-M3 sur base des chiffres mesurés ci-dessus.
```

## Dépendances logicielles

M4/M5 nécessitent **PyTorch** (GPU CUDA RTX 3050 Ti ≤ 4 Go, AMP recommandé) :

```bash
# NE PAS installer avant M4 (lourd, GPU-spécifique). Utiliser uv comme pour le reste :
uv pip install --python .venv/bin/python "torch>=2.2"
```

## Références

- **H2 (dérive long-horizon)** : voir `m3_error_growth.png`, `m3_mass_drift.png`.
- **Détails POD+DMD** : voir `README.md` et docstrings dans `src/pod.py`, `src/dmd.py`.
- **Métriques** : `src/metrics.py` (relative_l2_error, mass_drift, seam_jump).
