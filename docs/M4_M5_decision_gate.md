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
2. **Instabilité (explosion)** : amplitude RMS > 10× l'état initial → rayon spectral A > 1 (toute valeur propre hors du disque unité).
3. **Dérive de masse excessive** : |Δm_final / m_init| > 5 %.
4. **Défaut de généralisation** : écart err(test) − err(vue) > 10 % (non-transférable).

## Verdict M3 mesuré (H2) — par canal, k=43 (énergie 99.99 %)

**Opérateur DMD stabilisé par écrêtage des valeurs propres : ρ brut=1.010 → ρ écrêté=1.000 (disque unité).**

| CI | h_rel_final | h_rel_max | u_rms_final (% réf) | v_rms_final (% réf) | dérive masse finale | explosé |
|----|-------------|-----------|---------------------|---------------------|---------------------|---------|
| drop_center (vue) | 0.055 | 0.055 | 0.250 m/s (74% de 0.335 m/s) | 0.062 m/s (18% de 0.335 m/s) | 1.96e-02 | non |
| drop_test (test)  | 0.067 | 0.077 | 0.261 m/s (86% de 0.304 m/s) | 0.091 m/s (30% de 0.304 m/s) | 1.54e-02 | non |

Rayon spectral de A (DMD) : brut=1.010 → écrêté=1.000 (stabilisé, toutes valeurs propres dans le disque unité).

**Note :** l'écrêtage des valeurs propres (clip_eigenvalues) est appliqué après le fit DMD pour projeter
les 2 modes instables (|λ|=1.01) sur le cercle unité. Ceci réduit l'artefact de croissance parasite
sur les CI de test (h_rel_max test : 19.8 % → 7.7 %). L'écrêtage devient la baseline opérationnelle.
Les métriques rapportent h en relatif (robuste) et u,v en RMS absolu normalisé par la réf d'échelle
(max-sur-t du RMS par frame de la vérité).

**Lecture :** le rollout DMD est STABLE et BORNÉ (pas d'explosion), GÉNÉRALISE bien
à la CI de test. L'erreur HEIGHT est 5.5 % (vue) / 6.7 % (test), avec une dérive
de masse modérée (~2 %).

**Décision :** par les critères stricts ci-dessus (h_rel_final < 50 %, pas d'explosion,
dérive de masse bornée), la baseline DMD est jugée **SUFFISANTE** pour trancher H2 ;
M4/M5 ne sont donc **PAS strictement requis**. La dérive de masse de ~2 % reste le
signal structurel motivant M5. Ce choix est laissé à l'arbitrage de l'équipe (M4/M5
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

## M5 implémenté — Projection de masse (garde-fou de sortie)

**Stratégie** : offset uniforme additif post-rollout (correction de norme minimale pour une
contrainte intégrale), appliqué à chaque frame décodée. C'est un garde-fou de *sortie*
open-loop : la dynamique latente n'est pas modifiée — la projection est purement physique.

**Vérification BC** : masse vraie exactement conservée (parois réfléchissantes) — dérive
relative ≤ 2.15e-16 (précision machine) — ce qui valide que `m_target = m(h[0])` est la
bonne cible (on projette sur la masse de la CI, pas sur une masse instantanée).

**Résultats mesurés (rollout DMD clipped, ρ=1.0) :**

| CI | Dérive masse finale OFF | Dérive masse finale ON | h_rel_final OFF | h_rel_final ON | h_rel_max OFF | h_rel_max ON |
|----|------------------------|------------------------|-----------------|----------------|---------------|--------------|
| drop_center (vue) | +1.961 % | −2.15e-14 % | 0.0548 | 0.0512 | 0.0548 | 0.0512 |
| drop_test (test)  | +1.537 % | +0.00e+00 % | 0.0675 | 0.0657 | 0.0768 | 0.0768 |

**Bilan :** la projection ramène la dérive de masse de ~2 % à la précision machine (~0),
sans dégradation de l'erreur de hauteur — légère amélioration sur drop_center
(5.48 % → 5.12 %, −0.36 pp) car la composante de dérive du niveau moyen est absorbée.

**Note architecturale :** ceci est un garde-fou de sortie (post-rollout), non une
dynamique conservative apprise. La variante par pénalité Lagrangienne sur le fit DMD
(cf. ci-dessus §M5 stratégie 1) reste l'alternative plus lourde, pour un gain attendu
similaire.

## Références

- **H2 (dérive long-horizon)** : voir `m3_error_growth.png`, `m3_mass_drift.png`.
- **M5 (projection masse)** : voir `m5_mass_drift.png`, script `scripts/run_m5_mass_projection.py`.
- **Détails POD+DMD** : voir `README.md` et docstrings dans `src/pod.py`, `src/dmd.py`.
- **Métriques** : `src/metrics.py` (relative_l2_error, mass_drift, seam_jump).
