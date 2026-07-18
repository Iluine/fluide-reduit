# Note — Budget mémoire VRAM (enveloppe paper-grade) — 2026-07-18

Objet : borner l'espace de design de la spec fovéa-z par la contrainte matérielle réelle
(RTX 3050 Ti, 4 Go). Calcul d'enveloppe PARAMÉTRIQUE, hypothèses nommées, à épingler par
Romain/la spec — pas des mesures. Consommateur : spec fovéa-z (section budget) +
tranche-moteur (critères de mort).

## Modèle : pyramide Harten bornée par la fovéa

Principe (architecture gravée) : un seul opérateur F sur niveaux multirésolution de
Harten ; la distance fixe un PLAFOND de LOD ; le raffinement fin n'existe que dans les
fenêtres actives (fovéa + énergie). Modèle géométrique : fovéa de n_fov cellules de côté
au niveau le plus fin ; chaque doublement de distance perd un niveau (règle
perspective-cohérente).

Résultat structurel (le point clé) : chaque anneau/coquille de distance contient un nombre
de cellules ~CONSTANT par niveau —

- 2D : anneau [2^k·r, 2^(k+1)·r] au niveau J−k : aire ~3·(2^k r)², cellule (2^k h)² →
  ~3·n_fov² cellules par niveau.
- 3D : coquille : volume ~7·(2^k r)³, cellule (2^k h)³ → ~7·n_fov³ par niveau.

D'où : **cellules_actives ≈ γ_d · n_fov^d · N_niv**, avec γ_2≈3, γ_3≈7, et
N_niv = log₂(taille_monde / taille_fovéa). **Le monde n'entre que par le log** — un monde
1000× plus étendu ne coûte que ~10 niveaux. Le budget est dominé par n_fov^d. C'est la
raison d'être du mariage Harten×fovéa, ici chiffrée.

## Formule d'enveloppe

M_total ≈ (2 + β) · c · b · γ_d · n_fov^d · N_niv

avec : c = nb de champs de z par cellule ; b = octets/champ (f32=4 — cible GPU ; f64=8
double tout) ; facteur 2 = double-buffering du pas de temps ; β = cache matérialisé
(dérivés : vorticité, pression…) en fraction de z, hypothèse β≈1.5.

## Scénarios (c=8 champs, f32, N_niv=10, β=1.5)

| d | n_fov | cellules actives | M_z | M_total (~) | verdict enveloppe |
|---|---|---|---|---|---|
| 2D | 512 | 7.9 M | 252 Mo | **~0.9 Go** | LARGE (reste rendu + framework) |
| 2D | 1024 | 31 M | 1.0 Go | ~3.5 Go | limite |
| 3D | 64  | 18 M | 0.6 Go | **~2.1 Go** | PASSE |
| 3D | 96  | 62 M | 2.0 Go | ~6.9 Go | CASSE |
| 3D | 128 | 147 M | 4.7 Go | ~16 Go | CASSE net |

**Le chiffre avec des dents : en 3D sur 4 Go, la fine-fovéa est bornée ≈ 64³ (c=8, f32,
hypothèses ci-dessus) ; 96³ ne passe pas.** Sensibilités : linéaire en c et b (c=4 f32
double le n_fov³ admissible ; f64 le divise par 2) ; cubique en n_fov ; linéaire en N_niv.

## Le registre (ledger) — négligeable en VRAM, borné en disque

Commit ≤ 400 floats (cap 409.6, §A13) ≈ 3.2 Ko (f64). À 1 commit/s : ~11.5 Mo/h de jeu.
Le ledger vit en RAM/disque, pas en VRAM ; sa boundedness pratique est un ARGUMENT du
verdict §A13 (coût par émission indépendant de l'historique), ici chiffrée en Mo/h.

## Leviers nommés, NON mesurés (à la spec de trancher lesquels compter)

- c variable par niveau (champs complets dans la fovéa, réduits au large) ;
- masques d'activité (wet/dry, repos sédimentaire) → cellules dormantes hors résidence ;
- compression des niveaux grossiers froids ; streaming VRAM↔RAM hors fovéa ;
- γ_d réel < géométrique si le plafond LOD par distance est plus agressif.

## Ce que cette note impose à la spec fovéa-z

1. Épingler (c, b, N_niv, n_fov^d cible) — le quadruplet EST le budget.
2. Section GPU-déterminisme OBLIGATOIRE : la bit-identité (É3, gels) est non-défaut sur
   GPU (ordre de réduction, atomics) ; la frontière éphémère/persistant doit dire QUI a
   droit au non-déterminisme et où vivent les commits (CPU-side ?).
3. La tranche-moteur (piste nommée au plan) mesure frame-time vs L_eff DANS cette
   enveloppe — critères de mort pré-écrits avant tout run.

Hypothèses à corriger par Romain si fausses : c≈8, règle un-niveau-par-doublement,
β≈1.5, double-buffer ×2.
