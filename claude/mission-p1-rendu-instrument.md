# ORDRE DE MISSION — noyau R1, rendu minimal d'instrumentation (P1, endossé §A37 le 2026-07-25)

> **Pour : session Claude Code.** Développement possible sur la VM Cowork (tout est
> CPU pur, tests sans GPU) ; la suite complète se re-passe sur iluin-tworings3 avant
> remontée. **Sources qui font foi** :
> `claude/spec-p1-rendu-instrument-2026-07-25.md` (la spec ENDOSSÉE, bloc REVUE
> inclus), `pocCascade2phys/PREREGISTRATION.md` §A37 (les décisions), §A20 (les
> invariants de projection), §C7 (la calibration).

## Le contrat, en trois lignes

R1 est une fonction **PURE, sans RNG, sans état** : champ albédo 64×64 float64
∈ [0,1] → image uint8 affichée à `taille_domaine_px`. Décisions gravées (§A37) :
**Option B** (pleine résolution, pas de compositeur), **Y = A** (identité),
**inverse-EOTF sRGB** gravé en formule, **rééchantillonnage bilinéaire** comme
étage de R1. L'observable Δχ et le pin sont INTOUCHÉS.

## AMENDEMENT-ÉTAGES — TRANCHÉ (Romain, 2026-07-25) : l'ordre AMENDÉ est retenu

La spec (§2.3) ordonne : encodage sRGB → quantification → rééchantillonnage.
Cet ordre interpole des valeurs quantifiées (double quantification) et en espace
gamma. **Ordre amendé proposé, à graver ici** :

```
A (64×64, [0,1]) → rééchantillonnage bilinéaire (espace LINÉAIRE = A, car Y = A)
                 → encodage sRGB (float) → quantification uint8 (une seule fois)
```

Justification : une moyenne bilinéaire de luminances est une luminance (l'intention
de D-P1-4) ; une seule quantification, en bout de chaîne.

**Décision Romain (2026-07-25), consignée avec son épistémique** : « pas de réponse
a priori — partons avec, et on verra au résultat si c'est un bon choix. » Le choix
n'est pas revendiqué supérieur : il est GRAVÉ, LOGGÉ dans la provenance de chaque
session, et re-jugeable — si P3 rend un transport ≠ 1, l'ordre des étages fait
partie des pièces attribuables (c'est pour cela que la provenance le porte).
L'ordre littéral de la spec §2.3 est remplacé par entrée de correction explicite
(bloc REVUE de la spec), jamais réécrit.

## Ce qui se construit — et rien d'autre

1. **NOUVEAU `src/arcC_rendu.py`** — le module R1, numpy pur, docstring citant
   §A37 :
   - `reechantillonne_bilineaire(a, taille_px)` : 64×64 → taille_px×taille_px,
     convention **centres de pixels** (demi-pixel), écrite en toutes lettres dans
     le docstring ; `taille_px == H` ⇒ identité **bit-exacte** (court-circuit
     explicite) ;
   - `srgb_encode(y)` : la formule standard écrite dans le code
     (y ≤ 0.0031308 → 12.92·y ; sinon 1.055·y^(1/2.4) − 0.055) ;
   - `quantifie_uint8(v)` : `np.rint`, une seule fois, en bout de chaîne ;
   - `rendu_r1(a, taille_px)` : la composition des étages dans l'ordre tranché
     ci-dessus, l'étage `Y = A` nommé par un commentaire (c'est lui que
     l'incrément R2 post-P3 remplacerait) ;
   - **GARDE FAIL-LOUD** : `a` non-64×64, non-fini, ou hors [0,1] ⇒ `ValueError`
     clair — jamais un clip silencieux (l'albédo est dans [0,1) par construction ;
     une valeur hors bande est un BUG amont, pas une donnée).
2. **EMPREINTE-VERROU (tradition kernels)** : `tests/test_arcC_rendu.py` grave
   `SHA256_ATTENDU` / `LONGUEUR_ATTENDUE` de la SOURCE de `src/arcC_rendu.py`
   (fichier lu en octets), commentaire datant la gravure et pointant §A37 —
   « P3 date son verdict sur R1 » exige que R1 soit verrouillé.
3. **`scripts/run_arcC_session.py`** — le branchement :
   - `affiche_r1(...)` À CÔTÉ de l'`affiche` viridis existant (l.102-107),
     même point d'insertion, **niveaux de gris** (une seule valeur par pixel,
     répliquée RGB si le backend l'exige), `origin="lower"` conservé,
     `interpolation="nearest"` côté matplotlib (le rééchantillonnage est DÉJÀ
     fait par R1 — l'affichage n'a plus le droit d'interpoler ; documenter que la
     fenêtre présente l'image À SA TAILLE, sans zoom de figure) ;
   - **sélecteur OBLIGATOIRE `--rendu {viridis,r1}`, SANS défaut** : chaque
     session future choisit son chemin consciemment ; viridis reste disponible
     (contrôle A/A, comparaisons historiques) ;
   - **PROVENANCE LOGGÉE** dans l'en-tête JSONL de session : chemin de rendu,
     sha256 de `arcC_rendu.py`, ordre des étages, valeurs de calibration de la
     session (ppd, taille_px réalisée, report `observation_cellule_pic_csf` —
     déjà obligatoire, y adjoindre le rendu).
4. **`tests/test_arcC_rendu.py`** — les falsifieurs de la spec §5, chacun nommant
   son consommateur en docstring :
   - **déterminisme** : deux appels, mêmes octets ;
   - **test à blanc** : `A_cand ≡ A_ref` ⇒ images bit-identiques (l'ABX serait au
     hasard par construction) ;
   - **identité de calibration** : `taille_px == 64` ⇒ sortie ≡ quantifiée de
     l'entrée encodée (le rééchantillonneur s'efface bit-exact) ;
   - **préservation de la porteuse** : une sinusoïde à k cycles/domaine
     rééchantillonnée reste à k cycles/domaine (pic FFT inchangé en fréquence
     NORMALISÉE ; tolérance nommée sur l'amplitude — le bilinéaire atténue, il ne
     déplace pas) ;
   - **monotonie de chaîne** : A₁ ≤ A₂ partout ⇒ rendu(A₁) ≤ rendu(A₂) partout
     (champs seedés, plusieurs tirages) ;
   - **valeurs connues sRGB** : 0.0 → 0 ; 1.0 → 255 ; + deux points intermédiaires
     calculés depuis la formule et codés en dur ;
   - **gardes fail-loud** : hors [0,1], NaN, mauvaise forme ⇒ `ValueError` ;
   - **empreinte** (point 2).

## Garde-fous (ils sont le contrat)

- **NE PAS TOUCHER** : `src/albedo.py` (Δχ, albedo — le pin vit là),
  `src/arcC_calibration.py` (contrat §C7 ; ses TESTS peuvent être étendus, pas
  ses fonctions), `src/arcC_abx.py` (logique pure), tout `src/f1_gpu/`, tout
  seuil, toute bande, tout combinateur.
- **AUCUNE MESURE** : pas de chrono (P2 est un pré-enregistrement séparé), pas de
  session humaine (P3 aura le sien), pas de Δχ recalculé.
- R1 reste **sans état** : aucune variable de module, aucun cache, aucune
  dépendance à l'horloge — les tests de déterminisme doivent le prouver, pas le
  supposer.
- Lint : `.venv/bin/ruff check` (ruff est INSTALLÉ — brief du 25/07) ; suite
  complète `pytest tests/` verte **sur iluin-tworings3** avant remontée.
- Commits : `git -c user.name="Claude (Cascade)" -c user.email="noreply@anthropic.com"
  commit` ; **push : Romain uniquement**.

## POINT D'ARRÊT — la mission s'arrête là

Tests verts ⇒ REMONTER à Romain avec : le sha256 gravé de R1, la sortie de la
suite, et UNE image d'exemple par chemin (viridis vs R1, même champ) pour
inspection visuelle — **sans aucune lecture prononcée dessus**. L'achat de P2
(chiffrage) et le pré-enregistrement de P3 (transport du pin, session ABX humaine)
sont des décisions de Romain, jamais un enchaînement.
