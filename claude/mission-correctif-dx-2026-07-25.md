# ORDRE DE MISSION — correctif Δx du kernel fidèle (D19-c, endossé le 2026-07-25)

> **Pour : session Claude Code, sur la machine de Romain (iluin-tworings3).**
> **Sources** : `claude/verif-echelle-grossier-2026-07-25.md` (le fait),
> `pocCascade2phys/PREREGISTRATION.md` §A33-CORRECTION + résultat (les décisions),
> `tests/test_portage_rhs_o2.py` (le falsificateur, déjà exécuté : CONFIRMÉ —
> err(Δx=1) = 3.297e-05, err(Δx=2) = 5.000e-01).

## Le fait, en deux lignes

Le kernel `_SOURCE_FIDELE` ne divise pas sa divergence par la maille : il est `_rhs_o2`
à Δx = 1 exactement. Employé pour le GROSSIER de T2 (physiquement Δx = 2), il le fait
avancer à ~2× la vitesse physique du fin, en lockstep. Confirmé par exécution.

## Le correctif ENDOSSÉ — et rien d'autre

**Le kernel devient Δx-conscient, comme sa référence.** Concrètement :

1. **`src/f1_gpu/substrat_fidele.py`** — dans `_CORPS_FIDELE` :
   - ajouter `const float inv_dx` à la signature du kernel (après `dt`, avant
     `w_base` — ordre à ta main, mais IDENTIQUE partout) ;
   - appliquer à l'assemblage : `Lh = -(dhx + dhy) * inv_dx;` idem `Lhu`, `Lhv`.
     (La référence divise chaque axe par son pas ; la grille du projet est carrée,
     dx = dy, un seul paramètre suffit — le nommer `inv_dx` et le documenter ainsi.)
   - `pas_f_fidele(...)` : nouvel argument `dx_maille: float = 1.0`, passé
     `np.float32(1.0 / dx_maille)` aux DEUX étages. Défaut 1.0 ⇒ tous les usages
     mono-niveau existants (témoin, T1, régime) sont inchangés par construction.
2. **`src/f1_gpu/fovea_2niveaux.py`** — `_etage(...)` prend `inv_dx` et le transmet ;
   `pas_deux_niveaux(...)` appelle le GROSSIER avec `dx_maille = float(DECIMATION)`
   (= 2.0) et la FOVÉA avec `dx_maille = 1.0`, aux QUATRE appels d'étage.
3. **`src/f1_gpu/production_fidele.py`** — mettre les appels en cohérence (grossier
   à 2.0, fin à 1.0). **NE PAS toucher `reduction_cfl_fidele`** : décision endossée —
   la CFL reste calculée à Δx = 1, CONSERVATRICE pour le grossier (dt plus petit que sa
   limite, jamais faux). Mettre à jour le docstring du choix 3 pour dire ce que le code
   FAIT désormais (le mensonge prose/code est ce qui a caché le défaut).
4. **Empreinte** : le verrou parlera — c'est son rôle. Re-graver dans
   `tests/test_f1_substrat_fidele.py` : `SHA256_ATTENDU`, `LONGUEUR_ATTENDUE`
   (recalculables : `sha256(_SOURCE_FIDELE.encode())`), avec un commentaire datant la
   re-gravure et pointant §A33-CORRECTION. Vérifier au grep si l'ancienne empreinte
   `6dd207ca…` est citée ailleurs (`tests/test_f1_mb_t2.py`, `test_f1_fovea_2niveaux.py`,
   `test_f1_temoin_fidele.py`, `test_f1_regime_fidele.py`, `test_f1_mb_t1.py`,
   docstring de `fovea_2niveaux.py`) et mettre à jour CHAQUE citation.
5. **Étendre `tests/test_portage_rhs_o2.py`** d'un troisième test : le kernel avec
   `inv_dx = 0.5` reproduit `_rhs_o2(dx=2)` (mêmes tolérances que l'existant). Les deux
   tests existants doivent rester verts tels quels (défaut 1.0).

## Garde-fous (ils sont le contrat)

- **JAMAIS un ajustement de dt, JAMAIS la mesure, AUCUN seuil, AUCUNE bande.**
- Ne toucher NI `_SOURCE` (fusionné) NI L3 NI EXNER — leurs empreintes restent.
- Le préfixe `_PREFIXE_BASE` reste extrait du figé, intouché
  (`test_prefixe_b_agnostique_vient_du_fusionne` doit passer sans modification).
- Suite complète verte sur la machine de Romain :
  `pytest tests/` — en particulier `test_portage_rhs_o2.py` (3 tests),
  `test_f1_substrat_fidele.py` (C-property terrain réel : insensible à l'échelle de L,
  doit passer identique), `test_f1_fovea_2niveaux.py`, `test_f1_temoin_fidele.py`.
- Commits : `git -c user.name="Claude (Cascade)" -c user.email="noreply@anthropic.com"
  commit` ; **push : Romain uniquement**.

## POINT D'ARRÊT — la mission s'arrête là

Tests verts ⇒ REMONTER à Romain. **Le re-run T2 (endossé : inchangé par ailleurs —
mêmes seuils, mêmes seeds, mêmes bandes, cellule §A15 intacte) est LANCÉ PAR ROMAIN,
pas par la session.** Le verdict du re-run remonte avant toute suite : D14–D18 se
reposent dessus. Aucun enchaînement automatique.
