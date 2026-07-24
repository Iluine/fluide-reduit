# Brief de reprise — projet Cascade (2026-07-19, après la mort du gate (iii))

> **Ce brief remplace `brief-reprise-2026-07-18.md`, périmé.** Celui-ci annonçait
> « prochain pas : pré-enregistrement tranche-moteur » — fait, et dix-neuf sections de
> journal écrites depuis (§A15 → §A33).

## Rôle et règles d'engagement

Critique senior adversarial, en français. Romain décide seul (il lit le code, n'écrit
pas le Python) ; Claude Code implémente ; la session lit, challenge, pré-enregistre,
revoit. **JAMAIS d'enchaînement automatique après un verdict ou une lecture** — point
d'arrêt, remonter, décision de Romain.

## Sources de vérité (dans cet ordre — elles portent TOUT l'état)

1. **pocCascade2phys** (local), branche `t1-spring-config` :
   - `PREREGISTRATION.md` — le contrat + journal append-only. **FIN DE FICHIER : §A33**
     (plancher structurel établi). Sections clés de la séquence : §A15 (tranche-moteur),
     §A24 (verdict V4), §A28 (mipmap), §A29–A30 (T1), §A31–A33 (T2, MORT-b, plancher).
   - `SPEC-FOVEA-Z.md` — **9 sections + §2-rev1 (propriété E, champs d'échelle),
     §2-rev2 (LA PYRAMIDE EST UN MIPMAP), §6-rev1, §9 (LA PROJECTION)**.
2. **pocPhysicator** (local), branche `arc-a-manche2-registre` :
   - `claude/lectures/*.lecture.json` — **les lectures versionnées** (`outputs/` est
     gitignoré : ces fichiers sont ce qui survit aux runs).
   - `claude/etat-des-lieux-2026-07-19.md` — MESURÉ / GATÉ / DETTE / **JAMAIS TOUCHÉ**.
   - `claude/veille-sota-2026-07-19.md` — PERSIST (ICML 2026) lu intégralement.
   - Documents de séance : `chiffrage-*.md`, `prereg-*.md`, `diagnostic-oom-t2.md`.

## État en une ligne

**Gate (i) SATISFAIT avec marge** (V4 à 14.490 ms, seuil 16.7, ancre proxy honnête à
0.924) ; **gate (ii) F0-cloud DÉFÉRÉ et déprioritisé** (son consommateur a évaporé) ;
**gate (iii) MORT** — plancher STRUCTUREL : à seuil nul et budget plein, le vivant
fovéal reste un ordre de grandeur au-dessus du pin. **É2 n'est pas satisfaisable par
l'architecture fovéale telle que construite.**

## Prochain pas exact : **LA SÉANCE É2 — PAPIER, PAS MESURE**

Les trois questions gravées en §A33, à traiter paper-grade :
1. **É2 doit-il comparer des ÉTATS ou des PROJECTIONS ?** S'il compare des projections,
   **il est NON MESURABLE tant que la moitié projection n'existe pas** (§A20/§9) — et le
   projet devrait le dire au lieu de mesurer un proxy d'état.
2. **La contamination du halo est-elle un DÉFAUT RÉPARABLE** (recouvrement plus profond,
   fovéa portant son propre bord, bord commis au registre) **ou INHÉRENTE** au couplage
   grossier→fin ?
3. **Le caveat d'échelle** — propagation `c·t ≈ 91 cellules` contre une fovéa de 32 :
   le 64²/32² est le PIRE cas. **Quelle contre-épreuve** le rendrait falsifiable sans
   devenir un quatrième diagnostic déguisé ?

**Engagement pris et tenu : pas de quatrième diagnostic.** Trois l'ont précédée
(décomposition spatiale, EPS = 0, et le diagnostic OOM) ; la suite est du papier.

## Ce qui est MORT, ce qui est VIVANT (ne pas re-litiger)

**MORT** : Option A (quarantaine du non-déterminisme) — MORT-b par-seed, 0.70–0.87
contre pin 0.0733 ; Option B **n'est pas le remède** (règle d'agrégation) ; **EPS n'est
pas le remède** (plancher à seuil nul) ; **`r_fovea` ne rachèterait rien** — l'erreur du
lointain **entre dans la fovéa par le halo** (hypothèse nommée, mécanisme le plus
probable).

**VIVANT et mesuré** : le registre (REGISTRE_FERME, ledger 1.17 Ko/commit, load 63 ms,
compaction = disque et non load) ; la propriété E (arbre emboîté) ; le mipmap ; L3 ;
**la prédiction GPU-side emboîtée est GRATUITE** ; le modèle de coût complet et attribué
terme à terme ; **l'ancre proxy est honnête (0.924)**.

## Faits d'instrument (mesurés — NE PAS re-découvrir)

- Mesures verdict-grade : **iluin-tworings3, terminal natif** (`.venv/bin/python`,
  Python 3.12.3, numpy 2.4.6, **CuPy 14.1.1**). VM Cowork = dev/lecture (kill 45 s),
  elle voit le GPU mais **jamais une mesure**. Le cloud **n'est pas** l'instrument.
- **Quatre empreintes de kernels**, recalculables et vérifiées :
  `_SOURCE` fusionné **e18015f5…** (8223 car.) ; `_SOURCE_L3` **9533a130…** (9635) ;
  `_SOURCE_FIDELE` **6dd207ca…** (9248) ; `_SOURCE_EXNER` **3533fd0b…** (1251).
  *Une empreinte doit être re-calculable par qui l'exige, sinon elle est décorative.*
- **`ruff` n'est pas installé** — ne pas revendiquer « ruff clean » ; contrôle F401 par AST.
- **OOM T2** : RAM hôte, pas VRAM (mempool CuPy 34 Mo). Cause : `_relax_episode` intègre
  jusqu'à `_T_END_RELAX` puis tronque ⇒ 254 000 pas / 25 Go pour un épisode.
  **Correctif : escalade de `t_end` depuis le bas** (jamais de mémo monotone),
  garde structurelle par épisode, **RLIMIT_AS réglé sur VmSize** (l'init CUDA réserve
  6.43 Go d'espace d'adressage pour 0.28 Go résident).
- **À EPS = 0, le cap 10 % mord à la place du seuil** (4096 candidats, 409 places) —
  forcer le budget à H·W, sinon on mesure le cap.
- **Δχ n'est PAS monotone en erreur d'état** : plus d'information peut donner plus
  d'écart (3/18 émissions). Les bras ne se décomposent pas additivement.
- **Restreindre Δχ à une sous-fenêtre est ILLÉGITIME** (base de bandes changée) —
  masquer la DIFFÉRENCE, pas le domaine ; référence identique ⇒ porteuses identiques.
- **6e-5 d'écart d'état ⇒ jusqu'à 0.077 de Δχ** (témoin f32) : l'observable est
  hypersensible, et l'écart d'état n'est pas l'instrument.
- Push GitHub : Romain uniquement. Commits :
  `git -c user.name="Claude (Cascade)" -c user.email="noreply@anthropic.com" commit`.

## Dettes et risques nommés, NON armés

σ_ω (réveil = décision de cadencement) ; `r_fovea` (gaté) ; **l'effondrement du `dt`
dans la queue d'assèchement** (2 ordres — M = 1 n'est pas invalidé, la fenêtre retenue
donne `dt` 0.081, mais la marge 4–12× de T1 fut mesurée sur un état qui ne séchait pas) ;
p99 non résolu ; **la moitié PROJECTION du projet n'existe pas** — ni image rendue, ni
son ; le substrat-jeu réel n'est pas figé (`c = 8` est une enveloppe de travail, et le
fidèle coûte **~2.5× le proxy PAR CHAMP**).

## Discipline (le projet vit d'elle)

Pré-enregistrer AVANT de mesurer ; le moins cher qui peut échouer, d'abord ; **chiffres
inconfortables reportés en évidence** ; portées nommées, jamais surclamées ; INDÉTERMINÉ
est un résultat ; **un échec d'instrument n'est pas un résultat** ; étiquetage MESURÉ /
TRANSPOSITION-HYPOTHÈSE / NON-ANCRÉ ; les états divergents et d'échec se **persistent**
toujours ; **aucun seuil ne se déplace après coup** ; **ne jamais rétrécir la mesure pour
qu'elle entre dans l'instrument** (l'inversion interdite) ; **une bande se pré-enregistre
pour sa config exacte, jamais recomposée après la mesure** ; concéder proprement et
immédiatement — la session critique a été falsifiée plusieurs fois dans cette séquence,
et le journal le porte.
