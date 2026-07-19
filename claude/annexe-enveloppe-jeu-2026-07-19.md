# Annexe — Enveloppe-jeu post-MORT-a (2026-07-19)

> Objet : document d'ENTRÉE de la séance de re-épinglage du quadruplet §6
> (SPEC-FOVEA-Z, suite au verdict §A15-lecture-M-a′). Calculs d'enveloppe sur ancre
> MESURÉE — pas des verdicts, pas des promesses. Étiquetage maison : [MESURÉ] /
> [TRANSPOSITION-HYPOTHÈSE] / [NON-ANCRÉ]. Contexte discussion Romain 2026-07-19 :
> « le mur rencontré n'est peut-être qu'un muret pour notre cas d'usage » — cette
> annexe chiffre le muret ET nomme ses pieds non mesurés.

## Ancre mesurée

**6.43 ns/cellule/frame** à c=8 (fusion compétente, 3050 Ti, 2D) — 5.056 ms /
786 432 cellules `[MESURÉ : M-a′, §A15-lecture-M-a′]`. Budget frame 16.7 ms (T2) ⇒
**~2.60 M cellules fines actives** (c=8) ; **~5.19 M** à c=4 (linéarité en systèmes :
le kernel calcule 2 systèmes indépendants — c=4 = 1 système, ÷2 structurel
`[MESURÉ-par-structure, à confirmer d'un run]`).

## Structure de coût du cône d'observation

FOV θ, un niveau de LOD par doublement de distance : chaque anneau [r, 2r] contient
~n² cellules, n = cellules en travers de l'arc. Coût total ≈ **n² × J** (J =
profondeur de raffinement sous h0). **La profondeur est LINÉAIRE (pas chère) ; la
largeur angulaire est QUADRATIQUE (chère).** C'est la forme quantitative de
l'intuition consignée au verdict : raffiner à des emplacements limités.

## Enveloppe 1920 / 3050 Ti (l'objectif : GPU minimal)

Scénario : monde h0 = 500 000 cellules (~707 de côté, 2D), 1 observateur, FOV 20°,
cap dur 1 fenêtre/niveau. Écran 1920 px sur les 20° ⇒ px/cellule = 1920/n.

| n (arc) | px/cellule | J (c=8) | finest (c=8) | J (c=4) | finest (c=4) |
|---|---|---|---|---|---|
| 512 | 3.75 | ~9–10 | **~h0/512** | ~19 | excédentaire* |
| 740 | 2.6 | ~4 | h0/16 | ~9 | h0/512 |
| 1920 | 1.0 | 0.7 | impossible | 1.4 | impossible |

\* excédentaire = la profondeur permise dépasse toute échelle utile ; borne = usage,
pas budget. Lecture : à n=512, le monde 707×h0 est entièrement couvert par le cône
et le plus fin ≈ h0/512 s'atteint à ~3 cellules-h0 de l'observateur (h0 = 1 m ⇒
~2 mm). Pixel-perfect sur tout le cône : impossible — on raffine des emplacements,
jamais le cône.

## Enveloppe 4K (3840 px sur les 20°)

Même px/cellule ⇒ n double ⇒ **coût par niveau ×4**.

- **Sur 3050 Ti** : n=1024 ⇒ J≈2 (c=8) / 4 (c=4) ⇒ finest h0/4 à h0/16 — **le muret
  redevient un MUR**. Cohérent avec l'objectif déclaré : la 3050 Ti est le GPU
  minimal pour 1920, pas une machine 4K.
- **Sur GPU moderne ~×4–5 en bande passante** `[TRANSPOSITION-HYPOTHÈSE : scaling
  premier-ordre en bande passante mémoire ; le kernel M-a′ (1 thread/cellule,
  halo ±2 sans shared memory) est vraisemblablement borné par les lectures
  halo-redondantes — la re-mesure M-a′ sur machine cible coûte des MINUTES, le
  harnais existe]` : budget ~10–13 M cellules ⇒ n=1024 : J≈10–12 ⇒ finest
  ~h0/1024–4096. **L'enveloppe 4K sur GPU ×4–5 ≈ l'enveloppe 1920 sur 3050 Ti** :
  le couple (résolution d'affichage, GPU) se compense en n².
- Le rendu 4K se paie AUSSI hors de ce budget (T2 : physique seule) `[NON compté]`.

## Les pieds non mesurés du muret (à garder en face)

1. **Niveau 0 CPU vivant à 500k cellules** — le harnais en mesurait 65k : ×7.6 non
   mesuré sur le chemin CPU de la frame.
2. **La 3D cube tout** — mémoire déjà bornée (~64³ fine-fovéa, note VRAM) ; le
   CALCUL 3D n'a aucune ancre.
3. **Zéro fenêtre d'énergie comptée** — l'architecture raffine là où l'énergie
   arrive (magie/transducteur, eau, impacts) : le budget d'emplacements est une
   ressource de GAMEPLAY à router, pas une constante. Chaque effet raffiné se paie
   sur les ~10 emplacements.
4. **r_fovea non pinné** — l'arbitre du troc n/J ; quadratique en n : si le pin
   exige plus fin angulairement, l'enveloppe bouge fort (§7, garde reconduite).

## Ce que cette annexe impose à la séance de re-épinglage

1. Épingler un quadruplet-jeu CANDIDAT : (c, n d'arc, J, politique de fenêtres
   d'énergie — réserve explicite) + machine(s) cible(s) nommée(s).
2. Pré-enregistrer **M-a-ter** : re-run du scan paramétrique à cette enveloppe
   (harnais + kernel existants — coût : minutes), critère de mort pré-écrit.
3. La spec ne re-fait foi qu'après M-a-ter SANS MORT (+ F0-cloud lu, + M-b re-scopé
   en tranche-2 sur l'enveloppe re-épinglée).
