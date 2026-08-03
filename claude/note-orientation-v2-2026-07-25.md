# NOTE D'ORIENTATION v2 — le monde comme forêt d'arbres élagués (2026-07-25)

> **Statut : PAPER GRADE — note d'orientation, PAS une spec, PAS un pré-enregistrement.**
> Consigne la discussion d'alignement Romain ↔ session critique du 2026-07-25, tenue
> après §A35 et l'ouverture de l'arc projection (`seance-fidelite-2026-07-25.md`).
> Rien de mesurable n'y est affirmé au-delà des faits déjà gravés au journal.
>
> Étiquetage : **[ENDOSSÉ]** = accord explicite de Romain (2026-07-25) ;
> **[PROPOSITION]** = position de la session, à débattre ; **[OUVERT]** = nommé, non
> tranché ; **[GRAVÉ]** = déjà au journal ; **[PROPRIÉTÉ]** = conséquence mathématique
> de la représentation, à énoncer avec sa portée exacte.

---

## 1. Le but, réaligné — la phrase qui a tout recalé

Un moteur de monde voxel pour un jeu sandbox : un seul état-monde `z`, image et son en
projections déterministes co-égales, **historique persistant non-contradictoire à
compute borné**. **[ENDOSSÉ]** *(Précision de portée, §A45 du 2026-08-03 : « co-égales »
se dit en **JUGEMENT** — même type de juge, mêmes invariants, même ABX de substitution —
et **NON en FALSIFIABILITÉ**. Le corollaire de §A20-1 (« `z` stable ⇒ image stable ⇒
l'image est falsifiable contre l'état ») est une propriété du SOLVE d'équilibre optique ;
§A20-2 a gravé que le solve acoustique est mort à l'arrivée, et une synthèse n'est pas un
solve. Le son n'a jamais eu ce référent-là.)* Le monde doit rester « vrai » au sens de Romain : *tout
doit rester plausible, et rien de ce qui a été vu ne doit se contredire* (la montagne
n'apparaît ni ne disparaît) — **mais les lois physiques n'ont pas besoin d'être
vraies : elles doivent juste sembler l'être**. La roche se brise en morceaux plausibles
sans mécanique de la rupture ; la turbulence a le bon spectre sans la bonne phase ; la
météo peut dériver de la forme du monde et de la température. Le but n'a jamais été de
simuler la vérité ; la vérité f64 est un instrument de harnais (étalon du registre),
pas le référent du monde vivant.

## 2. Les principes endossés

1. **[ENDOSSÉ] Conservation = squelette, plausibilité = chair.** Tout phénomène a le
   droit d'être phénoménologique dans sa FORME, jamais dans son BILAN. Masse, énergie,
   quantité de mouvement : les livres de comptes sont non négociables — c'est eux qui
   empêchent le plausible de dériver en absurde à long horizon (et c'est le cadre
   conservation-structuré déjà gravé comme pilier).
2. **[ENDOSSÉ] `z` est une forêt d'arbres de coefficients ÉLAGUÉS** (formulation de
   Romain, meilleure que l'« escalier » de la session) : les fenêtres de Harten sont la
   transformée en ondelettes du monde ; chaque champ est un arbre élagué SPATIALEMENT
   (la température coupée court partout sauf près du feu). La fovéa, le support
   d'échelle par champ, le masquage, l'énergie : autant de **règles d'élagage** d'un
   même objet. La politique de raffinement EST la politique de croissance des arbres.
3. **[ENDOSSÉ — REFORMULÉ PAR ROMAIN le 25/07 au soir, remplaçant les « trois
   régimes » de la session]** : **il n'y a qu'UNE dynamique — F, le même, tournant à
   la profondeur locale de l'arbre.** La « synthèse » n'est pas un régime permanent ;
   elle se relocalise en deux endroits précis :
   - **la NAISSANCE des branches** : quand l'arbre pousse (la fovéa arrive), les
     nouveaux coefficients reçoivent un **tirage conditionnel semé** (déterministe,
     re-dérivable, contraint par le ledger) — puis F prend le relais, le même F ;
   - **sous les feuilles** : là où l'état n'existe pas, la texture visible appartient
     à la **PROJECTION** (procédurale, conditionnée à l'état des feuilles) — jamais un
     état de plus. C'est la thèse fondatrice appliquée : l'image est un readout de `z`.

   **Condition maintenue par la session (prix nommé de l'unification)** : F unique
   n'est vrai que si **F est scale-aware à deux titres** — **Δx** (géométrie : le bug
   du 25/07 en était le premier barreau) et **fermeture sous-maille** (statistique :
   pour une dynamique non-linéaire, la moyenne du vrai flux ≠ le flux de la moyenne ;
   un F peu profond sans fermeture se trompe SUR LA MOYENNE — transport turbulent,
   dissipation des ressauts — et une moyenne fausse se voit de loin ; cf. la
   « lentille de diffusivité oracle », gravée). La fermeture est un TERME dans F, pas
   un second moteur. **[OUVERT : le modèle de fermeture, à choisir en §4.]**

   **Le falsifieur unifié survit inchangé** : AUCUNE TRANSITION N'EST DÉTECTABLE PAR
   L'OBSERVATEUR — les transitions étant désormais les naissances/élagages de
   branches et la frontière feuille/projection. Le halo de T2 était une frontière de
   croissance jugée contre la vérité pleine ; en v2 elle est jugée perceptuellement.
   τ_dec garde son rôle : il gouverne la liberté du tirage de naissance (ce que le
   tirage peut inventer vs ce que le ledger contraint) et la politique de commit.
4. **[ENDOSSÉ] τ_dec par champ — la frontière éphémère/persistant devient une grandeur
   mesurée.** Pour un substrat chaotique, la phase fine a une durée de vie prédictive
   finie : au-delà de τ_dec, les détails d'une simulation parfaite ne sont pas plus
   « vrais » qu'une synthèse conditionnée au même grossier — **la vérité fine y est
   épistémiquement vide**. C'est la justification théorique du contrat stratifié.
   Politique par champ dérivée de τ_dec vs temps de revisite : eau vive (τ courts) →
   synthèse généreuse ; sédiment/terrain (mémoire longue) → territoire du ledger.
   **Falsifieur pas cher, un par champ** : deux runs à graines fines perturbées, même
   grossier, chronométrer la divergence des détails. Le cadencement des commits sur la
   relaxation (innovation portée, [GRAVÉ] §A13-résultat) retombe ici : committer quand
   la phase s'est décorrélée et qu'il ne reste que le persistant.
5. **[ENDOSSÉ] Stockage : snapshot + queue.** Le monde fin n'existe nulle part en
   définition max — il est défini par **(graine, entrées+commits, F)** et matérialisé à
   la demande. Le ledger porte DEUX choses : les **entrées** (actions joueurs —
   l'imprévisible, enregistré toujours) et les **commits** (résumés supra-JND — pour
   borner le rejeu et figer l'histoire perceptuelle). Sur disque : snapshots
   périodiques de l'arbre élagué (« Z en totalité » = tout ce qui existe, pas la
   définition max) + la queue du ledger depuis le dernier snapshot. **Le snapshot
   matérialise, la queue reste la vérité** — le ledger ne « se vide » pas, ses segments
   anciens s'archivent par décision explicite (rejeu, multi, intégrité sha256 en
   dépendent). Coût par émission : reste O(k) [GRAVÉ, la thèse existentielle] ; le
   snapshot s'amortit en tâche de fond (créneau de la « consolidation onirique »).
   Fait mesuré à porter : sur le harnais, **la compaction achète du disque, pas du
   load** (0.063 s/heure de ledger) — la cadence X du snapshot est un paramètre à
   chiffrer à l'échelle monde, pas à deviner. Résidences : DISQUE = snapshots + ledger
   + graine ; RAM = fenêtre 0 globale + arbres actifs ; VRAM = fovéa + halos.
6. **[ENDOSSÉ] Météo = champs à support grossier natif + canal causal mince.** Brume,
   pluie, vent dérivent d'un état grossier (forme du monde, température — le « z
   réduit ») ; leur apparence fine est une PROJECTION (readout volumétrique), jamais un
   état fin. MAIS la météo doit pouvoir toucher causalement le monde fin (la pluie
   mouille le sédiment, laisse flaques et boue) : un terme source au grossier, descendu
   par le chemin normal — un monde qui ne garde pas trace de ses orages n'est pas vrai
   au sens de §1.

## 3. Propriétés de la représentation — à énoncer exactement

- **[PROPRIÉTÉ] La montagne ne peut pas flipper, par construction** : le raffinement
  est une synthèse de détails qui préserve exactement les coefficients parents
  (moyennes de cellules). Ajouter des coefficients fins (du ledger, de la graine, ou
  de F) ne change jamais ce qui a été vu de loin. La non-contradiction inter-LOD
  statique est un invariant structurel, pas un test.
- **[PROPRIÉTÉ] Masse conservée par construction** : les détails d'une multirésolution
  en moyennes de cellules sont à moyenne nulle — ajouter/retirer des détails
  synthétisés ne peut pas violer la masse au niveau parent. **Portée exacte : vrai
  pour la masse ; PAS automatique pour la quantité de mouvement et l'énergie**
  (non-linéarité) — celles-là gardent des livres de comptes explicites (§2.1).
- **[PROPRIÉTÉ, avec sa limite]** Le seuillage d'ondelettes donne des bornes d'erreur
  PAR PAS (Harten ; Cohen ; Müller — littérature de la multirésolution adaptative).
  Il ne donne PAS la fermeture à long horizon : seuiller∘évoluer ≠ évoluer∘seuiller —
  même non-commutation que le halo. **Les maths donnent le vocabulaire, pas la
  dispense de mesurer** ; la fermeture d'historique reste le problème empirique du
  projet (la manche 1 en est le prototype de preuve).

## 4. Le cœur dur restant — le downscaling conditionnel dynamique

**[OUVERT — le vrai problème de recherche de la v2, en DEUX morceaux jumeaux.]**
(1) **La fermeture sous-maille de F** (§2.3) : le terme qui rend F juste SUR LA
MOYENNE quand il tourne peu profond — sans lui, l'unification « un seul F » produit
des grossiers systématiquement biaisés, visibles de loin. (2) **Le tirage de
naissance** : la rivière regardée de loin puis approchée — les nouveaux détails
doivent être UN tirage plausible dont la moyenne colle à ce qui a été vu PENDANT
l'observation. Littérature à piller pour les deux : LES à sous-maille stochastique,
downscaling stochastique (météo), super-résolution conditionnée par trajectoire
grossière. Le JND donne la tolérance, le ledger donne les conditions aux limites,
l'habitat naturel du problème de fermeture `f_p ∝ L^(-0.78)`. Aucun modèle choisi ici.

## 5. Propositions de la session, NON endossées — à débattre

- **[PROPOSITION] Le pilote de raffinement comme file de priorité budgétée** : chaque
  branche candidate a un prix perceptuel (magnitude prédite des détails × surface JND
  dans sa bande/excentricité) et un coût en flux ; le budget frame sert la file par
  rendement JND-par-FLOP. La fovéa devient un équilibre émergent, la dégradation est
  gracieuse par construction. (Le débat distance/énergie/masquage se dissout dedans —
  mais la prédiction de magnitude est un morceau à spécifier.)
- **[PROPOSITION] La quantification du commis par format** (entiers/point fixe) pour
  un déterminisme machine-indépendant par construction — leçon du fait d'instrument
  CPU/BLAS, jamais instruite.
- **[PROPOSITION] Fovéa en pente, pas en falaise** : un anneau par niveau, chaque
  couture douce à 2× — la leçon du pire-cas 64²/32².

## 6. Provenance — le détournement ONERA, et la carte des emprunts

**[CONTEXTE, Romain 2026-07-25]** Les idées de multirésolution viennent d'un
détournement de travaux ONERA (Le Besnerais et al., *Experimental Fluid Mechanics
goes 3D*, Aerospacelab n°12, 2016 — PIV 3D/FOLKI3D, LocM-CoSaMP, 3DBOS) : le côté
**problèmes inverses / mesure**, pas les solveurs AMR. Transferts identifiés :

- **3DBOS (inversion régularisée) ↔ matérialisation à la demande** : parents exacts =
  contrainte dure (ondelette), commits du ledger = attache aux données, graine/prior =
  régulariseur. Différence assumée : le 3DBOS prend le MAP (lisse) ; Cascade veut **un
  tirage du posterior** (texturé) — « perceptuel, jamais L2 » est ce choix, dit en
  langage d'estimation. Et le λ ne se choisit pas à la L-curve : **le pin JND remplace
  la L-curve** — contribution propre du projet.
- **LocM-CoSaMP (recouvrement parcimonieux) ↔ le registre** : les commits = support +
  coefficients d'un codage parcimonieux de l'histoire ; le seuil JND = critère de
  sparsification. Le vocabulaire du compressed sensing s'applique.
- **FOLKI (coarse-to-fine par warping) ↔ tirage de naissance et dynamique des
  détails** : candidat concret — advecter les coefficients de détail le long du flot
  parent, puis corriger. Convergence indépendante avec *Wavelet Turbulence* (Kim,
  Thürey, James, Turk 2008), éprouvé en production VFX — deux routes, même endroit.
- **Culture GPU de FOLKI** (itérations denses simples > algorithmes malins) : déjà
  dans les kernels fusionnés et la prédiction emboîtée gratuite.

**Ce qui ne transfère PAS** : PIV/BOS reconstruisent un INSTANT depuis des
observations SIMULTANÉES ; Cascade reconstruit une TRAJECTOIRE cohérente avec des
observations PASSÉES sous un opérateur chaotique — la fermeture d'historique n'existe
pas en tomographie, elle reste le problème propre du projet (§4).

## 7. Réserve endossée — six pistes (2026-07-25, après §A36 ; à graver avec la prochaine entrée)

1. **[ENDOSSÉ, avec l'exception météo — Romain] SAUTER AU LIEU DE TOURNER.** Au-delà
   de τ_dec, intégrer et échantillonner sont équivalents ⇒ une région non observée ne
   se simule pas : au retour de l'observateur elle **saute** à un équilibre statistique
   conditionné (invariants de conservation + champs lents + commits). Les champs lents
   passent par des **opérateurs effectifs multi-Δt** (F^n compressé — géomorphologie).
   Coût de F : **O(observation) + O(sauts)**, plus O(monde × temps). **Exception
   endossée : la météo GLOBALE intègre toujours** (fenêtre la plus grossière, coût
   minuscule) — et cette exception est l'**ENABLER** des sauts : la trajectoire météo
   intégrée en continu est la variable de conditionnement qui rend le saut régional
   bien posé (on sait quel forçage la région a « vécu » pendant l'absence).
2. **[ACTÉ — sous-entendu depuis l'origine, désormais explicite] LE LEDGER EST AUSSI
   L'INTERFACE D'AUTEUR.** Contenu autoré = commits à t=0. Design, action joueur et
   histoire physique passent par le même mécanisme, avec les mêmes garanties ;
   l'éditeur de niveau est un client du registre.
3. **[ENDOSSÉ — dette de spec, née du correctif Δx] VERSIONNAGE DE F = CLAUSE DE
   SAUVEGARDE.** Chaque segment de queue porte l'empreinte du F qui fait foi pour son
   rejeu ; une migration de version passe par un snapshot, jamais par un re-rejeu
   silencieux.
4. **[ENDOSSÉ] LE REMBOBINAGE EST GRATUIT.** (snapshot, queue, F déterministe) = rejeu
   depuis tout point : outil de dev majeur (bisecter QUAND un artefact perceptuel est
   apparu), et mécanique de jeu potentielle, par construction.
5. **[ENDOSSÉ — travail papier, post-P3] CHECKLIST DE PLAUSIBILITÉ SANS RÉFÉRENCE.**
   La littérature de physique intuitive : l'humain détecte mal une dynamique fausse
   sans comparaison (la barre est plus basse que craint), mais attrape une classe
   précise de violations — discontinuités de mouvement, brisures de symétrie,
   régularité excessive, bilans faux. Deuxième juge du vivant, complémentaire de
   l'ABX de substitution.
6. **[NOTÉ — dépendent du coût et de la puissance de calcul]** L'**hystérésis
   d'élagage** (deux seuils : naître à ε₁, mourir à ε₂ < ε₁ — sinon scintillement au
   bord) ; et le **statut des PNJ-témoins** (engagent-ils le monde ?) — les deux
   réponses sont défendables, à trancher AVANT v1.1 qui en héritera.

## 8. Ce que cette note NE change PAS

L'arc P0–P3 reste l'instrument et le chemin critique [GRAVÉ §A35] — les décisions
P0-a..d de `seance-fidelite-2026-07-25.md` restent À TRANCHER formellement (cette note
les nourrit, elle ne les remplace pas ; l'endossement de §1–2 rend P0-b quasi mûre
mais elle appartient à Romain). **[PÉRIMÉ LE JOUR MÊME — §A36 FAIT FOI.]** Cette phrase
a été écrite AVANT la gravure de §A36 (2026-07-25), qui cite cette note et tranche
P0-a..d : « **LES GATES DE L'ARC : P0 ✓** » ; **P0-b = C-STRAT, VERSION F-UNIQUE**,
trois clauses. Personne n'a rebouclé, et cette ligne a été lue comme vivante jusqu'au
2026-08-03. Ne rien conclure d'ici sur le statut de P0 : lire §A36. La règle D17 s'applique à tout ce qui précède : chaque
falsifieur nommé ici attend une décision qui en dépend avant de courir. Le registre,
le pin, les morts (§A34), les dettes (chrono T1, σ_ω, p99, le son, 3D, substrat-jeu,
multi-observateur) : inchangés. Le SON reste co-égal en thèse et absent de l'arc —
nommé, pas oublié. **[Précision §A45, 2026-08-03]** « Co-égal » se lit en JUGEMENT, pas
en FALSIFIABILITÉ (cf. §1 ci-dessus). Et §A46 re-scope pour le canal auditif la clause
« éphémère par définition » de §A20-3, classe impulsive seulement — la classe continue
(rivière, vent, pluie) reste portée par τ_dec, qui acquiert par là un SECOND
consommateur, à tolérance temporelle beaucoup plus serrée que le premier.
