# PRÉ-ENREGISTREMENT — TRANCHE, coût du gather-plancher et résidence VRAM (2026-08-03)

> **Statut : PRÉ-ENREGISTREMENT, écrit AVANT tout run.** Aucune mesure n'a été
> prise. Le journal `pocCascade2phys/PREREGISTRATION.md` fait foi ; ce document
> raisonne sur lui et ne le remplace pas. Ancres sortantes au format §A50 —
> `fichier:NNN` suivi du fragment textuel qui fait foi.
> **L'endossement est le commit de Romain ; aucun run avant.**

---

## §0. Pourquoi ce document existe avant le run

La tranche produit **le premier chiffre du programme qui ne vienne pas du
papier**. Un chiffre est verdict-grade, donc ce qu'il voudra dire s'écrit
**avant** de le connaître. La garde qui commande ce geste est déjà au journal —
`PREREGISTRATION.md:1032-1033` :

> « `S_eff` **DÉPEND de `r_fovea`, perceptuel NON-PINNÉ** → hérite de
> l'arbitraire JND/fovéa […] **Choisir `r_fovea` pour que le budget passe =
> fabriquer le verdict.** »

Transposée ici : choisir la configuration, la taille d'écran ou le découpage du
chronomètre **après** avoir vu le résultat serait la même faute. Tout est fixé
ci-dessous.

---

## §1. La machine — mesurée, pas déclarée

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop |
| pilote | 570.211.01 · cupy 14.1.1 |
| **VRAM totale** | **3 781 Mo** (`cudaMemGetInfo`) |
| VRAM libre au repos | 3 672 Mo |

C'est la machine cible : `claude/annexe-enveloppe-jeu-2026-07-19.md:48-49` « la
3050 Ti est le GPU minimal pour 1920, pas une machine 4K ».

**FAIT CONSIGNÉ, qui corrige §A51 à la marge.** La séance-table a lu l'enveloppe
contre **4 Go**. Le budget réel est **3 781 Mo**, soit **7,7 % de moins**.
Aucune branche ne bouge — résidence conique ⇒ ~3,48 Go de reste ; résidence
omnidirectionnelle ⇒ ~2,5 Go ; **les deux restent en B1** — mais l'écart est
réel et n'est pas absorbé en silence.

---

## §2. La configuration, épinglée AVANT

**Quadruplet : V4**, seul candidat endossé et gaté — `PREREGISTRATION.md:4214`
« fovéale 9 slots (2 fins c=8 + 7 c=4) + **2 slots énergie c=8** », passé sous
`§A18-lecture-M-a-quater` « **PAS DE MORT, mais V4 AU SEUIL** », marge **0,7 %**
sur un budget de 16,7 ms.

| paramètre | valeur | pourquoi celle-là |
|---|---|---|
| `n_fov` | 512 | V4, scénario R2 |
| `N_niv` | 10 | V4 |
| slots | 9 fovéaux (2×c=8, 7×c=4) + 2 énergie c=8 | V4 **tel quel**, non retouché |
| `n0` | défaut du module | géométrie B3 inchangée |
| backend | **cupy** | `src/f1_gpu/backend.py:4-6` « pour la MESURE, `xp` est cupy EXCLUSIVEMENT […] le backend numpy n'existe que pour tester la LOGIQUE en VM (pas de mesure valide) » |
| champ gathéré | indice 3 (`s`) | `src/f1_gpu/pyramide.py:75` « `CHAMPS_PAR_SYSTEME: int = 4  # (h, hu, hv, s) — E4a` » |

Géométrie confirmée par `PREREGISTRATION.md:5386` « niveau_fin = 9 ⇒ le monde de
V4 au plus fin fait **131 072²** cellules, la fovéa (512) » : niveaux GPU 1..9,
fin = 9, `n0` = 256.

**UN PARAMÈTRE QUE V4 NE FIXE PAS — déclaré comme INFÉRENCE, pas comme gravure.**
Aucun texte ne dit à quels niveaux vivent les **2** slots énergie de V4 :
`:4214` écrit seulement « + 2 slots énergie c=8, placement emboîté ». Le seul
appui est S3, qui portait sur les **3** slots de V2 — `:3967` « **Répartition des
3 slots énergie : 2 au niveau le plus fin + 1 au second** — seule géométrie
compatible B3 (deux offsets par niveau) ».

> **Fixé ici : les 2 slots énergie au niveau LE PLUS FIN (j = 9).** C'est S3 lu
> littéralement — il remplit le plus fin d'abord — réduit de trois à deux slots.
> Compatible B3 : au niveau 9, un slot fovéal (offset nul) plus deux slots
> énergie aux offsets ±`n_fov`.
> **`[INFÉRENCE DE SESSION, NON GRAVÉE]`.** Si Romain veut 1+1 (niveaux 9 et 8),
> il le dit et ce document est réécrit AVANT run — jamais après lecture.

**Aucun autre paramètre n'est un degré de liberté de cette mesure.** V4 est pris
tel qu'il a été gaté ; si la configuration devait changer, ce document serait
réécrit et re-endossé avant run, jamais ajusté après lecture.

**Écran : une LOI, pas un point.** Le gather est mesuré à `cote_px` ∈
**{512, 1024, 1440, 1920}**, en coordonnées du niveau fin, centré sur la fovéa.
Raison : le coût d'un gather plus-proche-voisin est attendu linéaire en nombre
de pixels ; mesurer quatre tailles donne la **loi** et permet de déduire
n'importe quelle géométrie d'écran — dont 1920×1080 — sans avoir à trancher
d'avance la forme de l'écran, qui n'est pas une décision de cette mesure.
`cote_px = 512` sert de **témoin de dégénérescence** : à cette taille la seule
fenêtre fovéale fine couvre l'écran, donc le gather ne traverse qu'un niveau.
Si son coût par pixel ne se distingue pas des grandes tailles, la mesure ne
mesure pas ce qu'on croit et se lit INDÉTERMINÉE.

---

## §3. Les trois mesures

**M-1 — RÉSIDENCE VRAM de la pyramide V4.** Deux nombres, à ne pas confondre :
`PyramideFovea.octets_etat` (part **déterministe** — « octets des buffers d'ÉTAT
GPU préalloués (fenêtres + références) ») et `cudaMemGetInfo` avant/après
construction (le **tout**, mempool et temporaires compris — le docstring de
`octets_etat` prévient : « le mempool ajoute les temporaires recyclés »).
**L'écart entre les deux est lui-même un résultat** : c'est la part de résidence
qu'aucun compte sur papier ne pouvait prédire, et §A51 n'en comptait aucune.

**Confronte la prédiction de §A51**, qui comptait `2·Σb_primitifs` par cellule
active — le facteur 2 correspondant exactement à la paire
`fenetres`/`references`. Prédiction dérivée ici, AVANT run : 4 slots c=8 et
7 slots c=4, à 262 144 cellules par slot :

> `4 × 262 144 × 8 × 4 o × 2  +  7 × 262 144 × 4 × 4 o × 2` = **≈ 126 Mo**

C'est `octets_etat` qui se compare à ce nombre ; `cudaMemGetInfo` s'y compare
**par excès**, et les branches R-A/R-B/R-C ci-dessous portent sur `octets_etat`.

**M-2 — COÛT DU GATHER**, par `cote_px`, en ms/frame. Chronomètre : le chemin
cuda-Events de la maison (`src/f1_gpu/chrono.py`), jamais `perf_counter` sur du
GPU. Le protocole n'est pas choisi ici, il est **gravé** — `src/f1_gpu/chrono.py:3-6`
cite §A15 : « **warmup ≥ 30 frames EXCLU ; série ≥ 300 frames ; médiane ET p99
reportées toutes deux (chiffres inconfortables en évidence)** ». Il s'applique
tel quel.

*(Une première rédaction de ce document écrivait « 100 frames, médiane et p95 » —
**en dessous du protocole gravé sur deux points**. Corrigé avant endossement, et
consigné ici plutôt qu'effacé : affaiblir un protocole en le récrivant de mémoire
est précisément la dérive que le journal traque, et elle ne se voit qu'en allant
lire la source.)*

**M-3 — COÛT DU VERROU** (`_verrouiller_appelant`), mesuré **à part** et
déclaré. Le module l'annonce déjà : « le coût est mesuré et déclaré […] jamais
supposé négligeable ». S'il dépasse **1 %** de M-2, il est soustrait
explicitement et le fait est écrit.

**Ce qui est chronométré et ce qui ne l'est pas.** M-2 mesure le gather SEUL —
ni `frame()`, ni le readout albédo, ni un quelconque étage d'affichage. Les
trois postes sont chronométrés séparément et jamais additionnés en silence.

---

## §4. LES BRANCHES, ÉCRITES AVANT LA LECTURE

**Sur M-1 (résidence) :**
- **R-A — l'écart au prédit est < 15 %** : le modèle de compte de §A51 tient sur
  pièce ; la lecture d'enveloppe de la séance gagne son premier appui mesuré.
- **R-B — écart 15–50 %** : le modèle est du bon ordre mais lui manque un poste ;
  le poste manquant se nomme avant toute correction du modèle.
- **R-C — écart > 50 %** : le compte de §A51 est faux, et les deux branches B1
  doivent être relues. **Ce résultat ne se corrige pas en ajustant le modèle
  après coup** — il se consigne, et la cause se cherche ensuite.

**Sur M-2 (gather), ramené à 1920×1080 par la loi :**
- **G1 — < 1 ms** : le gather-plancher n'est pas un poste budgétaire. La
  question du coût de rendu se déplace entièrement vers ce que cette tranche NE
  mesure pas — ombrage, éclairage, composition réelle.
- **G2 — 1 à 5 ms** : poste notable sur 16,7 ms, **sur une marge V4 de 0,7 %**.
  Conséquence écrite d'avance : le budget T2 doit être rouvert, ou la porte 33,3
  prise ; aucune des deux n'est tranchée par cette mesure.
- **G3 — > 5 ms** : le gather-plancher menace à lui seul le budget. Comme le
  compositeur instrument sera **plus cher par construction** (stencil de
  prédiction), la lecture est que le rendu fovéal à 60 Hz n'est pas atteignable
  sur cette machine sans repli, et `PREREGISTRATION.md:4217` « **Porte 33,3 =
  REPLI PRÉ-NOMMÉ** » devient la branche par défaut.

**Branche transversale — INDÉTERMINÉE.** Si le témoin `cote_px = 512` ne se
distingue pas des grandes tailles (§2), ou si `violations_consignees()` est non
vide (§5), **aucune branche n'est prononcée** et la mesure est INDÉTERMINÉE.

---

## §5. Gardes de lecture

1. **La trace de violation est lue AVANT le chiffre.** Si
   `chemin_de_cout.violations_consignees()` est non vide au moment du prononcé,
   **le chiffre est nul** : un chemin perceptuel a atteint le chemin-de-coût et
   rien ne garantit plus ce qui a été mesuré.
2. **ANTI-SURCLAME (§A48 garde 4), mot pour mot.** Le résultat se dit « **le
   gather-plancher coûte X** », jamais « le rendu coûte X ». La dette
   « mesurable quand le compositeur existera » **reste ouverte** : ce chiffre
   l'avance, il ne la solde pas.
3. **PLANCHER, jamais LE chiffre** (§A48 garde 1). L'opérateur de couture réel
   ajoutera de l'arithmétique. Toute lecture d'enveloppe traite M-2 en
   **plancher + marge**.
4. **Un seuil manqué est un RÉSULTAT.** Aucune configuration, aucune taille
   d'écran, aucun découpage de chronomètre ne sera modifié pour changer la
   branche atteinte. Si G3 tombe, G3 se consigne.
5. **Le run ne réécrit pas ce document.** Les résultats vont dans un artefact
   séparé et daté, puis au journal.

---

## §5-bis. ADDENDUM (2026-08-03, après le run 1, AVANT le run 2)

**Le run 1 est INDÉTERMINÉ, et pour une cause que ce document n'avait pas
prévue.** Artefact : `claude/lectures/tranche-cout-rendu-2026-08-03.json`.

Deux postes sont apparus à l'écriture du driver, non isolés ici : le contrôle
fail-loud de couverture (M-4) et le **témoin de bande passante (M-5)**. Le second
décide de la lecture. À 1920², le gather coûte 20,0 ms quand l'écriture pleine de
l'écran en coûte **0,183 ms** — **ratio 109×**. Le chiffre mesurait la boucle
Python sur 11 slots et l'indexation avancée, **pas la structure creuse**. La
garde 1 de §A48 impose un plancher sur l'ARITHMÉTIQUE ; elle ne dit rien de
l'implémentation, et l'implémentation dominait d'un facteur cent.

**Prononcer G3 sur ce chiffre aurait condamné le rendu fovéal à 60 Hz sur la foi
de la lenteur du code de session.** C'est la faute que ce protocole existe pour
empêcher, dans le sens inverse de l'habituel : ici l'erreur allait vers le
pessimisme. Elle reste une erreur.

**M-2b — GATHER EN UN KERNEL.** Un thread par pixel, zéro copie (pointeurs
device), parcours FIN→GROSSIER avec arrêt au premier slot couvrant.
**L'ARITHMÉTIQUE EST INCHANGÉE** — division entière, plus proche voisin, aucune
interpolation — et l'équivalence avec la voie Python est **testée bit à bit**
(`tests/test_chemin_de_cout.py`), non supposée : si le kernel divergeait, on
aurait changé l'arithmétique en croyant ne changer que l'implémentation.

**Ce qui NE change pas** : les branches G1/G2/G3 de §4 s'appliquent à M-2b
telles quelles, ainsi que toutes les gardes de §5. Seule l'implémentation
mesurée change. **La lecture se fera sur M-2b**, M-2 restant consignée comme
plafond d'implémentation.

**Nouvelle garde de lecture, tirée du run 1** : la branche G n'est prononçable
que si le **ratio M-2b / M-5 est inférieur à 10**. Au-delà, l'implémentation
domine encore et la lecture reste INDÉTERMINÉE — le seuil est fixé ICI, avant de
connaître le résultat.

**Fait à déclarer avec le chiffre** : les métadonnées de slots sont mises en
cache et ne se reconstruisent que si `centre_fin` bouge. À fovéa immobile leur
coût est amorti à zéro ; **il ne l'est pas en régime mobile**, où la fovéa se
déplace à chaque frame. M-2b est donc un plancher de plus, et le dit.

---

## §6. Ce que cette mesure NE fait PAS

Elle ne mesure **aucune fidélité** — la garde 2 du chemin-de-coût existe pour
que ce soit impossible. Elle ne tranche ni la couture fovéa/grossier, ni le
contrat C-strat, ni la porte 33,3. Elle ne dit rien du **coût de rendu** au sens
plein : ombrage, éclairage et composition réelle restent hors périmètre et
`NON MESURÉ`. Elle ne touche ni seuil ni garde, et n'autorise aucun code
au-delà de ce que §A48 autorise déjà.

Document rédigé par la session Claude ; **l'endossement est le commit de
Romain, et aucun run n'a lieu avant.**
