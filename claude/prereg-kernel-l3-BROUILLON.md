# Pré-enregistrement du KERNEL L3 (borne de la remontée) — SOUMIS À ENDOSSEMENT

> **Statut : BROUILLON soumis. AUCUNE ligne de code écrite.** Achat A endossé
> §A18-complément (pocCascade2phys b1a8a13). Ce document fige les choix
> d'implémentation AVANT la première ligne, comme le garde-fou (4) l'exige.
> **Un point demande arbitrage — le §4 (stratégie de compaction).** Le reste
> est déterminé par l'équivalence ou déjà tranché au complément.

## 0. Nature et cap — à relire avant tout

- **Le kernel L3 est un KERNEL NEUF**, dans un fichier neuf. `substrat_fusionne.py`
  n'est pas touché : la borne M-a′ reste INTACTE et citée telle quelle, bloc
  `_SOURCE` à l'empreinte
  `e8fcaad47db0af010728d5c5d878e1629191bfef4e0382fa80aa3cefd9f27f04`.
- **CAP GRAVÉ : L3 est la DERNIÈRE escalade sur la remontée**, comme M-a′ le fut
  sur F. Aucune escalade ultérieure — ni CUDA Graphs, ni exotique, ni « une
  version mieux optimisée » — n'est autorisée après ce build.
- **Cette borne est JETABLE, et c'est dit ici comme dans le code** : elle mesure
  un motif de COÛT, elle n'est pas un composant de moteur. Le portage fidèle
  reste non acheté.
- Portée : **borne de la REMONTÉE SEULE**, config V2 actuelle, **géométrie
  INTOUCHÉE**. Ce n'est pas M-a-quater.

## 1. Ce que l'épilogue calcule (granularité — déterminé, pas au choix)

Le schéma B4 actuel émet **par ÉLÉMENT** : `d = fenêtre − référence`, masque
`|d| ≥ EPS_DETAIL`, puis valeurs f32 et indices u32 des éléments retenus. Un
élément = un champ d'une cellule d'un système d'une fenêtre.

L'épilogue L3 reconduit exactement cette granularité : chaque thread traite sa
cellule et ses **4 champs**, donc **jusqu'à 4 émissions par thread**. Ce n'est
pas un choix libre — c'est ce que l'équivalence avec l'extraction CuPy impose.
EPS_DETAIL reste 1e-4, `[NON-ANCRÉ]`, suspect tranche-2 inchangé.

**Lecture de la référence + écriture prédiquée** (la formulation endossée) :
après avoir calculé son état sortant, le thread lit `ref[idx]` pour chacun de ses
4 champs, calcule `d`, et n'écrit dans les buffers de sortie que si
`fabsf(d) >= EPS_DETAIL`. Le coût de la lecture de référence est donc PAYÉ dans
le kernel — c'est une passe mémoire supplémentaire, et elle doit l'être : le
schéma CuPy la paie aussi.

La mise à jour incrémentale de la référence (`ref += d` sur les éléments retenus)
est également faite dans l'épilogue, prédiquée de la même façon : sans elle, le
schéma ne serait plus incrémental et l'équivalence tomberait.

## 2. Dimensionnement des sorties (déterminé)

Préallocation au **pire cas** (B2, zéro réallocation en série) : tout peut être
émis. Pour la config V2 (17 blocs, 4 champs, 512²) cela fait **17 825 792
éléments**, soit **71.3 Mo de valeurs f32 + 71.3 Mo d'indices u32 = 142.6 Mo**.
Le gate de résidence est à 1.35 Go et V2 mesure 0.311 Go : la marge est
confortable. Aucune autre option n'est honnête — dimensionner en dessous du pire
cas exigerait de savoir d'avance combien sera émis, ce qu'on ne sait pas.

## 3. Transfert et compteur (déjà tranché au complément — reconduit ici)

**Compteur lu à RETARD D'UNE FRAME**, endossé §A18-complément (1). Mise en œuvre
figée : le compteur d'émission vit sur le device ; à chaque frame on lance une
**copie asynchrone** du compteur vers un **tampon hôte pinned**
(`cp.cuda.alloc_pinned_memory`, déjà employé par `transferts.py`), et le
transfert D2H de la frame courante est dimensionné par la valeur **arrivée à la
frame précédente**. Aucun `synchronize()`, aucun `int(tableau_device)` par
frame : la maladie s2 ne rouvre pas.

Conséquences nommées, et elles ne sont pas nulles :
- la taille transférée à la frame *n* est celle de *n−1* — étiquetée
  **[chemin vivant, lag d'une frame, sous-JND]** au complément ;
- si l'émission de *n* dépasse celle de *n−1*, une part des coefficients n'est
  pas transférée cette frame-là. Le schéma B4 étant **incrémental** (les résidus
  s'accumulent et repassent le seuil), rien n'est perdu définitivement — mais le
  driver doit **reporter l'écart** (émis contre transférés) plutôt que de le
  taire ;
- la première frame n'a pas de compteur précédent : tampon initialisé à zéro, la
  frame est dans le warmup (exclu de la série, B6).

## 4. STRATÉGIE DE COMPACTION — **LE POINT QUI DEMANDE ARBITRAGE**

Chaque émission doit réserver une place dans les buffers de sortie. Deux
stratégies, et le choix **détermine ce que la borne signifie**, sans seconde
chance puisque le cap interdit une réécriture.

**Option 1 — `atomicAdd` par émission.** Chaque champ retenu fait son propre
`atomicAdd(compteur, 1)`. Le plus simple à écrire et à prouver. Mais jusqu'à
17.8 millions d'atomics par frame se sérialisent sur une seule adresse : la
contention peut dominer la mesure, et l'on mesurerait alors la file d'attente,
pas le design L3.

**Option 2 — agrégation par warp** (`__ballot_sync` + `__popc`, un seul
`atomicAdd` par warp et par champ, puis répartition des rangs). Le nombre
d'atomics est divisé par 32 au maximum. C'est une technique standard, disponible
dès sm_30 (la machine est sm_86, warpSize 32) — pas de l'exotique.

**Ma recommandation : Option 2**, pour trois raisons.

1. **Le précédent M-a′.** La borne de F n'a pas mesuré le naïf CuPy — jugé non
   représentatif parce qu'il « matérialise des dizaines de temporaires » — mais
   le fusionné, c'est-à-dire l'implémentation raisonnable d'un moteur. L'atomique
   agrégé est à la compaction ce que le fusionné est au stencil : le standard, pas
   l'optimisation exotique.
2. **L'asymétrie du regret.** Le cap interdit une seconde tentative. Si l'Option 1
   est retenue et que la borne échoue, on ne saura jamais si la contention ou le
   design a tué V4 — et il sera interdit de le vérifier. Si l'Option 2 est
   retenue et que la borne échoue, la conclusion est propre : même bien
   implémentée, la remontée L3 ne descend pas.
3. **Ce que la borne doit trancher.** §A18-complément attend une borne dans
   1.0–2.0 ms, dont l'essentiel est la suppression des passes mémoire CuPy
   (≈ 6.5 des 8.523 ms). L'Option 1 risque d'ajouter un coût qui n'appartient à
   aucun des deux termes de la question.

**Coût de l'Option 2 : environ +25 lignes CUDA et +40 de tests** par rapport à
l'Option 1 — le chiffrage de l'achat A (0.8–1.2 séance pour le kernel) les
absorbe sans le faire glisser.

**Ce que je demande d'endosser : l'Option 2, ou son refus explicite au profit de
l'Option 1.** Quel que soit le choix, il devient partie du protocole : un chiffre
décevant se lira « ce design-là, compacté ainsi, coûte tant », jamais « il aurait
fallu mieux l'implémenter ».

## 5. Le deuxième système reste MOUILLÉ (reconduit du complément)

Le majorant E4a est reconduit : l'état de mesure garde ses **deux systèmes
mouillés**, générés comme aujourd'hui. Seule la PRÉDICTION traitera les champs
d'échelle en zéro — et cela relève de l'achat B, pas de cette borne.

**Vérification anti-minoration, incluse dans le build** : un système entièrement
nul est une fenêtre sèche ; toutes les branches d'un warp partent alors du même
côté et l'absence de divergence peut le rendre moins cher. Le build mesurera
donc le coût d'un bloc à système nul contre un bloc mouillé et **reportera
l'écart**. Si l'écart est significatif, il est consigné pour la lecture de
l'achat B — il ne modifie pas la borne A, qui est mesurée mouillée.

## 6. Protocole de mesure de la borne

Deux bras, même session, même état initial, config V2 batchée (17 blocs, 4
lancements/frame, CFL on-device, géométrie intouchée, fovéa **immobile** — c'est
la remontée qu'on isole, pas le déplacement) :

- **Bras REF** : kernel FUSIONNÉ M-a′ (intouché) — re-mesure de F seul. Sert
  aussi de contrôle de dérive machine face aux 14.432 ms gravés.
- **Bras L3** : kernel L3 (F + épilogue d'émission) + compaction + transfert.

**Lecture mécanique** : `remontée_L3 = A_L3 − A_REF`, comparée aux **8.523 ms**
du schéma CuPy et à la **cible ~1.5 ms** — les trois reportées, **sans
interprétation**. Sont reportés en plus : coefficients émis contre transférés
(l'écart du lag), le contrôle de dérive (A_REF contre 14.432), et l'écart
sec/mouillé. Le driver ne prononce rien ; la décision d'ouvrir ou non l'achat B
revient à Romain.

## 7. Verrous d'équivalence (ce qui interdit un harnais complaisant)

1. **État bit-identique** : sur le même entrant, le kernel L3 produit le même
   état que `pas_f_fusionne`. L'épilogue n'a pas le droit de changer F.
2. **Ensemble émis identique** : les coefficients émis par L3 sont exactement
   ceux que l'extraction CuPy actuelle produit — mêmes indices, mêmes valeurs.
   **Comparaison par ENSEMBLE TRIÉ, pas par séquence** : la compaction atomique
   ne garantit aucun ordre. Ce point est nommé ici parce qu'il déroge à la
   comparaison exacte habituelle du dépôt — la dérogation vaut **pour ce chemin
   et nulle part ailleurs**.
3. **Référence mise à jour à l'identique** : après un pas, la référence de L3
   égale celle du schéma CuPy (le schéma reste incrémental).
4. **Seuil respecté** : rien sous EPS_DETAIL n'est émis, tout ce qui est au-dessus
   l'est — testé aux deux bords (seuil infini : émission vide ; seuil nul :
   émission dense).
5. **Comptage exact** : le compteur d'émission égale le nombre d'éléments
   au-dessus du seuil.

## 8. Chiffrage reconduit et fichiers prévus

| Fichier (tous NEUFS) | LOC estimées |
|---|---|
| `src/f1_gpu/substrat_l3.py` (kernel + wrapper + compaction) | ~335 |
| `scripts/run_f1_borne_l3.py` (driver deux bras) | ~300 |
| `tests/test_f1_substrat_l3.py` (les 5 verrous) | ~300 |
| **Total** | **~935** |

Conforme au chiffrage de l'achat A (composants 1 et 4 : ~925 LOC, 1.1–1.7
séance). `substrat_fusionne.py`, `chrono.py`, `substrat_jetable.py`,
`pyramide.py` : **diffs vides**.

---

**POINT D'ARRÊT.** Ce pré-enregistrement est soumis. Le §4 demande un arbitrage
explicite ; les autres sections sont figées et n'attendent qu'un accord global.
Aucune ligne de code ne sera écrite avant endossement.
