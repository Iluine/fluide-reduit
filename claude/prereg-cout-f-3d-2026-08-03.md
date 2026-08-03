# PRÉ-ENREGISTREMENT — LE COÛT DE F EN 3D (2026-08-03)

> **Statut : PRÉ-ENREGISTREMENT, écrit AVANT tout run et AVANT toute ligne de
> code 3D.** Aucune mesure n'a été prise ; aucun kernel 3D n'existe. Le journal
> `pocCascade2phys/PREREGISTRATION.md` fait foi ; ce document raisonne sur lui et
> ne le remplace pas. Ancres sortantes au format §A50 — `fichier:NNN` suivi du
> fragment textuel qui fait foi ; en cas de désaccord, **le texte l'emporte**.
> **Ce document se commit SEUL.** Le code, le driver et l'artefact suivent dans
> des commits distincts — règle née de §A52, dont c'est la première application :
> le prereg et le driver de la tranche étaient nés dans le même commit (`4dda6a6`),
> ce qui rend l'antériorité du protocole invérifiable dans l'histoire git.
> **L'endossement est le commit de Romain ; aucun run avant.**

---

## §0. La question, et la décision qui en dépend

La question est celle que §A51 a laissée ouverte, mot pour mot —
`PREREGISTRATION.md:8254` :

> « **LECTURE TEMPS 3D : INDÉTERMINÉE.** Toutes les ancres de coût (1.685 /
> 0.843 ms) sont mesurées **en 2D** (scénario R2, « 2D mono-observateur, FOV
> 20° »). Un slot 3D a le même nombre de cellules mais un stencil à plus de
> voisins et trois directions de flux. **Le coût par cellule en 3D n'est mesuré
> nulle part.** »

**La décision nommée qui en dépend** (D17 — `PREREGISTRATION.md:6461` « *aucun
diagnostic sans une décision nommée qui en dépend explicitement.* ») :

> **V4 survit-il au passage en 3D, ou la porte 33,3 devient-elle LA décision ?**

Le repli est déjà nommé et n'est pas à inventer — `PREREGISTRATION.md:8183`
« **plus de deux foyers perçus simultanément ⟺ physique à 30 Hz** (`:4217`
« Porte 33,3 = REPLI PRÉ-NOMMÉ ») ». Cette mesure ne prend pas cette décision :
elle produit le chiffre sans lequel elle ne peut pas être prise.

---

## §1. CORRECTION PRÉALABLE — la marge de V4 n'est pas 0,7 %, elle est 13,2 %

**À consigner avant toute autre chose, parce que cette correction change la
taille de la question d'un facteur ~25.**

`PREREGISTRATION.md:8258` (§A51) écrit : « d'autant que V4 passe son gate en 2D
avec **0,7 % de marge** ». Ce chiffre est **exact à sa date et périmé au moment
où §A51 l'a cité**. Il vient de `§A18-lecture-M-a-quater` (19/07, médiane
16.589 ms, marge 0.111 ms). Le **même jour**, `§A23-2b` l'a superséé —
`PREREGISTRATION.md:5215` :

> « | **MORT (médiane > 16.7)** | **NON** | marge **2.199 ms** (contre 0.111
> auparavant) | »

et `PREREGISTRATION.md:5222` : « **VERDICT PRONONCÉ (Romain, 2026-07-19) : V4
SANS MORT.** Le critère MORT-a est pré-enregistré et porte sur la MÉDIANE :
14.490 < 16.7. »

**C'est exactement le troisième état du vérificateur que §A51-7 a lui-même
inventé** — *ancre exacte, contenu superséé* — et §A51 en est la victime dans la
page qui le nomme. La même valeur périmée a été recopiée une seconde fois dans
`claude/prereg-tranche-cout-rendu-2026-08-03.md:52`. Deux propagations, une
seule source : une session qui recopie au lieu de relire (§A41).

**Ce que la correction déplace, et pourquoi il faut s'en méfier plutôt que s'en
réjouir.** Sous 0,7 % de marge, tout facteur 3D supérieur à ~1,006 tuait V4 : la
mesure était réglée d'avance. Sous 13,2 %, la 3D dispose de **2,210 ms**. Le
chiffre corrigé est **favorable** — et la règle de §A52 s'applique dans ce sens
précis : *un chiffre défavorable n'est pas plus sûr qu'un chiffre favorable*.
La correction n'a donc pas été retenue parce qu'elle arrange : elle a été
**vérifiée dans l'artefact** avant d'être écrite (§4), et le verdict qui la porte
est un verdict de Romain, pas une lecture de session.

---

## §2. LA DÉCISION DE PÉRIMÈTRE, PRISE ET ASSUMÉE

Deux voies existaient : (a) mesurer le coût **structurel** avec un F jouet à
stencil 3D représentatif, ce qui donne un **plancher** sans trancher le schéma ;
(b) attendre que le schéma eau 3D soit choisi — `PREREGISTRATION.md:8339` « le
**schéma eau 3D** (dette « 3D » : pression matérialisée ou non, tampons de
flux) », arbitrage explicitement laissé à Romain.

**La voie (a) est prise.** Le fondement n'est pas le confort : c'est que
**l'ancre 2D elle-même est un F jouet**. `src/f1_gpu/substrat_jetable.py:5-6` :
« le choix de F « doit être **représentatif en COÛT** (stencil, c=8 champs), pas
en physique-jeu » ». Le 1,685 ms n'a jamais été le coût d'une physique ; c'est le
coût d'un motif. Mesurer un jouet 3D contre un jouet 2D, c'est **comparer deux
objets de même nature** — et c'est précisément ce qui rend le rapport plus solide
que chacun des deux nombres. Mesurer un F « réel » 3D contre le jetable 2D
confondrait la **dimension** avec le **schéma**, et produirait un chiffre
ininterprétable.

**Ce que la voie (a) coûte, dit d'avance** : elle ne prononce rien sur le schéma
eau 3D, rien sur la pression matérialisée, rien sur les tampons de flux aux
faces. Elle donne un **plancher de dimension**, pas un coût de production.

---

## §3. CE QUE LE F JOUET 3D EST, ET CE QU'IL N'EST PAS

**Ce qu'il EST** — la transposition, axe par axe, du motif E4a déjà mesuré :
padding réfléchissant (miroir + négation de la qdm normale, sur **trois** paires
de faces), désingularisation, pentes minmod, réconciliation positivité,
reconstruction MUSCL, flux HLL dry-aware, transport tangentiel upwind,
corrections de pression hydrostatiques (calculées, nulles à b ≡ 0), divergence,
plancher sec, 2 étages SSP-RK2, réduction CFL payée non consommée, dt figé, f32.

**Les trois seules choses qui changent, et elles sont la question :**

| # | ce qui change | pourquoi c'est la dimension, et non un choix |
|---|---|---|
| 1 | **3 axes de flux** au lieu de 2 | il n'y a pas de 3D à deux directions |
| 2 | **5 champs par système** (h, hu, hv, **hw**, s) au lieu de 4 | on ne fait pas de 3D avec deux composantes d'impulsion ; le troisième moment est intrinsèque |
| 3 | **2 tangentielles par axe** au lieu de 1 | conséquence mécanique de (1)+(2) : chaque face transporte deux composantes tangentielles |

**Ce qu'il N'EST PAS**, écrit avant pour ne pas être plaidé après : ce n'est
**pas** un schéma eau 3D ; ce n'est **pas** de l'incompressible (aucun solve de
Poisson, aucune pression globale) ; ce n'est **pas** une validation numérique
(aucune exigence de fidélité — `(a) ≠ (b)`, le portage FIDÈLE reste non acheté) ;
ce n'est **pas** un coût de production.

**La garde qui interdit de le rendre complaisant** —
`src/f1_gpu/substrat_fusionne.py:8-9` : « PAS changer le motif de coût — **toute
omission de calcul serait un harnais complaisant, interdit** ». Et le précédent
qui nomme la faute exacte à ne pas commettre, `substrat_jetable.py:6-8` : « 8
champs = (h, hu, hv, s)×2 sous stencil complet — **majorant honnête**,
l'alternative "3+5 passifs" **minorerait le coût et fabriquerait un M-a
complaisant** ». Ici la minoration tentante serait de garder 4 champs en 3D (pas
de `hw`) : **elle est interdite d'avance, et par écrit.**

---

## §4. LES ANCRES — LUES PAR MACHINE, JAMAIS RECOPIÉES

Toutes les valeurs ci-dessous sortent de deux artefacts JSON existants, lus par
`json.load`. **Le driver les relira lui-même et échouera bruyamment si elles ont
changé** ; aucune n'est saisie à la main dans le code.

| grandeur | valeur | artefact, chemin dans le JSON |
|---|---|---|
| ancre 2D, slot c=8 | **1,68531 ms** | `outputs/f1/ma_prime.json` → `frame_time.mediane_ms` (5,055920) ÷ `meta.cellule.n_fenetres` (3) |
| ancre 2D, bloc c=4 | **0,84265 ms** | idem ÷ 2 systèmes |
| V4, blocs | **15** (11 slots) | `outputs/f1/ma_quater.json` → `meta.candidat.n_blocs` |
| F 2D de V4 | **12,655 ms** | `ma_quater.json` → `lecture_mecanique.parts.f_batche_ms` |
| médiane frame V4 | **14,49003 ms** | `ma_quater.json` → `frame_time.mediane_ms` |
| **seuil de MORT** | **16,7 ms** (sur la MÉDIANE) | `ma_quater.json` → `lecture_mecanique.mort.seuil_ms` |
| non-F de la frame | **1,83503 ms** | 14,49003 − 12,655, recalculé |

**Trois faits de contrôle, consignés parce qu'ils auraient pu piéger :**

1. **`outputs/` n'est pas versionné** (`.gitignore:5` « `outputs/` »). Ces deux
   artefacts ne sont protégés par rien d'autre que leur présence sur le disque.
   Ils concordent avec le journal (14,490 / 16,7 / 12,655 / 15 blocs), ce qui a
   été vérifié ligne à ligne avant d'écrire ce document. Les artefacts de la
   tranche vivent, eux, dans `claude/lectures/` qui **est** versionné — c'est le
   régime à reconduire ici.
2. **`ma_quater.json` porte 14,490, pas 16,589.** Le fichier est postérieur
   (mtime 2026-07-20 23:03) à la première lecture (19/07). Ce n'est **pas** un
   écrasement d'artefact verdict-grade : c'est le run EPS 1e-2 de `§A23-2b`, dont
   le résultat est gravé et endossé. Vérifié avant de crier au loup.
3. **F reconstruit depuis l'ancre = 12,6398 ms** contre 12,655 gravé, écart
   **−0,12 %** : c'est l'arrondi de l'ancre (le journal écrit « F = ancre V4
   (1.685/0.845) »). Le dénominateur des seuils reste la valeur **gravée**
   12,655 — c'est elle qui a porté le verdict. L'écart est reporté, pas absorbé.

---

## §5. LES GRANDEURS MESURÉES

L'unité de comparaison est le **bloc** — un système sur une fenêtre de
262 144 cellules — parce que c'est l'unité du modèle de coût déjà gravé (V4 =
15 blocs) et parce qu'elle **n'exige aucune hypothèse de linéarité en champs**.
La transposition de taille est gravée, non inventée — `PREREGISTRATION.md:8200` :
« **Transposition 3D exacte : `512² = 64³ = 262 144` cellules** — le slot garde
sa taille, sans hypothèse. »

| # | mesure | contenu |
|---|---|---|
| **M-1** | 2D, S=1, 3 fenêtres 512² | reproduction de l'ancre bloc (attendu 0,843 ms) |
| **M-2** | 2D, S=2, 3 fenêtres 512² | reproduction de l'ancre slot (attendu 1,685 ms) ; témoin de linéarité 2D |
| **M-3** | **3D, S=1, 3 fenêtres 64³** | **la mesure** : coût d'un bloc 3D |
| **M-4** | 3D, S=2, 3 fenêtres 64³ | témoin de linéarité 3D |
| **M-5** | 2D à n ∈ {256, 512, 1024} · 3D à n ∈ {32, 64, 128} | **témoin de taille** : le coût par cellule est-il stable ? teste l'hypothèse « le slot garde sa taille » |
| **M-6** | comptes d'opérations statiques + registres/occupancy des deux kernels | **témoin d'attribution** : distinguer « la 3D coûte ça » de « mon kernel est mauvais » |

**La grandeur de verdict est un rapport INTRA-RUN :**

> **ρ = M-3 / M-1** — le coût d'un bloc quand le monde gagne une dimension.

M-1 et M-3 sont mesurés dans **le même run, sur la même machine, sous le même
chronomètre**. L'ancre archivée du 19/07 ne sert **pas** de dénominateur : elle
sert de **témoin de reproduction** (§7). Ce choix
neutralise d'un coup la dérive machine, le changement de version de cupy et
l'état thermique.

**Chronométrie** : protocole gravé, non réinventé — `src/f1_gpu/chrono.py:3-6`,
série **≥ 300 frames**, warmup exclu, **médiane ET p99**, cuda-Events. C'est
celui de M-a′ ; le déroger le rendrait incomparable. *(Rappel du 03/08 : une
première rédaction de la tranche avait proposé 100 frames / p95, sous le protocole
gravé — dérive rattrapée et consignée.)*

---

## §6. LES BRANCHES, ÉCRITES AVANT LA LECTURE

Seuils calculés **maintenant**, par machine, depuis les artefacts du §4, et
**immuables** :

| seuil | formule | valeur |
|---|---|---|
| **ρ_A** — mort inconditionnelle | `16,7 / 12,655` | **1,3196** |
| **ρ_B** — mort conditionnelle | `(16,7 − 1,83503) / 12,655` | **1,1746** |

> **R-1 — ρ < 1,1746 : V4 SURVIT au passage 3D, au titre du seul stencil.**
> La marge tombe de 2,210 ms à `16,7 − (12,655·ρ + 1,835)`. Ne prononce rien sur
> le `c`, rien sur le rendu, rien sur les autres pieds non mesurés.
>
> **R-2 — 1,1746 ≤ ρ < 1,3196 : MORT CONDITIONNELLE de V4 en 3D.**
> F seul tient, la frame complète ne tient plus — *sous réserve que le non-F 2D
> vaille en 3D*, ce qui n'est pas mesuré ici. Consommateur : la porte 33,3.
>
> **R-3 — ρ ≥ 1,3196 : MORT INCONDITIONNELLE de V4 en 3D.**
> F seul dépasse le seuil ; aucune autre part n'étant négative, aucune
> connaissance du reste de la frame n'est requise. **La porte 33,3 devient LA
> décision**, à prendre par Romain — jamais un enchaînement automatique.

**Le sens de la réserve de R-2 est nommé d'avance, et il est défavorable à V4.**
Le non-F (remontée, transferts, prédiction) est retenu **constant** entre 2D et
3D. C'est l'hypothèse neutre sur le nombre de cellules (identique) — mais la
prédiction et la remontée touchent des **halos**, dont le rapport surface/volume
explose en passant de 512² à 64³ (2 % de cellules de bord en 2D, ~20 % en 3D).
Le non-F 3D est donc vraisemblablement **supérieur** à 1,835 ms, ce qui rend
ρ_B un **majorant généreux** : si V4 meurt à 1,1746, il meurt plus fort en
vérité. Généreux **envers V4**, c'est-à-dire dans le sens qui ne fabrique pas
la mort.

**Sensibilité déclarée d'avance** : sous le régime `EPS_DETAIL = 1e-4`
(antérieur, superséé par §A23), le non-F valait 3,934 ms et ρ_B aurait valu
**1,0088**. Le régime en vigueur est 1e-2, endossé ; 1,1746 est le critère,
1,0088 est **reporté comme sensibilité**, jamais substitué.

---

## §7. LES CRITÈRES D'INDÉTERMINATION — vérifiés aussi sur le cas favorable

Règle née de §A52 : *un critère d'indétermination se vérifie sur le cas FAVORABLE
autant que sur le défavorable, sinon il punit la qualité qu'il contrôle.* Le
témoin de dégénérescence de la tranche était malformé et n'a pas tiré **par
chance** ; ici chaque critère est testé mentalement dans les deux sens avant
d'être adopté.

**(I-1) Reproduction de l'ancre.** Si `|M-2 − 1,685| / 1,685 > 25 %`, le harnais
n'est pas celui qui a produit l'ancre : **INDÉTERMINÉ**, cause nommée, aucun
verdict ni dans un sens ni dans l'autre.
*Vérification sur le cas favorable* : ce critère ne peut pas punir un bon kernel
3D — il ne porte que sur le chemin **2D**, inchangé et intouché. Un M-3 rapide
laisse I-1 muet. ✔

**(I-2) Équivalence de motif 3D.** Si le kernel 3D fusionné ne reproduit pas le
jetable 3D à `rtol = 1e-4, atol = 1e-5` (les tolérances de M-a′, reconduites),
il ne calcule pas ce qu'il déclare : **INDÉTERMINÉ**, et le run n'a pas lieu.
*Vérification sur le cas favorable* : un kernel **rapide** échouerait ce test
s'il était rapide **parce qu'il omet du travail** — c'est exactement l'effet
voulu. Un kernel rapide et correct passe. ✔

**(I-3) Attribution.** Si `ρ_mesuré > 2 × ρ_prédit_par_les_comptes_statiques`
(M-6), l'écart n'est pas l'arithmétique : registres et occupancy sont **reportés
en évidence**, et le résultat est prononcé **AVEC réserve d'implémentation
nommée**. Ce n'est pas une indétermination — un vrai coût de localité 3D est un
vrai coût — mais il ne peut pas être attribué à « la dimension » sans le dire.
*Vérification sur le cas favorable* : si `ρ_mesuré < ρ_prédit`, le kernel 3D bat
son propre compte d'opérations — suspect **dans l'autre sens** (omission
possible), donc I-3 se déclenche **symétriquement**, à `ρ_mesuré < 0,5 × ρ_prédit`.
Un critère qui ne surveille qu'un côté fabrique le côté qu'il ne surveille pas. ✔

**(I-4) Stabilité par cellule.** Si M-5 montre que le coût par cellule varie de
plus de 25 % entre `n = 32, 64, 128` en 3D, l'hypothèse « le slot garde sa
taille » (`:8200`) ne tient pas à 64³, et **ρ ne se transporte pas** au budget :
la lecture est **INDÉTERMINÉE** et la cause est le choix de `n_fov` en 3D, pas
la dimension.
*Vérification sur le cas favorable* : une instabilité **favorable** (64³ moins
cher par cellule que 128³) déclenche aussi I-4 — sans quoi on encaisserait une
aubaine de taille en la nommant « dimension ». ✔

---

## §8. GARDES ANTI-FABRICATION

1. **Aucun seuil ne bouge.** 16,7 ms, `rtol/atol` de M-a′, ≥ 300 frames : tous
   importés, aucun redéfini. Un seuil manqué est un RÉSULTAT.
2. **Les kernels 2D restent INTOUCHÉS.** `substrat_fusionne.py` et
   `substrat_jetable.py` ne sont pas modifiés d'une ligne ; le 3D vit dans des
   fichiers neufs. Vérification : `git diff` vide sur ces deux fichiers, contrôlé
   par le driver via l'empreinte de `_SOURCE` (motif déjà employé par M-a-quater
   — `meta.protocole.kernels` « kernels F et L3 … INTOUCHÉS (git diff +
   empreintes `_SOURCE` / `_SOURCE_L3`) »).
3. **Pas d'`assert` nu dans le chemin verdict** : exceptions explicites
   (disparition sous `python -O`).
4. **Fail-loud** : tout `else` de combinateur sort en INDÉTERMINÉ, jamais en
   PASS/FAIL par défaut.
5. **La leçon du run 1 de la tranche, transposée.** Le run 1 a mesuré la lenteur
   de ma boucle Python et aurait prononcé un verdict **pessimiste fabriqué**.
   Ici la faute symétrique serait de comparer un **fusionné 2D** à un **naïf 3D**.
   Interdit par construction : ρ ne se lit **que** entre deux kernels fusionnés
   de la même famille (un thread par cellule, halo relu, flux recalculés). Le
   jetable 3D existe pour I-2 uniquement, et **ne porte aucun chiffre de verdict**.
6. **L'artefact est un fichier**, écrit dans `claude/lectures/` (versionné). Une
   sortie de console n'est pas un artefact (§A52) — l'entrée de journal ne citera
   que des valeurs relues dans le JSON.

---

## §9. CE QUE CETTE MESURE NE PRONONCERA PAS

- **Rien sur le `c` 3D.** §A51 donne `c_fin 3D = 7,5 / c_grossier = 7,0`
  (`:8042`) contre 8 / 4 en 2D : **la dégressivité s'est effondrée**. C'est un
  **second multiplicateur**, indépendant de ρ, qui pèse potentiellement plus
  lourd que lui. Il n'est pas mesuré ici parce qu'il n'est pas mesurable : il
  dépend du **schéma eau 3D**, arbitrage de Romain. Il est **nommé**, chiffré
  nulle part, et **ne sera pas fondu dans ρ**. *(Le `c` de §A51 compte des
  éq-f32 **stockés** ; le `c` du modèle de coût compte des champs **traités par
  le stencil**. Les deux ne sont pas la même grandeur — les confondre serait une
  faute d'unité, et ce document ne les additionne pas.)*
- **Rien sur le rendu**, ni sur l'ombrage, ni sur le régime mobile, ni sur le
  streaming (facteur 5,8 toujours non mesuré).
- **Rien sur la VRAM 3D** : la résidence a été lue au papier par §A51 (B1 des
  deux côtés) et n'est pas rouverte.
- **Aucun verdict de gate.** Si R-3 tombe, ce document produit un **fait**, pas
  une décision : le repli 33,3 appartient à Romain.

---

## §10. CE QUI SERA DÛ APRÈS

- Le second multiplicateur (`c` 3D), qui attend le schéma eau 3D.
- Le non-F en 3D (halos, remontée, prédiction) — la réserve de R-2.
- Le vérificateur d'ancres de §A50, augmenté de l'état `fait périmé` : **§1 de ce
  document en est la troisième occurrence en deux jours**. Son consommateur est
  désormais nommé et concret.

---

*Document rédigé par la session Claude, AVANT tout run et AVANT toute ligne de
code 3D. L'endossement est le commit de Romain.*
