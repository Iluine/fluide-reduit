# VÉRIFICATION — le niveau grossier de T2 est-il avancé au Δx du fin ?

> **Statut : VÉRIFIÉ PAR LECTURE DE CODE, NON EXÉCUTÉ.** Aucun run, aucun GPU (la VM
> de session n'en a pas). Tout ce qui suit est re-vérifiable en quelques secondes avec
> `grep` par qui l'exige — c'est la condition que le projet impose à ses empreintes, elle
> s'applique à ce document.
>
> Ouverte le 2026-07-25 sur demande de Romain, après le soupçon nommé en §0 de
> `seance-e2-2026-07-24.md`.
>
> **Conclusion : le soupçon est CONFIRMÉ au niveau du code.** Il reste une confirmation
> par exécution, dont le falsificateur le moins cher est donné en §5.

---

## 1. La chaîne de preuve, ligne par ligne

### 1.1 Ce que le module DÉCLARE être

`src/f1_gpu/fovea_2niveaux.py`, docstring :

> « **GROSSIER : le 64² décimé ×2 (32² à dx=2)**, murs réfléchissants au VRAI bord du
> domaine (physique, = le rederive) ; **FIN : une fovéa pleine résolution (dx=1)** »

Le grossier est donc censé représenter **le même domaine physique** (extension 64) avec
une maille deux fois plus grande. Corroboré par la composition : `production_fidele.py`
l.79-80 upsample le grossier `×2` pour former la connaissance 64² comparée à la vérité.

### 1.2 Ce que l'opérateur de RÉFÉRENCE fait

`src/solver_wetdry.py::_rhs_o2` — l'opérateur d'Audusse que le kernel GPU **déclare
porter** (`substrat_fidele.py` : « Ce module EST le portage GPU de l'opérateur bien
équilibré d'Audusse `_rhs_o2` ») :

```python
l.261   dx, dy = grid.dx, grid.dy
l.328   dh    = (m[:, 1:] - m[:, :-1]) / dx
l.329   dhun  = (nl[:, 1:] - nr[:, :-1]) / dx
l.330   dhvt  = (tg[:, 1:] - tg[:, :-1]) / dx
l.336   dh    = (m[1:, :] - m[:-1, :]) / dy
l.343   return -(dxh + dyh), -(dxhun + dyhut), -(dxhvt + dyhvn)
```

**La référence divise par `Δx`.** C'est le schéma volumes finis correct :
`∂q/∂t = −(1/Δx)·(F_{i+1/2} − F_{i−1/2})`.

Son pas de temps aussi : `l.205  return cfl * min(grid.dx, grid.dy) / max(smax, 1e-12)`.

### 1.3 Ce que le kernel GPU fait

`src/f1_gpu/substrat_fidele.py`, `_SOURCE_FIDELE` :

```c
l.113   static ... divergence_axe_b(..., int dy, int dx, int mode)   /* ±1 : STENCIL */
l.204   *dh  = Fh_d - Fh_g;
l.205   *dn  = FnL_d - FnR_g;
l.206   *dt_ = Ft_d - Ft_g;
...
l.226   float Lh  = -(dhx + dhy);
l.234   float un = w_base*q_base[idx] + (1-w_base)*(h + dt*Lh);
```

**Aucune division par un pas d'espace.** Les paramètres nommés `dx`/`dy` sont des
**décalages de stencil** (`0,1` puis `1,0`), pas une maille. La signature du kernel ne
contient **aucune** longueur :

```c
l.209   (const float* q_in, const float* q_base, float* q_out,
         const float* b, const float* halo_q, const float* halo_b,
         const float dt, const float w_base, const int n, const long total,
         const int mode)
```

**Le portage est donc fidèle à `_rhs_o2` UNIQUEMENT pour `Δx = Δy = 1`** — ce qui est le
défaut gelé du projet (`src/sediment.py` l.83 : `_GRID = GridConfig()  # 64x64, dx=dy=1
(défaut FIGÉ)`).

### 1.4 Ce que le driver fait des deux niveaux

`fovea_2niveaux.py::pas_deux_niveaux` appelle **le même kernel** pour le grossier et le
fin, **avec le même `dt`** (quatre appels `_etage`, un seul `dt`).

`production_fidele.py` l.129-131 :

```python
dt = min(float(reduction_cfl_fidele(q_f, cp)),
         float(reduction_cfl_fidele(q_c, cp)))
```

et `substrat_fidele.py::reduction_cfl_fidele` l.271-281 : docstring « dt = 0.4·min(dx,dy)/…,
**dx=dy=1** », corps `return 0.4 / max(smax, 1e-12)`. **Le pas de temps du grossier est
lui aussi calculé à Δx = 1.**

Conséquence secondaire, corroborante : le choix 3 du docstring de `production_fidele`
(« le grossier, **à dx = 2**, a une limite ~2× plus large ») **n'est pas ce que le code
calcule** — le `min` compare deux limites toutes deux à Δx = 1. **L'intention `dx = 2`
existe en prose, et nulle part en code.**

---

## 2. Conséquence quantitative

Au niveau grossier, `Δx` vaut 2 et la division manque :

> **`L_grossier` calculé = 2 × `L_grossier` correct.**
> `dt` étant partagé en lockstep, **le grossier avance à ≈ 2× la vitesse physique du fin.**

Formulation équivalente, plus parlante : **le grossier ne simule pas le monde de 64
d'extension vu grossièrement — il simule un monde de 32 d'extension**, puis son résultat
est upsamplé ×2 et comparé à une vérité de 64 d'extension.

**Aucun réglage de `dt` ne peut compenser.** Corriger par `dt/2` au grossier donnerait le
bon `dt·L` par pas, mais le grossier n'avancerait alors que d'un demi-pas de temps
physique : **le lockstep serait rompu**, ce qui est la structure même de §A31.

**Ordre de grandeur, sur la fenêtre retenue d'un épisode** `[CALCUL, sur t = 48.44 et
c = 1.886 dérivés du journal]` : à la fin d'un épisode le grossier serait à `t ≈ 97`
pendant que le fin est à `48.44` — une désynchronisation de ~48 unités, pendant lesquelles
l'information parcourt ~91 cellules. **Ce n'est pas une correction du second ordre.**

*Atténuation à porter avec le chiffre, dans l'autre sens* `[NON-ANCRÉ]` : le substrat
relaxe. Deux états qui relaxent vers le même équilibre, l'un deux fois plus vite,
convergent — l'écart terminal peut donc être bien plus petit que la désynchronisation
brute. **Cela n'excuse rien** (le comparatif se fait aussi à des émissions intermédiaires,
et l'écart mesuré est justement de grande échelle et corrélé) mais interdit de prédire
l'amplitude à partir du facteur 2. **Je ne prédis pas d'amplitude.**

---

## 3. Pourquoi AUCUNE vérification existante ne pouvait l'attraper

Inventaire exhaustif des tests touchant ce chemin :

| test | ce qu'il vérifie | pourquoi il est aveugle |
|---|---|---|
| `test_c_property_deux_niveaux` | lac au repos ⇒ vitesses parasites < 1e-6 | **flux nuls** : toute mise à l'échelle de `L` préserve le repos |
| `test_c_property_terrain_reel_plein` / `_front_wet_dry` / `test_reste_au_repos_sur_plusieurs_pas` | idem, mono-niveau | idem |
| `test_decimer_moyenne_2x2` | la décimation est une moyenne 2×2 | ne teste que le rééchantillonnage |
| `test_construire_halo_…` | intérieur du halo = la fovéa | ne teste que l'assemblage |
| `test_refresh_inter_etage_matter` | rafraîchi **≠** figé (> 1e-3) | teste une **différence**, jamais une justesse |
| `test_bord_pluggable_meme_operateur_interieur` | mode 1 ≡ mode 0 à halo identique | identité interne |
| `test_empreinte_source_fidele_verrouillee` / `test_kernels_figes_intouches` | sha256 du CUDA | le code n'a pas bougé ≠ le code est juste |
| `test_cfl_calculee_non_triviale` | la CFL n'est pas dégénérée | à Δx = 1 |

**Et le fait le plus lourd** : `grep -n "_rhs_o2" tests/` ne retourne **rien**.
**Le portage GPU n'a JAMAIS été comparé à l'opérateur qu'il déclare porter.** Il a été
verrouillé par empreinte (on ne peut plus le changer sans qu'un test crie) mais jamais
confronté à sa référence. Le verrou protège la stabilité, pas la justesse — la distinction
que le projet fait partout ailleurs.

Toutes les propriétés vérifiées sont **des propriétés d'équilibre ou d'identité**. Le
défaut est **un facteur multiplicatif sur un terme qui vaut zéro à l'équilibre.**

---

## 4. Portée : ce qui serait contaminé, ce qui ne l'est pas

`pas_deux_niveaux` est **neuf en T2** (§A31 : « au 64² du rederive, la pyramide de V4
dégénère […] d'où deux niveaux »). Avant T2, aucun chemin de fidélité n'a jamais fait
tourner un niveau grossier.

**INTACT** (aucun usage du grossier fidèle) : le pin (Arc C) ; §A13 et §A14 (registre,
ledger, fenêtrage) ; le modèle de coût F1 terme à terme ; M-a-quater / V4 (proxy de coût,
champ plat) ; **T1** (mono-niveau 512² réfléchissant) et son ancre 0.924 ; le diagnostic
OOM de §A31 (un fait de RAM, indépendant de la physique) ; la propriété E ; le mipmap ; L3.

**CONTAMINÉ si la lecture tient** : tout ce qui descend de T2 —
**§A32 (MORT-b)**, **§A32-décomposition**, **§A33 (le plancher)**.

**NON contaminé à l'intérieur de T2** : **le bras témoin** (64² plein domaine, **un seul
niveau**, Δx = 1 ⇒ opérateur correct par construction). Il a rendu **0.0015–0.077**.

---

## 5. CONCESSION QUE JE DOIS À ROMAIN — je me suis trompé hier

Le document de séance affirme, en tête et en §4 : « **MORT-b n'est pas rouvert.** »
**Si cette lecture tient, cette phrase est fausse, et je la retire.**

Raisonnement : MORT-b se prononce sur le **bras de production**, qui **est** la structure
à deux niveaux. Son `Δχ` est donc porté par un grossier à mauvaise échelle. Par la gravure
du projet — *« un échec d'instrument n'est pas un résultat »*, *« corriger l'INSTRUMENT,
jamais la mesure »* (§A31-run / §A31-diagnostic) — **le verdict MORT-b serait un échec
d'instrument, pas un résultat**, exactement comme l'OOM avait suspendu le gate (iii) au
lieu de le rendre.

**Ce qui survit quand même**, et il faut le dire aussi précisément :
- **« Option B n'est pas le remède »** — cette conclusion s'appuie sur le **témoin**, qui
  est propre et qui **passe**. `(a)` (arithmétique f32) n'est pas la cause dominante :
  vrai, et non contaminé.
- **Le lieu** : la périphérie porte 22× (jusqu'à 169×) plus d'énergie d'erreur par cellule
  (§A32-décomposition). Le lieu reste juste — et c'est **exactement** là que le défaut
  vivrait.
- **La méthode** de §A32-décomposition (masquer la DIFFÉRENCE, pas le domaine) : intacte.

**Ce qui ne survit pas** : « **Option A est MORTE** ». Sa mort repose sur la traversée du
bras de production, et c'est le chiffre contaminé. Le seul bras propre de T2 — le témoin —
**ne traverse pas** (une seule valeur, 0.077, frôle le pin à 0.0733).

**Je concède également** que §0 de la séance présentait ceci comme un « soupçon » à
arbitrer, tout en maintenant « MORT-b n'est pas rouvert » : les deux étaient
**incompatibles**, et j'ai gardé la formulation confortable. C'est corrigé ici.

---

## 6. Le falsificateur le moins cher — un TEST UNITAIRE, pas une mesure

Il n'est **pas** verdict-grade, il ne chronomètre rien, il ne consomme pas d'épisode :
c'est la comparaison qui manque depuis C1.

> **Sur un même champ décimé `q_c, b_c` (32²), un seul pas, un seul `dt` :**
> 1. `L_ref2 = _rhs_o2(q_c, b_c, GridConfig(dx=2, dy=2), dry_eps)`
> 2. `L_ref1 = _rhs_o2(q_c, b_c, GridConfig(dx=1, dy=1), dry_eps)`
> 3. `L_gpu` = le terme spatial du kernel `_SOURCE_FIDELE` sur le même champ.
>
> **Prédiction si cette lecture tient** : `L_gpu ≡ L_ref1` (à l'arithmétique f32 près) et
> `L_ref1 = 2 · L_ref2` exactement. **Prédiction si elle est fausse** : `L_gpu ≡ L_ref2`.
>
> **Les deux branches sont décidables, sur un seul pas, et l'une des deux est vraie.**

Ce test **devrait exister de toute façon** : c'est la validation du portage C1 contre sa
référence, absente depuis l'origine. Il n'est pas un quatrième diagnostic — **c'est le
test unitaire manquant.** Le construire est utile **quel que soit** son résultat.

**Portée de la vérification exigée** : Δx = 1 et Δx = 2, sur un champ **non trivial**
(flux non nuls — sinon on reproduit l'aveuglement de §3).

---

## 7. Décision pour Romain

**(D19-a)** Le test unitaire de §6 est-il autorisé ? *(Recommandation : oui — il est le
moins cher qui peut échouer, il comble un trou de validation antérieur au problème, et
son coût est un test, pas un run.)*

**(D19-b)** Dans l'attente de son résultat : **suspendre** §A32 (MORT-b), §A32-décomposition
et §A33 par une entrée de **correction append-only** (tradition §A19-CORRECTION), les
FAITS mesurés restant lisibles et les ATTRIBUTIONS étant marquées en suspens ?

**(D19-c)** Si le test confirme : le correctif est **un paramètre de maille passé au
kernel** (le kernel devient `Δx`-conscient, comme sa référence), **jamais** un ajustement
de `dt`, **jamais** une modification de la mesure. Et T2 se re-court **inchangé par
ailleurs** — cellule §A15 intacte, seuils intacts, aucune bande recomposée.

**(D19-d)** Si le test infirme : consigner §0 de la séance comme **hypothèse ÉLIMINÉE**,
ce qui **renforce** §A33 (une cause concurrente de moins), et rouvrir les trois questions
telles qu'elles sont traitées dans le document de séance.

---

## 8. Ce que ce document ne dit pas

- Il **n'a rien exécuté** : c'est une lecture de code, pas une mesure. Elle peut être
  fausse pour une raison que la lecture ne voit pas — c'est pourquoi §6 existe.
- Il **ne prédit aucune amplitude**, et ne prétend pas que corriger le Δx ferait passer
  le gate (iii). Le témoin frôle déjà le pin à 0.077 ; **rien ici n'annonce un PASS.**
- Il **ne touche aucun seuil, ne recompose aucune bande, ne rétrécit aucune cellule.**
- Il **ne remet pas en cause le pin, le registre, le modèle de coût, ni T1** (§4).
