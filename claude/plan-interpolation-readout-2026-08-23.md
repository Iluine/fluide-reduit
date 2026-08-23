# Interpolation de readout — plan d'implémentation

> **Pour les exécutants agentiques :** SOUS-SKILL REQUIS — `superpowers:subagent-driven-development`
> (recommandé) ou `superpowers:executing-plans`, tâche par tâche. Les étapes
> utilisent la syntaxe `- [ ]`.

**But :** construire le composant qui produit un champ `s` interpolé entre deux
pas de physique, à côté du gather et jamais à sa place, pour que la porte 33,3
(physique 30 Hz, rendu 60 fps) ait son objet — et que `I` cesse d'être une
grandeur sans référent.

**Architecture :** un noyau affine unique `s_out = (1−α)·s_prev + α·s_cur` dont
les deux modes sont deux valeurs de `α` ; un tampon `s_prev` **hors du tenseur
que `F` consomme** ; une **vue** qui expose au gather les quatre attributs qu'il
touche, avec le `s` interpolé à la place du `s` courant — `chemin_de_cout` reste
identique à l'octet.

**Pile :** Python 3.12, `numpy` (backend B1 injecté ; `cupy` seulement pour la
mesure, qui n'a PAS lieu ici), `pytest`. Binaires via `.venv/bin/` de
`pocPhysicator` **exclusivement**.

**Spec :** `claude/spec-interpolation-readout-2026-08-23.md`
(corps `ee87000`, amendements `c72b9b4` et `f310d03`). **Lire le spec avant ce
plan** : il porte les motifs, ce plan ne porte que les gestes.

---

## Contraintes globales

Elles s'appliquent à **toutes** les tâches, implicitement.

- **Français partout** : code, commentaires, docstrings, messages d'erreur,
  messages de commit. Sans exception.
- **`s_out` et `s_prev` sont ÉPHÉMÈRES** : jamais écrits dans le tenseur que
  `F` consomme, jamais lus par `F`. C'est la garde de fond du `§4` du spec.
- **`chemin_de_cout.py` n'est PAS modifié.** Pas une ligne. Le `§1` du spec en
  dépend, et le verrou (e) le vérifie.
- **`pyramide.py` n'est PAS modifié.**
- **Jamais un `assert`** pour une garde : il disparaît sous `python -O`
  (`§A43`). Toujours une exception explicite.
- **Aucune mesure, aucun `cupy`, aucun chronomètre.** L'interdiction de `§A61`
  tient. Les tests sont de la LOGIQUE, en `numpy`, sur des tailles minuscules.
- **Backend injecté** : `xp` vient de la pyramide (`pyramide.xp`), jamais
  importé en dur.
- **Commits fréquents**, un par tâche, message en français.
- Lancer les tests avec `.venv/bin/pytest`, jamais un `pytest` global.

---

## Structure des fichiers

| fichier | responsabilité |
|---|---|
| **Créer** `src/f1_gpu/interpolation_readout.py` | le noyau affine, le tampon `s_prev`, la vue, la serrure de consommateur |
| **Créer** `tests/test_interpolation_readout.py` | les six verrous du `§7` du spec |
| **Ne pas toucher** `src/f1_gpu/chemin_de_cout.py` | garde `§A48` — sa non-modification EST le contrat |
| **Ne pas toucher** `src/f1_gpu/pyramide.py` | le tenseur de `F` reste hors de portée |

**Rappel de layout, lu et non supposé** — `src/f1_gpu/pyramide.py:385` :
`fenetres[j]` a la forme `(n_slots(j), n_systemes(j), 4, n_fov, n_fov)`, et le
champ `s` est **l'indice 3** (`(h, hu, hv, s)`, `CHAMPS_PAR_SYSTEME = 4`).

---

## Tâche 1 : le noyau affine et le tampon

**Fichiers :**
- Créer : `src/f1_gpu/interpolation_readout.py`
- Test : `tests/test_interpolation_readout.py`

**Interfaces :**
- Consomme : `pyramide.fenetres[j]` (lecture seule), `pyramide.geo.niveaux_gpu`,
  `pyramide.xp`.
- Produit :
  - `INDICE_CHAMP_S: int = 3`
  - `ALPHA_INTERPOLER: float = 0.5`, `ALPHA_EXTRAPOLER: float = 1.5`
  - `class TamponReadout(pyramide)` avec `.capturer(pyramide) -> None` et
    `.melanger(pyramide, alpha) -> VueInterpolee`
  - `.clamps: int` sur la vue

- [ ] **Étape 1 : écrire le test qui échoue — les bornes du noyau**

Dans `tests/test_interpolation_readout.py` :

```python
"""Tests — INTERPOLATION DE READOUT (`src/f1_gpu/interpolation_readout.py`),
les six verrous du §7 du spec vérifiés MÉCANIQUEMENT.

Backend numpy (B1), tailles minuscules (VM) : on teste ici la LOGIQUE et les
GARDES, JAMAIS un coût. `I` ne se mesure pas ici — §A61 l'interdit tant que
l'instrument n'est pas qualifié, et §A63 ne l'a pas qualifié.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.f1_gpu.interpolation_readout import (
    ALPHA_EXTRAPOLER,
    ALPHA_INTERPOLER,
    INDICE_CHAMP_S,
    TamponReadout,
)
from src.f1_gpu.pyramide import GeometriePyramide, PyramideFovea, Slot
from src.f1_gpu.transferts import TransfertComptable

N_FOV = 8
N0 = 16
N_NIV = 4

SLOTS = (
    Slot(niveau=1, c=4, role="fovea"),
    Slot(niveau=2, c=8, role="fovea"),
    Slot(niveau=3, c=8, role="fovea"),
)


def _pyramide():
    geo = GeometriePyramide(n_fov=N_FOV, n_niv=N_NIV, n0=N0, slots=SLOTS)
    transferts = TransfertComptable(np)
    pyr = PyramideFovea(np, geo, transferts, eps_detail=1e30)
    transferts.frame_suivante()
    return pyr


def _peindre_s(pyr, valeur: float) -> None:
    """Peint le champ `s` de toutes les fenêtres avec une valeur constante."""
    for j in pyr.geo.niveaux_gpu:
        pyr.fenetres[j][:, :, INDICE_CHAMP_S, :, :] = float(valeur)


def test_alpha_zero_rend_s_prev():
    """Verrou (a) — borne basse. Propriété du NOYAU : les images exactes
    CONTOURNENT le noyau en production (§3 du spec)."""
    pyr = _pyramide()
    _peindre_s(pyr, 2.0)
    tampon = TamponReadout(pyr)
    tampon.capturer(pyr)          # s_prev <- 2.0
    _peindre_s(pyr, 7.0)          # s_cur  <- 7.0
    vue = tampon.melanger(pyr, 0.0)
    for j in pyr.geo.niveaux_gpu:
        assert np.array_equal(vue.fenetres[j][:, :, 0], np.full_like(
            vue.fenetres[j][:, :, 0], 2.0))


def test_alpha_un_rend_s_cur():
    """Verrou (a) — borne haute."""
    pyr = _pyramide()
    _peindre_s(pyr, 2.0)
    tampon = TamponReadout(pyr)
    tampon.capturer(pyr)
    _peindre_s(pyr, 7.0)
    vue = tampon.melanger(pyr, 1.0)
    for j in pyr.geo.niveaux_gpu:
        assert np.array_equal(vue.fenetres[j][:, :, 0], np.full_like(
            vue.fenetres[j][:, :, 0], 7.0))


def test_melange_a_un_demi():
    """Le mode INTERPOLER : α = ½ entre deux états RÉELS."""
    pyr = _pyramide()
    _peindre_s(pyr, 2.0)
    tampon = TamponReadout(pyr)
    tampon.capturer(pyr)
    _peindre_s(pyr, 8.0)
    vue = tampon.melanger(pyr, ALPHA_INTERPOLER)
    for j in pyr.geo.niveaux_gpu:
        assert np.allclose(vue.fenetres[j][:, :, 0], 5.0)
```

- [ ] **Étape 2 : lancer les tests, vérifier qu'ils ÉCHOUENT**

```bash
cd /home/iluin/project/python/pocPhysicator
.venv/bin/pytest tests/test_interpolation_readout.py -v
```

Attendu : `ImportError` / `ModuleNotFoundError` —
`src.f1_gpu.interpolation_readout` n'existe pas.

- [ ] **Étape 3 : écrire le module minimal**

Créer `src/f1_gpu/interpolation_readout.py` :

```python
"""INTERPOLATION DE READOUT — le composant de la porte 33,3 (§A18).

Spec : `claude/spec-interpolation-readout-2026-08-23.md` (corps `ee87000`,
amendements `c72b9b4`, `f310d03`).

CE MODULE N'EST PAS UN COMPOSITEUR ET NE REMPLACE RIEN. Il ÉCRIT un champ `s`
interpolé ; le gather de `chemin_de_cout` le LIT. Ce dernier reste identique à
l'octet — c'est le §1 du spec, et la garde 1 de §A48 (« l'arithmétique est la
plus bête permise ») continue de décrire exactement ce que fait le gather,
parce que ce module ne touche pas à son arithmétique.

UN SEUL NOYAU, AFFINE — `s_out = (1−α)·s_prev + α·s_cur` :

    mode INTERPOLER   α = ½    entre n et n+1     — deux états RÉELS
    mode EXTRAPOLER   α = 1½   depuis n−1 et n    — état jamais existé

Conséquence GRAVÉE : `I` est INVARIANT AU MODE par construction (même
arithmétique, mêmes octets). Une mesure future ne comparera que la latence et
le perçu, jamais un confondant d'arithmétique. Le verrou (b) des tests le
mécanise — une garde promise dans un document est une garde absente (§A62-bis-2).

`s_out` ET `s_prev` SONT ÉPHÉMÈRES. Jamais écrits dans le tenseur que `F`
consomme, jamais lus par `F`. Une fuite corromprait la comptabilité de
conservation et ne se signalerait par AUCUN symptôme — la signature de §A53
retournée. Trois verrous : STRUCTUREL (tampons séparés, aucune référence en
écriture vers l'état), DE PILE (`_verrouiller_consommateur`), DE TEST.

LES IMAGES EXACTES NE PASSENT PAS PAR ICI — c'est le §3 du spec, et c'est un
CONTRAT D'APPELANT, pas une option. Sur 60 images/s à 30 Hz de physique, une
sur deux tombe sur un pas : celle-là appelle le gather DIRECTEMENT sur la
pyramide, sans toucher ce module. Le motif est la comptabilité gravée
(`prereg-ou-vit-le-rendu:87` porte `30·I`, pas `60·I`) : si les images exactes
traversaient le noyau, elles le paieraient aussi et la comptabilité
contredirait un document endossé. `I` est donc le coût d'UNE image
INTERPOLÉE — 30 par seconde, exactement le `30·` de `Δ`.

AUCUNE MESURE ICI. §A61 interdit toute mesure 3D absolue tant que l'instrument
n'est pas qualifié, et §A63 ne l'a pas qualifié. Ce module se construit ; il ne
se chronomètre pas."""
from __future__ import annotations

import sys

# Champ `s` dans `(h, hu, hv, s)` — `pyramide.py:75`, `CHAMPS_PAR_SYSTEME = 4`.
INDICE_CHAMP_S: int = 3

# Les deux modes, qui sont deux valeurs d'un seul noyau (§2 du spec).
ALPHA_INTERPOLER: float = 0.5
ALPHA_EXTRAPOLER: float = 1.5


class VueInterpolee:
    """Ce que le gather reçoit à la place de la pyramide.

    Elle expose EXACTEMENT les quatre attributs que `gather_chemin_de_cout`
    touche — `xp`, `geo`, `centre_fin`, `fenetres` — et rien d'autre. Le
    gather n'est pas modifié : il reçoit un objet de même surface dont le
    champ lu est le `s` interpolé.

    `fenetres[j]` a la forme `(n_slots(j), n_systemes(j), 1, n_fov, n_fov)` :
    l'axe des champs est un SINGLETON, donc le gather s'appelle avec
    `indice_champ=0`. Garder un vrai tableau contigu — plutôt qu'un proxy qui
    ré-indexerait en Python — évite de remettre dans le chemin le facteur 109
    que `chemin_de_cout` a dû tuer (boucle Python contre kernel)."""

    __slots__ = ("xp", "geo", "centre_fin", "fenetres", "clamps")

    def __init__(self, xp, geo, centre_fin, fenetres, clamps: int):
        self.xp = xp
        self.geo = geo
        self.centre_fin = centre_fin
        self.fenetres = fenetres
        self.clamps = int(clamps)


class TamponReadout:
    """Détient `s_prev` et `s_out`, TOUS DEUX HORS du tenseur de `F`.

    `s_prev` est un état de readout load-bearing — le premier du code. La
    défense est au §4-bis du spec, et elle tient sur deux jambes : ces gardes
    sont scellées au périmètre de l'INSTRUMENT (ce module est du code moteur),
    et `s_prev` ne porte que des COPIES de champs réellement produits par `F` —
    de l'HISTOIRE, pas de la SYNTHÈSE. La synthèse, `s_out`, reste sans état.

    DIMENSIONNEMENT — sur les PLANS `s`, pas sur les SLOTS : un slot `c = 8`
    porte DEUX systèmes (`SYSTEMES_PAR_C`, §A16), donc deux plans. Pour V4,
    11 slots font 15 plans."""

    def __init__(self, pyramide):
        xp = pyramide.xp
        self._s_prev = {}
        self._s_out = {}
        for j in pyramide.geo.niveaux_gpu:
            fen = pyramide.fenetres[j]
            forme = (fen.shape[0], fen.shape[1], 1, fen.shape[3], fen.shape[4])
            self._s_prev[j] = xp.zeros(forme, dtype=fen.dtype)
            self._s_out[j] = xp.zeros(forme, dtype=fen.dtype)

    def capturer(self, pyramide) -> None:
        """`s_prev` ← COPIE du `s` courant. À appeler AVANT le pas de `F`.

        C'est une copie RÉELLE, et ce n'est pas un choix : `F` écrit EN PLACE
        (`pyramide.py:501` — le tampon de sortie passé à `pas_f` EST le tampon
        d'entrée), donc aucun échange de pointeurs ne peut conserver l'état
        précédent. Ce coût est DANS `I` (§3-bis du spec)."""
        for j in pyramide.geo.niveaux_gpu:
            source = pyramide.fenetres[j][:, :, INDICE_CHAMP_S, :, :]
            self._s_prev[j][:, :, 0, :, :] = source

    def melanger(self, pyramide, alpha: float) -> VueInterpolee:
        """`s_out = (1−α)·s_prev + α·s_cur`, puis clamp `s ≥ 0` COMPTÉ.

        Le clamp ne lève JAMAIS : ceci est du code moteur, il tourne. Le seuil
        au-delà duquel un comptage invalide une mesure appartient au prereg de
        la lecture d'orientation, pas ici (§5 du spec)."""
        xp = pyramide.xp
        a = float(alpha)
        clamps = 0
        for j in pyramide.geo.niveaux_gpu:
            s_cur = pyramide.fenetres[j][:, :, INDICE_CHAMP_S, :, :]
            sortie = self._s_out[j][:, :, 0, :, :]
            sortie[...] = (1.0 - a) * self._s_prev[j][:, :, 0, :, :] + a * s_cur
            sous_zero = sortie < 0.0
            clamps += int(xp.count_nonzero(sous_zero))
            sortie[sous_zero] = 0.0
        return VueInterpolee(xp, pyramide.geo, pyramide.centre_fin,
                             self._s_out, clamps)
```

- [ ] **Étape 4 : lancer les tests, vérifier qu'ils PASSENT**

```bash
.venv/bin/pytest tests/test_interpolation_readout.py -v
```

Attendu : **3 passed**.

- [ ] **Étape 5 : commit**

```bash
git add src/f1_gpu/interpolation_readout.py tests/test_interpolation_readout.py
git commit -m "Interpolation de readout — noyau affine et tampon, verrou (a)

Un seul noyau, deux modes qui en sont deux valeurs de alpha (§2 du spec).
s_prev et s_out vivent dans des tampons SÉPARÉS, hors du tenseur que F
consomme — verrou structurel du §4. Dimensionné sur les PLANS s (un slot
c=8 porte deux systèmes, §A16), pas sur les slots.

capturer() prend une COPIE RÉELLE : F écrit EN PLACE (pyramide.py:501),
donc aucun échange de pointeurs ne conserve l'état précédent. Ce coût est
dans I (§3-bis).

Verrou (a) : alpha=0 rend s_prev, alpha=1 rend s_cur, alpha=1/2 mélange.
Bornes testées comme propriété du NOYAU — les images exactes le contournent
en production (§3).

Aucune mesure, aucun cupy : §A61 tient."
```

---

## Tâche 2 : le verrou (b) — l'invariance de `I` au mode, MÉCANISÉE

**Fichiers :**
- Modifier : `tests/test_interpolation_readout.py` (ajout en fin de fichier)

**Interfaces :**
- Consomme : `TamponReadout`, `ALPHA_INTERPOLER`, `ALPHA_EXTRAPOLER` (tâche 1).
- Produit : rien de neuf — ce verrou protège une propriété, il n'ajoute pas
  d'API.

**Pourquoi cette tâche est séparée** : elle ne teste pas un comportement, elle
teste que **le composant n'a qu'un seul chemin d'exécution**.

> ⚠ **CORRIGÉ EN RONDE 1 (voir l'amendement 3 du spec).** La première version
> de ce verrou comparait deux appels au MÊME `α` : elle testait le
> DÉTERMINISME, que tout noyau branché sur `α` satisfait. Deux mutants la
> passaient. Le verrou réel est l'**AFFINITÉ** — trois évaluations
> colinéaires. Le test de déterminisme est conservé sous son vrai nom. Un relecteur peut
légitimement accepter la tâche 1 et rejeter celle-ci.

- [ ] **Étape 1 : écrire le test qui échoue**

Ajouter à `tests/test_interpolation_readout.py` :

```python
def test_les_deux_modes_partagent_un_seul_chemin():
    """Verrou (b) — §2 du spec MÉCANISÉ.

    `I` est annoncé INVARIANT AU MODE par construction. Non vérifié, cet
    énoncé est une garde promise, donc une garde absente (§A62-bis-2). Ici
    on l'exécute : à `α` ÉGAL, les deux appels doivent rendre les MÊMES
    OCTETS — s'il existait un chemin propre à un mode, ils différeraient."""
    pyr = _pyramide()
    _peindre_s(pyr, 3.0)
    tampon = TamponReadout(pyr)
    tampon.capturer(pyr)
    _peindre_s(pyr, 9.0)

    for alpha in (ALPHA_INTERPOLER, ALPHA_EXTRAPOLER):
        premier = {j: np.array(tampon.melanger(pyr, alpha).fenetres[j],
                               copy=True)
                   for j in pyr.geo.niveaux_gpu}
        second = {j: np.array(tampon.melanger(pyr, alpha).fenetres[j],
                              copy=True)
                  for j in pyr.geo.niveaux_gpu}
        for j in pyr.geo.niveaux_gpu:
            assert premier[j].tobytes() == second[j].tobytes(), (
                f"le mode α={alpha} n'est pas déterministe au niveau {j}")


def test_extrapoler_depasse_et_le_clamp_compte():
    """Verrou (d) — §5 du spec. Le clamp COMPTE et REPORTE, il ne lève pas."""
    pyr = _pyramide()
    _peindre_s(pyr, 10.0)
    tampon = TamponReadout(pyr)
    tampon.capturer(pyr)          # s_prev = 10
    _peindre_s(pyr, 1.0)          # s_cur  = 1  ⇒ α=1½ donne −3,5
    vue = tampon.melanger(pyr, ALPHA_EXTRAPOLER)

    assert vue.clamps > 0, "l'extrapolation devait passer sous zéro"
    for j in pyr.geo.niveaux_gpu:
        assert (vue.fenetres[j] >= 0.0).all(), "le clamp n'a pas tenu"


def test_interpoler_ne_depasse_jamais():
    """Verrou (d), face FAVORABLE — un critère se vérifie aussi sur le cas
    qui doit PASSER (§A52), sinon il punit la qualité qu'il contrôle."""
    pyr = _pyramide()
    _peindre_s(pyr, 10.0)
    tampon = TamponReadout(pyr)
    tampon.capturer(pyr)
    _peindre_s(pyr, 1.0)
    vue = tampon.melanger(pyr, ALPHA_INTERPOLER)
    assert vue.clamps == 0, "α=½ entre deux états positifs ne peut pas clamper"
```

- [ ] **Étape 2 : lancer, vérifier l'échec**

```bash
.venv/bin/pytest tests/test_interpolation_readout.py -k "modes or clamp or depasse" -v
```

Attendu : les tests de clamp **ÉCHOUENT** si `melanger` ne compte pas ; le test
d'invariance **PASSE** déjà (le noyau est unique dès la tâche 1) — **et c'est
le résultat correct** : il est un verrou de non-régression, pas un moteur de
développement. **Le noter et ne pas le "faire échouer" artificiellement.**

- [ ] **Étape 3 : implémenter si nécessaire**

Le comptage de clamps est déjà écrit en tâche 1. Si les trois tests passent
sans modification, **ne rien changer** — écrire du code pour faire échouer un
test qui passe est une fabrication.

- [ ] **Étape 4 : lancer la suite complète du fichier**

```bash
.venv/bin/pytest tests/test_interpolation_readout.py -v
```

Attendu : **6 passed**.

- [ ] **Étape 5 : commit**

```bash
git add tests/test_interpolation_readout.py
git commit -m "Interpolation de readout — verrous (b) et (d) : l'invariance au mode et le clamp

Verrou (b) : §2 du spec MÉCANISÉ. « I est invariant au mode par
construction » était un énoncé ; il est maintenant exécuté — à alpha égal,
les deux modes rendent les mêmes octets. Une garde promise dans un document
est une garde absente (§A62-bis-2).

Verrou (d) : le clamp COMPTE et REPORTE, il ne lève pas — c'est du code
moteur, il tourne. Et il est vérifié sur les DEUX faces : le cas qui doit
clamper (extrapolation sous zéro) ET le cas qui ne doit pas (interpolation
entre deux états positifs). Un critère braqué sur le seul cas défavorable
punit la qualité qu'il contrôle (§A52)."
```

---

## Tâche 3 : la serrure de consommateur et le verrou éphémère

**Fichiers :**
- Modifier : `src/f1_gpu/interpolation_readout.py`
- Modifier : `tests/test_interpolation_readout.py`

**Interfaces :**
- Consomme : `TamponReadout.melanger` (tâche 1).
- Produit :
  - `class ReadoutInterditDansEtat(RuntimeError)`
  - `violations_consignees() -> tuple[str, ...]`
  - `_MODULES_ETAT: frozenset[str]`

- [ ] **Étape 1 : écrire le test qui échoue**

Ajouter à `tests/test_interpolation_readout.py` :

```python
def _appeler_depuis(nom_module: str, fonction, *args, **kwargs):
    """Exécute `fonction` depuis une frame dont `__name__` est `nom_module`.

    Même patron que `tests/test_chemin_de_cout.py` : on fabrique une frame
    portant le nom d'un module d'état, sans importer ce module."""
    code = compile("resultat = fonction(*args, **kwargs)", "<frame>", "exec")
    portee = {"__name__": nom_module, "fonction": fonction,
              "args": args, "kwargs": kwargs}
    exec(code, portee)
    return portee["resultat"]


def test_l_etat_de_la_pyramide_est_bit_identique_apres_appel():
    """Verrou (c), face STRUCTURELLE — la garde de fond du §4.

    Le seul verrou que §A53 ne sait pas contourner : si le module n'écrit
    nulle part dans le tenseur, aucune fuite n'est possible, quel que soit
    l'appelant."""
    pyr = _pyramide()
    _peindre_s(pyr, 4.0)
    tampon = TamponReadout(pyr)
    tampon.capturer(pyr)
    _peindre_s(pyr, 6.0)

    avant = {j: np.array(pyr.fenetres[j], copy=True)
             for j in pyr.geo.niveaux_gpu}
    tampon.melanger(pyr, ALPHA_EXTRAPOLER)
    for j in pyr.geo.niveaux_gpu:
        assert avant[j].tobytes() == pyr.fenetres[j].tobytes(), (
            f"le tenseur de F a été modifié au niveau {j} — FUITE")


def test_la_serrure_leve_depuis_un_module_d_etat():
    """Verrou (c), face DE PILE. `pyramide` est le module qui exécute `F` :
    s'il apparaît dans la pile, un état de readout est en train d'atteindre
    le chemin de la physique."""
    from src.f1_gpu.interpolation_readout import ReadoutInterditDansEtat

    pyr = _pyramide()
    tampon = TamponReadout(pyr)
    tampon.capturer(pyr)
    with pytest.raises(ReadoutInterditDansEtat):
        _appeler_depuis("pyramide", tampon.melanger, pyr, ALPHA_INTERPOLER)


def test_la_tentative_est_consignee_meme_si_l_exception_est_avalee():
    """La trace survit à un `except RuntimeError` bien intentionné.

    Découvert par mutation sur `chemin_de_cout` le 03/08 : la serrure hérite
    de `RuntimeError`, et un appelant qui enveloppe l'appel d'un
    `except RuntimeError` l'avale EN SILENCE. L'exception peut être étouffée ;
    la trace, non."""
    from src.f1_gpu.interpolation_readout import violations_consignees

    pyr = _pyramide()
    tampon = TamponReadout(pyr)
    tampon.capturer(pyr)
    avant = len(violations_consignees())
    try:
        _appeler_depuis("pyramide", tampon.melanger, pyr, ALPHA_INTERPOLER)
    except RuntimeError:
        pass
    assert len(violations_consignees()) == avant + 1
```

- [ ] **Étape 2 : lancer, vérifier l'échec**

```bash
.venv/bin/pytest tests/test_interpolation_readout.py -k "serrure or consignee or bit_identique" -v
```

Attendu : `ImportError` sur `ReadoutInterditDansEtat` et
`violations_consignees`. Le test `bit_identique` **passe déjà** — c'est le
verrou structurel, et il tient par construction depuis la tâche 1. Le noter.

- [ ] **Étape 3 : implémenter la serrure**

Ajouter à `src/f1_gpu/interpolation_readout.py`, après les constantes :

```python
# Modules dont la seule présence dans la pile signale que le chemin de l'ÉTAT
# est en train d'atteindre un produit de readout. La liste est la MÉCANISATION
# du §4 du spec : elle doit GRANDIR avec tout nouveau module qui écrit l'état
# ou exécute `F`. Le nom SIMPLE est comparé : `src.f1_gpu.pyramide` -> `pyramide`.
_MODULES_ETAT: frozenset[str] = frozenset({
    "pyramide",                     # exécute `F` via `pas_f` (`pyramide.py:501`)
    "transferts",                   # descend/remonte l'état
    "ledger",                       # comptabilité de conservation
    "exner_gpu",
    "substrat_fusionne",
    "substrat_fusionne_3d",
    "substrat_fusionne_3d_param",
    "substrat_jetable",
    "substrat_jetable_3d",
    "substrat_l3",
    "substrat_fidele",
})


class ReadoutInterditDansEtat(RuntimeError):
    """Levée quand le chemin de l'ÉTAT tente d'atteindre un produit de readout.

    Hérite de `RuntimeError` — la faute porte sur QUI appelle. N'est JAMAIS
    rattrapée dans ce module : une serrure qui se rattrape est une promesse,
    pas une serrure."""


# Trace INDÉLÉBILE des violations — voir `_verrouiller_consommateur`.
_VIOLATIONS: list[str] = []


def violations_consignees() -> tuple[str, ...]:
    """Violations survenues depuis le début du processus.

    **TOUT DRIVER QUI PRONONCE UN CHIFFRE DOIT LIRE CECI AVANT DE L'ÉCRIRE.**
    Une liste non vide signifie qu'un produit de readout a été atteint depuis
    le chemin de l'état : la comptabilité de conservation n'est plus garantie.

    POURQUOI LA TRACE EXISTE — la leçon de `chemin_de_cout`, trouvée par
    mutation le 03/08 et non par relecture. `ReadoutInterditDansEtat` hérite de
    `RuntimeError` ; un appelant qui écrit `except RuntimeError` — geste banal
    et bien intentionné — **avale la serrure en silence**, et la garde redevient
    une promesse. L'exception peut être étouffée ; la trace, non."""
    return tuple(_VIOLATIONS)


def _verrouiller_consommateur() -> None:
    """Refuse l'appel si la pile traverse un module d'état, et CONSIGNE la
    tentative AVANT de lever.

    Remonte les frames par `sys._getframe` plutôt que par `inspect.stack()` :
    ce dernier ouvre et lit les fichiers source de chaque frame, ce qui
    coûterait des millisecondes dans un chemin dont `I` sera un jour mesuré."""
    frame = sys._getframe(1)
    while frame is not None:
        nom = frame.f_globals.get("__name__", "")
        if nom.rsplit(".", 1)[-1] in _MODULES_ETAT:
            _VIOLATIONS.append(nom)          # consignée AVANT la levée
            raise ReadoutInterditDansEtat(
                f"interpolation de readout appelée depuis « {nom} », un "
                "module d'ÉTAT. §4 du spec : `s_out` et `s_prev` sont "
                "ÉPHÉMÈRES — jamais écrits dans l'état, jamais lus par `F`. "
                "Une fuite corromprait la comptabilité de conservation sans "
                "produire aucun symptôme. Tentative consignée — voir "
                "`violations_consignees()`.")
        frame = frame.f_back
```

Puis, **première ligne du corps de `melanger`** :

```python
        _verrouiller_consommateur()
```

- [ ] **Étape 4 : lancer la suite complète**

```bash
.venv/bin/pytest tests/test_interpolation_readout.py -v
```

Attendu : **9 passed**.

- [ ] **Étape 5 : commit**

```bash
git add src/f1_gpu/interpolation_readout.py tests/test_interpolation_readout.py
git commit -m "Interpolation de readout — verrou (c) : la garde éphémère, structurelle puis de pile

STRUCTUREL d'abord : le tenseur de F est prouvé bit-identique après appel.
C'est le seul verrou que §A53 ne sait pas contourner — si le module n'écrit
nulle part dans l'état, aucune fuite n'est possible quel que soit l'appelant.

DE PILE ensuite : _verrouiller_consommateur() lève si un module d'état est
dans la pile, et CONSIGNE la tentative AVANT de lever. La trace reprend la
leçon de chemin_de_cout, trouvée par mutation le 03/08 : la serrure hérite
de RuntimeError, donc un « except RuntimeError » bien intentionné l'avale en
silence. L'exception peut être étouffée ; la trace, non.

Jamais un assert : il disparaît sous python -O (§A43)."
```

---

## Tâche 4 : le verrou (e) — la non-régression de `§A48`

**Fichiers :**
- Modifier : `tests/test_interpolation_readout.py`

**Interfaces :**
- Consomme : `TamponReadout` (tâche 1), `gather_chemin_de_cout` (existant).
- Produit : rien — c'est le contrat du `§1` du spec, mécanisé.

**Pourquoi séparée** : c'est le verrou qui protège un fichier que ce plan
interdit de toucher. Il doit pouvoir être rejeté seul.

- [ ] **Étape 1 : écrire le test**

Ajouter à `tests/test_interpolation_readout.py` :

```python
def test_le_gather_est_inchange_et_lit_la_vue():
    """Verrou (e) — le contrat du §1 du spec.

    Deux choses en une, et les deux comptent : (1) le gather rend LES MÊMES
    OCTETS sur la pyramide qu'avant l'existence de ce module — le plancher
    « plus proche voisin » de §A48 garde 1 n'a pas bougé ; (2) le même gather,
    NON MODIFIÉ, sait lire la vue interpolée, parce qu'elle expose la même
    surface. `chemin_de_cout.py` n'a pas une ligne de différence."""
    from src.f1_gpu.chemin_de_cout import gather_chemin_de_cout

    pyr = _pyramide()
    _peindre_s(pyr, 4.0)
    cote = N_FOV

    # (1) le gather sur la pyramide, inchangé
    ecran_direct = gather_chemin_de_cout(pyr, cote, indice_champ=INDICE_CHAMP_S)
    assert np.allclose(ecran_direct, 4.0)

    # (2) le MÊME gather sur la vue : α = 1 doit reproduire le gather direct
    tampon = TamponReadout(pyr)
    tampon.capturer(pyr)
    vue = tampon.melanger(pyr, 1.0)
    ecran_vue = gather_chemin_de_cout(vue, cote, indice_champ=0)

    assert ecran_direct.tobytes() == ecran_vue.tobytes(), (
        "à α=1 la vue doit rendre exactement le gather direct")


def test_deux_appels_rendent_les_memes_octets():
    """Verrou (f) — fonction pure, anti-PERSIST. La SYNTHÈSE est sans état,
    même si `s_prev` en a un : c'est la seconde jambe de la défense du
    §4-bis — de l'HISTOIRE, pas de la SYNTHÈSE."""
    pyr = _pyramide()
    _peindre_s(pyr, 2.0)
    tampon = TamponReadout(pyr)
    tampon.capturer(pyr)
    _peindre_s(pyr, 5.0)

    premier = {j: np.array(tampon.melanger(pyr, ALPHA_INTERPOLER).fenetres[j],
                           copy=True) for j in pyr.geo.niveaux_gpu}
    second = {j: np.array(tampon.melanger(pyr, ALPHA_INTERPOLER).fenetres[j],
                          copy=True) for j in pyr.geo.niveaux_gpu}
    for j in pyr.geo.niveaux_gpu:
        assert premier[j].tobytes() == second[j].tobytes()
```

- [ ] **Étape 2 : lancer, vérifier**

```bash
.venv/bin/pytest tests/test_interpolation_readout.py -v
```

Attendu : **11 passed**. Si `gather_chemin_de_cout(vue, ...)` échoue, **NE PAS
modifier `chemin_de_cout.py`** — c'est `VueInterpolee` qui doit exposer
l'attribut manquant. Le contrat du `§1` est que le gather ne bouge pas.

- [ ] **Étape 3 : prouver que `chemin_de_cout.py` n'a pas bougé**

```bash
git diff --stat 8d839fb -- src/f1_gpu/chemin_de_cout.py src/f1_gpu/pyramide.py
```

`8d839fb` est la BASE de branche (le commit du plan), **pas `HEAD`**.
Comparer à `HEAD` ne verrait que l'arbre de travail et laisserait passer une
modification COMMITÉE dans une tâche antérieure — le contrat du `§1` doit
tenir sur toute la série, pas sur le dernier commit.

Attendu : **sortie vide**. Si elle ne l'est pas, la tâche est en échec quel
que soit l'état des tests.

- [ ] **Étape 4 : lancer la suite ENTIÈRE du dépôt**

```bash
.venv/bin/pytest
```

Attendu : aucune régression. **Reporter le compte exact**, ne pas le supposer.

- [ ] **Étape 5 : commit**

```bash
git add tests/test_interpolation_readout.py
git commit -m "Interpolation de readout — verrous (e) et (f) : la non-régression de §A48 et la pureté de la synthèse

Verrou (e) : le gather rend LES MÊMES OCTETS qu'avant l'existence de ce
module, et le MÊME gather NON MODIFIÉ sait lire la vue interpolée parce
qu'elle expose la même surface. chemin_de_cout.py n'a pas une ligne de
différence — prouvé par git diff --stat vide, pas supposé. Le plancher
« plus proche voisin » de §A48 garde 1 n'a pas bougé.

Verrou (f) : deux appels rendent les mêmes octets. La SYNTHÈSE est sans
état même si s_prev en a un — c'est la seconde jambe de la défense du
§4-bis : de l'HISTOIRE, pas de la SYNTHÈSE."
```

---

## Ce que ce plan ne fait PAS — **ARRÊT**

C'est le `§11` du spec, repris pour qu'aucune tâche ne l'enjambe :

- **Aucune mesure de `I`.** `§A61` l'interdit, `§A63` n'a pas qualifié
  l'instrument. Aucune tâche n'appelle `cupy`, ne chronomètre, ne lance de
  driver.
- **Aucun branchement dans une boucle de rendu réelle.** Le composant existe et
  est testé ; le câbler à 60 fps est un incrément, non couvert ici.
- **La lecture d'orientation n'est pas ordonnée** et son prereg n'est pas
  écrit.
- **Ni cadence, ni côté, ni Porte 3 de `§A58`.**
- **Les trois dûs du spec restent ouverts** : le placement « interpoler
  l'IMAGE » (`§9`, non chiffré) ; le fait que chronométrer un composant à état
  remet en cause la pureté de la chaîne de mesure (`§4-bis`) ; et l'élargissement
  de la liste de champs par R2+ (`§8`, « `s` seul est un plancher »).

**Rien ne s'enchaîne après la tâche 4.**
