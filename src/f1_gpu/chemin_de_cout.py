"""CHEMIN-DE-COÛT — assemblage naïf des fenêtres actives vers un champ écran.

CE MODULE N'EST PAS UN COMPOSITEUR. Il est autorisé par §A48 du journal
(`pocCascade2phys/PREREGISTRATION.md`) comme objet JETABLE, jugé au COÛT SEUL,
et il ne tranche RIEN — ni la couture fovéa/grossier, ni un contrat de fidélité.
Le compositeur INSTRUMENT, lui, reste post-P3 et se juge au pin transporté.

    « ce qu'il tranche : rien · son juge : "ça tient dans le budget" ·
      ce qu'il mesure : coût seul »   — §A48, table de la coupe

LES QUATRE GARDES DE §A48, ET OÙ ELLES VIVENT ICI
-------------------------------------------------

**Garde 1 — PLANCHER SUR STRUCTURE RÉELLE** (« naïf dans l'opérateur, réel dans
la structure »). L'arithmétique est la plus bête permise : PLUS PROCHE VOISIN
par division entière, INSERTION DURE du grossier vers le fin, aucune
interpolation, aucun mélange. Mais la structure lue est la vraie : on parcourt
les FENÊTRES ACTIVES de `PyramideFovea` (`src/f1_gpu/pyramide.py`), slot par
slot, aux origines B3 réelles — jamais un tableau grossier dense.

La garde est explicite là-dessus : « un upsample cohérent depuis un grossier
DENSE borne le coût d'un autre programme ; le gather CREUX est dispersé, et le
rapport cohérent/dispersé n'est pas une marge, c'est le terme dominant ». C'est
pourquoi `src/multiresolution.py:30` (`upsample`) et `src/summary.py:191`
(`_bilinear_upsample`) sont INUTILISABLES ici : tous deux partent d'un grossier
dense. Aucun n'est importé, et ce n'est pas un oubli.

Le chiffre produit est donc un PLANCHER de la classe gather — l'opérateur de
couture réel ajoutera de l'arithmétique (stencil de prédiction). Il se lit
« plancher + marge », JAMAIS comme LE chiffre.

**Garde 2 — SERRURE MÉCANISÉE.** `_verrouiller_appelant()` lève une
`RuntimeError` explicite si un module perceptuel figure dans la pile d'appel.
JAMAIS un `assert` : il disparaît sous `python -O` (fait consigné §A43 du
journal). `RuntimeError` et non le `ValueError` de la maison, parce que la garde
porte sur QUI APPELLE, pas sur quelle valeur.

Second verrou, en aval et préexistant : `src/arcC_rendu.py:275-279` refuse tout
champ hors 64x64 — « un champ d'une autre taille signifierait qu'un producteur
non prévu (compositeur ? niveau grossier ?) est arrivé jusqu'ici sans décision
[...] c'est un garde de PÉRIMÈTRE ». Écrit le 2026-07-25, AVANT §A48, et visant
nommément ce que §A48 autorise. Le chemin-de-coût ne passe donc jamais par
l'instrument, et ce module n'importe rien de `arcC_*`.

**Garde 3 — DISQUALIFICATION PAR LE NOM.** Fichier, classe et fonctions portent
`chemin_de_cout`, jamais « compositeur ». Jetable par construction : le jour où
le compositeur instrument existe, ce fichier se supprime sans dette.

**Garde 4 — CLAUSE DE COUCHER.** Post-P3, le compositeur instrument supersède ce
module POUR TOUT USAGE. La dette « mesurable quand le compositeur existera »
reste OUVERTE : le chiffre naïf l'avance, il ne la solde pas. L'anti-surclame
s'applique mot pour mot — le résultat se dit « le NOYAU plus un gather-plancher
coûtent X », jamais « le rendu coûte X ».
"""
from __future__ import annotations

import sys

import numpy as np

# Modules dont la seule présence dans la pile interdit ce chemin (garde 2).
# Le nom SIMPLE du module est comparé : `src.arcC_abx` -> `arcC_abx`.
_MODULES_PERCEPTUELS: frozenset[str] = frozenset({
    "arcC_abx",            # l'ABX lui-même
    "arcC_stimuli",        # fabrique des stimuli jugés par un observateur
    "arcC_orchestration",  # pilote une session perceptuelle
    "arcC_rendu",          # le NOYAU-INSTRUMENT R1, scellé
    "arcC_scelle",         # descellement des seuils P3'
    "arcC_calibration",    # points d'opération de l'instrument
})


class CheminDeCoutInterdit(RuntimeError):
    """Levée quand un chemin perceptuel tente d'atteindre le chemin-de-coût.

    Hérite de `RuntimeError` (garde 2 : la faute porte sur QUI appelle) et
    n'est JAMAIS rattrapée dans ce module : une serrure qui se rattrape est
    une promesse, pas une serrure."""


# Trace INDÉLÉBILE des violations de la garde 2 — voir `_verrouiller_appelant`.
_VIOLATIONS: list[str] = []


def violations_consignees() -> tuple[str, ...]:
    """Violations de la garde 2 survenues depuis le début du processus.

    **TOUT DRIVER QUI PRONONCE UN CHIFFRE DOIT LIRE CECI AVANT DE L'ÉCRIRE.**
    Une liste non vide annule le chiffre : le chemin-de-coût a été atteint
    depuis un chemin perceptuel, et rien ne garantit plus ce qu'il a mesuré.

    POURQUOI CETTE TRACE EXISTE — découvert par mutation, 2026-08-03, pas par
    relecture. `CheminDeCoutInterdit` hérite de `RuntimeError` comme §A48
    l'exige ; or ce module lève AUSSI des `RuntimeError` pour les trous de
    couverture. Un appelant qui écrit `except RuntimeError` autour du gather —
    geste banal et bien intentionné — **avale la serrure en silence** et la
    garde 2 redevient une promesse. Le mutant qui rattrapait les trous a
    neutralisé la serrure sans le vouloir : c'est exactement le glissement que
    §A48 garde 3 nomme.

    La réponse est celle que la maison emploie déjà pour le scellement
    (`src/arcC_scelle.py` : « si quelqu'un ouvre le fichier scellé avant le
    prononcé, la non-cécité se CONSIGNE au verdict ») : on n'essaie pas
    d'empêcher l'étouffement, on le rend **VISIBLE APRÈS COUP**. L'exception
    peut être avalée ; la trace, non.

    ÉTAT MUTABLE ASSUMÉ. `src/arcC_rendu.py` interdit toute variable de module
    mutable — c'est l'INSTRUMENT, et un instrument avec mémoire n'est pas
    rejouable. Ce module n'est pas un instrument : il est jetable, jugé au coût
    seul, et sa trace de violation EXIGE une mémoire. Les deux règles ne se
    contredisent pas, elles s'appliquent à deux objets que §A48 sépare."""
    return tuple(_VIOLATIONS)


def _verrouiller_appelant() -> None:
    """GARDE 2, mécanisée : refuse l'appel si la pile traverse un module
    perceptuel, et CONSIGNE la tentative avant de lever.

    Remonte les frames par `sys._getframe` plutôt que par `inspect.stack()` :
    ce dernier ouvre et lit les fichiers source de chaque frame, ce qui
    coûterait des millisecondes DANS la boucle qu'on chronomètre. Ici on ne lit
    que `f_globals['__name__']` — le coût est mesuré et déclaré par
    `tests/test_chemin_de_cout.py`, jamais supposé négligeable."""
    frame = sys._getframe(1)
    while frame is not None:
        nom = frame.f_globals.get("__name__", "")
        if nom.rsplit(".", 1)[-1] in _MODULES_PERCEPTUELS:
            # Consignée AVANT la levée : un `except` en aval ne l'efface pas.
            _VIOLATIONS.append(nom)
            raise CheminDeCoutInterdit(
                f"chemin-de-coût appelé depuis « {nom} », un chemin "
                "perceptuel. §A48 garde 2 : cet objet est jugé au COÛT SEUL et "
                "ne passe JAMAIS sous une mesure de fidélité. Le compositeur "
                "instrument, lui, attend P3 et le pin transporté. "
                "Tentative consignée — voir `violations_consignees()`.")
        frame = frame.f_back


def _fenetres_du_niveau(pyramide, j: int):
    """Fenêtres actives du niveau `j` — API publique de la géométrie, jamais
    l'état privé de la pyramide."""
    return pyramide.fenetres[j], pyramide.geo.origines(j, pyramide.centre_fin)


def gather_chemin_de_cout(pyramide, cote_px: int, *,
                          indice_systeme: int = 0,
                          indice_champ: int = 3) -> np.ndarray:
    """Assemble un champ écran `cote_px x cote_px` depuis les fenêtres actives.

    Sémantique : l'écran est une fenêtre carrée en COORDONNÉES DU NIVEAU FIN,
    centrée sur le centre fovéal courant. Avec `cote_px > n_fov` elle déborde la
    fenêtre fovéale fine et doit être complétée par les niveaux plus grossiers —
    c'est exactement le cas d'usage du gather (fin au centre, grossier autour),
    et c'est ce qui rend le coût mesuré représentatif.

    `indice_champ=3` : le sédiment `s`, dans l'ordre `(h, hu, hv, s)` gravé
    `src/f1_gpu/pyramide.py:75` (« `CHAMPS_PAR_SYSTEME: int = 4  # (h, hu, hv,
    s) — E4a` »). C'est le champ dont `src/albedo.py` tire le readout.

    ARITHMÉTIQUE (garde 1) : `coordonnée_fine // 2**(j_fin - j)` — plus proche
    voisin par troncature, insertion dure, écrasement du grossier par le fin
    dans l'ordre croissant des niveaux. Rien de plus cher n'est permis ici.

    FAIL-LOUD : un pixel qu'aucune fenêtre active ne couvre reste `NaN` et fait
    lever. Un trou de couverture est un fait de structure, pas une valeur à
    rattraper en douce — le rattraper masquerait précisément ce que la mesure
    doit voir."""
    _verrouiller_appelant()

    xp = pyramide.xp
    geo = pyramide.geo
    if cote_px <= 0:
        raise ValueError(f"gather_chemin_de_cout : cote_px={cote_px} <= 0.")

    j_fin = geo.niveau_fin
    demi = cote_px // 2
    origine_ecran = pyramide.centre_fin - demi

    # Coordonnées FINES de chaque ligne/colonne de l'écran.
    coord_fine = origine_ecran + xp.arange(cote_px, dtype=xp.int64)
    ecran = xp.full((cote_px, cote_px), xp.nan, dtype=xp.float32)

    # Du GROSSIER au FIN : le fin écrase, c'est l'insertion dure.
    for j in geo.niveaux_gpu:
        facteur = 2 ** (j_fin - j)
        coord_j = coord_fine // facteur          # plus proche voisin, troncature
        fenetres, origines = _fenetres_du_niveau(pyramide, j)

        for indice_slot, (oy, ox) in enumerate(origines):
            iy = coord_j - oy
            ix = coord_j - ox
            valides_y = xp.flatnonzero((iy >= 0) & (iy < geo.n_fov))
            valides_x = xp.flatnonzero((ix >= 0) & (ix < geo.n_fov))
            if valides_y.size == 0 or valides_x.size == 0:
                continue
            source = fenetres[indice_slot, indice_systeme, indice_champ]
            ecran[valides_y[:, None], valides_x[None, :]] = source[
                iy[valides_y][:, None], ix[valides_x][None, :]]

    manquants = int(xp.isnan(ecran).sum())
    if manquants:
        raise RuntimeError(
            f"gather_chemin-de-coût : {manquants} pixel(s) sur {cote_px**2} ne "
            "sont couverts par AUCUNE fenêtre active. Trou de couverture "
            "structurel — réduire `cote_px` ou revoir la liste de slots. Un "
            "remplissage par défaut fausserait le coût mesuré.")
    return ecran


def albedo_ecran(ecran_s: np.ndarray, s_half: float) -> np.ndarray:
    """Readout albédo appliqué APRÈS le gather : `A = 1 − exp(−s / s_half)`.

    L'ordre est délibéré et sans effet sur le résultat — le readout est LOCAL,
    donc il commute avec un plus-proche-voisin — mais il change le coût : une
    seule exponentielle par pixel écran au lieu d'une par cellule active.
    Le chemin-de-coût prend systématiquement le moins cher des équivalents :
    c'est ce qu'un PLANCHER doit faire.

    Réplique la formule de `src/albedo.py:27-30` sans l'importer : ce module ne
    dépend d'aucun chemin instrumenté (garde 3), et une dépendance vers un
    module perceptuel serait exactement le glissement que le nom protège."""
    xp = np if isinstance(ecran_s, np.ndarray) else sys.modules[
        type(ecran_s).__module__.split(".")[0]]
    if s_half <= 0.0:
        raise ValueError(f"albedo_ecran : s_half={s_half} <= 0.")
    return 1.0 - xp.exp(-ecran_s / s_half)
