"""DÉMO DU 2:1 — rendre VISIBLE ce que la boucle orchestre, pas seulement qu'elle tourne.

    .venv/bin/python -m scripts.run_demo_2a1 \\
        --cote interpoler --s-half 0.15 --dossier outputs/demo_2a1_2026-08-26

CE QUE CE FICHIER AJOUTE À SON FRÈRE `run_boucle_rendu_demo.py`, ET POURQUOI IL
EXISTE. Le frère prouve que la chaîne tourne et le dit lui-même : sur le substrat
jetable, un pas de physique déplace `s` de MOINS d'un niveau de gris, donc les
deux images d'un même tick sortent BIT-IDENTIQUES une fois quantifiées. On voit
la chaîne, on ne voit pas le 2:1. La clôture du chantier a nommé trois voies pour
y remédier — `pas_f` injecté, plus de bits, cumul sur plusieurs ticks — et a écrit
qu'« aucune n'est un réglage de driver », donc que chacune appelait un
pré-enregistrement. **§A64-4 a recalibré cette cérémonie** —
`PREREGISTRATION.md:10507` « Tout le reste est donc verrouillé par tests et
consigné par des messages de commit factuels » : cette démo ne
mesure rien, ne prononce rien, et aucune décision verdict-grade n'en dépend. Elle
se fait donc en build, verrouillée par des tests, sans prereg.

LA VOIE RETENUE EST LA MOINS CHÈRE DES TROIS : un `pas_f` injecté. Le paramètre
existe déjà sur `PyramideFovea`, et le dépôt en a le précédent exact — le pas
doublant des tests. Les deux autres voies coûtent un format d'image neuf ou une
logique d'accumulation ; celle-ci coûte une fonction.

CE QUE LE JOUET FAIT, ET CE QU'IL N'EST PAS. `_pas_f_translation` fait glisser le
champ `s` d'UNE CELLULE par `frame()`, dans le sens des `x` DÉCROISSANTS
(`SENS_TRANSLATION`, dont le motif est mesuré plus bas), et ne touche à aucun
autre champ. **Ce n'est pas de la physique et ça n'a pas à l'être** — même statut
que le pas doublant : ce qu'on regarde ici est la GÉOMÉTRIE TEMPORELLE de la
lecture, pas ce que `F` calcule. À noter, et ce n'est pas un défaut : chaque
niveau glisse d'une cellule DE SA PROPRE RÉSOLUTION, donc les niveaux grossiers
défilent plus vite en unités de monde.

CE QU'ON VOIT, ET C'EST LA SIGNATURE DU 2:1. Le champ peint est une DENT DE SCIE
en `x`. Sur ses rampes, le mélange d'état à `α = 0,5` vaut exactement la rampe
translatée d'UNE DEMI-CELLULE — `0,5·f(x) + 0,5·f(x+1) = f(x + 0,5)` pour `f`
affine, le champ glissant vers les `x` décroissants : l'écran intermédiaire montre une position qui n'a jamais été calculée par
la physique, et c'est précisément le service que le 2:1 rend. Sur la
DISCONTINUITÉ de la dent, en revanche, le mélange n'est pas une translation mais
un FONDU : on y voit un fantôme à mi-intensité. Ce n'est pas un bug, c'est ce
qu'interpoler l'ÉTAT veut dire, et c'est écrit ici pour que personne ne le
rediagnostique. À l'œil, la séquence alterne donc net / intermédiaire / net /
intermédiaire — deux images par pas de physique.

AUCUNE MESURE, AUCUN CHIFFRE DE COÛT — et la phrase est étroite parce que la
version large serait fausse, exactement comme dans le frère. `TransfertComptable`
est instanciée ici et chaque `frame()` traverse `_chronometrer`, qui lit
`time.perf_counter` : **un chronomètre tourne pendant cette démo**, atteint par
transitivité. Ce qui est vrai est que ce fichier ne lit lui-même aucune horloge —
mécanisé par `test_la_source_de_la_demo_2a1_ne_porte_ni_filet_large_ni_assert_ni_horloge`
— et qu'AUCUN CHIFFRE N'EN SORT : rien ici ne lit, ne compare, n'imprime ni ne
retourne `h2d_ms` / `d2h_ms`. C'est ce second point qui garde le régime de mesure,
et non une absence de chronomètre qui serait imaginaire.

LE DOSSIER DE SORTIE EST DONNÉ, JAMAIS FABRIQUÉ ICI, ET C'EST UNE CONSÉQUENCE DU
POINT PRÉCÉDENT. Un dossier horodaté par le driver voudrait dire `datetime` ou
`time` dans cette source — la famille même que le verrou d'horloge interdit. C'est
donc l'appelant qui horodate le nom, et ce fichier se contente de REFUSER un
dossier qui existe déjà : « jamais par-dessus un artefact existant » devient une
garde mécanisée au lieu d'une promesse.

CE DRIVER NE PRONONCE RIEN SUR LE CÔTÉ. Il l'exige, il ne le choisit pas — comme
son frère, et pour la même raison : le code ne tranche pas ce qu'un document a
laissé ouvert. Le côté employé pour un run donné, et son motif, vivent dans le
message de commit de ce run (§A64-3 : choix provisoire et révisable).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from scripts.run_boucle_rendu_demo import (
    COTES_DISPONIBLES,
    N_FOV,
    N_NIV,
    N0,
    encoder_png_gris,
    quantifier_en_octets,
)
from src.f1_gpu.boucle_rendu import BoucleRendu
from src.f1_gpu.interpolation_readout import INDICE_CHAMP_S
from src.f1_gpu.pyramide import GeometriePyramide, PyramideFovea
from src.f1_gpu.transferts import TransfertComptable

# LE PNG N'EST PAS RÉÉCRIT ICI. `quantifier_en_octets` et `encoder_png_gris` sont
# IMPORTÉS du frère : deux encodeurs PNG dans le même dépôt dériveraient l'un de
# l'autre sans que rien ne le signale, et celui-là est déjà tenu par ses verrous
# — relecture octet par octet, aucun `tIME`, aucune bibliothèque d'image.

# Côté d'écran couvrable, repris du frère par la même responsabilité d'appelant :
# le gather LÈVE sur un pixel qu'aucune fenêtre active ne couvre et la boucle ne
# le rattrape pas.
COTE_PX_DEMO: int = 128

# LA DENT DE SCIE. La période est en cellules du niveau considéré ; l'amplitude
# est en unités de `s`, donc positive — `albedo_ecran` vit dans `[0, 1)` pour tout
# `s ≥ 0`, et une translation est une PERMUTATION des valeurs : l'ensemble des
# valeurs peintes est invariant dans le temps. En bande au premier tick ⇒ en bande
# à tous, sans qu'aucune garde n'ait à le surveiller.
PERIODE_DENT: int = 16
AMPLITUDE_DENT: float = 0.30

# LE SENS DE LA TRANSLATION EST `-1`, ET C'EST UN FAIT MESURÉ, PAS UN GOÛT.
# La fovéa avance d'UNE CELLULE FINE PAR TICK (§5 de la spec de la boucle), dans
# le sens des `x` croissants. Un roll `+1` déplace le champ du MÊME pas dans le
# MÊME sens : les deux s'annulent, et le déplacement apparent à l'écran est de
# ZÉRO pixel par tick — constaté par corrélation croisée sur la ligne médiane de
# quatre écrans exacts avant d'écrire cette constante, pas supposé. La séquence
# aurait alors montré une image FIXE qui scintille : le symptôme d'un montage
# cassé alors que tout tourne. À `-1`, les deux mouvements s'ajoutent.
SENS_TRANSLATION: int = -1

# La longueur a un défaut, le côté et `s_half` n'en ont pas — même régime que le
# frère, et pour son motif : la longueur d'une démo n'est la valeur de rien.
N_TICKS_DEFAUT: int = 6


class DossierDejaPresent(ValueError):
    """Levée quand le dossier de sortie demandé existe déjà.

    Une classe NOMMÉE, comme `EcranNonRepresentable` et `CoteInconnu`, et pour le
    même motif : un `ValueError` nu obligerait son verrou à se rabattre sur le
    texte du message, serrure qu'une reformulation innocente ouvrirait.

    Hérite de `ValueError` et non de `RuntimeError` — un `except RuntimeError`
    d'appelant avalerait la garde en silence, et le run écraserait alors ce qu'il
    devait protéger."""


def _pas_f_translation(fenetres, xp, sortie=None):
    """Pas de physique JOUET : `s` glisse d'une cellule, le reste intact.

    LA TRANSLATION EST EXACTE — c'est une permutation des valeurs, aucune
    arithmétique flottante n'intervient. Les valeurs de `s` à la frame `n` sont exactement celles
    de la frame `n−1`, réordonnées : rien ne croît, rien ne sature, et la bande de
    l'albédo est invariante dans le temps. Un pas multiplicatif ferait exploser `s`
    en quelques ticks et un pas additif ferait dériver le fond ; ni l'un ni l'autre
    ne se regarde sur une séquence.

    Ce n'est PAS de la physique, au même titre que le pas doublant des tests : ce
    qu'on rend visible ici est la géométrie TEMPORELLE de la lecture — un pas de
    monde, deux écrans — et non ce que `F` calcule."""
    if sortie is None:
        sortie = xp.empty_like(fenetres)
    if sortie is not fenetres:
        sortie[...] = fenetres
    sortie[:, :, INDICE_CHAMP_S] = xp.roll(
        sortie[:, :, INDICE_CHAMP_S], SENS_TRANSLATION, axis=-1)
    return sortie, 0.0


def peindre_dent_de_scie(pyramide) -> None:
    """Peint `s` en DENT DE SCIE selon `x`, à tous les niveaux GPU.

    POURQUOI UNE DENT ET NON UNE RAMPE SEULE. Une rampe pure translatée reste la
    même rampe à une constante près : rien ne bougerait à l'écran, et la démo
    serait verte et vide. Une dent porte une DISCONTINUITÉ, donc un repère qui se
    déplace — c'est lui qu'on suit d'une image à l'autre.

    POURQUOI UNE DENT ET NON UN CRÉNEAU. Sur les rampes, le mélange à `α = 0,5`
    est EXACTEMENT la dent translatée d'une demi-cellule ; sur un créneau, il n'y
    aurait que des fondus. La dent montre donc les deux régimes dans la même
    image, et c'est le plus honnête des deux motifs : l'interpolation d'état rend
    un demi-pas là où le champ est affine, et un fantôme là où il saute."""
    for j in pyramide.geo.niveaux_gpu:
        fenetre = pyramide.fenetres[j]
        n = fenetre.shape[-1]
        x = np.arange(n, dtype=np.float32).reshape(1, 1, 1, n)
        dent = (x % PERIODE_DENT) * (AMPLITUDE_DENT / PERIODE_DENT)
        fenetre[:, :, INDICE_CHAMP_S, :, :] = dent


def construire_pyramide() -> PyramideFovea:
    """La pyramide de la démo — le substrat du frère, plus le SEUL `pas_f` injecté.

    Tout le reste est laissé aux défauts du module, `slots` compris : la géométrie
    est celle de M-a/M-c, inchangée. La seule chose que cette démo introduit est
    le pas jouet, et c'est ce qu'elle annonce.

    `frame_suivante()` clôt le bilan de la descente initiale, payée à la
    construction et hors-série — le même geste que le frère et que le harnais de
    tests. Il ne mesure rien : il sépare l'initialisation du régime établi."""
    geo = GeometriePyramide(n_fov=N_FOV, n_niv=N_NIV, n0=N0)
    transferts = TransfertComptable(np)
    pyramide = PyramideFovea(np, geo, transferts, pas_f=_pas_f_translation)
    transferts.frame_suivante()
    return pyramide


def rendre(cote: str, s_half: float, n_ticks: int, dossier: Path) -> int:
    """`n_ticks` ticks → `2 · n_ticks` PNG numérotés. Rend le nombre écrit.

    LE DOSSIER DOIT NE PAS EXISTER, et c'est la garde de ce fichier. Un run qui
    retombe dans un dossier peuplé mélange deux séquences dont rien ne dit
    ensuite laquelle est laquelle — et si la seconde est plus courte, il en reste
    la queue de la première, indiscernable à l'œil. `mkdir(exist_ok=False)` le
    refuse au lieu de le découvrir après coup.

    Le couple d'un tick sort en `α` CROISSANT et exactement un de ses deux termes
    vaut `α = 1,0` : les images se numérotent donc dans l'ordre où elles sortent,
    deux par tick, sans trou.

    AUCUNE LEVÉE N'EST RATTRAPÉE ICI : un trou de couverture du gather est un fait
    de structure, il traverse."""
    if n_ticks <= 0:
        raise ValueError(
            f"rendre : n_ticks={n_ticks} <= 0. Une démo qui n'écrit aucune image "
            "imprimerait « 0 images écrites » et ressemblerait à un run propre.")

    try:
        dossier.mkdir(parents=True, exist_ok=False)
    except FileExistsError as cause:
        raise DossierDejaPresent(
            f"rendre : {dossier} existe déjà. Ce driver n'écrit jamais par-dessus "
            "un artefact existant — donnez un dossier neuf (c'est l'appelant qui "
            "horodate le nom, pas ce fichier).") from cause

    pyramide = construire_pyramide()
    peindre_dent_de_scie(pyramide)
    boucle = BoucleRendu(pyramide, cote)

    compte = 0
    for _ in range(n_ticks):
        for ecran in boucle.tick(COTE_PX_DEMO):
            chemin = dossier / f"ecran_{cote}_{compte:04d}.png"
            chemin.write_bytes(
                encoder_png_gris(quantifier_en_octets(ecran, s_half)))
            compte += 1
    return compte


def construire_parseur() -> argparse.ArgumentParser:
    """`--cote`, `--s-half` et `--dossier` sont OBLIGATOIRES et sans défaut.

    Les deux premiers pour le motif du frère : ils figurent dans ce que la spec
    laisse ouvert, et un défaut trancherait en silence. Le troisième parce qu'un
    défaut ferait retomber deux runs successifs dans le même dossier — la faute
    même que `DossierDejaPresent` existe pour empêcher."""
    parseur = argparse.ArgumentParser(
        description="Démo du 2:1 : un pas de physique, deux écrans.")
    parseur.add_argument("--cote", required=True, choices=COTES_DISPONIBLES)
    parseur.add_argument("--s-half", required=True, type=float)
    parseur.add_argument("--dossier", required=True, type=Path)
    parseur.add_argument("--ticks", type=int, default=N_TICKS_DEFAUT)
    return parseur


def main(argv: list[str] | None = None) -> int:
    """LE SEUL CHIFFRE IMPRIMÉ EST LE NOMBRE D'IMAGES ÉCRITES.

    Un chiffre imprimé par un driver de démo se cite ensuite comme s'il était une
    mesure ; celui-ci n'en est pas une, et il est le seul."""
    options = construire_parseur().parse_args(argv)
    ecrites = rendre(options.cote, options.s_half, options.ticks,
                     options.dossier)
    print(f"{ecrites} images écrites")
    return 0


if __name__ == "__main__":
    sys.exit(main())
