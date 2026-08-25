"""DÉMO 2:1 v2 — LE CONTRASTE DES DEUX CÔTÉS, sur l'inondation shallow-water.

    .venv/bin/python -m scripts.run_demo_2a1_v2 \\
        --s-half 0.9 --dossier outputs/demo_2a1_v2_2026-08-26 --ticks 12

CE QUE LA v1 NE MONTRAIT PAS, ET QUI EST LE SUJET ICI. `run_demo_2a1.py` rend le
2:1 visible d'UN côté : on y voit qu'un pas de physique sert deux écrans. Ce qu'on
n'y voit pas, c'est ce que le CHOIX DU CÔTÉ change — et §A62 grave que le rendu
décide le côté. Cette v2 fait tourner les DEUX côtés en parallèle, même monde,
même balayage fovéal, sorties appariées, et laisse l'œil juger le contraste. Une
cadence seule ne dit rien ; deux cadences côte à côte disent tout.

`run_demo_2a1.py` N'EST PAS TOUCHÉ. C'est un frère, pas une réécriture : le PNG,
la quantification et la classe `DossierDejaPresent` sont IMPORTÉS de lui. Deux
copies d'un encodeur ou d'une garde dérivent l'une de l'autre sans que rien ne le
signale.

LE SUBSTRAT A MAINTENANT DE LA STRUCTURE, ET IL N'EST PAS INVENTÉ ICI. Le champ
affiché est la PROFONDEUR D'EAU `h − b` d'une rupture de barrage shallow-water
déjà calculée par ce dépôt — `data/ground_truth/dam_break.npz`, 201 frames 64×64,
produite par l'oracle Saint-Venant de `src/solver.py`. Aucun solveur n'est écrit
ici, aucun artefact existant n'est réécrit : le fichier est lu, jamais touché.
L'œil a donc un FRONT à juger, et non des rampes synthétiques.

COMMENT L'HISTOIRE ENTRE DANS LA PYRAMIDE, ET POURQUOI ÇA NE COÛTE QU'UNE CLASSE.
La pyramide descend déjà son niveau 0 CPU (`monde0`) vers les fenêtres de chaque
niveau, avec la géométrie correcte, par `predire_bloc_cpu`. Il suffit donc
d'écrire la frame courante de l'histoire dans `monde0` et de laisser la descente
faire le travail. `PasHistoire` est exactement ça : « le monde a avancé, relis-le
». Rien de la géométrie multi-niveaux n'est réimplémenté — c'est la règle §A17,
et c'est aussi ce qui rend le branchement bon marché.

CE QUE LE CHAMP `s` PORTE ICI, DIT FRANCHEMENT. La pyramide porte quatre champs
`(h, hu, hv, s)` et le readout capture le QUATRIÈME, `s`, le sédiment. C'est donc
dans `s` que la profondeur d'eau est écrite : le champ sert de PORTEUR SCALAIRE
pour une démo, il ne devient pas du sédiment pour autant. Le nommer évite qu'on
lise plus tard ces images comme un couplage Exner qui n'a pas eu lieu.

AUCUNE MESURE, AUCUN CHIFFRE ABSOLU EN SORTIE — et la phrase est étroite, comme
chez les deux frères. Un chronomètre tourne par transitivité
(`TransfertComptable._chronometrer`) ; ce qui tient le régime de mesure est
qu'aucun chiffre n'en sorte, et que cette source ne lise elle-même aucune horloge
— mécanisé par
`test_la_source_de_la_v2_ne_porte_ni_filet_large_ni_assert_ni_horloge`. Le seul
chiffre imprimé est le nombre d'images écrites.

L'HORODATAGE EST CHEZ L'APPELANT, acquis de la v1 qu'on ne rediagnostique pas :
un dossier horodaté par le driver voudrait dire `datetime` ou `time` dans cette
source, la famille même que le verrou d'horloge interdit. `--dossier` est donc
obligatoire, et un dossier existant est REFUSÉ.

CE DRIVER NE PRONONCE RIEN SUR LE CÔTÉ. Il les fait tourner tous les DEUX, ce qui
est le contraire de trancher. §A64-3 a fait du côté un choix provisoire et
révisable ; cette démo lui donne de quoi être révisé à l'œil, elle ne le révise
pas.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from scripts.run_demo_2a1 import DossierDejaPresent
from scripts.run_boucle_rendu_demo import (
    N_FOV,
    N_NIV,
    N0,
    encoder_png_gris,
    quantifier_en_octets,
)
from src.f1_gpu.boucle_rendu import (
    COTE_EXTRAPOLER,
    COTE_INTERPOLER,
    BoucleRendu,
)
from src.f1_gpu.interpolation_readout import INDICE_CHAMP_S
from src.f1_gpu.pyramide import (
    GeometriePyramide,
    PyramideFovea,
    predire_bloc_cpu,
)
from src.f1_gpu.transferts import TransfertComptable

# L'HISTOIRE, EN LECTURE SEULE. Artefact existant du dépôt : il est lu, jamais
# réécrit, jamais recopié dans `outputs/`.
HISTOIRE = Path("data/ground_truth/dam_break.npz")

# Les deux côtés, IMPORTÉS et jamais recopiés en littéral — un littéral gravé ici
# donnerait un driver qui accepte un côté que la boucle refuse. L'ordre est celui
# de l'`α` croissant, et il fixe l'ordre des colonnes de la sortie.
COTES: tuple[str, ...] = (COTE_INTERPOLER, COTE_EXTRAPOLER)

COTE_PX_DEMO: int = 128
N_TICKS_DEFAUT: int = 8

# L'ÉCHELLE ET LE PAS, TOUS DEUX MESURÉS AVANT D'ÊTRE ÉCRITS.
#
# LA FOVÉA NE VOIT PAS GRAND-CHOSE DU MONDE, et c'est le fait qui commande tout
# ce bloc. Les niveaux GPU sont 1, 2, 3 ; la fenêtre du niveau `j` couvre
# `n_fov / 2**j` cellules de `monde0`, soit 32, 16 et 8 cellules sur 128. Une
# inondation étalée sur TOUT le monde est donc quasi plate dans ce que l'écran
# montre — constaté : écart-type spatial 1,7 niveau de gris et 9 valeurs
# distinctes sur 256. L'œil n'a rien à juger, et on croirait la chaîne cassée.
#
# CE QUI RÈGLE LE PROBLÈME EST UNE COMPOSITION DE SCÈNE, PAS UN TRUCAGE :
# l'inondation décimée occupe le CARRÉ CENTRAL du monde de la pyramide, le reste
# est SEC. Le monde d'un moteur fovéal est plus grand que le domaine d'une
# simulation ; l'y poser est ce qu'on ferait dans un jeu. Le bénéfice est double —
# le bord mouillé/sec devient le repère que l'œil suit, et la structure passe de
# 1,7 à 37–66 d'écart-type, pour une étendue d'environ 225 niveaux sur 256.
DECIMATION: int = 4

# LE PAS D'HISTOIRE. Deux frames consécutives d'une rupture de barrage diffèrent
# peu ; or ce que les deux côtés montrent d'écart vaut UN DEMI PAS DE MONDE. À
# pas 1, l'écart entre côtés tombe à 0,19 niveau de gris — invisible, et le
# contraste que cette démo existe pour montrer n'existerait pas. Balayage mesuré
# de l'écart moyen entre côtés, à décimation 4 : pas 1 → 0,19 ; pas 5 → 1,01 ;
# pas 15 → 2,86 ; pas 25 → 3,5 à 4,9 selon le tick, avec des pointes à 25.
# Retenu : 25, qui consomme les 201 frames de l'histoire en huit ticks sans
# reboucler.
PAS_HISTOIRE: int = 25


class NiveauNonReconnu(ValueError):
    """Levée quand `pas_f` reçoit un bloc de fenêtres qu'il ne sait pas situer.

    FAIL-LOUD PLUTÔT QUE PAS SAUTÉ. `PasHistoire` identifie le niveau par
    IDENTITÉ du tableau de fenêtres. Si l'identité ne correspond à aucun niveau —
    pyramide réallouée, appel hors `frame()` — la seule alternative serait de ne
    rien faire, c'est-à-dire de laisser le monde immobile sans que rien ne le
    dise. Une démo figée qui ne lève pas est indiscernable d'une démo correcte
    dont le champ n'évolue pas.

    Hérite de `ValueError` comme les autres serrures de ce chantier : ce qui
    hérite de `RuntimeError` risque d'être avalé par un `except RuntimeError`
    d'appelant."""


def charger_profondeur(chemin: Path = HISTOIRE) -> np.ndarray:
    """L'histoire → `(T, N0, N0)` f32 : l'inondation posée dans le monde, terre sèche autour.

    `h` est la surface libre et `b` le fond : la profondeur est `h − b`, bornée à
    zéro. C'est elle qui porte le front, et c'est elle qui est POSITIVE — ce que
    `albedo_ecran` exige, vivant dans `[0, 1)` pour tout `s ≥ 0`.

    LA DÉCIMATION EST UN SOUS-ÉCHANTILLONNAGE STRICT, et rien n'est inventé :
    `[::DECIMATION]` prend une cellule sur deux, donc uniquement des valeurs que le
    solveur a calculées. Une interpolation bilinéaire, elle, fabriquerait des
    valeurs intermédiaires et LISSERAIT le front — c'est-à-dire précisément ce
    qu'on vient regarder.

    LE RESTE DU MONDE EST SEC, à zéro. Ce n'est pas un remplissage de commodité :
    c'est ce qui donne au domaine un BORD, et ce bord mouillé/sec est le repère
    que l'œil suit d'une image à l'autre."""
    with np.load(chemin) as donnees:
        hauteur = np.asarray(donnees["h"], dtype=np.float32)
        fond = np.asarray(donnees["b"], dtype=np.float32)
    profondeur = np.maximum(hauteur - fond[None, :, :], 0.0)[:, ::DECIMATION,
                                                             ::DECIMATION]
    cote = profondeur.shape[-1]
    if cote > N0:
        raise ValueError(
            f"charger_profondeur : le domaine décimé fait {cote} cellules et le "
            f"monde de la pyramide {N0} — il n'y tient pas.")
    monde = np.zeros((profondeur.shape[0], N0, N0), dtype=np.float32)
    origine = (N0 - cote) // 2
    monde[:, origine:origine + cote, origine:origine + cote] = profondeur
    return monde


class PasHistoire:
    """« Le monde a avancé, relis-le » — le `pas_f` de cette démo.

    UNE INSTANCE PAR PYRAMIDE, JAMAIS PARTAGÉE, et ce n'est pas une précaution
    de style. Les deux côtés tournent en parallèle : un objet partagé serait
    appelé par les deux boucles dans le même tick et avancerait l'histoire DEUX
    fois — chaque côté ne verrait qu'une frame sur deux, décalées l'une de
    l'autre. L'appariement que cette démo existe pour montrer serait faux par
    construction, sans lever. Le verrou d'appariement le prouve sur pièce.

    L'AVANCE SE FAIT SUR L'IDENTITÉ DU PREMIER NIVEAU, PAS SUR UN COMPTEUR.
    `frame()` appelle `pas_f` une fois par niveau, dans l'ordre de
    `geo.niveaux_gpu`, et INCONDITIONNELLEMENT — l'appel est hors du test de
    déplacement. Un compteur modulo le nombre de niveaux marcherait donc
    aujourd'hui et se désynchroniserait en silence à la première surprise. Se
    caler sur l'identité du tableau du premier niveau ne dérive pas."""

    def __init__(self, images: np.ndarray):
        self.images = images
        self.indice = 0
        self.pyramide = None
        self._premier = None

    def attacher(self, pyramide) -> None:
        """La pyramide n'existe pas encore quand `pas_f` lui est passé — d'où
        cette seconde étape, explicite plutôt que devinée."""
        self.pyramide = pyramide
        self._premier = pyramide.fenetres[list(pyramide.geo.niveaux_gpu)[0]]

    def _niveau_de(self, fenetres) -> int:
        for niveau, bloc in self.pyramide.fenetres.items():
            if bloc is fenetres:
                return niveau
        raise NiveauNonReconnu(
            "PasHistoire : bloc de fenêtres non situé dans la pyramide "
            "attachée. Ne rien faire laisserait le monde immobile sans que rien "
            "ne le dise — une démo figée est indiscernable d'une démo correcte.")

    def __call__(self, fenetres, xp, sortie=None):
        niveau = self._niveau_de(fenetres)
        if fenetres is self._premier:
            self.indice = (self.indice + PAS_HISTOIRE) % len(self.images)
            self.pyramide.monde0[:, INDICE_CHAMP_S] = self.images[self.indice]

        geo = self.pyramide.geo
        bloc = predire_bloc_cpu(
            self.pyramide.monde0, geo, niveau,
            geo.origines(niveau, self.pyramide.centre_fin), 0, geo.n_fov,
            geo.n_systemes_du_niveau(niveau))

        if sortie is None:
            sortie = xp.empty_like(fenetres)
        if sortie is not fenetres:
            sortie[...] = fenetres
        sortie[:, :, INDICE_CHAMP_S] = xp.asarray(bloc[:, :, INDICE_CHAMP_S])
        return sortie, 0.0


def construire_pyramide(images: np.ndarray) -> tuple[PyramideFovea, PasHistoire]:
    """Une pyramide amorcée SUR L'HISTOIRE, et pas sur le monde jetable.

    POURQUOI L'AMORÇAGE, ET IL A ÉTÉ VÉRIFIÉ SUR LE PREMIER COUPLE. Le readout
    capture `s_prev` post-roll et pré-`pas_f`. Sans amorçage, le `s_prev` du tout
    premier tick serait le monde JETABLE descendu par `__init__`, tandis que
    `s_cur` serait déjà la frame 0 de l'histoire : l'écran intermédiaire fondrait
    deux mondes sans rapport, une fois, au début — le genre d'artefact qu'on
    attribue ensuite à l'interpolation. On écrit donc la frame 0 dans `monde0` et
    on redescend les fenêtres, exactement comme `__init__` le fait."""
    geo = GeometriePyramide(n_fov=N_FOV, n_niv=N_NIV, n0=N0)
    transferts = TransfertComptable(np)
    pas = PasHistoire(images)
    pyramide = PyramideFovea(np, geo, transferts, pas_f=pas)

    pyramide.monde0[:, INDICE_CHAMP_S] = images[0]
    for niveau in geo.niveaux_gpu:
        bloc = predire_bloc_cpu(
            pyramide.monde0, geo, niveau,
            geo.origines(niveau, pyramide.centre_fin), 0, geo.n_fov,
            geo.n_systemes_du_niveau(niveau))
        pyramide.fenetres[niveau][...] = np.asarray(bloc)
        pyramide.references[niveau][...] = np.asarray(bloc)

    pas.attacher(pyramide)
    transferts.frame_suivante()
    return pyramide, pas


def rendre_couple(s_half: float, n_ticks: int, dossier: Path) -> int:
    """Les DEUX côtés en parallèle → `4 · n_ticks` PNG. Rend le nombre écrit.

    L'APPARIEMENT EST STRUCTUREL, PAS DÉCLARÉ. Les deux boucles partent du MÊME
    monde, avancent du MÊME balayage fovéal et consomment la MÊME histoire, chacune
    avec sa propre instance de `PasHistoire` ; à index égal, les deux séries
    montrent le même instant. La preuve la plus dure en est que l'écran EXACT d'un
    tick est le même des deux côtés — `interpoler` le rend en second, `extrapoler`
    en premier, et ce sont les mêmes octets. C'est ce que vérifie le verrou
    d'appariement, et il tombe si l'une des deux histoires dérive d'une frame.

    AUCUNE LEVÉE N'EST RATTRAPÉE ICI : un trou de couverture du gather traverse."""
    if n_ticks <= 0:
        raise ValueError(
            f"rendre_couple : n_ticks={n_ticks} <= 0. Une démo qui n'écrit "
            "aucune image imprimerait « 0 images écrites » et ressemblerait à un "
            "run propre — un silence, pas un résultat.")

    try:
        dossier.mkdir(parents=True, exist_ok=False)
    except FileExistsError as cause:
        raise DossierDejaPresent(
            f"rendre_couple : {dossier} existe déjà. Ce driver n'écrit jamais "
            "par-dessus un artefact existant — donnez un dossier neuf (c'est "
            "l'appelant qui horodate le nom, pas ce fichier).") from cause

    images = charger_profondeur()
    boucles = {}
    for cote in COTES:
        pyramide, _ = construire_pyramide(images)
        boucles[cote] = BoucleRendu(pyramide, cote)

    compte = 0
    for tick in range(n_ticks):
        for cote in COTES:
            for rang, ecran in enumerate(boucles[cote].tick(COTE_PX_DEMO)):
                indice = 2 * tick + rang
                chemin = dossier / f"ecran_{cote}_{indice:04d}.png"
                chemin.write_bytes(
                    encoder_png_gris(quantifier_en_octets(ecran, s_half)))
                compte += 1
    return compte


def construire_parseur() -> argparse.ArgumentParser:
    """`--s-half` et `--dossier` sont OBLIGATOIRES et sans défaut ; `--cote` n'existe
    pas, puisque ce driver fait tourner les deux."""
    parseur = argparse.ArgumentParser(
        description="Démo 2:1 v2 : les deux côtés, sur l'inondation.")
    parseur.add_argument("--s-half", required=True, type=float)
    parseur.add_argument("--dossier", required=True, type=Path)
    parseur.add_argument("--ticks", type=int, default=N_TICKS_DEFAUT)
    return parseur


def main(argv: list[str] | None = None) -> int:
    """LE SEUL CHIFFRE IMPRIMÉ EST LE NOMBRE D'IMAGES ÉCRITES."""
    options = construire_parseur().parse_args(argv)
    ecrites = rendre_couple(options.s_half, options.ticks, options.dossier)
    print(f"{ecrites} images écrites")
    return 0


if __name__ == "__main__":
    sys.exit(main())
