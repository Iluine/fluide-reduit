"""DÉMO 2:1 v3 — UNE INONDATION QUI ÉPOUSE LE TERRAIN RÉEL DU DÉPÔT.

    .venv/bin/python -m scripts.run_demo_2a1_v3 \\
        --dossier outputs/demo_2a1_v3_2026-08-27 --ticks 120

CE QUE LA v2 NE MONTRAIT PAS. `run_demo_2a1_v2.py` fait tourner les DEUX côtés
sur une inondation REJOUÉE — 201 frames de `dam_break.npz`, décimées d'un facteur
4, posées dans le carré central du monde. L'œil y voit un front, mais un front
DANS UNE BOÎTE : 16 cellules de monde sur 128, une bathymétrie plate, et une
histoire qui a été calculée ailleurs, pour autre chose. Trois manques, traités
ici un par un — l'échelle, la scène, le rendu.

`run_demo_2a1.py` ET `run_demo_2a1_v2.py` NE SONT PAS TOUCHÉS. C'est un troisième
frère, pas une réécriture. La garde `DossierDejaPresent` et la mécanique des
chunks PNG sont IMPORTÉES d'eux : deux copies d'une garde ou d'un encodeur
dérivent l'une de l'autre sans que rien ne le signale.

LE CÔTÉ EST TRANCHÉ, ET CE FICHIER NE FAIT PLUS TOURNER QUE LUI. `interpoler`.
La v2 existait pour donner à ce choix de quoi être révisé à l'œil ; il l'a été,
et le motif est l'image. À `α = 0,5` l'écran intermédiaire est une combinaison
CONVEXE de deux états qui ont réellement existé — aucune valeur inventée, aucun
clamp possible par construction — là où `α = 1,5` extrapole au-delà de `n` et
peut sortir de la bande. SA DETTE EST NOMMÉE ET NON PAYÉE : `interpoler` rend
l'écran intermédiaire en RETARD d'un demi pas de monde sur l'état le plus récent.
Cette latence ne se juge pas sur des images ; elle se juge à la main d'un joueur,
et il n'y a pas d'entrée dans ce moteur. Le jour où il y en aura une, c'est là
qu'il faudra l'éprouver — pas ici.

── 1. L'ÉCHELLE ────────────────────────────────────────────────────────────
LA FOVÉA PLAFONNE CE QU'UN ÉCRAN PEUT MONTRER, et c'est ce plafond qui commande
toute la géométrie de cette démo. L'écran est une fenêtre carrée en coordonnées
du niveau FIN (`gather_chemin_de_cout`) ; la fenêtre la plus large est celle du
niveau GPU le plus grossier, le niveau 1, qui couvre `n_fov / 2` cellules de
monde. Avec le `n_fov = 64` des deux frères, AUCUN écran ne peut montrer plus de
32 cellules de monde — quelle que soit la taille du monde, quel que soit
`cote_px`. C'est la raison structurelle pour laquelle la v2 était petite, et
aucune composition de scène ne l'aurait levée.

`n_fov` EST DONC MONTÉ À 256, ET C'EST LE SEUL PARAMÈTRE DE GÉOMÉTRIE TOUCHÉ.
La liste de slots reste le défaut du module (enveloppe dense tranche-1, γ₂ = 3
slots par niveau GPU) — inventer une liste de slots serait une géométrie NEUVE,
jamais mesurée. `n_fov` gouverne la seule grandeur en cause : combien de monde
une fenêtre voit. Le plafond passe à 128 cellules.

LES QUATRE NOMBRES CI-DESSOUS ONT ÉTÉ MESURÉS ENSEMBLE, PAS CHOISIS SÉPARÉMENT.
`n_niv = 3` (niveau 0 CPU + niveaux GPU 1 et 2) donne `2**j_fin = 4` pixels
d'écran par cellule de monde : `cote_px = 384` montre donc 96 cellules, sous le
plafond de 128 avec la marge que le balayage exige. Ce balayage n'est pas un
détail — `BoucleRendu` avance `centre_fin` d'un pixel fin par tick et la caméra
DÉRIVE de 30 cellules de monde sur 120 ticks, en diagonale : les origines en `x`
suivent le centre fovéal, celles en `y` sont FIXES, et l'écran glisse donc à
travers des fenêtres immobiles en `y`. C'est cette dérive qui mange la marge, et
elle a été exercée plutôt que supposée : `384` tient les 120 ticks, `448` en
tient 96 puis LÈVE, `512` lève dès le premier. Choisir la valeur qui tient de
justesse aurait donné une démo qui casse quand on allonge le run. Un trou de
couverture LÈVE et n'est pas rattrapé.

── 2. LA SCÈNE ─────────────────────────────────────────────────────────────
LA BATHYMÉTRIE EST LE TERRAIN RÉEL DU DÉPÔT, `sediment.default_terrain` — celui
des tests `test_f1_substrat_fidele.py`, asset figé, appelé et jamais recopié.
Il est paramétré par la grille et il est HONNÊTE de dire ce qu'il est à cette
taille : un plan incliné de chute totale 0,5 le long de `x`, plus TROIS bosses
gaussiennes (σ = 6 cellules, amplitudes 0,25 / 0,30 / 0,20). Il N'A PAS DE
VALLÉES — le brief en espérait, ce terrain-là n'en porte pas, et aucune n'a été
ajoutée. Ce qu'il porte est un versant et trois obstacles, et c'est autour de
ces obstacles que l'eau se sépare.

CE QUI REND LE RELIEF VISIBLE N'EST PAS LA TAILLE DE LA GRILLE, C'EST LA
PROFONDEUR D'EAU — et c'est le fait qui a décidé la scène. Les bosses culminent
à 0,30 au-dessus du plan ; sous deux mètres d'eau, comme dans `dam_break.npz`,
elles sont noyées et invisibles. La retenue est donc posée à `η = 0,62`, ce qui
donne une lame de 0,15 à 0,20 : les bosses ÉMERGENT, l'eau les contourne, et le
sillage qu'elle laisse derrière chacune est ce que l'œil suit.

σ EST EN CELLULES ABSOLUES, PAS EN FRACTION DU DOMAINE — d'où le monde de 160 et
non de 512. Agrandir la grille RÉTRÉCIT les bosses relativement au domaine : à
256 cellules elles ne font plus que 2 % du côté et le terrain redevient une
rampe nue. « La grille la plus grande que le CPU tient » n'est donc pas le bon
critère ici ; le CPU tient bien plus grand que ce que ce terrain-là supporte.
160 cellules de monde, 96 visibles, 4 pixels par cellule : une bosse fait une
centaine de pixels de large à l'écran, les trois sont dans le cadre au départ.

── 3. LE RENDU ─────────────────────────────────────────────────────────────
DEUX COUCHES, DEUX CHAMPS DE LA PYRAMIDE, UN SEUL GATHER CHACUNE — et rien de la
géométrie multi-niveaux n'est réimplémenté (règle §A17). Le terrain est écrit
dans le champ d'indice 0 et l'eau dans le champ `s`. Les deux sortent de la MÊME
pyramide, au MÊME `centre_fin`, par le MÊME `gather_chemin_de_cout` : ils sont
alignés par construction et non par une arithmétique d'écran recopiée ici.

CE QUE LES CHAMPS PORTENT, DIT FRANCHEMENT. La pyramide porte
`(h, hu, hv, s)`. Le readout capture le quatrième, `s`, le sédiment : c'est donc
dans `s` que la PROFONDEUR D'EAU est écrite, comme dans la v2, et pour la même
raison — c'est le seul champ que `melanger` sait interpoler. Le champ d'indice 0
est nommé `h` dans cet ordre ; il sert ici de PORTEUR STATIQUE pour la
BATHYMÉTRIE. Ni l'un ni l'autre ne devient ce que son nom dit : ces images ne
sont pas un couplage Exner, et le champ 0 ne porte aucune hauteur d'eau. Le
nommer évite qu'on les lise plus tard comme autre chose.

LE PORTEUR STATIQUE SE TIENT TOUT SEUL À TRAVERS LE BALAYAGE, et c'est pourquoi
`pas_f` n'a pas à s'en occuper : `frame()` roule les fenêtres en `x` et redescend
la bande entrante depuis `monde0` pour TOUS les champs. Le terrain écrit une fois
dans `monde0` y reste juste, tick après tick.

L'ENCODEUR PNG EST RGB ET IL EST ÉCRIT ICI, SUR LES CHUNKS IMPORTÉS. Pillow est
présent dans le venv mais `requirements.txt:21` le laisse COMMENTÉ — « optionnel,
export GIF » — et une démo n'a pas à dépendre d'un paquet non déclaré. Ce qui est
importé est ce qui existe : `_chunk_png`, la signature, le niveau de compression,
l'octet de filtre. Ce qui change tient en un octet du header, le type couleur
`2` au lieu de `0`, et en trois canaux par pixel au lieu d'un.

AUCUNE MESURE, AUCUN CHIFFRE ABSOLU EN SORTIE, et la phrase est étroite comme
chez les deux frères. Un chronomètre tourne par transitivité
(`TransfertComptable._chronometrer`) ; ce qui tient le régime §A64 est qu'aucun
chiffre n'en sorte. Le SEUL chiffre imprimé est le nombre d'images écrites.

ET LA NON-LECTURE D'HORLOGE EST ICI UN FAIT, PAS UNE SERRURE — dit franchement
parce que les deux frères, eux, la MÉCANISENT. Aucun module d'horloge n'est
importé ci-dessous, ce qui se vérifie en lisant la liste d'imports ; mais le
brief de cette v3 fixe trois verrous et pas quatre, et le contrôle de source des
frères n'en fait pas partie. Une prose qui promettrait ici une garde absente
vaudrait moins que ce constat étroit.

L'HORODATAGE EST CHEZ L'APPELANT, acquis des frères qu'on ne rediagnostique pas :
`--dossier` est obligatoire, et un dossier existant est REFUSÉ.
"""
from __future__ import annotations

import argparse
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

from config import GridConfig
from scripts.run_demo_2a1 import DossierDejaPresent
from scripts.run_boucle_rendu_demo import (
    NIVEAU_ZLIB,
    OCTET_FILTRE_AUCUN,
    PROFONDEUR_BITS,
    SIGNATURE_PNG,
    EcranNonRepresentable,
    _chunk_png,
)
from src.f1_gpu.boucle_rendu import COTE_INTERPOLER, BoucleRendu
from src.f1_gpu.chemin_de_cout import gather_chemin_de_cout
from src.f1_gpu.interpolation_readout import INDICE_CHAMP_S
from src.f1_gpu.pyramide import (
    GeometriePyramide,
    PyramideFovea,
    predire_bloc_cpu,
)
from scipy.ndimage import uniform_filter

from src.f1_gpu.transferts import TransfertComptable
from src.sediment import default_terrain
from src.solver_wetdry import simulate_wetdry_o2

# LE CÔTÉ, IMPORTÉ ET JAMAIS RECOPIÉ EN LITTÉRAL. Un littéral gravé ici donnerait
# un driver qui accepte un côté que la boucle refuse.
COTE: str = COTE_INTERPOLER

# ── LA GÉOMÉTRIE — voir le §1 de l'en-tête ──────────────────────────────────
N_FOV: int = 256         # plafond de vision = n_fov / 2 = 128 cellules de monde
N_NIV: int = 3           # niveau 0 CPU + niveaux GPU 1 et 2 ; 4 px par cellule
N0: int = 160            # côté du monde, en cellules
COTE_PX: int = 384       # 96 cellules visibles ; 512 ne tient pas la couverture
N_TICKS_DEFAUT: int = 120

# ── LA SCÈNE — voir le §2 de l'en-tête ──────────────────────────────────────
# LA RETENUE. `η = 0,62` contre un terrain qui monte à 0,60 : la lame fait 0,15 à
# 0,20 au pied du mur, les bosses (0,30 au-dessus du plan) émergent. Le mur est à
# 33 % du côté, soit la cellule 52 — DANS le cadre initial, qui commence à la
# cellule 32 : la retenue occupe un cinquième du cadre au départ, la rupture se
# voit, et l'eau n'arrive pas déjà faite par le bord. La première bosse (centre
# en `x = 48`) est SOUS la retenue, à deux centièmes près : elle affleure, et le
# front la contourne dès le premier tick au lieu de l'atteindre à mi-course.
NIVEAU_RETENUE: float = 0.62
FRACTION_MUR: float = 0.33

# LE PAS DE MONDE, EN TEMPS PHYSIQUE. Le front court à `sqrt(g·h) ≈ 1,4` cellule
# par seconde ; traverser les 96 cellules du cadre demande donc ~70 s de temps
# physique, soit 0,6 s par tick sur 120 ticks. Mesuré : à 0,6, le compte de
# cellules mouillées passe de 8 320 à ~23 500 et PLAFONNE vers le tick 60, quand
# la lame a rempli le cadre. Les soixante ticks suivants ne sont donc plus de la
# conquête mais de l'étalement — ils sont gardés parce que c'est là que l'eau se
# range autour des bosses, et que c'est ce rangement qu'on vient voir. Le pas de
# temps interne reste celui du solveur (CFL adaptative) ; celui-ci n'est qu'une
# durée d'intégration.
DT_TICK: float = 0.6

# ── LE RENDU — voir le §3 de l'en-tête ──────────────────────────────────────
# LE CHAMP D'INDICE 0 EST LE PORTEUR STATIQUE DE LA BATHYMÉTRIE. Il s'appelle `h`
# dans l'ordre `(h, hu, hv, s)` et il ne porte AUCUNE hauteur d'eau ici.
INDICE_CHAMP_PORTEUR_TERRAIN: int = 0

# L'OMBRAGE. Azimut et élévation d'un hillshade classique. L'EXAGÉRATION N'EST
# PAS UN TRUCAGE mais un changement d'unité assumé : le relief vaut 0,60 sur 160
# cellules de 1 unité de côté, soit une pente de 0,4 % — sans exagération, le
# terrain sort uniformément gris et l'ombrage ne montre rien. Elle est appliquée
# à la PENTE, jamais à l'altitude ; aucune altitude affichée n'en dépend, puisque
# aucune altitude n'est affichée.
AZIMUT_SOLEIL: float = 315.0
ELEVATION_SOLEIL: float = 45.0
EXAGERATION_RELIEF: float = 120.0

# LA PENTE SE LIT SUR UNE CELLULE DE MONDE, PAS SUR UN PIXEL — et c'est un
# correctif, pas un réglage. Le gather échantillonne au PLUS PROCHE VOISIN
# (`coordonnée_fine // 2**(j_fin − j)`) : l'écran terrain est un ESCALIER, plat
# sur `2**j_fin` pixels puis marche. Un gradient pris pixel à pixel y lit donc
# zéro trois fois sur quatre et une marche la quatrième — ce qui sort en damier
# régulier sur tout le relief, constaté au premier essai. Moyenner sur la
# largeur exacte d'une cellule de monde avant de dériver rend la pente du
# TERRAIN et non celle de l'escalier. La moyenne ne touche QUE l'ombrage :
# aucune profondeur d'eau ne passe par là.

# L'EAU. Deux échelles distinctes et c'est délibéré. L'OPACITÉ sature vite
# (`1 − exp(−h/0,02)`) parce que la lame est mince presque partout — mesuré entre
# 0,01 et 0,05 sur l'essentiel du domaine — et qu'une opacité linéaire y rendrait
# l'eau invisible. La TEINTE, elle, se déploie sur toute la profondeur utile
# (0,25) : c'est elle qui porte l'information de profondeur, du cyan pâle des
# bords au bleu sombre des creux.
PROFONDEUR_OPACITE: float = 0.02
PROFONDEUR_TEINTE: float = 0.25
SEUIL_MOUILLE: float = 1e-3      # celui du solveur : sous lui, la cellule est sèche

ROCHE_SECHE: tuple[float, float, float] = (0.52, 0.47, 0.40)
ROCHE_LUMIERE: tuple[float, float, float] = (0.40, 0.38, 0.34)
EAU_FAIBLE: tuple[float, float, float] = (0.42, 0.72, 0.82)
EAU_PROFONDE: tuple[float, float, float] = (0.04, 0.17, 0.45)

TYPE_COULEUR_RGB: int = 2        # PNG « truecolor », trois canaux par pixel
NIVEAUX: float = 255.0


class NiveauNonReconnu(ValueError):
    """Levée quand `pas_f` reçoit un bloc de fenêtres qu'il ne sait pas situer.

    FAIL-LOUD PLUTÔT QUE PAS SAUTÉ, comme dans la v2. `PasInondation` identifie
    le niveau par IDENTITÉ du tableau de fenêtres ; si l'identité ne correspond à
    aucun niveau, la seule alternative serait de ne rien faire, c'est-à-dire de
    laisser le monde immobile sans que rien ne le dise. Une démo figée qui ne
    lève pas est indiscernable d'une démo correcte dont le champ n'évolue pas.

    Hérite de `ValueError` comme les autres serrures de ce chantier : ce qui
    hérite de `RuntimeError` risque d'être avalé par un `except RuntimeError`
    d'appelant."""


def terrain_du_monde() -> np.ndarray:
    """La bathymétrie : l'asset figé du dépôt, APPELÉ et jamais recopié."""
    return default_terrain(GridConfig(H=N0, W=N0)).astype(np.float64)


def retenue_initiale(terrain: np.ndarray) -> np.ndarray:
    """La condition initiale : une retenue à `η` constant derrière un mur en `x`.

    `max(η − b, 0)` et non `η − b` : au-delà du mur, et partout où le terrain
    dépasse déjà `η`, la cellule est SÈCHE. C'est ce qui donne à la retenue un
    bord épousant le relief plutôt qu'une marche rectangulaire, et c'est le bord
    que la rupture met en mouvement.

    Le terrain descend vers les `x` croissants (`default_terrain` : chute de 0,5
    de `x = 0` à `x = W−1`), donc la retenue est à l'AMONT et l'eau part vers la
    droite — dans le même sens que le balayage fovéal, qui suit le front."""
    profondeur = np.zeros_like(terrain)
    mur = int(FRACTION_MUR * N0)
    profondeur[:, :mur] = np.maximum(NIVEAU_RETENUE - terrain[:, :mur], 0.0)
    return profondeur


class PasInondation:
    """« Le monde a avancé, relis-le » — le `pas_f` de cette démo.

    CE QUI AVANCE ICI EST UN SOLVEUR, PAS UNE BANDE REJOUÉE. La v2 lisait la
    frame suivante d'un `.npz` ; ici chaque tick intègre `simulate_wetdry_o2` sur
    `DT_TICK` de temps physique et repart de l'état rendu. C'est le solveur
    mouillé/sec well-balanced du dépôt, appelé et jamais recopié — aucun schéma
    n'est écrit dans ce fichier.

    L'INTÉGRATION EST DÉCOUPÉE PAR TICK, ET CE N'EST PAS UNE COMMODITÉ.
    `simulate_wetdry_o2` empile un instantané par pas accepté : sur 120 ticks
    d'un seul appel, la pile ferait plusieurs gigaoctets pour un seul état
    utile. Découper borne la mémoire au coût d'une troncature de `dt` en fin de
    chaque tranche — le solveur repart de l'état exact qu'il vient de rendre.

    L'AVANCE SE FAIT SUR L'IDENTITÉ DU PREMIER NIVEAU, PAS SUR UN COMPTEUR.
    `frame()` appelle `pas_f` une fois par niveau, dans l'ordre de
    `geo.niveaux_gpu`, et INCONDITIONNELLEMENT — l'appel est hors du test de
    déplacement. Un compteur modulo le nombre de niveaux marcherait aujourd'hui
    et se désynchroniserait en silence à la première surprise ; se caler sur
    l'identité du tableau du premier niveau ne dérive pas.

    LE TERRAIN N'EST PAS TOUCHÉ ICI. `sortie[...] = fenetres` recopie tous les
    champs et seul `s` est réécrit : le porteur statique traverse intact, et
    `frame()` redescend lui-même la bande entrante du balayage."""

    def __init__(self, terrain: np.ndarray):
        self.terrain = terrain
        self.profondeur = retenue_initiale(terrain)
        self.debit_x = np.zeros_like(terrain)
        self.debit_y = np.zeros_like(terrain)
        self.grille = GridConfig(H=N0, W=N0)
        self.pyramide = None
        self._premier = None

    def masse(self) -> float:
        """La masse d'eau du domaine — lue par les verrous, jamais imprimée."""
        return float(self.profondeur.sum())

    def cellules_mouillees(self) -> int:
        """Le compte de cellules au-dessus du seuil sec du solveur."""
        return int((self.profondeur > SEUIL_MOUILLE).sum())

    def attacher(self, pyramide) -> None:
        """La pyramide n'existe pas encore quand `pas_f` lui est passé — d'où
        cette seconde étape, explicite plutôt que devinée."""
        self.pyramide = pyramide
        self._premier = pyramide.fenetres[list(pyramide.geo.niveaux_gpu)[0]]

    def avancer(self) -> None:
        """Un pas de monde : `DT_TICK` de temps physique, état repris en place."""
        _, profondeurs, debits_x, debits_y = simulate_wetdry_o2(
            self.profondeur, self.debit_x, self.debit_y, self.terrain,
            self.grille, t_end=DT_TICK)
        self.profondeur = profondeurs[-1]
        self.debit_x = debits_x[-1]
        self.debit_y = debits_y[-1]

    def _niveau_de(self, fenetres) -> int:
        for niveau, bloc in self.pyramide.fenetres.items():
            if bloc is fenetres:
                return niveau
        raise NiveauNonReconnu(
            "PasInondation : bloc de fenêtres non situé dans la pyramide "
            "attachée. Ne rien faire laisserait le monde immobile sans que rien "
            "ne le dise — une démo figée est indiscernable d'une démo correcte.")

    def __call__(self, fenetres, xp, sortie=None):
        niveau = self._niveau_de(fenetres)
        if fenetres is self._premier:
            self.avancer()
            self.pyramide.monde0[:, INDICE_CHAMP_S] = self.profondeur

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


def construire_pyramide() -> tuple[PyramideFovea, PasInondation]:
    """Une pyramide amorcée SUR LA SCÈNE, terrain et retenue déjà en place.

    POURQUOI L'AMORÇAGE. Le readout capture `s_prev` post-roll et pré-`pas_f`.
    Sans amorçage, le `s_prev` du tout premier tick serait le monde JETABLE
    descendu par `__init__` tandis que `s_cur` serait déjà la retenue : l'écran
    intermédiaire fondrait deux mondes sans rapport, une fois, au début — le
    genre d'artefact qu'on attribue ensuite à l'interpolation. On écrit donc la
    scène dans `monde0` et on redescend les fenêtres, exactement comme
    `__init__` le fait.

    LES DEUX CHAMPS SONT AMORCÉS ENSEMBLE. Le terrain n'a pas d'autre occasion
    d'entrer dans les fenêtres : `pas_f` ne le réécrit jamais."""
    geo = GeometriePyramide(n_fov=N_FOV, n_niv=N_NIV, n0=N0)
    transferts = TransfertComptable(np)
    pas = PasInondation(terrain_du_monde())
    pyramide = PyramideFovea(np, geo, transferts, pas_f=pas)

    pyramide.monde0[:, INDICE_CHAMP_PORTEUR_TERRAIN] = pas.terrain
    pyramide.monde0[:, INDICE_CHAMP_S] = pas.profondeur
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


def pixels_par_cellule() -> int:
    """Combien de pixels d'écran vaut une cellule de monde — DÉRIVÉ de la
    géométrie, jamais gravé en littéral. L'écran vit au niveau fin ; une cellule
    de monde y occupe `2**j_fin` pixels. Un littéral ici se désaccorderait en
    silence le jour où `N_NIV` bouge, et l'ombrage relirait l'escalier."""
    return 2 ** GeometriePyramide(n_fov=N_FOV, n_niv=N_NIV, n0=N0).niveau_fin


def ombrage(terrain_ecran: np.ndarray) -> np.ndarray:
    """Hillshade classique sur l'écran terrain → `[0, 1]`.

    Le gradient est pris SUR L'ÉCRAN, en pixels, APRÈS moyenne sur la largeur
    d'une cellule de monde — voir le commentaire de `EXAGERATION_RELIEF` : sans
    elle, la dérivée lit l'escalier du plus proche voisin et sort en damier.
    Rien ici ne prétend à une pente physique — aucune n'est mesurée ni imprimée.

    `uniform_filter` est APPELÉ et non recopié ; `scipy` est déclaré
    (`requirements.txt:3`), contrairement à Pillow."""
    lisse = uniform_filter(np.asarray(terrain_ecran, dtype=np.float64),
                           size=pixels_par_cellule())
    dz_y, dz_x = np.gradient(lisse * EXAGERATION_RELIEF)
    pente = np.arctan(np.hypot(dz_x, dz_y))
    aspect = np.arctan2(-dz_x, dz_y)
    azimut = np.deg2rad(360.0 - AZIMUT_SOLEIL + 90.0)
    elevation = np.deg2rad(ELEVATION_SOLEIL)
    clair = (np.sin(elevation) * np.cos(pente)
             + np.cos(elevation) * np.sin(pente) * np.cos(azimut - aspect))
    return np.clip(clair, 0.0, 1.0)


def composer_rgb(terrain_ecran: np.ndarray,
                 eau_ecran: np.ndarray) -> np.ndarray:
    """Les deux écrans → un tableau `uint8` `(h, w, 3)`.

    LE CONTRÔLE DE FINITUDE PORTE SUR LES ENTRÉES, ET C'EST LOAD-BEARING. Un
    contrôle posé sur la couleur composée serait AVEUGLE au `+inf` : une
    profondeur infinie donne une opacité de 1,0, valeur finie, dans la bande, qui
    quantifie en un pixel d'eau profonde parfaitement ordinaire. Le pixel
    silencieux ne se voit qu'à l'entrée. C'est le motif de `quantifier_en_octets`
    chez le frère, et le précédent de la maison (`src/arcC_rendu.py:57-60`) :
    FAIL-LOUD, jamais un clip silencieux.

    LA PROFONDEUR NÉGATIVE EST UNE FAUTE, PAS UNE VALEUR À RATTRAPER. L'écran
    interpolé ne peut pas arriver négatif — `melanger` clampe `s ≥ 0` et le
    compte — mais l'écran EXACT est un gather DIRECT sur la pyramide : il ne
    passe par aucun clamp et porte le `s` de la physique tel quel."""
    for nom, champ in (("terrain", terrain_ecran), ("eau", eau_ecran)):
        if not np.isfinite(champ).all():
            raise EcranNonRepresentable(
                f"écran {nom} non fini : "
                f"{int((~np.isfinite(champ)).sum())} cellule(s) sur "
                f"{champ.size} valent NaN ou ±inf. Une composition les rendrait "
                "en pixels d'apparence normale — la signature §A53. C'est un "
                "BUG AMONT, pas une donnée à rattraper en douce.")
    if float(np.min(eau_ecran)) < 0.0:
        raise EcranNonRepresentable(
            f"profondeur d'eau négative : min={float(np.min(eau_ecran))}. La "
            "physique a rendu une lame négative — BUG AMONT "
            "(`src/arcC_rendu.py:57-60`), jamais un clip silencieux.")

    profondeur = np.asarray(eau_ecran, dtype=np.float64)
    clair = ombrage(terrain_ecran)[..., None]
    roche = (np.asarray(ROCHE_SECHE, dtype=np.float64)
             + np.asarray(ROCHE_LUMIERE, dtype=np.float64) * clair)

    opacite = (1.0 - np.exp(-profondeur / PROFONDEUR_OPACITE))[..., None]
    teinte = np.clip(profondeur / PROFONDEUR_TEINTE, 0.0, 1.0)[..., None]
    eau = ((1.0 - teinte) * np.asarray(EAU_FAIBLE, dtype=np.float64)
           + teinte * np.asarray(EAU_PROFONDE, dtype=np.float64))

    mouille = (profondeur > SEUIL_MOUILLE)[..., None]
    image = np.where(mouille, roche * (1.0 - opacite) + eau * opacite, roche)
    return np.rint(np.clip(image, 0.0, 1.0) * NIVEAUX).astype(np.uint8)


def encoder_png_rgb(octets: np.ndarray) -> bytes:
    """Un tableau `uint8` `(hauteur, largeur, 3)` → les octets d'un PNG RGB.

    LE MÊME PNG QUE CELUI DU FRÈRE, À DEUX CHOSES PRÈS : le type couleur vaut
    `2` (truecolor) au lieu de `0`, et une ligne fait `3 · largeur` octets. Le
    reste — signature, chunks, CRC, filtre `0`, niveau de compression fixé —
    est IMPORTÉ et non recopié. Le filtre nul rend l'encodage trivialement
    relisible : un test rouvre le fichier sans bibliothèque d'image.

    DÉTERMINISTE PAR CONSTRUCTION : rien ici ne lit d'horloge, d'adresse ou
    d'environnement, et aucun chunk `tIME` n'est écrit."""
    if octets.ndim != 3 or octets.shape[-1] != 3 or octets.dtype != np.uint8:
        raise EcranNonRepresentable(
            f"image de forme {octets.shape} et de dtype {octets.dtype} : "
            "attendu un tableau `(h, w, 3)` `uint8`. Une mauvaise forme est un "
            "BUG AMONT — l'encoder quand même écrirait un fichier dont les "
            "octets ne sont pas les pixels.")

    hauteur, largeur, _ = octets.shape
    entete = struct.pack(">IIBBBBB", largeur, hauteur, PROFONDEUR_BITS,
                         TYPE_COULEUR_RGB, 0, 0, 0)
    brut = b"".join(bytes([OCTET_FILTRE_AUCUN]) + ligne.tobytes()
                    for ligne in octets)
    return (SIGNATURE_PNG
            + _chunk_png(b"IHDR", entete)
            + _chunk_png(b"IDAT", zlib.compress(brut, NIVEAU_ZLIB))
            + _chunk_png(b"IEND", b""))


def rendre(n_ticks: int, dossier: Path) -> int:
    """Le côté `interpoler` seul → `2 · n_ticks` PNG. Rend le nombre écrit.

    LE TERRAIN EST GATHERÉ UNE FOIS PAR TICK, APRÈS `tick()`, ET C'EST L'ORDRE
    QUI LE REND JUSTE. `tick()` appelle `frame()` en premier — le centre fovéal
    n'est donc plus déplacé quand les deux écrans d'eau sont rendus (§4-6 : un
    seul centre par tick, pour deux écrans). Gatherer le terrain après le tick le
    prend au MÊME `centre_fin` que les deux écrans qu'il habille.

    AUCUNE LEVÉE N'EST RATTRAPÉE ICI : un trou de couverture du gather traverse."""
    if n_ticks <= 0:
        raise ValueError(
            f"rendre : n_ticks={n_ticks} <= 0. Une démo qui n'écrit aucune "
            "image imprimerait « 0 images écrites » et ressemblerait à un run "
            "propre — un silence, pas un résultat.")

    try:
        dossier.mkdir(parents=True, exist_ok=False)
    except FileExistsError as cause:
        raise DossierDejaPresent(
            f"rendre : {dossier} existe déjà. Ce driver n'écrit jamais "
            "par-dessus un artefact existant — donnez un dossier neuf (c'est "
            "l'appelant qui horodate le nom, pas ce fichier).") from cause

    pyramide, _ = construire_pyramide()
    boucle = BoucleRendu(pyramide, COTE)

    compte = 0
    for tick in range(n_ticks):
        ecrans = boucle.tick(COTE_PX)
        terrain_ecran = gather_chemin_de_cout(
            pyramide, COTE_PX, indice_champ=INDICE_CHAMP_PORTEUR_TERRAIN)
        for rang, eau_ecran in enumerate(ecrans):
            indice = 2 * tick + rang
            chemin = dossier / f"ecran_{COTE}_{indice:04d}.png"
            chemin.write_bytes(
                encoder_png_rgb(composer_rgb(terrain_ecran, eau_ecran)))
            compte += 1
    return compte


def construire_parseur() -> argparse.ArgumentParser:
    """`--dossier` est OBLIGATOIRE et sans défaut ; `--cote` n'existe pas,
    puisque le côté est tranché et qu'il n'y a plus rien à comparer."""
    parseur = argparse.ArgumentParser(
        description="Démo 2:1 v3 : une inondation sur le terrain réel.")
    parseur.add_argument("--dossier", required=True, type=Path)
    parseur.add_argument("--ticks", type=int, default=N_TICKS_DEFAUT)
    return parseur


def main(argv: list[str] | None = None) -> int:
    """LE SEUL CHIFFRE IMPRIMÉ EST LE NOMBRE D'IMAGES ÉCRITES."""
    options = construire_parseur().parse_args(argv)
    ecrites = rendre(options.ticks, options.dossier)
    print(f"{ecrites} images écrites")
    return 0


if __name__ == "__main__":
    sys.exit(main())
