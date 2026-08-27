"""DÉMO 2:1 v4 — L'EAU QUI CREUSE : LA PREMIÈRE ÉCRITURE PERSISTANTE À L'ÉCRAN.

    .venv/bin/python -m scripts.run_demo_2a1_v4 \\
        --dossier outputs/demo_2a1_v4_2026-08-27 --ticks 200

CE QUE CE PAS DÉBLOQUE. La v3 montre une inondation sur un terrain qui ne bouge
pas : le monde subit l'eau, il n'en garde rien. Ici le lit ÉVOLUE — l'eau
l'érode et y dépose, tick après tick, et le lit modifié redevient la
bathymétrie sur laquelle l'eau coule au tick suivant. Le terrain de la fin
n'est plus celui du début, et c'est la première fois que ce dépôt le montre.

`run_demo_2a1_v3.py` N'EST PAS TOUCHÉ. C'est un quatrième frère. La scène, la
géométrie, l'ombrage, la composition roche/eau et l'encodeur PNG RGB en sont
IMPORTÉS — la v4 n'est que le couplage en plus, et deux copies d'un rendu
dérivent l'une de l'autre sans que rien ne le signale.

AUCUN PAS D'EXNER N'EST ÉCRIT ICI. `_integrate_exner` et `_exner_step` de
`src/sediment.py` sont appelés tels quels, avec les coefficients CO-CALIBRÉS du
module (`KD_CALIBRE_V2`, `KE_CALIBRE_V2`, gelés). Le branchement n'a coûté
qu'un raccord : `simulate_wetdry_o2` rend déjà `(times, hs, hus, hvs)` sur la
tranche d'un tick, et c'est EXACTEMENT la signature que `_integrate_exner`
consomme. La v3 jetait ces séquences ; la v4 les lui passe.

LA CALIBRATION N'A PAS ÉTÉ RELANCÉE, ET C'EST MOTIVÉ. `cocalibrate_kd_ke` coûte
~19 minutes et vise DEUX signatures d'un autre protocole — des épisodes à pulse
gaussien, pas une rupture de retenue. Les coefficients gelés produisent déjà,
sur cette scène, un creusement qui atteint 10 % du relief : la condition du
brief (« visible sans détruire le terrain ») est tenue sans y toucher.

── SIX FAITS MESURÉS QUI ONT DÉCIDÉ CE FICHIER ─────────────────────────────

FAIT 1 — CETTE LOI N'ÉRODE QUE CE QU'ELLE A DÉPOSÉ, JAMAIS LA ROCHE. Le terme
d'érosion vaut `k_e · s · 1[θ>θc] · (θ/θc − 1)` : il est PROPORTIONNEL À `s`.
Partie de `s = 0`, l'érosion est identiquement nulle — mesuré, pas déduit :
sur 60 ticks depuis un lit nu, le compte de cellules creusées reste à ZÉRO à
tous les ticks, et `s` ne fait que croître. Une démo « l'eau qui creuse » lancée
sur un lit nu n'aurait montré que de l'accrétion, sans rien qui lève.

D'OÙ LE MANTEAU MEUBLE, QUI EST UNE COMPOSITION DE SCÈNE ET NON UN TRUCAGE. Le
lit part couvert d'une couche érodible uniforme d'épaisseur `MANTEAU` : le
socle `b0` est la ROCHE, le manteau est ce que l'eau peut emporter. C'est ce
qu'on poserait dans un jeu, et c'est ce qui rend les deux sens possibles —
mesuré : 22 491 cellules creusées et 2 967 déposées au tick 200. Étant
UNIFORME, il ne change rien à l'écoulement initial : c'est un décalage
constant, et la lame d'eau de départ est celle de la v3, cellule pour cellule.

FAIT 2 — LES DEUX SENS N'ONT PAS LA MÊME AMPLITUDE, ET L'ÉCART EST D'UN FACTEUR
TROIS. `s` est clampé à zéro par `_exner_step` : une cellule creusée jusqu'à la
roche ne descend pas plus bas, donc le creusement maximal vaut EXACTEMENT
`MANTEAU` — et il l'atteint, 0,05994 mesuré au tick 120. Le dépôt, lui, plafonne
à 0,0195, médiane 0,0122. Le panneau porte donc DEUX ÉCHELLES, une par branche,
déclarées au bloc d'`AMPLITUDE_CREUSE` : à échelle commune le premier run livré
ne portait AUCUN pixel vert alors que le champ comptait 4 808 cellules de
dépôt.

FAIT 3 — LE SÉDIMENT N'EST PAS CONSERVÉ, ET IL NE PEUT PAS L'ÊTRE. Cette loi
est une SOURCE-PUITS locale, pas la divergence d'un flux : rien n'est
transporté d'une cellule à l'autre. `ds/dt` dépose là où l'écoulement est lent
et érode là où il est rapide, sans que la matière érodée aille nulle part.
Mesuré : `Σs` passe de 1 536 à 660 sur 120 ticks — une perte de 57 %, monotone,
pas une dérive numérique. Aucune tolérance honnête ne borne ça, et un verrou de
conservation posé ici serait soit rouge d'emblée, soit vide. Le premier verrou
garde donc ce que ce couplage a de réel et de fragile : la RÉTROACTION du lit
sur l'eau. Voir `tests/test_demo_2a1_v4.py`.

FAIT 4 — L'ÉPAISSEUR DU MANTEAU S'ANNULE DU MOTIF, ET C'EST STRUCTUREL. Le
terme d'érosion étant PROPORTIONNEL à `s`, `s` décroît de façon exponentielle :
un manteau deux fois plus épais s'érode deux fois plus vite, et le champ `s/s₀`
ne dépend pas de `s₀`. Mesuré à 200 ticks sur trois épaisseurs — 0,06 / 0,12 /
0,20 : la fraction de cellules raclées jusqu'à la roche vaut 35,1 % / 34,4 % /
38,1 %, le creusement maximal vaut EXACTEMENT le manteau à chaque fois, et
l'écart-type relatif du changement reste 0,40 ± 0,03. CONSÉQUENCE DE MOTEUR, pas
de démo : avec cette loi, la PROFONDEUR que l'eau creuse n'est pas réglée par
l'écoulement mais par l'épaisseur qu'on a posée. On ne peut pas régler l'une
sans l'autre. C'est aussi ce qui a tué le remède évident à la saturation du
panneau — épaissir le manteau ne diffère rien du tout.

FAIT 5 — LE PEIGNE DU FRONT N'EST PAS L'ŒUVRE DU LIT. À partir du tick ~60,
l'image montre des stries horizontales d'UNE cellule au front mouillé/sec, qui
s'allongent avec le run. Elles sont dans les CHAMPS, pas dans le rendu — les
deux panneaux les portent au même endroit. DEUX HYPOTHÈSES ONT ÉTÉ TESTÉES ET
TOUTES DEUX ÉCARTÉES. La cadence du lit d'abord : intégrer Exner un tick sur 16
au lieu de chaque tick ne réduit la rugosité ligne à ligne que de 0,0134 à
0,0077, sans changer la structure ni les comptes d'érosion. Le couplage ensuite :
un TÉMOIN à lit GELÉ, même scène, même durée, produit la même rugosité d'eau —
0,0463 contre 0,0538 au tick 120, seize pour cent d'écart. Le peigne appartient
donc au front du solveur mouillé/sec sur une pente faible et un long run ; le
couplage l'amplifie modestement, il ne le crée pas. Écrit ici pour qu'il ne soit
pas rediagnostiqué comme un défaut d'Exner.

FAIT 6 — LE DÉPÔT EST UN TRANSITOIRE, ET LE CADRE FINIT TOUT ROUGE. Compté sur
les 240 images livrées, pas sur un échantillon : le panneau ne porte AUCUN pixel
vert avant le tick 15 ni après le tick 42, culmine à 8 008 dans cette fenêtre, et
reste à zéro sur les 77 ticks suivants. Ce n'est pas le rendu qui s'éteint, c'est
la scène :
une fois l'écoulement établi sur tout le cadre, `θ > θc` presque partout, la
branche d'érosion domine et RE-EMPORTE les dépôts des premiers ticks. Aucun
horizon ne donne à la fois la grande amplitude et les deux signes — le dépôt
vit tôt, le creusement vit tard. Le run livré porte les DEUX régimes puisqu'il
écrit chaque tick : les éventails de dépôt sont dans les ticks 15 à 42, le
creusement structuré vers 40-60, et la dernière image est un aplat rouge qu'il
faut lire comme « tout le cadre a été décapé », pas comme un panneau muet.

── CE QUE LES CHAMPS PORTENT, DIT FRANCHEMENT ──────────────────────────────
La pyramide porte `(h, hu, hv, s)` et la v3 avait déjà détourné deux de ces
noms. La v4 en détourne un troisième, et aucun ne devient ce que son nom dit :

  · champ 0 (`h`) — LE LIT COURANT `b0 + s`. Il n'est plus statique comme en v3 :
    il est réécrit à chaque tick, puisque c'est lui qui bouge.
  · champ 1 (`hu`) — LE CHANGEMENT DU LIT `s − MANTEAU`, la seule grandeur
    SIGNÉE de la scène. Négative = creusé, positive = déposé.
  · champ 3 (`s`) — la PROFONDEUR D'EAU, comme en v2 et en v3, parce que c'est
    le seul champ que `melanger` sait interpoler.

CONSÉQUENCE À NE PAS LIRE DE TRAVERS : seule l'EAU passe par le noyau `α = 0,5`.
Le lit et son changement sont des gathers DIRECTS sur la pyramide, donc exacts.
Sur l'image intermédiaire d'un tick, l'eau est en retard d'un demi pas de monde
sur le lit qu'elle habille. C'est la dette du côté `interpoler`, scellée en v3
(`373f86f`), et elle se voit ici pour la première fois entre deux couches.

`b − b0` DU BRIEF EST LU `lit courant − lit de DÉPART`, et c'est la seule
lecture qui donne deux signes : mesuré depuis la ROCHE, `b − b0` vaut `s`, qui
est positif partout par construction — un panneau qui ne serait jamais rouge.

DEUX PANNEAUX ET NON TROIS. Le brief en demande un « troisième », comptant le
terrain et l'eau pour deux ; le rendu hérité de la v3 les compose déjà en UNE
image. L'image livrée porte donc la scène à gauche et le lit à droite, au même
`centre_fin` et par le même gather.

AUCUNE MESURE, AUCUN CHIFFRE D'INSTRUMENT EN SORTIE ; le seul chiffre imprimé
est le nombre d'images écrites. `--dossier` est obligatoire, un dossier existant
est REFUSÉ, et l'horodatage est chez l'appelant.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from config import GridConfig
from scripts.run_demo_2a1 import DossierDejaPresent
from scripts.run_demo_2a1_v3 import (
    COTE,
    COTE_PX,
    N_NIV,
    N0,
    composer_rgb,
    encoder_png_rgb,
    retenue_initiale,
    terrain_du_monde,
)
from src.f1_gpu.boucle_rendu import BoucleRendu
from src.f1_gpu.chemin_de_cout import gather_chemin_de_cout
from src.f1_gpu.interpolation_readout import INDICE_CHAMP_S
from src.f1_gpu.pyramide import (
    GeometriePyramide,
    PyramideFovea,
    predire_bloc_cpu,
)
from src.f1_gpu.transferts import TransfertComptable
from src.sediment import SedimentParams, _integrate_exner
from src.solver_wetdry import simulate_wetdry_o2

# L'HORIZON LIVRÉ EST DE 120 TICKS, ET LE BRIEF EN DEMANDAIT ~200 — l'écart est
# mesuré, pas économisé. Le changement de lit a ATTEINT son amplitude à 120
# (`|Δ|max / relief` = 0,0999 contre 0,1000 au-delà) : les quatre-vingts ticks
# suivants n'ajoutent aucune information de lit. Ils en RETIRENT : la fraction de
# cellules raclées jusqu'à la roche passe de 0,6 % au tick 100 à 35 % au tick
# 200, et un panneau où un tiers du domaine est au fond de l'échelle est un
# aplat rouge sans structure — le contraire de ce que cette démo doit montrer.
# Le peigne du FAIT 5, lui, ne cesse de croître. 120 est donc le point où le lit
# a tout dit et où l'image le dit encore.
N_TICKS_DEFAUT: int = 120

# `n_fov` N'EST PAS IMPORTÉ DE LA v3, ET C'EST LA v4 QUI L'A PAYÉ. La v3 mesure
# sa couverture contre un horizon de 120 ticks et grave `n_fov = 256` ; le
# premier run de la v4, à 200 ticks, a LEVÉ au tick 128 — le fail-loud du gather
# a fait exactement son travail, et aucune image fausse n'a été écrite.
#
# CE QUI PLAFONNE LA DURÉE D'UN RUN N'EST PAS LA TAILLE DE L'ÉCRAN. La caméra
# avance d'un pixel fin par tick et les origines en `y` sont FIXES : l'écran
# glisse hors des fenêtres, et rétrécir l'écran n'achète qu'un délai linéaire et
# maigre — mesuré, tick de rupture 128 / 144 / 160 / 176 / 192 pour `cote_px`
# 384 / 352 / 320 / 288 / 256. Même à 256 px, soit un tiers de cadre en moins
# que la v3, la rupture tombe AVANT les 200 ticks demandés. Le plafond absolu de
# cette géométrie est le tick ~320, écran nul compris.
#
# LE LEVIER EST DONC `n_fov`, qui gouverne l'étendue en `y` des fenêtres — le
# même levier que la v3 avait actionné pour le plafond de VISION, employé ici
# pour le plafond de DURÉE. Mesuré à `cote_px = 384` : 256 lève au tick 128,
# 384 tient 230 ticks. Le cadre reste celui de la v3 — 96 cellules de monde — et
# c'est ce qui permet de comparer les deux démos à l'œil.
#
# CE QUE LE PASSAGE À 384 COÛTE VRAIMENT, ET LE PREMIER CHIFFRE ÉCRIT ICI ÉTAIT
# FAUX. Une sonde sur la pyramide NUE donne 317 ms par tick à `n_fov = 256` et
# 802 à 384 — un facteur 2,5 qui aurait été rédhibitoire. Mais cette sonde
# mesure l'applicateur PAR DÉFAUT, le `F` jetable, que ce driver n'emploie pas :
# `PasErosion` le remplace, et ne fait qu'une recopie depuis `predire_bloc_cpu`.
# Le driver complet — pyramide, solveur, Exner, trois gathers et deux PNG —
# coûte 186 ms par tick à `n_fov = 384`, soit ~25 s pour les 120 ticks livrés.
# Le coût du `F` jetable n'est pas celui de cette démo, et le confondre a failli
# faire renoncer au seul paramètre qui règle la durée d'un run.
#
# IL EST DIMENSIONNÉ POUR 200 TICKS ALORS QUE LE RUN LIVRÉ EN FAIT 120, et c'est
# délibéré : le brief nomme ~200 ticks, `--ticks` les accepte, et un driver qui
# lève sur la valeur que son propre brief demande serait un piège. La marge est
# payée en temps de run, pas en justesse.
N_FOV: int = 384

# LE PAS DE MONDE. Celui de la v3 n'est PAS importé : la v3 le justifie par la
# traversée du cadre en 120 ticks, la v4 tourne sur 200 et le lit a sa propre
# échelle de temps. Les deux valeurs coïncident aujourd'hui ; les lier ferait
# bouger l'une quand l'autre change, sans que rien ne le dise.
DT_TICK: float = 0.6

# LE MANTEAU MEUBLE — voir le FAIT 1 de l'en-tête. 0,06 pour un relief de 0,60 :
# le creusement plafonne à 10 % du relief, assez pour se lire dans un panneau
# dédié, trop peu pour effacer les bosses (0,30 au-dessus du plan). Un manteau
# plus épais rendrait le terrain méconnaissable, un plus mince rendrait le
# creusement invisible ; celui-ci a été mesuré sur 200 ticks avant d'être écrit.
MANTEAU: float = 0.06

# ── LES CHAMPS DÉTOURNÉS — voir l'en-tête ───────────────────────────────────
INDICE_CHAMP_LIT: int = 0          # `h` : le lit courant `b0 + s`
INDICE_CHAMP_CHANGEMENT: int = 1   # `hu` : `s − MANTEAU`, signé

# ── LE PANNEAU DU LIT ───────────────────────────────────────────────────────
# DEUX BRANCHES, DEUX ÉCHELLES, ET IL FAUT LE DIRE HAUT. Une carte divergente
# dont les branches ne partagent pas leur échelle se lit de travers par défaut :
# un rouge et un vert de même intensité n'y valent PAS la même épaisseur de lit.
# C'est déclaré ici parce que c'est le seul endroit où ça se déclare.
#
# CE N'EST PAS UN CHOIX ESTHÉTIQUE, C'EST UNE MESURE. Les deux sens n'ont pas la
# même amplitude et ne PEUVENT pas l'avoir : le creusement est borné par le clamp
# `s ≥ 0`, donc par `MANTEAU`, et il l'atteint (0,05994 mesuré au tick 120) ;
# le dépôt, lui, ne dépasse pas 0,0195 sur le même horizon, avec une médiane à
# 0,0122. À échelle commune, la moitié déposée tombe SOUS le seuil de teinte —
# constaté sur le premier run livré, qui portait ZÉRO pixel vert à tous les
# ticks alors que le champ comptait 4 808 cellules de dépôt. Un panneau muet
# d'un côté est pire qu'une échelle dissymétrique déclarée.
AMPLITUDE_CREUSE: float = MANTEAU
AMPLITUDE_DEPOSE: float = 0.02

# LE SEUIL DE NEUTRALITÉ, EN FRACTION DE L'AMPLITUDE. Sous lui, la cellule est
# rendue grise plutôt que teintée : sans ce plateau, l'arrondi en 8 bits donne à
# tout le domaine une teinte faible et l'œil ne distingue plus les zones où le
# lit n'a rien fait de celles où il a à peine bougé.
FRACTION_NEUTRE: float = 0.02

CREUSE: tuple[float, float, float] = (0.78, 0.16, 0.13)   # rouge
NEUTRE: tuple[float, float, float] = (0.58, 0.56, 0.53)   # gris
DEPOSE: tuple[float, float, float] = (0.13, 0.55, 0.22)   # vert

LARGEUR_SEPARATEUR: int = 4
GRIS_SEPARATEUR: int = 24
NIVEAUX: float = 255.0


class NiveauNonReconnu(ValueError):
    """Levée quand `pas_f` reçoit un bloc de fenêtres qu'il ne sait pas situer.

    FAIL-LOUD PLUTÔT QUE PAS SAUTÉ, comme chez les frères : ne rien faire
    laisserait le monde immobile sans que rien ne le dise, et une démo figée est
    indiscernable d'une démo correcte dont le champ n'évolue pas. Hérite de
    `ValueError` — ce qui hérite de `RuntimeError` risque d'être avalé par un
    `except RuntimeError` d'appelant."""


class PasErosion:
    """« Le monde a avancé ET le lit a bougé, relis-les » — le `pas_f` de la v4.

    LE COUPLAGE EST À DEUX SENS, ET IL TIENT EN TROIS LIGNES D'`avancer`. Le lit
    courant `b0 + s` est reformé À CHAQUE TICK et passé au solveur comme
    bathymétrie ; les instantanés que le solveur rend sur la tranche sont passés
    à `_integrate_exner`, qui rend le `s` suivant. Geler le lit à `b0` donnerait
    une démo d'apparence identique où le monde n'apprend rien — c'est ce que le
    premier verrou exerce.

    LA CADENCE DU LIT EST CELLE DE L'EAU, PAS CELLE DES ÉPISODES DU MODULE.
    `run_episode` fait tourner 600 pas solveur sur un lit GELÉ avant de mettre le
    lit à jour une fois ; ici la mise à jour a lieu à chaque tick, sur les trois
    à cinquante pas que le solveur accepte dans `DT_TICK`. La loi est la même —
    `_integrate_exner` est appelée telle quelle, avec son `save_every` — seule la
    fenêtre d'intégration change. C'est ce qu'il faut pour que l'écran voie le
    lit bouger au lieu de le voir sauter.

    L'AVANCE SE FAIT SUR L'IDENTITÉ DU PREMIER NIVEAU, PAS SUR UN COMPTEUR.
    `frame()` appelle `pas_f` une fois par niveau et INCONDITIONNELLEMENT ; un
    compteur modulo le nombre de niveaux marcherait aujourd'hui et se
    désynchroniserait en silence à la première surprise.

    TROIS CHAMPS SONT RECOPIÉS DEPUIS LE BLOC PRÉDIT, LÀ OÙ LA v3 N'EN RECOPIAIT
    QU'UN. En v3 le terrain était statique et la descente de `frame()` suffisait
    à le tenir juste ; ici le lit BOUGE, et un champ laissé au soin du seul roll
    afficherait le lit du tick où il est entré dans la fenêtre — un décalage
    silencieux, croissant, invisible à l'œil."""

    def __init__(self, socle: np.ndarray):
        self.socle = socle
        self.manteau = np.full_like(socle, MANTEAU)
        self.profondeur = retenue_initiale(socle)
        self.debit_x = np.zeros_like(socle)
        self.debit_y = np.zeros_like(socle)
        self.grille = GridConfig(H=N0, W=N0)
        self.params = SedimentParams()
        self.pyramide = None
        self._premier = None

    def lit(self) -> np.ndarray:
        """Le lit courant : la roche plus ce qui reste du manteau."""
        return self.socle + self.manteau

    def changement(self) -> np.ndarray:
        """Le changement du lit depuis le DÉPART — la seule grandeur signée."""
        return self.manteau - MANTEAU

    def attacher(self, pyramide) -> None:
        """La pyramide n'existe pas encore quand `pas_f` lui est passé — d'où
        cette seconde étape, explicite plutôt que devinée."""
        self.pyramide = pyramide
        self._premier = pyramide.fenetres[list(pyramide.geo.niveaux_gpu)[0]]

    def avancer(self) -> None:
        """Un pas de monde : l'eau coule sur le lit, puis le lit encaisse."""
        temps, hauteurs, debits_x, debits_y = simulate_wetdry_o2(
            self.profondeur, self.debit_x, self.debit_y, self.lit(),
            self.grille, t_end=DT_TICK)
        self.profondeur = hauteurs[-1]
        self.debit_x = debits_x[-1]
        self.debit_y = debits_y[-1]
        self.manteau = _integrate_exner(
            self.manteau, np.asarray(temps), hauteurs, debits_x, debits_y,
            self.params)

    def _ecrire_monde(self) -> None:
        monde = self.pyramide.monde0
        monde[:, INDICE_CHAMP_LIT] = self.lit()
        monde[:, INDICE_CHAMP_CHANGEMENT] = self.changement()
        monde[:, INDICE_CHAMP_S] = self.profondeur

    def _niveau_de(self, fenetres) -> int:
        for niveau, bloc in self.pyramide.fenetres.items():
            if bloc is fenetres:
                return niveau
        raise NiveauNonReconnu(
            "PasErosion : bloc de fenêtres non situé dans la pyramide attachée. "
            "Ne rien faire laisserait le monde immobile sans que rien ne le "
            "dise — une démo figée est indiscernable d'une démo correcte.")

    def __call__(self, fenetres, xp, sortie=None):
        niveau = self._niveau_de(fenetres)
        if fenetres is self._premier:
            self.avancer()
            self._ecrire_monde()

        geo = self.pyramide.geo
        bloc = predire_bloc_cpu(
            self.pyramide.monde0, geo, niveau,
            geo.origines(niveau, self.pyramide.centre_fin), 0, geo.n_fov,
            geo.n_systemes_du_niveau(niveau))

        if sortie is None:
            sortie = xp.empty_like(fenetres)
        if sortie is not fenetres:
            sortie[...] = fenetres
        for indice in (INDICE_CHAMP_LIT, INDICE_CHAMP_CHANGEMENT,
                       INDICE_CHAMP_S):
            sortie[:, :, indice] = xp.asarray(bloc[:, :, indice])
        return sortie, 0.0


def construire_pyramide() -> tuple[PyramideFovea, PasErosion]:
    """Une pyramide amorcée sur la scène, lit et retenue déjà en place.

    POURQUOI L'AMORÇAGE, motif hérité de la v2 et inchangé : le readout capture
    `s_prev` post-roll et pré-`pas_f`. Sans amorçage, le `s_prev` du tout premier
    tick serait le monde JETABLE descendu par `__init__` tandis que `s_cur` serait
    déjà la retenue — l'écran intermédiaire fondrait deux mondes sans rapport,
    une fois, au début."""
    geo = GeometriePyramide(n_fov=N_FOV, n_niv=N_NIV, n0=N0)
    transferts = TransfertComptable(np)
    pas = PasErosion(terrain_du_monde())
    pyramide = PyramideFovea(np, geo, transferts, pas_f=pas)

    pas.pyramide = pyramide
    pas._ecrire_monde()
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


def colorer_changement(ecran_changement: np.ndarray) -> np.ndarray:
    """L'écran du changement de lit → `uint8` `(h, w, 3)`, rouge/gris/vert.

    RAMPE DIVERGENTE À DEUX BRANCHES, chacune interpolée du gris neutre vers sa
    couleur pleine sur `AMPLITUDE_LIT`. Le plateau neutre autour de zéro est ce
    qui distingue « le lit n'a rien fait » de « le lit a à peine bougé ».

    LES DEUX BRANCHES N'ONT PAS LA MÊME ÉCHELLE — voir le bloc d'`AMPLITUDE_CREUSE` :
    un rouge et un vert de même intensité ne valent pas la même épaisseur.

    LE HORS-BANDE EST ÉCRÊTÉ ET NON LEVÉ, contrairement au rendu de la scène, et
    la différence est motivée : l'échelle est un choix d'affichage FIXE, pas une
    borne physique. Un dépôt qui dépasserait `AMPLITUDE_DEPOSE` est un fait de la
    scène, pas une faute ; le rendre en vert plein est la lecture juste. Ce qui
    resterait une faute — un `NaN`, un `±inf` — est refusé en amont par
    `composer_rgb`, qui voit le même tick."""
    changement = np.asarray(ecran_changement, dtype=np.float64)
    creuse = changement < 0.0
    amplitude = np.where(creuse, AMPLITUDE_CREUSE, AMPLITUDE_DEPOSE)
    intensite = np.clip(np.abs(changement) / amplitude, 0.0, 1.0)
    intensite = np.where(intensite < FRACTION_NEUTRE, 0.0, intensite)[..., None]

    neutre = np.asarray(NEUTRE, dtype=np.float64)
    pleine = np.where(creuse[..., None],
                      np.asarray(CREUSE, dtype=np.float64),
                      np.asarray(DEPOSE, dtype=np.float64))
    image = neutre * (1.0 - intensite) + pleine * intensite
    return np.rint(np.clip(image, 0.0, 1.0) * NIVEAUX).astype(np.uint8)


def accoler(gauche: np.ndarray, droite: np.ndarray) -> np.ndarray:
    """Les deux panneaux côte à côte, séparés par un filet sombre.

    LE FILET N'EST PAS DÉCORATIF : sans lui, les deux panneaux se touchent et un
    lecteur ne sait pas où finit la scène et où commence le lit — les deux ont
    des zones grises."""
    hauteur = gauche.shape[0]
    filet = np.full((hauteur, LARGEUR_SEPARATEUR, 3), GRIS_SEPARATEUR,
                    dtype=np.uint8)
    return np.hstack((gauche, filet, droite))


def rendre(n_ticks: int, dossier: Path) -> int:
    """Le côté `interpoler` seul → `2 · n_ticks` PNG à deux panneaux.

    LES TROIS GATHERS D'UN TICK PARTAGENT UN SEUL `centre_fin`. `tick()` appelle
    `frame()` en premier, donc le centre n'est plus déplacé quand les écrans sont
    rendus (§4-6 : un seul centre par tick, pour deux écrans) ; gatherer le lit
    et son changement après le tick les prend au MÊME centre que l'eau.

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
        ecran_lit = gather_chemin_de_cout(
            pyramide, COTE_PX, indice_champ=INDICE_CHAMP_LIT)
        ecran_changement = gather_chemin_de_cout(
            pyramide, COTE_PX, indice_champ=INDICE_CHAMP_CHANGEMENT)
        panneau_lit = colorer_changement(ecran_changement)
        for rang, ecran_eau in enumerate(ecrans):
            indice = 2 * tick + rang
            chemin = dossier / f"ecran_{COTE}_{indice:04d}.png"
            scene = composer_rgb(ecran_lit, ecran_eau)
            chemin.write_bytes(encoder_png_rgb(accoler(scene, panneau_lit)))
            compte += 1
    return compte


def construire_parseur() -> argparse.ArgumentParser:
    """`--dossier` est OBLIGATOIRE et sans défaut ; `--cote` n'existe pas, le
    côté ayant été tranché en v3."""
    parseur = argparse.ArgumentParser(
        description="Démo 2:1 v4 : l'eau qui creuse le lit qu'elle parcourt.")
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
