"""Les verrous de la démo 2:1 v4 — l'eau qui creuse le lit qu'elle parcourt.

Régime §A64-4 : cette démo ne mesure rien, aucune décision verdict-grade n'en
dépend, la cérémonie est celle du build. LE BRIEF EN DEMANDE TROIS, IL Y EN A
TROIS — mais LE PREMIER N'EST PAS CELUI QU'IL DEMANDAIT, et l'échange se motive
plutôt qu'il ne se cache.

LE BRIEF DEMANDAIT LA CONSERVATION DU SÉDIMENT, « Exner est conservatif ». LA
PRÉMISSE EST FAUSSE POUR CE MODULE. `_exner_step` est une SOURCE-PUITS locale
— `ds/dt = 1[wet]·(k_d·h·1[θ<θc]·(1−θ/θc) − k_e·s·1[θ>θc]·(θ/θc−1))` — et non
la divergence d'un flux de sédiment : rien n'est transporté d'une cellule à
l'autre, la matière érodée ne va NULLE PART. Mesuré sur la scène de la démo :
`Σs` passe de 1 536 à 660 en 120 ticks, une perte de 57 %, monotone. Un verrou
de conservation posé là serait rouge d'emblée, ou vide s'il desserrait sa
tolérance jusqu'à passer. On ne déplace pas un seuil pour faire passer un gate,
et on n'écrit pas un verrou qui ne peut rien attraper.

CE QUI PREND SA PLACE GARDE LE VRAI RISQUE DE CETTE DÉMO : la RÉTROACTION. Le
couplage à deux sens est tout le sujet — l'eau écrit dans le lit, le lit
redevient la bathymétrie de l'eau. Un driver qui oublierait de reformer
`b0 + s` avant chaque appel au solveur produirait des images RIGOUREUSEMENT
IDENTIQUES d'aspect : le lit changerait toujours à l'écran, mais le monde
n'apprendrait rien. Aucun autre verrou ne voit ça, et l'œil non plus.

Les trois :

  1. LE LIT RÉTROAGIT SUR L'EAU — prouvé contre un TÉMOIN à lit gelé, pas
     déclaré. Sans rétroaction l'écart est exactement nul.
  2. LE LIT CHANGE RÉELLEMENT, DANS LES DEUX SENS — relu dans l'artefact, sur
     le panneau de droite des PNG livrés. Les deux signes sont exigés : le seul
     sens était le piège de cette démo (voir le FAIT 1 du driver).
  3. LE DOSSIER EXISTANT EST REFUSÉ — garde héritée des trois frères, exercée.

CE QUE CES TROIS NE GARDENT PAS : ni le peigne du front (FAIT 5 du driver : il
appartient au solveur, un témoin à lit gelé le reproduit), ni la couverture du
gather (elle LÈVE d'elle-même — c'est ce qui a arrêté le premier run de la v4),
ni l'aspect du rendu.

La lecture des chunks PNG et le dépaquetage RGB sont IMPORTÉS des fichiers de
tests des frères, jamais recopiés."""
from __future__ import annotations

import numpy as np
import pytest

from config import GridConfig
from scripts import run_demo_2a1_v4
from scripts.run_demo_2a1 import DossierDejaPresent
from scripts.run_demo_2a1_v3 import COTE, COTE_PX, terrain_du_monde
from src.solver_wetdry import simulate_wetdry_o2
from tests.test_demo_2a1_v3 import _pixels_rgb

# TICKS DU HARNAIS. Le run livré en fait 120 ; vingt suffisent aux deux verrous
# qui en ont besoin. Le rendu coûte 810 ms par tick à `n_fov = 384`, et la
# trajectoire est DÉTERMINISTE : l'image `k` d'un run de vingt ticks est
# l'image `k` du run livré, octet pour octet. Les planchers ci-dessous ont donc
# été mesurés sur le run livré et s'appliquent tels quels.
TICKS_HARNAIS = 20

# LE PLANCHER DE RÉTROACTION, MESURÉ ET NON CHOISI. Écart L1 entre la lame d'eau
# couplée et celle du témoin à lit gelé, en pour-cent de la lame : 0,018 % à 5
# ticks, 0,599 % à 20, 2,772 % à 40, 6,355 % à 60, 14,753 % à 120 — croissance
# monotone. SANS rétroaction il vaudrait EXACTEMENT zéro, les deux histoires
# étant le même code sur la même bathymétrie. Le verrou tourne à 40 ticks, où la
# mesure donne 2,77 % : un plancher à 1,0 % est près de trois fois dessous et
# très au-dessus de zéro.
TICKS_RETROACTION = 40
PLANCHER_RETROACTION = 0.010

# LES DEUX PLANCHERS DU LIT, MESURÉS SUR LA DERNIÈRE IMAGE DU HARNAIS (tick 19
# du run livré, la trajectoire étant déterministe) : 36 656 pixels creusés et
# 7 872 déposés, sur un panneau de 148 224. Les planchers sont posés à un peu
# moins de la moitié. Un lit immobile donnerait ZÉRO des deux côtés — le panneau
# serait gris uni.
#
# VINGT TICKS N'EST PAS UN CHOIX DE CONFORT ICI, C'EST LE SEUL HORIZON QUI PORTE
# LES DEUX SIGNES. Le dépôt est un TRANSITOIRE (FAIT 6 du driver), et la fenêtre
# a été balayée sur les 240 images livrées et non échantillonnée : AUCUN pixel
# vert avant le tick 15 ni après le tick 42. Le harnais tombe au tick 19, au
# cœur de cette fenêtre. Calé plus tard il ne pourrait plus exiger le vert, et le
# verrou perdrait exactement ce qu'il existe pour garder.
PLANCHER_CREUSE = 16000
PLANCHER_DEPOSE = 3000

# LA MARGE DU PRÉDICAT DE COULEUR, EN NIVEAUX 8 BITS. Le gris neutre du panneau
# porte déjà `rouge − vert = +5` : une marge sous 5 compterait tout le domaine
# comme creusé. 16 est trois fois au-dessus de ce biais et sous le `+158` du
# rouge plein comme sous le `−107` du vert plein.
MARGE_COULEUR = 16


def _panneau_du_lit(image: np.ndarray) -> np.ndarray:
    """Le panneau de DROITE, découpé sur les constantes du driver et non sur un
    littéral : la largeur du séparateur bouge, le découpage suit."""
    debut = COTE_PX + run_demo_2a1_v4.LARGEUR_SEPARATEUR
    attendue = debut + COTE_PX
    assert image.shape[1] == attendue, (
        f"image large de {image.shape[1]} px, attendu {attendue} — le montage "
        "à deux panneaux a changé et ce découpage ne vise plus le bon.")
    return image[:, debut:, :]


def _compter_lit(image: np.ndarray) -> tuple[int, int]:
    """(pixels creusés, pixels déposés) du panneau de droite.

    LE PRÉDICAT SUIT LA RAMPE DIVERGENTE : rouge `(199, 41, 33)` d'un côté,
    vert `(33, 140, 56)` de l'autre, gris `(148, 143, 135)` au milieu. Un pixel
    est creusé si `rouge − vert` dépasse la marge, déposé si `vert − rouge` la
    dépasse, neutre sinon.

    IL EST DISSYMÉTRIQUE ET LE VERROU VIT AVEC. Le gris neutre penche déjà de
    `+5` vers le rouge : à marge égale, le prédicat déclare « creusé » dès une
    intensité de ~0,07 et « déposé » seulement à partir de ~0,19. Les deux
    planchers sont donc mesurés SÉPARÉMENT sur les images livrées, jamais
    dérivés l'un de l'autre."""
    panneau = _panneau_du_lit(image)
    ecart = panneau[..., 0].astype(np.int32) - panneau[..., 1].astype(np.int32)
    return (int((ecart > MARGE_COULEUR).sum()),
            int((ecart < -MARGE_COULEUR).sum()))


@pytest.fixture(scope="module")
def sortie(tmp_path_factory):
    """Un seul run pour le verrou qui a besoin d'images. La démo est
    déterministe : rien ne dépend de l'ordre des tests."""
    dossier = tmp_path_factory.mktemp("v4") / "sortie"
    run_demo_2a1_v4.rendre(TICKS_HARNAIS, dossier)
    return dossier


def test_le_lit_retroagit_sur_l_eau():
    """VERROU 1 — le couplage est à DEUX sens, prouvé contre un témoin.

    LE TÉMOIN EST LE MÊME CODE SUR UN LIT GELÉ. Il part du même état, avance du
    même `DT_TICK`, appelle le même `simulate_wetdry_o2` — et reçoit la
    bathymétrie du DÉPART à chaque tick au lieu du lit courant. C'est la seule
    différence, et c'est exactement la faute que ce verrou existe pour attraper :
    un driver qui oublierait de reformer `b0 + s` produirait le témoin en croyant
    produire la démo, sans qu'aucune image ne s'en ressente.

    IL NE PREND PAS LA FIXTURE : aucune image n'est lue ici, et la demander
    coûterait un rendu de vingt ticks dont rien ne se sert.

    LE TÉMOIN N'EST PAS UNE RECOPIE D'`avancer` — c'est sa CONTREFACTUELLE. La
    boucle ci-dessous n'intègre aucun Exner et ne réécrit aucune loi ; elle
    appelle le solveur importé, et c'est tout ce qu'il lui reste à faire une fois
    la rétroaction retirée."""
    couple = run_demo_2a1_v4.PasErosion(terrain_du_monde())
    grille = GridConfig(H=run_demo_2a1_v4.N0, W=run_demo_2a1_v4.N0)
    lit_gele = couple.lit().copy()
    eau, debit_x, debit_y = (couple.profondeur.copy(), couple.debit_x.copy(),
                             couple.debit_y.copy())

    for _ in range(TICKS_RETROACTION):
        couple.avancer()
        _, eaux, debits_x, debits_y = simulate_wetdry_o2(
            eau, debit_x, debit_y, lit_gele, grille,
            t_end=run_demo_2a1_v4.DT_TICK)
        eau, debit_x, debit_y = eaux[-1], debits_x[-1], debits_y[-1]

    lame = float(np.abs(eau).sum())
    assert lame > 0.0, (
        "le témoin n'a plus d'eau : l'écart relatif qui suit ne voudrait rien "
        "dire.")
    ecart = float(np.abs(couple.profondeur - eau).sum()) / lame
    assert ecart >= PLANCHER_RETROACTION, (
        f"l'eau ne voit pas le lit bouger : écart de {100 * ecart:.3f} % contre "
        f"le témoin à lit gelé, sous le plancher de "
        f"{100 * PLANCHER_RETROACTION:.1f} % (2,77 % mesuré à "
        f"{TICKS_RETROACTION} ticks). Un écart nul signifie que la bathymétrie "
        "passée au solveur ne suit pas le lit — le monde n'apprend rien.")


def test_le_lit_change_dans_les_deux_sens(sortie):
    """VERROU 2 — creusement ET dépôt, comptés sur le panneau livré.

    LES DEUX SENS SONT EXIGÉS, ET C'EST LE CŒUR DE CE VERROU. Le terme d'érosion
    de `_exner_step` est proportionnel à `s` : lancée sur un lit NU, cette démo
    n'érode rien du tout — mesuré, le compte de cellules creusées reste à zéro à
    tous les ticks — et n'aurait montré que de l'accrétion, sans que rien ne
    lève. C'est le manteau meuble initial qui rend l'érosion possible ; un verrou
    qui n'aurait compté que « le lit a changé » aurait été VERT sur cette faute.

    SUR LES PIXELS RELUS, PAS SUR LE CHAMP : c'est la chaîne entière — Exner,
    gather, rampe divergente, encodeur — qui est ainsi exercée. Un panneau qui
    cesserait d'écrire le lit laisserait ce verrou rouge, là où un verrou posé
    sur `PasErosion.changement()` resterait vert."""
    images = sorted(sortie.glob(f"ecran_{COTE}_*.png"))
    assert len(images) == 2 * TICKS_HARNAIS, (
        f"{len(images)} images pour {TICKS_HARNAIS} ticks — la cadence 2:1 "
        "veut exactement deux écrans par tick.")

    creuse, depose = _compter_lit(_pixels_rgb(images[-1]))
    assert creuse >= PLANCHER_CREUSE, (
        f"{creuse} pixels creusés sur le panneau final, sous le plancher de "
        f"{PLANCHER_CREUSE}. Un lit immobile en donnerait ZÉRO, et un lit qui "
        "ne fait que se déposer aussi — c'est exactement ce qu'un manteau "
        "initial nul produirait.")
    assert depose >= PLANCHER_DEPOSE, (
        f"{depose} pixels déposés sur le panneau final, sous le plancher de "
        f"{PLANCHER_DEPOSE}. Le dépôt est l'autre moitié de la loi ; sans lui "
        "le panneau ne montre qu'un décapage.")


def test_un_dossier_existant_est_refuse(tmp_path):
    """VERROU 3 — hérité des trois frères, et exercé plutôt que promis.

    Il monte son propre cas : rien à voir avec le run du harnais. La garde est
    la classe IMPORTÉE `DossierDejaPresent` — l'attraper par son nom, et non par
    `FileExistsError`, est ce qui distingue le refus délibéré du driver d'une
    collision de système de fichiers survenue ailleurs."""
    dossier = tmp_path / "deja_la"
    dossier.mkdir()
    with pytest.raises(DossierDejaPresent):
        run_demo_2a1_v4.rendre(1, dossier)
