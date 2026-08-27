"""Les verrous de la démo 2:1 v3 — l'inondation sur le terrain réel.

Régime §A64-4 : cette démo ne mesure rien, aucune décision verdict-grade n'en
dépend, la cérémonie est celle du build. LE BRIEF EN DEMANDE TROIS, IL Y EN A
TROIS. Chacun garde une chose que l'œil ne garde pas :

  1. L'EAU RESTE DANS LE DOMAINE. Une inondation qui fuit — par une paroi mal
     posée, par un plancher sec qui mange la lame — donne des images qui
     ressemblent encore à une inondation. C'est un verrou de COMPORTEMENT, pas
     une mesure : il n'y a pas de seuil pré-enregistré derrière, et le nombre
     qu'il lit ne sort nulle part.
  2. LE FRONT AVANCE entre le premier et le dernier tick. Sans lui, un `pas_f`
     qui n'avancerait plus rien — la faute que `NiveauNonReconnu` existe pour
     rendre bruyante, mais qui a d'autres portes — donnerait 240 images
     identiques d'une retenue immobile, et rien ne le dirait.
  3. LE DOSSIER EXISTANT EST REFUSÉ — garde héritée des deux frères, exercée ici
     parce qu'une garde qu'aucun test n'exerce est une promesse.

CE QUE CES TROIS VERROUS NE GARDENT PAS, dit plutôt que laissé deviner : ni le
choix du côté (il est tranché, il n'y a plus rien à comparer), ni la couverture
du gather sur toute la durée (elle LÈVE d'elle-même, et la levée traverse le
driver sans être rattrapée), ni l'aspect du rendu — aucun test ne juge une
palette.

La lecture des chunks PNG est IMPORTÉE de `test_boucle_rendu`, jamais recopiée ;
seul le dépaquetage des trois canaux est écrit ici, parce que le lecteur du frère
est gris."""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from scripts import run_demo_2a1_v3
from scripts.run_demo_2a1 import DossierDejaPresent
from tests.test_boucle_rendu import _chunks_png

# TICKS DU HARNAIS. Le run livré en fait 120 ; vingt suffisent au verrou du front
# et coûtent une dizaine de secondes au lieu de deux minutes. Le choix est
# ARBITRÉ PAR LA COURBE, pas par le confort : le compte de pixels d'eau croît de
# façon monotone jusqu'au tick 60 environ (rapport 1,16 au tick 5 ; 1,39 au 10 ;
# 1,84 au 20 ; 2,52 au 40 ; 3,37 au 60, puis plateau), donc vingt ticks sont déjà
# franchement dans la partie croissante.
TICKS_HARNAIS = 20

# LE PLANCHER DU FRONT, MESURÉ ET NON CHOISI. Rapport du compte de pixels d'eau
# entre la dernière image et la première : 1,84 sur vingt ticks du run livré, et
# EXACTEMENT 1,0 si le monde est figé — c'est ce que le verrou sépare. 1,4 est à
# mi-chemin des deux régimes, très au-dessus du cas mort et très en dessous du
# cas vivant. Un verrou qui exigeait seulement « le compte a changé » passerait
# sur un frémissement de la retenue au repos.
PLANCHER_FRONT = 1.4

# LA TOLÉRANCE DE CONSERVATION, MESURÉE SUR LE RUN LIVRÉ. Le rapport
# `masse / masse₀` balaie [0,997753 ; 1,008082] sur les 120 ticks — écart maximal
# 0,81 %. La masse n'est PAS exactement conservée et la raison est nommée : le
# plancher sec du solveur (`_floor_dry`) remet à zéro les cellules sous
# `dry_eps`, ce qui retire de la masse, et la reconstruction hydrostatique en
# rend au front. 3 % laisse près de quatre fois la dérive constatée ; une vraie
# fuite — paroi ouverte, domaine qui se vide — se compte en dizaines de pour
# cent, pas en unités. Ce n'est pas un seuil pré-enregistré et rien n'en dépend.
TOLERANCE_MASSE = 0.03


def _pixels_rgb(chemin: Path) -> np.ndarray:
    """Relit un PNG RGB 8 bits À LA MAIN, sans bibliothèque d'image.

    Même idiome que `_pixels` chez le frère, et pour le même motif : `pillow` est
    en ligne COMMENTÉE dans `requirements.txt`, et une dépendance non déclarée
    dans un verrou est une dette invisible. Le filtre de chaque ligne vaut 0, donc
    les octets compressés SONT les pixels — un octet de filtre puis `3 · largeur`
    octets par ligne."""
    chunks = dict(_chunks_png(chemin.read_bytes()))
    largeur, hauteur = struct.unpack(">II", chunks["IHDR"][:8])
    brut = np.frombuffer(zlib.decompress(chunks["IDAT"]), dtype=np.uint8)
    return brut.reshape(hauteur, 1 + 3 * largeur)[:, 1:].reshape(
        hauteur, largeur, 3).astype(np.int16)


def _pixels_d_eau(image: np.ndarray) -> int:
    """Combien de pixels sont de l'eau — LU SUR L'IMAGE FINIE, pas sur le champ.

    LE PRÉDICAT EST `bleu > rouge`, ET CE QU'IL GARANTIT EST DISSYMÉTRIQUE —
    dit ainsi parce que la version symétrique serait fausse. La roche vaut
    `bleu − rouge = −0,12 − 0,06 · ombrage`, NÉGATIF à toute luminosité : aucun
    pixel de roche ne peut être compté comme de l'eau, il n'y a donc AUCUN faux
    positif. L'inverse n'est pas vrai. Le mélange vaut
    `(B−R)_roche · (1 − opacité) + (B−R)_eau · opacité` et il croise zéro : sur
    la roche la plus claire (`−0,18`) mêlée à l'eau la plus pâle (`+0,40`), la
    bascule est à `opacité ≈ 0,31`, soit une lame d'environ 0,007 — plus mince
    que ça, l'eau est comptée comme de la roche. LE COMPTE EST DONC UN
    SOUS-COMPTE, et c'est la bonne dissymétrie pour ce verrou : il ne peut pas
    inventer d'avance de front, seulement en manquer. Le plancher de 1,4 a été
    mesuré avec ce même prédicat, sur les mêmes images.

    Compter sur l'IMAGE et non sur `PasInondation.cellules_mouillees()` est
    load-bearing : c'est la chaîne entière — gather, composition, quantification,
    encodeur — qui est ainsi exercée. Un verrou posé sur le champ flottant
    resterait vert si le rendu cessait d'écrire l'eau."""
    return int((image[..., 2] > image[..., 0]).sum())


@pytest.fixture(scope="module")
def sortie(tmp_path_factory):
    """Un seul run pour le verrou qui a besoin d'images. La démo est
    déterministe : rien ne dépend de l'ordre des tests."""
    dossier = tmp_path_factory.mktemp("v3") / "sortie"
    run_demo_2a1_v3.rendre(TICKS_HARNAIS, dossier)
    return dossier


def test_l_eau_reste_dans_le_domaine():
    """VERROU 1 — la masse d'eau ne dérive pas au-delà de la tolérance mesurée.

    IL NE PREND PAS LA FIXTURE, ET C'EST DÉLIBÉRÉ : il n'a besoin d'aucune
    image, et la demander lui ferait payer un rendu de dix secondes dont rien
    ici ne lit le résultat.

    IL PILOTE LE SOLVEUR SEUL, SANS PYRAMIDE, ET SUR L'HORIZON COMPLET. Les deux
    choix vont ensemble. La pyramide ne touche pas à la physique — `pas_f`
    intègre puis RECOPIE le résultat dans les fenêtres — donc la faire tourner
    ici ne garderait rien de plus et coûterait deux minutes. Ce que ça permet est
    ce qui compte : l'horizon des 120 ticks du run livré tient en huit secondes,
    et une fuite qui n'apparaît qu'au tick 90 est vue. Un verrou tronqué à vingt
    ticks aurait déclaré verte une inondation qui se vide ensuite.

    LES DEUX BORNES SONT EXIGÉES, pas seulement la perte. Une masse qui MONTE est
    une faute au même titre qu'une masse qui baisse : de l'eau apparue de nulle
    part est un remplissage silencieux, la signature §A53. Le run livré monte
    d'ailleurs plus qu'il ne descend (+0,81 % contre −0,22 %)."""
    pas = run_demo_2a1_v3.PasInondation(run_demo_2a1_v3.terrain_du_monde())
    masse_initiale = pas.masse()
    assert masse_initiale > 0.0, (
        "la retenue est vide au tick 0 : le verrou de conservation serait "
        "trivialement vert sur un domaine sans eau.")

    ecart_max = 0.0
    for _ in range(run_demo_2a1_v3.N_TICKS_DEFAUT):
        pas.avancer()
        ecart_max = max(ecart_max, abs(pas.masse() / masse_initiale - 1.0))

    assert ecart_max <= TOLERANCE_MASSE, (
        f"la masse d'eau dérive de {100 * ecart_max:.2f} % sur "
        f"{run_demo_2a1_v3.N_TICKS_DEFAUT} ticks, au-delà des "
        f"{100 * TOLERANCE_MASSE:.0f} % tolérés (0,81 % sur le run livré). "
        "L'eau sort du domaine ou y apparaît : les images ressembleraient "
        "encore à une inondation.")


def test_le_front_avance_entre_le_premier_et_le_dernier_tick(sortie):
    """VERROU 2 — le compte de pixels d'eau croît d'un facteur plancher.

    SUR LES PIXELS RELUS, PAS SUR LE CHAMP FLOTTANT, et la distinction est celle
    que le frère a payée : c'est la chaîne complète qu'on veut exercer, jusqu'aux
    octets du PNG. Un rendu qui cesserait d'écrire l'eau laisserait ce verrou
    rouge, là où un verrou posé sur `cellules_mouillees()` resterait vert.

    PREMIÈRE ET DERNIÈRE IMAGE DE LA SÉRIE, sans trier les deux sortes. Un tick
    rend DEUX écrans — l'interpolé puis l'exact — qui sont séparés d'un demi pas
    de monde ; sur un plancher de 1,4 et une croissance mesurée à 1,84, ce demi
    pas ne pèse rien, et l'exiger d'écrans de même sorte n'ajouterait aucune
    garde."""
    images = sorted(sortie.glob(f"ecran_{run_demo_2a1_v3.COTE}_*.png"))
    assert len(images) == 2 * TICKS_HARNAIS, (
        f"{len(images)} images pour {TICKS_HARNAIS} ticks — la cadence 2:1 "
        "veut exactement deux écrans par tick.")

    premier = _pixels_d_eau(_pixels_rgb(images[0]))
    dernier = _pixels_d_eau(_pixels_rgb(images[-1]))
    assert premier > 0, (
        "aucun pixel d'eau sur la première image : la retenue n'est pas dans le "
        "cadre, et le rapport qui suit ne voudrait rien dire.")
    assert dernier >= PLANCHER_FRONT * premier, (
        f"le front n'avance pas : {premier} pixels d'eau à la première image, "
        f"{dernier} à la dernière, soit un rapport de {dernier / premier:.2f} "
        f"sous le plancher de {PLANCHER_FRONT} (1,84 mesuré sur le run livré, "
        "1,0 exactement si le monde est figé).")


def test_un_dossier_existant_est_refuse(tmp_path):
    """VERROU 3 — hérité des deux frères, et exercé plutôt que promis.

    Il monte son propre cas : rien à voir avec le run du harnais. La garde est
    la classe IMPORTÉE `DossierDejaPresent` — l'attraper par son nom, et non par
    `FileExistsError`, est ce qui distingue le refus délibéré du driver d'une
    collision de système de fichiers survenue ailleurs."""
    dossier = tmp_path / "deja_la"
    dossier.mkdir()
    with pytest.raises(DossierDejaPresent):
        run_demo_2a1_v3.rendre(1, dossier)
