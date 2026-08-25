"""Les verrous de la démo 2:1 v2 — le contraste des côtés sur l'inondation.

Régime §A64-4 : cette démo ne mesure rien, aucune décision verdict-grade n'en
dépend, la cérémonie est celle du build. LE BRIEF EN DEMANDAIT TROIS OU QUATRE ;
il y en a SIX, et l'écart se justifie plutôt qu'il ne se cache. Les quatre premiers
sont ceux du brief. Le cinquième est le contrôle de source hérité des frères, que
la docstring du driver CITE — s'il n'existait pas, cette citation promettrait une
garde absente. Le sixième existe parce que deux mutants ont survécu aux quatre
premiers, sur des mécaniques que les docstrings du driver promettaient sans que
rien ne les tienne. Chacun garde une chose que l'œil ne garde pas :

  1. LES DEUX CÔTÉS DIFFÈRENT VISIBLEMENT sur au moins un tick. C'est LA propriété
     livrable de cette v2 — la v1 montrait la cadence d'un seul côté, et une
     cadence seule ne dit rien du choix qu'elle sert.
  2. LES DEUX SÉRIES SONT APPARIÉES. Prouvé par le plus dur : l'écran EXACT d'un
     tick est le MÊME OCTET des deux côtés. Il tombe si l'une des deux histoires
     dérive d'une seule frame.
  3. LE DOSSIER EXISTANT EST REFUSÉ — garde héritée de la v1, exercée ici parce
     qu'une garde qu'aucun test n'exerce est une promesse.
  4. LE FRONT EST PRÉSENT À L'IMAGE. Sans lui, la v2 tournerait sur un champ plat
     et personne ne le verrait : c'est exactement ce qui est arrivé à la première
     échelle essayée (écart-type 1,7 niveau de gris).

Les helpers de relecture PNG et de contrôle de source sont IMPORTÉS des fichiers
de tests des frères, jamais recopiés."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from scripts import run_demo_2a1_v2
from scripts.run_demo_2a1 import DossierDejaPresent
from tests.test_boucle_rendu import (
    ATTRIBUTS_HORLOGE,
    MODULES_HORLOGE,
    _asserts_et_filets_larges,
    _identifiants,
)
from src.f1_gpu.boucle_rendu import COTE_INTERPOLER, BoucleRendu
from src.f1_gpu.interpolation_readout import INDICE_CHAMP_S
from src.f1_gpu.pyramide import predire_bloc_cpu
from tests.test_demo_2a1 import _pixels

S_HALF = 0.9           # celui du run livré : la profondeur balaie la bande sans saturer
TICKS = 4              # quatre ticks : assez pour un tick franc, court à exécuter

# LES DEUX PLANCHERS, MESURÉS ET NON CHOISIS.
#
# CONTRASTE ENTRE CÔTÉS : la moyenne par tick vaut 2,4 à 10 niveaux de gris sur le
# run livré, avec des pointes à 36. Elle N'EST PAS uniforme — l'écart entre côtés
# vaut un demi pas de monde, et le monde bouge inégalement au fil d'une rupture de
# barrage. Exiger le plancher à CHAQUE tick reviendrait à exiger que l'histoire
# soit régulière, ce qu'aucune physique ne promet ; le verrou exige donc AU MOINS
# UN tick au-dessus, ce que le brief demande dans ces termes. À pas d'histoire 1,
# la moyenne tombe à 0,19 : le plancher sépare franchement les deux régimes.
SEUIL_CONTRASTE = 4.0

# FRONT PRÉSENT : écart-type spatial de l'écran exact. Mesuré 37 à 66 sur le run
# livré ; 1,7 quand l'inondation était étalée sur tout le monde de la pyramide,
# c'est-à-dire quand il n'y avait rien à voir. Un plancher à 10 est cinq fois
# au-dessus du cas mort et trois fois sous le cas vivant.
SEUIL_FRONT = 10.0


def _series(dossier: Path) -> tuple[list[Path], list[Path]]:
    return (sorted(dossier.glob("ecran_interpoler_*.png")),
            sorted(dossier.glob("ecran_extrapoler_*.png")))


@pytest.fixture(scope="module")
def sortie(tmp_path_factory):
    """Un seul run pour les trois verrous qui en ont besoin — le quatrième monte
    son propre cas. La démo est déterministe : rien ne dépend de l'ordre."""
    dossier = tmp_path_factory.mktemp("v2") / "sortie"
    run_demo_2a1_v2.rendre_couple(S_HALF, TICKS, dossier)
    return dossier


def test_les_deux_cotes_different_visiblement_sur_au_moins_un_tick(sortie):
    """LA PROPRIÉTÉ LIVRABLE DE LA v2.

    À index égal, les deux séries montrent le même instant lu de deux façons :
    l'une par le côté qui interpole, l'autre par celui qui extrapole. Si elles ne
    se séparaient pas, le choix du côté serait sans effet visible et cette démo
    n'aurait rien à montrer.

    AU MOINS UN TICK, ET C'EST LE BON QUANTIFICATEUR. L'écart entre côtés vaut un
    demi pas de monde ; le monde d'une rupture de barrage bouge vite puis se
    calme. Exiger le plancher à chaque tick exigerait de la physique une régularité
    qu'elle ne promet pas, et le verrou se mettrait à garder l'histoire au lieu de
    garder le rendu."""
    interpoler, extrapoler = _series(sortie)
    assert len(interpoler) == len(extrapoler) == 2 * TICKS

    ecarts = [float(np.abs(_pixels(a) - _pixels(b)).mean())
              for a, b in zip(interpoler, extrapoler)]
    assert max(ecarts) >= SEUIL_CONTRASTE, (
        f"aucun index ne sépare les deux côtés d'au moins {SEUIL_CONTRASTE} "
        f"niveaux de gris (maximum observé : {max(ecarts):.2f}). Le choix du côté "
        "est alors sans effet visible, et cette démo ne montre rien")


def test_les_deux_series_sont_appariees_index_par_index(sortie):
    """L'APPARIEMENT, PROUVÉ PAR LE PLUS DUR QU'ON PUISSE EXIGER.

    Les deux boucles partent du même monde, avancent du même balayage et
    consomment la même histoire ; à index égal, elles montrent le même instant.
    Le dire ne coûte rien — ce verrou le PROUVE : l'écran `α = 1,0` d'un tick est
    rendu par un gather DIRECT sur la pyramide, identique des deux côtés puisque
    l'état l'est. `interpoler` le place en second de son couple, `extrapoler` en
    premier du sien ; ce sont les mêmes octets.

    IL TOMBE SUR LA FAUTE QUI COMPTE. Si les deux `PasHistoire` se partageaient un
    index, ou si l'avance de frame se désynchronisait, les deux côtés verraient des
    frames différentes et cette égalité octet-pour-octet disparaîtrait — sans
    qu'aucune levée ne se produise. C'est le seul endroit où cette faute se voit."""
    interpoler, extrapoler = _series(sortie)

    for tick in range(TICKS):
        exact_interpoler = interpoler[2 * tick + 1].read_bytes()
        exact_extrapoler = extrapoler[2 * tick].read_bytes()
        assert exact_interpoler == exact_extrapoler, (
            f"tick {tick} : l'écran exact diffère entre les deux séries "
            f"({interpoler[2 * tick + 1].name} contre {extrapoler[2 * tick].name}). "
            "Les deux côtés ne regardent donc pas le même instant, et toute "
            "comparaison entre les séries compare deux mondes")


def test_un_dossier_existant_est_refuse(tmp_path):
    """Garde héritée de la v1, exercée ici : une garde qu'aucun test n'exerce est
    une promesse. Le danger visé n'est pas l'écrasement franc, qui se verrait,
    mais le MÉLANGE de deux runs dans une même numérotation."""
    dossier = tmp_path / "deja_la"
    dossier.mkdir()

    with pytest.raises(DossierDejaPresent):
        run_demo_2a1_v2.rendre_couple(S_HALF, TICKS, dossier)


def test_le_front_est_present_a_l_image(sortie):
    """SANS CE VERROU, LA v2 POURRAIT TOURNER SUR UN CHAMP PLAT.

    Et ce n'est pas une hypothèse : à la première échelle essayée — l'inondation
    étalée sur tout le monde de la pyramide — l'écran exact avait un écart-type de
    1,7 niveau de gris et 9 valeurs distinctes sur 256. La chaîne tournait, les
    verrous d'appariement passaient, et il n'y avait RIEN à voir. Le substrat
    structuré est l'objet même de cette v2 ; il lui faut donc sa garde."""
    interpoler, _ = _series(sortie)

    sigmas = [float(_pixels(chemin).std()) for chemin in interpoler]
    assert min(sigmas) >= SEUIL_FRONT, (
        f"un écran a un écart-type spatial de {min(sigmas):.2f} niveaux, sous le "
        f"plancher {SEUIL_FRONT} : le champ est plat, il n'y a pas de front à "
        "juger et la démo ne montre que du gris")


def test_la_source_de_la_v2_ne_porte_ni_filet_large_ni_assert_ni_horloge():
    """Les trois gardes de source des frères, appliquées au driver neuf.

    §4-8 — aucun `except RuntimeError`, `Exception` ni nu. §A43 — aucun `assert`.
    Et aucune horloge : l'horodatage du dossier appartient à l'appelant, ce qui est
    précisément la conséquence de cette garde et non une commodité."""
    source = Path(run_demo_2a1_v2.__file__)
    asserts, larges = _asserts_et_filets_larges(
        ast.parse(source.read_text(encoding="utf-8")))

    assert not asserts, f"`assert` ligne(s) {asserts} — disparaît sous `python -O`"
    assert not larges, f"filet trop large : {larges}"

    identifiants = _identifiants(source)
    horloges = sorted((identifiants & MODULES_HORLOGE)
                      | (identifiants & ATTRIBUTS_HORLOGE))
    assert not horloges, f"horloge dans le driver v2 : {horloges}"
    assert "cupy" not in identifiants, "la démo est CPU numpy (aucun GPU ici)"


def test_la_mecanique_de_l_histoire_amorce_et_avance_d_un_pas_par_tick():
    """LE CINQUIÈME VERROU, ET IL EXISTE PARCE QUE DEUX MUTANTS ONT SURVÉCU.

    Les quatre verrous ci-dessus regardent les IMAGES. Deux fautes de la mécanique
    qui les produit leur échappent entièrement, et les deux étaient PROMISES par
    les docstrings du driver sans être gardées nulle part — une garde promise est
    une garde absente (§A62-bis-2) :

      (a) L'AVANCE À CHAQUE NIVEAU au lieu du seul premier. `frame()` appelle
          `pas_f` une fois par niveau GPU ; avancer à chacun consomme l'histoire
          TROIS fois trop vite. Les deux côtés se trompant symétriquement,
          l'appariement tient, le contraste grandit même, et rien ne se voit. La
          démo raconte alors une cadence qui n'est pas celle de la boucle.
      (b) L'AMORÇAGE RETIRÉ. Sans lui, le `s_prev` du premier tick est le monde
          JETABLE et son `s_cur` la frame 0 de l'histoire : le premier écran
          intermédiaire fond deux mondes sans rapport. Une seule image sur toute
          la séquence, celle qu'on regarde en premier.

    Ce verrou tient les deux sur la MÉCANIQUE plutôt que sur le rendu, parce que
    c'est là qu'elles vivent."""
    images = run_demo_2a1_v2.charger_profondeur()
    pyramide, pas = run_demo_2a1_v2.construire_pyramide(images)

    # (b) amorçage : les fenêtres portent la frame 0, pas le monde jetable.
    #
    # LE TÉMOIN EST CONSTRUIT INDÉPENDAMMENT, et c'est la correction d'une
    # première écriture CIRCULAIRE : elle descendait `pyramide.monde0` — c'est-à-
    # dire ce que la pyramide porte, quel qu'il soit — et comparait le résultat
    # aux fenêtres. Amorçage retiré, les deux côtés de l'égalité devenaient le
    # monde jetable et le verrou restait VERT. Constaté par mutation. Le témoin
    # impose donc `images[0]` de l'extérieur.
    assert pas.indice == 0
    geo = pyramide.geo
    niveau = list(geo.niveaux_gpu)[0]
    monde_temoin = np.array(pyramide.monde0, copy=True)
    monde_temoin[:, INDICE_CHAMP_S] = images[0]
    attendu = np.asarray(predire_bloc_cpu(
        monde_temoin, geo, niveau, geo.origines(niveau, pyramide.centre_fin),
        0, geo.n_fov, geo.n_systemes_du_niveau(niveau)))
    premier = pyramide.fenetres[niveau]
    assert np.array_equal(premier[:, :, INDICE_CHAMP_S],
                          attendu[:, :, INDICE_CHAMP_S]), (
        "les fenêtres ne portent pas la frame 0 de l'histoire à la construction : "
        "le premier écran intermédiaire fondra le monde jetable et l'inondation")

    # (a) avance : exactement un pas d'histoire par tick, quel que soit le
    # nombre de niveaux que `frame()` traverse.
    boucle = BoucleRendu(pyramide, COTE_INTERPOLER)
    for tick in range(1, 4):
        boucle.tick(run_demo_2a1_v2.COTE_PX_DEMO)
        assert pas.indice == tick * run_demo_2a1_v2.PAS_HISTOIRE, (
            f"après {tick} tick(s), l'histoire est à la frame {pas.indice} au "
            f"lieu de {tick * run_demo_2a1_v2.PAS_HISTOIRE} : elle n'avance pas "
            "d'un pas par tick, donc la démo ne montre pas la cadence de la boucle")

    # (c) la garde fail-loud, EXERCÉE. Elle vit sur un chemin que le
    # fonctionnement normal n'emprunte jamais : une mutation qui la remplace par
    # un repli silencieux passe donc TOUS les autres verrous — constaté. Une
    # docstring qui promet « fail-loud » sans qu'aucun test n'appelle le chemin
    # promet une garde absente.
    with pytest.raises(run_demo_2a1_v2.NiveauNonReconnu):
        pas(np.zeros_like(premier), np)
