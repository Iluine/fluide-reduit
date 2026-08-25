"""Les TROIS verrous de la démo 2:1 — pas un de plus, et le plafond est délibéré.

`PREREGISTRATION.md:10507` « Tout le reste est donc verrouillé par tests et
consigné par des messages de commit factuels » : cette démo ne mesure rien et
aucune décision verdict-grade n'en dépend, donc la cérémonie qui l'entoure est
celle du build. Trois verrous suffisent, et chacun garde une chose que
l'inspection à l'œil ne garde pas :

  1. LES DEUX IMAGES D'UN TICK DIFFÈRENT VISIBLEMENT — au-dessus d'un plancher de
     niveaux de gris, et non par un simple « les fichiers ne sont pas identiques ».
     C'est LA propriété livrable, celle que la démo précédente n'avait pas, et la
     seule raison d'exister de ce fichier. La nuance a été payée : voir la
     docstring du verrou.
  2. LE DOSSIER EXISTANT EST REFUSÉ. « Jamais par-dessus un artefact existant »
     est une promesse tant qu'aucun test ne l'exerce.
  3. LA SOURCE NE PORTE NI HORLOGE, NI `assert`, NI FILET LARGE. Le contrôle du
     driver frère, appliqué au neuf : les helpers sont IMPORTÉS de son fichier de
     tests, jamais recopiés — deux copies d'un contrôle dérivent l'une de l'autre
     sans que rien ne le signale.

Ce que ces verrous NE gardent pas, et c'est assumé : ni l'aspect des images, ni
le déplacement du motif, ni la valeur de `s_half`. Ce sont des réglages de run,
ils vivent dans la ligne de commande et dans le message de commit."""
from __future__ import annotations

import ast
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from scripts import run_demo_2a1
from tests.test_boucle_rendu import (
    ATTRIBUTS_HORLOGE,
    MODULES_HORLOGE,
    _asserts_et_filets_larges,
    _chunks_png,
    _identifiants,
)

S_HALF = 0.15          # celui du run livré : l'albédo balaie la bande sans saturer
TICKS = 2              # deux ticks suffisent pour comparer un couple

# LE SEUIL DE VISIBILITÉ, ET POURQUOI IL Y EN A UN. Un verrou qui exigeait
# seulement « les deux fichiers diffèrent » PASSAIT sur le substrat jetable —
# constaté, pas supposé : une poignée de pixels y diffèrent d'UN niveau, ce qui
# suffit à changer les octets du PNG et ne se voit sur aucun écran. Le verrou
# promettait alors la visibilité et n'attrapait que la non-identité. L'écart
# moyen mesuré vaut 0,00 niveau sous le `F` jetable et ~18 sous la translation :
# tout plancher franchement au-dessus du bruit de quantification les sépare, et
# 4 laisse trois fois la marge sans rien mesurer.
SEUIL_VISIBLE = 4.0


def _pixels(chemin: Path) -> np.ndarray:
    """Relit un PNG gris 8 bits À LA MAIN, sans bibliothèque d'image.

    Même idiome que `test_le_png_se_relit_octet_par_octet_sans_bibliotheque_d_
    image` : `pillow` est en ligne COMMENTÉE dans `requirements.txt`, et une
    dépendance non déclarée dans un verrou est une dette invisible."""
    chunks = dict(_chunks_png(chemin.read_bytes()))
    largeur, hauteur = struct.unpack(">II", chunks["IHDR"][:8])
    brut = np.frombuffer(zlib.decompress(chunks["IDAT"]), dtype=np.uint8)
    return brut.reshape(hauteur, largeur + 1)[:, 1:].astype(np.int16)


def test_les_deux_images_d_un_tick_different_visiblement(tmp_path):
    """LA PROPRIÉTÉ LIVRABLE, et le seul motif d'exister de ce driver.

    Sur le substrat du frère, un pas de physique déplace `s` de moins d'un niveau
    de gris : la chaîne tourne, le 2:1 ne se voit pas. Le `pas_f` de translation
    injecté ici fait évoluer `s` franchement, et c'est ce verrou qui l'atteste —
    SUR LES PIXELS RELUS, pas sur le champ flottant. La distinction est
    load-bearing : c'est exactement à la quantification 8 bits que la démo
    précédente perdait la différence, et un verrou posé en amont l'aurait
    déclarée verte.

    ET IL A FALLU DEUX ÉCRITURES. La première exigeait seulement que les deux
    FICHIERS diffèrent — elle SURVIVAIT au remplacement du pas de translation par
    le `F` jetable, parce qu'une poignée de pixels y diffèrent d'un niveau, ce qui
    change les octets du PNG et ne se voit nulle part. Le verrou promettait la
    visibilité et n'attrapait que la non-identité : une garde promise est une
    garde absente. Le seuil le rend effectif — et il le rend au sens strict, la
    mutation étant maintenant tuée."""
    dossier = tmp_path / "sortie"
    ecrites = run_demo_2a1.rendre("interpoler", S_HALF, TICKS, dossier)
    images = sorted(dossier.glob("*.png"))

    assert ecrites == 2 * TICKS == len(images)
    for premier in range(0, len(images), 2):
        gauche = _pixels(images[premier])
        droite = _pixels(images[premier + 1])
        ecart = float(np.abs(gauche - droite).mean())
        assert ecart >= SEUIL_VISIBLE, (
            f"tick {premier // 2} : les deux images ne diffèrent que de {ecart:.2f} "
            f"niveau(x) de gris en moyenne, sous le plancher {SEUIL_VISIBLE}. Le "
            "2:1 est alors invisible à l'écran — précisément le défaut que ce "
            "driver existe pour corriger")


def test_un_dossier_existant_est_refuse(tmp_path):
    """« Jamais par-dessus un artefact » — mécanisé, pas promis.

    Le danger n'est pas l'écrasement franc, qui se verrait : c'est le MÉLANGE.
    Un run plus court retombant dans un dossier peuplé y laisse la queue du
    précédent, numérotée dans la même série et indiscernable à l'œil."""
    dossier = tmp_path / "deja_la"
    dossier.mkdir()

    with pytest.raises(run_demo_2a1.DossierDejaPresent):
        run_demo_2a1.rendre("interpoler", S_HALF, TICKS, dossier)


def test_la_source_de_la_demo_2a1_ne_porte_ni_filet_large_ni_assert_ni_horloge():
    """Les trois gardes de source du frère, appliquées au driver neuf.

    §4-8 — aucun `except RuntimeError`, `Exception` ni nu : le gather lève
    `RuntimeError` pour ses trous de couverture et ne se rattrape pas.

    §A43 — aucun `assert` : il disparaît sous `python -O`.

    AUCUNE HORLOGE, et ce verrou porte une conséquence directe de conception. Le
    brief demandait un dossier de sortie HORODATÉ ; un horodatage fabriqué par le
    driver voudrait dire `datetime` ou `time` dans cette source, c'est-à-dire la
    famille même que ce contrôle interdit. C'est pourquoi `--dossier` est
    obligatoire et l'horodatage appartient à l'appelant. Le contrôle porte sur les
    MODULES importés ET sur les attributs appelés : `import time` seul ne dit rien
    de `datetime.datetime.now()`."""
    source = Path(run_demo_2a1.__file__)
    asserts, larges = _asserts_et_filets_larges(
        ast.parse(source.read_text(encoding="utf-8")))

    assert not asserts, f"`assert` ligne(s) {asserts} — disparaît sous `python -O`"
    assert not larges, f"filet trop large : {larges}"

    identifiants = _identifiants(source)
    horloges = sorted((identifiants & MODULES_HORLOGE)
                      | (identifiants & ATTRIBUTS_HORLOGE))
    assert not horloges, (
        f"horloge dans le driver de démo : {horloges}. La cadence est LOGIQUE, et "
        "une date lue est la porte par laquelle un horodatage entre dans un PNG")
    assert "cupy" not in identifiants, "la démo est CPU numpy (aucun GPU ici)"
