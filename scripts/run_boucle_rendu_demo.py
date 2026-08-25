"""DÉMO DE LA BOUCLE DE RENDU — la chaîne physique → readout → écran, en images.

Ce driver fait TOURNER `src/f1_gpu/boucle_rendu.py` sur le substrat jetable et
écrit ses écrans en PNG. C'est une DÉMO : elle montre que la chaîne tourne, elle
ne mesure rien.

    .venv/bin/python -m scripts.run_boucle_rendu_demo \\
        --cote extrapoler --s-half 0.05

(Depuis la racine du dépôt, et par `-m` : les imports du projet passent par
`src.f1_gpu.…`, donc la racine doit être sur le chemin d'import.)

AUCUNE MESURE, AUCUN CHRONOMÈTRE, AUCUN CHIFFRE DE COÛT. §A61 tient —
l'instrument n'est pas qualifié, et §A63 ne l'a pas qualifié. La cadence de la
boucle est LOGIQUE (§1 de `claude/spec-boucle-rendu-2026-08-24.md`) : « 2:1 »
est un RAPPORT DE COMPTE, deux écrans par `frame()`, pas deux écrans par
33,3 ms. Aucune horloge murale n'entre ici, et le SEUL chiffre imprimé est le
nombre d'images écrites. Un chiffre imprimé par un driver de démo se cite
ensuite comme s'il était une mesure ; celui-ci n'en est pas une et il est le
seul.

CE DRIVER NE PRONONCE RIEN. Ni sur le CÔTÉ — aucune comparaison entre côtés,
aucun jugement de « meilleur » : le côté appartient à la lecture d'orientation,
qui n'est pas ordonnée et dont le prereg n'est pas écrit (§6 de la spec). Ni sur
les compteurs `clamps` / `non_finis` que le tick rend : aucun seuil, aucun
avertissement, le seuil qui invaliderait une mesure appartient au prereg de la
lecture d'orientation (§3 de la spec), pas à ce code.

CE QUE LES IMAGES MONTRENT, ET CE QU'IL NE FAUT PAS Y LIRE — un fait MESURÉ,
écrit ici pour qu'il ne soit pas redécouvert comme un bug. Sur le substrat
jetable, un pas de physique déplace `s` de MOINS d'un niveau de gris : les
DEUX images d'un même tick, une fois quantifiées en 8 bits, sortent
BIT-IDENTIQUES. Elles ne le sont PAS en flottant. Or « tout écran interpolé
devient identique à l'exact » est exactement le symptôme que le §3 de la spec
décrit pour l'ORDRE DÉGÉNÉRÉ du montage — qui ne lève pas et laisse
`clamps == 0`. Deux causes, un seul symptôme à l'œil ; ce qui les sépare est le
champ flottant, et c'est
`tests/test_boucle_rendu.py::test_les_deux_ecrans_d_un_tick_ne_sont_pas_le_meme_champ`
qui tient la différence. Ce qui bouge d'une image à l'autre dans `outputs/`,
c'est le balayage de la fovéa — une cellule fine par tick.

POURQUOI LA CONVERSION `s` → IMAGE VIT ICI, ET NON DANS LA BOUCLE. Le §1 de la
spec endossée la range dans le driver :
`claude/spec-boucle-rendu-2026-08-24.md:29` « **LA CONVERSION `s` → IMAGE VIT
DEHORS, DANS LE DRIVER.** » — et la ligne 36 du même paragraphe : « **La boucle
ne convertit rien et n'importe pas `albedo_ecran`** ». Le motif est que `s_half`
est un paramètre d'APPELANT sans AUCUNE valeur endossée dans le corpus
(`src/albedo.py:27` comme `src/f1_gpu/chemin_de_cout.py:344` en font tous deux
un paramètre), et que le graver dans du code moteur serait choisir un paramètre
hors de toute décision. C'est ce driver qui est l'appelant, et c'est son
utilisateur qui apporte `s_half`.

LE PNG EST ÉCRIT À LA MAIN — `zlib` + `struct`, stdlib —, ni `pillow` ni
`matplotlib`, et ce n'est pas une coquetterie :
  - `pillow` figure en ligne COMMENTÉE dans `requirements.txt`. Il se trouve
    installé dans le venv, mais en dépendre ajouterait une dépendance NON
    DÉCLARÉE, invisible à quiconque lit le fichier de dépendances ;
  - `matplotlib.savefig` — la seule voie image du dépôt aujourd'hui
    (`src/io_utils.py`) — n'est pas bit-déterministe entre versions et insère
    des métadonnées : inutilisable pour un verrou « deux runs, mêmes octets » ;
  - un PNG minimal (signature, `IHDR`, `IDAT`, `IEND`) tient en une trentaine de
    lignes et il est DÉTERMINISTE PAR CONSTRUCTION.

AUCUN HORODATAGE, ni dans les pixels ni dans les métadonnées : aucun chunk
`tIME`, aucun `tEXt`. Écrire un PNG n'est pas une mesure ; l'horodater en serait
une. C'est aussi ce qui rend le verrou (f) possible — un `tIME` rendrait deux
runs différents par construction.
"""
from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path

import numpy as np

from src.f1_gpu.boucle_rendu import (
    COTE_EXTRAPOLER,
    COTE_INTERPOLER,
    BoucleRendu,
)
from src.f1_gpu.chemin_de_cout import albedo_ecran
from src.f1_gpu.pyramide import GeometriePyramide, PyramideFovea
from src.f1_gpu.transferts import TransfertComptable

# LES DEUX CÔTÉS SONT IMPORTÉS, JAMAIS RECOPIÉS EN LITTÉRAL. Un littéral gravé
# ici donnerait un driver qui accepte un côté que la boucle refuse : `CoteInconnu`
# lèverait au montage, loin de sa cause. L'ordre suit celui de l'`α` croissant.
COTES_DISPONIBLES: tuple[str, ...] = (COTE_INTERPOLER, COTE_EXTRAPOLER)

# ── LA GÉOMÉTRIE DE LA DÉMO — MODESTE, CPU NUMPY, ET RIEN D'INVENTÉ ─────────
#
# `slots` n'est PAS fourni : `GeometriePyramide` retombe alors sur l'enveloppe
# DENSE tranche-1 (`slots_uniformes(n_niv, gamma2)`, γ₂ = 3 slots par niveau
# GPU — le fovéal aux offsets B3 nuls, deux d'énergie). Inventer une liste de
# slots ici serait une géométrie NEUVE, jamais mesurée ; le défaut du module
# est celui de M-a/M-c, inchangé.
N_FOV: int = 64          # côté d'une fenêtre active
N0: int = 128            # côté du niveau 0 CPU
N_NIV: int = 4           # niveau 0 CPU + trois niveaux GPU

# CÔTÉ D'ÉCRAN COUVRABLE, et c'est une responsabilité d'APPELANT. Le gather LÈVE
# sur un pixel qu'aucune fenêtre active ne couvre et la boucle ne le rattrape
# pas (§4-7 de la spec) — un remplissage par défaut fausserait ce que la mesure
# doit voir. La couverture se dégrade à mesure que la fovéa balaie ; celle-ci
# est constatée par `test_le_cote_px_de_la_demo_est_couvrable_sur_toute_la_
# duree_du_run` sur quatre fois la durée de la démo, pas supposée.
COTE_PX_DEMO: int = 128

# LE NOMBRE DE TICKS A UN DÉFAUT, LE CÔTÉ ET `s_half` N'EN ONT PAS — et la
# différence n'est pas de commodité. Le côté et `s_half` figurent tous deux dans
# la liste du §6 de la spec, « ce que ce document ne tranche pas » : leur donner
# un défaut trancherait en silence une question laissée ouverte. La LONGUEUR de
# la démo, elle, n'est la valeur de rien — elle ne décide d'aucune grandeur, et
# aucun document n'en attend une.
N_TICKS_DEFAUT: int = 4

# `outputs/` est ignoré par `.gitignore:5` : les images de la démo n'entrent
# jamais dans l'histoire du dépôt, et aucun binaire n'y est committé.
DOSSIER_SORTIE: Path = Path("outputs/boucle_rendu")

# ── LE PNG, À LA MAIN ───────────────────────────────────────────────────────
SIGNATURE_PNG: bytes = b"\x89PNG\r\n\x1a\n"
PROFONDEUR_BITS: int = 8          # 8 bits par pixel
TYPE_COULEUR_GRIS: int = 0        # niveaux de gris, un canal
OCTET_FILTRE_AUCUN: int = 0       # filtre `None` : les octets SONT les pixels
NIVEAUX_GRIS: float = 255.0       # `A ∈ [0, 1]` → `[0, 255]`

# Niveau de compression FIXÉ. Le laisser au défaut de `zlib` marcherait aussi,
# mais un défaut de bibliothèque peut bouger d'une version à l'autre : ce qui
# est écrit ici est ce qui sera relu dans un an.
NIVEAU_ZLIB: int = 6


class LongueurDeDemoInvalide(ValueError):
    """Levée quand le nombre de ticks demandé ne peut produire aucune image.

    Une CLASSE NOMMÉE et non un `ValueError` nu, comme partout dans ce chantier
    (`CoteInconnu`, `MontageBoucleInvalide`, `EcranNonRepresentable`). Un
    `ValueError` nu obligeait son verrou à se rabattre sur le TEXTE du message
    pour le distinguer — une serrure tenue par une chaîne de caractères, qu'une
    reformulation innocente du message ouvrirait sans bruit.

    Hérite de `ValueError` pour le motif du §4-8 : ce qui hérite de
    `RuntimeError` sur ce chemin risque d'être avalé par un `except
    RuntimeError` d'appelant."""


class EcranNonRepresentable(ValueError):
    """Levée quand un écran ne peut pas devenir une image sans mentir.

    Trois fautes, une seule classe, parce qu'elles ont une seule signature :
    un `s` non fini, un albédo hors de la bande `[0, 1]`, une forme ou un
    `dtype` que le PNG ne sait pas porter. Dans les trois cas, quantifier quand
    même produirait un pixel d'APPARENCE NORMALE — la signature §A53 — et le
    précédent de la maison est `src/arcC_rendu.py:57-60` : « FAIL-LOUD, jamais
    un clip silencieux : l'albédo vit dans [0,1) par construction […] ; une
    valeur hors bande, un NaN ou une mauvaise forme sont un BUG AMONT, pas une
    donnée à rattraper en douce. »

    Hérite de `ValueError` comme `CoteInconnu`, et pour le même motif : le §4-8
    de la spec rappelle que tout ce qui hérite de `RuntimeError` sur ce chemin
    risque d'être avalé par un `except RuntimeError` d'appelant.

    ET CE N'EST PAS LE RÉGIME DU CLAMP. `melanger` applique `s ≥ 0` après le
    mélange et le COMPTE (§5 du spec readout) : il ne lève pas, il consigne. Ce
    sont deux régimes différents et ils ne se confondent pas — le clamp couvre
    l'écran INTERPOLÉ, cette classe couvre ce que le clamp ne voit pas."""


def quantifier_en_octets(ecran_s: np.ndarray, s_half: float) -> np.ndarray:
    """Un champ de sédiment `s` → un tableau `uint8` de niveaux de gris.

    La lecture albédo est `chemin_de_cout.albedo_ecran`, APPELÉE et jamais
    recopiée : deux copies de `1 − exp(−s/s_half)` dériveraient l'une de l'autre
    sans que rien ne le signale.

    `s_half` N'EST PAS GARDÉ ICI, ET C'EST DÉLIBÉRÉ. `albedo_ecran` lève déjà
    sur `s_half <= 0` (`src/f1_gpu/chemin_de_cout.py:358-359`). Redoubler la
    garde donnerait deux seuils à tenir d'accord, et le jour où l'un bouge, le
    message nommerait le mauvais fichier.

    LE CONTRÔLE DE FINITUDE PORTE SUR `s`, EN ENTRÉE, ET C'EST LOAD-BEARING. Un
    contrôle posé sur l'albédo serait AVEUGLE au `+inf` : `A = 1 − exp(−∞)` vaut
    `1,0`, une valeur finie, DANS la bande, qui quantifie en un pixel blanc
    parfaitement ordinaire. Le pixel silencieux ne se voit qu'à l'entrée.

    LA BANDE, ELLE, SE CONTRÔLE EN SORTIE, et elle garde la moitié du couple que
    le clamp ne couvre pas. `A = 1 − exp(−s/s_half)` est `< 1` par construction
    mais NÉGATIF dès que `s < 0` ; `np.rint(A·255).astype(np.uint8)` d'un
    négatif BOUCLE en un octet clair. L'écran INTERPOLÉ ne peut pas arriver
    négatif — `melanger` clampe `s ≥ 0` et compte —, mais l'écran EXACT est un
    gather DIRECT sur la pyramide (§2-1) : il ne passe par aucun clamp et porte
    le `s` de la physique tel quel.

    ET LA MOITIÉ HAUTE DE CETTE BANDE EST INATTEIGNABLE — dit ici plutôt que
    laissé croire à une branche vivante. Avec `s` FINI (garanti par le contrôle
    ci-dessus) et `s_half > 0` (garanti par `albedo_ecran`), `exp(−s/s_half)`
    est positif ou nul, donc `A <= 1,0` TOUJOURS : `float(albedo.max()) > 1.0`
    ne peut pas se produire, et aucun verrou ne l'exerce. La condition est
    gardée quand même, comme défense en profondeur si `albedo_ecran` changeait
    de formule — mais elle n'est pas un chemin que ce code emprunte."""
    if not np.isfinite(ecran_s).all():
        raise EcranNonRepresentable(
            f"écran non fini : {int((~np.isfinite(ecran_s)).sum())} cellule(s) "
            f"sur {ecran_s.size} valent NaN ou ±inf. `A = 1 − exp(−s/s_half)` y "
            "est indéfini, et une quantification en 8 bits en ferait un pixel "
            "d'apparence normale — la signature §A53. C'est un BUG AMONT, pas "
            "une donnée à rattraper en douce (`src/arcC_rendu.py:57-60`).")

    albedo = albedo_ecran(ecran_s, s_half)

    if float(albedo.min()) < 0.0 or float(albedo.max()) > 1.0:
        raise EcranNonRepresentable(
            f"albédo hors de la bande [0, 1] : min={float(albedo.min())}, "
            f"max={float(albedo.max())}. `A < 0` signifie `s < 0` — la "
            "physique a rendu un sédiment négatif — et `np.rint(A·255)` "
            "BOUCLERAIT en `uint8` sur un pixel clair d'apparence normale. "
            "Hors bande = BUG AMONT (`src/arcC_rendu.py:57-60`), jamais un "
            "clip silencieux.")

    return np.rint(albedo * NIVEAUX_GRIS).astype(np.uint8)


def _chunk_png(type_: bytes, donnees: bytes) -> bytes:
    """Un chunk PNG : longueur, type, données, CRC32 sur `type + données`."""
    return (struct.pack(">I", len(donnees)) + type_ + donnees
            + struct.pack(">I", zlib.crc32(type_ + donnees) & 0xFFFFFFFF))


def encoder_png_gris(octets: np.ndarray) -> bytes:
    """Un tableau `uint8` `(hauteur, largeur)` → les octets d'un PNG gris 8 bits.

    TROIS CHUNKS, ET RIEN D'AUTRE : `IHDR`, `IDAT`, `IEND`. Pas de `tIME`, pas
    de `tEXt` — aucun horodatage ni dans les pixels ni dans les métadonnées.

    LE FILTRE DE CHAQUE LIGNE VAUT 0 (`None`). Le format autorise cinq filtres
    par ligne ; en choisir un autre donnerait un fichier toujours valide et
    toujours lisible, mais dont les octets compressés ne seraient plus les
    pixels. Zéro partout rend l'encodage TRIVIALEMENT relisible — c'est ce qui
    permet à un test de rouvrir le fichier sans bibliothèque d'image.

    DÉTERMINISTE PAR CONSTRUCTION : rien ici ne lit d'horloge, d'adresse ou
    d'environnement ; le niveau de compression est fixé."""
    if octets.ndim != 2 or octets.dtype != np.uint8:
        raise EcranNonRepresentable(
            f"image de forme {octets.shape} et de dtype {octets.dtype} : "
            "attendu un tableau 2D `uint8`. Une mauvaise forme est un BUG "
            "AMONT (`src/arcC_rendu.py:57-60`) — l'encoder quand même écrirait "
            "un fichier dont les octets ne sont pas les pixels.")

    hauteur, largeur = octets.shape
    entete = struct.pack(">IIBBBBB", largeur, hauteur, PROFONDEUR_BITS,
                         TYPE_COULEUR_GRIS, 0, 0, 0)
    brut = b"".join(bytes([OCTET_FILTRE_AUCUN]) + ligne.tobytes()
                    for ligne in octets)
    return (SIGNATURE_PNG
            + _chunk_png(b"IHDR", entete)
            + _chunk_png(b"IDAT", zlib.compress(brut, NIVEAU_ZLIB))
            + _chunk_png(b"IEND", b""))


def construire_pyramide() -> PyramideFovea:
    """La pyramide de la démo — SUBSTRAT EXISTANT, aucun `pas_f` inventé.

    `PyramideFovea` retombe sur son applicateur par DÉFAUT, `appliquer_jetable`
    (le `F` jetable a1/B5, `src/f1_gpu/pyramide.py:372-375`). Injecter un autre
    applicateur ferait de cette démo une expérience, et il n'y a pas de prereg
    pour ça.

    `transferts.frame_suivante()` clôt le bilan de la DESCENTE INITIALE, payée à
    la construction et hors-série (B4) — le même geste que le harnais de tests.
    Il ne mesure rien : il sépare l'initialisation du régime établi."""
    geo = GeometriePyramide(n_fov=N_FOV, n_niv=N_NIV, n0=N0)
    transferts = TransfertComptable(np)
    pyramide = PyramideFovea(np, geo, transferts)
    transferts.frame_suivante()
    return pyramide


def rendre(cote: str, s_half: float, n_ticks: int, dossier: Path) -> int:
    """`n_ticks` ticks → `2 · n_ticks` PNG numérotés. Rend le nombre écrit.

    Le couple d'un tick sort en `α` CROISSANT et exactement un de ses deux
    termes vaut `α = 1,0` (§2-2 de la spec, endossé par son §8 point 1) : les
    images se numérotent donc dans l'ordre où elles sortent, deux par tick, sans
    trou. La numérotation avance d'un par IMAGE et non par tick — numéroter par
    tick perdrait l'ordre du couple, qui est justement ce que le §2-2 rend
    lisible sans savoir quel côté tourne.

    LE CÔTÉ EST DANS LE NOM DE FICHIER, ET C'EST DE LA PROVENANCE, PAS UN
    VERDICT. Sans lui, deux runs de côtés différents dans le même dossier se
    recouvriraient à moitié et rien ne dirait de quel côté vient quelle image.

    AUCUNE LEVÉE N'EST RATTRAPÉE ICI (§4-7, §4-8) : un trou de couverture du
    gather est un fait de structure, il traverse."""
    if n_ticks <= 0:
        raise LongueurDeDemoInvalide(
            f"rendre : n_ticks={n_ticks} <= 0. Une démo qui n'écrit aucune "
            "image imprimerait « 0 images écrites » et ressemblerait à un run "
            "propre — un silence, pas un résultat.")

    dossier.mkdir(parents=True, exist_ok=True)
    boucle = BoucleRendu(construire_pyramide(), cote)

    compte = 0
    for _ in range(n_ticks):
        for ecran in boucle.tick(COTE_PX_DEMO):
            chemin = dossier / f"ecran_{cote}_{compte:04d}.png"
            chemin.write_bytes(
                encoder_png_gris(quantifier_en_octets(ecran, s_half)))
            compte += 1
    return compte


def construire_parseur() -> argparse.ArgumentParser:
    """Les arguments de la démo — et ce qui n'a pas de défaut n'en aura pas."""
    parseur = argparse.ArgumentParser(
        description="Démo de la boucle de rendu 2:1 — écrit deux images PNG "
                    "par pas de physique. Ne mesure rien.")
    parseur.add_argument(
        "--cote", required=True, choices=COTES_DISPONIBLES,
        help="côté de la boucle. OBLIGATOIRE et SANS DÉFAUT : le côté "
             "appartient à la lecture d'orientation, qui n'est pas ordonnée et "
             "dont le prereg n'est pas écrit (§6 de la spec). Un défaut "
             "trancherait §A62 en silence.")
    parseur.add_argument(
        "--s-half", required=True, type=float, dest="s_half",
        help="paramètre de la lecture albédo `A = 1 − exp(−s/s_half)`. "
             "OBLIGATOIRE et SANS DÉFAUT, pour le MÊME motif que le côté : "
             "`s_half` figure dans la liste du §6 de la spec, et AUCUNE valeur "
             "n'en est endossée dans le corpus — c'est un paramètre "
             "d'APPELANT. `albedo_ecran` lève elle-même sur `s_half <= 0`.")
    parseur.add_argument(
        "--ticks", type=int, default=N_TICKS_DEFAUT,
        help="nombre de pas de physique ; chacun rend DEUX images "
             f"(défaut : {N_TICKS_DEFAUT}). Un défaut est légitime ici — la "
             "longueur d'une démo n'est la valeur de rien et ne décide "
             "d'aucune grandeur.")
    parseur.add_argument(
        "--dossier", type=Path, default=DOSSIER_SORTIE,
        help=f"dossier de sortie, créé si absent (défaut : {DOSSIER_SORTIE}).")
    return parseur


def main(argv: list[str] | None = None) -> int:
    """Le seul chiffre imprimé est le nombre d'images écrites (§A61)."""
    args = construire_parseur().parse_args(argv)
    compte = rendre(args.cote, args.s_half, args.ticks, args.dossier)
    print(f"{compte} images écrites")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
