"""INTERPOLATION DE READOUT — le composant de la porte 33,3 (§A18).

Spec : `claude/spec-interpolation-readout-2026-08-23.md` (corps `ee87000`,
amendements `c72b9b4`, `f310d03`).

CE MODULE N'EST PAS UN COMPOSITEUR ET NE REMPLACE RIEN. Il ÉCRIT un champ `s`
interpolé ; le gather de `chemin_de_cout` le LIT. Ce dernier reste identique à
l'octet — c'est le §1 du spec, et la garde 1 de §A48 (« l'arithmétique est la
plus bête permise ») continue de décrire exactement ce que fait le gather,
parce que ce module ne touche pas à son arithmétique.

UN SEUL NOYAU, AFFINE — `s_out = (1−α)·s_prev + α·s_cur` :

    mode INTERPOLER   α = ½    entre n et n+1     — deux états RÉELS
    mode EXTRAPOLER   α = 1½   depuis n−1 et n    — état jamais existé

Conséquence GRAVÉE : `I` est INVARIANT AU MODE par construction (même
arithmétique, mêmes octets). Une mesure future ne comparera que la latence et
le perçu, jamais un confondant d'arithmétique. Le verrou (b) des tests le
mécanise — une garde promise dans un document est une garde absente (§A62-bis-2).

`s_out` ET `s_prev` SONT ÉPHÉMÈRES. Jamais écrits dans le tenseur que `F`
consomme, jamais lus par `F`. Une fuite corromprait la comptabilité de
conservation et ne se signalerait par AUCUN symptôme — la signature de §A53
retournée. Trois verrous : STRUCTUREL (tampons séparés, aucune référence en
écriture vers l'état), DE PILE (`_verrouiller_consommateur`), DE TEST.

LES IMAGES EXACTES NE PASSENT PAS PAR ICI — c'est le §3 du spec, et c'est un
CONTRAT D'APPELANT, pas une option. Sur 60 images/s à 30 Hz de physique, une
sur deux tombe sur un pas : celle-là appelle le gather DIRECTEMENT sur la
pyramide, sans toucher ce module. Le motif est la comptabilité gravée
(`prereg-ou-vit-le-rendu:87` porte `30·I`, pas `60·I`) : si les images exactes
traversaient le noyau, elles le paieraient aussi et la comptabilité
contredirait un document endossé. `I` est donc le coût d'UNE image
INTERPOLÉE — 30 par seconde, exactement le `30·` de `Δ`.

AUCUNE MESURE ICI. §A61 interdit toute mesure 3D absolue tant que l'instrument
n'est pas qualifié, et §A63 ne l'a pas qualifié. Ce module se construit ; il ne
se chronomètre pas."""
from __future__ import annotations

import sys

# Champ `s` dans `(h, hu, hv, s)` — `pyramide.py:75`, `CHAMPS_PAR_SYSTEME = 4`.
INDICE_CHAMP_S: int = 3

# Les deux modes, qui sont deux valeurs d'un seul noyau (§2 du spec).
ALPHA_INTERPOLER: float = 0.5
ALPHA_EXTRAPOLER: float = 1.5


# Modules dont la seule présence dans la pile signale que le chemin de l'ÉTAT
# est en train d'atteindre un produit de readout. La liste est la MÉCANISATION
# du §4 du spec : elle doit GRANDIR avec tout nouveau module qui écrit l'état
# ou exécute `F`. Le nom SIMPLE est comparé : `src.f1_gpu.pyramide` -> `pyramide`.
#
# CE QUE CETTE GARDE NE PEUT PAS GARDER — une garde par liste ne couvre que les
# modules NOMMÉS ici ; tout module d'état futur, pas encore écrit ou pas encore
# ajouté, lui échappe jusqu'à ce que quelqu'un l'y mette. C'est pourquoi le
# verrou de pile n'est JAMAIS le premier verrou, mais le second : le verrou
# STRUCTUREL — `s_prev` et `s_out` vivent hors du tenseur que `F` consomme, et
# `melanger` n'écrit nulle part dans ce tenseur — tient quel que soit
# l'appelant, nommé ici ou non, et c'est LUI qui porte la garantie. La preuve :
# cinq modules manquaient à la première rédaction de cette liste
# (`temoin_fidele`, `production_fidele`, `fovea_2niveaux`, `solver_wetdry`,
# `sediment`) et ont été trouvés par une REVUE qui a confronté la liste au
# contenu réel de `src/` — pas par la garde elle-même, qui ne peut par
# construction rien dire de ce qu'elle ne nomme pas.
_MODULES_ETAT: frozenset[str] = frozenset({
    "exner_gpu",
    "fovea_2niveaux",                # exécute `pas_deux_niveaux`
    "ledger",                        # comptabilité de conservation
    "production_fidele",             # exécute `pas_f_fidele`
    "pyramide",                      # exécute `F` via `pas_f` (`pyramide.py:501`)
    "sediment",                      # chemin re-dérivé CPU
    "solver_wetdry",                 # chemin re-dérivé CPU
    "substrat_fidele",
    "substrat_fusionne",
    "substrat_fusionne_3d",
    "substrat_fusionne_3d_param",
    "substrat_jetable",
    "substrat_jetable_3d",
    "substrat_l3",
    "temoin_fidele",                 # exécute `pas_f_fidele`
    "transferts",                    # descend/remonte l'état
})


class ReadoutInterditDansEtat(RuntimeError):
    """Levée quand le chemin de l'ÉTAT tente d'atteindre un produit de readout.

    Hérite de `RuntimeError` — la faute porte sur QUI appelle. N'est JAMAIS
    rattrapée dans ce module : une serrure qui se rattrape est une promesse,
    pas une serrure."""


# Trace INDÉLÉBILE des violations — voir `_verrouiller_consommateur`.
_VIOLATIONS: list[str] = []


def violations_consignees() -> tuple[str, ...]:
    """Violations survenues depuis le début du processus.

    **TOUT DRIVER QUI PRONONCE UN CHIFFRE DOIT LIRE CECI AVANT DE L'ÉCRIRE.**
    Une liste non vide signifie qu'un produit de readout a été atteint depuis
    le chemin de l'état : la comptabilité de conservation n'est plus garantie.

    POURQUOI LA TRACE EXISTE — la leçon de `chemin_de_cout`, trouvée par
    mutation le 03/08 et non par relecture. `ReadoutInterditDansEtat` hérite de
    `RuntimeError` ; un appelant qui écrit `except RuntimeError` — geste banal
    et bien intentionné — **avale la serrure en silence**, et la garde redevient
    une promesse. L'exception peut être étouffée ; la trace, non."""
    return tuple(_VIOLATIONS)


def _verrouiller_consommateur() -> None:
    """Refuse l'appel si la pile traverse un module d'état, et CONSIGNE la
    tentative AVANT de lever.

    Remonte les frames par `sys._getframe` plutôt que par `inspect.stack()` :
    ce dernier ouvre et lit les fichiers source de chaque frame, ce qui
    coûterait des millisecondes dans un chemin dont `I` sera un jour mesuré."""
    frame = sys._getframe(1)
    while frame is not None:
        nom = frame.f_globals.get("__name__", "")
        if nom.rsplit(".", 1)[-1] in _MODULES_ETAT:
            _VIOLATIONS.append(nom)          # consignée AVANT la levée
            raise ReadoutInterditDansEtat(
                f"interpolation de readout appelée depuis « {nom} », un "
                "module d'ÉTAT. §4 du spec : `s_out` et `s_prev` sont "
                "ÉPHÉMÈRES — jamais écrits dans l'état, jamais lus par `F`. "
                "Une fuite corromprait la comptabilité de conservation sans "
                "produire aucun symptôme. Tentative consignée — voir "
                "`violations_consignees()`.")
        frame = frame.f_back


class VueInterpolee:
    """Ce que le gather reçoit à la place de la pyramide.

    Elle expose EXACTEMENT les quatre attributs que `gather_chemin_de_cout`
    touche — `xp`, `geo`, `centre_fin`, `fenetres` — et rien d'autre. Le
    gather n'est pas modifié : il reçoit un objet de même surface dont le
    champ lu est le `s` interpolé.

    `fenetres[j]` a la forme `(n_slots(j), n_systemes(j), 1, n_fov, n_fov)` :
    l'axe des champs est un SINGLETON, donc le gather s'appelle avec
    `indice_champ=0`. Garder un vrai tableau contigu — plutôt qu'un proxy qui
    ré-indexerait en Python — évite de remettre dans le chemin le facteur 109
    que `chemin_de_cout` a dû tuer (boucle Python contre kernel)."""

    __slots__ = ("xp", "geo", "centre_fin", "fenetres", "clamps")

    def __init__(self, xp, geo, centre_fin, fenetres, clamps: int):
        self.xp = xp
        self.geo = geo
        self.centre_fin = centre_fin
        self.fenetres = fenetres
        self.clamps = int(clamps)


class TamponReadout:
    """Détient `s_prev` et `s_out`, TOUS DEUX HORS du tenseur de `F`.

    `s_prev` est un état de readout load-bearing — le premier du code. La
    défense est au §4-bis du spec, et elle tient sur deux jambes : ces gardes
    sont scellées au périmètre de l'INSTRUMENT (ce module est du code moteur),
    et `s_prev` ne porte que des COPIES de champs réellement produits par `F` —
    de l'HISTOIRE, pas de la SYNTHÈSE. La synthèse, `s_out`, reste sans état.

    DIMENSIONNEMENT — sur les PLANS `s`, pas sur les SLOTS : un slot `c = 8`
    porte DEUX systèmes (`SYSTEMES_PAR_C`, §A16), donc deux plans. Pour V4,
    11 slots font 15 plans."""

    def __init__(self, pyramide):
        xp = pyramide.xp
        self._s_prev = {}
        self._s_out = {}
        for j in pyramide.geo.niveaux_gpu:
            fen = pyramide.fenetres[j]
            forme = (fen.shape[0], fen.shape[1], 1, fen.shape[3], fen.shape[4])
            self._s_prev[j] = xp.zeros(forme, dtype=fen.dtype)
            self._s_out[j] = xp.zeros(forme, dtype=fen.dtype)

    def capturer(self, pyramide) -> None:
        """`s_prev` ← COPIE du `s` courant. À appeler AVANT le pas de `F`.

        C'est une copie RÉELLE, et ce n'est pas un choix : `F` écrit EN PLACE
        (`pyramide.py:501` — le tampon de sortie passé à `pas_f` EST le tampon
        d'entrée), donc aucun échange de pointeurs ne peut conserver l'état
        précédent. Ce coût est DANS `I` (§3-bis du spec)."""
        for j in pyramide.geo.niveaux_gpu:
            source = pyramide.fenetres[j][:, :, INDICE_CHAMP_S, :, :]
            self._s_prev[j][:, :, 0, :, :] = source

    def melanger(self, pyramide, alpha: float) -> VueInterpolee:
        """`s_out = (1−α)·s_prev + α·s_cur`, puis clamp `s ≥ 0` COMPTÉ.

        Le clamp ne lève JAMAIS : ceci est du code moteur, il tourne. Le seuil
        au-delà duquel un comptage invalide une mesure appartient au prereg de
        la lecture d'orientation, pas ici (§5 du spec)."""
        _verrouiller_consommateur()
        xp = pyramide.xp
        a = float(alpha)
        clamps = 0
        for j in pyramide.geo.niveaux_gpu:
            s_cur = pyramide.fenetres[j][:, :, INDICE_CHAMP_S, :, :]
            sortie = self._s_out[j][:, :, 0, :, :]
            sortie[...] = (1.0 - a) * self._s_prev[j][:, :, 0, :, :] + a * s_cur
            sous_zero = sortie < 0.0
            clamps += int(xp.count_nonzero(sous_zero))
            sortie[sous_zero] = 0.0
        return VueInterpolee(xp, pyramide.geo, pyramide.centre_fin,
                             self._s_out, clamps)
