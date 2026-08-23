"""INTERPOLATION DE READOUT — le composant de la porte 33,3 (§A18).

Spec : `claude/spec-interpolation-readout-2026-08-23.md` (corps `ee87000`,
amendements `c72b9b4`, `f310d03`, `4c5252a`).

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

⚠ LE REGISTRE SOUS FOVÉA MOBILE — CE QUI LE PERD. Les fenêtres sont
FOVÉA-RELATIVES — `origines(j, centre_fin)` fait dépendre l'origine `ox` du
centre — et `PyramideFovea.frame()` translate le contenu
(`fen[...] = xp.roll(fen, -dx, axis=-1)`, `pyramide.py:491`) PUIS applique `F`
(`pyramide.py:501`), dans la MÊME boucle par niveau. Une capture prise AVANT
`frame()` n'est JAMAIS roulée : à l'adresse locale `(iy, ix)`, `s_cur` porte
alors la cellule monde `x` et `s_prev` la cellule monde `x − dx`. Le mélange
fantôme d'une cellule fine par pas, et sur les colonnes ENTRANTES il produit une
valeur QUI N'A JAMAIS EXISTÉ — sans clamp, sans levée, `clamps == 0`. La
signature exacte de §A53.

LA VOIE 2 EST TRANCHÉE (§13 du spec, AMENDEMENT 6), ET C'EST UNE DÉRIVATION, PAS
UNE PRÉFÉRENCE. Les deux autres voies meurent de contredire le §2, qui est
endossé — un seul noyau affine : rouler `s_prev` avec la fenêtre exigerait un
masque de validité sur les colonnes entrantes, donc un `α` PAR CELLULE, donc un
AUTRE noyau ; mélanger en coordonnées MONDE changerait l'arithmétique du noyau
(lecture décalée), donc un AUTRE noyau. Contredire un document endossé n'est pas
une option de design, et c'est le motif — pas le goût — qui a choisi la voie 2.

LE MÉCANISME — `ApplicateurCaptureRegistre`, en bas de ce module. Il ENVELOPPE
`pas_f`, qui est INJECTABLE (`pyramide.py:392-395` : le stepper est un ATTRIBUT).
`frame()` l'appelle une fois PAR NIVEAU, exactement POST-ROLL et PRÉ-PAS DE
PHYSIQUE (`pyramide.py:501`) — le seul point de la boucle où `s_prev` (état
`n−1`) et `s_cur` (état `n`) vivent dans les MÊMES coordonnées fenêtre par
construction, COLONNES ENTRANTES COMPRISES, puisque celles-ci reçoivent du roll
leur contenu `n−1` (§13-2 du spec). Pas de masque, pas de cellule qui n'a jamais
coexisté, noyau du §2 INTACT. Et `pyramide.py` n'est pas touché d'une ligne :
c'est cette contrainte qui a choisi le levier.

QUI ÉCRIT QUOI — la retouche du §4 exigée par le §13-4, et elle est LOAD-BEARING.
Sous la voie 2 le SENS D'ÉCRITURE S'INVERSE :

    `frame()` (par l'applicateur) ÉCRIT dans le tampon de readout ; le module,
    lui, ne fait que LIRE l'état.

Sans cette phrase, la garde « le module ne détient aucune référence en écriture
vers l'état » devient FAUSSE À LA LETTRE dans l'autre sens — un lecteur y verrait
une interdiction que la voie 2 viole, alors qu'elle la respecte dans le seul sens
qui compte. LA GARANTIE, ÉCRITE DANS CE SENS-LÀ : RIEN DU READOUT NE REMONTE
JAMAIS DANS L'ÉTAT. `capturer_niveau` LIT la fenêtre et écrit dans `_s_prev` ;
`melanger` LIT la fenêtre et écrit dans `_s_out` ; aucune des deux n'écrit un
seul octet dans le tenseur que `F` consomme, et le verrou STRUCTUREL de test le
vérifie sur les trois entrées.

`ReadoutHorsRegistre` RESTE, en DÉFENSE EN PROFONDEUR — voir sa docstring.

DOUZE VERROUS VERTS N'ONT PAS VU CE DÉFAUT parce qu'AUCUN n'appelait `frame()` :
la fovéa était immobile dans tous les tests. Un verrou ne garde que ce que son
état de test allume (§A53) — et l'état de test le plus coûteux à oublier est
celui que le composant rencontrera en production. Le verrou qui manquait à tout
le banc est désormais là : `test_le_registre_tient_sous_fovea_mobile`, avec une
fovéa QUI BOUGE et une vérification du REGISTRE lui-même — pas seulement
l'absence d'exception.

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
#
# UNE SECONDE REVUE, à la clôture de branche, en a trouvé TROIS DE PLUS
# (`solver`, `terrains`, `rederive_borne`) : la liste était encore FAUSSE après
# la ronde qui venait de la corriger. Deux rondes, deux fois par revue sur
# pièces, jamais par la garde — c'est le motif, pas l'accident. Le test
# `test_la_serrure_mord_depuis_chaque_module_declare` est désormais PARAMÉTRÉ
# sur toute la liste : il ne peut pas dire ce qui MANQUE (rien ne le peut), mais
# il interdit qu'une entrée déclarée soit un nom mort.
#
# CRITÈRE D'ENTRÉE, écrit pour que la prochaine ronde n'ait pas à le redeviner :
# le module AVANCE l'état (un pas de temps, `F`, un pilote qui les enchaîne) ou
# ÉCRIT dans un tenseur d'état vivant. Ne sont PAS des modules d'état les
# constructeurs de conditions initiales (`wetdry_vocab`,
# `solver.initial_condition_*`), ni le chemin de READOUT lui-même
# (`chemin_de_cout`, `summary*`, `multiresolution`, `mass_projection`,
# `arcC_*`), ni les surrogates latents (`dmd.rollout` avance un `z`, pas l'état).
_MODULES_ETAT: frozenset[str] = frozenset({
    "exner_gpu",
    "fovea_2niveaux",                # exécute `pas_deux_niveaux`
    "ledger",                        # comptabilité de conservation
    "production_fidele",             # exécute `pas_f_fidele`
    "pyramide",                      # exécute `F` via `pas_f` (`pyramide.py:501`)
    "rederive_borne",                # PILOTE `run_episode` du chemin re-dérivé
    "sediment",                      # chemin re-dérivé CPU
    "solver",                        # `lax_friedrichs_step`/`simulate` — avanceur d'état CPU
    "solver_wetdry",                 # chemin re-dérivé CPU
    "substrat_fidele",
    "substrat_fusionne",
    "substrat_fusionne_3d",
    "substrat_fusionne_3d_param",
    "substrat_jetable",
    "substrat_jetable_3d",
    "substrat_l3",
    "temoin_fidele",                 # exécute `pas_f_fidele`
    "terrains",                      # `rest_residual` avance l'état (`terrains.py:180`)
    "transferts",                    # descend/remonte l'état
})


class ReadoutInterditDansEtat(RuntimeError):
    """Levée quand le chemin de l'ÉTAT tente d'atteindre un produit de readout.

    Hérite de `RuntimeError` — la faute porte sur QUI appelle. N'est JAMAIS
    rattrapée dans ce module : une serrure qui se rattrape est une promesse,
    pas une serrure."""


class ReadoutHorsRegistre(RuntimeError):
    """Levée quand `s_prev` et `s_cur` ne décrivent plus les MÊMES cellules.

    LE REGISTRE, ET POURQUOI IL SE PERD. Les fenêtres de la pyramide sont
    FOVÉA-RELATIVES : `GeometriePyramide.origines(j, centre_fin)` fait dépendre
    l'origine `ox` du centre fovéal. Quand `PyramideFovea.frame()` déplace la
    fovéa, il translate le CONTENU des fenêtres (`xp.roll(fen, -dx, axis=-1)`,
    `pyramide.py:491`) puis applique `F` (`pyramide.py:501`). `s_prev` est
    indexé en coordonnées LOCALES à la fenêtre et n'est jamais roulé : à
    l'adresse locale `(iy, ix)`, `s_cur` porte alors la cellule monde `x` et
    `s_prev` la cellule monde `x − dx`.

    POURQUOI LEVER PLUTÔT QUE LAISSER PASSER. Un mélange hors registre est
    silencieux de bout en bout : il fantôme d'une cellule fine par pas, et sur
    les colonnes ENTRANTES il synthétise une valeur qui n'a jamais existé — sans
    clamp (les deux opérandes sont positives), sans levée, avec `clamps == 0`.
    Rien dans l'aval ne le signalerait, et un chiffre en sortirait quand même.
    C'est exactement la faute que §A53 décrit : un terme faux qui passe tous les
    verrous numériques. Une image manquante se voit ; un fantôme d'une cellule,
    non.

    CE QUE CETTE EXCEPTION NE FAIT PAS : elle ne remet pas `s_prev` en registre.
    C'est `ApplicateurCaptureRegistre` qui le fait, en capturant post-roll et
    pré-pas de physique, PAR NIVEAU (voie 2, §13 du spec).

    POURQUOI ELLE RESTE ALORS QUE LE MÉCANISME EXISTE — DÉFENSE EN PROFONDEUR, et
    les deux gardes ne gardent pas la même chose :

      - l'APPLICATEUR met en registre le cycle qui passe par lui. Il ne peut rien
        dire d'un cycle qui ne passe PAS par lui : `TamponReadout.capturer` reste
        appelable, et une capture prise avant `frame()` est exactement le bug du
        §12-1, à l'octet près. Le mécanisme corrige un usage ; il n'en interdit
        aucun autre.
      - CETTE EXCEPTION mord sur le CENTRE FOVÉAL, qu'il y ait un applicateur ou
        non : `_centre_capture` est écrit à CHAQUE capture, donc une capture en
        registre (prise DANS `frame()`, après `centre_fin += delta_x`) le laisse
        égal au centre courant et la garde se tait ; une capture à l'ancienne
        (prise AVANT `frame()`) le laisse en retard et la garde lève.

    Retirer l'exception parce que le mécanisme existe rendrait donc SILENCIEUSE
    la seule faute que le mécanisme ne couvre pas — celle de l'appelant qui ne
    l'utilise pas. C'est le motif §A62-bis-2 retourné : une garde qu'on retire
    parce qu'« on ne fera plus la faute » est une garde absente."""


class ReadoutSansCapture(RuntimeError):
    """Levée quand `melanger` est appelé sans qu'aucun `capturer` ait eu lieu.

    Exception SŒUR de `ReadoutHorsRegistre` — même famille, même motif : sans
    capture, `s_prev` vaut les zéros de l'allocation, donc `s_out = α·s_cur`.
    Le résultat n'est PAS un mélange entre deux états : c'est un état atténué,
    faux, positif, et donc `clamps == 0`. Silencieux par construction, comme le
    hors-registre. Deux exceptions distinctes plutôt qu'une seule parce que le
    DIAGNOSTIC diffère — protocole non amorcé d'un côté, protocole rompu par un
    déplacement de fovéa de l'autre — et qu'un message qui confond les deux
    envoie chercher la faute au mauvais endroit."""


class ReadoutCaptureImpossible(RuntimeError):
    """Levée quand la capture EN REGISTRE ne peut pas être posée correctement.

    Deux fautes, toutes deux STRUCTURELLES — elles portent sur le montage, pas
    sur les données, et elles se voient à l'installation ou au premier appel :

      - `pas_f` reçoit un tableau qui n'est AUCUNE des fenêtres de la pyramide
        sur laquelle l'applicateur a été installé. L'applicateur identifie le
        niveau par l'IDENTITÉ du tableau (B2 : les buffers sont préalloués à la
        construction et `frame()` n'écrit QUE dedans — `fen[...] = …`, jamais un
        rebind) ; un tableau inconnu signifie que cette invariante a cessé de
        tenir, et capturer « au petit bonheur » écrirait `s_prev` du mauvais
        niveau, sans symptôme ;
      - un applicateur est déjà installé sur cette pyramide. Deux applicateurs
        empilés écrivent DEUX tampons dont un seul sera lu, et le lecteur n'a
        aucun moyen de savoir lequel il tient.

    Lever plutôt que deviner : un montage faux qui tourne quand même rend un
    chiffre, et c'est la faute §A53 une fois de plus."""


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
    touche — `xp`, `geo`, `centre_fin`, `fenetres` — plus DEUX compteurs de
    compte rendu (`clamps`, `non_finis`) que le gather ne lit pas. Le gather
    n'est pas modifié : il reçoit un objet de même surface dont le champ lu est
    le `s` interpolé.

    `fenetres[j]` a la forme `(n_slots(j), n_systemes(j), 1, n_fov, n_fov)` :
    l'axe des champs est un SINGLETON, donc le gather s'appelle avec
    `indice_champ=0`. Garder un vrai tableau contigu — plutôt qu'un proxy qui
    ré-indexerait en Python — évite de remettre dans le chemin le facteur 109
    que `chemin_de_cout` a dû tuer (boucle Python contre kernel).

    ⚠ `fenetres` ALIASE le tampon interne `_s_out` du `TamponReadout` — ce n'est
    PAS une copie. Deux `melanger` successifs rendent deux `VueInterpolee`
    distinctes qui pointent sur LES MÊMES tableaux : le second appel écrase le
    contenu que le premier avait rendu. Une vue est valide JUSQU'AU `melanger`
    suivant, et pas au-delà. Un appelant qui veut conserver un résultat doit le
    COPIER lui-même (c'est ce que font les tests de déterminisme, avec
    `np.array(..., copy=True)`). Copier ici serait plus sûr, mais chargerait une
    copie de plus dans `I` à chaque image interpolée : le choix appartient au
    propriétaire, il est documenté plutôt que pris en douce."""

    __slots__ = ("xp", "geo", "centre_fin", "fenetres", "clamps", "non_finis")

    def __init__(self, xp, geo, centre_fin, fenetres, clamps: int,
                 non_finis: int = 0):
        self.xp = xp
        self.geo = geo
        self.centre_fin = centre_fin
        self.fenetres = fenetres
        self.clamps = int(clamps)
        self.non_finis = int(non_finis)


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
        # Centre fovéal AU MOMENT DE LA CAPTURE — `None` tant qu'aucune capture
        # n'a eu lieu. Ce champ unique porte les DEUX gardes de registre :
        # `None` ⇒ `ReadoutSansCapture`, valeur différente ⇒ `ReadoutHorsRegistre`.
        self._centre_capture: int | None = None
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
        précédent. Ce coût est DANS `I` (§3-bis du spec).

        MÉMORISE AUSSI le centre fovéal : les fenêtres sont fovéa-relatives,
        donc `s_prev` n'a de sens QU'À CE CENTRE. `melanger` le revérifie.

        ⚠ C'EST LA CAPTURE À L'ANCIENNE — FOVÉA IMMOBILE SEULEMENT. Prise depuis
        le chemin d'appelant, elle est nécessairement PRÉ-`frame()`, donc
        pré-roll : dès que la fovéa bouge, `s_prev` reste dans les coordonnées
        de l'ancienne fenêtre et `ReadoutHorsRegistre` lève (et c'est ce qu'il
        faut : cf. sa docstring). Le cycle À FOVÉA MOBILE passe par
        `installer_capture_en_registre` / `ApplicateurCaptureRegistre`, qui
        capture PAR NIVEAU à l'intérieur de `frame()`, post-roll et pré-pas de
        physique. Cette méthode-ci reste : `frame(0)` et les bancs à fovéa
        immobile en vivent, et elle est le point de capture le plus simple qui
        soit correct dans ce régime."""
        for j in pyramide.geo.niveaux_gpu:
            source = pyramide.fenetres[j][:, :, INDICE_CHAMP_S, :, :]
            self._s_prev[j][:, :, 0, :, :] = source
        self._centre_capture = int(pyramide.centre_fin)

    def capturer_niveau(self, niveau: int, fenetres_du_niveau,
                        centre_fin: int) -> None:
        """`s_prev[niveau]` ← COPIE du `s` courant d'UN SEUL niveau.

        LE POINT D'ENTRÉE DE LA VOIE 2, et il n'est appelé QUE par
        `ApplicateurCaptureRegistre` — c'est-à-dire depuis l'intérieur de
        `frame()`, POST-ROLL et PRÉ-PAS DE PHYSIQUE. La justification du point
        de capture est au §13-2 du spec et dans l'en-tête de ce module ; ce qui
        se joue ICI est la GRANULARITÉ : la capture est PAR NIVEAU parce que
        `frame()` roule et fait avancer un niveau à la fois, dans la même
        itération. Un geste global — avant ou après la boucle — serait
        forcément pré-roll pour certains niveaux et post-`F` pour d'autres.

        `centre_fin` EST PASSÉ, PAS RELU SUR LA PYRAMIDE, parce que le seul
        appelant légitime le connaît déjà et que le rendre explicite dit ce qui
        est mémorisé : le centre AU MOMENT DE LA CAPTURE. `frame()` incrémente
        `centre_fin` AVANT sa boucle par niveau, donc une capture en registre
        mémorise le centre D'APRÈS le déplacement — et c'est exactement ce qui
        rend `ReadoutHorsRegistre` muette pour ce cycle et bavarde pour une
        capture à l'ancienne.

        AUCUN VERROU DE PILE ICI, ET C'EST VOULU (§13-4). Cette méthode s'exécute
        avec `pyramide` dans la pile par construction : `_verrouiller_consommateur`
        y lèverait à chaque frame. Elle n'a rien à y faire, car le sens d'écriture
        est état → readout, LE SENS SÛR — on LIT la fenêtre, on ÉCRIT le tampon de
        readout. Le verrou reste sur `melanger`, qui est l'autre sens : la
        PRODUCTION d'un `s_out` que le chemin de l'état ne doit jamais atteindre.
        """
        cible = self._s_prev.get(niveau)
        if cible is None:
            raise ReadoutCaptureImpossible(
                f"capture en registre demandée pour le niveau {niveau}, absent "
                f"du tampon (niveaux alloués : {sorted(self._s_prev)}). Le "
                "tampon se dimensionne à la construction sur "
                "`pyramide.geo.niveaux_gpu` ; un niveau qui n'y est pas signale "
                "que le tampon et la pyramide ne décrivent pas la même "
                "géométrie.")
        cible[:, :, 0, :, :] = fenetres_du_niveau[:, :, INDICE_CHAMP_S, :, :]
        self._centre_capture = int(centre_fin)

    def etat_precedent(self, niveau: int):
        """`s_prev` du niveau, en `(n_slots, n_systemes, n_fov, n_fov)`.

        LECTURE — pour vérifier le registre (c'est ce que fait le verrou de
        fovéa mobile) ou pour un diagnostic. Ce n'est PAS une référence vers
        l'état de `F` : `s_prev` vit dans le tampon de readout, hors du tenseur
        que `F` consomme, et le §13-4 tient parce que rien de ce tampon ne
        remonte jamais dans l'état."""
        return self._s_prev[niveau][:, :, 0, :, :]

    def melanger(self, pyramide, alpha: float) -> VueInterpolee:
        """`s_out = (1−α)·s_prev + α·s_cur`, puis clamp `s ≥ 0` COMPTÉ.

        Le clamp ne lève JAMAIS : ceci est du code moteur, il tourne. Le seuil
        au-delà duquel un comptage invalide une mesure appartient au prereg de
        la lecture d'orientation, pas ici (§5 du spec).

        DEUX GARDES DE REGISTRE, AVANT toute arithmétique — et elles LÈVENT,
        elles, parce qu'elles ne portent pas sur une VALEUR hors domaine (§5)
        mais sur un PROTOCOLE rompu, dont le produit ne décrirait aucun état :

          - `ReadoutSansCapture` — aucun `capturer` n'a eu lieu : `s_prev` est
            l'allocation à zéro, et `s_out = α·s_cur` serait un état atténué,
            faux et positif ;
          - `ReadoutHorsRegistre` — le centre fovéal a bougé depuis la capture :
            `s_prev` et `s_cur` ne décrivent plus les mêmes cellules du monde.
            Sous une capture EN REGISTRE cette garde ne mord jamais (le centre
            est mémorisé DANS `frame()`, après le déplacement) ; elle reste pour
            l'appelant qui capture à l'ancienne — défense en profondeur.

        LE CLAMP NE PROMET `s ≥ 0` QUE POUR LES VALEURS FINIES, et les non
        finies sont COMPTÉES À PART (`VueInterpolee.non_finis`). Le fait
        d'abord : `sortie < 0.0` est FAUX pour un NaN, donc un NaN traversait le
        clamp sans être vu ni compté, alors que la docstring promettait
        « clamp `s ≥ 0` » — la promesse est ici restreinte à ce qu'elle tient.
        Le traitement, tranché et écrit :

          - COMPTER à part, dans un compteur DISTINCT. Les verser dans `clamps`
            polluerait un compteur dont un prereg futur tirera un seuil : une
            extrapolation sous zéro et un état corrompu sont deux phénomènes
            sans rapport, et un seuil calé sur leur somme ne mesurerait ni l'un
            ni l'autre.
          - NE PAS clamper le NaN à 0. Un NaN ou un ±inf dans `s_out` ne peut
            venir que de `s_prev` ou de `s_cur`, donc d'un état déjà corrompu
            EN AMONT par `F` : l'écraser à zéro effacerait la preuve d'une
            faute de physique en la déguisant en cellule sèche — la signature
            §A53 une fois de plus.
          - NE PAS lever. §5 du spec a tranché pour le readout : code moteur,
            il tourne, il compte et il reporte. Une garde qui lève ici tuerait
            le rendu sur une faute qui n'est pas la sienne.

        Le masque est calculé AVANT le clamp, sinon `−inf < 0.0` étant VRAI,
        l'infini serait ramené à zéro avant d'avoir pu être vu. Un `−inf` est
        donc compté DEUX FOIS, dans `non_finis` ET dans `clamps` — c'est exact :
        il est réellement clampé, et `non_finis` ne retire rien à `clamps`, il
        ajoute ce que `clamps` est incapable de dire. L'arithmétique des valeurs
        finies est INCHANGÉE : l'affinité du §2 reste vérifiable octet pour
        octet, et ce comptage supplémentaire est un balayage de plus par niveau,
        DANS `I` comme le reste du noyau (§3-bis).

        ⚠ LA VUE RENDUE ALIASE `_s_out` — ce n'est pas une copie. L'appel
        SUIVANT à `melanger` réécrit ces mêmes tableaux, donc invalide la vue
        précédente. Voir `VueInterpolee`."""
        _verrouiller_consommateur()
        if self._centre_capture is None:
            raise ReadoutSansCapture(
                "`melanger` appelé sans `capturer` préalable : `s_prev` est "
                "encore l'allocation à ZÉRO, donc `s_out = α·s_cur` — un état "
                "atténué, faux, positif, et compté `clamps == 0`. Le protocole "
                "est `capturer(pyramide)` AVANT le pas de `F`, puis `melanger` "
                "après lui ; sans le premier terme, le noyau affine ne mélange "
                "rien.")
        centre_courant = int(pyramide.centre_fin)
        if centre_courant != self._centre_capture:
            decalage = centre_courant - self._centre_capture
            raise ReadoutHorsRegistre(
                f"`s_prev` n'est plus en registre avec `s_cur` : le centre "
                f"fovéal est passé de {self._centre_capture} à "
                f"{centre_courant} (décalage de {decalage} cellule(s) fine(s)) "
                "depuis la capture. Les fenêtres sont FOVÉA-RELATIVES "
                "(`origines(j, centre_fin)`), et `PyramideFovea.frame()` "
                "translate le contenu (`xp.roll(fen, -dx, axis=-1)`, "
                "`pyramide.py:491`) alors que `s_prev`, indexé en coordonnées "
                "locales à la fenêtre, n'est PAS roulé. Mélanger ici "
                "combinerait, à la même adresse locale, deux cellules du monde "
                "DIFFÉRENTES — un fantôme d'une cellule fine par pas, et sur "
                "les colonnes entrantes une valeur qui n'a JAMAIS existé, sans "
                "clamp ni symptôme. Cette garde ne RÉPARE rien, et elle n'a "
                "plus à le faire : le mécanisme de capture EN REGISTRE existe "
                "— `installer_capture_en_registre(pyramide)` enveloppe `pas_f` "
                "et capture post-roll / pré-pas de physique, PAR NIVEAU (voie "
                "2, §13 du spec). Cette levée signale une capture À L'ANCIENNE, "
                "prise AVANT `frame()`.")
        xp = pyramide.xp
        a = float(alpha)
        clamps = 0
        non_finis = 0
        for j in pyramide.geo.niveaux_gpu:
            s_cur = pyramide.fenetres[j][:, :, INDICE_CHAMP_S, :, :]
            sortie = self._s_out[j][:, :, 0, :, :]
            sortie[...] = (1.0 - a) * self._s_prev[j][:, :, 0, :, :] + a * s_cur
            non_finis += int(xp.count_nonzero(~xp.isfinite(sortie)))
            sous_zero = sortie < 0.0
            clamps += int(xp.count_nonzero(sous_zero))
            sortie[sous_zero] = 0.0
        return VueInterpolee(xp, pyramide.geo, pyramide.centre_fin,
                             self._s_out, clamps, non_finis)


class ApplicateurCaptureRegistre:
    """L'APPLICATEUR DE CAPTURE EN REGISTRE — la voie 2 du §13 du spec, montée.

    IL ENVELOPPE `pas_f`. `PyramideFovea` expose son applicateur de `F` comme un
    ATTRIBUT INJECTABLE (`pyramide.py:392-395`, signature
    `pas_f(fenetres, xp, sortie) -> (etat, dt_cfl)`), et `frame()` l'appelle une
    fois par niveau (`pyramide.py:501`). Envelopper cet attribut donne donc, sans
    modifier UNE SEULE LIGNE de `pyramide.py`, un point d'exécution situé :

        APRÈS le roll et la descente des colonnes entrantes (`pyramide.py:491`),
        AVANT le pas de physique (`pyramide.py:501`),
        DANS la boucle par niveau, et une fois par niveau.

    POURQUOI LÀ ET PAS AILLEURS — §13-2 du spec. Post-roll, `s_prev` (état `n−1`)
    et `s_cur` (état `n`, après `pas_f`) vivent dans LES MÊMES COORDONNÉES
    FENÊTRE PAR CONSTRUCTION, **y compris les colonnes entrantes, qui reçoivent
    du roll leur contenu `n−1`**. Pas de masque de validité, pas de cellule qui
    n'a jamais coexisté, et le NOYAU AFFINE du §2 reste intact — c'est ce que la
    voie 2 achète, et c'est la raison pour laquelle les voies « rouler `s_prev` »
    (un `α` par cellule) et « coordonnées monde » (lecture décalée) sont mortes :
    elles changeraient le noyau qu'un document endossé grave.

    LES TROIS AUTRES POINTS DE CAPTURE CONCEVABLES, ET CE QU'ILS CASSENT :

      - AVANT `frame()` (`TamponReadout.capturer`) : PRÉ-ROLL — c'est le bug du
        §12-1, `s_prev` en coordonnées de l'ancienne fenêtre ;
      - APRÈS `pas_f`, dans la même enveloppe : `s_prev` porterait l'état `n`,
        pas `n−1` — le mélange serait l'identité, silencieusement ;
      - APRÈS `frame()`, en un geste global : la boucle a déjà roulé ET fait
        avancer TOUS les niveaux ; il n'existe plus, hors de la boucle, d'instant
        où l'état `n−1` post-roll est encore lisible.

    LA COPIE EST DUE, ET LE §13-3 LE TRANCHE : capturer pré-`pas_f` pendant que
    `F` écrit EN PLACE (`self.pas_f(fen, xp, fen)`) impose la copie d2d — le
    régime « échange de pointeurs » meurt, et il meurt désormais par la GÉOMÉTRIE
    du point de capture, plus seulement par l'écriture en place. `I` = noyau +
    copie, 30/s, et la copie se chiffre PAR NIVEAU, sur ce site-ci.

    IDENTIFICATION DU NIVEAU — par IDENTITÉ du tableau. `frame()` passe `fen`
    sans dire quel niveau c'est ; la correspondance est faite sur `id()` des
    tableaux de `pyramide.fenetres`, ce qui est licite parce que B2 les préalloue
    à la construction et que `frame()` n'écrit QUE dedans (`fen[...] = …`,
    `fen[..., k:] = …`) sans jamais les rebinder. L'applicateur garde une
    référence FORTE sur chacun — un `id()` ne peut donc pas être recyclé sous lui
    — et re-vérifie l'identité (`is`) avant de capturer. Un tableau inconnu ne
    donne PAS lieu à une devinette : `ReadoutCaptureImpossible`.

    UNE FRAME INTERROMPUE LAISSE UN TAMPON HÉTÉROGÈNE, et ce n'est pas gardé
    ici : si `frame()` lève au milieu de sa boucle (par exemple sur le recul
    d'origine, `pyramide.py:479`), les niveaux déjà traversés ont capturé et les
    autres non. Le tampon ne peut pas être plus cohérent que l'état qu'il
    reflète, lequel est dans le même désordre — la pyramide a fait avancer
    certains niveaux et pas d'autres. Rien à mécaniser ici : l'appelant qui
    rattrape une levée de `frame()` tient une pyramide à jeter, pas un readout à
    sauver. Écrit pour que la prochaine revue ne le redécouvre pas.

    AUCUNE MESURE ICI, et pas de chronomètre : §A61 tient (§11 du spec)."""

    __slots__ = ("tampon", "_pyramide", "_pas_f", "_niveau_par_id")

    def __init__(self, pyramide, pas_f_sous_jacent, tampon=None):
        self._pyramide = pyramide
        self._pas_f = pas_f_sous_jacent
        self.tampon = TamponReadout(pyramide) if tampon is None else tampon
        self._niveau_par_id = {}
        for j in pyramide.geo.niveaux_gpu:
            fenetre = pyramide.fenetres[j]
            self._niveau_par_id[id(fenetre)] = (int(j), fenetre)

    def __call__(self, fenetres, xp, sortie=None):
        """Capture le niveau PUIS délègue au `pas_f` enveloppé.

        L'ORDRE DES DEUX LIGNES EST TOUT LE MÉCANISME : la capture est ce qui se
        passe entre le roll (déjà fait par `frame()`) et le pas de physique (fait
        juste après, ici). Les inverser rendrait `s_prev` égal à `s_cur`."""
        entree = self._niveau_par_id.get(id(fenetres))
        if entree is None or entree[1] is not fenetres:
            connus = sorted(j for j, _ in self._niveau_par_id.values())
            raise ReadoutCaptureImpossible(
                "`pas_f` a reçu un tableau qui n'est aucune des fenêtres de la "
                "pyramide sur laquelle la capture en registre a été installée "
                f"(niveaux connus : {connus}). "
                "Le niveau est identifié par l'IDENTITÉ du tableau, ce que B2 "
                "autorise : les buffers sont préalloués à la construction et "
                "`frame()` n'écrit que DEDANS. Un tableau inconnu signifie que "
                "cette invariante a cessé de tenir ; capturer quand même "
                "écrirait `s_prev` du mauvais niveau, sans aucun symptôme.")
        niveau, _ = entree
        # POST-ROLL, PRÉ-PAS DE PHYSIQUE — §13-2 du spec. Le centre fovéal passé
        # est celui d'APRÈS le déplacement : `frame()` incrémente `centre_fin`
        # avant d'entrer dans sa boucle par niveau.
        self.tampon.capturer_niveau(niveau, fenetres, self._pyramide.centre_fin)
        return self._pas_f(fenetres, xp, sortie)


def installer_capture_en_registre(pyramide,
                                  tampon=None) -> ApplicateurCaptureRegistre:
    """Enveloppe l'applicateur de `F` de `pyramide` par la capture en registre.

    C'EST LE SEUL GESTE DE MONTAGE, et il est fait DEPUIS L'EXTÉRIEUR de
    `pyramide.py`, sur l'attribut `pas_f` que ce dernier expose déjà comme point
    d'injection (`pyramide.py:392-395`). Le `pas_f` en place — celui du
    constructeur, jetable par défaut, fusionné pour M-a-ter — est conservé et
    appelé tel quel : la physique n'est ni remplacée ni contournée, elle est
    PRÉCÉDÉE.

    Rend l'applicateur, dont `.tampon` est le `TamponReadout` que `melanger`
    consommera après le `frame()`. À appeler UNE FOIS, après la construction de
    la pyramide et avant le premier `frame()` : un `frame()` intercalé
    laisserait `s_prev` non capturé pour ce pas, et `melanger` lèverait
    `ReadoutSansCapture` — bruyamment, ce qui est le comportement voulu."""
    if isinstance(pyramide.pas_f, ApplicateurCaptureRegistre):
        raise ReadoutCaptureImpossible(
            "une capture en registre est DÉJÀ installée sur cette pyramide. "
            "Deux applicateurs empilés captureraient dans deux tampons "
            "distincts dont un seul serait lu, et rien ne dirait lequel : "
            "réutiliser l'applicateur rendu par la première installation.")
    applicateur = ApplicateurCaptureRegistre(pyramide, pyramide.pas_f, tampon)
    pyramide.pas_f = applicateur
    return applicateur
