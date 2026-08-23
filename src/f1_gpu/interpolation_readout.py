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

⚠ SOUS FOVÉA MOBILE, CE MODULE REFUSE DE PRODUIRE. C'est la vérité de son état
ACTUEL, et ce n'est pas une régression : c'est la mise au jour d'un défaut qui
existait déjà, muet. Les fenêtres sont FOVÉA-RELATIVES — `origines(j,
centre_fin)` fait dépendre l'origine `ox` du centre — et `PyramideFovea.frame()`
translate le contenu (`fen[...] = xp.roll(fen, -dx, axis=-1)`,
`pyramide.py:491`) PUIS applique `F` (`pyramide.py:501`), dans la MÊME boucle.
`s_prev`, lui, est indexé en coordonnées LOCALES à la fenêtre et n'est JAMAIS
roulé. Après un `frame()` qui déplace la fovéa, l'adresse locale `(iy, ix)`
porte donc la cellule monde `x` dans `s_cur` et la cellule monde `x − dx` dans
`s_prev` : le mélange fantômerait d'une cellule fine par pas, et sur les
colonnes ENTRANTES il produirait une valeur QUI N'A JAMAIS EXISTÉ — sans clamp,
sans levée, `clamps == 0`. La signature exacte de §A53.

CE QUE LA GARDE FAIT, ET TOUT CE QU'ELLE FAIT : transformer une corruption
SILENCIEUSE en échec BRUYANT (`ReadoutHorsRegistre`). Elle ne RÉPARE rien. Le
point de capture correct est post-roll et pré-`pas_f`, PAR NIVEAU, à l'intérieur
de `frame()` — c'est une décision de DESIGN, qui appartient au propriétaire du
projet et non à cette garde. Tant qu'elle n'est pas prise, le composant ne sert
qu'à fovéa IMMOBILE (`frame(0)`, ou aucun `frame` entre `capturer` et
`melanger`).

DOUZE VERROUS VERTS N'ONT PAS VU CE DÉFAUT parce qu'AUCUN n'appelait `frame()` :
la fovéa était immobile dans tous les tests. Un verrou ne garde que ce que son
état de test allume (§A53) — et l'état de test le plus coûteux à oublier est
celui que le composant rencontrera en production.

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
    Le point de capture correct est post-roll et pré-`pas_f`, PAR NIVEAU, à
    l'intérieur de `frame()` ; l'y placer est une décision de DESIGN, réservée au
    propriétaire du projet."""


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
        donc `s_prev` n'a de sens QU'À CE CENTRE. `melanger` le revérifie."""
        for j in pyramide.geo.niveaux_gpu:
            source = pyramide.fenetres[j][:, :, INDICE_CHAMP_S, :, :]
            self._s_prev[j][:, :, 0, :, :] = source
        self._centre_capture = int(pyramide.centre_fin)

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
                "clamp ni symptôme. Cette garde ne RÉPARE rien : le point de "
                "capture correct est post-roll et pré-`pas_f`, PAR NIVEAU, à "
                "l'intérieur de `frame()` — décision de DESIGN, réservée au "
                "propriétaire du projet.")
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
