"""BOUCLE DE RENDU — l'orchestration 2:1 de la porte 33,3.

Spec : `claude/spec-boucle-rendu-2026-08-24.md`, endossée. Son §8 porte le
STATUT COURANT et amende l'épigraphe de tête ainsi que le §7 : l'épinglage des
deux côtés du §2 EST endossé, la maison amende en bas plutôt que de réécrire en
haut.

CE MODULE EST UN CONSOMMATEUR. Il branche `interpolation_readout` et
`chemin_de_cout` dans une boucle qui produit DEUX écrans par pas de physique. Il
ne modifie ni l'un, ni l'autre, ni `pyramide.py` — et il n'en importe rien
d'autre que leur surface publique.

LA CADENCE EST LOGIQUE (§1 de la spec). « 2:1 » est un RAPPORT DE COMPTE : deux
écrans par `frame()`, PAS deux écrans par 33,3 ms. Aucune horloge murale
n'entre ici — ni `time`, ni `cuda.Event`, ni attente, ni régulation. §A61 en
interdit l'usage tant que l'instrument n'est pas qualifié, et une cadence
logique n'en a pas besoin.

LA CONVERSION `s` → IMAGE VIT DEHORS (§1). Le tick rend des champs de SÉDIMENT ;
la lecture albédo qui en fait une image est `chemin_de_cout.albedo_ecran`, dont
le `s_half` est un paramètre d'APPELANT sans aucune valeur endossée dans le
corpus. Ce module ne convertit rien et n'importe pas `albedo_ecran` : le graver
ici serait choisir un paramètre hors de toute décision.

CE QUI N'EST PAS TRANCHÉ ICI, ET NE DOIT PAS L'ÊTRE (§6) : LE CÔTÉ. Il
appartient à la lecture d'orientation, qui n'est pas ordonnée et dont le prereg
n'est pas écrit. La boucle est PARAMÉTRABLE parce que le côté n'est pas
choisi — c'est une garde, pas une souplesse — et `cote` n'a donc AUCUNE valeur
par défaut : l'absence de défaut EST le refus de trancher §A62.

AUCUNE MESURE ICI, aucun chiffre de `I` : §A61 tient. La phrase a dit « aucun
chronomètre » et c'était plus large que ce qui est tenu : `tick` appelle
`pyramide.frame()`, dont chaque transfert passe par `_chronometrer` et
`time.perf_counter` (`src/f1_gpu/transferts.py:66-68`). Un chronomètre tourne
donc SOUS ce module, atteint par transitivité. Ce qui est tenu, et mécanisé,
est plus étroit : ce fichier ne LIT lui-même aucune horloge — ni `time`, ni
`cuda.Event`, ni attente, ni régulation, contrôlé sur la SOURCE par
`tests/test_boucle_rendu.py::test_la_source_du_module_ne_porte_ni_filet_large_ni_assert_ni_horloge`
— et aucun chiffre de durée ne remonte d'ici : ni `tick`, ni `CoupleEcrans`, ni
`CompteRenduTick` n'en portent un. C'est ce second point qui garde §A61.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

from src.f1_gpu.chemin_de_cout import gather_chemin_de_cout
from src.f1_gpu.interpolation_readout import (
    ALPHA_EXTRAPOLER,
    ALPHA_INTERPOLER,
    INDICE_CHAMP_S,
    ApplicateurCaptureRegistre,
    installer_capture_en_registre,
)

if TYPE_CHECKING:                       # pas d'import au runtime : ce module
    from src.f1_gpu.pyramide import PyramideFovea   # ne dépend pas de l'état

# UN ÉCRAN EST UN TABLEAU 2D DU BACKEND INJECTÉ (`pyramide.xp`) : numpy dans les
# bancs, cupy en production. Le nommer `np.ndarray` serait FAUX sous cupy, et ce
# serait la seule ligne du module à contredire le principe du backend injecté —
# ce module ne connaît aucun backend, il transmet celui de la pyramide.
Ecran = Any

# Les deux côtés, nommés. Le côté ne choisit QUE la paire d'`α` (§2-2).
COTE_INTERPOLER: str = "interpoler"
COTE_EXTRAPOLER: str = "extrapoler"

# L'`α` de l'image EXACTE. Ce n'est PAS une recopie d'une constante du module de
# readout : `interpolation_readout` n'en porte aucune pour cette valeur, et pour
# une raison de fond — cet `α` n'est JAMAIS passé à `melanger`. Le spec readout
# a tranché que les images exactes CONTOURNENT le noyau (§2-1 de la spec), donc
# `α = 1,0` ne nomme pas ici un mélange à faire, mais un mélange à NE PAS faire.
# Les deux `α` qui atteignent réellement le noyau, eux, sont IMPORTÉS.
ALPHA_EXACT: float = 1.0

# `VueInterpolee.fenetres` a la forme `(n_slots, n_systemes, 1, n_fov, n_fov)` :
# l'axe des champs est un SINGLETON, donc le gather s'appelle avec l'indice 0.
INDICE_CHAMP_VUE: int = 0

# Le balayage de la fovéa, une cellule fine par tick (E4c, `frame(1)`). La fovéa
# BOUGE : c'est ce qui fait de la capture EN REGISTRE la condition d'existence
# de cette boucle, et non un raffinement (§3 de la spec).
PAS_FOVEA: int = 1

# LA TABLE DES COUPLES — la seule source du couple, et le seul endroit où le
# côté a un effet. Dans les deux côtés le couple sort en `α` CROISSANT et
# EXACTEMENT UN de ses deux termes vaut `α = 1,0` (§2-2, endossé par le §8
# point 1). C'est cette uniformité qui rend le tick UNIQUE : un chemin de code,
# aucun branchement par côté hors de cette table.
COUPLES_PAR_COTE: dict[str, tuple[float, float]] = {
    COTE_INTERPOLER: (ALPHA_INTERPOLER, ALPHA_EXACT),   # `n − ½` puis `n`
    COTE_EXTRAPOLER: (ALPHA_EXACT, ALPHA_EXTRAPOLER),   # `n` puis `n + ½`
}


class CoteInconnu(ValueError):
    """Levée quand le côté demandé n'est aucun de ceux de `COUPLES_PAR_COTE`.

    Hérite de `ValueError` — la faute porte sur un ARGUMENT, comme
    `PyramideFovea.frame` pour un `delta_x` négatif et `gather_chemin_de_cout`
    pour un `cote_px` nul. Délibérément HORS de la famille `RuntimeError` : le
    §4-8 de la spec rappelle que tout ce qui hérite de `RuntimeError` sur ce
    chemin risque d'être avalé par un `except RuntimeError` d'appelant."""


class MontageBoucleInvalide(RuntimeError):
    """Levée quand le montage de la boucle ne peut pas être posé correctement.

    Fautes STRUCTURELLES — elles portent sur le montage, pas sur les données, et
    se voient à la construction : un applicateur qui n'en est pas un, un
    applicateur installé sur une AUTRE pyramide, une table de couples qui a
    cessé de vérifier l'uniformité du §2-2.

    Lever plutôt que deviner : un montage faux qui tourne quand même rend un
    chiffre, et c'est la faute §A53 une fois de plus. Un applicateur étranger,
    en particulier, mélangerait le `s_prev` d'un monde avec le `s_cur` d'un
    autre — deux états qui n'ont jamais coexisté, sans clamp, sans levée, avec
    `clamps == 0`."""


class CoupleEcrans(NamedTuple):
    """Les deux écrans d'un tick, EN `α` CROISSANT (§2-2).

    Un `NamedTuple` à DEUX champs, et pas davantage : l'appelant doit pouvoir
    déballer les deux écrans sans cérémonie (`petit, grand = boucle.tick(...)`).
    Les compteurs de compte rendu vivent à côté, dans `BoucleRendu.compte_rendu`.

    Les champs ne s'appellent ni « exact » ni « interpolé » PARCE QUE LE CÔTÉ
    N'EST PAS TRANCHÉ (§6) : selon le côté, l'exact est le second (interpoler)
    ou le premier (extrapoler). Nommer par l'`α` dit ce qui est vrai des deux
    côtés ; nommer par le rôle obligerait l'appelant à savoir quel côté tourne
    pour lire un nom de champ."""

    ecran_alpha_petit: Ecran
    ecran_alpha_grand: Ecran


class CompteRenduTick(NamedTuple):
    """Ce que la `VueInterpolee` du tick a COMPTÉ, remonté tel quel.

    `clamps` : cellules finies ramenées à zéro par le clamp `s ≥ 0`.
    `non_finis` : cellules NaN ou ±inf, comptées à part — un `−inf` est compté
    dans les DEUX, et c'est exact (il est réellement clampé).

    AUCUN SEUIL, AUCUN AVERTISSEMENT, AUCUNE DÉCISION n'est attaché à ces
    nombres ici. Le seuil au-delà duquel un comptage invalide une mesure
    appartient au prereg de la lecture d'orientation, pas à ce code (§3 de la
    spec, §5 du spec readout). La boucle les REND ; elle ne prononce pas."""

    clamps: int
    non_finis: int


class BoucleRendu:
    """L'orchestrateur 2:1 — un pas de physique, deux écrans.

    MONTAGE. Construire la pyramide, puis construire la boucle : elle installe
    la capture EN REGISTRE (`installer_capture_en_registre`) une fois, ou reçoit
    un applicateur DÉJÀ installé et n'installe alors rien. La détection est
    faite sur le paramètre reçu, JAMAIS en rattrapant `ReadoutCaptureImpossible`
    pour deviner : une serrure qui se rattrape est une promesse (§4-8).

    POURQUOI LA CAPTURE EN REGISTRE EST OBLIGATOIRE ICI. Les bancs du readout
    tournaient à fovéa IMMOBILE — douze verrous verts n'ont pas vu le bug de
    registre parce qu'aucun n'appelait `frame()`. Cette boucle appelle
    `frame(1)` : la fovéa BOUGE à chaque tick. Sans la capture en registre,
    `s_prev` serait nécessairement pré-roll et `melanger` lèverait
    `ReadoutHorsRegistre` à chaque tick.

    LA BOUCLE N'APPELLE JAMAIS `TamponReadout.capturer` (§3). Appelée APRÈS
    `frame()`, cette méthode ne lève pas — `_centre_capture` prend le centre
    courant, donc `ReadoutHorsRegistre` se tait — mais rend `s_prev == s_cur` :
    tout écran interpolé deviendrait identique à l'exact, `clamps == 0`, et rien
    ne le signalerait. Le seul point de capture est l'applicateur.

    ET L'ORDRE INVERSE — installer APRÈS un `frame()` — n'est pas gardé ici, et
    ce n'est pas un oubli : le tick fait TOUJOURS `frame()` PUIS `melanger`,
    donc la capture du tick courant a lieu avant le mélange du tick courant,
    quel que soit ce qui a précédé la construction. Un garde-fou sur
    `centre_fin` refuserait au passage un applicateur légitimement déjà en
    service — la faute qu'il prétendrait attraper n'existe pas dans ce chemin.

    CE QUE LA BOUCLE NE FAIT PAS : elle ne copie AUCUN écran (§4-4), ne conserve
    AUCUNE vue au-delà de son tick, n'écrit JAMAIS dans le tampon de readout
    (c'est `frame()` qui le fait, par l'applicateur — la boucle LIT), n'écrit
    JAMAIS `pyramide.centre_fin` hors de `frame()` (§4-6), ne rattrape AUCUNE
    levée (§4-7, §4-8) et n'emploie AUCUN `assert` pour une garde (§4-9, §A43 :
    un `assert` disparaît sous `python -O`, donc une garde qui en dépend est
    absente exactement dans le régime où l'on mesurera).
    """

    def __init__(self, pyramide: "PyramideFovea", cote: str,
                 applicateur: ApplicateurCaptureRegistre | None = None):
        """`cote` est SANS VALEUR PAR DÉFAUT — voir l'en-tête du module.

        `applicateur` : `None` pour que la boucle installe la capture en
        registre elle-même, ou l'`ApplicateurCaptureRegistre` déjà posé sur
        CETTE pyramide."""
        couple = COUPLES_PAR_COTE.get(cote)
        if couple is None:
            raise CoteInconnu(
                f"côté {cote!r} inconnu — les seuls côtés sont "
                f"{sorted(COUPLES_PAR_COTE)}. Le côté n'a AUCUNE valeur par "
                "défaut et n'en aura pas : il appartient à la lecture "
                "d'orientation, qui n'est pas ordonnée et dont le prereg n'est "
                "pas écrit (§6 de `claude/spec-boucle-rendu-2026-08-24.md`). "
                "Un défaut trancherait §A62 dans du code moteur.")

        # L'UNIFORMITÉ DU §2-2, mécanisée sur la table plutôt que promise par
        # elle. Ces deux propriétés sont ce qui rend le tick unique ; les perdre
        # en éditant la table casserait la comptabilité `30·I` (deux `melanger`
        # par tick) ou l'ordre de sortie du couple, dans les deux cas SANS
        # symptôme au niveau des pixels.
        if not couple[0] < couple[1]:
            raise MontageBoucleInvalide(
                f"le couple du côté {cote!r} vaut {couple} : il ne sort pas en "
                "`α` CROISSANT. Le §2-2 de la spec le grave pour les deux "
                "côtés — c'est ce qui rend l'ordre des deux écrans lisible "
                "sans savoir quel côté tourne.")
        if list(couple).count(ALPHA_EXACT) != 1:
            raise MontageBoucleInvalide(
                f"le couple du côté {cote!r} vaut {couple} : il doit contenir "
                f"EXACTEMENT un `α = {ALPHA_EXACT}`. Zéro ferait passer les "
                "DEUX écrans par le noyau — la comptabilité gravée deviendrait "
                "`60·I` au lieu de `30·I` et contredirait un document endossé "
                "(§2-1) ; deux rendraient le tick redondant.")

        if applicateur is None:
            # Une capture déjà installée fait lever `ReadoutCaptureImpossible`
            # ici, et la levée TRAVERSE : deviner par `except` serait la serrure
            # avalée du §4-8. L'appelant qui tient déjà un applicateur le passe.
            applicateur = installer_capture_en_registre(pyramide)
        else:
            if not isinstance(applicateur, ApplicateurCaptureRegistre):
                raise MontageBoucleInvalide(
                    "`applicateur` n'est pas un `ApplicateurCaptureRegistre` "
                    f"mais un {type(applicateur).__name__}. La confusion est "
                    "facile — `pyramide.pas_f` porte le même nom d'attribut "
                    "avant et après l'installation — et elle donnerait une "
                    "boucle sans aucun tampon de readout, qui échouerait plus "
                    "tard et ailleurs.")
            if pyramide.pas_f is not applicateur:
                raise MontageBoucleInvalide(
                    "l'applicateur reçu n'est pas celui installé sur cette "
                    "pyramide (`pyramide.pas_f` est un autre objet). Il "
                    "capturerait `s_prev` d'un monde pendant que `melanger` "
                    "lirait le `s_cur` d'un autre : deux états qui n'ont jamais "
                    "coexisté, sans clamp, sans levée, avec `clamps == 0`.")

        self.pyramide = pyramide
        self.cote = cote
        self.couple_alphas: tuple[float, float] = couple
        self.applicateur = applicateur
        # `None` tant qu'aucun tick n'a eu lieu : un `CompteRenduTick(0, 0)`
        # ressemblerait à un tick propre qui n'a jamais tourné.
        self.compte_rendu: CompteRenduTick | None = None

    def tick(self, cote_px: int) -> CoupleEcrans:
        """Un pas de physique, DEUX écrans — dans l'ordre, et le même des deux
        côtés :

          1. `pyramide.frame(1)` — l'applicateur capture `s_prev` par niveau,
             post-roll et pré-pas de physique ; l'état passe de `n−1` à `n` ;
          2. l'écran de l'`α` le PLUS PETIT du couple ;
          3. l'écran de l'`α` le plus grand.

        Selon le côté, l'un des deux est l'EXACT — gather DIRECT sur la
        pyramide, `indice_champ = INDICE_CHAMP_S` — et l'autre l'INTERPOLÉ —
        `melanger(α)` puis gather sur la vue, `indice_champ = INDICE_CHAMP_VUE`.
        UN SEUL `melanger` PAR TICK (§2-1), garanti par l'uniformité du §2-2
        vérifiée au montage.

        LE BRANCHEMENT CI-DESSOUS PORTE SUR « CET `α` EST-IL L'EXACT ? », PAS
        SUR LE CÔTÉ. C'est la différence que le §2-2 exige : un `if` par côté
        dupliquerait la logique du tick en deux chemins qui dériveraient l'un de
        l'autre. Ici le côté n'a d'effet qu'à travers `couple_alphas`.

        LES DEUX ÉCRANS PARTAGENT UN SEUL CENTRE FOVÉAL (§4-6) : rien n'écrit
        `centre_fin` entre eux. Le déplacer là serait SILENCIEUX du côté exact —
        le gather relirait `geo.origines(j, centre_fin)` alors que le contenu
        des fenêtres n'a pas été roulé — et bruyant du côté interpolé
        (`ReadoutHorsRegistre`) : c'est cette asymétrie qui rend la garde
        nécessaire. Conséquence NOMMÉE et non tranchée (§5) : la caméra se
        rafraîchit une fois par TICK, pour DEUX écrans.

        Un trou de couverture du gather LÈVE et n'est pas rattrapé (§4-7) :
        choisir un `cote_px` couvrable appartient à l'appelant."""
        # LE COMPTE RENDU DU TICK PRÉCÉDENT EST INVALIDÉ D'ABORD, avant le
        # premier geste qui peut lever. Sans cette ligne, un tick interrompu —
        # le gather d'un écran qui trouve un trou de couverture, alors que
        # `melanger` a déjà eu lieu et que `frame()` a déjà avancé l'état —
        # laisserait `compte_rendu` porter les compteurs du tick PRÉCÉDENT. Un
        # appelant qui rattrape par classe nommée en amont et le relit lirait
        # alors un chiffre qui ne décrit AUCUN tick, sans aucun symptôme : la
        # signature §A53 en miniature, dans un attribut dont un prereg futur
        # tirera un seuil.
        #
        # SANS ORDINAL : cette phrase a dit « le SECOND écran », ce qui décrit
        # un scénario inatteignable. Les deux gathers d'un tick partagent `geo`,
        # `cote_px` et le MÊME `centre_fin` (§4-6) : leur prédicat de couverture
        # est identique, donc le PREMIER échoue toujours en premier. Ce qui rend
        # le cas réel n'est pas le rang du gather, c'est que `melanger` et
        # `frame()` le PRÉCÈDENT tous deux du côté `interpoler`.
        self.compte_rendu = None
        self.pyramide.frame(PAS_FOVEA)

        ecrans: list[Ecran] = []
        clamps = 0
        non_finis = 0
        for alpha in self.couple_alphas:
            if alpha == ALPHA_EXACT:
                # L'IMAGE EXACTE CONTOURNE LE NOYAU (§2-1) : gather DIRECT sur
                # la pyramide, jamais `melanger(1.0)`. « `α = 1` gratuit » veut
                # dire hors du noyau, donc hors de `I` — la ligne `P-b` du
                # prereg `où-vit-le-rendu` porte `30·I`, pas `60·I`.
                ecrans.append(gather_chemin_de_cout(
                    self.pyramide, cote_px, indice_champ=INDICE_CHAMP_S))
            else:
                vue = self.applicateur.tampon.melanger(self.pyramide, alpha)
                clamps += vue.clamps
                non_finis += vue.non_finis
                ecrans.append(gather_chemin_de_cout(
                    vue, cote_px, indice_champ=INDICE_CHAMP_VUE))
            # AUCUNE COPIE D'ÉCRAN, ET CE N'EST PAS UNE COMMODITÉ (§4-4). Le
            # gather ALLOUE sa sortie (`ecran = xp.full(...)`, `chemin_de_cout`)
            # et la retourne : l'écran n'aliase rien, donc une copie ne
            # garderait RIEN et chargerait une copie de plus dans `I` à chaque
            # image. Ce qui aliase, c'est la VUE — `fenetres` pointe sur le
            # tampon interne `_s_out`, réécrit au `melanger` suivant. C'est
            # pourquoi la boucle ne conserve aucune vue au-delà de son tick.

        self.compte_rendu = CompteRenduTick(clamps, non_finis)
        return CoupleEcrans(*ecrans)
