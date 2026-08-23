"""Tests — INTERPOLATION DE READOUT (`src/f1_gpu/interpolation_readout.py`),
les six verrous du §7 du spec vérifiés MÉCANIQUEMENT.

Backend numpy (B1), tailles minuscules (VM) : on teste ici la LOGIQUE et les
GARDES, JAMAIS un coût. `I` ne se mesure pas ici — §A61 l'interdit tant que
l'instrument n'est pas qualifié, et §A63 ne l'a pas qualifié.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.f1_gpu.interpolation_readout import (
    ALPHA_EXTRAPOLER,
    ALPHA_INTERPOLER,
    INDICE_CHAMP_S,
    ReadoutHorsRegistre,
    ReadoutInterditDansEtat,
    ReadoutSansCapture,
    TamponReadout,
    _MODULES_ETAT,
)
from src.f1_gpu.pyramide import GeometriePyramide, PyramideFovea, Slot
from src.f1_gpu.transferts import TransfertComptable

N_FOV = 8
N0 = 16
N_NIV = 4

SLOTS = (
    Slot(niveau=1, c=4, role="fovea"),
    Slot(niveau=2, c=8, role="fovea"),
    Slot(niveau=3, c=8, role="fovea"),
)


def _pyramide():
    geo = GeometriePyramide(n_fov=N_FOV, n_niv=N_NIV, n0=N0, slots=SLOTS)
    transferts = TransfertComptable(np)
    pyr = PyramideFovea(np, geo, transferts, eps_detail=1e30)
    transferts.frame_suivante()
    return pyr


def _peindre_s(pyr, valeur: float) -> None:
    """Peint le champ `s` de toutes les fenêtres avec une valeur constante."""
    for j in pyr.geo.niveaux_gpu:
        pyr.fenetres[j][:, :, INDICE_CHAMP_S, :, :] = float(valeur)


def _peindre_s_gradient(pyr) -> None:
    """Peint `s` avec un motif NON UNIFORME, dépendant de la position.

    Un champ constant rend le gather INSENSIBLE à `centre_fin` : un décalage
    de la vue passerait alors tous les verrous du fichier. Mesuré par la revue
    de la tâche 4, pas supposé."""
    for j in pyr.geo.niveaux_gpu:
        fen = pyr.fenetres[j]
        n = fen.shape[-1]
        y = np.arange(n, dtype=np.float32).reshape(1, 1, n, 1)
        x = np.arange(n, dtype=np.float32).reshape(1, 1, 1, n)
        fen[:, :, INDICE_CHAMP_S, :, :] = 100.0 * j + 10.0 * y + x


def test_alpha_zero_rend_s_prev():
    """Verrou (a) — borne basse. Propriété du NOYAU : les images exactes
    CONTOURNENT le noyau en production (§3 du spec)."""
    pyr = _pyramide()
    _peindre_s(pyr, 2.0)
    tampon = TamponReadout(pyr)
    tampon.capturer(pyr)          # s_prev <- 2.0
    _peindre_s(pyr, 7.0)          # s_cur  <- 7.0
    vue = tampon.melanger(pyr, 0.0)
    for j in pyr.geo.niveaux_gpu:
        assert np.array_equal(vue.fenetres[j][:, :, 0], np.full_like(
            vue.fenetres[j][:, :, 0], 2.0))


def test_alpha_un_rend_s_cur():
    """Verrou (a) — borne haute."""
    pyr = _pyramide()
    _peindre_s(pyr, 2.0)
    tampon = TamponReadout(pyr)
    tampon.capturer(pyr)
    _peindre_s(pyr, 7.0)
    vue = tampon.melanger(pyr, 1.0)
    for j in pyr.geo.niveaux_gpu:
        assert np.array_equal(vue.fenetres[j][:, :, 0], np.full_like(
            vue.fenetres[j][:, :, 0], 7.0))


def test_melange_a_un_demi():
    """Le mode INTERPOLER : α = ½ entre deux états RÉELS."""
    pyr = _pyramide()
    _peindre_s(pyr, 2.0)
    tampon = TamponReadout(pyr)
    tampon.capturer(pyr)
    _peindre_s(pyr, 8.0)
    vue = tampon.melanger(pyr, ALPHA_INTERPOLER)
    for j in pyr.geo.niveaux_gpu:
        assert np.allclose(vue.fenetres[j][:, :, 0], 5.0)


def test_melanger_est_deterministe():
    """Déterminisme de `melanger` — et RIEN DE PLUS.

    Ce test s'appelait « les deux modes partagent un seul chemin » et ne
    vérifiait pas cela : comparer deux appels au MÊME `α` ne peut pas
    distinguer un noyau unique d'un noyau branché sur `α`, puisque les deux
    sont déterministes à `α` fixé. Renommé pour dire ce qu'il teste
    réellement — une propriété vraie et utile. Le chemin unique est verrouillé
    par `test_un_seul_noyau_affine_les_alphas_sont_colineaires`.
    """
    pyr = _pyramide()
    _peindre_s(pyr, 3.0)
    tampon = TamponReadout(pyr)
    tampon.capturer(pyr)
    _peindre_s(pyr, 9.0)

    for alpha in (ALPHA_INTERPOLER, ALPHA_EXTRAPOLER):
        premier = {j: np.array(tampon.melanger(pyr, alpha).fenetres[j],
                               copy=True)
                   for j in pyr.geo.niveaux_gpu}
        second = {j: np.array(tampon.melanger(pyr, alpha).fenetres[j],
                              copy=True)
                  for j in pyr.geo.niveaux_gpu}
        for j in pyr.geo.niveaux_gpu:
            assert premier[j].tobytes() == second[j].tobytes(), (
                f"le mode α={alpha} n'est pas déterministe au niveau {j}")


def test_extrapoler_depasse_et_le_clamp_compte():
    """Verrou (d) — §5 du spec. Le clamp COMPTE et REPORTE, il ne lève pas."""
    pyr = _pyramide()
    _peindre_s(pyr, 10.0)
    tampon = TamponReadout(pyr)
    tampon.capturer(pyr)          # s_prev = 10
    _peindre_s(pyr, 1.0)          # s_cur  = 1  ⇒ α=1½ donne −3,5
    vue = tampon.melanger(pyr, ALPHA_EXTRAPOLER)

    assert vue.clamps > 0, "l'extrapolation devait passer sous zéro"
    for j in pyr.geo.niveaux_gpu:
        assert (vue.fenetres[j] >= 0.0).all(), "le clamp n'a pas tenu"


def test_interpoler_ne_depasse_jamais():
    """Verrou (d), face FAVORABLE — un critère se vérifie aussi sur le cas
    qui doit PASSER (§A52), sinon il punit la qualité qu'il contrôle."""
    pyr = _pyramide()
    _peindre_s(pyr, 10.0)
    tampon = TamponReadout(pyr)
    tampon.capturer(pyr)
    _peindre_s(pyr, 1.0)
    vue = tampon.melanger(pyr, ALPHA_INTERPOLER)
    assert vue.clamps == 0, "α=½ entre deux états positifs ne peut pas clamper"


def test_un_seul_noyau_affine_les_alphas_sont_colineaires():
    """Verrou (b), RÉPARÉ — §2 du spec RÉELLEMENT mécanisé.

    « `I` est invariant au mode par construction » ne tient que s'il n'existe
    qu'UN noyau. La conséquence testable est l'AFFINITÉ : `s_out` est linéaire
    en `α`, donc trois évaluations sont colinéaires —

        s_out(α) = s_out(0) + α·(s_out(1) − s_out(0))

    Un chemin propre à un mode brise cette identité, et CE VERROU EST LE SEUL
    QUI L'ATTRAPE — mesuré par mutation, pas supposé :

      - décalage CONDITIONNÉ AU MODE (`if α > 1`) : `1 failed, 6 passed`, et
        l'unique échec est ce test. Les six autres verrous le laissent passer.
      - décalage INCONDITIONNEL (`− 0,001` partout) : `3 failed, 4 passed` —
        attrapé par les verrous de BORNES de la tâche 1, et ce test-ci le laisse
        passer, car un décalage indépendant de `α` s'annule dans l'identité
        (`s_out(0)` et `s_out(1)` le portent aussi).

    Les deux familles sont donc NON REDONDANTES : chacune est l'unique
    détecteur de sa classe. Ce n'est pas une faiblesse du verrou, c'est sa
    portée exacte — et l'écrire évite de croire qu'il garde plus qu'il ne garde
    (`§A53` : un verrou ne garde que ce que son état de test allume).

    LES VALEURS SONT CHOISIES POUR QUE LE CLAMP NE MORDE JAMAIS (`s_prev = 3`,
    `s_cur = 9`, donc `s_out ≥ 3` sur tout `α ∈ [0 ; 1,5]`). Le clamp brise
    LÉGITIMEMENT l'affinité — le tester dans sa zone testerait autre chose.
    """
    pyr = _pyramide()
    _peindre_s(pyr, 3.0)
    tampon = TamponReadout(pyr)
    tampon.capturer(pyr)
    _peindre_s(pyr, 9.0)

    zero = {j: np.array(tampon.melanger(pyr, 0.0).fenetres[j], copy=True)
            for j in pyr.geo.niveaux_gpu}
    un = {j: np.array(tampon.melanger(pyr, 1.0).fenetres[j], copy=True)
          for j in pyr.geo.niveaux_gpu}

    for alpha in (ALPHA_INTERPOLER, ALPHA_EXTRAPOLER):
        vue = tampon.melanger(pyr, alpha)
        assert vue.clamps == 0, (
            f"α={alpha} a clampé : le test sort de sa zone de validité")
        for j in pyr.geo.niveaux_gpu:
            attendu = zero[j] + alpha * (un[j] - zero[j])
            assert np.allclose(vue.fenetres[j], attendu), (
                f"α={alpha} au niveau {j} sort de la droite affine : "
                "il existe un chemin propre à un mode")


def _appeler_depuis(nom_module: str, fonction, *args, **kwargs):
    """Exécute `fonction` depuis une frame dont `__name__` est `nom_module`.

    Même patron que `tests/test_chemin_de_cout.py` : on fabrique une frame
    portant le nom d'un module d'état, sans importer ce module."""
    code = compile("resultat = fonction(*args, **kwargs)", "<frame>", "exec")
    portee = {"__name__": nom_module, "fonction": fonction,
              "args": args, "kwargs": kwargs}
    exec(code, portee)
    return portee["resultat"]


def test_l_etat_de_la_pyramide_est_bit_identique_apres_appel():
    """Verrou (c), face STRUCTURELLE — la garde de fond du §4.

    Le seul verrou que §A53 ne sait pas contourner : si le module n'écrit
    nulle part dans le tenseur, aucune fuite n'est possible, quel que soit
    l'appelant.

    C'EST LE TEST QUI A ÉTÉ CORRIGÉ, PAS SA DOCSTRING — et le choix se motive.
    La première rédaction prenait son instantané APRÈS `TamponReadout(...)` et
    APRÈS `capturer`, donc ne couvrait que `melanger` alors que sa prose
    annonçait « le module ». Restreindre la prose à `melanger` aurait laissé
    DEUX points d'entrée non gardés — et ce sont eux qui touchent le tenseur de
    plus près : `__init__` le lit pour se dimensionner, `capturer` en copie un
    champ. Une fuite y serait exactement aussi silencieuse. Le test couvre donc
    désormais les TROIS entrées, en deux instantanés, parce que le pas de `F`
    (ici simulé par une repeinture) doit légitimement modifier le tenseur entre
    la capture et le mélange."""
    pyr = _pyramide()
    _peindre_s(pyr, 4.0)

    # Instantané 1 — couvre `__init__` ET `capturer`.
    avant_capture = {j: np.array(pyr.fenetres[j], copy=True)
                     for j in pyr.geo.niveaux_gpu}
    tampon = TamponReadout(pyr)
    tampon.capturer(pyr)
    for j in pyr.geo.niveaux_gpu:
        assert avant_capture[j].tobytes() == pyr.fenetres[j].tobytes(), (
            f"`__init__`/`capturer` ont modifié le tenseur de F au niveau {j} "
            "— FUITE")

    _peindre_s(pyr, 6.0)          # tient lieu du pas de `F`

    # Instantané 2 — couvre `melanger`.
    avant = {j: np.array(pyr.fenetres[j], copy=True)
             for j in pyr.geo.niveaux_gpu}
    tampon.melanger(pyr, ALPHA_EXTRAPOLER)
    for j in pyr.geo.niveaux_gpu:
        assert avant[j].tobytes() == pyr.fenetres[j].tobytes(), (
            f"le tenseur de F a été modifié au niveau {j} — FUITE")


def test_la_serrure_leve_depuis_un_module_d_etat():
    """Verrou (c), face DE PILE. `pyramide` est le module qui exécute `F` :
    s'il apparaît dans la pile, un état de readout est en train d'atteindre
    le chemin de la physique.

    Conservé comme cas NOMMÉ et canonique ; la couverture de TOUTE la liste est
    portée par `test_la_serrure_mord_depuis_chaque_module_declare`."""
    from src.f1_gpu.interpolation_readout import ReadoutInterditDansEtat

    pyr = _pyramide()
    tampon = TamponReadout(pyr)
    tampon.capturer(pyr)
    with pytest.raises(ReadoutInterditDansEtat):
        _appeler_depuis("pyramide", tampon.melanger, pyr, ALPHA_INTERPOLER)


def test_la_tentative_est_consignee_meme_si_l_exception_est_avalee():
    """La trace survit à un `except RuntimeError` bien intentionné.

    Découvert par mutation sur `chemin_de_cout` le 03/08 : la serrure hérite
    de `RuntimeError`, et un appelant qui enveloppe l'appel d'un
    `except RuntimeError` l'avale EN SILENCE. L'exception peut être étouffée ;
    la trace, non."""
    from src.f1_gpu.interpolation_readout import violations_consignees

    pyr = _pyramide()
    tampon = TamponReadout(pyr)
    tampon.capturer(pyr)
    avant = len(violations_consignees())
    try:
        _appeler_depuis("pyramide", tampon.melanger, pyr, ALPHA_INTERPOLER)
    except RuntimeError:
        pass
    assert len(violations_consignees()) == avant + 1


def test_le_gather_lit_la_vue_octet_pour_octet():
    """Verrou (e) — ce que ce test prouve, ET CE QU'IL NE PROUVE PAS.

    IL PROUVE : le MÊME gather, non modifié, sait lire la vue interpolée — à
    `α = 1` elle rend octet pour octet ce que le gather rend sur la pyramide.
    La surface exposée par `VueInterpolee` est donc suffisante ET correctement
    propagée.

    IL NE PROUVE PAS que « le gather rend les mêmes octets qu'avant
    l'existence de ce module » — et AUCUN test ne le peut, il faudrait une
    référence historique. Cette garantie est portée ENTIÈREMENT par
    `git diff --stat <base de branche> -- src/f1_gpu/chemin_de_cout.py`, vide,
    et par le fait qu'aucun commit de la série n'a jamais touché ce fichier.
    L'écrire ici évite la faute `§A62-bis-2` : une garde promise est une garde
    absente, et un test qui promet plus qu'il ne tient est de la même famille.

    LE CHAMP EST PEINT NON UNIFORME, et c'est LOAD-BEARING : sous un champ
    constant le gather est insensible à `centre_fin`, et un décalage de la vue
    passait les DOUZE verrous de ce fichier. Mesuré par la revue, pas supposé.
    """
    from src.f1_gpu.chemin_de_cout import gather_chemin_de_cout

    pyr = _pyramide()
    _peindre_s_gradient(pyr)
    cote = N_FOV

    ecran_direct = gather_chemin_de_cout(pyr, cote, indice_champ=INDICE_CHAMP_S)
    assert not np.allclose(ecran_direct, ecran_direct.flat[0]), (
        "le champ témoin doit être NON UNIFORME, sinon `centre_fin` ne "
        "discrimine rien et ce verrou ne garde plus ce qu'il annonce")

    tampon = TamponReadout(pyr)
    tampon.capturer(pyr)
    vue = tampon.melanger(pyr, 1.0)
    ecran_vue = gather_chemin_de_cout(vue, cote, indice_champ=0)

    assert ecran_direct.tobytes() == ecran_vue.tobytes(), (
        "à α=1 la vue doit rendre exactement le gather direct")


def test_deux_appels_rendent_les_memes_octets():
    """Verrou (f) — fonction pure, anti-PERSIST. La SYNTHÈSE est sans état,
    même si `s_prev` en a un : c'est la seconde jambe de la défense du
    §4-bis — de l'HISTOIRE, pas de la SYNTHÈSE."""
    pyr = _pyramide()
    _peindre_s(pyr, 2.0)
    tampon = TamponReadout(pyr)
    tampon.capturer(pyr)
    _peindre_s(pyr, 5.0)

    premier = {j: np.array(tampon.melanger(pyr, ALPHA_INTERPOLER).fenetres[j],
                           copy=True) for j in pyr.geo.niveaux_gpu}
    second = {j: np.array(tampon.melanger(pyr, ALPHA_INTERPOLER).fenetres[j],
                          copy=True) for j in pyr.geo.niveaux_gpu}
    for j in pyr.geo.niveaux_gpu:
        assert premier[j].tobytes() == second[j].tobytes()


# ─────────────────────────────────────────────────────────────────────────────
# LE TEST QUI MANQUAIT — aucun des douze verrous ci-dessus n'appelle `frame()`
# ─────────────────────────────────────────────────────────────────────────────
#
# La fovéa était IMMOBILE dans tout ce fichier, et c'est précisément l'état de
# test qui cachait le défaut : `s_prev` est indexé en coordonnées locales à la
# fenêtre, `frame()` roule le contenu des fenêtres quand la fovéa avance, donc
# après un déplacement les deux opérandes du mélange ne décrivent plus les mêmes
# cellules du monde. Douze verrous verts, aucun symptôme, `clamps == 0`.


def test_melanger_leve_hors_registre_apres_un_frame_qui_deplace_la_fovea():
    """LE défaut de la série, rendu BRUYANT — `capturer` → `frame()` → `melanger`.

    C'est le cycle de PRODUCTION : on capture avant le pas de `F`, `frame()`
    déplace la fovéa PUIS applique `F`, et on mélange après. Sans la garde, ce
    cycle rendrait un champ fantômé d'une cellule fine, dont les colonnes
    entrantes porteraient des valeurs qui n'ont jamais existé — sans clamp,
    sans levée, sans compteur non nul. Rien en aval ne pourrait le voir."""
    pyr = _pyramide()
    _peindre_s_gradient(pyr)
    tampon = TamponReadout(pyr)
    centre_avant = int(pyr.centre_fin)
    tampon.capturer(pyr)

    pyr.frame(1)                       # déplace la fovéa d'UNE cellule fine
    assert int(pyr.centre_fin) == centre_avant + 1, (
        "le test suppose que `frame(1)` avance le centre fovéal ; s'il ne le "
        "fait plus, ce verrou ne garde plus rien")

    with pytest.raises(ReadoutHorsRegistre):
        tampon.melanger(pyr, ALPHA_INTERPOLER)


def test_un_frame_sans_deplacement_de_fovea_ne_leve_pas():
    """Le JUMEAU — la face qui doit PASSER (§A52).

    Un critère qui ne se vérifie que sur le cas défavorable punit la qualité
    qu'il contrôle : il faut prouver que la garde ne mord pas sur l'usage
    LÉGITIME. `frame(0)` est possible dans cette API, et ce n'est pas une
    supposition — c'est LU dans le code puis VÉRIFIÉ ici : `pyramide.py:463`
    ne rejette que `delta_x < 0`, `frame(0)` laisse donc `centre_fin`
    inchangé, ne roule aucune fenêtre (`dx == 0` à tous les niveaux) et
    applique tout de même `pas_f`. C'est exactement le cycle
    capturer → F → melanger, à fovéa immobile — le seul régime où ce composant
    est utilisable aujourd'hui."""
    pyr = _pyramide()
    _peindre_s_gradient(pyr)
    tampon = TamponReadout(pyr)
    centre_avant = int(pyr.centre_fin)
    tampon.capturer(pyr)

    diagnostic = pyr.frame(0)          # F appliqué, fovéa IMMOBILE
    assert diagnostic["niveaux_deplaces"] == [], (
        "`frame(0)` ne doit déplacer aucun niveau, sinon ce jumeau ne prouve "
        "plus ce qu'il annonce")
    assert int(pyr.centre_fin) == centre_avant

    vue = tampon.melanger(pyr, ALPHA_INTERPOLER)   # ne doit PAS lever
    assert vue.centre_fin == pyr.centre_fin


def test_melanger_sans_capturer_leve():
    """`s_prev` à l'allocation vaut ZÉRO — mélanger là-dessus est faux et muet.

    Sans capture, `s_out = α·s_cur` : un état atténué, faux, positif, donc
    `clamps == 0` et aucun symptôme. La garde le refuse au lieu de le rendre."""
    pyr = _pyramide()
    _peindre_s(pyr, 5.0)
    tampon = TamponReadout(pyr)

    with pytest.raises(ReadoutSansCapture):
        tampon.melanger(pyr, ALPHA_INTERPOLER)


@pytest.mark.parametrize("nom_module", sorted(_MODULES_ETAT))
def test_la_serrure_mord_depuis_chaque_module_declare(nom_module):
    """Verrou (c), face DE PILE — PARAMÉTRÉ SUR TOUTE LA LISTE.

    Un seul nom était couvert (`pyramide`), et pendant ce temps la liste était
    FAUSSE : trois modules d'état — `solver` (l'avanceur CPU
    `lax_friedrichs_step`/`simulate`), `terrains` (`rest_residual` appelle
    `simulate`), `rederive_borne` (pilote `run_episode`) — y manquaient, alors
    que leurs frères `solver_wetdry` et `sediment` y étaient déjà.

    CE QUE CE TEST DIT, ET CE QU'IL NE DIT PAS. Il prouve que CHAQUE entrée
    déclarée mord réellement — donc qu'aucune n'est un nom mort, mal orthographié
    ou rendu inopérant par un changement du comparateur. Il ne peut RIEN dire de
    ce qui MANQUE à la liste : aucun test ne le peut, c'est la limite écrite dans
    le module lui-même, et c'est pourquoi le verrou STRUCTUREL — et non celui-ci
    — porte la garantie. Mais il est ce qu'une liste doit avoir : le jour où on
    ajoute une entrée, elle est vérifiée, et non seulement écrite."""
    pyr = _pyramide()
    tampon = TamponReadout(pyr)
    tampon.capturer(pyr)
    with pytest.raises(ReadoutInterditDansEtat):
        _appeler_depuis(nom_module, tampon.melanger, pyr, ALPHA_INTERPOLER)


def test_un_non_fini_est_compte_a_part_et_ne_gonfle_pas_les_clamps():
    """`sortie < 0.0` est FAUX pour un NaN — le clamp ne le voyait pas.

    La docstring promettait « clamp `s ≥ 0` » et un test assertait
    `(… >= 0.0).all()` : un NaN traversait les deux sans être compté. Traitement
    tranché et écrit dans `melanger` — COMPTÉ à part (`non_finis`), NI clampé
    (l'écraser à zéro déguiserait une faute de `F` en cellule sèche), NI levé
    (§5 : le readout compte et reporte, il ne tue pas le rendu pour une faute
    d'amont). `clamps` reste réservé à ce qu'il mesure : l'extrapolation sous
    zéro."""
    pyr = _pyramide()
    _peindre_s(pyr, 2.0)
    tampon = TamponReadout(pyr)
    tampon.capturer(pyr)
    _peindre_s(pyr, 8.0)

    niveaux = list(pyr.geo.niveaux_gpu)
    for j in niveaux:                  # une cellule empoisonnée par niveau
        pyr.fenetres[j][0, 0, INDICE_CHAMP_S, 0, 0] = np.nan

    vue = tampon.melanger(pyr, ALPHA_INTERPOLER)

    assert vue.non_finis == len(niveaux), (
        f"un non-fini par niveau attendu, {vue.non_finis} compté(s)")
    assert vue.clamps == 0, (
        "un NaN n'est pas un clamp : le compteur de clamps ne doit pas le "
        "porter, sinon le seuil qu'un prereg futur en tirera mesurera deux "
        "phénomènes sans rapport")
    for j in niveaux:
        assert np.isnan(vue.fenetres[j][0, 0, 0, 0, 0]), (
            "le NaN doit TRAVERSER : l'écraser à zéro effacerait la preuve "
            "d'un état corrompu en amont")
        reste = np.delete(vue.fenetres[j][:, :, 0].ravel(), 0)
        assert np.allclose(reste, 5.0), (
            "le NaN ne doit contaminer que sa propre cellule")
