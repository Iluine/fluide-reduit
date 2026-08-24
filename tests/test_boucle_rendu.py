"""Tests — BOUCLE DE RENDU (`src/f1_gpu/boucle_rendu.py`), les verrous de la
tâche 1 du plan, vérifiés MÉCANIQUEMENT.

Autorité : `claude/spec-boucle-rendu-2026-08-24.md`, endossée (son §8 porte le
statut courant et amende l'épigraphe et le §7).

Backend numpy, tailles minuscules : on teste ici la LOGIQUE et les GARDES,
JAMAIS un coût. Aucune horloge, aucun chiffre de `I` — §A61 tient, et le §1 de
la spec interdit toute horloge murale dans ce chantier.

LES HELPERS SONT IMPORTÉS DE `tests/test_interpolation_readout.py`, PAS
RECOPIÉS. Ils sont éprouvés et le plan demande de les réutiliser ; les recopier
créerait deux définitions qui dériveraient l'une de l'autre sans que rien ne le
signale. En particulier `_peindre_s_gradient` : un champ `s` CONSTANT rend le
gather insensible à `centre_fin`, et un décalage de vue passerait alors tous les
verrous de ce fichier sans bruit (fait mesuré par la revue de la tâche 4 du
readout, pas supposé).
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from src.f1_gpu import boucle_rendu
from src.f1_gpu.boucle_rendu import (
    ALPHA_EXACT,
    COTE_EXTRAPOLER,
    COTE_INTERPOLER,
    COUPLES_PAR_COTE,
    BoucleRendu,
    CoteInconnu,
    MontageBoucleInvalide,
)
from src.f1_gpu.chemin_de_cout import gather_chemin_de_cout
from src.f1_gpu.interpolation_readout import (
    ALPHA_EXTRAPOLER,
    ALPHA_INTERPOLER,
    ReadoutCaptureImpossible,
    TamponReadout,
    _MODULES_ETAT,
    installer_capture_en_registre,
)
from tests.test_interpolation_readout import (
    _pas_f_doublant,
    _peindre_s_gradient,
    _pyramide,
    _pyramide_doublante,
)

# Côté d'écran COUVRABLE par les fenêtres de la pyramide minuscule sur toute la
# durée des runs de ce fichier — vérifié jusqu'à six ticks. Le gather LÈVE sur
# un trou de couverture et la boucle ne le rattrape pas (§4-7) : choisir un
# `cote_px` couvrable est une responsabilité d'appelant, pas une garde à écrire.
COTE_PX = 16

# Côté d'écran qui DÉBORDE le niveau le plus grossier de ce banc, donc trou de
# couverture structurel dès le premier tick — le matériau du verrou §4-7.
COTE_PX_TROP_GRAND = 32

N_TICKS = 3

COTES = (COTE_INTERPOLER, COTE_EXTRAPOLER)

# Les filets que le §4-8 interdit : ils avalent les cinq serrures du readout et
# du chemin-de-coût, qui héritent toutes de `RuntimeError`.
FILETS_INTERDITS = frozenset({"RuntimeError", "Exception", "BaseException"})

# Les compteurs de `TransfertComptable` comparés par le verrou (a) : des OCTETS
# et des NOMBRES D'APPELS. Les champs `h2d_ms` / `d2h_ms` en sont exclus
# délibérément — ils sortent de `perf_counter` et un chronomètre n'a rien à
# faire dans un verrou (§A61).
COMPTEURS_TRANSFERT = ("h2d_octets", "d2h_octets", "h2d_n", "d2h_n")


def _memes_octets(a, b) -> bool:
    """Égalité BIT À BIT — mêmes forme, même dtype, mêmes octets.

    Ni `allclose` ni `array_equal` : le premier tolère une dérive, le second
    déclare inégaux deux NaN identiques et masquerait donc une égalité réelle
    d'un état corrompu en amont. `tobytes()` compare ce que la garantie dit :
    des octets."""
    return (a.shape == b.shape and a.dtype == b.dtype
            and a.tobytes() == b.tobytes())


def _octets(tableau) -> tuple:
    """Forme, dtype et OCTETS — l'identité complète d'un tableau."""
    return (tableau.shape, tableau.dtype, tableau.tobytes())


def _etat_en_octets(pyr) -> dict:
    """L'état observable de la pyramide, en octets — CINQ morceaux, et la liste
    est celle que la garantie §4-1/§13-4 doit couvrir, pas un échantillon :

      - `centre_fin` — la position de la fovéa ;
      - `fenetres[j]` — TOUS les champs de TOUTES les fenêtres de TOUS les
        niveaux, pas seulement le champ `s` ;
      - `references[j]` — la référence du schéma diff, que la remontée met à
        jour incrémentalement ;
      - `monde0` — LE MORCEAU QUI MANQUAIT, et il est obligatoire. C'est la
        source de TOUTE prédiction (`pyramide.py:421`), relue à chaque roll par
        `_predire_niveau_cpu` pour remplir les colonnes ENTRANTES : ce que `F`
        consommera aux ticks SUIVANTS. Une écriture du chemin de rendu dans
        `monde0`, hors de la bande qui entre pendant les `N_TICKS` de ce
        verrou, est TOTALEMENT invisible aux trois morceaux ci-dessus — ni
        `fenetres`, ni `references`, ni `centre_fin` ne bougent, et le run nu
        rend les mêmes octets. C'est exactement la fuite que le §4-1 interdit,
        et le verrou la laissait passer ;
      - les COMPTEURS de `TransfertComptable` — le §4 du spec readout fait de
        la comptabilité le motif même de la garde (« corromprait la
        comptabilité de conservation sans produire aucun symptôme »).

    ⚠ SUR LES COMPTEURS, SEULS LES OCTETS ET LES APPELS SONT COMPARÉS, JAMAIS
    LES CHAMPS `_ms`. Ces derniers viennent de `perf_counter`
    (`transferts.py:66-68`) : les comparer rendrait ce verrou instable, et un
    CHRONOMÈTRE DANS UN VERROU serait un contresens dans une maison où §A61
    tient. Ce qui est comparé est un COMPTE, pas une durée.

    Seuls les bilans CLÔTURÉS sont lisibles par l'API publique
    (`frame_suivante()`), et `frame()` n'en clôt aucun : l'appelant clôt une
    fois, symétriquement des deux côtés, avant de photographier."""
    photo = {"centre_fin": int(pyr.centre_fin)}
    for j in pyr.geo.niveaux_gpu:
        photo[("fenetres", j)] = _octets(pyr.fenetres[j])
        photo[("references", j)] = _octets(pyr.references[j])
    photo["monde0"] = _octets(pyr.monde0)
    photo["transferts_n_bilans"] = len(pyr.transferts.bilans)
    for rang, bilan in enumerate(pyr.transferts.bilans):
        photo[("transferts", rang)] = tuple(
            (cle, bilan[cle]) for cle in COMPTEURS_TRANSFERT)
    return photo


def _espionner_melanger(monkeypatch) -> list:
    """Enregistre chaque `VueInterpolee` rendue par `TamponReadout.melanger`, et
    rend la liste des appels sous forme `(alpha, vue)`.

    POURQUOI UN ESPION PLUTÔT QU'UN ATTRIBUT DE LA BOUCLE. Le §4-4 interdit à la
    boucle de CONSERVER une vue au-delà de son tick ; lui faire exposer sa
    dernière vue pour rendre les tests commodes contredirait la garde qu'ils
    doivent vérifier. L'espion prend la vue là où elle naît, sans rien demander
    à la boucle. Il porte aussi le COMPTE des appels — c'est lui qui mécanise
    « un seul `melanger` par tick » (§2-1).

    SUR LA CLASSE, PAS SUR L'INSTANCE, ET CE N'EST PAS UN DÉTAIL. Poser
    l'espion en attribut d'instance dépendrait de deux faits que rien ne
    garantit : que `TamponReadout` n'ait pas de `__slots__` — le patron est
    DÉJÀ appliqué à deux des trois classes de son module, `VueInterpolee` et
    `ApplicateurCaptureRegistre` — et que la boucle déréférence `melanger` à
    chaque appel plutôt que d'en garder une référence liée. Le jour où
    `TamponReadout` gagne un `__slots__`, quatre verrous tomberaient SANS que
    la boucle ait changé. `monkeypatch.setattr` sur la classe est insensible
    aux deux, et il est RESTAURÉ automatiquement en fin de test — l'attribut
    d'instance, lui, n'était jamais retiré."""
    appels: list = []
    melanger_originel = TamponReadout.melanger

    def melanger_espion(self, pyramide, alpha):
        vue = melanger_originel(self, pyramide, alpha)
        appels.append((float(alpha), vue))
        return vue

    monkeypatch.setattr(TamponReadout, "melanger", melanger_espion)
    return appels


# ─────────────────────────────────────────────────────────────────────────────
# LE CÔTÉ — paramètre sans défaut, et le couple qu'il choisit
# ─────────────────────────────────────────────────────────────────────────────

def test_le_cote_n_a_aucune_valeur_par_defaut():
    """L'ABSENCE DE DÉFAUT EST LE REFUS DE TRANCHER §A62, mécanisé.

    Le côté appartient à la lecture d'orientation, qui n'est PAS ordonnée et
    dont le prereg n'est pas écrit (§6 de la spec). Un défaut — n'importe
    lequel — trancherait en douce, dans du code moteur, une question que deux
    documents laissent explicitement ouverte : l'appelant hériterait d'un choix
    que personne n'a fait et rien ne le lui dirait. La boucle est paramétrable
    parce que le côté n'est pas choisi : c'est une GARDE, pas une souplesse."""
    parametre = inspect.signature(BoucleRendu.__init__).parameters["cote"]
    assert parametre.default is inspect.Parameter.empty, (
        "`cote` a reçu une valeur par défaut : le côté serait alors tranché "
        "par le code, alors que §6 de la spec et §A62 le laissent ouvert")


def test_un_cote_inconnu_est_refuse_au_montage():
    """`if` + levée d'une classe NOMMÉE (§4-9) — jamais un `assert`, qui
    disparaît sous `python -O`, donc exactement dans le régime où l'on
    mesurera. Et jamais un silence : un côté mal orthographié qui tomberait sur
    un défaut rendrait des images du mauvais côté sans aucun symptôme."""
    pyr = _pyramide()
    with pytest.raises(CoteInconnu):
        BoucleRendu(pyr, "interpolet")


@pytest.mark.parametrize("cote, attendu", [
    (COTE_INTERPOLER, (ALPHA_INTERPOLER, ALPHA_EXACT)),
    (COTE_EXTRAPOLER, (ALPHA_EXACT, ALPHA_EXTRAPOLER)),
])
def test_le_couple_sort_en_alpha_croissant_avec_exactement_un_alpha_exact(
        cote, attendu):
    """§2-2, ENDOSSÉ par le §8 point 1 : le côté ne choisit QUE la paire d'`α`.

    Les deux valeurs sont celles du module `interpolation_readout`, IMPORTÉES —
    ce test les compare aux constantes, il ne les recopie pas en littéral : une
    dérive de `ALPHA_INTERPOLER` ou de `ALPHA_EXTRAPOLER` doit se propager, pas
    se faire contredire par un chiffre gravé ici.

    Deux propriétés, toutes deux gravées : le couple sort en `α` CROISSANT, et
    EXACTEMENT UN des deux vaut `α = 1,0`. C'est cette uniformité qui rend le
    tick unique — un chemin de code, aucun branchement par côté."""
    boucle = BoucleRendu(_pyramide(), cote)
    assert boucle.couple_alphas == attendu
    assert boucle.couple_alphas[0] < boucle.couple_alphas[1], (
        "le couple doit sortir en `α` croissant (§2-2)")
    assert list(boucle.couple_alphas).count(ALPHA_EXACT) == 1, (
        "exactement un écran du couple doit valoir `α = 1,0` (§2-2) — deux "
        "rendraient le tick redondant, zéro ferait payer le noyau deux fois")


def test_les_deux_cotes_couvrent_la_table_des_couples():
    """La table est la seule source du couple : aucun côté ne doit exister sans
    couple, ni couple sans côté. Un troisième côté ajouté à la table sans être
    exposé, ou l'inverse, serait un chemin mort — et un chemin mort est
    l'endroit où une revue future croit lire une garde."""
    assert set(COUPLES_PAR_COTE) == set(COTES)


# ─────────────────────────────────────────────────────────────────────────────
# VERROU (a) — L'ÉTAT NE VOIT PAS LE RENDU
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cote", COTES)
@pytest.mark.parametrize("fabrique", [_pyramide, _pyramide_doublante],
                         ids=["f_jetable", "f_doublant"])
def test_l_etat_ne_voit_pas_le_rendu(cote, fabrique):
    """VERROU (a) — un run de `N` ticks AVEC rendu et un run de `N` `frame()`
    NUS laissent un état BIT-IDENTIQUE.

    IL PORTE DEUX CHOSES, et la seconde est neuve :

      1. LA GARANTIE §13-4 / §4-1 DE LA SPEC — « rien du readout ne remonte
         jamais dans l'état ». Elle est ici testée DE L'EXTÉRIEUR : on ne
         regarde aucun tampon interne, on compare deux histoires du monde. Une
         fuite de `s_out`, de `s_prev` ou d'un écran dans le tenseur que `F`
         consomme corromprait la comptabilité de conservation SANS produire
         aucun symptôme — la signature §A53 retournée ;

      2. L'EXEMPTION DU §4-3 DE LA SPEC, mécanisée. Le §4-3 sort
         `boucle_rendu` de `_MODULES_ETAT` : sans cette exemption,
         `_verrouiller_consommateur` lèverait `ReadoutInterditDansEtat` à
         CHAQUE `melanger`, puisque `tick` appelle `frame()`. Mais une exemption
         écrite dans un document et non mécanisée est une garde PROMISE, donc
         une garde ABSENTE (`PREREGISTRATION.md:10152`). Ce verrou EST sa
         mécanisation : il prouve de l'extérieur ce que le §4-3 affirme — la
         boucle APPELLE l'état, elle ne l'avance pas autrement, et rien du
         readout n'y a fuité. Un verrou (a) faible ou approximatif rendrait
         l'exemption indéfendable.

    D'où la comparaison sur les OCTETS de TOUS les champs de TOUTES les fenêtres
    de TOUS les niveaux, plus `references[j]` et `centre_fin` : la garantie porte
    sur l'état, pas sur un échantillon de l'état. Un `allclose` laisserait
    passer exactement le genre de fuite qu'on cherche."""
    avec = fabrique()
    _peindre_s_gradient(avec)
    depart = _etat_en_octets(avec)

    boucle = BoucleRendu(avec, cote)
    for _ in range(N_TICKS):
        boucle.tick(COTE_PX)

    nu = fabrique()
    _peindre_s_gradient(nu)
    for _ in range(N_TICKS):
        nu.frame(1)

    # Clôture SYMÉTRIQUE du bilan de transfert — une fois de chaque côté, après
    # la même quantité de travail. `frame()` n'en clôt aucun, donc les `N` ticks
    # se sont accumulés dans le bilan ouvert ; le clore le rend lisible par
    # l'API publique. Une clôture asymétrique ferait diverger le NOMBRE de
    # bilans et le verrou échouerait pour une raison qui n'est pas la sienne.
    avec.transferts.frame_suivante()
    nu.transferts.frame_suivante()

    apres = _etat_en_octets(avec)
    cles_fenetres = [cle for cle in depart
                     if isinstance(cle, tuple) and cle[0] == "fenetres"]
    assert any(apres[cle] != depart[cle] for cle in cles_fenetres), (
        "AUCUNE fenêtre n'a bougé sur les N ticks : la comparaison qui suit "
        "serait vraie sans rien garder — un verrou ne garde que ce que son "
        "état de test allume (§A53). Le delta est exigé sur les FENÊTRES et "
        "non sur la photo entière : `centre_fin` avance de `N_TICKS` par "
        "construction du tick, donc un garde-fou posé sur le dictionnaire "
        "complet serait lui-même VIDE — exactement la faute qu'il prétend "
        "attraper (§A62-bis-2)")

    assert set(apres) == set(_etat_en_octets(nu))
    for cle, valeur in _etat_en_octets(nu).items():
        assert apres[cle] == valeur, (
            f"{cle} diffère entre le run RENDU et le run NU : quelque chose du "
            "readout a atteint l'état. §4-1/§13-4 de la spec, et l'exemption "
            "de `_MODULES_ETAT` du §4-3 tombe avec ce verrou")


def test_boucle_rendu_reste_hors_des_modules_etat():
    """§4-3 de la spec, RULING 3 — et ce test existe pour la revue qui viendra.

    Le critère d'entrée gravé dans `interpolation_readout` dit « le module
    AVANCE l'état ». Appliqué À LA LETTRE, il ferait entrer `boucle_rendu`,
    puisque `tick` appelle `frame()` — et `_verrouiller_consommateur` lèverait
    alors `ReadoutInterditDansEtat` à chaque `melanger` : la boucle deviendrait
    IMPOSSIBLE. La liste s'est révélée fausse deux fois en deux revues (cinq
    modules manquants, puis trois) ; la prochaine appliquera le critère à la
    lettre et cassera la boucle sans voir pourquoi.

    Ce que tranche le §4-3 : la boucle EST le rendu, le consommateur légitime et
    nommé. Elle appelle l'état, elle ne l'avance pas elle-même, et rien d'elle
    n'écrit dans le tenseur que `F` consomme — ce que le verrou (a) ci-dessus
    prouve mécaniquement. Ce test-ci ne prouve rien de plus : il fait ÉCHOUER
    l'ajout avec le motif sous les yeux, plutôt que de laisser une cascade de
    `ReadoutInterditDansEtat` envoyer chercher la faute dans la boucle."""
    assert "boucle_rendu" not in _MODULES_ETAT, (
        "`boucle_rendu` a été ajouté à `_MODULES_ETAT` : `melanger` lèvera "
        "désormais `ReadoutInterditDansEtat` à chaque tick et la boucle est "
        "morte. Lire le §4-3 de `claude/spec-boucle-rendu-2026-08-24.md` AVANT "
        "de trancher : l'exemption est motivée, et le verrou "
        "`test_l_etat_ne_voit_pas_le_rendu` la mécanise")


# ─────────────────────────────────────────────────────────────────────────────
# VERROU (b) — DÉTERMINISME
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cote", COTES)
def test_deux_boucles_identiques_rendent_des_ecrans_bit_identiques(cote):
    """VERROU (b) — même graine, même côté, mêmes octets à l'écran.

    La graine est celle du monde jetable (`GRAINE_MONDE`, défaut du
    constructeur de `PyramideFovea`) : deux pyramides construites pareil
    partent du même contenu. Si deux boucles identiques divergeaient, aucune
    comparaison entre côtés — le verrou (d) de la tâche 2 — ne voudrait plus
    rien dire : on ne saurait pas si l'écart mesuré vient du côté ou du bruit.

    Sur les OCTETS, pas sur une tolérance : une divergence de dernier bit
    signale un chemin de code non déterministe, et c'est ce qu'on veut voir."""
    boucle_a = BoucleRendu(_pyramide_doublante(), cote)
    boucle_b = BoucleRendu(_pyramide_doublante(), cote)
    _peindre_s_gradient(boucle_a.pyramide)
    _peindre_s_gradient(boucle_b.pyramide)

    for indice_tick in range(N_TICKS):
        couple_a = boucle_a.tick(COTE_PX)
        couple_b = boucle_b.tick(COTE_PX)
        for rang, (ecran_a, ecran_b) in enumerate(zip(couple_a, couple_b)):
            assert _memes_octets(ecran_a, ecran_b), (
                f"tick {indice_tick}, écran {rang} : deux boucles identiques "
                "ont rendu des octets différents")
        assert boucle_a.compte_rendu == boucle_b.compte_rendu


# ─────────────────────────────────────────────────────────────────────────────
# VERROU (c), REFORMULÉ PAR LE §4-5 — LA VUE ALIASE, L'ÉCRAN NON
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cote", COTES)
def test_la_vue_est_perimee_par_le_tick_suivant_mais_pas_l_ecran(
        cote, monkeypatch):
    """LA VUE ALIASE `_s_out`, L'ÉCRAN EST MATÉRIALISÉ — §4-4 et §4-5.

    LE VERROU QUE CE TEST REMPLACE, ET POURQUOI. Le plan demandait « les deux
    écrans d'un tick diffèrent, donc la copie a eu lieu ». Ce verrou-là est
    ÉTEINT PAR CONSTRUCTION : `gather_chemin_de_cout` ALLOUE sa sortie
    (`ecran = xp.full(...)`), donc l'écran n'aliase rien et le test passerait
    avec ou sans copie. C'est la famille de verrous que le spec readout a
    trouvés trois fois plus étroits que leur prose, tous par mutation et aucun
    par relecture.

    Ce qui aliase vraiment, c'est la VUE. Les deux moitiés comptent :

      1. `vue.fenetres` du premier tick a CHANGÉ après le second — l'alias est
         RÉEL, et une boucle qui conserverait une vue d'un tick à l'autre
         rendrait un contenu périmé sans le savoir ;
      2. l'ÉCRAN déjà gatherisé du premier tick est INCHANGÉ — l'alias ne
         s'étend pas jusqu'à lui, et c'est ce qui rend la copie défensive
         MORTE : elle ne garderait rien et chargerait une copie de plus dans
         `I` à chaque image interpolée.

    L'identité des tableaux est vérifiée en plus du changement : `is` dit
    l'alias directement, là où « a changé » pourrait aussi s'expliquer par deux
    tampons distincts qui divergent."""
    pyr = _pyramide_doublante()
    applicateur = installer_capture_en_registre(pyr)
    appels = _espionner_melanger(monkeypatch)
    _peindre_s_gradient(pyr)
    boucle = BoucleRendu(pyr, cote, applicateur=applicateur)

    couple_1 = boucle.tick(COTE_PX)
    alpha_1, vue_1 = appels[-1]
    fenetres_1 = {j: np.array(vue_1.fenetres[j], copy=True)
                  for j in pyr.geo.niveaux_gpu}
    ecrans_1 = [np.array(e, copy=True) for e in couple_1]

    couple_2 = boucle.tick(COTE_PX)
    alpha_2, vue_2 = appels[-1]
    assert alpha_1 == alpha_2, "les deux ticks doivent mélanger au même `α`"

    for j in pyr.geo.niveaux_gpu:
        assert vue_1.fenetres[j] is vue_2.fenetres[j], (
            f"niveau {j} : les deux vues ne pointent pas sur le MÊME tableau — "
            "l'alias documenté de `_s_out` n'existe plus, et la garde §4-4 "
            "décrit un mécanisme disparu")
        assert not _memes_octets(vue_1.fenetres[j], fenetres_1[j]), (
            f"niveau {j} : `vue.fenetres` du premier tick n'a PAS changé après "
            "le second — l'alias n'est pas exercé, donc ce verrou ne garde rien")

    for rang, ecran in enumerate(couple_1):
        assert _memes_octets(ecran, ecrans_1[rang]), (
            f"écran {rang} du premier tick modifié par le second tick : le "
            "gather ne matérialise plus sa sortie, et tout le raisonnement du "
            "§4-4 sur la copie morte tombe")

    # Et les deux écrans d'un tick diffèrent bien — vrai, mais ce N'EST PAS le
    # verrou : c'est une conséquence du `s` qui évolue (`_pas_f_doublant`), pas
    # une preuve de copie. Écrit ici pour que personne ne le reprenne comme tel.
    assert not _memes_octets(couple_2[0], couple_2[1])


# ─────────────────────────────────────────────────────────────────────────────
# UN SEUL `melanger` PAR TICK, ET L'EXACT CONTOURNE LE NOYAU (§2-1)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cote, alpha_attendu", [
    (COTE_INTERPOLER, ALPHA_INTERPOLER),
    (COTE_EXTRAPOLER, ALPHA_EXTRAPOLER),
])
def test_un_seul_melanger_par_tick_et_jamais_a_alpha_exact(
        cote, alpha_attendu, monkeypatch):
    """§2-1, RULING 1 — la comptabilité gravée porte `30·I`, pas `60·I`.

    Deux `melanger` par tick feraient payer le noyau à deux images par pas de
    physique : la ligne `P-b` de `claude/prereg-ou-vit-le-rendu-2026-08-04.md`
    deviendrait fausse, et elle est endossée. Le spec readout a tranché que les
    images EXACTES contournent le noyau ; « `α = 1` gratuit » veut dire hors du
    noyau, donc hors de `I`.

    Ce verrou compte les appels — c'est la seule façon de le voir : une boucle
    qui appellerait `melanger(1.0)` rendrait exactement les MÊMES octets à
    l'écran (le noyau affine à `α = 1` rend `s_cur`), donc aucune comparaison
    de pixels ne pourrait la distinguer. Le coût, lui, aurait doublé."""
    pyr = _pyramide_doublante()
    applicateur = installer_capture_en_registre(pyr)
    appels = _espionner_melanger(monkeypatch)
    _peindre_s_gradient(pyr)
    boucle = BoucleRendu(pyr, cote, applicateur=applicateur)

    for indice_tick in range(N_TICKS):
        boucle.tick(COTE_PX)
        assert len(appels) == indice_tick + 1, (
            f"tick {indice_tick} : {len(appels) - indice_tick} appel(s) à "
            "`melanger` — il en faut EXACTEMENT un par tick (§2-1)")
        assert appels[-1][0] == alpha_attendu
    assert all(alpha != ALPHA_EXACT for alpha, _ in appels), (
        "`melanger` a été appelé à `α = 1,0` : l'image exacte doit contourner "
        "le noyau par un gather DIRECT sur la pyramide (§2-1)")


@pytest.mark.parametrize("cote, rang_exact", [
    (COTE_INTERPOLER, 1),
    (COTE_EXTRAPOLER, 0),
])
def test_l_ecran_exact_est_le_gather_direct_sur_la_pyramide(cote, rang_exact):
    """L'écran à `α = 1,0` est, À L'OCTET, le gather direct sur la pyramide.

    Deux choses en une : il est bien produit hors du noyau (§2-1), et il est
    gatherisé au centre fovéal COURANT — celui d'après le `frame()` du tick,
    pas celui d'avant (§2-3 : la latence de VUE est NULLE des deux côtés).
    Comparer au gather refait APRÈS le tick n'a de sens que si le tick n'a rien
    déplacé entre-temps ; c'est précisément ce que le §4-6 exige et ce que le
    verrou du centre fovéal ci-dessous vérifie séparément.

    Le rang de l'exact dans le couple suit du §2-2 : second en interpoler
    (`0,5` puis `1,0`), premier en extrapoler (`1,0` puis `1,5`)."""
    pyr = _pyramide_doublante()
    _peindre_s_gradient(pyr)
    boucle = BoucleRendu(pyr, cote)
    couple = boucle.tick(COTE_PX)
    direct = gather_chemin_de_cout(pyr, COTE_PX)
    assert _memes_octets(couple[rang_exact], direct)
    assert not _memes_octets(couple[1 - rang_exact], direct), (
        "l'écran interpolé est identique à l'exact : soit `s` n'évolue pas "
        "dans ce banc, soit `s_prev` vaut `s_cur` — dans les deux cas le "
        "verrou ne distingue plus rien")


# ─────────────────────────────────────────────────────────────────────────────
# LE CENTRE FOVÉAL — un pas par TICK, pas un par écran (§4-6, §5)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cote", COTES)
def test_le_centre_foveal_avance_d_un_pas_par_tick_et_pas_par_ecran(cote):
    """§4-6 : aucune écriture de `centre_fin` hors de `frame()`, et le tick fait
    `frame(1)` — donc UN déplacement par tick, pour DEUX écrans.

    Déplacer la fovéa ENTRE les deux écrans d'un tick serait SILENCIEUX du côté
    exact — le gather relirait `geo.origines(j, centre_fin)` alors que le
    CONTENU des fenêtres n'a pas été roulé, et rendrait un champ faux sans lever
    — tandis que `ReadoutHorsRegistre` mordrait du côté interpolé. C'est cette
    asymétrie qui rend la garde nécessaire.

    Ce que ce verrou CONSTATE sans le trancher : les deux images d'un tick
    partagent la même position de caméra. C'est le dû nommé du §5 de la spec —
    un pas de rafraîchissement de la caméra par TICK et non par ÉCRAN —
    endossé comme dû SANS ÉCHÉANCE (§8, point 3). Il n'est pas refermé ici, il
    est rendu visible."""
    pyr = _pyramide_doublante()
    _peindre_s_gradient(pyr)
    boucle = BoucleRendu(pyr, cote)
    for _ in range(N_TICKS):
        avant = int(pyr.centre_fin)
        boucle.tick(COTE_PX)
        assert int(pyr.centre_fin) == avant + 1, (
            "le centre fovéal doit avancer d'exactement un pas par tick — "
            "`frame(1)`, et rien d'autre n'écrit `centre_fin` (§4-6)")


# ─────────────────────────────────────────────────────────────────────────────
# LE MONTAGE — installation, applicateur reçu, et ce qui ne se rattrape pas
# ─────────────────────────────────────────────────────────────────────────────

def test_l_applicateur_deja_installe_est_recu_sans_reinstallation():
    """Le montage accepte un applicateur DÉJÀ posé et n'en installe pas un
    second. Le cas est réel : un appelant qui tient déjà la capture en registre
    (un driver, la tâche 2 comparant les deux côtés sur une même pyramide) ne
    doit pas avoir à la démonter pour construire une boucle."""
    pyr = _pyramide_doublante()
    applicateur = installer_capture_en_registre(pyr)
    boucle = BoucleRendu(pyr, COTE_INTERPOLER, applicateur=applicateur)
    assert boucle.applicateur is applicateur
    assert pyr.pas_f is applicateur
    _peindre_s_gradient(pyr)
    boucle.tick(COTE_PX)          # ne doit PAS lever `ReadoutSansCapture`


def test_installer_sur_une_pyramide_deja_equipee_leve_sans_etre_rattrape():
    """LA SERRURE N'EST PAS AVALÉE (§4-8). Détecter « déjà installé » en
    rattrapant `ReadoutCaptureImpossible` pour deviner serait exactement le
    geste que le §4-8 interdit : une serrure qui se rattrape est une promesse.

    Le constructeur n'installe que si aucun applicateur ne lui est donné ; dans
    ce cas la levée de `installer_capture_en_registre` TRAVERSE."""
    pyr = _pyramide_doublante()
    installer_capture_en_registre(pyr)
    with pytest.raises(ReadoutCaptureImpossible):
        BoucleRendu(pyr, COTE_INTERPOLER)


def test_un_applicateur_etranger_a_la_pyramide_est_refuse():
    """Un applicateur installé sur une AUTRE pyramide capturerait `s_prev` d'un
    monde et mélangerait avec le `s_cur` d'un autre : deux états qui n'ont
    jamais coexisté, sans clamp, sans levée, avec `clamps == 0`. La faute est
    STRUCTURELLE et se voit AU MONTAGE — on la refuse là, pas dans un chiffre
    plus tard."""
    pyr_a = _pyramide_doublante()
    pyr_b = _pyramide_doublante()
    applicateur_b = installer_capture_en_registre(pyr_b)
    with pytest.raises(MontageBoucleInvalide):
        BoucleRendu(pyr_a, COTE_INTERPOLER, applicateur=applicateur_b)


def test_un_objet_qui_n_est_pas_un_applicateur_est_refuse():
    """Recevoir le `pas_f` nu — confusion facile, l'attribut porte le même nom —
    donnerait une boucle sans aucun tampon de readout, qui échouerait plus tard
    et ailleurs. Refusé au montage, avec le type sous les yeux."""
    pyr = _pyramide_doublante()
    with pytest.raises(MontageBoucleInvalide):
        BoucleRendu(pyr, COTE_INTERPOLER, applicateur=_pas_f_doublant)


@pytest.mark.parametrize("cote", COTES)
def test_le_trou_de_couverture_du_gather_n_est_pas_rattrape(cote):
    """§4-7 — FAIL-LOUD. Un pixel qu'aucune fenêtre active ne couvre est un fait
    de STRUCTURE : la boucle ne le rattrape pas, ne le remplit pas, ne le compte
    pas pour l'ignorer. Un remplissage par défaut fausserait ce que la mesure
    doit voir, et le silence ferait croire à une couverture qui n'existe pas.

    Le gather lève un `RuntimeError` nu pour ce cas ; la boucle n'écrit ni
    `except RuntimeError` ni `except Exception` (§4-8), donc la levée traverse
    telle quelle.

    LES DEUX CÔTÉS SONT EXERCÉS, ET CE N'EST PAS DE LA SYMÉTRIE DÉCORATIVE : le
    premier écran du couple est l'INTERPOLÉ en interpoler et l'EXACT en
    extrapoler (§2-2), donc les deux appels au gather du tick ne lèvent en
    PREMIER que sous un côté chacun. Testé sur le seul côté `interpoler`, ce
    verrou laissait passer un `except RuntimeError` posé autour du gather de
    l'écran EXACT — mesuré par mutation, pas supposé."""
    pyr = _pyramide_doublante()
    _peindre_s_gradient(pyr)
    boucle = BoucleRendu(pyr, cote)
    with pytest.raises(RuntimeError, match="couvert"):
        boucle.tick(COTE_PX_TROP_GRAND)


def test_la_source_ne_porte_ni_except_large_ni_assert():
    """§4-8 ET §4-9, MÉCANISÉS SUR LA SOURCE — et c'est le seul endroit où ils
    peuvent l'être complètement.

    POURQUOI PAR L'AST PLUTÔT QUE PAR UN RUN. Un rattrapage ne se voit que sur
    le chemin qui lève, et un tick n'en emprunte que deux : un `except
    RuntimeError` posé sur une branche que le test n'atteint pas est INVISIBLE
    à tout verrou de comportement. C'est exactement la mutation qui a survécu à
    la première version de `test_le_trou_de_couverture_du_gather_n_est_pas_
    rattrape`. Lire la source dit ce qui est ÉCRIT, indépendamment des chemins
    qu'un test se trouve exercer — la maison le fait déjà pour l'attribution de
    R1 (`tests/test_arcC_rendu.py::test_empreinte_source_r1`).

    POURQUOI L'AST ET PAS UN `grep`. Un motif textuel confondrait le mot
    « assert » d'une docstring — ce fichier-ci en est plein — avec une
    instruction. L'AST ne voit que du code.

    CE QUI RESTE PERMIS : un rattrapage par CLASSE NOMMÉE, motif écrit à côté.
    Le §4-8 ne l'interdit pas ; il interdit les filets larges, parce que
    `CheminDeCoutInterdit`, `ReadoutInterditDansEtat`, `ReadoutHorsRegistre`,
    `ReadoutSansCapture` et `ReadoutCaptureImpossible` héritent TOUTES de
    `RuntimeError`, et que le gather lève `RuntimeError` pour ses trous — un
    `except RuntimeError` bien intentionné avale les cinq serrures d'un coup.
    C'est la leçon de la mutation du 03/08, trouvée par mutation et non par
    relecture."""
    arbre = ast.parse(Path(boucle_rendu.__file__).read_text(encoding="utf-8"))

    asserts = [n for n in ast.walk(arbre) if isinstance(n, ast.Assert)]
    assert not asserts, (
        f"`assert` trouvé ligne(s) {[n.lineno for n in asserts]} de "
        "`boucle_rendu.py` — il DISPARAÎT sous `python -O`, donc une garde qui "
        "en dépend est absente exactement dans le régime où l'on mesurera "
        "(§4-9, §A43). Une garde s'écrit `if` + levée d'une classe nommée")

    larges = []
    for noeud in ast.walk(arbre):
        if not isinstance(noeud, ast.ExceptHandler):
            continue
        if noeud.type is None:
            larges.append((noeud.lineno, "except nu"))
            continue
        for nom in ast.walk(noeud.type):
            # `ast.Name` attrape `except RuntimeError` ; `ast.Attribute` attrape
            # `except builtins.RuntimeError` et `except ex.Exception`. Un verrou
            # qui existe POUR ce que la relecture ne voit pas ne peut pas se
            # permettre de ne reconnaître qu'une des deux formes d'écriture.
            if isinstance(nom, ast.Name) and nom.id in FILETS_INTERDITS:
                larges.append((noeud.lineno, nom.id))
            elif isinstance(nom, ast.Attribute) and nom.attr in FILETS_INTERDITS:
                larges.append((noeud.lineno, nom.attr))
    assert not larges, (
        f"filet trop large dans `boucle_rendu.py` : {larges}. Les cinq "
        "serrures du readout et du chemin-de-coût héritent de `RuntimeError`, "
        "et le gather lève `RuntimeError` pour ses trous de couverture : un "
        "tel `except` les avale toutes en silence et les garde redeviennent "
        "des promesses (§4-8). Rattraper par CLASSE NOMMÉE, motif à côté")


# ─────────────────────────────────────────────────────────────────────────────
# LES COMPTEURS DE COMPTE RENDU — rendus, non prononcés
# ─────────────────────────────────────────────────────────────────────────────

def test_un_tick_qui_leve_ne_laisse_pas_un_compte_rendu_perime():
    """UN TICK INTERROMPU NE DOIT PAS LAISSER LE CHIFFRE DU TICK PRÉCÉDENT.

    Le cas est réel et il est silencieux. Côté `interpoler`, le couple sort
    `(0,5 ; 1,0)` : si le gather du SECOND écran trouve un trou de couverture,
    `melanger` a DÉJÀ eu lieu et `frame()` a DÉJÀ avancé l'état — mais rien
    n'aurait remis `compte_rendu` à jour. Un appelant qui rattrape par classe
    nommée en amont et le relit lirait alors les compteurs d'un tick qui n'est
    plus, sur un état qui a changé : un chiffre qui ne décrit AUCUN tick, sans
    aucun symptôme. C'est la signature §A53 en miniature, et elle porte sur
    l'attribut dont un prereg futur tirera un seuil.

    La correction est une ligne — `compte_rendu = None` AVANT le premier geste
    qui peut lever — et ce verrou est ce qui la tient."""
    pyr = _pyramide_doublante()
    _peindre_s_gradient(pyr)
    boucle = BoucleRendu(pyr, COTE_INTERPOLER)
    boucle.tick(COTE_PX)
    assert boucle.compte_rendu is not None

    with pytest.raises(RuntimeError, match="couvert"):
        boucle.tick(COTE_PX_TROP_GRAND)
    assert boucle.compte_rendu is None, (
        "après un tick qui a levé, `compte_rendu` porte encore les compteurs "
        "du tick PRÉCÉDENT — un chiffre qui ne décrit aucun tick, rendu sans "
        "symptôme sur un état qui a pourtant avancé")



@pytest.mark.parametrize("cote", COTES)
def test_le_tick_rend_les_compteurs_de_la_vue_sans_prononcer_dessus(
        cote, monkeypatch):
    """Le tick rend `clamps` et `non_finis` de la vue du tick COURANT.

    CE QUE CE VERROU NE DEMANDE PAS, ET C'EST VOULU : aucun seuil, aucun
    avertissement, aucune décision. Le seuil au-delà duquel un comptage
    invalide une mesure appartient au prereg de la lecture d'orientation, pas à
    ce code (§3 de la spec, §5 du spec readout). Ce qui est vérifié est que les
    compteurs REMONTENT jusqu'à l'appelant, à l'octet près de ce que `melanger`
    a compté — un compteur perdu en route serait un silence, et c'est le seul
    défaut que ce code peut avoir sur ce point.

    L'appelant déballe les deux écrans SANS CÉRÉMONIE (`a, b = tick(...)`) : le
    couple reste un couple, les compteurs vivent à côté."""
    pyr = _pyramide_doublante()
    applicateur = installer_capture_en_registre(pyr)
    appels = _espionner_melanger(monkeypatch)
    _peindre_s_gradient(pyr)
    boucle = BoucleRendu(pyr, cote, applicateur=applicateur)

    assert boucle.compte_rendu is None, (
        "avant le premier tick, il n'y a aucun compte rendu à rendre — un zéro "
        "y ressemblerait à un tick propre qui n'a jamais eu lieu")

    for _ in range(N_TICKS):
        premier, second = boucle.tick(COTE_PX)      # déballage sans cérémonie
        _, vue = appels[-1]
        assert boucle.compte_rendu.clamps == vue.clamps
        assert boucle.compte_rendu.non_finis == vue.non_finis
        assert premier.shape == second.shape == (COTE_PX, COTE_PX)
