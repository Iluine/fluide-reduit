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

LA TÂCHE 3 AJOUTE LA SECTION FINALE — LA SORTIE IMAGE ET LE DRIVER
(`scripts/run_boucle_rendu_demo.py`). Elle porte le verrou (f) du plan et les
gardes du driver. Le patron « tester un driver » est celui de la maison :
`tests/test_p2_chiffrage.py:20` fait `from scripts import run_p2_chiffrage`, et
`scripts/` est un package.

⚠ AUCUN PNG DE RÉFÉRENCE N'EST VERSIONNÉ, et ce n'est pas une facilité :
`.gitignore:5` porte `outputs/`, donc un golden-fichier vivrait hors du dépôt ou
forcerait un binaire dedans. Le verrou (f) est « DEUX RUNS, MÊMES OCTETS », entre
deux constructions COMPLÈTES et INDÉPENDANTES — jamais deux lectures du même
appel, qui seraient vertes et vides.
"""
from __future__ import annotations

import ast
import inspect
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from scripts import run_boucle_rendu_demo
from scripts.run_boucle_rendu_demo import (
    COTES_DISPONIBLES,
    COTE_PX_DEMO,
    NIVEAUX_GRIS,
    OCTET_FILTRE_AUCUN,
    SIGNATURE_PNG,
    EcranNonRepresentable,
    LongueurDeDemoInvalide,
    construire_parseur,
    encoder_png_gris,
    quantifier_en_octets,
    rendre,
)
from src.f1_gpu import boucle_rendu
from src.f1_gpu.boucle_rendu import (
    ALPHA_EXACT,
    COTE_EXTRAPOLER,
    COTE_INTERPOLER,
    COUPLES_PAR_COTE,
    INDICE_CHAMP_VUE,
    BoucleRendu,
    CoteInconnu,
    MontageBoucleInvalide,
)
from src.f1_gpu.chemin_de_cout import albedo_ecran, gather_chemin_de_cout
from src.f1_gpu.interpolation_readout import (
    ALPHA_EXTRAPOLER,
    ALPHA_INTERPOLER,
    INDICE_CHAMP_S,
    ReadoutCaptureImpossible,
    TamponReadout,
    VueInterpolee,
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


def _asserts_et_filets_larges(arbre: ast.AST) -> tuple[list[int], list[tuple]]:
    """Les `assert` et les `except` TROP LARGES d'un arbre, par l'AST.

    UN SEUL EXEMPLAIRE, ET C'EST LE POINT. Ce balayage sert DEUX verrous — la
    source de `boucle_rendu.py` et celle du driver — et il a d'abord été écrit
    DEUX FOIS, logique pour logique, seuls les messages changeant. Le jour où
    une troisième forme d'écriture d'`except` devra être reconnue, deux copies
    divergeraient sans que rien ne le signale : c'est exactement le motif par
    lequel l'en-tête de ce fichier refuse de recopier les helpers du readout.

    `ast.Name` attrape `except RuntimeError` ; `ast.Attribute` attrape
    `except builtins.RuntimeError` et `except ex.Exception`. Un verrou qui
    existe POUR ce que la relecture ne voit pas ne peut pas se permettre de ne
    reconnaître qu'une des deux formes. `noeud.type is None` est l'`except` nu.

    Rend `(lignes des asserts, [(ligne, nom du filet)])` — les MESSAGES
    appartiennent aux verrous, qui n'ont pas le même motif à écrire."""
    asserts = [n.lineno for n in ast.walk(arbre) if isinstance(n, ast.Assert)]

    larges: list[tuple] = []
    for noeud in ast.walk(arbre):
        if not isinstance(noeud, ast.ExceptHandler):
            continue
        if noeud.type is None:
            larges.append((noeud.lineno, "except nu"))
            continue
        for nom in ast.walk(noeud.type):
            if isinstance(nom, ast.Name) and nom.id in FILETS_INTERDITS:
                larges.append((noeud.lineno, nom.id))
            elif isinstance(nom, ast.Attribute) and nom.attr in FILETS_INTERDITS:
                larges.append((noeud.lineno, nom.attr))
    return asserts, larges


def _ecritures_visant(arbre: ast.AST, attributs: frozenset) -> list[tuple]:
    """Les AFFECTATIONS dont la cible touche l'un des `attributs` nommés.

    Rend `[(ligne, nom d'attribut)]`. Trois formes d'affectation sont
    parcourues — `Assign`, `AugAssign`, `AnnAssign` —, et pour chacune la
    CIBLE est balayée EN ENTIER, sans regarder le `ctx` du nœud `Attribute`.
    C'est nécessaire et ce n'est pas de la prudence décorative : dans
    `x._s_out[i] = v`, le `Subscript` porte `ctx=Store` mais l'`Attribute`
    `_s_out` porte `ctx=Load`. Un contrôle qui ne retiendrait que les
    `Attribute` en `Store` verrait `x._s_out = v` et manquerait la forme
    INDEXÉE — celle qu'un vrai bug écrirait.

    CE QUE CE CONTRÔLE NE VOIT PAS, dit ici plutôt que sous-entendu : une
    mutation passant par une MÉTHODE (`x._s_out.fill(0)`, `np.copyto(...)`,
    `x._s_out[:] = ...` via un alias local pris plus haut sous un autre nom).
    Il tient la forme d'écriture directe, qui est celle que le §4-2 nomme ;
    il n'est pas un contrôle de mutabilité général."""
    trouves: list[tuple] = []
    for noeud in ast.walk(arbre):
        if isinstance(noeud, ast.Assign):
            cibles = noeud.targets
        elif isinstance(noeud, (ast.AugAssign, ast.AnnAssign)):
            cibles = [noeud.target]
        else:
            continue
        for cible in cibles:
            for interne in ast.walk(cible):
                if (isinstance(interne, ast.Attribute)
                        and interne.attr in attributs):
                    trouves.append((noeud.lineno, interne.attr))
    return trouves


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

    D'OÙ LA PORTÉE DE LA COMPARAISON : elle est celle de `_etat_en_octets`, dont
    la docstring l'énumère et la motive morceau par morceau. **Elle n'est PAS
    réénumérée ici** — deux énumérations du même contrat dérivent l'une de
    l'autre sans que rien ne le signale, et c'est déjà arrivé à celle-ci : elle a
    dit « toutes les fenêtres, plus `references` et `centre_fin` » pendant tout
    le temps où la photo en couvrait déjà cinq morceaux, `monde0` compris. Une
    prose plus étroite que la portée est le sens sûr, mais c'est une affirmation
    FAUSSE dans le verrou même qui mécanise l'exemption du §4-3.

    Ce qui se dit ici et nulle part ailleurs : la comparaison porte sur les
    OCTETS, la garantie porte sur l'état et non sur un échantillon de l'état, et
    un `allclose` laisserait passer exactement le genre de fuite qu'on cherche."""
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


def test_la_source_du_module_ne_porte_ni_filet_large_ni_assert_ni_horloge():
    """§4-8, §4-9 ET §1, MÉCANISÉS SUR LA SOURCE — et c'est le seul endroit où
    ils peuvent l'être complètement.

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
    relecture.

    ET LE §1 — AUCUNE HORLOGE MURALE. L'en-tête du module PROMET « Aucune
    horloge murale n'entre ici — ni `time`, ni `cuda.Event`, ni attente, ni
    régulation », et la spec le grave au §1 : la cadence est LOGIQUE, « 2:1 »
    est un rapport de COMPTE. Cette promesse n'était tenue par RIEN. La
    machinerie existait pourtant dans ce fichier même, appliquée au SEUL
    driver — alors que c'est le MODULE que le §1 vise en premier, et lui qui
    portera un jour le chemin GPU, où `cuda.Event` est la tentation réelle. Une
    garde promise dans un document est une garde absente (§4-5) : c'en était
    une, à trois lignes de son propre outillage.

    ET PAS D'INTERDICTION DE `cupy` ICI, contrairement au verrou du driver.
    Elle serait FAUSSE pour ce module : la démo est CPU numpy, mais la boucle
    doit pouvoir tourner sous cupy — elle ne connaît aucun backend, elle
    transmet celui de la pyramide. Recopier l'assertion du driver aurait
    interdit au module ce pour quoi il est écrit."""
    arbre = ast.parse(Path(boucle_rendu.__file__).read_text(encoding="utf-8"))
    asserts, larges = _asserts_et_filets_larges(arbre)

    assert not asserts, (
        f"`assert` trouvé ligne(s) {asserts} de "
        "`boucle_rendu.py` — il DISPARAÎT sous `python -O`, donc une garde qui "
        "en dépend est absente exactement dans le régime où l'on mesurera "
        "(§4-9, §A43). Une garde s'écrit `if` + levée d'une classe nommée")

    assert not larges, (
        f"filet trop large dans `boucle_rendu.py` : {larges}. Les cinq "
        "serrures du readout et du chemin-de-coût héritent de `RuntimeError`, "
        "et le gather lève `RuntimeError` pour ses trous de couverture : un "
        "tel `except` les avale toutes en silence et les garde redeviennent "
        "des promesses (§4-8). Rattraper par CLASSE NOMMÉE, motif à côté")

    identifiants = _identifiants(Path(boucle_rendu.__file__))
    horloges = sorted((identifiants & MODULES_HORLOGE)
                      | (identifiants & ATTRIBUTS_HORLOGE))
    assert not horloges, (
        f"horloge dans `boucle_rendu.py` : {horloges}. L'en-tête du module "
        "promet « Aucune horloge murale n'entre ici — ni `time`, ni "
        "`cuda.Event`, ni attente, ni régulation », et le §1 de la spec le "
        "grave : la cadence est LOGIQUE, « 2:1 » est un rapport de COMPTE et "
        "non deux écrans par 33,3 ms. §A61 en interdit l'usage tant que "
        "l'instrument n'est pas qualifié")


def test_la_boucle_n_ecrit_jamais_dans_le_tampon_de_readout():
    """§4-2, MÉCANISÉ — LE SEUL §4-x QUE LA SPEC FASSE GARDE ET QUE RIEN NE
    TENAIT.

    `claude/spec-boucle-rendu-2026-08-24.md:167-168` : « La boucle n'écrit
    **jamais** elle-même dans le tampon de readout — c'est `frame()`, par
    l'applicateur, qui le fait. La boucle LIT. »

    POURQUOI AUCUN VERROU DE COMPORTEMENT NE PEUT LE TENIR, et c'est ce qui
    rend le contrôle de SOURCE nécessaire ici comme il l'est pour les filets
    larges. Une écriture dans `applicateur.tampon._s_prev` AVANT le `melanger`
    du tick est invisible :
      - au verrou (a) — `_etat_en_octets` photographie `centre_fin`,
        `fenetres`, `references`, `monde0` et les compteurs de transfert ; le
        TAMPON DE READOUT n'y est pas, et il n'a rien à y faire : ce n'est pas
        de l'état de physique ;
      - au verrou (e) — son témoin `_temoin_du_noyau` relit `etat_precedent`
        depuis LE MÊME tampon, donc depuis la corruption elle-même, et
        concorde avec elle. C'est le patron « comparer deux fois le résultat du
        même appel », par un chemin plus long.
    Le risque réel est nul aujourd'hui — la boucle ne nomme jamais ces
    attributs — mais §4-5 est explicite : une garde promise dans un document
    est une garde absente, et celle-ci était promise.

    LA PORTÉE DU CONTRÔLE EST CELLE DE `_ecritures_visant`, dont la docstring
    dit ce qu'il ne voit pas : une mutation passant par une méthode
    (`.fill()`, `np.copyto`) lui échappe. Il tient la forme d'écriture DIRECTE,
    qui est celle que le §4-2 nomme."""
    ecritures = _ecritures_visant(
        ast.parse(Path(boucle_rendu.__file__).read_text(encoding="utf-8")),
        ATTRIBUTS_TAMPON_READOUT)
    assert not ecritures, (
        f"écriture visant le tampon de readout dans `boucle_rendu.py` : "
        f"{ecritures}. Le §4-2 grave que la boucle LIT et n'écrit JAMAIS "
        "elle-même dans ce tampon — c'est `frame()` qui le fait, par "
        "l'applicateur. Ni le verrou (a) ni le verrou (e) ne verraient cette "
        "écriture : le tampon n'est pas dans la photo d'état, et le témoin de "
        "(e) relit le tampon corrompu et concorde avec lui")


# ─────────────────────────────────────────────────────────────────────────────
# LES COMPTEURS DE COMPTE RENDU — rendus, non prononcés
# ─────────────────────────────────────────────────────────────────────────────

def test_un_tick_qui_leve_ne_laisse_pas_un_compte_rendu_perime():
    """UN TICK INTERROMPU NE DOIT PAS LAISSER LE CHIFFRE DU TICK PRÉCÉDENT.

    Le cas est réel et il est silencieux. Côté `interpoler`, le couple sort
    `(0,5 ; 1,0)` : si le gather d'un écran trouve un trou de couverture,
    `melanger` a DÉJÀ eu lieu et `frame()` a DÉJÀ avancé l'état — mais rien
    n'aurait remis `compte_rendu` à jour.

    SANS ORDINAL, ET C'EST UNE CORRECTION. Cette phrase a dit « le gather du
    SECOND écran », ce qui décrit un scénario INATTEIGNABLE : les deux gathers
    d'un tick partagent `geo`, `cote_px` et le MÊME `centre_fin` — rien ne
    l'écrit entre eux (§4-6) — donc leur prédicat de couverture est identique et
    le PREMIER échoue toujours en premier. La substance tenait, l'ordinal était
    faux ; c'est le motif « prose plus large que la portée » qui se réintroduit
    en miniature dans le correctif qui le poursuivait. Un appelant qui rattrape par classe
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


# ─────────────────────────────────────────────────────────────────────────────
# VERROU (d) — LE CÔTÉ NE TOUCHE PAS LA PHYSIQUE
# ─────────────────────────────────────────────────────────────────────────────

def test_le_cote_ne_touche_pas_la_physique():
    """VERROU (d) — `N` ticks côté INTERPOLER et `N` ticks côté EXTRAPOLER
    depuis le même état initial : les états physiques sont BIT-IDENTIQUES, et
    seuls les écrans diffèrent.

    TROIS MOITIÉS, ET LA DEUXIÈME EST LOAD-BEARING SANS ÊTRE DANS LE PLAN :

      1. LES ÉTATS SONT BIT-IDENTIQUES. Le côté ne choisit QUE la paire d'`α`
         (§2-2) : il n'a aucune prise sur `frame()`, donc aucune sur la
         physique. Si le côté touchait l'état, toute comparaison entre côtés —
         celle que la lecture d'orientation fera un jour — comparerait deux
         MONDES et non deux lectures du même monde, et son verdict porterait sur
         un écart que personne n'aurait voulu. La portée de la comparaison est
         celle de `_etat_en_octets` ; elle n'est PAS réénumérée ici, deux
         énumérations du même contrat dérivant l'une de l'autre sans signal.

      2. L'ÉCRAN `α = 1,0` EST COMMUN AUX DEUX CÔTÉS — `interpoler.
         ecran_alpha_grand` et `extrapoler.ecran_alpha_petit`, à l'octet, à
         CHAQUE tick. Le plan ne la demande pas. CE QU'ELLE MÉCANISE EST
         L'UNIFORMITÉ GRAVÉE DU §2-2 — « exactement un des deux écrans vaut
         `α = 1,0`, et le couple sort en `α` croissant » — ENDOSSÉE AU §8
         POINT 1. Que cet écran soit NUMÉRIQUEMENT LE MÊME des deux côtés en est
         la conséquence directe : même état `n`, même gather DIRECT sur la
         pyramide (§2-1), même `centre_fin`. Une uniformité affirmée dans un
         document et vérifiée nulle part est une garde PROMISE, donc une garde
         ABSENTE (`PREREGISTRATION.md:10152`).

         CE QUI NE DÉPEND PAS DE CETTE MOITIÉ — ET LA CITATION À NE PAS FAIRE.
         Le §2-3 épingle la mesure du §10 du spec readout comme l'ÉCART PAR
         IMAGE ENTRE LES DEUX CÔTÉS (`n − ½` contre `n + ½`), et cet écart
         repose sur les PAIRES d'`α` du §2-2, PAS sur la communauté de l'écran
         exact ; c'est cet épinglage-là que le §8 point 2 prononce. Ce qui
         s'appuie sur la communauté est la TROISIÈME grandeur du §2-3 — au
         niveau du flux 2:1, l'écart moyen entre les deux couples tombe à une
         demi-période — que la spec déclare NON ÉPINGLÉE, dérivation de session,
         et EMPLOYÉE NULLE PART dans le document. Si cette moitié tombait, c'est
         ELLE qui tomberait, pas l'épinglage. Écrit parce qu'une première
         rédaction de ce verrou a cité le §2-3 pour l'épinglage : la citation
         était authentique et prise dans le MAUVAIS paragraphe. Une lecture
         écartée sans être nommée revient par la porte de derrière — c'est le
         mot du §2-3 lui-même, appliqué ici à sa propre citation.

      3. LES ÉCRANS NON EXACTS DIFFÈRENT — `α = 0,5` contre `α = 1,5`. Sans
         cette moitié, un côté qui n'aurait AUCUN effet passerait les deux
         premières : les deux couples seraient identiques terme à terme et le
         verrou serait vert en ne gardant rien.

    POURQUOI `_pyramide_doublante` ET NON LA FABRIQUE JETABLE, ET C'EST MESURÉ.
    Sous le `F` jetable, `s` N'ÉVOLUE PAS : `s_prev == s_cur`, tout mélange
    affine rend `s_cur` quel que soit `α`, et la moitié 3 devient VERTE ET VIDE
    — constaté sur les trois ticks de ce banc avant d'écrire ce verrou, pas
    supposé. Le pas doublant (`s ← 2·s`) fait évoluer `s`, et le gradient non
    uniforme de `_peindre_s_gradient` rend le gather sensible à `centre_fin`."""
    pyr_interpoler = _pyramide_doublante()
    pyr_extrapoler = _pyramide_doublante()
    _peindre_s_gradient(pyr_interpoler)
    _peindre_s_gradient(pyr_extrapoler)
    depart = _etat_en_octets(pyr_interpoler)

    boucle_interpoler = BoucleRendu(pyr_interpoler, COTE_INTERPOLER)
    boucle_extrapoler = BoucleRendu(pyr_extrapoler, COTE_EXTRAPOLER)

    for indice_tick in range(N_TICKS):
        couple_interpoler = boucle_interpoler.tick(COTE_PX)
        couple_extrapoler = boucle_extrapoler.tick(COTE_PX)

        # MOITIÉ 2 — l'exact est le SECOND en interpoler (`0,5` puis `1,0`) et
        # le PREMIER en extrapoler (`1,0` puis `1,5`) : c'est l'uniformité du
        # §2-2, endossée au §8 point 1, qui rend cet écran commun.
        assert _memes_octets(couple_interpoler.ecran_alpha_grand,
                             couple_extrapoler.ecran_alpha_petit), (
            f"tick {indice_tick} : l'écran `α = 1,0` N'EST PAS commun aux deux "
            "côtés. Ce qui cesse de tenir est l'UNIFORMITÉ GRAVÉE DU §2-2 — "
            "exactement un écran à `α = 1,0` dans chaque couple, même état `n`, "
            "même gather DIRECT sur la pyramide, même `centre_fin` — endossée "
            "au §8 point 1. NE PAS lire cet échec comme la chute de l'épinglage "
            "du §10 : celui-ci est l'ÉCART PAR IMAGE entre les deux côtés et "
            "repose sur les PAIRES d'`α`, pas sur cette communauté. Ce qui "
            "dépend d'elle est la TROISIÈME grandeur du §2-3, que la spec "
            "déclare NON épinglée, dérivation de session, employée nulle part")

        # MOITIÉ 3 — sans elle, un côté sans aucun effet passerait tout le reste.
        assert not _memes_octets(couple_interpoler.ecran_alpha_petit,
                                 couple_extrapoler.ecran_alpha_grand), (
            f"tick {indice_tick} : les écrans `α = 0,5` et `α = 1,5` sont "
            "IDENTIQUES à l'octet. Soit le côté n'a plus aucun effet sur la "
            "paire d'`α`, soit `s` n'évolue pas dans ce banc — dans les deux "
            "cas les moitiés 1 et 2 seraient vertes en ne gardant rien")

    # Clôture SYMÉTRIQUE du bilan de transfert, une fois de chaque côté après la
    # même quantité de travail : `frame()` n'en clôt aucun, et une clôture
    # asymétrique ferait diverger le NOMBRE de bilans — le verrou échouerait
    # alors pour une raison qui n'est pas la sienne.
    pyr_interpoler.transferts.frame_suivante()
    pyr_extrapoler.transferts.frame_suivante()

    apres = _etat_en_octets(pyr_interpoler)
    cles_fenetres = [cle for cle in depart
                     if isinstance(cle, tuple) and cle[0] == "fenetres"]
    assert any(apres[cle] != depart[cle] for cle in cles_fenetres), (
        "AUCUNE fenêtre n'a bougé sur les N ticks : la comparaison d'états qui "
        "suit serait vraie sans rien garder. Le delta est exigé sur les "
        "FENÊTRES et non sur la photo entière — `centre_fin` avance de "
        "`N_TICKS` par construction du tick, donc un garde-fou posé sur le "
        "dictionnaire complet serait lui-même VIDE (§A62-bis-2)")

    # MOITIÉ 1 — égalité EXACTE, jamais un `allclose` : une divergence de
    # dernier bit signale déjà que le côté a atteint un chemin de calcul de
    # l'état, et c'est exactement ce qu'on cherche.
    etat_extrapoler = _etat_en_octets(pyr_extrapoler)
    assert set(apres) == set(etat_extrapoler)
    for cle, valeur in etat_extrapoler.items():
        assert apres[cle] == valeur, (
            f"{cle} diffère entre le run INTERPOLER et le run EXTRAPOLER : le "
            "côté a touché la PHYSIQUE. Il ne doit choisir QUE la paire d'`α` "
            "(§2-2) ; sinon comparer deux côtés reviendrait à comparer deux "
            "mondes, et le verdict de la lecture d'orientation porterait sur un "
            "écart que personne n'a voulu")


# ─────────────────────────────────────────────────────────────────────────────
# VERROU (e) — CONFORMITÉ AU NOYAU DU §2, À L'OCTET
# ─────────────────────────────────────────────────────────────────────────────

def _temoin_du_noyau(pyr, applicateur, alpha: float, centre_fin: int):
    """Le noyau du §2 recalculé À LA MAIN, sans passer par le module testé, et
    présenté au gather sous la surface qu'il attend.

    `s_prev` est lu dans le tampon de readout (`etat_precedent`, post-roll et en
    REGISTRE), `s_cur` dans les fenêtres de la pyramide. Le mélange est écrit
    dans un tableau NEUF — jamais dans `_s_out`, qui appartient au tampon —
    puis emballé dans une `VueInterpolee` dont l'axe des champs est un
    SINGLETON, d'où `indice_champ=INDICE_CHAMP_VUE` chez l'appelant.

    L'ORDRE ET LE TYPAGE DE L'ARITHMÉTIQUE SONT CEUX DE `melanger`, À LA LETTRE :
    `a = float(alpha)` puis `(1.0 - a) * s_prev + a * s_cur`. En numpy ≥ 2 un
    scalaire Python NE PROMEUT PAS un tableau f32 — le résultat reste f32 —
    mais une RÉASSOCIATION (`s_prev + a * (s_cur - s_prev)`, algébriquement
    identique) changerait les octets. Un témoin qui ne reproduit pas l'ordre ne
    verrouille rien : il mesure sa propre arithmétique.

    Les compteurs de la vue témoin valent zéro et ne servent à rien : le gather
    ne les lit pas. Ce qui compte, c'est que le CENTRE FOVÉAL passé ici soit
    celui du tick — il est LU par l'appelant et vérifié là-bas."""
    a = float(alpha)
    fenetres_temoin = {}
    for j in pyr.geo.niveaux_gpu:
        s_prev = applicateur.tampon.etat_precedent(j)
        s_cur = pyr.fenetres[j][:, :, INDICE_CHAMP_S, :, :]
        # ALLOCATION PAR `pyr.xp`, PAS PAR `np` : la `VueInterpolee` rendue
        # ci-dessous DÉCLARE `pyr.xp`, et un témoin qui allouerait en numpy
        # pour une vue déclarée cupy se contredirait sur la seule ligne de ce
        # fichier qui touche au backend.
        bloc = pyr.xp.empty((s_cur.shape[0], s_cur.shape[1], 1,
                             s_cur.shape[2], s_cur.shape[3]), dtype=s_cur.dtype)
        bloc[:, :, 0, :, :] = (1.0 - a) * s_prev + a * s_cur
        fenetres_temoin[j] = bloc
    return VueInterpolee(pyr.xp, pyr.geo, centre_fin, fenetres_temoin, 0, 0)


@pytest.mark.parametrize("cote", COTES)
def test_chaque_ecran_du_couple_vaut_le_noyau_du_paragraphe_2(cote, monkeypatch):
    """VERROU (e) — sur le pas jouet DOUBLANT, CHACUN des deux écrans du couple
    vaut À L'OCTET `(1−α)·s_prev + α·s_cur` gatherisé. Aucun autre noyau n'est
    entré en douce.

    LE DOUBLANT EST EXACT EN f32 (`s ← 2·s` : la mantisse ne bouge pas, seul
    l'exposant), donc l'égalité octet pour octet est une exigence tenable et non
    une tolérance déguisée — c'est le doublement du 13ᵉ verrou du readout.

    LES DEUX ÉCRANS, PAS SEULEMENT L'INTERPOLÉ, ET POUR L'EXACT CE N'EST PAS UNE
    FORMALITÉ. L'écran `α = 1,0` vient d'un gather DIRECT sur la pyramide, hors
    du noyau (§2-1) ; le vérifier contre `(1−1)·s_prev + 1·s_cur = s_cur` prouve
    que ce CONTOURNEMENT est NUMÉRIQUEMENT le noyau à `α = 1`. C'est la seule
    chose qui interdise qu'un autre chemin soit entré en douce du côté exact —
    un champ voisin, un système voisin, un centre décalé rendraient tous des
    pixels plausibles sans lever.

    CE QUE CE VERROU NE GARDE PAS, ET C'EST DÉLIBÉRÉ. Il ne peut pas distinguer
    le gather direct d'un `melanger(1.0)` suivi d'un gather sur la vue : le
    noyau affine à `α = 1` rend `s_cur`, donc les OCTETS sont les mêmes. C'est
    précisément le point du §2-1 — le contournement est une affaire de COÛT
    (`30·I` et non `60·I`), pas de pixels — et ce qui le garde est le COMPTE des
    appels, dans `test_un_seul_melanger_par_tick_et_jamais_a_alpha_exact`. Les
    deux verrous sont complémentaires ; écrit ici pour qu'aucune revue ne croie
    que l'un rend l'autre inutile.

    LE CLAMP NE DOIT PAS MORDRE, ET C'EST VÉRIFIÉ PLUTÔT QUE SUPPOSÉ. `melanger`
    applique `s ≥ 0` APRÈS le mélange et le COMPTE ; le témoin, lui, ne clampe
    pas. Avec `s ≥ 0` et le doublant, `α = 1,5` donne `2,5·s ≥ 0` et `α = 0,5`
    donne `1,5·s ≥ 0` : le clamp est inerte. Si un jour il mordait, le témoin et
    le module divergeraient pour une raison INVISIBLE dans le message d'échec —
    d'où l'assertion sur `clamps`, qui nomme la cause avant qu'on la cherche.

    LE CENTRE FOVÉAL EST LU, PAS SUPPOSÉ. Le gather témoin doit tourner au MÊME
    `centre_fin` que le tick. Le §4-6 l'assure — rien n'écrit `centre_fin` hors
    de `frame()` — mais une garde qu'on suppose est une garde qu'on n'a pas :
    le centre mémorisé par la vue du tick est comparé à celui lu après le tick,
    et c'est ce dernier qui est passé au témoin."""
    pyr = _pyramide_doublante()
    applicateur = installer_capture_en_registre(pyr)
    appels = _espionner_melanger(monkeypatch)
    _peindre_s_gradient(pyr)
    boucle = BoucleRendu(pyr, cote, applicateur=applicateur)

    for indice_tick in range(N_TICKS):
        couple = boucle.tick(COTE_PX)

        assert boucle.compte_rendu.clamps == 0, (
            f"tick {indice_tick} : le clamp `s ≥ 0` a mordu "
            f"({boucle.compte_rendu.clamps} cellule(s)). Le témoin de ce verrou "
            "ne clampe pas : il divergerait du module pour une raison qui "
            "n'apparaîtrait nulle part dans l'échec. Choisir des valeurs où le "
            "clamp est inerte fait partie du banc")
        assert boucle.compte_rendu.non_finis == 0, (
            f"tick {indice_tick} : {boucle.compte_rendu.non_finis} cellule(s) "
            "non finie(s) — l'état est corrompu en amont et l'égalité à "
            "l'octet ne dirait plus rien du noyau")

        # LE CENTRE FOVÉAL, LU. La vue du tick a mémorisé `pyramide.centre_fin`
        # au moment du `melanger` ; s'il valait encore celui d'après le tick,
        # alors rien ne l'a écrit entre les deux et le témoin peut gatheriser
        # au même centre. C'est la garde du §4-6 constatée ici, pas supposée.
        centre_du_tick = int(pyr.centre_fin)
        _, vue_du_tick = appels[-1]
        assert int(vue_du_tick.centre_fin) == centre_du_tick, (
            f"tick {indice_tick} : le centre fovéal a bougé entre le `melanger` "
            f"du tick ({int(vue_du_tick.centre_fin)}) et la lecture d'après "
            f"tick ({centre_du_tick}). Le §4-6 l'interdit, et le témoin de ce "
            "verrou ne pourrait plus gatheriser au centre du tick")

        for alpha, ecran in zip(boucle.couple_alphas, couple):
            temoin = _temoin_du_noyau(pyr, applicateur, alpha, centre_du_tick)
            attendu = gather_chemin_de_cout(temoin, COTE_PX,
                                            indice_champ=INDICE_CHAMP_VUE)
            assert _memes_octets(ecran, attendu), (
                f"tick {indice_tick}, `α = {alpha}` (côté {cote}) : l'écran "
                "rendu N'EST PAS, à l'octet, `(1−α)·s_prev + α·s_cur` "
                "gatherisé. Un autre noyau est entré en douce — ou, pour "
                f"`α = {ALPHA_EXACT}`, le gather direct du §2-1 n'est plus "
                "numériquement le noyau à `α = 1`")


# ─────────────────────────────────────────────────────────────────────────────
# TÂCHE 3 — LA SORTIE IMAGE ET LE DRIVER (`scripts/run_boucle_rendu_demo.py`)
# ─────────────────────────────────────────────────────────────────────────────

# `s_half` DE BANC, ET IL FAUT DIRE POURQUOI IL EST ÉCRIT ICI. Le §6 de la spec
# range `s_half` parmi ce qu'elle NE TRANCHE PAS, et le corpus n'en porte AUCUNE
# valeur (`src/albedo.py:27` et `src/f1_gpu/chemin_de_cout.py:344` en font tous
# deux un paramètre d'appelant). Cette valeur n'endosse donc rien : c'est un
# choix de BANC, pris pour que la quantification 8 bits ne sature ni en bas ni
# en haut sur le substrat jetable, et aucun verrou de ce fichier ne prononce sur
# elle. Le driver, lui, l'EXIGE de son appelant — il n'en porte aucun défaut.
S_HALF_BANC = 0.05

# Deux ticks suffisent au verrou (f) : ce qu'il compare est l'octet, pas la
# durée. Un run plus long ne rendrait pas l'égalité plus vraie, et le banc doit
# rester minuscule (§A61 : aucune mesure ici, donc rien à amortir).
N_TICKS_GOLDEN = 2

# Ce que le PNG du driver a le droit de contenir, et rien d'autre. `tIME` porte
# une date par définition ; `tEXt` peut en porter une. Écrire un PNG n'est pas
# une mesure — l'horodater en serait une, et §A61 tient.
CHUNKS_AUTORISES = ("IHDR", "IDAT", "IEND")

# LA VALEUR DU FORMAT PNG, GRAVÉE ICI ET NON IMPORTÉE DU DRIVER. Le filtre de
# ligne « None » vaut 0 dans la spécification PNG : c'est un fait du FORMAT, pas
# un choix de ce dépôt. L'importer du driver rendrait le verrou vide — une
# mutation de la constante mutant du même coup l'attente, et le test comparerait
# le module à lui-même. C'est la leçon de la tâche 2, appliquée : un verrou qui
# compare deux fois le résultat du même appel est vert et vide. Constaté par
# mutation sur ce fichier même, pas supposé.
FILTRE_AUCUN_PNG = 0

# CE QUE CES DEUX JEUX TIENNENT, ET CE QU'ILS NE TIENNENT PAS — la prose
# d'origine disait « ce dont l'absence rend le driver INCAPABLE DE CHRONOMÉTRER
# QUOI QUE CE SOIT », et c'était FAUX. Le driver importe `TransfertComptable`,
# l'instancie dans `construire_pyramide`, et chaque `frame()` de la démo passe
# par `_chronometrer`, qui appelle `time.perf_counter`
# (`src/f1_gpu/transferts.py:66-68`). UN CHRONOMÈTRE TOURNE PENDANT LA DÉMO.
#
# Ce qui est tenu, et c'est tout : ni le driver ni `boucle_rendu.py` ne LISENT
# eux-mêmes une horloge. Ce qui garde §A61 est ailleurs et il faut le dire —
# AUCUN CHIFFRE N'EN SORT : rien ne lit, ne compare, n'imprime ni ne retourne
# `h2d_ms` / `d2h_ms`, et le seul `print` du driver est verrouillé par égalité
# EXACTE. Le fait est NOMMÉ plutôt que nié : une prose qui nie un chronomètre
# qui tourne réellement est plus dangereuse que le chronomètre.
#
# `Event` couvre le `cuda.Event` que le §1 de la spec nomme explicitement, et
# `sleep` / `synchronize` l'attente et la régulation qu'il nomme aussi : ces
# trois-là manquaient, et c'était un trou de couverture pour les DEUX sources.
MODULES_HORLOGE = frozenset({"time", "datetime", "timeit", "calendar"})
ATTRIBUTS_HORLOGE = frozenset({
    "perf_counter", "perf_counter_ns", "monotonic", "monotonic_ns", "time_ns",
    "process_time", "now", "utcnow", "today",
    "Event", "sleep", "synchronize"})

# §4-2 — LE TAMPON DE READOUT NE S'ÉCRIT PAS DEPUIS LA BOUCLE. Les deux noms
# privés du `TamponReadout` : `_s_prev` est ce que `capturer` remplit,
# `_s_out` ce que `melanger` produit et que `VueInterpolee.fenetres` ALIASE.
ATTRIBUTS_TAMPON_READOUT = frozenset({"_s_prev", "_s_out"})


def _chunks_png(octets: bytes) -> list[tuple[str, bytes]]:
    """Découpe un PNG en `(type, données)`, CRC VÉRIFIÉ, sans aucune
    bibliothèque d'image.

    POURQUOI À LA MAIN PLUTÔT QU'AVEC `pillow`. `pillow` figure en ligne
    COMMENTÉE dans `requirements.txt` : il se trouve installé dans le venv, mais
    l'importer — fût-ce dans un test — ferait dépendre le verrou d'une
    dépendance NON DÉCLARÉE, exactement ce que la décision D7 de la tâche
    refuse pour le driver. Un lecteur qui ne lit que la stdlib prouve la même
    chose et ne doit rien à personne.

    Le CRC est vérifié parce qu'un PNG dont le CRC est faux n'est pas un PNG :
    sans ce contrôle, « le fichier s'ouvre » resterait une affirmation."""
    if not octets.startswith(SIGNATURE_PNG):
        raise ValueError("signature PNG absente — ce n'est pas un PNG.")
    chunks: list[tuple[str, bytes]] = []
    position = len(SIGNATURE_PNG)
    while position < len(octets):
        (longueur,) = struct.unpack(">I", octets[position:position + 4])
        type_ = octets[position + 4:position + 8]
        donnees = octets[position + 8:position + 8 + longueur]
        (crc,) = struct.unpack(
            ">I", octets[position + 8 + longueur:position + 12 + longueur])
        if crc != zlib.crc32(type_ + donnees) & 0xFFFFFFFF:
            raise ValueError(f"CRC faux sur le chunk {type_!r}.")
        chunks.append((type_.decode("ascii"), donnees))
        position += 12 + longueur
    return chunks


def _identifiants(source: Path) -> set[str]:
    """Tous les noms et attributs qui apparaissent dans le CODE d'un fichier.

    PAR L'AST, PAS PAR UN `grep`, et pour la raison que
    `test_la_source_du_module_ne_porte_ni_filet_large_ni_assert_ni_horloge` a
    déjà écrite : les
    docstrings de ce dépôt NOMMENT ce qu'elles interdisent. L'en-tête de
    `boucle_rendu.py` écrit `albedo_ecran` en toutes lettres pour dire qu'il ne
    l'importe pas — un motif textuel en ferait un usage."""
    arbre = ast.parse(source.read_text(encoding="utf-8"))
    noms: set[str] = set()
    for noeud in ast.walk(arbre):
        if isinstance(noeud, ast.Name):
            noms.add(noeud.id)
        elif isinstance(noeud, ast.Attribute):
            noms.add(noeud.attr)
        elif isinstance(noeud, ast.alias):
            noms.add(noeud.name.split(".")[0])
            noms.add(noeud.name)
            if noeud.asname:
                noms.add(noeud.asname)
        elif isinstance(noeud, ast.ImportFrom) and noeud.module:
            noms.add(noeud.module)
    return noms


# ─── D5 — LA CONVERSION `s` → IMAGE VIT DEHORS, ET C'EST MÉCANISÉ ────────────

def test_la_boucle_n_importe_pas_albedo_ecran_et_le_driver_si():
    """§1 DE LA SPEC ENDOSSÉE, LES DEUX MOITIÉS.

    `claude/spec-boucle-rendu-2026-08-24.md:29` « **LA CONVERSION `s` → IMAGE
    VIT DEHORS, DANS LE DRIVER.** » — et la ligne 36 du même §1 : « **La boucle
    ne convertit rien et n'importe pas `albedo_ecran`** ».

    LE MOTIF, ET IL N'EST PAS DE STYLE. `s_half` n'a AUCUNE valeur endossée
    dans le corpus : il est paramètre d'appelant dans `src/albedo.py:27` comme
    dans `src/f1_gpu/chemin_de_cout.py:344`. Le graver dans le module moteur
    serait choisir un paramètre hors de toute décision — le motif même par
    lequel le §5 du spec readout refuse son propre seuil.

    LES DEUX MOITIÉS COMPTENT. Vérifier seulement l'absence côté boucle
    laisserait passer un driver qui réimplémente `1 − exp(−s/s_half)` à la main
    : la lecture albédo se dédoublerait, et les deux copies dériveraient sans
    que rien ne le signale. Vérifier seulement la présence côté driver ne dirait
    rien de la garde. Ce verrou porte les deux."""
    de_la_boucle = _identifiants(Path(boucle_rendu.__file__))
    assert "albedo_ecran" not in de_la_boucle, (
        "`boucle_rendu.py` référence `albedo_ecran` dans son CODE : le §1 de la "
        "spec endossée range la conversion `s` → image DEHORS, dans le driver, "
        "parce que `s_half` n'a aucune valeur endossée dans le corpus")

    du_driver = _identifiants(Path(run_boucle_rendu_demo.__file__))
    assert "albedo_ecran" in du_driver, (
        "le driver n'utilise pas `albedo_ecran` : la lecture albédo est CELLE "
        "de `chemin_de_cout.py:344`, elle ne se réimplémente pas — deux copies "
        "de `1 − exp(−s/s_half)` dériveraient sans symptôme")


# ─── LE CÔTÉ ET `s_half` — DEUX ARGUMENTS OBLIGATOIRES, MÊME MOTIF ───────────

def test_le_cote_et_s_half_sont_obligatoires_et_sans_defaut():
    """AUCUN DES DEUX N'EST TRANCHÉ, DONC AUCUN DES DEUX N'A DE DÉFAUT.

    Le CÔTÉ appartient à la lecture d'orientation, qui n'est pas ordonnée et
    dont le prereg n'est pas écrit (§6 de la spec) ; `s_half` figure dans la
    MÊME liste du §6 (« **`s_half`**, et toute valeur de lecture albédo »). Un
    défaut, sur l'un comme sur l'autre, trancherait en silence une question que
    la spec laisse explicitement ouverte, et l'appelant hériterait d'un choix
    que personne n'a fait.

    C'est le pendant, côté driver, de
    `test_le_cote_n_a_aucune_valeur_par_defaut` côté module."""
    actions = {action.dest: action for action in construire_parseur()._actions}
    for nom in ("cote", "s_half"):
        assert actions[nom].required, (
            f"`--{nom.replace('_', '-')}` n'est pas obligatoire : le driver "
            "tranche alors en silence un paramètre que le §6 de la spec laisse "
            "explicitement ouvert")
        assert actions[nom].default is None, (
            f"`--{nom.replace('_', '-')}` porte un défaut ({actions[nom].default!r})")


def test_les_choix_du_cote_sont_les_constantes_du_module_pas_des_litteraux():
    """LES DEUX CÔTÉS SONT IMPORTÉS, JAMAIS RECOPIÉS.

    Le même geste que `test_le_couple_sort_en_alpha_croissant...` fait sur les
    `α` : une dérive du nom d'un côté dans `boucle_rendu` doit se PROPAGER au
    driver, pas se faire contredire par une chaîne gravée dans un `argparse`.
    Un littéral recopié donnerait un driver qui accepte un côté que la boucle
    refuse — `CoteInconnu` levée après le montage, loin de sa cause."""
    actions = {action.dest: action for action in construire_parseur()._actions}
    assert tuple(actions["cote"].choices) == COTES_DISPONIBLES
    assert set(COTES_DISPONIBLES) == set(COUPLES_PAR_COTE), (
        "les côtés offerts par le driver ne sont pas ceux de la table des "
        "couples de `boucle_rendu`")

    du_driver = _identifiants(Path(run_boucle_rendu_demo.__file__))
    assert {"COTE_INTERPOLER", "COTE_EXTRAPOLER"} <= du_driver, (
        "le driver ne référence pas les constantes de côté de `boucle_rendu` : "
        "il les a probablement recopiées en littéraux")

    # ET LA MOITIÉ QUI MANQUAIT, TROUVÉE PAR MUTATION. Les deux assertions
    # ci-dessus sont VIDES contre un driver qui garde l'`import` et recopie
    # quand même les valeurs : `COTES_DISPONIBLES` vient du driver, donc la
    # comparaison confronte le module à lui-même, et les identifiants importés
    # restent dans l'AST. Un littéral vaut aujourd'hui la constante — c'est
    # précisément pourquoi seule la SOURCE peut les distinguer.
    litteraux = [
        noeud.value
        for noeud in ast.walk(ast.parse(
            Path(run_boucle_rendu_demo.__file__).read_text(encoding="utf-8")))
        if isinstance(noeud, ast.Constant) and isinstance(noeud.value, str)
        and noeud.value in COTES_DISPONIBLES]
    assert not litteraux, (
        f"le nom d'un côté est écrit en littéral dans le driver : {litteraux}. "
        "Les deux côtés s'IMPORTENT de `boucle_rendu` ; recopiés, ils "
        "survivraient à un renommage de la constante et le driver offrirait un "
        "côté que la boucle refuse")


@pytest.mark.parametrize("arguments, motif", [
    ([], "côté absent"),
    (["--s-half", str(S_HALF_BANC)], "côté absent, `s_half` présent"),
    (["--cote", "interpolet", "--s-half", str(S_HALF_BANC)], "côté mal écrit"),
    (["--cote", COTE_INTERPOLER], "`s_half` absent"),
])
def test_un_appel_incomplet_ou_faux_fait_sortir_le_driver(
        arguments, motif, tmp_path):
    """SORTIR PLUTÔT QUE DE TOURNER SUR UN DÉFAUT — les quatre entrées.

    Le côté mal orthographié est le cas vicieux : sans `choices`, il tomberait
    sur un défaut ou plus loin, et rendrait des images du mauvais côté sans
    symptôme — ou lèverait `CoteInconnu` au montage, loin de sa cause.

    LA SORTIE NE SUFFIT PAS, ET LA PROSE NE DOIT PAS ÊTRE PLUS LARGE QUE CE QUI
    EST TENU. Cette docstring a dit « aucun de ces appels ne produit d'image »
    alors que le verrou ne constatait qu'un `SystemExit` : un driver qui aurait
    créé son dossier — ou écrit une image — AVANT de valider ses arguments
    serait passé. Le dossier de sortie est donc donné, et son ABSENCE après
    coup est constatée : c'est elle qui dit qu'aucun geste d'écriture n'a
    précédé la validation."""
    sortie = tmp_path / "sortie"
    with pytest.raises(SystemExit):
        run_boucle_rendu_demo.main(arguments + ["--dossier", str(sortie)])
    assert not sortie.exists(), (
        f"{motif} : le driver a créé son dossier de sortie avant de valider "
        "ses arguments — un geste d'écriture a précédé la garde")


def test_la_garde_de_s_half_est_celle_d_albedo_ecran_et_n_est_pas_redoublee():
    """`albedo_ecran` LÈVE DÉJÀ SUR `s_half <= 0` — on la laisse lever.

    `src/f1_gpu/chemin_de_cout.py:358-359` :
    « if s_half <= 0.0: raise ValueError(f"albedo_ecran : s_half={s_half} <= 0.") »

    Redoubler la garde dans le driver donnerait deux seuils à tenir d'accord,
    et le jour où l'un bouge, le message d'erreur nommerait le mauvais fichier.
    Le `match` porte sur le NOM DE LA FONCTION QUI LÈVE : c'est lui qui
    distingue « la garde d'amont a mordu » de « le driver a posé la sienne »."""
    ecran = np.full((2, 2), 1.0, dtype=np.float32)
    with pytest.raises(ValueError, match="albedo_ecran"):
        quantifier_en_octets(ecran, 0.0)
    with pytest.raises(ValueError, match="albedo_ecran"):
        quantifier_en_octets(ecran, -1.0)


# ─── D9 — UN NON-FINI NE DEVIENT PAS UN PIXEL SILENCIEUX ─────────────────────

@pytest.mark.parametrize("valeur, nom", [
    (np.nan, "NaN"),
    (np.inf, "+inf"),
    (-np.inf, "−inf"),
])
def test_un_ecran_non_fini_leve_par_classe_nommee(valeur, nom):
    """LA SIGNATURE §A53, ATTRAPÉE AVANT LA QUANTIFICATION.

    `albedo_ecran` applique `A = 1 − exp(−s / s_half)`. Sur un `s` non fini, `A`
    est indéfini, et une quantification en 8 bits en ferait un pixel
    d'APPARENCE NORMALE. Le précédent de la maison est
    `src/arcC_rendu.py:57-60` : « FAIL-LOUD, jamais un clip silencieux :
    l'albédo vit dans [0,1) par construction […] ; une valeur hors bande, un
    NaN ou une mauvaise forme sont un BUG AMONT, pas une donnée à rattraper en
    douce. »

    LE CONTRÔLE PORTE SUR `s`, PAS SUR `A`, ET C'EST LOAD-BEARING. Un contrôle
    posé sur l'albédo serait AVEUGLE au `+inf` : `A = 1 − exp(−∞) = 1,0`, une
    valeur finie, DANS la bande, qui quantifie en un pixel blanc parfaitement
    ordinaire. C'est exactement le pixel silencieux que ce verrou doit
    interdire, et il ne se voit qu'à l'entrée.

    PAR UNE CLASSE NOMMÉE, jamais un `assert` (§4-9, §A43) et jamais un clip
    silencieux. Le clamp de `melanger`, lui, COMPTE et ne lève pas (§5 du spec
    readout) : deux régimes différents, à ne pas confondre."""
    ecran = np.full((2, 2), 1.0, dtype=np.float32)
    ecran[1, 1] = valeur
    with pytest.raises(EcranNonRepresentable, match="non fini"):
        quantifier_en_octets(ecran, S_HALF_BANC)


def test_un_albedo_hors_bande_leve_plutot_que_de_boucler_en_uint8():
    """LA MÊME FAUTE, SON AUTRE FORME — et elle est aussi silencieuse.

    `A = 1 − exp(−s/s_half)` est `< 1` par construction, mais NÉGATIF dès que
    `s < 0`. Une quantification `np.rint(A·255).astype(np.uint8)` d'un négatif
    BOUCLE : `−1,7 · 255` devient un octet clair, un pixel d'apparence normale
    de plus. C'est la même signature §A53 que le non-fini, par une autre porte.

    OÙ CETTE FAUTE PEUT NAÎTRE, ET POURQUOI LES DEUX RÉGIMES NE SE CONFONDENT
    PAS. `melanger` applique le clamp `s ≥ 0` APRÈS le mélange et le COMPTE
    (§5 du spec readout) : l'écran INTERPOLÉ ne peut donc pas arriver négatif
    ici. L'écran EXACT, lui, est un gather DIRECT sur la pyramide (§2-1) — il
    ne passe par aucun clamp, et porte le `s` de la physique tel quel. Ce
    verrou garde donc exactement la moitié du couple que le clamp ne couvre
    pas, et il LÈVE là où le clamp COMPTE : deux régimes, deux réponses."""
    ecran = np.full((2, 2), 1.0, dtype=np.float32)
    ecran[0, 1] = -1.0
    with pytest.raises(EcranNonRepresentable, match="hors de la bande"):
        quantifier_en_octets(ecran, S_HALF_BANC)


def test_la_quantification_est_celle_d_albedo_ecran_a_255_niveaux():
    """CE QUE LE DRIVER ÉCRIT EST BIEN L'ALBÉDO DE `chemin_de_cout`.

    Sans ce verrou, tous les autres tiendraient sur des pixels arbitraires :
    ils vérifieraient qu'un PNG déterministe sort, sans rien dire de ce qu'il
    montre. Le témoin est `albedo_ecran` appelée à côté, jamais une formule
    recopiée."""
    ecran = np.linspace(0.0, 0.5, 16, dtype=np.float32).reshape(4, 4)
    octets = quantifier_en_octets(ecran, S_HALF_BANC)
    attendu = np.rint(albedo_ecran(ecran, S_HALF_BANC) * NIVEAUX_GRIS)
    assert _memes_octets(octets, attendu.astype(np.uint8)), (
        "la quantification n'est pas `rint(albedo_ecran(...) · 255)` en "
        "`uint8`. `_memes_octets` et non `np.array_equal` : la ligne 113 "
        "de ce fichier grave « Ni `allclose` ni `array_equal` », et celle-ci "
        "en était la SEULE entorse — introduite en tâche 3 dans un fichier que "
        "la tâche 2 avait laissé à zéro")


# ─── LE PNG — ÉCRIT À LA MAIN, RELU À LA MAIN ────────────────────────────────

def test_le_png_ne_porte_que_ihdr_idat_iend_donc_aucun_horodatage():
    """D7 MÉCANISÉ : PAS DE `tIME`, PAS DE `tEXt`, PAS DE DATE.

    Une liste BLANCHE, pas une liste noire — interdire `tIME` nommément
    laisserait passer `tEXt`, `zTXt`, `iTXt` et `eXIf`, qui peuvent tous porter
    une date. Ce qui n'est pas dans la liste n'a pas à être là.

    Écrire un PNG n'est pas une mesure ; l'horodater en serait une, et §A61
    tient tant que l'instrument n'est pas qualifié. C'est aussi ce qui rend le
    verrou (f) possible : un `tIME` rendrait deux runs différents par
    construction, et l'égalité à l'octet serait inatteignable."""
    octets = encoder_png_gris(np.arange(16, dtype=np.uint8).reshape(4, 4))
    types = tuple(type_ for type_, _ in _chunks_png(octets))
    assert types == CHUNKS_AUTORISES, (
        f"chunks trouvés {types}, attendus exactement {CHUNKS_AUTORISES} — "
        "tout autre chunk peut porter une date, et §A61 tient")


def test_le_png_se_relit_octet_par_octet_sans_bibliotheque_d_image():
    """« LES PNG S'OUVRENT » CESSE D'ÊTRE UNE AFFIRMATION.

    Le fichier est décodé À LA MAIN — signature, `IHDR` (dimensions,
    profondeur 8, type couleur 0 = gris), `IDAT` décompressé — et les pixels
    sont comparés au tableau d'entrée. Le filtre de CHAQUE ligne doit valoir 0 :
    un filtre non nul rendrait un fichier toujours valide, toujours lisible,
    mais dont les octets ne seraient plus les pixels — et le verrou (f), qui ne
    compare que des octets, ne le verrait JAMAIS.

    Aucun `pillow` : il est en ligne COMMENTÉE dans `requirements.txt`, et une
    dépendance non déclarée dans un verrou est une dette invisible (D7)."""
    pixels = np.arange(48, dtype=np.uint8).reshape(6, 8)
    chunks = dict(_chunks_png(encoder_png_gris(pixels)))

    largeur, hauteur, profondeur, couleur, compression, filtre, entrelacement = \
        struct.unpack(">IIBBBBB", chunks["IHDR"])
    assert (largeur, hauteur) == (pixels.shape[1], pixels.shape[0])
    assert (profondeur, couleur) == (8, 0), "attendu : 8 bits, niveaux de gris"
    assert (compression, filtre, entrelacement) == (0, 0, 0)

    brut = zlib.decompress(chunks["IDAT"])
    assert len(brut) == hauteur * (largeur + 1), (
        "la taille des données décompressées ne vaut pas `h · (l + 1)` : il "
        "manque ou il y a de trop un octet de filtre par ligne")
    assert OCTET_FILTRE_AUCUN == FILTRE_AUCUN_PNG, (
        f"le driver déclare le filtre « None » à {OCTET_FILTRE_AUCUN} : la "
        "spécification PNG le fixe à 0, et tout autre code annonce une "
        "transformation que l'encodeur n'applique pas")
    for indice_ligne in range(hauteur):
        debut = indice_ligne * (largeur + 1)
        assert brut[debut] == FILTRE_AUCUN_PNG, (
            f"ligne {indice_ligne} : octet de filtre {brut[debut]} ≠ "
            f"{FILTRE_AUCUN_PNG} — le fichier ANNONCE un filtre qu'il n'a pas "
            "appliqué : il reste valide, il s'ouvre, et ses octets ne sont plus "
            "les pixels")
        assert bytes(brut[debut + 1:debut + 1 + largeur]) == \
            pixels[indice_ligne].tobytes()


# ─── VERROU (f) — DEUX RUNS, MÊMES OCTETS ────────────────────────────────────

@pytest.mark.parametrize("cote", COTES)
def test_verrou_f_deux_runs_independants_donnent_des_png_bit_identiques(
        cote, tmp_path):
    """VERROU (f) — GOLDEN MINUSCULE, SUR DEUX CONSTRUCTIONS COMPLÈTES.

    CE QUE CE VERROU DOIT ATTRAPER, ET COMMENT IL EST CONSTRUIT POUR L'ATTRAPER.
    Comparer deux fois le résultat du MÊME appel serait vert et vide — la leçon
    de la tâche 2, apprise par mutation. Ici les deux runs construisent chacun
    LEUR pyramide, LEUR boucle et LEURS fichiers : tout ce qui pourrait dépendre
    d'une horloge, d'une adresse mémoire ou d'un état résiduel de processus a
    deux occasions de diverger.

    ET PAS D'UN ORDRE D'ITÉRATION DÉPENDANT DU HACHAGE — cette docstring l'a
    prétendu, et c'était une moitié de prose non tenue. Les deux runs vivent
    dans le MÊME processus, donc sous le même `PYTHONHASHSEED` : aucun ordre
    dérivé du hachage ne peut différer de l'un à l'autre. Le voir exigerait un
    second run en SOUS-PROCESSUS, que ce verrou ne fait pas.

    SUR `outputs/`, JAMAIS. `.gitignore:5` porte `outputs/` : un golden-fichier
    y serait hors du dépôt, et l'y committer serait un binaire dans l'histoire.
    Le verrou est « deux runs, mêmes octets », et les deux runs vivent sur
    `tmp_path`.

    CE QUE CE VERROU NE PEUT PAS ATTRAPER, DIT ICI PLUTÔT QUE SUPPOSÉ. Deux
    runs du même code ne voient aucune mutation DÉTERMINISTE de l'encodage :
    changer le niveau de compression, inverser deux lignes, écrire un `tEXt`
    constant laissent l'égalité intacte. Ce sont
    `test_le_png_ne_porte_que_ihdr_idat_iend...` et
    `test_le_png_se_relit_octet_par_octet...` qui couvrent cette moitié. Les
    trois se partagent le travail ; aucun ne rend les autres inutiles."""
    premier = rendre(cote, S_HALF_BANC, N_TICKS_GOLDEN, tmp_path / "run_a")
    second = rendre(cote, S_HALF_BANC, N_TICKS_GOLDEN, tmp_path / "run_b")
    assert premier == second == 2 * N_TICKS_GOLDEN

    noms_a = sorted(p.name for p in (tmp_path / "run_a").iterdir())
    noms_b = sorted(p.name for p in (tmp_path / "run_b").iterdir())
    assert noms_a == noms_b, "les deux runs n'ont pas écrit les mêmes noms"

    for nom in noms_a:
        octets_a = (tmp_path / "run_a" / nom).read_bytes()
        octets_b = (tmp_path / "run_b" / nom).read_bytes()
        assert octets_a == octets_b, (
            f"{nom} : deux runs indépendants ont écrit des octets différents. "
            "Une horloge, une adresse ou un ordre non déterministe est entré "
            "dans l'image ou dans ses métadonnées")


@pytest.mark.parametrize("cote", COTES)
def test_le_driver_ecrit_deux_png_par_tick_numerotes_sans_trou(cote, tmp_path):
    """2N IMAGES POUR N TICKS, NUMÉROTÉES DANS L'ORDRE DE SORTIE.

    Le couple sort en `α` CROISSANT et exactement un de ses termes vaut
    `α = 1,0` (§2-2, endossé par le §8 point 1) : les images se numérotent donc
    dans l'ordre où elles sortent, deux par tick, SANS TROU. Un trou dans la
    numérotation dirait qu'un écran n'a pas été écrit — et personne ne saurait
    lequel des deux `α` manque.

    LA NUMÉROTATION PART DE ZÉRO ET AVANCE D'UN PAR IMAGE, pas par tick :
    numéroter par tick perdrait l'ordre du couple, qui est précisément ce que
    le §2-2 rend lisible sans savoir quel côté tourne."""
    compte = rendre(cote, S_HALF_BANC, 3, tmp_path)
    assert compte == 6

    fichiers = sorted(tmp_path.iterdir())
    assert len(fichiers) == compte
    numeros = [int(p.stem.rsplit("_", 1)[1]) for p in fichiers]
    assert numeros == list(range(compte)), (
        f"numérotation {numeros} — attendu {list(range(compte))}, deux images "
        "par tick dans l'ordre de sortie, sans trou")
    for chemin in fichiers:
        assert cote in chemin.name, (
            "le nom de fichier ne porte pas le côté : deux runs de côtés "
            "différents dans le même dossier se recouvriraient à moitié, et "
            "rien ne dirait de quel côté vient quelle image")
        assert chemin.read_bytes().startswith(SIGNATURE_PNG)


def test_le_driver_n_imprime_que_le_nombre_d_images_ecrites(tmp_path, capsys):
    """LE SEUL CHIFFRE IMPRIMÉ EST « n images écrites » — §A61, mécanisé.

    Ni durée, ni taille, ni compte de clamps, ni compte de non-finis. §A61
    tient : l'instrument 3D n'est pas qualifié, aucune mesure n'est autorisée,
    et un chiffre imprimé par un driver de démo se cite ensuite comme s'il en
    était une. Le driver ne prononce rien non plus sur `clamps` / `non_finis` :
    le seuil qui invaliderait une mesure appartient au prereg de la lecture
    d'orientation (§3 de la spec), pas à ce code.

    L'ÉGALITÉ EST EXACTE, PAS UN `in`. Un `in` laisserait passer une ligne de
    plus — et c'est exactement par une ligne de plus que le premier chiffre non
    autorisé entrerait."""
    code = run_boucle_rendu_demo.main([
        "--cote", COTE_EXTRAPOLER, "--s-half", str(S_HALF_BANC),
        "--ticks", "2", "--dossier", str(tmp_path)])
    assert code == 0
    capture = capsys.readouterr()
    assert capture.out == "4 images écrites\n", (
        f"sortie {capture.out!r} — le seul chiffre autorisé est le nombre "
        "d'images écrites")
    assert capture.err == ""


def test_la_source_du_driver_ne_porte_ni_filet_large_ni_assert_ni_horloge():
    """LES TROIS GARDES DE SOURCE DU DRIVER, PAR L'AST.

    §4-8 — aucun `except RuntimeError`, `Exception` ni nu : les cinq serrures
    du readout et du chemin-de-coût en héritent toutes, et le gather lève
    `RuntimeError` pour ses trous de couverture (§4-7, qui ne se rattrape pas).
    Un filet large les avale d'un coup.

    §4-9 / §A43 — aucun `assert` : il disparaît sous `python -O`, donc une
    garde qui en dépend est absente exactement dans le régime où l'on mesurera.

    §1 ET §A61 — AUCUNE HORLOGE LUE ICI. « 2:1 » est un RAPPORT DE COMPTE : deux
    écrans par `frame()`, pas deux écrans par 33,3 ms. Cette phrase a dit
    « aucune horloge murale n'entre DANS CE CHANTIER » et c'était plus large que
    ce que ce contrôle tient : un chronomètre y tourne par transitivité
    (`TransfertComptable._chronometrer`, `src/f1_gpu/transferts.py:66-68`), et ce
    qui garde §A61 est qu'aucun chiffre n'en sorte. Ce que ce verrou tient est
    plus étroit et suffit : le driver ne LIT lui-même aucune horloge. Et une
    date lue serait de surcroît la porte par laquelle un horodatage entrerait
    dans un PNG. Ce contrôle porte sur les MODULES importés ET sur les attributs
    appelés : `import time` seul ne dit rien de `datetime.datetime.now()`.

    Et aucun `cupy` : la démo est CPU numpy, il n'y a pas de GPU à ce banc."""
    source = Path(run_boucle_rendu_demo.__file__)
    asserts, larges = _asserts_et_filets_larges(
        ast.parse(source.read_text(encoding="utf-8")))

    assert not asserts, (
        f"`assert` ligne(s) {asserts} du driver — il disparaît sous `python -O`")
    assert not larges, f"filet trop large dans le driver : {larges}"

    identifiants = _identifiants(source)
    horloges = sorted((identifiants & MODULES_HORLOGE)
                      | (identifiants & ATTRIBUTS_HORLOGE))
    assert not horloges, (
        f"horloge dans le driver : {horloges}. La cadence est LOGIQUE (§1) et "
        "§A61 interdit toute mesure tant que l'instrument n'est pas qualifié — "
        "et une date lue est la porte par laquelle un horodatage entre dans un "
        "PNG")
    assert "cupy" not in identifiants, "la démo est CPU numpy (aucun GPU ici)"


def test_le_cote_px_de_la_demo_est_couvrable_sur_toute_la_duree_du_run():
    """§4-7 — LE TROU DE COUVERTURE NE SE RATTRAPE PAS, DONC IL S'ÉVITE.

    Le gather LÈVE sur un pixel qu'aucune fenêtre active ne couvre, et la
    boucle ne le rattrape pas : choisir un `cote_px` couvrable est une
    responsabilité d'APPELANT, et le driver est cet appelant. Ce verrou constate
    la couverture sur une durée FRANCHEMENT plus longue que celle de la démo —
    la fovéa avance d'une cellule fine par tick, donc la couverture se dégrade
    avec le temps, et un run un peu plus long ne doit pas tomber sur un trou
    juste après la fin du banc.

    Il ne mesure RIEN : il constate qu'aucune levée ne se produit."""
    pyramide = run_boucle_rendu_demo.construire_pyramide()
    boucle = BoucleRendu(pyramide, COTE_INTERPOLER)
    for _ in range(4 * run_boucle_rendu_demo.N_TICKS_DEFAUT):
        boucle.tick(COTE_PX_DEMO)


def test_les_deux_ecrans_d_un_tick_ne_sont_pas_le_meme_champ():
    """LE CAS DÉGÉNÉRÉ DU §3, ATTRAPÉ SUR LE CHAMP FLOTTANT — pas sur l'image.

    LE FAIT QUI REND CE VERROU NÉCESSAIRE, ET IL A ÉTÉ MESURÉ. Sur le substrat
    jetable de cette démo, un pas de physique déplace `s` de MOINS d'un niveau
    de gris : les deux images d'un tick, une fois quantifiées en 8 bits, sont
    BIT-IDENTIQUES — vérifiable en LANÇANT la démo et en ouvrant les PNG
    qu'elle écrira (`outputs/` est ignoré par `.gitignore:5` et n'existe pas
    dans l'arbre : ce n'est pas un artefact qu'on va relire, c'est un dossier
    qu'un run produit). Or
    « tout écran interpolé devient identique à l'exact » est exactement le
    symptôme que le §3 de la spec décrit pour l'ORDRE DÉGÉNÉRÉ (`frame()` puis
    `TamponReadout.capturer`), qui ne lève PAS et laisse `clamps == 0`. Deux
    causes, un seul symptôme visible : sans ce verrou, la seconde se cacherait
    derrière la première, et la démo aurait l'air de tourner.

    CE QUI SÉPARE LES DEUX : le champ FLOTTANT. La coïncidence de quantification
    laisse `ecran_alpha_petit` et `ecran_alpha_grand` DIFFÉRENTS à l'octet
    flottant ; l'ordre dégénéré, lui, rend `s_prev == s_cur`, donc les deux
    écrans STRICTEMENT égaux. C'est là que le verrou porte.

    CE QUE CETTE DOCSTRING A REVENDIQUÉ ET QUI ÉTAIT FAUX — consigné, pas
    effacé. Elle a écrit « ET AUCUN VERROU DE LA TÂCHE 1 NE LE COUVRE […] il
    faut comparer les deux écrans ENTRE EUX, ce qu'aucun autre ne fait ». C'est
    faux DEUX FOIS, dans ce fichier même :
      - `test_la_vue_est_perimee_par_le_tick_suivant_mais_pas_l_ecran` finit sur
        `assert not _memes_octets(couple_2[0], couple_2[1])` — littéralement les
        deux écrans d'un tick comparés entre eux ;
      - `test_l_ecran_exact_est_le_gather_direct_sur_la_pyramide` TOMBE sous
        l'ordre dégénéré : `melanger(α)` d'un `s_prev == s_cur` rend `s_cur`,
        donc l'interpolé devient le gather direct et son assertion
        `not _memes_octets(couple[1 - rang_exact], direct)` mord.
    Seul l'argument sur `test_chaque_ecran_du_couple_vaut_le_noyau_du_
    paragraphe_2` tenait — son témoin dégénère avec l'état et reste vert. La
    faute est celle que ce chantier poursuit depuis §A62-bis-2, ici retournée en
    revendication de NOUVEAUTÉ : une prose plus large que ce qui est tenu.

    L'INCRÉMENT RÉEL EST UNE COUVERTURE DE SUBSTRAT, ET IL VAUT D'ÊTRE GARDÉ.
    Tous les verrous de la tâche 1 tournent sur `_pyramide_doublante()` +
    `_peindre_s_gradient` : un substrat SYNTHÉTIQUE, bâti pour que `s` évolue
    franchement d'un tick à l'autre. Celui-ci tourne sur le montage et le
    substrat DE LA DÉMO — `construire_pyramide()` et le `F` jetable par défaut —
    c'est-à-dire là où la coïncidence de quantification a réellement été
    mesurée, et là où `s` évolue de moins d'un niveau de gris par pas. Un `s`
    qui cesserait d'évoluer sous le `F` JETABLE ne se verrait dans aucun banc
    synthétique : c'est cette moitié-là, et elle seule, que ce verrou ajoute."""
    for cote in COTES:
        boucle = BoucleRendu(run_boucle_rendu_demo.construire_pyramide(), cote)
        for indice_tick in range(2):
            petit, grand = boucle.tick(COTE_PX_DEMO)
            assert not _memes_octets(petit, grand), (
                f"côté {cote}, tick {indice_tick} : les deux écrans du couple "
                "sont le MÊME champ à l'octet. `s_prev == s_cur` — c'est "
                "l'ordre dégénéré du §3, qui ne lève pas, laisse `clamps == 0` "
                "et ne se signale par rien d'autre que ceci")


def test_le_dossier_de_sortie_est_celui_que_la_tache_exige():
    """`outputs/boucle_rendu/` — UNE EXIGENCE, PAS UN CHOIX DU DRIVER.

    Le texte de la tâche l'écrit noir sur blanc : « N ticks → 2N PNG numérotés
    dans `outputs/boucle_rendu/` ». Ce chemin n'était tenu par RIEN — un mutant
    qui l'aurait changé en `outputs/demo/` passait la suite entière en vert, et
    les images seraient sorties ailleurs que là où on ira les chercher.

    LE CHEMIN EST ÉCRIT ICI EN LITTÉRAL, ET C'EST VOULU. L'importer du driver
    ferait comparer le module à lui-même — le patron exact qui a laissé
    survivre deux mutants au premier tour de cette tâche. Ce qui est comparé est
    l'EXIGENCE, pas la valeur courante du module.

    LES DEUX PORTES SONT TENUES : la constante, et le défaut réellement
    ENREGISTRÉ par `argparse`. Un driver qui garderait la constante juste et
    donnerait un autre défaut à `--dossier` déclarerait quand même le mauvais
    emplacement, et la seconde assertion est celle qui mord.

    CE QUE CE VERROU NE TIENT PAS, NOMMÉ PLUTÔT QUE SOUS-ENTENDU. Sa seconde
    assertion a d'abord dit « c'est ce défaut qui décide où `main` écrit » —
    d'un cran plus large que ce qu'elle attrape. Elle constate le défaut
    ENREGISTRÉ dans le parseur ; elle ne constate PAS le câblage
    `main` → `args.dossier` → `rendre`. Un driver qui passerait `DOSSIER_SORTIE`
    en dur à `rendre` en ignorant `args.dossier` survivrait à ce verrou, parce
    que tous les verrous d'emplacement de ce fichier appellent `rendre`
    directement. Le trou est nommé et NON comblé : le renforcer demanderait de
    faire écrire `main` hors de `tmp_path`, ou d'espionner `rendre`, et ni l'un
    ni l'autre n'est demandé ici."""
    exige = Path("outputs/boucle_rendu")
    assert run_boucle_rendu_demo.DOSSIER_SORTIE == exige, (
        f"`DOSSIER_SORTIE` vaut {run_boucle_rendu_demo.DOSSIER_SORTIE} — la "
        f"tâche exige {exige}")

    actions = {action.dest: action for action in construire_parseur()._actions}
    assert actions["dossier"].default == exige, (
        f"le défaut de `--dossier` vaut {actions['dossier'].default} — la "
        f"tâche exige {exige}. C'est l'emplacement que le driver DÉCLARE quand "
        "l'appelant n'en donne aucun ; que `main` l'emprunte réellement n'est "
        "pas ce que ce verrou constate")


@pytest.mark.parametrize("n_ticks", [0, -1])
def test_un_nombre_de_ticks_nul_ou_negatif_est_refuse(n_ticks, tmp_path):
    """UNE DÉMO QUI N'ÉCRIT RIEN NE DOIT PAS RESSEMBLER À UNE DÉMO QUI TOURNE.

    Sans cette garde, `range(0)` et `range(-1)` sont tous deux vides : le
    driver sortirait en code 0 sur « 0 images écrites », c'est-à-dire un
    silence présenté comme un résultat. Le seul chiffre que ce driver imprime
    doit vouloir dire quelque chose.

    TROUVÉ PAR MUTATION, PAS PAR RELECTURE : la garde était écrite dans
    `rendre` et aucun verrou ne la tenait — un mutant qui la retirait passait
    la suite en vert. Une garde que rien ne tient est une garde absente.

    SUR LA CLASSE, PAS SUR LE TEXTE DU MESSAGE. Ce verrou s'est d'abord rabattu
    sur `pytest.raises(ValueError, match="n_ticks")`, parce que la garde levait
    un `ValueError` NU : la serrure tenait alors à une chaîne de caractères,
    qu'une reformulation innocente du message aurait ouverte sans bruit. La
    garde porte désormais une classe nommée, comme partout ailleurs dans ce
    chantier, et le verrou porte sur elle."""
    with pytest.raises(LongueurDeDemoInvalide):
        rendre(COTE_INTERPOLER, S_HALF_BANC, n_ticks, tmp_path)
