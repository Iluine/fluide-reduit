"""Tests F1 — L'INVARIANT DU TRANSFERT L3, et sa tenue sous ping-pong
(§A21, pocCascade2phys 299adfc).

INVARIANT GRAVÉ, au-dessus du correctif :

    **la référence du device n'avance QUE sur ce qui a été effectivement
    TRANSFÉRÉ, jamais sur ce qui a été ÉMIS.**

Le kernel L3 est INTOUCHÉ et fait `reference[p] += d` à l'émission : il
ne peut pas tenir l'invariant lui-même, n'ayant aucun moyen de savoir ce
qui sera transféré. L'invariant est donc tenu par le COMPACTEUR, en
garantissant la BIJECTION entre ce qui est émis et ce qui est reçu —
chaque couple transféré une fois et une seule. Ces tests éprouvent cette
bijection.

────────────────────────────────────────────────────────────────────────
CE QUE LE PING-PONG CHANGE
────────────────────────────────────────────────────────────────────────
Avant, la TAILLE venait d'une frame et les DONNÉES d'une autre : `taille`
était le compteur de n−1, le tampon contenait l'émission de n. Deux
pertes symétriques en découlaient, toutes deux EXHIBÉES par les cas
construits ci-dessous (§A19-CORRECTION) :
  (α) frame émettant PLUS : les couples au-delà de `taille` n'étaient
      jamais transférés, alors que `reference[p] += d` avait déjà tourné.
      Perte DÉFINITIVE ;
  (β) frame émettant MOINS : les positions [émis, taille) portaient
      encore les couples de la frame précédente, retransférés et
      RÉAPPLIQUÉS.

Désormais la frame n écrit dans le jeu `n mod 2` et le transfert porte
sur le jeu de n−1 avec le compteur de n−1 : **taille et données du même
tour**. Les MÊMES cas construits, avec les MÊMES attendus, prouvent
maintenant l'ABSENCE des deux défauts.

────────────────────────────────────────────────────────────────────────
LA FORME EXACTE DE L'INVARIANT — le décalage d'une frame est CONSERVÉ
────────────────────────────────────────────────────────────────────────
Le correctif ne supprime pas le retard, il le rend HONNÊTE. Ce que le CPU
détient APRÈS la frame n est exactement ce que le device avait émis À LA
FIN de la frame n−1 :

    livre_cpu(après n)  ≡  livre_device(après n−1)

Rien de moins (α réglé), rien de plus (β réglé). Seule la dernière frame
d'une série reste en attente, ce que `diagnostic_retard` reporte.

Les tests SYNCHRONISENT après `cloturer_frame` pour rendre le retard
exactement d'une frame. `int(compacteur.compteur[0])` est une lecture de
TEST : la production ne la fait jamais — c'est tout l'objet du choix 6,
et `test_aucune_synchronisation_ajoutee` l'impose."""
import numpy as np
import pytest

from src.f1_gpu.backend import cupy_disponible, vers_cpu
from src.f1_gpu.substrat_fusionne import pas_f_fusionne
from src.f1_gpu.substrat_jetable import etat_initial_jetable
from src.f1_gpu.substrat_l3 import (
    CompacteurL3,
    pas_f_l3,
    taille_sortie_max,
)

gpu_requis = pytest.mark.skipif(
    not cupy_disponible(), reason="cupy/device CUDA indisponible")

EPS_FIGE: float = 1e-4          # inchangé — rien n'est réglé ici
N_BLOCS: int = 2
N_COTE: int = 12
GRAINE: int = 91

# Amplitudes RÉELLES lues dans outputs/f1/ (aucun run nouveau) : borne_l3
# varie d'un facteur 2.3 sur ses deux groupes, ma_quater d'un facteur 4.8
# sur le second. La série d'invariant les reproduit.
FACTEUR_BORNE_L3: float = 2.3
FACTEUR_MA_QUATER: float = 4.8
CARDINAL_BAS: int = 100
CARDINAUX_VARIABLES: tuple[int, ...] = (
    CARDINAL_BAS, 230, CARDINAL_BAS, 478, CARDINAL_BAS,
    478, 230, CARDINAL_BAS, 230, 478)


def _etat_initial(cp):
    """(B, 1, 4, n, n) f32 — un système suffit : ce qui est éprouvé est le
    TRANSFERT, pas le motif de calcul."""
    brut = etat_initial_jetable(N_BLOCS, N_COTE, graine=GRAINE)[:, :1]
    return cp.asarray(np.ascontiguousarray(brut))


def _etat_apres(cp, q):
    """L'état que le pas VA produire, obtenu par le kernel FUSIONNÉ —
    dont la sortie est bit-identique à celle de L3 (verrou déjà en place,
    `test_etat_bit_identique_au_fusionne`). Sert à fabriquer une
    référence qui donne d = 0 EXACTEMENT partout où on ne veut pas
    d'émission."""
    copie = q.copy()
    pas_f_fusionne(copie, cp, sortie=copie, tampon_etage=cp.empty_like(copie))
    return copie


def _reference_emettant(cp, q, indices_emetteurs):
    """Référence construite pour que EXACTEMENT les indices donnés
    émettent : ailleurs d = etat_apres − reference = 0, donc |d| < eps.

    Le décalage vaut 10·eps, franchement au-dessus du seuil — on ne teste
    pas la frontière du seuil, on choisit un CARDINAL d'émission."""
    reference = _etat_apres(cp, q)
    reference.reshape(-1)[cp.asarray(indices_emetteurs)] -= cp.float32(
        10.0 * EPS_FIGE)
    return reference


def _frame(cp, q, reference, compacteur):
    """Une frame, avec la discipline de `_appliquer_f`
    (run_f1_ma_quater.py) reproduite À L'IDENTIQUE. Retourne
    (taille, émis, transféré, émission) — les deux derniers en copies
    HÔTE, l'émission étant lue dans le jeu COURANT du ping-pong."""
    taille = compacteur.taille_precedente()
    compacteur.noter_taille(taille)
    pas_f_l3(q, cp, reference, compacteur, sortie=q,
             tampon_etage=cp.empty_like(q), eps=EPS_FIGE)
    emis = int(compacteur.compteur[0])           # lecture de TEST
    transfere = None
    if taille:
        valeurs, indices = compacteur.vues_a_transferer(taille)
        transfere = (vers_cpu(valeurs).copy(), vers_cpu(indices).copy())
    emission = (vers_cpu(compacteur.valeurs[:emis]).copy(),
                vers_cpu(compacteur.indices[:emis]).copy())
    compacteur.cloturer_frame()
    # Déterminisme du test : le retard devient EXACTEMENT d'une frame.
    cp.cuda.Stream.null.synchronize()
    return taille, emis, transfere, emission


def _cumuler(grand_livre, couples):
    """`np.add.at` sur les indices aplatis, exactement comme
    `ReconstructeurNiveau0.appliquer`."""
    if couples is None:
        return
    valeurs, indices = couples
    np.add.at(grand_livre, indices.astype(np.int64), valeurs)


def _sans_bruit(x, seuil: float = 1e-9) -> int:
    """Nombre d'entrées franchement non nulles — le signal construit vaut
    10·eps = 1e-3, mille fois au-dessus de ce seuil."""
    return int(np.count_nonzero(np.abs(x) > seuil))


def _jouer_serie(cp, q, compacteur, cardinaux, graine: int):
    """Joue une série de frames aux CARDINAUX d'émission imposés, en
    tenant les deux grands livres, et éprouve l'invariant à CHAQUE frame :

        livre_cpu(après n) ≡ livre_device(après n−1)

    Retourne le journal [(taille, émis), ...]."""
    total = taille_sortie_max(q)
    rng = np.random.default_rng(graine)
    livre_cpu = np.zeros(total)
    livre_device = np.zeros(total)
    device_a_la_frame_precedente = np.zeros(total)
    journal = []
    for cardinal in cardinaux:
        emetteurs = rng.permutation(total)[:cardinal]
        reference = _reference_emettant(cp, q, emetteurs)
        taille, emis, transfere, emission = _frame(cp, q, reference,
                                                   compacteur)
        assert emis == cardinal          # le cas construit tient
        _cumuler(livre_cpu, transfere)
        # L'INVARIANT, éprouvé frame par frame : ni manque, ni doublon.
        np.testing.assert_allclose(livre_cpu, device_a_la_frame_precedente,
                                   rtol=0, atol=1e-12)
        _cumuler(livre_device, emission)
        device_a_la_frame_precedente = livre_device.copy()
        journal.append((taille, emis))
    return journal


# ----- (α) la troncature ne perd plus rien -----

@gpu_requis
def test_alpha_la_troncature_ne_perd_plus_rien():
    """MÊME cas construit qu'en §A19-CORRECTION : 5 émissions, puis 40,
    puis trois frames à 40. La frame 2 émet huit fois plus que la 1 —
    c'est exactement la condition qui faisait perdre 35 couples.

    ATTENDU, inchangé : si (α) subsistait, la frame 2 creuserait un écart
    de grand livre de 35 couples qu'aucune frame propre ne comblerait.
    Sous ping-pong, l'écart est NUL à chaque frame — la frame 2 transfère
    les 5 couples de la frame 1, la frame 3 les 40 de la frame 2."""
    import cupy as cp
    q = _etat_initial(cp)
    compacteur = CompacteurL3(cp, taille_sortie_max(q))
    journal = _jouer_serie(cp, q, compacteur, (5, 40, 40, 40, 40), graine=5)

    # La condition de (α) A BIEN EU LIEU : la frame 2 émet 40 là où la
    # frame 1 n'en avait émis que 5. Sans quoi le test serait vide.
    assert journal[1] == (5, 40)
    # ... et le transfert de la frame 2 porte sur les 5 de la frame 1,
    # pas sur une tranche tronquée des 40 siens.
    assert journal[2] == (40, 40)


@gpu_requis
def test_alpha_le_dernier_emis_reste_en_attente_et_rien_de_plus():
    """La seule chose qui reste non transférée à la fin d'une série est
    l'émission de la DERNIÈRE frame — le retard d'une frame, honnête.
    C'est la borne exacte de ce que le ping-pong laisse en suspens, et
    `diagnostic_retard` la reporte."""
    import cupy as cp
    q = _etat_initial(cp)
    compacteur = CompacteurL3(cp, taille_sortie_max(q))
    total = taille_sortie_max(q)
    rng = np.random.default_rng(13)
    livre_cpu, livre_device = np.zeros(total), np.zeros(total)
    derniere_emission = None
    for cardinal in (24, 60, 24):
        emetteurs = rng.permutation(total)[:cardinal]
        reference = _reference_emettant(cp, q, emetteurs)
        _, emis, transfere, emission = _frame(cp, q, reference, compacteur)
        _cumuler(livre_cpu, transfere)
        _cumuler(livre_device, emission)
        derniere_emission = emission
    en_attente = np.zeros(total)
    _cumuler(en_attente, derniere_emission)
    np.testing.assert_allclose(livre_device - livre_cpu, en_attente,
                               rtol=0, atol=1e-12)
    assert _sans_bruit(en_attente) == 24


# ----- (β) la queue de tampon n'est plus lisible -----

@gpu_requis
def test_beta_la_queue_de_tampon_nest_plus_reappliquee():
    """MÊME cas construit qu'en §A19-CORRECTION : 40 émissions, 40, puis
    5. La frame 3 émet huit fois moins que la 2 tout en transférant 40 —
    c'est exactement la condition qui réappliquait 35 couples périmés.

    ATTENDU, inchangé : si (β) subsistait, les positions [5, 40) du
    transfert seraient des couples de la frame précédente réappliqués.
    Sous ping-pong, les 40 couples transférés sont EXACTEMENT l'émission
    de la frame 2 — complète, et pas une de plus."""
    import cupy as cp
    q = _etat_initial(cp)
    compacteur = CompacteurL3(cp, taille_sortie_max(q))
    total = taille_sortie_max(q)
    rng = np.random.default_rng(11)

    emissions, transferts, journal = [], [], []
    for cardinal in (40, 40, 5):
        emetteurs = rng.permutation(total)[:cardinal]
        reference = _reference_emettant(cp, q, emetteurs)
        taille, emis, transfere, emission = _frame(cp, q, reference,
                                                   compacteur)
        emissions.append(emission)
        transferts.append(transfere)
        journal.append((taille, emis))

    # La condition de (β) A BIEN EU LIEU : 40 transférés pour 5 émis.
    assert journal[2] == (40, 5)
    # Le transfert de la frame 3 est l'émission de la frame 2, TERME À
    # TERME — donc aucune queue périmée, et aucun couple manquant.
    valeurs_transferees, indices_transferes = transferts[2]
    valeurs_emises, indices_emis = emissions[1]
    np.testing.assert_array_equal(valeurs_transferees, valeurs_emises)
    np.testing.assert_array_equal(indices_transferes, indices_emis)


@gpu_requis
def test_beta_les_deux_jeux_sont_distincts_et_basculent():
    """La cause immédiate de (β) — un seul jeu de buffers, dont la queue
    survivait à la remise à zéro du compteur — n'existe plus : le kernel
    de la frame n écrit ailleurs que là où la frame n−1 a écrit."""
    import cupy as cp
    compacteur = CompacteurL3(cp, 64)
    premier = (compacteur.valeurs.data.ptr, compacteur.indices.data.ptr)
    compacteur.cloturer_frame()
    second = (compacteur.valeurs.data.ptr, compacteur.indices.data.ptr)
    assert premier != second                    # bascule effective
    compacteur.cloturer_frame()
    assert (compacteur.valeurs.data.ptr,
            compacteur.indices.data.ptr) == premier      # ping-pong


# ----- l'INVARIANT lui-même, sur les amplitudes réelles -----

@gpu_requis
def test_invariant_sur_serie_a_tailles_fortement_variables():
    """Pas une régression sur deux cas : l'INVARIANT, éprouvé frame par
    frame sur une série dont les tailles varient aux AMPLITUDES RÉELLES
    lues dans outputs/f1/ — 2.3× (borne_l3) et 4.8× (ma_quater).

    La série alterne croissances et décroissances, donc déclenche
    alternativement les conditions de (α) et de (β) à chaque changement.
    À chaque frame : livre_cpu(après n) ≡ livre_device(après n−1)."""
    import cupy as cp
    q = _etat_initial(cp)
    compacteur = CompacteurL3(cp, taille_sortie_max(q))
    journal = _jouer_serie(cp, q, compacteur, CARDINAUX_VARIABLES,
                           graine=29)

    cardinaux = [emis for _, emis in journal]
    assert min(cardinaux) == CARDINAL_BAS
    # les deux amplitudes réelles sont bien exercées
    assert max(cardinaux) / min(cardinaux) == pytest.approx(
        FACTEUR_MA_QUATER, abs=0.03)
    assert 230 / CARDINAL_BAS == pytest.approx(FACTEUR_BORNE_L3, abs=0.01)
    # et le diagnostic dit bien que les tailles ne sont PAS stables :
    # l'invariant tient malgré cela, ce qui est tout le propos.
    diagnostic = compacteur.diagnostic_retard()
    assert diagnostic["tailles_stables"] is False
    assert "une fois et une seule" in diagnostic["note"]


# ----- le choix 6 est PRÉSERVÉ : aucune synchronisation ajoutée -----

@gpu_requis
def test_aucune_synchronisation_ajoutee(monkeypatch):
    """PROUVÉ, pas affirmé. Les points d'entrée de synchronisation sont
    remplacés par des levées, puis une série de frames est jouée par le
    chemin de production : compacteur + kernel L3 avec la réduction CFL
    ON-DEVICE (celle que le pipeline emploie — la réduction du jetable,
    elle, synchronise par conception et n'est pas ce chemin).

    SEULE attente autorisée (M8, review 28/07) : l'évènement du memcpy du
    compteur — 4 octets soumis une frame plus tôt, attente quasi nulle en
    régime. Elle est VOULUE : sans elle, `taille_precedente()` ne tenait
    que par les synchronisations incidentes des voisins. Les attentes de
    stream/device restent interdites, et le test VÉRIFIE DÉSORMAIS DES
    DONNÉES (le trou noté par la review) : le compteur lu à chaque frame
    est celui de la frame précédente, pas un zéro périmé."""
    import cupy as cp

    from scripts.run_f1_s2_batche import reduction_cfl_on_device

    def interdit(*_args, **_kwargs):
        raise AssertionError(
            "synchronisation pendant la boucle de frame — le choix 6 "
            "interdit toute attente par frame (maladie s2).")

    # Le montage se fait AVANT l'armement : fabriquer la référence passe
    # par la réduction du JETABLE, qui synchronise par conception. Ce
    # qu'on éprouve est la BOUCLE DE FRAME, pas le montage.
    q = _etat_initial(cp)
    compacteur = CompacteurL3(cp, taille_sortie_max(q))
    reference = _reference_emettant(cp, q, np.arange(48))
    tampon = cp.empty_like(q)

    # `cp.cuda.Device` est un type immuable : sa synchronisation passe de
    # toute façon par `runtime.deviceSynchronize`, qui est couvert.
    # `eventSynchronize` n'est PAS interdit : c'est l'attente bornée M8.
    monkeypatch.setattr(cp.cuda.Stream, "synchronize", interdit)
    monkeypatch.setattr(cp.cuda.runtime, "deviceSynchronize", interdit)
    monkeypatch.setattr(cp.cuda.runtime, "streamSynchronize", interdit)
    monkeypatch.setattr(cp.cuda.runtime, "memcpy", interdit)   # D2H bloquant

    # LE PIÈGE EST-IL ARMÉ ? Un piège qui n'attrape rien ne prouverait
    # rien. On vérifie d'abord qu'il mord sur une synchronisation
    # délibérée, AVANT de conclure de son silence.
    with pytest.raises(AssertionError, match="choix 6"):
        cp.cuda.Stream.null.synchronize()

    for _ in range(6):
        taille = compacteur.taille_precedente()
        compacteur.noter_taille(taille)
        pas_f_l3(q, cp, reference, compacteur, sortie=q,
                 tampon_etage=tampon, eps=EPS_FIGE,
                 reduction=reduction_cfl_on_device)
        if taille:
            compacteur.vues_a_transferer(taille)
        compacteur.cloturer_frame()

    # VÉRIFICATION DE DONNÉES (le trou noté par la review : ce test ne
    # vérifiait RIEN). Vérité-terrain : la MÊME série rejouée sur un
    # compacteur frais par le chemin instrumenté (lecture directe du
    # compteur, synchronisante — artefact de TEST, hors boucle armée).
    # Les tailles vues par la boucle sans-sync doivent être les cardinaux
    # d'émission de la frame précédente — un compteur périmé (zéro, ou
    # taille d'une frame plus vieille) casse ici. C'est exactement la
    # donnée que l'évènement M8 garantit.
    monkeypatch.undo()
    q2 = _etat_initial(cp)
    compacteur2 = CompacteurL3(cp, taille_sortie_max(q2))
    reference2 = _reference_emettant(cp, q2, np.arange(48))
    tampon2 = cp.empty_like(q2)
    emissions = []
    for _ in range(6):
        taille = compacteur2.taille_precedente()
        compacteur2.noter_taille(taille)
        pas_f_l3(q2, cp, reference2, compacteur2, sortie=q2,
                 tampon_etage=tampon2, eps=EPS_FIGE,
                 reduction=reduction_cfl_on_device)
        cp.cuda.Stream.null.synchronize()
        emissions.append(int(compacteur2.compteur[0]))
        if taille:
            compacteur2.vues_a_transferer(taille)
        compacteur2.cloturer_frame()
    assert compacteur.tailles_vues == [0] + emissions[:-1]
    assert emissions[0] == 48        # la frame 1 du cas construit tient


@gpu_requis
def test_le_pingpong_double_les_buffers_et_le_reporte():
    """Le coût est REPORTÉ, jamais tu : `octets_buffers()` compte les DEUX
    jeux, et c'est lui que la résidence des drivers consomme."""
    import cupy as cp
    taille = 1 << 16
    compacteur = CompacteurL3(cp, taille)
    un_jeu = compacteur.valeurs.nbytes + compacteur.indices.nbytes
    assert compacteur.octets_buffers() == 2 * un_jeu
    assert compacteur.octets_buffers() == taille * (4 + 4) * 2
    assert compacteur.diagnostic_retard()["octets_buffers"] == (
        compacteur.octets_buffers())


# ----- le nota faux a été RETIRÉ -----

def test_le_nota_rien_nest_perdu_nest_plus_porte_comme_vrai():
    """Le nota « le schéma étant incrémental, rien n'est perdu » était
    FAUX — (α) l'a réfuté. Il est RETIRÉ (§A21), pas nuancé.

    Ce test n'interdit pas de le CITER — la docstring le cite pour dire
    qu'il était faux, et c'est la bonne façon de retirer une affirmation.
    Il interdit qu'il soit encore PORTÉ COMME VRAI : toute citation doit
    être accompagnée de sa réfutation, et la note reportée au JSON — le
    seul texte qu'un lecteur de résultats verra — ne doit plus le
    contenir du tout."""
    doc = CompacteurL3.__doc__
    assert "rien n'est perdu" in doc          # cité...
    assert "était FAUX" in doc                # ... et réfuté sur place
    assert "RETIRÉ" in doc
    if cupy_disponible():
        import cupy as cp
        note = CompacteurL3(cp, 64).diagnostic_retard()["note"]
        assert "rien n'est perdu" not in note
        assert "une fois et une seule" in note


def test_linvariant_grave_est_porte_par_la_classe():
    """L'invariant §A21 est l'ÉNONCÉ dont le ping-pong est une
    implémentation — il doit être lisible là où il est tenu, pas
    seulement dans le journal."""
    doc = CompacteurL3.__doc__
    assert "effectivement TRANSFÉRÉ" in doc
    assert "jamais sur ce qui a été ÉMIS" in doc


# ----- le prix du correctif, chiffré et reporté -----

def test_le_surcout_pingpong_sur_v4_vaut_120_mio():
    """Le prix du correctif sur la géométrie de mesure, calculé sans rien
    allouer : un SECOND jeu de buffers d'émission, dimensionnés au pire
    cas (choix 5). Chiffré AVANT le run, pour que la résidence mesurée
    puisse être confrontée à une attente écrite d'avance.

    Résidence V2 mesurée 0.311 Go, gate 1.35 Go : le surcoût de 0.126 Go
    laisse la marge intacte. Le driver reporte le surcoût À PART."""
    from scripts.run_f1_ma_quater import GATE_VRAM_GO, geometrie_v4
    from scripts.run_f1_s2_mobile import plan_groupes

    geo = geometrie_v4()
    elements = sum(p["n_slots"] * p["n_systemes"] * 4 * geo.n_fov ** 2
                   for p in plan_groupes(geo))
    un_jeu = elements * (4 + 4)          # valeurs f32 + indices u32
    assert un_jeu == 120 * 2 ** 20       # 120 Mio, le surcoût exact
    assert GATE_VRAM_GO == 1.35
    residence_v2_go, surcout_go = 0.311, un_jeu / 1000 ** 3
    assert surcout_go == pytest.approx(0.126, abs=0.001)
    assert residence_v2_go + surcout_go < GATE_VRAM_GO


# ----- l'offset voyage avec les données, sous le même invariant -----

@gpu_requis
def test_loffset_suit_les_donnees_et_non_la_frame_courante():
    """RÉGRESSION TROUVÉE PAR LE VERROU DE LA SONDE (§A21-complément).

    Les indices émis sont relatifs à la tranche passée au kernel. Le
    ping-pong livre les données de n−1 pendant la frame n : les indexer
    par l'offset de n les écrirait dans le mauvais slot, en silence.
    L'offset relève donc du MÊME invariant que la taille et les données —
    il vient du tour ÉMETTEUR."""
    import cupy as cp
    compacteur = CompacteurL3(cp, 64)
    compacteur.noter_offset(1000)          # frame n : tour à l'offset 1000
    assert compacteur.offset_precedent() == 0        # aucun prédécesseur
    compacteur.cloturer_frame()
    compacteur.noter_offset(2000)          # frame n+1 : offset 2000
    # ... mais ce qu'on transfère est l'émission de la frame n :
    assert compacteur.offset_precedent() == 1000
    compacteur.cloturer_frame()
    compacteur.noter_offset(3000)
    assert compacteur.offset_precedent() == 2000
