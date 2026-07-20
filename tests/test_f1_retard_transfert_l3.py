"""Tests F1 — VÉRIFICATION DE DEUX BUGS CANDIDATS du retard d'une frame
(§A19-CORRECTION, pocCascade2phys 917de09).

Ce sont des tests de CORRECTION, pas de mesure : ils construisent des cas
où les tailles d'émission varient de façon CHOISIE, et regardent ce que le
CPU reçoit réellement. Aucun kernel n'est touché — le kernel L3 et le
kernel fusionné sont appelés tels quels.

────────────────────────────────────────────────────────────────────────
LA DISCIPLINE EXAMINÉE — celle de `_appliquer_f` (run_f1_ma_quater.py)
────────────────────────────────────────────────────────────────────────
    taille = compacteur.taille_precedente()   # compteur de la frame n−1
    ... pas_f_l3(...) ...                     # émission de la frame n
    if taille:  transférer valeurs[:taille], indices[:taille]
    compacteur.cloturer_frame()

La TAILLE vient de la frame précédente ; les DONNÉES viennent de celle-ci.
Deux régimes en découlent, et ce sont les deux candidats :

**(α) TRONCATURE** — la frame n émet PLUS que n−1 : les couples au-delà
de `taille` ne sont pas transférés. Or le kernel a déjà fait
`reference[p] += d` pour eux (épilogue L3, choix 2). Le device tient donc
pour acquis que le CPU sait ce qu'il n'a jamais reçu, et le détail ne
repassera plus le seuil : il est absorbé dans la référence.

**(β) QUEUE DE TAMPON** — la frame n émet MOINS que n−1 : `cloturer_frame`
remet le COMPTEUR à zéro mais ne réécrit pas les buffers. Les positions
[émis_n, taille) portent donc encore les couples de la frame n−1, et le
transfert de taille `taille` les emporte une seconde fois.

────────────────────────────────────────────────────────────────────────
ATTENDUS, ÉCRITS D'AVANCE
────────────────────────────────────────────────────────────────────────
Dans un cas STATIQUE (aucun déplacement de fovéa, aucune prédiction), le
schéma B4 pose l'identité `vivant CPU == reference device`. Tout écart y
est donc imputable au seul transfert.

  - si (α) est RÉEL : après une frame tronquée, l'écart vivant↔référence
    est non nul et **INVARIANT** sur les frames suivantes — y compris
    celles qui ne tronquent rien. La perte est DÉFINITIVE, pas différée ;
  - si (α) est FAUX : l'écart se résorbe dès qu'une frame non tronquée
    passe (les résidus repasseraient le seuil, comme le nota l'affirme) ;
  - si (β) est RÉEL : la frame qui émet moins transfère, aux positions
    [émis, taille), des couples BIT-IDENTIQUES à ceux de la frame
    précédente, et le CPU les applique une SECONDE fois ;
  - si (β) est FAUX : le transfert ne contient que des couples émis par
    la frame courante.

Les tests SYNCHRONISENT après `cloturer_frame`, ce qui rend le retard
exactement d'une frame — le cas le PLUS FAVORABLE au schéma. Un défaut
qui subsiste là est structurel, pas un aléa d'ordonnancement.

────────────────────────────────────────────────────────────────────────
CE QUE LES CAS CONSTRUITS ONT MONTRÉ — LES DEUX SONT RÉELS
────────────────────────────────────────────────────────────────────────
(α) et (β) sont EXHIBÉS tous les deux. Pour (α), la perte est en outre
DÉFINITIVE : la frame suivante, qui ne tronque rien, ne la comble pas.

**Cela CONTREDIT le nota endossé** — « le schéma B4 étant incrémental,
les résidus repassent le seuil : rien n'est perdu » (`CompacteurL3` et
`diagnostic_retard`). Le nota suppose que le détail non transféré reste
un résidu côté GPU. Il ne le reste pas : l'épilogue L3 fait
`reference[p] += d` pour TOUT couple émis, transféré ou non. Le device
inscrit donc dans sa référence une connaissance que le CPU n'a jamais
reçue, et l'écart ne repassera jamais le seuil — il est absorbé. Le nota
est CITÉ ici, pas corrigé : sa correction est une décision de Romain.

La condition de déclenchement est exactement `tailles_stables`, que
`diagnostic_retard` reporte DÉJÀ. Sa mesure est juste, c'est sa LECTURE
qui est fausse. Et dans les runs DÉJÀ ENREGISTRÉS (aucun run nouveau —
lecture de `outputs/f1/`), elle est fausse partout :
    ma_quater : stables=False, tailles 447 965 → 844 768 (groupe 0)
                stables=False, tailles 222 755 → 1 064 750 (groupe 1)
    borne_l3  : stables=False, facteurs 2.3× et 2.3× sur 330 frames
Ce n'est donc pas un cas limite : à chaque frame où le compte croît, des
centaines de milliers de coefficients sont perdus définitivement ; à
chaque frame où il décroît, autant de coefficients périmés sont
réappliqués.

────────────────────────────────────────────────────────────────────────
CE QUE CELA IMPLIQUE POUR LE SCHÉMA DE PRODUCTION, PAS SEULEMENT LE
HARNAIS
────────────────────────────────────────────────────────────────────────
La garantie de convergence de B4 repose entièrement sur « rien n'est
perdu, les résidus repassent le seuil ». Elle est VIDE sous ce transfert :
l'erreur du modèle lointain n'est bornée ni par EPS ni par la cadence,
puisqu'elle s'accumule par un chemin que ni l'un ni l'autre ne contrôle.
Toute lecture de fidélité du lointain (fovéa-z) faite sur ce transfert
mesure ce défaut en plus du schéma.

La cause commune des deux défauts tient en une phrase : **la TAILLE vient
d'une frame et les DONNÉES d'une autre.** Tout correctif doit les faire
venir de la même.

CORRECTIF NATUREL, NON CONSTRUIT (décision de Romain) : des buffers
d'émission en PING-PONG. Le kernel de la frame n écrit dans le jeu
n mod 2 ; la frame n+1 transfère le jeu de la frame n avec le compteur de
la frame n — taille et données du même tour, sans aucune synchronisation
ajoutée, donc le choix 6 est préservé.
  COÛT : +120 Mio de VRAM préallouée (les buffers actuels pèsent 64 + 56
  Mio pour la géométrie V4 : 8 388 608 et 7 340 032 éléments à 4+4 octets),
  et une frame de PLUS de péremption sur les coefficients — aujourd'hui
  les données transférées sont fraîches, seule leur taille est vieille.
  Ce second point n'est PAS gratuit : l'attribution (a), « le retard d'une
  frame suffit à lui seul », n'est pas armée. Échanger une perte
  définitive contre une frame de péremption est un arbitrage, pas une
  évidence.
Les autres voies sont plus chères ou changent le kernel : lire le
compteur en synchrone rétablit la synchronisation par frame que s2 a
guérie ; estampiller chaque coefficient d'un numéro de frame ajoute 4
octets par coefficient (+33 % de D2H) ET modifie le kernel L3.

INDÉPENDANCE À DIRE : ces deux défauts et le fork de réparation de la
RECONSTRUCTION (lire `reference` contre miroir CPU) sont DISJOINTS.
Corriger le transfert ne répare pas le décalage spatial du
reconstructeur ; réparer le reconstructeur ne rend pas au CPU les
coefficients que le transfert a perdus. Le fork n'est pas tranché ici et
rien de ce fichier ne le tranche."""
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


def _etat_initial(cp):
    """(B, 1, 4, n, n) f32 — un système suffit : le défaut examiné est
    dans le TRANSFERT, pas dans le motif de calcul."""
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
    """Une frame, avec la discipline de `_appliquer_f` reproduite À
    L'IDENTIQUE. Retourne (taille_utilisée, émis_cette_frame, transféré,
    tampon_apres) — les deux dernières en copies HÔTE.

    `int(compacteur.compteur[0])` est une lecture de TEST (elle synchronise) :
    la production ne la fait jamais, c'est tout l'objet du choix 6."""
    taille = compacteur.taille_precedente()
    compacteur.noter_taille(taille)
    pas_f_l3(q, cp, reference, compacteur, sortie=q,
             tampon_etage=cp.empty_like(q), eps=EPS_FIGE)
    emis = int(compacteur.compteur[0])
    transfere = None
    if taille:
        valeurs, indices = compacteur.vues_a_transferer(taille)
        transfere = (vers_cpu(valeurs).copy(), vers_cpu(indices).copy())
    tampon = (vers_cpu(compacteur.valeurs).copy(),
              vers_cpu(compacteur.indices).copy())
    compacteur.cloturer_frame()
    # Déterminisme du test : le retard devient EXACTEMENT d'une frame,
    # c'est-à-dire le cas le plus favorable au schéma.
    cp.cuda.Stream.null.synchronize()
    return taille, emis, transfere, tampon


def _cumuler(grand_livre, couples):
    """`np.add.at` sur les indices aplatis, exactement comme
    `ReconstructeurNiveau0.appliquer`."""
    if couples is None:
        return
    valeurs, indices = couples
    np.add.at(grand_livre, indices.astype(np.int64), valeurs)


def _grands_livres(total: int):
    """DEUX grands livres d'incréments, tenus en parallèle :

      - CÔTÉ DEVICE : ce que le kernel a réellement ajouté à `reference`
        (`reference[p] += d`), donc ce que le schéma AFFIRME que le CPU
        sait ;
      - CÔTÉ CPU : ce que le CPU a réellement appliqué.

    Le schéma B4 les veut IDENTIQUES. Cette comptabilité est immune aux
    références que le cas construit fabrique entre les frames : elle ne
    porte que sur les incréments émis et reçus."""
    return np.zeros(total), np.zeros(total)


def _sans_bruit(x, seuil: float = 1e-9) -> int:
    """Nombre d'entrées franchement non nulles — le signal construit vaut
    10·eps = 1e-3, mille fois au-dessus de ce seuil."""
    return int(np.count_nonzero(np.abs(x) > seuil))


# ----- (α) la troncature perd-elle DÉFINITIVEMENT ? -----

@gpu_requis
def test_alpha_la_troncature_perd_definitivement():
    """Cas construit : 5 émissions, puis 40, puis TROIS frames à 40.

    La frame 2 émet 40 alors que la frame 1 n'en avait émis que 5 : elle
    ne transfère que 5 couples sur 40. Les trois frames suivantes émettent
    autant que la 2 — elles ne tronquent donc RIEN et laissent au schéma
    toute latitude pour rattraper.

    ATTENDU si (α) est réel : la frame 2 creuse un écart de grand livre de
    35 couples, et AUCUNE des frames propres qui suivent ne le comble.
    ATTENDU si (α) est faux : l'écart se résorbe, les résidus « repassant
    le seuil » comme le nota l'affirme."""
    import cupy as cp
    q = _etat_initial(cp)
    total = taille_sortie_max(q)
    compacteur = CompacteurL3(cp, total)
    rng = np.random.default_rng(5)
    tirage = rng.permutation(total)
    petits, grands = tirage[:5], tirage[5:45]
    livre_device, livre_cpu = _grands_livres(total)

    reference = _reference_emettant(cp, q, petits)
    taille_1, emis_1, transfere_1, tampon_1 = _frame(cp, q, reference,
                                                     compacteur)
    assert (taille_1, emis_1) == (0, 5)      # frame 1 : aucun prédécesseur
    _cumuler(livre_device, (tampon_1[0][:emis_1], tampon_1[1][:emis_1]))
    _cumuler(livre_cpu, transfere_1)
    manque_1 = livre_device - livre_cpu

    reference[...] = _reference_emettant(cp, q, grands)
    taille_2, emis_2, transfere_2, tampon_2 = _frame(cp, q, reference,
                                                     compacteur)
    assert taille_2 == 5 and emis_2 == 40    # LA TRONCATURE A LIEU
    _cumuler(livre_device, (tampon_2[0][:emis_2], tampon_2[1][:emis_2]))
    _cumuler(livre_cpu, transfere_2)
    manque_2 = livre_device - livre_cpu

    # (α) EXHIBÉ : la frame 2 a fait `reference += d` pour 40 couples et
    # n'en a transféré que 5. Les 35 autres manquent au CPU...
    assert _sans_bruit(manque_2 - manque_1) == emis_2 - taille_2 == 35

    # ... et AUCUNE des frames suivantes ne les rattrape, bien qu'aucune
    # ne tronque : le détail est absorbé dans la référence, il ne
    # repassera plus le seuil. Trois frames propres, pas une seule.
    for _ in range(3):
        reference[...] = _reference_emettant(cp, q, grands)
        taille_n, emis_n, transfere_n, tampon_n = _frame(cp, q, reference,
                                                         compacteur)
        assert taille_n == emis_n == 40       # aucune troncature ici
        _cumuler(livre_device, (tampon_n[0][:emis_n], tampon_n[1][:emis_n]))
        _cumuler(livre_cpu, transfere_n)
        np.testing.assert_allclose(livre_device - livre_cpu, manque_2,
                                   rtol=0, atol=1e-12)
    assert _sans_bruit(manque_2) > 0


@gpu_requis
def test_alpha_sans_troncature_rien_nest_perdu():
    """Contre-épreuve indispensable : à tailles d'émission STABLES, le
    schéma est exact. Le défaut est bien la VARIABILITÉ des tailles, pas
    le retard d'une frame en lui-même — c'est ce que `diagnostic_retard`
    appelle `tailles_stables`."""
    import cupy as cp
    q = _etat_initial(cp)
    total = taille_sortie_max(q)
    compacteur = CompacteurL3(cp, total)
    rng = np.random.default_rng(7)
    emetteurs = rng.permutation(total)[:24]
    livre_device, livre_cpu = _grands_livres(total)

    reference = _reference_emettant(cp, q, emetteurs)
    _frame(cp, q, reference, compacteur)          # amorçage (taille 0)

    for _ in range(3):
        reference[...] = _reference_emettant(cp, q, emetteurs)
        taille, emis, transfere, tampon = _frame(cp, q, reference,
                                                 compacteur)
        assert taille == emis == 24
        _cumuler(livre_device, (tampon[0][:emis], tampon[1][:emis]))
        _cumuler(livre_cpu, transfere)
        np.testing.assert_allclose(livre_device - livre_cpu, 0.0,
                                   rtol=0, atol=1e-12)


# ----- (β) la queue de tampon est-elle retransférée ? -----

@gpu_requis
def test_beta_la_queue_de_tampon_est_reappliquee():
    """Cas construit : 40 émissions, 40, puis 5.

    La frame 3 n'émet que 5 couples mais transfère `taille` = 40 : les
    positions [5, 40) du tampon n'ont pas été réécrites.

    ATTENDU si (β) est réel : ces 35 couples sont BIT-IDENTIQUES à ceux de
    la frame 2, et le CPU les applique une SECONDE fois — son grand livre
    DÉPASSE celui du device. ATTENDU si (β) est faux : le transfert ne
    contient que les 5 couples émis par cette frame."""
    import cupy as cp
    q = _etat_initial(cp)
    total = taille_sortie_max(q)
    compacteur = CompacteurL3(cp, total)
    rng = np.random.default_rng(11)
    tirage = rng.permutation(total)
    grands, petits = tirage[:40], tirage[40:45]
    livre_device, livre_cpu = _grands_livres(total)

    reference = _reference_emettant(cp, q, grands)
    _frame(cp, q, reference, compacteur)          # amorçage (taille 0)

    reference[...] = _reference_emettant(cp, q, grands)
    taille_2, emis_2, transfere_2, tampon_2 = _frame(cp, q, reference,
                                                     compacteur)
    assert taille_2 == emis_2 == 40               # frame propre
    _cumuler(livre_device, (tampon_2[0][:emis_2], tampon_2[1][:emis_2]))
    _cumuler(livre_cpu, transfere_2)
    exces_2 = livre_cpu - livre_device

    reference[...] = _reference_emettant(cp, q, petits)
    taille_3, emis_3, transfere_3, tampon_3 = _frame(cp, q, reference,
                                                     compacteur)
    assert taille_3 == 40 and emis_3 == 5         # la frame émet MOINS
    _cumuler(livre_device, (tampon_3[0][:emis_3], tampon_3[1][:emis_3]))
    _cumuler(livre_cpu, transfere_3)
    exces_3 = livre_cpu - livre_device

    # (β) EXHIBÉ, d'abord sur les octets transférés : la queue est celle
    # de la frame précédente, à l'identique.
    valeurs_3, indices_3 = transfere_3
    np.testing.assert_array_equal(valeurs_3[emis_3:], tampon_2[0][emis_3:40])
    np.testing.assert_array_equal(indices_3[emis_3:], tampon_2[1][emis_3:40])
    # ... puis sur la conséquence : le CPU a appliqué 35 incréments que
    # cette frame n'a PAS émis. Son grand livre dépasse celui du device.
    assert _sans_bruit(exces_3 - exces_2) == taille_3 - emis_3 == 35


@gpu_requis
def test_beta_le_compteur_est_remis_a_zero_mais_pas_les_buffers():
    """La cause immédiate, isolée : `cloturer_frame` remet le COMPTEUR à
    zéro et laisse les buffers tels quels. C'est légitime en soi — les
    réécrire coûterait une passe mémoire par frame — mais cela rend la
    queue lisible, et c'est le transfert dimensionné sur la frame
    précédente qui la lit."""
    import cupy as cp
    q = _etat_initial(cp)
    compacteur = CompacteurL3(cp, taille_sortie_max(q))
    emetteurs = np.arange(16)
    reference = _reference_emettant(cp, q, emetteurs)
    _frame(cp, q, reference, compacteur)

    avant = vers_cpu(compacteur.valeurs).copy()
    compacteur.cloturer_frame()
    cp.cuda.Stream.null.synchronize()
    assert int(compacteur.compteur[0]) == 0                  # compteur remis
    np.testing.assert_array_equal(                        # buffers intacts
        vers_cpu(compacteur.valeurs), avant)


# ----- ce que le nota endossé affirme, et ce que les cas montrent -----

def test_le_nota_endosse_affirme_que_rien_nest_perdu():
    """Le nota vit dans `diagnostic_retard` et dans le docstring de
    `CompacteurL3`. Ce test le CITE sans le corriger : la correction du
    nota est une décision de Romain, pas un effet de bord de test."""
    from src.f1_gpu.substrat_l3 import CompacteurL3 as classe
    assert "rien n'est perdu définitivement" in classe.__doc__
    if cupy_disponible():
        import cupy as cp
        compacteur = CompacteurL3(cp, 64)
        assert "rien n'est perdu" in compacteur.diagnostic_retard()["note"]


def test_la_condition_qui_declenche_les_deux_defauts_est_deja_reportee():
    """`tailles_stables` est EXACTEMENT le prédicat qui décide : tailles
    stables ⇒ ni troncature ni queue ; instables ⇒ l'un des deux à chaque
    changement de taille. Le diagnostic le reporte déjà — c'est sa
    LECTURE qui est fausse, pas sa mesure."""
    if not cupy_disponible():
        pytest.skip("cupy/device CUDA indisponible")
    import cupy as cp
    compacteur = CompacteurL3(cp, 64)
    for taille in (12, 12, 12):
        compacteur.noter_taille(taille)
    assert compacteur.diagnostic_retard()["tailles_stables"] is True
    compacteur.noter_taille(30)
    assert compacteur.diagnostic_retard()["tailles_stables"] is False
