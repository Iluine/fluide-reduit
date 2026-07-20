"""Tests F1 — kernel L3 (F fusionné + émission des détails), achat A
(§A18-complément-2, pocCascade2phys d720da6).

LE RISQUE PROPRE À CE KERNEL : il réécrit le chemin de la remontée dans
du CUDA neuf. En omettre une part rendrait la borne artificiellement
basse — et ferait ouvrir l'achat B sur un chiffre faux. Les cinq verrous
du pré-enregistrement le ferment :
  1. l'état produit est BIT-IDENTIQUE à `pas_f_fusionne` (l'épilogue n'a
     pas le droit de changer F) ;
  2. l'ensemble émis est exactement celui de l'extraction CuPy — comparé
     TRIÉ, l'ordre atomique n'étant pas garanti, mais l'UNICITÉ DES
     INDICES est vérifiée AVANT le tri (consigne §A18-complément-2 : un
     tri masquerait une double émission) ;
  3. la référence est mise à jour à l'identique (schéma incrémental) ;
  4. le seuil est respecté aux deux bords ;
  5. le comptage est exact.
S'y ajoute l'équivalence STRUCTURELLE (le préfixe device vient bien du
fusionné) et le comportement du compacteur à retard d'une frame."""
import numpy as np
import pytest

from src.f1_gpu.backend import cupy_disponible, vers_cpu
from src.f1_gpu.substrat_fusionne import _SOURCE as _SOURCE_FUSIONNE
from src.f1_gpu.substrat_fusionne import pas_f_fusionne
from src.f1_gpu.substrat_jetable import etat_initial_jetable
from src.f1_gpu.substrat_l3 import (
    _PREFIXE_FUSIONNE,
    _SOURCE_L3,
    CompacteurL3,
    extraction_cupy_reference,
    pas_f_l3,
    taille_sortie_max,
)

gpu_requis = pytest.mark.skipif(
    not cupy_disponible(), reason="cupy/device CUDA indisponible")

EPS = 1e-4


def _etat_et_reference(cp, n_fenetres=2, n=12, graine=71, ecart=0.999):
    """État jetable + référence légèrement décalée, pour qu'une partie
    des coefficients dépasse le seuil et l'autre non."""
    q0 = etat_initial_jetable(n_fenetres, n, graine)
    return cp.asarray(q0), cp.asarray(q0) * cp.float32(ecart)


def _emission(compacteur, cp) -> tuple[np.ndarray, np.ndarray]:
    """(valeurs, indices) émis, rapatriés — hors chrono, pour les tests."""
    cp.cuda.runtime.deviceSynchronize()
    taille = int(compacteur.compteur[0])
    return (vers_cpu(compacteur.valeurs[:taille]),
            vers_cpu(compacteur.indices[:taille]))


# ----- équivalence STRUCTURELLE : le motif n'est pas recopié -----

def test_le_prefixe_device_vient_du_fusionne():
    """Les fonctions device (desing, minmod, lire, pentes_axe, flux_1d,
    divergence_axe) ne sont pas dupliquées : le source L3 REPREND le
    préfixe du fusionné. L'équivalence de motif est donc structurelle,
    pas surveillée."""
    marqueur = 'extern "C" __global__'
    assert _PREFIXE_FUSIONNE == _SOURCE_FUSIONNE[
        : _SOURCE_FUSIONNE.index(marqueur)]
    assert _SOURCE_L3.startswith(_PREFIXE_FUSIONNE)
    for fonction in ("desing", "minmod", "lire", "pentes_axe", "flux_1d",
                     "divergence_axe"):
        assert fonction in _PREFIXE_FUSIONNE


def test_le_kernel_m_a_prime_reste_intouche():
    """Le fusionné garde son empreinte : la borne M-a′ est citée, jamais
    re-mesurée ni modifiée.

    L'empreinte porte sur la CHAÎNE CUDA EXTRAITE (`_SOURCE`), PAS sur le
    fichier — c'est dit ici pour que la preuve soit re-calculable par qui
    l'exige (§A21)."""
    import hashlib
    assert len(_SOURCE_FUSIONNE) == 8223
    empreinte = hashlib.sha256(_SOURCE_FUSIONNE.encode()).hexdigest()
    assert empreinte == (
        "e18015f57e14263414239b18c51d25cd335250f58dde2ccb892a6e2c1d6f30b4")


def test_le_kernel_l3_a_son_propre_verrou_dempreinte():
    """VERROU AJOUTÉ, autorisé §A21. Le kernel L3 portait `reference += d`
    sans aucun verrou d'empreinte : seules l'équivalence structurelle et
    l'égalité bit-à-bit de l'état le tenaient. C'est lui qui décide de ce
    que le CPU saura ; il ne pouvait pas rester sans.

    Même forme que celui du fusionné : sha256 de la CHAÎNE CUDA EXTRAITE
    (`_SOURCE_L3`), pas du fichier — de sorte que le Python du module
    (buffers, comptabilité) puisse bouger sans que le CUDA ne bouge, et
    que ce soit PROUVÉ plutôt qu'affirmé."""
    import hashlib
    assert len(_SOURCE_L3) == 9635
    empreinte = hashlib.sha256(_SOURCE_L3.encode()).hexdigest()
    assert empreinte == (
        "9533a130b6624ce1b5b76d7d8e206aaae2e55fcc34e2a8d4eafa315e8014781a")


# ----- verrou 1 : l'état est BIT-IDENTIQUE au fusionné -----

@gpu_requis
@pytest.mark.parametrize("n_systemes", [1, 2])
def test_etat_bit_identique_au_fusionne(n_systemes):
    """L'épilogue n'a pas le droit de changer F."""
    import cupy as cp
    q0 = np.ascontiguousarray(
        etat_initial_jetable(3, 12, graine=73)[:, :n_systemes])
    attendu = cp.asarray(q0)
    pas_f_fusionne(attendu, cp, sortie=attendu,
                   tampon_etage=cp.empty_like(attendu))
    obtenu = cp.asarray(q0)
    reference = cp.asarray(q0) * cp.float32(0.999)
    compacteur = CompacteurL3(cp, taille_sortie_max(obtenu))
    pas_f_l3(obtenu, cp, reference, compacteur, sortie=obtenu,
             tampon_etage=cp.empty_like(obtenu), eps=EPS)
    assert np.array_equal(vers_cpu(obtenu), vers_cpu(attendu))


@gpu_requis
def test_etat_bit_identique_sur_plusieurs_pas():
    """Composition : un écart d'1 ULP au pas 1 se verrait amplifié."""
    import cupy as cp
    q0 = etat_initial_jetable(2, 10, graine=79)
    attendu, obtenu = cp.asarray(q0), cp.asarray(q0)
    reference = cp.asarray(q0) * cp.float32(0.999)
    tampon_a, tampon_o = cp.empty_like(attendu), cp.empty_like(obtenu)
    compacteur = CompacteurL3(cp, taille_sortie_max(obtenu))
    for _ in range(3):
        pas_f_fusionne(attendu, cp, sortie=attendu, tampon_etage=tampon_a)
        pas_f_l3(obtenu, cp, reference, compacteur, sortie=obtenu,
                 tampon_etage=tampon_o, eps=EPS)
        compacteur.cloturer_frame()
    assert np.array_equal(vers_cpu(obtenu), vers_cpu(attendu))


# ----- verrou 2 : l'ensemble émis, avec UNICITÉ AVANT TRI -----

@gpu_requis
def test_indices_emis_sont_uniques_avant_tout_tri():
    """CONSIGNE §A18-complément-2 : la comparaison triée pourrait masquer
    une DOUBLE ÉMISSION (deux threads écrivant le même élément, ou deux
    warps se chevauchant sur la même plage). Chaque élément n'émettant
    qu'une fois par frame par construction, l'unicité est exigible — et
    c'est elle qui prouve que l'agrégation de warp attribue des rangs
    disjoints."""
    import cupy as cp
    q, reference = _etat_et_reference(cp)
    compacteur = CompacteurL3(cp, taille_sortie_max(q))
    pas_f_l3(q, cp, reference, compacteur, sortie=q,
             tampon_etage=cp.empty_like(q), eps=EPS)
    _, indices = _emission(compacteur, cp)
    assert indices.size > 0                       # le test doit mordre
    assert np.unique(indices).size == indices.size


@gpu_requis
def test_ensemble_emis_egale_lextraction_cupy():
    """Verrou 2 : mêmes indices, mêmes valeurs que le schéma B4 actuel.
    Comparaison TRIÉE — dérogation isolée à ce chemin, l'ordre atomique
    n'étant pas garanti — précédée du contrôle d'unicité."""
    import cupy as cp
    q0 = etat_initial_jetable(2, 12, graine=71)
    reference_l3 = cp.asarray(q0) * cp.float32(0.999)
    reference_cupy = reference_l3.copy()

    etat_l3 = cp.asarray(q0)
    compacteur = CompacteurL3(cp, taille_sortie_max(etat_l3))
    pas_f_l3(etat_l3, cp, reference_l3, compacteur, sortie=etat_l3,
             tampon_etage=cp.empty_like(etat_l3), eps=EPS)
    valeurs_l3, indices_l3 = _emission(compacteur, cp)

    etat_cupy = cp.asarray(q0)
    pas_f_fusionne(etat_cupy, cp, sortie=etat_cupy,
                   tampon_etage=cp.empty_like(etat_cupy))
    valeurs_ref, indices_ref = extraction_cupy_reference(
        etat_cupy, reference_cupy, cp, eps=EPS)
    valeurs_ref, indices_ref = vers_cpu(valeurs_ref), vers_cpu(indices_ref)

    assert np.unique(indices_l3).size == indices_l3.size   # avant tri
    ordre_l3, ordre_ref = np.argsort(indices_l3), np.argsort(indices_ref)
    assert np.array_equal(indices_l3[ordre_l3], indices_ref[ordre_ref])
    assert np.array_equal(valeurs_l3[ordre_l3], valeurs_ref[ordre_ref])


# ----- verrou 3 : la référence est mise à jour à l'identique -----

@gpu_requis
def test_reference_incrementale_identique_au_schema_cupy():
    """Après le pas, la référence de L3 égale celle du schéma CuPy : le
    schéma reste incrémental, rien n'est perdu silencieusement."""
    import cupy as cp
    q0 = etat_initial_jetable(2, 12, graine=83)
    reference_l3 = cp.asarray(q0) * cp.float32(0.998)
    reference_cupy = reference_l3.copy()

    etat_l3 = cp.asarray(q0)
    compacteur = CompacteurL3(cp, taille_sortie_max(etat_l3))
    pas_f_l3(etat_l3, cp, reference_l3, compacteur, sortie=etat_l3,
             tampon_etage=cp.empty_like(etat_l3), eps=EPS)

    etat_cupy = cp.asarray(q0)
    pas_f_fusionne(etat_cupy, cp, sortie=etat_cupy,
                   tampon_etage=cp.empty_like(etat_cupy))
    extraction_cupy_reference(etat_cupy, reference_cupy, cp, eps=EPS)
    assert np.array_equal(vers_cpu(reference_l3), vers_cpu(reference_cupy))


# ----- verrou 4 : le seuil, aux deux bords -----

@gpu_requis
def test_seuil_infini_nemet_rien():
    import cupy as cp
    q, reference = _etat_et_reference(cp)
    compacteur = CompacteurL3(cp, taille_sortie_max(q))
    pas_f_l3(q, cp, reference, compacteur, sortie=q,
             tampon_etage=cp.empty_like(q), eps=1e30)
    _, indices = _emission(compacteur, cp)
    assert indices.size == 0


@gpu_requis
def test_seuil_nul_emet_tout_letat():
    """Seuil 0 : tout élément dont l'écart est non nul émet. Avec une
    référence décalée partout, c'est l'état entier."""
    import cupy as cp
    q, reference = _etat_et_reference(cp, ecart=0.5)
    attendu = taille_sortie_max(q)
    compacteur = CompacteurL3(cp, attendu)
    pas_f_l3(q, cp, reference, compacteur, sortie=q,
             tampon_etage=cp.empty_like(q), eps=0.0)
    _, indices = _emission(compacteur, cp)
    assert np.unique(indices).size == indices.size
    assert indices.size <= attendu
    # les zéros exacts (cellules sèches) ne dépassent pas |d| >= 0 ... si :
    # |0| >= 0 est vrai, donc tout élément émet.
    assert indices.size == attendu


# ----- verrou 5 : le comptage est exact -----

@gpu_requis
def test_compteur_egale_le_nombre_delements_au_dessus_du_seuil():
    import cupy as cp
    q0 = etat_initial_jetable(2, 12, graine=89)
    reference_l3 = cp.asarray(q0) * cp.float32(0.999)
    reference_cupy = reference_l3.copy()

    etat_l3 = cp.asarray(q0)
    compacteur = CompacteurL3(cp, taille_sortie_max(etat_l3))
    pas_f_l3(etat_l3, cp, reference_l3, compacteur, sortie=etat_l3,
             tampon_etage=cp.empty_like(etat_l3), eps=EPS)
    cp.cuda.runtime.deviceSynchronize()

    etat_cupy = cp.asarray(q0)
    pas_f_fusionne(etat_cupy, cp, sortie=etat_cupy,
                   tampon_etage=cp.empty_like(etat_cupy))
    _, indices_ref = extraction_cupy_reference(
        etat_cupy, reference_cupy, cp, eps=EPS)
    assert int(compacteur.compteur[0]) == int(indices_ref.size)


# ----- le compacteur : retard d'une frame -----

@gpu_requis
def test_compteur_lu_a_retard_dune_frame():
    """Choix 6 : la frame courante dimensionne son transfert sur la
    valeur de la frame PRÉCÉDENTE. À la première frame, rien n'est encore
    arrivé — le tampon vaut 0 (et la frame est dans le warmup)."""
    import cupy as cp
    q, reference = _etat_et_reference(cp)
    compacteur = CompacteurL3(cp, taille_sortie_max(q))
    assert compacteur.taille_precedente() == 0          # avant tout pas

    pas_f_l3(q, cp, reference, compacteur, sortie=q,
             tampon_etage=cp.empty_like(q), eps=EPS)
    emis = int(compacteur.compteur[0])
    compacteur.cloturer_frame()
    cp.cuda.runtime.deviceSynchronize()
    assert compacteur.taille_precedente() == emis        # arrivé pour la
    assert int(compacteur.compteur[0]) == 0              # suivante ; remis à 0


@gpu_requis
def test_diagnostic_retard_repose_sur_la_variabilite():
    """Le retard d'une frame se diagnostique par la VARIABILITÉ des
    tailles, pas par un cumul : chaque frame transférant ce que la
    précédente a émis, un cumul serait tautologique.

    DEPUIS §A21 (ping-pong), `tailles_stables` ne CONDITIONNE plus la
    complétude du transfert — il ne décrit plus que la variabilité de
    l'émission. Le diagnostic reste utile, sa portée a changé, et la note
    ne doit plus invoquer l'incrémentalité pour rassurer."""
    import cupy as cp
    q, _ = _etat_et_reference(cp)
    compacteur = CompacteurL3(cp, taille_sortie_max(q))
    for taille in (0, 120, 120, 120):
        compacteur.noter_taille(taille)
    compacteur.compteur_final = 120
    diagnostic = compacteur.diagnostic_retard()
    assert diagnostic["frames_observees"] == 4
    assert diagnostic["taille_min"] == 120        # les zéros sont écartés
    assert diagnostic["taille_max"] == 120
    assert diagnostic["tailles_stables"] is True
    assert diagnostic["compteur_final_non_transfere"] == 120
    assert "PING-PONG" in diagnostic["note"]
    assert "une fois et une seule" in diagnostic["note"]
    assert "rien n'est perdu" not in diagnostic["note"]   # nota RETIRÉ

    variable = CompacteurL3(cp, taille_sortie_max(q))
    for taille in (100, 180, 140):
        variable.noter_taille(taille)
    assert variable.diagnostic_retard()["tailles_stables"] is False


@gpu_requis
def test_taille_precedente_est_bornee_par_les_buffers():
    """Une valeur aberrante ne peut pas faire lire hors des buffers."""
    import cupy as cp
    q, _ = _etat_et_reference(cp)
    compacteur = CompacteurL3(cp, taille_sortie_max(q))
    compacteur.hote[0] = 10 ** 9
    assert compacteur.taille_precedente() == compacteur.taille_max


# ----- gardes fail-loud -----

@gpu_requis
def test_gardes_de_shape_et_dtype():
    import cupy as cp
    q, reference = _etat_et_reference(cp)
    compacteur = CompacteurL3(cp, taille_sortie_max(q))
    with pytest.raises(ValueError, match="float32"):
        pas_f_l3(q.astype(cp.float64), cp, reference, compacteur)
    for n_systemes in (0, 3):
        mauvais = cp.zeros((1, n_systemes, 4, 8, 8), dtype=cp.float32)
        with pytest.raises(ValueError, match="E4a"):
            pas_f_l3(mauvais, cp, cp.empty_like(mauvais), compacteur)


@gpu_requis
def test_garde_reference_de_meme_forme():
    import cupy as cp
    q, _ = _etat_et_reference(cp)
    compacteur = CompacteurL3(cp, taille_sortie_max(q))
    with pytest.raises(ValueError, match="référence"):
        pas_f_l3(q, cp, cp.zeros((1, 1, 4, 4, 4), dtype=cp.float32),
                 compacteur)


@gpu_requis
def test_garde_buffers_demission_au_pire_cas():
    """Dimensionner en dessous du pire cas est refusé (choix 5)."""
    import cupy as cp
    q, reference = _etat_et_reference(cp)
    with pytest.raises(ValueError, match="PIRE CAS"):
        pas_f_l3(q, cp, reference,
                 CompacteurL3(cp, taille_sortie_max(q) - 1))
