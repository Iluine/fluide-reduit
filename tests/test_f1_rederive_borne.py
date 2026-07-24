"""Tests F1 — M-b tranche-2 : consommer le rederive SANS LE MODIFIER
(§A31-run).

Le rederive déborde (25 Go au seed 103, épisode 4) et il est INTOUCHÉ. La
consigne gravée est donc : dire comment le CONSOMMER autrement.

Le protocole : `_relax_episode` intègre jusqu'à `_T_END_RELAX = 130` puis
TRONQUE à `N_settle + 1` pas. Les pas retenus sont produits AVANT que `t_end`
ne joue — il ne gouverne que l'arrêt de boucle et l'écrêtage du dernier `dt`.
Donc consommer le rederive avec le PLUS PETIT `t_end` qui atteint `N_settle`
pas rend le MÊME résultat pour une fraction de la mémoire. L'échelle plafonne
à `_T_END_RELAX` : le domaine de résultats est identique, RuntimeError
compris.

Ce que ces tests protègent :
  1. RÉSULTAT IDENTIQUE — bit-pour-bit contre `run_episode`/`run_history`
     tels quels (c'est TOUTE la validité du protocole) ;
  2. AUCUNE FUITE GLOBALE — la constante est restaurée, exception comprise ;
  3. le plafond gravé n'est jamais dépassé ;
  4. l'épisode qui tuait le run se calcule."""
from dataclasses import replace

import numpy as np
import pytest

import src.sediment as sediment
from config import GridConfig
from src.f1_gpu.cellule_mb import centres_cellule
from src.f1_gpu.rederive_borne import (
    ECHELLE_T_END,
    ConsommateurRederive,
)
from src.sediment import (
    SedimentParams,
    _T_END_RELAX,
    default_terrain,
    run_episode,
    run_history,
)


def _terrain():
    return default_terrain(GridConfig())


# ----- 1. résultat IDENTIQUE : toute la validité du protocole -----

def test_episode_borne_est_bit_exact_contre_run_episode():
    """Le coeur : consommer avec un `t_end` réduit rend EXACTEMENT ce que
    `run_episode` rend aujourd'hui. Sans cela, le protocole n'a aucune
    valeur — il changerait la vérité-sol."""
    b0 = _terrain()
    params = SedimentParams()
    centre = centres_cellule(101, 1)[0]
    attendu = run_episode(np.zeros_like(b0), b0, centre, params)
    obtenu = ConsommateurRederive(params).episode(
        np.zeros_like(b0), b0, centre)
    np.testing.assert_array_equal(obtenu, attendu)


def test_histoire_bornee_est_bit_exacte_contre_run_history():
    """Même exigence au niveau de l'histoire, checkpoints compris."""
    b0 = _terrain()
    params = replace(SedimentParams(), N_settle=40)
    attendu = run_history(101, 3, b0, params, checkpoints={2, 3})
    obtenu = ConsommateurRederive(params).histoire(
        b0, centres_cellule(101, 3), checkpoints={2, 3})
    assert set(obtenu) == set(attendu)
    for n in attendu:
        np.testing.assert_array_equal(obtenu[n], attendu[n])


def test_le_t_end_retenu_est_le_plus_petit_qui_suffit():
    """L'échelle s'arrête au premier `t_end` qui atteint N_settle pas — pas
    au premier qui « marche à peu près »."""
    b0 = _terrain()
    params = replace(SedimentParams(), N_settle=10)
    c = ConsommateurRederive(params)
    c.episode(np.zeros_like(b0), b0, centres_cellule(101, 1)[0])
    assert c.t_end_courant in ECHELLE_T_END
    i = ECHELLE_T_END.index(c.t_end_courant)
    if i > 0:                       # le précédent devait être insuffisant
        with pytest.raises(RuntimeError):
            c_bas = ConsommateurRederive(params,
                                         echelle=(ECHELLE_T_END[i - 1],))
            c_bas.episode(np.zeros_like(b0), b0, centres_cellule(101, 1)[0])


# ----- 2. aucune fuite globale -----

def test_la_constante_globale_est_restauree():
    b0 = _terrain()
    params = replace(SedimentParams(), N_settle=10)
    ConsommateurRederive(params).episode(np.zeros_like(b0), b0, (0.5, 0.5))
    assert sediment._T_END_RELAX == _T_END_RELAX


def test_la_constante_est_restauree_meme_si_ca_leve():
    b0 = _terrain()
    params = replace(SedimentParams(), N_settle=10)
    c = ConsommateurRederive(params, echelle=(0.001,))
    with pytest.raises(RuntimeError):
        c.episode(np.zeros_like(b0), b0, (0.5, 0.5))
    assert sediment._T_END_RELAX == _T_END_RELAX


# ----- 3. le plafond gravé -----

def test_l_echelle_plafonne_au_t_end_grave():
    """Ne JAMAIS dépasser 130 : au-delà, le domaine de résultats ne serait
    plus celui du rederive (un épisode que l'original refuse doit rester
    refusé)."""
    assert ECHELLE_T_END[-1] == _T_END_RELAX
    assert all(a < b for a, b in zip(ECHELLE_T_END, ECHELLE_T_END[1:]))


def test_un_episode_impossible_leve_comme_l_original():
    """Si même le plafond ne suffit pas, on lève — on ne rend pas un
    résultat tronqué."""
    b0 = _terrain()
    params = replace(SedimentParams(), N_settle=10_000_000)
    with pytest.raises(RuntimeError):
        ConsommateurRederive(params).episode(np.zeros_like(b0), b0,
                                             (0.5, 0.5))


def test_le_barreau_redescend_entre_episodes():
    """RÉGRESSION, mesurée : au seed 103 l'épisode 3 exige t_end=130 et
    l'épisode 4 se contente de 64. Un mémo monotone ferait hériter le 130 et
    tuerait le processus — le besoin n'est PAS monotone le long d'une
    histoire."""
    b0 = _terrain()
    c = ConsommateurRederive(SedimentParams())
    s = np.zeros_like(b0)
    centres = centres_cellule(103, 4)
    for ep in range(4):
        s = c.episode(s, b0, centres[ep])
    retenus = [b["t_end"] for b in c.barreaux]
    assert retenus[2] > retenus[3], f"barreaux={retenus}"


# ----- 3bis. LA GARDE STRUCTURELLE (§A31-diagnostic, amendement 1) -----

def test_la_garde_structurelle_est_verifiee_a_chaque_episode():
    """La garde lit le temps du DERNIER PAS RETENU et l'exige strictement
    sous `t_end − marge`. Elle ne compare à AUCUNE exécution non bornée —
    c'est tout son intérêt : là où le risque est maximal (l'épisode qui
    OOM), l'exécution non bornée est justement impossible."""
    b0 = _terrain()
    params = replace(SedimentParams(), N_settle=10)
    c = ConsommateurRederive(params)
    c.episode(np.zeros_like(b0), b0, (0.5, 0.5))
    g = c.gardes[-1]
    assert g["t_dernier_retenu"] + g["marge"] <= g["t_end"]
    assert g["ok"] is True


def test_la_garde_rejette_un_barreau_ou_la_fenetre_touche_t_end():
    """Cas construit sur des nombres MESURÉS : à N_settle=10 et t_end=2,0 le
    dernier pas retenu tombe à t=1,9331 pour une marge de ~0,19 — la fenêtre
    frôle `t_end`, donc le barreau est REJETÉ et l'escalade continue.

    Ce cas montre aussi que la garde est STRICTEMENT PLUS FORTE que la sonde
    échantillonnée qu'elle remplace : celle-ci acceptait ce barreau (11 pas
    disponibles ≥ N_settle+1)."""
    b0 = _terrain()
    params = replace(SedimentParams(), N_settle=10)
    c = ConsommateurRederive(params, echelle=(2.0, 2.5, 4.0))
    c.episode(np.zeros_like(b0), b0, (0.5, 0.5))
    assert c.t_end_courant > 2.0, "le barreau 2,0 aurait dû être rejeté"
    rejets = [g for g in c.gardes if not g["ok"]]
    assert rejets and rejets[0]["t_end"] == 2.0


def test_la_fenetre_retenue_ne_depend_pas_de_t_end():
    """Le fondement du protocole, vérifié directement : le temps du dernier
    pas retenu est le MÊME à trois `t_end` distincts."""
    b0 = _terrain()
    params = replace(SedimentParams(), N_settle=10)
    temps = []
    for t_end in (2.5, 4.0, 8.0):
        c = ConsommateurRederive(params, echelle=(t_end,))
        c.episode(np.zeros_like(b0), b0, (0.5, 0.5))
        temps.append(c.gardes[-1]["t_dernier_retenu"])
    assert temps[0] == temps[1] == temps[2]


def test_au_plafond_grave_aucune_garde_n_est_requise():
    """Au barreau plafond, l'appel EST l'original (même `t_end` gravé) : le
    borner n'a plus de sens, et refuser là où l'original accepte créerait une
    divergence de domaine. La garde ne s'applique qu'aux barreaux BORNÉS."""
    b0 = _terrain()
    params = replace(SedimentParams(), N_settle=10)
    c = ConsommateurRederive(params, echelle=(_T_END_RELAX,))
    c.episode(np.zeros_like(b0), b0, (0.5, 0.5))
    g = c.gardes[-1]
    assert g["ok"] is True and g["plafond"] is True


def test_la_garde_ne_double_pas_le_calcul():
    """La garde lit les temps de l'appel RÉEL (un seul par barreau) — elle
    ne relance pas l'épisode comme le faisait la sonde échantillonnée."""
    b0 = _terrain()
    params = replace(SedimentParams(), N_settle=10)
    c = ConsommateurRederive(params, echelle=(4.0,))
    c.episode(np.zeros_like(b0), b0, (0.5, 0.5))
    assert c.appels == 1


# ----- 4. l'épisode qui tuait le run -----

def test_l_episode_fatal_seed_103_se_calcule():
    """Seed 103 épisode 4 : l'épisode qui demandait 25 Go et tuait le
    processus. Il doit se calculer, et sous une borne mémoire franche."""
    import resource
    b0 = _terrain()
    params = SedimentParams()
    c = ConsommateurRederive(params)
    s = np.zeros_like(b0)
    centres = centres_cellule(103, 4)
    avant = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    for ep in range(4):
        s = c.episode(s, b0, centres[ep])
    apres = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    assert np.isfinite(s).all()
    assert apres - avant < 2000        # Mo : très loin des 25 Go
