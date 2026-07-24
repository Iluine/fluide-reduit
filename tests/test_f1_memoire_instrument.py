"""Tests F1 — M-b tranche-2 : la sonde mémoire PERSISTÉE (§A31-run).

La leçon gravée du FAIL cloud non persisté s'applique ici : on PERSISTE les
états d'échec. Un processus tué par l'OOM killer ne lève rien, ne déroule
aucun `finally`, n'écrit aucun `except` — tout ce qui n'est pas déjà SUR LE
DISQUE au moment du SIGKILL est perdu.

Ce que ces tests protègent :
  1. le jalon de DÉBUT est sur disque AVANT que la phase ne travaille —
     c'est le seul moyen de savoir quelle phase tenait le processus ;
  2. le pic est échantillonné et persisté PENDANT la phase, pas seulement à
     sa fin (une phase qui meurt n'a pas de fin) ;
  3. le pic est attribué PAR BRAS ET PAR PHASE ;
  4. la sonde fonctionne sans cupy (le poste suspect est la RAM hôte)."""
import json

from src.f1_gpu.memoire_instrument import (
    SondeMemoire,
    lire_jalons,
    pic_par_phase,
)


def _chemin(tmp_path):
    return tmp_path / "memoire.jsonl"


# ----- 1. le début est persisté AVANT le travail -----

def test_le_jalon_de_debut_est_sur_disque_pendant_la_phase(tmp_path):
    """Si le processus meurt DANS la phase, le disque doit déjà dire
    laquelle. Vérifié pendant, pas après."""
    chemin = _chemin(tmp_path)
    sonde = SondeMemoire(chemin)
    with sonde.phase("rederive", "seed=103"):
        jalons = lire_jalons(chemin)
        assert jalons, "rien sur disque pendant la phase"
        assert jalons[0]["bras"] == "rederive"
        assert jalons[0]["phase"] == "seed=103"
        assert jalons[0]["etat"] == "debut"


def test_la_fin_est_persistee_aussi(tmp_path):
    chemin = _chemin(tmp_path)
    sonde = SondeMemoire(chemin)
    with sonde.phase("production", "seed=101"):
        pass
    etats = [j["etat"] for j in lire_jalons(chemin)]
    assert "debut" in etats and "fin" in etats


def test_une_phase_qui_leve_persiste_quand_meme_son_echec(tmp_path):
    """Une exception (MemoryError contenue, par ex.) laisse une trace
    exploitable — l'échec est un résultat, pas un trou."""
    chemin = _chemin(tmp_path)
    sonde = SondeMemoire(chemin)
    try:
        with sonde.phase("rederive", "seed=103"):
            raise MemoryError("simulacre")
    except MemoryError:
        pass
    fins = [j for j in lire_jalons(chemin) if j["etat"] == "fin"]
    assert fins and fins[-1]["exception"] == "MemoryError"


# ----- 2. le pic est échantillonné PENDANT -----

def test_le_pic_est_echantillonne_pendant_la_phase(tmp_path):
    """Le battement de coeur écrit le pic courant sans attendre la fin :
    un SIGKILL laisse le dernier échantillon."""
    chemin = _chemin(tmp_path)
    sonde = SondeMemoire(chemin, periode_s=0.01)
    with sonde.phase("rederive", "gonflement"):
        bloc = bytearray(80 * 1024 * 1024)      # +80 Mo, touchés
        for i in range(0, len(bloc), 4096):
            bloc[i] = 1
        sonde.attendre_un_echantillon()
        battements = [j for j in lire_jalons(chemin) if j["etat"] == "vivant"]
        assert battements, "aucun battement persisté pendant la phase"
        assert battements[-1]["rss_pic_mo"] > 0
        del bloc


def test_le_pic_de_phase_voit_le_gonflement(tmp_path):
    chemin = _chemin(tmp_path)
    sonde = SondeMemoire(chemin, periode_s=0.01)
    with sonde.phase("temoin", "repos"):
        sonde.attendre_un_echantillon()
    repos = sonde.dernier_pic_mo
    with sonde.phase("rederive", "gonflement"):
        bloc = bytearray(120 * 1024 * 1024)
        for i in range(0, len(bloc), 4096):
            bloc[i] = 1
        sonde.attendre_un_echantillon()
        del bloc
    assert sonde.dernier_pic_mo > repos + 50      # ~120 Mo, marge large


# ----- 3. attribution PAR BRAS ET PAR PHASE -----

def test_pic_attribue_par_bras_et_par_phase(tmp_path):
    chemin = _chemin(tmp_path)
    sonde = SondeMemoire(chemin, periode_s=0.01)
    for bras in ("rederive", "production"):
        with sonde.phase(bras, "seed=101"):
            sonde.attendre_un_echantillon()
    pics = pic_par_phase(lire_jalons(chemin))
    assert ("rederive", "seed=101") in pics
    assert ("production", "seed=101") in pics
    assert pics[("rederive", "seed=101")]["rss_pic_mo"] > 0


def test_jalons_sont_du_jsonl_ligne_par_ligne(tmp_path):
    """JSON *lines* à dessein : un fichier tronqué par un SIGKILL au milieu
    d'une écriture garde toutes les lignes précédentes lisibles — ce qu'un
    JSON unique ne permettrait pas."""
    chemin = _chemin(tmp_path)
    sonde = SondeMemoire(chemin)
    with sonde.phase("rederive", "a"):
        pass
    lignes = chemin.read_text(encoding="utf-8").strip().splitlines()
    assert len(lignes) >= 2
    for ligne in lignes:
        json.loads(ligne)


def test_lire_jalons_tolere_une_derniere_ligne_tronquee(tmp_path):
    """Exactement le cas SIGKILL : la dernière écriture est coupée."""
    chemin = _chemin(tmp_path)
    sonde = SondeMemoire(chemin)
    with sonde.phase("rederive", "a"):
        pass
    with open(chemin, "a", encoding="utf-8") as f:
        f.write('{"bras": "rederive", "etat": "viv')     # tronqué
    jalons = lire_jalons(chemin)
    assert jalons and all("etat" in j for j in jalons)


# ----- 4. sans cupy -----

def test_sonde_sans_cupy(tmp_path):
    """Le poste suspect est la RAM HÔTE : la sonde doit marcher sans GPU."""
    chemin = _chemin(tmp_path)
    sonde = SondeMemoire(chemin, cp=None)
    with sonde.phase("rederive", "a"):
        pass
    j = lire_jalons(chemin)[-1]
    assert j["vram_pool_mo"] is None
    assert j["rss_mo"] > 0
