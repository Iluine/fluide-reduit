"""Manche 2 (§A13) — Task 0 : cœur du harnais registre-commis.

Mission « registre-commis » : mesurer si une chaîne de commits quadtree (budget
`k`) + replay déterministe reste indistinguable (sous JND) du monde vrai le
long d'émissions successives. Ce module est le cœur INSTRUMENTAL — aucune
mesure de verdict ici (les JND vivent dans `outputs/arcC/pins_spatial.json`,
consommés PLUS TARD par les tâches suivantes de la manche, pas dans ce module).

Protocole §A13-1 (gravé), citation : « à chaque émission t_i : mesurer Δχ,
committer, ré-ancrer, continuer ». Émissions aux épisodes t_i = i·Δt,
i = 1..n_émissions ; t₀ = état vierge s = 0 (convention `run_history`,
src/sediment.py) — la chaîne-vérité EST l'histoire gelée du seed : mêmes
centres, même `run_episode`, aucune divergence entre moteur et vérité tant
qu'aucun commit n'a encore touché l'état-moteur (c'est pourquoi dchi_1 = 0.0
EXACT, cf. `chaine_registre`).

Le protocole par émission, ordre STRICT :
  1. mesurer : Δχ entre le readout albedo de l'état-moteur et celui du monde
     vrai, AVANT tout commit — le readout de chaîne est
     `float(delta_chi(albedo(etat_moteur, S_HALF_OP), albedo(etat_vrai,
     S_HALF_OP))["max_carrier"])`, la référence est TOUJOURS le monde vrai ;
  2. committer : compression/copie de l'ÉTAT-MOTEUR (jamais du vrai — c'est ce
     qui rend la composition d'erreur possible, l'objet du claim de la
     manche) ;
  3. ré-ancrer : l'état-moteur devient la reconstruction du commit ;
  4. continuer : moteur ET vérité évoluent de Δt épisodes via `run_episode`,
     avec les MÊMES centres (issus du même `seed`) — la vérité ne voit jamais
     le commit.

Deux bras de commit (`chaine_registre(..., arm=...)`) :
  - "v2"   : `summarize_qt(etat_moteur, k)` / `regenerate_qt` (quadtree
    adaptatif, `src/summary_quadtree.py`) — budget `k`, garde
    `size_floats_qt(commit) <= k` (RuntimeError sinon, défensif : déjà garanti
    par le contrat de `summarize_qt`).
  - "ferm" : copie exacte du champ complet (contrôle §A13-4-1, sans perte —
    doit fermer Δχ=0.0 EXACT à toutes les émissions, quel que soit `k` : c'est
    le test d'intégration du protocole, pas un budget réaliste).

Aucune nouvelle constante de seuil n'est définie ici : seules les ancres déjà
gravées (`B0`, `GRID`, `PARAMS`, `RELIEF`, `S_HALF_OP`, importées de
`scripts.run_arcA_revalidate`, mêmes valeurs que `run_arcA_measure_qt.py`
lignes 106-111) et le motif de tirage des centres (`_draw_rollout_pairs`,
importé de `scripts.run_arcA_measure`, réutilisé VERBATIM) sont réutilisées."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.run_arcA_measure import _draw_rollout_pairs
from scripts.run_arcA_revalidate import B0, GRID, PARAMS, RELIEF, S_HALF_OP  # noqa: F401
from src.albedo import albedo, delta_chi
from src.sediment import run_episode
from src.summary_quadtree import regenerate_qt, size_floats_qt, summarize_qt

# Taille (floats-équivalents) du champ complet 64x64 -- valeur STRUCTURELLE
# (dérivée de l'ancre GRID), PAS un seuil : c'est `taille_commits` du bras
# "ferm" (contrôle de plomberie sans perte, hors comptabilité clause-1).
_TAILLE_CHAMP_COMPLET: int = GRID.H * GRID.W

_ARMS: tuple[str, ...] = ("v2", "ferm")


def centres_episodes(seed: int, n_episodes: int) -> tuple[tuple[float, float], ...]:
    """Wrapper mince sur `_draw_rollout_pairs(seed, n_episodes)`
    (scripts/run_arcA_measure.py, motif VERBATIM de `run_history` :
    `default_rng(seed)` FRAIS, tirages (x, y) i.i.d. uniformes dans
    [0.15, 0.85]², x puis y par épisode). Le centre de l'épisode t (1-indexé)
    est la t-ième paire : `centres_episodes(seed, n)[t - 1]`."""
    return tuple(_draw_rollout_pairs(seed, n_episodes))


def chaine_verite(seed: int, n_episodes: int) -> dict[int, np.ndarray]:
    """Monde vrai plein-état : simule s=0 -> n_episodes via `run_episode` et
    les centres de `centres_episodes(seed, n_episodes)`, et retourne l'état à
    CHAQUE épisode 1..n_episodes (copies float64). C'est l'histoire gelée du
    seed -- bit-identique à ce que produirait `run_history(seed, n_episodes,
    B0, PARAMS, checkpoints=set(range(1, n_episodes + 1)))` (mêmes ancres
    B0/PARAMS, mêmes centres, même `run_episode`)."""
    centres = centres_episodes(seed, n_episodes)
    s = np.zeros_like(B0, dtype=np.float64)
    out: dict[int, np.ndarray] = {}
    for i, centre in enumerate(centres, start=1):
        s = run_episode(s, B0, centre, PARAMS)
        out[i] = np.array(s, dtype=np.float64, copy=True)
    return out


def chaine_registre(seed: int, dt: int, n_emissions: int, k: int, arm: str = "v2",
                    verite: dict[int, np.ndarray] | None = None) -> dict:
    """Le protocole gravé (cf. docstring de tête), sur `n_emissions` émissions
    aux épisodes t_i = i·dt.

    `verite` (optionnel) : dict précalculé (typiquement
    `chaine_verite(seed, n_emissions * dt)`) donnant l'état vrai à CHAQUE
    épisode -- permet de PARTAGER le monde vrai entre plusieurs appels (par
    ex. plusieurs budgets `k`, ou plusieurs bras) sans le resimuler : si
    fourni, il est utilisé TEL QUEL (`verite[t_i]`), jamais resimulé. Si
    `None` (défaut), la vérité est simulée ICI, en lockstep avec le moteur
    (mêmes centres, mêmes segments) -- bit-identique à
    `chaine_verite(seed, n_emissions * dt)`.

    Retourne (dict, AUCUN timestamp) :
    `{"seed", "dt", "n_emissions", "k", "arm", "episodes_emissions",
    "delta_chi", "taille_commits", "registre", "readouts_moteur",
    "etats_vrais"}` -- cf. docstring de tête pour le détail de chaque clé
    (readouts_moteur = copies AVANT commit ; etats_vrais = copies aux mêmes
    émissions). Pour arm="ferm", `taille_commits` = [4096, ...] (champ
    complet ; contrôle de plomberie, pas une cellule du verdict).

    Garde : pour arm="v2", `size_floats_qt(commit) <= k` (RuntimeError sinon
    -- défensif, déjà garanti par le contrat de `summarize_qt`)."""
    if arm not in _ARMS:
        raise ValueError(f"chaine_registre : arm inconnu {arm!r} (attendu un de {_ARMS}).")

    n_ep_total = n_emissions * dt
    centres = centres_episodes(seed, n_ep_total)

    s_moteur = np.zeros_like(B0, dtype=np.float64)
    s_vrai = np.zeros_like(B0, dtype=np.float64) if verite is None else None

    episodes_emissions: list[int] = []
    delta_chi_list: list[float] = []
    taille_commits: list[int] = []
    registre: list = []
    readouts_moteur: list[np.ndarray] = []
    etats_vrais: list[np.ndarray] = []

    ep_fait = 0
    for i in range(1, n_emissions + 1):
        t_i = i * dt
        segment = centres[ep_fait:t_i]
        for centre in segment:
            s_moteur = run_episode(s_moteur, B0, centre, PARAMS)
            if verite is None:
                s_vrai = run_episode(s_vrai, B0, centre, PARAMS)
        ep_fait = t_i

        etat_vrai_i = verite[t_i] if verite is not None else s_vrai
        readout_moteur_i = np.array(s_moteur, dtype=np.float64, copy=True)

        # 1. mesurer -- AVANT tout commit (readout de chaîne, référence = vrai).
        dchi_i = float(delta_chi(albedo(readout_moteur_i, S_HALF_OP),
                                 albedo(etat_vrai_i, S_HALF_OP))["max_carrier"])

        # 2. committer -- l'état-MOTEUR (jamais le vrai).
        if arm == "v2":
            commit = summarize_qt(s_moteur, k)
            taille = size_floats_qt(commit)
            if taille > k:
                raise RuntimeError(
                    f"chaine_registre : commit hors budget à l'émission {i} "
                    f"(t_i={t_i}) -- taille={taille} > k={k}.")
            # 3. ré-ancrer.
            s_moteur = regenerate_qt(commit)
        else:  # arm == "ferm"
            commit = np.array(s_moteur, dtype=np.float64, copy=True)
            taille = _TAILLE_CHAMP_COMPLET
            # 3. ré-ancrer -- copie du commit (sans perte, contrôle §A13-4-1).
            s_moteur = np.array(commit, dtype=np.float64, copy=True)

        episodes_emissions.append(t_i)
        delta_chi_list.append(dchi_i)
        taille_commits.append(taille)
        registre.append(commit)
        readouts_moteur.append(readout_moteur_i)
        etats_vrais.append(np.array(etat_vrai_i, dtype=np.float64, copy=True))

        # 4. continuer : évoluer moteur ET vérité de dt épisodes, MÊMES
        # centres -- fait au TOUR SUIVANT de la boucle (segment ci-dessus, au
        # début de l'itération i+1), ou par le pré-chargement `verite` s'il
        # est fourni. Rien à faire ici explicitement.

    return {
        "seed": seed, "dt": dt, "n_emissions": n_emissions, "k": k, "arm": arm,
        "episodes_emissions": episodes_emissions,
        "delta_chi": delta_chi_list,
        "taille_commits": taille_commits,
        "registre": registre,
        "readouts_moteur": readouts_moteur,
        "etats_vrais": etats_vrais,
    }


def rederiver_emissions(registre: list, seed: int, dt: int, n_emissions: int,
                        arm: str = "v2") -> list[np.ndarray]:
    """Le foncteur de reconstruction É2/É3 : reconstruit les readouts-moteur de
    CHAQUE émission à partir de (`registre`, `seed`) UNIQUEMENT -- aucun accès
    au monde vrai ni aux états vivants d'un appel de `chaine_registre` (anti-
    fuite, cf. tests) :
      readout@t_1     = évolution de s=0 sur dt épisodes (centres du seed) ;
      readout@t_{i+1} = évolution de `regenerate_qt(registre[i-1])` (ou copie
                        si `arm == "ferm"`) de t_i à t_{i+1}, mêmes centres.
    Retourne la liste des `n_emissions` champs.

    SIGNATURE STRICTE : exactement ces paramètres (`registre`, `seed`, `dt`,
    `n_emissions`, `arm`) -- pas de paramètre « champ vrai », pas d'accès
    global aux états d'un autre appel de `chaine_registre` (testé par
    inspection de signature)."""
    if arm not in _ARMS:
        raise ValueError(f"rederiver_emissions : arm inconnu {arm!r} (attendu un de {_ARMS}).")

    n_ep_total = n_emissions * dt
    centres = centres_episodes(seed, n_ep_total)

    readouts: list[np.ndarray] = []

    s = np.zeros_like(B0, dtype=np.float64)
    for centre in centres[0:dt]:
        s = run_episode(s, B0, centre, PARAMS)
    readouts.append(np.array(s, dtype=np.float64, copy=True))

    for i in range(1, n_emissions):
        commit = registre[i - 1]
        if arm == "v2":
            s = regenerate_qt(commit)
        else:  # arm == "ferm"
            s = np.array(commit, dtype=np.float64, copy=True)
        segment = centres[i * dt:(i + 1) * dt]
        for centre in segment:
            s = run_episode(s, B0, centre, PARAMS)
        readouts.append(np.array(s, dtype=np.float64, copy=True))

    return readouts
