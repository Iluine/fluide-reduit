"""Task S1 — SONDE FENÊTRAGE-FOVÉA (§A14, texte gravé fait foi).

Citation (`PREREGISTRATION.md` §A14, pocCascade2phys, gravé commit `f08c340`
2026-07-18, AVANT toute ligne de code de sonde) :

    « Deux mécanismes que la manche 2 (émissions plein-domaine) ne pouvait
    pas voir : (1) contamination — entre émissions, la dynamique fait entrer
    dans la fenêtre de l'information venue du dehors non-contraint ;
    (2) couture — discontinuité au bord de fenêtre au ré-ancrage. Sonde =
    lecture remontée, AUCUN verdict, aucun branchement automatique. »

Cellule reconduite §A13-3 : seed 101, Δt = 4, 6 émissions ; substrat v2
gelé ; exécution machine-instrument (terminal natif). Fenêtre FIXE 32x32
alignée quadtree, quadrant (0,0) par défaut, garde anti-vacuité pré-écrite
(ordre FIXE (0,0)→(0,1)→(1,0)→(1,1), premier vivant retenu — règle
mécanique, zéro choix après lecture). Bras UNIQUE, NU (décision Romain
2026-07-18) : à chaque émission t_i — mesurer Δχ_fen, commit =
summarize(état∣fenêtre, k_fen), ré-ancrage DANS la fenêtre seulement ;
DEHORS le moteur continue SANS contrainte. La série Δχ_fen,i est l'objet de
la lecture.

Choix d'implémentation PRÉ-ENREGISTRÉS §A14 (endossés avec le texte gravé) :
  (i)   k_fen = 64 — aire-proportionnel au k* = 256 de la grille
        (32²/64² x 256) ;
  (ii)  ré-ancrage par remplacement de zone : la couture est ASSUMÉE
        (mécanisme mesuré, pas un défaut d'instrument) ;
  (iii) nouveau script consommant les primitives du cœur Task 0
        (`scripts/run_arcA_m2_registre.py` : `chaine_verite`,
        `centres_episodes`, ancres B0/GRID/PARAMS/S_HALF_OP ; substrat
        `summarize_qt`/`regenerate_qt`/`size_floats_qt` ; instruments
        `albedo`/`delta_chi` ; lecture `lecture_sonde`/`charge_pins_severe`
        de la sonde §A13-3) — cœur INTOUCHÉ. NOTE : `chaine_registre` /
        `rederiver_emissions` (nommés par le plan S1, rédigé AVANT le gravage
        §A14) implémentent le protocole PLEIN-DOMAINE et ne peuvent pas
        exprimer le protocole fenêtré — leurs signatures strictes anti-fuite
        sont RECONDUITES ici (`chaine_fenetre`,
        `rederiver_emissions_fenetre`), pas leurs corps ;
  (iv)  lecture au jnd_sev, formes de la sonde §A13-3 (S1 traverse / S2
        borné-contractant / AUTRE), Δχ_fen,1 ≡ 0 structurel exclu (aucun
        commit avant la première émission).

Choix d'implémentation NOMMÉS PAR CE BUILD (à endosser par Romain AVANT
tout run — points que le texte gravé fixe en intention mais pas en
mécanique) :
  (v)   COMMIT FENÊTRÉ PAR PLONGEMENT : le domaine de `summarize_qt` est
        gravé FIXE 64x64 (spec quadtree, intouchée). La fenêtre 32x32
        alignée quadtree est un bloc dyadique de profondeur 1 : le commit
        est `summarize_qt` du champ 64x64 valant l'état∣fenêtre dans la
        fenêtre et EXACTEMENT 0 dehors. Les 3 quadrants nuls sont constants
        (gain 0.0 exact, jamais splittés) : le commit ne porte AUCUNE
        information du dehors. Le budget k_fen = 64 s'applique au commit
        plongé ENTIER (surcoût structurel du plongement — racine + 3
        feuilles nulles + topologie, ~4 floats-éq — compté DANS k_fen :
        comptabilité conservatrice). Garde fail-loud à l'extraction : une
        reconstruction non nulle hors fenêtre est REJETÉE (RuntimeError).
  (vi)  GARDE ANTI-VACUITÉ OPÉRATIONNALISÉE : fenêtre vivante ssi
        Δχ(readout t₀∣fen, readout vrai@t_i∣fen)["max_carrier"] >= jnd_sev
        pour AU MOINS UNE émission t_i (readout t₀ = albedo(s=0) = 0
        partout ; la référence de `delta_chi` est le vrai@t_i, dont
        viennent les bandes porteuses). Quadrants essayés dans l'ordre
        FIXE, PREMIER vivant retenu, arrêt de l'essai au premier vivant ;
        AUCUN vivant -> RuntimeError (remonter, jamais mesurer).
  (vii) VÉRIFICATION D'INSTRUMENT OPÉRATIONNALISÉE (§A14 : « les bandes
        porteuses de max_carrier sont-elles définies sur 32x32 ? ») —
        AUCUNE simulation, uniquement structure + artefacts GELÉS :
        (a) structurel : chacune des 5 bandes radiales gravées possède au
        moins un bin de la grille FFT 32x32 ; (b) empirique gelé : bandes
        porteuses (critère gravé >= 10 % de la puissance AC, via
        `delta_chi(a, a)["carriers"]` — jamais réimplémenté) du readout
        vrai restreint à CHAQUE quadrant, sur les instantanés GELÉS s_L10
        et s_L20 de `outputs/arcA/histories/h160_s101.npz` (seed de la
        sonde, épisodes 10 et 20 ∈ [4, 24] la plage des émissions) ;
        (c) échelle : si toutes les porteuses plein-domaine sont des bandes
        plus grandes que la fenêtre (λ_max = 64/k_lo > 32 px, i.e. la seule
        bande "1"), la fenêtre ne peut pas les contenir. MUETTE ssi (a)
        échoue, ou aucune porteuse fenêtre sur aucun (instantané, quadrant),
        ou (c) — alors REMONTER avant toute mesure (RuntimeError dans
        `main`, aucun run).
  (viii) REDERIVATION FENÊTRÉE : signature STRICTE reconduite
        (`registre`, `seed`, `dt`, `n_emissions`, `quadrant`) — le moteur
        fenêtré est entièrement rejouable de (registre, seed) seuls : replay
        de s=0, remplacement de la fenêtre par l'extraction du commit à
        chaque émission, évolution Δt (le dehors non-contraint est
        déterministe du seed).

Escalade PRÉ-ÉCRITE (§A14, jamais improvisée) : si la série traverse
(contamination avérée), bras DEUX-ÉTAGES — commit fin dedans (k_fen = 64) +
commit grossier dehors (K_OUT_ESCALADE = 64, un niveau LOD d'écart par
unité d'aire), ré-ancrage des deux zones. Paramètres FIGÉS ici, AVANT toute
lecture ; le bras n'est PAS implémenté dans ce build (bras nu seul —
provenance §A14 : le remède embarqué masquerait le mécanisme à exposer).

Diagnostic non-verdictal : Δχ plein-domaine de la chaîne fenêtrée (chiffre
l'erreur totale, dehors non-contraint inclus).

Usage (machine-instrument, terminal natif, APRÈS endossement des choix
(v)-(viii)) :
  .venv/bin/python scripts/run_arcA_sonde_fenetre.py

POINT D'ARRÊT OBLIGATOIRE après le run : lecture remontée à Romain, aucun
enchaînement automatique."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.run_arcA_m2_registre import centres_episodes, chaine_verite
from scripts.run_arcA_m2_sonde import _comparaisons_ic, charge_pins_severe, lecture_sonde
from scripts.run_arcA_revalidate import B0, GRID, PARAMS, S_HALF_OP
from src.albedo import _BAND_BOUNDS, _BAND_LABELS, _radial_bins, albedo, delta_chi
from src.sediment import run_episode
from src.summary_quadtree import SummaryQT, regenerate_qt, size_floats_qt, summarize_qt

OUT_DIR = ROOT / "outputs" / "arcA"
OUT_JSON_PATH = OUT_DIR / "m2_sonde_fenetre.json"
OUT_PNG_PATH = OUT_DIR / "arcA_m2_sonde_fenetre.png"
H160_S101_PATH = OUT_DIR / "histories" / "h160_s101.npz"

# Paramètres FIGÉS §A14 -- ne pas paramétrer au-delà.
SEED_SONDE: int = 101
DT_SONDE: int = 4
N_EMISSIONS_SONDE: int = 6
COTE_FENETRE: int = 32
K_FEN: int = 64  # choix (i) : aire-proportionnel, 32²/64² x 256
# Escalade PRÉ-ÉCRITE §A14 (bras deux-étages, NON implémenté dans ce build) :
# figé AVANT toute lecture, ne se re-règle pas après.
K_OUT_ESCALADE: int = 64

# Ordre FIXE gravé de la garde anti-vacuité -- (qy, qx), indexation
# array[y, x] du dépôt : (0,0)=haut-gauche, (0,1)=haut-droite,
# (1,0)=bas-gauche, (1,1)=bas-droite.
QUADRANTS_ORDRE: tuple[tuple[int, int], ...] = ((0, 0), (0, 1), (1, 0), (1, 1))

# Instantanés GELÉS consommés par la vérification d'instrument (choix (vii)) :
# épisodes 10 et 20 de l'histoire gelée du seed de la sonde, dans la plage
# des émissions t ∈ [4, 24].
_SNAPSHOTS_INSTRUMENT: tuple[str, ...] = ("s_L10", "s_L20")


def fenetre_slices(quadrant: tuple[int, int]) -> tuple[slice, slice]:
    """Quadrant (qy, qx) de `QUADRANTS_ORDRE` -> (slice lignes, slice
    colonnes) du bloc dyadique 32x32 correspondant (alignement quadtree :
    profondeur 1 du domaine 64x64). Quadrant inconnu -> ValueError."""
    if quadrant not in QUADRANTS_ORDRE:
        raise ValueError(
            f"fenetre_slices : quadrant inconnu {quadrant!r} "
            f"(attendu un de {QUADRANTS_ORDRE}).")
    qy, qx = quadrant
    return (slice(qy * COTE_FENETRE, (qy + 1) * COTE_FENETRE),
            slice(qx * COTE_FENETRE, (qx + 1) * COTE_FENETRE))


def plonger_fenetre(champ_fen: np.ndarray, quadrant: tuple[int, int]) -> np.ndarray:
    """Choix (v) : champ 32x32 -> champ (GRID.H, GRID.W) EXACTEMENT nul hors
    fenêtre, copie du champ dans la fenêtre. Shape != 32x32 -> ValueError
    (fail-loud)."""
    champ_fen = np.asarray(champ_fen)
    if champ_fen.shape != (COTE_FENETRE, COTE_FENETRE):
        raise ValueError(
            f"plonger_fenetre : fenêtre fixe {COTE_FENETRE}x{COTE_FENETRE} "
            f"(reçu shape={champ_fen.shape}).")
    plein = np.zeros((GRID.H, GRID.W), dtype=np.float64)
    sy, sx = fenetre_slices(quadrant)
    plein[sy, sx] = champ_fen
    return plein


def commettre_fenetre(champ_fen: np.ndarray, quadrant: tuple[int, int],
                      k_fen: int) -> SummaryQT:
    """Commit fenêtré du protocole (§A14 : « commit = summarize(état∣fenêtre,
    k_fen) ») via le plongement (v) : `summarize_qt(plonger_fenetre(...),
    k_fen)`, budget k_fen sur le commit plongé ENTIER (surcoût structurel
    compté dedans). Garde : `size_floats_qt(commit) <= k_fen` (RuntimeError
    sinon -- défensif, déjà garanti par le contrat de `summarize_qt`)."""
    commit = summarize_qt(plonger_fenetre(champ_fen, quadrant), k_fen)
    taille = size_floats_qt(commit)
    if taille > k_fen:
        raise RuntimeError(
            f"commettre_fenetre : commit hors budget -- taille={taille} > "
            f"k_fen={k_fen}.")
    return commit


def extraire_fenetre(commit: SummaryQT, quadrant: tuple[int, int]) -> np.ndarray:
    """Fenêtre 32x32 de la reconstruction `regenerate_qt(commit)`, avec la
    garde fail-loud du choix (v) : si la reconstruction est non nulle HORS
    fenêtre, le commit porte de l'information du dehors -> RuntimeError
    (jamais de ré-ancrage depuis un commit contaminé)."""
    recon = regenerate_qt(commit)
    sy, sx = fenetre_slices(quadrant)
    dehors = np.array(recon, dtype=np.float64, copy=True)
    dehors[sy, sx] = 0.0
    if np.any(dehors != 0.0):
        raise RuntimeError(
            "extraire_fenetre : reconstruction non nulle HORS fenêtre -- le "
            f"commit porte de l'information du dehors (quadrant {quadrant}).")
    return np.array(recon[sy, sx], dtype=np.float64, copy=True)


def verifier_instrument_fenetre(chemin_histoire: str | Path = H160_S101_PATH) -> dict:
    """Vérification d'instrument §A14, DUE en build AVANT run (choix (vii)) --
    structure + artefacts GELÉS uniquement, AUCUNE simulation :
      (a) `bandes_definies_32` : chaque bande radiale gravée a >= 1 bin sur
          la grille FFT 32x32 ;
      (b) `porteuses_fenetre` : bandes porteuses (critère gravé de
          `delta_chi`, jamais réimplémenté -- lu via
          `delta_chi(a, a)["carriers"]`) du readout albedo vrai restreint à
          chaque quadrant, sur les instantanés gelés s_L10/s_L20 ;
      (c) `porteuses_plein_domaine` + `bande_plus_grande_que_fenetre`
          (λ_max = GRID.W / k_lo > COTE_FENETRE) : porteuses basses plus
          grandes que la fenêtre.

    `muette` (bool) ssi (a) échoue, ou aucune bande porteuse fenêtre sur
    aucun couple (instantané, quadrant), ou toutes les porteuses
    plein-domaine sont plus grandes que la fenêtre sur tous les
    instantanés ; `raisons` liste alors les causes -- REMONTER avant toute
    mesure, la sonde est muette PAR CONSTRUCTION."""
    k_int = _radial_bins((COTE_FENETRE, COTE_FENETRE))
    bandes_definies = {
        label: bool(np.any((k_int >= lo) & (k_int <= hi)))
        for label, (lo, hi) in zip(_BAND_LABELS, _BAND_BOUNDS)}
    bande_plus_grande = {
        label: bool((GRID.W / lo) > COTE_FENETRE)
        for label, (lo, _hi) in zip(_BAND_LABELS, _BAND_BOUNDS)}

    porteuses_plein: dict[str, dict[str, bool]] = {}
    porteuses_fenetre: dict[str, dict[str, dict[str, bool]]] = {}
    with np.load(chemin_histoire, allow_pickle=False) as data:
        for nom in _SNAPSHOTS_INSTRUMENT:
            etat = np.array(data[nom], dtype=np.float64, copy=True)
            readout = albedo(etat, S_HALF_OP)
            porteuses_plein[nom] = dict(delta_chi(readout, readout)["carriers"])
            porteuses_fenetre[nom] = {}
            for qy, qx in QUADRANTS_ORDRE:
                fen = readout[fenetre_slices((qy, qx))]
                porteuses_fenetre[nom][f"{qy},{qx}"] = dict(
                    delta_chi(fen, fen)["carriers"])

    raisons: list[str] = []
    if not all(bandes_definies.values()):
        absentes = [b for b, ok in bandes_definies.items() if not ok]
        raisons.append(
            f"bandes sans bin sur la grille 32x32 : {absentes}")
    aucune_porteuse_fenetre = all(
        not any(carriers.values())
        for par_quadrant in porteuses_fenetre.values()
        for carriers in par_quadrant.values())
    if aucune_porteuse_fenetre:
        raisons.append(
            "aucune bande porteuse dans la fenêtre sur les instantanés gelés "
            f"{list(_SNAPSHOTS_INSTRUMENT)}")
    plein_hors_fenetre = all(
        all(bande_plus_grande[b] for b, est_porteuse in carriers.items()
            if est_porteuse)
        and any(carriers.values())
        for carriers in porteuses_plein.values())
    if plein_hors_fenetre:
        raisons.append(
            "toutes les porteuses plein-domaine sont des bandes plus grandes "
            f"que la fenêtre ({COTE_FENETRE} px)")

    return {
        "bandes_definies_32": bandes_definies,
        "bande_plus_grande_que_fenetre": bande_plus_grande,
        "porteuses_plein_domaine": porteuses_plein,
        "porteuses_fenetre": porteuses_fenetre,
        "muette": bool(raisons),
        "raisons": raisons,
    }


def choisir_fenetre(verite: dict[int, np.ndarray], episodes: list[int],
                    jnd_sev: float) -> dict:
    """Garde anti-vacuité pré-écrite §A14, opérationnalisée par le choix
    (vi) : quadrants essayés dans l'ordre FIXE `QUADRANTS_ORDRE`, un
    quadrant est VIVANT ssi Δχ(readout t₀∣fen, readout vrai@t_i∣fen)
    ["max_carrier"] >= jnd_sev pour AU MOINS UNE émission t_i de `episodes`
    (readout t₀ = albedo(0) = 0 partout ; référence = vrai@t_i). PREMIER
    vivant retenu, essais arrêtés là (règle mécanique, zéro choix après
    lecture). AUCUN vivant -> RuntimeError (fenêtre morte partout = sonde
    triviale : remonter, jamais mesurer).

    Retourne `{"quadrant": (qy, qx), "detail": [{"quadrant", "vivante",
    "dchi_vs_t0"}, ...]}` (detail = quadrants ESSAYÉS, dans l'ordre)."""
    readout_t0_fen = albedo(
        np.zeros((COTE_FENETRE, COTE_FENETRE), dtype=np.float64), S_HALF_OP)
    detail: list[dict] = []
    for quadrant in QUADRANTS_ORDRE:
        sy, sx = fenetre_slices(quadrant)
        serie = []
        for t_i in episodes:
            ref = albedo(
                np.asarray(verite[t_i], dtype=np.float64)[sy, sx], S_HALF_OP)
            serie.append(float(delta_chi(readout_t0_fen, ref)["max_carrier"]))
        vivante = any(d >= jnd_sev for d in serie)
        detail.append({"quadrant": list(quadrant), "vivante": vivante,
                       "dchi_vs_t0": serie})
        if vivante:
            return {"quadrant": quadrant, "detail": detail}
    raise RuntimeError(
        "choisir_fenetre : AUCUN quadrant vivant (dynamique infra-JND vs t0 "
        "dans les 4 fenêtres) -- sonde triviale, remonter avant toute mesure "
        f"(§A14). Détail : {detail}")


def chaine_fenetre(seed: int, dt: int, n_emissions: int, k_fen: int,
                   quadrant: tuple[int, int],
                   verite: dict[int, np.ndarray] | None = None) -> dict:
    """Le protocole §A14 (bras unique, NU) : vérité et moteur partent du même
    t₀ (s = 0), mêmes seeds de forçage. À chaque émission t_i = i·dt :
      1. mesurer Δχ_fen(readout moteur∣fenêtre, readout vrai∣fenêtre) AVANT
         tout commit (+ diagnostic non-verdictal Δχ plein-domaine) ;
      2. commit = summarize(état-moteur∣fenêtre, k_fen)
         (`commettre_fenetre`, plongement (v) -- l'état-MOTEUR, jamais le
         vrai) ;
      3. ré-ancrage DANS la fenêtre seulement : état-moteur∣fenêtre <-
         extraction du commit ; DEHORS, le moteur continue SANS contrainte
         (la couture au bord est ASSUMÉE, choix (ii)) ;
      4. continuer : moteur ET vérité évoluent de dt épisodes
         (`run_episode`, mêmes centres du seed) -- la vérité ne voit jamais
         le commit.

    `verite` (optionnel) : dict précalculé (`chaine_verite(seed,
    n_emissions * dt)`) -- utilisé TEL QUEL, jamais resimulé ; si None, la
    vérité est simulée ICI en lockstep (bit-identique).

    Retourne (dict, AUCUN timestamp) : `{"seed", "dt", "n_emissions",
    "k_fen", "quadrant", "episodes_emissions", "delta_chi_fen",
    "delta_chi_plein", "taille_commits", "registre", "readouts_moteur",
    "etats_vrais"}` (readouts_moteur = copies plein-état AVANT commit)."""
    sy, sx = fenetre_slices(quadrant)
    n_ep_total = n_emissions * dt
    centres = centres_episodes(seed, n_ep_total)

    s_moteur = np.zeros_like(B0, dtype=np.float64)
    s_vrai = np.zeros_like(B0, dtype=np.float64) if verite is None else None

    episodes_emissions: list[int] = []
    delta_chi_fen: list[float] = []
    delta_chi_plein: list[float] = []
    taille_commits: list[int] = []
    registre: list[SummaryQT] = []
    readouts_moteur: list[np.ndarray] = []
    etats_vrais: list[np.ndarray] = []

    ep_fait = 0
    for i in range(1, n_emissions + 1):
        t_i = i * dt
        for centre in centres[ep_fait:t_i]:
            s_moteur = run_episode(s_moteur, B0, centre, PARAMS)
            if verite is None:
                s_vrai = run_episode(s_vrai, B0, centre, PARAMS)
        ep_fait = t_i

        etat_vrai_i = np.asarray(
            verite[t_i] if verite is not None else s_vrai, dtype=np.float64)
        readout_moteur_i = np.array(s_moteur, dtype=np.float64, copy=True)

        # 1. mesurer -- AVANT tout commit, référence = vrai (fenêtré + plein).
        alb_moteur = albedo(readout_moteur_i, S_HALF_OP)
        alb_vrai = albedo(etat_vrai_i, S_HALF_OP)
        delta_chi_fen.append(float(
            delta_chi(alb_moteur[sy, sx], alb_vrai[sy, sx])["max_carrier"]))
        delta_chi_plein.append(float(
            delta_chi(alb_moteur, alb_vrai)["max_carrier"]))

        # 2. committer -- l'état-MOTEUR∣fenêtre (jamais le vrai).
        commit = commettre_fenetre(s_moteur[sy, sx], quadrant, k_fen)

        # 3. ré-ancrer DANS la fenêtre seulement ; dehors inchangé.
        s_moteur[sy, sx] = extraire_fenetre(commit, quadrant)

        episodes_emissions.append(t_i)
        taille_commits.append(size_floats_qt(commit))
        registre.append(commit)
        readouts_moteur.append(readout_moteur_i)
        etats_vrais.append(np.array(etat_vrai_i, dtype=np.float64, copy=True))

        # 4. continuer : fait au tour suivant (segment ci-dessus), ou par le
        # pré-chargement `verite` s'il est fourni.

    return {
        "seed": seed, "dt": dt, "n_emissions": n_emissions, "k_fen": k_fen,
        "quadrant": quadrant,
        "episodes_emissions": episodes_emissions,
        "delta_chi_fen": delta_chi_fen,
        "delta_chi_plein": delta_chi_plein,
        "taille_commits": taille_commits,
        "registre": registre,
        "readouts_moteur": readouts_moteur,
        "etats_vrais": etats_vrais,
    }


def rederiver_emissions_fenetre(registre: list, seed: int, dt: int,
                                n_emissions: int,
                                quadrant: tuple[int, int]) -> list[np.ndarray]:
    """Foncteur de reconstruction É2/É3 reconduit (choix (viii)) : les
    readouts-moteur plein-état de CHAQUE émission, depuis (`registre`,
    `seed`) UNIQUEMENT -- aucun accès au monde vrai ni aux états vivants
    d'un appel de `chaine_fenetre` :
      readout@t_1     = évolution de s=0 sur dt épisodes (centres du seed) ;
      readout@t_{i+1} = évolution, de t_i à t_{i+1} (mêmes centres), du
                        readout@t_i dont la FENÊTRE a été remplacée par
                        l'extraction de `registre[i-1]` (le dehors
                        non-contraint est déterministe du seed).

    SIGNATURE STRICTE : exactement (`registre`, `seed`, `dt`, `n_emissions`,
    `quadrant`) -- pas de paramètre « champ vrai » (testé par inspection de
    signature)."""
    sy, sx = fenetre_slices(quadrant)
    centres = centres_episodes(seed, n_emissions * dt)

    readouts: list[np.ndarray] = []

    s = np.zeros_like(B0, dtype=np.float64)
    for centre in centres[0:dt]:
        s = run_episode(s, B0, centre, PARAMS)
    readouts.append(np.array(s, dtype=np.float64, copy=True))

    for i in range(1, n_emissions):
        s = np.array(s, dtype=np.float64, copy=True)
        s[sy, sx] = extraire_fenetre(registre[i - 1], quadrant)
        for centre in centres[i * dt:(i + 1) * dt]:
            s = run_episode(s, B0, centre, PARAMS)
        readouts.append(np.array(s, dtype=np.float64, copy=True))

    return readouts


def figure_sonde_fenetre(document: dict, path: str | Path) -> None:
    """`arcA_m2_sonde_fenetre.png` : Δχ_fen,i vs i (points + ligne pleine),
    Δχ plein-domaine en pointillés gris (diagnostic non-verdictal), lignes
    au jnd_sev et aux bornes de l'IC sévère, titre seed/Δt/k_fen/quadrant."""
    delta_fen = document["delta_chi_fen"]
    delta_plein = document["delta_chi_plein"]
    meta = document["meta"]
    jnd_sev = meta["jnd_sev"]
    ic_bas, ic_haut = meta["ic"]
    xs = list(range(1, len(delta_fen) + 1))

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(xs, delta_fen, "-o", color="#0072B2", linewidth=1.8, markersize=6,
            label="Δχ_fen,i (fenêtre 32x32, bras nu)")
    ax.plot(xs, delta_plein, "--s", color="#7F7F7F", linewidth=1.2,
            markersize=4, label="Δχ plein-domaine (diagnostic)")
    ax.axhline(jnd_sev, color="#D55E00", linestyle="-", linewidth=1.4,
               label=f"JND sévère (pin) = {jnd_sev * 100:.2f} %")
    ax.axhline(ic_bas, color="#D55E00", linestyle="--", linewidth=1.0,
               label=f"IC sévère [{ic_bas * 100:.2f}, {ic_haut * 100:.2f}] %")
    ax.axhline(ic_haut, color="#D55E00", linestyle="--", linewidth=1.0)
    ax.set_xticks(xs)
    ax.set_xlabel("émission i")
    ax.set_ylabel("Δχ (max_carrier, albedo moteur vs vrai)")
    ax.set_title(
        f"Sonde §A14 -- seed={meta['seed']}, Δt={meta['dt']}, "
        f"k_fen={meta['k_fen']}, fen={tuple(meta['quadrant'])}, "
        f"lecture={document['lecture']}")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    # Vérification d'instrument §A14 -- AVANT toute mesure, fail-loud.
    verification = verifier_instrument_fenetre()
    if verification["muette"]:
        raise RuntimeError(
            "SONDE MUETTE PAR CONSTRUCTION (§A14) -- remonter avant toute "
            f"mesure. Raisons : {verification['raisons']}")

    pins = charge_pins_severe()
    jnd_sev = pins["jnd"]
    ic_bas, ic_haut = pins["ic"]

    n_ep_total = DT_SONDE * N_EMISSIONS_SONDE
    episodes = [i * DT_SONDE for i in range(1, N_EMISSIONS_SONDE + 1)]

    t0 = time.perf_counter()
    verite = chaine_verite(SEED_SONDE, n_ep_total)
    t_verite = time.perf_counter() - t0

    garde = choisir_fenetre(verite, episodes, jnd_sev)
    quadrant = garde["quadrant"]

    t1 = time.perf_counter()
    res = chaine_fenetre(SEED_SONDE, DT_SONDE, N_EMISSIONS_SONDE, K_FEN,
                         quadrant, verite=verite)
    t_moteur = time.perf_counter() - t1

    delta_fen = [float(d) for d in res["delta_chi_fen"]]
    delta_plein = [float(d) for d in res["delta_chi_plein"]]
    lecture = lecture_sonde(delta_fen, jnd_sev)
    detail_ic = _comparaisons_ic(delta_fen, ic_bas, ic_haut)

    detail_lecture = {
        "traverse_jnd_sev": lecture["traverse_jnd_sev"],
        "s1": lecture["s1"],
        "s2": lecture["s2"],
        "max_emissions_2_3": lecture["max_emissions_2_3"],
        "max_emissions_4_6": lecture["max_emissions_4_6"],
        "traverse_ic_bas": detail_ic["traverse_ic_bas"],
        "traverse_ic_haut": detail_ic["traverse_ic_haut"],
    }

    chronometrage = {
        "verite_s": t_verite,
        "moteur_s": t_moteur,
        "par_episode_verite_s": t_verite / n_ep_total,
        "par_episode_moteur_s": t_moteur / n_ep_total,
    }

    document = {
        "meta": {
            "seed": SEED_SONDE, "dt": DT_SONDE, "k_fen": K_FEN,
            "n_emissions": N_EMISSIONS_SONDE,
            "cote_fenetre": COTE_FENETRE,
            "quadrant": list(quadrant),
            "k_out_escalade_fige": K_OUT_ESCALADE,
            "jnd_sev": jnd_sev, "ic": [ic_bas, ic_haut],
        },
        "verification_instrument": verification,
        "garde_anti_vacuite": garde["detail"],
        "delta_chi_fen": delta_fen,
        "delta_chi_plein": delta_plein,
        "taille_commits": [int(t) for t in res["taille_commits"]],
        "episodes_emissions": [int(e) for e in res["episodes_emissions"]],
        "chronometrage": chronometrage,
        "lecture": lecture["lecture"],
        "detail_lecture": detail_lecture,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON_PATH.write_text(
        json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    figure_sonde_fenetre(document, OUT_PNG_PATH)

    print("=" * 78)
    print("TASK S1 -- SONDE FENÊTRAGE-FOVÉA (§A14) : bras NU, fenêtre fixe 32x32")
    print("=" * 78)
    print(f"  seed={SEED_SONDE}  dt={DT_SONDE}  k_fen={K_FEN}  "
          f"n_emissions={N_EMISSIONS_SONDE}  quadrant={quadrant}")
    print(f"  jnd_sev = {jnd_sev!r}  ic = [{ic_bas!r}, {ic_haut!r}]")
    print("  série Δχ_fen,i (i=1..6) [diagnostic plein-domaine entre crochets] :")
    for i, (dfen, dplein) in enumerate(zip(delta_fen, delta_plein), start=1):
        print(f"    i={i}  Δχ_fen={dfen!r}  "
              f">=jnd_sev={lecture['traverse_jnd_sev'][i - 1]}  "
              f"[Δχ_plein={dplein!r}]")
    print(f"  LECTURE : {document['lecture']}  "
          f"(s1={lecture['s1']}, s2={lecture['s2']}, "
          f"max_2_3={lecture['max_emissions_2_3']!r}, "
          f"max_4_6={lecture['max_emissions_4_6']!r})")
    print(f"  chrono : vérité={t_verite:.3f} s  moteur={t_moteur:.3f} s")
    print(f"[REPORT] -> {OUT_JSON_PATH}")
    print(f"[REPORT] -> {OUT_PNG_PATH}")
    print("RAPPEL §A14 : sonde = lecture remontée, AUCUN verdict -- POINT "
          "D'ARRÊT OBLIGATOIRE, remonter la série avant toute suite.")


if __name__ == "__main__":
    main()
