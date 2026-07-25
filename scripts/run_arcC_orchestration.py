"""Arc C / Task 3 — orchestrateur de campagne (§C8 de PREREGISTRATION.md,
pocCascade2phys, commits `c494d50`→`d7c95e3`), reproduit verbatim dans
`.superpowers/sdd/arcC-task3-runner-brief.md`. Base : Task 2 COMPLETE
(`2ce40a0`).

**Ce script ne mesure AUCUN humain** dans ses chemins TESTÉS : la campagne
est validée sur sujet SYNTHÉTIQUE (`src/arcC_synthetic.py`), sur les VRAIS
20 champs sources (npz commités manche 1, via `src/arcC_abx.py`). Le chemin
`--sujet humain` délègue à la coquille interactive
(`scripts/run_arcC_session.py`) -- comme la coquille elle-même, il n'est PAS
testé ici (matplotlib + clavier), et n'est PAS exécuté par ce build.

Principe d'architecture : ORCHESTRE les primitives PURES déjà livrées et
testées de `src/arcC_abx.py` (`run_escalier`, `evalue_validite_session`,
`verifie_arret_2_invalides`, `positions_catch`, `BanqueBancs`,
`ecrit_log_jsonl`) et de `src/arcC_stimuli.py` (`regenere_budget`,
`delta_chi_stim`) -- rien n'est réimplémenté ici, ce module compose.

Trois conséquences §C8 câblées (brief, verbatim) :

  D-1 — ANCRE = quadtree budget 32 (`ANCRE_BUDGET`). Le staircase tourne à
  budget 32 (là où Task 2 mettait 256, l'objet jugé par la surface k*, pas
  l'axe de mesure Δχ).

  D-2 — Exclusion nommée par-source, AVANT toute présentation
  (`construit_manifeste_exclusions`) : ancre Δχ(t=1, budget=32) < 2 x
  `HAUT_JND_PLAUSIBLE` (= `SEUIL_EXCLUSION` = 0.16) -> source EXCLUE et
  NOMMÉE au manifeste avec son Δχ (jamais retirée silencieusement du tirage
  de roving) ; >= `N_EXCLUSIONS_STOP` (10) exclues -> statut STOP explicite
  (`STATUT_STOP_TROP_EXCLUES`), jamais une exception.

  D-3 — Catch en RÉSERVE SÉPARÉE. Collision résolue : `BUDGET_CATCH`
  (`src/arcC_abx.py`) vaut DÉJÀ `ANCRE_BUDGET` (32) -- la séparation
  catch/near-threshold ne vient PLUS d'un budget distinct (comme en Task 2,
  où le staircase tournait à 256 et le catch à 32, structurellement séparés
  par construction), mais de la SÉLECTION : `selectionne_stimulus_fort`
  (argmax, non-monotone-safe) choisit TOUJOURS le point le plus fort du banc
  de sa source, qui est >= la valeur à t=1 de ce même banc (un max est >= à
  n'importe quel point de la même courbe) -- et cette valeur à t=1 est
  l'ancre elle-même, qui vaut >= `SEUIL_EXCLUSION` (0.16) pour toute source
  INCLUSE (D-2 l'a garanti). Donc tout catch, sur toute source incluse, reste
  >= 0.16, largement au-dessus du niveau near-JND (`HAUT_JND_PLAUSIBLE` =
  0.08) que l'escalier balaie normalement -- SANS AMBIGUÏTÉ, par construction
  géométrique, pas par un budget séparé. `assert BUDGET_CATCH == ANCRE_BUDGET`
  ci-dessous rend cette collision EXPLICITE (et vérifiée), pas accidentelle.
  Validité §C5 INCHANGÉE (`evalue_validite_session`, pas de recalibration) ;
  option nommée NON activée par défaut (`--catch-sources-fortes`,
  `selectionne_sources_fortes`) pour restreindre le tirage catch aux sources
  à ancre la plus forte si la difficulté catch s'avère structurelle.

  D-4 — Contingence géométrie-plafond (`--geometrie {pic-csf,plafond}`,
  `calcule_taille_affichage_px`) : MÊME ancre/famille/sources, SEULE la
  taille d'affichage change (calculée depuis `plafond_texture`, §C7)."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_arcA_measure import _load_history
from src.arcC_abx import (BUDGET_CATCH, PAIRES_SOURCES, REGIME_LAXISTE, REGIME_SEVERE,
                          STATUT_CONTINUE, STATUT_STOP_2_INVALIDES, BanqueBancs,
                          EssaiPropose, ParametresEscalier, Regime, Reponse,
                          ecrit_log_jsonl, evalue_validite_session, run_escalier,
                          verifie_arret_2_invalides)
from src.arcC_calibration import (C_DEG_CIBLE_DEFAUT, PORTEUSE_CYC_PAR_DOMAINE_DEFAUT,
                                  SEUIL_ACUITE_ARCMIN_DEFAUT, observation_cellule_pic_csf,
                                  plafond_texture, taille_domaine_px)
from src.arcC_rendu import CHEMINS_RENDU
from src.arcC_stimuli import delta_chi_stim, regenere_budget
from src.arcC_synthetic import fabrique_sujet_synthetique

OUT_DIR = ROOT / "outputs" / "arcC"
LOGS_DIR = OUT_DIR / "logs"

# --- Paramètres NOMMÉS gravés (§C8 -- ne PAS changer sans remonter) ---------

ANCRE_BUDGET: int = 32                              # D-1
HAUT_JND_PLAUSIBLE: float = 0.08                    # D-2, estimation Romain §C8
SEUIL_EXCLUSION: float = 2.0 * HAUT_JND_PLAUSIBLE   # D-2 = 0.16
N_STAIRCASES: int = 3                               # §C5, minimum par régime
N_EXCLUSIONS_STOP: int = 10                          # D-2, seuil de STOP
FRACTION_SOURCES_FORTES: float = 0.5                # D-3, option --catch-sources-fortes
GEOMETRIES: tuple[str, ...] = ("pic-csf", "plafond")  # D-4

# Position du BRAS TÉMOIN viridis dans la session P3, GRAVÉE au prereg P3
# (précisions du 2026-07-25 soir) : la 3e des 4 staircases — deux staircases R1,
# PUIS le témoin, PUIS la dernière R1. Motif gravé : répartir fatigue et
# apprentissage sur les deux bras. Indice 0-based, donc 2.
POSITION_TEMOIN_VIRIDIS: int = 2
REGIMES_CANONIQUES: tuple[Regime, ...] = (REGIME_SEVERE, REGIME_LAXISTE)
CHOIX_REGIME: tuple[str, ...] = (REGIME_SEVERE.nom, REGIME_LAXISTE.nom, "tous")


def chemins_staircases(rendu: str, n_staircases: int, temoin_viridis: bool) -> tuple[str, ...]:
    """L'ORDRE EFFECTIF des chemins d'affichage, un par staircase — fonction
    PURE, pour que la position gravée du témoin soit testable sans écran.

    Sans `temoin_viridis`, toutes les staircases suivent `rendu`. Avec, la
    staircase d'indice `POSITION_TEMOIN_VIRIDIS` bascule sur `viridis` :
    (r1, r1, TÉMOIN, r1) à quatre staircases, la disposition gravée.

    Lève si la campagne ne peut pas porter cette disposition — il faut une 3e
    staircase pour le témoin ET au moins une APRÈS lui. Le gravé dit « la 3e
    des 4 : deux staircases R1, PUIS le témoin, PUIS la dernière R1 » : à trois
    staircases le témoin serait le DERNIER, ce qui est exactement la
    disposition que le motif gravé écarte (répartir fatigue et apprentissage
    sur les deux bras — un témoin en fin de session mesure un sujet fatigué
    contre trois staircases fraîches). Le défaut `N_STAIRCASES = 3` est le
    minimum §C5 par régime ; P3 exige donc `--n-staircases 4` explicitement, et
    cette garde est ce qui le dit à voix haute plutôt que de rendre en silence
    une disposition qui n'est pas celle du prereg."""
    if temoin_viridis and n_staircases < POSITION_TEMOIN_VIRIDIS + 2:
        raise ValueError(
            f"chemins_staircases : --temoin-viridis exige au moins "
            f"{POSITION_TEMOIN_VIRIDIS + 2} staircases (reçu {n_staircases}) -- la "
            "disposition est GRAVÉE au prereg P3 : R1, R1, TÉMOIN, R1. Il faut une 3e "
            "staircase pour le témoin ET au moins une après lui ; à "
            f"{POSITION_TEMOIN_VIRIDIS + 1} le témoin serait le DERNIER, ce que le motif "
            "gravé écarte.\n"
            f"    CORRECTION : relance avec --n-staircases {POSITION_TEMOIN_VIRIDIS + 2}")
    chemins = [rendu] * n_staircases
    if temoin_viridis:
        chemins[POSITION_TEMOIN_VIRIDIS] = "viridis"
    return tuple(chemins)

STATUT_STOP_TROP_EXCLUES: str = "STOP_TROP_DE_SOURCES_EXCLUES"

# D-3 : collision RENDUE EXPLICITE (cf. docstring module) -- si un jour
# `BUDGET_CATCH` ou `ANCRE_BUDGET` divergeaient, cette assertion casserait
# l'import (préférable à une divergence silencieuse de la garantie
# "catch >= SEUIL_EXCLUSION").
assert BUDGET_CATCH == ANCRE_BUDGET, (
    f"run_arcC_orchestration : BUDGET_CATCH ({BUDGET_CATCH}) doit valoir ANCRE_BUDGET "
    f"({ANCRE_BUDGET}) -- la séparation catch/near-threshold (§C8 D-3) repose sur la "
    "SÉLECTION (bout fort), pas sur un budget distinct ; un écart ici invaliderait la "
    "garantie 'tout catch >= SEUIL_EXCLUSION'.")


# --- Provenance (§C10 : commit_harnais/date_session/sujet stampés au manifeste) --


def obtient_commit_harnais(root: Path = ROOT) -> str:
    """SHA du commit pocPhysicator ayant produit la session (`git rev-parse
    HEAD`, `root` = racine du dépôt) -- c'est le HARNAIS qui connaît cette
    information au moment de la session, jamais le post-traitement (§C10,
    correctif provenance). Hors dépôt git (ou `git` absent) -> `"inconnu"`,
    mais l'ÉCHEC est LOGGUÉ (imprimé), jamais avalé silencieusement."""
    try:
        resultat = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                                  capture_output=True, text=True, check=True)
        return resultat.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        print(f"[AVERTISSEMENT] obtient_commit_harnais : impossible de déterminer le commit "
              f"git de {root} ({exc}) -- commit_harnais='inconnu' (§C10, échec loggué, "
              "jamais avalé).")
        return "inconnu"


# --- D-2 : ancre par-source + manifeste d'exclusions ------------------------


def mesure_ancre_source(seed: int, L: int, budget: int = ANCRE_BUDGET) -> float:
    """Ancre Δχ(t=1, `budget`) d'une source (seed, L) : `delta_chi_stim(s_true,
    régénéré(budget))` -- IDENTIQUE AU BIT PRÈS à `courbe_delta_chi(s_true,
    budget, ts=[1.0])[0]` (Task 1) : `melange(..., t=1.0)` renvoie
    `régénéré(budget)` bit-exact, donc les deux chemins mesurent EXACTEMENT
    la même chose (pas une réimplémentation, une composition directe des
    primitives Task 1, plus rapide qu'un banc fin complet -- un seul point,
    pas 161)."""
    hist = _load_history(seed)
    s_true = np.asarray(hist[f"s_L{L}"], dtype=np.float64)
    s_regen = regenere_budget(s_true, budget)
    return delta_chi_stim(s_true, s_regen)


def construit_manifeste_exclusions(
        *, sources: tuple[tuple[int, int], ...] = PAIRES_SOURCES,
        budget: int = ANCRE_BUDGET, seuil_exclusion: float = SEUIL_EXCLUSION,
        n_exclusions_stop: int = N_EXCLUSIONS_STOP) -> dict:
    """Contrôle d'ancrage PAR SOURCE, AVANT toute présentation (§C8 D-2) :
    mesure l'ancre de chacune des `sources`, exclut celles STRICTEMENT sous
    `seuil_exclusion` -- NOMMÉES au manifeste avec leur Δχ (jamais retirées
    silencieusement). `statut` = `STATUT_STOP_TROP_EXCLUES` (explicite,
    jamais une exception) ssi `n_exclues >= n_exclusions_stop`."""
    entrees: list[dict] = []
    for seed, L in sources:
        ancre = mesure_ancre_source(seed, L, budget)
        exclu = ancre < seuil_exclusion
        entrees.append(dict(seed=int(seed), L=int(L), delta_chi_ancre=float(ancre),
                            exclu=bool(exclu)))
    sources_exclues = [(e["seed"], e["L"]) for e in entrees if e["exclu"]]
    sources_incluses = [(e["seed"], e["L"]) for e in entrees if not e["exclu"]]
    statut = (STATUT_STOP_TROP_EXCLUES if len(sources_exclues) >= n_exclusions_stop
             else STATUT_CONTINUE)
    return dict(entrees=entrees, sources_exclues=sources_exclues,
               sources_incluses=sources_incluses, n_exclues=len(sources_exclues),
               n_sources=len(sources), budget=budget, seuil_exclusion=seuil_exclusion,
               n_exclusions_stop=n_exclusions_stop, statut=statut)


# --- D-3 : option nommée « sources fortes » pour le pool catch --------------


def selectionne_sources_fortes(entrees: list[dict],
                               fraction: float = FRACTION_SOURCES_FORTES
                               ) -> tuple[tuple[int, int], ...]:
    """Restreint le tirage des catch aux sources à ancre la plus FORTE parmi
    les INCLUSES (D-3, option nommée -- pas activée par défaut, cf.
    `--catch-sources-fortes`) : trie par ancre décroissante, garde la
    fraction haute (`FRACTION_SOURCES_FORTES` = 50 %, au moins 1 source).
    Jamais une source déjà EXCLUE (D-2) -- l'exclusion prime."""
    incluses = [e for e in entrees if not e["exclu"]]
    if not incluses:
        return ()
    tri = sorted(incluses, key=lambda e: e["delta_chi_ancre"], reverse=True)
    n_fortes = max(1, math.ceil(fraction * len(tri)))
    return tuple((e["seed"], e["L"]) for e in tri[:n_fortes])


# --- D-4 : géométrie pic-CSF vs plafond (même ancre/famille/sources) --------


def calcule_taille_affichage_px(
        geometrie: str, ppd: float, *,
        porteuse_cyc_par_domaine: float = PORTEUSE_CYC_PAR_DOMAINE_DEFAUT,
        c_deg_cible: float = C_DEG_CIBLE_DEFAUT,
        seuil_acuite_arcmin: float = SEUIL_ACUITE_ARCMIN_DEFAUT) -> int:
    """Taille d'affichage (px) du domaine 64x64, selon le MODE géométrique
    (§C7/§C8 D-4) : `pic-csf` pose la porteuse au pic de la CSF
    (`taille_domaine_px`) ; `plafond` calcule la taille au PLAFOND d'acuité
    (`plafond_texture`, `domaine_px_max`) -- SEULE cette taille change entre
    les deux modes, jamais l'ancre/la famille/les sources (garanti par
    construction : cette fonction ne touche à aucun des trois)."""
    if geometrie == "pic-csf":
        return taille_domaine_px(ppd, porteuse_cyc_par_domaine, c_deg_cible)
    if geometrie == "plafond":
        plafond = plafond_texture(ppd, seuil_acuite_arcmin)
        return int(round(plafond["domaine_px_max"]))
    raise ValueError(
        f"calcule_taille_affichage_px : geometrie inconnue {geometrie!r} "
        f"(attendu {GEOMETRIES!r}).")


# --- Seeds de campagne (dérivées, consignées) -------------------------------


def derive_seeds(base_seed: int, regime_idx: int, k: int) -> dict:
    """Dérive (`seed_roving`, `seed_catch`, `seed_sujet`) pour la staircase
    `k` du régime d'indice `regime_idx`, à partir d'un unique `base_seed` --
    déterministe, distinct par (régime, staircase) (même schéma que le
    test-clé Task 2 : `seed + 7919*indice`, `+1`/`+2` pour roving/catch,
    étendu ici avec `+3` pour le sujet synthétique et un facteur 1000 par
    régime pour ne jamais collisionner entre les deux régimes)."""
    graine = base_seed + (regime_idx * 1000 + k) * 7919
    return dict(seed_roving=graine + 1, seed_catch=graine + 2, seed_sujet=graine + 3)


# --- Une staircase, un régime, une campagne ---------------------------------

FabriqueRepondre = Callable[[int, str, int], tuple[Callable[[EssaiPropose], Reponse],
                                                   Callable[[], None]]]


def _fabrique_repondre_synthetique(theta_sim: float, sigma_sim: float) -> FabriqueRepondre:
    """Fabrique par défaut (sujet SYNTHÉTIQUE, ce build) : construit
    `fabrique_sujet_synthetique(theta_sim, sigma_sim, seed=seed_sujet)` par
    staircase -- AUCUN nettoyage nécessaire (pas de ressource UI), `cleanup`
    est un no-op."""
    def fabrique(numero_staircase: int, regime_nom: str, seed_sujet: int
                ) -> tuple[Callable[[EssaiPropose], Reponse], Callable[[], None]]:
        repondre = fabrique_sujet_synthetique(theta_sim, sigma_sim, seed=seed_sujet)
        return repondre, (lambda: None)
    return fabrique


def _fabrique_repondre_humain(taille_px: int, out_dir: Path, rendu: str,
                              chemins: tuple[str, ...] | None = None) -> FabriqueRepondre:
    """Délègue à la coquille interactive (`scripts/run_arcC_session.py`) --
    **NON testé ici** (matplotlib + clavier), **non exécuté par ce build**
    (brief : « AUCUN humain lancé dans ce build »). Une figure/axes est créée
    PAR staircase (régime déterminé par `regime_nom`) et fermée par le
    `cleanup` retourné -- `orchestre_regime` l'appelle après chaque
    staircase.

    CORRECTIF §C9 pièce 1 (timing) : `_construit_repondre_humain` renvoie
    désormais aussi le `JournalTiming` PUR (`src/arcC_timing.py`) de la
    staircase -- `cleanup` écrit son sidecar (`session_<regime>_<k>.
    timing.jsonl`, MÊME convention de nom que le log d'essais écrit ensuite
    par `orchestre_regime`, `out_dir` fourni ici pour cette raison) et
    imprime le résumé réalisé-vs-nominal (+ AVERTISSEMENT >25%, jamais un
    gate dur).

    P1 / §A37 : `rendu` est un paramètre REQUIS, transmis à `fabrique_affiche`
    (source UNIQUE du dispatch d'affichage, côté coquille) -- une campagne ne
    peut donc pas hériter d'un chemin d'affichage en silence, et la coquille et
    la campagne ne peuvent pas diverger.

    `chemins` (prereg P3) donne le chemin PAR STAIRCASE : c'est ce qui
    matérialise le BRAS TÉMOIN viridis en 3e position. `None` = toutes les
    staircases suivent `rendu`. Les fonctions d'affichage sont construites une
    fois par chemin, jamais par staircase : la staircase 0 et la staircase 3
    doivent traverser le MÊME objet, sinon le bras R1 ne serait pas un bras."""
    import os

    import matplotlib
    import matplotlib.pyplot as plt

    from scripts.run_arcC_session import (_cree_figure, _construit_repondre_humain,
                                          fabrique_affiche)
    from src.arcC_backend import assert_backend_interactif, selectionne_backend_qt
    from src.arcC_timing import ecrit_timing_jsonl, formate_resume_timing, resume_timing

    # Plomberie session live : backend Qt interactif + garde fail-loud (sinon
    # `waitforbuttonpress` boucle sans fin sur Agg, mode d'échec du 1er pré-vol).
    # `plt.ion()` -> fenêtres affichées. Une seule fois pour toute la campagne.
    selectionne_backend_qt()
    assert_backend_interactif(matplotlib.get_backend(), est_replay=False,
                              display=os.environ.get("DISPLAY"))
    plt.ion()
    affiches = {chemin: fabrique_affiche(chemin, taille_px)
                for chemin in set(chemins or (rendu,))}

    def fabrique(numero_staircase: int, regime_nom: str, seed_sujet: int
                ) -> tuple[Callable[[EssaiPropose], Reponse], Callable[[], None]]:
        chemin = rendu if chemins is None else chemins[numero_staircase]
        affiche = affiches[chemin]
        regime = REGIME_SEVERE if regime_nom == REGIME_SEVERE.nom else REGIME_LAXISTE
        fig, axes = _cree_figure(regime, taille_px)
        fig.suptitle("a : X ressemble à A     b : X ressemble à B     —     Échap / fermer : arrêter",
                     fontsize=9)
        fig.show()  # affiche la fenêtre de CETTE staircase (mode interactif)
        rng_masque = np.random.default_rng(seed_sujet)
        repondre, journal_timing = _construit_repondre_humain(fig, axes, regime, rng_masque,
                                                              affiche)

        def cleanup() -> None:
            plt.close(fig)
            timing_path = out_dir / f"session_{regime_nom}_{numero_staircase}.timing.jsonl"
            ecrit_timing_jsonl(timing_path, journal_timing)
            print(formate_resume_timing(resume_timing(journal_timing, regime)))
            print(f"[REPORT timing] -> {timing_path}")

        return repondre, cleanup
    return fabrique


def orchestre_regime(*, regime: Regime, regime_idx: int, n_staircases: int, base_seed: int,
                     sources_incluses: tuple[tuple[int, int], ...],
                     sources_catch: tuple[tuple[int, int], ...] | None,
                     budget: int, params: ParametresEscalier, banques: BanqueBancs,
                     fabrique_repondre: FabriqueRepondre, out_dir: Path,
                     chemins: tuple[str, ...] | None = None,
                     provenances: dict | None = None) -> dict:
    """Enchaîne `n_staircases` staircases pour UN régime : seeds dérivées
    (`derive_seeds`), roving restreint aux sources INCLUSES (D-2) et, pour
    le catch, au pool `sources_catch` (D-3, `None` = même pool que
    `sources_incluses`), écrit chaque log brut, arrête la boucle DÈS que
    `verifie_arret_2_invalides` (§C5) signale `STATUT_STOP_2_INVALIDES` --
    la 3e staircase (ou plus) N'EST PAS lancée dans ce cas.

    P3 (chantier 6) : `chemins` consigne le chemin d'affichage de CHAQUE
    staircase (bras témoin viridis en 3e position) ; `provenances` (chemin ->
    bloc de provenance) déclenche l'écriture d'un SIDECAR PAR STAIRCASE,
    `session_<regime>_<k>.conditions.json`, portant la provenance du chemin
    traversé ET le `sha256_log` du log qui vient d'être écrit (chantier 3) --
    la lecture versionnée du verdict citera ce sha, plus d'appariement par nom.
    Les deux sont `None` par défaut : une campagne synthétique n'affiche rien
    et n'a aucun chemin à dater."""
    sessions: list[dict] = []
    historique_valide: list[bool] = []
    statut = STATUT_CONTINUE
    for k in range(n_staircases):
        seeds = derive_seeds(base_seed, regime_idx, k)
        repondre, cleanup = fabrique_repondre(k, regime.nom, seeds["seed_sujet"])
        try:
            resultat = run_escalier(
                numero_staircase=k, regime_nom=regime.nom, repondre=repondre,
                seed_roving=seeds["seed_roving"], seed_catch=seeds["seed_catch"],
                budget=budget, params=params, banques=banques,
                sources=sources_incluses, sources_catch=sources_catch)
        finally:
            cleanup()

        validite = evalue_validite_session(resultat.essais)
        log_path = out_dir / f"session_{regime.nom}_{k}.jsonl"
        ecrit_log_jsonl(log_path, resultat.essais)
        historique_valide.append(bool(validite["valide"]))
        chemin_rendu = None if chemins is None else chemins[k]
        session = dict(
            numero_staircase=k, regime=regime.nom, seed_roving=seeds["seed_roving"],
            seed_catch=seeds["seed_catch"], seed_sujet=seeds["seed_sujet"],
            log_path=str(log_path), chemin_rendu=chemin_rendu,
            n_essais=len(resultat.essais),
            n_reversals=len(resultat.reversals), complet=bool(resultat.complet),
            seuil=resultat.seuil, validite=validite)

        if provenances is not None and chemin_rendu is not None:
            from scripts.run_arcC_session import sha256_fichier
            conditions_path = out_dir / f"session_{regime.nom}_{k}.conditions.json"
            ecrit_manifeste_json(conditions_path, dict(
                numero_staircase=k, regime=regime.nom, log_path=str(log_path),
                provenance_rendu=provenances[chemin_rendu],
                sha256_log=sha256_fichier(log_path)))
            session["conditions_path"] = str(conditions_path)
        sessions.append(session)

        statut = verifie_arret_2_invalides(historique_valide)
        if statut == STATUT_STOP_2_INVALIDES:
            break

    return dict(sessions=sessions, statut_orchestration=statut,
               n_staircases_lancees=len(sessions))


def orchestre_campagne(
        *, base_seed: int, n_staircases: int = N_STAIRCASES, geometrie: str = "pic-csf",
        catch_sources_fortes: bool = False, ppd: float, sujet: str = "synthetique",
        fabrique_repondre: FabriqueRepondre | None = None,
        theta_sim: float | None = None, sigma_sim: float | None = None,
        out_dir: Path = LOGS_DIR, params: ParametresEscalier = ParametresEscalier(),
        budget: int = ANCRE_BUDGET, seuil_exclusion: float = SEUIL_EXCLUSION,
        n_exclusions_stop: int = N_EXCLUSIONS_STOP,
        fraction_sources_fortes: float = FRACTION_SOURCES_FORTES,
        luminosite: str | None = None, conditions: str | None = None,
        long_ref_px: float | None = None, long_ref_mm: float | None = None,
        distance_mm: float | None = None, provenance_rendu: dict | None = None,
        regimes: tuple[str, ...] | None = None, chemins: tuple[str, ...] | None = None,
        provenances: dict | None = None) -> dict:
    """Orchestrateur de campagne complet (§C8) : applique D-2 (exclusion
    nommée -- STOP si >= `n_exclusions_stop`), D-3 (catch réserve forte,
    optionnellement restreint aux sources fortes), D-1 (ancre = `budget`,
    par défaut `ANCRE_BUDGET`), D-4 (`geometrie`), puis enchaîne
    `n_staircases` x {sévère, laxiste} (`orchestre_regime`).

    Sujet : soit `fabrique_repondre` fourni explicitement (tout sujet,
    y compris `_fabrique_repondre_humain` -- non testé ici), soit
    `theta_sim`/`sigma_sim` (sujet SYNTHÉTIQUE, ce build/tests) ; l'un des
    deux est requis.

    CORRECTIF §C9 pièces 2 & 3 : `luminosite`/`conditions` (REQUISES par
    `main()` quand `--sujet humain`, cf. `valide_conditions_requises_si_
    humain` -- optionnelles ici, aucune validation à ce niveau, la logique
    pure ne refuse rien) + rappel des paramètres de calibration
    (`ppd`/`long_ref_px`/`long_ref_mm`/`distance_mm`) sont consignés au
    manifeste sous `conditions_validite`, TOUJOURS présent (y compris sur
    le statut STOP-trop-exclues).

    CORRECTIF §C10 (provenance, consommée par `scripts/run_arcC_pins.py`) :
    `sujet` (`"humain"`/`"synthetique"`, défaut inchangé pour compat
    ascendante), `commit_harnais` (`obtient_commit_harnais`, SHA pocPhysicator
    au moment de CETTE session) et `date_session` (ISO 8601, `datetime.now()`)
    sont stampés au manifeste -- c'est le HARNAIS qui les connaît, jamais le
    post-traitement Task 3 (pins).

    P1 / §A37 : `provenance_rendu` (bloc construit par
    `run_arcC_session.provenance_rendu` -- chemin d'affichage, empreinte de
    `src/arcC_rendu.py`, ordre des étages, calibration réalisée) est stampé
    sous `config_affichage`. Défaut `None` = campagne SYNTHÉTIQUE (aucun écran,
    aucun chemin d'affichage à dater) ; `main()` l'exige dès `--sujet humain`.
    Motif : « P3 date son verdict sur R1 » -- une campagne humaine dont on ne
    sait pas quel chemin elle a traversé est inattribuable.

    P3 (chantier 6) : `regimes` restreint les régimes lancés (`None` = les
    deux, comportement historique ; P3 utilisera le SÉVÈRE seul, c'est jnd_sev
    que T2 consomme). L'indice de régime passé à `derive_seeds` reste celui du
    tuple CANONIQUE, jamais celui de la sélection : restreindre les régimes ne
    doit RIEN changer aux seeds d'un régime conservé, sinon `--regime severe`
    ne rejouerait pas la même campagne que `--regime tous`. `chemins` et
    `provenances` matérialisent le bras témoin (cf. `orchestre_regime`) ;
    l'ordre effectif est consigné sous `config_affichage`."""
    if geometrie not in GEOMETRIES:
        raise ValueError(
            f"orchestre_campagne : geometrie inconnue {geometrie!r} (attendu {GEOMETRIES!r}).")
    if fabrique_repondre is None:
        if theta_sim is None or sigma_sim is None:
            raise ValueError(
                "orchestre_campagne : fournir soit `fabrique_repondre`, soit "
                "`theta_sim`+`sigma_sim` (sujet synthétique).")
        fabrique_repondre = _fabrique_repondre_synthetique(theta_sim, sigma_sim)

    manifeste_exclusions = construit_manifeste_exclusions(
        budget=budget, seuil_exclusion=seuil_exclusion, n_exclusions_stop=n_exclusions_stop)
    taille_px = calcule_taille_affichage_px(geometrie, ppd)

    manifeste = dict(
        sujet=sujet, commit_harnais=obtient_commit_harnais(), date_session=datetime.now().isoformat(),
        parametres_graves=dict(
            ancre_budget=budget, haut_jnd_plausible=HAUT_JND_PLAUSIBLE,
            seuil_exclusion=seuil_exclusion, n_staircases=n_staircases,
            budget_catch=BUDGET_CATCH, geometrie=geometrie,
            catch_sources_fortes=catch_sources_fortes,
            fraction_sources_fortes=(fraction_sources_fortes if catch_sources_fortes else None)),
        config_affichage=dict(geometrie=geometrie, ppd=ppd, taille_domaine_px=taille_px,
                              provenance_rendu=provenance_rendu,
                              chemins_staircases=(list(chemins) if chemins else None),
                              regimes_lances=list(regimes) if regimes else None),
        conditions_validite=dict(
            luminosite=luminosite, conditions=conditions,
            calibration=dict(ppd=ppd, long_ref_px=long_ref_px, long_ref_mm=long_ref_mm,
                             distance_mm=distance_mm)),
        exclusions=manifeste_exclusions,
        base_seed=base_seed)

    if manifeste_exclusions["statut"] == STATUT_STOP_TROP_EXCLUES:
        manifeste["statut_global"] = STATUT_STOP_TROP_EXCLUES
        manifeste["regimes"] = {}
        return manifeste

    sources_incluses = tuple(tuple(s) for s in manifeste_exclusions["sources_incluses"])
    sources_catch = (selectionne_sources_fortes(manifeste_exclusions["entrees"],
                                                fraction_sources_fortes)
                     if catch_sources_fortes else None)

    banques = BanqueBancs()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resultats: dict[str, dict] = {}
    for regime_idx, regime in enumerate(REGIMES_CANONIQUES):
        # `regime_idx` reste l'indice CANONIQUE, pas celui de la sélection --
        # `derive_seeds` en dépend, et restreindre les régimes ne doit pas
        # re-seeder ceux qu'on garde.
        if regimes is not None and regime.nom not in regimes:
            continue
        resultats[regime.nom] = orchestre_regime(
            regime=regime, regime_idx=regime_idx, n_staircases=n_staircases,
            base_seed=base_seed, sources_incluses=sources_incluses,
            sources_catch=sources_catch, budget=budget, params=params, banques=banques,
            fabrique_repondre=fabrique_repondre, out_dir=out_dir,
            chemins=chemins, provenances=provenances)

    manifeste["statut_global"] = STATUT_CONTINUE
    manifeste["regimes"] = resultats
    return manifeste


# --- Manifeste JSON ----------------------------------------------------------


def ecrit_manifeste_json(path: str | Path, manifeste: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifeste, indent=2, ensure_ascii=False), encoding="utf-8")


def lit_manifeste_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --- CLI ---------------------------------------------------------------------


def valide_conditions_requises_si_humain(
        sujet: str, luminosite: str | None, conditions: str | None,
        erreur: Callable[[str], None]) -> None:
    """§C9 (pièces 2 & 3) : luminosité + conditions d'environnement REQUISES
    *par écrit* dès que `sujet == "humain"` (« pas par habitude ») -- appelle
    `erreur(message)` (typiquement `parser.error`, qui lève `SystemExit`) si
    l'une des deux manque. OPTIONNELLES pour `sujet == "synthetique"` (non
    pertinentes, aucun humain devant l'écran) -- `erreur` n'est alors JAMAIS
    appelé."""
    if sujet == "humain" and (luminosite is None or conditions is None):
        erreur(
            "--sujet humain requiert --luminosite ET --conditions (§C9 : consignées par "
            "écrit, pas par habitude, AVANT toute donnée humaine).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--n-staircases", type=int, default=N_STAIRCASES)
    parser.add_argument("--geometrie", choices=list(GEOMETRIES), default="pic-csf")
    parser.add_argument("--catch-sources-fortes", action="store_true", default=False,
                        help="Restreint le tirage des catch aux sources à ancre la plus "
                             "forte (§C8 D-3, option nommée) -- OFF par défaut, à activer "
                             "seulement sur décision remontée (catch structurellement trop "
                             "dur, cf. brief).")
    parser.add_argument("--sujet", choices=["synthetique", "humain"], default="synthetique")
    # P1 / §A37 : SANS DÉFAUT, exigé dès `--sujet humain` (même idiome que
    # --luminosite/--conditions, §C9). Laissé optionnel en synthétique : cette
    # campagne-là n'affiche rien du tout, il n'y a aucun chemin à déclarer.
    parser.add_argument("--rendu", choices=list(CHEMINS_RENDU), default=None,
                        help="Chemin d'affichage (§A37) : 'viridis' (historique, pseudo-"
                             "couleur) ou 'r1' (noyau rendu-instrument). REQUIS si "
                             "--sujet humain.")
    parser.add_argument("--regime", choices=list(CHOIX_REGIME), default="tous",
                        help="Régime(s) lancé(s). Défaut 'tous' = comportement historique. "
                             "P3 utilise 'severe' : c'est jnd_sev que T2 consomme, le "
                             "laxiste n'est pas re-mesuré.")
    parser.add_argument("--temoin-viridis", action="store_true", default=False,
                        help="Insère UNE staircase témoin en chemin viridis, en 3e position "
                             "(position GRAVÉE au prereg P3). Valide UNIQUEMENT avec "
                             "--sujet humain --rendu r1.")
    parser.add_argument("--theta-sim", type=float, default=None,
                        help="Requis si --sujet synthetique.")
    parser.add_argument("--sigma-sim", type=float, default=None,
                        help="Requis si --sujet synthetique.")
    # Géométrie d'écran (§C7) -- mesurée physiquement, PAS de défaut inventé.
    parser.add_argument("--long-ref-px", type=float, required=True)
    parser.add_argument("--long-ref-mm", type=float, required=True)
    parser.add_argument("--distance-mm", type=float, required=True)
    # Conditions de validité d'exécution (§C9 pièces 2 & 3) -- REQUISES si
    # --sujet humain (`valide_conditions_requises_si_humain`), sans objet si
    # --sujet synthetique (aucun humain devant l'écran).
    parser.add_argument("--luminosite", type=str, default=None,
                        help="Luminosité écran NOTÉE (§C9 pièce 2), p. ex. '120 nits'/"
                             "'OSD 75%%'. REQUISE si --sujet humain.")
    parser.add_argument("--conditions", type=str, default=None,
                        help="Conditions d'environnement NOTÉES (§C9 pièce 3) -- repère de "
                             "distance + éclairage ambiant. REQUISES si --sujet humain.")
    parser.add_argument("--out-dir", type=Path, default=LOGS_DIR)
    parser.add_argument("--manifeste", type=Path, default=OUT_DIR / "manifeste_campagne.json")
    args = parser.parse_args()

    valide_conditions_requises_si_humain(args.sujet, args.luminosite, args.conditions,
                                         parser.error)

    from src.arcC_calibration import pixels_par_degre
    ppd = pixels_par_degre(args.long_ref_px, args.long_ref_mm, args.distance_mm)

    # P3 (chantier 6) : le bras témoin n'a de sens que dans une session humaine
    # en chemin R1 -- ailleurs, « témoin viridis » ne témoignerait de rien.
    if args.temoin_viridis and not (args.sujet == "humain" and args.rendu == "r1"):
        parser.error(
            "--temoin-viridis est valide UNIQUEMENT avec --sujet humain --rendu r1 (prereg "
            f"P3 : le bras témoin est une GARDE DE VALIDITÉ du bras R1 ; reçu --sujet "
            f"{args.sujet!r} --rendu {args.rendu!r}).")

    regimes = None if args.regime == "tous" else (args.regime,)
    provenance = None
    chemins = None
    provenances = None
    if args.sujet == "humain":
        # P1 / §A37 : le chemin d'affichage se DÉCLARE avant toute présentation
        # humaine (même idiome que --luminosite/--conditions, §C9).
        if args.rendu is None:
            parser.error(
                "--rendu est REQUIS avec --sujet humain (§A37 : chaque session choisit son "
                f"chemin d'affichage consciemment, attendu {CHEMINS_RENDU!r}) -- sans objet "
                "en sujet synthétique, qui n'affiche rien.")
        print("[AVERTISSEMENT] --sujet humain : chemin délégué à la coquille interactive, "
              "NON testé par ce build. Aucune session humaine n'est lancée par les tests.")
        taille_px = calcule_taille_affichage_px(args.geometrie, ppd)
        from scripts.run_arcC_session import assert_geometrie_comparable, provenance_rendu
        # Garde de COMPARABILITÉ (§A38-CORRECTION) : MÊME garde que la coquille
        # isolée, importée d'elle -- une campagne et une session ne peuvent pas
        # diverger sur ce qui rend un verdict lisible contre l'IC gravé.
        assert_geometrie_comparable(args.geometrie, est_session_humaine=True)
        provenance = provenance_rendu(
            args.rendu, ppd=ppd, taille_px=taille_px,
            obs=observation_cellule_pic_csf(ppd))
        print(f"[provenance §A37] rendu={provenance['chemin_rendu']}  "
              f"sha256(arcC_rendu.py)={provenance['sha256_arcC_rendu']}  "
              f"etages={provenance['ordre_etages']}")
        chemins = chemins_staircases(args.rendu, args.n_staircases, args.temoin_viridis)
        provenances = {chemin: provenance_rendu(chemin, ppd=ppd, taille_px=taille_px,
                                                obs=observation_cellule_pic_csf(ppd))
                       for chemin in set(chemins)}
        print(f"[chemins §P3] ordre effectif des staircases : {list(chemins)}"
              + ("  (témoin viridis en 3e position, GRAVÉE)" if args.temoin_viridis else ""))
        fabrique_repondre = _fabrique_repondre_humain(taille_px, args.out_dir, args.rendu,
                                                      chemins)
    else:
        if args.theta_sim is None or args.sigma_sim is None:
            parser.error("--sujet synthetique requiert --theta-sim et --sigma-sim.")
        fabrique_repondre = _fabrique_repondre_synthetique(args.theta_sim, args.sigma_sim)

    from scripts.run_arcC_session import SessionInterrompue
    try:
        manifeste = orchestre_campagne(
            base_seed=args.base_seed, n_staircases=args.n_staircases, geometrie=args.geometrie,
            catch_sources_fortes=args.catch_sources_fortes, ppd=ppd, sujet=args.sujet,
            fabrique_repondre=fabrique_repondre, out_dir=args.out_dir,
            luminosite=args.luminosite, conditions=args.conditions,
            long_ref_px=args.long_ref_px, long_ref_mm=args.long_ref_mm,
            distance_mm=args.distance_mm, provenance_rendu=provenance,
            regimes=regimes, chemins=chemins, provenances=provenances)
    except SessionInterrompue as exc:
        print(f"[CAMPAGNE INTERROMPUE] {exc} -- AUCUN manifeste écrit (campagne inachevée = "
              "fait de session, pas de référent, §C10). Reprends au pré-vol si c'était "
              "volontaire, sinon remonte.")
        return

    ecrit_manifeste_json(args.manifeste, manifeste)

    print("=" * 78)
    print("ARC C / TASK 3 -- ORCHESTRATION DE CAMPAGNE")
    print("=" * 78)
    print(f"statut_global={manifeste['statut_global']}  "
          f"sources_exclues={manifeste['exclusions']['n_exclues']}/"
          f"{manifeste['exclusions']['n_sources']}")
    for regime_nom, regime_data in manifeste.get("regimes", {}).items():
        print(f"  régime={regime_nom} : {regime_data['n_staircases_lancees']} staircases, "
              f"statut={regime_data['statut_orchestration']}")
    print(f"[REPORT] -> {args.manifeste}")


if __name__ == "__main__":
    main()
