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
import sys
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
                                  SEUIL_ACUITE_ARCMIN_DEFAUT, plafond_texture,
                                  taille_domaine_px)
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


def _fabrique_repondre_humain(taille_px: int, out_dir: Path) -> FabriqueRepondre:
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
    gate dur)."""
    import matplotlib.pyplot as plt

    from scripts.run_arcC_session import _cree_figure, _construit_repondre_humain
    from src.arcC_timing import ecrit_timing_jsonl, formate_resume_timing, resume_timing

    def fabrique(numero_staircase: int, regime_nom: str, seed_sujet: int
                ) -> tuple[Callable[[EssaiPropose], Reponse], Callable[[], None]]:
        regime = REGIME_SEVERE if regime_nom == REGIME_SEVERE.nom else REGIME_LAXISTE
        fig, axes = _cree_figure(regime, taille_px)
        rng_masque = np.random.default_rng(seed_sujet)
        repondre, journal_timing = _construit_repondre_humain(fig, axes, regime, rng_masque)

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
                     fabrique_repondre: FabriqueRepondre, out_dir: Path) -> dict:
    """Enchaîne `n_staircases` staircases pour UN régime : seeds dérivées
    (`derive_seeds`), roving restreint aux sources INCLUSES (D-2) et, pour
    le catch, au pool `sources_catch` (D-3, `None` = même pool que
    `sources_incluses`), écrit chaque log brut, arrête la boucle DÈS que
    `verifie_arret_2_invalides` (§C5) signale `STATUT_STOP_2_INVALIDES` --
    la 3e staircase (ou plus) N'EST PAS lancée dans ce cas."""
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
        sessions.append(dict(
            numero_staircase=k, regime=regime.nom, seed_roving=seeds["seed_roving"],
            seed_catch=seeds["seed_catch"], seed_sujet=seeds["seed_sujet"],
            log_path=str(log_path), n_essais=len(resultat.essais),
            n_reversals=len(resultat.reversals), complet=bool(resultat.complet),
            seuil=resultat.seuil, validite=validite))

        statut = verifie_arret_2_invalides(historique_valide)
        if statut == STATUT_STOP_2_INVALIDES:
            break

    return dict(sessions=sessions, statut_orchestration=statut,
               n_staircases_lancees=len(sessions))


def orchestre_campagne(
        *, base_seed: int, n_staircases: int = N_STAIRCASES, geometrie: str = "pic-csf",
        catch_sources_fortes: bool = False, ppd: float,
        fabrique_repondre: FabriqueRepondre | None = None,
        theta_sim: float | None = None, sigma_sim: float | None = None,
        out_dir: Path = LOGS_DIR, params: ParametresEscalier = ParametresEscalier(),
        budget: int = ANCRE_BUDGET, seuil_exclusion: float = SEUIL_EXCLUSION,
        n_exclusions_stop: int = N_EXCLUSIONS_STOP,
        fraction_sources_fortes: float = FRACTION_SOURCES_FORTES) -> dict:
    """Orchestrateur de campagne complet (§C8) : applique D-2 (exclusion
    nommée -- STOP si >= `n_exclusions_stop`), D-3 (catch réserve forte,
    optionnellement restreint aux sources fortes), D-1 (ancre = `budget`,
    par défaut `ANCRE_BUDGET`), D-4 (`geometrie`), puis enchaîne
    `n_staircases` x {sévère, laxiste} (`orchestre_regime`).

    Sujet : soit `fabrique_repondre` fourni explicitement (tout sujet,
    y compris `_fabrique_repondre_humain` -- non testé ici), soit
    `theta_sim`/`sigma_sim` (sujet SYNTHÉTIQUE, ce build/tests) ; l'un des
    deux est requis."""
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
        parametres_graves=dict(
            ancre_budget=budget, haut_jnd_plausible=HAUT_JND_PLAUSIBLE,
            seuil_exclusion=seuil_exclusion, n_staircases=n_staircases,
            budget_catch=BUDGET_CATCH, geometrie=geometrie,
            catch_sources_fortes=catch_sources_fortes,
            fraction_sources_fortes=(fraction_sources_fortes if catch_sources_fortes else None)),
        config_affichage=dict(geometrie=geometrie, ppd=ppd, taille_domaine_px=taille_px),
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
    regimes: dict[str, dict] = {}
    for regime_idx, regime in enumerate((REGIME_SEVERE, REGIME_LAXISTE)):
        regimes[regime.nom] = orchestre_regime(
            regime=regime, regime_idx=regime_idx, n_staircases=n_staircases,
            base_seed=base_seed, sources_incluses=sources_incluses,
            sources_catch=sources_catch, budget=budget, params=params, banques=banques,
            fabrique_repondre=fabrique_repondre, out_dir=out_dir)

    manifeste["statut_global"] = STATUT_CONTINUE
    manifeste["regimes"] = regimes
    return manifeste


# --- Manifeste JSON ----------------------------------------------------------


def ecrit_manifeste_json(path: str | Path, manifeste: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifeste, indent=2, ensure_ascii=False), encoding="utf-8")


def lit_manifeste_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --- CLI ---------------------------------------------------------------------


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
    parser.add_argument("--theta-sim", type=float, default=None,
                        help="Requis si --sujet synthetique.")
    parser.add_argument("--sigma-sim", type=float, default=None,
                        help="Requis si --sujet synthetique.")
    # Géométrie d'écran (§C7) -- mesurée physiquement, PAS de défaut inventé.
    parser.add_argument("--long-ref-px", type=float, required=True)
    parser.add_argument("--long-ref-mm", type=float, required=True)
    parser.add_argument("--distance-mm", type=float, required=True)
    parser.add_argument("--out-dir", type=Path, default=LOGS_DIR)
    parser.add_argument("--manifeste", type=Path, default=OUT_DIR / "manifeste_campagne.json")
    args = parser.parse_args()

    from src.arcC_calibration import pixels_par_degre
    ppd = pixels_par_degre(args.long_ref_px, args.long_ref_mm, args.distance_mm)

    if args.sujet == "humain":
        print("[AVERTISSEMENT] --sujet humain : chemin délégué à la coquille interactive, "
              "NON testé par ce build. Aucune session humaine n'est lancée par les tests.")
        taille_px = calcule_taille_affichage_px(args.geometrie, ppd)
        fabrique_repondre = _fabrique_repondre_humain(taille_px, args.out_dir)
    else:
        if args.theta_sim is None or args.sigma_sim is None:
            parser.error("--sujet synthetique requiert --theta-sim et --sigma-sim.")
        fabrique_repondre = _fabrique_repondre_synthetique(args.theta_sim, args.sigma_sim)

    manifeste = orchestre_campagne(
        base_seed=args.base_seed, n_staircases=args.n_staircases, geometrie=args.geometrie,
        catch_sources_fortes=args.catch_sources_fortes, ppd=ppd,
        fabrique_repondre=fabrique_repondre, out_dir=args.out_dir)

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
