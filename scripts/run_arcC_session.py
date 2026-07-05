"""Arc C / Task 2 — coquille interactive MINCE pour le harnais ABX (§C0/§C7 de
PREREGISTRATION.md, pocCascade2phys, reproduit verbatim dans
`.superpowers/sdd/arcC-task2-abx-brief.md`).

Principe d'architecture LOAD-BEARING (brief) : ce script ne contient AUCUNE
logique de mesure -- il appelle `src/arcC_abx.py` (logique PURE, testée
exhaustivement sur sujet synthétique) en substituant un HUMAIN au sujet
synthétique. Rien ici n'est testé (matplotlib + clavier ne sont pas
headless-testables) ; **ce script n'est PAS exécuté en Task 2** -- il est
construit pour Task 3 (vraies sessions), avec un mode `--replay` qui rejoue
un log déjà écrit (sans capter aucune entrée humaine) pour PROUVER le
déterminisme de la logique pure sur un cas réel.

Format ABX affiché (résolution d'ambiguïté du contrôleur, faute de précision
du brief sur le nombre de stimuli affichés -- décision nommée, cf. rapport
Task 2) : trois stimuli, A = champ VRAI (Δχ=0), B = champ DÉGRADÉ (le
stimulus présenté, Δχ mesuré), X = COPIE de A ou de B selon
`essai.reponse_correcte` (déjà tirée par la logique pure, §C3 -- contrebalance
la position de la bonne réponse). Le sujet répond "A" ou "B" : quelle
position X reproduit. AUCUNE réassignation supplémentaire de A/B n'est
nécessaire ici : la position (A=vrai fixe, B=dégradé fixe) est FIXE par
essai, seule l'identité de X varie -- c'est `reponse_correcte`, déjà seedée
et consignée par `run_escalier`, qui porte tout le contrebalancement.

Affichage : albedo (`src.albedo.albedo`, point d'opération `S_HALF_OP` gravé
manche 1) rendu viridis, vmin=0, vmax=1, origin="lower" (= `render.py`/
`io_utils.py`, cf. brief). Taille d'affichage calibrée via
`src/arcC_calibration.py` (§C7) -- PARAMÈTRES DE GÉOMÉTRIE D'ÉCRAN (distance,
longueur de référence) à fournir en CLI, mesurés physiquement avant la
session (Task 3, pas improvisés ici).

CORRECTIF (§C9 de PREREGISTRATION.md, pocCascade2phys `7110e31`, reproduit
dans `.superpowers/sdd/arcC-task3fix-timing-conditions-brief.md`), avant
toute donnée humaine : (1) durées d'exposition RÉALISÉES loggées par essai
(`src/arcC_timing.py`, accumulateur PUR testé -- sidecar `<log>.timing.jsonl`
+ résumé réalisé-vs-nominal imprimé, AVERTISSEMENT si écart >25 %, jamais un
gate dur) ; (2) `--luminosite`/`--conditions` REQUISES en session live
(sidecar `<log>.conditions.json`) -- absentes seulement en `--replay` (aucune
capture humaine)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from scripts.run_arcA_measure import _load_history
from src.albedo import albedo
from src.arcC_abx import (REGIME_LAXISTE, REGIME_SEVERE, BanqueBancs, EssaiPropose,
                          ParametresEscalier, Regime, Reponse, ResultatEscalier,
                          ecrit_log_jsonl, evalue_validite_session, lit_log_jsonl,
                          rejoue_escalier, run_escalier)
from src.arcC_calibration import (C_DEG_CIBLE_DEFAUT, PORTEUSE_CYC_PAR_DOMAINE_DEFAUT,
                                  observation_cellule_pic_csf, pixels_par_degre)
from src.arcC_stimuli import melange, regenere_budget
from src.arcC_timing import (JournalTiming, ecrit_timing_jsonl, enregistre_phase,
                             formate_resume_timing, resume_timing)
from scripts.run_arcA_revalidate import S_HALF_OP
# Task 3 (§C8) : ancre budget-32 (D-1) et flag géométrie-plafond (D-4) --
# IMPORTÉS de l'orchestrateur (source unique de vérité pour ces paramètres
# gravés), pas redéfinis ici (coquille = mince, cf. docstring module).
from scripts.run_arcC_orchestration import ANCRE_BUDGET, GEOMETRIES, calcule_taille_affichage_px

OUT_DIR = ROOT / "outputs" / "arcC" / "logs"


# --- Rendu d'un essai (albedo, viridis, taille calibrée) --------------------


def _images_essai(essai: EssaiPropose) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construit (A, B, X) en albedo pour CET essai : A = champ VRAI (Δχ=0),
    B = champ au `t` présenté (le stimulus dégradé, Δχ mesuré), X = copie de
    A ou de B selon `essai.reponse_correcte` (cf. docstring module)."""
    hist = _load_history(essai.seed_source)
    s_true = np.asarray(hist[f"s_L{essai.L_source}"], dtype=np.float64)
    s_regen = regenere_budget(s_true, essai.budget)
    s_b = melange(s_true, s_regen, essai.t)
    a_a = albedo(s_true, S_HALF_OP)
    a_b = albedo(s_b, S_HALF_OP)
    a_x = a_a if essai.reponse_correcte == "A" else a_b
    return a_a, a_b, a_x


def _affiche(ax: plt.Axes, image: np.ndarray, titre: str) -> None:
    """Affiche `image` (albedo, viridis, vmin=0/vmax=1/origin=lower, cf.
    `render.py`/`io_utils.py`). La taille CALIBRÉE (§C7, `taille_domaine_px`)
    gouverne la taille de la FIGURE entière, fixée une fois à la création
    (`_cree_figure`) -- pas par image individuelle."""
    ax.clear()
    ax.imshow(image, cmap="viridis", vmin=0.0, vmax=1.0, origin="lower")
    ax.set_title(titre)
    ax.set_xticks([])
    ax.set_yticks([])


def _masque_bruite(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Masque bruité intercalé (régime laxiste, §C0) : bruit uniforme [0,1],
    indépendant à chaque présentation (RNG passée par l'appelant -- seedée si
    on veut un masque reproductible au replay, cf. `--replay`)."""
    return rng.uniform(0.0, 1.0, size=shape)


# --- Sujet HUMAIN (capture clavier) ------------------------------------------


def _construit_repondre_humain(fig: plt.Figure, axes: list[plt.Axes], regime: Regime,
                               rng_masque: np.random.Generator
                               ) -> tuple["callable", JournalTiming]:
    """Fabrique `repondre(essai) -> "A"|"B"` branché sur un humain : affiche
    A/B (+X) selon le régime (simultané côte à côte / séquentiel avec masque
    et délai) et bloque sur une touche 'a'/'b' (`plt.waitforbuttonpress` +
    callback clavier). AUCUNE logique de mesure/scoring ici -- délègue tout
    le scoring à `run_escalier` (logique pure).

    CORRECTIF §C9 pièce 1 (durées réalisées) : mesure `time.perf_counter()`
    autour de CHAQUE phase présentée -- sévère : la présentation simultanée
    (une seule phase `"simultane"`, AUCUNE cible temporelle, inspection
    libre) ; laxiste : X, masque, A, masque, B, masque (`"exposition_{X,A,B}"`
    / `"retention_{X,A,B}"`, comparées à `regime.exposition_s`/
    `retention_s`) -- accumulées via `enregistre_phase` (accumulateur PUR,
    `src/arcC_timing.py`) dans un `JournalTiming` renvoyé à l'appelant (qui
    écrit le sidecar + imprime le résumé, cf. `main()` ici et
    `_fabrique_repondre_humain`, `scripts/run_arcC_orchestration.py` --
    PARTAGÉ, un seul endroit instrumenté)."""
    reponse_capturee: dict[str, str] = {}
    journal_timing = JournalTiming()

    def on_key(event) -> None:
        if event.key in ("a", "b"):
            reponse_capturee["valeur"] = event.key.upper()

    fig.canvas.mpl_connect("key_press_event", on_key)

    def repondre(essai: EssaiPropose) -> Reponse:
        a_a, a_b, a_x = _images_essai(essai)
        if regime.simultane:
            t_debut = time.perf_counter()
            _affiche(axes[0], a_a, "A")
            _affiche(axes[1], a_b, "B")
            _affiche(axes[2], a_x, "X")
            fig.canvas.draw()
            # Inspection libre, SANS limite de temps (régime sévère, §C0) --
            # AUCUNE cible temporelle : phase mesurée pour information
            # seulement (`nominal=None`, cf. `resume_timing`, catégorie
            # "autre", jamais comparée/alertée).
            enregistre_phase(journal_timing, indice_essai=essai.indice_essai,
                             regime_nom=regime.nom, nom_phase="simultane",
                             t_debut=t_debut, t_fin=time.perf_counter(), nominal=None)
        else:
            for image, titre in ((a_x, "X"), (a_a, "A"), (a_b, "B")):
                t_debut = time.perf_counter()
                _affiche(axes[0], image, titre)
                fig.canvas.draw()
                plt.pause(regime.exposition_s or 0.0)
                enregistre_phase(journal_timing, indice_essai=essai.indice_essai,
                                 regime_nom=regime.nom, nom_phase=f"exposition_{titre}",
                                 t_debut=t_debut, t_fin=time.perf_counter(),
                                 nominal=regime.exposition_s)
                if regime.masque_bruite:
                    t_debut = time.perf_counter()
                    _affiche(axes[0], _masque_bruite(image.shape, rng_masque), "")
                    fig.canvas.draw()
                    plt.pause(regime.retention_s or 0.0)
                    enregistre_phase(journal_timing, indice_essai=essai.indice_essai,
                                     regime_nom=regime.nom, nom_phase=f"retention_{titre}",
                                     t_debut=t_debut, t_fin=time.perf_counter(),
                                     nominal=regime.retention_s)

        reponse_capturee.clear()
        while "valeur" not in reponse_capturee:
            plt.waitforbuttonpress(timeout=0.1)
        return reponse_capturee["valeur"]  # type: ignore[return-value]

    return repondre, journal_timing


# --- CLI ---------------------------------------------------------------------


def _cree_figure(regime: Regime, taille_px: int) -> tuple[plt.Figure, list[plt.Axes]]:
    """Taille de figure calibrée (§C7) : `taille_px` (par stimulus, en
    pixels d'écran) converti en pouces matplotlib via le DPI par défaut de la
    figure (100, cf. `plt.rcParams["figure.dpi"]`)."""
    n_axes = 3 if regime.simultane else 1
    dpi = plt.rcParams.get("figure.dpi", 100.0)
    taille_pouces = max(taille_px / dpi, 1.0)
    fig, axes = plt.subplots(1, n_axes, figsize=(taille_pouces * n_axes, taille_pouces))
    if n_axes == 1:
        axes = [axes]
    return fig, list(axes)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--regime", choices=["severe", "laxiste"], required=True)
    parser.add_argument("--numero-staircase", type=int, required=True)
    parser.add_argument("--seed-roving", type=int, required=True)
    parser.add_argument("--seed-catch", type=int, required=True)
    parser.add_argument("--seed-masque", type=int, default=0,
                        help="Seed du bruit du masque (régime laxiste) -- consignée pour "
                             "replay (le masque LUI-MÊME n'est pas rejoué à l'identique par "
                             "`--replay`, seul le SCORE/la séquence de stimuli le sont).")
    parser.add_argument("--budget", type=int, default=ANCRE_BUDGET,
                        help=f"Défaut = ANCRE_BUDGET (§C8 D-1) = {ANCRE_BUDGET}.")
    parser.add_argument("--geometrie", choices=list(GEOMETRIES), default="pic-csf",
                        help="Contingence géométrie-plafond (§C8 D-4) -- MÊME ancre/famille/"
                             "sources que 'pic-csf', seule la taille d'affichage change.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Chemin du log JSONL de sortie (défaut : "
                             "outputs/arcC/logs/session_<staircase>_<regime>.jsonl).")
    parser.add_argument("--replay", type=Path, default=None,
                        help="Rejoue un log existant (AUCUNE capture humaine) -- "
                             "prouve le déterminisme (`rejoue_escalier`).")
    # Géométrie d'écran (§C7) -- mesurées physiquement, PAS de défaut inventé.
    parser.add_argument("--long-ref-px", type=float, required=True)
    parser.add_argument("--long-ref-mm", type=float, required=True)
    parser.add_argument("--distance-mm", type=float, required=True)
    parser.add_argument("--c-deg-cible", type=float, default=C_DEG_CIBLE_DEFAUT)
    parser.add_argument("--porteuse-cyc-par-domaine", type=float,
                        default=PORTEUSE_CYC_PAR_DOMAINE_DEFAUT)
    # Conditions de validité d'exécution (§C9 pièces 2 & 3) -- REQUISES en
    # session LIVE (un humain devant l'écran) ; sans objet en --replay (aucune
    # capture humaine, rien à consigner).
    parser.add_argument("--luminosite", type=str, default=None,
                        help="Luminosité écran NOTÉE (§C9 pièce 2), p. ex. '120 nits'/"
                             "'OSD 75%%'. REQUISE en session live.")
    parser.add_argument("--conditions", type=str, default=None,
                        help="Conditions d'environnement NOTÉES (§C9 pièce 3) -- repère de "
                             "distance + éclairage ambiant. REQUISES en session live.")
    args = parser.parse_args()

    if args.replay is None and (args.luminosite is None or args.conditions is None):
        parser.error(
            "--luminosite et --conditions sont REQUISES en session live (§C9 : consignées "
            "par écrit, pas par habitude, AVANT toute donnée humaine) -- absentes seulement "
            "en --replay (aucune capture humaine, rien à consigner).")

    regime = REGIME_SEVERE if args.regime == "severe" else REGIME_LAXISTE
    ppd = pixels_par_degre(args.long_ref_px, args.long_ref_mm, args.distance_mm)
    taille_px = calcule_taille_affichage_px(
        args.geometrie, ppd, porteuse_cyc_par_domaine=args.porteuse_cyc_par_domaine,
        c_deg_cible=args.c_deg_cible)
    obs = observation_cellule_pic_csf(ppd, porteuse_cyc_par_domaine=args.porteuse_cyc_par_domaine,
                                      c_deg_cible=args.c_deg_cible)
    print(f"[calibration §C7] geometrie={args.geometrie}  ppd={ppd:.3f}  "
          f"taille_domaine_px={taille_px}  cellule_pic_csf={obs['cellule_arcmin_pic_csf']:.3f} "
          f"arcmin (plafond={obs['seuil_acuite_arcmin']:.3f} arcmin, ratio="
          f"{obs['ratio_cellule_sur_seuil']:.3f}) -- observation, aucune décision.")

    banques = BanqueBancs()
    params = ParametresEscalier()

    if args.replay is not None:
        essais_log = lit_log_jsonl(args.replay)
        resultat = rejoue_escalier(
            essais_log, numero_staircase=args.numero_staircase, regime_nom=regime.nom,
            seed_roving=args.seed_roving, seed_catch=args.seed_catch, budget=args.budget,
            params=params, banques=banques)
        print(f"[replay] {len(resultat.essais)} essais rejoués depuis {args.replay} -- "
              f"seuil={resultat.seuil}, complet={resultat.complet} (aucune capture humaine).")
        return

    fig, axes = _cree_figure(regime, taille_px)
    rng_masque = np.random.default_rng(args.seed_masque)
    repondre, journal_timing = _construit_repondre_humain(fig, axes, regime, rng_masque)

    t0 = time.time()
    resultat: ResultatEscalier = run_escalier(
        numero_staircase=args.numero_staircase, regime_nom=regime.nom, repondre=repondre,
        seed_roving=args.seed_roving, seed_catch=args.seed_catch, budget=args.budget,
        params=params, banques=banques)
    plt.close(fig)

    validite = evalue_validite_session(resultat.essais)
    out_path = args.out or (OUT_DIR / f"session_{args.numero_staircase}_{regime.nom}.jsonl")
    ecrit_log_jsonl(out_path, resultat.essais)

    # Correctif §C9 pièce 1 : sidecar timing (durées réalisées, hors log
    # d'essais -- ne touche PAS le déterminisme des logs existants) + résumé
    # réalisé-vs-nominal (avertissement >25%, jamais un gate dur).
    timing_path = out_path.parent / f"{out_path.stem}.timing.jsonl"
    ecrit_timing_jsonl(timing_path, journal_timing)
    resume = resume_timing(journal_timing, regime)

    # Correctif §C9 pièces 2 & 3 : luminosité + conditions d'environnement,
    # consignées par écrit (sidecar, même convention que le timing).
    conditions_path = out_path.parent / f"{out_path.stem}.conditions.json"
    conditions_path.write_text(json.dumps(dict(
        luminosite=args.luminosite, conditions=args.conditions,
        long_ref_px=args.long_ref_px, long_ref_mm=args.long_ref_mm,
        distance_mm=args.distance_mm, c_deg_cible=args.c_deg_cible,
        porteuse_cyc_par_domaine=args.porteuse_cyc_par_domaine),
        indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 78)
    print(f"ARC C / TASK 3 -- SESSION staircase={args.numero_staircase} régime={regime.nom}")
    print("=" * 78)
    print(f"n_essais={len(resultat.essais)}  n_reversals={len(resultat.reversals)}  "
          f"complet={resultat.complet}  seuil={resultat.seuil}")
    print(f"validité §C5 : {validite}")
    print(formate_resume_timing(resume))
    print(f"Temps de session : {time.time() - t0:.1f}s")
    print(f"[REPORT] -> {out_path}")
    print(f"[REPORT timing] -> {timing_path}")
    print(f"[REPORT conditions] -> {conditions_path}")


if __name__ == "__main__":
    main()
