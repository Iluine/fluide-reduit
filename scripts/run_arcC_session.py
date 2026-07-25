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
capture humaine).

P1 / §A37 (2026-07-25) — DEUX CHEMINS D'AFFICHAGE, choisis EXPLICITEMENT.
`--rendu` est OBLIGATOIRE et SANS DÉFAUT : chaque session future choisit son
chemin consciemment, jamais par héritage.

  - `viridis` : le chemin HISTORIQUE, conservé tel quel (contrôle A/A,
    comparaisons avec les sessions d'Arc C déjà mesurées). C'est LUI qui a
    mesuré le pin gravé jnd_sev 7.33 % -- en PSEUDO-COULEUR, avec
    l'interpolation par DÉFAUT d'imshow, sans gestion d'EOTF (fait de code
    gravé §A37).
  - `r1` : le noyau de rendu-instrument (`src/arcC_rendu.py`) -- albédo ->
    luminance, niveaux de gris, rééchantillonnage bilinéaire GRAVÉ dans R1,
    inverse-EOTF sRGB, quantification unique. Ici matplotlib n'a plus le droit
    d'interpoler quoi que ce soit (`interpolation="nearest"`) : le
    rééchantillonnage est DÉJÀ fait, à la taille calibrée §C7. La fenêtre
    présente donc l'image À SA TAILLE, sans zoom de figure.

PROVENANCE (§A37 : « P3 date son verdict sur R1 »). Le sidecar
`<log>.conditions.json` porte désormais, en plus des conditions §C9, le bloc
`provenance_rendu` : chemin choisi, sha256 de `src/arcC_rendu.py`, ordre des
étages, ppd, taille d'affichage RÉALISÉE et le report §C7
`observation_cellule_pic_csf`. ÉCART ASSUMÉ ET REMONTÉ par rapport à l'ordre de
mission, qui demandait « l'en-tête JSONL de session » : le log d'essais N'A PAS
d'en-tête (`ecrit_log_jsonl`/`lit_log_jsonl`, `src/arcC_abx.py`, écrivent et
relisent UNE LIGNE PAR ESSAI), et lui en ajouter un casserait `rejoue_escalier`
-- or `src/arcC_abx.py` est INTOUCHABLE par garde-fou. Le sidecar de conditions
est le manifeste de session existant ; il portait déjà les entrées de
calibration. Aucune information de la liste gravée n'est perdue, seul son
FICHIER change."""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.arcC_backend import assert_backend_interactif, selectionne_backend_qt

from scripts.run_arcA_measure import _load_history
from src.albedo import albedo
from src.arcC_abx import (REGIME_LAXISTE, REGIME_SEVERE, BanqueBancs, EssaiPropose,
                          ParametresEscalier, Regime, Reponse, ResultatEscalier,
                          ecrit_log_jsonl, evalue_validite_session, lit_log_jsonl,
                          rejoue_escalier, run_escalier)
from src.arcC_calibration import (C_DEG_CIBLE_DEFAUT, PORTEUSE_CYC_PAR_DOMAINE_DEFAUT,
                                  observation_cellule_pic_csf, pixels_par_degre)
from src import arcC_rendu
from src.arcC_rendu import CHEMINS_RENDU, rendu_r1
from src.arcC_stimuli import melange, regenere_budget
from src.arcC_timing import (JournalTiming, ecrit_timing_jsonl, enregistre_phase,
                             formate_resume_timing, formate_timing_essai, resume_timing)
from scripts.run_arcA_revalidate import S_HALF_OP
# Task 3 (§C8) : ancre budget-32 (D-1) et flag géométrie-plafond (D-4) --
# IMPORTÉS de l'orchestrateur (source unique de vérité pour ces paramètres
# gravés), pas redéfinis ici (coquille = mince, cf. docstring module).
# `ecrit_manifeste_json` réutilisé tel quel (Task 3 CORRECTIF §C9, finding
# Minor) pour le sidecar `<log>.conditions.json` -- écriture JSON DRY, un
# seul endroit qui sait écrire un dict en JSON indenté (import module-level
# sans cycle : `run_arcC_orchestration` n'importe cette coquille QUE
# paresseusement, dans `_fabrique_repondre_humain`, jamais au chargement du
# module).
from scripts.run_arcC_orchestration import (ANCRE_BUDGET, GEOMETRIES, calcule_taille_affichage_px,
                                             ecrit_manifeste_json)

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


def _affiche_r1(ax: plt.Axes, image: np.ndarray, titre: str, *, taille_px: int) -> None:
    """Affiche `image` (albedo) à travers le NOYAU R1 (§A37) : niveaux de gris,
    `origin="lower"` conservé (même convention que le chemin viridis).

    `interpolation="nearest"` est LOAD-BEARING, pas une préférence : R1 a DÉJÀ
    rééchantillonné le champ 64x64 vers `taille_px` (bilinéaire gravé, D-P1-2).
    Laisser matplotlib re-interpoler par-dessus remettrait dans la chaîne
    exactement l'étage non gravé que P1 est venu retirer. La fenêtre présente
    donc l'image À SA TAILLE : tout zoom de figure (redimensionnement manuel de
    la fenêtre, DPI inattendu) ré-introduit un rééchantillonnage hors contrat --
    la géométrie de session §C7 suppose que ça n'arrive pas.

    Niveaux de gris via `cmap="gray"` + `vmin=0`/`vmax=255` : UNE valeur par
    pixel, la réplication RGB est laissée au backend (matplotlib la fait), rien
    n'est fabriqué ici."""
    ax.clear()
    ax.imshow(rendu_r1(image, taille_px), cmap="gray", vmin=0, vmax=255,
              origin="lower", interpolation="nearest")
    ax.set_title(titre)
    ax.set_xticks([])
    ax.set_yticks([])


def fabrique_affiche(rendu: str, taille_px: int) -> "callable":
    """Retourne la fonction d'affichage `(ax, image, titre) -> None` du chemin
    demandé -- AUCUN défaut : `rendu` inconnu lève, il n'existe pas de valeur
    implicite (§A37 : chaque session choisit son chemin consciemment).

    Un SEUL point de dispatch, partagé par la coquille et par la campagne
    (`scripts/run_arcC_orchestration.py`) : le chemin d'affichage ne peut donc
    pas diverger entre une session isolée et une session de campagne."""
    if rendu == "viridis":
        return _affiche
    if rendu == "r1":
        return partial(_affiche_r1, taille_px=taille_px)
    raise ValueError(
        f"run_arcC_session : chemin de rendu inconnu {rendu!r} (attendu {CHEMINS_RENDU!r}).")


def _sha256_module_rendu() -> str:
    """Empreinte sha256 de la SOURCE de `src/arcC_rendu.py` (fichier lu en
    octets), recalculée à CHAQUE session.

    Elle ne juge rien ici : le VERROU de non-dérive vit dans
    `tests/test_arcC_rendu.py` (tradition des kernels F1). Ce que cette
    empreinte fait, c'est DATER la session -- §A37 grave que « P3 date son
    verdict sur R1 », donc le pin transporté devra pouvoir nommer l'objet R1
    exact qu'il a traversé."""
    return hashlib.sha256(Path(arcC_rendu.__file__).read_bytes()).hexdigest()


def provenance_rendu(rendu: str, *, ppd: float, taille_px: int, obs: dict) -> dict:
    """Bloc de PROVENANCE du chemin d'affichage (§A37), consigné au sidecar de
    conditions et imprimé en tête de session.

    Pour `viridis`, `sha256_arcC_rendu` et `ordre_etages` disent la VÉRITÉ du
    chemin historique -- R1 n'y participe pas, et la chaîne est celle du fait de
    code gravé : pseudo-couleur, interpolation par défaut d'imshow, aucun EOTF.
    Y écrire l'empreinte de R1 serait une provenance fausse."""
    if rendu == "r1":
        sha = _sha256_module_rendu()
        etages = arcC_rendu.ORDRE_ETAGES
    elif rendu == "viridis":
        sha = None
        etages = ("imshow(cmap=viridis, vmin=0, vmax=1) -- pseudo-couleur, "
                  "reechantillonnage PAR DEFAUT d'imshow (non grave), aucun EOTF")
    else:
        raise ValueError(
            f"provenance_rendu : chemin de rendu inconnu {rendu!r} (attendu {CHEMINS_RENDU!r}).")
    return dict(chemin_rendu=rendu, sha256_arcC_rendu=sha, ordre_etages=etages,
                ppd=ppd, taille_domaine_px_realisee=taille_px,
                observation_cellule_pic_csf=obs)


def _masque_bruite(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Masque bruité intercalé (régime laxiste, §C0) : bruit uniforme [0,1],
    indépendant à chaque présentation (RNG passée par l'appelant -- seedée si
    on veut un masque reproductible au replay, cf. `--replay`)."""
    return rng.uniform(0.0, 1.0, size=shape)


# --- Sujet HUMAIN (capture clavier) ------------------------------------------


class SessionInterrompue(RuntimeError):
    """Levée quand le sujet interrompt la session (fenêtre FERMÉE ou touche Échap)
    AVANT d'avoir répondu -- la staircase est incomplète, rien de définitif n'est
    écrit. Attrapée par `main()` (coquille) et par la campagne
    (`run_arcC_orchestration.py`) pour un arrêt PROPRE : pas un hang, pas une
    fenêtre blanche qui se rouvre en boucle (le défaut du 1er essai d'affichage)."""


def _construit_repondre_humain(fig: plt.Figure, axes: list[plt.Axes], regime: Regime,
                               rng_masque: np.random.Generator, affiche: "callable"
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
    PARTAGÉ, un seul endroit instrumenté).

    P1 / §A37 : `affiche` est un paramètre REQUIS (`fabrique_affiche`), jamais
    une valeur par défaut -- un chemin d'affichage hérité en silence est
    exactement ce que le sélecteur obligatoire interdit. Le MASQUE bruité du
    régime laxiste passe par la MÊME fonction que les stimuli : sous `--rendu
    r1` il traverse donc R1 comme eux, et la session ne mélange pas deux
    chaînes d'affichage."""
    etat: dict = {"reponse": None, "ferme": False}
    journal_timing = JournalTiming()

    def on_key(event) -> None:
        if event.key in ("a", "b"):
            etat["reponse"] = event.key.upper()
        elif event.key == "escape":
            etat["ferme"] = True  # abandon volontaire au clavier

    def on_close(_event) -> None:
        etat["ferme"] = True  # fermeture de la fenêtre = abandon (arrêt propre, pas un hang)

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", on_close)

    def repondre(essai: EssaiPropose) -> Reponse:
        a_a, a_b, a_x = _images_essai(essai)
        n_avant = len(journal_timing.enregistrements)
        if regime.simultane:
            t_debut = time.perf_counter()
            affiche(axes[0], a_a, "A")
            affiche(axes[1], a_b, "B")
            affiche(axes[2], a_x, "X")
            fig.canvas.draw()
            fig.canvas.flush_events()  # rendu immédiat (sévère : pas de plt.pause pour pomper)
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
                affiche(axes[0], image, titre)
                fig.canvas.draw()
                plt.pause(regime.exposition_s or 0.0)
                enregistre_phase(journal_timing, indice_essai=essai.indice_essai,
                                 regime_nom=regime.nom, nom_phase=f"exposition_{titre}",
                                 t_debut=t_debut, t_fin=time.perf_counter(),
                                 nominal=regime.exposition_s)
                if regime.masque_bruite:
                    t_debut = time.perf_counter()
                    affiche(axes[0], _masque_bruite(image.shape, rng_masque), "")
                    fig.canvas.draw()
                    plt.pause(regime.retention_s or 0.0)
                    enregistre_phase(journal_timing, indice_essai=essai.indice_essai,
                                     regime_nom=regime.nom, nom_phase=f"retention_{titre}",
                                     t_debut=t_debut, t_fin=time.perf_counter(),
                                     nominal=regime.retention_s)

        # Affichage LIVE des durées réalisées de CET essai (§C9 pièce 1,
        # pré-vol B-bis) : rend « quelques essais + Ctrl-C » suffisant pour
        # vérifier le timing sans attendre la complétion (le sidecar n'est
        # écrit qu'à la fin de la staircase).
        print(formate_timing_essai(essai.indice_essai,
                                   journal_timing.enregistrements[n_avant:]), flush=True)

        etat["reponse"] = None
        while etat["reponse"] is None:
            if etat["ferme"]:
                raise SessionInterrompue(
                    f"Session interrompue (fenêtre fermée ou Échap) à l'essai "
                    f"{essai.indice_essai}, régime {regime.nom} -- aucune réponse, staircase "
                    "incomplète, rien de définitif écrit.")
            plt.pause(0.05)  # pompe l'event loop Qt + repeint ; laisse on_key/on_close tirer
        return etat["reponse"]  # type: ignore[return-value]

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
    # P1 / §A37 : SANS DÉFAUT, volontairement. Exigé même en `--replay` (qui
    # n'affiche rien) : le sélecteur est le lieu où une session DÉCLARE son
    # chemin, et une exception « sauf en replay » rouvrirait la porte à
    # l'héritage silencieux que ce flag existe pour fermer.
    parser.add_argument("--rendu", choices=list(CHEMINS_RENDU), required=True,
                        help="Chemin d'affichage (§A37) : 'viridis' = chemin historique "
                             "(pseudo-couleur, celui du pin gravé) ; 'r1' = noyau "
                             "rendu-instrument (luminance, sRGB, bilinéaire gravé).")
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

    provenance = provenance_rendu(args.rendu, ppd=ppd, taille_px=taille_px, obs=obs)
    print(f"[provenance §A37] rendu={provenance['chemin_rendu']}  "
          f"sha256(arcC_rendu.py)={provenance['sha256_arcC_rendu']}  "
          f"etages={provenance['ordre_etages']}")

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

    # Plomberie session live : fenêtre GUI requise (affichage + capture clavier).
    # Bascule sur un backend Qt interactif si dispo, PUIS garde fail-loud si le
    # backend reste non-interactif -- sinon `waitforbuttonpress` boucle sans fin
    # (le mode d'échec du 1er pré-vol B, sur Agg).
    selectionne_backend_qt()
    assert_backend_interactif(matplotlib.get_backend(), est_replay=False,
                              display=os.environ.get("DISPLAY"))

    fig, axes = _cree_figure(regime, taille_px)
    fig.suptitle("a : X ressemble à A     b : X ressemble à B     —     Échap / fermer : arrêter",
                 fontsize=9)
    plt.ion()          # mode interactif : la fenêtre s'affiche et pompe les événements
    fig.show()
    rng_masque = np.random.default_rng(args.seed_masque)
    repondre, journal_timing = _construit_repondre_humain(
        fig, axes, regime, rng_masque, fabrique_affiche(args.rendu, taille_px))

    t0 = time.time()
    try:
        resultat: ResultatEscalier = run_escalier(
            numero_staircase=args.numero_staircase, regime_nom=regime.nom, repondre=repondre,
            seed_roving=args.seed_roving, seed_catch=args.seed_catch, budget=args.budget,
            params=params, banques=banques)
    except SessionInterrompue as exc:
        plt.close(fig)
        print(f"[SESSION INTERROMPUE] {exc}")
        return
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
    # consignées par écrit (sidecar, même convention que le timing) --
    # réutilise `ecrit_manifeste_json` (helper JSON DRY de l'orchestrateur).
    # P1 / §A37 : le bloc `provenance_rendu` s'y adjoint (chemin, empreinte de
    # R1, ordre des étages, calibration réalisée) -- le dict n'est donc plus
    # plat ; aucun lecteur n'existe pour ce sidecar (vérifié au grep), rien ne
    # casse. Motif de l'écart au « en-tête JSONL » demandé : cf. docstring
    # module (le log d'essais n'a pas d'en-tête, et `arcC_abx.py` est
    # intouchable).
    conditions_path = out_path.parent / f"{out_path.stem}.conditions.json"
    ecrit_manifeste_json(conditions_path, dict(
        luminosite=args.luminosite, conditions=args.conditions,
        long_ref_px=args.long_ref_px, long_ref_mm=args.long_ref_mm,
        distance_mm=args.distance_mm, c_deg_cible=args.c_deg_cible,
        porteuse_cyc_par_domaine=args.porteuse_cyc_par_domaine,
        provenance_rendu=provenance))

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
