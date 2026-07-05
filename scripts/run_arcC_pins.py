"""Arc C / Task 3 — post-traitement d'une campagne → `pins_spatial.json`,
CONFORME au contrat de sortie gravé §C10 de PREREGISTRATION.md (pocCascade2phys,
`schemas/arcC-pins-spatial-v1.schema.json`), reproduit verbatim dans
`.superpowers/sdd/arcC-task3conform-schema-brief.md`. Base : `scripts/
run_arcC_orchestration.py` (manifeste de campagne, provenance §C10 incluse).

**Ce script ne mesure AUCUN humain** (données synthétiques en test) et
**n'imprime/n'écrit AUCUN verdict de manche** : il rapporte des MESURES
(`JND_sev^spat`, `JND_lax^spat` + IC, par régime) depuis les logs bruts d'une
campagne -- l'`additionalProperties: false` du schéma l'interdit
STRUCTURELLEMENT (aucun champ hors contrat ne peut être ajouté). La LECTURE
§C4 (côté 3/4 %, contingence géométrie-plafond) reste au contrôleur, sur
données RÉELLES, Task 5.

Contrat de sortie -- LE SCHÉMA EST LA VÉRITÉ (`schemas/
arcC-pins-spatial-v1.schema.json`, copie BYTE POUR BYTE du fichier gravé dans
pocCascade2phys) : `ecrit_pins_json` VALIDE (`jsonschema.Draft7Validator`)
AVANT d'écrire quoi que ce soit -- un document non conforme lève une
`ValueError` détaillée (fail loud), JAMAIS un référent malformé n'est écrit.

Conséquence structurelle notable : le schéma n'a AUCUNE représentation pour
une campagne arrêtée avant le lancement des régimes (ancien statut
`STOP_TROP_DE_SOURCES_EXCLUES`, `regimes={}`) -- `construit_pins` refuse
explicitement ce cas (`RuntimeError`, avant même la validation JSON Schema).
De même, si un régime s'arrête après seulement 2 staircases (§C5, 2 sessions
invalides consécutives, `STATUT_STOP_2_INVALIDES`), le tableau `staircases`
qui en résulte a `minItems=2 < 3` -- la validation du schéma refusera
d'écrire (fail loud, PAS de contournement : le contrat exige >= 3 staircases
BRUTES par régime, y compris sur un régime INVALIDE, la branche à-cheval
recalculant dessus).

Principe d'architecture : ORCHESTRE les primitives PURES déjà livrées et
testées de `src/arcC_abx.py` (`evalue_dispersion`, `lit_log_jsonl`), `src/
arcC_calibration.py` (`observation_cellule_pic_csf`) et `src/arcC_timing.py`
(`resume_timing`, `lit_timing_jsonl`) -- rien n'est réimplémenté ici.

Méthode d'IC primaire (NOMMÉE, inchangée depuis la 1ʳᵉ version de ce script) :
bornes MIN/MAX des seuils des staircases complètes de la condition
(`METHODE_IC = "min_max_seuils"`) -- à `n=3` (typique, §C5), un bootstrap
ré-échantillonnerait 3 points, illusion de précision que l'échantillon ne
supporte pas ; min/max expose directement la DISPERSION OBSERVÉE. La
méthode combinée (`ic_combine`, branche à-cheval UNIQUEMENT, §C10) est
`mean±2SEM` sur les seuils -- gravée par le schéma, jamais appliquée hors de
cette branche (6 staircases par régime)."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from jsonschema import Draft7Validator

from src.arcC_abx import (REGIME_LAXISTE, STATUT_CONTINUE, STATUT_STOP_2_INVALIDES,
                          evalue_dispersion, lit_log_jsonl)
from src.arcC_calibration import observation_cellule_pic_csf
from src.arcC_timing import JournalTiming, lit_timing_jsonl, resume_timing
from scripts.run_arcC_orchestration import OUT_DIR, lit_manifeste_json

SCHEMA_PATH: Path = ROOT / "schemas" / "arcC-pins-spatial-v1.schema.json"

METHODE_IC: str = "min_max_seuils"
METHODE_IC_COMBINEE: str = "mean±2SEM — branche à-cheval uniquement"

STATUT_RESOLU: str = "RESOLU"
STATUT_INDETERMINE: str = "INDETERMINE"
STATUT_INVALIDE: str = "INVALIDE"

# Régime nécessaire à la branche à-cheval (§C10) : `ic_combine` sur 6 seuils.
N_STAIRCASES_BRANCHE_COMBINEE: int = 6


# --- Schéma : chargement + validation (fail loud AVANT écriture) ------------


def charge_schema(path: str | Path = SCHEMA_PATH) -> dict:
    """Charge `schemas/arcC-pins-spatial-v1.schema.json` -- copie BYTE POUR
    BYTE du contrat gravé (pocCascade2phys). Ne PAS reconstruire de mémoire :
    lu tel quel depuis le disque."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def valide_pins(pins: dict, schema: dict | None = None) -> None:
    """Valide `pins` contre le contrat §C10 (`jsonschema.Draft7Validator`) --
    lève une `ValueError` avec le détail de TOUTES les erreurs (chemin +
    message) si non conforme. AUCUN affaiblissement : c'est cette fonction
    qui garantit qu'un `pins_spatial.json` malformé n'est JAMAIS écrit."""
    schema = schema if schema is not None else charge_schema()
    validateur = Draft7Validator(schema)
    erreurs = sorted(validateur.iter_errors(pins), key=lambda e: [str(p) for p in e.path])
    if erreurs:
        detail = "\n".join(
            f"  - {'.'.join(str(p) for p in e.path) or '<racine>'} : {e.message}"
            for e in erreurs)
        raise ValueError(
            "valide_pins : pins_spatial.json NON CONFORME au contrat §C10 gravé "
            f"({schema.get('$id', 'schema inconnu')}) -- refus d'écrire (fail loud), "
            f"{len(erreurs)} erreur(s) :\n{detail}")


# --- IC primaire (min/max) + IC combiné (mean±2SEM, branche à-cheval) -------


def calcule_ic(seuils: list[float]) -> list[float] | None:
    """IC PRIMAIRE (§C10 `methode_ic_primaire`) : `[min(seuils), max(seuils)]`
    -- `None` si `seuils` est vide (rien à borner, jamais un IC fabriqué)."""
    if not seuils:
        return None
    return [float(min(seuils)), float(max(seuils))]


def calcule_ic_combine(seuils: list[float]) -> list[float] | None:
    """IC COMBINÉ (§C10 `methode_ic_combinee`, branche à-cheval UNIQUEMENT) :
    `[moyenne - 2*SEM, moyenne + 2*SEM]`, `SEM = std(ddof=1)/sqrt(n)` --
    `None` si `< 2` seuils (SEM incalculable, jamais fabriqué)."""
    arr = np.asarray(seuils, dtype=np.float64)
    if arr.size < 2:
        return None
    moyenne = float(arr.mean())
    sem = float(arr.std(ddof=1) / math.sqrt(arr.size))
    return [moyenne - 2.0 * sem, moyenne + 2.0 * sem]


def _valeur_finie_ou_none(x: float | None) -> float | None:
    """`None`/NaN/±inf -> `None` (jamais un flottant non-JSON-safe rapporté :
    `dispersion_relative` peut valoir `inf` quand la moyenne des seuils est
    nulle, cf. `evalue_dispersion`)."""
    if x is None:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return float(x)


# --- Agrégation d'UN régime -> objet `regime` du schéma (statut 3-voies) ----


def _calcule_taux_catch_global(sessions: list[dict]) -> float | None:
    """Taux de réussite catch PONDÉRÉ sur TOUS les essais catch du régime
    (somme des `n_catch`/`n_catch_ok` de chaque staircase, PAS la moyenne des
    taux par staircase -- pondère par le nombre d'essais réel). `None` si
    aucun catch dans tout le régime (jamais un taux fabriqué sur zéro essai)."""
    n_catch = sum(s["validite"]["n_catch"] for s in sessions)
    n_catch_ok = sum(s["validite"]["n_catch_ok"] for s in sessions)
    return (n_catch_ok / n_catch) if n_catch > 0 else None


def construit_staircases_brutes(sessions: list[dict]) -> list[dict]:
    """`staircases[]` du schéma : seuils BRUTS par staircase (complètes ET
    incomplètes -- la branche à-cheval recalcule dessus, §C10). Mappe
    DIRECTEMENT les champs déjà calculés par l'orchestrateur (`run_escalier`
    + `evalue_validite_session`) -- pas de relecture des logs bruts (déjà
    fait, en amont, par le harnais qui a produit la session)."""
    return [
        dict(numero=s["numero_staircase"], seuil=s["seuil"], n_essais=s["n_essais"],
            n_renversements=s["n_reversals"], renversements_moyennes=6,
            taux_catch=s["validite"]["taux_reussite_catch"], valide=s["validite"]["valide"])
        for s in sessions]


def determine_statut_regime(sessions: list[dict], statut_orchestration: str
                            ) -> tuple[str, str | None, dict]:
    """Statut 3-voies (§C10) :
      - au moins une session INVALIDE (§C5, catch < 90%) OU arrêt 2-invalides
        OU aucune staircase complète (pin structurellement incalculable) ->
        `INVALIDE` (motif explicite, `jnd`/`ic` = `None`) ;
      - sinon staircases incohérentes (`evalue_dispersion.coherent` faux :
        `< 3` staircases complètes OU dispersion `> 30%`) -> `INDETERMINE`
        (motif adapté à la cause, `jnd`/`ic` NON `None` -- valeur centrale
        rapportée mais marquée non fiable, `null` réservé à `INVALIDE`) ;
      - sinon `RESOLU`.
    Renvoie `(statut, motif_statut_ou_None, dispersion)` -- `dispersion` =
    sortie de `evalue_dispersion` (réutilisée pour `dispersion_relative`)."""
    sessions_invalides = any(not s["validite"]["valide"] for s in sessions)
    arret_2_invalides = statut_orchestration == STATUT_STOP_2_INVALIDES
    seuils_complets = [s["seuil"] for s in sessions if s["complet"] and s["seuil"] is not None]
    dispersion = evalue_dispersion(seuils_complets)

    if sessions_invalides or arret_2_invalides or dispersion["n"] == 0:
        if arret_2_invalides:
            motif = ("arrêt anticipé (§C5) : 2 sessions consécutives invalides "
                     "(catch < 90 %) -- protocole/instrument à remonter.")
        elif sessions_invalides:
            motif = "au moins une session invalide (§C5, taux_catch < 90 %)."
        else:
            motif = "aucune staircase complète dans ce régime -- pin non calculable."
        return STATUT_INVALIDE, motif, dispersion

    if not dispersion["coherent"]:
        if dispersion["n"] < 3:
            motif = f"moins de 3 staircases complètes (n={dispersion['n']} < 3, §C5)."
        else:
            motif = (f"dispersion inter-staircases {dispersion['dispersion_relative'] * 100:.1f}"
                     "% > 30 % (§C5-3) -- valeur centrale rapportée, non fiable.")
        return STATUT_INDETERMINE, motif, dispersion

    return STATUT_RESOLU, None, dispersion


def construit_regime_pins(regime_nom: str, sessions: list[dict], statut_orchestration: str,
                          branche_combinee_active: bool) -> dict:
    """Construit l'objet `regime` du schéma (§C10, `#/definitions/regime`)
    pour UN régime : statut 3-voies (`determine_statut_regime`), pin/IC
    primaire (valeur centrale = `dispersion["moyenne"]`, IC = `calcule_ic` sur
    les seuils complets -- `None` ssi `INVALIDE`), staircases brutes,
    dispersion/taux-catch de rollup, et `ic_combine` (branche à-cheval,
    UNIQUEMENT si `branche_combinee_active` ET statut `!= INVALIDE` ET
    >= 2 seuils complets -- sinon champ ABSENT, jamais fabriqué)."""
    statut, motif, dispersion = determine_statut_regime(sessions, statut_orchestration)
    seuils_complets = [s["seuil"] for s in sessions if s["complet"] and s["seuil"] is not None]

    if statut == STATUT_INVALIDE:
        jnd, ic = None, None
    else:
        jnd = dispersion["moyenne"]
        ic = calcule_ic(seuils_complets)

    resultat = dict(
        statut=statut, jnd=jnd, ic=ic, staircases=construit_staircases_brutes(sessions),
        dispersion_relative=_valeur_finie_ou_none(dispersion["dispersion_relative"]),
        taux_catch_global=_calcule_taux_catch_global(sessions))
    if motif is not None:
        resultat["motif_statut"] = motif

    if branche_combinee_active and statut != STATUT_INVALIDE:
        ic_combine = calcule_ic_combine(seuils_complets)
        if ic_combine is not None:
            resultat["ic_combine"] = ic_combine

    return resultat


def calcule_branche_combinee_active(manifeste_regimes: dict) -> bool:
    """`true` ssi la session SUPPLÉMENTAIRE de la cellule à-cheval a tourné
    dans LES DEUX régimes (§C10) : `>= 6` staircases (brutes) par régime.
    `false` sinon (cas standard, 3 staircases)."""
    return all(len(regime_data["sessions"]) >= N_STAIRCASES_BRANCHE_COMBINEE
              for regime_data in manifeste_regimes.values())


def calcule_controle_directionnel(regime_severe: dict, regime_laxiste: dict) -> bool:
    """`ic_disjoints_mauvais_ordre` (§C10) : `true` SSI les deux régimes sont
    `RESOLU` ET l'IC laxiste est ENTIÈREMENT sous l'IC sévère (attendu :
    JND_lax >= JND_sev -- violation = disjoints ET mauvais ordre). IC qui se
    chevauchent -> `false` (bruit, pas un défaut, §C10) ; un régime
    non-RESOLU -> `false` (le contrôle ne peut pas se prononcer)."""
    if regime_severe["statut"] != STATUT_RESOLU or regime_laxiste["statut"] != STATUT_RESOLU:
        return False
    ic_severe, ic_laxiste = regime_severe["ic"], regime_laxiste["ic"]
    return bool(ic_laxiste[1] < ic_severe[0])


# --- Calibration, sources, params, timing, artefacts ------------------------


def construit_calibration(manifeste: dict) -> dict:
    """`calibration` du schéma (§C10) : `long_ref_px`/`long_ref_mm`/
    `distance_mm`/`px_par_degre` depuis `conditions_validite.calibration` ;
    `taille_domaine_deg` = `taille_domaine_px / ppd` ; `cellule_arcmin` =
    `observation_cellule_pic_csf(ppd)["cellule_arcmin_pic_csf"]` (REPORT §C7
    pièce 3, à la géométrie pic-CSF, indépendant de la géométrie RÉELLEMENT
    utilisée par la campagne) ; `luminosite`/`conditions` = valeur notée, ou
    `"(synetique)"` si `None` (sujet synthétique -- le schéma exige des
    STRING, jamais un référent commité, §C10)."""
    cv = manifeste["conditions_validite"]
    calib = cv["calibration"]
    ppd = calib["ppd"]
    taille_domaine_px = manifeste["config_affichage"]["taille_domaine_px"]
    cellule_arcmin = observation_cellule_pic_csf(ppd)["cellule_arcmin_pic_csf"]
    return dict(
        long_ref_px=calib["long_ref_px"], long_ref_mm=calib["long_ref_mm"],
        distance_mm=calib["distance_mm"], px_par_degre=ppd,
        taille_domaine_deg=taille_domaine_px / ppd, cellule_arcmin=cellule_arcmin,
        luminosite=(cv["luminosite"] if cv["luminosite"] is not None else "(synthetique)"),
        conditions=(cv["conditions"] if cv["conditions"] is not None else "(synthetique)"))


def construit_params(manifeste: dict, branche_combinee_active: bool) -> dict:
    """`params` du schéma (§C10) -- constantes gravées + `seuil_exclusion` du
    manifeste + `branche_combinee_active` (calculée, cf.
    `calcule_branche_combinee_active`)."""
    pg = manifeste["parametres_graves"]
    return dict(
        unites="fraction (0.04 = 4 %)", ancre_budget=pg["ancre_budget"],
        seuil_exclusion=pg["seuil_exclusion"], methode_ic_primaire=METHODE_IC,
        methode_ic_combinee=METHODE_IC_COMBINEE,
        branche_combinee_active=branche_combinee_active)


def construit_sources(manifeste: dict) -> dict:
    """`sources` du schéma (§C10) : `utilisees` = sources totales - exclues ;
    `exclues` = liste NOMMÉE `{id, chi_ancre}` (`id="s{seed}_L{L}"`) -- JAMAIS
    une exclusion silencieuse."""
    exclusions = manifeste["exclusions"]
    exclues = [dict(id=f"s{e['seed']}_L{e['L']}", chi_ancre=e["delta_chi_ancre"])
              for e in exclusions["entrees"] if e["exclu"]]
    return dict(utilisees=exclusions["n_sources"] - exclusions["n_exclues"], exclues=exclues)


def _chemin_timing(log_path: str) -> Path:
    """Chemin du sidecar timing d'UNE staircase -- MÊME convention que
    l'orchestrateur (`scripts/run_arcC_orchestration.py`,
    `_fabrique_repondre_humain.cleanup`) : `session_<regime>_<k>.timing.jsonl`
    à côté du log `session_<regime>_<k>.jsonl`."""
    p = Path(log_path)
    return p.with_name(p.stem + ".timing.jsonl")


def construit_timing_laxiste(sessions_laxiste: list[dict]) -> dict:
    """`timing_laxiste` du schéma (§C9 pièce 1) : lit les sidecars
    `*.timing.jsonl` du régime laxiste (mêmes chemins que les logs), les
    fusionne (`resume_timing` une seule fois sur l'union des enregistrements)
    -- `exposition_moyenne_s` = moyenne réalisée de la catégorie exposition,
    `derive_max_pct` = pire écart |réalisé/nominal-1| en % toutes catégories
    comparées confondues. AUCUN sidecar trouvé (sujet synthétique, sans glue
    matplotlib) -> `0.0`/`0.0`, informatif nul, NOTÉ (imprimé) -- conforme au
    schéma, jamais une exception (le timing n'est qu'informatif hors régime
    laxiste réellement mesuré sur humain)."""
    journal_combine = JournalTiming()
    n_sidecars_trouves = 0
    for s in sessions_laxiste:
        chemin = _chemin_timing(s["log_path"])
        if chemin.exists():
            n_sidecars_trouves += 1
            journal_combine.enregistrements.extend(lit_timing_jsonl(chemin).enregistrements)

    if n_sidecars_trouves == 0:
        print("[NOTE timing §C9] aucun sidecar *.timing.jsonl trouvé pour le régime laxiste "
              "(sujet synthétique ou glue absente) -- exposition_moyenne_s=0.0, "
              "derive_max_pct=0.0 (informatif nul, conforme au schéma).")
        return dict(exposition_moyenne_s=0.0, derive_max_pct=0.0)

    resume = resume_timing(journal_combine, REGIME_LAXISTE)
    exposition_moyenne_s = resume["categories"]["exposition"]["moyenne_realisee_s"]
    ecarts = [abs(cat["ecart_relatif"]) for cat in resume["categories"].values()
             if cat["ecart_relatif"] is not None]
    return dict(
        exposition_moyenne_s=(exposition_moyenne_s if exposition_moyenne_s is not None else 0.0),
        derive_max_pct=(max(ecarts) * 100.0 if ecarts else 0.0))


def construit_artefacts(manifeste: dict, figures_paths: list[str]) -> dict:
    """`artefacts` du schéma : `logs` = TOUS les chemins de logs bruts (les
    deux régimes confondus) ; `figures` = chemins fournis par l'appelant
    (`main()` -- déterminés AVANT génération, mêmes noms que les fichiers
    réellement écrits par `figure_psychometrique`)."""
    logs = [s["log_path"] for regime_data in manifeste["regimes"].values()
           for s in regime_data["sessions"]]
    return dict(logs=logs, figures=list(figures_paths))


# --- Document racine ---------------------------------------------------------


def construit_pins(manifeste: dict, *, manifeste_path: str | Path,
                   figures_paths: list[str] | None = None) -> dict:
    """Construit le document `pins_spatial.json` CONFORME au schéma §C10
    depuis un manifeste de campagne (`scripts.run_arcC_orchestration.
    orchestre_campagne`). NE VALIDE PAS lui-même (cf. `ecrit_pins_json`, qui
    appelle `valide_pins` avant d'écrire) -- mais REFUSE explicitement
    (`RuntimeError`) le cas structurel sans représentation dans le schéma :
    campagne arrêtée AVANT le lancement des régimes (`statut_global !=
    CONTINUE`, `regimes` vide/absent -- ancien `STOP_TROP_DE_SOURCES_
    EXCLUES`) ou régime manquant."""
    if manifeste.get("statut_global") != STATUT_CONTINUE or not manifeste.get("regimes"):
        raise RuntimeError(
            "construit_pins : campagne sans régimes exploitables "
            f"(statut_global={manifeste.get('statut_global')!r}) -- le contrat §C10 "
            "(schemas/arcC-pins-spatial-v1.schema.json) n'a AUCUNE représentation pour une "
            "campagne arrêtée avant le lancement des régimes (ex. STOP_TROP_DE_SOURCES_EXCLUES) "
            "-- refus explicite, aucun pins_spatial.json ne peut être produit pour cet état.")
    for regime_nom in ("severe", "laxiste"):
        if regime_nom not in manifeste["regimes"]:
            raise RuntimeError(f"construit_pins : régime {regime_nom!r} absent du manifeste.")

    branche_active = calcule_branche_combinee_active(manifeste["regimes"])
    regimes_pins = {
        regime_nom: construit_regime_pins(
            regime_nom, regime_data["sessions"], regime_data["statut_orchestration"],
            branche_active)
        for regime_nom, regime_data in manifeste["regimes"].items()}

    return dict(
        schema="arcC-pins-spatial-v1",
        date_session=manifeste["date_session"], sujet=manifeste["sujet"],
        base_seed=manifeste["base_seed"], commit_harnais=manifeste["commit_harnais"],
        manifeste=str(manifeste_path),
        calibration=construit_calibration(manifeste),
        params=construit_params(manifeste, branche_active),
        sources=construit_sources(manifeste),
        regimes=regimes_pins,
        timing_laxiste=construit_timing_laxiste(manifeste["regimes"]["laxiste"]["sessions"]),
        controle_directionnel=dict(
            attendu=("JND_lax >= JND_sev (portée : violation = IC disjoints dans le mauvais "
                    "ordre)"),
            ic_disjoints_mauvais_ordre=calcule_controle_directionnel(
                regimes_pins["severe"], regimes_pins["laxiste"])),
        artefacts=construit_artefacts(manifeste, figures_paths if figures_paths else []))


# --- pins_spatial.json (I/O + validation fail-loud) --------------------------


def ecrit_pins_json(path: str | Path, pins: dict, *, schema: dict | None = None) -> None:
    """Valide `pins` contre le contrat §C10 (`valide_pins`, lève si non
    conforme) PUIS écrit -- dans cet ORDRE STRICT, jamais l'inverse : un
    `pins_spatial.json` malformé n'est JAMAIS écrit sur disque
    (`allow_nan=False` en ceinture-bretelles : un NaN/inf qui aurait
    échappé à la validation ferait lever `json.dumps`, pas produire un JSON
    invalide silencieusement)."""
    valide_pins(pins, schema)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(pins, indent=2, ensure_ascii=False, allow_nan=False),
                encoding="utf-8")


def lit_pins_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --- Figures psychométriques (sobres, français, viridis) --------------------


def _points_psychometriques(essais: list, n_bins: int = 12) -> list[dict]:
    """Agrège les essais NORMAUX (jamais les catch, hors logique d'escalier)
    d'un régime en `n_bins` classes de Δχ (bords = quantiles, pour répartir
    la MASSE d'essais plutôt qu'un pas linéaire arbitraire) : point =
    (Δχ moyen de la classe, proportion correcte, n)."""
    normaux = [e for e in essais if e.type_essai == "normal"]
    if not normaux:
        return []
    delta = np.array([e.delta_chi_mesure for e in normaux], dtype=np.float64)
    correct = np.array([1.0 if e.correct else 0.0 for e in normaux], dtype=np.float64)
    bords = np.unique(np.quantile(delta, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(bords) < 2:
        return [dict(delta_chi_moyen=float(delta.mean()), proportion_correcte=float(correct.mean()),
                    n=int(delta.size))]
    indices = np.clip(np.digitize(delta, bords[1:-1]), 0, len(bords) - 2)
    points = []
    for i in range(len(bords) - 1):
        masque = indices == i
        if not masque.any():
            continue
        points.append(dict(delta_chi_moyen=float(delta[masque].mean()),
                           proportion_correcte=float(correct[masque].mean()),
                           n=int(masque.sum())))
    return points


def figure_psychometrique(regime_nom: str, sessions: list[dict], resultat_regime: dict,
                          path: Path) -> None:
    """`arcC_psychometrique_<regime>.png` : points (Δχ, proportion correcte)
    agrégés des essais NORMAUX de toutes les staircases du régime, les
    seuils individuels (viridis, une couleur par staircase) et le pin (si
    RESOLU). Sobre, français -- AUCUN verdict de manche imprimé ou tracé
    (pas de ligne de seuil JND 3-4%, cf. §C4 réservée au contrôleur/Task 5)."""
    essais_tous = []
    for s in sessions:
        essais_tous.extend(lit_log_jsonl(s["log_path"]))
    points = _points_psychometriques(essais_tous)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    if points:
        xs = [p["delta_chi_moyen"] * 100.0 for p in points]
        ys = [p["proportion_correcte"] for p in points]
        tailles = [24.0 + 4.0 * p["n"] for p in points]
        ax.scatter(xs, ys, s=tailles, color="#404040", alpha=0.65, zorder=3,
                  edgecolors="none", label="essais agrégés (Δχ, proportion correcte)")

    n_stair = max(len(sessions), 1)
    couleurs = plt.cm.viridis(np.linspace(0.15, 0.85, n_stair))
    for session, couleur in zip(sessions, couleurs):
        if session["seuil"] is not None:
            ax.axvline(session["seuil"] * 100.0, color=couleur, linestyle=":", linewidth=1.3,
                      alpha=0.85,
                      label=f"seuil staircase {session['numero_staircase']}")

    if resultat_regime["jnd"] is not None:
        ax.axvline(resultat_regime["jnd"] * 100.0, color="#B03060", linestyle="-", linewidth=2.0,
                  label=f"pin = {resultat_regime['jnd'] * 100.0:.2f} %", zorder=4)

    ax.set_xlabel("Δχ (%, mesuré)")
    ax.set_ylabel("Proportion correcte")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Arc C / Task 3 -- psychométrique, régime {regime_nom} "
                f"(statut={resultat_regime['statut']})")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --- CLI ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifeste", type=Path, default=OUT_DIR / "manifeste_campagne.json")
    parser.add_argument("--out", type=Path, default=OUT_DIR / "pins_spatial.json")
    parser.add_argument("--figures-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    manifeste = lit_manifeste_json(args.manifeste)
    figures_paths = [str(args.figures_dir / f"arcC_psychometrique_{regime_nom}.png")
                     for regime_nom in ("severe", "laxiste") if regime_nom in manifeste.get("regimes", {})]
    pins = construit_pins(manifeste, manifeste_path=args.manifeste, figures_paths=figures_paths)
    ecrit_pins_json(args.out, pins)  # valide §C10 AVANT d'écrire, fail loud sinon

    print("=" * 78)
    print("ARC C / TASK 3 -- PINS SPATIAUX (§C10, AUCUN verdict de manche)")
    print("=" * 78)
    for regime_nom, regime_data in manifeste.get("regimes", {}).items():
        resultat = pins["regimes"][regime_nom]
        print(f"  régime={regime_nom} : statut={resultat['statut']}  jnd={resultat['jnd']}  "
              f"ic={resultat['ic']}")
        path_fig = args.figures_dir / f"arcC_psychometrique_{regime_nom}.png"
        figure_psychometrique(regime_nom, regime_data["sessions"], resultat, path_fig)
        print(f"  [REPORT] -> {path_fig}")
    print(f"[REPORT] -> {args.out}")


if __name__ == "__main__":
    main()
