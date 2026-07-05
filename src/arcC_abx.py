"""Arc C / Task 2 — logique PURE du harnais ABX + escalier adaptatif.

Contrat : addendum §C0 (régimes), §C3 (escalier, sélection de stimulus), §C5
(catch trials, gardes anti-fabrication) de PREREGISTRATION.md (pocCascade2phys,
commits `c494d50`/`5d13bc4`/`f39b402`), reproduit verbatim dans
`.superpowers/sdd/arcC-task2-abx-brief.md`. **AUCUN sujet humain ici** : ce
module est validé en test sur un sujet SYNTHÉTIQUE (`src/arcC_synthetic.py`) ;
les vraies sessions sont Task 3.

Principe d'architecture LOAD-BEARING (brief) : ce module est de la logique
PURE -- escalier 2-down-1-up, scoring ABX, sélection de stimulus par Δχ
mesuré (non-monotone-safe), insertion des catch trials, calculs de validité
§C5, dispersion inter-staircases, format des logs (JSONL) + replay. ZÉRO
dépendance à matplotlib/clavier. Un **sujet** est une fonction
`repondre(essai: EssaiPropose) -> "A"|"B"` -- la coquille interactive
(`scripts/run_arcC_session.py`, Task 3) ne fait qu'appeler cette logique en
substituant un humain au sujet synthétique de test.

Les DEUX RÉGIMES (§C0, `Regime`/`REGIME_SEVERE`/`REGIME_LAXISTE`) ne
changent QUE la PRÉSENTATION (simultané/séquentiel, timings, masque) --
`run_escalier` ne prend le nom du régime que comme MÉTADONNÉE loggée
(`EssaiJournal.regime`), jamais comme paramètre de la logique d'escalier
elle-même (cf. tests, famille 8 : même sujet synthétique -> même seuil, quel
que soit le régime passé en métadonnée)."""
from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import numpy as np

from scripts.run_arcA_measure import L_LIST, SEEDS, _load_history
from src.arcC_stimuli import courbe_delta_chi

Reponse = Literal["A", "B"]

# --- Grille des champs sources (roving, §C3) ---------------------------------
# Les 20 (seed, L) -- réutilise SEEDS/L_LIST de la manche 1 (aucune redéfinition
# locale, cf. convention `scripts/build_arcC_stimuli.py`).
PAIRES_SOURCES: tuple[tuple[int, int], ...] = tuple(itertools.product(SEEDS, L_LIST))

# BUDGETS reste LOCAL (pas de réimport croisé de script à script pour cette
# grille, cf. `scripts/build_arcC_stimuli.py`) : budget fixe par staircase
# (choix nommé, brief) -- 256 bracket la fenêtre JND [~1.8-7.9] % (catalogue
# Task 1, `outputs/arcC/stimuli_catalogue.npz`). BUDGET_CATCH = le plus BAS
# de la grille manche 1 = dégradation la plus forte = Δχ le plus grand
# atteignable, sans ambiguïté (§C5).
BUDGET_DEFAUT: int = 256
BUDGET_CATCH: int = 32
N_POINTS_BANC: int = 161  # >= 100 points exigés (§C3) -- banc fin balayé densément


# --- Banc fin (Δχ mesuré, densément échantillonné en t) ----------------------


@dataclass(frozen=True, eq=False)
class BancFin:
    """Banc fin (seed, L, budget) : Δχ(t) MESURÉ (R1, `courbe_delta_chi`) sur
    une grille dense de `t` -- la sélection de stimulus par niveau cible
    (`selectionne_stimulus`) cherche dans CE banc, jamais en interpolant."""
    seed: int
    L: int
    budget: int
    ts: np.ndarray
    delta_chi: np.ndarray


def construit_banc_fin(seed: int, L: int, budget: int, *,
                       n_points: int = N_POINTS_BANC) -> BancFin:
    """Construit le banc fin (seed, L, budget) : champ source chargé via
    `_load_history` (garde-fou `b0` inclus, manche 1), `ts = linspace(0, 1,
    n_points)`, Δχ(t) mesuré par `courbe_delta_chi` (Task 1, R1) -- une passe
    R1 par t, aucune interpolation, aucune hypothèse de monotonie."""
    hist = _load_history(seed)
    s_true = np.asarray(hist[f"s_L{L}"], dtype=np.float64)
    ts = np.linspace(0.0, 1.0, n_points)
    delta_chi = courbe_delta_chi(s_true, budget, ts)
    return BancFin(seed=seed, L=L, budget=budget, ts=ts, delta_chi=delta_chi)


class BanqueBancs:
    """Cache mémoïsé des bancs fins `(seed, L, budget) -> BancFin`, construits
    À LA DEMANDE (lazy) : le roving ne visite pas nécessairement les 20
    sources dans une staircase courte, inutile de tout précalculer. Chaque
    banc est une fonction PURE de sa clé -- le cache ne casse aucun
    déterminisme (mêmes entrées -> même banc, qu'il soit ou non déjà en
    cache)."""

    def __init__(self, n_points: int = N_POINTS_BANC) -> None:
        self._n_points = n_points
        self._cache: dict[tuple[int, int, int], BancFin] = {}

    def banc(self, seed: int, L: int, budget: int) -> BancFin:
        cle = (seed, L, budget)
        if cle not in self._cache:
            self._cache[cle] = construit_banc_fin(seed, L, budget, n_points=self._n_points)
        return self._cache[cle]


# --- Sélection de stimulus par Δχ mesuré (non-monotone-safe, §C3) -----------


def _indice_extremum(valeurs: np.ndarray, ts: np.ndarray, cible: float | None) -> int:
    """indice minimisant `|valeurs - cible|` (si `cible` fourni) ou maximisant
    `valeurs` (si `cible=None`) ; tie-break déterministe = plus petit `t`.
    Recherche EXHAUSTIVE (`np.abs`/`.min()` sur tout le tableau) -- ne suppose
    AUCUNE monotonie de `valeurs` en fonction de `ts` (§C3)."""
    valeurs = np.asarray(valeurs, dtype=np.float64)
    ts = np.asarray(ts, dtype=np.float64)
    score = -valeurs if cible is None else np.abs(valeurs - float(cible))
    meilleur = score.min()
    indices = np.flatnonzero(score == meilleur)
    return int(indices[np.argmin(ts[indices])])


def selectionne_stimulus(ts: np.ndarray, delta_chi_mesures: np.ndarray,
                         delta_chi_cible: float) -> int:
    """`argmin |Δχ_mesuré − Δχ*|`, tie-break déterministe = plus petit `t`
    (§C3). Non-monotone-safe : recherche exhaustive, valide quel que soit le
    profil (observé) de la courbe Δχ(t)."""
    return _indice_extremum(delta_chi_mesures, ts, delta_chi_cible)


def selectionne_stimulus_fort(ts: np.ndarray, delta_chi_mesures: np.ndarray) -> int:
    """`argmax Δχ_mesuré`, tie-break déterministe = plus petit `t` -- le
    stimulus CATCH (§C5) : le plus grand Δχ ATTEIGNABLE du champ courant, SANS
    supposer qu'il soit à `t=1` (non-monotone-safe, §C3)."""
    return _indice_extremum(delta_chi_mesures, ts, None)


# --- Régimes (§C0) -- PRÉSENTATION uniquement, jamais la logique pure -------


@dataclass(frozen=True)
class Regime:
    """Un régime = un objet de config de PRÉSENTATION (timings, masque,
    simultané/séquentiel) -- consommé PAR LA COQUILLE uniquement. `run_escalier`
    ne reçoit que `regime.nom` (métadonnée loggée), jamais l'objet complet :
    la logique d'escalier est IDENTIQUE aux deux régimes (cf. tests, famille 8)."""
    nom: str
    simultane: bool
    inspection_libre: bool
    exposition_s: float | None
    retention_s: float | None
    masque_bruite: bool


REGIME_SEVERE = Regime(nom="severe", simultane=True, inspection_libre=True,
                       exposition_s=None, retention_s=None, masque_bruite=False)
REGIME_LAXISTE = Regime(nom="laxiste", simultane=False, inspection_libre=False,
                        exposition_s=2.0, retention_s=5.0, masque_bruite=True)


# --- Essais (proposé / journalisé) ------------------------------------------


@dataclass(frozen=True, eq=False)
class EssaiPropose:
    """Ce qu'un sujet (synthétique ou humain, via la coquille) reçoit pour
    répondre : la difficulté de la paire (`delta_chi`, mesuré) -- le sujet ne
    voit JAMAIS `reponse_correcte` pour décider (elle ne sert qu'au scoring
    après coup), exactement comme un humain verrait les deux stimuli sans
    connaître la bonne réponse."""
    numero_staircase: int
    indice_essai: int
    seed_source: int
    L_source: int
    budget: int
    t: float
    delta_chi: float
    type_essai: str          # "normal" | "catch"
    reponse_correcte: Reponse


@dataclass(frozen=True, eq=False)
class EssaiJournal:
    """Enregistrement COMPLET d'un essai, un par ligne de log (JSONL) :
    n° staircase, régime, champ source (seed, L), budget, `t` présenté, Δχ
    MESURÉ (jamais le cible, jamais `t` seul comme critère), type, réponse,
    correct/incorrect, latence, niveau courant de l'escalier, flag
    renversement. `latence_s` est le SEUL champ EXCLU de la comparaison
    bit-exacte de déterminisme (temps d'exécution réel de `repondre`, jamais
    reproductible bit-à-bit -- cf. tests de replay)."""
    numero_staircase: int
    indice_essai: int
    regime: str
    seed_source: int
    L_source: int
    budget: int
    t: float
    delta_chi_mesure: float
    type_essai: str
    reponse: Reponse
    correct: bool
    latence_s: float
    niveau_delta_chi: float
    est_renversement: bool

    def to_dict(self) -> dict:
        return dict(
            numero_staircase=self.numero_staircase, indice_essai=self.indice_essai,
            regime=self.regime, seed_source=self.seed_source, L_source=self.L_source,
            budget=self.budget, t=self.t, delta_chi_mesure=self.delta_chi_mesure,
            type_essai=self.type_essai, reponse=self.reponse, correct=self.correct,
            latence_s=self.latence_s, niveau_delta_chi=self.niveau_delta_chi,
            est_renversement=self.est_renversement)

    @staticmethod
    def from_dict(d: dict) -> "EssaiJournal":
        return EssaiJournal(**d)

    def egal_hors_latence(self, autre: "EssaiJournal") -> bool:
        """Égalité de TOUS les champs SAUF `latence_s` (cf. docstring classe) --
        c'est la comparaison utilisée par les tests de déterminisme/replay."""
        a, b = self.to_dict(), autre.to_dict()
        a.pop("latence_s")
        b.pop("latence_s")
        return a == b


# --- Escalier 2-down-1-up (§C3) ----------------------------------------------


@dataclass(frozen=True)
class ParametresEscalier:
    """Paramètres de l'escalier -- décisions nommées (cf. rapport) :
      - `facteur_pas` : pas MULTIPLICATIF CONSTANT en espace Δχ (pas de
        réduction de pas après renversement -- simplicité, testabilité de la
        règle 2-down-1-up sur une séquence construite à la main, cf. tests).
      - `n_reversals_cible` (8) > `n_reversals_pour_seuil` (6, §C3 gravé) :
        on court jusqu'à 8 renversements puis on moyenne les 6 DERNIERS
        (§C3 : « moyenne des 6 derniers renversements » -- implique qu'on
        peut en collecter plus et n'en garder que la fin, pratique standard
        pour écarter les premiers renversements, plus bruités/loin de la
        convergence).
      - `n_essais_max` (200) : garde-fou anti-boucle infinie (le sujet
        synthétique logistique + un pas raisonnable converge largement avant).
      - `taille_bloc_catch` : 1 catch tiré uniformément par bloc de 10
        essais (~10 %, §C5) -- cf. `positions_catch`.
    """
    delta_chi_initial: float = 0.05
    facteur_pas: float = 1.30
    n_reversals_cible: int = 8
    n_reversals_pour_seuil: int = 6
    n_essais_max: int = 200
    delta_chi_min: float = 1e-4
    delta_chi_max: float = 0.5
    taille_bloc_catch: int = 10
    seuil_reussite_catch: float = 0.90


@dataclass
class EtatEscalier:
    """État MUTABLE interne de l'escalier -- `avance_escalier` le met à jour
    en place, un essai NORMAL à la fois."""
    niveau: float
    compteur_correct: int = 0
    dernier_sens: str | None = None
    reversals: list[float] = field(default_factory=list)


def avance_escalier(etat: EtatEscalier, correct: bool, delta_chi_presente: float,
                    params: ParametresEscalier) -> bool:
    """Applique la règle 2-down-1-up à UN essai NORMAL déjà scoré
    (`correct`/incorrect) : 2 bonnes réponses CONSÉCUTIVES -> niveau plus
    difficile (Δχ plus petit, `/facteur_pas`) ; 1 mauvaise -> niveau plus
    facile (Δχ plus grand, `*facteur_pas`). Renversement = le SENS du
    mouvement qui vient d'être décidé diffère du DERNIER mouvement effectué ;
    la valeur enregistrée est `delta_chi_presente` (le Δχ RÉELLEMENT présenté
    à CET essai, §C3 -- pas le niveau-cible abstrait), au moment où le
    changement de sens est détecté (avant d'appliquer le nouveau pas). Modifie
    `etat` EN PLACE ; renvoie `est_renversement` pour cet essai."""
    est_renversement = False
    if correct:
        etat.compteur_correct += 1
        if etat.compteur_correct >= 2:
            sens = "descend"
            if etat.dernier_sens is not None and sens != etat.dernier_sens:
                etat.reversals.append(delta_chi_presente)
                est_renversement = True
            etat.dernier_sens = sens
            etat.niveau = min(params.delta_chi_max,
                              max(params.delta_chi_min, etat.niveau / params.facteur_pas))
            etat.compteur_correct = 0
    else:
        sens = "monte"
        if etat.dernier_sens is not None and sens != etat.dernier_sens:
            etat.reversals.append(delta_chi_presente)
            est_renversement = True
        etat.dernier_sens = sens
        etat.niveau = min(params.delta_chi_max,
                          max(params.delta_chi_min, etat.niveau * params.facteur_pas))
        etat.compteur_correct = 0
    return est_renversement


def calcule_seuil(reversals: list[float],
                  n_reversals_pour_seuil: int = 6) -> tuple[float | None, bool]:
    """Seuil = moyenne des `n_reversals_pour_seuil` (6, gravé §C3) DERNIERS
    renversements. `< n_reversals_pour_seuil` renversements -> (None, False) :
    staircase INCOMPLÈTE, AUCUN seuil inventé."""
    complet = len(reversals) >= n_reversals_pour_seuil
    if not complet:
        return None, False
    return float(np.mean(reversals[-n_reversals_pour_seuil:])), True


# --- Catch trials (§C5) -------------------------------------------------


def positions_catch(n_essais: int, rng_catch: np.random.Generator,
                    taille_bloc: int = 10) -> np.ndarray:
    """Masque booléen "catch" (longueur `n_essais`) : PAR BLOCS de
    `taille_bloc` essais consécutifs, UNE position catch tirée uniformément
    (`rng_catch.integers(0, taille_bloc)`) -- garantit EXACTEMENT 1 catch par
    bloc plein (~10 % pour `taille_bloc=10`), position reproductible (seed de
    `rng_catch`). Le dernier bloc, s'il est PARTIEL (n_essais % taille_bloc
    != 0), peut contenir 0 ou 1 catch selon que la position tirée tombe dans
    la partie tronquée -- c'est la « tolérance sur l'arrondi » (brief,
    tests exigés #4)."""
    n_blocs = -(-n_essais // taille_bloc)  # ceil division, sans dépendance math
    masque = np.zeros(n_blocs * taille_bloc, dtype=bool)
    for i in range(n_blocs):
        pos = int(rng_catch.integers(0, taille_bloc))
        masque[i * taille_bloc + pos] = True
    return masque[:n_essais]


def evalue_validite_session(essais: list[EssaiJournal],
                            seuil_reussite_catch: float = 0.90) -> dict:
    """Validité §C5 : fraction de catch trials réussis >= `seuil_reussite_catch`
    (90 %, gravé) -> VALIDE, sinon INVALIDE. Aucun catch dans la session ->
    INVALIDE explicite (pas de garde qui ne peut jamais se déclencher)."""
    catches = [e for e in essais if e.type_essai == "catch"]
    n = len(catches)
    n_ok = sum(1 for e in catches if e.correct)
    taux = (n_ok / n) if n > 0 else float("nan")
    valide = (n > 0) and (taux >= seuil_reussite_catch)
    return dict(n_catch=n, n_catch_ok=n_ok, taux_reussite_catch=taux,
               seuil_reussite_catch=seuil_reussite_catch, valide=valide)


STATUT_CONTINUE: str = "CONTINUE"
STATUT_STOP_2_INVALIDES: str = "STOP_2_SESSIONS_INVALIDES"


def verifie_arret_2_invalides(historique_valide: list[bool]) -> str:
    """2 sessions invalides CONSÉCUTIVES -> `STATUT_STOP_2_INVALIDES` (remonter
    fatigue/protocole), sinon `STATUT_CONTINUE`. Statut explicite -- jamais
    une exception silencieuse ni une moyenne complaisante."""
    if len(historique_valide) >= 2 and not historique_valide[-1] and not historique_valide[-2]:
        return STATUT_STOP_2_INVALIDES
    return STATUT_CONTINUE


# --- Dispersion inter-staircases + cellule "instrument muet" (§C5) ----------


STATUT_RESOLU: str = "RESOLU"
STATUT_INSTRUMENT_NE_PEUT_PAS_REPONDRE: str = "INSTRUMENT_NE_PEUT_PAS_REPONDRE"


def evalue_dispersion(seuils: list[float], seuil_dispersion_relative: float = 0.30,
                      n_min_staircases: int = 3) -> dict:
    """Dispersion inter-staircases = COEFFICIENT DE VARIATION (écart-type
    ÉCHANTILLON `ddof=1` / moyenne) des seuils des staircases COMPLÈTES d'une
    condition -- choix nommé (le brief autorise CV ou étendue relative ;
    ddof=1 est le choix standard pour une variabilité estimée sur un petit
    échantillon, n=3 typiquement). `coherent` exige `n >= n_min_staircases`
    (3, gravé) ET dispersion <= `seuil_dispersion_relative` (30 %, gravé) --
    sinon la condition est INDÉTERMINÉE (`pin=None`), jamais une moyenne
    complaisante sur des staircases dispersées ou en nombre insuffisant."""
    arr = np.asarray(seuils, dtype=np.float64)
    n = int(arr.size)
    moyenne = float(arr.mean()) if n > 0 else float("nan")
    ecart_type = float(arr.std(ddof=1)) if n >= 2 else 0.0
    dispersion_relative = (ecart_type / moyenne) if moyenne != 0.0 else float("inf")
    coherent = (n >= n_min_staircases) and (dispersion_relative <= seuil_dispersion_relative)
    return dict(n=n, moyenne=moyenne, ecart_type=ecart_type,
               dispersion_relative=dispersion_relative,
               seuil_dispersion_relative=seuil_dispersion_relative,
               n_min_staircases=n_min_staircases, coherent=coherent,
               pin=(moyenne if coherent else None))


def statut_condition(validites_sessions: list[bool], dispersion: dict) -> str:
    """Cellule "l'instrument ne peut pas répondre" EXPLICITE (§C5-5) : au
    moins une session invalide OU staircases incohérentes ->
    `STATUT_INSTRUMENT_NE_PEUT_PAS_REPONDRE` (jamais un pin fabriqué),
    sinon `STATUT_RESOLU` (le pin = `dispersion["pin"]`)."""
    if not all(validites_sessions):
        return STATUT_INSTRUMENT_NE_PEUT_PAS_REPONDRE
    if not dispersion["coherent"]:
        return STATUT_INSTRUMENT_NE_PEUT_PAS_REPONDRE
    return STATUT_RESOLU


# --- Session d'une staircase (roving + catch + escalier + logs) ------------


@dataclass(frozen=True, eq=False)
class ResultatEscalier:
    """Résultat complet d'une staircase : les essais journalisés, les
    renversements (Δχ mesurés), le seuil (`None` si incomplet), et la
    provenance (seeds, budget, régime, paramètres) -- nécessaire pour le
    replay (`rejoue_escalier`)."""
    numero_staircase: int
    regime: str
    essais: list[EssaiJournal]
    reversals: list[float]
    seuil: float | None
    complet: bool
    seed_roving: int
    seed_catch: int
    budget: int
    params: ParametresEscalier


def _tire_source(rng: np.random.Generator) -> tuple[int, int]:
    """Rôdage du champ source (roving, §C3) : tire uniformément l'un des 20
    (seed, L) via `rng` -- RNG SEEDÉE et CONSIGNÉE (`seed_roving` de
    `run_escalier`), le sujet ne peut pas mémoriser une texture."""
    idx = int(rng.integers(0, len(PAIRES_SOURCES)))
    return PAIRES_SOURCES[idx]


def _tire_reponse_correcte(rng: np.random.Generator) -> Reponse:
    """Contrebalancement anti-biais de position (A/B) -- tiré de la MÊME RNG
    de roving (`seed_roving`), dans l'ordre : source PUIS position, par essai."""
    return "A" if int(rng.integers(0, 2)) == 0 else "B"


def autre_reponse(r: Reponse) -> Reponse:
    """L'autre position ("A"<->"B") -- utilisé par un sujet qui répond
    INCORRECTEMENT (cf. `src/arcC_synthetic.py`)."""
    return "B" if r == "A" else "A"


def run_escalier(*, numero_staircase: int, regime_nom: str,
                 repondre: Callable[[EssaiPropose], Reponse],
                 seed_roving: int, seed_catch: int,
                 budget: int = BUDGET_DEFAUT,
                 params: ParametresEscalier = ParametresEscalier(),
                 banques: BanqueBancs | None = None) -> ResultatEscalier:
    """Fait tourner UNE staircase 2-down-1-up complète : à chaque essai, tire
    le champ source (roving, `seed_roving`), décide si l'essai est un catch
    (`seed_catch`, blocs de 10, §C5), sélectionne le stimulus (banc fin,
    `selectionne_stimulus`/`selectionne_stimulus_fort`), appelle `repondre`
    (synthétique en test, humain via la coquille), score la réponse, met à
    jour l'escalier (essais NORMAUX uniquement -- les catch sont HORS logique
    d'escalier, §C5) et journalise. S'arrête à `n_reversals_cible`
    renversements ou `n_essais_max` essais (garde-fou), selon la première
    échéance atteinte.

    `regime_nom` est une PURE MÉTADONNÉE loggée (`EssaiJournal.regime`) --
    n'affecte JAMAIS la logique d'escalier (cf. tests, famille 8)."""
    banques = banques if banques is not None else BanqueBancs()
    rng_roving = np.random.default_rng(seed_roving)
    rng_catch = np.random.default_rng(seed_catch)
    masque_catch = positions_catch(params.n_essais_max, rng_catch, params.taille_bloc_catch)

    etat = EtatEscalier(niveau=params.delta_chi_initial)
    essais: list[EssaiJournal] = []
    indice = 0
    while len(etat.reversals) < params.n_reversals_cible and indice < params.n_essais_max:
        seed_src, L_src = _tire_source(rng_roving)
        reponse_correcte = _tire_reponse_correcte(rng_roving)
        est_catch = bool(masque_catch[indice])

        if est_catch:
            banc = banques.banc(seed_src, L_src, BUDGET_CATCH)
            i_stim = selectionne_stimulus_fort(banc.ts, banc.delta_chi)
            budget_essai = BUDGET_CATCH
        else:
            banc = banques.banc(seed_src, L_src, budget)
            i_stim = selectionne_stimulus(banc.ts, banc.delta_chi, etat.niveau)
            budget_essai = budget
        t_presente = float(banc.ts[i_stim])
        delta_chi_presente = float(banc.delta_chi[i_stim])

        essai_propose = EssaiPropose(
            numero_staircase=numero_staircase, indice_essai=indice, seed_source=seed_src,
            L_source=L_src, budget=budget_essai, t=t_presente, delta_chi=delta_chi_presente,
            type_essai="catch" if est_catch else "normal", reponse_correcte=reponse_correcte)

        t0 = time.perf_counter()
        reponse = repondre(essai_propose)
        latence = time.perf_counter() - t0
        if reponse not in ("A", "B"):
            raise ValueError(f"run_escalier : réponse invalide {reponse!r} (attendu 'A'/'B').")
        correct = (reponse == reponse_correcte)

        niveau_avant = etat.niveau
        if est_catch:
            est_renversement = False  # catch : HORS logique d'escalier (§C5)
        else:
            est_renversement = avance_escalier(etat, correct, delta_chi_presente, params)

        essais.append(EssaiJournal(
            numero_staircase=numero_staircase, indice_essai=indice, regime=regime_nom,
            seed_source=seed_src, L_source=L_src, budget=budget_essai, t=t_presente,
            delta_chi_mesure=delta_chi_presente, type_essai=essai_propose.type_essai,
            reponse=reponse, correct=correct, latence_s=latence,
            niveau_delta_chi=niveau_avant, est_renversement=est_renversement))
        indice += 1

    seuil, complet = calcule_seuil(etat.reversals, params.n_reversals_pour_seuil)
    return ResultatEscalier(numero_staircase=numero_staircase, regime=regime_nom, essais=essais,
                            reversals=list(etat.reversals), seuil=seuil, complet=complet,
                            seed_roving=seed_roving, seed_catch=seed_catch, budget=budget,
                            params=params)


# --- Logs bruts (JSONL) + replay déterministe -------------------------------


def ecrit_log_jsonl(path: str | Path, essais: list[EssaiJournal]) -> None:
    """Un enregistrement JSON par essai, une ligne par essai (JSONL) --
    format sérialisable, choix nommé (lisible, appendable, un parseur JSON
    standard par ligne ; alternative np.savez structuré écartée : les essais
    sont hétérogènes en type -- str/bool/float mêlés -- JSONL est plus direct)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for e in essais:
            f.write(json.dumps(e.to_dict(), ensure_ascii=False))
            f.write("\n")


def lit_log_jsonl(path: str | Path) -> list[EssaiJournal]:
    """Lit un log JSONL écrit par `ecrit_log_jsonl` -- inverse exact."""
    essais: list[EssaiJournal] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if ligne:
                essais.append(EssaiJournal.from_dict(json.loads(ligne)))
    return essais


def rejoue_escalier(essais_log: list[EssaiJournal], *, numero_staircase: int, regime_nom: str,
                    seed_roving: int, seed_catch: int, budget: int = BUDGET_DEFAUT,
                    params: ParametresEscalier = ParametresEscalier(),
                    banques: BanqueBancs | None = None) -> ResultatEscalier:
    """Rejoue une staircase depuis son LOG : reconstruit la MÊME séquence de
    stimuli (mêmes `seed_roving`/`seed_catch`/`budget`/`params`) et SUBSTITUE
    le sujet par un rejeu des réponses enregistrées, dans l'ORDRE. Vérifie à
    CHAQUE essai que le stimulus reconstruit (source, `t`, type) coïncide
    avec celui du log -- sinon les seeds/budget fournis sont incohérents avec
    ce log précis (erreur de protocole, `RuntimeError`, jamais une divergence
    silencieuse)."""
    it = iter(essais_log)

    def _sujet_rejeu(essai: EssaiPropose) -> Reponse:
        try:
            enregistre = next(it)
        except StopIteration:
            raise RuntimeError(
                "rejoue_escalier : le log fourni est plus court que la reconstruction -- "
                "seeds/budget/params incohérents avec ce log.") from None
        if (enregistre.seed_source != essai.seed_source or enregistre.L_source != essai.L_source
                or enregistre.t != essai.t or enregistre.type_essai != essai.type_essai
                or enregistre.indice_essai != essai.indice_essai):
            raise RuntimeError(
                f"rejoue_escalier : essai {essai.indice_essai} reconstruit "
                f"(seed={essai.seed_source}, L={essai.L_source}, t={essai.t}, "
                f"type={essai.type_essai}) diverge du log (seed={enregistre.seed_source}, "
                f"L={enregistre.L_source}, t={enregistre.t}, type={enregistre.type_essai}) -- "
                "seeds/budget/params fournis incohérents avec ce log, STOP.")
        return enregistre.reponse

    return run_escalier(numero_staircase=numero_staircase, regime_nom=regime_nom,
                        repondre=_sujet_rejeu, seed_roving=seed_roving, seed_catch=seed_catch,
                        budget=budget, params=params, banques=banques)
