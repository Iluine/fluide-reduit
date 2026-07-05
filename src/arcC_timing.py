"""Arc C / Task 3 — CORRECTIF avant session : accumulateur PUR des durées
d'exposition RÉALISÉES par essai (§C9 pièce 1 de PREREGISTRATION.md,
pocCascade2phys `7110e31`), reproduit verbatim dans
`.superpowers/sdd/arcC-task3fix-timing-conditions-brief.md`.

**Aucun sujet humain mesuré ici** : ce module est de la logique PURE, ZÉRO
dépendance matplotlib -- testé sur des durées CONSTRUITES à la main
(`tests/test_arcC_timing.py`). La glue matplotlib (mesure de
`time.perf_counter()` autour de chaque phase présentée) vit dans
`scripts/run_arcC_session.py` (`_construit_repondre_humain`, PARTAGÉE par
l'orchestrateur via `_fabrique_repondre_humain`,
`scripts/run_arcC_orchestration.py`) -- NON testée, comme le reste de la
coquille (matplotlib/clavier ne sont pas headless-testables).

Principe (§C9 pièce 1) : le régime laxiste empile des timers matplotlib
(`plt.pause`) NOMINAUX (`Regime.exposition_s`/`retention_s`) mais
approximatifs (dépendants de la machine/charge) -- un « 2 s » réalisé à
3,5 s FAUSSE le paramètre qui DÉFINIT le régime, sur des staircases déjà
consommées. Cet accumulateur consigne les durées RÉELLEMENT mesurées
(`t_fin - t_debut`) par (essai, phase), les compare aux nominales du
régime (`resume_timing`), et surface un AVERTISSEMENT proéminent si
l'écart dépasse 25 % (`SEUIL_AVERTISSEMENT_ECART`) -- **PAS un gate dur**
(Romain juge au pré-vol B-bis, §C9), mais l'écart doit CRIER, pas se
cacher.

Convention de nom de phase (glue, cf. `run_arcC_session.py`) : régime
sévère = une seule phase `"simultane"` (inspection libre, AUCUNE cible
temporelle -- `nominal=None`, rien à comparer) ; régime laxiste = six
phases par essai, `"exposition_{X,A,B}"` (comparées à `regime.
exposition_s`) et `"retention_{X,A,B}"` (comparées à `regime.
retention_s`) -- la qualification par stimulus (X/A/B) garde chaque
(essai, phase) UNIQUE dans le sidecar (§C9, test #2)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.arcC_abx import Regime

# >25 % d'écart réalisé/nominal (§C9 pièce 1) -- CRIER, pas un gate dur.
SEUIL_AVERTISSEMENT_ECART: float = 0.25

PREFIXE_EXPOSITION: str = "exposition"
PREFIXE_RETENTION: str = "retention"


@dataclass(frozen=True, eq=False)
class EnregistrementPhase:
    """Une phase mesurée d'UN essai -- `duree` = `t_fin - t_debut` (mesuré,
    jamais recalculé ailleurs), `nominal` = durée NOMINALE de RÉFÉRENCE pour
    cette phase (`regime.exposition_s`/`retention_s`), `None` si la phase
    n'a aucune cible temporelle (ex. `"simultane"`, sévère, inspection
    libre, §C0)."""
    indice_essai: int
    regime: str
    phase: str
    t_debut: float
    t_fin: float
    duree: float
    nominal: float | None

    def to_dict(self) -> dict:
        return dict(indice_essai=self.indice_essai, regime=self.regime, phase=self.phase,
                   t_debut=self.t_debut, t_fin=self.t_fin, duree=self.duree,
                   nominal=self.nominal)

    @staticmethod
    def from_dict(d: dict) -> "EnregistrementPhase":
        return EnregistrementPhase(**d)


@dataclass
class JournalTiming:
    """Accumulateur MUTABLE des `EnregistrementPhase` d'une staircase --
    `enregistre_phase` l'alimente EN PLACE, phase par phase (même
    convention que `EtatEscalier`/`avance_escalier`, `src/arcC_abx.py`)."""
    enregistrements: list[EnregistrementPhase] = field(default_factory=list)


def enregistre_phase(journal: JournalTiming, *, indice_essai: int, regime_nom: str,
                     nom_phase: str, t_debut: float, t_fin: float,
                     nominal: float | None = None) -> EnregistrementPhase:
    """Mesure `duree = t_fin - t_debut` (>= 0, sinon `ValueError` -- horloge
    incohérente) pour la phase `nom_phase` de l'essai `indice_essai`,
    l'ajoute à `journal.enregistrements` EN PLACE et renvoie l'enregistrement
    créé."""
    if t_fin < t_debut:
        raise ValueError(
            f"enregistre_phase : t_fin ({t_fin}) < t_debut ({t_debut}) -- horloge incohérente "
            f"(essai {indice_essai}, phase {nom_phase!r}).")
    enr = EnregistrementPhase(indice_essai=indice_essai, regime=regime_nom, phase=nom_phase,
                              t_debut=t_debut, t_fin=t_fin, duree=t_fin - t_debut,
                              nominal=nominal)
    journal.enregistrements.append(enr)
    return enr


def _categorie_phase(nom_phase: str) -> str:
    """Catégorise `nom_phase` (préfixe, cf. docstring module) en
    `"exposition"`/`"retention"`/`"autre"` -- `"autre"` regroupe les phases
    sans cible temporelle (ex. `"simultane"`, sévère)."""
    if nom_phase.startswith(PREFIXE_EXPOSITION):
        return PREFIXE_EXPOSITION
    if nom_phase.startswith(PREFIXE_RETENTION):
        return PREFIXE_RETENTION
    return "autre"


def resume_timing(journal: JournalTiming, regime: Regime,
                  seuil_avertissement: float = SEUIL_AVERTISSEMENT_ECART) -> dict:
    """Résumé réalisé-vs-nominal PAR CATÉGORIE de phase (`"exposition"`
    comparée à `regime.exposition_s`, `"retention"` à `regime.retention_s`,
    `"autre"` = informationnel seul, jamais comparé) -- moyenne réalisée,
    écart relatif SIGNÉ `(moyenne-nominal)/nominal`, et `avertissement` =
    `|écart relatif| > seuil_avertissement` (25 % par défaut, §C9 pièce 1 --
    « CRIER, pas se cacher », PAS un gate dur). Catégorie vide ou sans
    nominal -> aucun écart calculable, jamais un avertissement fabriqué sur
    du vide."""
    nominaux = {PREFIXE_EXPOSITION: regime.exposition_s, PREFIXE_RETENTION: regime.retention_s}
    groupes: dict[str, list[float]] = {PREFIXE_EXPOSITION: [], PREFIXE_RETENTION: [], "autre": []}
    for enr in journal.enregistrements:
        groupes[_categorie_phase(enr.phase)].append(enr.duree)

    categories: dict[str, dict] = {}
    en_alerte: list[str] = []
    for categorie, durees in groupes.items():
        nominal = nominaux.get(categorie)
        n = len(durees)
        moyenne = float(np.mean(durees)) if n > 0 else None
        if moyenne is not None and nominal is not None and nominal > 0.0:
            ecart_relatif = (moyenne - nominal) / nominal
            avertissement = abs(ecart_relatif) > seuil_avertissement
        else:
            ecart_relatif = None
            avertissement = False
        categories[categorie] = dict(n=n, moyenne_realisee_s=moyenne, nominal_s=nominal,
                                     ecart_relatif=ecart_relatif, avertissement=avertissement)
        if avertissement:
            en_alerte.append(categorie)

    return dict(regime=regime.nom, seuil_avertissement=seuil_avertissement,
               categories=categories, categories_en_alerte=en_alerte,
               avertissement_global=bool(en_alerte))


def formate_resume_timing(resume: dict) -> str:
    """Résumé IMPRIMABLE en fin de session (§C9 pièce 1) -- une ligne par
    catégorie comparée (nominal connu, au moins une mesure), + un
    AVERTISSEMENT proéminent si `avertissement_global` (jamais un gate dur
    -- Romain juge au pré-vol B-bis)."""
    lignes = [f"[timing §C9] régime={resume['regime']} -- durées réalisées vs nominales :"]
    for categorie, info in resume["categories"].items():
        if info["nominal_s"] is None or info["n"] == 0:
            continue
        marque = (f"  *** AVERTISSEMENT (>{resume['seuil_avertissement'] * 100:.0f}%) ***"
                 if info["avertissement"] else "")
        lignes.append(
            f"  {categorie} (n={info['n']}) : réalisé={info['moyenne_realisee_s']:.3f}s "
            f"nominal={info['nominal_s']:.3f}s écart={info['ecart_relatif'] * 100.0:+.1f}%{marque}")
    if resume["avertissement_global"]:
        lignes.append(
            "!!! AVERTISSEMENT §C9 pièce 1 : écart réalisé/nominal > "
            f"{resume['seuil_avertissement'] * 100:.0f}% sur {resume['categories_en_alerte']} -- "
            "le paramètre qui DÉFINIT le régime laxiste peut être faussé. Vérifier le "
            "mécanisme de timing (pré-vol B-bis) AVANT toute campagne.")
    return "\n".join(lignes)


def ecrit_timing_jsonl(path: str | Path, journal: JournalTiming) -> None:
    """Un enregistrement JSON par ligne (JSONL), même convention que
    `ecrit_log_jsonl` (`src/arcC_abx.py`)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for enr in journal.enregistrements:
            f.write(json.dumps(enr.to_dict(), ensure_ascii=False))
            f.write("\n")


def lit_timing_jsonl(path: str | Path) -> JournalTiming:
    """Lit un sidecar timing écrit par `ecrit_timing_jsonl` -- inverse
    exact."""
    journal = JournalTiming()
    with Path(path).open("r", encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if ligne:
                journal.enregistrements.append(EnregistrementPhase.from_dict(json.loads(ligne)))
    return journal
