"""F1 tranche-1 — GÉNÉRATEUR du ledger M-d (E6 — NON-verdictal).

Décision gravée E6 (§A15-complément) : « Génération du ledger M-d :
Δt=1 épisode/commit, CPU natif de nuit, NON-verdictal (MORT-d ne
chronomètre que le LOAD, jamais la génération ; la taille du ledger —
3600 commits, ~11.5 Mo — est préservée, c'est elle que le parse paie). »

La chaîne est la chaîne RÉELLE fenêtrée : `run_episode` (cœur intouché,
consommé par import) + commits fenêtrés §A14 (`commettre_fenetre` /
`extraire_fenetre` — primitives de la sonde S1, RECONDUITES, jamais
réimplémentées ; plongement (v), budget k_fen = 64, cap aire-proportionnel
mesuré). À chaque t = 1..3600 : run_episode -> entrée (a) événement seedé
-> commit fenêtré -> ré-ancrage fenêtre -> entrée (c).

Choix d'implémentation NOMMÉS par ce build (à endosser AVANT le run) :
  (g-i)   quadrant FIGÉ (0, 0) : la garde anti-vacuité §A14 sert la mesure
          Δχ ; M-d ne lit AUCUN Δχ — le contenu du commit n'influence pas
          le coût de load, seul le format/volume compte ;
  (g-ii)  DEUX ledgers écrits en un seul passage (une seule nuit de
          physique) : `ledger_3600/` (sans snapshot) et
          `ledger_3600_snapshot/` (identique + UNE entrée (d)
          snapshot-racine à t = 1800, champ complet post-ré-ancrage —
          le bras ferm mesuré §A13-4-1 en est l'appui) — c'est la paire
          qu'exige le diagnostic E7-2 (attendu : écart de load ≈ 0) ;
  (g-iii) REPRISE : point de reprise `md_generation_reprise.npz` (état
          moteur + t) tous les 50 commits — le run de nuit est
          interruptible ; à la reprise, les têtes des deux ledgers doivent
          concorder avec le point de reprise, sinon RuntimeError
          (supprimer outputs/f1/ledger_3600* et relancer proprement —
          jamais de rafistolage silencieux d'un ledger) ;
  (g-iv)  graine = 101 (la cellule-instrument reconduite partout).

Coût attendu (chiffrage §4, non-verdictal) : ~2-5 h CPU natif. La durée de
génération est reportée à titre de DIAGNOSTIC (ce n'est pas une donnée de
mesure ; MORT-d ne la regarde jamais). Usage (terminal natif, nuit) :
  .venv/bin/python scripts/run_f1_md_generer.py"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from scripts.run_arcA_m2_registre import centres_episodes  # noqa: E402
from scripts.run_arcA_revalidate import B0, PARAMS  # noqa: E402
from scripts.run_arcA_sonde_fenetre import (  # noqa: E402
    COTE_FENETRE,
    K_FEN,
    commettre_fenetre,
    extraire_fenetre,
    fenetre_slices,
)
from src.f1_gpu.ledger import EcrivainLedger  # noqa: E402
from src.sediment import run_episode  # noqa: E402

OUT_DIR = ROOT / "outputs" / "f1"
LEDGER_SANS_PATH = OUT_DIR / "ledger_3600"
LEDGER_AVEC_PATH = OUT_DIR / "ledger_3600_snapshot"
REPRISE_PATH = OUT_DIR / "md_generation_reprise.npz"
OUT_JSON_PATH = OUT_DIR / "md_generation.json"

GRAINE: int = 101                 # (g-iv) cellule-instrument reconduite
N_COMMITS: int = 3600             # gravé (M-d : 1 h simulée à 1 commit/s)
T_SNAPSHOT: int = 1800            # (g-ii) snapshot-racine au milieu
QUADRANT: tuple[int, int] = (0, 0)  # (g-i) figé
PAS_REPRISE: int = 50             # (g-iii)
NIVEAU_FENETRE: int = 1           # profondeur dyadique de la fenêtre 32² / 64²


def _ouvrir_ledgers() -> tuple[EcrivainLedger, EcrivainLedger, np.ndarray, int]:
    """Ouvre (ou crée) les deux ledgers + l'état de reprise (g-iii)."""
    neuf = not (LEDGER_SANS_PATH / "index.jsonl").exists()
    if neuf:
        ecrivain_sans = EcrivainLedger(LEDGER_SANS_PATH, creer=True)
        ecrivain_avec = EcrivainLedger(LEDGER_AVEC_PATH, creer=True)
        return ecrivain_sans, ecrivain_avec, np.zeros_like(B0), 0
    ecrivain_sans = EcrivainLedger(LEDGER_SANS_PATH, creer=False)
    ecrivain_avec = EcrivainLedger(LEDGER_AVEC_PATH, creer=False)
    if not REPRISE_PATH.exists():
        raise RuntimeError(
            "run_f1_md_generer : ledgers présents sans point de reprise — "
            "supprimer outputs/f1/ledger_3600* et relancer (g-iii).")
    with np.load(REPRISE_PATH, allow_pickle=False) as reprise:
        s_moteur = np.array(reprise["s_moteur"], dtype=np.float64, copy=True)
        t_fait = int(reprise["t_fait"])
    if (ecrivain_sans.derniere_cle[0] != t_fait
            or ecrivain_avec.derniere_cle[0] != t_fait):
        raise RuntimeError(
            f"run_f1_md_generer : têtes de ledger ({ecrivain_sans.derniere_cle},"
            f" {ecrivain_avec.derniere_cle}) désaccordées du point de reprise "
            f"t={t_fait} — supprimer outputs/f1/ledger_3600* et relancer.")
    return ecrivain_sans, ecrivain_avec, s_moteur, t_fait


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ecrivain_sans, ecrivain_avec, s_moteur, t_fait = _ouvrir_ledgers()
    if t_fait >= N_COMMITS:
        print(f"ledger complet ({t_fait}/{N_COMMITS}) — rien à faire.")
        return

    sy, sx = fenetre_slices(QUADRANT)
    rect = (QUADRANT[0] * COTE_FENETRE, QUADRANT[1] * COTE_FENETRE,
            COTE_FENETRE)
    centres = centres_episodes(GRAINE, N_COMMITS)

    t0 = time.perf_counter()
    for t in range(t_fait + 1, N_COMMITS + 1):
        # Δt = 1 épisode/commit (E6) : un épisode réel, puis commit fenêtré.
        s_moteur = run_episode(s_moteur, B0, centres[t - 1], PARAMS)
        commit = commettre_fenetre(s_moteur[sy, sx], QUADRANT, K_FEN)
        s_moteur[sy, sx] = extraire_fenetre(commit, QUADRANT)
        for ecrivain in (ecrivain_sans, ecrivain_avec):
            ecrivain.ajouter_evenement(t, GRAINE, "pulse_sediment")
            ecrivain.ajouter_commit(t, commit, NIVEAU_FENETRE, rect, K_FEN)
        if t == T_SNAPSHOT:
            # (g-ii) snapshot-racine : champ complet POST-ré-ancrage,
            # sans perte (motif du bras ferm) — variante AVEC seulement.
            ecrivain_avec.ajouter_snapshot(t, s_moteur)
        if t % PAS_REPRISE == 0 or t == N_COMMITS:
            np.savez(REPRISE_PATH, s_moteur=s_moteur, t_fait=t)
            ecoule = time.perf_counter() - t0
            fait = t - t_fait
            print(f"  commit {t}/{N_COMMITS}  "
                  f"({ecoule / fait:.2f} s/commit, "
                  f"reste ~{(N_COMMITS - t) * ecoule / fait / 3600:.2f} h)",
                  flush=True)

    duree_s = time.perf_counter() - t0
    octets_sans = sum(
        p.stat().st_size for p in LEDGER_SANS_PATH.iterdir())
    octets_avec = sum(
        p.stat().st_size for p in LEDGER_AVEC_PATH.iterdir())
    document = {
        "meta": {"generation": "E6 — NON-verdictale (MORT-d ne chronomètre "
                               "que le load)",
                 "graine": GRAINE, "n_commits": N_COMMITS,
                 "dt_episodes_par_commit": 1, "k_fen": K_FEN,
                 "quadrant": list(QUADRANT), "t_snapshot": T_SNAPSHOT},
        "octets_ledger_sans": octets_sans,
        "octets_ledger_avec": octets_avec,
        "duree_generation_s_diagnostic": duree_s,
    }
    OUT_JSON_PATH.write_text(
        json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=" * 78)
    print(f"GÉNÉRATION E6 TERMINÉE : {N_COMMITS} commits, "
          f"{octets_sans / 1e6:.1f} Mo (sans) / {octets_avec / 1e6:.1f} Mo "
          f"(avec snapshot), {duree_s / 3600:.2f} h (diagnostic).")
    print(f"[REPORT] -> {OUT_JSON_PATH}")
    print("Non-verdictal. Prochaine étape possible (décision Romain) : "
          "scripts/run_f1_md.py (le LOAD, la mesure).")


if __name__ == "__main__":
    main()
