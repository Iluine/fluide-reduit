"""Manche 2 (§A13) — VERDICT §A13-2 (combinateur mécanique).

Citation verbatim (`PREREGISTRATION.md` §A13-2, pocCascade2phys `811b11d`,
gravé AVANT toute mesure) :

  « Lecture au pin sévère IC ENTIER (6.03 / 7.33 / 8.67 %), par colonne :
  une colonne « ferme » ssi k*_chaîne(Δt) ≤ cap (409.6) pour TOUS les Δt
  testés ET la série Δχ_i ne croît pas à travers le JND (pas de composition).
  - REGISTRE-FERME : les trois colonnes ferment → commis PASS (portée §A13-5).
  - MUR-REGISTRE : les trois colonnes ouvertes (composition à travers le JND
    à tout budget ≤ cap) → commis FAIL de CETTE famille → conséquences §A13-0.
  - INDÉTERMINÉ-JND : colonnes divergentes → surface k*_chaîne(Δt, JND) portée.
  - INSTRUMENT-MUET : contrôles §A13-4 en violation → verdict scellé, remonter.
  Diagnostic non-verdictal : la même table à JND_lax (IC [9.37, 13.87]) — le
  delta sev↔lax au niveau chaîne chiffre le stockage-deux-étages (§C0) sur
  l'axe temporel-registre. »

  §A13-1 : « k*_chaîne(Δt) = plus petit k tel que max_i Δχ_i < JND pour la
  médiane des seeds, aucune seed > 2×JND (clause §A2 reconduite). »

DÉFINITIONS MÉCANIQUES PRÉ-ENREGISTRÉES (gravées ICI, avant toute lecture de
la grille — à endosser en revue AVANT exécution ; combinateur pur, aucune
branche « à l'œil ») :

  (a) max_i porte sur i ≥ 2 (Δχ_1 ≡ 0 STRUCTUREL, cf. cœur Task 0 et sonde —
      l'inclure rendrait toute chaîne « partiellement fermée » par
      construction ; convention identique à la lecture de la sonde).
  (b) k*_chaîne(Δt, JND) = plus petit k ∈ {128, 256, 400} tel que
      médiane_seeds(max_{i≥2} Δχ_i) < JND ET aucune seed avec
      max_{i≥2} Δχ_i > 2·JND ; sinon ∞ (hors grille — la grille est bornée
      par le cap : 400 est le dernier budget sous cap 409.6).
  (c) « composition à travers le JND » à (Δt, k, JND) ssi la série MÉDIANE
      (médiane des seeds, émission par émission) traverse : ∃ i ≥ 2,
      mediane_i ≥ JND. (« Croissante à travers le JND » = traverser ; une
      série bornée sous JND ne compose pas, quelle que soit sa forme —
      cohérent avec §A13-1 « bornée/contractante = ferme ; croissante à
      travers le JND = mur ».)
  (d) Colonne (JND) FERMÉE ssi ∀ Δt : k*_chaîne(Δt, JND) fini (≤ cap).
      Colonne OUVERTE-MUR ssi ∀ Δt, ∀ k ≤ cap : composition (c) présente.
      Une colonne peut n'être NI fermée NI ouverte-MUR (ex. bloquée par la
      clause 2·JND d'une seed sans traversée de la médiane) → elle compte
      comme divergence → INDÉTERMINÉ-JND (sortie sûre, jamais un PASS/FAIL
      par défaut — leçon du combinateur §A12 : « else = sortie sûre »).
  (e) INSTRUMENT-MUET est lu depuis `m2_measures.npz` (posé par l'assemble
      depuis les attendus §A13-4) et COURT-CIRCUITE tout : verdict scellé.
  (f) Le diagnostic JND_lax applique (a)-(d) à l'IC laxiste ENTIER —
      non-verdictal, consigné pour le stockage-deux-étages.

Usage :
  .venv/bin/python scripts/run_arcA_m2_verdict.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.run_arcA_m2_grille import (
    BUDGETS,
    DTS,
    MEASURES_PATH,
    N_EMISSIONS,
    SEEDS,
    charge_pins,
)

OUT_JSON_PATH = ROOT / "outputs" / "arcA" / "m2_verdict.json"
OUT_PNG_PATH = ROOT / "outputs" / "arcA" / "arcA_m2_kstar_chaine.png"

# Cap gravé (§A13-2 : « ≤ cap (409.6) ») — 10 % de 4096, clause anti-trivialité
# §A3 reconduite en famille 2 (comptabilité clause-1).
CAP_FLOATS: float = 409.6

# ∞ sérialisable (JSON n'a pas d'Infinity portable) — convention 16L₀.
K_INF = "inf"


def kstar_chaine(delta_chi: np.ndarray, jnd: float) -> tuple[int | str, dict]:
    """Définitions (a)+(b) : `delta_chi` de forme (n_seeds, n_budgets,
    n_emissions) pour UN Δt. Retourne (k* ou "inf", détail par budget)."""
    detail = {}
    kstar: int | str = K_INF
    for ki, k in enumerate(BUDGETS):
        maxes = delta_chi[:, ki, 1:].max(axis=1)  # (a) : i >= 2.
        med = float(np.median(maxes))
        pire_seed = float(maxes.max())
        ok = bool(med < jnd and pire_seed <= 2.0 * jnd)
        detail[str(k)] = {"mediane_max": med, "pire_seed_max": pire_seed,
                          "ferme": ok}
        if ok and kstar == K_INF:
            kstar = k
    return kstar, detail


def composition_partout(delta_chi: np.ndarray, jnd: float) -> bool:
    """Définition (c) appliquée à TOUS les budgets ≤ cap d'UN Δt : la série
    MÉDIANE traverse le JND pour chaque k. `delta_chi` (n_seeds, n_budgets,
    n_emissions)."""
    for ki in range(len(BUDGETS)):
        mediane_serie = np.median(delta_chi[:, ki, :], axis=0)
        if not bool((mediane_serie[1:] >= jnd).any()):
            return False
    return True


def verdict_colonne(delta_chi: np.ndarray, jnd: float) -> dict:
    """Définition (d) pour UNE colonne (un JND) : `delta_chi`
    (n_seeds, n_dts, n_budgets, n_emissions)."""
    kstars: dict = {}
    details: dict = {}
    for di, dt in enumerate(DTS):
        ks, det = kstar_chaine(delta_chi[:, di, :, :], jnd)
        kstars[str(dt)] = ks
        details[str(dt)] = det
    fermee = all(ks != K_INF for ks in kstars.values())
    ouverte_mur = all(composition_partout(delta_chi[:, di, :, :], jnd)
                      for di in range(len(DTS)))
    return {"jnd": jnd, "kstar_chaine": kstars, "fermee": fermee,
            "ouverte_mur": ouverte_mur, "detail_budgets": details}


def combinateur(colonnes: list[dict], instrument_muet: bool) -> str:
    """Le combinateur §A13-2, pur (définitions (d)+(e)) — else = sortie sûre."""
    if instrument_muet:
        return "INSTRUMENT_MUET"
    if all(c["fermee"] for c in colonnes):
        return "REGISTRE_FERME"
    if all(c["ouverte_mur"] for c in colonnes):
        return "MUR_REGISTRE"
    return "INDETERMINE_JND"


def figure_kstar(document: dict, delta_chi: np.ndarray, path: Path) -> None:
    """3 panneaux (un par Δt) : série Δχ_i MÉDIANE par budget + lignes IC
    sévère ; titre = verdict + k*_chaîne par colonne."""
    pins_sev = document["pins"]["severe"]
    fig, axes = plt.subplots(1, len(DTS), figsize=(13.5, 4.4), sharey=True)
    couleurs = {128: "#009E73", 256: "#0072B2", 400: "#CC79A7"}
    xs = list(range(1, N_EMISSIONS + 1))
    for di, (dt, ax) in enumerate(zip(DTS, axes)):
        for ki, k in enumerate(BUDGETS):
            med = np.median(delta_chi[:, di, ki, :], axis=0)
            ax.plot(xs, med, "-o", color=couleurs[k], markersize=4,
                    linewidth=1.5, label=f"k={k} (médiane)")
        ax.axhline(pins_sev["jnd"], color="#D55E00", linewidth=1.3)
        for borne in pins_sev["ic"]:
            ax.axhline(borne, color="#D55E00", linestyle="--", linewidth=0.9)
        ax.set_title(f"Δt = {dt} — k* = "
                     + "/".join(str(document['colonnes_severe'][j]['kstar_chaine'][str(dt)])
                                for j in range(3)))
        ax.set_xlabel("émission i")
        ax.set_xticks(xs)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Δχ_i (médiane des seeds)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Manche 2 §A13 — VERDICT : {document['verdict']} "
                 f"(k* affichés : ic_bas/pin/ic_haut)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    if not MEASURES_PATH.exists():
        raise RuntimeError(f"verdict : {MEASURES_PATH} absent — assembler d'abord.")
    with np.load(MEASURES_PATH) as data:
        delta_chi = np.array(data["delta_chi"], dtype=np.float64)
        instrument_muet = bool(data["instrument_muet"])
        assert list(data["seeds"]) == list(SEEDS)
        assert list(data["dts"]) == list(DTS)
        assert list(data["budgets"]) == list(BUDGETS)
    if np.isnan(delta_chi).any():
        raise RuntimeError("verdict : NaN dans le tenseur de mesures — grille "
                           "incomplète, STOP (jamais de verdict partiel).")

    pins = charge_pins()
    ic_sev = [pins["severe"]["ic"][0], pins["severe"]["jnd"], pins["severe"]["ic"][1]]
    ic_lax = [pins["laxiste"]["ic"][0], pins["laxiste"]["jnd"], pins["laxiste"]["ic"][1]]

    colonnes_sev = [verdict_colonne(delta_chi, jnd) for jnd in ic_sev]
    verdict = combinateur(colonnes_sev, instrument_muet)
    colonnes_lax = [verdict_colonne(delta_chi, jnd) for jnd in ic_lax]

    document = {
        "verdict": verdict,
        "pins": pins,
        "colonnes_severe": colonnes_sev,
        "diagnostic_laxiste_non_verdictal": colonnes_lax,
        "cap_floats": CAP_FLOATS,
        "instrument_muet": instrument_muet,
        "provenance": {"measures": str(MEASURES_PATH.name),
                       "pins": "outputs/arcC/pins_spatial.json",
                       "contrat": "PREREGISTRATION.md §A13 (pocCascade2phys 811b11d)"},
    }
    OUT_JSON_PATH.write_text(json.dumps(document, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    figure_kstar(document, delta_chi, OUT_PNG_PATH)

    print("=" * 78)
    print(f"MANCHE 2 (§A13) — VERDICT §A13-2 : {verdict}")
    print("=" * 78)
    for nom, jnd, col in zip(("ic_bas", "pin", "ic_haut"), ic_sev, colonnes_sev):
        print(f"  {nom} ({jnd * 100:.2f} %) : k*_chaîne(Δt) = "
              f"{col['kstar_chaine']}  fermée={col['fermee']}  "
              f"ouverte_mur={col['ouverte_mur']}")
    print(f"[REPORT] -> {OUT_JSON_PATH}")
    print(f"[REPORT] -> {OUT_PNG_PATH}")
    print("RAPPEL : point d'arrêt OBLIGATOIRE — remonter le verdict, pas "
          "d'enchaînement (§A13-3).")


if __name__ == "__main__":
    main()
