"""Arc C / Task C-1a Task 1 — mesure du ratio χ_rms / C_michelson sur nos
stimuli RÉELS, bande "4-7" cycles/domaine (pic-CSF, §C7).

Mesure d'analyse PURE : aucune simulation, aucun RNG, aucune nouvelle donnée
humaine. C-1 (encadrement bibliographique, fait hors-code) doit convertir les
seuils CSF de la littérature -- exprimés en **contraste de Michelson** de
gratings sinusoïdaux -- vers l'échelle **Δχ = RMS/⟨A⟩** de notre pin
perceptuel. Le facteur de conversion load-bearing est ce ratio, MESURÉ ici sur
les 19 sources incluses de la campagne manche 1 (jamais supposé) -- cf. brief
`.superpowers/sdd/arcC-c1a-task1-ratio-brief.md`.

Réutilise TEL QUEL (jamais réimplémenté) :
  - `src.albedo.{albedo, chi_bands, _radial_bins, _BAND_BOUNDS, _BAND_LABELS}`
    (binning radial R1 -- LE juge de fidélité spectrale du dépôt) ;
  - `scripts.run_arcA_revalidate.S_HALF_OP` (point d'opération gravé manche 1) ;
  - `src.arcC_stimuli.regenere_budget` (champ régénéré, budget d'ancre 32,
    quadtree famille 2).

Pour chaque source incluse, le ratio est mesuré sur DEUX champs albedo : le
champ VRAI et sa version régénérée (budget 32, diagnostic de robustesse du
ratio à la reconstruction). Chaque famille est agrégée avec la DISPERSION
complète (médiane, IQR, min/max, liste par-source) -- garde d'honnêteté
imposée par la mission : si le ratio varie fortement d'une source à l'autre,
C-1 en hérite comme incertitude, jamais une valeur ponctuelle fabriquée.

Sanity gravé par la mission (testé, `tests/test_arcC_ratio_rms_michelson.py`) :
sur un grating sinusoïdal pur dans la bande 4-7, ratio analytique = 1/√2
(< 1 % d'erreur numérique attendue).

Sortie : `outputs/arcC/ratio_rms_michelson.json` (structure gravée par le
brief)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.run_arcA_revalidate import S_HALF_OP
from src.albedo import _BAND_BOUNDS, _BAND_LABELS, _radial_bins, albedo, chi_bands
from src.arcC_stimuli import regenere_budget

OUT_DIR = ROOT / "outputs" / "arcC"
MANIFESTE_PATH = OUT_DIR / "manifeste_campagne.json"
HISTORIES_DIR = ROOT / "outputs" / "arcA" / "histories"

# Bande porteuse d'intérêt (pic-CSF, §C7) -- toute la mesure porte sur CETTE
# bande, jamais réimplémentée (bornes lues dans `src.albedo._BAND_BOUNDS`).
BANDE: str = "4-7"
# Budget d'ancre du champ régénéré (quadtree famille 2, manche 1).
ANCRE_BUDGET: int = 32
# Sanity gravé par la mission : ratio analytique d'un grating sinusoïdal pur
# dans la bande 4-7 = 1/√2 (cf. docstring module + test dédié).
SANITY_SINUS_PUR_RATIO_ATTENDU: float = math.sqrt(0.5)


# --- Sources incluses (manifeste de campagne) --------------------------------


def charge_sources_incluses(manifeste_path: str | Path = MANIFESTE_PATH
                            ) -> list[tuple[int, int]]:
    """Lit `manifeste_campagne.json` -> liste TRIÉE `(seed, L)` des sources
    INCLUSES (`exclu == False`) -- 19 attendues manche 1, la seule exclue est
    `(seed=103, L=10)` (aucune exclusion silencieuse : lue depuis le
    manifeste, pas codée en dur)."""
    manifeste = json.loads(Path(manifeste_path).read_text(encoding="utf-8"))
    entrees = manifeste["exclusions"]["entrees"]
    incluses = [(e["seed"], e["L"]) for e in entrees if not e["exclu"]]
    return sorted(incluses)


def charge_champ_s_true(seed: int, L: int, histories_dir: str | Path = HISTORIES_DIR
                        ) -> np.ndarray:
    """Charge le champ de sédiment settled VRAI `s_L{L}` depuis
    `h_s{seed}.npz` -- tableau (64, 64) float64, aucune transformation."""
    path = Path(histories_dir) / f"h_s{seed}.npz"
    with np.load(path) as npz:
        return np.asarray(npz[f"s_L{L}"], dtype=np.float64)


# --- Cœur du calcul, par champ albedo (fonction PURE) ------------------------


def _composante_bande(A: np.ndarray, bande: str = BANDE) -> np.ndarray:
    """Composante spatiale de la bande radiale `bande` -- réutilise le
    binning R1 (`_radial_bins`/`_BAND_BOUNDS`/`_BAND_LABELS`), AUCUNE
    réimplémentation. Moyenne spatiale NULLE par construction (le bin DC
    k_r=0 est hors de toute bande AC nommée, dont "4-7")."""
    A = np.asarray(A, dtype=np.float64)
    F = np.fft.fft2(A)
    k_int = _radial_bins(A.shape)
    lo, hi = _BAND_BOUNDS[_BAND_LABELS.index(bande)]
    mask = (k_int >= lo) & (k_int <= hi)
    return np.real(np.fft.ifft2(F * mask))


def mesure_ratio_champ(A: np.ndarray) -> dict:
    """Cœur du calcul (fonction PURE) sur un champ albedo `A` (H, W) -- cf.
    brief `.superpowers/sdd/arcC-c1a-task1-ratio-brief.md` §"Cœur du calcul" :
      - contraste de Michelson de la bande "4-7" (le pédestal `mean_A` EST
        réinjecté avant min/max -- c'est ce qui donne la valeur analytique du
        sanity 1/√2) ;
      - χ_rms = définition R1 EXACTE (`chi_bands(A)[0]["4-7"]`), jamais
        recalculée à la main ;
      - ratio = χ_rms / C_michelson ;
      - diagnostics : `est_porteuse` (bande "4-7" porte >= 10 % de la
        puissance AC totale, définition de `src.albedo.delta_chi`),
        `composante_deborde_zero` (drapeau d'honnêteté : la reconstruction
        partielle d'une seule bande peut sous-tirer < 0 sur un champ réel à
        fort relief -- on ne clippe RIEN, on le SIGNALE),
        `bande_porteuse_max` (bande AC la plus énergétique, au cas où "4-7"
        ne domine pas partout).

    Retourne {chi_rms, c_michelson, ratio, mean_albedo, est_porteuse,
    composante_deborde_zero, bande_porteuse_max}."""
    A = np.asarray(A, dtype=np.float64)
    mean_A = float(np.mean(A))

    band_ac = _composante_bande(A, BANDE)
    A_max = mean_A + float(band_ac.max())
    A_min = mean_A + float(band_ac.min())
    c_michelson = (A_max - A_min) / (A_max + A_min)

    chi, power = chi_bands(A)
    chi_rms = chi[BANDE]
    ratio = chi_rms / c_michelson

    total_ac = sum(power.values())
    est_porteuse = bool(total_ac > 0 and power[BANDE] / total_ac >= 0.10)
    bande_porteuse_max = max(power, key=power.get)

    return dict(
        chi_rms=chi_rms, c_michelson=c_michelson, ratio=ratio, mean_albedo=mean_A,
        est_porteuse=est_porteuse, composante_deborde_zero=bool(A_min < 0.0),
        bande_porteuse_max=bande_porteuse_max)


# --- Agrégation par famille (vrai / régénéré) --------------------------------


def _construit_entree_source(seed: int, L: int, A: np.ndarray) -> dict:
    """Entrée `par_source` (JSON) : identifiants `(seed, L)` + mesure
    (`mesure_ratio_champ`)."""
    return dict(seed=seed, L=L, **mesure_ratio_champ(A))


def agrege_famille(entrees: list[dict]) -> dict:
    """Agrège une famille (vrai / régénéré) de mesures par-source :
    `ratio_median` (np.median), `ratio_q1`/`ratio_q3` (np.percentile 25/75),
    `ratio_iqr`, `ratio_min`/`ratio_max`, `n`, et la liste COMPLÈTE
    `par_source` -- garde d'honnêteté imposée par la mission : la DISPERSION
    est rapportée, pas seulement la valeur centrale."""
    ratios = np.array([e["ratio"] for e in entrees], dtype=np.float64)
    q1, q3 = np.percentile(ratios, [25.0, 75.0])
    return dict(
        ratio_median=float(np.median(ratios)), ratio_q1=float(q1), ratio_q3=float(q3),
        ratio_iqr=float(q3 - q1), ratio_min=float(ratios.min()), ratio_max=float(ratios.max()),
        n=len(entrees), par_source=entrees)


def construit_mesures(sources: list[tuple[int, int]], *,
                      histories_dir: str | Path = HISTORIES_DIR,
                      s_half: float = S_HALF_OP, ancre_budget: int = ANCRE_BUDGET) -> dict:
    """Construit le document complet (§"Agrégation & sortie" du brief) : pour
    chaque source incluse `(seed, L)`, mesure le ratio sur le champ VRAI
    (`albedo(s_true, s_half)`) et sur le champ régénéré au budget d'ancre
    (`albedo(regenere_budget(s_true, ancre_budget), s_half)`), agrège chaque
    famille séparément. Fonction PURE (aucune RNG) -- deux appels sur les
    mêmes sources produisent des dicts numériquement identiques."""
    sources = sorted(sources)
    entrees_vrai: list[dict] = []
    entrees_regen: list[dict] = []
    for seed, L in sources:
        s_true = charge_champ_s_true(seed, L, histories_dir)
        A_vrai = albedo(s_true, s_half)
        A_regen = albedo(regenere_budget(s_true, ancre_budget), s_half)
        entrees_vrai.append(_construit_entree_source(seed, L, A_vrai))
        entrees_regen.append(_construit_entree_source(seed, L, A_regen))

    meta = dict(
        description=("Ratio chi_rms/C_michelson sur la bande 4-7, stimuli reels manche 1 "
                     "(Arc C / C-1a Task 1)."),
        bande=BANDE, s_half_op=s_half, ancre_budget=ancre_budget,
        source_manifeste="outputs/arcC/manifeste_campagne.json",
        source_champs="outputs/arcA/histories/h_s{seed}.npz cle s_L{L}",
        n_sources_incluses=len(sources),
        sanity_sinus_pur_ratio_attendu=SANITY_SINUS_PUR_RATIO_ATTENDU)

    return dict(meta=meta, vrai=agrege_famille(entrees_vrai),
               regenere_budget_32=agrege_famille(entrees_regen))


# --- I/O + CLI ----------------------------------------------------------------


def ecrit_resultat_json(path: str | Path, resultat: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(resultat, indent=2, ensure_ascii=False, allow_nan=False),
                encoding="utf-8")


def main() -> None:
    sources = charge_sources_incluses()
    resultat = construit_mesures(sources)

    out_path = OUT_DIR / "ratio_rms_michelson.json"
    ecrit_resultat_json(out_path, resultat)

    print("=" * 78)
    print("ARC C / C-1a TASK 1 -- RATIO CHI_RMS / C_MICHELSON (bande 4-7)")
    print("=" * 78)
    print(f"  sources incluses : {resultat['meta']['n_sources_incluses']}  "
          f"(s_half_op={resultat['meta']['s_half_op']!r})")
    for label, cle in (("vrai", "vrai"), ("régénéré (budget 32)", "regenere_budget_32")):
        agg = resultat[cle]
        print(f"  {label:<24s} : n={agg['n']:<3d} médiane={agg['ratio_median']:.4f}  "
              f"IQR=[{agg['ratio_q1']:.4f}, {agg['ratio_q3']:.4f}] (iqr={agg['ratio_iqr']:.4f})  "
              f"min={agg['ratio_min']:.4f}  max={agg['ratio_max']:.4f}")
    print(f"[REPORT] -> {out_path}")


if __name__ == "__main__":
    main()
