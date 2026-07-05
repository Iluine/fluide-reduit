"""Arc C / C-1a Task 3 — ratio r0 = 2*RMS/(peak-to-peak) SCALE-INVARIANT
(bande "4-7"), mesure COMPAGNE du ratio natif χ_rms/C_michelson, cf. brief
`.superpowers/sdd/arcC-c1a-task3-r0-brief.md`.

Mesure d'analyse PURE ADDITIVE : aucune simulation, aucun RNG, aucune
nouvelle donnée humaine. Décision Romain GRAVÉE (PREREGISTRATION.md
pocCascade2phys §C12, commit `122e82a`) : le contraste de Michelson de la
porteuse est > 1 sur 19/19 sources incluses (nos stimuli sont des textures
fort-contraste, pas des gratings faible-contraste) -- le ratio natif
(`outputs/arcC/ratio_rms_michelson.json`) vit donc dans le régime de
saturation-pédestal et n'est PAS le facteur légitime pour convertir des
seuils CSF (petits contrastes). `r0` est la limite petit-contraste,
mean-INDÉPENDANTE, du même signal de bande -- le seul facteur légitime pour
C-1. §C12 exige : garder le natif INTACT (byte-identique, jamais réécrit) et
ajouter r0 en PASSE COMPAGNE, dans un fichier SÉPARÉ.

Réutilise TEL QUEL (jamais réimplémenté), par IMPORT depuis
`scripts.run_arcC_ratio_rms_michelson` :
  - `charge_sources_incluses` / `charge_champ_s_true` / `HISTORIES_DIR` --
    mêmes 19 sources incluses, mêmes champs `h_s{seed}.npz` que le natif ;
  - `_composante_bande` -- LA MÊME composante spatiale de la bande "4-7"
    (moyenne nulle) que le natif : r0 est calculé dessus, pas sur une
    reconstruction parallèle ;
  - `mesure_ratio_champ` -- réutilisé UNIQUEMENT pour son diagnostic
    `est_porteuse` (bande "4-7" porte >= 10 % de la puissance AC totale) ;
  - `ANCRE_BUDGET` (= 32), `BANDE` (= "4-7").

Définition de r0 (le seul calcul propre à cette task) : pour un champ albedo
`A`, `band_ac = _composante_bande(A)` (moyenne nulle, réutilisée du natif),
`rms = std(band_ac)`, `pic_a_pic = band_ac.max() - band_ac.min()`,
`r0 = 2*rms/pic_a_pic`. r0 ne dépend NI de ⟨A⟩ NI du pédestal Michelson ->
scale-invariant et pédestal-invariant PAR CONSTRUCTION. Sanity gravé : sur un
grating sinusoïdal pur (`a*cos`), `rms = a/√2`, `pic_a_pic = 2a` ->
`r0 = 1/√2 ≈ 0.7071` -- même sanity numérique que le natif.

Pour chaque source incluse, r0 est mesuré sur les DEUX champs albedo (VRAI et
régénéré au budget d'ancre 32, `src.arcC_stimuli.regenere_budget`), chaque
famille agrégée avec la DISPERSION complète (médiane, IQR, min/max, liste
par-source) -- garde d'honnêteté explicitement imposée par §C12.

Sortie : `outputs/arcC/ratio_r0_scale_invariant.json` (gitignoré, structure
gravée par le brief), COMPAGNON de `outputs/arcC/ratio_rms_michelson.json`
(natif, jamais touché par ce module)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.run_arcA_revalidate import S_HALF_OP
from scripts.run_arcC_ratio_rms_michelson import (ANCRE_BUDGET, BANDE, HISTORIES_DIR,
                                                  _composante_bande, charge_champ_s_true,
                                                  charge_sources_incluses, mesure_ratio_champ)
from src.albedo import albedo
from src.arcC_stimuli import regenere_budget

OUT_DIR = ROOT / "outputs" / "arcC"
# Ratio natif COMPAGNON -- jamais réécrit par ce module (garde §C12).
RATIO_NATIF_PATH = "outputs/arcC/ratio_rms_michelson.json"
# Sanity gravé par la mission (même valeur numérique que le natif, cf. son
# module) : ratio analytique r0 d'un grating sinusoïdal pur dans la bande
# 4-7 = 1/√2. `math.sqrt(0.5)` (et non `1.0 / np.sqrt(2.0)`) pour matcher
# bit-à-bit la constante gravée par le brief (dernier bit IEEE-754 différent
# selon l'ordre des opérations -- déjà vérifié côté natif, Task 1).
SANITY_SINUS_PUR_R0_ATTENDU: float = math.sqrt(0.5)


# --- Cœur du calcul, par champ albedo (fonction PURE) ------------------------


def mesure_r0_champ(A: np.ndarray) -> dict:
    """Cœur du calcul r0 (fonction PURE) sur un champ albedo `A` (H, W) --
    cf. brief `.superpowers/sdd/arcC-c1a-task3-r0-brief.md` §"Définition de
    r0" :
      - `band_ac = _composante_bande(A)` -- LA MÊME composante spatiale de
        la bande "4-7" que le natif, RÉUTILISÉE (jamais reconstruite) ;
      - `rms_bande = std(band_ac)` -- moyenne nulle par construction, donc
        RMS = écart-type (ddof=0 par défaut de `np.std`) ;
      - `pic_a_pic = band_ac.max() - band_ac.min()` ;
      - `r0 = 2*rms_bande/pic_a_pic` -- mean-INDÉPENDANT (ni ⟨A⟩ ni le
        pédestal Michelson n'entrent) : scale-invariant et
        pédestal-invariant PAR CONSTRUCTION ;
      - `est_porteuse` : diagnostic REPRIS du natif (`mesure_ratio_champ`),
        pas recalculé -- même seuil (bande "4-7" porte >= 10 % de la
        puissance AC totale).

    Retourne {r0, rms_bande, pic_a_pic, est_porteuse}."""
    A = np.asarray(A, dtype=np.float64)

    band_ac = _composante_bande(A, BANDE)
    rms_bande = float(np.std(band_ac))
    pic_a_pic = float(band_ac.max() - band_ac.min())
    r0 = 2.0 * rms_bande / pic_a_pic

    est_porteuse = mesure_ratio_champ(A)["est_porteuse"]

    return dict(r0=r0, rms_bande=rms_bande, pic_a_pic=pic_a_pic, est_porteuse=est_porteuse)


# --- Agrégation par famille (vrai / régénéré) --------------------------------


def _construit_entree_source(seed: int, L: int, A: np.ndarray) -> dict:
    """Entrée `par_source` (JSON) : identifiants `(seed, L)` + `r0` +
    `est_porteuse` -- forme gravée par le brief (pas `rms_bande`/`pic_a_pic`,
    diagnostics internes non exposés dans la sortie)."""
    m = mesure_r0_champ(A)
    return dict(seed=seed, L=L, r0=m["r0"], est_porteuse=m["est_porteuse"])


def agrege_famille_r0(entrees: list[dict]) -> dict:
    """Agrège une famille (vrai / régénéré) de mesures par-source :
    `r0_median` (np.median), `r0_q1`/`r0_q3` (np.percentile 25/75), `r0_iqr`,
    `r0_min`/`r0_max`, `n`, et la liste COMPLÈTE `par_source` -- garde
    d'honnêteté imposée par §C12 : la DISPERSION inter-sources est
    rapportée, pas seulement la valeur centrale."""
    r0s = np.array([e["r0"] for e in entrees], dtype=np.float64)
    q1, q3 = np.percentile(r0s, [25.0, 75.0])
    return dict(
        r0_median=float(np.median(r0s)), r0_q1=float(q1), r0_q3=float(q3),
        r0_iqr=float(q3 - q1), r0_min=float(r0s.min()), r0_max=float(r0s.max()),
        n=len(entrees), par_source=entrees)


def construit_mesures_r0(sources: list[tuple[int, int]], *,
                         histories_dir: str | Path = HISTORIES_DIR,
                         s_half: float = S_HALF_OP, ancre_budget: int = ANCRE_BUDGET) -> dict:
    """Construit le document complet (§"Agrégation & sortie" du brief) :
    pour chaque source incluse `(seed, L)`, mesure r0 sur le champ VRAI
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
        description=("Ratio r0 = 2*RMS/(peak-to-peak) de la bande 4-7 (scale-invariant, "
                     "regime petit-contraste) — COMPAGNON de ratio_rms_michelson.json (natif "
                     "intact). Arc C / C-1a Task 3, decision PREREGISTRATION.md §C12."),
        definition_r0=("r0 = 2*RMS(composante_bande) / (max-min de la composante_bande) ; "
                       "mean-independant, scale-invariant ; limite petit-contraste du ratio "
                       "natif."),
        bande=BANDE, s_half_op=s_half, ancre_budget=ancre_budget,
        compagnon_de=RATIO_NATIF_PATH,
        n_sources_incluses=len(sources),
        sanity_sinus_pur_r0_attendu=SANITY_SINUS_PUR_R0_ATTENDU)

    return dict(meta=meta, vrai=agrege_famille_r0(entrees_vrai),
               regenere_budget_32=agrege_famille_r0(entrees_regen))


# --- I/O + CLI ----------------------------------------------------------------


def ecrit_resultat_json(path: str | Path, resultat: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(resultat, indent=2, ensure_ascii=False, allow_nan=False),
                encoding="utf-8")


def main() -> None:
    sources = charge_sources_incluses()
    resultat = construit_mesures_r0(sources)

    out_path = OUT_DIR / "ratio_r0_scale_invariant.json"
    ecrit_resultat_json(out_path, resultat)

    print("=" * 78)
    print("ARC C / C-1a TASK 3 -- RATIO r0 SCALE-INVARIANT (bande 4-7, §C12)")
    print("=" * 78)
    print(f"  sources incluses : {resultat['meta']['n_sources_incluses']}  "
          f"(s_half_op={resultat['meta']['s_half_op']!r})")
    for label, cle in (("vrai", "vrai"), ("régénéré (budget 32)", "regenere_budget_32")):
        agg = resultat[cle]
        print(f"  {label:<24s} : n={agg['n']:<3d} médiane={agg['r0_median']:.4f}  "
              f"IQR=[{agg['r0_q1']:.4f}, {agg['r0_q3']:.4f}] (iqr={agg['r0_iqr']:.4f})  "
              f"min={agg['r0_min']:.4f}  max={agg['r0_max']:.4f}")
    print(f"[REPORT] -> {out_path}  (compagnon de {RATIO_NATIF_PATH}, natif intact)")


if __name__ == "__main__":
    main()
