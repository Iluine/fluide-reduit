"""Arc C / Task 3 — post-traitement d'une campagne → pins spatiaux (§C8 de
PREREGISTRATION.md, pocCascade2phys, commits `c494d50`→`d7c95e3`), reproduit
verbatim dans `.superpowers/sdd/arcC-task3-runner-brief.md`. Base :
`scripts/run_arcC_orchestration.py` (manifeste de campagne).

**Ce script ne mesure AUCUN humain** (données synthétiques en test) et
**n'imprime AUCUN verdict de manche** : il CALCULE le pin (`JND_sev^spat`,
`JND_lax^spat`) + son IC, par régime, depuis les logs bruts d'une campagne --
la LECTURE §C4 (côté 3/4 %, contingence géométrie-plafond) est au
contrôleur, sur données RÉELLES, Task 5.

Principe d'architecture : ORCHESTRE les primitives PURES déjà livrées et
testées de `src/arcC_abx.py` (`evalue_dispersion`, `statut_condition`,
`lit_log_jsonl`) -- rien n'est réimplémenté ici.

Méthode d'IC (NOMMÉE, brief l'autorise CV/étendue OU bootstrap si justifié) :
bornes MIN/MAX des seuils des staircases complètes de la condition
(`METHODE_IC = "min_max_seuils"`). Choix : à `n=3` (typique, §C5), un
bootstrap ré-échantillonnerait 3 points -- il ne peut produire qu'une poignée
de valeurs DISTINCTES (au plus les C(3+2,2)=10 multi-ensembles de taille 3
tirés parmi 3 valeurs), donnant une illusion de précision statistique que
l'échantillon ne supporte pas. Les bornes min/max, elles, exposent
directement la DISPERSION OBSERVÉE (déjà le même esprit que le contrôle CV
ddof=1 de `evalue_dispersion`, §C5 Task 2) -- honnête sur un petit
échantillon, sans invention de méthode plus sophistiquée que ce que les
données permettent."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.arcC_abx import (STATUT_CONTINUE, STATUT_RESOLU, evalue_dispersion, lit_log_jsonl,
                          statut_condition)
from scripts.run_arcC_orchestration import OUT_DIR, lit_manifeste_json

METHODE_IC: str = "min_max_seuils"


# --- Agrégation seuils -> pin + IC, par régime -------------------------------


def calcule_ic(seuils: list[float]) -> dict | None:
    """IC NOMMÉ (cf. docstring module) : bornes min/max des seuils fournis.
    `None` si `seuils` est vide (rien à borner)."""
    if not seuils:
        return None
    return dict(methode=METHODE_IC, borne_inf=float(min(seuils)), borne_sup=float(max(seuils)))


def agrege_regime(sessions: list[dict]) -> dict:
    """Agrège les staircases d'UN régime (`sessions`, format
    `orchestre_regime`) : seuils des staircases COMPLÈTES uniquement
    (`complet=True`, `seuil is not None`) -> `evalue_dispersion` (CV ddof=1
    <= 30%, >= 3 staircases) -> `statut_condition` (exige EN PLUS que TOUTES
    les sessions -- complètes ou non -- soient VALIDES §C5, sans
    recalibration). `jnd`/`ic` = `None` si le statut n'est pas `RESOLU`
    (jamais un pin fabriqué)."""
    seuils_complets = [s["seuil"] for s in sessions if s["complet"] and s["seuil"] is not None]
    validites = [bool(s["validite"]["valide"]) for s in sessions]
    dispersion = evalue_dispersion(seuils_complets)
    statut = statut_condition(validites, dispersion)
    ic = calcule_ic(seuils_complets) if statut == STATUT_RESOLU else None
    return dict(
        jnd=dispersion["pin"] if statut == STATUT_RESOLU else None, ic=ic, statut=statut,
        n_staircases=len(sessions), n_staircases_completes=len(seuils_complets),
        seuils=seuils_complets, dispersion_relative=dispersion["dispersion_relative"],
        validites_sessions=validites)


def construit_pins(manifeste: dict) -> dict:
    """Construit le document `pins_spatial.json` depuis un manifeste de
    campagne (`scripts.run_arcC_orchestration.orchestre_campagne`) : par
    régime, `agrege_regime` ; si la campagne s'est arrêtée en amont (D-2,
    >= 10 sources exclues), propage `statut_global` avec `regimes={}` (rien à
    agréger -- aucune staircase n'a tourné)."""
    if manifeste.get("statut_global") != STATUT_CONTINUE:
        return dict(
            parametres_graves={**manifeste["parametres_graves"], "methode_ic": METHODE_IC},
            exclusions=manifeste["exclusions"], statut_global=manifeste["statut_global"],
            regimes={})
    regimes = {regime_nom: agrege_regime(regime_data["sessions"])
              for regime_nom, regime_data in manifeste["regimes"].items()}
    return dict(
        parametres_graves={**manifeste["parametres_graves"], "methode_ic": METHODE_IC},
        exclusions=manifeste["exclusions"], statut_global=manifeste["statut_global"],
        regimes=regimes)


# --- pins_spatial.json --------------------------------------------------------


def ecrit_pins_json(path: str | Path, pins: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(pins, indent=2, ensure_ascii=False), encoding="utf-8")


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
    pins = construit_pins(manifeste)
    ecrit_pins_json(args.out, pins)

    print("=" * 78)
    print("ARC C / TASK 3 -- PINS SPATIAUX (post-traitement, AUCUN verdict de manche)")
    print("=" * 78)
    print(f"statut_global={pins['statut_global']}")
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
