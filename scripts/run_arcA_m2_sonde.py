"""Manche 2 (§A13) — Task 1 : SONDE (§A13-3).

Citation verbatim (`PREREGISTRATION.md` §A13-3, pocCascade2phys, gravé
commit `811b11d`, AVANT toute mesure) :

    « SONDE d'abord (Task 1, ~minutes) : 1 seed, Δt = 4, k = 256,
    6 émissions. Lectures pré-écrites : (S1) Δχ_i traverse JND_sev dès les
    premières émissions → la composition est massive, le verdict d'existence
    est quasi rendu, la grille complète se re-dimensionne en conséquence
    (remonter avant d'acheter) ; (S2) Δχ_i borné/contractant → la grille
    s'achète. La sonde ne PRONONCE rien : elle dimensionne. »

Ce script exécute UNE cellule de la sonde (seed=101, Δt=4, k=256,
6 émissions, bras "v2") via le cœur §A13 (`scripts/run_arcA_m2_registre.py`,
Task 0, importé — jamais réimplémenté), applique la lecture pré-écrite
mécanique S1/S2/AUTRE (booléenne, aucune prose libre), chronomètre
séparément la chaîne-vérité et la chaîne-moteur (alimente le gate d'achat
Task 2), et écrit `outputs/arcA/m2_sonde.json` + la figure
`outputs/arcA/arcA_m2_sonde.png`.

La sonde ne tranche AUCUN verdict de la manche (§A13-2, lu au pin sévère
IC ENTIER, réservé à une task ultérieure) — `lecture_sonde` ci-dessous est
une fonction PURE, mécanique, qui ne fait que catégoriser la forme de la
série Δχ_i observée (S1 / S2 / AUTRE) pour dimensionner la suite, comme le
dit le §A13-3 : « la sonde ne PRONONCE rien ».

Lecture (mécanique, booléenne) :
  - traverse_jnd_sev[i] = (Δχ_i >= jnd_sev), pour chaque émission i=1..6 ;
  - **S1** (composition massive) ssi traverse_jnd_sev est vrai pour AU MOINS
    UNE des émissions i ∈ {2, 3} (« dès les premières émissions » ;
    i=1 est structurellement 0 par construction du protocole — cf.
    `chaine_registre` — et donc exclu de la lecture) ;
  - **S2** (borné/contractant) ssi AUCUN Δχ_i (i >= 2) ne dépasse jnd_sev ET
    max(Δχ_4..Δχ_6) <= max(Δχ_2..Δχ_3) * 1.5 (pas de ressaut après le
    plateau initial) ;
  - sinon (ni S1 ni S2, typiquement une croissance sous JND qui viole le
    facteur 1.5) : **AUTRE** — la forme est remontée telle quelle, à décrire
    par les chiffres, pas absorbée dans une case pré-écrite qui ne lui
    correspond pas.
  - Les comparaisons aux deux bornes de l'IC sévère (ic_bas, ic_haut) sont
    consignées à titre DESCRIPTIF SEULEMENT dans le JSON de sortie (champ
    `detail_lecture`) — le verdict final §A13-2 lira l'IC entier, la sonde
    ne tranche pas dessus.

Usage :
  .venv/bin/python scripts/run_arcA_m2_sonde.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.run_arcA_m2_registre import chaine_registre, chaine_verite

OUT_DIR = ROOT / "outputs" / "arcA"
OUT_JSON_PATH = OUT_DIR / "m2_sonde.json"
OUT_PNG_PATH = OUT_DIR / "arcA_m2_sonde.png"
PINS_PATH = ROOT / "outputs" / "arcC" / "pins_spatial.json"

# Paramètres FIGÉS de la sonde (§A13-3) -- ne pas paramétrer au-delà.
SEED_SONDE: int = 101
DT_SONDE: int = 4
K_SONDE: int = 256
N_EMISSIONS_SONDE: int = 6

# Littéral de contrôle (point central admis du pin sévère, `outputs/arcC/
# pins_spatial.json` -> regimes.severe.jnd) -- sert UNIQUEMENT à l'assert de
# cohérence de `charge_pins_severe` (détecter un artefact déplacé) ; la
# VALEUR consommée par le calcul est TOUJOURS celle lue dans le fichier, pas
# ce littéral.
_JND_SEV_ATTENDU_LITTERAL: float = 0.07334065957385666
_TOLERANCE_LITTERAL: float = 1e-12


def charge_pins_severe(path: str | Path = PINS_PATH) -> dict:
    """Charge `jnd_sev` et son IC depuis `outputs/arcC/pins_spatial.json`
    (`regimes.severe.jnd`, `regimes.severe.ic`) -- LIT le fichier, ne code
    aucune valeur en dur pour le calcul. Assert de cohérence à 1e-12 près sur
    le point central admis contre le littéral gravé
    `_JND_SEV_ATTENDU_LITTERAL` (0.07334065957385666), pour détecter un
    artefact déplacé (fichier régénéré avec une autre campagne, etc.).

    Retourne `{"jnd": float, "ic": (ic_bas, ic_haut)}`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    severe = data["regimes"]["severe"]
    jnd = float(severe["jnd"])
    ic_bas, ic_haut = (float(x) for x in severe["ic"])

    ecart = abs(jnd - _JND_SEV_ATTENDU_LITTERAL)
    if ecart > _TOLERANCE_LITTERAL:
        raise AssertionError(
            f"charge_pins_severe : jnd_sev lu ({jnd!r}) diverge du littéral "
            f"gravé ({_JND_SEV_ATTENDU_LITTERAL!r}) de {ecart!r} (> "
            f"{_TOLERANCE_LITTERAL!r}) -- artefact déplacé ? Fichier : {path}")

    return {"jnd": jnd, "ic": (ic_bas, ic_haut)}


def lecture_sonde(delta_chi: Sequence[float], jnd_sev: float) -> dict:
    """Lecture pré-écrite mécanique (§A13-3), fonction PURE (aucune E/S,
    aucun accès disque) -- catégorise la série `delta_chi` (une valeur par
    émission, i=1..n, `delta_chi[i-1]` = Δχ_i) en "S1" / "S2" / "AUTRE".

    Convention d'indexation : `delta_chi[0]` = Δχ_1 (structurellement 0.0,
    cf. `chaine_registre`) -- EXCLU de la lecture (ni S1 ni S2 ne le
    consultent) ; `delta_chi[1]`, `delta_chi[2]` = Δχ_2, Δχ_3 ("les
    premières émissions" de S1) ; `delta_chi[3:6]` = Δχ_4..Δχ_6 (le reste,
    pour le facteur de contraction de S2). La valeur de Δχ_1 (0.0 ou toute
    autre) n'influence donc JAMAIS le résultat.

    - **S1** ssi `delta_chi[i] >= jnd_sev` pour au moins un i ∈ {1, 2}
      (0-indexé -- Δχ_2 ou Δχ_3) ;
    - **S2** ssi aucun `delta_chi[i] >= jnd_sev` pour i >= 1 (Δχ_2..Δχ_6) ET
      `max(delta_chi[3:6]) <= max(delta_chi[1:3]) * 1.5` ;
    - sinon **AUTRE**.

    Retourne `{"lecture": "S1"|"S2"|"AUTRE", "traverse_jnd_sev": [bool, ...],
    "s1": bool, "s2": bool, "max_emissions_2_3": float, "max_emissions_4_6":
    float}`."""
    delta_chi = list(delta_chi)
    traverse_jnd_sev = [bool(d >= jnd_sev) for d in delta_chi]

    # i ∈ {2, 3} (1-indexé) = indices {1, 2} (0-indexé). Δχ_1 (indice 0)
    # structurellement 0, exclu.
    fenetre_precoce = traverse_jnd_sev[1:3]
    s1 = any(fenetre_precoce)

    reste = traverse_jnd_sev[1:]  # i >= 2
    aucun_depassement = not any(reste)
    max_2_3 = max(delta_chi[1:3])
    max_4_6 = max(delta_chi[3:6])
    s2 = aucun_depassement and (max_4_6 <= max_2_3 * 1.5)

    if s1:
        lecture = "S1"
    elif s2:
        lecture = "S2"
    else:
        lecture = "AUTRE"

    return {
        "lecture": lecture,
        "traverse_jnd_sev": traverse_jnd_sev,
        "s1": s1,
        "s2": s2,
        "max_emissions_2_3": float(max_2_3),
        "max_emissions_4_6": float(max_4_6),
    }


def _comparaisons_ic(delta_chi: Sequence[float], ic_bas: float, ic_haut: float) -> dict:
    """Comparaisons DESCRIPTIVES (pas de lecture, pas de verdict) de la série
    `delta_chi` aux deux bornes de l'IC sévère -- §A13-2 rappelle que « tout
    verdict est lu sur l'IC entier, jamais au point central » : la sonde ne
    tranche rien ici, elle consigne juste ces deux vues pour le lecteur."""
    return {
        "traverse_ic_bas": [bool(d >= ic_bas) for d in delta_chi],
        "traverse_ic_haut": [bool(d >= ic_haut) for d in delta_chi],
    }


def figure_sonde(document: dict, path: str | Path) -> None:
    """`arcA_m2_sonde.png` : Δχ_i vs i (points + ligne), ligne horizontale
    pleine à `jnd_sev`, lignes pointillées aux deux bornes de l'IC sévère,
    titre mentionnant seed/Δt/k (§A13-3)."""
    delta_chi = document["delta_chi"]
    meta = document["meta"]
    jnd_sev = meta["jnd_sev"]
    ic_bas, ic_haut = meta["ic"]
    xs = list(range(1, len(delta_chi) + 1))

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(xs, delta_chi, "-o", color="#0072B2", linewidth=1.8, markersize=6,
            label="Δχ_i (chaîne registre-commis, bras v2)")
    ax.axhline(jnd_sev, color="#D55E00", linestyle="-", linewidth=1.4,
               label=f"JND sévère (pin) = {jnd_sev * 100:.2f} %")
    ax.axhline(ic_bas, color="#D55E00", linestyle="--", linewidth=1.0,
               label=f"IC sévère [{ic_bas * 100:.2f}, {ic_haut * 100:.2f}] %")
    ax.axhline(ic_haut, color="#D55E00", linestyle="--", linewidth=1.0)
    ax.set_xticks(xs)
    ax.set_xlabel("émission i")
    ax.set_ylabel("Δχ_i (max_carrier, albedo moteur vs vrai)")
    ax.set_title(
        f"Sonde §A13-3 -- seed={meta['seed']}, Δt={meta['dt']}, k={meta['k']}, "
        f"lecture={document['lecture']}")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    pins = charge_pins_severe()
    jnd_sev = pins["jnd"]
    ic_bas, ic_haut = pins["ic"]

    n_ep_total = DT_SONDE * N_EMISSIONS_SONDE

    t0 = time.perf_counter()
    verite = chaine_verite(SEED_SONDE, n_ep_total)
    t_verite = time.perf_counter() - t0

    t1 = time.perf_counter()
    res = chaine_registre(SEED_SONDE, DT_SONDE, N_EMISSIONS_SONDE, K_SONDE, arm="v2",
                          verite=verite)
    t_moteur = time.perf_counter() - t1

    delta_chi = [float(d) for d in res["delta_chi"]]
    lecture = lecture_sonde(delta_chi, jnd_sev)
    detail_ic = _comparaisons_ic(delta_chi, ic_bas, ic_haut)

    detail_lecture = {
        "traverse_jnd_sev": lecture["traverse_jnd_sev"],
        "s1": lecture["s1"],
        "s2": lecture["s2"],
        "max_emissions_2_3": lecture["max_emissions_2_3"],
        "max_emissions_4_6": lecture["max_emissions_4_6"],
        "traverse_ic_bas": detail_ic["traverse_ic_bas"],
        "traverse_ic_haut": detail_ic["traverse_ic_haut"],
    }

    chronometrage = {
        "verite_s": t_verite,
        "moteur_s": t_moteur,
        "par_episode_verite_s": t_verite / n_ep_total,
        "par_episode_moteur_s": t_moteur / n_ep_total,
    }

    document = {
        "meta": {
            "seed": SEED_SONDE, "dt": DT_SONDE, "k": K_SONDE,
            "n_emissions": N_EMISSIONS_SONDE, "jnd_sev": jnd_sev,
            "ic": [ic_bas, ic_haut],
        },
        "delta_chi": delta_chi,
        "taille_commits": [int(t) for t in res["taille_commits"]],
        "episodes_emissions": [int(e) for e in res["episodes_emissions"]],
        "chronometrage": chronometrage,
        "lecture": lecture["lecture"],
        "detail_lecture": detail_lecture,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON_PATH.write_text(json.dumps(document, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    figure_sonde(document, OUT_PNG_PATH)

    print("=" * 78)
    print("MANCHE 2 (§A13) TASK 1 -- SONDE (§A13-3) : 1 seed, Δt=4, k=256, 6 émissions")
    print("=" * 78)
    print(f"  seed={SEED_SONDE}  dt={DT_SONDE}  k={K_SONDE}  "
          f"n_emissions={N_EMISSIONS_SONDE}  arm=v2")
    print(f"  jnd_sev = {jnd_sev!r}  ic = [{ic_bas!r}, {ic_haut!r}]")
    print("  série Δχ_i (i=1..6) :")
    for i, d in enumerate(delta_chi, start=1):
        print(f"    i={i}  Δχ_i={d!r}  >=jnd_sev={lecture['traverse_jnd_sev'][i - 1]}")
    print(f"  LECTURE : {document['lecture']}  "
          f"(s1={lecture['s1']}, s2={lecture['s2']}, "
          f"max_2_3={lecture['max_emissions_2_3']!r}, "
          f"max_4_6={lecture['max_emissions_4_6']!r})")
    print(f"  chrono : vérité={t_verite:.3f} s ({chronometrage['par_episode_verite_s']:.3f} "
          f"s/épisode)  moteur={t_moteur:.3f} s "
          f"({chronometrage['par_episode_moteur_s']:.3f} s/épisode)")
    print(f"[REPORT] -> {OUT_JSON_PATH}")
    print(f"[REPORT] -> {OUT_PNG_PATH}")
    print("RAPPEL §A13-3 : la sonde ne PRONONCE rien, elle dimensionne -- point "
          "d'arrêt OBLIGATOIRE, remonter la courbe avant toute suite.")


if __name__ == "__main__":
    main()
