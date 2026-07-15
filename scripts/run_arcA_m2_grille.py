"""Manche 2 (§A13) — Tasks 2/3 : gate d'achat + GRILLE complète (mesures).

Citations verbatim (`PREREGISTRATION.md` §A13, pocCascade2phys, gravé commit
`811b11d`, AVANT toute mesure) :

  §A13-1 : « Balayages : Δt ∈ {1, 4, 16} épisodes [...] ; n_émissions = 6 par
  chaîne ; budgets k ∈ {128, 256, 400} (là où k* vit au pin, §C12) ;
  seeds {101..105}. »

  §A13-3 : « Gate d'achat (Task 2) : benchmark de coût réel AVANT la grille
  (règle W0 : ≤ une nuit ~10 h). Estimation a priori ~4-5 h CPU. Si > gate :
  grille réduite PRÉ-ÉCRITE (n_émissions 6→4, puis budget 128 retiré) —
  jamais improvisée. »

  §A13-4 (contrôles, attendus gravés AVEC LEUR PORTÉE) :
    1. « ferm-chaîne (plomberie) : commit = champ complet (sans perte) →
       chaîne moteur ≡ chaîne vraie bit-identique, Δχ_i = 0 exact partout.
       Portée : tout Δt, tout L. »
    2. « shuf-commit (discriminant) : commit permuté (histogramme préservé) →
       première re-dérivation supra-JND. Portée : budgets SOUS CAP
       uniquement. »
    3. « corruption (détecteur de contradiction) : UNE masse de bloc commise
       altérée → la re-dérivation de CETTE émission montre un supra-JND
       localisé. Portée : détection d'instrument, ne participe pas au
       verdict. »
    « Toute violation → INSTRUMENT-MUET, scellé, remonter. »

Ce script produit les MESURES (aucun verdict §A13-2 ici — le combinateur vit
dans `scripts/run_arcA_m2_verdict.py`, séparation identique à
measure/verdict de la manche 1 et de 16L₀).

CHOIX D'IMPLÉMENTATION PRÉ-ENREGISTRÉS (gravés ICI, avant toute mesure de
grille — à endosser en revue AVANT exécution, comme le facteur 1.5 de la
sonde ; aucun n'est re-réglable après coup) :

  (i)   Partage de la vérité : `chaine_verite(seed, 96)` est calculée UNE
        fois par seed (96 = 6·16, le plus long Δt) et sert les trois Δt et
        les trois budgets — licite car `centres_episodes` est stable par
        préfixe (rng frais par seed, tirages séquentiels) et
        `chaine_registre(..., verite=...)` consomme la vérité TELLE QUELLE.
        Round-trip npz float64 = bit-exact (np.save/np.load sans perte).
  (ii)  Contrôles shuf/corruption POST-HOC sur le registre stocké, via le
        foncteur É2 `rederiver_emissions` (signature stricte anti-fuite) —
        AUCUNE modification du cœur Task 0 (`run_arcA_m2_registre.py`,
        revu Approved). « Re-dérivation » est le mot É2 du contrat : le
        discriminant teste que la re-dérivation dépend du CONTENU des
        commits, pas de leur histogramme.
  (iii) shuf-commit : permutation des moyennes de feuilles de CHAQUE commit
        (histogramme préservé par commit, topologie intacte),
        `default_rng(9001 + 1000*dt + seed)` — même famille de graine que le
        shuf manche 1 (9001 + 1000*L + seed) ; UNE permutation tirée par
        commit, séquentiellement sur le même flux. Attendu (portée gravée :
        seed 101, Δt = 4, budgets sous cap {128, 256, 400}) : la PREMIÈRE
        re-dérivation dépendante d'un commit (émission 2) donne
        Δχ(re-dérivé, vrai) ≥ jnd_sev (point central du pin — contrôle
        d'instrument, pas un verdict : le verdict §A13-2 lit l'IC entier).
  (iv)  corruption : sur le registre de la cellule (seed 101, Δt = 4,
        k = 256), la moyenne de la feuille de PLUS GRANDE AIRE du commit de
        l'émission 3 est altérée de +10·S_HALF_OP (sature l'albédo local —
        garantit un effet supra-bruit) ; attendus : re-dérivations des
        émissions 1..3 BIT-IDENTIQUES à la re-dérivation non corrompue
        (la corruption du commit i touche les re-dérivations > i seulement),
        émission 4 : Δχ(corrompu, non-corrompu) ≥ jnd_sev ET ≥ 50 % de
        l'énergie |Δalbédo| dans le rectangle de la feuille corrompue
        AVANT évolution (localisation mesurée au ré-ancrage — après
        évolution la dynamique étale, c'est physique, pas un défaut
        d'instrument).
  (v)   Spot-check É2/É3 (portée : cellule seed 101, Δt = 4, k = 256) :
        `rederiver_emissions(registre, ...)` reproduit BIT-À-BIT les
        readouts-moteur émis de la chaîne (É2-déterminisme au niveau
        harnais) ; double re-dérivation bit-identique (É3). Les propriétés
        É1/É3 générales sont couvertes par les tests Task 0.
  (vi)  Gate d'achat : benchmark RÉEL = la cellule la moins chère
        (seed 101, Δt = 1, k = 256, 6 émissions) chronométrée sur LA machine
        qui exécutera la grille, extrapolée par la comptabilité EXACTE des
        épisodes de la grille (vérité 480 + moteur 1890 + ferm 126 +
        rederive 24 + shuf 72 + corruption 48 = 2640 épisodes) ; PASS ssi
        projection ≤ 10 h (36 000 s). Si FAIL : grille réduite pré-écrite
        §A13-3, jamais improvisée.

Usage (phases résumables, convention measure_qt) :
  .venv/bin/python scripts/run_arcA_m2_grille.py --gate
  .venv/bin/python scripts/run_arcA_m2_grille.py --verite --seed 101
  .venv/bin/python scripts/run_arcA_m2_grille.py --cell --seed 101 --dt 4
  .venv/bin/python scripts/run_arcA_m2_grille.py --ferm --dt 4
  .venv/bin/python scripts/run_arcA_m2_grille.py --controles
  .venv/bin/python scripts/run_arcA_m2_grille.py --assemble
  .venv/bin/python scripts/run_arcA_m2_grille.py --tout   # séquence complète
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.run_arcA_m2_registre import (
    chaine_registre,
    chaine_verite,
    rederiver_emissions,
)
from scripts.run_arcA_revalidate import GRID, S_HALF_OP
from src.albedo import albedo, delta_chi
from src.summary_quadtree import SummaryQT, regenerate_qt, size_floats_qt

OUT_DIR = ROOT / "outputs" / "arcA"
PARTS_DIR = OUT_DIR / "m2_parts"
GATE_PATH = OUT_DIR / "m2_gate_achat.json"
MEASURES_PATH = OUT_DIR / "m2_measures.npz"
CONTROLES_PATH = OUT_DIR / "m2_controles.json"
PINS_PATH = ROOT / "outputs" / "arcC" / "pins_spatial.json"

# --- Paramètres FIGÉS de la grille (§A13-1) — ne pas paramétrer au-delà. -----
SEEDS: tuple[int, ...] = (101, 102, 103, 104, 105)
DTS: tuple[int, ...] = (1, 4, 16)
BUDGETS: tuple[int, ...] = (128, 256, 400)
N_EMISSIONS: int = 6
N_EP_VERITE: int = N_EMISSIONS * max(DTS)  # 96 — sert tous les Δt (choix (i))

# Gate d'achat (§A13-3) : règle W0, une nuit.
GATE_SECONDS: float = 10.0 * 3600.0

# Contrôles (portées gravées, choix (ii)-(v)).
SEED_CONTROLE: int = 101
DT_SHUF: int = 4
K_CORRUPTION: int = 256
EMISSION_CORROMPUE: int = 3  # 1-indexée ; touche les re-dérivations > 3.
AMPLITUDE_CORRUPTION_S: float = 10.0  # × S_HALF_OP, ajouté à UNE moyenne.
FRACTION_ENERGIE_LOCALISEE: float = 0.50

# Comptabilité EXACTE des épisodes de la grille (choix (vi)) — dérivée des
# paramètres figés, PAS un seuil : sert l'extrapolation du gate.
_EP_VERITE: int = N_EP_VERITE * len(SEEDS)                        # 480
_EP_MOTEUR: int = sum(N_EMISSIONS * dt for dt in DTS) * len(BUDGETS) * len(SEEDS)  # 1890
_EP_FERM: int = sum(N_EMISSIONS * dt for dt in DTS)               # 126 (seed 101)
_EP_REDERIVE_SPOT: int = N_EMISSIONS * DT_SHUF                    # 24
_EP_SHUF: int = N_EMISSIONS * DT_SHUF * len(BUDGETS)              # 72
_EP_CORRUPTION: int = 2 * N_EMISSIONS * DT_SHUF                   # 48 (base + corrompu)
EPISODES_GRILLE: int = (_EP_VERITE + _EP_MOTEUR + _EP_FERM
                        + _EP_REDERIVE_SPOT + _EP_SHUF + _EP_CORRUPTION)


def charge_pins() -> dict:
    """Charge les deux régimes (sévère + laxiste) depuis
    `outputs/arcC/pins_spatial.json` — mêmes clés que `charge_pins_severe`
    de la sonde, étendu au laxiste (diagnostic §A13-2). Fail-loud si absent."""
    data = json.loads(PINS_PATH.read_text(encoding="utf-8"))
    out = {}
    for regime in ("severe", "laxiste"):
        r = data["regimes"][regime]
        out[regime] = {"jnd": float(r["jnd"]),
                       "ic": tuple(float(x) for x in r["ic"])}
    return out


# --- Vérité partagée (choix (i)) ---------------------------------------------


def _verite_path(seed: int) -> Path:
    return PARTS_DIR / f"verite_s{seed}.npz"


def phase_verite(seed: int) -> Path:
    """Calcule et cache `chaine_verite(seed, 96)` (96 états float64, ~3 Mo).
    Round-trip npz float64 bit-exact — la vérité rechargée est LA vérité."""
    path = _verite_path(seed)
    if path.exists():
        print(f"[SKIP] vérité seed={seed} déjà présente : {path}")
        return path
    t0 = time.perf_counter()
    verite = chaine_verite(seed, N_EP_VERITE)
    dt_s = time.perf_counter() - t0
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{f"t{t}": v for t, v in verite.items()},
                        seed=np.int64(seed), n_episodes=np.int64(N_EP_VERITE))
    print(f"[OK] vérité seed={seed} : {N_EP_VERITE} épisodes en {dt_s:.1f} s "
          f"({dt_s / N_EP_VERITE:.3f} s/épisode) -> {path}")
    return path


def charge_verite(seed: int) -> dict[int, np.ndarray]:
    path = _verite_path(seed)
    if not path.exists():
        raise RuntimeError(
            f"vérité seed={seed} absente ({path}) — lancer --verite --seed "
            f"{seed} d'abord (phases résumables, jamais implicites).")
    with np.load(path) as data:
        return {int(k[1:]): np.array(data[k], dtype=np.float64, copy=True)
                for k in data.files if k.startswith("t")}


# --- Cellules v2 (le corps de la grille) --------------------------------------


def _cell_path(seed: int, dt: int) -> Path:
    return PARTS_DIR / f"cell_s{seed}_dt{dt}.npz"


def _registre_vers_arrays(registre: list[SummaryQT], prefix: str) -> dict:
    """Aplatit un registre (liste de SummaryQT) en arrays npz nommés
    `{prefix}_i{n}_topo` / `{prefix}_i{n}_means` (n = émission 1-indexée) —
    stockage AUDIT + contrôles post-hoc (choix (ii)), aucun pickle."""
    out: dict = {}
    for n, commit in enumerate(registre, start=1):
        out[f"{prefix}_i{n}_topo"] = np.asarray(commit.topology, dtype=np.uint8)
        out[f"{prefix}_i{n}_means"] = np.asarray(commit.means, dtype=np.float64)
    return out


def _arrays_vers_registre(data, prefix: str, n_emissions: int) -> list[SummaryQT]:
    registre = []
    for n in range(1, n_emissions + 1):
        topo = np.array(data[f"{prefix}_i{n}_topo"], dtype=np.uint8, copy=True)
        means = np.array(data[f"{prefix}_i{n}_means"], dtype=np.float64, copy=True)
        registre.append(SummaryQT(topology=topo, means=means, shape=(GRID.H, GRID.W)))
    return registre


def phase_cell(seed: int, dt: int) -> Path:
    """Les 3 budgets de la cellule (seed, Δt) : chaîne v2 complète par budget,
    vérité partagée (choix (i)). Sauve les séries Δχ_i, tailles de commits,
    et le REGISTRE de chaque budget (audit + contrôles post-hoc)."""
    path = _cell_path(seed, dt)
    if path.exists():
        print(f"[SKIP] cellule seed={seed} dt={dt} déjà présente : {path}")
        return path
    verite = charge_verite(seed)
    payload: dict = {"seed": np.int64(seed), "dt": np.int64(dt),
                     "n_emissions": np.int64(N_EMISSIONS),
                     "budgets": np.array(BUDGETS, dtype=np.int64)}
    t0 = time.perf_counter()
    for k in BUDGETS:
        res = chaine_registre(seed, dt, N_EMISSIONS, k, arm="v2", verite=verite)
        payload[f"delta_chi_k{k}"] = np.array(res["delta_chi"], dtype=np.float64)
        payload[f"taille_commits_k{k}"] = np.array(res["taille_commits"], dtype=np.int64)
        payload.update(_registre_vers_arrays(res["registre"], f"reg_k{k}"))
    dt_s = time.perf_counter() - t0
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    print(f"[OK] cellule seed={seed} dt={dt} (3 budgets) en {dt_s:.1f} s -> {path}")
    return path


# --- Contrôle 1 : ferm-chaîne (§A13-4-1) --------------------------------------


def _ferm_path(dt: int) -> Path:
    return PARTS_DIR / f"ferm_dt{dt}.npz"


def phase_ferm(dt: int) -> Path:
    """ferm-chaîne, seed 101 (portée gravée choix (ii) : le contrôle de
    plomberie rejoue le protocole avec commit sans perte — attendu
    Δχ_i = 0.0 EXACT partout, tout Δt)."""
    path = _ferm_path(dt)
    if path.exists():
        print(f"[SKIP] ferm dt={dt} déjà présent : {path}")
        return path
    verite = charge_verite(SEED_CONTROLE)
    res = chaine_registre(SEED_CONTROLE, dt, N_EMISSIONS, k=GRID.H * GRID.W,
                          arm="ferm", verite=verite)
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path,
                        delta_chi=np.array(res["delta_chi"], dtype=np.float64),
                        seed=np.int64(SEED_CONTROLE), dt=np.int64(dt))
    print(f"[OK] ferm dt={dt} : Δχ = {res['delta_chi']} -> {path}")
    return path


# --- Contrôles 2/3 + spot-check É2/É3 (post-hoc, choix (ii)-(v)) --------------


def _permute_registre(registre: list[SummaryQT], seed: int, dt: int) -> list[SummaryQT]:
    """shuf-commit (choix (iii)) : pour CHAQUE commit, permutation des
    moyennes de feuilles (histogramme préservé, topologie intacte) —
    permutations tirées séquentiellement de `default_rng(9001 + 1000*dt +
    seed)` (même famille de graine que le shuf manche 1)."""
    rng = np.random.default_rng(9001 + 1000 * dt + seed)
    out = []
    for commit in registre:
        perm = rng.permutation(commit.means.size)
        out.append(SummaryQT(topology=np.array(commit.topology, copy=True),
                             means=np.array(commit.means, copy=True)[perm],
                             shape=commit.shape))
    return out


def _aires_feuilles(commit: SummaryQT) -> np.ndarray:
    """Aire (en cellules) de chaque feuille, dans l'ordre préordre de
    `means` — re-parcours du préordre (même ordre d'enfants que
    `_decoder_preordre`, cf. src/summary_quadtree.py : _ORDRE_ENFANTS)."""
    topo = commit.topology
    aires: list[int] = []
    idx = [0]

    def _walk(depth: int) -> None:
        bit = int(topo[idx[0]])
        idx[0] += 1
        if bit == 0:
            cote = GRID.H // (2 ** depth)
            aires.append(cote * cote)
            return
        for _ in range(4):
            _walk(depth + 1)

    _walk(0)
    if idx[0] != topo.size or len(aires) != commit.means.size:
        raise RuntimeError(
            f"_aires_feuilles : parcours préordre incohérent (idx={idx[0]}/"
            f"{topo.size}, feuilles={len(aires)}/{commit.means.size}).")
    return np.array(aires, dtype=np.int64)


def _rect_feuille(commit: SummaryQT, cible: int) -> tuple[int, int, int]:
    """(row, col, cote) de la feuille d'indice préordre `cible` — pour la
    mesure de localisation (choix (iv))."""
    topo = commit.topology
    idx = [0]
    n_feuille = [0]
    resultat: list[tuple[int, int, int]] = []

    def _walk(depth: int, row: int, col: int) -> None:
        bit = int(topo[idx[0]])
        idx[0] += 1
        if bit == 0:
            if n_feuille[0] == cible:
                resultat.append((row, col, GRID.H // (2 ** depth)))
            n_feuille[0] += 1
            return
        demi = GRID.H // (2 ** (depth + 1))
        for drow, dcol in ((0, 0), (0, 1), (1, 0), (1, 1)):
            _walk(depth + 1, row + drow * demi, col + dcol * demi)

    _walk(0, 0, 0)
    if not resultat:
        raise RuntimeError(f"_rect_feuille : feuille {cible} introuvable.")
    return resultat[0]


def _dchi(a: np.ndarray, b: np.ndarray) -> float:
    return float(delta_chi(albedo(a, S_HALF_OP), albedo(b, S_HALF_OP))["max_carrier"])


def phase_controles() -> Path:
    """Contrôles post-hoc (§A13-4-2/3 + spot-check É2/É3, choix (ii)-(v)).
    Requiert la cellule (101, Δt=4) et la vérité seed 101 (parts).
    Écrit `m2_controles.json` — les attendus y sont vérifiés MÉCANIQUEMENT,
    toute violation est un fait consigné (l'assemble la transforme en
    INSTRUMENT-MUET, §A13-4)."""
    pins = charge_pins()
    jnd_sev = pins["severe"]["jnd"]

    cell = np.load(_cell_path(SEED_CONTROLE, DT_SHUF))
    verite = charge_verite(SEED_CONTROLE)
    doc: dict = {"portees": {
        "shuf": {"seed": SEED_CONTROLE, "dt": DT_SHUF, "budgets": list(BUDGETS)},
        "corruption": {"seed": SEED_CONTROLE, "dt": DT_SHUF, "k": K_CORRUPTION,
                       "emission_corrompue": EMISSION_CORROMPUE},
        "spot_e2_e3": {"seed": SEED_CONTROLE, "dt": DT_SHUF, "k": K_CORRUPTION},
    }, "jnd_sev_reference_controle": jnd_sev}

    # --- Spot-check É2 (re-dérivation ≡ readouts émis) + É3 (double run). ----
    registre = _arrays_vers_registre(cell, f"reg_k{K_CORRUPTION}", N_EMISSIONS)
    rederive_1 = rederiver_emissions(registre, SEED_CONTROLE, DT_SHUF,
                                     N_EMISSIONS, arm="v2")
    rederive_2 = rederiver_emissions(registre, SEED_CONTROLE, DT_SHUF,
                                     N_EMISSIONS, arm="v2")
    e3_ok = all(np.array_equal(a, b) for a, b in zip(rederive_1, rederive_2))
    # É2-déterminisme harnais : la chaîne vivante refaite ici doit émettre des
    # readouts bit-identiques à la re-dérivation depuis le registre stocké.
    res_vif = chaine_registre(SEED_CONTROLE, DT_SHUF, N_EMISSIONS,
                              K_CORRUPTION, arm="v2", verite=verite)
    e2_ok = all(np.array_equal(a, b)
                for a, b in zip(rederive_1, res_vif["readouts_moteur"]))
    doc["spot_e2_e3"] = {"e2_rederive_egale_emis": bool(e2_ok),
                         "e3_double_rederive_bit_identique": bool(e3_ok)}

    # --- Contrôle 2 : shuf-commit (budgets sous cap). -------------------------
    shuf_out = {}
    for k in BUDGETS:
        reg_k = _arrays_vers_registre(cell, f"reg_k{k}", N_EMISSIONS)
        reg_shuf = _permute_registre(reg_k, SEED_CONTROLE, DT_SHUF)
        # Garde histogramme préservé + budget inchangé (fail-loud).
        for orig, perm in zip(reg_k, reg_shuf):
            if not np.array_equal(np.sort(orig.means), np.sort(perm.means)):
                raise RuntimeError("shuf : histogramme NON préservé — bug de contrôle.")
            if size_floats_qt(perm) != size_floats_qt(orig):
                raise RuntimeError("shuf : taille de commit modifiée — bug de contrôle.")
        red_shuf = rederiver_emissions(reg_shuf, SEED_CONTROLE, DT_SHUF,
                                       N_EMISSIONS, arm="v2")
        # Δχ vs monde VRAI aux émissions (référence = vérité, comme la chaîne).
        serie = [_dchi(red_shuf[i], verite[(i + 1) * DT_SHUF])
                 for i in range(N_EMISSIONS)]
        # « Première re-dérivation » dépendante d'un commit = émission 2
        # (readout@t_1 re-dérivé ne dépend d'AUCUN commit — cf.
        # rederiver_emissions ; convention sonde : Δχ_1 exclu).
        shuf_out[str(k)] = {"delta_chi_rederive": serie,
                            "premiere_rederivation_supra_jnd": bool(serie[1] >= jnd_sev)}
    doc["shuf"] = shuf_out

    # --- Contrôle 3 : corruption (détection seulement). -----------------------
    reg_base = _arrays_vers_registre(cell, f"reg_k{K_CORRUPTION}", N_EMISSIONS)
    red_base = rederiver_emissions(reg_base, SEED_CONTROLE, DT_SHUF,
                                   N_EMISSIONS, arm="v2")
    reg_corr = [SummaryQT(topology=np.array(c.topology, copy=True),
                          means=np.array(c.means, copy=True), shape=c.shape)
                for c in reg_base]
    commit_cible = reg_corr[EMISSION_CORROMPUE - 1]
    aires = _aires_feuilles(commit_cible)
    feuille = int(np.argmax(aires))
    commit_cible.means[feuille] += AMPLITUDE_CORRUPTION_S * S_HALF_OP
    # Localisation mesurée AU RÉ-ANCRAGE (avant évolution — choix (iv)).
    a_base = albedo(regenerate_qt(reg_base[EMISSION_CORROMPUE - 1]), S_HALF_OP)
    a_corr = albedo(regenerate_qt(commit_cible), S_HALF_OP)
    diff = np.abs(a_corr - a_base)
    row, col, cote = _rect_feuille(commit_cible, feuille)
    energie_totale = float(diff.sum())
    energie_locale = float(diff[row:row + cote, col:col + cote].sum())
    if energie_totale <= 0.0:
        raise RuntimeError("corruption : altération sans effet sur l'albédo — "
                           "amplitude à revoir, contrôle muet.")
    red_corr = rederiver_emissions(reg_corr, SEED_CONTROLE, DT_SHUF,
                                   N_EMISSIONS, arm="v2")
    prefixe_intact = all(np.array_equal(red_corr[i], red_base[i])
                         for i in range(EMISSION_CORROMPUE))
    dchi_apres = _dchi(red_corr[EMISSION_CORROMPUE], red_base[EMISSION_CORROMPUE])
    doc["corruption"] = {
        "feuille_corrompue": {"index_preordre": feuille, "row": row, "col": col,
                              "cote": cote, "aire": int(aires[feuille])},
        "prefixe_1_a_3_bit_identique": bool(prefixe_intact),
        "dchi_emission_4_corrompu_vs_base": dchi_apres,
        "supra_jnd_emission_4": bool(dchi_apres >= jnd_sev),
        "fraction_energie_localisee_reancrage": energie_locale / energie_totale,
        "localisation_ok": bool(energie_locale / energie_totale
                                >= FRACTION_ENERGIE_LOCALISEE),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CONTROLES_PATH.write_text(json.dumps(doc, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(f"[OK] contrôles post-hoc -> {CONTROLES_PATH}")
    return CONTROLES_PATH


# --- Gate d'achat (Task 2, §A13-3) --------------------------------------------


def phase_gate() -> Path:
    """Benchmark RÉEL (choix (vi)) : cellule la moins chère (seed 101,
    Δt = 1, k = 256, 6 émissions — 6 épisodes vérité + 6 moteur), chronométrée
    ICI, extrapolée à la comptabilité exacte de la grille. PASS ssi ≤ 10 h."""
    t0 = time.perf_counter()
    verite = chaine_verite(SEED_CONTROLE, N_EMISSIONS)  # Δt=1 → 6 épisodes.
    t_verite = time.perf_counter() - t0
    t1 = time.perf_counter()
    chaine_registre(SEED_CONTROLE, 1, N_EMISSIONS, 256, arm="v2", verite=verite)
    t_moteur = time.perf_counter() - t1
    par_ep = (t_verite + t_moteur) / (2 * N_EMISSIONS)
    projection_s = par_ep * EPISODES_GRILLE
    doc = {
        "benchmark": {"seed": SEED_CONTROLE, "dt": 1, "k": 256,
                      "n_emissions": N_EMISSIONS,
                      "verite_s": t_verite, "moteur_s": t_moteur,
                      "par_episode_s": par_ep},
        "comptabilite_episodes": {
            "verite": _EP_VERITE, "moteur_v2": _EP_MOTEUR, "ferm": _EP_FERM,
            "rederive_spot": _EP_REDERIVE_SPOT, "shuf": _EP_SHUF,
            "corruption": _EP_CORRUPTION, "total": EPISODES_GRILLE},
        "projection_s": projection_s,
        "projection_h": projection_s / 3600.0,
        "gate_s": GATE_SECONDS,
        "verdict_gate": "PASS" if projection_s <= GATE_SECONDS else "FAIL",
        "regle_si_fail": "grille réduite PRÉ-ÉCRITE §A13-3 : n_émissions 6→4, "
                         "puis budget 128 retiré — jamais improvisée.",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    GATE_PATH.write_text(json.dumps(doc, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"[OK] gate d'achat : {par_ep:.3f} s/épisode × {EPISODES_GRILLE} "
          f"épisodes = {projection_s / 3600.0:.2f} h -> {doc['verdict_gate']} "
          f"(gate 10 h) -> {GATE_PATH}")
    return GATE_PATH


# --- Assemble ------------------------------------------------------------------


def phase_assemble() -> Path:
    """Assemble les parts en `m2_measures.npz` (layout (seed, dt, k) →
    delta_chi[6]) + relit `m2_controles.json` et le gate — consigne
    `instrument_muet` si un attendu §A13-4 est violé (le VERDICT lui-même
    vit dans run_arcA_m2_verdict.py)."""
    manquants = []
    payload: dict = {"seeds": np.array(SEEDS, dtype=np.int64),
                     "dts": np.array(DTS, dtype=np.int64),
                     "budgets": np.array(BUDGETS, dtype=np.int64),
                     "n_emissions": np.int64(N_EMISSIONS)}
    # delta_chi : tenseur (n_seeds, n_dts, n_budgets, N_EMISSIONS).
    tenseur = np.full((len(SEEDS), len(DTS), len(BUDGETS), N_EMISSIONS), np.nan)
    tailles = np.full((len(SEEDS), len(DTS), len(BUDGETS), N_EMISSIONS), -1,
                      dtype=np.int64)
    for si, seed in enumerate(SEEDS):
        for di, dt in enumerate(DTS):
            p = _cell_path(seed, dt)
            if not p.exists():
                manquants.append(str(p))
                continue
            with np.load(p) as data:
                for ki, k in enumerate(BUDGETS):
                    tenseur[si, di, ki, :] = data[f"delta_chi_k{k}"]
                    tailles[si, di, ki, :] = data[f"taille_commits_k{k}"]
    if manquants:
        raise RuntimeError("assemble : parts manquantes :\n  " + "\n  ".join(manquants))
    payload["delta_chi"] = tenseur
    payload["taille_commits"] = tailles

    # Contrôle ferm (Δχ = 0.0 exact partout, tout Δt).
    ferm_viole = []
    for dt in DTS:
        p = _ferm_path(dt)
        if not p.exists():
            raise RuntimeError(f"assemble : contrôle ferm dt={dt} manquant ({p}).")
        with np.load(p) as data:
            serie = data["delta_chi"]
            payload[f"ferm_dt{dt}"] = serie
            if not np.all(serie == 0.0):
                ferm_viole.append(dt)

    if not CONTROLES_PATH.exists():
        raise RuntimeError(f"assemble : {CONTROLES_PATH} manquant — lancer --controles.")
    controles = json.loads(CONTROLES_PATH.read_text(encoding="utf-8"))
    shuf_viole = [k for k, v in controles["shuf"].items()
                  if not v["premiere_rederivation_supra_jnd"]]
    spot = controles["spot_e2_e3"]
    violations = {
        "ferm_dts_violes": ferm_viole,
        "shuf_budgets_violes": shuf_viole,
        "e2_viole": not spot["e2_rederive_egale_emis"],
        "e3_viole": not spot["e3_double_rederive_bit_identique"],
        # corruption : détection SEULEMENT (§A13-4-3) — consignée, PAS muette.
        "corruption_detection": controles["corruption"],
    }
    instrument_muet = bool(ferm_viole or shuf_viole
                           or violations["e2_viole"] or violations["e3_viole"])
    payload["instrument_muet"] = np.bool_(instrument_muet)

    np.savez_compressed(MEASURES_PATH, **payload)
    resume = {"instrument_muet": instrument_muet, "violations": {
        k: v for k, v in violations.items() if k != "corruption_detection"}}
    print(f"[OK] assemble -> {MEASURES_PATH}")
    print(json.dumps(resume, indent=2, ensure_ascii=False))
    if instrument_muet:
        print("!! INSTRUMENT-MUET (§A13-4) : verdict scellé, remonter — ne PAS "
              "lancer le combinateur comme si de rien n'était.")
    return MEASURES_PATH


# --- CLI -----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--verite", action="store_true")
    parser.add_argument("--cell", action="store_true")
    parser.add_argument("--ferm", action="store_true")
    parser.add_argument("--controles", action="store_true")
    parser.add_argument("--assemble", action="store_true")
    parser.add_argument("--tout", action="store_true",
                        help="Séquence complète (gate déjà PASSé requis).")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dt", type=int, default=None)
    args = parser.parse_args()

    if args.gate:
        phase_gate()
        return
    if args.verite:
        if args.seed is None:
            parser.error("--verite exige --seed")
        phase_verite(args.seed)
        return
    if args.cell:
        if args.seed is None or args.dt is None:
            parser.error("--cell exige --seed et --dt")
        phase_cell(args.seed, args.dt)
        return
    if args.ferm:
        if args.dt is None:
            parser.error("--ferm exige --dt")
        phase_ferm(args.dt)
        return
    if args.controles:
        phase_controles()
        return
    if args.assemble:
        phase_assemble()
        return
    if args.tout:
        if not GATE_PATH.exists():
            raise RuntimeError("--tout : gate d'achat absent — lancer --gate "
                               "d'abord (Task 2 AVANT la grille, §A13-3).")
        gate = json.loads(GATE_PATH.read_text(encoding="utf-8"))
        if gate["verdict_gate"] != "PASS":
            raise RuntimeError("--tout : gate d'achat FAIL — grille réduite "
                               "pré-écrite §A13-3, décision à remonter.")
        for seed in SEEDS:
            phase_verite(seed)
        for seed in SEEDS:
            for dt in DTS:
                phase_cell(seed, dt)
        for dt in DTS:
            phase_ferm(dt)
        phase_controles()
        phase_assemble()
        return
    parser.error("préciser une phase : --gate | --verite | --cell | --ferm | "
                 "--controles | --assemble | --tout")


if __name__ == "__main__":
    main()
