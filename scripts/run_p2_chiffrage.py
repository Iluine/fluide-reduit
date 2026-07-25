"""P2 — CHIFFRAGE DU NOYAU DE RENDU (runner des deux cellules).

Pré-enregistrement qui fait foi : `claude/prereg-p2-chiffrage-rendu.md`
(ENDOSSÉ, gravé §A38). Les BANDES, les BRANCHES et les protocoles y sont
gravés : ce fichier les RECOPIE et les prononce mécaniquement, il ne les
interprète pas et n'en ajuste aucun.

PORTÉE, dite d'abord (prereg §Portée) : par D-P1-1 (Option B), la COMPOSITION
est EXCLUE — le rééchantillonnage à l'échelle écran appartient au compositeur,
différé post-P3. P2 chiffre le NOYAU (encodage sRGB + quantification, Y = A) et
l'instrument R1 tel qu'il servira en session. **P2 est la première ancre de la
moitié manquante, PAS la moitié entière** : « le rendu coûte X » serait une
surclame ; P2 dit « le NOYAU coûte X ».

  Cellule 1 — `rendu_r1` numpy, champ albédo RÉEL (h_s101, s_L10, celui des
    images d'exemple de P1), `taille_px` de la géométrie de session §C7.
    Bande [0.05, 5] ms. Consommateur (D17) : l'intégrité des timings de
    présentation du régime sévère (§C0).
  Cellule 2 — étages sRGB + quantification (Y = A) en f32 sur 1920×1080, GPU.
    L'ÉQUIVALENCE au numpy passe D'ABORD, tolérance ZÉRO. Bande [0.02, 0.5] ms.
    Consommateur (D17) : la ligne de budget « ~16.7 ms pour un rendu jamais
    chiffré ».

QUI LANCE QUOI. Les 300 appels officiels sont VERDICT-GRADE : iluin-tworings3,
terminal natif, **lancés par Romain**. `--n-appels`/`--n-chauffe` existent pour
qu'une session de build puisse vérifier que le runner TOURNE sans produire de
faux chiffre ; leurs défauts SONT les valeurs gravées, et le JSON porte
`verdict_grade` = (300 appels, 30 de chauffe) — un run raccourci se dénonce
lui-même dans son propre fichier.

R1 EST INTOUCHÉ : ce runner l'IMPORTE (cellule 1) et RECOPIE ses formules
(cellule 2, contrainte du prereg — la sonde GPU ne peut pas importer un module
qui exige du float64). L'empreinte de R1 est recalculée dans le JSON de sortie,
et le sha attendu est cité en commentaire au-dessus de la recopie : si R1 bouge,
la recopie doit être re-confrontée."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.run_arcA_measure import _load_history
from scripts.run_arcA_revalidate import S_HALF_OP
from src import arcC_rendu
from src.albedo import albedo
from src.arcC_calibration import (C_DEG_CIBLE_DEFAUT, PORTEUSE_CYC_PAR_DOMAINE_DEFAUT,
                                  observation_cellule_pic_csf, pixels_par_degre,
                                  taille_domaine_px)
from src.arcC_rendu import quantifie_uint8, rendu_r1, srgb_encode
from src.f1_gpu.backend import cupy_disponible

# --- Constantes RECOPIÉES du prereg (elles ne s'ajustent pas) ----------------

BANDE_CELLULE1_MS: tuple[float, float] = (0.05, 5.0)
BANDE_CELLULE2_MS: tuple[float, float] = (0.02, 0.5)
# Seuil de la lecture pré-écrite de la cellule 1 (§C0, timings de présentation).
SEUIL_TIMINGS_P3_MS: float = 5.0
# Marge V4 gravée (gate (i) satisfait : 14.490 ms sur 16.7).
MARGE_V4_MS: float = 2.199
N_APPELS_GRAVE: int = 300
N_CHAUFFE_GRAVE: int = 30
# Échelle V4 de la cellule 2 (H, W) — ~2.07e6 px.
FORME_V4: tuple[int, int] = (1080, 1920)
# Champ réel de la cellule 1 : celui des images d'exemple de P1.
SEED_SOURCE: int = 101
CLE_CHAMP: str = "s_L10"

VERDICT_DANS_BANDE: str = "DANS_BANDE"
VERDICT_AUTRE: str = "AUTRE"


# --- Chronométrie ------------------------------------------------------------


def chronometre(appel, *, n_appels: int, n_chauffe: int, synchronise=None) -> dict:
    """Médiane et p95 (ms) sur `n_appels` appels après `n_chauffe` de chauffe,
    chrono monotone hôte (`time.perf_counter`).

    `synchronise` (cellule 2) est appelé AVANT CHAQUE LECTURE D'HORLOGE — les
    deux, celle d'entrée comme celle de sortie : sans la première, le temps
    d'une file d'attente laissée par l'itération précédente s'imputerait à
    l'itération courante (tradition B6)."""
    for _ in range(n_chauffe):
        appel()
    durees_ms: list[float] = []
    for _ in range(n_appels):
        if synchronise is not None:
            synchronise()
        t0 = time.perf_counter()
        appel()
        if synchronise is not None:
            synchronise()
        durees_ms.append((time.perf_counter() - t0) * 1000.0)
    tableau = np.asarray(durees_ms, dtype=np.float64)
    return dict(n_appels=n_appels, n_chauffe=n_chauffe,
                mediane_ms=float(np.median(tableau)),
                p95_ms=float(np.percentile(tableau, 95.0)),
                min_ms=float(tableau.min()), max_ms=float(tableau.max()))


# --- Bandes et lectures, prononcées MÉCANIQUEMENT ----------------------------


def classe_bande(mediane_ms: float, bande: tuple[float, float]) -> str:
    """`DANS_BANDE` si la médiane est dans la bande pré-écrite (bornes
    incluses), `AUTRE` sinon — hors bande se REMONTE tel quel, il ne se
    rattrape pas."""
    return VERDICT_DANS_BANDE if bande[0] <= mediane_ms <= bande[1] else VERDICT_AUTRE


def lecture_cellule1(mediane_ms: float) -> dict:
    """Lecture pré-écrite de la cellule 1 (prereg), prononcée mécaniquement et
    TOTALE : `≤ 5 ms` ou `> 5 ms`, aucune zone muette."""
    if mediane_ms <= SEUIL_TIMINGS_P3_MS:
        branche, texte = "1", (
            "mediane <= 5 ms : le rendu est INVISIBLE dans les timings de session P3 "
            "(regime severe, §C0). Aucune modification de protocole n'est requise.")
    else:
        branche, texte = "2", (
            "mediane > 5 ms : le protocole P3 doit PRE-CALCULER les stimuli rendus. "
            "C'est une decision de PROTOCOLE, prise AVANT la session humaine, jamais "
            "pendant (prereg P3, pre-requis 3).")
    return dict(branche=branche, texte=texte, seuil_ms=SEUIL_TIMINGS_P3_MS)


def lecture_cellule2(mediane_ms: float) -> dict:
    """Lectures pré-écrites de la cellule 2 (prereg), les DEUX branches.

    RENDRE « ≪ » MÉCANIQUE — le seul point du prereg qui n'était pas un
    opérateur, et il est REMONTÉ comme tel. La branche 1 dit « médiane ≪ marge
    V4 (2.199 ms) », la branche 2 « comparable ou supérieure » : entre les deux
    il n'existe aucun nombre gravé. Règle appliquée, dérivée des seuls chiffres
    DÉJÀ gravés, sans en inventer un :

      - médiane <= 0.5 ms (le HAUT de la bande gravée de cette cellule, soit
        0.227 x la marge) ⇒ branche 1 : à cette échelle « ≪ » est garanti par
        la bande elle-même ;
      - médiane >= 2.199 ms (la marge) ⇒ branche 2 ;
      - entre les deux ⇒ AUCUNE branche prononcée, le ratio est surfacé et la
        lecture remonte à Romain. Prononcer là serait choisir un seuil.
    """
    ratio = mediane_ms / MARGE_V4_MS
    if mediane_ms <= BANDE_CELLULE2_MS[1]:
        return dict(branche="1", ratio_sur_marge_v4=ratio, texte=(
            "mediane << marge V4 (2.199 ms) : LECTURE GRAVEE D'AVANCE -- le cout inconnu "
            "du rendu ne vit PAS dans l'encodage ; il vit dans la COMPOSITION et dans "
            "l'optique au-dela de Y = A (R2+). La dette se DEPLACE, elle ne retrecit pas. "
            "INTERDICTION PRE-ECRITE d'en conclure « le rendu tient »."))
    if mediane_ms >= MARGE_V4_MS:
        return dict(branche="2", ratio_sur_marge_v4=ratio, texte=(
            "mediane comparable ou superieure a la marge V4 : le repli 33.3 ms (§A18, "
            "porte pre-nommee) passe de « nomme » a « a instruire » -- decision NEUVE, "
            "jamais un enchainement."))
    return dict(branche=None, ratio_sur_marge_v4=ratio, texte=(
        "mediane entre le haut de la bande gravee (0.5 ms) et la marge V4 (2.199 ms) : "
        "AUCUNE branche prononcee. Le prereg n'a grave aucun nombre dans cet "
        "intervalle ; le ratio est surface, la lecture appartient a Romain."))


# --- Cellule 1 ---------------------------------------------------------------


def champ_reel() -> np.ndarray:
    """Le champ albédo de la cellule 1 : `albedo(s, S_HALF_OP)` sur `s_L10` de
    l'histoire seed 101 — celui des images d'exemple de P1, pour que le
    chiffrage porte sur l'objet que l'inspection visuelle a déjà vu."""
    hist = _load_history(SEED_SOURCE)
    return albedo(np.asarray(hist[CLE_CHAMP], dtype=np.float64), S_HALF_OP)


def cellule1(taille_px: int, *, n_appels: int, n_chauffe: int) -> dict:
    """Cellule 1 du prereg : `rendu_r1` numpy sur champ réel, à la taille de
    session. Aucun GPU."""
    champ = champ_reel()
    mesure = chronometre(lambda: rendu_r1(champ, taille_px),
                         n_appels=n_appels, n_chauffe=n_chauffe)
    return dict(objet="rendu_r1 numpy (champ reel h_s101/s_L10)", taille_px=taille_px,
                forme_entree=list(champ.shape), bande_ms=list(BANDE_CELLULE1_MS),
                verdict_bande=classe_bande(mesure["mediane_ms"], BANDE_CELLULE1_MS),
                lecture=lecture_cellule1(mesure["mediane_ms"]), **mesure)


# --- Cellule 2 : la sonde GPU ------------------------------------------------


# FORMULES RECOPIÉES de `src/arcC_rendu.py`, empreinte attendue
#   c5ac875798d7cdb770702e9d612683ab3448ec8556fa2e81b3d902d426e4fe10
# La recopie est IMPOSÉE par le prereg : la sonde travaille en f32 (échelle V4)
# alors que R1 exige du float64 (sa garde de re-jouabilité). Si l'empreinte
# ci-dessus n'est plus celle du JSON produit, cette recopie doit être
# re-confrontée AVANT toute lecture.
def etages_srgb_quantif_cupy(cp, champ_f32):
    """Étages 3 et 4 de R1 en f32 sur GPU : inverse-EOTF sRGB puis
    quantification uint8. `Y = A` (étage 2, identité) est déjà appliqué du fait
    que l'entrée EST l'albédo. PAS de rééchantillonnage : la composition est
    exclue de P2 (D-P1-1)."""
    seuil = np.float32(arcC_rendu.SEUIL_LINEAIRE_SRGB)
    pente = np.float32(arcC_rendu.PENTE_LINEAIRE_SRGB)
    gain = np.float32(arcC_rendu.GAIN_SRGB)
    decalage = np.float32(arcC_rendu.DECALAGE_SRGB)
    inv_gamma = np.float32(1.0 / arcC_rendu.GAMMA_SRGB)
    puissance = gain * cp.power(cp.maximum(champ_f32, np.float32(0.0)), inv_gamma) - decalage
    v = cp.where(champ_f32 <= seuil, pente * champ_f32, puissance)
    return cp.rint(v * np.float32(arcC_rendu.NIVEAU_MAX_UINT8)).astype(cp.uint8), v


def champ_equivalence() -> np.ndarray:
    """Champ test de l'équivalence : une RAMPE `linspace(0, 1)` sur 1920x1080.
    Choix nommé — elle couvre la bande entière, les deux extrémités exactes
    (0.0 et 1.0) ET la branche linéaire de sRGB (~6500 points sous 0.0031308),
    ce qu'un tirage aléatoire ne garantit pas. Déterministe par construction."""
    total = FORME_V4[0] * FORME_V4[1]
    return np.linspace(0.0, 1.0, total, dtype=np.float64).reshape(FORME_V4)


def verifie_equivalence(cp) -> dict:
    """ÉQUIVALENCE D'ABORD, TOLÉRANCE ZÉRO (prereg cellule 2) : la sonde f32
    doit rendre les MÊMES uint8 que la chaîne numpy f64 de R1 sur le champ
    test. L'arrondi f32/f64 de la puissance 1/2.4 peut casser le bit-exact —
    si c'est le cas on le REMONTE CHIFFRÉ (combien de pixels, de combien),
    jamais on ne le tolère en silence.

    Contrôle adjoint : les valeurs encodées f32 restent dans [0, 1]. Hors
    bande, `astype(uint8)` enroulerait silencieusement (255.4 -> 255 mais
    256.0 -> 0) — c'est exactement la classe de défaut que la garde fail-loud
    de R1 refuse, et la sonde ne doit pas la ré-ouvrir."""
    champ = champ_equivalence()
    encode64 = srgb_encode(champ)
    attendu = quantifie_uint8(encode64)
    obtenu_gpu, encode_gpu = etages_srgb_quantif_cupy(cp, cp.asarray(champ, dtype=cp.float32))
    obtenu = cp.asnumpy(obtenu_gpu)
    v_min = float(cp.asnumpy(encode_gpu.min()))
    v_max = float(cp.asnumpy(encode_gpu.max()))

    differents = attendu != obtenu
    n_differents = int(differents.sum())
    ecart_max = int(np.abs(attendu.astype(np.int16) - obtenu.astype(np.int16)).max())

    # Rend l'écart LISIBLE plutôt que seulement compté : distance des pixels en
    # désaccord au point de bascule de `np.rint` (k + 0.5). Un désaccord collé à
    # la bascule est le désaccord MINIMAL possible entre deux arithmétiques —
    # ce n'est pas une excuse, c'est le chiffre qui permet à Romain de lire
    # « arrondi f32/f64 » plutôt que « formule différente ».
    echelle = encode64 * float(arcC_rendu.NIVEAU_MAX_UINT8)
    distances = np.abs(echelle - (np.floor(echelle) + 0.5))[differents]
    return dict(
        equivalent=bool(n_differents == 0 and 0.0 <= v_min and v_max <= 1.0),
        n_pixels_total=int(attendu.size), n_pixels_differents=n_differents,
        fraction_differents=n_differents / attendu.size, ecart_max_niveaux=ecart_max,
        distance_max_a_la_bascule=(float(distances.max()) if n_differents else None),
        encode_f32_min=v_min, encode_f32_max=v_max, tolerance="ZERO (prereg cellule 2)")


def cellule2(cp, *, n_appels: int, n_chauffe: int) -> dict:
    """Cellule 2 du prereg. L'équivalence GATE la mesure : si elle échoue, on
    n'entre PAS dans le chrono — un chiffre sur une sonde qui ne calcule pas la
    même chose que R1 ne mesurerait rien d'attribuable."""
    equivalence = verifie_equivalence(cp)
    base = dict(objet="etages sRGB + quantification uint8 (Y = A), f32, GPU",
                forme=list(FORME_V4), n_pixels=FORME_V4[0] * FORME_V4[1],
                bande_ms=list(BANDE_CELLULE2_MS), equivalence=equivalence)
    if not equivalence["equivalent"]:
        return dict(base, verdict_bande=VERDICT_AUTRE, lecture=dict(
            branche=None, texte=(
                "EQUIVALENCE ECHOUEE : la sonde f32 ne reproduit pas les uint8 de R1. "
                "Le chrono n'a PAS ete lance -- un chiffre sur une sonde qui ne calcule "
                "pas la meme chose que R1 ne mesure rien d'attribuable.")))

    champ = cp.asarray(np.random.default_rng(SEED_SOURCE).uniform(0.0, 1.0, size=FORME_V4),
                       dtype=cp.float32)
    peripherique = cp.cuda.Device()
    mesure = chronometre(lambda: etages_srgb_quantif_cupy(cp, champ),
                         n_appels=n_appels, n_chauffe=n_chauffe,
                         synchronise=peripherique.synchronize)
    return dict(base, verdict_bande=classe_bande(mesure["mediane_ms"], BANDE_CELLULE2_MS),
                lecture=lecture_cellule2(mesure["mediane_ms"]), **mesure)


# --- Provenance et CLI -------------------------------------------------------


def empreinte_r1() -> dict:
    """Empreinte de la SOURCE de `src/arcC_rendu.py`, recalculée à CHAQUE run —
    le chiffrage porte sur un objet daté (§A37 : « P3 date son verdict sur
    R1 » ; P2 chiffre le même objet)."""
    octets = Path(arcC_rendu.__file__).read_bytes()
    return dict(sha256=hashlib.sha256(octets).hexdigest(), octets=len(octets))


def _versions() -> dict:
    versions = dict(python=platform.python_version(), numpy=np.__version__, cupy=None)
    try:
        import cupy
        versions["cupy"] = cupy.__version__
    except Exception:
        pass
    return versions


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cellule", choices=["1", "2", "toutes"], default="toutes")
    parser.add_argument("--n-appels", type=int, default=N_APPELS_GRAVE,
                        help=f"GRAVÉ = {N_APPELS_GRAVE}. Le baisser produit un run NON "
                             "verdict-grade, marqué comme tel dans le JSON.")
    parser.add_argument("--n-chauffe", type=int, default=N_CHAUFFE_GRAVE,
                        help=f"GRAVÉ = {N_CHAUFFE_GRAVE}.")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "arcC" / "p2_chiffrage.json")
    # Géométrie d'écran (§C7) — mesurées physiquement, PAS de défaut inventé.
    parser.add_argument("--long-ref-px", type=float, required=True)
    parser.add_argument("--long-ref-mm", type=float, required=True)
    parser.add_argument("--distance-mm", type=float, required=True)
    parser.add_argument("--c-deg-cible", type=float, default=C_DEG_CIBLE_DEFAUT)
    parser.add_argument("--porteuse-cyc-par-domaine", type=float,
                        default=PORTEUSE_CYC_PAR_DOMAINE_DEFAUT)
    args = parser.parse_args()

    ppd = pixels_par_degre(args.long_ref_px, args.long_ref_mm, args.distance_mm)
    taille_px = taille_domaine_px(ppd, args.porteuse_cyc_par_domaine, args.c_deg_cible)
    obs = observation_cellule_pic_csf(ppd, porteuse_cyc_par_domaine=args.porteuse_cyc_par_domaine,
                                      c_deg_cible=args.c_deg_cible)

    rapport = dict(
        prereg="claude/prereg-p2-chiffrage-rendu.md (ENDOSSE, §A38)",
        date=datetime.now().isoformat(), machine=platform.node(), versions=_versions(),
        empreinte_r1=empreinte_r1(),
        verdict_grade=bool(args.n_appels == N_APPELS_GRAVE
                           and args.n_chauffe == N_CHAUFFE_GRAVE),
        calibration=dict(ppd=ppd, taille_domaine_px=taille_px,
                         observation_cellule_pic_csf=obs),
        portee=("P2 chiffre le NOYAU, PAS la composition (D-P1-1) -- premiere ancre de la "
                "moitie manquante, jamais la moitie entiere."))

    print("=" * 78)
    print("P2 -- CHIFFRAGE DU NOYAU DE RENDU")
    print("=" * 78)
    print(f"machine={rapport['machine']}  verdict_grade={rapport['verdict_grade']}  "
          f"(n_appels={args.n_appels}, chauffe={args.n_chauffe} ; gravé "
          f"{N_APPELS_GRAVE}/{N_CHAUFFE_GRAVE})")
    print(f"empreinte R1 = {rapport['empreinte_r1']['sha256']} "
          f"({rapport['empreinte_r1']['octets']} octets)")
    print(f"ppd={ppd:.3f}  taille_domaine_px={taille_px}")

    if args.cellule in ("1", "toutes"):
        c1 = cellule1(taille_px, n_appels=args.n_appels, n_chauffe=args.n_chauffe)
        rapport["cellule_1"] = c1
        print(f"\n[cellule 1] mediane={c1['mediane_ms']:.4f} ms  p95={c1['p95_ms']:.4f} ms  "
              f"bande={list(BANDE_CELLULE1_MS)} -> {c1['verdict_bande']}")
        print(f"            lecture (branche {c1['lecture']['branche']}) : "
              f"{c1['lecture']['texte']}")

    if args.cellule in ("2", "toutes"):
        if not cupy_disponible():
            rapport["cellule_2"] = dict(
                verdict_bande=VERDICT_AUTRE,
                lecture=dict(branche=None, texte=(
                    "CuPy/device CUDA indisponible -- AUCUNE mesure GPU. Le repli numpy "
                    "est interdit pour la mesure (B1) : un chiffre CPU serait un faux "
                    "chiffre. Machine attendue : iluin-tworings3, terminal natif.")))
            print("\n[cellule 2] CuPy indisponible -- aucune mesure, verdict AUTRE.")
        else:
            import cupy as cp
            c2 = cellule2(cp, n_appels=args.n_appels, n_chauffe=args.n_chauffe)
            rapport["cellule_2"] = c2
            eq = c2["equivalence"]
            print(f"\n[cellule 2] equivalence f32 vs numpy f64 : equivalent={eq['equivalent']} "
                  f"({eq['n_pixels_differents']}/{eq['n_pixels_total']} pixels differents, "
                  f"ecart max {eq['ecart_max_niveaux']} niveau(x), distance max a la "
                  f"bascule de np.rint = {eq['distance_max_a_la_bascule']} ; encode f32 "
                  f"dans [{eq['encode_f32_min']:.9f}, {eq['encode_f32_max']:.9f}])")
            if eq["equivalent"]:
                print(f"            mediane={c2['mediane_ms']:.4f} ms  "
                      f"p95={c2['p95_ms']:.4f} ms  bande={list(BANDE_CELLULE2_MS)} -> "
                      f"{c2['verdict_bande']}")
            print(f"            lecture (branche {c2['lecture']['branche']}) : "
                  f"{c2['lecture']['texte']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rapport, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[REPORT] -> {args.out}")
    print("POINT D'ARRET : la lecture versionnee "
          "(claude/lectures/p2_chiffrage_rendu.lecture.json) se recopie APRES le run "
          "verdict-grade de Romain, jamais par la session de build.")


if __name__ == "__main__":
    main()
