"""F1 — M-b TRANCHE-1 / C5 : driver du gate T1 (F fidèle à l'échelle V4).

Ce que ce driver mesure, et ce qu'il REPORTE. T1 est SÉPARABLE (§A27) : une
mesure de TEMPS, sans rederive (celui-ci est tranche-2). Le driver assemble
la frame complète V4 avec le F FIDÈLE (C1) + Exner cadencé (C2), la chronomètre
(B6), et prononce le verdict T1 : **médiane de la frame complète > 16,7 ⇒ V4
mort**.

DIAGNOSTIC PRINCIPAL, EN TÊTE (amendement A, §A29) : le RAPPORT DE
REPRÉSENTATIVITÉ **F_fidèle / F_proxy** (proxy = 12,655, l'ancre jetable).
Apples-to-apples : les DEUX sont réfléchissants, à l'échelle V4, sur les 15
blocs × 512². C'est la raison d'être originelle de T1 — le kernel figé
peut-il servir de proxy du moteur réel ? La bande vient au SECOND rang, parce
qu'elle est LARGE et peu falsifiable (§A29) ; c'est le verdict ABSOLU qui
porte la réserve de halo.

CADENCE EXNER DÉRIVÉE (§A29-C1) : un Exner tous les `CADENCE_EXNER` = 2 pas
(save_every du rederive, non choisie). Le poste Exner de la bande est amorti
d'autant.

RÉGIME DÛ AVANT LECTURE (§A28 caveat + amendement B) : la série doit exercer
le régime (fronts wet/dry, marge CFL 4–12×). `exiger_regime_pass` refuse
fail-loud toute lecture sans PASS — un chrono hors régime ne vaut rien.

BORD RÉFLÉCHISSANT en tranche-1 (mur) ; le halo-parent est tranche-2. La
réserve de halo est reportée à part (§A29-C1 : 0,4–1,0 ms en halo-parent).

Kernels FIGÉS et C1/C2/C4 intouchés ; k reste 4 ; rederive NON appelé.

Machine : iluin-tworings3, terminal natif UNIQUEMENT. Usage :
  .venv/bin/python scripts/run_f1_mb_t1.py

POINT D'ARRÊT OBLIGATOIRE AVANT LE RUN (§A29-C1) : ce driver est LIVRÉ mais
non exécuté tant que la bande re-gelée et les choix ne sont pas endossés."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_f1_ma_ter import VERIFS_EXIGEES, liberer_vram  # noqa: E402
from src.f1_gpu.backend import exiger_cupy  # noqa: E402
from src.f1_gpu.chrono import (  # noqa: E402
    SERIE_FRAMES,
    WARMUP_FRAMES,
    chronometrer_frames,
)
from src.f1_gpu.exner_gpu import CADENCE_EXNER, cadence_exner_derivee, pas_exner  # noqa: E402
from src.f1_gpu.regime_fidele import (  # noqa: E402
    DT_FRAME,
    etat_synthetise,
    exiger_regime_pass,
    mesurer_regime,
    verifier_regime_serie,
)
from src.f1_gpu.substrat_fidele import (  # noqa: E402
    MODE_REFLECHISSANT,
    SHA256_SOURCE_FIDELE,
    pas_f_fidele,
)
from src.f1_gpu.verifs import exiger_pass, vram_nvidia_smi_octets  # noqa: E402

OUT_DIR = ROOT / "outputs" / "f1"
OUT_JSON_PATH = OUT_DIR / "mb_t1.json"

# ── Config exacte, GRAVÉE avant tout run ─────────────────────────────────
N_FOV: int = 512                        # échelle V4
N_BLOCS: int = 15                       # blocs actifs V4 (= Σ n_systemes)
F_PROXY_MS: float = 12.655              # ancre jetable, PROXY (§A28)
SEUIL_MORT_MS: float = 16.7             # T2 inchangé
REMONTEE_L3_AMORTIE_MS: float = 1.280   # borne L3 / k=4, ancre fixe
TRANSFERTS_MS: float = 0.110            # à EPS 1e-2, ancre fixe
DERIVE_MACHINE_MS: float = 0.18         # bruit connu

# ── BANDE T1 PRÉ-ENREGISTRÉE, RE-GELÉE (§A25/§A29) ───────────────────────
# Composée de postes bornés AVANT le run. Mise à jour depuis §A28 pour :
#  - Exner amorti par la cadence DÉRIVÉE (÷2) : [1,5;3,0] -> [0,75;1,5] ;
#  - le reste inchangé. Re-gelée ici, plus jamais recomposée après coup.
POSTE_F_FIDELE: tuple[float, float] = (10.8, 15.2)   # proxy ×[0,85;1,2]
POSTE_EXNER_AMORTI: tuple[float, float] = (0.75, 1.5)  # /CADENCE_EXNER
BANDE_T1_MS: tuple[float, float] = (
    POSTE_F_FIDELE[0] + POSTE_EXNER_AMORTI[0] + REMONTEE_L3_AMORTIE_MS
    + TRANSFERTS_MS - DERIVE_MACHINE_MS,
    POSTE_F_FIDELE[1] + POSTE_EXNER_AMORTI[1] + REMONTEE_L3_AMORTIE_MS
    + TRANSFERTS_MS + DERIVE_MACHINE_MS,
)                                                     # ≈ (12,76 ; 18,27)

# Réserve de halo, à part (§A29-C1) : le bord réfléchissant sous-compte le
# halo-parent de tranche-2. Le seuil de réserve GLISSE à 16,7 − haut = 15,7.
RESERVE_HALO_MS: tuple[float, float] = (0.4, 1.0)
SEUIL_RESERVE_MS: float = SEUIL_MORT_MS - RESERVE_HALO_MS[1]   # 15,7

# VERDICT À TROIS VALEURS PRÉ-ÉCRITES (§A29-C2/C4/C5) — le LABEL vit DANS le
# champ verdict, pas à côté : la réserve était reportée à côté, l'info y
# était mais pas le cadrage. C'est le cadrage qui rend un chiffre lisible en
# diagonale (le 16,589 de M-a-quater comme succès).
LABEL_MORT: str = "MORT"
LABEL_SOUS_RESERVE: str = "SANS MORT SOUS RÉSERVE DE HALO"
LABEL_SANS_MORT: str = "SANS MORT"


def verdict_t1(mediane_ms: float) -> str:
    """Les trois branches gravées, appliquées SANS interprétation :
      médiane > 16,7            ⇒ MORT
      médiane ∈ [15,7 ; 16,7]   ⇒ SANS MORT SOUS RÉSERVE DE HALO
      médiane < 15,7            ⇒ SANS MORT."""
    if mediane_ms > SEUIL_MORT_MS:
        return LABEL_MORT
    if mediane_ms >= SEUIL_RESERVE_MS:
        return LABEL_SOUS_RESERVE
    return LABEL_SANS_MORT


def etat_synthetise_v4(cp, graine: int = 101):
    """État à l'échelle V4 : 15 blocs × 512², chacun exerçant le régime
    (fronts wet/dry, CFL sollicitée). Batché en (N_BLOCS, 1, 3, n, n)."""
    qs, bs = [], []
    for i in range(N_BLOCS):
        q, b = etat_synthetise(N_FOV, cp, graine=graine + i)
        qs.append(q)
        bs.append(b)
    return cp.concatenate(qs, axis=0), cp.concatenate(bs, axis=0)


def _passe_regime(cp, q0, b, frames: int) -> list[dict]:
    """Passe RÉGIME (hors chrono) : évolue l'état par le F fidèle et mesure
    la marge CFL À CHAQUE frame (amendement B — sur la série). Séparée du
    chrono pour ne pas y injecter la synchro de la réduction."""
    q = q0.copy()
    tampon = cp.empty_like(q)
    serie = [mesurer_regime(q, b, cp, DT_FRAME)]
    for _ in range(frames):
        q, _ = pas_f_fidele(q, b, cp, dt=DT_FRAME, sortie=q,
                            tampon_etage=tampon, mode=MODE_REFLECHISSANT)
        serie.append(mesurer_regime(q, b, cp, DT_FRAME))
    return serie


def _chrono_f_fidele(cp, q0, b) -> dict:
    """Chrono B6 du F FIDÈLE seul (bord réfléchissant) — c'est lui qui donne
    le rapport de représentativité F_fidèle/F_proxy, apples-to-apples."""
    q = q0.copy()
    tampon = cp.empty_like(q)

    def frame() -> None:
        nonlocal q
        q, _ = pas_f_fidele(q, b, cp, dt=DT_FRAME, sortie=q,
                            tampon_etage=tampon, mode=MODE_REFLECHISSANT)

    return chronometrer_frames(cp, frame)["stats"]


def _chrono_frame_complete(cp, q0, b) -> dict:
    """Chrono B6 de la frame complète MESURÉE : F fidèle + Exner CADENCÉ
    (un tous les CADENCE_EXNER pas). La remontée L3 et les transferts sont
    des ancres DÉRIVÉES (série de sondes close), ajoutées à la lecture."""
    q = q0.copy()
    tampon = cp.empty_like(q)
    s = cp.zeros((q.shape[0], q.shape[1], q.shape[-1], q.shape[-1]),
                 dtype=cp.float32)
    compteur = {"n": 0}

    def frame() -> None:
        nonlocal q
        q, _ = pas_f_fidele(q, b, cp, dt=DT_FRAME, sortie=q,
                            tampon_etage=tampon, mode=MODE_REFLECHISSANT)
        compteur["n"] += 1
        if compteur["n"] % CADENCE_EXNER == 0:
            pas_exner(q, s, cp, DT_FRAME)

    return chronometrer_frames(cp, frame)["stats"]


def lecture_t1(f_fidele: dict, frame_complete: dict,
               regime: dict) -> dict:
    """Lecture MÉCANIQUE de T1. DIAGNOSTIC PRINCIPAL EN TÊTE (amendement A) :
    F_fidèle/F_proxy. Bande au SECOND rang. Verdict absolu + réserve de halo.
    REFUSE fail-loud sans le PASS de régime (B9)."""
    exiger_regime_pass(regime)
    f_fidele_med = f_fidele["mediane_ms"]
    mediane = frame_complete["mediane_ms"]
    p99 = frame_complete["p99_ms"]
    # la frame MESURÉE (F + Exner cadencé) + ancres dérivées fixes
    mediane_complete = mediane + REMONTEE_L3_AMORTIE_MS + TRANSFERTS_MS
    bas, haut = BANDE_T1_MS
    return {
        # ── EN TÊTE : le rapport de représentativité ──
        "representativite": {
            "f_fidele_ms": f_fidele_med,
            "f_proxy_ms": F_PROXY_MS,
            "rapport_f_fidele_sur_proxy": f_fidele_med / F_PROXY_MS,
            "note": ("apples-to-apples de COÛT : F fidèle et proxy tous deux "
                     "RÉFLÉCHISSANTS, échelle V4, 15 blocs × 512². Rapport "
                     "~1 ⇒ le figé était un proxy de COÛT fidèle ; > 1 ⇒ le "
                     "moteur réel coûte plus. C'est la raison d'être "
                     "originelle de T1 (§A29 amendement A)."),
            "portee": (
                "PORTÉE, à porter AVEC le rapport : ≈1 dit qu'ils COÛTENT "
                "pareil, PAS qu'ils FONT pareil. Le proxy transporte 8 "
                "champs à fond PLAT (E4a jetable) ; le fidèle évolue 3 "
                "champs + reconstruction b-aware d'Audusse à terrain réel. "
                "Un rapport proche de 1 NE VALIDE PAS E4a — ce sont deux "
                "physiques différentes de coût voisin."),
        },
        # ── VERDICT T1 : trois valeurs pré-écrites DANS le champ ──
        "t1": {
            "verdict": verdict_t1(mediane_complete),
            "mediane_frame_complete_ms": mediane_complete,
            "mediane_mesuree_f_plus_exner_ms": mediane,
            "seuil_mort_ms": SEUIL_MORT_MS,
            "seuil_reserve_ms": SEUIL_RESERVE_MS,
            "mort": bool(mediane_complete > SEUIL_MORT_MS),
            "branches_preecrites": {
                LABEL_MORT: "médiane > 16,7 ⇒ le repli 33.3 devient LA "
                            "décision",
                LABEL_SOUS_RESERVE: "médiane ∈ [15,7 ; 16,7] ⇒ le "
                                    "halo-parent de tranche-2 peut la porter "
                                    "au-dessus de 16,7",
                LABEL_SANS_MORT: "médiane < 15,7 ⇒ marge même sous la "
                                 "réserve de halo",
            },
        },
        "p99_en_evidence": {
            "valeur_ms": p99,
            "ecart_a_la_mediane_ms": p99 - mediane,
            "note": ("REPORTÉE EN ÉVIDENCE (§A25) : le stutter suspendu au "
                     "régime. Un pas de F bursty se lirait ici."),
        },
        # ── RÉSERVE DE HALO (§A29-C1), à part ──
        "reserve_halo": {
            "reserve_ms": list(RESERVE_HALO_MS),
            "seuil_glissant_ms": SEUIL_RESERVE_MS,
            "verdict_sous_reserve": (
                "si la médiane dépasse 15,7, le halo-parent de tranche-2 "
                "(+0,4–1,0 ms) peut à lui seul la porter au-dessus de 16,7"),
            "mediane_plus_reserve_haute_ms": mediane_complete
            + RESERVE_HALO_MS[1],
            "mort_sous_reserve": bool(
                mediane_complete + RESERVE_HALO_MS[1] > SEUIL_MORT_MS),
        },
        # ── BANDE, au SECOND rang ──
        "bande_modele": {
            "bande_ms": [bas, haut],
            "hors_bande": bool(not bas <= mediane_complete <= haut),
            "postes": {
                "f_fidele": list(POSTE_F_FIDELE),
                "exner_amorti": list(POSTE_EXNER_AMORTI),
                "remontee_l3_amortie": REMONTEE_L3_AMORTIE_MS,
                "transferts": TRANSFERTS_MS,
                "derive_machine": DERIVE_MACHINE_MS,
            },
            "note": ("LARGE et peu falsifiable (§A29) — au second rang. Hors "
                     "bande ⇒ AUTRE (modèle de coût faux), distinct du "
                     "verdict de mort. Exner amorti par la cadence DÉRIVÉE "
                     f"÷{CADENCE_EXNER}."),
            "cadence_exner": cadence_exner_derivee(),
        },
        "regime": regime,
        "rappel_portee": (
            "T1 REPORTE une mesure de TEMPS. La fidélité (MORT-b) est "
            "tranche-2 ; le rederive n'est pas appelé. La décision — V4-b "
            "vit ou le repli 33.3 — revient à Romain."),
    }


def main() -> None:
    cp = exiger_cupy()
    exiger_pass(VERIFS_EXIGEES)
    print(f"M-b T1 : échelle V4 {N_BLOCS} blocs × {N_FOV}², bord "
          f"réfléchissant, Exner 1/{CADENCE_EXNER}, dt=temps de frame "
          f"({DT_FRAME*1000:.3f} ms). F fidèle empreinte "
          f"{SHA256_SOURCE_FIDELE[:8]}...", flush=True)

    q0, b = etat_synthetise_v4(cp)

    print("  passe RÉGIME (sur la série, amendement B) ...", flush=True)
    serie = _passe_regime(cp, q0, b, SERIE_FRAMES)
    regime = verifier_regime_serie(serie)
    exiger_regime_pass(regime)          # fail-loud AVANT tout chrono lu
    liberer_vram(cp)

    print("  chrono F FIDÈLE seul (représentativité) ...", flush=True)
    f_fidele = _chrono_f_fidele(cp, q0, b)
    liberer_vram(cp)

    print("  chrono FRAME COMPLÈTE (F + Exner cadencé) ...", flush=True)
    frame_complete = _chrono_frame_complete(cp, q0, b)
    liberer_vram(cp)

    lecture = lecture_t1(f_fidele, frame_complete, regime)
    mempool = cp.get_default_memory_pool()
    document = {
        "meta": {
            "mesure": ("M-b tranche-1 / T1 — F fidèle à l'échelle V4 "
                       "(§A29-C1, pocCascade2phys c7df35c)"),
            "config": {"n_fov": N_FOV, "n_blocs": N_BLOCS,
                       "bord": "réfléchissant (mur, tranche-1)",
                       "cadence_exner": CADENCE_EXNER,
                       "dt_frame_ms": DT_FRAME * 1000},
            "kernels": ("F fidèle _SOURCE_FIDELE 6dd207ca (C1) INTOUCHÉ ; "
                        "figés e18015f5/9533a130 intouchés ; Exner 3533fd0b"),
            "frames": {"warmup_exclu": WARMUP_FRAMES, "serie": SERIE_FRAMES},
            "verifs_exigees": list(VERIFS_EXIGEES),
        },
        "chrono_f_fidele": f_fidele,
        "chrono_frame_complete": frame_complete,
        "residence": {
            "mempool_total_octets": int(mempool.total_bytes()),
            "nvidia_smi_octets": vram_nvidia_smi_octets(),
        },
        "lecture_mecanique": lecture,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON_PATH.write_text(
        json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")

    rep = lecture["representativite"]
    t1 = lecture["t1"]
    print("=" * 78)
    print("F1 -- M-b TRANCHE-1 / T1 (lecture mécanique)")
    print("=" * 78)
    print(f"  [EN TÊTE] F_fidèle/F_proxy = {rep['f_fidele_ms']:.3f} / "
          f"{rep['f_proxy_ms']} = {rep['rapport_f_fidele_sur_proxy']:.3f}")
    print(f"    portée : {rep['portee']}")
    print(f"  T1 : médiane frame complète = "
          f"{t1['mediane_frame_complete_ms']:.3f} ms")
    print(f"    VERDICT (trois valeurs pré-écrites) : {t1['verdict']}")
    print(f"  p99 (évidence) = {lecture['p99_en_evidence']['valeur_ms']:.3f} ms")
    rh = lecture["reserve_halo"]
    print(f"  réserve halo : médiane + {RESERVE_HALO_MS[1]} = "
          f"{rh['mediane_plus_reserve_haute_ms']:.3f}  -> mort sous réserve = "
          f"{rh['mort_sous_reserve']}")
    print(f"  [second rang] bande {BANDE_T1_MS} -> hors bande = "
          f"{lecture['bande_modele']['hors_bande']}")
    print("  Le driver REPORTE — la décision revient à Romain.")
    print(f"[REPORT] -> {OUT_JSON_PATH}")
    print("POINT D'ARRÊT OBLIGATOIRE : lecture remontée à Romain.")


if __name__ == "__main__":
    main()
