"""F1 — BORNE L3 de la remontée (achat A, §A18-complément / -2,
pocCascade2phys d720da6).

MESURE JETABLE, DITE TELLE QUELLE. Cette borne mesure un motif de COÛT —
ce que coûte la remontée quand elle est émise par le kernel — comme M-a′
borna F. Elle n'est pas un composant de moteur. Le portage fidèle reste
non acheté, et le CAP GRAVÉ tient : **L3 est la DERNIÈRE escalade sur la
remontée**.

CE QUE CETTE BORNE N'EST PAS : ce n'est PAS M-a-quater. La géométrie est
INTOUCHÉE (V2 actuelle, offsets B3), il n'y a ni emboîtement, ni champs
d'échelle, ni prédiction — l'achat B est gaté sur la lecture de celle-ci.
La fovéa est IMMOBILE : c'est la remontée qu'on isole, pas le
déplacement.

LES DEUX BRAS (même session, même état initial, même structure
d'allocation — les références sont allouées dans LES DEUX, sans quoi la
différence mesurerait aussi une différence de résidence) :
  - **REF** : kernel FUSIONNÉ M-a′, INTOUCHÉ (bloc _SOURCE à l'empreinte
    e8fcaad47db0af010728d5c5d878e1629191bfef4e0382fa80aa3cefd9f27f04).
    Mesure F seul. Sert AUSSI de contrôle de dérive machine face aux
    14.432 ms gravés en s2.
  - **L3** : kernel L3 (F + épilogue d'émission) + compaction par
    agrégation de warp + transfert des coefficients.

LECTURE MÉCANIQUE : **remontée_L3 = A_L3 − A_REF**, comparée aux
**8.523 ms** du schéma CuPy actuel et à la **cible ~1.5 ms** — les trois
reportées SANS INTERPRÉTATION. La décision d'ouvrir ou non l'achat B
revient à Romain ; si la borne ne descend pas vers la cible, V4 meurt
pour un achat au lieu de trois et le repli 33.3 devient LA décision.

DIAGNOSTICS REPORTÉS À CÔTÉ, tous non verdictaux :
  - **écart émis / transférés** — le retard d'une frame (choix 6) peut
    laisser des coefficients pour le tour suivant ; le schéma B4 étant
    incrémental, rien n'est perdu, mais l'écart est rendu lisible ;
  - **dérive machine** — A_REF contre les 14.432 ms gravés ;
  - **écart sec / mouillé** — vérification ANTI-MINORATION exigée au
    protocole : un système entièrement nul est une fenêtre sèche, toutes
    les branches d'un warp partent du même côté, et l'absence de
    divergence peut le rendre moins cher. Le deuxième système reste
    MOUILLÉ dans les deux bras (majorant E4a reconduit) ; cette
    micro-mesure chiffre ce que coûterait de le laisser sec, pour la
    lecture de l'achat B où les champs d'échelle seront nuls.

Chrono B6, vérifs #1/#2 reconduites, natif. Sortie JSON SANS timestamp.

Machine : iluin-tworings3, terminal natif UNIQUEMENT. Usage :
  .venv/bin/python scripts/run_f1_borne_l3.py

POINT D'ARRÊT OBLIGATOIRE après le run : lecture remontée à Romain,
aucun verdict, aucun enchaînement — l'achat B est une décision."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_f1_ma_ter import (  # noqa: E402
    B_FRAME_MS,
    N_NIV,
    VERIFS_EXIGEES,
    liberer_vram,
    slots_v2,
)
from scripts.run_f1_s1_ancre_1sys import N_FOV, etat_mono_systeme  # noqa: E402
from scripts.run_f1_s2_batche import (  # noqa: E402
    LANCEMENTS_PAR_FRAME,
    pas_f_fusionne_async,
    reduction_cfl_on_device,
)
from scripts.run_f1_s2_mobile import plan_groupes  # noqa: E402
from src.f1_gpu.backend import exiger_cupy  # noqa: E402
from src.f1_gpu.chrono import (  # noqa: E402
    SERIE_FRAMES,
    WARMUP_FRAMES,
    chronometrer_frames,
)
from src.f1_gpu.pyramide import EPS_DETAIL, GRAINE_MONDE, GeometriePyramide  # noqa: E402
from src.f1_gpu.substrat_jetable import etat_initial_jetable  # noqa: E402
from src.f1_gpu.substrat_l3 import (  # noqa: E402
    CompacteurL3,
    pas_f_l3,
    taille_sortie_max,
)
from src.f1_gpu.transferts import TransfertComptable  # noqa: E402
from src.f1_gpu.verifs import exiger_pass, vram_nvidia_smi_octets  # noqa: E402

OUT_DIR = ROOT / "outputs" / "f1"
OUT_JSON_PATH = OUT_DIR / "borne_l3.json"

# Chiffres GRAVÉS — consommés en lecture mécanique SEULEMENT.
REMONTEE_CUPY_MS: float = 8.523      # schéma B4 actuel (§A16-lecture-sonde)
CIBLE_REMONTEE_MS: float = 1.5       # « ~1.5 » (§A18-complément)
A_BATCHE_GRAVE_MS: float = 14.432    # s2, contrôle de dérive machine


class BrasBorne:
    """Un bras : les buffers groupés de V2 (17 blocs, 2 groupes) + les
    RÉFÉRENCES, allouées dans les DEUX bras — on retire du travail,
    jamais de la structure. Fovéa immobile, aucune prédiction."""

    def __init__(self, cp, geo: GeometriePyramide, avec_emission: bool,
                 transferts: TransfertComptable | None = None,
                 graine: int = GRAINE_MONDE, eps: float = EPS_DETAIL):
        self.cp = cp
        self.geo = geo
        self.avec_emission = bool(avec_emission)
        self.transferts = transferts
        self.eps = float(eps)
        self.plans = plan_groupes(geo)
        self.fenetres: list = []
        self.tampons: list = []
        self.references: list = []
        for plan in self.plans:
            if plan["n_systemes"] == 2:
                bloc = cp.asarray(etat_initial_jetable(
                    plan["n_slots"], geo.n_fov, graine))
            else:
                bloc = etat_mono_systeme(cp, plan["n_slots"], geo.n_fov)
            self.fenetres.append(bloc)
            self.tampons.append(cp.empty_like(bloc))
            # Référence légèrement décalée : sans écart, rien n'émettrait
            # et la borne mesurerait une remontée vide.
            self.references.append(bloc * cp.float32(0.999))
        self.compacteurs = [
            CompacteurL3(cp, taille_sortie_max(bloc))
            for bloc in self.fenetres] if avec_emission else []
        self.dt_cfl: list = [None] * len(self.plans)

    def frame(self) -> None:
        """Une frame : F sur chaque groupe (4 lancements), plus — pour le
        bras L3 — le transfert des coefficients dimensionné par le
        compteur de la frame PRÉCÉDENTE, puis la clôture du compteur."""
        for indice, (fen, tampon) in enumerate(
                zip(self.fenetres, self.tampons)):
            if self.avec_emission:
                compacteur = self.compacteurs[indice]
                taille = compacteur.taille_precedente()
                compacteur.noter_taille(taille)
                _, dt_cfl = pas_f_l3(
                    fen, self.cp, self.references[indice], compacteur,
                    sortie=fen, tampon_etage=tampon, eps=self.eps,
                    reduction=reduction_cfl_on_device)
                if taille:
                    valeurs, indices = compacteur.vues_a_transferer(taille)
                    self.transferts.remonter(valeurs)
                    self.transferts.remonter(indices)
                compacteur.cloturer_frame()
            else:
                _, dt_cfl = pas_f_fusionne_async(fen, self.cp, fen, tampon)
            self.dt_cfl[indice] = dt_cfl

    def relever_emissions(self) -> None:
        """Relève le compteur de la DERNIÈRE frame (celui qui n'a pas eu
        de frame suivante pour être transféré) — HORS chrono uniquement,
        c'est la seule lecture bloquante de tout le bras."""
        self.cp.cuda.runtime.deviceSynchronize()
        for compacteur in self.compacteurs:
            compacteur.compteur_final = int(compacteur.hote[0])

    def audit_cfl(self) -> dict:
        valeurs = [float(dt) if dt is not None else None
                   for dt in self.dt_cfl]
        return {"dt_cfl_par_groupe": valeurs,
                "toutes_finies_positives": bool(
                    all(v is not None and np.isfinite(v) and v > 0.0
                        for v in valeurs))}

    def octets_etat(self) -> int:
        total = sum(b.nbytes for b in self.fenetres)
        total += sum(b.nbytes for b in self.references)
        for compacteur in self.compacteurs:
            # Les DEUX jeux du ping-pong (§A21). NOTA : la borne L3 est
            # une mesure ACQUISE et n'est pas re-jouée ; cette ligne suit
            # le compacteur pour rester juste, elle ne recompte rien.
            total += compacteur.octets_buffers()
        return int(total)

    def n_blocs(self) -> int:
        return int(sum(b.shape[0] * b.shape[1] for b in self.fenetres))


def mesurer_bras(cp, nom: str, geo: GeometriePyramide,
                 avec_emission: bool) -> dict:
    """Un bras, chrono B6. Les transferts ne sont comptés que pour L3."""
    transferts = TransfertComptable(cp) if avec_emission else None
    bras = BrasBorne(cp, geo, avec_emission, transferts=transferts)

    if avec_emission:
        def frame() -> None:
            bras.frame()
            transferts.frame_suivante()
    else:
        frame = bras.frame

    chrono = chronometrer_frames(cp, frame)
    bras.relever_emissions()

    mempool = cp.get_default_memory_pool()
    resultat = {
        "nom": nom,
        "avec_emission": avec_emission,
        "n_blocs": bras.n_blocs(),
        "frames": {"warmup_exclu": WARMUP_FRAMES, "serie": SERIE_FRAMES},
        "frame_time": chrono["stats"],
        "audit_cfl": bras.audit_cfl(),
        "residence": {
            "mempool_used_octets": int(mempool.used_bytes()),
            "mempool_total_octets": int(mempool.total_bytes()),
            "etat_prealloue_octets": bras.octets_etat(),
            "nvidia_smi_octets": vram_nvidia_smi_octets(),
        },
    }
    if avec_emission:
        resultat["transferts_regime"] = transferts.stats_regime(
            exclure_premiers=WARMUP_FRAMES)
        resultat["retard_transfert"] = [c.diagnostic_retard()
                                        for c in bras.compacteurs]
    return resultat


def mesurer_ecart_sec_mouille(cp, geo: GeometriePyramide,
                              n_blocs: int = 3) -> dict:
    """Vérification ANTI-MINORATION (protocole §A18-complément, point 2).

    Un système entièrement nul est une fenêtre SÈCHE : `desing` rend 0, le
    plancher sec s'applique, et surtout toutes les branches d'un warp
    partent du même côté — l'absence de divergence peut rendre le calcul
    moins cher. Le harnais garde ses deux systèmes MOUILLÉS (majorant E4a
    reconduit) ; cette micro-mesure chiffre ce que coûterait de ne pas le
    faire, pour la lecture de l'achat B où les champs d'échelle seront
    nuls par définition. Reportée, jamais appliquée à la borne."""
    mouille = cp.asarray(etat_initial_jetable(n_blocs, geo.n_fov,
                                              GRAINE_MONDE))
    sec = mouille.copy()
    sec[:, 1] = 0.0            # deuxième système entièrement sec

    def chronometrer(etat):
        tampon = cp.empty_like(etat)
        travail = etat.copy()

        def frame() -> None:
            pas_f_fusionne_async(travail, cp, travail, tampon)

        return chronometrer_frames(cp, frame)["stats"]["mediane_ms"]

    ms_mouille = chronometrer(mouille)
    ms_sec = chronometrer(sec)
    return {
        "n_blocs": n_blocs,
        "mediane_mouille_ms": ms_mouille,
        "mediane_second_systeme_sec_ms": ms_sec,
        "ecart_relatif": (ms_sec - ms_mouille) / ms_mouille,
        "note": ("le harnais reste MOUILLÉ (majorant E4a) ; un écart "
                 "négatif significatif signifierait qu'un système nul "
                 "coûte moins — à retenir pour l'achat B, où les champs "
                 "d'échelle sont nuls par définition. Reporté, non "
                 "appliqué à la borne."),
    }


def lecture_mecanique(ref: dict, l3: dict, sec_mouille: dict) -> dict:
    """Consommation MÉCANIQUE — REPORTÉE, jamais interprétée."""
    a_ref = ref["frame_time"]["mediane_ms"]
    a_l3 = l3["frame_time"]["mediane_ms"]
    remontee = a_l3 - a_ref
    retards = l3.get("retard_transfert", [])
    return {
        "a_ref_ms": a_ref,
        "a_l3_ms": a_l3,
        "remontee_l3_ms": remontee,
        "comparaisons": {
            "remontee_cupy_gravee_ms": REMONTEE_CUPY_MS,
            "cible_ms": CIBLE_REMONTEE_MS,
            "facteur_vs_cupy": (REMONTEE_CUPY_MS / remontee
                                if remontee > 0 else None),
            "note": ("les trois grandeurs sont REPORTÉES côte à côte ; ce "
                     "driver ne conclut pas. L'ouverture de l'achat B est "
                     "une décision de Romain à la lecture."),
        },
        "derive_machine": {
            "a_ref_ms": a_ref,
            "a_batche_grave_ms": A_BATCHE_GRAVE_MS,
            "ecart_relatif": (a_ref - A_BATCHE_GRAVE_MS) / A_BATCHE_GRAVE_MS,
            "note": ("contrôle : le bras REF re-mesure F seul avec le "
                     "kernel intouché — un écart notable signalerait une "
                     "dérive de machine, pas un effet de L3"),
        },
        "retard_transfert": {
            "par_groupe": retards,
            "toutes_tailles_stables": bool(
                retards and all(r["tailles_stables"] for r in retards)),
            "note": ("retard d'une frame (choix 6). Le diagnostic utile "
                     "est la VARIABILITÉ des tailles, pas un cumul (qui "
                     "serait tautologique : chaque frame transfère ce que "
                     "la précédente a émis). Tailles stables => retard "
                     "neutre, seule la dernière frame reste en attente. "
                     "Le schéma B4 étant incrémental, rien n'est perdu."),
        },
        "ecart_sec_mouille": sec_mouille,
        "budget_restant": {
            "formule": f"{B_FRAME_MS} - A_l3",
            "valeur_ms": B_FRAME_MS - a_l3,
            "note": ("reporté tel quel, y compris négatif ; ce n'est PAS "
                     "M-a-quater — ni prédiction ni géométrie emboîtée "
                     "n'entrent dans ce chiffre"),
        },
        "rappel_portee": (
            "borne JETABLE de la remontée seule, géométrie intouchée, "
            "fovéa immobile. CAP : L3 est la dernière escalade sur la "
            "remontée. Aucun verdict, aucun enchaînement."),
    }


def main() -> None:
    cp = exiger_cupy()
    exiger_pass(VERIFS_EXIGEES)

    geo = GeometriePyramide(n_fov=N_FOV, n_niv=N_NIV, slots=slots_v2())
    bras: list[dict] = []
    for nom, emission in (("REF_fusionne", False), ("L3_emission", True)):
        print(f"bras {nom} ...", flush=True)
        bras.append(mesurer_bras(cp, nom, geo, emission))
        liberer_vram(cp)

    print("vérification anti-minoration (sec vs mouillé) ...", flush=True)
    sec_mouille = mesurer_ecart_sec_mouille(cp, geo)
    liberer_vram(cp)

    par_nom = {b["nom"]: b for b in bras}
    lecture = lecture_mecanique(par_nom["REF_fusionne"],
                                par_nom["L3_emission"], sec_mouille)

    document = {
        "meta": {
            "mesure": "borne L3 de la remontée (achat A, §A18-complément-2, "
                      "pocCascade2phys d720da6)",
            "nature": ("mesure JETABLE — borne d'un motif de coût, comme "
                       "M-a′ borna F ; PAS un composant de moteur"),
            "cap": "L3 est la DERNIÈRE escalade sur la remontée",
            "protocole": {
                "config": "V2 actuelle, géométrie INTOUCHÉE (offsets B3)",
                "fovea": "IMMOBILE — c'est la remontée qu'on isole",
                "bras_ref": ("kernel FUSIONNÉ M-a′ INTOUCHÉ, empreinte "
                             "e8fcaad4...7f04 — contrôle de dérive"),
                "bras_l3": ("kernel L3 : F + épilogue d'émission, "
                            "compaction par AGRÉGATION DE WARP (arbitrage "
                            "§A18-complément-2), transfert à retard d'une "
                            "frame"),
                "structure": ("références allouées dans les DEUX bras — on "
                              "retire du travail, jamais de la structure"),
                "eps_detail": EPS_DETAIL,
                "lancements_par_frame": LANCEMENTS_PAR_FRAME,
                "chrono": "B6 (cuda-Events, warmup exclu, médiane ET p99)",
            },
            "n_est_pas": ("M-a-quater : ni emboîtement, ni champs "
                          "d'échelle, ni prédiction — l'achat B est gaté "
                          "sur cette lecture"),
            "cellule": {"n_fov": N_FOV, "n_niv": N_NIV},
            "verifs_exigees": list(VERIFS_EXIGEES),
        },
        "bras": bras,
        "lecture_mecanique": lecture,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON_PATH.write_text(
        json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 78)
    print("F1 -- BORNE L3 de la remontée (lecture mécanique)")
    print("=" * 78)
    for b in bras:
        ft = b["frame_time"]
        print(f"  {b['nom']:<14} médiane={ft['mediane_ms']:7.3f} ms   "
              f"p99={ft['p99_ms']:7.3f} ms")
    print("-" * 78)
    print(f"  remontée_L3 = A_L3 - A_REF = "
          f"{lecture['remontee_l3_ms']:.3f} ms")
    print(f"    schéma CuPy gravé : {REMONTEE_CUPY_MS} ms   "
          f"cible : ~{CIBLE_REMONTEE_MS} ms")
    derive = lecture["derive_machine"]
    print(f"  dérive machine : A_REF={derive['a_ref_ms']:.3f} vs "
          f"{A_BATCHE_GRAVE_MS} gravés ({derive['ecart_relatif']:+.1%})")
    retard = lecture["retard_transfert"]
    tailles = [f"{r['taille_min']}-{r['taille_max']}"
               for r in retard["par_groupe"]]
    print(f"  retard d'une frame : tailles par groupe {tailles}  "
          f"stables={retard['toutes_tailles_stables']}")
    print(f"  sec vs mouillé : {sec_mouille['ecart_relatif']:+.1%} "
          f"[anti-minoration, reporté]")
    print("  Les grandeurs sont REPORTÉES SANS INTERPRÉTATION — "
          "l'ouverture de l'achat B est une décision de Romain.")
    print(f"[REPORT] -> {OUT_JSON_PATH}")
    print("POINT D'ARRÊT OBLIGATOIRE : lecture remontée à Romain, aucun "
          "verdict, aucun enchaînement automatique.")


if __name__ == "__main__":
    main()
