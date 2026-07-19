"""F1 — DIAGNOSTIC D'INSTRUMENT D-1 / D-2 (§A19-lecture-sonde-EPS,
pocCascade2phys 07ef6cc).

POURQUOI CE DIAGNOSTIC EXISTE. La sonde EPS a rendu la branche (ii) :
MUETTE. Le bras de vérification k=1 a en outre FALSIFIÉ l'explication que
ce harnais portait — « l'observable est gouverné par la péremption L1 » :
Δχ est identique à TROIS DÉCIMALES entre k=1 et k=4, et le Δχ non décimé
est tout aussi plat (≈0.069). Ni EPS ni k ne bougent l'aiguille : c'est un
**PLANCHER STRUCTUREL**.

CE QUI REND CE PLANCHER GATANT, et non une curiosité : M-b partage les
mêmes primitives de readout (albedo + delta_chi/max_carrier) et son seuil
de mort est **0.0733 par-seed**. Un plancher d'instrument à 0.069 siège à
6 % SOUS le seuil qui tue Option A. Acheter 3–5 séances de portage fidèle
avec ce plancher inexpliqué, ce serait risquer de prononcer MORT-b sur un
ARTEFACT. Le chiffrage de M-b est donc SUSPENDU à cette lecture.

────────────────────────────────────────────────────────────────────────
D-1 — D'OÙ VIENT LE PLANCHER
────────────────────────────────────────────────────────────────────────
**(a) TEST À BLANC** : on pose `vivant := vérité` et on mesure Δχ par le
chemin de lecture COMPLET (décimation, albedo, delta_chi). Attendu
pré-écrit : **Δχ = 0 EXACT**. Tout résidu serait un plancher de READOUT —
propriété des primitives elles-mêmes, donc TRANSFÉRABLE à M-b, qui les
partage. C'est un contrôle négatif : il ne peut pas « réussir mieux » que
zéro, il ne peut que révéler un défaut.

**(b) COMPARAISON RESTREINTE** : on ne compare que les cellules
EFFECTIVEMENT COUVERTES — celles qu'une fenêtre active a réellement
remontées. Les autres n'ont jamais rien reçu et divergent librement
depuis l'initialisation : elles produiraient un écart constant,
indépendant de tout réglage, ce qui est exactement la signature observée.
Mise en œuvre : les cellules non couvertes sont remplacées par la vérité,
donc leur écart est nul et seules les couvertes contribuent — la
structure spectrale est préservée (on ne peut pas faire de FFT sur un
sous-ensemble troué). Deux vues sont reportées : couverture du CYCLE
courant (k frames) et couverture CUMULÉE depuis le début.

Lectures PRÉ-ÉCRITES : si le plancher S'EFFONDRE une fois restreint, il
est un plancher de **COUVERTURE**, propre à cette sonde et NON
transférable à M-b. S'il PERSISTE, la couverture n'explique rien et le
plancher reste à attribuer. Le driver ne tire AUCUNE autre conclusion.
Le seuil d'effondrement n'est pas chiffré par le gravé : il est fixé ici
à la MOITIÉ de la comparaison complète (`FACTEUR_EFFONDREMENT`), AVANT le
run — un facteur ne se discute pas au cas par cas une fois la donnée vue.

Le mécanisme a été EXERCÉ au build sur une configuration RÉDUITE (PAS la
cellule de mesure) : test à blanc exactement nul, et comparaison
restreinte tombant de ~0.069 à ~0.004 pour une couverture de quelques
pour cent. Ces chiffres ne sont PAS un résultat et ne préjugent pas de la
cellule — ils disent seulement que le chemin de mesure fonctionne et que
la question est bien posée.

────────────────────────────────────────────────────────────────────────
D-2 — D'OÙ VIENNENT LES +0.66 ms
────────────────────────────────────────────────────────────────────────
À EPS=1e-4 la sonde a donné **17.249 ms** là où M-a-quater donnait
**16.589** : +0.66 ms, soit quatre fois la marge de V4, dans le mauvais
sens. Les transferts concordent (2.635 contre 2.727) — l'écart est
ailleurs. Deux causes possibles, et le contrôle T1 de M-b étant VERDICTAL
(Q3), il faut trancher AVANT.

Le bras re-mesure le chrono « nu » à configuration M-a-quater IDENTIQUE,
avec l'appareil de reconstruction explicitement HORS BOUCLE — et il le
PROUVE plutôt que de l'affirmer : `preuve_appareil_hors_boucle` vérifie,
sur le pipeline réellement chronométré, que la capture est désactivée et
qu'aucun coefficient n'a été retenu pendant toute la série.

Lectures PRÉ-ÉCRITES : écart ABSORBÉ ⇒ **APPAREIL** de sonde entrant dans
le chrono, sans conséquence sur V4 ; écart PERSISTANT ⇒ **TERME DE
PRODUCTION NON COMPTÉ** — la marge de V4 est entamée, et le contrôle T1
de M-b s'appliquera à ce total.

────────────────────────────────────────────────────────────────────────
PORTÉE, ET CE QUI EST INTERDIT ICI
────────────────────────────────────────────────────────────────────────
Diagnostic d'INSTRUMENT, pas une nouvelle tentative de régler EPS : la
branche (ii) est APPLIQUÉE, **rien n'est réglé**. EPS reste FIGÉ à 1e-4,
k reste FIGÉ, aucune décision de cadencement n'est prise ni suggérée. Le
driver refuse fail-loud toute autre valeur.

LEVIER VERROUILLÉ, rappelé pour qu'il ne soit pas saisi par inadvertance :
le balayage montre que passer de 1e-4 à 1e-2 rendrait ~2.5 ms de
transferts et ~1.8 ms de médiane — exactement ce qui manque à V4. C'est
INUTILISABLE : la non-discrimination signifie que l'observable ne VOIT
pas EPS, pas qu'il n'y a rien à voir. « Puisque rien n'est établi, prenons
le moins cher » serait du raisonnement motivé, explicitement refusé.

Kernels F et L3, chrono, substrats, pyramide : INTOUCHÉS. Coût : minutes,
harnais existant. Sortie JSON SANS timestamp.

Machine : iluin-tworings3, terminal natif UNIQUEMENT. Usage :
  .venv/bin/python scripts/run_f1_diagnostic_instrument.py

POINT D'ARRÊT OBLIGATOIRE après le run : lecture remontée à Romain. Le
chiffrage de M-b reste SUSPENDU."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_f1_ma_quater import (  # noqa: E402
    CADENCE_L1,
    PipelineMaQuater,
    geometrie_v4,
)
from scripts.run_f1_ma_ter import VERIFS_EXIGEES, liberer_vram  # noqa: E402
from scripts.run_f1_s2_mobile import DELTA_X_MOBILE  # noqa: E402
from scripts.run_f1_sonde_eps import (  # noqa: E402
    CHAMP_SEDIMENT,
    EPS_EN_VIGUEUR,
    SEUIL_IC_BAS,
    ReconstructeurNiveau0,
    _slot_observe,
    champ_verite,
    delta_chi_readout,
    exiger_cadence_admise,
)
from src.f1_gpu.backend import exiger_cupy  # noqa: E402
from src.f1_gpu.chrono import (  # noqa: E402
    SERIE_FRAMES,
    WARMUP_FRAMES,
    chronometrer_frames,
)
from src.f1_gpu.pyramide import N0_DEFAUT  # noqa: E402
from src.f1_gpu.transferts import TransfertComptable  # noqa: E402
from src.f1_gpu.verifs import exiger_pass, vram_nvidia_smi_octets  # noqa: E402

OUT_DIR = ROOT / "outputs" / "f1"
OUT_JSON_PATH = OUT_DIR / "diagnostic_instrument.json"

# Chiffres GRAVÉS — consommés en lecture mécanique SEULEMENT.
PLANCHER_OBSERVE: float = 0.069        # sonde EPS, plat sur tout le balayage
SEUIL_MORT_MB: float = 0.0733          # T3 par-seed, que M-b partage
MEDIANE_SONDE_MS: float = 17.249       # sonde à EPS=1e-4
MEDIANE_MA_QUATER_MS: float = 16.589   # M-a-quater, même config
ECART_A_ATTRIBUER_MS: float = MEDIANE_SONDE_MS - MEDIANE_MA_QUATER_MS

# Tolérance d'absorption de D-2 : en deçà, l'écart est tenu pour absorbé
# (bruit de mesure) ; au-delà, il persiste. Nommée AVANT le run.
TOLERANCE_ABSORPTION_MS: float = 0.15

# Seuil d'« effondrement » de D-1(b), NON couvert par le gravé — le
# pré-enregistrement dit « si le plancher s'effondre » sans le chiffrer.
# Choisi et nommé AVANT le run : le plancher est tenu pour effondré si la
# comparaison restreinte tombe sous la MOITIÉ de la comparaison complète.
# Un facteur 2 ne se discute pas au cas par cas une fois la donnée vue.
FACTEUR_EFFONDREMENT: float = 0.5

# EPS et k restent FIGÉS — la branche (ii) est appliquée, rien n'est réglé.
EPS_FIGE: float = EPS_EN_VIGUEUR
CADENCE_FIGEE_DIAGNOSTIC: int = CADENCE_L1


def exiger_rien_regle(eps: float, k: int) -> None:
    """Garde de portée : ce diagnostic ne règle RIEN. La branche (ii) est
    appliquée — EPS reste 1e-4 et k reste figé. Toute autre valeur serait
    une tentative de réglage déguisée en diagnostic."""
    if eps != EPS_FIGE:
        raise RuntimeError(
            f"diagnostic : eps={eps} != {EPS_FIGE}. La branche (ii) est "
            "APPLIQUÉE — rien n'est réglé. Ce driver diagnostique "
            "l'instrument, il ne cherche pas un meilleur EPS.")
    exiger_cadence_admise(k)


class ReconstructeurAvecCouverture(ReconstructeurNiveau0):
    """Le reconstructeur de la sonde, plus le suivi des cellules
    EFFECTIVEMENT COUVERTES — celles qu'un coefficient a réellement
    touchées. Les autres n'ont jamais rien reçu depuis l'initialisation :
    ce sont elles que D-1(b) soupçonne de porter le plancher."""

    def __init__(self, pipeline: PipelineMaQuater):
        super().__init__(pipeline)
        formes = [vivant.shape for vivant in self.vivant]
        self.couvert_cycle = [np.zeros(forme, dtype=bool) for forme in formes]
        self.couvert_cumule = [np.zeros(forme, dtype=bool)
                               for forme in formes]

    def appliquer(self, coefficients: list) -> None:
        super().appliquer(coefficients)
        for indice_groupe, valeurs, indices in coefficients:
            if valeurs.size:
                for masque in (self.couvert_cycle[indice_groupe],
                               self.couvert_cumule[indice_groupe]):
                    masque.reshape(-1)[indices] = True

    def nouveau_cycle(self) -> None:
        """Réinitialise la couverture du cycle courant (tous les k)."""
        for masque in self.couvert_cycle:
            masque.fill(False)

    def masque(self, cumule: bool, indice_groupe: int,
               position: int) -> np.ndarray:
        source = self.couvert_cumule if cumule else self.couvert_cycle
        return source[indice_groupe][position, 0, CHAMP_SEDIMENT]


def champ_restreint(vivant: np.ndarray, verite: np.ndarray,
                    couvert: np.ndarray) -> np.ndarray:
    """Vivant restreint aux cellules COUVERTES : les autres prennent la
    valeur de vérité, donc leur écart est nul et seules les couvertes
    contribuent au Δχ. La structure spectrale est préservée — on ne peut
    pas faire de FFT sur un sous-ensemble troué."""
    return np.where(couvert, vivant, verite)


def mesurer_d1(cp, frames: int, facteur: int, k: int) -> dict:
    """D-1 : test à blanc (a) et comparaison restreinte (b), sur la même
    trajectoire que la sonde."""
    geo = geometrie_v4()
    transferts = TransfertComptable(cp)
    pipeline = PipelineMaQuater(cp, geo, transferts, k=k, eps=EPS_FIGE,
                                capturer_coefficients=True)
    reconstructeur = ReconstructeurAvecCouverture(pipeline)
    indice, position = _slot_observe(pipeline)

    a_blanc: list[float] = []
    complet: list[float] = []
    restreint_cycle: list[float] = []
    restreint_cumule: list[float] = []
    fractions_cycle: list[float] = []
    fractions_cumule: list[float] = []

    for numero in range(frames):
        if numero % k == 0:
            reconstructeur.nouveau_cycle()
        pipeline.frame(delta_x=DELTA_X_MOBILE)
        reconstructeur.appliquer(pipeline.coefficients_frame)
        vivant = reconstructeur.champ(indice, position)
        verite = champ_verite(pipeline, indice, position)

        # (a) TEST À BLANC : vivant := vérité, attendu Δχ = 0 EXACT.
        a_blanc.append(delta_chi_readout(verite, verite, facteur))

        complet.append(delta_chi_readout(vivant, verite, facteur))
        for cumule, serie, fractions in (
                (False, restreint_cycle, fractions_cycle),
                (True, restreint_cumule, fractions_cumule)):
            couvert = reconstructeur.masque(cumule, indice, position)
            serie.append(delta_chi_readout(
                champ_restreint(vivant, verite, couvert), verite, facteur))
            fractions.append(float(couvert.mean()))

    def resume(serie: list[float]) -> dict:
        return {"max": float(np.max(serie)),
                "median": float(np.percentile(serie, 50))}

    return {
        "frames": frames,
        "test_a_blanc": {
            **resume(a_blanc),
            "attendu": "Δχ = 0 EXACT",
            "exactement_nul": bool(np.all(np.asarray(a_blanc) == 0.0)),
            "lecture_preecrite": (
                "tout résidu serait un plancher de READOUT — propriété "
                "des primitives elles-mêmes, donc TRANSFÉRABLE à M-b qui "
                f"les partage (seuil de mort {SEUIL_MORT_MB})"),
        },
        "comparaison_complete": resume(complet),
        "comparaison_restreinte_cycle": {
            **resume(restreint_cycle),
            "fraction_couverte_mediane": float(
                np.percentile(fractions_cycle, 50)),
        },
        "comparaison_restreinte_cumulee": {
            **resume(restreint_cumule),
            "fraction_couverte_mediane": float(
                np.percentile(fractions_cumule, 50)),
        },
    }


def lecture_d1(mesure: dict) -> dict:
    """Lectures PRÉ-ÉCRITES de D-1 — aucune autre conclusion."""
    blanc = mesure["test_a_blanc"]
    complet = mesure["comparaison_complete"]["max"]
    restreints = {
        "cycle": mesure["comparaison_restreinte_cycle"]["max"],
        "cumule": mesure["comparaison_restreinte_cumulee"]["max"],
    }
    effondre = {nom: bool(valeur < complet * FACTEUR_EFFONDREMENT)
                for nom, valeur in restreints.items()}
    return {
        "a_plancher_de_readout": {
            "delta_chi_a_blanc_max": blanc["max"],
            "exactement_nul": blanc["exactement_nul"],
            "lecture": (
                "Δχ à blanc EXACTEMENT NUL : aucun plancher de readout — "
                "les primitives ne fabriquent rien, le plancher observé "
                "vient d'ailleurs"
                if blanc["exactement_nul"] else
                "RÉSIDU NON NUL à blanc : plancher de READOUT, propriété "
                "des primitives — TRANSFÉRABLE à M-b, qui les partage "
                f"(seuil de mort {SEUIL_MORT_MB}, plancher sonde "
                f"{PLANCHER_OBSERVE})"),
        },
        "b_plancher_de_couverture": {
            "facteur_effondrement": FACTEUR_EFFONDREMENT,
            "seuil_effondrement": complet * FACTEUR_EFFONDREMENT,
            "delta_chi_complet_max": complet,
            "delta_chi_restreint_max": restreints,
            "plancher_seffondre": effondre,
            "seuil_ic_bas": SEUIL_IC_BAS,
            "restreint_sous_le_seuil": {
                nom: bool(valeur < SEUIL_IC_BAS)
                for nom, valeur in restreints.items()},
            "note_seuil": (
                "reporté seulement : que le Δχ restreint repasse ou non "
                "sous 0.0603 ne change RIEN à la lecture de la sonde, "
                "dont la branche (ii) est appliquée. Aucune relecture "
                "d'EPS n'est ouverte par ce chiffre."),
            "lecture": (
                "le plancher S'EFFONDRE une fois la comparaison "
                "restreinte aux cellules couvertes : plancher de "
                "COUVERTURE, propre à cette sonde — NON transférable à "
                "M-b"
                if any(effondre.values()) else
                "le plancher PERSISTE malgré la restriction : la "
                "couverture n'explique rien, le plancher reste à "
                "attribuer"),
        },
        "rappel_portee": (
            "D-1 rapporte ces deux lectures et RIEN d'autre. Aucune "
            "conclusion supplémentaire n'est tirée, aucun réglage n'est "
            "proposé — la branche (ii) est appliquée."),
    }


def mesurer_d2(cp, k: int) -> dict:
    """D-2 : chrono « nu » à configuration M-a-quater IDENTIQUE, appareil
    de reconstruction HORS BOUCLE — et la preuve que c'est bien le cas."""
    geo = geometrie_v4()
    transferts = TransfertComptable(cp)
    pipeline = PipelineMaQuater(cp, geo, transferts, k=k, eps=EPS_FIGE)
    transferts.frame_suivante()

    def frame() -> None:
        pipeline.frame(delta_x=DELTA_X_MOBILE)
        transferts.frame_suivante()

    chrono = chronometrer_frames(cp, frame)
    regime = transferts.stats_regime(exclure_premiers=1 + WARMUP_FRAMES)
    return {
        "frame_time": chrono["stats"],
        "transferts_median_ms": regime["transfert_total_ms"]["mediane_ms"],
        "preuve_appareil_hors_boucle": {
            "capture_desactivee": pipeline.capturer_coefficients is False,
            "aucun_coefficient_retenu": pipeline.coefficients_frame == [],
            "note": ("PROUVÉ, pas affirmé : la capture est désactivée sur "
                     "le pipeline réellement chronométré, et aucun "
                     "coefficient n'a été retenu au terme de la série — "
                     "l'appareil de reconstruction n'est pas entré dans "
                     "la boucle"),
        },
    }


def lecture_d2(mesure: dict) -> dict:
    """Lectures PRÉ-ÉCRITES de D-2 — l'écart est absorbé, ou il persiste."""
    mediane = mesure["frame_time"]["mediane_ms"]
    ecart_vs_ma_quater = mediane - MEDIANE_MA_QUATER_MS
    absorbe = bool(abs(ecart_vs_ma_quater) <= TOLERANCE_ABSORPTION_MS)
    return {
        "mediane_nue_ms": mediane,
        "mediane_ma_quater_ms": MEDIANE_MA_QUATER_MS,
        "mediane_sonde_ms": MEDIANE_SONDE_MS,
        "ecart_a_attribuer_ms": ECART_A_ATTRIBUER_MS,
        "ecart_restant_ms": ecart_vs_ma_quater,
        "tolerance_absorption_ms": TOLERANCE_ABSORPTION_MS,
        "ecart_absorbe": absorbe,
        "lecture": (
            "écart ABSORBÉ : c'était l'APPAREIL de sonde entrant dans le "
            "chrono — sans conséquence sur V4"
            if absorbe else
            "écart PERSISTANT : TERME DE PRODUCTION NON COMPTÉ — la marge "
            "de V4 est entamée, et le contrôle T1 de M-b (verdictal, Q3) "
            "s'appliquera à ce total"),
        "preuve_appareil_hors_boucle": mesure["preuve_appareil_hors_boucle"],
    }


def main() -> None:
    cp = exiger_cupy()
    exiger_pass(VERIFS_EXIGEES)
    exiger_rien_regle(EPS_FIGE, CADENCE_FIGEE_DIAGNOSTIC)

    geo = geometrie_v4()
    facteur = max(geo.n_fov // N0_DEFAUT, 1)
    print(f"diagnostic d'instrument : EPS={EPS_FIGE:g} FIGÉ, "
          f"k={CADENCE_FIGEE_DIAGNOSTIC} FIGÉ — rien n'est réglé",
          flush=True)

    print("  D-1 : test à blanc + comparaison restreinte ...", flush=True)
    mesure_d1 = mesurer_d1(cp, SERIE_FRAMES, facteur,
                           CADENCE_FIGEE_DIAGNOSTIC)
    liberer_vram(cp)

    print("  D-2 : chrono nu à configuration M-a-quater ...", flush=True)
    mesure_d2 = mesurer_d2(cp, CADENCE_FIGEE_DIAGNOSTIC)
    liberer_vram(cp)

    lectures = {"d1": lecture_d1(mesure_d1), "d2": lecture_d2(mesure_d2)}
    mempool = cp.get_default_memory_pool()

    document = {
        "meta": {
            "mesure": "diagnostic d'instrument D-1/D-2 "
                      "(§A19-lecture-sonde-EPS, pocCascade2phys 07ef6cc)",
            "portee": (
                "diagnostic d'INSTRUMENT — PAS une tentative de régler "
                "EPS. La branche (ii) est APPLIQUÉE : rien n'est réglé, "
                f"EPS reste {EPS_FIGE:g} et k reste "
                f"{CADENCE_FIGEE_DIAGNOSTIC}"),
            "pourquoi": (
                f"plancher {PLANCHER_OBSERVE} contre seuil de mort M-b "
                f"{SEUIL_MORT_MB} (partagent les primitives de readout) : "
                "6 % sous le seuil qui tue Option A. Le chiffrage de M-b "
                "est SUSPENDU à cette lecture"),
            "note_falsifiee": (
                "l'explication « l'observable est gouverné par la "
                "péremption L1 » a été FALSIFIÉE par le bras k=1 (Δχ "
                "identique à 3 décimales entre k=1 et k=4) — elle n'est "
                "plus portée nulle part"),
            "levier_verrouille": (
                "passer de 1e-4 à 1e-2 rendrait ~2.5 ms de transferts et "
                "~1.8 ms de médiane — exactement ce qui manque à V4. "
                "INUTILISABLE : la non-discrimination signifie que "
                "l'observable ne VOIT pas EPS, pas qu'il n'y a rien à "
                "voir. « Prenons le moins cher » serait du raisonnement "
                "motivé, refusé."),
            "cellule": {"n_fov": geo.n_fov, "n_niv": geo.n_niv,
                        "decimation": facteur},
            "frames": {"warmup_exclu": WARMUP_FRAMES, "serie": SERIE_FRAMES},
            "kernels": "F, L3, chrono, substrats, pyramide INTOUCHÉS",
            "verifs_exigees": list(VERIFS_EXIGEES),
        },
        "d1_plancher": mesure_d1,
        "d2_chrono": mesure_d2,
        "residence": {
            "mempool_total_octets": int(mempool.total_bytes()),
            "nvidia_smi_octets": vram_nvidia_smi_octets(),
        },
        "lecture_mecanique": lectures,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON_PATH.write_text(
        json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")

    d1, d2 = lectures["d1"], lectures["d2"]
    print("=" * 78)
    print("F1 -- DIAGNOSTIC D'INSTRUMENT D-1 / D-2")
    print("=" * 78)
    blanc = d1["a_plancher_de_readout"]
    print(f"  D-1(a) test à blanc : Δχ max = "
          f"{blanc['delta_chi_a_blanc_max']:.6f}  "
          f"(exactement nul = {blanc['exactement_nul']})")
    print(f"    {blanc['lecture']}")
    couverture = d1["b_plancher_de_couverture"]
    print(f"  D-1(b) restreint : complet="
          f"{couverture['delta_chi_complet_max']:.4f}  "
          f"cycle={couverture['delta_chi_restreint_max']['cycle']:.4f}  "
          f"cumulé={couverture['delta_chi_restreint_max']['cumule']:.4f}")
    print(f"    {couverture['lecture']}")
    print(f"  D-2 : médiane nue = {d2['mediane_nue_ms']:.3f} ms  "
          f"(M-a-quater {MEDIANE_MA_QUATER_MS}, sonde {MEDIANE_SONDE_MS})")
    print(f"    écart restant = {d2['ecart_restant_ms']:+.3f} ms  "
          f"absorbé = {d2['ecart_absorbe']}")
    print(f"    {d2['lecture']}")
    preuve = d2["preuve_appareil_hors_boucle"]
    print(f"    appareil hors boucle PROUVÉ : capture désactivée="
          f"{preuve['capture_desactivee']}, aucun coefficient retenu="
          f"{preuve['aucun_coefficient_retenu']}")
    print("  Le driver REPORTE — aucune autre conclusion, aucun réglage.")
    print(f"[REPORT] -> {OUT_JSON_PATH}")
    print("POINT D'ARRÊT OBLIGATOIRE : lecture remontée à Romain. Le "
          "chiffrage de M-b reste SUSPENDU.")


if __name__ == "__main__":
    main()
