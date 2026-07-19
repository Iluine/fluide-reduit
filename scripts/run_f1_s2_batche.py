"""F1 — MICRO-SONDE s2 : F BATCHÉ inter-niveaux + CFL asynchrone
(§A17 G1, pocCascade2phys b1f9cf9).

CE QUE s2 DÉPARTAGE : la sonde d'attribution a mesuré F seul (bras A,
immobile, remontée OFF) à 20.438 ms contre 14.33 prédits — +6.11 ms
inexpliqués. Deux des trois suspects gravés sont des artefacts
d'ORCHESTRATION DU HARNAIS, pas du coût architectural : 18 lancements de
kernel là où le moteur réel en ferait 4, et 9 synchronisations CPU dues à
`float(...max())` dans la réduction CFL. s2 re-mesure le MÊME bras A avec
l'orchestration que B5 décrivait déjà, et lit ce qui reste.

LÉGITIMITÉ VS CAP (endossée §A17 G1, rappelée ici) : ce n'est PAS une
escalade du kernel — celui-ci est INTOUCHÉ, empreinte
e8fcaad47db0af010728d5c5d878e1629191bfef4e0382fa80aa3cefd9f27f04. B5
documentait explicitement que « le moteur réel batcherait des fenêtres
identiques » ; le découpage per-niveau était de la plomberie de harnais.
La CFL reste PAYÉE (le même enchaînement de passes tourne) ; seule la
synchronisation CPU, qui est un artefact de mesure, disparaît.

L'ORCHESTRATION BATCHÉE : tous les niveaux sont à 512², donc groupables
par construction. Deux groupes, un appel chacun, 2 étages RK2 ⇒ 4
lancements/frame au lieu de 18 :
  - groupe 2-systèmes : B=5 (2 fovéale fins c=8 + 3 énergie c=8) ;
  - groupe 1-système : B=7 (les 7 fovéale supérieurs c=4).
Soit 5×2 + 7×1 = 17 blocs — exactement les 17 de V2, ni plus ni moins.

LA CFL ON-DEVICE, ET COMMENT ELLE RESTE AUDITABLE : `reduction_cfl` du
jetable finit par `float(...)`, qui rapatrie un scalaire et synchronise le
device. `reduction_cfl_on_device` ci-dessous est sa TRANSCRIPTION EXACTE —
mêmes passes, même arithmétique, même ordre — à ceci près que le résultat
reste un tableau device 0-d au lieu d'être converti en float Python. Rien
n'est omis : le travail GPU est identique, seule la barrière tombe. Le
résultat reste CONSOMMABLE EN AUDIT parce que le driver conserve ce
tableau 0-d et ne le rapatrie qu'UNE FOIS, APRÈS la série chronométrée
(`audit_cfl`) : on vérifie alors qu'il est fini et strictement positif,
donc que la réduction a bien tourné et produit une valeur sensée. Un
`float()` par frame aurait mesuré la barrière ; un `float()` en fin de
série ne mesure rien du tout et prouve la même chose. Deux tests
verrouillent l'absence d'omission : `reduction_cfl_on_device` rapatriée
égale `reduction_cfl` au bit près, et le pas batché produit un état
bit-identique à `pas_f_fusionne`.

Le kernel est réutilisé PAR IMPORT (`_obtenir_kernel`, `_TAILLE_BLOC` de
`substrat_fusionne`) — le module n'est pas modifié d'un octet ; seule
l'orchestration Python, qui est du harnais, est réécrite ici.

RÉSIDENCE COMPARABLE : les buffers de RÉFÉRENCE sont préalloués comme
dans le bras A, bien que la remontée soit OFF et qu'ils ne participent à
aucun calcul. Même doctrine que la sonde précédente : on retire du
TRAVAIL, jamais de la STRUCTURE — sinon s2 mesurerait une pyramide plus
petite, pas la même ré-orchestrée.

CE QUE s2 NE DÉPARTAGE PAS — limite de lecture, nommée avant le run : en
passant de la boucle per-niveau aux deux groupes, TROIS choses tombent
ensemble — les 18 lancements deviennent 4, les 9 synchronisations CFL
disparaissent, et l'orchestration CPU per-niveau (calcul des origines B3,
indexation des buffers) n'a plus lieu. s2 mesure leur effet CUMULÉ, pas
leurs parts respectives. C'est exactement ce que le pré-enregistrement
demande — « artefact d'orchestration VS coût architectural » — mais si
A_batché rentre dans la bande, la lecture est « les trois ensemble
expliquaient les 6.11 ms », jamais « c'était les lancements » ou « c'était
les syncs ». Départager exigerait des bras supplémentaires, non
pré-enregistrés : aucun n'est ajouté ici.

ATTENDU PRÉ-ÉCRIT (gravé §A17 G1) : A_batché dans ±10 % de la prédiction
re-calibrée 5×1.685 + 7×ancre_s1(b) ⇒ les +6.11 ms étaient
l'ORCHESTRATION, et le modèle re-calibré TIENT ; hors ⇒ AUTRE remonté
(coût architectural résiduel non compris — remontée, pas de design à
l'aveugle). L'ancre est LUE dans le JSON de s1, jamais recopiée : s2
refuse de démarrer sans lui.

s2 FIXE LE BUDGET MACHINERIE = 16.7 − A_batché. Ce chiffre est REPORTÉ,
jamais interprété : c'est la séance §A17 qui le confronte aux leviers.

Chrono B6, vérifs #1/#2, natif. Sortie JSON SANS timestamp.

Machine : iluin-tworings3, terminal natif UNIQUEMENT. Usage (APRÈS s1) :
  .venv/bin/python scripts/run_f1_s2_batche.py

POINT D'ARRÊT OBLIGATOIRE après le run : lecture remontée à Romain."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_f1_ma_ter import (  # noqa: E402
    B_FRAME_MS,
    MS_PAR_SLOT,
    VERIFS_EXIGEES,
    liberer_vram,
)
from scripts.run_f1_s1_ancre_1sys import (  # noqa: E402
    N_FOV,
    TOLERANCE_RELATIVE,
    etat_mono_systeme,
)
from scripts.run_f1_s1_ancre_1sys import OUT_JSON_PATH as S1_JSON_PATH  # noqa: E402
from src.f1_gpu.backend import exiger_cupy  # noqa: E402
from src.f1_gpu.chrono import (  # noqa: E402
    SERIE_FRAMES,
    WARMUP_FRAMES,
    chronometrer_frames,
)
from src.f1_gpu.pyramide import GRAINE_MONDE  # noqa: E402
# Le kernel est RÉUTILISÉ par import — `substrat_fusionne` n'est pas
# modifié (diff vide prouvé) ; seule l'orchestration Python est réécrite.
from src.f1_gpu.substrat_fusionne import (  # noqa: E402
    _TAILLE_BLOC,
    _obtenir_kernel,
)
from src.f1_gpu.substrat_jetable import (  # noqa: E402
    CFL_JETABLE,
    DT_JETABLE,
    GRAVITE,
    _desing,
    etat_initial_jetable,
)
from src.f1_gpu.verifs import exiger_pass, vram_nvidia_smi_octets  # noqa: E402

OUT_DIR = ROOT / "outputs" / "f1"
OUT_JSON_PATH = OUT_DIR / "s2_batche.json"

# Les deux groupes de V2 (17 blocs : 5x2 + 7x1).
N_BLOCS_2SYS: int = 5      # 2 fovéale fins c=8 + 3 énergie c=8
N_BLOCS_1SYS: int = 7      # les 7 fovéale supérieurs c=4
LANCEMENTS_PAR_FRAME: int = 4          # 2 groupes x 2 étages RK2
LANCEMENTS_BRAS_A_PER_NIVEAU: int = 18  # 9 niveaux x 2 étages (la référence)

ANCRE_2SYS_MS: float = MS_PAR_SLOT[8]   # 1.685, mesurée M-a′

# Écart admis entre le dt CFL on-device (division finale f32) et celui du
# jetable (division f64 après rapatriement) — dernier bit f32 sur une
# grandeur NON consommée. `smax`, lui, est exigé bit-identique.
TOL_CFL_RELATIVE: float = 1e-6


def reduction_cfl_on_device(q, xp):
    """Transcription de `substrat_jetable.reduction_cfl` — mêmes passes,
    même arithmétique, même ordre — SAUF que le résultat reste un tableau
    device 0-d au lieu d'un float Python.

    Le `float()` du jetable rapatrie un scalaire, ce qui SYNCHRONISE le
    device : c'est une barrière par appel. Ici la réduction est
    intégralement PAYÉE (sqrt, désingularisation, maximum, .max()) mais
    rien ne redescend vers l'hôte, donc rien ne bloque. Le résultat reste
    auditable : l'appelant garde le tableau 0-d et le rapatrie UNE fois,
    hors chrono (`audit_cfl`).

    UN ÉCART, NOMMÉ (mesuré ≈ 2.9e-8 en relatif) : `smax`, tout le
    travail lourd de la réduction, est BIT-IDENTIQUE au jetable — c'est
    testé. Seule la DIVISION FINALE diffère : le jetable la fait en f64
    (son `float()` a promu le scalaire avant de diviser), celle-ci la
    fait en f32 puisqu'elle reste on-device. Trois raisons de ne pas
    forcer le f64 ici : la valeur n'est PAS consommée (le kernel reçoit
    `DT_JETABLE` figé — l'état produit est bit-identique, testé) ; une
    conversion f32->f64 ajouterait un lancement que le jetable ne paie
    pas sur device, donc fausserait à la hausse ce que s2 mesure ; et
    l'écart est du dernier bit f32 sur une grandeur de diagnostic.
    Verrouillé par tests : `smax` identique, dt égal à TOL_CFL_RELATIVE
    près, état final bit-identique."""
    h, hu, hv = q[:, :, 0], q[:, :, 1], q[:, :, 2]
    c = xp.sqrt(GRAVITE * xp.maximum(h, 0.0))
    u = _desing(h, hu, xp)
    v = _desing(h, hv, xp)
    smax = xp.maximum(xp.abs(u) + c, xp.abs(v) + c).max()
    return CFL_JETABLE / xp.maximum(smax, 1e-12)


def pas_f_fusionne_async(q, cp, sortie, tampon_etage):
    """`substrat_fusionne.pas_f_fusionne` avec la réduction CFL laissée
    ON-DEVICE. Le KERNEL est le même objet (importé, jamais recompilé
    différemment) et les deux étages RK2 sont lancés à l'identique :
    mêmes arguments, même dt figé `DT_JETABLE`, même grille, même taille
    de bloc. La SEULE différence avec l'original est que `dt_cfl` reste un
    tableau device.

    Verrouillé par test : l'état produit est BIT-IDENTIQUE à celui de
    `pas_f_fusionne` sur le même entrant."""
    if q.dtype != cp.float32:
        raise ValueError(f"pas_f_fusionne_async : dtype {q.dtype} != float32.")
    if q.ndim != 5 or q.shape[2] != 4 or q.shape[1] not in (1, 2):
        raise ValueError(
            f"pas_f_fusionne_async : shape {q.shape} != (B, S, 4, n, n) "
            "avec S ∈ {1, 2} systèmes (E4a).")
    dt_cfl = reduction_cfl_on_device(q, cp)      # payée, NON synchronisante

    n = int(q.shape[-1])
    total = int(q.shape[0] * q.shape[1] * n * n)
    kernel = _obtenir_kernel(cp)
    grille = ((total + _TAILLE_BLOC - 1) // _TAILLE_BLOC,)
    dt = np.float32(DT_JETABLE)
    n_arg, total_arg = np.int32(n), np.int64(total)
    kernel(grille, (_TAILLE_BLOC,),
           (q, q, tampon_etage, dt, np.float32(0.0), n_arg, total_arg))
    kernel(grille, (_TAILLE_BLOC,),
           (tampon_etage, q, sortie, dt, np.float32(0.5), n_arg, total_arg))
    return sortie, dt_cfl


class GroupesBatches:
    """Le bras A ré-orchestré : deux groupes de blocs, un appel chacun.

    B2 — TOUT est préalloué à la construction : fenêtres, tampons
    d'étage, ET références. Les références ne participent à aucun calcul
    (remontée OFF) ; elles sont là pour que la RÉSIDENCE soit celle du
    bras A. On retire du travail, jamais de la structure."""

    def __init__(self, cp, n_blocs_2sys: int = N_BLOCS_2SYS,
                 n_blocs_1sys: int = N_BLOCS_1SYS, n_fov: int = N_FOV):
        self.cp = cp
        deux = cp.asarray(etat_initial_jetable(n_blocs_2sys, n_fov,
                                               GRAINE_MONDE))
        un = etat_mono_systeme(cp, n_blocs_1sys, n_fov)
        self.fenetres = [deux, un]
        self.tampons = [cp.empty_like(bloc) for bloc in self.fenetres]
        # Références : préallouées pour la résidence, jamais lues (B4 OFF).
        self.references = [cp.array(bloc, copy=True) for bloc in self.fenetres]
        self.dt_cfl = [None, None]

    def frame(self) -> None:
        """Une frame du bras A batché : 2 appels, 4 lancements de kernel,
        aucune synchronisation CPU. Ni déplacement (fovéa immobile) ni
        remontée (B4 OFF) — comme le bras A d'origine."""
        for indice, (fen, tampon) in enumerate(
                zip(self.fenetres, self.tampons)):
            _, dt_cfl = pas_f_fusionne_async(fen, self.cp, fen, tampon)
            self.dt_cfl[indice] = dt_cfl

    def audit_cfl(self) -> dict:
        """Rapatrie les réductions CFL UNE fois, APRÈS la série — hors
        chrono, donc sans rien fausser. Atteste que la réduction a bien
        tourné et produit des valeurs finies et strictement positives (si
        elle avait été omise, ces tableaux seraient absents ; si elle
        avait dégénéré, la valeur le montrerait)."""
        valeurs = [float(dt) if dt is not None else None
                   for dt in self.dt_cfl]
        return {
            "dt_cfl_par_groupe": valeurs,
            "toutes_finies_positives": bool(
                all(v is not None and np.isfinite(v) and v > 0.0
                    for v in valeurs)),
            "note": ("rapatriement UNIQUE hors chrono — pendant la série, "
                     "la réduction est payée sans jamais synchroniser"),
        }

    def octets_etat(self) -> int:
        return int(sum(bloc.nbytes for bloc in self.fenetres)
                   + sum(bloc.nbytes for bloc in self.references))

    def n_blocs(self) -> int:
        return int(sum(bloc.shape[0] * bloc.shape[1]
                       for bloc in self.fenetres))


def lire_ancre_s1(chemin: Path = S1_JSON_PATH) -> dict:
    """Ancre 1-système MESURÉE, lue dans le JSON de s1 — jamais recopiée
    ici. Fail-loud si s1 n'a pas tourné : l'ordre s1 -> s2 est une
    DÉPENDANCE, pas une convention."""
    if not chemin.exists():
        raise RuntimeError(
            f"s2 exige l'ancre MESURÉE de s1, absente ({chemin}). "
            "Lancer d'abord `.venv/bin/python scripts/"
            "run_f1_s1_ancre_1sys.py` — la prédiction re-calibrée de s2 "
            "consomme cette valeur, la recopier à la main serait un "
            "chiffre non traçable. Aucune mesure ici.")
    document = json.loads(chemin.read_text(encoding="utf-8"))
    try:
        retenue = document["lecture_mecanique"]["ancre_retenue_pour_s2"]
        valeur = float(retenue["valeur_ms"])
    except (KeyError, TypeError, ValueError) as erreur:
        raise RuntimeError(
            f"s2 : JSON s1 illisible ou incomplet ({chemin}) — "
            f"{erreur!r}. Aucune mesure.") from erreur
    if not np.isfinite(valeur) or valeur <= 0.0:
        raise RuntimeError(
            f"s2 : ancre s1 non exploitable ({valeur}). Aucune mesure.")
    return {"ancre_1sys_ms": valeur, "origine": retenue["origine"],
            "source": str(chemin)}


def lecture_mecanique(a_batche_ms: float, ancre_s1: dict) -> dict:
    """Consommation MÉCANIQUE des attendus §A17 G1 — le driver REPORTE.
    Le budget machinerie est calculé et reporté, JAMAIS interprété."""
    ancre_1sys = ancre_s1["ancre_1sys_ms"]
    prediction = (N_BLOCS_2SYS * ANCRE_2SYS_MS + N_BLOCS_1SYS * ancre_1sys)
    ecart_relatif = (a_batche_ms - prediction) / prediction
    return {
        "a_batche_ms": a_batche_ms,
        "prediction_recalibree": {
            "formule": (f"{N_BLOCS_2SYS} x {ANCRE_2SYS_MS} (2-sys, M-a′) "
                        f"+ {N_BLOCS_1SYS} x {ancre_1sys:.3f} "
                        "(1-sys, MESURÉE en s1)"),
            "valeur_ms": prediction,
            "ancre_1sys_ms": ancre_1sys,
            "ancre_1sys_origine": ancre_s1["origine"],
            "ancre_1sys_source": ancre_s1["source"],
        },
        "tolerance_relative": TOLERANCE_RELATIVE,
        "ecart_relatif": ecart_relatif,
        "dans_tolerance": bool(abs(ecart_relatif) <= TOLERANCE_RELATIVE),
        "branche_preecrite_si_dans_tolerance": (
            "les +6.11 ms de la sonde étaient l'ORCHESTRATION (artefact "
            "de harnais) — le modèle re-calibré TIENT"),
        "branche_preecrite_si_hors_tolerance": (
            "AUTRE remonté : coût architectural résiduel non compris — "
            "remontée, PAS de design à l'aveugle (§A17 G1)"),
        "budget_machinerie": {
            "formule": f"{B_FRAME_MS} - A_batché",
            "valeur_ms": B_FRAME_MS - a_batche_ms,
            "seuil_t2_ms": B_FRAME_MS,
            "note": ("REPORTÉ, jamais interprété : c'est la séance §A17 "
                     "qui le confronte aux leviers L1/L3. Un budget "
                     "négatif ou nul se lit tel quel, sans commentaire "
                     "de ce driver."),
        },
        "limite_de_lecture": (
            "TROIS artefacts tombent ENSEMBLE dans ce bras (18 -> 4 "
            "lancements, 9 syncs CFL supprimées, orchestration CPU "
            "per-niveau supprimée) : s2 mesure leur effet CUMULÉ, jamais "
            "leurs parts. Si A_batché rentre dans la bande, la lecture "
            "est « les trois ensemble expliquaient les 6.11 ms » — pas "
            "« c'était les lancements ». Départager exigerait des bras "
            "non pré-enregistrés ; aucun n'est ajouté."),
        "rappel_portee": (
            "micro-sonde d'ouverture de séance : elle ANCRE le design, "
            "elle ne le décide pas. Aucun levier tranché, aucune "
            "escalade — le kernel est intouché."),
    }


def main() -> None:
    cp = exiger_cupy()
    exiger_pass(VERIFS_EXIGEES)
    ancre_s1 = lire_ancre_s1()      # fail-loud AVANT toute allocation

    print(f"ancre 1-système lue de s1 : {ancre_s1['ancre_1sys_ms']:.3f} ms/slot",
          flush=True)
    groupes = GroupesBatches(cp)
    print(f"bras A batché : {groupes.n_blocs()} blocs, "
          f"{LANCEMENTS_PAR_FRAME} lancements/frame ...", flush=True)

    chrono = chronometrer_frames(cp, groupes.frame)
    audit = groupes.audit_cfl()     # rapatriement UNIQUE, hors chrono

    mempool = cp.get_default_memory_pool()
    a_batche_ms = chrono["stats"]["mediane_ms"]
    lecture = lecture_mecanique(a_batche_ms, ancre_s1)

    document = {
        "meta": {
            "mesure": "s2 — F batché inter-niveaux + CFL asynchrone "
                      "(§A17 G1, pocCascade2phys b1f9cf9)",
            "protocole": {
                "bras": ("A de la sonde d'attribution (fovéa IMMOBILE, "
                         "remontée B4 OFF) RE-ORCHESTRÉ"),
                "F": ("kernel FUSIONNÉ M-a′ INTOUCHÉ, réutilisé par "
                      "import ; seule l'orchestration Python est réécrite"),
                "groupes": {"2_systemes": N_BLOCS_2SYS,
                            "1_systeme": N_BLOCS_1SYS,
                            "blocs_total": groupes.n_blocs()},
                "lancements_par_frame": LANCEMENTS_PAR_FRAME,
                "lancements_bras_a_reference": LANCEMENTS_BRAS_A_PER_NIVEAU,
                "cfl": ("PAYÉE on-device, NON synchronisante (pas de "
                        "float() par frame) — auditée une fois hors série"),
                "chrono": "B6 (cuda-Events, warmup exclu, médiane ET p99)",
                "residence": ("références préallouées comme au bras A bien "
                              "que la remontée soit OFF — on retire du "
                              "travail, pas de la structure"),
            },
            "legitimite_vs_cap": (
                "pas une escalade kernel (diff vide, empreinte "
                "e8fcaad47db0af010728d5c5d878e1629191bfef4e0382fa80aa3cef"
                "d9f27f04) — B5 prévoyait le batch du moteur réel ; le "
                "per-niveau était de la plomberie de harnais"),
            "frames": {"warmup_exclu": WARMUP_FRAMES, "serie": SERIE_FRAMES},
            "verifs_exigees": list(VERIFS_EXIGEES),
        },
        "frame_time": chrono["stats"],
        "audit_cfl": audit,
        "residence": {
            "mempool_used_octets": int(mempool.used_bytes()),
            "mempool_total_octets": int(mempool.total_bytes()),
            "etat_prealloue_octets": groupes.octets_etat(),
            "nvidia_smi_octets": vram_nvidia_smi_octets(),
        },
        "lecture_mecanique": lecture,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON_PATH.write_text(
        json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    liberer_vram(cp)

    prediction = lecture["prediction_recalibree"]
    budget = lecture["budget_machinerie"]
    print("=" * 78)
    print("F1 -- s2 : F batché inter-niveaux + CFL async (lecture mécanique)")
    print("=" * 78)
    print(f"  orchestration : {groupes.n_blocs()} blocs en "
          f"{LANCEMENTS_PAR_FRAME} lancements/frame "
          f"(bras A : {LANCEMENTS_BRAS_A_PER_NIVEAU}), CFL sans sync CPU")
    print(f"  A_batché  = {a_batche_ms:.3f} ms   "
          f"p99 = {chrono['stats']['p99_ms']:.3f} ms")
    print(f"  prédiction re-calibrée = {prediction['valeur_ms']:.3f} ms")
    print(f"    ({prediction['formule']})")
    print(f"  écart = {lecture['ecart_relatif']:+.1%}  "
          f"dans ±{TOLERANCE_RELATIVE:.0%} = {lecture['dans_tolerance']}")
    print(f"  budget machinerie = {budget['valeur_ms']:.3f} ms "
          f"(= {B_FRAME_MS} - A_batché)  [REPORTÉ, non interprété]")
    print(f"  audit CFL : dt par groupe = {audit['dt_cfl_par_groupe']}  "
          f"finies>0 = {audit['toutes_finies_positives']}")
    print("  RAPPEL : micro-sonde d'ouverture — elle ANCRE le design, "
          "elle ne le décide pas ; kernel intouché.")
    print(f"[REPORT] -> {OUT_JSON_PATH}")
    print("POINT D'ARRÊT OBLIGATOIRE : lecture remontée à Romain, aucun "
          "levier tranché, aucun enchaînement automatique.")


if __name__ == "__main__":
    main()
