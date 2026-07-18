"""F1 tranche-1 — composant (e) : les 4 vérifications d'instrument §A15 +
câblage « tranche muette => remonter » (choix B9).

Texte gravé (§A15, « Instrument ») : « Vérification d'instrument DUE avant
toute lecture (sinon remonter, tranche muette) : 1. Résidence VRAM
mesurable et concordante (allocateur du stack vs nvidia-smi) ; 2.
Sensibilité du chrono : n_fov 256->512 doit bouger le frame-time ; 3. Le
chemin rederive CPU f64 reproduit le gel bit-exact sur la machine (le test
de gel existant, reconduit) ; 4. Bande passante PCIe soutenable MESURÉE à
vide. »

B9 — registre `outputs/f1/verifs.json` : chaque vérification y écrit
{pass, details} sous son nom ; chaque driver de mesure appelle
`exiger_pass(...)` AVANT toute lecture et lève RuntimeError (« TRANCHE
MUETTE ») si une vérification exigée manque ou a échoué. Le registre est
un artefact d'instrument, pas une donnée de mesure.

Implémentations :
  #1 `verif_vram_concordance` : allocation-témoin via cupy, delta
     `used_bytes()` du mempool vs delta `nvidia-smi memory.used` —
     concordants à TOLERANCE_VRAM_OCTETS près (64 Mo : granularité
     d'allocation du driver + bruit d'autres processus, tolérance
     d'instrument nommée) ;
  #2 (sensibilité chrono) : produite PAR le driver M-a depuis son scan
     (seuil B9 : médiane(512) > médiane(256) d'au moins
     SEUIL_SENSIBILITE_RELATIF = 5 % à N_niv identique) — `verifier_
     sensibilite_chrono` fait le calcul, le driver l'enregistre ;
  #3 `verif_gel_bit_exact` : reconduction du test EXISTANT
     `tests/test_arcA_histories_16L0.py::test_replay_prefixe_bit_identique`
     via subprocess pytest (~1 min CPU) — c'est AUSSI la garde de
     coexistence numpy 2.4.6 après l'installation CuPy (E2 : « re-jouer la
     vérif #3 après install ») ; à rejouer NATIF ;
  #4 bande passante à vide : `transferts.mesurer_bande_passante_a_vide`,
     enregistrée ici par le driver des vérifications."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_RACINE = Path(__file__).resolve().parents[2]

CHEMIN_REGISTRE_DEFAUT: Path = _RACINE / "outputs" / "f1" / "verifs.json"

NOMS_VERIFS: tuple[str, ...] = (
    "vram_concordance", "sensibilite_chrono", "gel_bit_exact",
    "bande_passante_a_vide")

TOLERANCE_VRAM_OCTETS: int = 64 * 1024 * 1024   # B9, tolérance d'instrument
TAILLE_TEMOIN_OCTETS: int = 256 * 1024 * 1024   # allocation-témoin vérif #1
SEUIL_SENSIBILITE_RELATIF: float = 0.05         # B9, vérif #2

_TEST_GEL: str = ("tests/test_arcA_histories_16L0.py"
                  "::test_replay_prefixe_bit_identique")


def lire_registre(chemin: str | Path = CHEMIN_REGISTRE_DEFAUT) -> dict:
    """Registre des vérifications ({} si aucun enregistrement)."""
    chemin = Path(chemin)
    if not chemin.exists():
        return {}
    return json.loads(chemin.read_text(encoding="utf-8"))


def enregistrer_verif(nom: str, passe: bool, details: dict,
                      chemin: str | Path = CHEMIN_REGISTRE_DEFAUT) -> dict:
    """Écrit {pass, details} sous `nom` dans le registre (les autres
    entrées sont conservées). Nom inconnu -> ValueError."""
    if nom not in NOMS_VERIFS:
        raise ValueError(
            f"enregistrer_verif : nom inconnu {nom!r} "
            f"(attendu un de {NOMS_VERIFS}).")
    chemin = Path(chemin)
    registre = lire_registre(chemin)
    registre[nom] = {"pass": bool(passe), "details": details}
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(json.dumps(registre, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    return registre


def exiger_pass(noms: tuple[str, ...],
                chemin: str | Path = CHEMIN_REGISTRE_DEFAUT) -> None:
    """Câblage « tranche muette » (B9) : RuntimeError si une vérification
    exigée est absente du registre ou n'a pas PASS — AUCUNE lecture ne se
    produit sans instrument vérifié (§A15 : « sinon remonter, tranche
    muette »)."""
    registre = lire_registre(chemin)
    manquantes = [n for n in noms if n not in registre]
    echouees = [n for n in noms
                if n in registre and not registre[n]["pass"]]
    if manquantes or echouees:
        raise RuntimeError(
            "TRANCHE MUETTE (§A15/B9) — vérifications d'instrument "
            f"manquantes : {manquantes}, échouées : {echouees} "
            f"(registre : {Path(chemin)}). Remonter, aucune lecture.")


def vram_nvidia_smi_octets() -> int:
    """`nvidia-smi --query-gpu=memory.used` en octets (fail-loud si
    l'outil manque : la vérif #1 est alors impossible, remonter)."""
    resultat = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True)
    return int(resultat.stdout.strip().splitlines()[0]) * 1024 * 1024


def verif_vram_concordance(
        cp, taille_temoin: int = TAILLE_TEMOIN_OCTETS) -> tuple[bool, dict]:
    """Vérif #1 : le delta de `used_bytes()` du mempool CuPy et le delta de
    `nvidia-smi` autour d'une allocation-témoin doivent concorder à
    TOLERANCE_VRAM_OCTETS près. Retourne (passe, details)."""
    mempool = cp.get_default_memory_pool()
    cp.cuda.runtime.deviceSynchronize()
    smi_avant = vram_nvidia_smi_octets()
    pool_avant = mempool.used_bytes()
    temoin = cp.zeros(taille_temoin, dtype=cp.uint8)
    temoin[::4096] = 1  # toucher les pages (allocation réellement résidente)
    cp.cuda.runtime.deviceSynchronize()
    smi_apres = vram_nvidia_smi_octets()
    pool_apres = mempool.used_bytes()
    del temoin
    delta_smi = smi_apres - smi_avant
    delta_pool = pool_apres - pool_avant
    ecart = abs(delta_smi - delta_pool)
    details = {
        "taille_temoin_octets": int(taille_temoin),
        "delta_mempool_octets": int(delta_pool),
        "delta_nvidia_smi_octets": int(delta_smi),
        "ecart_octets": int(ecart),
        "tolerance_octets": int(TOLERANCE_VRAM_OCTETS),
        "smi_base_octets": int(smi_avant),
    }
    passe = (delta_pool >= taille_temoin
             and ecart <= TOLERANCE_VRAM_OCTETS)
    return passe, details


def verifier_sensibilite_chrono(mediane_256_ms: float,
                                mediane_512_ms: float) -> tuple[bool, dict]:
    """Vérif #2 (calcul ; l'ENREGISTREMENT appartient au driver M-a,
    AVANT sa lecture) : n_fov 256->512 doit bouger la médiane d'au moins
    SEUIL_SENSIBILITE_RELATIF, vers le haut (un chrono muet ou inversé =
    instrument non crédible)."""
    ecart_relatif = (mediane_512_ms - mediane_256_ms) / max(
        mediane_256_ms, 1e-12)
    details = {"mediane_256_ms": float(mediane_256_ms),
               "mediane_512_ms": float(mediane_512_ms),
               "ecart_relatif": float(ecart_relatif),
               "seuil_relatif": SEUIL_SENSIBILITE_RELATIF}
    return ecart_relatif >= SEUIL_SENSIBILITE_RELATIF, details


def verif_gel_bit_exact(racine: str | Path = _RACINE) -> tuple[bool, dict]:
    """Vérif #3 : rejoue le test de gel EXISTANT (reconduit, jamais
    réimplémenté) dans un subprocess pytest du MÊME interpréteur — la
    preuve que le chemin rederive CPU f64 n'a pas bougé après
    l'installation du stack GPU (E2). ~1 min CPU : machine-instrument
    NATIVE uniquement (la VM tue à 45 s)."""
    resultat = subprocess.run(
        [sys.executable, "-m", "pytest", _TEST_GEL, "-q", "--no-header"],
        capture_output=True, text=True, cwd=str(racine))
    sortie = (resultat.stdout + resultat.stderr).strip().splitlines()
    details = {"test": _TEST_GEL, "code_retour": resultat.returncode,
               "dernieres_lignes": sortie[-5:]}
    return resultat.returncode == 0, details
