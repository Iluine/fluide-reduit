"""F1 tranche-1 — driver des VÉRIFICATIONS D'INSTRUMENT (§A15, composant (e)).

Exécute et ENREGISTRE (registre `outputs/f1/verifs.json`, B9) :
  #1 résidence VRAM : mempool CuPy vs nvidia-smi (allocation-témoin) ;
  #4 bande passante PCIe à vide : brut aller/retour, pageable + pinned,
     médiane ET p99 — le dénominateur de M-c, mesuré AVANT le schéma diff ;
  #3 gel bit-exact CPU f64 : le test EXISTANT
     `test_replay_prefixe_bit_identique` rejoué en subprocess (~1 min CPU)
     — garde de coexistence numpy 2.4.6 APRÈS l'installation CuPy (E2).
La vérif #2 (sensibilité du chrono) appartient au driver M-a : elle a
besoin du scan n_fov 256->512.

Machine : iluin-tworings3, terminal natif UNIQUEMENT (la VM tue à 45 s —
la vérif #3 seule dure ~1 min). Usage :
  .venv/bin/python scripts/run_f1_verifs.py

Aucune lecture de mesure ici : ce driver ARME l'instrument. Un FAIL =
tranche muette — remonter à Romain, ne rien mesurer."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.f1_gpu.backend import exiger_cupy  # noqa: E402
from src.f1_gpu.transferts import mesurer_bande_passante_a_vide  # noqa: E402
from src.f1_gpu.verifs import (  # noqa: E402
    CHEMIN_REGISTRE_DEFAUT,
    enregistrer_verif,
    verif_gel_bit_exact,
    verif_vram_concordance,
)


def _bande_passante_pass(document: dict) -> bool:
    """PASS de la vérif #4 = EXISTENCE d'une mesure complète et saine
    (tous débits finis et > 0) — aucun seuil de qualité inventé : le
    chiffre lui-même est le dénominateur de M-c, il sera dans la lecture."""
    for taille in document["tailles"]:
        for memoire in ("pageable", "pinned"):
            for direction in ("h2d", "d2h"):
                debit = taille[memoire][direction]["go_par_s"]
                if not (debit > 0.0 and debit == debit):
                    return False
    return True


def main() -> None:
    cp = exiger_cupy()
    resultats: dict[str, bool] = {}

    passe, details = verif_vram_concordance(cp)
    enregistrer_verif("vram_concordance", passe, details)
    resultats["vram_concordance"] = passe

    document_bp = mesurer_bande_passante_a_vide(cp)
    passe_bp = _bande_passante_pass(document_bp)
    enregistrer_verif("bande_passante_a_vide", passe_bp, document_bp)
    resultats["bande_passante_a_vide"] = passe_bp

    print("vérif #3 (gel bit-exact, ~1 min CPU) ...", flush=True)
    passe_gel, details_gel = verif_gel_bit_exact()
    enregistrer_verif("gel_bit_exact", passe_gel, details_gel)
    resultats["gel_bit_exact"] = passe_gel

    print("=" * 78)
    print("F1 TRANCHE-1 -- VÉRIFICATIONS D'INSTRUMENT (§A15) : #1, #4, #3")
    print("=" * 78)
    for nom, passe in resultats.items():
        print(f"  {nom:<24} : {'PASS' if passe else 'FAIL'}")
    print(f"[REGISTRE] -> {CHEMIN_REGISTRE_DEFAUT}")
    if not all(resultats.values()):
        print("TRANCHE MUETTE : au moins une vérification a échoué -- "
              "REMONTER à Romain, aucune mesure.")
        raise SystemExit(1)
    print("Instrument armé (#2 restera due au driver M-a). "
          "POINT D'ARRÊT : remonter avant tout run de mesure.")


if __name__ == "__main__":
    main()
