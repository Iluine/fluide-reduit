"""Scellement PROCÉDURAL des seuils P3′ (mission chantier 8, 8a.5 — M4).

Les seuils des bras d'une session P3′ ne s'écrivent plus en clair au
manifeste : ils vont dans un fichier À PART, dont le sha256 est consigné au
manifeste ; seule la lecture mécanique, gardes toutes vertes, le descelle.

NATURE NOMMÉE (mission 8a.5) : scellement PROCÉDURAL, pas cryptographique —
il protège du regard ACCIDENTEL et rend toute ALTÉRATION détectable
(sha256) ; la CONSULTATION, elle, ne laisse AUCUNE trace — un fichier se lit,
c'est la leçon M4 elle-même. Si quelqu'un ouvre le fichier scellé avant le
prononcé, la non-cécité se CONSIGNE au verdict, comme au prereg v2.

UN exemplaire, module DÉDIÉ (pas dans `arcC_abx.py`) : importé par
l'orchestration (qui scelle) ET la lecture (qui descelle), testé seul — la
raison est le foyer unique testable."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def scelle_seuils(path: str | Path, seuils: dict) -> str:
    """Écrit `seuils` (dict JSON-sérialisable, p. ex. {"severe": {"0": 0.07,
    ...}}) dans le fichier scellé et renvoie son sha256 — c'est CE sha que le
    manifeste consigne. L'écriture est atomique au sens utile ici : le sha est
    calculé sur les octets réellement écrits, relus depuis le disque."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(nature="seuils_scelles_p3prime (M4, mission 8a.5)",
                                 seuils=seuils),
                            indent=2, ensure_ascii=False), encoding="utf-8")
    return hashlib.sha256(p.read_bytes()).hexdigest()


def descelle_seuils(path: str | Path, sha256_attendu: str) -> dict:
    """Relit le fichier scellé et VÉRIFIE son sha256 contre celui du manifeste
    — fail-loud : un fichier absent, altéré ou substitué lève, jamais un
    silence. Renvoie le dict `seuils` tel que scellé.

    L'appelant (la lecture mécanique) ne doit desceller qu'avec TOUTES ses
    autres gardes vertes — c'est son contrat, pas celui de ce module."""
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"descelle_seuils : fichier scellé INTROUVABLE ({p}) -- "
                           "rien à desceller, STOP.")
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    if sha != sha256_attendu:
        raise RuntimeError(
            "descelle_seuils : sha256 du fichier scellé != sha consigné au manifeste "
            f"({sha} != {sha256_attendu}) -- fichier ALTÉRÉ ou substitué depuis la "
            "session, STOP.")
    return json.loads(p.read_text(encoding="utf-8"))["seuils"]
