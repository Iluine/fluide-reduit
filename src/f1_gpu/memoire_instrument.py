"""F1 — M-b tranche-2 : sonde mémoire PERSISTÉE, par bras et par phase
(§A31-run).

POURQUOI PERSISTER, ET POURQUOI EN JSON *LINES*. L'OOM killer envoie un
SIGKILL : le processus ne lève rien, ne déroule aucun `finally`, n'exécute
aucun `except`. Tout ce qui n'est pas DÉJÀ SUR LE DISQUE à cet instant est
perdu — c'est exactement la leçon gravée du FAIL cloud non persisté. La
sonde écrit donc :
  - le jalon de DÉBUT avant que la phase ne travaille (savoir QUI tenait le
    processus quand il est mort) ;
  - un BATTEMENT périodique portant le pic courant (savoir COMBIEN il tenait
    juste avant de mourir) ;
  - le jalon de FIN, exception comprise.
Une ligne par jalon : un fichier tronqué en pleine écriture garde toutes ses
lignes précédentes lisibles, ce qu'un JSON unique ne permettrait pas.

CE QUE LA SONDE MESURE. RAM hôte : RSS courant (`/proc/self/status`) et pic
échantillonné par un fil de fond (vrai maximum de la phase, y compris un pic
transitoire relâché avant la fin). VRAM : `used_bytes`/`total_bytes` du
mempool CuPy quand un `cp` est fourni — `None` sinon, la sonde devant
fonctionner sans GPU puisque le poste suspect est la RAM hôte.

Instrument, pas mesure : rien ici n'entre dans une lecture de fidélité."""
from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

_PAGE = 1024 * 1024      # les tailles sont reportées en Mo


def rss_mo() -> float:
    """RSS courant du processus, en Mo (Linux : `/proc/self/status`)."""
    with open(f"/proc/{os.getpid()}/status", encoding="utf-8") as f:
        for ligne in f:
            if ligne.startswith("VmRSS:"):
                return int(ligne.split()[1]) / 1024
    return 0.0


def _vram_mo(cp) -> tuple[float | None, float | None]:
    """(utilisé, réservé) du mempool CuPy en Mo, ou (None, None) sans GPU."""
    if cp is None:
        return None, None
    pool = cp.get_default_memory_pool()
    return pool.used_bytes() / _PAGE, pool.total_bytes() / _PAGE


def lire_jalons(chemin: str | Path) -> list[dict]:
    """Relit les jalons. TOLÈRE une dernière ligne tronquée — c'est
    précisément ce que laisse un SIGKILL survenu pendant une écriture."""
    chemin = Path(chemin)
    if not chemin.exists():
        return []
    jalons = []
    for ligne in chemin.read_text(encoding="utf-8").splitlines():
        ligne = ligne.strip()
        if not ligne:
            continue
        try:
            jalons.append(json.loads(ligne))
        except json.JSONDecodeError:
            continue          # ligne coupée en vol : on garde le reste
    return jalons


def pic_par_phase(jalons: list[dict]) -> dict[tuple[str, str], dict]:
    """Pic RSS et VRAM attribués PAR (bras, phase) — le maximum de tous les
    jalons portant cette clé, battements compris."""
    pics: dict[tuple[str, str], dict] = {}
    for j in jalons:
        cle = (j.get("bras"), j.get("phase"))
        courant = pics.setdefault(
            cle, {"rss_pic_mo": 0.0, "vram_pic_mo": None})
        courant["rss_pic_mo"] = max(courant["rss_pic_mo"],
                                    j.get("rss_pic_mo") or 0.0,
                                    j.get("rss_mo") or 0.0)
        vram = j.get("vram_pool_mo")
        if vram is not None:
            courant["vram_pic_mo"] = max(courant["vram_pic_mo"] or 0.0, vram)
    return pics


class SondeMemoire:
    """Sonde persistée. Usage :

        sonde = SondeMemoire(chemin, cp=cp)
        with sonde.phase("rederive", "seed=103"):
            ...

    Le fil d'échantillonnage ne tourne QUE pendant une phase."""

    def __init__(self, chemin: str | Path, cp=None, periode_s: float = 0.25):
        self.chemin = Path(chemin)
        self.chemin.parent.mkdir(parents=True, exist_ok=True)
        self._cp = cp
        self._periode = periode_s
        self._pic = 0.0
        self._arret: threading.Event | None = None
        self._fil: threading.Thread | None = None
        self._echantillons = 0
        self.dernier_pic_mo = 0.0

    # ── écriture ──────────────────────────────────────────────────────────
    def _ecrire(self, **champs) -> None:
        """Une ligne, flushée ET fsyncée : sans fsync, le tampon de l'OS
        emporterait le dernier jalon avec le processus."""
        utilise, reserve = _vram_mo(self._cp)
        jalon = {"t": time.time(), "rss_mo": round(rss_mo(), 1),
                 "rss_pic_mo": round(self._pic, 1),
                 "vram_pool_mo": None if reserve is None else round(reserve, 1),
                 "vram_utilise_mo": (None if utilise is None
                                     else round(utilise, 1)), **champs}
        with open(self.chemin, "a", encoding="utf-8") as f:
            f.write(json.dumps(jalon, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    # ── échantillonnage ───────────────────────────────────────────────────
    def _boucle(self, bras: str, phase: str) -> None:
        while self._arret is not None and not self._arret.is_set():
            self._pic = max(self._pic, rss_mo())
            self._echantillons += 1
            self._ecrire(bras=bras, phase=phase, etat="vivant")
            self._arret.wait(self._periode)

    def attendre_un_echantillon(self, timeout_s: float = 5.0) -> None:
        """Bloque jusqu'à ce qu'au moins un battement de plus soit persisté
        (rend les tests déterministes sans dormir à l'aveugle)."""
        cible = self._echantillons + 2
        limite = time.time() + timeout_s
        while self._echantillons < cible and time.time() < limite:
            time.sleep(self._periode / 4)

    @contextmanager
    def phase(self, bras: str, phase: str):
        """Encadre une phase : début persisté AVANT le travail, battements
        pendant, fin (exception comprise) après."""
        self._pic = rss_mo()
        self._echantillons = 0
        self._ecrire(bras=bras, phase=phase, etat="debut")
        self._arret = threading.Event()
        self._fil = threading.Thread(target=self._boucle, args=(bras, phase),
                                     daemon=True)
        self._fil.start()
        exception = None
        try:
            yield self
        except BaseException as exc:            # noqa: BLE001 — on RE-lève
            exception = type(exc).__name__
            raise
        finally:
            self._arret.set()
            self._fil.join(timeout=2.0)
            self._pic = max(self._pic, rss_mo())
            self.dernier_pic_mo = self._pic
            self._ecrire(bras=bras, phase=phase, etat="fin",
                         exception=exception)
            self._arret, self._fil = None, None
