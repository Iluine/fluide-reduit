"""F1 tranche-1 — composant (c) : compteurs PCIe + bande passante à vide.

Protocole gravé (§A15 M-c) : « Compter les octets/frame réels du trafic
CPU<->GPU : descente (prédiction Harten des fenêtres actives) + remontée
(coefficients de détail). » Vérif d'instrument #4 : « Bande passante PCIe
soutenable MESURÉE à vide (transfert brut aller/retour) — c'est le
dénominateur de M-c, mesuré AVANT de mesurer le schéma diff. »

`TransfertComptable` est LA porte unique des transferts du harnais : CuPy ne
transfère RIEN implicitement (raison du choix T6/E2), donc tout octet qui
traverse le PCIe passe par `descendre`/`remonter` — le comptage est exact
PAR CONSTRUCTION. Temps par transfert : cuda-Events (backend cupy) ;
`perf_counter` sous numpy (tests VM uniquement, B1 — le comptage d'OCTETS,
lui, est identique sous les deux backends, c'est lui que les tests
vérifient). AUCUN timestamp dans les bilans."""
from __future__ import annotations

import time

import numpy as np

from src.f1_gpu.backend import est_cupy, synchroniser
from src.f1_gpu.chrono import stats_ms

# Tailles de la vérif #4 (octets) : de 64 Ko à 32 Mo — encadre le trafic
# attendu du schéma diff (colonnes ~10-200 Ko) ET le plein-fovéa (8.4 Mo).
TAILLES_A_VIDE: tuple[int, ...] = tuple(
    64 * 1024 * (4 ** i) for i in range(5))  # 64 Ko .. 16 Mo
REPETITIONS_A_VIDE: int = 30


class TransfertComptable:
    """Compteur H2D/D2H par direction et par frame (composant (c)).

    Usage : `descendre(tableau_cpu) -> tableau_gpu` (H2D),
    `remonter(tableau_gpu) -> tableau_cpu` (D2H), puis `frame_suivante()`
    clôt le bilan de la frame courante et l'empile dans `bilans`. Le bilan
    d'une frame : octets, temps (ms) et nombre d'appels, PAR DIRECTION."""

    def __init__(self, xp):
        self._xp = xp
        self._cupy = est_cupy(xp)
        if self._cupy:
            self._ev_debut = xp.cuda.Event()
            self._ev_fin = xp.cuda.Event()
        self.bilans: list[dict] = []
        self._courant = self._bilan_vide()

    @staticmethod
    def _bilan_vide() -> dict:
        return {"h2d_octets": 0, "d2h_octets": 0, "h2d_ms": 0.0,
                "d2h_ms": 0.0, "h2d_n": 0, "d2h_n": 0}

    def _chronometrer(self, operation):
        """(résultat, ms) de l'opération de transfert — cuda-Events sous
        cupy (l'event de fin synchronise : le transfert est mesuré entier),
        perf_counter sous numpy (tests, B1)."""
        if self._cupy:
            self._ev_debut.record()
            resultat = operation()
            self._ev_fin.record()
            self._ev_fin.synchronize()
            ms = float(self._xp.cuda.get_elapsed_time(
                self._ev_debut, self._ev_fin))
        else:
            t0 = time.perf_counter()
            resultat = operation()
            ms = (time.perf_counter() - t0) * 1e3
        return resultat, ms

    def descendre(self, tableau_cpu: np.ndarray):
        """H2D : copie explicite vers le device (`xp.asarray` sous cupy).
        Compte `nbytes` et le temps dans la direction descente."""
        tableau_cpu = np.ascontiguousarray(tableau_cpu)
        resultat, ms = self._chronometrer(
            lambda: self._xp.asarray(tableau_cpu))
        self._courant["h2d_octets"] += int(tableau_cpu.nbytes)
        self._courant["h2d_ms"] += ms
        self._courant["h2d_n"] += 1
        return resultat

    def remonter(self, tableau_gpu, hote_out: np.ndarray | None = None) -> np.ndarray:
        """D2H : rapatriement explicite. Compte `nbytes` et le temps dans la
        direction remontée.

        `hote_out` (review 28/07, M10) : sans lui, `.get()` alloue un tampon
        hôte PAGEABLE neuf à chaque appel — un « d2h pinned » qui passerait
        par là ne mesurerait PAS un transfert vers mémoire pinned (c'était le
        cas de la vérif d'instrument #4 : étiquette fausse, biais
        conservateur). Fournir le tampon page-locked ici (`get(out=...)`)
        rend l'étiquette vraie."""
        if self._cupy:
            if hote_out is not None:
                resultat, ms = self._chronometrer(
                    lambda: tableau_gpu.get(out=hote_out))
            else:
                resultat, ms = self._chronometrer(tableau_gpu.get)
        elif hote_out is not None:
            def _copie():
                np.copyto(hote_out, tableau_gpu)
                return hote_out
            resultat, ms = self._chronometrer(_copie)
        else:
            resultat, ms = self._chronometrer(
                lambda: np.array(tableau_gpu, copy=True))
        self._courant["d2h_octets"] += int(tableau_gpu.nbytes)
        self._courant["d2h_ms"] += ms
        self._courant["d2h_n"] += 1
        return resultat

    def frame_suivante(self) -> dict:
        """Clôt le bilan de la frame courante, l'empile et le renvoie."""
        bilan = self._courant
        self.bilans.append(bilan)
        self._courant = self._bilan_vide()
        return bilan

    def stats_regime(self, exclure_premiers: int = 0) -> dict:
        """Statistiques du régime établi sur `bilans[exclure_premiers:]`
        (le driver exclut warmup/descente initiale, B4) : médiane et p99
        des octets et du temps par frame, par direction + total."""
        bilans = self.bilans[exclure_premiers:]
        if not bilans:
            raise ValueError("stats_regime : aucun bilan de frame à agréger.")
        stats: dict = {"n_frames": len(bilans)}
        for cle in ("h2d_octets", "d2h_octets"):
            serie = np.asarray([b[cle] for b in bilans], dtype=np.float64)
            stats[cle] = {"mediane": float(np.percentile(serie, 50)),
                          "p99": float(np.percentile(serie, 99))}
        for cle in ("h2d_ms", "d2h_ms"):
            stats[cle] = stats_ms([b[cle] for b in bilans])
        total_ms = [b["h2d_ms"] + b["d2h_ms"] for b in bilans]
        stats["transfert_total_ms"] = stats_ms(total_ms)
        total_octets = np.asarray(
            [b["h2d_octets"] + b["d2h_octets"] for b in bilans],
            dtype=np.float64)
        stats["transfert_total_octets"] = {
            "mediane": float(np.percentile(total_octets, 50)),
            "p99": float(np.percentile(total_octets, 99))}
        return stats


def mesurer_bande_passante_a_vide(cp, tailles: tuple[int, ...] = TAILLES_A_VIDE,
                                  repetitions: int = REPETITIONS_A_VIDE) -> dict:
    """Vérif d'instrument #4 (§A15) : transferts BRUTS aller/retour à vide,
    par taille, mémoire PAGEABLE et PINNED, médiane ET p99 (B6) — le
    dénominateur de M-c, mesuré AVANT le schéma diff. Backend cupy
    OBLIGATOIRE (une bande passante numpy n'aurait aucun sens).

    Retourne {"tailles": [{"octets", "pageable": {h2d, d2h}, "pinned":
    {h2d, d2h}}, ...]} où chaque direction porte stats_ms + "go_par_s"
    (médiane). AUCUN timestamp."""
    if not est_cupy(cp):
        raise RuntimeError(
            "mesurer_bande_passante_a_vide : backend cupy obligatoire (B1).")
    comptable = TransfertComptable(cp)
    resultats: list[dict] = []
    for taille in tailles:
        hote_pageable = np.zeros(taille, dtype=np.uint8)
        # Tampon hôte PINNED : allocation page-locked cupy, vue numpy dessus.
        memoire_pinned = cp.cuda.alloc_pinned_memory(taille)
        hote_pinned = np.frombuffer(memoire_pinned, dtype=np.uint8, count=taille)
        entree: dict = {"octets": int(taille)}
        for nom, hote in (("pageable", hote_pageable), ("pinned", hote_pinned)):
            h2d_ms: list[float] = []
            d2h_ms: list[float] = []
            synchroniser(cp)
            for _ in range(repetitions):
                device = comptable.descendre(hote)
                comptable.frame_suivante()
                h2d_ms.append(comptable.bilans[-1]["h2d_ms"])
                # M10 : le d2h « pinned » atterrit dans LE tampon pinned
                # (`get(out=...)`) — avant, `.get()` allouait un pageable
                # neuf et l'étiquette était fausse. Le pageable garde son
                # chemin historique (allocation comprise, comme en prod).
                comptable.remonter(device,
                                   hote_out=(hote if nom == "pinned" else None))
                comptable.frame_suivante()
                d2h_ms.append(comptable.bilans[-1]["d2h_ms"])
                del device
            entree[nom] = {
                "h2d": _stats_direction(taille, h2d_ms),
                "d2h": _stats_direction(taille, d2h_ms)}
        resultats.append(entree)
    return {"repetitions": repetitions, "tailles": resultats}


def _stats_direction(octets: int, serie_ms: list[float]) -> dict:
    """stats_ms + débit médian en Go/s (Go = 1e9 octets, nommé)."""
    stats = stats_ms(serie_ms)
    stats["go_par_s"] = float(octets / (stats["mediane_ms"] * 1e-3) / 1e9)
    return stats
