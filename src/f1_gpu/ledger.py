"""F1 tranche-1 — composant (d) : ledger v1 (format §4 de la spec) —
écriture append-only, parse+validation intégrale, load E7 (choix B7, B8).

Format gravé (SPEC-FOVEA-Z §4, ENDOSSÉE) : séquence append-only (H1)
d'entrées à ordre TOTAL, clé = (t_sim quantifié, numéro de séquence),
mono-écrivain v1. Types : (a) événement seedé {t_sim, seed,
type_procédural} ; (b) entrée-joueur {t_sim, payload, id_observateur ≡ 0,
champ réservé v1.1} ; (c) commit d'émission {t_sim, fenêtre (niveau, rect
aligné-dyadique), SummaryQT (topologie u8, moyennes f64), k} ;
(d) snapshot-racine {t_sim, commit PLEIN sans perte}.

B7 — sérialisation : un répertoire {`index.jsonl` (une ligne JSON par
entrée : type, t_sim, seq, meta, descripteurs de payloads) + `data.bin`
(payloads binaires concaténés)}, les DEUX strictement append-only.
Descripteur de payload : {nom, offset, taille, sha256, dtype, shape} —
le sha256 couvre `dtype|shape|octets` (l'interprétation est protégée
comme le contenu, cf. `_empreinte_payload`).
La clé (t_sim, seq) est strictement croissante — garantie à l'APPEND
(RuntimeError sinon, H1 côté API) et RE-VÉRIFIÉE au parse.

B8 — load E7 (gravé §A15-complément) : « parse+validation INTÉGRALE du
ledger + regenerate(dernier commit) + replay du segment courant + reprise
du vivant ». Consignes E7 attachées : 30 s borne le coût du LEDGER (parse,
intégrité, regenerate), pas celui de la physique — dit en face. Le DEHORS
de la fenêtre n'est pas reconstruit (éphémère, §3). L'évoluteur du segment
courant est INJECTÉ (production : `run_episode` via le driver ; tests :
évoluteur trivial) — le cœur manche 2 reste intouché, consommé par import.

Le parse est INTÉGRAL par construction : chaque payload est relu du disque
et son sha256 recalculé, chaque commit est reconstruit et re-gardé au
budget (`size_floats_qt <= k`) — c'est ce coût-là que MORT-d chronomètre."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.summary_quadtree import SummaryQT, regenerate_qt, size_floats_qt

NOM_INDEX: str = "index.jsonl"
NOM_DATA: str = "data.bin"

TYPES_ENTREES: tuple[str, ...] = (
    "evenement_seede", "entree_joueur", "commit_emission", "snapshot_racine")


def _empreinte_payload(dtype: str, shape, octets: bytes) -> str:
    """sha256 du payload, EN-TÊTE D'INTERPRÉTATION INCLUS (revue
    adversariale 2026-07-19) : hasher les octets seuls laisserait passer
    une corruption de `dtype`/`shape` de même taille dans l'index (ex.
    float64 relu int64 — bits identiques, valeurs aberrantes, aucun
    contrôle déclenché). L'empreinte couvre donc `dtype|shape|octets` :
    toute réinterprétation devient détectable au parse."""
    entete = f"{dtype}|{[int(v) for v in shape]}|".encode()
    return hashlib.sha256(entete + octets).hexdigest()


class EcrivainLedger:
    """Écrivain append-only (H1 côté API : aucune méthode de mutation ni de
    troncature n'existe). `creer=True` exige un répertoire vierge ;
    `creer=False` REPREND un ledger existant (relecture de l'index pour la
    dernière clé — le run de nuit E6 est interruptible/reprenable)."""

    def __init__(self, repertoire: str | Path, creer: bool = False):
        self.repertoire = Path(repertoire)
        self._chemin_index = self.repertoire / NOM_INDEX
        self._chemin_data = self.repertoire / NOM_DATA
        if creer:
            if self._chemin_index.exists() or self._chemin_data.exists():
                raise RuntimeError(
                    f"EcrivainLedger : {self.repertoire} contient déjà un "
                    "ledger (append-only H1 : jamais d'écrasement).")
            self.repertoire.mkdir(parents=True, exist_ok=True)
            self._chemin_index.touch()
            self._chemin_data.touch()
            self._derniere_cle = (-1, -1)
            self._offset = 0
        else:
            if not self._chemin_index.exists():
                raise FileNotFoundError(
                    f"EcrivainLedger : pas de ledger dans {self.repertoire} "
                    "(creer=False = reprise d'un ledger existant).")
            lignes = self._chemin_index.read_text(
                encoding="utf-8").splitlines()
            if lignes:
                derniere = json.loads(lignes[-1])
                self._derniere_cle = (int(derniere["t_sim"]),
                                      int(derniere["seq"]))
            else:
                self._derniere_cle = (-1, -1)
            self._offset = self._chemin_data.stat().st_size

    @property
    def derniere_cle(self) -> tuple[int, int]:
        return self._derniere_cle

    def _ajouter(self, type_: str, t_sim: int, meta: dict,
                 payloads: dict[str, np.ndarray]) -> dict:
        if type_ not in TYPES_ENTREES:
            raise ValueError(
                f"EcrivainLedger : type inconnu {type_!r} "
                f"(attendu un de {TYPES_ENTREES}).")
        seq = self._derniere_cle[1] + 1
        cle = (int(t_sim), seq)
        if cle <= self._derniere_cle:
            raise RuntimeError(
                f"EcrivainLedger : clé {cle} <= dernière {self._derniere_cle}"
                " — la clé (t_sim, seq) doit croître strictement (H1/B7).")
        descripteurs = []
        with self._chemin_data.open("ab") as flux_data:
            for nom, tableau in payloads.items():
                tableau = np.ascontiguousarray(tableau)
                octets = tableau.tobytes()
                flux_data.write(octets)
                descripteurs.append({
                    "nom": nom, "offset": self._offset,
                    "taille": len(octets),
                    "sha256": _empreinte_payload(
                        str(tableau.dtype), tableau.shape, octets),
                    "dtype": str(tableau.dtype),
                    "shape": list(tableau.shape)})
                self._offset += len(octets)
            flux_data.flush()
        entree = {"type": type_, "t_sim": int(t_sim), "seq": seq,
                  "meta": meta, "payloads": descripteurs}
        with self._chemin_index.open("a", encoding="utf-8") as flux_index:
            flux_index.write(json.dumps(entree, ensure_ascii=False) + "\n")
            flux_index.flush()
        self._derniere_cle = cle
        return entree

    def ajouter_evenement(self, t_sim: int, graine: int,
                          type_procedural: str) -> dict:
        """Entrée (a) — événement seedé (format §4, [MESURÉ])."""
        return self._ajouter("evenement_seede", t_sim,
                             {"graine": int(graine),
                              "type_procedural": type_procedural}, {})

    def ajouter_entree_joueur(self, t_sim: int, payload: dict,
                              id_observateur: int = 0) -> dict:
        """Entrée (b) — format D6, RÉSERVÉ v1 (jamais émis par le
        générateur E6 ; présent pour que le format v1 soit complet)."""
        return self._ajouter("entree_joueur", t_sim,
                             {"payload": payload,
                              "id_observateur": int(id_observateur)}, {})

    def ajouter_commit(self, t_sim: int, commit: SummaryQT, niveau: int,
                       rect: tuple[int, int, int], k: int) -> dict:
        """Entrée (c) — commit d'émission fenêtré : SummaryQT (topologie u8
        + moyennes f64) + fenêtre (niveau, rect aligné-dyadique
        (y0, x0, côté)) + budget k. Garde budget à l'écriture (défensive,
        re-vérifiée au parse)."""
        taille = size_floats_qt(commit)
        if taille > k:
            raise RuntimeError(
                f"ajouter_commit : commit hors budget ({taille} > k={k}).")
        return self._ajouter(
            "commit_emission", t_sim,
            {"niveau": int(niveau), "rect": [int(v) for v in rect],
             "k": int(k), "shape": list(commit.shape)},
            {"topologie": np.asarray(commit.topology, dtype=np.uint8),
             "moyennes": np.asarray(commit.means, dtype=np.float64)})

    def ajouter_snapshot(self, t_sim: int, champ: np.ndarray) -> dict:
        """Entrée (d) — snapshot-racine : commit PLEIN sans perte (le bras
        ferm mesuré §A13-4-1 en est l'appui)."""
        champ = np.asarray(champ, dtype=np.float64)
        return self._ajouter("snapshot_racine", t_sim,
                             {"shape": list(champ.shape)},
                             {"champ": champ})


def _reconstruire_commit(entree: dict) -> SummaryQT:
    """SummaryQT depuis les tableaux relus d'une entrée (c)."""
    return SummaryQT(topology=entree["tableaux"]["topologie"],
                     means=entree["tableaux"]["moyennes"],
                     shape=tuple(entree["meta"]["shape"]))


def parser_ledger(repertoire: str | Path) -> dict:
    """Parse + validation INTÉGRALE (B8 — la part « coût du ledger » de
    MORT-d) : toutes les lignes de l'index, clé (t_sim, seq) strictement
    croissante, offsets contigus couvrant exactement `data.bin`, CHAQUE
    payload relu + sha256 recalculé, chaque commit reconstruit + garde
    budget, chaque snapshot reconstruit + garde de shape. La moindre
    incohérence -> ValueError (fail-loud, jamais de lecture partielle).

    Retourne {"entrees": [entrée + "tableaux"], "n_par_type",
    "octets_data"} — AUCUN timestamp."""
    repertoire = Path(repertoire)
    lignes = (repertoire / NOM_INDEX).read_text(encoding="utf-8").splitlines()
    donnees = (repertoire / NOM_DATA).read_bytes()

    entrees: list[dict] = []
    n_par_type = {t: 0 for t in TYPES_ENTREES}
    cle_prec = (-1, -1)
    offset_attendu = 0
    for i, ligne in enumerate(lignes):
        entree = json.loads(ligne)
        if entree["type"] not in TYPES_ENTREES:
            raise ValueError(
                f"parser_ledger : type inconnu {entree['type']!r} (seq {i}).")
        cle = (int(entree["t_sim"]), int(entree["seq"]))
        if cle <= cle_prec:
            raise ValueError(
                f"parser_ledger : clé non croissante {cle} après {cle_prec}.")
        if entree["seq"] != i:
            raise ValueError(
                f"parser_ledger : seq {entree['seq']} != rang {i} (trou ou "
                "désordre dans l'index).")
        cle_prec = cle
        tableaux: dict[str, np.ndarray] = {}
        for desc in entree["payloads"]:
            if desc["offset"] != offset_attendu:
                raise ValueError(
                    f"parser_ledger : offset {desc['offset']} != attendu "
                    f"{offset_attendu} (payload {desc['nom']!r}, seq {i}).")
            octets = donnees[desc["offset"]:desc["offset"] + desc["taille"]]
            if len(octets) != desc["taille"]:
                raise ValueError(
                    f"parser_ledger : data.bin tronqué (seq {i}).")
            if _empreinte_payload(desc["dtype"], desc["shape"],
                                  octets) != desc["sha256"]:
                raise ValueError(
                    f"parser_ledger : sha256 invalide (payload "
                    f"{desc['nom']!r}, seq {i}) — ledger corrompu (octets "
                    "OU interprétation dtype/shape).")
            tableaux[desc["nom"]] = np.frombuffer(
                octets, dtype=np.dtype(desc["dtype"])).reshape(desc["shape"])
            offset_attendu += desc["taille"]
        entree["tableaux"] = tableaux
        if entree["type"] == "commit_emission":
            commit = _reconstruire_commit(entree)
            taille = size_floats_qt(commit)
            if taille > entree["meta"]["k"]:
                raise ValueError(
                    f"parser_ledger : commit hors budget au parse "
                    f"({taille} > k={entree['meta']['k']}, seq {i}).")
        if entree["type"] == "snapshot_racine":
            attendue = tuple(entree["meta"]["shape"])
            if tableaux["champ"].shape != attendue:
                raise ValueError(
                    f"parser_ledger : snapshot shape {tableaux['champ'].shape}"
                    f" != meta {attendue} (seq {i}).")
        n_par_type[entree["type"]] += 1
        entrees.append(entree)
    if offset_attendu != len(donnees):
        raise ValueError(
            f"parser_ledger : data.bin porte {len(donnees)} octets, l'index "
            f"n'en couvre que {offset_attendu} — ledger incohérent.")
    return {"entrees": entrees, "n_par_type": n_par_type,
            "octets_data": len(donnees)}


def charger_ledger(repertoire: str | Path, evoluer=None,
                   televerser=None) -> dict:
    """Le LOAD E7/B8 — la séquence que MORT-d chronomètre :
      1. parse + validation intégrale (`parser_ledger`) ;
      2. `regenerate_qt(dernier commit d'émission)` — l'état∣fenêtre
         post-ré-ancrage = f(dernier commit) seul (conséquence mécanique du
         ré-ancrage, E7 endossé ; le dehors est éphémère, non reconstruit) ;
      3. replay du SEGMENT COURANT : les événements seedés (a) de clé
         postérieure au dernier commit, appliqués via `evoluer(etat,
         entree)` (INJECTÉ — production : `run_episode` ; segment non vide
         sans évoluteur -> RuntimeError) ;
      4. reprise du vivant : `televerser(etat)` si fourni (H2D, compté
         dans le chrono de load par le driver).

    Retourne {"etat", "n_entrees", "n_par_type", "t_dernier_commit",
    "n_segment", "octets_data"}."""
    document = parser_ledger(repertoire)
    commits = [e for e in document["entrees"]
               if e["type"] == "commit_emission"]
    if not commits:
        raise RuntimeError(
            "charger_ledger : aucun commit d'émission — rien à re-dériver.")
    dernier = commits[-1]
    etat = regenerate_qt(_reconstruire_commit(dernier))

    cle_commit = (dernier["t_sim"], dernier["seq"])
    segment = [e for e in document["entrees"]
               if e["type"] == "evenement_seede"
               and (e["t_sim"], e["seq"]) > cle_commit]
    if segment and evoluer is None:
        raise RuntimeError(
            f"charger_ledger : segment courant de {len(segment)} événements "
            "sans évoluteur injecté (B8).")
    for evenement in segment:
        etat = evoluer(etat, evenement)

    if televerser is not None:
        televerser(etat)

    return {"etat": etat, "n_entrees": len(document["entrees"]),
            "n_par_type": document["n_par_type"],
            "t_dernier_commit": int(dernier["t_sim"]),
            "n_segment": len(segment),
            "octets_data": document["octets_data"]}
