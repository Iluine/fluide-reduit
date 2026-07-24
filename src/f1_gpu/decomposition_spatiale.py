"""F1 — M-b : décomposition SPATIALE de Δχ (§A32).

DIAGNOSTIC sur données déjà acquises. Le verdict MORT-b n'est pas rouvert :
ni l'observable ni le critère ne bougent. La question est étroite — l'écart
0,70–0,87 vient-il de la FOVÉA (25 % de l'aire, pleine résolution) ou de la
PÉRIPHÉRIE (75 %, décimée 2×) ?

────────────────────────────────────────────────────────────────────────
CE QU'ON NE FAIT PAS, ET POURQUOI — le piège §A14
────────────────────────────────────────────────────────────────────────
Restreindre `delta_chi` à une sous-fenêtre est illégitime, deux fois :

1. **Le complément n'est pas rectangulaire.** La fovéa est une fenêtre 32²
   dans un 64² : son complément est une couronne en L. `chi_bands` fait une
   `fft2` et exige un tableau rectangulaire — carré, même, puisque
   `_radial_bins` suppose H = W. Zéro-remplir la couronne changerait
   `mean_a`, qui DIVISE χ : le nombre obtenu ne mesurerait plus rien.
2. **Même la fovéa seule change de base.** À 32², la dernière bande
   `(16, ∞)` s'arrête au Nyquist 22,6 au lieu de 45,25. Les bandes ne sont
   plus les mêmes, et les PORTEUSES — définies par la puissance de la
   référence — seraient redéfinies sur un autre domaine. Comparer un Δχ à
   32² et un Δχ à 64² comparerait deux instruments.

────────────────────────────────────────────────────────────────────────
CE QU'ON FAIT : masquer LA DIFFÉRENCE, pas le domaine
────────────────────────────────────────────────────────────────────────
Le candidat hybride vaut la production DANS la région étudiée et le rederive
AILLEURS, sur le domaine PLEIN :

    hybride = rederive + (production − rederive) · 1_région

L'observable est appelé tel quel, sur 64², contre la MÊME référence. Trois
conséquences qui font toute la légitimité :
  - même domaine ⇒ mêmes bandes radiales ;
  - même référence ⇒ **bandes PORTEUSES IDENTIQUES** dans les trois
    évaluations (elles ne dépendent que de la référence) — c'est exactement
    ce que la restriction détruisait ;
  - masque plein ⇒ on retombe au bit près sur le Δχ déjà mesuré (testé).

RÉSERVE, à porter avec le résultat : masquer la différence introduit une
discontinuité au bord de la fovéa, et une discontinuité FUIT en spectre. Les
deux Δχ partiels ne s'additionnent donc pas au Δχ plein — Δχ n'est de toute
façon pas additif (rapports de RMS). Ils se LISENT l'un contre l'autre, pas
en somme.

D'où un SECOND REGARD, sans artefact : la partition d'énergie de la
différence, EXACTEMENT additive (‖d‖²_fovéa + ‖d‖²_périph = ‖d‖²), normalisée
par cellule puisque la fovéa ne fait que 25 % de l'aire. C'est un diagnostic,
pas le critère.

La BANDE DOMINANTE est reportée aussi : si le max porteur tombe sur la bande
la plus fine, c'est la signature d'une décimation 2×."""
from __future__ import annotations

import numpy as np

from scripts.run_arcA_revalidate import S_HALF_OP
from src.albedo import albedo, delta_chi
from src.f1_gpu.cellule_mb import dchi


def masque_fenetre(n: int, oy: int, ox: int, cote: int) -> np.ndarray:
    """Masque booléen (n, n) vrai sur la fenêtre [oy, oy+cote) × [ox, ox+cote)."""
    m = np.zeros((n, n), dtype=bool)
    m[oy:oy + cote, ox:ox + cote] = True
    return m


def champ_hybride(candidat, reference, masque) -> np.ndarray:
    """`candidat` dans le masque, `reference` dehors — sur le domaine PLEIN.
    Revient à masquer la DIFFÉRENCE, jamais le domaine."""
    candidat = np.asarray(candidat, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    return np.where(masque, candidat, reference)


def _detail_bandes(candidat, reference) -> dict:
    """Δχ par bande + porteuses, via les primitives EXISTANTES."""
    r = delta_chi(albedo(np.asarray(candidat, dtype=np.float64), S_HALF_OP),
                  albedo(np.asarray(reference, dtype=np.float64), S_HALF_OP))
    return {"par_bande": dict(r["delta_chi"]), "porteuses": dict(r["carriers"]),
            "max_carrier": float(r["max_carrier"])}


def decomposer(candidat, reference, masque) -> dict:
    """Δχ PLEIN, Δχ FOVÉA-SEULE et Δχ PÉRIPHÉRIE-SEULE, tous sur 64² contre
    la même référence — donc sous des porteuses identiques."""
    fovea = champ_hybride(candidat, reference, masque)
    peripherie = champ_hybride(candidat, reference, ~np.asarray(masque))
    plein_d = _detail_bandes(candidat, reference)
    fovea_d = _detail_bandes(fovea, reference)
    periph_d = _detail_bandes(peripherie, reference)
    porteuses = plein_d["porteuses"]
    candidates = [b for b, p in porteuses.items() if p]
    dominante = (max(candidates, key=lambda b: plein_d["par_bande"][b])
                 if candidates else None)
    return {
        "dchi_plein": dchi(candidat, reference),
        "dchi_fovea": fovea_d["max_carrier"],
        "dchi_peripherie": periph_d["max_carrier"],
        "bande_dominante": dominante,
        "par_bande_plein": plein_d["par_bande"],
        "par_bande_fovea": fovea_d["par_bande"],
        "par_bande_peripherie": periph_d["par_bande"],
        "porteuses": porteuses,
        "porteuses_fovea": fovea_d["porteuses"],
        "porteuses_peripherie": periph_d["porteuses"],
    }


def fidelite_structurelle(candidat, reference) -> dict:
    """DISCRIMINANT contre l'hypothèse « corruption » (§A32) : l'amplitude
    0,70–0,87 tombe dans la plage du contrôle de corruption délibérée
    (shuf-commit). Un champ mélangé est DÉCORRÉLÉ de la vérité ; un champ
    structurellement juste mais mal amplifié reste fortement corrélé.

    Corrélation de Pearson et erreur RMS relative — hors de l'espace
    instrument, donc DIAGNOSTIC seul, jamais le critère."""
    c = np.asarray(candidat, dtype=np.float64).ravel()
    r = np.asarray(reference, dtype=np.float64).ravel()
    dc, dr = c - c.mean(), r - r.mean()
    denom = float(np.sqrt((dc ** 2).sum() * (dr ** 2).sum()))
    correlation = float((dc * dr).sum() / denom) if denom > 0 else 0.0
    norme_r = float(np.sqrt((r ** 2).sum()))
    return {
        "correlation": correlation,
        "erreur_rms_relative": (float(np.sqrt(((c - r) ** 2).sum())) / norme_r
                                if norme_r > 0 else float("inf")),
        "moyenne_candidat": float(c.mean()),
        "moyenne_reference": float(r.mean()),
    }


def partition_energie(candidat, reference, masque) -> dict:
    """Partition EXACTEMENT additive de l'énergie de la différence — le
    second regard, sans artefact de bord. Normalisée par cellule : la fovéa
    ne fait que 25 % de l'aire, et comparer des énergies brutes favoriserait
    mécaniquement la périphérie."""
    d = (np.asarray(candidat, dtype=np.float64)
         - np.asarray(reference, dtype=np.float64)) ** 2
    masque = np.asarray(masque, dtype=bool)
    e_f = float(d[masque].sum())
    e_p = float(d[~masque].sum())
    a_f, a_p = int(masque.sum()), int((~masque).sum())
    total = e_f + e_p
    return {
        "energie_fovea": e_f, "energie_peripherie": e_p,
        "energie_totale": total,
        "aire_fovea": a_f, "aire_peripherie": a_p,
        "energie_par_cellule_fovea": e_f / a_f if a_f else 0.0,
        "energie_par_cellule_peripherie": e_p / a_p if a_p else 0.0,
        "fraction_fovea": e_f / total if total > 0 else 0.0,
    }
