"""F1 — M-b tranche-2 : le BRAS PRODUCTION (§A31-build).

Le bras production est le pipeline TEL QU'IL VIVRAIT : fovéation 2-niveaux
(le grossier 64² décimé, murs réfléchissants au vrai bord ; la fovéa mobile
E4c à pleine résolution, bord HALO-PARENT rafraîchi entre étages RK2), Exner
à la cadence LUE, puis remontée L3 SEUILLÉE à EPS 1e-2 vers la connaissance
du CPU (option 1 : la `reference`, jamais une copie qui puisse diverger).

Il porte donc les TROIS sources d'écart : (a) f32, (b) le seuil EPS, (c) la
structure fovéale. Le bras témoin (`temoin_fidele`) n'en porte qu'une, (a) —
c'est leur DIFFÉRENCE qui rend la lecture à trois branches décidable, et
c'est pourquoi les deux tournent sur la MÊME cellule, avec le MÊME
observable Δχ (§A31-build, apples-to-apples).

INVARIANT LIANT : aucun kernel neuf, rien de neuf dans `pas_f_fidele`. Ce
module ORCHESTRE `pas_deux_niveaux` (qui orchestre lui-même le kernel figé
de C1, empreinte 6dd207ca). La cadence Exner est LUE de `save_every`.

────────────────────────────────────────────────────────────────────────
CHOIX D'IMPLÉMENTATION NON COUVERTS PAR LE GRAVÉ — remontés, pas préemptés
────────────────────────────────────────────────────────────────────────
1. **Cadence de remontée = UNE PAR ÉPISODE.** Le gravé fixe les émissions
   (Δt = 4) mais pas la cadence du canal. L'épisode est l'unité atomique de
   la cellule (b_eff est GELÉ pendant un épisode, le sédiment n'a de sens
   qu'entre épisodes) : remonter une fois par épisode fait mordre le seuil
   4 fois entre deux émissions, ce qu'une remontée par émission diluerait.
2. **Où mord le cap 10 %.** Le gravé donne « k_fen aire-proportionnel,
   cap 10 % » pour le commit fenêtré. Dans CE pipeline le commit est
   l'émission L3 : le cap borne le nombre de détails transportés, et les
   plus GROS |Δ| passent en premier. C'est la lecture qui rend le cap
   load-bearing sur la fidélité plutôt que décoratif.
3. **dt en lockstep = min(CFL fin, CFL grossier).** Les deux niveaux
   avancent d'un même dt ; le prendre au plus contraint des deux est
   conservateur pour les deux (le grossier, à dx = 2, a une limite ~2× plus
   large). Aucune sous-cyclage : le mipmap gravé (§A28) donne M = 1.
4. **Échelle de lecture = le 64² PLEIN.** Le pré-enregistrement §4 disait
   « échelle décimée, comme la sonde » — écrit AVANT que la réconciliation
   2-niveaux soit endossée, et visant le grossier-contre-grossier de la
   sonde. Ici la connaissance du CPU est un champ 64² COMPOSÉ (fovéa fine +
   grossier upsamplé) et le rederive est un 64² plein : les deux ne sont
   définis qu'à cette échelle, qui est aussi la plus SÉVÈRE.
5. **Géométrie de la fovéa** : côté 32 (le grossier reste non trivial),
   ordonnée fixe centrée, abscisse mobile de 1 cellule/épisode — ce qui
   RÉALISE la portée déjà gravée dans la lecture (~24 cellules ≈ un tiers
   du domaine sur les 6 émissions)."""
from __future__ import annotations

import numpy as np

from src.f1_gpu.cellule_mb import budget_k_fen
from src.f1_gpu.exner_gpu import CADENCE_EXNER, pas_exner
from src.f1_gpu.fovea_2niveaux import DECIMATION, decimer, pas_deux_niveaux
from src.f1_gpu.substrat_fidele import reduction_cfl_fidele
from src.f1_gpu.temoin_fidele import _pulse_gpu
from src.sediment import SedimentParams, _T_END_RELAX

# Seuil de remontée L3 de production, gravé (§A23 puis reconduit §A31).
EPS_PRODUCTION: float = 1e-2

# Géométrie de la fovéa mobile E4c sur le 64² (choix 5 du docstring).
N_FOVEA: int = 32
OY_FOVEA: int = 16                     # ordonnée fixe, centrée
OX_FOVEA_0: int = 4                    # abscisse de départ
PAS_FOVEA: int = 1                     # cellules par épisode


def position_fovea(episode: int, n: int = 64) -> tuple[int, int]:
    """Position (oy, ox) de la fovéa à l'épisode `episode` (0-indexé).
    Mobile E4c : l'abscisse avance de `PAS_FOVEA` par épisode, bornée au
    domaine (la fovéa ne sort jamais du 64²)."""
    ox = OX_FOVEA_0 + PAS_FOVEA * episode
    return OY_FOVEA, int(min(max(ox, 0), n - N_FOVEA))


def composer_plein(q_coarse, q_fine, oy: int, ox: int, cp):
    """Le champ 64² tel que le système le CONNAÎT : grossier upsamplé ×2
    partout, fovéa fine collée à (oy, ox). C'est exactement la source (c) —
    hors fovéa, la « vérité » du système est une interpolation du parent."""
    plein = cp.repeat(cp.repeat(q_coarse, DECIMATION, axis=-1),
                      DECIMATION, axis=-2)
    n_f = int(q_fine.shape[-1])
    plein[:, :, :, oy:oy + n_f, ox:ox + n_f] = q_fine
    return cp.ascontiguousarray(plein)


def remonter_seuillee(champ, reference, eps: float, budget: int) -> int:
    """Remontée L3 SEUILLÉE, schéma incrémental B4, côté connaissance CPU
    (option 1) : ne transporte que les écarts |champ − reference| ≥ `eps`,
    au plus `budget` d'entre eux — les plus GROS d'abord (cap 10 %, choix 2).
    Met à jour `reference` EN PLACE, uniquement là où le détail a été
    transporté. Retourne le nombre de détails transportés.

    Aucune physique CPU ici : le CPU applique ce qu'il REÇOIT. Le miroir CPU
    (option 2) reste interdit."""
    d = np.asarray(champ, dtype=np.float32) - reference
    plats = np.flatnonzero(np.abs(d).ravel() >= eps)
    if plats.size == 0:
        return 0
    if plats.size > budget:
        amplitudes = np.abs(d.ravel()[plats])
        plats = plats[np.argsort(amplitudes)[::-1][:budget]]
    reference.ravel()[plats] += d.ravel()[plats]
    return int(plats.size)


def evoluer_episode_production(s, b0, centre_frac, cp, oy: int, ox: int,
                               params: SedimentParams = SedimentParams(),
                               n_settle: int | None = None):
    """Un épisode du bras PRODUCTION : même structure que le témoin (pulse ->
    relaxation -> Exner sur snapshots à la cadence LUE), mais la relaxation
    tourne sur la fovéation 2-niveaux au lieu du domaine plein.

    `s`, `b0` numpy f32 (H, W) ; retourne le nouveau `s` numpy f32 (H, W)."""
    n_settle = params.N_settle if n_settle is None else n_settle
    H, W = b0.shape
    b_eff = cp.asarray((b0 + s).astype(np.float32).reshape(1, 1, H, W))
    q = _pulse_gpu((H, W), centre_frac, params, cp)

    # décomposition 2-niveaux de l'état initial de l'épisode
    q_c = decimer(q, cp)
    b_c = decimer(b_eff.reshape(1, 1, 1, H, W), cp)[:, :, 0]
    q_f = cp.ascontiguousarray(q[:, :, :, oy:oy + N_FOVEA, ox:ox + N_FOVEA])
    b_f = cp.ascontiguousarray(b_eff[:, :, oy:oy + N_FOVEA, ox:ox + N_FOVEA])

    t = 0.0
    snapshots = [composer_plein(q_c, q_f, oy, ox, cp)]
    temps = [0.0]
    for _ in range(n_settle):
        # lockstep : le dt du plus contraint des deux niveaux (choix 3)
        dt = min(float(reduction_cfl_fidele(q_f, cp)),
                 float(reduction_cfl_fidele(q_c, cp)))
        if t + dt > _T_END_RELAX:
            dt = _T_END_RELAX - t
        q_c, q_f = pas_deux_niveaux(q_c, b_c, q_f, b_f, oy, ox, cp, dt)
        t += dt
        snapshots.append(composer_plein(q_c, q_f, oy, ox, cp))
        temps.append(t)

    s_dev = cp.asarray(s.astype(np.float32).reshape(1, 1, H, W))
    for i in range(CADENCE_EXNER, n_settle + 1, CADENCE_EXNER):
        dt_exner = float(temps[i] - temps[i - CADENCE_EXNER])
        pas_exner(snapshots[i], s_dev, cp, dt_exner)
    return cp.asnumpy(s_dev)[0, 0]


def evoluer_histoire_production(seed: int, n_episodes: int, b0, cp, centres,
                                params: SedimentParams = SedimentParams(),
                                checkpoints: set[int] | None = None,
                                eps: float = EPS_PRODUCTION,
                                budget: int | None = None,
                                rendre_champ: bool = False) -> dict:
    """L'histoire du bras PRODUCTION. Centres FOURNIS (ceux du rederive, cf.
    `cellule_mb.centres_cellule`) — la production ne les tire pas.

    Retourne {n: connaissance_cpu} aux checkpoints : ce que le CPU SAIT,
    c'est-à-dire la `reference` accumulée par la remontée seuillée — PAS le
    champ live. C'est l'objet que Δχ compare au rederive.

    `budget` : places de la remontée. Par défaut celui de la cellule
    (aire-proportionnel, cap 10 %). LE FORCER EST NÉCESSAIRE pour le bras
    EPS = 0 : à seuil nul, `|d| >= 0` est vrai PARTOUT, donc H·W candidats se
    présentent pour 10 % de places et c'est le CAP qui morde, pas le seuil —
    le « plancher » mesurerait le cap. Un canal vraiment transparent exige
    `budget >= H·W`.

    `rendre_champ` : rend le champ `s` LIVE au lieu de la connaissance.
    Sert à VÉRIFIER la transparence du canal (à EPS=0 et budget plein, les
    deux doivent coïncider), jamais à mesurer — Δχ se lit toujours sur ce
    que le CPU SAIT."""
    if checkpoints is None:
        checkpoints = {n_episodes}
    H, W = b0.shape
    budget = budget_k_fen(H * W) if budget is None else int(budget)
    s = np.zeros_like(b0, dtype=np.float32)
    connaissance = np.zeros_like(b0, dtype=np.float32)   # option 1
    out: dict[int, np.ndarray] = {}
    for ep in range(1, n_episodes + 1):
        oy, ox = position_fovea(ep - 1, n=H)
        s = evoluer_episode_production(s, b0, centres[ep - 1], cp, oy, ox,
                                       params)
        remonter_seuillee(s, connaissance, eps=eps, budget=budget)
        if ep in checkpoints:
            out[ep] = s.copy() if rendre_champ else connaissance.copy()
    return out
