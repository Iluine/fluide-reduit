"""F1 — M-b tranche-2 : CONSOMMER le rederive sans le modifier (§A31-run).

────────────────────────────────────────────────────────────────────────
LE POSTE, ÉTABLI PAR LA MESURE
────────────────────────────────────────────────────────────────────────
`_relax_episode` appelle `simulate_wetdry_o2(..., t_end=_T_END_RELAX=130)`,
qui accumule `hs, hus, hvs` à CHAQUE pas accepté (3 × 64² × f64 = 98 Ko/pas),
puis TRONQUE à `N_settle + 1 = 601` pas. Au seed 103 épisode 4 le pas CFL
s'effondre en cours d'épisode (dt médian 5,5e-3 à t = 64) : ~254 000 pas
matérialisés, **25 Go**, pour en retenir 601 — ~99,8 % calculé puis jeté.
C'est ce gâchis, et non la physique, qui rendait le seed incalculable.

────────────────────────────────────────────────────────────────────────
LE PROTOCOLE DE CONSOMMATION — le rederive reste INTOUCHÉ
────────────────────────────────────────────────────────────────────────
`t_end` ne gouverne QUE deux choses dans la boucle du solveur : l'arrêt
(`while t < t_end`) et l'écrêtage du dernier pas (`if t + dt > t_end`). Les
pas produits AVANT que `t` n'approche `t_end` n'en dépendent donc pas. D'où :

  **consommer le rederive avec le PLUS PETIT `t_end` de l'échelle qui
  atteint `N_settle` pas rend le MÊME résultat, pour une fraction de la
  mémoire.**

Vérifié empiriquement avant d'être codé (les 601 pas retenus sont
bit-identiques entre `t_end = 55` et `t_end = 64`), et gardé par un test
bit-exact contre `run_episode`/`run_history` tels quels.

L'ÉCHELLE PLAFONNE À `_T_END_RELAX` : le domaine de résultats est
rigoureusement celui du rederive — un épisode que l'original refuse
(RuntimeError « t_end insuffisant ») est refusé ici aussi, au même endroit,
par le même garde-fou. Ce garde-fou est d'ailleurs l'ORACLE du protocole :
c'est lui qui dit si un `t_end` suffit, et il appartient au rederive.

CE QUE CE MODULE NE FAIT PAS. Il ne réécrit aucune physique — pas de miroir
CPU. Il appelle `run_episode` INTOUCHÉ. `git diff` sur `src/sediment.py` et
`src/solver_wetdry.py` reste vide.

CHOIX D'IMPLÉMENTATION REMONTÉ, non couvert par le gravé. Le seul point
d'entrée du `t_end` est la constante de module `sediment._T_END_RELAX`, lue
à l'appel : la borner impose donc de la REMPLACER LE TEMPS DE L'APPEL, puis
de la restaurer (try/finally, testé y compris sur exception). C'est une
mutation d'état global : elle n'est pas thread-safe, et un appel concurrent
au rederive depuis un autre fil verrait la valeur bornée. T2 est
mono-fil — mais je le dis plutôt que de le taire, parce qu'une mutation
globale invisible serait pire qu'une modification franche. Si Romain préfère
la modification franche (exposer le `max_steps` déjà présent en dur dans
`simulate_wetdry_o2`), elle est décrite dans
`claude/rederive-borne-memoire.md` et reste à sa main."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

import src.sediment as sediment
from src.sediment import SedimentParams, _T_END_RELAX, run_episode

# Échelle géométrique, plafonnée au `t_end` GRAVÉ (jamais au-delà).
ECHELLE_T_END: tuple[float, ...] = (4.0, 8.0, 16.0, 32.0, 64.0, _T_END_RELAX)


class ConsommateurRederive:
    """Consomme le rederive épisode par épisode sous `t_end` borné.

    L'ESCALADE REPART DU BAS À CHAQUE ÉPISODE. Un mémo monotone (garder le
    `t_end` du dernier succès) était mon premier réflexe et il est FAUX :
    mesuré au seed 103, l'épisode 3 exige 130 tandis que l'épisode 4 se
    contente de 64 — hériter du 130 tuait le processus sur l'épisode
    suivant. Les besoins ne sont pas monotones le long d'une histoire ; les
    barreaux bas coûtent quelques dizaines de pas, on les repaie.

    Cette anti-corrélation est d'ailleurs ce qui rend le protocole sûr : un
    épisode qui exige un `t_end` ÉLEVÉ est un épisode dont le `dt` reste
    grand, donc PEU de pas ; l'épisode coûteux (dt effondré) est justement
    celui qu'un `t_end` BAS satisfait."""

    def __init__(self, params: SedimentParams = SedimentParams(),
                 echelle: tuple[float, ...] = ECHELLE_T_END):
        self.params = params
        self.echelle = tuple(echelle)
        # marge : on exige N_settle + 1 pas disponibles, cf. `_barreau`.
        self._params_marge = replace(params, N_settle=params.N_settle + 1)
        self.t_end_courant: float = self.echelle[0]
        self.barreaux: list[dict] = []

    def _barreau(self, s, b0, centre_frac) -> float:
        """Le plus petit `t_end` de l'échelle offrant N_settle + 1 pas.

        POURQUOI UNE MARGE D'UN PAS. `_relax_episode` retient `N_settle + 1`
        entrées et le solveur ÉCRÊTE son dernier `dt` pour atterrir sur
        `t_end` (`if t + dt > t_end: dt = t_end - t`). Si le nombre de pas
        disponibles valait EXACTEMENT `N_settle`, la dernière entrée retenue
        serait ce pas écrêté — différent de ce que le rederive non borné
        produit au même rang. Exiger un pas de plus garantit que les
        `N_settle + 1` entrées retenues sont toutes des pas CFL ordinaires,
        donc bit-identiques. L'échelle cherchant le PLUS PETIT barreau qui
        passe, ce cas limite n'est pas rare : il est visé."""
        derniere_erreur: RuntimeError | None = None
        for t_end in self.echelle:
            try:
                self._appeler(s, b0, centre_frac, t_end, self._params_marge)
            except RuntimeError as exc:
                # L'oracle est le garde-fou du rederive lui-même : « t_end
                # insuffisant, seulement N pas acceptés ». On monte d'un cran.
                derniere_erreur = exc
                continue
            return t_end
        raise RuntimeError(
            f"rederive_borne : le plafond gravé t_end={self.echelle[-1]} ne "
            f"suffit pas pour centre={centre_frac} — l'original échouerait "
            f"identiquement. Cause d'origine : {derniere_erreur}")

    def episode(self, s, b0, centre_frac) -> np.ndarray:
        """Un épisode du rederive INTOUCHÉ, consommé sous `t_end` borné."""
        t_end = self._barreau(s, b0, centre_frac)
        self.t_end_courant = t_end
        self.barreaux.append({"t_end": t_end})
        return self._appeler(s, b0, centre_frac, t_end, self.params)

    def _appeler(self, s, b0, centre_frac, t_end: float,
                 params: SedimentParams) -> np.ndarray:
        """Appelle `run_episode` INTOUCHÉ sous `t_end` borné, puis restaure
        la constante — y compris si l'appel lève."""
        ancien = sediment._T_END_RELAX
        sediment._T_END_RELAX = t_end
        try:
            return run_episode(s, b0, centre_frac, params)
        finally:
            sediment._T_END_RELAX = ancien

    def histoire(self, b0, centres, checkpoints: set[int] | None = None,
                 sonde=None, bras: str = "rederive") -> dict:
        """L'histoire complète, épisode par épisode. `centres` sont ceux du
        rederive (vérifiés contre `run_history`, cf. `cellule_mb`).

        `sonde` (optionnelle) : une `SondeMemoire` — chaque épisode devient
        une phase persistée, pour que la mort du processus laisse dire QUEL
        épisode la tenait."""
        n_episodes = len(centres)
        if checkpoints is None:
            checkpoints = {n_episodes}
        s = np.zeros_like(b0, dtype=np.float64)
        out: dict[int, np.ndarray] = {}
        for ep in range(1, n_episodes + 1):
            if sonde is None:
                s = self.episode(s, b0, centres[ep - 1])
            else:
                with sonde.phase(bras, f"episode={ep}"):
                    s = self.episode(s, b0, centres[ep - 1])
            if ep in checkpoints:
                out[ep] = s.copy()
        return out
