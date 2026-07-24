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

GARDÉ PAR UNE ASSERTION STRUCTURELLE, PAR ÉPISODE (§A31-diagnostic) : le
temps du dernier pas RETENU doit rester strictement sous `t_end − marge`
(cf. `_garde`). Le pas écrêté est alors hors fenêtre PAR CONSTRUCTION, sans
aucune comparaison à une exécution non bornée — laquelle est justement
IMPOSSIBLE là où le risque est maximal, l'épisode qui fait déborder la RAM.
La bit-exactitude contre `run_episode`/`run_history` tels quels reste testée,
en SECOND RANG (elle ne peut s'exercer que là où le danger est absent).

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
        self.t_end_courant: float = self.echelle[0]
        self.barreaux: list[dict] = []
        self.gardes: list[dict] = []
        self.appels: int = 0

    def episode(self, s, b0, centre_frac) -> np.ndarray:
        """Un épisode du rederive INTOUCHÉ, consommé sous `t_end` borné.

        Escalade jusqu'au premier barreau qui (i) offre assez de pas —
        l'oracle est le `RuntimeError` du rederive — ET (ii) satisfait LA
        GARDE STRUCTURELLE (§A31-diagnostic)."""
        derniere_erreur: RuntimeError | None = None
        for t_end in self.echelle:
            try:
                resultat, temps = self._appeler(s, b0, centre_frac, t_end)
            except RuntimeError as exc:
                # L'oracle est le garde-fou du rederive lui-même : « t_end
                # insuffisant, seulement N pas acceptés ». On monte d'un cran.
                derniere_erreur = exc
                continue
            garde = self._garde(temps, t_end)
            self.gardes.append(garde)
            if not garde["ok"]:
                continue          # fenêtre trop près de t_end : on monte
            self.t_end_courant = t_end
            self.barreaux.append({"t_end": t_end})
            return resultat
        raise RuntimeError(
            f"rederive_borne : le plafond gravé t_end={self.echelle[-1]} ne "
            f"suffit pas pour centre={centre_frac} — l'original échouerait "
            f"identiquement. Cause d'origine : {derniere_erreur}")

    def _garde(self, temps, t_end: float) -> dict:
        """LA GARDE STRUCTURELLE (§A31-diagnostic, amendement 1).

        Le solveur ÉCRÊTE son dernier `dt` pour atterrir exactement sur
        `t_end` (`if t + dt > t_end: dt = t_end - t`). Ce pas écrêté n'existe
        pas dans le rederive non borné : s'il tombait DANS la fenêtre
        retenue, le résultat différerait.

        La garde l'exclut PAR CONSTRUCTION : elle exige que le temps du
        dernier pas RETENU soit strictement sous `t_end − marge`, avec pour
        marge le `dt` de ce pas — c'est-à-dire qu'un pas entier de plus
        tienne encore avant `t_end`. La fenêtre est alors strictement
        intérieure, sans dépendance à l'endroit où `t_end` tombe.

        Ce qu'elle vaut de mieux que la sonde échantillonnée qu'elle
        remplace : elle ne compare à AUCUNE exécution non bornée. Or c'est
        précisément là où le risque est maximal — l'épisode qui fait
        déborder la RAM — que l'exécution non bornée est IMPOSSIBLE. Une
        garde qui n'est vérifiable que là où le danger est absent ne garde
        rien.

        AU PLAFOND GRAVÉ, aucune garde : l'appel EST l'original (même
        `t_end`), rien n'est borné, et refuser là où l'original accepte
        créerait une divergence de domaine."""
        n = self.params.N_settle
        t_dernier = float(temps[n])
        marge = float(temps[n] - temps[n - 1])
        plafond = t_end >= self.echelle[-1]
        ok = bool(plafond or (t_dernier + marge <= t_end))
        return {"t_end": t_end, "t_dernier_retenu": t_dernier,
                "marge": marge, "plafond": plafond, "ok": ok}

    def _appeler(self, s, b0, centre_frac, t_end: float):
        """Appelle `run_episode` INTOUCHÉ sous `t_end` borné et CAPTURE au
        passage les temps de `_relax_episode` — un seul appel par barreau.

        L'espion ne modifie rien : il délègue à la fonction d'origine et note
        ce qu'elle retourne. `_T_END_RELAX` et `_relax_episode` sont tous
        deux restaurés, y compris si l'appel lève."""
        captures: list = []
        vrai_relax = sediment._relax_episode

        def espion(b_eff, centre, params):
            resultat = vrai_relax(b_eff, centre, params)
            captures.append(np.asarray(resultat[0]))
            return resultat

        ancien_t, ancien_relax = sediment._T_END_RELAX, sediment._relax_episode
        sediment._T_END_RELAX = t_end
        sediment._relax_episode = espion
        try:
            self.appels += 1
            sortie = run_episode(s, b0, centre_frac, self.params)
        finally:
            sediment._T_END_RELAX = ancien_t
            sediment._relax_episode = ancien_relax
        return sortie, captures[-1]

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
