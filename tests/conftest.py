"""Configuration pytest — force le backend matplotlib NON-INTERACTIF (Agg) pour
toute la suite.

Motif : l'installation de PySide6 (backend GUI de la session HUMAINE Arc C, cf.
`src/arcC_backend.py`) fait basculer le backend PAR DÉFAUT de matplotlib vers
QtAgg dès qu'un toolkit Qt + un display sont présents. Les tests ne doivent PAS
en dépendre : ils créent des figures via `savefig` (Agg suffit, aucun écran
requis) et doivent rester déterministes et headless-safe (CI sans display). La
session live, elle, bascule explicitement sur QtAgg dans son chemin d'exécution
(`run_arcC_session.py`/`run_arcC_orchestration.py`) -- jamais atteint par les
tests. Ce forçage Agg AVANT tout import de figure découple les deux."""
import matplotlib

matplotlib.use("Agg")
