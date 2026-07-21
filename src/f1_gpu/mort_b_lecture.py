"""F1 — M-b tranche-2 : la LECTURE à trois branches de MORT-b (§A31).

Ce module ne mesure rien : il porte la DÉCISION pré-écrite que le bras
témoin rend possible. L'écart live↔rederive a TROIS sources (§A31) :
  (a) l'arithmétique f32 GPU ;
  (b) la remontée seuillée à EPS 1e-2 (perte délibérée) ;
  (c) la structure fovéale elle-même.

Le remède gravé, Option B « tout-déterministe », ne corrige que (a). Une
traversée due à (b) ou (c) ferait donc prononcer la mort d'Option A en
appliquant un remède incapable de guérir la cause. Le BRAS TÉMOIN isole
(a) : fidèle F GPU f32 au 64² PLEIN DOMAINE, un seul niveau, murs
réfléchissants (config identique au rederive, seule l'arithmétique diffère).

TROIS BRANCHES, PAR-SEED, contre le pin sévère 0,0733 :
  - production TRAVERSE et témoin PASSE ⇒ cause (b)+(c), **OPTION B N'EST
    PAS LE REMÈDE** — la décision portera sur EPS et/ou la structure ;
  - les DEUX traversent ⇒ cause (a), **Option B EST le remède**, MORT-b se
    prononce comme gravé ;
  - AUCUN ne traverse ⇒ **Option A tient, gate (iii) satisfait**.

MORT-b est PAR-SEED : un seul seed qui traverse en production tue Option A.
La branche dit POURQUOI, et donc si le remède gravé s'applique."""
from __future__ import annotations

# Pin sévère IC entier (7,33 %), gravé — MORT-b s'y compare.
PIN_SEVERE: float = 0.0733

LABEL_OPTION_A_TIENT: str = "OPTION A TIENT — gate (iii) satisfait"
LABEL_OPTION_B_REMEDE: str = "MORT-b — Option B EST le remède (cause a, f32)"
LABEL_OPTION_B_PAS_REMEDE: str = (
    "MORT-b — Option B N'EST PAS le remède (cause b+c : EPS et/ou structure)")
LABEL_ANOMALIE: str = (
    "ANOMALIE — production PASSE mais témoin TRAVERSE (à remonter)")


def _traverse(dchi_max: float) -> bool:
    return bool(dchi_max > PIN_SEVERE)


def branche_seed(production_dchi_max: float,
                 temoin_dchi_max: float) -> dict:
    """La branche pré-écrite d'UN seed, appliquée sans interprétation.

    production TRAVERSE = max de série Δχ (live↔rederive) > 0,0733.
    témoin TRAVERSE = idem pour le bras témoin (f32 plein domaine)."""
    prod = _traverse(production_dchi_max)
    tem = _traverse(temoin_dchi_max)
    if not prod:
        # production ne traverse pas : ce seed n'accuse pas Option A.
        # (si le témoin traverse quand même, c'est une anomalie — le
        #  témoin a MOINS de sources que la production.)
        label = LABEL_ANOMALIE if tem else LABEL_OPTION_A_TIENT
    elif tem:
        # les deux traversent : la cause est (a), le remède gravé s'applique.
        label = LABEL_OPTION_B_REMEDE
    else:
        # production traverse, témoin non : la cause est (b)+(c).
        label = LABEL_OPTION_B_PAS_REMEDE
    return {
        "production_dchi_max": production_dchi_max,
        "temoin_dchi_max": temoin_dchi_max,
        "production_traverse": prod,
        "temoin_traverse": tem,
        "label": label,
        "option_a_morte": bool(prod),
        "option_b_est_le_remede": bool(prod and tem),
    }


def lecture_mort_b(par_seed: dict[int, dict]) -> dict:
    """Agrège les branches par-seed en une lecture MÉCANIQUE. `par_seed` :
    {seed: {"production": dchi_max, "temoin": dchi_max}}.

    MORT-b PAR-SEED : Option A morte si UN seed traverse en production. Le
    remède gravé (Option B) ne s'applique QUE si les seeds traversants ont
    aussi leur témoin traversant (cause a) ; sinon le remède est ailleurs
    (EPS/structure)."""
    branches = {seed: branche_seed(m["production"], m["temoin"])
                for seed, m in par_seed.items()}
    traversants = [s for s, b in branches.items() if b["production_traverse"]]
    option_a_morte = bool(traversants)
    # Le remède gravé (Option B) ne guérit que si TOUS les traversants sont
    # cause (a) — un seul traversant b+c le rend inapplicable en l'état.
    tous_cause_a = all(branches[s]["option_b_est_le_remede"]
                       for s in traversants)
    anomalies = [s for s, b in branches.items()
                 if b["label"] == LABEL_ANOMALIE]
    if not option_a_morte:
        verdict = LABEL_OPTION_A_TIENT
    elif tous_cause_a:
        verdict = LABEL_OPTION_B_REMEDE
    else:
        verdict = LABEL_OPTION_B_PAS_REMEDE
    return {
        "pin_severe": PIN_SEVERE,
        "par_seed": branches,
        "seeds_traversants": traversants,
        "option_a_morte": option_a_morte,
        "verdict": verdict,
        "option_b_est_le_remede": bool(option_a_morte and tous_cause_a),
        "anomalies": anomalies,
        "note_remede": (
            "Option B (tout-déterministe) ne corrige que (a) l'arithmétique "
            "f32. Elle n'est le remède QUE si tout seed traversant a son "
            "témoin traversant. Un traversant à témoin PASSANT accuse (b) "
            "EPS et/ou (c) structure — remède ailleurs."),
    }


def portees_de_la_mesure() -> dict:
    """Les portées à PORTER DANS la lecture (§A31, §A31-build), pas seulement
    au docstring : ce que la mesure à 64²/2-niveaux ne valide pas, l'effet
    dominant des colonnes entrantes, le RE-SCOPE du gate (iii), et la réserve
    sur ce que le témoin a déjà montré au build."""
    return {
        "l1_l3_non_validees_a_v4": (
            "à 64² sur DEUX niveaux, L1 étalée et L3 ne sont exercées que "
            "MARGINALEMENT : la mesure de fidélité NE LES VALIDE PAS à "
            "l'échelle V4 (131 072²). Un PASS ici ne dit rien de leur "
            "comportement à grande échelle."),
        "colonnes_entrantes_dominantes": (
            "la fovéa mobile E4c parcourt ~24 cellules sur les 6 émissions, "
            "soit ~un TIERS du domaine 64². La contribution des colonnes "
            "ENTRANTES (prédites depuis le grossier à chaque déplacement) y "
            "est FORTE — l'écart mesuré est en bonne part celui de la "
            "prédiction de bord, pas du cœur fovéal."),
        "gate_iii_re_scope_a_64": (
            "gate (iii) RE-SCOPÉ AVANT le run (§A31-build) : il était gravé "
            "« M-b sans mort SUR V4 » alors que M-b tourne à 64² — "
            "structurellement insatisfiable. Il devient « M-b sans mort À "
            "64², sur fovéation 2-niveaux ». Le contrat à l'échelle V4 "
            "devient une TRANSPOSITION NOMMÉE, NON MESURÉE : son "
            "falsificateur exigerait un rederive à l'échelle V4, interdit "
            "par « rederive intouché ». Conséquence : un PASS ici ne peut "
            "PAS s'énoncer « le contrat tient à l'échelle V4 », seulement "
            "« sans mort à 64² sur deux niveaux »."),
        "reserve_temoin_etat_vs_mesure": (
            "au BUILD, le témoin a montré (a) = 6e-5 d'écart relatif contre "
            "le rederive f64 — trois ordres SOUS le pin 0,0733. La branche "
            "« les deux traversent ⇒ Option B est le remède » en devient "
            "très improbable, et une traversée de production sera presque "
            "certainement (b)+(c). RÉSERVE : 6e-5 est une vérification "
            "d'ÉTAT au build, PAS la mesure Δχ pré-enregistrée — "
            "l'instrument n'est pas le même, le témoin doit tourner comme "
            "prévu et sa branche reste lue sur SA mesure."),
    }
