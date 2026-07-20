# Lectures versionnées

`outputs/` est gitignoré : les sorties brutes des sondes vivent sur le disque
de la machine-instrument, pas dans le dépôt. Le journal porte les *lectures*,
mais pas les *données* — donc un run n'est pas ré-auditable une fois la
machine nettoyée.

**Décision (§A23, remarque de Romain) : on versionne ici la LECTURE de chaque
run, pas les tableaux complets.** Concrètement, pour chaque
`outputs/f1/<run>.json` on garde `<run>.lecture.json` avec :

- `meta` — protocole, portées, empreintes citées ;
- `lecture_mecanique` — la lecture pré-écrite appliquée (branches, règle,
  plancher, verdicts reportés) ;
- une `table_compacte` quand le run a un balayage (une ligne par point,
  scalaires seulement) ;
- `residence` — l'empreinte VRAM.

Ce qu'on NE versionne PAS : les tableaux par-frame, les séries brutes, les
états. Ils restent sur disque sous `outputs/f1/` ; leur taille n'apporte
rien à l'audit d'une lecture, et les versionner alourdirait le dépôt sans
gain de traçabilité.

Ainsi une lecture reste re-lisible et confrontable à son pré-enregistrement
longtemps après que la machine a oublié le run.
