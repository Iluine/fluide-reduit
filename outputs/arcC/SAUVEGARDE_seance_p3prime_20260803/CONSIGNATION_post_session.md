# Consignation post-séance P3′ — notes pour gravure §A44 (PAS un gravé)
- Séance : 2026-08-03, 10:06→10:28, commande tapée par Romain le jour même.
- `--luminosite "80"` : unité précisée par Romain a posteriori (horloge lue à la consignation, voir date ci-dessous) : **OSD 80 %** (réglage moniteur), pas des nits.
- `--conditions` : chaîne elliptique à la CLI (mots de la spec §C9 pièce 3, pas les valeurs). Éclairage ambiant réel (note Romain, 10:35 horloge lue — corrigé : 10:36 avait été écrit AVANT lecture de l'horloge, faute nommée) : **plafonnier + lumière du jour, volet légèrement clos pour une lumière uniforme et stable**. Repère de distance (note Romain, 10:43 horloge lue) : **mesure directe à la règle** (775 mm), pas de repère physique permanent — la tenue de la distance en cours de séance repose sur la posture, non re-vérifiée pendant les essais.
- Collision de chemins (famille B1) : la séance a écrasé 7 fichiers trackés de la campagne du 05/07 ; anciens contenus intacts à HEAD 75b2289 ; cette sauvegarde est byte-identique aux fichiers de séance en place (sha témoin vérifié).
- Push des branches : PAS fait avant la séance (§A43 le plaçait avant) — fait de processus à consigner.
10:34
- Défaut de câblage n°2 (même famille que la collision) : l'orchestration p3prime a écrit son manifeste au nom par défaut HISTORIQUE (`manifeste_campagne.json`) ; la lecture p3prime attend `manifeste_p3prime.json`. Lecture relancée avec `--manifeste` explicite (chemin canonique en place), voie prévue par l'outil.
