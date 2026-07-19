# Re-épinglage du quadruplet §6 — ENDOSSÉ (Romain, 2026-07-19), GRAVÉ

> **Statut : ENDOSSÉ — R1=V2 (c dégressif), R2+R3=oui (scénario + bande ±15 %),
> R4=σ_ω reconduite, consommateur consigné. La version qui fait foi :
> pocCascade2phys PREREGISTRATION.md §A16 + SPEC-FOVEA-Z §6-rev1.
> Ce fichier est l'archive de travail.
> Entrées : annexe-enveloppe-jeu-2026-07-19.md (b7bef0a), modèle de coût MESURÉ
> (6.43 ns/cellule/frame à c=8 ; 1 slot ≡ fenêtre 512² : 1.685 ms à c=8,
> 0.843 ms à c=4 [linéarité en systèmes, structurelle]).

## Ce qui change de structure (pas seulement de chiffres)

L'ancien quadruplet (c, b, n_fov, N_niv) décrivait une enveloppe DENSE
(γ₂·n_fov²·N_niv). Le re-épinglage adopte la structure consignée au verdict :
**cap dur d'emplacements** — le budget est une liste de SLOTS (niveau, c), somme
des coûts ≤ 16.7 ms (T2, inchangé — aucun seuil ne bouge). Le routeur respecte le
cap ; dépasser = déraffiner ailleurs, jamais dépasser la frame. `[NON-ANCRÉ :
politique de routeur — le mécanisme exact est spec §2/§4, hors re-épinglage]`

## Scénario-jeu de référence (reconduit de l'annexe)

2D, mono-observateur, FOV 20°, monde h0 = 500k cellules (~707 de côté),
n = 512 cellules d'arc (≈3.75 px/cellule à 1920), J = 9 niveaux sous h0
⇒ finest ≈ h0/512, cône couvrant tout le monde. Machine : 3050 Ti (GPU minimal
1920 — l'enveloppe 4K sur GPU ×4–5 s'en déduit, annexe).

## Variantes candidates (l'arithmétique, avant la décision)

| | composition | coût prédit | marge | réserve énergie |
|---|---|---|---|---|
| **V1** | c=4 partout : 9 slots fovéale + 5 slots énergie | **11.8 ms** | 29 % | 5 slots |
| **V2** | c dégressif : c=8 aux 2 niveaux fins, c=4 au-dessus (fovéale 9.27 ms) + 3 slots énergie c=8 fins | **14.3 ms** | 14 % | 3 slots (riches) |
| **V3** | c=8 partout : 9 slots fovéale, 0 énergie | 15.2 ms | 9 % | AUCUNE |

V3 est nommée pour mémoire et probablement à rejeter : zéro réserve = zéro
gameplay raffiné, et 9 % de marge pour porter 5 pieds non mesurés. V1 dépense le
levier c explicitement (c'est le lieu légitime : décision, pas glissement) ;
V2 garde la physique riche là où l'œil est.

## Les 5 pieds non mesurés (les 4 de l'annexe + 1 NEUF)

1. Niveau 0 CPU vivant 500k (harnais : 65k) ;
2. 3D cube tout (calcul sans ancre) ;
3. Fenêtres d'énergie = ressource de gameplay (les réserves ci-dessus les comptent
   ENFIN — c'était le pied 3, il devient un poste budgétaire explicite) ;
4. r_fovea non pinné (arbitre n/J, quadratique en n) ;
5. **NEUF — CADENCEMENT TEMPOREL** : M-a mesure UNE application de F par fenêtre
   par frame (B5, dt figé). Un moteur à CFL locale sous-cadence les niveaux
   grossiers et/ou sur-cadence les fins — « 1 pas/frame/niveau » est
   `[NON-ANCRÉ : hypothèse de design]`. Si le vrai cadencement fin exige k sous-pas,
   le coût fovéale fin est ×k. Falsificateur nommé (NON armé) : la question
   perceptuelle « le sous-cadencement des niveaux fins est-il visible ? » est un
   RÉFÉRENT TEMPOREL — **proximité avec la condition de réveil de la dette σ_ω,
   décision Romain requise (R4)** : réveil du fork, ou dette reconduite avec ce
   consommateur nommé en plus.

## M-a-ter — pré-enregistrement (à figer ici)

- Protocole : harnais tranche-1, kernel fusionné M-a′, config = LA variante
  endossée (liste de slots (niveau, c) — nécessite une extension mineure du
  driver : slots paramétriques + c par niveau [1 ou 2 systèmes]) ; fovéa mobile
  E4c ; chrono B6 ; vérifs #1/#2 reconduites ; natif.
- **CRITÈRE : médiane frame-time > 16.7 ms ⇒ MORT du quadruplet-jeu candidat**
  (le seuil T2 gravé, inchangé). Diagnostic non-verdictal : la variante non
  retenue mesurée aussi (coût marginal nul, information pour la spec).
- Résidence : gate §6 reconduit (< 1.35 Go — trivial, moins de cellules).
- Prédiction gravée AVANT run (test du modèle de coût) : coût prédit de la
  variante ±15 %. Hors de la bande = AUTRE à remonter (le modèle linéaire en
  slots serait faux — information capitale, pas un échec).

## Portabilité des lectures tranche-1 (pas de re-mesure)

- M-c : PASS porté A FORTIORI (moins de fenêtres ⇒ moins de trafic ; dépendance
  EPS_DETAIL reconduite telle quelle) ;
- M-d : PASS porté tel quel (le ledger ne dépend pas du quadruplet) ;
- M-b/tranche-2 : re-scopée à l'enveloppe re-épinglée, achetable SEULEMENT si
  M-a-ter sans mort ;
- F0-cloud : inchangé, dû à la prochaine session cloud.

## La spec re-fait foi quand

(i) M-a-ter SANS MORT sur le quadruplet-jeu endossé ; (ii) F0-cloud lu ;
(iii) M-b (tranche-2) sans mort sur cette enveloppe. — Reformulation du gate iii
§8 à l'enveloppe re-épinglée ; rien d'autre ne change dans la spec.

## Décisions Romain — TRANCHÉES (2026-07-19)

- **R1 = V2** : c dégressif — fovéale 9 slots (2 fins à c=8, 7 à c=4) + 3 slots
  énergie c=8 fins. Prédit : 9.271 + 5.055 = **14.33 ms** (85.8 % du budget).
- **R2** : scénario épinglé v1-jeu — 20°, n=512 d'arc, J=9, monde 500k, 3050 Ti.
- **R3** : bande ±15 % ⇒ [12.18, 16.48] ms — ENTIÈREMENT sous le seuil 16.7 :
  si le modèle tient, M-a-ter passe ; une mort impliquerait AUSSI un AUTRE du
  modèle. Cohérence nommée avant run.
- **R4** : σ_ω RECONDUITE — condition de réveil AFFINÉE : le fork rouvre quand la
  spec décide le CADENCEMENT (la question perceptuelle du sous-cadencement fin en
  main, branche (a″) préférée). Aucun réveil silencieux.
