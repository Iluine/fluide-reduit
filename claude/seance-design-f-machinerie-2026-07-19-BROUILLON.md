# Séance design — F-multi-niveaux + machinerie (E4d/orchestration) — ENDOSSÉ

> **Statut : ENDOSSÉ (Romain, 2026-07-19) — G1 oui (s1+s2), G2 = L1+L3 à
> l'étude (L2/L4 restent nommés), G3 = porte 33.3 ms ouverte-nommée. Version
> faisant foi : pocCascade2phys PREREGISTRATION.md §A17.** Entrée : §A16-lecture-sonde-attribution (bd8b0f1) —
> décomposition MESURÉE de la frame V2 : **F 20.438 / prédiction 2.135 /
> remontée 8.523 = 31.096 ms**, budget 16.7 (T2, inchangé). La séance s'ouvre par
> deux micro-sondes (§0) qui ANCRENT le design ; les leviers (§1) se trancheront
> avec leurs chiffres, pas avant. Gravure au journal après endossement.

## §0 — Micro-sondes d'ouverture (pré-enregistrées ici, AVANT tout code)

### s1 — Ancre du bloc 1-système (la « linéarité » enfin mesurée)

- Protocole : micro-mesure M-a′-style, kernel INTOUCHÉ — (a) 1 niveau, B=1, S=1
  (un slot c=4 isolé) ; (b) B=7, S=1 (les 7 slots c=4 de V2 en UN lancement).
  Chrono B6, natif.
- Attendus pré-écrits : le modèle actuel suppose 0.843 ms/slot (moitié du 2-sys).
  Lecture : les DEUX ancres mesurées (isolée, batchée) REMPLACENT l'étiquette
  « structurelle » — hors ±10 % de 0.843, la linéarité en systèmes est fausse et
  le modèle per-slot se re-calibre sur mesures. Sonde : lecture remontée.

### s2 — F batché inter-niveaux + CFL asynchrone (l'orchestration du moteur réel)

- Protocole : bras A (immobile, remontée OFF) RE-ORCHESTRÉ : tous les blocs
  2-systèmes en un appel (B=5 : 2 fovéale fins + 3 énergie), tous les 1-système
  en un appel (B=7) — possible par construction, TOUS les niveaux sont à 512² ;
  2 étages ⇒ 4 lancements/frame au lieu de 18. Réduction CFL CALCULÉE mais
  restant on-device (pas de float()) — payée, non consommée, NON synchronisante.
  Kernel intouché (diff vide à prouver).
- **Légitimité vs cap, nommée pour endossement** : ce n'est pas une escalade du
  kernel — B5 documentait explicitement que « le moteur réel batcherait des
  fenêtres identiques » ; le per-niveau était de la plomberie de harnais. La
  CFL reste payée (le kernel de réduction tourne) ; seule la synchronisation
  CPU (artefact) tombe.
- Attendus pré-écrits : prédiction re-calibrée = 5 × 1.685 + 7 × ancre_s1 ;
  lecture : A_batché dans ±10 % de cette prédiction ⇒ l'écart des 6.11 ms était
  l'orchestration (artefact) et le modèle re-calibré TIENT ; hors ⇒ AUTRE
  (coût architectural résiduel non compris — remontée, pas de design à l'aveugle).
- Ce que s2 fixe : **le budget machinerie = 16.7 − A_batché** — le chiffre que
  tout le §1 vise.

## §1 — Leviers de design (nommés ; se tranchent APRÈS s1/s2, avec les chiffres)

- **L1 — Remontée cadencée** : tous les k frames (k ∈ {2, 4, 8}) au lieu de
  chaque frame — coût ÷k. CONSÉQUENCE NOMMÉE : la fraîcheur du niveau 0 vivant
  décroît — c'est une décision de CADENCEMENT : si k se choisit PERCEPTUELLEMENT,
  la condition de réveil σ_ω (R4) est ATTEINTE ; si k se choisit par budget avec
  étiquette [NON-ANCRÉ perceptuel] et falsificateur différé, le réveil peut être
  différé — décision Romain explicite au moment du choix, jamais un glissement.
- **L2 — Remontée à activité** : seules les fenêtres dont l'activité supra-seuil
  a changé remontent. En jeu, la plupart des fenêtres sont calmes ; au harnais,
  le forçage touche tout — LEVIER NON MESURABLE sur l'instrument actuel,
  étiquette [TRANSPOSITION-HYPOTHÈSE], falsificateur = substrat à activité
  localisée (coût à chiffrer si retenu).
- **L3 — Détails émis par le kernel F** : le kernel calcule déjà l'état ; émettre
  |d| ≥ eps en sortie fusionnerait l'extraction (supprime les passes CuPy
  diff/abs/mask/nonzero — le gros des 8.5 ms). TOUCHE LE KERNEL : à décider en
  face du cap — le cap protège la MESURE de la borne M-a′ (qui reste intacte et
  citée telle quelle) ; un kernel-moteur avec émission de détails est un BUILD
  NEUF avec son propre pré-enregistrement, pas une re-mesure de la borne.
- **L4 — Prédiction** : 2.135 ms pour 1 cellule/frame est cher ; suspect =
  orchestration CPU per-niveau du déplacement (même maladie que F pré-s2).
  Mesure différée : s2 est immobile — un bras s2-mobile est le falsificateur
  naturel si L4 devient load-bearing.

## §2 — L'arithmétique cible (l'honnêteté d'avance)

Si s2 rend F ≈ 14.3 : budget machinerie = 2.4 ms, contre 10.66 mesurés — il faut
×4.4. L1 seul (k=8) + prédiction inchangée ≈ 3.2 ms : ENCORE TROP. Le chemin
crédible combine au moins deux leviers (ex. L1 modéré + L3, ou L1 + L4 réglée)
— ou alors le candidat V2 lui-même s'amincit (moins de slots), ou la PORTE
BUDGET (33.3 ms / 30 fps) se rouvre comme DÉCISION DE RE-ÉPINGLAGE assumée
(la mort de V2 à 16.7 reste gravée ; un nouveau candidat à un nouveau budget
est une décision neuve par la porte de devant, jamais une relecture). Rien de
tout cela ne se décide ici : s1/s2 d'abord.

## Décisions Romain — TRANCHÉES (2026-07-19)

- **G1 : s1 + s2 ENDOSSÉES** (protocoles, attendus, légitimité vs cap) — build
  Claude Code débloqué (≤ 1 séance, chiffrage remonté sinon).
- **G2 : L1 (cadencée) + L3 (détails émis par le kernel) à l'étude chiffrée** ;
  L2 et L4 restent nommés, non instruits (L4 a son falsificateur naturel — bras
  s2-mobile — si elle devient load-bearing).
- **G3 : porte 33.3 ms OUVERTE-NOMMÉE** — pas décidée ; si §2 l'exige, elle
  s'étudie par la porte de devant (re-épinglage assumé, la mort de V2 à 16.7
  reste gravée).
