# Diagnostic — l'OOM de T2, et la correction de l'INSTRUMENT

Source : pocCascade2phys §A31-run (commit 98d372e). Gate (iii) **NON MESURÉ**
(échec d'instrument), pas raté. La cellule §A15 **n'a pas bougé** :
3 seeds {101, 102, 103}, Δt = 4, 6 émissions, 24 épisodes.

---

## 1. Discrimination — RAM HÔTE, verbatim

Le run ne produit **aucune** trace Python : ni `cupy.cuda.memory.OutOfMemoryError`,
ni traceback. La console montre seulement, après les trois lignes de seed :

```
Processus arrêté
```

`dmesg` tranche :

```
[mar. 21 juil. 23:54:05 2026] Out of memory: Killed process 37234 (python)
  total-vm:33600364kB, anon-rss:23569692kB, file-rss:65920kB, shmem-rss:10080kB,
  UID:1000 pgtables:50664kB oom_score_adj:200
[mer. 22 juil. 00:09:39 2026] Out of memory: Killed process 39341 (python)
  total-vm:33592780kB, anon-rss:25088636kB, file-rss:65280kB, shmem-rss:6240kB,
  UID:1000 pgtables:51620kB oom_score_adj:200
```

**`anon-rss` 23,5 puis 25,1 Go ⇒ RAM HÔTE.** L'hypothèse GPU est écartée :
le mempool CuPy plafonne à **34 Mo** pendant tout le run (mesuré), et une
saturation VRAM lèverait une exception Python, pas un SIGKILL.

L'inspection portée par §A31-run est confirmée : rien dans T2 n'explique un
OOM à 64². Le poste est bien le seul objet neuf — **le rederive CPU f64**.

## 2. Le poste, localisé au centre près

**Seed 103, épisode 4, centre (0,3524 ; 0,6131).**

`_relax_episode` appelle `simulate_wetdry_o2(..., t_end = _T_END_RELAX = 130)`,
qui accumule `hs, hus, hvs` à **chaque pas accepté** (3 × 64² × f64 =
98 Ko/pas), puis **tronque à `N_settle + 1 = 601`**. Le `dt` CFL ne s'effondre
pas au départ — il s'effondre *en cours d'épisode* :

| `t_end` | pas acceptés | `dt` médian | trajectoire |
|---|---|---|---|
| 8 | 45 | 1,78e-1 | 4 Mo |
| 32 | 269 | 1,05e-1 | 26 Mo |
| 64 | 3 173 | **5,48e-3** | 312 Mo |
| 130 (le réel) | **~254 000** | — | **~25 Go** |

Les 601 pas retenus couvrent **t = 48,44** sur 130. Besoin réel : **59 Mo**.
**~99,8 % est calculé puis jeté** — et c'est ce gâchis, non la physique, qui
rendait le seed incalculable. `run_history` est hors de cause (il ne garde
que les checkpoints, vérifié).

## 3. Correction de l'instrument — le rederive reste INTOUCHÉ

`git diff` sur `src/sediment.py` et `src/solver_wetdry.py` : **vide**.

**Le protocole.** `t_end` ne gouverne que deux choses dans la boucle du
solveur : l'arrêt (`while t < t_end`) et l'écrêtage du dernier `dt`. Les pas
produits avant que `t` n'approche `t_end` n'en dépendent donc pas. On
consomme le rederive au **plus petit barreau d'une échelle géométrique**
(4, 8, 16, 32, 64, 130) qui offre assez de pas.

- **Plafond = `_T_END_RELAX` gravé.** Le domaine de résultats est
  rigoureusement celui du rederive : un épisode qu'il refuse reste refusé,
  par son propre `RuntimeError` — lequel est d'ailleurs **l'oracle** du
  protocole. Il appartient au rederive, pas à moi.
- **Marge d'un pas.** On exige `N_settle + 1` pas disponibles, non
  `N_settle`. Si le compte tombait *exactement* à `N_settle`, la dernière
  entrée retenue serait le pas **écrêté** — différent du rederive non borné.
  L'échelle cherchant le plus petit barreau qui passe, ce cas limite n'est
  pas rare : il est **visé**. La marge l'élimine.
- **Bit-exactitude gardée par test** contre `run_episode` et `run_history`
  tels quels (`N_settle = 600`, comparaison `assert_array_equal`).

**Deux erreurs de ma part, corrigées par la mesure et non par le
raisonnement :**

1. J'avais d'abord proposé de modifier `simulate_wetdry_o2` (exposer son
   `max_steps`). §A31-run l'exclut ; le protocole de consommation fait le
   même travail sans toucher la vérité-sol.
2. Mon premier mémo était **monotone** (garder le `t_end` du dernier succès).
   Faux, et mesuré tel : au seed 103 l'épisode 3 exige 130 tandis que
   l'épisode 4 se contente de 64 — hériter du 130 tuait le processus à
   l'épisode suivant. **L'escalade repart du bas à chaque épisode.**

Cette anti-corrélation est ce qui rend le protocole sûr : un épisode qui
exige un `t_end` élevé est un épisode dont le `dt` reste grand, donc **peu**
de pas ; l'épisode coûteux (dt effondré) est justement celui qu'un `t_end`
bas satisfait.

### Résultat mesuré, cellule entière, sous garde `RLIMIT_AS = 4 Go`

```
seed 101: 6 checkpoints  pic= 1131 Mo  barreaux=[64.0, 130.0]
seed 102: 6 checkpoints  pic= 2109 Mo  barreaux=[64.0, 130.0]
seed 103: 6 checkpoints  pic= 2109 Mo  barreaux=[64.0, 130.0]
ru_maxrss global = 2230 Mo
```

**25 Go → 2,2 Go**, la cellule complète, sous une borne matérielle. La garde
`RLIMIT_AS` n'est pas un correctif mais une **preuve** : le processus n'aurait
pas pu dépasser 4 Go sans lever.

## 4. La sonde mémoire persistée

`src/f1_gpu/memoire_instrument.py`. L'OOM killer envoie un SIGKILL : aucun
`finally`, aucun `except`, aucun tampon vidé. Tout ce qui n'est pas **déjà
sur le disque** est perdu — la leçon du FAIL cloud non persisté, appliquée.

- jalon de **DÉBUT** écrit *avant* que la phase ne travaille (savoir **qui**
  tenait le processus) ;
- **BATTEMENT** périodique portant le pic courant (savoir **combien**) ;
- jalon de **FIN**, exception comprise ;
- **JSON lines + `fsync`** : un fichier tronqué en pleine écriture garde
  toutes ses lignes précédentes ; `lire_jalons` tolère la dernière ligne
  coupée — exactement ce que laisse un SIGKILL.

Pic attribué **par bras et par phase** (`pic_par_phase`), et porté **dans le
report** (`memoire_pic_mo`). Un run partiel qui atteint le pic le reporte.

## 5. Ce qui n'a pas bougé

Cellule §A15 **intacte** — rétrécir la mesure pour la faire entrer dans
l'instrument est l'inversion interdite, et MORT-b étant par-seed, un seul
seed viderait le critère. Le correctif porte sur l'**instrument**, comme
demandé. k reste 4 ; `attribution_b` gaté ; pas de miroir CPU (aucune
physique réécrite : `run_episode` est appelé tel quel) ; kernels et C1–C5
intouchés (`e18015f5` / `9533a130` / `6dd207ca` / `3533fd0b`).

## 6. Choix d'implémentation remonté

Le seul point d'entrée du `t_end` est la constante de module
`sediment._T_END_RELAX`, lue à l'appel. La borner impose de la **remplacer le
temps de l'appel** puis de la restaurer (`try/finally`, testé y compris sur
exception). C'est une **mutation d'état global** : non thread-safe, et un
appel concurrent au rederive depuis un autre fil verrait la valeur bornée.
T2 est mono-fil — mais je le dis plutôt que de le taire, une mutation globale
invisible étant pire qu'une modification franche. Si Romain préfère la
modification franche (exposer le `max_steps` déjà présent en dur), elle reste
décrite dans `claude/rederive-borne-memoire.md` et à sa main.

**POINT D'ARRÊT.** Diagnostic remonté, instrument corrigé et vérifié. Aucun
run de T2.
