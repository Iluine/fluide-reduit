# Constat — le rederive n'est pas calculable au seed 103 (M-b tranche-2)

**Constat, pas correctif.** Le remède touche le rederive INTOUCHÉ : il est
nommé ici, **non appliqué**. Décision à Romain.

Source : pocCascade2phys §A31-build (commit fedbe78). Deux runs T2 tués par
l'OOM killer (21/07 23:54, 22/07 00:09) — `anon-rss` **23,5 Go** puis
**25,1 Go**.

---

## 1. Ce que ce n'est pas

Ni fuite, ni faute de l'assemblage. Profil mesuré du workload sur les seeds
101 et 102, cellule complète (24 épisodes, N_settle=600) : **pic 856 Mo**
RSS, pool GPU 34 Mo, les deux seeds passent en 2,9 min. Mon diagnostic
initial n'a rien trouvé **parce qu'il n'avait pas parcouru le seed 103**.

## 2. Le point exact

**Seed 103, épisode 4, centre (0,3524 ; 0,6131).** Le dt CFL ne s'effondre
pas au départ — il s'effondre *en cours d'épisode* :

| `t_end` | pas acceptés | dt médian | trajectoire stockée |
|---|---|---|---|
| 8 | 45 | 1,78e-1 | ~4 Mo |
| 16 | 107 | 1,45e-1 | 10 Mo |
| 32 | 269 | 1,05e-1 | 26 Mo |
| 64 | 3 173 | **5,48e-3** | 312 Mo |
| 130 (le réel) | **> 51 000** | — | **> 5 Go** (OOM à ~254 000 pas) |

`simulate_wetdry_o2` accumule `hs, hus, hvs` à **chaque** pas accepté
(3 × 64² × f64 = 98 Ko/pas) jusqu'à `t_end`, plafond `max_steps = 1 000 000`
— soit ~98 Go de plafond théorique à 64². Rien ne borne la mémoire.

## 3. Le gâchis, et pourquoi il est total

`_relax_episode` **tronque à `N_settle + 1 = 601` pas** (`return times[:n],
hs[:n], ...`). Le solveur calcule donc ~254 000 pas pour en utiliser 601 :

- les 601 pas utilisés couvrent **t = 48,44** sur 130 ;
- besoin réel : **59 Mo**. Alloué : **25 Go**. **~99,8 % jeté.**

Ce n'est pas seulement du gaspillage : c'est *lui* qui rend le seed 103
incalculable. Le résultat retenu, lui, tient en 59 Mo.

## 4. Preuve que borner est result-preserving

Les 601 premiers pas ne dépendent **pas** de `t_end` (ils sont produits avant
que `t_end` ne joue) :

```
t_end=55,0 -> 1 148 pas ; t au pas 600 = 48,4365
t_end=64,0 -> 3 173 pas ; t au pas 600 = 48,4365
601 premiers pas IDENTIQUES (times, hs, hus, hvs) : True
```

Donc arrêter le solveur à `N_settle` pas acceptés rend **bit-pour-bit** ce que
`_relax_episode` retourne aujourd'hui.

## 5. Le remède nommé — NON APPLIQUÉ

`simulate_wetdry_o2` porte **déjà** un `max_steps = 1_000_000`, en dur.
Le remède minimal : l'exposer en paramètre (défaut inchangé, donc
rétro-compatible) et que `_relax_episode` passe `max_steps=params.N_settle`.

- **Result-preserving** : §4 ;
- **garde-fou préservé** : le `RuntimeError` de `_relax_episode` teste
  `n_available < N_settle` ; si `t_end` est atteint en moins de `N_settle`
  pas (dt large), `n_available` reste inférieur et l'erreur lève comme avant ;
- **mémoire** : 25 Go -> 59 Mo.

**Mais il touche `src/solver_wetdry.py` ET `src/sediment.py`** — le rederive
INTOUCHÉ, la vérité-sol. Je ne l'applique pas. Trois voies, à trancher :

1. **Autoriser le remède** (le rederive change de *coût*, pas de *résultat* —
   §4 le prouve, et un test de non-régression peut le graver) ;
2. **Déclarer le seed 103 INATTEINGNABLE** — mais MORT-b est PAR-SEED : le
   seed manquant peut être celui qui traverse. T2 ne se prononcerait alors
   que sous réserve, et la réserve serait lourde ;
3. **Changer la cellule** — écarté : elle est gravée §A15.

## 6. Ce qui est déjà en place

Le driver écrit son report **après chaque seed** et reprend sous empreinte
identique (commit `200a71d`). Un run relancé **banque 101 et 102**, meurt sur
103, et laisse un report partiel qui ne prononce **aucune** lecture. Aucun
travail n'est reperdu.

**POINT D'ARRÊT.** Constat remonté, remède nommé et non appliqué.
