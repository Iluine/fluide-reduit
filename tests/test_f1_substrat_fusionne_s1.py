"""Tests F1 — S1 : les TROIS VERROUS de l'élargissement de la garde de
`pas_f_fusionne` à {1, 2} systèmes (§A16-complément, pocCascade2phys
03f2534, gravé AVANT tout run M-a-ter).

Motif gravé : la parité est un artefact d'INSTRUMENT (un assert), pas le
calcul — le c dégressif de V2 produit des blocs à UN système en
permanence. Ce que S1 autorise est la garde Python, RIEN d'autre. D'où :

  (a) le kernel CUDA est à diff VIDE — vérifié SANS GPU, deux fois :
      contre une empreinte figée, et contre le texte extrait du commit
      pré-S1 lui-même ;
  (b) le chemin 2-SYSTÈMES est BIT-IDENTIQUE avant/après — c'est lui qui
      porte l'ancre M-a′ (1.685 ms/slot) et la comparabilité de toute la
      tranche-1 ;
  (c) le chemin 1-SYSTÈME est NEUF (les 7 slots c=4 de V2 SONT des
      systèmes seuls — le design, pas un contournement) : sa correction
      est testée contre le jetable mono-système.

La référence « avant » est le commit COMMIT_PRE_S1, FIGÉ ici — surtout
pas HEAD, qui contiendra S1 et rendrait (b) tautologique. Tailles
minuscules : plomberie/équivalence, JAMAIS une mesure."""
import hashlib
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.f1_gpu.backend import cupy_disponible, vers_cpu
from src.f1_gpu.substrat_jetable import etat_initial_jetable, pas_f_jetable

RACINE = Path(__file__).resolve().parents[1]

# Empreinte du kernel CUDA (`_SOURCE`) relevée AVANT S1, sur 884078e.
SHA256_KERNEL_PRE_S1: str = (
    "e18015f57e14263414239b18c51d25cd335250f58dde2ccb892a6e2c1d6f30b4")
COMMIT_PRE_S1: str = "884078e"
CHEMIN_MODULE: str = "src/f1_gpu/substrat_fusionne.py"

gpu_requis = pytest.mark.skipif(
    not cupy_disponible(), reason="cupy/device CUDA indisponible")


def _source_pre_s1() -> str | None:
    """Contenu du module AVANT S1 (commit figé). None si git ou l'objet
    est indisponible — le test se saute alors, il ne ment pas."""
    try:
        acheve = subprocess.run(
            ["git", "show", f"{COMMIT_PRE_S1}:{CHEMIN_MODULE}"],
            cwd=str(RACINE), capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    return acheve.stdout


def _corps_du_kernel(texte_module: str) -> str:
    """Extrait la chaîne `_SOURCE` (le CUDA) d'un texte de module."""
    debut = texte_module.index('_SOURCE: str = r"""')
    fin = texte_module.index("_TAILLE_BLOC", debut)
    return texte_module[debut:fin]


def _charger_pre_s1():
    """Charge le module pré-S1 sous un nom distinct — les deux versions
    coexistent, chacune avec son propre cache de RawKernel."""
    source = _source_pre_s1()
    if source is None:
        pytest.skip("git indisponible : référence pré-S1 non récupérable")
    with tempfile.TemporaryDirectory() as dossier:
        chemin = Path(dossier) / "substrat_fusionne_pre_s1.py"
        chemin.write_text(source, encoding="utf-8")
        spec = importlib.util.spec_from_file_location(
            "substrat_fusionne_pre_s1", chemin)
        module = importlib.util.module_from_spec(spec)
        sys.modules["substrat_fusionne_pre_s1"] = module
        spec.loader.exec_module(module)
    return module


# ----- verrou (a) : kernel CUDA à diff VIDE (aucun GPU requis) -----

def test_a_empreinte_du_kernel_inchangee():
    """S1-(a) : le texte du kernel CUDA n'a pas bougé d'un octet. C'est
    une propriété du SOURCE, testable sans device — donc vérifiée à
    chaque exécution de la suite, pas seulement sur la machine-instrument."""
    from src.f1_gpu.substrat_fusionne import _SOURCE
    empreinte = hashlib.sha256(_SOURCE.encode()).hexdigest()
    assert empreinte == SHA256_KERNEL_PRE_S1, (
        "le kernel CUDA a changé — S1 n'autorisait QUE la garde Python ; "
        "l'ancre M-a′ ne serait plus comparable.")


def test_a_kernel_identique_au_commit_pre_s1():
    """Même exigence prise à la SOURCE plutôt qu'à une empreinte figée
    (qu'on pourrait mettre à jour à tort) : le CUDA extrait du commit
    pré-S1 est caractère pour caractère celui d'aujourd'hui."""
    avant = _source_pre_s1()
    if avant is None:
        pytest.skip("git indisponible : référence pré-S1 non récupérable")
    from src.f1_gpu import substrat_fusionne
    actuel = Path(substrat_fusionne.__file__).read_text(encoding="utf-8")
    assert _corps_du_kernel(avant) == _corps_du_kernel(actuel)


def test_a_la_garde_est_bien_ce_qui_a_change():
    """Contre-épreuve de (a) : le module, lui, A changé — sinon le test
    ci-dessus serait vide de sens (on comparerait deux fichiers
    identiques)."""
    avant = _source_pre_s1()
    if avant is None:
        pytest.skip("git indisponible : référence pré-S1 non récupérable")
    from src.f1_gpu import substrat_fusionne
    actuel = Path(substrat_fusionne.__file__).read_text(encoding="utf-8")
    assert avant != actuel
    assert "q.shape[1] != 2" in avant
    assert "q.shape[1] not in (1, 2)" in actuel


# ----- verrou (b) : chemin 2-systèmes BIT-IDENTIQUE avant/après -----

@gpu_requis
def test_b_deux_systemes_bit_identique_avant_apres():
    """S1-(b) : sur le MÊME état, la version pré-S1 et la version élargie
    produisent le même résultat AU BIT PRÈS — c'est ce chemin qui porte
    l'ancre M-a′ (1.685 ms/slot) et la comparabilité de la tranche-1."""
    import cupy as cp
    pre_s1 = _charger_pre_s1()
    from src.f1_gpu.substrat_fusionne import pas_f_fusionne

    q = etat_initial_jetable(2, 12, graine=17)
    avant, dt_avant = pre_s1.pas_f_fusionne(cp.asarray(q), cp)
    apres, dt_apres = pas_f_fusionne(cp.asarray(q), cp)
    assert np.array_equal(vers_cpu(avant), vers_cpu(apres))
    assert dt_avant == dt_apres


@gpu_requis
def test_b_deux_systemes_bit_identique_sur_plusieurs_pas():
    """Composition : l'identité bit-à-bit tient sur 3 pas enchaînés (un
    écart d'1 ULP au pas 1 se verrait amplifié)."""
    import cupy as cp
    pre_s1 = _charger_pre_s1()
    from src.f1_gpu.substrat_fusionne import pas_f_fusionne

    q_avant = cp.asarray(etat_initial_jetable(1, 10, graine=19))
    q_apres = cp.asarray(etat_initial_jetable(1, 10, graine=19))
    for _ in range(3):
        q_avant, _ = pre_s1.pas_f_fusionne(q_avant, cp)
        q_apres, _ = pas_f_fusionne(q_apres, cp)
    assert np.array_equal(vers_cpu(q_avant), vers_cpu(q_apres))


# ----- verrou (c) : chemin 1-système, NEUF, contre le jetable -----

@gpu_requis
def test_c_un_systeme_equivaut_au_jetable_mono_systeme():
    """S1-(c) : le chemin NEUF (S=1) reproduit `pas_f_jetable` sur un état
    mono-système, à la tolérance f32 nommée — même contrat que le chemin
    2-systèmes (§A15-lecture-M-a)."""
    import cupy as cp
    from src.f1_gpu.substrat_fusionne import (
        TOL_EQUIVALENCE_ATOL,
        TOL_EQUIVALENCE_RTOL,
        pas_f_fusionne,
    )
    q1 = np.ascontiguousarray(
        etat_initial_jetable(2, 12, graine=23)[:, :1])
    assert q1.shape == (2, 1, 4, 12, 12)
    attendu, dt_ref = pas_f_jetable(q1, np)
    obtenu, dt_fus = pas_f_fusionne(cp.asarray(q1), cp)
    np.testing.assert_allclose(vers_cpu(obtenu), attendu,
                               rtol=TOL_EQUIVALENCE_RTOL,
                               atol=TOL_EQUIVALENCE_ATOL)
    assert dt_fus == pytest.approx(dt_ref, rel=1e-5)


@gpu_requis
def test_c_un_systeme_egale_bit_a_bit_le_bloc_du_deux_systemes():
    """Preuve que S=1 ne calcule RIEN de différent : le résultat d'un
    appel mono-système est bit-à-bit celui du bloc correspondant d'un
    appel 2-systèmes portant le même contenu. Le kernel ne voit que des
    blocs (fenêtre, système) aplatis — S n'entre pas dans le calcul."""
    import cupy as cp
    from src.f1_gpu.substrat_fusionne import pas_f_fusionne

    q2 = etat_initial_jetable(1, 12, graine=29)
    q1 = np.ascontiguousarray(q2[:, :1])
    sortie2, _ = pas_f_fusionne(cp.asarray(q2), cp)
    sortie1, _ = pas_f_fusionne(cp.asarray(q1), cp)
    assert np.array_equal(vers_cpu(sortie1)[:, 0], vers_cpu(sortie2)[:, 0])


@gpu_requis
def test_c_un_systeme_in_place_et_repos_bien_equilibre():
    """Le chemin S=1 garde les propriétés du motif : sortie in-place
    autorisée, et C-property au repos (η constante, u = 0)."""
    import cupy as cp
    from src.f1_gpu.substrat_fusionne import pas_f_fusionne

    q = np.zeros((1, 1, 4, 12, 12), dtype=np.float32)
    q[:, :, 0] = 1.0
    q[:, :, 3] = 0.2
    q_gpu = cp.asarray(q)
    meme, _ = pas_f_fusionne(q_gpu, cp, sortie=q_gpu)
    assert meme is q_gpu
    np.testing.assert_allclose(vers_cpu(meme), q, atol=1e-6)


# ----- la garde reste fail-loud pour tout le reste -----

@gpu_requis
def test_garde_refuse_toujours_les_autres_comptes_de_systemes():
    """L'élargissement est BORNÉ à {1, 2} : 0, 3, 4 systèmes restent
    refusés (E4a), et le dtype non-f32 aussi."""
    import cupy as cp
    from src.f1_gpu.substrat_fusionne import pas_f_fusionne

    for n_systemes in (0, 3, 4):
        with pytest.raises(ValueError, match="E4a"):
            pas_f_fusionne(
                cp.zeros((1, n_systemes, 4, 8, 8), dtype=cp.float32), cp)
    with pytest.raises(ValueError, match="float32"):
        pas_f_fusionne(cp.zeros((1, 1, 4, 8, 8), dtype=cp.float64), cp)
