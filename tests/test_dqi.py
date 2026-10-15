import numpy as np
import pytest

from dqitt import paths
from dqitt.circuit import DQICircuit
from dqitt.dqi import all_x, expected_satisfied, ideal_state, theory_weights
from dqitt.problem import brute_force, build, code_distance, decoder_failure, load

DATA = paths.DATA / "dqitt"


@pytest.fixture(scope="module")
def tiny():
    return build(load(f"{DATA}/tiny"))


@pytest.fixture(scope="module")
def mini():
    return build(load(f"{DATA}/mini"))


def test_linsat_shape_and_optimum(tiny, mini):
    assert (tiny.p, tiny.n, tiny.m) == (3, 3, 6)
    assert brute_force(tiny)[0] == 6  # triangle is 3-colourable with eligibility
    assert (mini.p, mini.n, mini.m) == (3, 8, 20)
    assert brute_force(mini)[0] == 18  # two K4 cliques: one conflict each unsatisfiable


def test_conflict_rows_are_signed_incidence(mini):
    conf = build(load(f"{DATA}/mini"), eligibility=False)
    assert ((conf.B != 0).sum(axis=1) == 2).all()
    assert (conf.B.sum(axis=1) % conf.p == 0).all()
    assert code_distance(conf) == 3  # girth of K4


def test_decoder_failure_layers(tiny):
    eps = decoder_failure(tiny, 2)
    assert eps[0] == eps[1] == 0
    assert eps[2] == 1.0  # every weight-2 error collides through a triangle


def test_ideal_state_normalized_and_beats_random(mini):
    X = all_x(mini.p, mini.n)
    random = mini.count(X).mean()
    prev = random
    for l in (1, 2, 3):
        psi, X, w = ideal_state(mini, l)
        assert np.isclose(np.linalg.norm(psi), 1)
        s = expected_satisfied(psi, mini, X)
        assert s > prev
        prev = s


def test_theory_matches_exact_when_code_distance_large():
    # path graph a-b-c: ker(B^T) trivial => d_code = inf, tridiagonal formula is exact
    from dqitt.problem import Linsat

    p = 3
    B = np.array([[1, p - 1, 0], [0, 1, p - 1]])
    lin = Linsat(p, B, np.tile(np.arange(p) != 0, (2, 1)))
    for l in (1, 2):
        w_th, frac_th = theory_weights(lin.m, l, p, p - 1)
        psi, X, w = ideal_state(lin, l)
        assert np.isclose(expected_satisfied(psi, lin, X) / lin.m, frac_th)
        assert np.allclose(np.abs(w), np.abs(w_th), atol=1e-8)


def test_circuit_equals_ideal_at_l1(tiny):
    psi, X, w = ideal_state(tiny, 1)
    pr = DQICircuit(tiny, 1, w, device="lightning.qubit").probs_x()
    assert np.allclose(pr, np.abs(psi) ** 2, atol=1e-10)


def test_circuit_f2_case():
    # 2 teachers -> p = 2: plain max-cut on a triangle, 1 qubit per symbol
    from dqitt.problem import Instance

    inst = Instance(["a", "b", "c"], ["s", "s", "s"], ["t1", "t2"], [{0, 1}] * 3, [(0, 1), (1, 2), (0, 2)])
    lin = build(inst)
    assert lin.p == 2 and lin.m == 3
    psi, X, w = ideal_state(lin, 1)
    pr = DQICircuit(lin, 1, w).probs_x()
    assert np.allclose(pr, np.abs(psi) ** 2, atol=1e-10)
