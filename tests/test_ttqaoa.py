import numpy as np
import pytest

from dqitt import paths

pytest.importorskip("qmos_qaoa")
from dqitt.ttqaoa import bits_to_assignment, energies, qubo  # noqa: E402

INST01 = paths.DATA / "ttqaoa" / "ensembles" / "inst_01"


@pytest.fixture(scope="module")
def problem():
    return qubo(INST01)


def test_energy_of_empty_schedule(problem):
    # every section penalised W_s^2 (0-1)^2 = 9; quota W_q^2 sum_t (q_min^2 + q_max^2) = 5+5+5+17
    assert energies(problem)[0] == pytest.approx(9 * 8 + 32)


def test_bits_to_assignment_maps_ttqaoa_variable_order(problem):
    bits = np.zeros(problem.n_qubits, int)
    t, s = problem.feasible_vars[0]
    bits[0] = 1
    a, valid = bits_to_assignment(bits, problem)
    assert valid
    assert a[s] == t
    assert (np.delete(a, s) == -1).all()


def test_bits_to_assignment_flags_sections_with_two_teachers(problem):
    s = problem.feasible_vars[0][1]
    bits = np.array([ss == s for _, ss in problem.feasible_vars], int)  # every teacher of section s
    assert not bits_to_assignment(bits, problem)[1]


def test_qaoa_probs_uniform_at_zero_angles(problem):
    from dqitt.ttqaoa import qaoa_probs

    pr = qaoa_probs(problem, np.zeros((1, 2)))
    assert pr.shape == (2**problem.n_qubits,)
    assert np.allclose(pr, 2.0**-problem.n_qubits)
