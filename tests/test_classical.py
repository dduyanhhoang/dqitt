import numpy as np
import pytest

from dqitt import paths
from dqitt.classical import cpsat_optimum, cpsat_ttqaoa_notebook, optimum
from dqitt.problem import Instance, load, violations

ENS = paths.DATA / "ttqaoa" / "ensembles"
TWO = Instance(["a", "b"], ["s", "s"], ["t0", "t1"], [{0, 1}, {0}], [(0, 1)], quota=[(0, 1), (1, 2)])


def test_violations_accepts_batches():
    assert violations(TWO, np.array([[1, 0], [0, 0], [1, -1]])).tolist() == [0, 3, 1]


def test_optimum_lists_every_best_assignment():
    v, best = optimum(TWO)
    assert v == 0
    assert best.tolist() == [[1, 0]]


def test_optimum_may_use_an_ineligible_teacher_to_meet_quota():
    # one section, t1 is not eligible but must teach at least one section
    inst = Instance(["a"], ["s"], ["t0", "t1"], [{0}], [], quota=[(0, 1), (1, 1)])
    v, best = optimum(inst)
    assert v == 1
    assert [1] in best.tolist() and [0] in best.tolist()


@pytest.mark.parametrize("name", ["inst_01", "inst_08", "inst_17"])
def test_cpsat_matches_brute_force(name):
    inst = load(ENS / name)
    v, best = optimum(inst)
    v_cp, a = cpsat_optimum(inst)
    assert v_cp == v
    assert violations(inst, a) == v


def test_notebook_model_returns_status_and_assignment():
    status, a = cpsat_ttqaoa_notebook(load(ENS / "inst_01"))
    assert status in {"OPTIMAL", "FEASIBLE", "INFEASIBLE"}
    assert a is None or a.shape == (8,)
