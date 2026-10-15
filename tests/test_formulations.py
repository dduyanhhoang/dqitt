import numpy as np
import pytest

from dqitt import paths
from dqitt.formulations import binary, to_assignment
from dqitt.problem import Instance, code_distance, load, violations

ENS = paths.DATA / "ttqaoa" / "ensembles"

# a, b share a slot; a can take t0 or t1, b only t0. t0 may teach at most one section.
TWO = Instance(["a", "b"], ["s", "s"], ["t0", "t1"], [{0, 1}, {0}], [(0, 1)], quota=[(0, 1), (1, 2)])


@pytest.fixture(scope="module")
def inst01():
    return load(ENS / "inst_01")


def test_load_reads_quota(inst01):
    assert inst01.quota == [(1, 2), (1, 2), (1, 2), (1, 4)]


def test_load_treats_missing_slot_as_unavailable():
    toy = load(paths.DATA / "ttqaoa" / "toy")
    assert toy.eligible[toy.sections.index("AI1914")] == set()  # slot A35 absent from InstructorSlot


def test_violations_counts_each_broken_constraint():
    assert violations(TWO, np.array([1, 0])) == 0
    # both on t0: conflict, t0 over quota, t1 under quota
    assert violations(TWO, np.array([0, 0])) == 3
    # b unassigned: missing teacher, t0 under quota is fine (min 0), t1 count 1 ok
    assert violations(TWO, np.array([1, -1])) == 1


def test_binary_hard_rows_and_field():
    lin, pairs = binary(TWO, quota="hard")
    assert pairs == [(0, 0), (0, 1), (1, 0)]
    assert lin.p == 3  # counts reach 2, so p must exceed 2
    assert lin.m == 3 + 2 + 1 + 2  # domain + one-hot + conflict + quota


def test_binary_hard_scores_valid_schedule_full():
    lin, pairs = binary(TWO, quota="hard")
    assert lin.count(np.array([0, 1, 1])) == lin.m  # a->t1, b->t0
    assert lin.count(np.array([1, 0, 1])) == lin.m - 3  # conflict + both quotas broken


def test_to_assignment_rejects_non_one_hot():
    _, pairs = binary(TWO, quota="hard")
    assert to_assignment(np.array([0, 1, 1]), pairs, 2).tolist() == [1, 0]
    assert to_assignment(np.array([1, 1, 1]), pairs, 2) is None
    assert to_assignment(np.array([0, 2, 1]), pairs, 2) is None  # value outside {0, 1}


def test_binary_hard_distance_is_three(inst01):
    lin, _ = binary(inst01, quota="hard")
    assert code_distance(lin, max_w=3) == 3


def test_binary_soft_quota_repeats_rows_so_distance_is_two(inst01):
    lin, _ = binary(inst01, quota="soft")
    assert code_distance(lin, max_w=2) == 2
