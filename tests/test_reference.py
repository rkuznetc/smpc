from conftest import COMMON

from smpc import reference


def test_intersection(bank, telecom):
    assert reference.intersection(bank, telecom) == COMMON


def test_intersection_target_sum(bank, telecom):
    # TARGET общих 100006..100010 = [0, 1, 0, 0, 1]
    assert reference.intersection_target_sum(bank, telecom) == 2


def test_bin_stats_totals(bank, telecom):
    stats = reference.bin_stats(bank, telecom, "DAYS_LAST_PHONE_CHANGE")
    assert stats["N"].sum() == len(COMMON)
    assert stats["defaults"].sum() == 2
