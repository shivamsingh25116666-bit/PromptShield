"""Tests for the evaluation benchmark suite."""

from tests.evaluate import run_evaluation


def test_reproducible_evaluation_metrics():
    summary = run_evaluation()

    assert summary["total_cases"] >= 40
    assert summary["correct_classifications"] > 0
    assert summary["accuracy"] >= 85.0
    assert summary["precision"] >= 90.0
    assert summary["recall"] >= 85.0
    assert summary["false_positives"] == 0

    pv = summary["policy_verification"]
    assert pv["critical_attacks_tested"] >= 10
    assert pv["critical_attacks_blocked"] == pv["critical_attacks_tested"]
    assert pv["harmless_normal_allowed"] == pv["harmless_normal_tested"]
