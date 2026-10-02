"""Meaningful edge cases for partial outcomes and joint accounting."""

import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MNAR = load_module("mnar_revision", "mnar.py")
JOINT = load_module("joint_revision", "joint.py")


class SensitivityTests(unittest.TestCase):
    def test_partial_information_is_preserved_under_extreme_shifts(self):
        s = np.array([6, 1, 6, np.nan, np.nan, 99, 2, 99.0])
        o = np.array([6, np.nan, np.nan, 6, np.nan, 99, 6, 3.0])
        one, zero, patterns, full = MNAR.partial_outcomes(s, o)
        self.assertEqual(one.sum(), 1)
        self.assertEqual(zero.sum(), 3)
        self.assertEqual(sum(p.sum() for p in patterns), 4)
        A = np.full(len(s), 0.1)
        low = MNAR.expected_outcome(one, zero, A, -1000)
        high = MNAR.expected_outcome(one, zero, A, 1000)
        np.testing.assert_array_equal(low[one], [1])
        np.testing.assert_array_equal(high[zero], [0, 0, 0])
        self.assertAlmostEqual(low.mean(), 1 / 8)
        self.assertAlmostEqual(high.mean(), 5 / 8)

    def test_mnar_monotonicity_and_weighted_bounds(self):
        q = np.array([0, 0, 0, 3, 3, 3])
        one = np.array([1, 0, 0, 0, 0, 1], dtype=bool)
        zero = np.array([0, 1, 0, 1, 0, 0], dtype=bool)
        A = np.full((6, 81), 0.2)
        W = np.tile(np.array([2, 1, 3, 2, 4, 1.0])[:, None], (1, 81))
        cache = q, one, zero, A, W
        np.testing.assert_allclose(MNAR.risk_for_delta(cache, 0, -1000), 2 / 6)
        np.testing.assert_allclose(MNAR.risk_for_delta(cache, 0, 1000), 5 / 6)
        self.assertTrue(
            np.all(MNAR.risk_for_delta(cache, 3, 1) > MNAR.risk_for_delta(cache, 3, -1))
        )
        profiles = [MNAR.aggregate_profile({"A": cache}, 0, z)[0] for z in [-2, 0, 2]]
        self.assertTrue(profiles[0] > profiles[1] > profiles[2])

    def test_joint_accounting_does_not_use_average_margin_products(self):
        e = JOINT.joint_components(
            np.array([0.1, 0.6]),
            np.array([0.2, 0.2]),
            np.array([0.8, 0.1]),
            np.array([0.1, 0.5]),
            np.array([0.08, 0.06]),
            np.array([0.02, 0.1]),
        )
        np.testing.assert_allclose(e[:, 8], e[:, 9] + e[:, 10] + e[:, 11], atol=1e-15)
        self.assertGreater(
            abs(np.mean(e[:, 6]) - (np.mean(e[:, 4]) - np.mean(e[:, 0]) * np.mean(e[:, 2]))), 0.01
        )


if __name__ == "__main__":
    unittest.main()
