import sys, unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from survey import ratio, covariance, equal_system, decomposition, holm, groups, joint
from adjustment import fit_logit
from target import target_mask
import pandas as pd
from scipy.special import expit
from scipy.optimize import minimize


class SurveyTests(unittest.TestCase):
    def test_ratio_and_fay_independent_calculation(self):
        W = np.ones((4, 81))
        W[:, 1:] = np.tile([[1.5], [0.5], [1.5], [0.5]], (1, 80))
        y = np.array([1.0, 0, 0, 0])
        q = ratio(y, np.ones(4, bool), W)
        self.assertAlmostEqual(q[0], 0.25)
        self.assertAlmostEqual(q[1], 0.375)
        self.assertAlmostEqual(covariance(q[:, None])[0, 0], 80 * (0.375 - 0.25) ** 2 / 20)

    def test_contrast_keeps_covariance(self):
        q = np.vstack([np.array([0.1, 0.2])] + [np.array([0.11, 0.21])] * 80)
        c = covariance(q)
        a = np.array([1.0, -1.0])
        self.assertAlmostEqual(a @ c @ a, 0, places=12)

    def test_independent_system_aggregation(self):
        a = np.arange(81)[:, None] / 1000
        b = a * 2
        p, c = equal_system([a, b])
        self.assertAlmostEqual(p[0], 0)
        np.testing.assert_allclose(c, (covariance(a) + covariance(b)) / 4)

    def test_school_identity_and_known_slopes(self):
        x = np.array([0.0, 1, 2, 3])
        y = 2 * x + 3
        g = np.array([1, 1, 2, 2])
        W = np.ones((4, 81))
        e = decomposition(x, y, g, W)
        np.testing.assert_allclose(e[:, 0:3], 2.0)
        np.testing.assert_allclose(e[:, 3] + e[:, 4], e[:, 0])

    def test_ties_and_multiplicity(self):
        np.testing.assert_equal(groups(np.array([0, 1, 2, 3]), [1, 2, 3]), [0, 0, 1, 2])
        np.testing.assert_allclose(holm([0.04, 0.01]), [0.04, 0.02])
        stat, df, p = joint(np.array([1.0, 0]), np.eye(2))
        self.assertEqual(df, 2)
        self.assertAlmostEqual(stat, 1)

    def test_zero_denominator_is_error(self):
        with self.assertRaises(ValueError):
            ratio(np.ones(3), np.zeros(3, bool), np.ones((3, 81)))

    def test_regularized_logit_separation_against_optimizer(self):
        X = np.column_stack([np.ones(8), np.array([0, 0, 0, 0, 1, 1, 1, 1])])
        y = np.array([0.0, 1, 0, 1, 0, 0, 0, 0])
        w = np.arange(1.0, 9)
        b, it = fit_logit(X, y, w)
        wn = w / w.mean()

        def objective(z):
            return np.sum(wn * (np.logaddexp(0, X @ z) - y * (X @ z))) + 0.5 * z[1] ** 2

        expected = minimize(objective, [0.0, 0.0], method="BFGS", tol=1e-10)
        np.testing.assert_allclose(b, expected.x, atol=1e-6)
        self.assertTrue(np.isfinite(b).all())

    def test_selection_eligibility_missing_invalid_zero_repweights(self):
        d = pd.DataFrame(
            {
                "ESCS": [0.0, 0, np.nan, 0, 0, 0, 0, np.inf],
                "Option_ICTQ": [1, 0, 1, 1, 1, 1, 1, 1],
                "W_FSTUWT": [1.0, 1, 1, 0, 1, 1, 1, 1],
                "rep": [1.0, 1, 1, 1, np.nan, -1, 0, 1],
            }
        )
        np.testing.assert_equal(
            target_mask(d, ["W_FSTUWT", "rep"]),
            [True, False, False, False, False, False, True, False],
        )


if __name__ == "__main__":
    unittest.main()
