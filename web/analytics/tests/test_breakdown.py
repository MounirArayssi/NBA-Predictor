"""Unit tests for analytics.breakdown.

    python -m unittest discover -s web/analytics/tests -t web
"""
import unittest

import numpy as np

from analytics import breakdown as B


class FourFactors(unittest.TestCase):
    def test_hand_checked_game(self):
        # 40-85 FG, 12 threes, 18-24 FT, 14 TOV, 10 OREB vs 33 opponent DREB.
        f = B.four_factors(40, 85, 12, 24, 18, 14, 10, 33)
        self.assertAlmostEqual(float(f["efg"]), 46 / 85)
        self.assertAlmostEqual(float(f["tov"]), 14 / (85 + 0.44 * 24 + 14))
        self.assertAlmostEqual(float(f["orb"]), 10 / 43)
        self.assertAlmostEqual(float(f["ftr"]), 18 / 85)


class Fit(unittest.TestCase):
    def test_recovers_exact_linear_relation(self):
        rng = np.random.default_rng(0)
        F = rng.normal([0.54, 0.13, 0.25, 0.2], 0.03, size=(500, 4))
        coef = np.array([180.0, -120.0, 40.0, 30.0])
        y = 5 + F @ coef
        fit = B.fit_rating(F, y)
        self.assertAlmostEqual(fit["intercept"], 5, places=6)
        for k, b in zip(B.FACTORS, coef):
            self.assertAlmostEqual(fit["coef"][k], b, places=6)
        self.assertAlmostEqual(fit["r2"], 1.0, places=9)

    def test_ignores_rows_with_missing_values(self):
        F = np.array([[0.5, 0.1, 0.2, 0.2], [np.nan, 0.1, 0.2, 0.2], [0.6, 0.12, 0.3, 0.25],
                      [0.55, 0.14, 0.22, 0.18], [0.52, 0.11, 0.27, 0.21], [0.58, 0.13, 0.24, 0.3]])
        y = np.array([110, 999, 120, 112, 113, 119.0])
        self.assertEqual(B.fit_rating(F, y)["n"], 5)


class Split(unittest.TestCase):
    def test_parts_sum_to_gap_exactly(self):
        coef = {"efg": 180.0, "tov": -120.0, "orb": 40.0, "ftr": 30.0}
        lf = {"efg": 0.54, "tov": 0.13, "orb": 0.25, "ftr": 0.20}
        f = {"efg": 0.56, "tov": 0.12, "orb": 0.24, "ftr": 0.22}
        parts = B.split_rating(118.0, f, 114.5, lf, coef)
        self.assertAlmostEqual(sum(parts.values()), 3.5)
        self.assertAlmostEqual(parts["efg"], 3.6)      # +2 pts eFG x 1.8
        self.assertAlmostEqual(parts["tov"], 1.2)      # 1 pt fewer turnovers
        self.assertAlmostEqual(parts["orb"], -0.4)
        self.assertAlmostEqual(parts["ftr"], 0.6)
        self.assertAlmostEqual(parts["other"], 3.5 - 5.0)


class Matches(unittest.TestCase):
    def test_tolerance(self):
        self.assertTrue(B.matches(107.83, 107.8, 0.05))
        self.assertFalse(B.matches(107.83, 107.7, 0.05))
        self.assertFalse(B.matches(None, 1, 0.05))
        self.assertTrue(B.matches(2, 2.0, 0.0))


if __name__ == "__main__":
    unittest.main()
