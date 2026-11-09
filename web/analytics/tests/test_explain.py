"""Tests for analytics.explain: SHAP additivity, margin derivation, verification,
plus integration checks on the exported explanations and game data.

    python -m unittest discover -s web/analytics/tests -t web
"""
import json
import os
import unittest

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from analytics import explain as E

WEB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(WEB, "src", "data")
COLS = list(E.FEATURES)


def toy_models(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(400, len(COLS)))
    yh = 110 + 3 * X[:, 0] - 2 * X[:, 7] + rng.normal(size=400)
    ya = 108 + 3 * X[:, 1] + 1.5 * X[:, 12] + rng.normal(size=400)
    kw = dict(n_estimators=40, max_depth=3, random_state=1)
    return GradientBoostingRegressor(**kw).fit(X, yh), GradientBoostingRegressor(**kw).fit(X, ya), X


class Attribution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mh, cls.ma, X = toy_models()
        cls.X = X[:25]
        cls.A = E.margin_attribution(cls.mh, cls.ma, cls.X)

    def test_additivity_reconstructs_unrounded_margin(self):
        A = self.A
        recon = A["base_margin"] + A["margin"].sum(1)
        np.testing.assert_allclose(recon, self.mh.predict(self.X) - self.ma.predict(self.X), atol=1e-6)

    def test_margin_is_home_minus_away(self):
        np.testing.assert_allclose(self.A["margin"], self.A["home"] - self.A["away"])
        self.assertAlmostEqual(self.A["base_margin"], self.A["base_home"] - self.A["base_away"])

    def test_sign_convention(self):
        # Raising feature 0 raises the home score -> positive margin contribution on average.
        hi = self.X.copy(); hi[:, 0] = 3
        lo = self.X.copy(); lo[:, 0] = -3
        self.assertGreater(E.margin_attribution(self.mh, self.ma, hi)["margin"][:, 0].mean(),
                           E.margin_attribution(self.mh, self.ma, lo)["margin"][:, 0].mean())

    def test_groups_sum_to_total(self):
        g = E.group_contributions(self.A["margin"][0], COLS)
        self.assertAlmostEqual(sum(g.values()), float(self.A["margin"][0].sum()), places=9)


class Verification(unittest.TestCase):
    def test_rounding_rules(self):
        ph, pa = np.array([110.0, 105.0]), np.array([100.0, 99.0])
        self.assertTrue(E.score_match(np.array([110.4, 104.6]), np.array([99.6, 99.4]), ph, pa).all())
        self.assertFalse(E.score_match(np.array([110.6]), np.array([100.0]), ph[:1], pa[:1])[0])
        self.assertTrue(E.margin_match(np.array([10.99]), ph[:1], pa[:1])[0])
        self.assertFalse(E.margin_match(np.array([11.2]), ph[:1], pa[:1])[0])

    def test_group_needs_most_forecasts_to_match(self):
        ph = np.full(10, 110.0); pa = np.full(10, 100.0)
        good = E.verify_group(np.full(10, 110.2), np.full(10, 100.1), ph, pa)
        chance = E.verify_group(np.r_[np.full(2, 110.2), np.full(8, 112.0)], np.full(10, 100.1), ph, pa)
        self.assertTrue(good[0])
        self.assertFalse(chance[0])        # 2 of 10 match: a different model

    def test_missing_inputs(self):
        X, ok = E.feature_matrix([{c: 1 for c in COLS}, {COLS[0]: 1}, None], COLS)
        self.assertEqual(ok.tolist(), [True, False, False])

    def test_summary_is_deterministic_and_signed(self):
        s = E.summary_sentence("DEN", "IND", 2.0, {"strength": 4.2, "form": -1.1, "availability": 0.2,
                                                   "rest": 0, "playoff": 0}, 5.1)
        self.assertIn("DEN by 5.1", s)
        self.assertIn("team strength toward DEN (+4.2)", s)
        self.assertIn("recent form away from DEN (-1.1)", s)
        self.assertNotIn("availability", s)          # below 0.5 points: not mentioned


@unittest.skipUnless(os.path.exists(os.path.join(DATA, "explanations.json")), "run the exporters first")
class ExportedData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ex = json.load(open(os.path.join(DATA, "explanations.json"), encoding="utf8"))
        cls.games = {g["id"]: g for g in json.load(open(os.path.join(DATA, "games.json"), encoding="utf8"))}

    def test_explanations_reconstruct_and_match_stored_forecast(self):
        n = 0
        for gid, e in self.ex["games"].items():
            if not e["ok"]:
                continue
            n += 1
            total = e["base"] + sum(v[2] for v in e["c"].values())
            self.assertAlmostEqual(total, e["pm"], delta=0.02, msg=gid)          # additivity (rounded export)
            fc = self.games[gid]["fc"]
            self.assertLessEqual(abs(e["pm"] - (fc["ph"] - fc["pa"])), E.MARGIN_TOL, gid)
            for c, v in e["c"].items():
                self.assertAlmostEqual(v[2], v[0] - v[1], delta=0.002, msg=f"{gid} {c}")
        self.assertGreater(n, 0)

    def test_saved_inputs_belong_to_the_game(self):
        for gid, e in self.ex["games"].items():
            if not e["ok"]:
                continue
            ins = self.games[gid].get("inputs")
            if ins and ins["home"].get("elo") is not None:
                self.assertAlmostEqual(e["x"]["home_elo"], ins["home"]["elo"], delta=0.01, msg=gid)
                self.assertAlmostEqual(e["x"]["away_elo"], ins["away"]["elo"], delta=0.01, msg=gid)

    def test_unverified_versions_are_never_explained(self):
        for name, st in self.ex["stats"].items():
            if not st.get("verified"):
                self.assertEqual(st.get("explained", 0), 0, name)
        for e in self.ex["games"].values():
            if not e["ok"]:
                self.assertNotIn("c", e)


if __name__ == "__main__":
    unittest.main()
