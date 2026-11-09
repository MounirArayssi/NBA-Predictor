"""Unit tests for analytics.metrics, using small hand-checked examples.

    python -m unittest discover -s web/analytics/tests -t web
"""
import math
import unittest
from statistics import stdev

from analytics import metrics as M


def row(id, d, h, a, ph, pa, p, hs, as_, s="2025-26", po=False):
    return dict(id=id, d=d, s=s, po=po, h=h, a=a, ph=ph, pa=pa, p=p, hs=hs, as_=as_)


# BOS home vs NYK, projected 110-100 (BOS 70%), final 95-105: NYK wins.
G1 = row("g1", "2025-11-01", "BOS", "NYK", 110, 100, 0.70, 95, 105)
# NYK home vs BOS, projected 104-108 (NYK 40%), final 99-120: BOS wins.
G2 = row("g2", "2025-11-03", "NYK", "BOS", 104, 108, 0.40, 99, 120)
# BOS home vs MIA, projected 112-105 (BOS 75%), final 118-101: BOS wins.
G3 = row("g3", "2025-11-05", "BOS", "MIA", 112, 105, 0.75, 118, 101)
ROWS = [G1, G2, G3]


class Orientation(unittest.TestCase):
    def test_home_and_away_orientation(self):
        g = {x["id"]: x for x in M.team_games(ROWS, "BOS")}
        # g1: BOS home. actual margin -10, predicted +10 -> margin error -20
        self.assertTrue(g["g1"]["home"])
        self.assertEqual((g["g1"]["tp"], g["g1"]["op"], g["g1"]["ts"], g["g1"]["os"]), (110, 100, 95, 105))
        self.assertEqual(g["g1"]["me"], -20)
        self.assertEqual(g["g1"]["score_err"], -15)
        self.assertAlmostEqual(g["g1"]["p"], 0.70)
        self.assertFalse(g["g1"]["won"])
        self.assertFalse(g["g1"]["ok"])
        # g2: BOS away. team projected 108-104 (+4), actual 120-99 (+21) -> +17
        self.assertFalse(g["g2"]["home"])
        self.assertEqual((g["g2"]["tp"], g["g2"]["op"], g["g2"]["ts"], g["g2"]["os"]), (108, 104, 120, 99))
        self.assertEqual(g["g2"]["me"], 17)
        self.assertAlmostEqual(g["g2"]["p"], 0.60)     # 1 - home probability
        self.assertTrue(g["g2"]["won"] and g["g2"]["ok"])

    def test_opponent_view_is_mirror(self):
        b = {x["id"]: x for x in M.team_games(ROWS, "BOS")}
        n = {x["id"]: x for x in M.team_games(ROWS, "NYK")}
        for k in ("g1", "g2"):
            self.assertAlmostEqual(b[k]["me"], -n[k]["me"])
            self.assertAlmostEqual(b[k]["p"], 1 - n[k]["p"])
            self.assertNotEqual(b[k]["won"], n[k]["won"])

    def test_date_order(self):
        self.assertEqual([x["id"] for x in M.team_games(list(reversed(ROWS)), "BOS")], ["g1", "g2", "g3"])


class Summary(unittest.TestCase):
    def setUp(self):
        self.s = M.team_summary(M.team_games(ROWS, "BOS"))

    def test_errors(self):
        me = [-20, 17, 10]                      # g3: actual +17, predicted +7
        self.assertEqual(self.s["n"], 3)
        self.assertAlmostEqual(self.s["bias"], round(sum(me) / 3, 2))
        self.assertAlmostEqual(self.s["mae"], round(sum(map(abs, me)) / 3, 2))
        self.assertAlmostEqual(self.s["sd"], round(stdev(me), 2))   # sample SD (n-1)

    def test_wins_and_probability(self):
        self.assertAlmostEqual(self.s["exp_w"], 0.70 + 0.60 + 0.75)
        self.assertEqual(self.s["act_w"], 2)
        self.assertAlmostEqual(self.s["gap"], round(2 - 2.05, 2))
        brier = ((0.70 - 0) ** 2 + (0.60 - 1) ** 2 + (0.75 - 1) ** 2) / 3
        self.assertAlmostEqual(self.s["brier"], round(brier, 4))
        ll = (-math.log(0.30) - math.log(0.60) - math.log(0.75)) / 3
        self.assertAlmostEqual(self.s["logloss"], round(ll, 4))
        self.assertAlmostEqual(self.s["acc"], round(100 * 2 / 3, 1))

    def test_small_sample_safeguards(self):
        self.assertEqual(self.s["tier"], "small")
        self.assertIsNone(self.s["mae_ci"])          # < 5 games: no interval
        self.assertIsNone(M.bias_statement("BOS", self.s))
        self.assertEqual(M.tier(19), "small")
        self.assertEqual(M.tier(20), "moderate")
        self.assertEqual(M.tier(60), "stable")

    def test_empty(self):
        self.assertEqual(M.team_summary([]), dict(n=0, tier="small"))

    def test_bias_statement_needs_interval_excluding_zero(self):
        rows = [row(f"x{i}", f"2025-12-{i + 1:02d}", "BOS", "LAL", 110, 108, 0.6, 115, 100) for i in range(25)]
        s = M.team_summary(M.team_games(rows, "BOS"))   # every game +13 better than forecast
        self.assertEqual(s["tier"], "moderate")
        self.assertIn("better than the forecast margin", M.bias_statement("BOS", s))
        noisy = [row(f"y{i}", f"2025-12-{i + 1:02d}", "BOS", "LAL", 110, 108, 0.6, 110 + (10 if i % 2 else -10), 108)
                 for i in range(26)]
        self.assertIsNone(M.bias_statement("BOS", M.team_summary(M.team_games(noisy, "BOS"))))

    def test_bootstrap_is_reproducible(self):
        vals = [float(i % 7) for i in range(40)]
        self.assertEqual(M._boot_ci(vals, lambda v: sum(v) / len(v)), M._boot_ci(vals, lambda v: sum(v) / len(v)))


class Surprises(unittest.TestCase):
    def test_actual_winner_and_probability(self):
        o = M.outcome(G1)                       # NYK (away) won; home p 0.70 -> NYK 0.30
        self.assertEqual(o["winner"], "NYK")
        self.assertAlmostEqual(o["p_winner"], 0.30)
        self.assertAlmostEqual(o["surprise"], -math.log(0.30))
        o3 = M.outcome(G3)
        self.assertEqual(o3["winner"], "BOS")
        self.assertAlmostEqual(o3["p_winner"], 0.75)

    def test_scoreline_error(self):
        self.assertEqual(M.scoreline_error(G1), 20)   # actual -10, predicted +10
        self.assertEqual(M.scoreline_error(G2), 17)   # actual home -21, predicted -4

    def test_ordering(self):
        self.assertEqual([r["id"] for r in M.rank_outcome(ROWS)], ["g1", "g2", "g3"])  # 0.30, 0.60, 0.75
        self.assertEqual([r["id"] for r in M.rank_scoreline(ROWS)], ["g1", "g2", "g3"])  # 20, 17, 10
        self.assertEqual(len(M.rank_outcome(ROWS, 2)), 2)

    def test_extreme_probability_is_clamped_only_in_the_log(self):
        r = M.annotate(row("z", "2025-11-09", "BOS", "NYK", 120, 90, 1.0, 90, 100))
        self.assertEqual(r["pw"], 0.0)                         # displayed value untouched
        self.assertAlmostEqual(r["su"], round(-math.log(M.P_FLOOR), 3))
        self.assertTrue(math.isfinite(r["su"]))

    def test_non_final_and_incomplete_rows_are_excluded(self):
        pending = dict(G1, id="p", hs=None, as_=None)
        noprob = dict(G1, id="q", p=None)
        self.assertEqual({r["id"] for r in M.rank_outcome(ROWS + [pending, noprob])}, {"g1", "g2", "g3"})
        self.assertEqual(len(M.team_games(ROWS + [pending], "BOS")), 3)

    def test_team_surprises(self):
        t = M.team_surprises(ROWS, "BOS", k=5)
        self.assertEqual([r["id"] for r in t["wins"]], ["g2", "g3"])   # BOS won at 0.60 and 0.75
        self.assertEqual([r["id"] for r in t["losses"]], ["g1"])
        self.assertEqual([r["id"] for r in t["recent"]], ["g1"])       # only miss: winner had < 50%

    def test_duplicates_rejected(self):
        with self.assertRaises(ValueError):
            M.dedupe([G1, dict(G1)])


if __name__ == "__main__":
    unittest.main()
