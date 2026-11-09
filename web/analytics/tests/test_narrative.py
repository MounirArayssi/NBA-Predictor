"""Unit tests for analytics.narrative.

    python -m unittest discover -s web/analytics/tests -t web
"""
import unittest

from analytics import narrative as N

LG = {"f": {"efg": 0.533, "tov": 0.133, "orb": 0.273, "ftr": 0.228}}


def unit(parts, f, ok=True, orb_missing=0):
    return {"ok": ok, "parts": parts, "f": f, "orb_missing": orb_missing}


# Sacramento, 2026-11-08: bad at everything on offense, leaky defense.
SAC_OFF = unit({"efg": -1.1, "tov": -0.5, "orb": -1.0, "ftr": -0.78, "other": -1.05},
               {"efg": 0.526, "tov": 0.137, "orb": 0.251, "ftr": 0.200})
SAC_DEF = unit({"efg": 6.0, "tov": 1.7, "orb": -0.1, "ftr": 1.4, "other": -0.4},
               {"efg": 0.573, "tov": 0.121, "orb": 0.271, "ftr": 0.278})


class Units(unittest.TestCase):
    def test_defense_biggest_factor_spelled_out(self):
        s = N.unit_sentences("def", SAC_DEF, LG)
        self.assertEqual(s[0], "Their defense lets opponents shoot 57% eFG against a 53% league average, "
                               "which alone costs about 6 points per 100 possessions.")
        self.assertEqual(s[1], "Smaller factors: forcing turnovers −1.7, free throws allowed −1.4.")

    def test_offense_below_average_at_everything(self):
        s = N.unit_sentences("off", SAC_OFF, LG)
        self.assertTrue(s[0].startswith("Their offense shoots 53% eFG"))
        self.assertIn("costs about 1.1 points", s[0])
        self.assertEqual(s[1], "Their offense is below average at everything: shooting, turnovers, rebounding and free throws.")

    def test_good_defense_wording(self):
        x = unit({"efg": -1.9, "tov": 0.2, "orb": 1.2, "ftr": 0.1, "other": 0.0},
                 {"efg": 0.52, "tov": 0.13, "orb": 0.28, "ftr": 0.22})
        self.assertTrue(N.unit_sentences("def", x, LG)[0].startswith(
            "Their defense holds opponents to 52% eFG against a 53% league average, which alone adds about 1.9"))

    def test_quiet_unit(self):
        x = unit({"efg": 0.4, "tov": -0.2, "orb": 0.1, "ftr": 0.0, "other": 0.1},
                 {"efg": 0.54, "tov": 0.13, "orb": 0.27, "ftr": 0.23})
        self.assertEqual(N.unit_sentences("off", x, LG), ["Their offense is close to league average on all four factors."])

    def test_missing_rebounds_not_mentioned_and_no_everything_claim(self):
        x = unit({"efg": -2.0, "tov": -0.5, "orb": 0.0, "ftr": -0.6, "other": 0.0},
                 {"efg": 0.52, "tov": 0.14, "orb": None, "ftr": 0.2}, orb_missing=2)
        s = " ".join(N.unit_sentences("off", x, LG))
        self.assertNotIn("everything", s)
        self.assertNotIn("rebound", s)

    def test_unverified_unit_says_nothing(self):
        self.assertEqual(N.unit_sentences("off", unit(None, {}, ok=False), LG), [])


class Team(unittest.TestCase):
    def test_full_story(self):
        s = {
            "margin": {"ok": True, "n": 9, "w_n": 1, "cur": -14.1, "pf": 109.2, "pa": 123.3, "w": 0.45, "last": -10.0},
            "off": SAC_OFF, "def": SAC_DEF,
            "missing": {"ok": True, "v": 15.3, "players": [{"n": "Domantas Sabonis", "ppg": 15.3}]},
            "rest": {"ok": True, "v": 2},
        }
        story = N.team_story("SAC", s, LG)
        self.assertEqual(story[0], "SAC is 1–8, losing by 14 a game on average (scoring 109, allowing 123). "
                                   "Last season they were a −10 team, which still counts for 55% of their season margin.")
        self.assertEqual(story[-1], "Domantas Sabonis (15 ppg) is out.")
        self.assertFalse(any("back-to-back" in x for x in story))

    def test_back_to_back_and_unverified_inputs_skipped(self):
        s = {"margin": {"ok": False}, "off": {"ok": False}, "def": {"ok": False},
             "missing": {"ok": True, "v": 0, "players": []}, "rest": {"ok": True, "v": 1}}
        self.assertEqual(N.team_story("BKN", s, LG), ["They're on the second night of a back-to-back."])

    def test_two_players_out(self):
        mi = {"ok": True, "v": 20.6, "players": [{"n": "A", "ppg": 12.0}, {"n": "B", "ppg": 8.6}]}
        self.assertEqual(N.availability_sentence("BKN", mi), "A and B are out (21 rotation points a game combined).")


if __name__ == "__main__":
    unittest.main()
