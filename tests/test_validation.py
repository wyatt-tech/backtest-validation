import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import validation as v  # noqa: E402


class TestBasics(unittest.TestCase):
    def test_sharpe_known_series(self):
        r = np.array([0.01, -0.01, 0.02, 0.0])
        expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
        self.assertAlmostEqual(v.sharpe(r), expected, places=12)
        self.assertEqual(v.sharpe([0.01, 0.01, 0.01]), 0.0)  # zero variance -> 0, not a division error

    def test_psr_rises_with_the_mean(self):
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 0.01, 1000)
        lo = v.probabilistic_sharpe_ratio(noise)
        hi = v.probabilistic_sharpe_ratio(noise + 0.002)
        self.assertTrue(0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0)
        self.assertGreater(hi, lo)

    def test_expected_max_sharpe_grows_with_trials(self):
        a = v.expected_max_sharpe(1e-4, 10)
        b = v.expected_max_sharpe(1e-4, 1000)
        self.assertGreater(b, a)
        self.assertEqual(v.expected_max_sharpe(1e-4, 1), 0.0)

    def test_benjamini_hochberg_textbook_case(self):
        mask = v.benjamini_hochberg([0.01, 0.02, 0.03, 0.5], alpha=0.05)
        self.assertEqual(mask.tolist(), [True, True, True, False])

    def test_bonferroni(self):
        self.assertEqual(v.bonferroni([0.01, 0.03], alpha=0.05).tolist(), [True, False])  # cut = 0.05 / 2 = 0.025


class TestWalkForward(unittest.TestCase):
    def test_purge_gap_and_order(self):
        purge = 7
        folds = list(v.purged_walkforward(600, n_splits=5, purge=purge))
        self.assertEqual(len(folds), 5)
        for train, test in folds:
            self.assertEqual(test.min() - train.max() - 1, purge)  # exactly `purge` observations between them
            self.assertTrue((train < test.min()).all())

    def test_too_short_yields_nothing(self):
        self.assertEqual(list(v.purged_walkforward(3, n_splits=5)), [])


class TestCalibrationAndPower(unittest.TestCase):
    """Small, fast versions of the module's self-test."""

    def test_noise_searches_are_mostly_rejected(self):
        rng = np.random.default_rng(1)
        M, T, L = 40, 500, 40
        fp = 0
        for _ in range(M):
            R = rng.normal(0, 0.01, size=(T, L))
            srs = np.array([v.sharpe(R[:, l]) for l in range(L)])
            if v.deflated_sharpe_ratio(R[:, int(np.argmax(srs))], all_trial_srs=srs) > 0.95:
                fp += 1
        self.assertLessEqual(fp / M, 0.15)

    def test_a_real_edge_passes_alone(self):
        rng = np.random.default_rng(3)
        edge = rng.normal(2.0 / np.sqrt(252) * 0.01, 0.01, size=(2500, 1))
        r = v.evaluate_search(edge, n_boot=300, seed=1)
        self.assertTrue(r["verdict"].startswith("EDGE"), r)


if __name__ == "__main__":
    unittest.main()
