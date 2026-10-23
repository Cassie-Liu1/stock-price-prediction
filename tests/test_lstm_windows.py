import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lstm_stock_price_prediction import target_indices, make_windows


class LSTMWindowTests(unittest.TestCase):
    def test_exact_horizon_and_window(self):
        features = np.arange(400, dtype=float).reshape(-1, 1)
        ids = np.array([149, 150])
        x, y = make_windows(features, features, ids, 90, 60)
        self.assertEqual(x.shape, (2, 90, 1))
        self.assertEqual(x[0, 0, 0], 0)
        self.assertEqual(x[0, -1, 0], 89)
        self.assertEqual(y[0, 0] - x[0, -1, 0], 60)

    def test_selection_labels_precede_later_forecast_origins(self):
        ids = target_indices(4006, 2804, 3405, 90, 60)
        self.assertEqual(len(ids["train"]), 2655)
        self.assertEqual(len(ids["validation"]), 542)
        self.assertEqual(len(ids["test"]), 542)
        self.assertLessEqual(ids["train"][-1], ids["validation"][0] - 60)
        self.assertLessEqual(ids["validation"][-1], ids["test"][0] - 60)
        for a, b in [("train", "validation"), ("validation", "test")]:
            self.assertEqual(len(np.intersect1d(ids[a], ids[b])), 0)

    def test_short_validation_period_is_rejected(self):
        with self.assertRaises(ValueError):
            target_indices(400, 200, 220, 90, 60)


if __name__ == "__main__":
    unittest.main()
