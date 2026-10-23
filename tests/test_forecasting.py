"""Checks for chronological target alignment and residual forecasting behavior."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from transformer_stock_price_prediction_original import WindowDataset, TransformerPredictor
from walk_forward import make_folds


class ForecastingTests(unittest.TestCase):
    def test_walk_forward_tests_are_disjoint_and_cover_tail(self):
        folds = make_folds(4006, 2400, 252, 400)
        self.assertEqual(len(folds), 4)
        self.assertEqual(folds[-1][2], 4006)
        for train_end, val_end, test_end in folds:
            self.assertLess(train_end, val_end)
            self.assertLess(val_end, test_end)
        for left, right in zip(folds, folds[1:]):
            self.assertEqual(left[2], right[1])
            self.assertGreater(right[0], left[0])

    def test_later_split_uses_past_context_and_exact_targets(self):
        values = np.arange(20, dtype=np.float32).reshape(-1, 1)
        data = WindowDataset(values[10 - 4:15], 4)
        self.assertEqual(len(data), 5)
        for i in range(len(data)):
            x, y = data[i]
            self.assertEqual(y.item(), 10 + i)
            self.assertEqual(x[-1].item(), 9 + i)
            self.assertTrue(torch.all(x < y))

    def test_initial_residual_model_matches_previous_close(self):
        model = TransformerPredictor(4).eval()
        x = torch.randn(3, 4, 1)
        with torch.inference_mode():
            torch.testing.assert_close(model(x), x[:, -1, :])

    def test_residual_prediction_tracks_constant_price_shift(self):
        model = TransformerPredictor(4).eval()
        torch.nn.init.normal_(model.output.weight, std=0.01)
        x = torch.arange(12, dtype=torch.float32).reshape(3, 4, 1)
        with torch.inference_mode():
            torch.testing.assert_close(model(x + 10), model(x) + 10)


if __name__ == "__main__":
    unittest.main()
