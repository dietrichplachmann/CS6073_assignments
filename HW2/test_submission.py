"""Inference checks using deterministic weights and known ordered predictions."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models import DNN
from submission import generate_submission


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        model_dir = self.root / "DNN"
        model_dir.mkdir()
        model = DNN()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            # Normalized pixel0 > 0.5 selects 7, otherwise 3.
            model.network[1].weight[0, 0] = 1
            model.network[4].weight[0, 0] = 1
            model.network[7].weight[7, 0] = 1
            model.network[7].bias[3] = 0.5
        torch.save({"model_name": "DNN", "state_dict": model.state_dict(),
                    "epoch": 1, "learning_rate": 0.001,
                    "validation_metrics": {"accuracy": 1.0}}, model_dir / "best_weights.pt")
        (self.root / "best_model.json").write_text(json.dumps({
            "model": "DNN", "weights": "DNN/best_weights.pt",
        }))
        self.frame = pd.DataFrame(np.zeros((5, 784), dtype=np.uint8),
                                  columns=[f"pixel{i}" for i in range(784)])
        self.frame["pixel0"] = [0, 255, 64, 200, 0]
        self.test_csv = self.root / "test.csv"
        self.frame.to_csv(self.test_csv, index=False)
        self.template = self.root / "sample_submission.csv"
        pd.DataFrame({"ImageId": range(1, 6), "Label": 0}).to_csv(self.template, index=False)

    def generate(self, output="submission.csv", batch_size=3):
        with contextlib.redirect_stdout(io.StringIO()):
            return generate_submission(run_dir=self.root, test_csv=self.test_csv,
                                       template_csv=self.template,
                                       output_csv=self.root / output,
                                       batch_size=batch_size, device="cpu")

    def test_preprocessing_order_batching_and_checkpoint_restore(self):
        first = pd.read_csv(self.generate())
        second = pd.read_csv(self.generate("second.csv", batch_size=1))
        self.assertEqual(list(first.columns), ["ImageId", "Label"])
        self.assertEqual(first.ImageId.tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(first.Label.tolist(), [3, 7, 3, 7, 3])
        pd.testing.assert_frame_equal(first, second)

    def test_missing_pixels_do_not_create_submission(self):
        self.frame = self.frame.astype(float)
        self.frame.loc[0, "pixel0"] = np.nan
        self.frame.to_csv(self.test_csv, index=False)
        with self.assertRaises(ValueError):
            self.generate()
        self.assertFalse((self.root / "submission.csv").exists())

    def test_template_order_and_existing_output_are_protected(self):
        self.generate()
        with self.assertRaises(FileExistsError):
            self.generate()
        pd.DataFrame({"ImageId": [5, 4, 3, 2, 1], "Label": 0}).to_csv(self.template, index=False)
        with self.assertRaises(ValueError):
            self.generate("bad.csv")
        self.assertFalse((self.root / "bad.csv").exists())


if __name__ == "__main__":
    unittest.main()
