"""Run with: python -m unittest discover -s tests -v"""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "cnn"))
import train_kepler_cnn as cnn


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        self.cfg = cnn.TrainConfig(
            base_dir=str(self.root),
            dataset_path=str(self.root / "dataset.npz"),
            splits_path=str(self.root / "splits.npz"),
            manifest_path=str(self.root / "manifest.csv"),
            output_root=str(self.root / "disabled"),
            device="cpu", num_workers=0, batch_size=6, epochs=2,
            threshold_grid_size=5,
        )

    def make_dataset(self):
        rng = np.random.default_rng(7)
        global_view = rng.normal(size=(24, 32, 3)).astype(np.float32)
        local_view = rng.normal(size=(24, 16, 3)).astype(np.float32)
        global_view[:, :, 2] = local_view[:, :, 2] = 1
        scalar = rng.normal(size=(24, 7)).astype(np.float32)
        labels = np.arange(24) % 2
        np.savez(self.cfg.dataset_path, X_global=global_view, X_local=local_view,
                 X_scalar=scalar, y=labels,
                 X=np.concatenate([global_view.reshape(24, -1), local_view.reshape(24, -1), scalar], axis=1))
        np.savez(self.cfg.splits_path, train_idx=np.arange(12), val_idx=np.arange(12, 18), test_idx=np.arange(18, 24))
        pd.DataFrame({"candidate_id": [f"test-{i}" for i in range(24)],
                      "kepid": np.arange(24), "y": labels}).to_csv(self.cfg.manifest_path, index=False)

    def fake_wandb(self):
        run = MagicMock()
        run.__enter__.return_value = run
        run.__exit__.return_value = False
        run.config = {}
        run.summary = {}
        return SimpleNamespace(init=Mock(return_value=run)), run

    def wandb_import(self, replacement):
        original_import = __import__

        def import_module(name, *args, **kwargs):
            if name == "wandb":
                if replacement is None:
                    raise ImportError("Optional wandb package is unavailable")
                return replacement
            return original_import(name, *args, **kwargs)

        # Preserve modules loaded lazily by PyTorch while exercising optional imports.
        return patch("builtins.__import__", side_effect=import_module)

    def test_disabled_training_and_logged_training_agree(self):
        self.make_dataset()
        with self.wandb_import(None), redirect_stdout(io.StringIO()):
            baseline = cnn.train_model(self.cfg)

        wandb, run = self.fake_wandb()
        cfg = replace(self.cfg, output_root=str(self.root / "tracked"), wandb_mode="offline")
        with self.wandb_import(wandb), redirect_stdout(io.StringIO()):
            tracked = cnn.train_model(cfg)

        for name in ("history.csv", "val_predictions.csv", "test_predictions.csv"):
            pd.testing.assert_frame_equal(pd.read_csv(baseline / name), pd.read_csv(tracked / name))
        history = pd.read_csv(tracked / "history.csv")
        summary = json.loads((tracked / "summary.json").read_text())
        self.assertAlmostEqual(history.iloc[0].learning_rate, cfg.learning_rate)
        self.assertAlmostEqual(history.iloc[1].learning_rate, cfg.learning_rate / 2)
        self.assertEqual(run.log.call_count, cfg.epochs + 1)
        for epoch, call in enumerate(run.log.call_args_list[:-1], 1):
            logged = call.args[0]
            self.assertEqual(call.kwargs["step"], epoch)
            self.assertFalse(any(key.startswith("test/") for key in logged))
            for split in ("train", "val"):
                for metric in ("loss", "accuracy", "f1", "pr_auc", "roc_auc", "threshold"):
                    self.assertAlmostEqual(logged[f"{split}/{metric}"], history.iloc[epoch - 1][f"{split}_{metric}"])
        final = run.log.call_args_list[-1].args[0]
        self.assertEqual(final["best_threshold"], summary["best_threshold"])
        self.assertEqual(final["best_epoch"], summary["best_epoch"])
        for metric, value in summary["test_metrics"].items():
            self.assertEqual(final[f"test/{metric}"], value)
        self.assertEqual(run.summary, final)
        self.assertEqual(wandb.init.call_args.kwargs["mode"], "offline")
        run.__exit__.assert_called_once_with(None, None, None)

    def test_requested_tracking_explains_missing_optional_dependency(self):
        with self.wandb_import(None):
            with self.assertRaisesRegex(ImportError, "requirements-wandb.txt"):
                with cnn.tracking_run(replace(self.cfg, wandb_mode="offline"), self.root):
                    pass

    def test_tracking_closes_on_training_failure(self):
        wandb, run = self.fake_wandb()
        cfg = replace(self.cfg, wandb_mode="offline")
        with self.wandb_import(wandb), self.assertRaisesRegex(RuntimeError, "training failed"):
            with cnn.tracking_run(cfg, self.root):
                raise RuntimeError("training failed")
        self.assertIs(run.__exit__.call_args.args[0], RuntimeError)


if __name__ == "__main__":
    unittest.main()
