"""Exercise auxiliary training, provenance checks, and the shared v0/transformer interfaces."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import torch

from scripts.data.kepler_prepare_aux_v1 import FEATURES, EXCLUDED, prepare
from training.cnn import train_kepler_cnn as cnn


class AuxiliaryCNNTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cnn_aux_test_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        rng = np.random.default_rng(7)
        n = 24
        global_view = rng.normal(size=(n, 32, 3)).astype(np.float32)
        local_view = rng.normal(size=(n, 16, 3)).astype(np.float32)
        global_view[:, :, 2] = local_view[:, :, 2] = 1
        scalar = rng.normal(size=(n, 7)).astype(np.float32)
        labels = np.arange(n) % 2
        self.v0 = self.root / "dataset_kepler_v0_fixture.npz"
        self.splits = self.root / "splits_kepler_v0_seed42.npz"
        self.manifest = self.root / "manifest_kepler_v0.csv"
        np.savez(self.v0, X_global=global_view, X_local=local_view, X_scalar=scalar, y=labels,
                 X=np.concatenate([global_view.reshape(n, -1), local_view.reshape(n, -1), scalar], axis=1),
                 preprocessing_version=np.array("v0"))
        np.savez(self.splits, train_idx=np.arange(12), val_idx=np.arange(12, 18), test_idx=np.arange(18, 24))
        manifest = pd.DataFrame({"candidate_id": [f"K{i:05d}.01" for i in range(n)],
                                 "kepid": np.arange(n), "y": labels,
                                 "koi_disposition": np.where(labels, "CONFIRMED", "FALSE POSITIVE")})
        manifest.to_csv(self.manifest, index=False)
        metadata = manifest.rename(columns={"candidate_id": "kepoi_name"}).drop(columns="y")
        for name, *_ in (*FEATURES, *EXCLUDED):
            metadata[name] = rng.uniform(1, 10, n)
        metadata_path = self.root / "metadata.csv"
        metadata.to_csv(metadata_path, index=False)
        with redirect_stdout(io.StringIO()):
            prepare(self.v0, self.splits, self.manifest, metadata_path, self.root / "v1", self.root / "audit")
        self.v1 = self.root / "v1/dataset_kepler_v1_fixture.npz"
        self.cfg = cnn.TrainConfig(base_dir=str(self.root), dataset_path=str(self.v1),
                                   splits_path=str(self.root / "v1/splits_kepler_v1_seed42.npz"),
                                   manifest_path=str(self.root / "v1/manifest_kepler_v1.csv"),
                                   output_root=str(self.root / "training"), prepare_version="v1",
                                   use_aux_branch=True, device="cpu", num_workers=0,
                                   batch_size=6, epochs=2, threshold_grid_size=5)

    def dataset(self, arrays, split):
        return cnn.KeplerTensorDataset(arrays["x_global"], arrays["x_local"], arrays["x_scalar"], arrays["x_flat"],
                                       arrays["y"], arrays[split + "_idx"], x_aux=arrays.get("x_aux"))

    def test_full_aux_training_logs_and_reload(self):
        tracker = MagicMock()
        tracker.__enter__.return_value = tracker
        tracker.__exit__.return_value = False
        tracker.config = {}
        tracker.summary = {}
        wandb = SimpleNamespace(init=Mock(return_value=tracker))
        original_import = __import__

        def import_module(name, *args, **kwargs):
            return wandb if name == "wandb" else original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_module), redirect_stdout(io.StringIO()):
            run_dir = cnn.train_model(replace(self.cfg, wandb_mode="offline"))
        checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu")
        self.assertEqual(checkpoint["model_config"]["aux_features"], 11)
        self.assertTrue(checkpoint["model_config"]["use_aux_branch"])
        self.assertEqual(checkpoint["aux_preprocessing"], json.loads((run_dir / "aux_preprocessing.json").read_text()))
        self.assertEqual(checkpoint["model_config"], json.loads((run_dir / "model_config.json").read_text()))
        self.assertEqual(tracker.config["aux_feature_count"], 11)
        self.assertEqual(tracker.config["aux_feature_names"], checkpoint["aux_preprocessing"]["feature_names"])
        cnn.set_seed(self.cfg.seed)
        initial = cnn.ExoMinerStyleCNN(**checkpoint["model_config"])
        self.assertFalse(torch.equal(initial.aux_branch.net[0].weight, checkpoint["model_state"]["aux_branch.net.0.weight"]))
        initial.load_state_dict(checkpoint["model_state"], strict=True)
        arrays, _ = cnn.load_training_data(self.cfg)
        data = self.dataset(arrays, "test")
        self.assertEqual(len(data[0]), 7)
        loader = cnn.make_dataloader(data, 6, False, 0, torch.device("cpu"))
        result = cnn.evaluate_split(initial, loader, torch.nn.BCEWithLogitsLoss(), torch.device("cpu"), checkpoint["val_threshold"])
        predictions = pd.read_csv(run_dir / "test_predictions.csv")
        np.testing.assert_allclose(result["probs"], predictions.probability, rtol=1e-6)
        summary = json.loads((run_dir / "summary.json").read_text())
        self.assertTrue(summary["use_aux_branch"])
        self.assertEqual(summary["aux_feature_count"], 11)
        for metric in ("accuracy", "f1", "pr_auc", "roc_auc"):
            self.assertAlmostEqual(result["metrics"][metric], summary["test_metrics"][metric])
            self.assertEqual(tracker.summary["test/" + metric], summary["test_metrics"][metric])

    def test_branch_affects_predictions_and_receives_gradients(self):
        arrays, _ = cnn.load_training_data(self.cfg)
        batch = next(iter(cnn.make_dataloader(self.dataset(arrays, "train"), 6, False, 0, torch.device("cpu"))))
        for wide in (False, True):
            cnn.set_seed(42)
            model = cnn.ExoMinerStyleCNN(7, arrays["x_flat"].shape[1], 0.2, 0.1, 0.15, wide,
                                         aux_features=11, use_aux_branch=True)
            model.eval()
            logits = model(*batch[:4], batch[6])
            changed = model(*batch[:4], batch[6] + torch.arange(11).float())
            self.assertFalse(torch.allclose(logits, changed))
            torch.nn.functional.binary_cross_entropy_with_logits(logits, batch[4]).backward()
            gradient = model.aux_branch.net[0].weight.grad
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0)
            with self.assertRaisesRegex(ValueError, "X_aux is required"):
                model(*batch[:4])

    def test_aux_loader_preserves_values_and_rejects_mismatches(self):
        arrays, _ = cnn.load_training_data(self.cfg)
        with np.load(self.v1, allow_pickle=True) as source:
            original = {key: source[key].copy() for key in source.files}
        np.testing.assert_array_equal(arrays["x_aux"], original["X_aux"])
        for case in ("missing", "nonfinite", "candidate_order", "feature_order"):
            bad = {key: value.copy() for key, value in original.items()}
            if case == "missing":
                bad.pop("X_aux")
            elif case == "nonfinite":
                bad["X_aux"][0, 0] = np.nan
            elif case == "candidate_order":
                bad["aux_candidate_ids"] = bad["aux_candidate_ids"][::-1]
            else:
                bad["aux_feature_names"] = bad["aux_feature_names"][::-1]
            path = self.root / (case + ".npz")
            np.savez(path, **bad)
            with self.subTest(case=case), self.assertRaises(ValueError):
                cnn.load_training_data(replace(self.cfg, dataset_path=str(path)))
        for change_train in (True, False):
            with np.load(self.splits) as source:
                splits = {key: source[key].copy() for key in source.files}
            if change_train:
                splits["train_idx"] = splits["train_idx"][::-1]
            else:
                splits["val_idx"], splits["test_idx"] = splits["test_idx"], splits["val_idx"]
            path = self.root / ("bad_splits_" + str(change_train) + ".npz")
            np.savez(path, **splits)
            with self.subTest(change_train=change_train), self.assertRaisesRegex(ValueError, "indices|source hash"):
                cnn.load_training_data(replace(self.cfg, splits_path=str(path)))

    def test_v0_and_transformer_interfaces_remain_compatible(self):
        from training.transformer import train_kepler_transformer as transformer
        v0_cfg = replace(self.cfg, dataset_path=str(self.v0), splits_path=str(self.splits),
                         manifest_path=str(self.manifest), use_aux_branch=False, prepare_version="v0")
        baseline, _ = cnn.load_training_data(v0_cfg)
        ignored_aux, _ = cnn.load_training_data(replace(self.cfg, use_aux_branch=False))
        self.assertNotIn("x_aux", baseline)
        for key in baseline:
            np.testing.assert_array_equal(baseline[key], ignored_aux[key])
        data = self.dataset(baseline, "val")
        self.assertEqual(len(data[0]), 6)
        cnn.set_seed(5)
        default = cnn.ExoMinerStyleCNN(7, baseline["x_flat"].shape[1], 0.2, 0.1, 0.15, False)
        cnn.set_seed(5)
        disabled = cnn.ExoMinerStyleCNN(7, baseline["x_flat"].shape[1], 0.2, 0.1, 0.15, False,
                                       aux_features=11, use_aux_branch=False, aux_dropout=0.9)
        for key, tensor in default.state_dict().items():
            self.assertTrue(torch.equal(tensor, disabled.state_dict()[key]))
        self.assertFalse(any(key.startswith("aux_branch") for key in default.state_dict()))
        loader = cnn.make_dataloader(data, 6, False, 0, torch.device("cpu"))
        batch = next(iter(loader))
        with self.assertRaisesRegex(ValueError, "disabled"):
            default(*batch[:4], torch.zeros(6, 11))
        cfg = transformer.TrainConfig(base_dir=str(self.root), dataset_path=str(self.v0),
                                      splits_path=str(self.splits), manifest_path=str(self.manifest), output_root=str(self.root))
        shared, _ = cnn.load_training_data(cfg)
        self.assertNotIn("x_aux", shared)
        model = transformer.ExoFormer(global_seq_len=32, local_seq_len=16, scalar_features=7,
                                      d_model=16, num_heads=4, global_depth=1, local_depth=1,
                                      mlp_ratio=2.0, dropout=0.0, attention_dropout=0.0,
                                      token_dropout=0.0, scalar_dropout=0.0)
        result = cnn.evaluate_split(model, loader, torch.nn.BCEWithLogitsLoss(), torch.device("cpu"), 0.5)
        self.assertTrue(np.isfinite(result["probs"]).all())

    def test_cli_auxiliary_mode_requires_explicit_artifacts(self):
        with patch("sys.argv", ["train", "--aux_branch"]):
            with self.assertRaisesRegex(ValueError, "explicit"):
                cnn.build_config(cnn.parse_args())
        with patch("sys.argv", ["train", "--aux_branch", "--dataset_path", self.cfg.dataset_path,
                                "--splits_path", self.cfg.splits_path, "--manifest_path", self.cfg.manifest_path,
                                "--aux_dropout", "0.25"]):
            config = cnn.build_config(cnn.parse_args())
            self.assertTrue(config.use_aux_branch)
            self.assertEqual(config.aux_dropout, 0.25)


if __name__ == "__main__":
    unittest.main()
