"""Verify training-only fitting, candidate alignment, and preservation of v0 artifacts."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.data.kepler_prepare_aux_v1 import (
    FEATURES, EXCLUDED, FEATURE_NAMES, align_metadata, clean_features,
    fit_auxiliary, prepare, validate_inputs,
)


class AuxiliaryPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="kepler_aux_test_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = pd.DataFrame({
            "candidate_id": [f"K{i:05d}.01" for i in range(6)],
            # Two different candidates around one training star exercise candidate-level joins.
            "kepid": [10, 10, 20, 21, 30, 31], "y": [0, 1, 0, 1, 0, 1],
            "koi_disposition": ["FALSE POSITIVE", "CONFIRMED"] * 3,
        })
        self.metadata = self.manifest.rename(columns={"candidate_id": "kepoi_name"}).drop(columns="y")
        for i, spec in enumerate((*FEATURES, *EXCLUDED)):
            self.metadata[spec[0]] = np.arange(1, 7, dtype=float) + i
        self.arrays = {
            "X": np.arange(18, dtype=np.float32).reshape(6, 3),
            "X_global": np.ones((6, 8, 3), dtype=np.float32),
            "X_local": np.ones((6, 4, 3), dtype=np.float32),
            "X_scalar": np.ones((6, 2), dtype=np.float32), "y": self.manifest.y.to_numpy(),
            "feature_names": np.array(["a", "b", "c"], dtype=object),
            "scalar_feature_names": np.array(["a", "b"], dtype=object),
            "preprocessing_version": np.array("v0"),
        }
        self.splits = {"train_idx": np.array([0, 1]), "val_idx": np.array([2, 3]), "test_idx": np.array([4, 5])}

    def test_fitting_ignores_held_out_values(self):
        raw = np.tile(np.arange(1, 7, dtype=float)[:, None], (1, len(FEATURES)))
        raw[0, 0], raw[1, 0] = 2.0, 4.0
        raw[2, 0] = np.nan
        raw[0, 1] = np.nan
        first, mask, fitted = fit_auxiliary(raw, self.splits["train_idx"])
        self.assertEqual(fitted["medians"][0], 3.0)
        self.assertEqual(fitted["means"][0], 3.0)
        self.assertEqual(fitted["scales"][0], 1.0)
        np.testing.assert_array_equal(first[:3, 0], [-1.0, 1.0, 0.0])
        self.assertTrue(mask[2, 0])
        # Perturb every validation/test feature. Learned statistics and training outputs must not move.
        altered = raw.copy()
        altered[2:] = 1e12
        changed, _, refitted = fit_auxiliary(altered, self.splits["train_idx"])
        self.assertEqual(fitted, refitted)
        np.testing.assert_array_equal(first[:2], changed[:2])
        empty = raw.copy()
        empty[:2, 3] = np.nan
        with self.assertRaisesRegex(ValueError, "No observed training values"):
            fit_auxiliary(empty, self.splits["train_idx"])

    def test_candidate_join_and_bad_identity_rejection(self):
        shuffled = self.metadata.sample(frac=1, random_state=5)
        aligned = align_metadata(self.manifest, shuffled)
        np.testing.assert_array_equal(aligned.kepoi_name, self.manifest.candidate_id)
        np.testing.assert_array_equal(aligned.koi_steff, self.metadata.koi_steff)
        duplicate = pd.concat([self.metadata, self.metadata.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "unique"):
            align_metadata(self.manifest, duplicate)
        with self.assertRaisesRegex(ValueError, "missing 1"):
            align_metadata(self.manifest, self.metadata.iloc[1:])
        wrong = self.metadata.copy()
        wrong.loc[0, "kepid"] = 999
        with self.assertRaisesRegex(ValueError, "star IDs"):
            align_metadata(self.manifest, wrong)
        wrong = self.metadata.copy()
        wrong.loc[0, "koi_disposition"] = "CONFIRMED"
        with self.assertRaisesRegex(ValueError, "dispositions"):
            align_metadata(self.manifest, wrong)

    def test_domain_rules_and_split_leakage_rejection(self):
        metadata = self.metadata.copy()
        metadata.loc[2, "koi_smass"] = 0.0
        metadata.loc[3, "koi_fwm_srao"] = -3.0
        metadata.loc[4, "koi_dicco_msky"] = 0.0
        raw, audit = clean_features(align_metadata(self.manifest, metadata), self.splits)
        self.assertTrue(np.isnan(raw[2, FEATURE_NAMES.index("koi_smass")]))
        self.assertEqual(raw[3, FEATURE_NAMES.index("koi_fwm_srao")], -3.0)
        self.assertEqual(raw[4, FEATURE_NAMES.index("koi_dicco_msky")], 0.0)
        self.assertEqual(len(audit), 13 * 4)
        leaked = self.manifest.copy()
        leaked.loc[2, "kepid"] = 10
        with self.assertRaisesRegex(ValueError, "overlapping stars"):
            validate_inputs(self.arrays, leaked, self.splits)
        bad_splits = copy.deepcopy(self.splits)
        bad_splits["test_idx"][0] = 0
        with self.assertRaisesRegex(ValueError, "exactly once"):
            validate_inputs(self.arrays, self.manifest, bad_splits)

    def test_build_preserves_v0_and_audit_only_writes_no_dataset(self):
        dataset = self.root / "dataset_kepler_v0_fixture.npz"
        splits = self.root / "splits_kepler_v0_seed42.npz"
        manifest = self.root / "manifest_kepler_v0.csv"
        metadata = self.root / "metadata.csv"
        np.savez_compressed(dataset, **self.arrays)
        np.savez_compressed(splits, **self.splits)
        self.manifest.to_csv(manifest, index=False)
        self.metadata.sample(frac=1, random_state=5).to_csv(metadata, index=False)
        sources = {p: p.read_bytes() for p in (dataset, splits, manifest, metadata)}
        output = self.root / "v1"
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(dataset, splits, manifest, metadata, output, self.root / "audit_only", audit_only=True)
        self.assertFalse(output.exists())
        with contextlib.redirect_stdout(io.StringIO()):
            result = prepare(dataset, splits, manifest, metadata, output, self.root / "audit")
        self.assertEqual(result["status"], "complete")
        for p, original in sources.items():
            self.assertEqual(p.read_bytes(), original)
        self.assertEqual((output / "manifest_kepler_v1.csv").read_bytes(), sources[manifest])
        self.assertEqual((output / "splits_kepler_v1_seed42.npz").read_bytes(), sources[splits])
        with np.load(output / "dataset_kepler_v1_fixture.npz", allow_pickle=True) as built:
            for key, original in self.arrays.items():
                if key != "preprocessing_version":
                    np.testing.assert_array_equal(built[key], original)
                    self.assertEqual(built[key].dtype, original.dtype)
            self.assertEqual(built["preprocessing_version"].item(), "v1")
            self.assertEqual(built["view_preprocessing_version"].item(), "v0")
            self.assertEqual(built["X_aux"].shape, (6, 11))
            self.assertTrue(np.isfinite(built["X_aux"]).all())
            np.testing.assert_array_equal(built["aux_candidate_ids"], self.manifest.candidate_id)
            stored = json.loads(built["aux_preprocessing_json"].item())
            self.assertEqual(stored, json.loads((output / "aux_preprocessing_v1.json").read_text()))
        with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
            prepare(dataset, splits, manifest, metadata, output, self.root / "unused_audit")


if __name__ == "__main__":
    unittest.main()
