#!/usr/bin/env python3
"""Add audited catalog features to a separate v1 dataset using the saved v0 rows/splits."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFINITIONS_URL = "https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html"
# Transform choices are fixed before training, not selected using validation/test labels.
# Names and units follow the NASA cumulative KOI column definitions.
FEATURES = (
    ("koi_steff", "stellar", "K", "identity", "positive"),
    ("koi_slogg", "stellar", "log10(cm/s^2)", "identity", "finite"),
    ("koi_smet", "stellar", "dex", "identity", "finite"),
    ("koi_srad", "stellar", "solar radii", "log10", "positive"),
    ("koi_smass", "stellar", "solar masses", "log10", "positive"),
    ("koi_teq", "irradiation", "K", "log10", "positive"),
    ("koi_insol", "irradiation", "Earth flux", "log10", "positive"),
    ("koi_fwm_srao", "centroid", "seconds of RA time", "asinh", "finite"),
    ("koi_fwm_sdeco", "centroid", "arcsec", "asinh", "finite"),
    ("koi_dicco_msky", "centroid", "arcsec", "log1p", "nonnegative"),
    ("koi_dikco_msky", "centroid", "arcsec", "log1p", "nonnegative"),
)
EXCLUDED = (
    ("koi_fwm_sra", "sky_position", "hours of RA", "excluded", "finite"),
    ("koi_fwm_sdec", "sky_position", "degrees", "excluded", "finite"),
)
FEATURE_NAMES = [spec[0] for spec in FEATURES]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def validate_inputs(arrays: dict[str, np.ndarray], manifest: pd.DataFrame,
                    splits: dict[str, np.ndarray]) -> None:
    n = len(manifest)
    if (n == 0 or manifest.candidate_id.isna().any() or not manifest.candidate_id.is_unique
            or manifest.candidate_id.astype(str).str.strip().eq("").any()):
        raise ValueError("The v0 manifest must contain unique, nonempty candidate IDs")
    if manifest.kepid.isna().any() or not np.isin(manifest.y, [0, 1]).all():
        raise ValueError("The v0 manifest must contain star IDs and binary labels")
    if str(arrays["preprocessing_version"].item()) != "v0":
        raise ValueError("This preparation script requires a v0 source dataset")
    for key in ("X", "X_global", "X_local", "X_scalar", "y"):
        if len(arrays[key]) != n or not np.isfinite(arrays[key]).all():
            raise ValueError(f"v0 {key}: row count mismatch or nonfinite values")
    if not np.array_equal(arrays["y"], manifest.y.to_numpy()):
        raise ValueError("Dataset labels do not match the saved manifest order")
    all_indices = []
    star_sets = []
    for split in ("train", "val", "test"):
        indices = splits[f"{split}_idx"]
        if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer) or len(indices) == 0
                or (indices < 0).any() or (indices >= n).any()):
            raise ValueError(f"Invalid {split} indices")
        all_indices.append(indices)
        star_sets.append(set(manifest.iloc[indices].kepid))
    if not np.array_equal(np.sort(np.concatenate(all_indices)), np.arange(n)):
        raise ValueError("Saved splits must cover every candidate exactly once")
    if any(star_sets[i] & star_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Saved splits contain overlapping stars")


def align_metadata(manifest: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Join by candidate, since one star can have several different KOIs."""
    required = {"kepoi_name", "kepid", "koi_disposition", *FEATURE_NAMES}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata columns missing: {sorted(missing)}")
    names = metadata.kepoi_name.astype("string").str.strip()
    if names.isna().any() or names.eq("").any() or names.duplicated().any():
        raise ValueError("Metadata must contain unique, nonempty KOI names")
    indexed = metadata.assign(kepoi_name=names).set_index("kepoi_name", verify_integrity=True)
    missing_ids = set(manifest.candidate_id) - set(indexed.index)
    if missing_ids:
        raise ValueError(f"Metadata is missing {len(missing_ids)} saved candidates")
    aligned = indexed.loc[manifest.candidate_id].reset_index()
    if not np.array_equal(aligned.kepid.to_numpy(), manifest.kepid.to_numpy()):
        raise ValueError("Matched metadata star IDs disagree with the v0 manifest")
    if not np.array_equal(aligned.koi_disposition.to_numpy(), manifest.koi_disposition.to_numpy()):
        raise ValueError("Matched metadata dispositions disagree with the v0 manifest; preserve the original catalog snapshot")
    return aligned


def clean_features(aligned: pd.DataFrame, splits: dict[str, np.ndarray]) -> tuple[np.ndarray, pd.DataFrame]:
    columns = []
    audit_rows = []
    for name, group, unit, transform, domain in (*FEATURES, *EXCLUDED):
        original = aligned[name] if name in aligned else pd.Series(np.nan, index=aligned.index)
        numeric = pd.to_numeric(original, errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(numeric)
        invalid_domain = finite & ((numeric <= 0) if domain == "positive" else
                                   (numeric < 0) if domain == "nonnegative" else False)
        clean = numeric.copy()
        clean[~finite | invalid_domain] = np.nan
        included = name in FEATURE_NAMES
        if included:
            columns.append(clean)
        for split, indices in (("all", np.arange(len(aligned))),
                               *((s, splits[f"{s}_idx"]) for s in ("train", "val", "test"))):
            observed = np.isfinite(clean[indices])
            row = {"feature": name, "group": group, "unit": unit, "transform": transform,
                   "included_in_X_aux": included, "column_present": name in aligned, "split": split,
                   "rows": len(indices), "missing_after_cleaning": int((~observed).sum()),
                   "missing_fraction": float((~observed).mean()),
                   "nonfinite_or_nonnumeric": int((~finite[indices]).sum()),
                   "invalid_domain": int(invalid_domain[indices].sum())}
            # Distribution summaries are training-only; held-out coverage is descriptive, never fitted.
            if split == "train" and observed.any():
                values = clean[indices][observed]
                row.update(train_min=float(values.min()), train_median=float(np.median(values)),
                           train_max=float(values.max()), train_unique=int(len(np.unique(values))))
            audit_rows.append(row)
    return np.column_stack(columns), pd.DataFrame(audit_rows)


def transform_features(raw: np.ndarray) -> np.ndarray:
    result = raw.astype(np.float64, copy=True)
    for column, (_, _, _, transform, _) in enumerate(FEATURES):
        if transform == "log10":
            result[:, column] = np.log10(result[:, column])
        elif transform == "log1p":
            result[:, column] = np.log1p(result[:, column])
        elif transform == "asinh":
            result[:, column] = np.arcsinh(result[:, column])
    return result


def fit_auxiliary(raw: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit medians/means/scales exclusively on training rows after fixed transforms."""
    transformed = transform_features(raw)
    missing = ~np.isfinite(transformed)
    train = transformed[train_idx]
    if (~np.isfinite(train)).all(axis=0).any():
        names = np.array(FEATURE_NAMES)[(~np.isfinite(train)).all(axis=0)].tolist()
        raise ValueError(f"No observed training values for features: {names}")
    medians = np.nanmedian(train, axis=0)
    filled = np.where(missing, medians, transformed)
    means = filled[train_idx].mean(axis=0)
    std = filled[train_idx].std(axis=0, ddof=0)
    scales = np.where(std < 1e-12, 1.0, std)
    scaled = ((filled - means) / scales).astype(np.float32)
    if not np.isfinite(scaled).all():
        raise ValueError("Auxiliary features are not finite after preprocessing")
    fitted = {
        "fit_split": "train", "fit_rows": len(train_idx), "feature_names": FEATURE_NAMES,
        "feature_groups": [spec[1] for spec in FEATURES], "units": [spec[2] for spec in FEATURES],
        "transforms": [spec[3] for spec in FEATURES], "valid_domains": [spec[4] for spec in FEATURES],
        "imputation": "training median in transformed space",
        "medians": medians.tolist(), "means": means.tolist(), "scales": scales.tolist(),
        "standard_deviation_ddof": 0, "constant_training_features": [FEATURE_NAMES[i] for i in np.flatnonzero(std < 1e-12)],
        "X_aux_is_standardized": True, "missing_mask_is_separate": True,
        "transform_notes": "log10 compresses positive scales; log1p permits zero offsets; asinh preserves the sign of shifts. "
                           "Transforms operate on the numeric values in the listed native units; no data-dependent clipping.",
    }
    return scaled, missing, fitted


def prepare(dataset_path: Path, splits_path: Path, manifest_path: Path, metadata_path: Path,
            output_dir: Path, audit_dir: Path, audit_only: bool = False) -> dict[str, Any]:
    paths = {key: path.resolve() for key, path in {
        "dataset": dataset_path, "splits": splits_path, "manifest": manifest_path, "metadata": metadata_path}.items()}
    output_dir, audit_dir = output_dir.resolve(), audit_dir.resolve()
    if not audit_only and output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing v1 output: {output_dir}")
    if output_dir == audit_dir or output_dir in audit_dir.parents or audit_dir in output_dir.parents:
        raise ValueError("Audit and dataset directories must be separate")
    if audit_dir.exists():
        raise FileExistsError(f"Use a new audit directory: {audit_dir}")
    sources = {key: {"path": str(path), "sha256": sha256(path)} for key, path in paths.items()}
    with np.load(paths["dataset"], allow_pickle=True) as source:
        arrays = {key: source[key].copy() for key in source.files}
    with np.load(paths["splits"], allow_pickle=False) as source:
        splits = {key: source[key].copy() for key in source.files}
    manifest = pd.read_csv(paths["manifest"])
    metadata = pd.read_csv(paths["metadata"], comment="#", low_memory=False)
    validate_inputs(arrays, manifest, splits)
    aligned = align_metadata(manifest, metadata)
    raw, feature_audit = clean_features(aligned, splits)
    audit_dir.mkdir(parents=True)
    feature_audit.to_csv(audit_dir / "feature_audit.csv", index=False)
    raw_table = aligned[["kepoi_name", "kepid", *[x[0] for x in (*FEATURES, *EXCLUDED) if x[0] in aligned]]].copy()
    raw_table = raw_table.rename(columns={"kepoi_name": "candidate_id"})
    raw_table.to_csv(audit_dir / "aligned_catalog_features.csv", index=False)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "audit_complete",
        "source_files": sources, "script_sha256": sha256(Path(__file__)),
        "definitions_url": DEFINITIONS_URL, "metadata_rows": len(metadata), "matched_candidates": len(manifest),
        "split_sizes": {s: len(splits[f"{s}_idx"]) for s in ("train", "val", "test")},
        "selected_features": FEATURE_NAMES, "excluded_features": {x[0]: "Absolute sky position; excluded by modeling choice" for x in EXCLUDED},
        "rows_with_any_aux_missing": int((~np.isfinite(raw)).any(axis=1).sum()),
        "rows_with_all_aux_missing": int((~np.isfinite(raw)).all(axis=1).sum()),
        "candidate_rows_dropped": 0, "labels_and_star_ids_match": True,
        "audit_dir": str(audit_dir), "output_dir": None,
        "interpretation": "Catalog measurements support a metadata-assisted Kepler classification comparison. "
                          "Centroid measurements are vetting diagnostics; this does not establish blind discovery performance. "
                          "No disposition, score, false-positive flag, identifier, or absolute sky position is an X_aux input.",
    }
    write_json(audit_dir / "audit.json", summary)
    # Fail with the audit available if a feature cannot be fitted using training rows.
    aux, missing, fitted = fit_auxiliary(raw, splits["train_idx"])
    fitted.update(source_files=sources, fit_indices_sha256=hashlib.sha256(
        np.asarray(splits["train_idx"], dtype="<i8").tobytes()).hexdigest(), definitions_url=DEFINITIONS_URL)
    write_json(audit_dir / "aux_preprocessing.json", fitted)

    if not audit_only:
        dataset_name = paths["dataset"].name.replace("dataset_kepler_v0_", "dataset_kepler_v1_", 1)
        if dataset_name == paths["dataset"].name:
            raise ValueError("Source dataset filename must begin with dataset_kepler_v0_")
        split_name = paths["splits"].name.replace("splits_kepler_v0_", "splits_kepler_v1_", 1)
        if split_name == paths["splits"].name:
            raise ValueError("Source split filename must begin with splits_kepler_v0_")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".aux_v1_", dir=output_dir.parent) as temporary:
            staging = Path(temporary) / "artifacts"
            staging.mkdir()
            arrays.update(preprocessing_version=np.array("v1"), view_preprocessing_version=np.array("v0"),
                          X_aux=aux, X_aux_missing=missing, aux_feature_names=np.array(FEATURE_NAMES),
                          aux_feature_groups=np.array([x[1] for x in FEATURES]),
                          aux_candidate_ids=manifest.candidate_id.to_numpy(dtype=str),
                          aux_preprocessing_json=np.array(json.dumps(fitted, allow_nan=False)))
            np.savez_compressed(staging / dataset_name, **arrays)
            shutil.copy2(paths["manifest"], staging / "manifest_kepler_v1.csv")
            shutil.copy2(paths["splits"], staging / split_name)
            write_json(staging / "aux_preprocessing_v1.json", fitted)
            for key, source in sources.items():
                if sha256(paths[key]) != source["sha256"]:
                    raise RuntimeError(f"Input changed during preparation: {paths[key]}")
            if output_dir.exists():
                raise FileExistsError(f"Output appeared during preparation: {output_dir}")
            staging.rename(output_dir)
        summary.update(status="complete", output_dir=str(output_dir), X_aux_shape=list(aux.shape),
                       output_files={name: {"path": str(output_dir / name), "sha256": sha256(output_dir / name)}
                                     for name in (dataset_name, "manifest_kepler_v1.csv", split_name, "aux_preprocessing_v1.json")},
                       completed_at=datetime.now(timezone.utc).isoformat())
    summary["sources_unchanged"] = all(sha256(paths[key]) == source["sha256"] for key, source in sources.items())
    if not summary["sources_unchanged"]:
        raise RuntimeError("Source changed during audit")
    write_json(audit_dir / "audit.json", summary)
    lines = ["# Kepler auxiliary-feature audit", "",
             f"Status: {summary['status']}. Matched {len(manifest):,} of {len(manifest):,} saved candidates; no rows dropped.", "",
             "The main X_aux has 11 columns: five stellar parameters, two irradiation estimates, and four centroid measurements. "
             "Absolute sky coordinates koi_fwm_sra and koi_fwm_sdec are audited but excluded by modeling choice.", "",
             f"Column meanings and units: [NASA KOI definitions]({DEFINITIONS_URL}). RA shift uses seconds of time; Dec shift uses arcseconds. "
             "koi_teq is a planet equilibrium-temperature estimate, not stellar temperature.", "",
             "| Feature | Group | Unit | Transform | Missing in training | Missing overall |", "| --- | --- | --- | --- | --- | --- |"]
    for name, group, unit, transform, _ in FEATURES:
        rows = feature_audit[feature_audit.feature == name].set_index("split")
        lines.append(f"| {name} | {group} | {unit} | {transform} | {int(rows.loc['train', 'missing_after_cleaning'])} | "
                     f"{int(rows.loc['all', 'missing_after_cleaning'])} |")
    lines.extend(["", f"Rows with any selected value missing after cleaning: {summary['rows_with_any_aux_missing']}. "
                  f"Rows with all selected values missing: {summary['rows_with_all_aux_missing']}. They retain their original light-curve inputs.", "",
                  "Nonfinite/nonnumeric values and violations of the recorded physical domain become missing. "
                  "Positive mass/radius/temperature/flux values must exceed zero; sky-offset magnitudes may be zero. "
                  "No clipping or label-dependent feature selection is performed.", "",
                  "Fixed transforms compress large ranges; imputation medians, means, and population standard deviations are fitted only on training rows. "
                  "X_aux is already standardized. X_aux_missing is a separate boolean mask, not appended to model inputs. "
                  "The fitted transform and training-index/source hashes are saved in JSON and embedded in the NPZ.", "",
                  "The source v0 arrays are copied exactly except for the version tag; the views retain version v0. "
                  "The manifest and split files are copied byte-for-byte under v1 names. Original data and cache files are unchanged.", "",
                  summary["interpretation"], "",
                  "Step 5 prepares inputs only. The existing CNN does not consume X_aux yet; the auxiliary branch is step 6. "
                  "A matched comparison, with all training seeds retained, is step 7.", ""])
    (audit_dir / "AUDIT_REPORT.md").write_text("\n".join(lines))
    print(f"{summary['status']}: {len(manifest)} matched candidates, {aux.shape[1]} auxiliary features")
    print(f"Audit: {audit_dir}")
    if not audit_only:
        print(f"v1 dataset: {output_dir / dataset_name}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="Explicit v0 dataset NPZ")
    parser.add_argument("--splits", required=True, type=Path, help="Existing saved v0 split NPZ")
    parser.add_argument("--manifest", required=True, type=Path, help="Matching v0 manifest CSV")
    parser.add_argument("--metadata", type=Path, default=ROOT / "data/raw/kepler/metadata/koi_cumulative.csv")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "data/processed/kepler/v1")
    parser.add_argument("--audit_dir", type=Path, default=ROOT / "outputs/metadata_audit" /
                        datetime.now(timezone.utc).strftime("kepler_aux_v1_%Y%m%d_%H%M%S_utc"))
    parser.add_argument("--audit_only", action="store_true", help="Audit and fit training-only transforms without writing a v1 dataset")
    args = parser.parse_args()
    prepare(args.dataset, args.splits, args.manifest, args.metadata, args.output_dir, args.audit_dir, args.audit_only)


if __name__ == "__main__":
    main()
