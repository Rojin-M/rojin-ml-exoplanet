#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import logging
import math
import os
import random
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

try:
    import lightkurve as lk
except ImportError as exc:
    raise ImportError(
        "lightkurve is required for this script. "
        "Install with: pip install lightkurve"
    ) from exc


KOI_API = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
    "query=select+*+from+cumulative&format=csv"
)
PREPROCESS_VERSION = "v0"
_WORKER_CFG: Optional["PrepConfig"] = None
_WORKER_PATHS: Optional["Paths"] = None
_WORKER_LOGGER: Optional[logging.Logger] = None


@dataclass
class PrepConfig:
    base_dir: str
    seed: int = 42
    target_n: int = 10000
    oversample_factor: float = 2.5

    global_bins: int = 200
    local_bins: int = 100
    global_window: float = 0.5
    local_window_mult: float = 4.0
    detrend_window_days: float = 1.5
    sigma_clip_sigma: float = 5.0

    min_period: float = 0.5
    max_period: float = 30.0
    min_duration_hours: float = 0.1
    max_duration_hours: float = 20.0

    use_bls: bool = True
    bls_min_period: float = 0.5
    bls_max_period: float = 30.0
    bls_grid_size: int = 2000

    test_size: float = 0.15
    val_size: float = 0.15

    cadence_preference: str = "long"
    force_redownload_metadata: bool = False
    force_recompute_candidate_cache: bool = False
    max_retry_per_star: int = 2
    prepare_workers: int = 0


class Paths:
    def __init__(self, base_dir: str) -> None:
        self.base = Path(base_dir)
        self.raw_meta = self.base / "data" / "raw" / "kepler" / "metadata"
        self.raw_lc = self.base / "data" / "raw" / "kepler" / "lightcurves"
        self.interim = self.base / "data" / "interim" / "kepler"
        self.interim_version = self.interim / PREPROCESS_VERSION
        self.interim_folded = self.interim_version / "folded"
        self.interim_bls = self.interim_version / "bls"
        self.interim_views = self.interim_version / "views"
        self.processed = self.base / "data" / "processed" / "kepler" / PREPROCESS_VERSION
        self.logs = self.base / "logs"
        self.outputs = self.base / "outputs"

        for p in [
            self.raw_meta,
            self.raw_lc,
            self.interim,
            self.interim_version,
            self.interim_folded,
            self.interim_bls,
            self.interim_views,
            self.processed,
            self.logs,
            self.outputs,
        ]:
            p.mkdir(parents=True, exist_ok=True)

    @property
    def raw_koi_csv(self) -> Path:
        return self.raw_meta / "koi_cumulative.csv"

    @property
    def binary_koi_csv(self) -> Path:
        return self.raw_meta / "koi_binary.csv"

    @property
    def manifest_csv(self) -> Path:
        return self.processed / f"manifest_kepler_{PREPROCESS_VERSION}.csv"


def setup_logger(log_path: Path, logger_name: str = "kepler_prep") -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


def setup_null_logger(logger_name: str) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.CRITICAL)
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def safe_float(x: Any) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        v = float(x)
        if math.isfinite(v):
            return v
        return None
    except Exception:
        return None


def candidate_id_from_row(row: pd.Series) -> str:
    for col in ["kepoi_name", "kepoi_name", "kepler_name"]:
        if col in row and pd.notna(row[col]):
            return str(row[col]).strip().replace(" ", "_")
    return f"kepid_{int(row['kepid'])}_period_{float(row['koi_period']):.6f}"


def star_cache_path(paths: Paths, kepid: int) -> Path:
    return paths.raw_lc / f"{int(kepid)}.npz"


def candidate_folded_path(paths: Paths, candidate_id: str) -> Path:
    return paths.interim_folded / f"{candidate_id}.npz"


def candidate_bls_path(paths: Paths, candidate_id: str) -> Path:
    return paths.interim_bls / f"{candidate_id}.json"


def candidate_views_path(paths: Paths, candidate_id: str) -> Path:
    return paths.interim_views / f"{candidate_id}.npz"


def dataset_name(cfg: PrepConfig) -> str:
    return (
        f"dataset_kepler_{PREPROCESS_VERSION}_target{cfg.target_n}"
        f"_g{cfg.global_bins}"
        f"_l{cfg.local_bins}"
        f"_gw{cfg.global_window}"
        f"_lwm{cfg.local_window_mult}"
        f"_bls{int(cfg.use_bls)}"
        f"_seed{cfg.seed}.npz"
    )


def splits_name(cfg: PrepConfig) -> str:
    return f"splits_kepler_{PREPROCESS_VERSION}_seed{cfg.seed}.npz"


def download_koi_metadata(cfg: PrepConfig, paths: Paths, logger: logging.Logger) -> pd.DataFrame:
    if paths.raw_koi_csv.exists() and not cfg.force_redownload_metadata:
        logger.info("Using cached KOI metadata: %s", paths.raw_koi_csv)
        return pd.read_csv(paths.raw_koi_csv, low_memory=False)

    logger.info("Downloading KOI cumulative table from NASA Exoplanet Archive...")
    df = pd.read_csv(KOI_API, low_memory=False)
    df.to_csv(paths.raw_koi_csv, index=False)
    logger.info("Saved KOI metadata: %s rows=%d cols=%d", paths.raw_koi_csv, len(df), df.shape[1])
    return df


def prepare_binary_koi_table(
    df: pd.DataFrame,
    cfg: PrepConfig,
    paths: Paths,
    logger: logging.Logger,
) -> pd.DataFrame:
    keep_dispositions = {"CONFIRMED", "FALSE POSITIVE"}
    out = df[df["koi_disposition"].isin(keep_dispositions)].copy()

    required = ["kepid", "koi_disposition", "koi_period", "koi_time0bk", "koi_duration"]
    for col in required:
        if col not in out.columns:
            raise ValueError(f"Required KOI column missing: {col}")

    for col in [
        "kepid",
        "koi_period",
        "koi_time0bk",
        "koi_duration",
        "koi_snr",
        "koi_depth",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["kepid", "koi_period", "koi_time0bk", "koi_duration"]).copy()
    out = out[(out["koi_period"] >= cfg.min_period) & (out["koi_period"] <= cfg.max_period)].copy()
    out = out[
        (out["koi_duration"] >= cfg.min_duration_hours) & (out["koi_duration"] <= cfg.max_duration_hours)
    ].copy()

    out["y"] = (out["koi_disposition"] == "CONFIRMED").astype(np.int64)
    out["candidate_id"] = out.apply(candidate_id_from_row, axis=1)

    keep_cols = [
        "kepid",
        "candidate_id",
        "kepoi_name",
        "kepler_name",
        "koi_disposition",
        "y",
        "koi_period",
        "koi_time0bk",
        "koi_duration",
        "koi_depth",
        "koi_snr",
    ]
    keep_cols = [c for c in keep_cols if c in out.columns]
    out = out[keep_cols].copy()

    out.to_csv(paths.binary_koi_csv, index=False)
    logger.info(
        "Prepared binary KOI table: %s rows=%d pos=%d neg=%d unique_kepid=%d",
        paths.binary_koi_csv,
        len(out),
        int(out["y"].sum()),
        int((out["y"] == 0).sum()),
        out["kepid"].nunique(),
    )
    return out


def validate_star_cache(path: Path) -> bool:
    try:
        if not path.exists():
            return False
        with np.load(path, allow_pickle=False) as data:
            time = data["time"]
            flux = data["flux"]
        return (
            isinstance(time, np.ndarray)
            and isinstance(flux, np.ndarray)
            and len(time) > 100
            and len(flux) == len(time)
            and np.isfinite(time).sum() > 100
            and np.isfinite(flux).sum() > 100
        )
    except Exception:
        return False


def download_kepler_lightcurve_for_star(
    kepid: int,
    cfg: PrepConfig,
    paths: Paths,
    logger: logging.Logger,
) -> Tuple[bool, Optional[str]]:
    cache_path = star_cache_path(paths, kepid)

    if validate_star_cache(cache_path):
        return True, "cache_hit"

    if cache_path.exists():
        logger.warning("Corrupted existing cache detected, removing: %s", cache_path)
        cache_path.unlink(missing_ok=True)

    target_name = f"KIC {int(kepid)}"

    for attempt in range(cfg.max_retry_per_star):
        try:
            sr = lk.search_lightcurve(target_name, author="Kepler")
            if len(sr) == 0:
                return False, "no_search_results"

            if cfg.cadence_preference == "long":
                try:
                    sr = sr[sr.exptime > 1000]
                except Exception:
                    pass
            elif cfg.cadence_preference == "short":
                try:
                    sr = sr[sr.exptime < 1000]
                except Exception:
                    pass

            if len(sr) == 0:
                return False, "no_matching_cadence"

            lc_collection = sr.download_all(download_dir=str(paths.raw_lc))
            if lc_collection is None or len(lc_collection) == 0:
                return False, "download_empty"

            stitched = lc_collection.stitch(corrector_func=lambda x: x.remove_nans())
            time = np.asarray(stitched.time.value, dtype=np.float64)
            flux = np.asarray(stitched.flux.value, dtype=np.float64)

            mask = np.isfinite(time) & np.isfinite(flux)
            time = time[mask]
            flux = flux[mask]

            if len(time) < 200:
                return False, "too_few_points"

            flux_median = np.nanmedian(flux)
            if not np.isfinite(flux_median) or abs(flux_median) < 1e-12:
                return False, "bad_flux_median"

            flux = flux / flux_median

            tmp_path = cache_path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp_path, time=time, flux=flux)
            shutil.move(tmp_path, cache_path)

            if not validate_star_cache(cache_path):
                cache_path.unlink(missing_ok=True)
                return False, "saved_cache_invalid"

            return True, "download_ok"

        except Exception as exc:
            if attempt == cfg.max_retry_per_star - 1:
                logger.warning("Download failed for kepid=%s after retries: %s", kepid, exc)
                return False, f"exception:{type(exc).__name__}"
    return False, "unknown_failure"


def robust_scale_series(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    med = np.nanmedian(x)
    q1 = np.nanpercentile(x, 25)
    q3 = np.nanpercentile(x, 75)
    iqr = max(float(q3 - q1), 1e-6)
    y = (x - med) / iqr
    y = np.clip(y, -10.0, 10.0)
    return y.astype(np.float32)


def sigma_clip_and_detrend(
    time: np.ndarray,
    flux: np.ndarray,
    cfg: PrepConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)

    mask = np.isfinite(time) & np.isfinite(flux)
    time = time[mask]
    flux = flux[mask]
    if len(time) < 200:
        return time, flux

    med = np.nanmedian(flux)
    mad = np.nanmedian(np.abs(flux - med))
    robust_sigma = max(1.4826 * mad, 1e-6)
    clip_mask = np.abs(flux - med) <= (cfg.sigma_clip_sigma * robust_sigma)
    if clip_mask.sum() < 50:
        clip_mask = np.ones_like(flux, dtype=bool)

    clipped_flux = flux.copy()
    if (~clip_mask).any() and clip_mask.sum() >= 2:
        clipped_flux[~clip_mask] = np.interp(time[~clip_mask], time[clip_mask], flux[clip_mask])

    median_dt = np.nanmedian(np.diff(time))
    if not np.isfinite(median_dt) or median_dt <= 0:
        return time, clipped_flux

    window_points = int(round(cfg.detrend_window_days / median_dt))
    window_points = max(window_points, 51)
    if window_points % 2 == 0:
        window_points += 1

    trend = (
        pd.Series(clipped_flux)
        .rolling(window=window_points, center=True, min_periods=max(5, window_points // 10))
        .median()
        .to_numpy(dtype=np.float64)
    )
    if np.isnan(trend).any():
        finite_trend = np.isfinite(trend)
        if finite_trend.sum() >= 2:
            trend[~finite_trend] = np.interp(time[~finite_trend], time[finite_trend], trend[finite_trend])
        elif finite_trend.sum() == 1:
            trend[:] = trend[finite_trend][0]
        else:
            trend[:] = med

    trend = np.clip(trend, 1e-6, None)
    detrended_flux = clipped_flux / trend
    return time, detrended_flux


def fold_phase(time: np.ndarray, flux: np.ndarray, period: float, t0: float) -> Tuple[np.ndarray, np.ndarray]:
    phase = ((time - t0 + 0.5 * period) % period) / period - 0.5
    order = np.argsort(phase)
    return phase[order].astype(np.float32), flux[order].astype(np.float32)


def fill_binned_series(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float32).copy()
    observed = np.asarray(observed, dtype=bool)
    if (~observed).any():
        good_idx = np.where(observed)[0]
        if len(good_idx) > 1:
            out[~observed] = np.interp(np.where(~observed)[0], good_idx, out[good_idx]).astype(np.float32)
        elif len(good_idx) == 1:
            out[:] = out[good_idx[0]]
    return out.astype(np.float32)


def bin_curve_stats(
    phase: np.ndarray,
    flux: np.ndarray,
    n_bins: int,
    low: float,
    high: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(low, high, n_bins + 1, dtype=np.float32)
    mean = np.zeros(n_bins, dtype=np.float32)
    std = np.zeros(n_bins, dtype=np.float32)

    idx = np.digitize(phase, edges) - 1
    valid = (idx >= 0) & (idx < n_bins) & np.isfinite(flux)
    idx = idx[valid]
    vals = flux[valid]

    if len(vals) == 0:
        return mean, std, np.zeros(n_bins, dtype=np.float32)

    sums = np.bincount(idx, weights=vals, minlength=n_bins).astype(np.float32)
    sums_sq = np.bincount(idx, weights=vals * vals, minlength=n_bins).astype(np.float32)
    counts = np.bincount(idx, minlength=n_bins).astype(np.float32)
    observed = counts > 0
    mean[observed] = sums[observed] / counts[observed]

    good_std = counts > 1
    if good_std.any():
        variance = sums_sq[good_std] / counts[good_std] - mean[good_std] ** 2
        std[good_std] = np.sqrt(np.clip(variance, 0.0, None)).astype(np.float32)

    mean = fill_binned_series(mean, observed)
    std = fill_binned_series(std, good_std) if good_std.any() else std
    mask = observed.astype(np.float32)
    return mean, std, mask


def bin_curve(phase: np.ndarray, flux: np.ndarray, n_bins: int, low: float, high: float) -> np.ndarray:
    mean, _, _ = bin_curve_stats(phase, flux, n_bins, low, high)
    return mean


def engineered_features(global_view: np.ndarray, local_view: np.ndarray) -> np.ndarray:
    lv = np.asarray(local_view, dtype=np.float32)

    depth = float(np.nanmedian(lv) - np.nanmin(lv))

    mid = len(lv) // 2
    left = lv[:mid]
    right = lv[-mid:][::-1] if mid > 0 else lv
    symmetry = float(np.mean(np.abs(left - right[: len(left)]))) if len(left) > 0 else 0.0

    odd = lv[::2]
    even = lv[1::2]
    odd_even = float(abs(np.nanmean(odd) - np.nanmean(even))) if len(even) > 0 else 0.0

    return np.array([depth, symmetry, odd_even], dtype=np.float32)


def compute_bls_features(time: np.ndarray, flux: np.ndarray, cfg: PrepConfig) -> Dict[str, float]:
    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)

    mask = np.isfinite(time) & np.isfinite(flux)
    time = time[mask]
    flux = flux[mask]

    if len(time) < 200:
        return {"bls_snr": 0.0, "bls_depth": 0.0, "bls_duration": 0.0, "bls_period": 0.0}

    flux = flux / np.nanmedian(flux)
    dy = np.full_like(flux, np.nanstd(flux), dtype=np.float64)
    dy[~np.isfinite(dy)] = 1e-3
    dy = np.clip(dy, 1e-6, None)

    periods = np.linspace(cfg.bls_min_period, cfg.bls_max_period, cfg.bls_grid_size)
    min_duration_days = 0.05 / 24.0
    # BLS requires durations to stay below the minimum searched period.
    max_duration_days = min(15.0 / 24.0, max(min_duration_days, 0.5 * cfg.bls_min_period))
    durations = np.linspace(min_duration_days, max_duration_days, 20)

    bls = BoxLeastSquares(time, flux, dy=dy)
    res = bls.power(periods, durations)

    if len(res.power) == 0 or not np.isfinite(res.power).any():
        return {"bls_snr": 0.0, "bls_depth": 0.0, "bls_duration": 0.0, "bls_period": 0.0}

    idx = int(np.nanargmax(res.power))
    return {
        "bls_snr": float(res.power[idx]) if np.isfinite(res.power[idx]) else 0.0,
        "bls_depth": float(res.depth[idx]) if np.isfinite(res.depth[idx]) else 0.0,
        "bls_duration": float(res.duration[idx] * 24.0) if np.isfinite(res.duration[idx]) else 0.0,
        "bls_period": float(res.period[idx]) if np.isfinite(res.period[idx]) else 0.0,
    }


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_candidate_cache(
    row: pd.Series,
    cfg: PrepConfig,
    paths: Paths,
    logger: logging.Logger,
) -> Tuple[bool, str]:
    candidate_id = str(row["candidate_id"])
    kepid = int(row["kepid"])

    folded_path = candidate_folded_path(paths, candidate_id)
    views_path = candidate_views_path(paths, candidate_id)
    bls_path = candidate_bls_path(paths, candidate_id)

    if (
        folded_path.exists()
        and views_path.exists()
        and ((not cfg.use_bls) or bls_path.exists())
        and not cfg.force_recompute_candidate_cache
    ):
        return True, "candidate_cache_hit"

    star_path = star_cache_path(paths, kepid)
    if not validate_star_cache(star_path):
        return False, "missing_or_invalid_star_cache"

    try:
        with np.load(star_path, allow_pickle=False) as data:
            time = data["time"].astype(np.float64)
            flux = data["flux"].astype(np.float64)
        time, flux = sigma_clip_and_detrend(time, flux, cfg)

        period = float(row["koi_period"])
        t0 = float(row["koi_time0bk"])
        duration_hours = float(row["koi_duration"])
        local_half_window = min(0.25, cfg.local_window_mult * (duration_hours / 24.0) / period)

        phase, folded_flux = fold_phase(time, flux, period=period, t0=t0)
        folded_flux = robust_scale_series(folded_flux)

        global_mean, global_std, global_mask = bin_curve_stats(
            phase,
            folded_flux,
            cfg.global_bins,
            -cfg.global_window,
            cfg.global_window,
        )
        local_mean, local_std, local_mask = bin_curve_stats(
            phase,
            folded_flux,
            cfg.local_bins,
            -local_half_window,
            local_half_window,
        )
        eng = engineered_features(global_mean, local_mean)
        global_channels = np.stack([global_mean, global_std, global_mask], axis=-1).astype(np.float32)
        local_channels = np.stack([local_mean, local_std, local_mask], axis=-1).astype(np.float32)

        np.savez_compressed(
            folded_path,
            phase=phase.astype(np.float32),
            flux=folded_flux.astype(np.float32),
            period=np.float32(period),
            t0=np.float32(t0),
            duration_hours=np.float32(duration_hours),
            preprocessing_version=np.array(PREPROCESS_VERSION),
        )

        np.savez_compressed(
            views_path,
            global_view=global_mean.astype(np.float32),
            local_view=local_mean.astype(np.float32),
            global_channels=global_channels,
            local_channels=local_channels,
            global_std=global_std.astype(np.float32),
            local_std=local_std.astype(np.float32),
            global_mask=global_mask.astype(np.float32),
            local_mask=local_mask.astype(np.float32),
            engineered=eng.astype(np.float32),
            local_half_window=np.float32(local_half_window),
            preprocessing_version=np.array(PREPROCESS_VERSION),
        )

        if cfg.use_bls:
            bls = compute_bls_features(time, flux, cfg)
            save_json(bls_path, bls)

        return True, "candidate_cache_built"

    except Exception as exc:
        logger.warning("Candidate cache failed for %s: %s", candidate_id, exc)
        return False, f"candidate_exception:{type(exc).__name__}"


def recommend_prepare_workers() -> int:
    cpu_count = os.cpu_count() or 1
    if cpu_count <= 2:
        return 1
    if cpu_count <= 6:
        return max(1, cpu_count - 1)
    return min(8, cpu_count - 2)


def resolve_prepare_workers(cfg: PrepConfig) -> int:
    if cfg.prepare_workers and cfg.prepare_workers > 0:
        return cfg.prepare_workers
    return recommend_prepare_workers()


def init_prepare_worker(cfg: PrepConfig) -> None:
    global _WORKER_CFG, _WORKER_PATHS, _WORKER_LOGGER
    _WORKER_CFG = cfg
    _WORKER_PATHS = Paths(cfg.base_dir)
    _WORKER_LOGGER = setup_null_logger("kepler_prepare_worker")


def build_candidate_cache_worker(record: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    if _WORKER_CFG is None or _WORKER_PATHS is None or _WORKER_LOGGER is None:
        raise RuntimeError("Prepare worker not initialized.")
    row = pd.Series(record)
    ok, reason = build_candidate_cache(row, _WORKER_CFG, _WORKER_PATHS, _WORKER_LOGGER)
    return ok, reason, record


def build_final_dataset_and_manifest(
    candidates: pd.DataFrame,
    cfg: PrepConfig,
    paths: Paths,
    logger: logging.Logger,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
    feature_names = (
        [f"g_mean_{i}" for i in range(cfg.global_bins)]
        + [f"g_std_{i}" for i in range(cfg.global_bins)]
        + [f"g_mask_{i}" for i in range(cfg.global_bins)]
        + [f"l_mean_{i}" for i in range(cfg.local_bins)]
        + [f"l_std_{i}" for i in range(cfg.local_bins)]
        + [f"l_mask_{i}" for i in range(cfg.local_bins)]
        + ["feat_depth", "feat_symmetry", "feat_odd_even"]
        + (["bls_snr", "bls_depth", "bls_duration", "bls_period"] if cfg.use_bls else [])
    )
    scalar_feature_names = ["feat_depth", "feat_symmetry", "feat_odd_even"] + (
        ["bls_snr", "bls_depth", "bls_duration", "bls_period"] if cfg.use_bls else []
    )

    X_list: List[np.ndarray] = []
    X_global_list: List[np.ndarray] = []
    X_local_list: List[np.ndarray] = []
    X_scalar_list: List[np.ndarray] = []
    y_list: List[int] = []
    manifest_rows: List[Dict[str, Any]] = []

    for _, row in tqdm(candidates.iterrows(), total=len(candidates), desc="Assembling dataset"):
        candidate_id = str(row["candidate_id"])
        kepid = int(row["kepid"])

        views_path = candidate_views_path(paths, candidate_id)
        bls_path = candidate_bls_path(paths, candidate_id)

        if not views_path.exists():
            continue
        if cfg.use_bls and not bls_path.exists():
            continue

        try:
            with np.load(views_path, allow_pickle=False) as data:
                if "global_channels" in data and "local_channels" in data:
                    g_channels = data["global_channels"].astype(np.float32)
                    l_channels = data["local_channels"].astype(np.float32)
                else:
                    global_std = (
                        data["global_std"].astype(np.float32)
                        if "global_std" in data
                        else np.zeros(cfg.global_bins, dtype=np.float32)
                    )
                    global_mask = (
                        data["global_mask"].astype(np.float32)
                        if "global_mask" in data
                        else np.ones(cfg.global_bins, dtype=np.float32)
                    )
                    local_std = (
                        data["local_std"].astype(np.float32)
                        if "local_std" in data
                        else np.zeros(cfg.local_bins, dtype=np.float32)
                    )
                    local_mask = (
                        data["local_mask"].astype(np.float32)
                        if "local_mask" in data
                        else np.ones(cfg.local_bins, dtype=np.float32)
                    )
                    g_channels = np.stack(
                        [
                            data["global_view"].astype(np.float32),
                            global_std,
                            global_mask,
                        ],
                        axis=-1,
                    ).astype(np.float32)
                    l_channels = np.stack(
                        [
                            data["local_view"].astype(np.float32),
                            local_std,
                            local_mask,
                        ],
                        axis=-1,
                    ).astype(np.float32)
                g_mean = g_channels[:, 0]
                g_std = g_channels[:, 1]
                g_mask = g_channels[:, 2]
                l_mean = l_channels[:, 0]
                l_std = l_channels[:, 1]
                l_mask = l_channels[:, 2]
                eng = data["engineered"].astype(np.float32)
                local_half_window = float(data["local_half_window"])

            pieces = [g_mean, g_std, g_mask, l_mean, l_std, l_mask, eng]
            scalar_pieces = [eng]

            if cfg.use_bls:
                bls = load_json(bls_path)
                bls_vals = np.array(
                    [
                        float(bls.get("bls_snr", 0.0)),
                        float(bls.get("bls_depth", 0.0)),
                        float(bls.get("bls_duration", 0.0)),
                        float(bls.get("bls_period", 0.0)),
                    ],
                    dtype=np.float32,
                )
                pieces.append(bls_vals)
                scalar_pieces.append(bls_vals)

            x = np.concatenate(pieces, axis=0).astype(np.float32)
            x_scalar = np.concatenate(scalar_pieces, axis=0).astype(np.float32)
            if not np.all(np.isfinite(x)):
                continue
            if not (
                np.all(np.isfinite(g_channels))
                and np.all(np.isfinite(l_channels))
                and np.all(np.isfinite(x_scalar))
            ):
                continue

            X_list.append(x)
            X_global_list.append(g_channels)
            X_local_list.append(l_channels)
            X_scalar_list.append(x_scalar)
            y_list.append(int(row["y"]))

            manifest_rows.append(
                {
                    "candidate_id": candidate_id,
                    "kepid": kepid,
                    "y": int(row["y"]),
                    "koi_disposition": row.get("koi_disposition", ""),
                    "koi_period": float(row["koi_period"]),
                    "koi_time0bk": float(row["koi_time0bk"]),
                    "koi_duration": float(row["koi_duration"]),
                    "koi_depth": safe_float(row.get("koi_depth", np.nan)),
                    "koi_snr": safe_float(row.get("koi_snr", np.nan)),
                    "local_half_window": local_half_window,
                }
            )

        except Exception as exc:
            logger.warning("Failed assembling candidate %s: %s", candidate_id, exc)
            continue

    if not X_list:
        raise RuntimeError("No usable candidates were assembled into the final dataset.")

    X = np.vstack(X_list).astype(np.float32)
    X_global = np.stack(X_global_list, axis=0).astype(np.float32)
    X_local = np.stack(X_local_list, axis=0).astype(np.float32)
    X_scalar = np.vstack(X_scalar_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    manifest = pd.DataFrame(manifest_rows)

    dataset_path = paths.processed / dataset_name(cfg)
    np.savez_compressed(
        dataset_path,
        X=X,
        X_global=X_global,
        X_local=X_local,
        X_scalar=X_scalar,
        y=y,
        feature_names=np.array(feature_names, dtype=object),
        scalar_feature_names=np.array(scalar_feature_names, dtype=object),
        preprocessing_version=np.array(PREPROCESS_VERSION),
    )
    manifest.to_csv(paths.manifest_csv, index=False)

    logger.info(
        "Saved processed dataset: %s X=%s X_global=%s X_local=%s X_scalar=%s y=%s pos=%d/%d",
        dataset_path,
        X.shape,
        X_global.shape,
        X_local.shape,
        X_scalar.shape,
        y.shape,
        int(y.sum()),
        len(y),
    )
    logger.info("Saved manifest: %s", paths.manifest_csv)

    return X, y, manifest, feature_names


def save_grouped_splits(manifest: pd.DataFrame, cfg: PrepConfig, paths: Paths, logger: logging.Logger) -> None:
    idx = np.arange(len(manifest))
    y = manifest["y"].values
    groups = manifest["kepid"].values

    gss_outer = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.seed)
    train_val_idx, test_idx = next(gss_outer.split(idx, y=y, groups=groups))

    train_val = manifest.iloc[train_val_idx].reset_index(drop=True)

    val_frac_of_train_val = cfg.val_size / (1.0 - cfg.test_size)
    gss_inner = GroupShuffleSplit(n_splits=1, test_size=val_frac_of_train_val, random_state=cfg.seed)
    inner_idx = np.arange(len(train_val))
    train_idx_rel, val_idx_rel = next(
        gss_inner.split(inner_idx, y=train_val["y"].values, groups=train_val["kepid"].values)
    )

    train_idx = train_val_idx[train_idx_rel]
    val_idx = train_val_idx[val_idx_rel]

    split_path = paths.processed / splits_name(cfg)
    np.savez_compressed(
        split_path,
        train_idx=train_idx.astype(np.int64),
        val_idx=val_idx.astype(np.int64),
        test_idx=test_idx.astype(np.int64),
    )

    def summarize(name: str, ids: np.ndarray) -> Dict[str, Any]:
        part = manifest.iloc[ids]
        return {
            "split": name,
            "n": int(len(part)),
            "pos": int(part["y"].sum()),
            "pos_rate": float(part["y"].mean()),
            "unique_kepid": int(part["kepid"].nunique()),
        }

    summary = pd.DataFrame(
        [
            summarize("train", train_idx),
            summarize("val", val_idx),
            summarize("test", test_idx),
        ]
    )
    logger.info("Saved splits: %s", split_path)
    logger.info("\n%s", summary.to_string(index=False))


def shuffled_candidate_pool(koi_df: pd.DataFrame, cfg: PrepConfig) -> pd.DataFrame:
    koi_df = koi_df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    requested_pool_n = min(len(koi_df), int(math.ceil(cfg.target_n * cfg.oversample_factor)))
    return koi_df.head(requested_pool_n).copy()


def prepare_candidate_pool(cfg: PrepConfig, paths: Paths, logger: logging.Logger) -> pd.DataFrame:
    raw_df = download_koi_metadata(cfg, paths, logger)
    koi_df = prepare_binary_koi_table(raw_df, cfg, paths, logger)
    pool_df = shuffled_candidate_pool(koi_df, cfg)
    logger.info(
        "Candidate pool prepared: pool_rows=%d target_n=%d unique_kepid=%d",
        len(pool_df),
        cfg.target_n,
        pool_df["kepid"].nunique(),
    )
    return pool_df


def run_download_stage(cfg: PrepConfig, paths: Paths, logger: logging.Logger) -> pd.DataFrame:
    pool_df = prepare_candidate_pool(cfg, paths, logger)

    unique_kepids = list(pd.unique(pool_df["kepid"].astype(int)))
    download_stats = {
        "requested_unique_kepid": len(unique_kepids),
        "cache_hit": 0,
        "download_ok": 0,
        "failed": 0,
    }

    valid_star_ids: set[int] = set()

    for kepid in tqdm(unique_kepids, desc="Caching star light curves"):
        ok, reason = download_kepler_lightcurve_for_star(kepid, cfg, paths, logger)
        if ok:
            valid_star_ids.add(int(kepid))
            if reason == "cache_hit":
                download_stats["cache_hit"] += 1
            else:
                download_stats["download_ok"] += 1
        else:
            download_stats["failed"] += 1
            logger.warning("Skipping star kepid=%s reason=%s", kepid, reason)

    logger.info("Download stats: %s", json.dumps(download_stats, indent=2))
    downloaded_pool_df = pool_df[pool_df["kepid"].astype(int).isin(valid_star_ids)].reset_index(drop=True)
    logger.info(
        "Download stage complete: usable_rows=%d unique_kepid=%d",
        len(downloaded_pool_df),
        downloaded_pool_df["kepid"].nunique() if len(downloaded_pool_df) else 0,
    )
    return downloaded_pool_df


def run_prepare_stage(
    cfg: PrepConfig,
    paths: Paths,
    logger: logging.Logger,
    pool_df: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
    if pool_df is None:
        pool_df = prepare_candidate_pool(cfg, paths, logger)

    usable_rows: List[pd.Series] = []
    candidate_failures = 0
    candidate_records = [
        record
        for record in pool_df.to_dict(orient="records")
        if validate_star_cache(star_cache_path(paths, int(record["kepid"])))
    ]
    workers = resolve_prepare_workers(cfg)
    logger.info(
        "Prepare stage worker setup: workers=%d cpu_count=%s candidates=%d",
        workers,
        os.cpu_count(),
        len(candidate_records),
    )

    if workers <= 1:
        for record in tqdm(candidate_records, total=len(candidate_records), desc="Building candidate caches"):
            row = pd.Series(record)
            ok, reason = build_candidate_cache(row, cfg, paths, logger)
            if ok:
                usable_rows.append(row)
                if len(usable_rows) >= cfg.target_n:
                    break
            else:
                candidate_failures += 1
                logger.warning("Skipping candidate %s reason=%s", row["candidate_id"], reason)
    else:
        chunksize = max(1, len(candidate_records) // (workers * 4)) if candidate_records else 1
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=init_prepare_worker,
            initargs=(cfg,),
        ) as executor:
            future_to_record = {
                executor.submit(build_candidate_cache_worker, record): record for record in candidate_records
            }
            for future in tqdm(
                as_completed(future_to_record),
                total=len(candidate_records),
                desc="Building candidate caches",
            ):
                ok, reason, record = future.result()
                row = pd.Series(record)
                if ok:
                    usable_rows.append(row)
                else:
                    candidate_failures += 1
                    logger.warning("Skipping candidate %s reason=%s", row["candidate_id"], reason)

    usable_df = pd.DataFrame(usable_rows).reset_index(drop=True)
    if len(usable_df) > cfg.target_n:
        usable_df = usable_df.head(cfg.target_n).reset_index(drop=True)

    logger.info(
        "Usable candidate cache summary: usable=%d target_n=%d failures=%d unique_kepid=%d",
        len(usable_df),
        cfg.target_n,
        candidate_failures,
        usable_df["kepid"].nunique() if len(usable_df) else 0,
    )

    if len(usable_df) < max(1000, int(0.5 * cfg.target_n)):
        logger.warning(
            "Usable candidate count is much lower than requested. "
            "Consider increasing oversample_factor or relaxing filters."
        )

    X, y, manifest, feature_names = build_final_dataset_and_manifest(usable_df, cfg, paths, logger)
    save_grouped_splits(manifest, cfg, paths, logger)
    logger.info("Prepare stage complete: X=%s y=%s feature_count=%d", X.shape, y.shape, len(feature_names))
    return X, y, manifest, feature_names


def run_full_pipeline(
    cfg: PrepConfig,
    paths: Paths,
    logger: logging.Logger,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
    pool_df = run_download_stage(cfg, paths, logger)
    return run_prepare_stage(cfg, paths, logger, pool_df=pool_df)


def build_common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--base_dir", type=str, required=True, help="Base project/data directory.")
    parser.add_argument("--target_n", type=int, default=10000, help="Desired number of usable candidates.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--global_bins", type=int, default=200)
    parser.add_argument("--local_bins", type=int, default=100)
    parser.add_argument("--global_window", type=float, default=0.5)
    parser.add_argument("--local_window_mult", type=float, default=4.0)
    parser.add_argument("--detrend_window_days", type=float, default=1.5)
    parser.add_argument("--sigma_clip_sigma", type=float, default=5.0)
    parser.add_argument("--use_bls", action="store_true")
    parser.add_argument("--force_redownload_metadata", action="store_true")
    parser.add_argument("--force_recompute_candidate_cache", action="store_true")
    parser.add_argument(
        "--prepare_workers",
        type=int,
        default=0,
        help="Worker processes for feature preparation. 0 = auto, 1 = serial.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> PrepConfig:
    return PrepConfig(
        base_dir=args.base_dir,
        target_n=args.target_n,
        seed=args.seed,
        global_bins=args.global_bins,
        local_bins=args.local_bins,
        global_window=args.global_window,
        local_window_mult=args.local_window_mult,
        detrend_window_days=args.detrend_window_days,
        sigma_clip_sigma=args.sigma_clip_sigma,
        use_bls=args.use_bls,
        force_redownload_metadata=args.force_redownload_metadata,
        force_recompute_candidate_cache=args.force_recompute_candidate_cache,
        prepare_workers=args.prepare_workers,
    )


def init_pipeline(args: argparse.Namespace, log_name: str, logger_name: str) -> Tuple[PrepConfig, Paths, logging.Logger]:
    cfg = config_from_args(args)
    paths = Paths(cfg.base_dir)
    logger = setup_logger(paths.logs / log_name, logger_name=logger_name)
    set_seed(cfg.seed)
    logger.info("Config:\n%s", json.dumps(asdict(cfg), indent=2))
    return cfg, paths, logger
