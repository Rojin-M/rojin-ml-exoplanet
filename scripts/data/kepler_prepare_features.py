#!/usr/bin/env python3
from __future__ import annotations

from kepler_pipeline_lib import build_common_parser, init_pipeline, run_prepare_stage


def main() -> None:
    parser = build_common_parser("Build cached candidate features and the final processed Kepler dataset.")
    args = parser.parse_args()

    cfg, paths, logger = init_pipeline(
        args,
        log_name="kepler_prepare_features.log",
        logger_name="kepler_prepare_features",
    )
    logger.info("Starting Kepler feature preparation stage")
    X, y, _, feature_names = run_prepare_stage(cfg, paths, logger)
    logger.info("Finished feature preparation: X=%s y=%s feature_count=%d", X.shape, y.shape, len(feature_names))


if __name__ == "__main__":
    main()
