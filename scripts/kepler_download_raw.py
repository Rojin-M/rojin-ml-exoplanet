#!/usr/bin/env python3
from __future__ import annotations

from kepler_pipeline_lib import build_common_parser, init_pipeline, run_download_stage


def main() -> None:
    parser = build_common_parser("Download KOI metadata and cache raw Kepler light curves.")
    args = parser.parse_args()

    cfg, paths, logger = init_pipeline(
        args,
        log_name="kepler_download_raw.log",
        logger_name="kepler_download_raw",
    )
    logger.info("Starting Kepler raw download stage")
    run_download_stage(cfg, paths, logger)
    logger.info("Finished raw download stage")


if __name__ == "__main__":
    main()
