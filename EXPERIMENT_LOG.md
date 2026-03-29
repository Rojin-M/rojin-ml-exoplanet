# Kepler Experiment Log

This document records what we changed in the project from the first raw-data stage up to the current CNN and HGB experiments. It is meant to be the single place to answer:

- what the pipeline does
- what we changed and why
- which experiments helped
- which experiments hurt
- what the current best setup is

Written from the current repo state on `2026-03-29`.

## Goal

Build a Kepler-first, ExoMiner-inspired exoplanet-vetting pipeline with:

- stable raw-data caches
- reusable preprocessing
- grouped train/val/test splits by star
- strong baselines from both CNN and tabular models

## Repo Evolution

The project started from a single large preprocessing script. It was then reorganized into separate stages while keeping the same data layout:

- [`scripts/kepler_download_raw.py`](scripts/kepler_download_raw.py)
  Downloads or reuses metadata and raw per-star light-curve caches.
- [`scripts/kepler_prepare_features.py`](scripts/kepler_prepare_features.py)
  Builds candidate-level caches and final processed arrays from raw caches.
- [`scripts/kepler_pipeline_lib.py`](scripts/kepler_pipeline_lib.py)
  Shared implementation for download, preprocessing, caching, and splits.
- [`training/cnn/train_kepler_cnn.py`](training/cnn/train_kepler_cnn.py)
  ExoMiner-inspired CNN training entrypoint.
- [`training/tabular/train_hgb.py`](training/tabular/train_hgb.py)
  HistGradientBoosting baselines.

Important design decision:

- `data/raw/` is treated as stable and reusable.
- `data/interim/kepler/v0/` stores preprocessing caches for version `v0`.
- `data/processed/kepler/` stores the train-ready dataset and split artifacts.

## Raw Data Stage

The raw stage finished successfully and reused the existing cache:

- usable KOI candidates after filtering: `5518`
- positives: `2128`
- negatives: `3390`
- unique stars (`kepid`): `4929`
- raw cache hits during final raw-stage run: `4929 / 4929`
- raw download failures: `0`

This means the raw stage is stable and should usually not be touched again unless the source cache is corrupted or intentionally regenerated.

## Preprocessing Timeline

### 1. Split the monolithic pipeline

We separated expensive downloads from changeable feature work:

- raw download/cache
- feature preparation
- training

This made it possible to rerun feature generation without redownloading Kepler data.

### 2. Added preprocessing versioning

Interim artifacts now live under:

- `data/interim/kepler/v0/folded/`
- `data/interim/kepler/v0/views/`
- `data/interim/kepler/v0/bls/`

This avoids mixing candidate caches across different preprocessing versions.

### 3. Improved the feature-prep pipeline

The current `v0` feature pipeline:

- loads cached per-star raw `time` and `flux`
- sigma-clips and detrends before folding
- folds on the KOI ephemeris
- builds `global` and `local` views
- uses multi-channel views: `mean`, `std`, `mask`
- computes engineered scalar features
- optionally computes BLS scalar features
- saves final arrays:
  - `X`
  - `X_global`
  - `X_local`
  - `X_scalar`
  - `y`

### 4. Fixed a BLS bug

During early feature runs, BLS failed for many candidates because the searched maximum duration could exceed the minimum searched period. This was fixed so BLS duration settings are valid.

### 5. Parallelized candidate preparation

Candidate cache building was moved to process-based parallelism and the progress reporting was fixed so the progress bar updates while work is actually completing.

## Current Dataset Snapshot

All training runs below used the same processed artifact:

- dataset: `data/processed/kepler/dataset_kepler_v0_target10000_g200_l100_gw0.5_lwm4.0_bls1_seed42.npz`
- splits: `data/processed/kepler/splits_kepler_v0_seed42.npz`
- manifest: `data/processed/kepler/manifest_kepler_v0.csv`

Shapes:

- `X`: `(5518, 907)`
- `X_global`: `(5518, 200, 3)`
- `X_local`: `(5518, 100, 3)`
- `X_scalar`: `(5518, 7)`
- `y`: `(5518,)`

Split summary:

- train: `3853` examples, `1505` positives, `3449` unique stars
- val: `836` examples, `327` positives, `740` unique stars
- test: `829` examples, `296` positives, `740` unique stars
- stars with multiple candidates: `433`

Why the split matters:

- grouping by `kepid` reduces star-level leakage
- train-only normalization is used in both CNN and tabular training

## Data Leakage and Overfitting Notes

Current status:

- No obvious code-level data leak was found.
- Grouped splits by `kepid` are in place.
- CNN normalization uses `train_idx` statistics only.
- HGB and CNN both tune thresholds on validation, not test.

But:

- the HGB full model shows strong train-vs-test overfitting
- the CNN still shows some train-vs-test gap, but less than HGB
- repeated model selection on the same test split can still create experiment-selection bias

So the setup is clean enough for a bachelor-level project, but future work should still be careful not to overuse the test set during tuning.

## Model Experiment Strategy

All model experiments below used the same:

- processed dataset
- grouped train/val/test split
- candidate pool
- label definition

So the model comparisons are fair in the sense that architecture and training choices changed, but the underlying data stayed fixed.

### What we compared

We compared two model families:

- `CNN`
  ExoMiner-inspired multi-view neural model using `X_global`, `X_local`, and `X_scalar`
- `HGB`
  HistGradientBoosting baselines using either `X_scalar` only or the full flattened `X`

### What we used for model selection

- validation set for early stopping
- validation set for threshold tuning
- test set only for final comparison after each run

Primary metrics we cared about:

- `PR-AUC`
  Important because this is a detection/vetting problem
- `F1`
  Useful thresholded summary
- `accuracy`
  Easy to report, but not the only metric
- `ROC-AUC`
  Secondary ranking metric

## CNN Experiment History

### CNN setup we started from

The CNN was designed to be ExoMiner-inspired rather than a literal ExoMiner reproduction:

- separate `global` and `local` view branches
- scalar feature branch
- grouped split by star
- class-weighted BCE loss
- cosine learning-rate schedule
- early stopping
- threshold tuning on validation

### What we tuned in the CNN

We changed and compared:

- branch architecture
  - stronger transit branches
  - pooling strategy
  - branch interaction features in the fusion head
- extra feature usage
  - whether to add a wide branch over the full flattened `X`
- training regularization
  - gradient clipping
  - optional label smoothing
  - optional input noise
  - optional flat-feature dropout
- reruns of the best architecture
  - to check whether gains were stable or just a lucky run

### CNN run-by-run results

| Run | What changed | Test accuracy | Test F1 | Test PR-AUC | Test ROC-AUC | What we learned |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `cnn_20260329_141634` | First strong multi-branch CNN baseline | 0.8818 | 0.8409 | 0.8813 | 0.9423 | Good starting point, but still weaker than full HGB |
| `cnn_20260329_143801` | Added a wide branch over flat `X` | 0.8384 | 0.7767 | 0.8153 | 0.8983 | Clearly bad idea; severe overfitting |
| `cnn_20260329_143904` | Removed wide branch and strengthened transit branches | 0.8999 | 0.8685 | 0.9259 | 0.9608 | Major improvement; CNN became competitive |
| `cnn_20260329_144204` | Same no-wide architecture, rerun | 0.9011 | 0.8682 | 0.9315 | 0.9640 | Best ranking metrics among CNN runs |
| `cnn_20260329_145145` | Strict anti-overfit settings: label smoothing + input noise | 0.8963 | 0.8631 | 0.9128 | 0.9545 | Too much regularization; hurt results |
| `cnn_20260329_145431` | Plain no-wide CNN rerun | 0.8987 | 0.8627 | 0.9238 | 0.9588 | Solid rerun, but not best |
| `cnn_20260329_145916` | Plain no-wide CNN rerun | 0.9059 | 0.8734 | 0.9266 | 0.9626 | Best thresholded CNN result overall |

### CNN findings

What helped:

- keeping the model focused on the ExoMiner-style multi-view inputs
- strengthening the `global` and `local` transit branches
- better pooling inside the transit branches
- feature interactions between the global and local embeddings

What did not help:

- adding a wide branch over the flat combined feature vector
- enabling strict regularization by default

Interpretation:

- The CNN benefits from a cleaner inductive bias around transit morphology.
- Once we gave it a large flat branch, it behaved more like a memorizing tabular model and generalized worse.
- The final plain no-wide CNN is both stronger and conceptually cleaner for the project.

Important nuance:

- best CNN for thresholded metrics like accuracy and F1: `cnn_20260329_145916`
- best CNN for ranking metrics like PR-AUC and ROC-AUC: `cnn_20260329_144204`

## HGB Experiment History

### What we tuned in HGB

We used HistGradientBoosting as the tabular baseline and compared:

- `scalar` features only
- full flattened feature vector `X`

We also tested a small validation-driven search over HGB regularization/complexity:

- `learning_rate`
- `max_iter`
- `max_leaf_nodes`
- `max_depth`
- `min_samples_leaf`
- `l2_regularization`

### HGB run-by-run results

| Run | Feature set / search mode | Test accuracy | Test F1 | Test PR-AUC | Test ROC-AUC | What we learned |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `hgb_scalar_20260329_142755` | Scalar-only HGB | 0.7998 | 0.7530 | 0.7782 | 0.8772 | Scalar-only input is not enough |
| `hgb_full_20260329_142827` | Full flattened features | 0.8938 | 0.8581 | 0.9225 | 0.9598 | Strong baseline; initially beat the first CNN |
| `hgb_full_20260329_145130` | Full HGB with regularization search | 0.8938 | 0.8581 | 0.9225 | 0.9598 | Search picked the original base candidate |
| `hgb_full_20260329_145700` | Repeated full HGB with regularization search | 0.8938 | 0.8581 | 0.9225 | 0.9598 | Same outcome again; base config still best |

### HGB findings

What helped:

- using the full flattened processed representation instead of only scalar features

What did not help:

- stronger regularization candidates
- smaller or more constrained trees on this split

Interpretation:

- The processed feature representation is rich enough that a tree ensemble can perform very strongly.
- On this split, more constrained HGB variants did not improve validation quality, even though the model clearly overfits more than the CNN.
- That makes HGB full a strong and important baseline, but not the final best model.

Important nuance:

- HGB full had the best validation PR-AUC across all runs (`0.9483`)
- but the best CNN generalized slightly better to the held-out test split

## Direct Model Comparison

The most useful final comparison is:

- best thresholded CNN: `cnn_20260329_145916`
- best ranking CNN: `cnn_20260329_144204`
- best HGB: `hgb_full_20260329_145700`

What this comparison tells us:

- `CNN` is now the strongest overall model on the held-out test split
- `HGB full` remains a very strong baseline and is close enough that blending/stacking may still help
- `HGB scalar` is useful mostly as a weak-reference baseline showing that scalar features alone are not sufficient

## Current Best Models

### Best CNN for thresholded classification metrics

Artifact:

- `outputs/cnn/dataset_kepler_v0_target10000_g200_l100_gw0.5_lwm4.0_bls1_seed42_cnn_20260329_145916/summary.json`

Test metrics:

- accuracy: `0.9059`
- F1: `0.8734`
- PR-AUC: `0.9266`
- ROC-AUC: `0.9626`

### Best CNN for ranking metrics

Artifact:

- `outputs/cnn/dataset_kepler_v0_target10000_g200_l100_gw0.5_lwm4.0_bls1_seed42_cnn_20260329_144204/summary.json`

Test metrics:

- accuracy: `0.9011`
- F1: `0.8682`
- PR-AUC: `0.9315`
- ROC-AUC: `0.9640`

### Best HGB

Artifact:

- `outputs/tabular/dataset_kepler_v0_target10000_g200_l100_gw0.5_lwm4.0_bls1_seed42_hgb_full_20260329_145700/summary.json`

Test metrics:

- accuracy: `0.8938`
- F1: `0.8581`
- PR-AUC: `0.9225`
- ROC-AUC: `0.9598`

## What Clearly Helped

- Separating raw download from feature generation
- Treating `data/raw/` as stable and reusable
- Versioning interim caches under `data/interim/kepler/v0/`
- Detrending before folding
- Multi-channel `global` and `local` views
- Saving `X_global`, `X_local`, and `X_scalar` separately
- Grouped splits by `kepid`
- Stronger CNN transit branches without the extra wide flat branch

## What Clearly Hurt

- Using only scalar features for HGB
- Adding the CNN wide branch over full `X`
- Making CNN anti-overfit regularization too strict

## Current Recommended Commands

### Raw stage

```bash
python scripts/kepler_download_raw.py \
  --base_dir /local00/student/moradian/rojin-ml-exoplanet \
  --target_n 10000
```

### Feature stage

```bash
python scripts/kepler_prepare_features.py \
  --base_dir /local00/student/moradian/rojin-ml-exoplanet \
  --target_n 10000 \
  --global_bins 200 \
  --local_bins 100 \
  --prepare_workers 8 \
  --use_bls
```

### Best current CNN command

```bash
python training/cnn/train_kepler_cnn.py \
  --base_dir /local00/student/moradian/rojin-ml-exoplanet \
  --batch_size 256 \
  --epochs 60 \
  --num_workers 4
```

### Best current HGB baseline

```bash
python training/tabular/train_hgb.py \
  --base_dir /local00/student/moradian/rojin-ml-exoplanet \
  --feature_set full
```

## What We Have Not Done Yet

- blending or stacking CNN and HGB predictions
- nested cross-validation or a second untouched holdout split
- a `v1` preprocessing version with richer ExoMiner-style auxiliary views
- centroid or stellar-metadata branches

## Current Bottom Line

The project now has:

- a stable raw-data cache
- a reusable `v0` preprocessing pipeline
- a strong ExoMiner-inspired CNN
- a strong full-feature HGB baseline

Best practical conclusion right now:

- `HGB full` is the strongest tabular baseline and a very good sanity check
- the current plain no-wide CNN is the strongest overall model on the held-out test set
- the next most worthwhile experiment is probably a careful CNN+HGB blend/stack rather than more preprocessing churn

## Validation Status and Practical-Work Readiness

Validation checks performed on the current repo state:

- the numbers in this log were cross-checked against the saved `summary.json` files in `outputs/cnn/` and `outputs/tabular/`
- train, validation, and test index sets have no overlap
- grouped star IDs (`kepid`) also have no overlap across train/val/test
- the repeated HGB full runs are genuinely the same result, not a logging mistake
- the wide-branch CNN regression and the strict anti-overfit CNN regression are real and reproducible in the saved artifacts

Practical assessment:

- Yes, this is already good enough for a bachelor practical work.
- The project has a real data pipeline, not just notebook-level modeling.
- It has a meaningful experimental story:
  - raw caching
  - reusable preprocessing
  - a strong tabular baseline
  - a stronger CNN
  - failed ablations that taught us something
- The results are strong enough to discuss seriously, but still modest and believable for student work.

Why it is strong for a student project:

- the dataset preparation is reproducible
- the split strategy is defensible
- there is a real comparison between model families
- there are both positive and negative experimental findings
- the best model does not come from vague handwaving; it comes from tracked runs

Caveats that should be stated honestly in a report:

- the same test split has been consulted multiple times during iteration, so future tuning should mostly stop here
- results are from a single split, not from repeated seeds or cross-validation
- no stacking/blending result is included yet
- no detailed qualitative failure analysis is included yet

Recommendation for practical work:

- Yes, we can hold here.
- Use the current best plain CNN and the full HGB baseline as the main comparison.
- Treat the strict anti-overfit and wide-branch runs as informative negative results.
- If one extra experiment is wanted, do a simple CNN+HGB blend; otherwise, stop tuning and start writing.
