# rojin-ml-exoplanet

Kepler-first exoplanet ML pipeline with reusable raw-data caching, versioned preprocessing, and train-ready artifacts for CNN and tabular baselines.

Detailed project timeline, experiment history, and results summary:

- [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md)

## Project structure

```text
rojin-ml-exoplanet/
├── README.md
├── EXPERIMENT_LOG.md
├── requirements.txt
├── scripts/
│   ├── kepler_download_raw.py
│   ├── kepler_pipeline_lib.py
│   └── kepler_prepare_features.py
├── training/
│   ├── cnn/
│   │   └── train_kepler_cnn.py
│   ├── stacking/
│   │   └── stack_cnn_hgb.py
│   ├── transformer/
│   │   └── train_kepler_transformer.py
│   └── tabular/
│       └── train_hgb.py
└── setup/
    └── setup_torch_cuda.py
```

## Quick start

Clone the repository and move into the project root:

```bash
git clone <your-repo-url>
cd rojin-ml-exoplanet
export PROJECT_ROOT="$(pwd)"
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install Python dependencies:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Install a compatible PyTorch build for your machine:

```bash
python setup/setup_torch_cuda.py
```

This setup step:

- detects the NVIDIA GPU with `nvidia-smi` when available
- installs a matching PyTorch wheel
- works as a safer entrypoint than manually guessing a CUDA build

## Workflow

The recommended workflow separates stable, expensive work from experiment-heavy work:

1. Download and cache raw Kepler data once.
2. Build reusable candidate-level features and processed datasets.
3. Train models from the processed artifacts.

Fast path:

- If your clone already contains `data/processed/kepler/`, you can skip the raw-download and feature-preparation stages and start directly with the training commands below.
- Rebuild steps are only needed when the processed artifacts are missing or when you intentionally want to regenerate the dataset.

## Step 1: Download raw Kepler data

You only need this step if the processed dataset is not already available in your clone, or if you want to regenerate everything from source data.

This stage downloads or reuses:

- KOI metadata
- raw per-star light-curve caches

Run:

```bash
python scripts/kepler_download_raw.py \
  --base_dir "$PROJECT_ROOT" \
  --target_n 10000
```

Notes:

- Raw light curves are cached per star.
- Once the raw cache exists, later feature and training iterations should not need to redownload it.

## Step 2: Build processed features

You only need this step if the processed dataset is missing or if you want to regenerate it from the raw cache.

This stage:

- reads the raw per-star cache
- detrends and folds each candidate light curve
- builds `global` and `local` transit views
- computes scalar features and optional BLS features
- writes reusable interim candidate caches
- writes the final processed dataset and grouped splits

Run:

```bash
python scripts/kepler_prepare_features.py \
  --base_dir "$PROJECT_ROOT" \
  --target_n 10000 \
  --global_bins 200 \
  --local_bins 100 \
  --prepare_workers 8 \
  --use_bls
```

If preprocessing logic changes and you want to rebuild candidate-level caches without touching the raw cache:

```bash
python scripts/kepler_prepare_features.py \
  --base_dir "$PROJECT_ROOT" \
  --target_n 10000 \
  --global_bins 200 \
  --local_bins 100 \
  --prepare_workers 8 \
  --use_bls \
  --force_recompute_candidate_cache
```

### What the preprocessing currently produces

The processed dataset includes:

- `X`
  Flat combined feature vector
- `X_global`
  Global transit view array
- `X_local`
  Local transit view array
- `X_scalar`
  Engineered and optional BLS scalar features
- `y`
  Binary labels

The current `v0` preprocessing is ExoMiner-inspired in spirit:

- transit-centered folding
- separate global and local views
- multi-channel view representation
- grouped splits by star

## Data layout

All generated artifacts live under the base directory you pass with `--base_dir`.

```text
$PROJECT_ROOT/
├── data/
│   ├── raw/
│   │   └── kepler/
│   │       ├── metadata/
│   │       └── lightcurves/
│   ├── interim/
│   │   └── kepler/
│   │       └── v0/
│   │           ├── folded/
│   │           ├── bls/
│   │           └── views/
│   └── processed/
│       └── kepler/
├── logs/
└── outputs/
```

Important points:

- `data/raw/` is the stable long-lived cache.
- `data/interim/kepler/v0/` stores candidate-level preprocessing caches for version `v0`.
- `data/processed/kepler/` stores train-ready arrays, splits, and manifests.
- If `data/processed/kepler/` is already included in the repository or distributed alongside it, users can start directly from the training section.

If you later introduce a preprocessing change that should not mix with existing interim artifacts, bump the version label in the code from `v0` to `v1`, `v2`, and so on.

## Training

If the processed dataset is already present in `data/processed/kepler/`, this is the first section you need.

### CNN

Train the current best ExoMiner-inspired multi-branch CNN:

```bash
python training/cnn/train_kepler_cnn.py \
  --base_dir "$PROJECT_ROOT" \
  --batch_size 256 \
  --epochs 60 \
  --num_workers 4
```

The CNN training script:

- auto-discovers the latest processed dataset and split files unless paths are passed explicitly
- uses separate `global`, `local`, and scalar branches
- applies class-weighted loss, cosine LR decay, gradient clipping, mixed precision on CUDA, early stopping, and validation-threshold tuning
- saves checkpoints, predictions, metric history, and a summary under `outputs/cnn/`

For ablations, you can enable the optional wide branch:

```bash
python training/cnn/train_kepler_cnn.py \
  --base_dir "$PROJECT_ROOT" \
  --epochs 60 \
  --wide_branch
```

For stricter regularization experiments, you can explicitly enable the optional regularization knobs:

```bash
python training/cnn/train_kepler_cnn.py \
  --base_dir "$PROJECT_ROOT" \
  --epochs 60 \
  --label_smoothing 0.02 \
  --view_noise_std 0.02 \
  --scalar_noise_std 0.01
```

### Tabular baseline

Train the full flattened-feature HGB baseline:

```bash
python training/tabular/train_hgb.py \
  --base_dir "$PROJECT_ROOT" \
  --feature_set full
```

Train the scalar-only HGB baseline:

```bash
python training/tabular/train_hgb.py \
  --base_dir "$PROJECT_ROOT" \
  --feature_set scalar
```

The HGB script:

- trains on the same processed dataset and grouped splits as the CNN
- supports `scalar` and `full` feature modes
- runs a small validation-driven regularization search by default
- tunes the classification threshold on validation
- saves summaries, predictions, model files, and candidate-search results under `outputs/tabular/`

### Stacking

Blend the saved CNN and HGB runs into a validation-tuned ensemble:

```bash
python training/stacking/stack_cnn_hgb.py \
  --base_dir "$PROJECT_ROOT" \
  --method blend \
  --selection_metric f1
```

By default this script:

- auto-discovers the latest CNN and HGB run folders under `outputs/`
- merges their validation and test prediction files by candidate
- tunes the blend weight and classification threshold on validation
- writes ensemble summaries and predictions under `outputs/stacking/`

If you want to stack specific runs explicitly:

```bash
python training/stacking/stack_cnn_hgb.py \
  --base_dir "$PROJECT_ROOT" \
  --cnn_run_dir outputs/cnn/<cnn-run-dir> \
  --hgb_run_dir outputs/tabular/<hgb-run-dir> \
  --method blend
```

### Transformer

Train the experimental transformer-based sequence model:

```bash
python training/transformer/train_kepler_transformer.py \
  --base_dir "$PROJECT_ROOT" \
  --batch_size 192 \
  --epochs 80 \
  --num_workers 4
```

This transformer is a modern extension, not a faithful ExoMiner reproduction:

- it treats the `global` and `local` views as short sequences
- it uses separate transformer branches plus a scalar-feature branch
- it keeps the same processed dataset and grouped splits as the CNN and HGB baselines

ExoMiner itself is not described as a transformer model. The original ExoMiner paper presents a modular deep-learning vetting system, and later ExoMiner++ descriptions still refer to multi-branch CNN-style architectures rather than transformers.

If you want the old single-configuration behavior:

```bash
python training/tabular/train_hgb.py \
  --base_dir "$PROJECT_ROOT" \
  --feature_set full \
  --single_config
```

## Recommended iteration loop

Use this as the default project loop:

1. Run `kepler_download_raw.py` once.
2. Iterate on `kepler_prepare_features.py` only when preprocessing changes.
3. Train and compare models from `data/processed/kepler/`.

If processed artifacts are already available in the repository, a new user can use this shorter loop instead:

1. Clone the repository.
2. Set up the environment.
3. Run the training commands directly.

## Reproducibility notes

The project is reproducible in the sense that:

- preprocessing is scripted
- grouped splits are saved
- model training reads from saved processed artifacts

If prebuilt processed artifacts are included in the repository, a fresh clone can skip the long raw-download and feature-generation steps and start from training immediately.

## Current status

The project currently contains:

- a stable raw-cache workflow
- a versioned `v0` preprocessing pipeline
- a strong CNN baseline
- a strong HGB tabular baseline

For the full experiment history, run inventory, and conclusions, see:

- [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md)
