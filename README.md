# Kepler transit-candidate classification

A reproducible pipeline comparing tabular HistGradientBoosting, a multi-view CNN, CNN–HGB blending, and a transformer on a fixed Kepler dataset. Metadata-enabled v1 extends the same light-curve inputs used in v0.

Read the [experimental storyline](STORYLINE.md) for the questions, findings, and limitations. The [published results](results/README.md) include the [complete model comparison](results/comparison/COMPARISON.md) from the latest completed reproduction. The workflow below generates your own runs and plots under `outputs/`.

CNN v1 is the main practical model. Transformer v1 has a tiny mean test AP lead (0.960631 versus 0.960548), while CNN v1 has higher mean accuracy, F1, and ROC-AUC and fewer parameters. Both metadata extensions improved all four metrics in every matched seed on the fixed split.

## Repository layout

```text
configs/              Fixed experiment settings, protocol, and data checksums
scripts/data/         Downloading, preprocessing, metadata preparation, verification
scripts/experiments/  Training launchers and the complete reproduction workflow
scripts/analysis/     Metric checks, aggregation, and plots
training/             Tabular, CNN, blending/stacking, and transformer implementations
data/processed/kepler/
  v0/                 Base views, scalar/flat inputs, labels, manifest, and split
  v1/                 Identical base inputs plus auxiliary metadata and its provenance
results/              Verified comparison tables, selected plots, and provenance
tests/                Data, training, tracking, and checkpoint checks
outputs/              Local generated runs and figures; ignored by Git
```

## 1. Set up the environment

The tested reference environment uses Linux, Python 3.9, and PyTorch 2.1.2. The pinned scientific dependencies are in `requirements-reference.txt`; `requirements.txt` records broader supported dependency ranges.

```bash
git clone https://github.com/Rojin-M/rojin-ml-exoplanet.git
cd rojin-ml-exoplanet
export PROJECT_ROOT="$(pwd)"
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-reference.txt
```

Install the CPU build:

```bash
python -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
```

For the CUDA 11.8 environment used by the reference GPU runs, install this build instead:

```bash
python -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
```

Other hardware may need a different build; see the [official PyTorch installation matrix](https://pytorch.org/get-started/previous-versions/#v212). The existing `python setup/setup_torch_cuda.py` helper detects hardware and replaces installed Torch packages; explicit installation above gives control over the version.

## 2. Verify the installation and processed data

Check the environment, dataset checksums, split, and implementation before training:

```bash
python -c 'import torch; print(torch.__version__); print("CUDA available:", torch.cuda.is_available())'
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
python scripts/data/check_processed.py
python -m unittest discover -s tests -v
```

**Both processed datasets are included.** Training and evaluation require no raw download, interim cache, catalog request, or W&B account. The dataset has 5,518 candidates and a fixed star-grouped train/validation/test split of 3,853/836/829.

## Run the story in order

Run commands from the repository root: the `rojin-ml-exoplanet/` directory containing `README.md` and `scripts/`.

If setup is already complete and you opened a new terminal, enter your clone and reactivate the existing environment first. Replace `/path/to/rojin-ml-exoplanet` with your clone's actual location:

```bash
cd /path/to/rojin-ml-exoplanet
source .venv/bin/activate
export PROJECT_ROOT="$(pwd)"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
```

Continue with the numbered commands below in that same terminal. `--device auto` selects one CUDA device when available; `--device cpu` and `--device cuda:0` choose explicitly. Seeds run sequentially by default, and W&B is disabled. Keep the same device, GPU UUID list, and tracking options across stages sharing an output directory. The optional parallel-GPU commands below must be selected when starting a new workflow.

The following numbered stages follow the storyline. A fresh clone writes to `outputs/reproduce/` by default. If you already have completed runs and intentionally want a new experiment, add the same new `--output_dir` to every stage; retain the existing results for comparison.

Preview the plan without starting training:

```bash
python scripts/experiments/reproduce.py --dry_run
```

### 3. Tabular baselines: scalar versus full inputs

Establish how much the flattened transit views add beyond the seven scalar features.

```bash
python scripts/experiments/reproduce.py --stage tabular --device auto
```

### 4. Multi-view CNN baseline

Train the preserved global/local/scalar CNN configuration.

```bash
python scripts/experiments/reproduce.py --stage cnn-baseline --device auto
```

### 5. CNN ablations: wide branch and regularization

Compare the baseline with the existing wide-input branch and additional regularization options.

```bash
python scripts/experiments/reproduce.py --stage cnn-ablations --device auto
```

### 6. Blend the CNN and full HGB

Select the blend weight and classification threshold using validation predictions from steps 3 and 4.

```bash
python scripts/experiments/reproduce.py --stage blend --device auto
```

### 7. CNN v0 across five seeds

Measure variability of the fixed baseline on the saved split.

```bash
python scripts/experiments/reproduce.py --stage cnn-v0 --device auto
```

### 8. CNN v1 with metadata across the same five seeds

Repeat the shared settings with the prepared auxiliary inputs and branch.

```bash
python scripts/experiments/reproduce.py --stage cnn-v1 --device auto
```

### 9. Compare CNN v0 and v1

Verify predictions and export paired differences, summary tables, and figures without training another model.

```bash
python scripts/experiments/reproduce.py --stage cnn-comparison --device auto
```

### 10. Transformer normalization and metadata study

Screen input normalization on validation, freeze the selection, then compare metadata across five seeds and generate plots.

```bash
python scripts/experiments/reproduce.py --stage transformer --device auto
```

### 11. Compare all models and inspect the findings

Recompute metrics from the saved predictions and export the final comparison.

```bash
python scripts/experiments/reproduce.py --stage comparison --device auto
```

The full workflow trains two HGB baselines, three individual CNN configurations, ten CNN seed runs, and twelve transformer configurations, plus the blend. CPU execution can take much longer than GPU execution. Completion is recorded in `outputs/reproduce/workflow.json`. Completed stages are reused with the same execution settings. Interrupted/failed stages are retained; choose a new `--output_dir` for a fresh attempt.

To execute steps 3–11 consecutively in a fresh output directory:

```bash
python scripts/experiments/reproduce.py --stage all --device auto
```

The primary CNN comparison is available after `cnn-comparison`. The optional transformer adds a two-seed validation screen of input normalization, followed by a five-seed metadata comparison. Its ten final test evaluations occur after all twelve training runs finish. The `comparison` stage requires all model families and exports the complete table.

| Output | Contents |
| --- | --- |
| `outputs/reproduce/comparison/COMPARISON.md` | Overall metrics and interpretation limits |
| `outputs/reproduce/comparison/comparison.png` | All-model comparison |
| `outputs/reproduce/comparison/runs.csv` | Individual run metrics |
| `outputs/reproduce/comparison/summary.csv` | Individual scores or five-seed mean/sample SD |
| `outputs/reproduce/cnn_comparison/` | Paired CNN v0/v1 tables and figures |
| `outputs/reproduce/transformer/results/` | Transformer screen and metadata tables |
| `outputs/reproduce/transformer/figures/` | Transformer comparisons and learning curves |
| `outputs/reproduce/cnn_baseline/`, `outputs/reproduce/cnn_wide/`, `outputs/reproduce/cnn_regularized/` | Individual runs and their `figures/` subdirectories |
| `outputs/reproduce/cnn_v0/`, `outputs/reproduce/cnn_v1/` | Five-seed checkpoints, predictions, histories, and metric tables |
| `outputs/figures/cnn/cnn_v0/`, `outputs/figures/cnn/cnn_v1/` | Five-seed CNN learning curves and aggregate plots |

The CNN aggregate plots in `outputs/figures/cnn/` use shared group names and are replaced by another aggregation of those groups. The workflow's metric tables remain in its own directory; step 12 exports a separate set of publication figures.

Inspect `outputs/reproduce/comparison/COMPARISON.md` and `summary.csv`, and compare them with the [published comparison](results/comparison/COMPARISON.md). AP is average precision. Error bars describe training-seed variability on the fixed split; they are not confidence intervals. GPU operations are seeded but not forced to be deterministic, so exact scores and selected epochs can vary. The storyline distinguishes historical experiments from the latest reproduction.

### 12. Export the results for sharing

After all stages complete, export the comparison, model-specific tables, selected figures, and provenance. This verifies saved prediction metrics and does not train models. A clone already contains the published `results/`, so export your reproduction separately:

```bash
python scripts/analysis/export_results.py --workflow_dir outputs/reproduce --output_dir outputs/published_results
```

The destination must be new or empty. To prepare an initial publication when `results/` is empty, use:

```bash
python scripts/analysis/export_results.py --workflow_dir outputs/reproduce --output_dir results
```

The export contains `comparison/`, `tabular/`, `cnn/`, `blend/`, and `transformer/`, with a guide in each model directory. Detailed runs, checkpoints, and histories remain under `outputs/`.

### Verify existing results without retraining

After the workflow completes, check its status and independently recompute the comparison from its prediction files:

```bash
python -c 'import json; print(json.load(open("outputs/reproduce/workflow.json"))["status"])'
python scripts/analysis/compare_all_models.py \
  --workflow_dir outputs/reproduce --output_dir outputs/recheck
```

The first command should print `complete`. The second writes a new comparison and checks prediction metrics against the saved summaries; it does not train models. Use a new output directory for another verification, because existing comparison exports are protected from overwriting.

<details>
<summary>Individual commands behind the workflow</summary>

These commands use a separate output directory. The configuration files contain the full settings; no input config is taken from an ignored historical run.

These manual commands do not create the `workflow.json` manifest required by the publication exporter in step 12. Use the numbered `reproduce.py` workflow to generate that manifest and prepare the publication export.

```bash
export EXPERIMENT_ROOT="$PROJECT_ROOT/outputs/manual"

python scripts/experiments/run_config.py --model tabular \
  --config configs/hgb_scalar.json --output_dir "$EXPERIMENT_ROOT/hgb_scalar"
python scripts/experiments/run_config.py --model tabular \
  --config configs/hgb_full.json --output_dir "$EXPERIMENT_ROOT/hgb_full"

python scripts/experiments/run_config.py --model cnn --config configs/cnn_v0.json \
  --output_dir "$EXPERIMENT_ROOT/cnn_baseline" --device auto
python scripts/experiments/run_config.py --model cnn --config configs/cnn_wide.json \
  --output_dir "$EXPERIMENT_ROOT/cnn_wide" --device auto
python scripts/experiments/run_config.py --model cnn --config configs/cnn_regularized.json \
  --output_dir "$EXPERIMENT_ROOT/cnn_regularized" --device auto

python scripts/experiments/run_config.py --model blend --config configs/blend.json \
  --cnn_record "$EXPERIMENT_ROOT/cnn_baseline/run.json" \
  --hgb_record "$EXPERIMENT_ROOT/hgb_full/run.json" \
  --output_dir "$EXPERIMENT_ROOT/blend"

python scripts/experiments/run_cnn_multiseed.py --source_config configs/cnn_v0.json \
  --group_dir "$EXPERIMENT_ROOT/cnn_v0" --device auto --wandb_mode disabled
python scripts/experiments/run_cnn_multiseed.py --source_config configs/cnn_v1.json \
  --group_dir "$EXPERIMENT_ROOT/cnn_v1" --device auto --wandb_mode disabled
python scripts/analysis/aggregate_cnn_multiseed.py \
  --experiment "$EXPERIMENT_ROOT/cnn_v0/experiment.json"
python scripts/analysis/aggregate_cnn_multiseed.py \
  --experiment "$EXPERIMENT_ROOT/cnn_v1/experiment.json"
python scripts/analysis/compare_cnn_variants.py \
  --baseline "$EXPERIMENT_ROOT/cnn_v0/experiment.json" \
  --candidate "$EXPERIMENT_ROOT/cnn_v1/experiment.json" \
  --output_dir "$EXPERIMENT_ROOT/cnn_comparison/results" \
  --figure_dir "$EXPERIMENT_ROOT/cnn_comparison/figures"

python scripts/experiments/run_transformer_study.py \
  --source_config configs/transformer_base.json \
  --protocol configs/transformer_protocol.json \
  --study_dir "$EXPERIMENT_ROOT/transformer" --device auto --wandb_mode disabled
python scripts/analysis/analyze_transformer_study.py \
  --study_dir "$EXPERIMENT_ROOT/transformer" \
  --figure_dir "$EXPERIMENT_ROOT/transformer/figures" \
  --cnn_v0 "$EXPERIMENT_ROOT/cnn_v0/experiment.json" \
  --cnn_v1 "$EXPERIMENT_ROOT/cnn_v1/experiment.json"

python scripts/analysis/compare_all_models.py --workflow_dir "$EXPERIMENT_ROOT" \
  --output_dir "$EXPERIMENT_ROOT/comparison"
```

`run_config.py` saves a `run.json` pointer, so blending never chooses an unrelated “latest” run. CNN learning curves are generated automatically. To regenerate one run's curves, obtain its exact directory from that pointer:

```bash
CNN_RUN=$(python -c 'import json, os; print(json.load(open(os.path.join(os.environ["EXPERIMENT_ROOT"], "cnn_baseline/run.json")))["run_dir"])')
python scripts/analysis/plot_cnn_history.py --run_dir "$CNN_RUN" \
  --output_dir "$EXPERIMENT_ROOT/replotted_cnn"
```

The transformer launcher calls the following evaluator for each deferred checkpoint after training finishes. Its CLI is available for a separately trained deferred-test run; it rejects a run already evaluated on test:

```bash
python scripts/experiments/evaluate_transformer_checkpoint.py --help
```

Every trainer and launcher also supports `--help` for individual options. Checkpoints are selected by validation AP and neural classification thresholds by validation F1; the blend configuration uses validation F1 for both weight and threshold selection.

</details>

<details>
<summary>Optional W&B tracking and parallel transformer execution</summary>

Local CSV/JSON files and plots are always produced. The published reproduction used disabled W&B tracking. To add offline W&B logs to the five-seed CNN and transformer studies in a new workflow:

```bash
python -m pip install -r requirements-wandb.txt
python scripts/experiments/reproduce.py --stage all --device auto \
  --wandb_mode offline --output_dir outputs/tracked
```

The three individual CNN runs use the tracking mode in their JSON configurations, which is disabled in the committed configs. The workflow flag above changes tracking for the multi-seed studies.

For multiple GPUs, inspect their UUIDs and pass the selected identifiers with `--gpu_uuids`. The transformer runs one job per selected GPU. The default remains sequential execution on the chosen device.

```bash
nvidia-smi --query-gpu=index,uuid,name --format=csv
python scripts/experiments/run_transformer_study.py --help
```

For example, after assigning your two GPU UUIDs to shell variables `GPU_A` and `GPU_B`, start a new workflow with parallel transformer seeds. GPU parallelism works with W&B disabled:

```bash
python scripts/experiments/reproduce.py --stage all --device cuda:0 \
  --gpu_uuids "$GPU_A" "$GPU_B" --wandb_mode disabled --output_dir outputs/parallel
```

Offline logging does not upload runs. Optional cloud use is a separate action: `wandb login`, a trainer's `--wandb_mode online`, or `wandb sync` with an explicit offline-run directory.

</details>

## Optional: rebuild data from its sources

**Skip this section when using the included processed datasets.** Raw light curves and interim caches are large and excluded from Git. Source catalog updates and data availability can change a rebuild; the committed processed artifacts and checksums define the fixed experiments above.

Raw acquisition and preprocessing may take hours, depending on archive availability, network speed, existing caches, and hardware. This is a planning estimate; the included processed datasets let you skip these steps.

The following commands use a separate rebuild directory, preserving the published processed files:

```bash
export REBUILD_ROOT="$PROJECT_ROOT/outputs/data_rebuild"
python scripts/data/kepler_download_raw.py --base_dir "$REBUILD_ROOT" --target_n 10000
python scripts/data/kepler_prepare_features.py \
  --base_dir "$REBUILD_ROOT" --target_n 10000 --seed 42 \
  --global_bins 200 --local_bins 100 --global_window 0.5 --local_window_mult 4.0 \
  --detrend_window_days 1.5 --sigma_clip_sigma 5.0 --prepare_workers 8 --use_bls

python scripts/data/kepler_prepare_aux_v1.py \
  --dataset "$REBUILD_ROOT/data/processed/kepler/v0/dataset_kepler_v0_target10000_g200_l100_gw0.5_lwm4.0_bls1_seed42.npz" \
  --splits "$REBUILD_ROOT/data/processed/kepler/v0/splits_kepler_v0_seed42.npz" \
  --manifest "$REBUILD_ROOT/data/processed/kepler/v0/manifest_kepler_v0.csv" \
  --metadata "$REBUILD_ROOT/data/raw/kepler/metadata/koi_cumulative.csv" \
  --output_dir "$REBUILD_ROOT/data/processed/kepler/v1" \
  --audit_dir "$REBUILD_ROOT/outputs/metadata_audit"
```

Raw caches are written under `$REBUILD_ROOT/data/raw/kepler/`; folded/BLS/view caches under `$REBUILD_ROOT/data/interim/kepler/v0/`. The metadata builder fits transformations on training rows and preserves the base candidates and split.

To rebuild candidate caches while reusing downloaded raw light curves:

```bash
python scripts/data/kepler_prepare_features.py \
  --base_dir "$REBUILD_ROOT" --target_n 10000 --seed 42 \
  --global_bins 200 --local_bins 100 --prepare_workers 8 --use_bls \
  --force_recompute_candidate_cache
```

To audit metadata against the published v0 without writing another dataset:

```bash
python scripts/data/kepler_prepare_aux_v1.py \
  --dataset "$PROJECT_ROOT/data/processed/kepler/v0/dataset_kepler_v0_target10000_g200_l100_gw0.5_lwm4.0_bls1_seed42.npz" \
  --splits "$PROJECT_ROOT/data/processed/kepler/v0/splits_kepler_v0_seed42.npz" \
  --manifest "$PROJECT_ROOT/data/processed/kepler/v0/manifest_kepler_v0.csv" \
  --metadata "$REBUILD_ROOT/data/raw/kepler/metadata/koi_cumulative.csv" \
  --audit_dir "$REBUILD_ROOT/outputs/published_v0_audit" --audit_only
```

Audit/output directories must be new. Use explicit dataset, split, and manifest paths in a separate config when training on a rebuilt dataset; the committed experiment configs continue to select the published artifacts.
