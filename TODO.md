# TODO

## Done

- Split the pipeline into separate raw-download and feature-preparation scripts.
- Kept the raw cache reusable and separate from preprocessing iterations.
- Added versioned interim preprocessing caches under `data/interim/kepler/v0/`.
- Improved preprocessing to be more ExoMiner-inspired:
  - detrending before folding
  - global/local transit views
  - multi-channel view outputs
  - optional BLS features
- Built the processed Kepler `v0` dataset and grouped train/val/test splits.
- Added CNN training under `training/cnn/`.
- Added HGB tabular baselines under `training/tabular/`.
- Compared multiple CNN variants and HGB baselines.
- Documented the experiment history in [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md).
- Reworked the public [`README.md`](README.md) so it is general and clone-friendly.
- Documented the fast-start workflow where users can train directly from committed `data/processed/kepler/` artifacts.
- Added a private local-notes area under `forme/` and ignored it in git.

## Next

- Freeze the current preprocessing and main train/test split for practical work.
- Push the processed dataset artifacts so new users can skip the expensive raw and feature stages.
- Use the best plain no-wide CNN as the main model.
- Use the full-feature HGB as the main tabular baseline.
- Start turning the experiment log into practical-work report sections.
- Add a short results table and main conclusions to the report draft.

## Nice to have

- Try a simple CNN + HGB blend on validation/test predictions.
- Add qualitative error analysis from `val_predictions.csv` and `test_predictions.csv`.
- Add plots for:
  - training history
  - PR curves / ROC curves
  - confusion matrices
- Compare a few more random seeds if time allows.

## Later / Thesis ideas

- Add richer ExoMiner-style auxiliary inputs.
- Try a `v1` preprocessing version with more views or channels.
- Add stacking / blending properly.
- Evaluate on additional splits or repeated seeds.
- Add a TESS preprocessing pipeline that writes the same processed array schema as Kepler.
- Generalize manifests and split logic from `kepid`/`koi_*` fields to mission-agnostic identifiers and metadata.
- Reuse the transformer architecture for TESS and compare against the Kepler transformer/CNN baselines.
- Build a joint Kepler+TESS processed dataset and test mixed-mission training.
- Try mission-aware transformer inputs, for example a mission token or mission scalar feature.
- Compare Kepler-only, TESS-only, joint training, and Kepler-to-TESS transfer learning.
