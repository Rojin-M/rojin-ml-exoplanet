# Supervisor Recommendations

This file tracks the next improvements suggested after the current Kepler practical-work baseline.

Supervisor reference:

- Weights & Biases: https://wandb.ai/site/

## Priority TODOs

- Add optional Weights & Biases logging to `training/cnn/train_kepler_cnn.py`.
- Log train/validation loss curves to W&B.
- Log train/validation PR-AUC, ROC-AUC, F1, accuracy, and learning rate to W&B.
- Log final test metrics and the best validation threshold to W&B.
- Add local plot generation from `history.csv` for practical-work figures.
- Plot train/validation loss curves.
- Plot train/validation PR-AUC and F1 curves.
- Add multi-seed CNN validation.
- Run the main CNN with several seeds, for example `42, 43, 44, 45, 46`.
- Aggregate mean and standard deviation across seeds for accuracy, F1, PR-AUC, and ROC-AUC.
- Add an error-bar summary plot for multi-seed results.

## ExoMiner-Style Feature TODOs

- Add a `v1` preprocessing version for auxiliary features.
- Keep raw downloads untouched when moving from `v0` to `v1`.
- Add stellar-parameter features from KOI metadata, for example:
  - `koi_steff`
  - `koi_slogg`
  - `koi_smet`
  - `koi_srad`
  - `koi_smass`
  - `koi_teq`
  - `koi_insol`
- Add centroid-like KOI metadata features, for example:
  - `koi_fwm_sra`
  - `koi_fwm_sdec`
  - `koi_fwm_srao`
  - `koi_fwm_sdeco`
  - `koi_dicco_msky`
  - `koi_dikco_msky`
- Save these features as a new processed array such as `X_aux`.
- Add an auxiliary dense branch to the CNN for `X_aux`.
- Compare `v0` CNN versus `v1` CNN with auxiliary stellar/centroid metadata.

## Thesis-Sized TODOs

- Consider full centroid or difference-image diagnostics as thesis work if the metadata-only centroid branch is not enough.
- Consider richer ExoMiner++-style inputs after the practical-work baseline is stable.
- Treat true TESS support and Kepler-to-TESS transfer learning as thesis-level extensions.
