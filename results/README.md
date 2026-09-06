# Reproduced experiment results

This directory contains verified results from the completed reproduction: 27 training runs (two HGB, three individual CNN, ten CNN seed runs, twelve transformer runs), one blend, and ten deferred transformer test evaluations. The overall comparison has 26 final evaluated entries; the two unselected transformer screen runs have validation evidence only.

| Evidence | Files |
| --- | --- |
| Overall comparison | [Table](comparison/COMPARISON.md), [plot](comparison/comparison.png), [individual runs](comparison/runs.csv), [means/SDs](comparison/summary.csv) |
| Scalar versus full HGB | [Tabular results](tabular/README.md) |
| CNN variants and metadata | [CNN results](cnn/README.md) |
| Probability blending | [Blend results](blend/README.md) |
| Normalization and transformer metadata | [Transformer results](transformer/README.md) |
| Verification and source hashes | [Provenance](provenance.json) |

CNN v1 is the main practical model used in this project. Its mean Accuracy, F1, ROC-AUC exceed transformer v1 in this reproduction. Mean test AP is 0.960548 for CNN v1 and 0.960631 for transformer v1. Compare all four metrics and seed variability before interpreting a small ranking difference. The [storyline](../STORYLINE.md) explains the model choice and experimental findings.

All planned seeds and unsuccessful alternatives remain in the tables. Learning-curve illustrations use seed 42 consistently. Figures are PNG for browsing and PDF for reports. Checkpoints, predictions, and detailed logs remain in local outputs/. Paths in CSV/JSON refer to the source workflow relative to the repository; timestamped run IDs preserve traceability.

To reproduce this export after training, run from the repository root:

```bash
python scripts/analysis/export_results.py --workflow_dir outputs/reproduce --output_dir outputs/published_results
```

The destination must be new or empty. For an initial publication, use `--output_dir results`. The exporter checks processed hashes, saved study evidence, and validation/test predictions, then writes derived tables and plots. It does not train models or change the source runs. New training runs can produce slightly different scores.

AP is scikit-learn average precision. Five-seed error bars are sample standard deviations of training runs on the same previously consulted Kepler split; they are not confidence intervals. Metadata comparisons change both inputs and model capacity. Model families use different training recipes and precision modes, so their comparison is descriptive. These experiments do not measure TESS transfer.
