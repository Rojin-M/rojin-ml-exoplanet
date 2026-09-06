# Transformer normalization and metadata

The two-seed screen gave mean validation AP 0.8548 with the original input normalization and 0.9445 with its bypass. The final metadata study produced:

| Test metric | v0 mean ± SD | v1 mean ± SD | Seeds improved |
| --- | ---: | ---: | ---: |
| Accuracy | 0.8741 ± 0.0074 | 0.9226 ± 0.0073 | 5/5 |
| F1 | 0.8386 ± 0.0078 | 0.8962 ± 0.0087 | 5/5 |
| Average precision | 0.8859 ± 0.0093 | 0.9606 ± 0.0025 | 5/5 |
| ROC-AUC | 0.9436 ± 0.0046 | 0.9785 ± 0.0016 | 5/5 |

Adding metadata and its branch changed mean test AP from 0.8859 to 0.9606 (+7.47 percentage points). The paired counts above show how consistently each metric improved across seeds.

- [Validation-only normalization screen](normalization_screen.csv) and [plot](figures/normalization_screen.png)
- [v0/v1 summary](comparison_metrics.csv), [all 12 training runs](per_run_metrics.csv), and [paired final metrics](paired_seed_metrics.csv)
- [Metadata plot](figures/v0_vs_v1.png) and [CNN/transformer context](figures/cnn_transformer_context.png)
- Seed-42 learning curves: [v0](figures/learning_curves_v0_seed42.png), [v1](figures/learning_curves_v1_seed42.png)

Selected normalization: `bypass_input_norm`, using two-seed mean validation AP. The screen CSV deliberately contains no test columns. Its two selected runs also belong to the final five-seed v0 group; their later test results appear in the full run table. Final test evaluation occurred after all 12 training runs finished. The final metadata comparison includes all five seeds per variant.

Learning curves always use seed 42. Training metrics use evolving weights/dropout; validation uses evaluation mode. Curve F1 thresholds are tuned per split/epoch. Every PNG has a matching PDF.

AP is scikit-learn average precision. Five-seed error bars are sample standard deviations of training runs on the same previously consulted Kepler split; they are not confidence intervals. Metadata comparisons change both inputs and model capacity. Model families use different training recipes and precision modes, so their comparison is descriptive. These experiments do not measure TESS transfer.
