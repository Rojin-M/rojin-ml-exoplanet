# CNN experiments

| Model | Accuracy | F1 | AP | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| CNN baseline | 0.9095 | 0.8760 | 0.9226 | 0.9605 |
| CNN wide branch | 0.8263 | 0.7729 | 0.8014 | 0.8925 |
| CNN regularized | 0.9059 | 0.8669 | 0.9138 | 0.9568 |

The three individual runs compare the baseline CNN, wide-branch variant, and regularized variant. The metadata comparison separately includes all five seeds for both v0 and v1.

| Test metric | v0 mean ± SD | v1 mean ± SD | Seeds improved |
| --- | ---: | ---: | ---: |
| Accuracy | 0.8926 ± 0.0115 | 0.9327 ± 0.0087 | 5/5 |
| F1 | 0.8594 ± 0.0114 | 0.9082 ± 0.0104 | 5/5 |
| Average precision | 0.9139 ± 0.0138 | 0.9605 ± 0.0042 | 5/5 |
| ROC-AUC | 0.9558 ± 0.0068 | 0.9802 ± 0.0015 | 5/5 |

Adding metadata and its branch changed mean test AP from 0.9139 to 0.9605 (+4.66 percentage points). The paired counts above show how consistently each metric improved across seeds.

- [Ablation metrics](ablations.csv) and [plot](figures/ablations.png)
- [v0/v1 summary](comparison_metrics.csv), [all seed metrics](per_seed_metrics.csv), and [paired metrics](paired_seed_metrics.csv)
- [v0/v1 plot](figures/v0_vs_v1.png) and [paired differences](figures/paired_differences.png)
- Seed-42 learning curves: [v0](figures/learning_curves_v0_seed42.png), [v1](figures/learning_curves_v1_seed42.png)

Seed 42 illustrates training behavior; it was chosen consistently, without selecting the best test score. Training curves use evolving weights/dropout, while validation uses evaluation mode. Learning-curve F1 thresholds are tuned separately per split/epoch; final test thresholds come from validation. Every PNG has a PDF with the same filename.

AP is scikit-learn average precision. Five-seed error bars are sample standard deviations of training runs on the same previously consulted Kepler split; they are not confidence intervals. Metadata comparisons change both inputs and model capacity. Model families use different training recipes and precision modes, so their comparison is descriptive. These experiments do not measure TESS transfer.
