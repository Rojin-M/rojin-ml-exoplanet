# CNN and HGB probability blend

| Model | Accuracy | F1 | AP | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| CNN baseline | 0.9095 | 0.8760 | 0.9226 | 0.9605 |
| HGB full | 0.8938 | 0.8581 | 0.9225 | 0.9598 |
| CNN + HGB blend | 0.9035 | 0.8754 | 0.9431 | 0.9701 |

| Model | True positives | False positives | False negatives | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| CNN baseline | 265 | 44 | 31 | 0.8576 | 0.8953 |
| HGB full | 266 | 58 | 30 | 0.8210 | 0.8986 |
| CNN + HGB blend | 281 | 65 | 15 | 0.8121 | 0.9493 |

Against the individual CNN component, test AP improved (0.9226 → 0.9431) and test F1 decreased (0.8760 → 0.8754). Validation-selected blending does not guarantee improvement on every test metric.

Validation F1 selected weights 0.52 CNN / 0.48 HGB and threshold 0.510. Higher recall can come with more false positives; inspect AP and F1 separately. The CNN component is the individual baseline, not the five-seed mean or CNN v1.

[Full metrics](comparison.csv) · [Selection](selection.json) · [Figure](figures/metrics.png) · [PDF](figures/metrics.pdf)
