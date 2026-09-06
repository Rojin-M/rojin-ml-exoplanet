# Tabular baselines

| Model | Accuracy | F1 | AP | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| HGB scalar | 0.7998 | 0.7530 | 0.7782 | 0.8772 |
| HGB full | 0.8938 | 0.8581 | 0.9225 | 0.9598 |

Both models use the same saved candidate pool and star-grouped split. The full input includes flattened global/local views and scalars. The full HGB configuration was selected on validation from the recorded regularization search. Each entry is one seed.

[Metrics](comparison.csv) · [Figure](figures/scalar_vs_full.png) · [PDF](figures/scalar_vs_full.pdf)
