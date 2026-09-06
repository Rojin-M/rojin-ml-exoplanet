# Model comparison

AP is average precision. Values are test scores on the same previously consulted Kepler split.
Five-seed entries report mean ± sample SD; individual runs have no seed uncertainty estimate.

| Model | Evidence | Runs | Accuracy | F1 | AP | ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| HGB scalar | individual run | 1 | 0.7998 | 0.7530 | 0.7782 | 0.8772 |
| HGB full | individual run | 1 | 0.8938 | 0.8581 | 0.9225 | 0.9598 |
| CNN baseline | individual run | 1 | 0.9095 | 0.8760 | 0.9226 | 0.9605 |
| CNN wide branch | individual run | 1 | 0.8263 | 0.7729 | 0.8014 | 0.8925 |
| CNN regularized | individual run | 1 | 0.9059 | 0.8669 | 0.9138 | 0.9568 |
| CNN + HGB blend | individual run | 1 | 0.9035 | 0.8754 | 0.9431 | 0.9701 |
| CNN v0 | five matched seeds | 5 | 0.8926 ± 0.0115 | 0.8594 ± 0.0114 | 0.9139 ± 0.0138 | 0.9558 ± 0.0068 |
| CNN v1 + metadata | five matched seeds | 5 | 0.9327 ± 0.0087 | 0.9082 ± 0.0104 | 0.9605 ± 0.0042 | 0.9802 ± 0.0015 |
| Transformer v0 | five matched seeds | 5 | 0.8741 ± 0.0074 | 0.8386 ± 0.0078 | 0.8859 ± 0.0093 | 0.9436 ± 0.0046 |
| Transformer v1 + metadata | five matched seeds | 5 | 0.9226 ± 0.0073 | 0.8962 ± 0.0087 | 0.9606 ± 0.0025 | 0.9785 ± 0.0016 |

Architecture, inputs, training recipes, and numerical precision can differ between model families. This is a descriptive comparison.
The paired v0/v1 studies provide stronger evidence about their combined metadata-and-branch extension; they do not isolate individual features or added capacity.
Repeated runs on this split do not create an untouched test set. Sample SD measures training-seed variability, not uncertainty on a new mission.
