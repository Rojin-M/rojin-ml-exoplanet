# From tabular baselines to metadata-assisted transit classification

This project classifies known Kepler transit candidates as confirmed planets or false positives. The experimental question is how much useful information comes from the light-curve representation, the model, and auxiliary catalog measurements.

The [README](README.md) gives the executable workflow. The [published results](results/README.md) and [complete model comparison](results/comparison/COMPARISON.md) contain the latest completed reproduction. Sections 2–8 describe the historical development, including 37 historical/study runs and an earlier isolated reproduction whose artifacts are archived separately. Section 9 reports the latest runs; historical scores are retained to show what repeated and what changed. Classification metrics were checked against saved predictions. AP means scikit-learn average precision, not trapezoidal area under a precision–recall curve.

CNN v1 remains the main practical model: it combines strong five-seed results with a smaller model. The latest transformer v1 has a tiny mean AP lead, while CNN v1 has higher mean accuracy, F1, and ROC-AUC. Both metadata extensions improve all four metrics in every matched seed.

## 1. Make the data reusable and define the evaluation

Raw light curves are cached per star. Preprocessing detrends and folds each candidate using its catalog ephemeris, constructs global and local views, and saves reusable candidate caches. Separating downloading from feature preparation made it possible to iterate without fetching the light curves again. A BLS duration/period constraint was corrected during development; candidate preparation was also parallelized. These are engineering improvements, not measured classification gains: no controlled before/after performance experiment was preserved for them.

The published v0 contains 5,518 candidates around 4,929 stars: 2,128 confirmed planets and 3,390 false positives. Candidates around the same star stay together in the saved split.

| Split | Candidates | Confirmed planets | Stars |
| --- | ---: | ---: | ---: |
| Training | 3,853 | 1,505 | 3,449 |
| Validation | 836 | 327 | 740 |
| Test | 829 | 296 | 740 |

| Input | Shape | Meaning |
| --- | --- | --- |
| `X_global` | 5,518 × 200 × 3 | Folded global view |
| `X_local` | 5,518 × 100 × 3 | Transit-centered local view |
| `X_scalar` | 5,518 × 7 | Engineered shape and BLS features |
| `X` | 5,518 × 907 | Flattened views plus scalar features |

The view channels contain mean flux, scatter, and a validity mask. Neural-model flux/scatter and scalar normalization use training rows; the mask is preserved. The feature named `feat_odd_even` compares alternating bins, **not separate odd and even transit events**.

The saved split and base arrays remain identical between v0 and v1. The models use validation data for model/threshold selection. The test split was inspected repeatedly during development, so it is a development-era holdout, not an untouched final assessment.

**Finding:** the pipeline supports comparisons on a fixed candidate pool with disjoint star groups. This reduces a clear leakage route; it does not establish that every possible source of bias has been eliminated.

## 2. Establish tabular baselines

**Question:** are a few scalar measurements enough, or does the full transit representation matter?

HistGradientBoosting (HGB) was trained on the scalar features and on the full flattened input. The full-feature workflow includes a small validation-driven regularization search; its selected configuration was the base configuration. The scalar reference used that base configuration without the search.

| Historical individual run | Test accuracy | Test F1 | Test AP | Test ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| HGB scalar | 0.7998 | 0.7530 | 0.7782 | 0.8772 |
| HGB full | 0.8938 | 0.8581 | 0.9225 | 0.9598 |

**Finding:** the full representation was substantially more useful in these runs. This is consistent with transit-shape information being lost when only the scalar summary is used. It does not identify which bins or channels matter most.

The full HGB reached training AP of about 0.999 while test AP was 0.9225, so fitting the training set almost perfectly did not remove the generalization gap. Three recorded full-HGB runs returned the same result; they are repetitions of the same seed/settings, not three independent seeds for uncertainty estimation.

Reproduce with the README's `tabular` stage and `configs/hgb_scalar.json` / `configs/hgb_full.json`.

## 3. Explore the multi-view CNN

**Question:** can separate global, local, and scalar branches learn a useful representation?

The CNN combines the branch embeddings for classification. Development explored stronger transit branches, a wide branch over flattened inputs, and additional regularization.

| Historical experiment | Test AP | Test F1 |
| --- | ---: | ---: |
| Initial CNN | 0.8813 | 0.8409 |
| Early wide-branch experiment | 0.8153 | 0.7767 |
| Stronger transit branches | 0.9259 | 0.8685 |
| Subsequent ranking-metric rerun | 0.9315 | 0.8682 |
| Added label smoothing and input noise | 0.9128 | 0.8631 |
| Plain rerun | 0.9238 | 0.8627 |
| Rerun later used for blending | 0.9266 | 0.8734 |

**Finding:** the final focused multi-view design was more promising than the initial and early wide-branch experiments. The additional regularization did not improve the recorded result. These are exploratory comparisons: several architecture details changed, so the table does not isolate one layer or prove that wide branches or regularization are generally harmful.

Some early runs do not have saved source snapshots, and an exact matching source version has not been recovered from Git. Their predictions and metrics remain valid historical evidence, but the repository does not claim to reconstruct those exact earlier architectures. The `cnn-baseline` and `cnn-ablations` stages run the preserved final architecture, its wide-branch variant, and its regularized variant. They are labeled separately in the reproduction results.

## 4. Combine CNN and HGB predictions

**Question:** do the two models make sufficiently different errors for a blend to help?

The historical blend selected its weight and classification threshold on validation F1: 0.58 CNN probability plus 0.42 HGB probability. Its test AP was **0.9415**, compared with 0.9266 and 0.9225 for its component runs; test F1 was **0.8851**.

**Finding:** the blend improved on these particular component runs. Complementary errors are a plausible explanation, but there was no multi-seed ensemble study. Three saved executions of the same blend gave the same result and are not counted as independent evidence. The implemented experiment is probability blending; a learned logistic stacking option exists in the code but has no reported experiment here.

The `blend` stage explicitly uses the reproduced CNN baseline and full HGB records. Its newly validation-selected weight can differ from the historical weight.

## 5. Measure CNN variability across seeds

**Question:** how representative is one good CNN run?

The chosen fixed CNN recipe was repeated with training seeds 42–46, keeping the saved data split unchanged. All five runs are included. Validation AP selected each checkpoint; validation F1 selected its threshold.

The original five-seed v0 mean test AP was **0.9105 ± 0.0115**, below the best earlier individual run. The SD is sample standard deviation across training seeds, not a confidence interval or uncertainty across datasets.

**Finding:** an attractive individual result is not a reliable summary of the model's typical performance. The multi-seed comparison provides a more defensible basis for evaluating the metadata extension.

Learning-curve training metrics are collected with changing weights and dropout during optimization; validation uses evaluation mode. Final saved train/validation/test metrics use the selected checkpoint. These quantities should not be confused when interpreting a train–validation gap.

## 6. Add auxiliary catalog metadata without changing the light curves

**Question:** does candidate context help beyond the existing transit representation?

The v1 builder preserves every v0 candidate, label, split, and base array. It adds 11 auxiliary inputs: five stellar measurements, two irradiation-related estimates, and four centroid measurements. Absolute sky coordinates were excluded by modeling choice. Feature definitions, units, transforms, fitted statistics, and source hashes are saved in `data/processed/kepler/v1/aux_preprocessing_v1.json`; field meanings follow the [NASA KOI column definitions](https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html).

Fixed transformations are followed by median imputation and standardization fitted only on training rows. No candidate is dropped for missing auxiliary values: 412 have at least one missing selected value, and 31 have all selected values missing before imputation. The missingness mask is stored for auditing and is not appended to model inputs.

The CNN's optional auxiliary branch adds a 64-value embedding to the existing classifier inputs. With five matched training seeds:

| Historical CNN test metric | v0 | v1 + metadata |
| --- | ---: | ---: |
| Accuracy | 0.8893 ± 0.0165 | 0.9332 ± 0.0069 |
| F1 | 0.8548 ± 0.0161 | 0.9100 ± 0.0080 |
| AP | 0.9105 ± 0.0115 | 0.9628 ± 0.0027 |
| ROC-AUC | 0.9538 ± 0.0059 | 0.9810 ± 0.0008 |

All four metrics improved in all five original matched seeds.

**Finding:** the combined metadata-and-branch extension improved this classifier consistently. Stellar context and centroid diagnostics may help distinguish planet-like signals from false positives, but the experiment does not establish which feature group is responsible. Inputs and model capacity both changed. This is catalog-assisted candidate classification, not classification from light curves alone or blind planet discovery.

## 7. Investigate the transformer

A transformer is a model architecture; it is separate from transfer learning between Kepler and TESS. This repository's transformer classifies the same Kepler candidates using global/local sequences and scalar inputs.

The original single run had test AP **0.8364**. Inspection identified an initial `LayerNorm(3)` applied across flux, scatter, and validity-mask channels. Such normalization can make differently scaled input patterns nearly indistinguishable. The experiment bypassed only this initial normalization, preserving internal encoder normalization and the remaining shared settings.

| Historical validation-only screen | Original input normalization AP | Bypassed AP |
| --- | ---: | ---: |
| Seed 42 | 0.8536 | 0.9471 |
| Seed 43 | 0.8650 | 0.9509 |
| Mean | 0.8593 | 0.9490 |

**Finding:** bypassing the input normalization improved both screening seeds. Suppression of useful amplitude information is a plausible mechanism, supported by inspecting the transformation, but the precise cause is not fully isolated. This is a two-seed validation result, not a five-seed test comparison of normalization modes.

The chosen normalization mode was frozen before the final metadata comparison. Its two screening runs were reused, three further v0 seeds were trained, and v1 used all five seeds. Test evaluation was deferred until all 12 training runs were complete. All completed controlled runs used float32; they should not be treated as an isolated comparison with the older autocast run.

| Historical transformer test metric | Selected v0 | v1 + metadata |
| --- | ---: | ---: |
| Accuracy | 0.8758 ± 0.0158 | 0.9288 ± 0.0098 |
| F1 | 0.8345 ± 0.0168 | 0.9037 ± 0.0130 |
| AP | 0.8842 ± 0.0149 | 0.9600 ± 0.0015 |
| ROC-AUC | 0.9422 ± 0.0079 | 0.9779 ± 0.0010 |

All four metrics improved in every original matched seed.

The historical `attention_dropout` argument remains recorded-only; effective attention dropout follows `dropout` (0.15 in this study). Cloned encoder initialization was also retained. Neither was changed while testing input normalization and metadata, so this study makes no claim about their effects.

## 8. Historical conclusions and the earlier isolated reproduction

The original overall comparison includes tabular baselines, blending, the original transformer, and both five-seed model families. The strongest original mean AP is CNN v1 at 0.9628; transformer v1 is close at 0.9600. The CNN has about 0.82 million parameters with metadata, versus about 2.19 million for the transformer. The different architectures use different training recipes and precision modes, so their small score difference is descriptive rather than proof of architectural superiority.

The historical conclusions were:

1. Full transit inputs were more useful than the scalar-only HGB baseline in the recorded experiments.
2. CNN development produced useful positive and negative exploratory results, but individual reruns varied enough to justify five-seed reporting.
3. A validation-tuned blend improved the particular CNN/HGB pair studied.
4. Metadata improved both neural model families consistently across the original five seeds.
5. The transformer's input representation mattered substantially in the controlled validation screen; architectural complexity alone did not ensure a strong result.

The complete frozen workflow was previously rerun from an isolated checkout containing processed data only. Those reproduction results were archived separately from the original reference. Both HGB baselines reproduced their original scores. That reproduction's CNN baseline had AP 0.9262; its wide and regularized variants had AP 0.7800 and 0.9057. Its blend reached AP 0.9445.

| Five-seed test AP | Original reference | Isolated reproduction |
| --- | ---: | ---: |
| CNN v0 | 0.9105 ± 0.0115 | 0.9144 ± 0.0130 |
| CNN v1 + metadata | 0.9628 ± 0.0027 | 0.9612 ± 0.0033 |
| Transformer v0 | 0.8842 ± 0.0149 | 0.8820 ± 0.0141 |
| Transformer v1 + metadata | 0.9600 ± 0.0015 | 0.9604 ± 0.0024 |

The reproduction again improved accuracy, F1, AP, and ROC-AUC in every matched seed for both metadata extensions. Its normalization screen improved mean validation AP from 0.8601 to 0.9363 and again selected the bypass. These are repeated training experiments on the same split, not new independent test data.

Seeded GPU execution is not bit-for-bit deterministic across environments. Exact scores and stopping epochs varied, while the main pattern survived. No hyperparameters were retuned in response to the reproduction test scores. Model/training computations and processed-file contents were preserved during repository cleanup; path handling and execution/analysis entry points changed.

## 9. Latest completed reproduction and the practical conclusion

The published export contains all 26 final evaluated runs: two HGB baselines, three individual CNN configurations, one blend, and five seeds for each neural v0/v1 variant. The two additional legacy-normalization transformer runs remain validation-only evidence. The [comparison table](results/comparison/COMPARISON.md) reports all model families, including variants that performed worse.

Both [HGB baselines](results/tabular/README.md) reproduced their original scores. The latest individual CNN baseline reached test AP 0.9226, compared with 0.8014 for the wide branch and 0.9138 for additional regularization. These final-architecture experiments again favor the baseline; they do not reconstruct every earlier exploratory architecture.

| Latest five-seed test result | Accuracy | F1 | AP | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| CNN v0 | 0.8926 ± 0.0115 | 0.8594 ± 0.0114 | 0.9139 ± 0.0138 | 0.9558 ± 0.0068 |
| CNN v1 + metadata | 0.9327 ± 0.0087 | 0.9082 ± 0.0104 | 0.9605 ± 0.0042 | 0.9802 ± 0.0015 |
| Transformer v0 | 0.8741 ± 0.0074 | 0.8386 ± 0.0078 | 0.8859 ± 0.0093 | 0.9436 ± 0.0046 |
| Transformer v1 + metadata | 0.9226 ± 0.0073 | 0.8962 ± 0.0087 | 0.9606 ± 0.0025 | 0.9785 ± 0.0016 |

Values are mean ± sample SD over seeds 42–46 on the same fixed split. In both the [CNN study](results/cnn/README.md) and [transformer study](results/transformer/README.md), metadata improved accuracy, F1, AP, and ROC-AUC in all five matched seeds. Mean AP increased by 4.66 percentage points for CNN and 7.47 points for the transformer. This repeats the strongest finding from the historical runs, while still comparing added inputs and branch capacity together.

The transformer's validation-only normalization screen again selected bypassing the initial `LayerNorm(3)`: mean validation AP increased from 0.8548 to 0.9445. The choice was frozen before test evaluation within this study. Its effect remains supported by two validation seeds, rather than a five-seed test comparison of normalization modes.

The latest [blend](results/blend/README.md) selected 0.52 CNN probability plus 0.48 HGB probability and a threshold of 0.51 on validation F1. Test AP improved to 0.9431 from the single CNN baseline's 0.9226 and HGB's 0.9225. Test F1, however, was 0.8754 versus the single CNN baseline's 0.8760. The blend found 281 of 296 planets with 65 false positives; the CNN found 265 with 44 false positives. The extra 16 detected planets came with 21 more false alarms. Validation F1 improvement therefore did not translate into test F1 improvement, even though ranking improved. This comparison uses the individual CNN component, not the five-seed CNN mean.

**CNN v1 is the main practical model.** Transformer v1 has the highest mean test AP at 0.960631 versus CNN v1's 0.960548, a difference of only 0.000083, far smaller than either model's observed seed SD. This does not establish a clear AP advantage. CNN v1 has higher mean accuracy, F1, and ROC-AUC and about 0.82 million parameters, compared with about 2.19 million for transformer v1. The comparison is descriptive because the model families use different training recipes and precision modes; the transformer remains a useful close competitor.

The published results come from repeated training on the same previously consulted holdout, not a new independent assessment. Further work can test feature groups, additional independent splits, or a new untouched dataset. TESS preparation and Kepler-to-TESS transfer are separate thesis extensions; none of the present results establishes cross-mission performance.
