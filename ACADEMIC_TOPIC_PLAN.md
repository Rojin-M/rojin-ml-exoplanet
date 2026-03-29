# Academic Topic Plan

## Honest mapping

Your current repo is:

- `AstroNet-style` in preprocessing
- `ExoMiner-inspired` in overall vetting style
- `not` a direct implementation of `ExoMiner` or `ExoMiner++`

Why:

- `scripts/kepler_pipeline_lib.py` builds folded `global/local` transit views, which is closest to AstroNet-style multi-view preprocessing.
- `training/cnn/train_kepler_cnn.py` uses a multi-branch CNN with scalar features, which is closer in spirit to ExoMiner-style vetting than to a simple one-branch classifier.
- The repo does **not** yet include the parts that would make it truly `ExoMiner++-style`: `TESS`, transfer learning, and mission-aware cross-mission vetting.

## Seminar

### Best topic

`Modern Exoplanet Vetting from ExoMiner to ExoMiner++`

### Best paper

- `ExoMiner++ 2.0: Vetting TESS Full-Frame Image Transit Signals`
  - https://arxiv.org/abs/2601.14877

Why:

- It is the newest ExoMiner-family paper.
- It gives you the cleanest modern seminar anchor.

## Practical Work

### Best topic

`A Modernized ExoMiner-Inspired Exoplanet Vetting Pipeline with AstroNet-Style Multi-View Preprocessing`

### Best paper anchors

- preprocessing anchor:
  - Christopher J. Shallue and Andrew Vanderburg, `Identifying Exoplanets with Deep Learning`
  - https://arxiv.org/abs/1712.05044

- vetting-style anchor:
  - Hamed Valizadegan et al., `ExoMiner: A Highly Accurate and Explainable Deep Learning Classifier that Validates 301 New Exoplanets`
  - https://arxiv.org/abs/2111.10009

### Best one-line description

`The practical work implements a modernized, ExoMiner-inspired vetting pipeline whose preprocessing is most directly descended from AstroNet-style global/local transit views.`

## Thesis

### Best topic

`Toward ExoMiner++-Style Cross-Mission Vetting: Extending a Kepler Pipeline to TESS`

### Best bridge papers

- `ExoMiner++ on TESS with Transfer Learning from Kepler`
  - https://arxiv.org/abs/2502.09790

- `ExoMiner++ 2.0: Vetting TESS Full-Frame Image Transit Signals`
  - https://arxiv.org/abs/2601.14877

### What the thesis should add

- `TESS` preprocessing
- `Kepler -> TESS` transfer learning
- mission-aware training
- richer diagnostic inputs
- stronger cross-mission vetting

### Where transformers fit

Transformers can be added on top as an extra thesis experiment, but they are **not** the core thing that makes the work `ExoMiner++-style`.

Optional transformer paper:

- `Exoplanet Transit Candidate Identification in TESS Full-Frame Images via a Transformer-Based Algorithm`
  - https://arxiv.org/abs/2502.07542
