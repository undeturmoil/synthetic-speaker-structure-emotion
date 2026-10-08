# Synthetic Speaker Structure Across Emotion

This repository contains code, configuration files, manifests, analysis scripts, and reproducibility materials associated with the study:

**Characterizing Emergent Synthetic Speaker Structure Across Emotion in Reference-Free Stochastic Voice Design**

## Overview

This study examines whether reproducible speaker-related structure emerges when a reference-free stochastic voice-design model generates synthetic voices under multiple emotion instructions.

The study focuses on three questions:

1. Whether reproducible candidate-related structure emerges across independently generated emotion conditions.
2. How strongly candidate-related structure is entangled with emotion.
3. Whether candidate-related structure can be recovered using statistical modeling without assuming that it corresponds directly to human-perceived speaker identity.

The term **operational candidate** refers to a reproducible generation trajectory defined by the combination of batch seed and batch position.

## Study design

The main experiments used:

- Qwen3-TTS VoiceDesign for reference-free speech generation
- Qwen3-TTS Base speaker embeddings for the primary representation
- Six generation conditions:
  - normal
  - happy
  - sad
  - calm
  - angry
  - surprise

Three candidate panels were used:

- **Development300**: 300 candidates, 1,800 utterances
- **External100**: 100 candidates, 600 utterances
- **Sealed100**: 100 candidates, 600 utterances

The Sealed100 panel was generated and evaluated only after the analysis procedure and frozen transformation had been fixed.

## Main findings

In the Development300 panel, total speaker-embedding variation was decomposed into:

- Candidate main effect: **23.52%**
- Emotion main effect: **14.62%**
- Cell-specific remainder: **61.87%**

The remainder should not be interpreted as a pure candidate-by-emotion interaction because each candidate-by-emotion cell contains only one observation and therefore also includes unseparated stochastic/error variation.

Cross-emotion candidate retrieval improved after emotion correction and multivariate modeling.

In the prospectively sealed 100-candidate replication panel, full-gallery LOEO mean reciprocal rank was:

- Raw representation: **0.1701**
- Emotion-corrected representation: **0.2078**
- Frozen multivariate representation: **0.2777**

The frozen multivariate method improved MRR relative to the emotion-corrected representation by **+0.0699**, with a 95% cluster-bootstrap confidence interval of **[+0.0201, +0.1217]**.

## Interpretation

The results support the presence of reproducible candidate-related organization in the primary synthetic-speech representation.

This structure is:

- emergent,
- graded,
- emotion-entangled,
- partially recoverable, and
- representation-dependent.

These findings concern representation-level structure and should not be interpreted as evidence that human listeners would necessarily perceive the same speaker identity across emotion conditions.

## Repository contents

The repository contains:

```text
configs/
manifests/
analysis/
artifacts/
results/
environment/
docs/
```

Current contents include:

- archived generation source and machine-readable campaign configurations
- candidate and seed manifests
- development and validation split definitions
- variance-decomposition scripts
- cross-emotion retrieval analysis
- emotion manipulation checks
- PLDA and multivariate analyses
- the frozen validation transform
- sealed replication results
- software-environment information
- reproducibility instructions

## Reproducibility

For detailed reproduction instructions and reproducibility scope, see [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md).

This repository serves as the versioned reproducibility package for the associated manuscript.

Exact model identifiers, model revisions, random seeds, analysis parameters, and frozen validation artifacts are preserved wherever available.

The sealed replication used an unchanged frozen transform and fixed hyperparameters selected before the sealed panel was evaluated.

## Data availability

Generated audio and full embedding arrays are not stored directly in this Git repository and may be distributed separately subject to licensing and archival constraints.

Analysis code, manifests, archived generation source, frozen artifacts, compact derived results, and execution-environment provenance are provided in this repository.

## Citation

Version 1.0.0 is permanently archived on Zenodo.

- Version DOI: https://doi.org/10.5281/zenodo.23228837
- Concept DOI (all versions): https://doi.org/10.5281/zenodo.23228836

For exact reproducibility of the archived v1.0.0 release, cite the version DOI.

## License

Original code and repository materials in this repository are licensed under the Apache License 2.0 unless otherwise noted. Third-party software, model weights, and externally sourced components remain subject to their respective licenses and are not relicensed by this repository. See [LICENSE](LICENSE).
