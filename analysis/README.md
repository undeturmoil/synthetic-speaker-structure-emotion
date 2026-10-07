# Analysis

This directory contains analysis code associated with the reported study results.

## Directory structure

```text
analysis/
├── README.md
├── verify_reported_results.py
└── source/
    ├── SHA256SUMS
    └── *.py
```

## As-run source analysis code

The Python files under:

```text
analysis/source/
```

are preserved copies of the analysis scripts used in the original study workspace.

These files are retained as provenance-preserving **as-run source code**. They have not been rewritten into a repository-portable form, and some therefore retain paths or assumptions from the original research environment.

Their SHA-256 checksums are recorded in:

```text
analysis/source/SHA256SUMS
```

The checksum file can be verified from the repository root with:

```bash
sha256sum -c analysis/source/SHA256SUMS
```

The archived source scripts cover the principal analyses included in the manuscript:

- Development300 candidate/emotion variance decomposition
- Development300 multivariate cross-validation
- Original External100 frozen-transform evaluation
- Original External100 inference
- Sealed100 inference
- emotion-wise retrieval analysis
- independent WavLM speaker-representation analysis
- emotion2vec extraction and manipulation checks
- acoustic manipulation checks

## Reported-result audit

The repository-level audit script is:

```text
analysis/verify_reported_results.py
```

Run it from the repository root:

```bash
python3 analysis/verify_reported_results.py
```

The script reads the archived result files under `results/` and checks that the values reported in the manuscript are compatible with the stored numerical results at their reported rounding precision.

The current audit covers 29 manuscript values, including:

- primary Development300 candidate/emotion variance shares
- secondary batch × position × emotion decomposition
- Original External100 raw, corrected, and frozen-multivariate retrieval
- Sealed100 full-gallery retrieval
- the primary Sealed100 multivariate-versus-corrected hypothesis test
- WavLM independent-representation results
- emotion2vec manipulation checks
- acoustic manipulation checks

A successful audit ends with:

```text
Checks passed: 29/29
ALL REPORTED RESULT CHECKS: PASS
```

## Important distinction

`analysis/source/` and `analysis/verify_reported_results.py` serve different purposes.

`analysis/source/` preserves the code used during the original study.

`analysis/verify_reported_results.py` is a repository audit utility created to verify that the manuscript's reported rounded values agree with the archived study results.

The audit utility should therefore not be interpreted as the original analysis implementation.

## Data dependencies

Several original analysis scripts depend on generated waveform data, speaker embeddings, model outputs, or workspace paths that are not all redistributed in this repository.

The repository instead preserves:

- study manifests and candidate definitions
- the frozen validation transform
- principal numerical result files
- as-run analysis source code
- validation and provenance checks

This separation avoids redistributing large model caches, model weights, generated waveform corpora, and unnecessary intermediate arrays while retaining the principal evidence required to audit the reported findings.
