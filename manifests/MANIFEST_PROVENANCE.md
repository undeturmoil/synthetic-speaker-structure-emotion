# Manifest Provenance

## Purpose

This directory preserves the candidate definitions and dataset partitioning used in the study.

An operational candidate is defined by a fixed combination of batch seed and batch position. Candidate identifiers therefore refer to reproducible stochastic generation trajectories rather than to human speaker ground truth.

## Dataset panels

The study contains three analysis panels:

| Panel | Candidates | Samples | Batch indices | Batch seeds |
|---|---:|---:|---|---|
| Development300 | 300 | 1,800 | 0–29 | 1502000–1502029 |
| External100 | 100 | 600 | 30–39 | 1502030–1502039 |
| Sealed100 | 100 | 600 | 40–49 | 1502040–1502049 |

Each candidate contains one generated utterance under each of six emotion instructions:

- normal
- happy
- sad
- calm
- angry
- surprise

The complete study therefore contains 500 operational candidates and 3,000 generated utterances.

## Source manifests

The original generation manifests are preserved without modification under:

```text
manifests/source/
```

The four source files are:

```text
study1_35f100_v1_manifest.json
study1_35f300_extension_v1_manifest.json
study1_35f100_external_v1_manifest.json
study1_35f100_sealed_v1_manifest.json
```

Their SHA-256 checksums are recorded in:

```text
manifests/source/SHA256SUMS
```

The source manifests retain generation metadata including candidate identifiers, batch seeds, batch positions, emotion labels, generation status, waveform checksums, acoustic summary values, and original workspace paths.

Original workspace paths appearing in these files are provenance records only and are not required to exist in a reproduction environment.

## Development-panel construction

Development300 combines two original generation campaigns.

The initial campaign contains:

```text
b00p00 – b09p09
100 candidates
600 samples
batch seeds 1502000 – 1502009
```

The extension campaign contains:

```text
b10p00 – b29p09
200 candidates
1,200 samples
batch seeds 1502010 – 1502029
```

Together they form the 300-candidate development panel.

## Manifest schema evolution

The original manifests were produced at different stages of the study and therefore contain minor schema differences.

The initial `study1_35f100_v1` manifest stores the global candidate number under:

```text
candidate_index
```

Later manifests store the same concept under:

```text
global_candidate_index
```

`build_public_manifests.py` supports both field names while preserving the underlying recorded values.

The sealed manifest also retains the schema label:

```text
study1-35f100-external-v1
```

while its `campaign_id` is:

```text
study1_35f100_sealed_v1
```

This historical schema label has intentionally not been altered in the archived source manifest.

## Public normalized manifests

The script:

```text
manifests/build_public_manifests.py
```

reads the archived source manifests, validates their internal consistency, and generates:

```text
manifests/development300.csv
manifests/external100.csv
manifests/sealed100.csv
manifests/manifest_build_report.json
```

The normalized CSV files contain one row per operational candidate rather than one row per generated utterance.

The build process verifies:

- 300 Development300 candidates and 1,800 samples
- 100 External100 candidates and 600 samples
- 100 Sealed100 candidates and 600 samples
- 500 candidates and 3,000 samples in total
- exactly six emotion records per candidate
- the expected six emotion labels
- successful generation status for every source record
- consistency of candidate ID, batch index, and batch position
- consistency of batch seed and batch index
- unique global candidate indices
- complete global candidate-index coverage from 0 through 499

## Development cross-validation folds

The development cross-validation structure was extracted directly from the `FOLDS` constant in the original analysis script:

```text
analysis/study1_35f300_multivariate_cv.py
```

The resulting public definition is stored in:

```text
manifests/folds.json
```

The five validation folds are:

```text
Fold 1: 0, 5, 10, 15, 20, 25
Fold 2: 1, 6, 11, 16, 21, 26
Fold 3: 2, 7, 12, 17, 22, 27
Fold 4: 3, 8, 13, 18, 23, 28
Fold 5: 4, 9, 14, 19, 24, 29
```

Each fold contains six batches and therefore 60 candidates.

Across the five folds, all 30 Development300 batches occur exactly once as a validation batch.

## Rebuilding the normalized manifests

From the repository root, run:

```bash
python3 manifests/build_public_manifests.py
```

A successful build reports:

```text
development: 300 candidates, 1800 samples, b00p00 -> b29p09 : PASS
external: 100 candidates, 600 samples, b30p00 -> b39p09 : PASS
sealed: 100 candidates, 600 samples, b40p00 -> b49p09 : PASS
TOTAL: 500 candidates, 3000 samples : PASS
```

The normalized manifests are derived files. The archived JSON files under `manifests/source/` remain the provenance-preserving source records.
