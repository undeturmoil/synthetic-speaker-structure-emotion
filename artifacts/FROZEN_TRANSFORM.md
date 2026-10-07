# Frozen Validation Transform

## Purpose

`frozen_transform.npz` contains the frozen transformation used for the external and sealed cross-emotion candidate-retrieval analyses reported in the associated manuscript.

The transformation was fixed before evaluation of the sealed 100-candidate replication panel. The sealed panel was not used to refit, retune, or modify this artifact.

## Origin

The artifact was copied from the study workspace:

```text
campaigns/study1_35f100_external_v1/frozen_validation/frozen_transform.npz
```

The copy included in this repository is byte-identical to the study artifact.

## SHA-256

```text
4f7b1d3c85e728ad6febd9288bdc4348a48f3e450a1cebc92d6314b509b8da3b
```

The checksum is also stored in:

```text
artifacts/frozen_transform.sha256
```

It can be verified from the repository root with:

```bash
cd artifacts
sha256sum -c frozen_transform.sha256
cd ..
```

## Contents

The NumPy archive contains the following arrays:

| Key | Shape | Data type | Role |
|---|---:|---|---|
| `emotion_effect` | `(6, 2048)` | `float32` | Frozen emotion-effect estimates for the six generation conditions |
| `grand` | `(2048,)` | `float32` | Frozen grand-mean vector |
| `basis` | `(2048, 128)` | `float32` | Frozen 128-dimensional multivariate projection basis |
| `gamma` | `(1,)` | `float64` | Frozen regularization/scaling hyperparameter (`0.1`) |
| `dimension` | `(1,)` | `int64` | Frozen projection dimensionality (`128`) |

The archive size is 1,027,879 bytes.

## Validation chronology

The relevant analysis chronology was:

```text
Development300
    |
    v
Analysis development and model selection
    |
    v
Original External100 evaluation
    |
    v
Transformation frozen
    |
    v
SHA-256 recorded
    |
    v
Sealed100 evaluation using the unchanged transform
```

The sealed replication therefore evaluated an already-fixed transformation rather than selecting or tuning a transformation against the sealed data.

## Interpretation

This artifact supports reproducibility of the frozen transformation used in the confirmatory analysis. Its checksum establishes the identity and integrity of the archived binary file.

The checksum alone should not be interpreted as independent proof of the historical freezing date. The freezing chronology is established by the study records and analysis workflow; the checksum provides a stable identifier for the exact artifact used in that workflow.
