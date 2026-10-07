# Reproduction Guide

This repository preserves the provenance needed to audit the reported Study 1 results and to reconstruct the main generation, embedding-extraction, and analysis pipeline.

The repository intentionally distinguishes three different meanings of "reproduction":

1. **repository-only audit** — checks archived source integrity, manifests, and reported numerical results;
2. **derived-data analysis reproduction** — reruns the original analyses when the generated audio and derived embedding arrays are available;
3. **end-to-end regeneration** — regenerates the synthetic speech corpus from the frozen VoiceDesign configuration and then repeats embedding extraction and analysis.

These levels should not be treated as equivalent.

---

## 1. Repository-only audit

This level requires only a clone of the repository.

From the repository root:

```bash
sha256sum -c configs/source/SHA256SUMS
sha256sum -c analysis/source/SHA256SUMS
python3 manifests/build_public_manifests.py
python3 analysis/verify_reported_results.py
```

A successful reported-result audit ends with:

```text
Checks passed: 29/29
ALL REPORTED RESULT CHECKS: PASS
```

This verifies that:

- the archived generation and analysis source files match their recorded SHA-256 checksums;
- the normalized public manifests rebuild consistently from the archived source manifests;
- the archived compact result files reproduce the manuscript-level numerical values checked by the audit script.

The audit script is a repository verification utility. It is not the original analysis implementation.

---

## 2. Study panels

The study contains three analysis panels:

| Panel | Candidates | Emotion conditions | Utterances | Batch indices | Batch seeds |
| --- | ---: | ---: | ---: | --- | --- |
| Development300 | 300 | 6 | 1,800 | 0-29 | 1502000-1502029 |
| External100 | 100 | 6 | 600 | 30-39 | 1502030-1502039 |
| Sealed100 | 100 | 6 | 600 | 40-49 | 1502040-1502049 |

The six emotion conditions are:

```text
normal
happy
sad
calm
angry
surprise
```

The normalized candidate-level manifests are:

```text
manifests/development300.csv
manifests/external100.csv
manifests/sealed100.csv
```

The original generation manifests are preserved without modification under:

```text
manifests/source/
```

---

## 3. Operational candidate definition

A candidate is an **RNG batch-slot pairing**, not human speaker ground truth.

For a given batch:

- the batch size is 10;
- one batch seed is assigned;
- the same batch seed is reset immediately before generating each emotion-specific batch;
- output position 0-9 within that batch defines the operational candidate slot.

The candidate identifier therefore represents:

```text
batch seed x fixed batch position
```

For example, candidate `b00p00` refers to position 0 from batch 0.

This design provides a reproducible pairing rule across emotion conditions. It does **not** imply that a human-identifiable speaker identity is deterministically guaranteed across different emotion prompts.

The archived public manifest builder verifies the recorded relationship among candidate ID, batch index, batch seed, batch position, and global candidate index.

---

## 4. Frozen VoiceDesign generation contract

The original VoiceDesign model was:

```text
Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
revision:
5ecdb67327fd37bb2e042aab12ff7391903235d3
```

The archived initial generation source is:

```text
configs/source/study1_35f100_generate.py
```

The Development300 extension, External100, and Sealed100 scripts import the frozen generation contract directly from that initial script:

```text
configs/source/study1_35f300_extension_generate.py
configs/source/study1_35f100_external_generate.py
configs/source/study1_35f100_sealed_generate.py
```

The frozen study configuration uses:

```text
target age:              35
design age:              50
gender:                  female
language:                Korean
batch size:              10
non_streaming_mode:      true
do_sample:               true
temperature:             0.9
top_p:                   1.0
top_k:                   50
repetition_penalty:      1.05
subtalker_dosample:      true
subtalker_temperature:   0.9
subtalker_top_p:         1.0
subtalker_top_k:         50
max_new_tokens:          2048
attention implementation: sdpa
model dtype:             bfloat16
device:                  cuda:0
```

The exact synthesis text, base voice instruction, six emotion instructions, tail instruction, seed lists, generation parameters, campaign metadata, sample records, waveform checksums, and configuration hashes are preserved in the original source manifests under:

```text
manifests/source/
```

Those manifests are the authoritative machine-readable campaign configuration records.

No separately normalized generation JSON is maintained, in order to avoid creating a second configuration source that could diverge from the original manifests.

---

## 5. Generated waveform contract

Each generation call produces a batch of 10 waveforms for one batch seed and one emotion condition.

The generation scripts require:

```text
sample rate: 24000 Hz
outputs per call: 10
```

Generation is repeated across the six emotion conditions while resetting the same batch seed before each corresponding 10-output batch.

The generated WAV corpus itself is not stored directly in this Git repository.

The archived source manifests retain recorded waveform paths and waveform checksums as provenance.

---

## 6. Qwen3-TTS Base speaker embedding extraction

The primary representation model was:

```text
Qwen/Qwen3-TTS-12Hz-1.7B-Base
revision:
fd4b254389122332181a7c3db7f27e918eec64e3
```

The archived embedding source files are:

```text
analysis/source/study1_35f100_baseline.py
analysis/source/study1_35f300_build_embeddings.py
analysis/source/study1_35f100_external_embeddings.py
analysis/source/study1_35f100_sealed_embeddings.py
```

The extraction contract is:

```text
audio loader:            soundfile
audio dtype:             float32
required sample rate:    24000 Hz
required channel shape:  mono
resampling:              none
model dtype:             bfloat16
device:                  cuda:0
attention implementation: sdpa
local model loading:     true
embedding API:           model.model.extract_speaker_embedding(audio=audio, sr=sr)
embedding dimension:     2048
stored embedding dtype:  float32
```

The resulting analysis panels have shapes:

```text
Development300: (300, 6, 2048)
External100:    (100, 6, 2048)
Sealed100:      (100, 6, 2048)
```

Development300 is assembled in two stages:

1. the initial 100-candidate panel is extracted by `study1_35f100_baseline.py`;
2. the 200-candidate extension is extracted by `study1_35f300_build_embeddings.py` and concatenated with the initial 100 candidates.

The full embedding arrays are not currently stored directly in this Git repository.

---

## 7. Development analysis

The archived Development300 analysis source includes:

```text
analysis/source/study1_35f300_variance_decomposition.py
analysis/source/study1_35f300_multivariate_cv.py
analysis/source/study1_35f_emotionwise_retrieval_v2.py
```

The primary representation decomposition separates total variation into:

- candidate main effect;
- emotion main effect;
- candidate-by-emotion plus remaining unseparated variation.

Because there is one observation per candidate-by-emotion cell, the final component must not be interpreted as a pure interaction term.

The Development300 cross-validation structure is archived in:

```text
manifests/folds.json
```

The five validation folds are based on interleaved generation batches.

---

## 8. Frozen transform and External100 validation

The frozen multivariate transform is archived under:

```text
artifacts/frozen_transform.npz
artifacts/frozen_transform.sha256
artifacts/FROZEN_TRANSFORM.md
```

The original External100 analysis source includes:

```text
analysis/source/study1_35f100_external_evaluate.py
analysis/source/study1_35f100_external_inference.py
```

The frozen multivariate configuration uses:

```text
gamma:      0.1
dimension:  128
```

External100 is independent of Development300 with respect to its candidate batch seeds and candidate slots.

---

## 9. Sealed100 replication

The Sealed100 generation script is archived at:

```text
configs/source/study1_35f100_sealed_generate.py
```

The Sealed100 embedding extraction source is:

```text
analysis/source/study1_35f100_sealed_embeddings.py
```

The Sealed100 inference source is:

```text
analysis/source/study1_35f100_sealed_inference.py
```

The sealed panel uses new batch indices 40-49 and new seeds 1502040-1502049.

The previously frozen analysis procedure and transform are applied without using Sealed100 to select or tune those settings.

---

## 10. Independent WavLM validation

The independent WavLM speaker-representation analysis is archived at:

```text
analysis/source/study1_35f100_wavlm_independent_validation.py
```

Its local helper dependencies are also preserved:

```text
analysis/source/study1_35f100_bundle_recovery.py
analysis/source/study1_35f_plda_baseline.py
```

The WavLM model identifier used by the archived analysis is:

```text
microsoft/wavlm-base-plus-sv
```

This analysis requires access to the generated waveform corpus.

---

## 11. Emotion and acoustic manipulation checks

The archived manipulation-check source includes:

```text
analysis/source/study1_emotion2vec_extract_2400.py
analysis/source/study1_manipulation_check_analyze.py
analysis/source/study1_acoustic_manipulation_check.py
```

The emotion2vec extraction source records:

```text
iic/emotion2vec_plus_large
```

The original emotion-check runtime environment is documented separately under:

```text
environment/original_dgx_spark/
```

---

## 12. Original execution environment

The original DGX Spark environment is preserved as provenance under:

```text
environment/original_dgx_spark/
```

This directory includes:

- original Dockerfile sources;
- package requirement snapshots;
- pip-freeze snapshots;
- Docker inspect metadata;
- model revision records;
- SHA-256 provenance files.

The environment record is intended to document the software stack used for the reported experiments.

It is **not** a claim that a different CPU architecture, CUDA stack, PyTorch build, GPU, or future model implementation will reproduce every floating-point value or waveform bit-for-bit.

---

## 13. Derived-data analysis reproduction

The original analysis scripts under:

```text
analysis/source/
```

are preserved as-run.

They intentionally retain the original `/factory` workspace layout and original local paths.

To rerun them directly, the required generated WAV files, embedding arrays, and intermediate artifacts must be restored to the expected locations or the scripts must be adapted to a new workspace.

Important derived inputs include:

```text
/factory/campaigns/study1_35f300_v1/analysis/embeddings_300x6x2048.npy

/factory/campaigns/study1_35f100_external_v1/analysis/embeddings_100x6x2048.npy

/factory/campaigns/study1_35f100_sealed_v1/analysis/embeddings_100x6x2048.npy
```

The Git repository therefore preserves the **analysis implementation and provenance**, while the large derived arrays and generated waveform corpus must be supplied separately for full reruns.

---

## 14. End-to-end regeneration

The repository preserves the major executable contracts required for end-to-end reconstruction:

```text
VoiceDesign generation source
        ->
source manifests
        ->
generated 24 kHz mono WAVs
        ->
Qwen3-TTS Base speaker embedding extraction
        ->
Development300 / External100 / Sealed100 embedding panels
        ->
archived statistical analyses
        ->
frozen transform and confirmatory validation
        ->
reported-result audit
```

A fresh Git clone alone is **not** sufficient to reproduce the original waveform corpus because the repository does not redistribute model weights, Hugging Face caches, or the generated WAV files.

Full regeneration additionally requires access to the recorded model revisions and a compatible runtime capable of executing the archived generation and embedding source.

Even with the same model revision and seed schedule, exact waveform identity across materially different hardware or software stacks should not be assumed unless empirically verified.

---

## 15. What this repository does and does not claim

The repository supports direct verification that the archived compact results agree with the manuscript-level reported values covered by the audit.

It also preserves:

- candidate and panel definitions;
- original generation manifests;
- frozen generation source;
- frozen model revisions;
- speaker-embedding extraction source;
- archived statistical analysis source;
- the frozen multivariate transform;
- the original execution environment provenance;
- cryptographic hashes for archived source files.

The repository does **not** claim that:

- operational candidate identity is human speaker ground truth;
- representation-level candidate structure necessarily implies stable perceptual speaker identity for human listeners;
- every original analysis can be rerun from a fresh clone without separately archived WAV files or derived arrays;
- exact waveform or floating-point bitwise identity is guaranteed on other hardware/software stacks.

These boundaries are intentional parts of the reproducibility record.
