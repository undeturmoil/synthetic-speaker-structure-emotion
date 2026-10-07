# Original DGX Spark Environment

This directory records the original software environment used for the
reported Study 1 experiments.

The purpose of these files is provenance preservation rather than claiming
platform-independent bitwise reproducibility.

## Hardware / architecture

The reported experiments were executed on NVIDIA DGX Spark using the ARM64
(aarch64) software stack.

## Environment chain

The original environment was constructed in three layers:

1. `local/qwen3-tts:1.7b-base`
   - Base Qwen3-TTS runtime.
   - Built from `Dockerfile.qwen3_tts_base`.
   - Python 3.12.
   - PyTorch 2.11.0 + CUDA 13.0.
   - Qwen-TTS 0.1.1.

2. `local/voice-character-factory:phase1`
   - Extends the Qwen3-TTS image with experiment-control dependencies.
   - Used for voice generation, Qwen speaker-embedding extraction, and
     core Study 1 processing.

3. `local/voice-character-factory:emotioncheck-v1`
   - Extends the phase1 environment with the external emotion-analysis
     dependencies used for manipulation checks.

## Locked model revisions

See `model_revisions.json`.

The VoiceDesign and Base Hugging Face revisions used in the experiments are
explicitly pinned there.

Speaker embeddings were extracted using:

`Qwen3TTSForConditionalGeneration.extract_speaker_embedding`

with:

`attn_implementation="sdpa"`

## Observed package states

`pip-freeze.phase1.txt` and `pip-freeze.emotioncheck-v1.txt` are direct
snapshots of the installed Python packages in the original Docker images.

These files record the environment that produced the reported results.
They are intentionally preserved separately from the shorter direct
dependency files used to construct the images.

## Docker provenance

The `docker-inspect.*.json` files are snapshots of the original local Docker
images used during the study.

Local image tags themselves are not globally resolvable registry references;
therefore the accompanying image metadata, source Dockerfiles, package
snapshots, and cryptographic hashes are provided as provenance records.

## Portability

No claim is made that another CPU architecture, CUDA version, PyTorch build,
or hardware platform will reproduce all reported floating-point values
bit-for-bit.

Portable environments should be validated separately before being described
as reproductions of the reported numerical results.
