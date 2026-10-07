# Configurations

This directory preserves the generation configuration provenance for Study 1.

## As-run generation source

The original generation scripts are preserved without modification under `configs/source/`.
Their SHA-256 checksums are recorded in `configs/source/SHA256SUMS`.

`study1_35f100_generate.py` defines the frozen 35-year-old female generation contract, including the synthesis text, age/sex instruction, six emotion instructions, generation parameters, RNG seeding procedure, VoiceDesign snapshot, and model-loading settings.

The Development300 extension, External100, and Sealed100 generation scripts import that same frozen contract directly from `study1_35f100_generate.py` and change only their campaign-specific batch ranges and seeds.

## Machine-readable generation configuration

The authoritative machine-readable campaign configurations are preserved in the original source manifests under `manifests/source/`.

These manifests contain the exact synthesis text, prompts, batch seeds, batch positions, emotion labels, generation parameters, campaign metadata, sample records, waveform checksums, and recorded configuration hashes.

No additional normalized generation JSON is maintained here, in order to avoid duplicating authoritative configuration data and creating a second source that could diverge from the original manifests.

## Operational candidate definition

For this study, a candidate is an RNG batch-slot pairing: a fixed batch seed and fixed position within a batch of 10 outputs.
For each emotion condition, the same batch seed is reset immediately before generating the corresponding 10-output batch.
Candidate identity therefore refers to this reproducible operational pairing and not to human speaker ground truth.

## Original local paths

The archived as-run scripts retain original local workspace and model-cache paths as provenance. These paths are not portable configuration requirements.
