#!/usr/bin/env python3

from pathlib import Path
from math import gcd
import csv
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np
import soundfile as sf
import torch

from scipy.signal import resample_poly
from funasr import AutoModel

try:
    import funasr
except Exception:
    funasr = None

try:
    import modelscope
except Exception:
    modelscope = None


ROOT = Path("/factory")

CAMPAIGN = (
    ROOT
    / "campaigns"
    / "study1_manipulation_check_v1"
)

MANIFEST = (
    CAMPAIGN
    / "manifest_2400.csv"
)

OUT = (
    CAMPAIGN
    / "emotion2vec_plus_large"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_ID = "iic/emotion2vec_plus_large"

EXPECTED_N = 2400
EXPECTED_D = 1024
EXPECTED_CLASSES = 9

EMOTIONS = [
    "normal",
    "happy",
    "sad",
    "calm",
    "angry",
    "surprise",
]

SEMANTIC_TARGET = {
    "normal": "neutral",
    "happy": "happy",
    "sad": "sad",
    "calm": None,
    "angry": "angry",
    "surprise": "surprised",
}


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

EMB_PART = OUT / "embeddings_2400x1024.partial.npy"
SCORE_PART = OUT / "scores_2400x9.partial.npy"
DONE_PART = OUT / "done_2400.partial.npy"

EMB_FINAL = OUT / "embeddings_2400x1024.npy"
SCORE_FINAL = OUT / "scores_2400x9.npy"

LABELS_JSON = OUT / "labels.json"
ROWS_CSV = OUT / "emotion2vec_scores_2400.csv"
PROVENANCE_JSON = OUT / "provenance.json"


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def sha256_file(path, block_size=1024 * 1024):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(block_size)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def package_version(module):
    if module is None:
        return None

    return getattr(
        module,
        "__version__",
        "unknown",
    )


def normalize_labels(labels):
    return [
        str(x).split("/")[-1]
        for x in labels
    ]


def load_manifest():
    if not MANIFEST.exists():
        raise RuntimeError(
            f"Manifest not found: {MANIFEST}"
        )

    with MANIFEST.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        rows = list(
            csv.DictReader(f)
        )

    if len(rows) != EXPECTED_N:
        raise RuntimeError(
            f"Expected {EXPECTED_N} manifest rows, "
            f"got {len(rows)}"
        )

    # Strong ordering / balance validation.
    dev = sum(
        r["dataset"] == "development"
        for r in rows
    )

    ext = sum(
        r["dataset"] == "external"
        for r in rows
    )

    if dev != 1800:
        raise RuntimeError(
            f"Development count != 1800: {dev}"
        )

    if ext != 600:
        raise RuntimeError(
            f"External count != 600: {ext}"
        )

    for emotion in EMOTIONS:
        n = sum(
            r["emotion"] == emotion
            for r in rows
        )

        if n != 400:
            raise RuntimeError(
                f"{emotion}: expected 400, got {n}"
            )

    return rows


def load_audio_16k(path):
    x, sr = sf.read(
        path,
        dtype="float32",
        always_2d=False,
    )

    if x.ndim == 2:
        x = x.mean(axis=1)

    if not np.all(
        np.isfinite(x)
    ):
        raise RuntimeError(
            f"NaN/Inf audio: {path}"
        )

    if sr != 16000:
        g = gcd(
            int(sr),
            16000,
        )

        x = resample_poly(
            x,
            16000 // g,
            int(sr) // g,
        ).astype(
            np.float32,
            copy=False,
        )

    return x


def prepare_array(
    path,
    shape,
    dtype,
):
    if path.exists():
        arr = np.load(
            path,
            mmap_mode="r+",
        )

        if arr.shape != shape:
            raise RuntimeError(
                f"{path} shape "
                f"{arr.shape} != {shape}"
            )

        if arr.dtype != np.dtype(dtype):
            raise RuntimeError(
                f"{path} dtype "
                f"{arr.dtype} != {dtype}"
            )

        return arr

    return np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=dtype,
        shape=shape,
    )


# ------------------------------------------------------------
# Start
# ------------------------------------------------------------

print("=" * 100)
print("STUDY1 MANIPULATION CHECK")
print("EMOTION2VEC+ LARGE FULL EXTRACTION")
print("=" * 100)

if EMB_FINAL.exists() or SCORE_FINAL.exists():
    if (
        EMB_FINAL.exists()
        and SCORE_FINAL.exists()
        and LABELS_JSON.exists()
        and ROWS_CSV.exists()
    ):
        print()
        print("Final outputs already exist.")
        print("Refusing to overwrite frozen extraction.")
        print("Embeddings:", EMB_FINAL)
        print("Scores    :", SCORE_FINAL)
        raise SystemExit(0)

    raise RuntimeError(
        "Partial final-output state detected. "
        "Inspect output directory manually."
    )


rows = load_manifest()

print()
print("Manifest:", MANIFEST)
print("Rows    :", len(rows))


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

print()
print("Loading model:", MODEL_ID)

model = AutoModel(
    model=MODEL_ID,
    device="cuda:0",
    disable_update=True,
)

print("Resolved path:", model.model_path)

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA unavailable"
    )

print("GPU:", torch.cuda.get_device_name(0))


# ------------------------------------------------------------
# Output arrays
# ------------------------------------------------------------

embeddings = prepare_array(
    EMB_PART,
    (
        EXPECTED_N,
        EXPECTED_D,
    ),
    np.float32,
)

scores = prepare_array(
    SCORE_PART,
    (
        EXPECTED_N,
        EXPECTED_CLASSES,
    ),
    np.float32,
)

done = prepare_array(
    DONE_PART,
    (
        EXPECTED_N,
    ),
    np.uint8,
)

n_done = int(
    np.sum(done)
)

print()
print(
    f"Resume state: "
    f"{n_done}/{EXPECTED_N} completed"
)


# ------------------------------------------------------------
# Extraction
# ------------------------------------------------------------

reference_labels = None

if LABELS_JSON.exists():
    reference_labels = json.loads(
        LABELS_JSON.read_text(
            encoding="utf-8"
        )
    )

tmp = Path(
    "/tmp/study1_emotion2vec_input.wav"
)

started = time.time()

for i, row in enumerate(rows):

    if int(done[i]) == 1:
        continue

    wav = (
        ROOT
        / row["wav_path"]
    )

    if not wav.exists():
        raise RuntimeError(
            f"Missing WAV: {wav}"
        )

    x = load_audio_16k(
        wav
    )

    sf.write(
        tmp,
        x,
        16000,
        subtype="PCM_16",
    )

    result = model.generate(
        input=str(tmp),
        granularity="utterance",
        extract_embedding=True,
    )

    if not result:
        raise RuntimeError(
            f"No result for row {i}: {wav}"
        )

    r = result[0]

    labels = normalize_labels(
        r.get(
            "labels",
            [],
        )
    )

    sc = np.asarray(
        r.get(
            "scores",
            [],
        ),
        dtype=np.float32,
    )

    feat = np.asarray(
        r.get(
            "feats",
        ),
        dtype=np.float32,
    ).reshape(-1)

    if feat.shape != (
        EXPECTED_D,
    ):
        raise RuntimeError(
            f"Unexpected embedding "
            f"shape {feat.shape} "
            f"at row {i}"
        )

    if sc.shape != (
        EXPECTED_CLASSES,
    ):
        raise RuntimeError(
            f"Unexpected score "
            f"shape {sc.shape} "
            f"at row {i}"
        )

    if len(labels) != EXPECTED_CLASSES:
        raise RuntimeError(
            f"Unexpected labels "
            f"{labels} at row {i}"
        )

    if not np.all(
        np.isfinite(feat)
    ):
        raise RuntimeError(
            f"Non-finite embedding "
            f"at row {i}"
        )

    if not np.all(
        np.isfinite(sc)
    ):
        raise RuntimeError(
            f"Non-finite scores "
            f"at row {i}"
        )

    if reference_labels is None:
        reference_labels = labels

        LABELS_JSON.write_text(
            json.dumps(
                reference_labels,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print()
        print(
            "Frozen label order:",
            reference_labels,
        )

    elif labels != reference_labels:
        raise RuntimeError(
            "Emotion label order changed:\n"
            f"expected={reference_labels}\n"
            f"actual={labels}"
        )

    embeddings[i] = feat
    scores[i] = sc

    # Write arrays first, done flag last.
    embeddings.flush()
    scores.flush()

    done[i] = 1
    done.flush()

    completed = int(
        np.sum(done)
    )

    if (
        completed == 1
        or completed % 25 == 0
        or completed == EXPECTED_N
    ):
        elapsed = (
            time.time()
            - started
        )

        pred_idx = int(
            np.argmax(sc)
        )

        pred = (
            reference_labels[
                pred_idx
            ]
        )

        print(
            f"[{completed:4d}/{EXPECTED_N}] "
            f"row={i:4d} "
            f"{row['dataset']:<11s} "
            f"{row['candidate_id']} "
            f"{row['emotion']:<9s} "
            f"pred={pred:<10s} "
            f"score={sc[pred_idx]:.6f} "
            f"elapsed={elapsed:.1f}s"
        )


# ------------------------------------------------------------
# Completion validation
# ------------------------------------------------------------

if not np.all(
    done == 1
):
    raise RuntimeError(
        "Extraction ended with incomplete rows."
    )

# Flush before validation.
embeddings.flush()
scores.flush()
done.flush()

E = np.asarray(
    embeddings
)

S = np.asarray(
    scores
)

if E.shape != (
    EXPECTED_N,
    EXPECTED_D,
):
    raise RuntimeError(
        E.shape
    )

if S.shape != (
    EXPECTED_N,
    EXPECTED_CLASSES,
):
    raise RuntimeError(
        S.shape
    )

if not np.all(
    np.isfinite(E)
):
    raise RuntimeError(
        "Final embeddings contain NaN/Inf"
    )

if not np.all(
    np.isfinite(S)
):
    raise RuntimeError(
        "Final scores contain NaN/Inf"
    )

# Score vectors should approximately sum to one.
score_sums = S.sum(
    axis=1
)

print()
print(
    "Score-sum range:",
    float(
        score_sums.min()
    ),
    "to",
    float(
        score_sums.max()
    ),
)


# ------------------------------------------------------------
# Write human-readable score table
# ------------------------------------------------------------

label_to_idx = {
    label: i
    for i, label
    in enumerate(
        reference_labels
    )
}

fieldnames = [
    "row_index",
    "dataset",
    "source_panel",
    "candidate_index",
    "candidate_id",
    "batch",
    "seed",
    "position",
    "instruction_emotion",
    "wav_path",
    "predicted_label",
    "predicted_score",
    "semantic_target_label",
    "semantic_target_score",
]

fieldnames += [
    f"score_{label}"
    for label
    in reference_labels
]


with ROWS_CSV.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:

    w = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    w.writeheader()

    for i, row in enumerate(
        rows
    ):

        sc = S[i]

        pred_idx = int(
            np.argmax(sc)
        )

        pred = (
            reference_labels[
                pred_idx
            ]
        )

        condition = (
            row["emotion"]
        )

        target = (
            SEMANTIC_TARGET[
                condition
            ]
        )

        target_score = ""

        if target is not None:
            if target not in label_to_idx:
                raise RuntimeError(
                    f"Target label "
                    f"{target} not present."
                )

            target_score = float(
                sc[
                    label_to_idx[
                        target
                    ]
                ]
            )

        out = {
            "row_index": i,
            "dataset":
                row["dataset"],
            "source_panel":
                row["source_panel"],
            "candidate_index":
                row["candidate_index"],
            "candidate_id":
                row["candidate_id"],
            "batch":
                row["batch"],
            "seed":
                row["seed"],
            "position":
                row["position"],
            "instruction_emotion":
                condition,
            "wav_path":
                row["wav_path"],
            "predicted_label":
                pred,
            "predicted_score":
                float(
                    sc[pred_idx]
                ),
            "semantic_target_label":
                (
                    target
                    if target is not None
                    else ""
                ),
            "semantic_target_score":
                target_score,
        }

        for label, j in (
            label_to_idx.items()
        ):
            out[
                f"score_{label}"
            ] = float(
                sc[j]
            )

        w.writerow(out)


# ------------------------------------------------------------
# Freeze partial arrays as final arrays
# ------------------------------------------------------------

# Close mmap references before rename.
del embeddings
del scores
del done

os.replace(
    EMB_PART,
    EMB_FINAL,
)

os.replace(
    SCORE_PART,
    SCORE_FINAL,
)

if DONE_PART.exists():
    DONE_PART.unlink()


# ------------------------------------------------------------
# Provenance
# ------------------------------------------------------------

provenance = {
    "status":
        "FROZEN COMPLETE EXTRACTION",

    "analysis_role":
        "Independent emotion manipulation check",

    "model": {
        "identifier":
            MODEL_ID,
        "resolved_model_path":
            str(
                model.model_path
            ),
        "embedding_dimension":
            EXPECTED_D,
        "labels":
            reference_labels,
        "granularity":
            "utterance",
        "extract_embedding":
            True,
    },

    "audio_preprocessing": {
        "source_sample_rate_hz":
            24000,
        "analysis_sample_rate_hz":
            16000,
        "resampler":
            "scipy.signal.resample_poly",
        "temporary_encoding":
            "PCM_16",
        "channel_policy":
            "mean channels if multichannel",
    },

    "data": {
        "manifest":
            str(MANIFEST),
        "utterances":
            EXPECTED_N,
        "development_utterances":
            1800,
        "external_utterances":
            600,
        "candidate_count":
            400,
        "emotion_conditions":
            EMOTIONS,
    },

    "semantic_target_mapping": {
        k: v
        for k, v
        in SEMANTIC_TARGET.items()
    },

    "software": {
        "python":
            sys.version,
        "platform":
            platform.platform(),
        "machine":
            platform.machine(),
        "torch":
            torch.__version__,
        "cuda_build":
            torch.version.cuda,
        "funasr":
            package_version(
                funasr
            ),
        "modelscope":
            package_version(
                modelscope
            ),
        "numpy":
            np.__version__,
    },

    "hardware": {
        "cuda_available":
            torch.cuda.is_available(),
        "gpu":
            (
                torch.cuda.get_device_name(
                    0
                )
                if torch.cuda.is_available()
                else None
            ),
    },

    "outputs": {
        "embeddings":
            str(
                EMB_FINAL
            ),
        "scores":
            str(
                SCORE_FINAL
            ),
        "score_table":
            str(
                ROWS_CSV
            ),
        "labels":
            str(
                LABELS_JSON
            ),
    },
}

# Hash only after files are finalized.
provenance[
    "sha256"
] = {
    "manifest":
        sha256_file(
            MANIFEST
        ),
    "embeddings":
        sha256_file(
            EMB_FINAL
        ),
    "scores":
        sha256_file(
            SCORE_FINAL
        ),
    "score_table":
        sha256_file(
            ROWS_CSV
        ),
}

PROVENANCE_JSON.write_text(
    json.dumps(
        provenance,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


# ------------------------------------------------------------
# Quick descriptive audit only
# ------------------------------------------------------------

print()
print("=" * 100)
print("EXTRACTION COMPLETE")
print("=" * 100)

print(
    "Embeddings:",
    EMB_FINAL,
)

print(
    "Scores    :",
    SCORE_FINAL,
)

print(
    "Labels    :",
    LABELS_JSON,
)

print(
    "CSV       :",
    ROWS_CSV,
)

print(
    "Provenance:",
    PROVENANCE_JSON,
)

print()
print(
    "Embedding shape:",
    np.load(
        EMB_FINAL,
        mmap_mode="r",
    ).shape,
)

print(
    "Score shape    :",
    np.load(
        SCORE_FINAL,
        mmap_mode="r",
    ).shape,
)

print()
print("SHA256:")
for k, v in (
    provenance[
        "sha256"
    ].items()
):
    print(
        f"  {k:<12s}: {v}"
    )

print()
print("DONE")
