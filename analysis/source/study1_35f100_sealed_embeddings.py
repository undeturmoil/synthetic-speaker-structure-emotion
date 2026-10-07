#!/usr/bin/env python3

import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


ROOT = Path("/factory")

MANIFEST = (
    ROOT / "state/study1_35f100_sealed_v1/manifest.json"
)

OUT = (
    ROOT / "campaigns/study1_35f100_sealed_v1/analysis"
)

VECTORS = OUT / "vectors"

BASE_SNAPSHOT = (
    "/home/sgbaeck/project/qwen-voice/cache/huggingface/hub/"
    "models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/"
    "snapshots/fd4b254389122332181a7c3db7f27e918eec64e3"
)

EMOTIONS = [
    "normal",
    "happy",
    "sad",
    "calm",
    "angry",
    "surprise",
]

EIDX = {
    e: i
    for i, e in enumerate(EMOTIONS)
}


def atomic_npy(path, arr):
    tmp = Path(str(path) + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    VECTORS.mkdir(parents=True, exist_ok=True)

    m = json.loads(
        MANIFEST.read_text(encoding="utf-8")
    )

    rows = m["samples"]

    if len(rows) != 600:
        raise RuntimeError(
            f"expected 600 rows, got {len(rows)}"
        )

    if not all(
        r["status"] == "DONE"
        for r in rows
    ):
        raise RuntimeError(
            "generation incomplete"
        )

    pending = []

    for r in rows:
        p = VECTORS / (
            f"{r['candidate_id']}__"
            f"{r['emotion']}.npy"
        )

        if p.exists():
            x = np.load(p)

            if x.shape != (2048,):
                raise RuntimeError(
                    f"{p}: {x.shape}"
                )
        else:
            pending.append(
                (r, p)
            )

    print(
        f"cached : {600-len(pending)}/600"
    )
    print(
        f"pending: {len(pending)}/600"
    )

    if pending:
        print(
            "Loading Base speaker encoder..."
        )

        model = (
            Qwen3TTSModel.from_pretrained(
                BASE_SNAPSHOT,
                device_map="cuda:0",
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                local_files_only=True,
            )
        )

        for i, (r, p) in enumerate(
            pending,
            1,
        ):
            audio, sr = sf.read(
                Path(r["output"]),
                dtype="float32",
                always_2d=False,
            )

            if sr != 24000:
                raise RuntimeError(
                    f"sr={sr}"
                )

            if audio.ndim != 1:
                raise RuntimeError(
                    audio.shape
                )

            with torch.inference_mode():
                t = (
                    model.model
                    .extract_speaker_embedding(
                        audio=audio,
                        sr=sr,
                    )
                )

            emb = (
                t.detach()
                .float()
                .cpu()
                .numpy()
                .reshape(-1)
                .astype(np.float32)
            )

            if emb.shape != (2048,):
                raise RuntimeError(
                    emb.shape
                )

            atomic_npy(
                p,
                emb,
            )

            print(
                f"[{i:03d}/{len(pending)}] "
                f"{r['candidate_id']} "
                f"{r['emotion']}"
            )

    X = np.zeros(
        (100, 6, 2048),
        dtype=np.float32,
    )

    filled = np.zeros(
        (100, 6),
        dtype=bool,
    )

    for r in rows:
        global_idx = int(
            r["global_candidate_index"]
        )

        local_idx = (
            global_idx - 400
        )

        if not 0 <= local_idx < 100:
            raise RuntimeError(
                f"unexpected global index: "
                f"{global_idx}"
            )

        ei = EIDX[
            r["emotion"]
        ]
        p = VECTORS / (
            f"{r['candidate_id']}__"
            f"{r['emotion']}.npy"
        )

        X[
            local_idx,
            ei,
            :
        ] = np.load(
            p
        ).astype(
            np.float32
        )

        filled[
            local_idx,
            ei
        ] = True

    if not filled.all():
        raise RuntimeError(
            f"missing cells="
            f"{np.sum(~filled)}"
        )

    path = (
        OUT
        / "embeddings_100x6x2048.npy"
    )

    np.save(
        path,
        X,
    )

    print()
    print("=" * 72)
    print(
        "EXTERNAL VALIDATION EMBEDDINGS COMPLETE"
    )
    print("=" * 72)
    print("shape:", X.shape)
    print("saved:", path)


if __name__ == "__main__":
    main()
