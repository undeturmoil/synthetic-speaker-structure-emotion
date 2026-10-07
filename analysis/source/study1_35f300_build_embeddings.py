#!/usr/bin/env python3
import csv
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


ROOT = Path("/factory")

PRIOR = (
    ROOT / "campaigns/study1_35f100_v1/"
    "analysis_baseline/embeddings_100x6x2048.npy"
)

MANIFEST = (
    ROOT / "state/study1_35f300_extension_v1/manifest.json"
)

OUT = (
    ROOT / "campaigns/study1_35f300_v1/analysis"
)
VECTORS = OUT / "extension_vectors"

BASE_SNAPSHOT = (
    "/home/sgbaeck/project/qwen-voice/cache/huggingface/hub/"
    "models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/"
    "snapshots/fd4b254389122332181a7c3db7f27e918eec64e3"
)

EMOTIONS = [
    "normal", "happy", "sad",
    "calm", "angry", "surprise",
]
EIDX = {e:i for i,e in enumerate(EMOTIONS)}


def atomic_npy(path, arr):
    tmp = Path(str(path) + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    VECTORS.mkdir(parents=True, exist_ok=True)

    prior = np.load(PRIOR)
    if prior.shape != (100, 6, 2048):
        raise RuntimeError(prior.shape)

    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = m["samples"]

    if len(rows) != 1200:
        raise RuntimeError(f"manifest rows={len(rows)}")

    if not all(r["status"] == "DONE" for r in rows):
        raise RuntimeError("extension generation incomplete")

    pending = []

    for r in rows:
        key = f"{r['candidate_id']}__{r['emotion']}.npy"
        p = VECTORS / key

        if p.exists():
            x = np.load(p)
            if x.shape != (2048,):
                raise RuntimeError(f"bad cached vector: {p} {x.shape}")
        else:
            pending.append((r, p))

    print(f"cached : {1200-len(pending)}/1200")
    print(f"pending: {len(pending)}/1200")

    if pending:
        print("Loading Base speaker encoder...")

        model = Qwen3TTSModel.from_pretrained(
            BASE_SNAPSHOT,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        )

        for i, (r, vector_path) in enumerate(pending, 1):
            wav_path = Path(r["output"])

            audio, sr = sf.read(
                wav_path,
                dtype="float32",
                always_2d=False,
            )

            if sr != 24000 or audio.ndim != 1:
                raise RuntimeError(
                    f"{wav_path}: sr={sr} shape={audio.shape}"
                )

            with torch.inference_mode():
                t = model.model.extract_speaker_embedding(
                    audio=audio,
                    sr=sr,
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
                    f"{wav_path}: {emb.shape}"
                )

            atomic_npy(vector_path, emb)

            print(
                f"[{i:04d}/{len(pending)}] "
                f"{r['candidate_id']} {r['emotion']}"
            )

    Xext = np.zeros((200, 6, 2048), dtype=np.float32)
    filled = np.zeros((200, 6), dtype=bool)
    records = []

    for r in rows:
        global_idx = int(r["global_candidate_index"])
        local_idx = global_idx - 100

        if not 0 <= local_idx < 200:
            raise RuntimeError(global_idx)

        ei = EIDX[r["emotion"]]

        p = VECTORS / (
            f"{r['candidate_id']}__{r['emotion']}.npy"
        )

        emb = np.load(p).astype(np.float32)

        Xext[local_idx, ei] = emb
        filled[local_idx, ei] = True

        records.append({
            "candidate_id": r["candidate_id"],
            "global_candidate_index": global_idx,
            "batch_index": r["batch_index"],
            "batch_seed": r["batch_seed"],
            "batch_position": r["batch_position"],
            "emotion": r["emotion"],
            "l2_norm": float(np.linalg.norm(emb)),
            "wav": r["output"],
            "vector": str(p),
        })

    if not filled.all():
        raise RuntimeError(
            f"missing embedding cells: {np.sum(~filled)}"
        )

    combined = np.concatenate(
        [prior.astype(np.float32), Xext],
        axis=0,
    )

    assert combined.shape == (300, 6, 2048)

    np.save(
        OUT / "embeddings_extension_200x6x2048.npy",
        Xext,
    )

    np.save(
        OUT / "embeddings_300x6x2048.npy",
        combined,
    )

    with (OUT / "extension_records.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(records[0].keys()),
        )
        w.writeheader()
        w.writerows(records)

    summary = {
        "prior_candidates": 100,
        "new_candidates": 200,
        "total_candidates": 300,
        "emotions": EMOTIONS,
        "samples": 1800,
        "shape": list(combined.shape),
        "batch_indices": list(range(30)),
        "batch_seeds": list(range(1502000, 1502030)),
    }

    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("35F300 EMBEDDING PANEL COMPLETE")
    print("=" * 72)
    print("extension:", Xext.shape)
    print("combined :", combined.shape)
    print("saved    :", OUT / "embeddings_300x6x2048.npy")


if __name__ == "__main__":
    main()
