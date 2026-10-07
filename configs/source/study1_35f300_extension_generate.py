#!/usr/bin/env python3
# coding: utf-8

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

import torch
from qwen_tts import Qwen3TTSModel

# Import the EXACT frozen generation contract from the completed 35F100 campaign.
from study1_35f100_generate import (
    TEXT,
    FEMALE_35_BASE,
    EMOTIONS,
    TAIL,
    GENERATION,
    VD_SNAPSHOT,
    seed_all,
    sync,
    wav_bytes,
    atomic_write,
    atomic_json,
)


ROOT = Path("/factory")
CAMPAIGN = "study1_35f300_extension_v1"

OUT = ROOT / "outputs" / CAMPAIGN
STATE = ROOT / "state" / CAMPAIGN
LOGS = ROOT / "logs" / CAMPAIGN
MANIFEST = STATE / "manifest.json"

BATCH_SIZE = 10

# Existing campaign: 1502000..1502009
# Extension only:
BATCH_INDICES = list(range(10, 30))
BATCH_SEEDS = [1502000 + i for i in BATCH_INDICES]


def config_payload():
    return {
        "campaign": CAMPAIGN,
        "extends_campaign": "study1_35f100_v1",
        "target_age": 35,
        "design_age": 50,
        "gender": "female",
        "batch_size": BATCH_SIZE,
        "batch_indices": BATCH_INDICES,
        "batch_seeds": BATCH_SEEDS,
        "text": TEXT,
        "female_35_base": FEMALE_35_BASE,
        "emotions": EMOTIONS,
        "tail": TAIL,
        "generation": GENERATION,
        "candidate_count": 200,
        "sample_count": 1200,
    }


def config_sha():
    b = json.dumps(
        config_payload(),
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")

    return hashlib.sha256(b).hexdigest()


def make_manifest():
    samples = []

    for batch_index, seed in zip(
        BATCH_INDICES,
        BATCH_SEEDS,
    ):
        for emotion in EMOTIONS:
            for pos in range(10):
                cid = f"b{batch_index:02d}p{pos:02d}"

                samples.append({
                    "candidate_id": cid,
                    "global_candidate_index":
                        batch_index * 10 + pos,
                    "batch_index": batch_index,
                    "batch_seed": seed,
                    "batch_position": pos,
                    "batch_size": 10,
                    "target_age": 35,
                    "design_age": 50,
                    "gender": "female",
                    "emotion": emotion,
                    "status": "PENDING",
                    "attempt_count": 0,
                    "sha256": None,
                    "duration_sec": None,
                    "rms": None,
                    "peak": None,
                    "output": str(
                        OUT
                        / emotion
                        / f"{cid}.wav"
                    ),
                })

    return {
        "schema": "study1-35f300-extension-v1",
        "campaign_id": CAMPAIGN,
        "config_sha256": config_sha(),
        "config": config_payload(),
        "design": {
            "new_batch_seeds": 20,
            "new_candidates": 200,
            "emotions": 6,
            "new_samples": 1200,
            "batch_calls": 120,
            "combined_with_prior_candidates": 300,
            "combined_samples": 1800,
            "identity_note": (
                "candidate_id is a fixed batch-seed x batch-position "
                "stochastic candidate, not human speaker ground truth"
            ),
        },
        "samples": samples,
    }


def load_manifest():
    if MANIFEST.exists():
        m = json.loads(
            MANIFEST.read_text(
                encoding="utf-8"
            )
        )

        if m.get("config_sha256") != config_sha():
            raise RuntimeError(
                "extension config changed after manifest creation"
            )

        return m

    m = make_manifest()
    atomic_json(MANIFEST, m)
    return m


def generate_group(model, seed, emotion):
    instruction = (
        FEMALE_35_BASE
        + EMOTIONS[emotion]
        + TAIL
    )

    seed_all(seed)

    sync()
    t0 = time.perf_counter()

    wavs, sr = model.generate_voice_design(
        text=[TEXT] * 10,
        language=["Korean"] * 10,
        instruct=[instruction] * 10,
        **GENERATION,
    )

    sync()

    elapsed = time.perf_counter() - t0

    if len(wavs) != 10:
        raise RuntimeError(
            f"expected B=10, got {len(wavs)}"
        )

    if int(sr) != 24000:
        raise RuntimeError(
            f"unexpected sample rate: {sr}"
        )

    return wavs, int(sr), elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--max-groups",
        type=int,
        default=None,
    )
    args = ap.parse_args()

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )
    STATE.mkdir(
        parents=True,
        exist_ok=True,
    )
    LOGS.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = load_manifest()
    samples = manifest["samples"]

    groups = []

    for batch_index, seed in zip(
        BATCH_INDICES,
        BATCH_SEEDS,
    ):
        for emotion in EMOTIONS:
            rows = [
                r
                for r in samples
                if r["batch_index"] == batch_index
                and r["emotion"] == emotion
            ]

            rows = sorted(
                rows,
                key=lambda r:
                    r["batch_position"],
            )

            if len(rows) != 10:
                raise RuntimeError(
                    f"batch={batch_index} "
                    f"emotion={emotion}: "
                    f"{len(rows)} rows"
                )

            groups.append(
                (
                    batch_index,
                    seed,
                    emotion,
                    rows,
                )
            )

    pending = []

    for g in groups:
        rows = g[3]

        complete = all(
            r["status"] == "DONE"
            and Path(
                r["output"]
            ).exists()
            for r in rows
        )

        if not complete:
            pending.append(g)

    if args.max_groups is not None:
        pending = pending[
            :args.max_groups
        ]

    print("=" * 72)
    print(CAMPAIGN)
    print("=" * 72)
    print("New batch seeds :", 20)
    print("New candidates  :", 200)
    print("New WAVs        :", 1200)
    print("Batch calls     :", 120)
    print("Pending calls   :", len(pending))
    print()

    if not pending:
        print("Nothing to generate.")
        return

    print("Loading VoiceDesign...")
    t0 = time.perf_counter()

    model = Qwen3TTSModel.from_pretrained(
        VD_SNAPSHOT,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )

    print(
        f"VoiceDesign loaded: "
        f"{time.perf_counter()-t0:.3f}s"
    )

    for gi, (
        batch_index,
        seed,
        emotion,
        rows,
    ) in enumerate(
        pending,
        1,
    ):
        print()
        print(
            f"[{gi}/{len(pending)}] "
            f"batch={batch_index:02d} "
            f"seed={seed} "
            f"emotion={emotion}"
        )

        for r in rows:
            r["attempt_count"] += 1
            r["status"] = "RUNNING"

        atomic_json(
            MANIFEST,
            manifest,
        )

        try:
            wavs, sr, elapsed = (
                generate_group(
                    model,
                    seed,
                    emotion,
                )
            )

            prepared = []

            for wav in wavs:
                prepared.append(
                    wav_bytes(
                        wav,
                        sr,
                    )
                )

            # Entire B10 group must pass QA
            # before any status becomes DONE.
            for pos, r in enumerate(rows):
                (
                    payload,
                    sha,
                    duration,
                    rms,
                    peak,
                ) = prepared[pos]

                path = Path(
                    r["output"]
                )

                if path.exists():
                    old_sha = hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()

                    if old_sha != sha:
                        raise RuntimeError(
                            f"existing file mismatch: {path}"
                        )
                else:
                    atomic_write(
                        path,
                        payload,
                    )

                r["status"] = "DONE"
                r["sha256"] = sha
                r["duration_sec"] = duration
                r["rms"] = rms
                r["peak"] = peak
                r["group_generation_sec"] = elapsed
                r["sec_per_sample"] = (
                    elapsed / 10.0
                )

            atomic_json(
                MANIFEST,
                manifest,
            )

            print(
                f"done {elapsed:.3f}s "
                f"{elapsed/10:.3f}s/sample"
            )

        except Exception:
            for r in rows:
                if r["status"] == "RUNNING":
                    r["status"] = "FAILED"

            atomic_json(
                MANIFEST,
                manifest,
            )
            raise

    counts = Counter(
        r["status"]
        for r in samples
    )

    done = sum(
        1
        for r in samples
        if r["status"] == "DONE"
        and Path(
            r["output"]
        ).exists()
    )

    print()
    print("=" * 72)
    print("STATUS")
    print("=" * 72)
    print(dict(counts))
    print(
        f"extension files: "
        f"{done}/1200"
    )


if __name__ == "__main__":
    main()
