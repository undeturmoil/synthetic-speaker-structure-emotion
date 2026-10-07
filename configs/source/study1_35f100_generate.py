#!/usr/bin/env python3
# coding: utf-8

import argparse
import hashlib
import io
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


ROOT = Path("/factory")
CAMPAIGN = "study1_35f100_v1"

OUT = ROOT / "outputs" / CAMPAIGN
STATE = ROOT / "state" / CAMPAIGN
LOGS = ROOT / "logs" / CAMPAIGN

MANIFEST = STATE / "manifest.json"

VD_SNAPSHOT = (
    "/home/sgbaeck/project/qwen-voice/cache/huggingface/hub/"
    "models--Qwen--Qwen3-TTS-12Hz-1.7B-VoiceDesign/"
    "snapshots/5ecdb67327fd37bb2e042aab12ff7391903235d3"
)

TEXT = (
    "오늘은 조금 일찍 집을 나섰습니다. 길을 걷다 보니 익숙한 풍경도 "
    "평소와 조금 다르게 느껴졌습니다. 잠시 생각을 정리한 뒤, 해야 할 "
    "일을 하나씩 차분하게 시작해 보려고 합니다."
)

BATCH_SIZE = 10

# 10 independent batch seeds × 10 fixed positions = 100 candidate slots.
BATCH_SEEDS = [1502000 + i for i in range(10)]

FEMALE_35_BASE = (
    "Create a native Korean woman, a mature adult around 50 years old, "
    "intended to be perceived as a person in the mid-thirties. "
    "Use a naturally grounded female speaking voice rather than a high, thin "
    "or girlish voice. Preserve believable vocal body and low-mid warmth "
    "appropriate to her age. "
    "Use a medium to medium-low female register with fuller vocal body, "
    "warm low-mid presence and less youthful brightness. "
    "Let maturity come mainly from vocal body and reduced youthful brightness, "
    "not frailty or slow speech. "
)

EMOTIONS = {
    "normal": (
        "Use an ordinary neutral conversational delivery: balanced, relaxed "
        "and emotionally unmarked. "
    ),

    "happy": (
        "Express genuine restrained happiness with a subtle audible smile and "
        "slightly brighter conversational energy. Keep it realistic, composed "
        "and appropriate to straight-drama acting. Do not sound excited, cute, "
        "cartoonish, bubbly or theatrical. "
    ),

    "sad": (
        "Express genuine restrained sadness with somewhat lower emotional "
        "energy, slightly heavier phrasing and quiet emotional weight. Remain "
        "composed and intelligible, as in realistic straight-drama acting. "
        "Do not cry, sob, tremble dramatically, become excessively breathy, "
        "or turn the performance into melodrama. "
    ),

    "calm": (
        "Use a distinctly calm, settled and reassuring delivery. Keep arousal "
        "low, the rhythm steady and slightly unhurried, and the emotional tone "
        "composed. Remain naturally conversational and attentive. Do not sound "
        "sleepy, flat, detached, sentimental or artificially soft. "
    ),

    "angry": (
        "Express controlled anger with restrained tension, firmer articulation "
        "and clear emotional pressure while maintaining self-control. Use "
        "realistic straight-drama acting. Anger should be recognizable without "
        "shouting, snarling, exaggerated harshness, extreme speed or theatrical aggression. "
    ),

    "surprise": (
        "Express clear but controlled surprise, with a brief natural lift in "
        "energy and intonation and a believable sense of unexpectedness. "
        "Quickly return to a composed conversational tone. Keep it realistic "
        "and understated, never gasping, shouting, cartoonish, exaggerated "
        "or theatrical. "
    ),
}

TAIL = (
    "Use natural standard Korean. Keep the speaker's underlying age and vocal "
    "body stable while expressing the requested emotion. "
    "Do not use announcer diction or theatrical character acting. "
    "Do not exaggerate pitch, speed, breathiness, roughness or articulation changes."
)

GENERATION = {
    "non_streaming_mode": True,
    "do_sample": True,
    "temperature": 0.9,
    "top_p": 1.0,
    "top_k": 50,
    "repetition_penalty": 1.05,
    "subtalker_dosample": True,
    "subtalker_temperature": 0.9,
    "subtalker_top_p": 1.0,
    "subtalker_top_k": 50,
    "max_new_tokens": 2048,
}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def config_payload():
    return {
        "campaign": CAMPAIGN,
        "target_age": 35,
        "design_age": 50,
        "gender": "female",
        "batch_size": BATCH_SIZE,
        "batch_seeds": BATCH_SEEDS,
        "text": TEXT,
        "female_35_base": FEMALE_35_BASE,
        "emotions": EMOTIONS,
        "tail": TAIL,
        "generation": GENERATION,
        "development_batches": list(range(8)),
        "sealed_holdout_batches": [8, 9],
    }


def config_sha():
    b = json.dumps(
        config_payload(),
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")

    return hashlib.sha256(b).hexdigest()


def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")

    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    os.replace(tmp, path)


def make_manifest():
    records = []

    for batch_index, seed in enumerate(BATCH_SEEDS):
        split = "development" if batch_index < 8 else "holdout"

        for emotion in EMOTIONS:
            for pos in range(10):
                cid = f"b{batch_index:02d}p{pos:02d}"

                records.append({
                    "candidate_id": cid,
                    "candidate_index": batch_index * 10 + pos,
                    "batch_index": batch_index,
                    "batch_seed": seed,
                    "batch_position": pos,
                    "batch_size": 10,
                    "split": split,
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
        "schema": "study1-35f100-v1",
        "campaign_id": CAMPAIGN,
        "config_sha256": config_sha(),
        "config": config_payload(),
        "design": {
            "candidates": 100,
            "emotions": 6,
            "samples": 600,
            "batch_calls": 60,
            "identity_note": (
                "candidate_id is an RNG batch-slot candidate, "
                "not human speaker ground truth"
            ),
        },
        "samples": records,
    }


def load_manifest():
    if MANIFEST.exists():
        m = json.loads(MANIFEST.read_text(encoding="utf-8"))

        if m.get("config_sha256") != config_sha():
            raise RuntimeError(
                "campaign config changed after manifest creation; "
                "refusing to resume"
            )

        return m

    m = make_manifest()
    atomic_json(MANIFEST, m)
    return m


def wav_bytes(wav, sr):
    x = np.asarray(wav, dtype=np.float32).squeeze()

    if x.ndim != 1:
        raise RuntimeError(f"unexpected waveform shape {x.shape}")

    if not np.all(np.isfinite(x)):
        raise RuntimeError("non-finite audio")

    if len(x) < 2400:
        raise RuntimeError(f"audio too short: {len(x)} samples")

    rms = float(np.sqrt(np.mean(x * x)))
    peak = float(np.max(np.abs(x)))

    if rms <= 1e-5:
        raise RuntimeError("near-silent output")

    buf = io.BytesIO()

    sf.write(
        buf,
        x,
        sr,
        format="WAV",
        subtype="PCM_16",
    )

    payload = buf.getvalue()

    y, check_sr = sf.read(
        io.BytesIO(payload),
        dtype="float32",
        always_2d=False,
    )

    if check_sr != 24000:
        raise RuntimeError(f"bad sample rate {check_sr}")

    if y.ndim != 1:
        raise RuntimeError(f"bad WAV channels: shape={y.shape}")

    return (
        payload,
        hashlib.sha256(payload).hexdigest(),
        float(len(y) / check_sr),
        rms,
        peak,
    )


def atomic_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = Path(str(path) + ".part")

    with tmp.open("wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, path)


def generate_group(model, seed, emotion):
    instruction = FEMALE_35_BASE + EMOTIONS[emotion] + TAIL

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
        raise RuntimeError(f"unexpected sample rate: {sr}")

    return wavs, int(sr), elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-groups", type=int, default=None)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    STATE.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()

    samples = manifest["samples"]

    groups = []

    for batch_index, seed in enumerate(BATCH_SEEDS):
        for emotion in EMOTIONS:
            rows = [
                r for r in samples
                if r["batch_index"] == batch_index
                and r["emotion"] == emotion
            ]

            rows = sorted(
                rows,
                key=lambda r: r["batch_position"],
            )

            if len(rows) != 10:
                raise RuntimeError(
                    f"group batch={batch_index} emotion={emotion}: "
                    f"{len(rows)} rows"
                )

            groups.append(
                (batch_index, seed, emotion, rows)
            )

    pending_groups = []

    for g in groups:
        rows = g[3]

        complete = all(
            r["status"] == "DONE"
            and Path(r["output"]).exists()
            for r in rows
        )

        if not complete:
            pending_groups.append(g)

    if args.max_groups is not None:
        pending_groups = pending_groups[:args.max_groups]

    print("=" * 72)
    print(CAMPAIGN)
    print("=" * 72)
    print("Declared samples : 600")
    print("Batch calls      : 60")
    print("Pending calls    :", len(pending_groups))
    print("Dev candidates   : 80  (batch00-07)")
    print("Holdout          : 20  (batch08-09)")
    print()

    if not pending_groups:
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

    for gi, (batch_index, seed, emotion, rows) in enumerate(
        pending_groups,
        1,
    ):
        print()
        print(
            f"[{gi}/{len(pending_groups)}] "
            f"batch={batch_index:02d} "
            f"seed={seed} emotion={emotion}"
        )

        for r in rows:
            r["attempt_count"] += 1
            r["status"] = "RUNNING"

        atomic_json(MANIFEST, manifest)

        wavs, sr, elapsed = generate_group(
            model,
            seed,
            emotion,
        )

        prepared = []

        for pos, wav in enumerate(wavs):
            payload, sha, duration, rms, peak = wav_bytes(
                wav,
                sr,
            )

            prepared.append(
                (payload, sha, duration, rms, peak)
            )

        # Only after all ten outputs pass QA do we write.
        for pos, r in enumerate(rows):
            payload, sha, duration, rms, peak = prepared[pos]
            path = Path(r["output"])

            if path.exists():
                existing = path.read_bytes()
                existing_sha = hashlib.sha256(existing).hexdigest()

                if existing_sha != sha:
                    raise RuntimeError(
                        f"existing file mismatch: {path}"
                    )
            else:
                atomic_write(path, payload)

            r["status"] = "DONE"
            r["sha256"] = sha
            r["duration_sec"] = duration
            r["rms"] = rms
            r["peak"] = peak
            r["group_generation_sec"] = elapsed
            r["sec_per_sample"] = elapsed / 10.0

        atomic_json(MANIFEST, manifest)

        print(
            f"done {elapsed:.3f}s "
            f"{elapsed/10:.3f}s/sample"
        )

    counts = Counter(r["status"] for r in samples)

    print()
    print("=" * 72)
    print("STATUS")
    print("=" * 72)
    print(dict(counts))

    done = sum(
        1
        for r in samples
        if r["status"] == "DONE"
        and Path(r["output"]).exists()
    )

    print(f"files: {done}/600")


if __name__ == "__main__":
    main()
