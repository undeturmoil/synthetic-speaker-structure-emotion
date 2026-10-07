#!/usr/bin/env python3
# coding: utf-8

import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


ROOT = Path("/factory")
CAMPAIGN = "study1_35f100_v1"

INPUT = ROOT / "outputs" / CAMPAIGN
MANIFEST = ROOT / "state" / CAMPAIGN / "manifest.json"

OUT = (
    ROOT
    / "campaigns"
    / CAMPAIGN
    / "analysis_baseline"
)

OUT.mkdir(parents=True, exist_ok=True)

BASE_SNAPSHOT = (
    "/home/sgbaeck/project/qwen-voice/cache/huggingface/hub/"
    "models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/"
    "snapshots/fd4b254389122332181a7c3db7f27e918eec64e3"
)

HISTORICAL_TOP50 = (
    ROOT
    / "campaigns"
    / "phase2_six_emotion_v1"
    / "reports"
    / "dimension_diagnostics"
    / "top50_identity_stable_female.csv"
)

EMOTIONS = [
    "normal",
    "happy",
    "sad",
    "calm",
    "angry",
    "surprise",
]


def normalize(x):
    d = np.linalg.norm(
        x,
        axis=-1,
        keepdims=True,
    )
    return x / np.maximum(d, 1e-12)


def correct_rank(row, index):
    order = np.argsort(-row)

    return (
        int(np.where(order == index)[0][0])
        + 1
    )


def retrieval_metrics(X):
    # X = candidates × emotions × dimensions

    Z = normalize(X)

    same = []
    cross = []
    ranks = []
    margins = []

    pair_rows = []

    n = X.shape[0]

    for ea, eb in combinations(
        range(len(EMOTIONS)),
        2,
    ):
        sim = Z[:, ea] @ Z[:, eb].T

        diag = np.diag(sim)

        offmask = ~np.eye(
            n,
            dtype=bool,
        )

        off = sim[offmask]

        same.extend(diag.tolist())
        cross.extend(off.tolist())

        ranks_ab = []
        ranks_ba = []
        margins_ab = []
        margins_ba = []

        for i in range(n):
            r = correct_rank(
                sim[i],
                i,
            )

            ranks.append(r)
            ranks_ab.append(r)

            impostor = np.delete(
                sim[i],
                i,
            ).max()

            m = float(
                sim[i, i] - impostor
            )

            margins.append(m)
            margins_ab.append(m)

            r = correct_rank(
                sim[:, i],
                i,
            )

            ranks.append(r)
            ranks_ba.append(r)

            impostor = np.delete(
                sim[:, i],
                i,
            ).max()

            m = float(
                sim[i, i] - impostor
            )

            margins.append(m)
            margins_ba.append(m)

        pair_rows.append({
            "emotion_a": EMOTIONS[ea],
            "emotion_b": EMOTIONS[eb],
            "diag_mean": float(np.mean(diag)),
            "offdiag_mean": float(np.mean(off)),
            "gap": float(np.mean(diag) - np.mean(off)),
            "top1_ab": float(np.mean(np.array(ranks_ab) == 1)),
            "top1_ba": float(np.mean(np.array(ranks_ba) == 1)),
            "mrr_ab": float(np.mean(1.0 / np.array(ranks_ab))),
            "mrr_ba": float(np.mean(1.0 / np.array(ranks_ba))),
            "margin_ab": float(np.mean(margins_ab)),
            "margin_ba": float(np.mean(margins_ba)),
        })

    ranks = np.asarray(ranks)

    return {
        "n_candidates": n,
        "chance_top1": 1.0 / n,
        "same_mean": float(np.mean(same)),
        "cross_mean": float(np.mean(cross)),
        "gap": float(
            np.mean(same)
            - np.mean(cross)
        ),
        "top1": float(np.mean(ranks == 1)),
        "top5": float(np.mean(ranks <= 5)),
        "mrr": float(
            np.mean(1.0 / ranks)
        ),
        "mean_correct_vs_best_impostor_margin": float(
            np.mean(margins)
        ),
        "pair_rows": pair_rows,
    }


def decomposition(X):
    # balanced candidate × emotion panel

    grand = X.mean(axis=(0, 1))

    cmean = X.mean(axis=1)
    emean = X.mean(axis=0)

    ceffect = cmean - grand
    eeffect = emean - grand

    residual = (
        X
        - grand[None, None, :]
        - ceffect[:, None, :]
        - eeffect[None, :, :]
    )

    n = X.shape[0]
    k = X.shape[1]

    ss_total = float(
        np.sum(
            (X - grand) ** 2
        )
    )

    ss_candidate = float(
        k * np.sum(
            ceffect ** 2
        )
    )

    ss_emotion = float(
        n * np.sum(
            eeffect ** 2
        )
    )

    ss_residual = float(
        np.sum(
            residual ** 2
        )
    )

    return {
        "ss_total": ss_total,
        "candidate_share": ss_candidate / ss_total,
        "emotion_share": ss_emotion / ss_total,
        "residual_share": ss_residual / ss_total,
        "residual_definition": (
            "candidate-by-emotion interaction "
            "plus remaining unseparated variation"
        ),
    }


def candidate_stability(X, ids):
    Z = normalize(X)

    rows = []

    for i, cid in enumerate(ids):
        pairs = []

        worst_pair = None
        worst_value = 999.0

        for ea, eb in combinations(
            range(len(EMOTIONS)),
            2,
        ):
            value = float(
                np.dot(
                    Z[i, ea],
                    Z[i, eb],
                )
            )

            pairs.append(value)

            if value < worst_value:
                worst_value = value
                worst_pair = (
                    f"{EMOTIONS[ea]}"
                    f"-{EMOTIONS[eb]}"
                )

        rows.append({
            "candidate_id": cid,
            "pairwise_mean_cos": float(np.mean(pairs)),
            "pairwise_min_cos": float(np.min(pairs)),
            "pairwise_max_cos": float(np.max(pairs)),
            "worst_emotion_pair": worst_pair,
            "instability_mean": float(
                np.mean(
                    [1.0 - x for x in pairs]
                )
            ),
            "instability_worst": float(
                1.0 - np.min(pairs)
            ),
        })

    return rows


def load_top50():
    if not HISTORICAL_TOP50.exists():
        return None

    dims = []

    with HISTORICAL_TOP50.open(
        newline="",
        encoding="utf-8",
    ) as f:
        for row in csv.DictReader(f):
            dims.append(
                int(row["dimension"])
            )

    return np.asarray(
        dims[:50],
        dtype=np.int64,
    )


def write_csv(path, rows):
    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        w.writeheader()
        w.writerows(rows)


def stripped_metrics(m):
    return {
        k: v
        for k, v in m.items()
        if k != "pair_rows"
    }


def main():
    manifest = json.loads(
        MANIFEST.read_text(
            encoding="utf-8",
        )
    )

    samples = manifest["samples"]

    if len(samples) != 600:
        raise SystemExit(
            f"manifest has {len(samples)} samples"
        )

    if not all(
        x["status"] == "DONE"
        for x in samples
    ):
        raise SystemExit(
            "generation incomplete"
        )

    candidate_ids = [
        f"b{b:02d}p{p:02d}"
        for b in range(10)
        for p in range(10)
    ]

    candidate_to_index = {
        c: i
        for i, c in enumerate(
            candidate_ids
        )
    }

    emotion_to_index = {
        e: i
        for i, e in enumerate(
            EMOTIONS
        )
    }

    X = np.zeros(
        (100, 6, 2048),
        dtype=np.float32,
    )

    records = []

    print("Loading Base speaker encoder...")

    model = Qwen3TTSModel.from_pretrained(
        BASE_SNAPSHOT,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )

    for n, row in enumerate(
        samples,
        1,
    ):
        cid = row["candidate_id"]
        emotion = row["emotion"]

        ci = candidate_to_index[cid]
        ei = emotion_to_index[emotion]

        path = Path(row["output"])

        audio, sr = sf.read(
            path,
            dtype="float32",
            always_2d=False,
        )

        if sr != 24000:
            raise RuntimeError(
                f"{path}: sr={sr}"
            )

        if audio.ndim != 1:
            raise RuntimeError(
                f"{path}: shape={audio.shape}"
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
                f"{path}: embedding={emb.shape}"
            )

        X[ci, ei] = emb

        records.append({
            "candidate_id": cid,
            "batch_index": row["batch_index"],
            "batch_position": row["batch_position"],
            "split": row["split"],
            "emotion": emotion,
            "l2_norm": float(
                np.linalg.norm(emb)
            ),
            "path": str(path),
        })

        print(
            f"[{n:03d}/600] "
            f"{cid} {emotion}"
        )

    np.save(
        OUT / "embeddings_100x6x2048.npy",
        X,
    )

    write_csv(
        OUT / "embedding_records.csv",
        records,
    )

    # Frozen split established before analysis:
    # batch00-07 development = candidates 0..79
    # batch08-09 holdout = candidates 80..99

    dev_idx = np.arange(
        0,
        80,
        dtype=int,
    )

    hold_idx = np.arange(
        80,
        100,
        dtype=int,
    )

    X_dev = X[dev_idx]
    X_hold = X[hold_idx]

    ids_dev = [
        candidate_ids[i]
        for i in dev_idx
    ]

    ids_hold = [
        candidate_ids[i]
        for i in hold_idx
    ]

    raw_dev = retrieval_metrics(
        X_dev
    )

    raw_hold = retrieval_metrics(
        X_hold
    )

    dec_dev = decomposition(
        X_dev
    )

    dec_hold = decomposition(
        X_hold
    )

    stable_dev = candidate_stability(
        X_dev,
        ids_dev,
    )

    stable_hold = candidate_stability(
        X_hold,
        ids_hold,
    )

    write_csv(
        OUT / "raw_dev_pair_retrieval.csv",
        raw_dev["pair_rows"],
    )

    write_csv(
        OUT / "raw_holdout_pair_retrieval.csv",
        raw_hold["pair_rows"],
    )

    write_csv(
        OUT / "candidate_stability_dev.csv",
        stable_dev,
    )

    write_csv(
        OUT / "candidate_stability_holdout.csv",
        stable_hold,
    )

    top50 = load_top50()

    historical = None

    if top50 is not None:
        np.savetxt(
            OUT / "historical_female_top50.txt",
            top50,
            fmt="%d",
        )

        hist_dev = retrieval_metrics(
            X_dev[:, :, top50]
        )

        hist_hold = retrieval_metrics(
            X_hold[:, :, top50]
        )

        write_csv(
            OUT / "historical_top50_dev_pair_retrieval.csv",
            hist_dev["pair_rows"],
        )

        write_csv(
            OUT / "historical_top50_holdout_pair_retrieval.csv",
            hist_hold["pair_rows"],
        )

        historical = {
            "development": stripped_metrics(
                hist_dev
            ),
            "sealed_holdout": stripped_metrics(
                hist_hold
            ),
        }

    summary = {
        "campaign": CAMPAIGN,

        "split": {
            "development": {
                "batches": list(range(8)),
                "candidates": 80,
            },
            "sealed_holdout": {
                "batches": [8, 9],
                "candidates": 20,
            },
        },

        "raw_2048": {
            "development": stripped_metrics(
                raw_dev
            ),
            "sealed_holdout": stripped_metrics(
                raw_hold
            ),
        },

        "decomposition": {
            "development": dec_dev,
            "sealed_holdout": dec_hold,
        },

        "historical_female_top50": historical,
    }

    (OUT / "summary.json").write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("STUDY1 35F100 BASELINE")
    print("=" * 72)

    print("\n[DEVELOPMENT 80 / RAW]")
    for k, v in stripped_metrics(
        raw_dev
    ).items():
        print(f"{k:38s}: {v}")

    print("\n[SEALED HOLDOUT 20 / RAW]")
    for k, v in stripped_metrics(
        raw_hold
    ).items():
        print(f"{k:38s}: {v}")

    print("\n[DECOMPOSITION / DEVELOPMENT]")
    for k, v in dec_dev.items():
        print(f"{k:38s}: {v}")

    print("\n[DECOMPOSITION / HOLDOUT]")
    for k, v in dec_hold.items():
        print(f"{k:38s}: {v}")

    if historical is not None:
        print("\n[HISTORICAL TOP50 / HOLDOUT]")
        for k, v in historical[
            "sealed_holdout"
        ].items():
            print(f"{k:38s}: {v}")

    print()
    print("saved:", OUT / "summary.json")


if __name__ == "__main__":
    main()
