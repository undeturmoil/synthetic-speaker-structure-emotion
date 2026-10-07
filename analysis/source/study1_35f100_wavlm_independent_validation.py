#!/usr/bin/env python3
# coding: utf-8

import json
from pathlib import Path

import numpy as np
import torch

from scipy.io import wavfile
from scipy.signal import resample_poly

from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioXVector,
)

import study1_35f100_bundle_recovery as br


ROOT = Path("/factory")

MODEL_ID = "microsoft/wavlm-base-plus-sv"

WAV_ROOT = (
    ROOT
    / "outputs/study1_35f100_external_v1"
)

OUT = (
    ROOT
    / "campaigns/study1_35f100_external_v1"
    / "wavlm_independent_validation"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

EMOTIONS = br.EMOTIONS

N = 100
E = 6
TARGET_SR = 16000
BATCH_SIZE = 8


def norm(x, axis=-1):
    return x / np.maximum(
        np.linalg.norm(
            x,
            axis=axis,
            keepdims=True,
        ),
        1e-12,
    )


def candidate_ids():
    return [
        f"b{b:02d}p{p:02d}"
        for b in range(30, 40)
        for p in range(10)
    ]


def load_audio(path):
    """
    Read PCM WAV without torchaudio/TorchCodec dependency
    and resample to WavLM's required 16 kHz.
    """

    sr, wav = wavfile.read(
        str(path)
    )

    # mono
    if wav.ndim == 2:
        wav = wav.mean(
            axis=1
        )

    # convert integer PCM to float32 [-1, 1]
    if np.issubdtype(
        wav.dtype,
        np.integer,
    ):
        info = np.iinfo(
            wav.dtype
        )

        scale = max(
            abs(info.min),
            abs(info.max),
        )

        wav = (
            wav.astype(np.float32)
            / float(scale)
        )

    else:
        wav = wav.astype(
            np.float32
        )

    if sr != TARGET_SR:
        # Our source WAVs are normally 24 kHz,
        # so 24k -> 16k becomes up=2, down=3.
        from math import gcd

        g = gcd(
            int(sr),
            int(TARGET_SR),
        )

        up = (
            TARGET_SR // g
        )

        down = (
            sr // g
        )

        wav = resample_poly(
            wav,
            up,
            down,
        ).astype(
            np.float32
        )

    return np.ascontiguousarray(
        wav,
        dtype=np.float32,
    )

def extract_wavlm():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "device:",
        device,
    )

    feature_extractor = (
        AutoFeatureExtractor.from_pretrained(
            MODEL_ID
        )
    )

    model = (
        AutoModelForAudioXVector.from_pretrained(
            MODEL_ID
        )
        .to(device)
        .eval()
    )

    commit_hash = getattr(
        model.config,
        "_commit_hash",
        None,
    )

    ids = candidate_ids()

    paths = []
    mapping = []

    for i, cid in enumerate(ids):
        for e, emotion in enumerate(
            EMOTIONS
        ):
            path = (
                WAV_ROOT
                / emotion
                / f"{cid}.wav"
            )

            if not path.is_file():
                raise RuntimeError(
                    f"missing: {path}"
                )

            paths.append(path)
            mapping.append(
                (i, e)
            )

    vectors = None

    for start in range(
        0,
        len(paths),
        BATCH_SIZE,
    ):
        batch_paths = paths[
            start:start+BATCH_SIZE
        ]

        audio = [
            load_audio(p)
            for p in batch_paths
        ]

        inputs = (
            feature_extractor(
                audio,
                sampling_rate=TARGET_SR,
                padding=True,
                return_tensors="pt",
            )
        )

        inputs = {
            k: v.to(device)
            for k, v in inputs.items()
        }

        with torch.inference_mode():
            out = model(
                **inputs
            )

            emb = (
                torch.nn.functional.normalize(
                    out.embeddings,
                    dim=-1,
                )
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        if vectors is None:
            vectors = np.empty(
                (
                    N,
                    E,
                    emb.shape[1],
                ),
                dtype=np.float32,
            )

        for j in range(
            emb.shape[0]
        ):
            i, e = mapping[
                start+j
            ]

            vectors[
                i,
                e,
            ] = emb[j]

        done = min(
            start+BATCH_SIZE,
            len(paths),
        )

        print(
            f"embedded {done}/600"
        )

    path = (
        OUT
        / "wavlm_embeddings_100x6.npy"
    )

    np.save(
        path,
        vectors,
    )

    return (
        vectors.astype(
            np.float64
        ),
        commit_hash,
    )


def ranks_from_scores(S):
    order = np.argsort(
        -S,
        axis=1,
    )

    target = np.arange(
        S.shape[0]
    )[:, None]

    rank = (
        np.argmax(
            order == target,
            axis=1,
        )
        + 1
    )

    return rank


def retrieval_metrics(W):
    ranks = []

    for qe in range(E):
        others = [
            e
            for e in range(E)
            if e != qe
        ]

        query = W[
            :,
            qe,
            :
        ]

        enrollment = norm(
            W[
                :,
                others,
                :
            ].mean(axis=1)
        )

        S = (
            query
            @ enrollment.T
        )

        ranks.extend(
            ranks_from_scores(
                S
            ).tolist()
        )

    ranks = np.asarray(
        ranks
    )

    return {
        "mrr":
            float(
                np.mean(
                    1.0 / ranks
                )
            ),

        "top1":
            float(
                np.mean(
                    ranks == 1
                )
            ),

        "top5":
            float(
                np.mean(
                    ranks <= 5
                )
            ),
    }


def same_cross_geometry(W):
    same = []
    cross = []

    for e1 in range(E):
        for e2 in range(
            e1+1,
            E,
        ):
            S = (
                W[:, e1, :]
                @ W[:, e2, :].T
            )

            same.extend(
                np.diag(S).tolist()
            )

            mask = ~np.eye(
                N,
                dtype=bool,
            )

            cross.extend(
                S[mask].tolist()
            )

    same = np.asarray(same)
    cross = np.asarray(cross)

    gap = (
        same.mean()
        - cross.mean()
    )

    pooled_sd = np.sqrt(
        0.5
        * (
            same.var(ddof=1)
            + cross.var(ddof=1)
        )
    )

    dprime = (
        gap
        / max(
            pooled_sd,
            1e-12,
        )
    )

    return {
        "same_mean":
            float(
                same.mean()
            ),

        "cross_mean":
            float(
                cross.mean()
            ),

        "gap":
            float(gap),

        "dprime":
            float(dprime),
    }


def frozen_multivariate_scores():
    """
    Reproduce the already-frozen primary method.
    WavLM is NOT involved here.
    """

    X = np.load(
        br.EXT_EMB
    ).astype(
        np.float64
    )

    V = br.load_multivariate(
        X
    )

    predictions = np.empty(
        (N, E),
        dtype=int,
    )

    all_ranks = []

    for qe in range(E):
        others = [
            e
            for e in range(E)
            if e != qe
        ]

        query = V[
            :,
            qe,
            :
        ]

        enrollment = br.norm(
            V[
                :,
                others,
                :
            ].mean(axis=1)
        )

        S = (
            query
            @ enrollment.T
        )

        predictions[
            :,
            qe,
        ] = np.argmax(
            S,
            axis=1,
        )

        all_ranks.extend(
            ranks_from_scores(
                S
            ).tolist()
        )

    all_ranks = np.asarray(
        all_ranks
    )

    mrr = float(
        np.mean(
            1.0
            / all_ranks
        )
    )

    return predictions, mrr


def exact_sign_flip_p(values):
    """
    Two-sided exact sign-flip test over batch-level means.
    At most 10 clusters -> 2^10 exact enumeration.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    observed = abs(
        values.mean()
    )

    n = len(values)

    count = 0
    total = 2 ** n

    for mask in range(total):
        signs = np.ones(n)

        for j in range(n):
            if (
                mask
                >> j
            ) & 1:
                signs[j] = -1.0

        stat = abs(
            np.mean(
                values
                * signs
            )
        )

        if stat >= (
            observed
            - 1e-15
        ):
            count += 1

    return (
        count
        / total
    )


def multivariate_adjudication(
    W,
    predictions,
):
    """
    For cases where frozen multivariate disagrees
    with the operational seed-position candidate:

      Compare WavLM cosine to:
        A) operational candidate enrollment
        B) multivariate-selected candidate enrollment

    Positive delta means independent WavLM agrees
    more strongly with the multivariate reassignment.
    """

    deltas = []
    records = []

    batch_values = {
        b: []
        for b in range(30, 40)
    }

    for qe in range(E):
        others = [
            e
            for e in range(E)
            if e != qe
        ]

        enrollment = norm(
            W[
                :,
                others,
                :
            ].mean(axis=1)
        )

        query = W[
            :,
            qe,
            :
        ]

        for i in range(N):
            pred = int(
                predictions[
                    i,
                    qe,
                ]
            )

            if pred == i:
                continue

            s_oper = float(
                query[i]
                @ enrollment[i]
            )

            s_multi = float(
                query[i]
                @ enrollment[pred]
            )

            delta = (
                s_multi
                - s_oper
            )

            batch = (
                30
                + i // 10
            )

            batch_values[
                batch
            ].append(
                delta
            )

            deltas.append(
                delta
            )

            records.append({
                "query_candidate":
                    i,

                "query_emotion":
                    EMOTIONS[qe],

                "multivariate_choice":
                    pred,

                "wavlm_operational":
                    s_oper,

                "wavlm_multivariate":
                    s_multi,

                "delta":
                    delta,
            })

    deltas = np.asarray(
        deltas
    )

    batch_means = []

    for b in range(30, 40):
        vals = np.asarray(
            batch_values[b]
        )

        if len(vals) == 0:
            raise RuntimeError(
                f"no disagreements in batch {b}"
            )

        batch_means.append(
            float(
                vals.mean()
            )
        )

    batch_means = np.asarray(
        batch_means
    )

    nonzero = (
        np.abs(deltas)
        > 1e-12
    )

    prefer_multi = float(
        np.mean(
            deltas[nonzero]
            > 0
        )
    )

    return {
        "n_disagreements":
            int(
                len(deltas)
            ),

        "wavlm_prefers_multivariate_fraction":
            prefer_multi,

        "mean_delta":
            float(
                deltas.mean()
            ),

        "median_delta":
            float(
                np.median(
                    deltas
                )
            ),

        "batch_mean_delta":
            batch_means.tolist(),

        "batch_positive_fraction":
            float(
                np.mean(
                    batch_means > 0
                )
            ),

        "exact_batch_signflip_p":
            float(
                exact_sign_flip_p(
                    batch_means
                )
            ),

        "records":
            records,
    }


def main():
    cache = (
        OUT
        / "wavlm_embeddings_100x6.npy"
    )

    commit_hash = None

    if cache.exists():
        print(
            "loading cached WavLM embeddings:",
            cache,
        )

        W = np.load(
            cache
        ).astype(
            np.float64
        )
    else:
        W, commit_hash = (
            extract_wavlm()
        )

    W = norm(W)

    print(
        "WavLM shape:",
        W.shape,
    )

    operational = (
        retrieval_metrics(
            W
        )
    )

    geometry = (
        same_cross_geometry(
            W
        )
    )

    predictions, frozen_mrr = (
        frozen_multivariate_scores()
    )

    print()
    print(
        "Frozen multivariate MRR check:",
        f"{frozen_mrr:.4f}",
    )

    # Protect against accidentally changing
    # the already-reported frozen method.
    if abs(
        frozen_mrr
        - 0.2856
    ) > 0.003:
        raise RuntimeError(
            "Frozen multivariate reproduction "
            f"failed: MRR={frozen_mrr}"
        )

    adjudication = (
        multivariate_adjudication(
            W,
            predictions,
        )
    )

    report = {
        "status":
            "INDEPENDENT SPEAKER-VERIFIER VALIDATION",

        "verifier": {
            "model_id":
                MODEL_ID,

            "resolved_commit":
                commit_hash,

            "sampling_rate":
                TARGET_SR,

            "use":
                (
                    "L2-normalized X-vector "
                    "embedding with cosine similarity"
                ),
        },

        "wavlm_operational_identity": {
            **operational,
            **geometry,
        },

        "frozen_primary_method": {
            "multivariate_external_mrr_check":
                frozen_mrr,
        },

        "independent_adjudication":
            adjudication,
    }

    report_path = (
        OUT
        / "wavlm_independent_validation.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 90)
    print(
        "WAVLM OPERATIONAL-ID RETRIEVAL"
    )
    print("=" * 90)

    print(
        "MRR  :",
        f"{operational['mrr']:.4f}"
    )

    print(
        "Top1 :",
        f"{operational['top1']:.4f}"
    )

    print(
        "Top5 :",
        f"{operational['top5']:.4f}"
    )

    print(
        "same/cross gap:",
        f"{geometry['gap']:.4f}"
    )

    print(
        "d-prime:",
        f"{geometry['dprime']:.4f}"
    )

    print()
    print("=" * 90)
    print(
        "INDEPENDENT ADJUDICATION OF "
        "MULTIVARIATE DISAGREEMENTS"
    )
    print("=" * 90)

    print(
        "n disagreements:",
        adjudication[
            "n_disagreements"
        ]
    )

    print(
        "WavLM prefers multivariate:",
        f"{adjudication['wavlm_prefers_multivariate_fraction']:.4f}"
    )

    print(
        "mean cosine delta:",
        f"{adjudication['mean_delta']:.6f}"
    )

    print(
        "median cosine delta:",
        f"{adjudication['median_delta']:.6f}"
    )

    print(
        "positive query-batch means:",
        f"{adjudication['batch_positive_fraction']:.4f}"
    )

    print(
        "exact batch sign-flip p:",
        f"{adjudication['exact_batch_signflip_p']:.6f}"
    )

    print()
    print(
        "saved:",
        report_path,
    )


if __name__ == "__main__":
    main()
