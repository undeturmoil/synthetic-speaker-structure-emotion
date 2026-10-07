#!/usr/bin/env python3
# coding: utf-8

import csv
import json
import os
import re
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

import study1_35f_plda_baseline as base


ROOT = Path("/factory")

DEV_EMB = (
    ROOT / "campaigns/study1_35f300_v1/"
    "analysis/embeddings_300x6x2048.npy"
)

CAMPAIGN = (
    ROOT / "campaigns/study1_35f100_external_v1"
)

WAV_ROOT = (
    ROOT / "outputs/study1_35f100_external_v1"
)

EXT_EMB = (
    CAMPAIGN /
    "analysis/embeddings_100x6x2048.npy"
)

MULTI_MODEL = (
    CAMPAIGN /
    "frozen_validation/frozen_transform.npz"
)

OUT = (
    CAMPAIGN /
    "bundle_recovery_v1"
)

EMOTIONS = [
    "normal",
    "happy",
    "sad",
    "calm",
    "angry",
    "surprise",
]

N = 100
E = 6
MAX_ITER = 30


def norm(X):
    return X / np.maximum(
        np.linalg.norm(
            X,
            axis=-1,
            keepdims=True,
        ),
        1e-12,
    )


# ------------------------------------------------------------
# WAV discovery
# ------------------------------------------------------------

CID_RE = re.compile(
    r"b(\d+)p(\d+)",
    re.I,
)


def discover_wavs():
    """
    Deterministic mapping for the external100 campaign.

    Candidate order:
      b30p00 ... b30p09
      b31p00 ... b31p09
      ...
      b39p00 ... b39p09

    Emotion order is exactly EMOTIONS.

    This mirrors the external embedding/vector naming scheme
    and avoids filesystem discovery ambiguity.
    """

    candidate_names = [
        f"b{b:02d}p{p:02d}"
        for b in range(30, 40)
        for p in range(10)
    ]

    if len(candidate_names) != N:
        raise RuntimeError(
            f"candidate count mismatch: {len(candidate_names)}"
        )

    paths = np.empty(
        (N, E),
        dtype=object,
    )

    missing = []

    for i, cid in enumerate(candidate_names):
        for e, emotion in enumerate(EMOTIONS):
            wav = (
                WAV_ROOT
                / emotion
                / f"{cid}.wav"
            )

            if not wav.is_file():
                missing.append(
                    str(wav)
                )

            paths[i, e] = wav

    if missing:
        print(
            f"Missing {len(missing)} WAV files."
        )

        for x in missing[:30]:
            print(" ", x)

        raise RuntimeError(
            "External WAV set is incomplete."
        )

    print(
        "WAV mapping verified:",
        f"{N * E} files"
    )

    print(
        "Candidate range:",
        candidate_names[0],
        "->",
        candidate_names[-1],
    )

    print(
        "WAV root:",
        WAV_ROOT,
    )

    return candidate_names, paths


# ------------------------------------------------------------
# Multivariate representation
# ------------------------------------------------------------

def load_multivariate(X):
    f = np.load(
        MULTI_MODEL
    )

    effect = f[
        "emotion_effect"
    ].astype(
        np.float64
    )

    grand = f[
        "grand"
    ].astype(
        np.float64
    )

    basis = f[
        "basis"
    ].astype(
        np.float64
    )

    Z = (
        X
        - effect[
            None,
            :,
            :
        ]
        - grand[
            None,
            None,
            :
        ]
    ) @ basis

    return norm(Z)


# ------------------------------------------------------------
# Final selected standard PLDA
# Fit ONLY on development300:
# corrected / PCA224 / alpha=.1
# ------------------------------------------------------------

def fit_selected_plda(
    Xdev,
    Xext,
):
    effect = base.emotion_effect(
        Xdev
    )

    D = base.apply_mode(
        Xdev,
        "corrected",
        effect,
    )

    T = base.apply_mode(
        Xext,
        "corrected",
        effect,
    )

    pca_mean, basis = (
        base.fit_pca(
            D,
            224,
        )
    )

    Dp = base.project_pca(
        D,
        pca_mean,
        basis,
    )

    Tp = base.project_pca(
        T,
        pca_mean,
        basis,
    )

    mu, B, W = (
        base.estimate_bw(
            Dp
        )
    )

    model = (
        base.fit_plda_from_bw(
            mu,
            B,
            W,
            0.1,
        )
    )

    Z = base.transform_plda(
        Tp,
        model,
    )

    return Z, model


# ------------------------------------------------------------
# Assignment helpers
# ------------------------------------------------------------

def cosine_objective(
    V,
    A,
):
    vals = []

    for e1, e2 in combinations(
        range(E),
        2,
    ):
        x = V[
            A[:, e1],
            e1,
            :
        ]

        y = V[
            A[:, e2],
            e2,
            :
        ]

        vals.extend(
            np.sum(
                x * y,
                axis=1,
            ).tolist()
        )

    return float(
        np.mean(vals)
    )


def optimize_cosine(
    V,
):
    """
    Balanced multi-partite matching.

    Each emotion sample is used exactly once.
    Six possible starting anchor emotions are tried.
    """

    best = None

    for anchor in range(E):
        A = np.empty(
            (N, E),
            dtype=int,
        )

        A[:, anchor] = np.arange(N)

        anchor_vec = V[
            :,
            anchor,
            :
        ]

        # Initial anchor-to-emotion Hungarian matching.
        for e in range(E):
            if e == anchor:
                continue

            S = (
                anchor_vec
                @ V[:, e, :].T
            )

            r, c = linear_sum_assignment(
                -S
            )

            A[r, e] = c

        # Iterative leave-one-emotion-out centroid matching.
        for _ in range(MAX_ITER):
            old = A.copy()

            for e in range(E):
                other = [
                    k
                    for k in range(E)
                    if k != e
                ]

                stack = np.stack([
                    V[
                        A[:, k],
                        k,
                        :
                    ]
                    for k in other
                ], axis=1)

                centroid = norm(
                    stack.mean(
                        axis=1
                    )
                )

                S = (
                    centroid
                    @ V[:, e, :].T
                )

                r, c = (
                    linear_sum_assignment(
                        -S
                    )
                )

                A[r, e] = c

            if np.array_equal(
                A,
                old,
            ):
                break

        obj = cosine_objective(
            V,
            A,
        )

        if (
            best is None
            or obj > best[
                "objective"
            ]
        ):
            best = {
                "anchor":
                    anchor,

                "objective":
                    obj,

                "assignment":
                    A.copy(),
            }

    return best


# ------------------------------------------------------------
# PLDA matching
# ------------------------------------------------------------

def plda_objective(
    Z,
    model,
    A,
):
    vals = []

    for qe in range(E):
        other = [
            e
            for e in range(E)
            if e != qe
        ]

        query = Z[
            A[:, qe],
            qe,
            :
        ]

        enroll = np.stack([
            Z[
                A[:, e],
                e,
                :
            ]
            for e in other
        ], axis=1).mean(
            axis=1
        )

        S = base.plda_score_matrix(
            query,
            enroll,
            model["lambda"],
            n_enroll=5,
        )

        vals.extend(
            np.diag(S).tolist()
        )

    return float(
        np.mean(vals)
    )


def optimize_plda(
    Z,
    model,
):
    best = None

    for anchor in range(E):
        A = np.empty(
            (N, E),
            dtype=int,
        )

        A[:, anchor] = np.arange(N)

        # Initial pairwise assignment.
        for e in range(E):
            if e == anchor:
                continue

            # candidate e = query rows
            # anchor candidate = enrollment cols
            S = base.plda_score_matrix(
                Z[:, e, :],
                Z[:, anchor, :],
                model["lambda"],
                n_enroll=1,
            ).T

            r, c = (
                linear_sum_assignment(
                    -S
                )
            )

            A[r, e] = c

        # Iterative assignment using mean of other 5.
        for _ in range(MAX_ITER):
            old = A.copy()

            for e in range(E):
                other = [
                    k
                    for k in range(E)
                    if k != e
                ]

                enrollment = (
                    np.stack([
                        Z[
                            A[:, k],
                            k,
                            :
                        ]
                        for k in other
                    ], axis=1)
                    .mean(axis=1)
                )

                # rows candidate e, cols bundle
                S = base.plda_score_matrix(
                    Z[:, e, :],
                    enrollment,
                    model["lambda"],
                    n_enroll=5,
                ).T

                r, c = (
                    linear_sum_assignment(
                        -S
                    )
                )

                A[r, e] = c

            if np.array_equal(
                A,
                old,
            ):
                break

        obj = plda_objective(
            Z,
            model,
            A,
        )

        if (
            best is None
            or obj > best[
                "objective"
            ]
        ):
            best = {
                "anchor":
                    anchor,

                "objective":
                    obj,

                "assignment":
                    A.copy(),
            }

    return best


# ------------------------------------------------------------
# Export
# ------------------------------------------------------------

def relative_symlink(
    src,
    dst,
):
    dst.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if dst.exists() or dst.is_symlink():
        dst.unlink()

    rel = os.path.relpath(
        src,
        start=dst.parent,
    )

    dst.symlink_to(
        rel
    )


def export_method(
    method_name,
    A,
    objective,
    candidate_names,
    wav_paths,
):
    root = (
        OUT /
        method_name
    )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for b in range(N):
        bundle_id = (
            f"voice_{b:03d}"
        )

        folder = (
            root /
            bundle_id
        )

        folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        row = {
            "bundle_id":
                bundle_id,
        }

        source_ids = []

        for e, emo in enumerate(
            EMOTIONS
        ):
            idx = int(
                A[b, e]
            )

            src = Path(
                wav_paths[
                    idx,
                    e,
                ]
            )

            sid = (
                candidate_names[idx]
            )

            source_ids.append(
                sid
            )

            dst = (
                folder /
                f"{emo}__{sid}.wav"
            )

            relative_symlink(
                src,
                dst,
            )

            row[
                f"{emo}_source"
            ] = sid

            row[
                f"{emo}_wav"
            ] = str(src)

        row[
            "unique_source_ids"
        ] = len(
            set(source_ids)
        )

        rows.append(
            row
        )

    csv_path = (
        root /
        "bundle_manifest.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    meta = {
        "method":
            method_name,

        "objective":
            float(objective),

        "n_bundles":
            N,

        "emotions":
            EMOTIONS,

        "note":
            (
                "WAV entries are relative symbolic "
                "links to original campaign files."
            ),
    }

    (
        root /
        "method.json"
    ).write_text(
        json.dumps(
            meta,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    Xdev = np.load(
        DEV_EMB
    ).astype(
        np.float64
    )

    X = np.load(
        EXT_EMB
    ).astype(
        np.float64
    )

    if X.shape != (
        N,
        E,
        2048,
    ):
        raise RuntimeError(
            X.shape
        )

    candidate_names, wav_paths = (
        discover_wavs()
    )

    # -----------------------------------------
    # Baseline: original operational candidate
    # -----------------------------------------
    operational = np.tile(
        np.arange(N)[:, None],
        (1, E),
    )

    export_method(
        "operational_seed_position",
        operational,
        float("nan"),
        candidate_names,
        wav_paths,
    )

    # -----------------------------------------
    # Frozen multivariate primary method
    # -----------------------------------------
    V = load_multivariate(
        X
    )

    multi = optimize_cosine(
        V
    )

    print()
    print(
        "Frozen multivariate:"
    )
    print(
        " anchor =",
        EMOTIONS[
            multi["anchor"]
        ]
    )
    print(
        " objective =",
        multi["objective"],
    )

    export_method(
        "frozen_multivariate",
        multi["assignment"],
        multi["objective"],
        candidate_names,
        wav_paths,
    )

    # -----------------------------------------
    # Final rank-aware PLDA baseline
    # -----------------------------------------
    Z, plda_model = (
        fit_selected_plda(
            Xdev,
            X,
        )
    )

    plda = optimize_plda(
        Z,
        plda_model,
    )

    print()
    print(
        "PLDA:"
    )
    print(
        " anchor =",
        EMOTIONS[
            plda["anchor"]
        ]
    )
    print(
        " objective =",
        plda["objective"],
    )

    export_method(
        "plda",
        plda["assignment"],
        plda["objective"],
        candidate_names,
        wav_paths,
    )

    # -----------------------------------------
    # Agreement between methods
    # -----------------------------------------
    agreement = float(
        np.mean(
            multi["assignment"]
            == plda["assignment"]
        )
    )

    report = {
        "multivariate": {
            "anchor":
                EMOTIONS[
                    multi["anchor"]
                ],

            "objective":
                multi[
                    "objective"
                ],
        },

        "plda": {
            "anchor":
                EMOTIONS[
                    plda["anchor"]
                ],

            "objective":
                plda[
                    "objective"
                ],
        },

        "elementwise_assignment_agreement":
            agreement,
    }

    (
        OUT /
        "bundle_recovery_report.json"
    ).write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Multivariate / PLDA "
        "element-wise agreement = "
        f"{agreement:.4f}"
    )

    print()
    print(
        "saved:",
        OUT,
    )


if __name__ == "__main__":
    main()
