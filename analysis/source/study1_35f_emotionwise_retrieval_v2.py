#!/usr/bin/env python3

from pathlib import Path
import csv
import json
import math

import numpy as np

# ------------------------------------------------------------
# IMPORT THE HISTORICAL IMPLEMENTATION DIRECTLY.
# No reimplementation of the multivariate model.
# ------------------------------------------------------------

from study1_35f300_multivariate_cv import (
    prepare_training,
    fit_gamma,
    correct,
    project,
    loeo,
    norm,
    rank,
)


ROOT = Path("/factory")

DEV = (
    ROOT
    / "campaigns/study1_35f300_v1"
    / "analysis/embeddings_300x6x2048.npy"
)

EXT = (
    ROOT
    / "campaigns/study1_35f100_external_v1"
    / "analysis/embeddings_100x6x2048.npy"
)

DEV_REPORT = (
    ROOT
    / "campaigns/study1_35f300_v1"
    / "multivariate_cv/report.json"
)

DEV_GRID = (
    ROOT
    / "campaigns/study1_35f300_v1"
    / "multivariate_cv/aggregate_grid.csv"
)

EXT_REPORT = (
    ROOT
    / "campaigns/study1_35f100_external_v1"
    / "frozen_validation/report.json"
)

FROZEN_TRANSFORM = (
    ROOT
    / "campaigns/study1_35f100_external_v1"
    / "frozen_validation/frozen_transform.npz"
)

OUT = (
    ROOT
    / "campaigns/study1_35f300_v1"
    / "emotionwise_retrieval_v2"
)

OUT.mkdir(parents=True, exist_ok=True)

EMOTIONS = [
    "normal",
    "happy",
    "sad",
    "calm",
    "angry",
    "surprise",
]

GAMMA = 0.1
DIM = 128

FOLDS = [
    [0, 5, 10, 15, 20, 25],
    [1, 6, 11, 16, 21, 26],
    [2, 7, 12, 17, 22, 27],
    [3, 8, 13, 18, 23, 28],
    [4, 9, 14, 19, 24, 29],
]

# Since we use the original code/data, reproduction should be essentially exact.
REPRO_TOL = 1e-10


# ============================================================
# Helpers
# ============================================================

def batch_indices(batches):
    idx = []
    for b in batches:
        for p in range(10):
            idx.append(b * 10 + p)
    return np.asarray(idx, dtype=int)


def get_cv_indices():
    all_idx = np.arange(300, dtype=int)
    out = []

    for valid_batches in FOLDS:
        valid_idx = batch_indices(valid_batches)

        mask = np.ones(300, dtype=bool)
        mask[valid_idx] = False

        train_idx = all_idx[mask]

        out.append(
            (train_idx, valid_idx)
        )

    return out


# ============================================================
# EXACT LOEO LOGIC + emotion-specific split
#
# Historical loeo():
#   1. normalize every utterance
#   2. average the five normalized enrollment utterances
#   3. normalize the enrollment mean again
#   4. cosine ranking
#
# This is deliberately different from averaging raw vectors first.
# ============================================================

def loeo_by_emotion(X):
    Z = norm(X)

    n = Z.shape[0]

    result = {}
    pooled_ranks = []

    for qe, emotion in enumerate(EMOTIONS):

        other = [
            e for e in range(6)
            if e != qe
        ]

        enroll = norm(
            Z[:, other, :].mean(
                axis=1
            )
        )

        sim = (
            Z[:, qe, :]
            @ enroll.T
        )

        ranks = []
        margins = []

        for i in range(n):
            r = rank(
                sim[i],
                i,
            )

            ranks.append(r)

            imp = np.delete(
                sim[i],
                i,
            ).max()

            margins.append(
                float(
                    sim[i, i]
                    - imp
                )
            )

        ranks = np.asarray(
            ranks,
            dtype=int,
        )

        pooled_ranks.append(ranks)

        result[emotion] = {
            "n": int(n),
            "mrr": float(
                np.mean(
                    1.0 / ranks
                )
            ),
            "top1": float(
                np.mean(
                    ranks == 1
                )
            ),
            "top5": float(
                np.mean(
                    ranks <= 5
                )
            ),
            "mean_rank": float(
                np.mean(ranks)
            ),
            "median_rank": float(
                np.median(ranks)
            ),
            "margin": float(
                np.mean(margins)
            ),
            "ranks": ranks,
        }

    all_ranks = np.concatenate(
        pooled_ranks
    )

    aggregate = {
        "mrr": float(
            np.mean(
                1.0 / all_ranks
            )
        ),
        "top1": float(
            np.mean(
                all_ranks == 1
            )
        ),
        "top5": float(
            np.mean(
                all_ranks <= 5
            )
        ),
    }

    return aggregate, result


# ============================================================
# Cross-emotion geometry
# ============================================================

def geometry_arrays(X):
    """
    Return scalar distributions for every directed emotion pair.

    Normalization follows the historical representation:
    each utterance is L2-normalized first.
    """
    Z = norm(X)

    n = Z.shape[0]

    same = {}
    different = {}

    for ea in range(6):
        for eb in range(6):

            S = (
                Z[:, ea, :]
                @ Z[:, eb, :].T
            )

            key = (
                EMOTIONS[ea],
                EMOTIONS[eb],
            )

            same[key] = np.diag(
                S
            ).copy()

            mask = ~np.eye(
                n,
                dtype=bool,
            )

            different[key] = (
                S[mask].copy()
            )

    return same, different


def summarize_geometry(
    same_blocks,
    different_blocks,
):
    same_matrix = np.zeros(
        (6, 6),
        dtype=float,
    )

    different_matrix = np.zeros(
        (6, 6),
        dtype=float,
    )

    gap_matrix = np.zeros(
        (6, 6),
        dtype=float,
    )

    dprime_matrix = np.zeros(
        (6, 6),
        dtype=float,
    )

    for ea, emo_a in enumerate(EMOTIONS):
        for eb, emo_b in enumerate(EMOTIONS):

            key = (
                emo_a,
                emo_b,
            )

            same = np.concatenate(
                [
                    block[key]
                    for block
                    in same_blocks
                ]
            )

            different = np.concatenate(
                [
                    block[key]
                    for block
                    in different_blocks
                ]
            )

            ms = float(
                same.mean()
            )

            md = float(
                different.mean()
            )

            vs = float(
                same.var(
                    ddof=1
                )
            )

            vd = float(
                different.var(
                    ddof=1
                )
            )

            pooled_sd = math.sqrt(
                max(
                    0.5
                    * (
                        vs
                        + vd
                    ),
                    1e-30,
                )
            )

            same_matrix[
                ea,
                eb,
            ] = ms

            different_matrix[
                ea,
                eb,
            ] = md

            gap_matrix[
                ea,
                eb,
            ] = ms - md

            dprime_matrix[
                ea,
                eb,
            ] = (
                ms - md
            ) / pooled_sd

    return {
        "same": same_matrix,
        "different":
            different_matrix,
        "gap": gap_matrix,
        "dprime":
            dprime_matrix,
    }


# ============================================================
# Aggregation across CV folds
# ============================================================

def combine_per_emotion(
    fold_results,
):
    out = {}

    for emotion in EMOTIONS:

        ranks = np.concatenate(
            [
                fr[emotion]["ranks"]
                for fr in fold_results
            ]
        )

        out[emotion] = {
            "n": int(
                len(ranks)
            ),
            "mrr": float(
                np.mean(
                    1.0 / ranks
                )
            ),
            "top1": float(
                np.mean(
                    ranks == 1
                )
            ),
            "top5": float(
                np.mean(
                    ranks <= 5
                )
            ),
            "mean_rank": float(
                np.mean(ranks)
            ),
            "median_rank": float(
                np.median(ranks)
            ),
        }

    all_ranks = np.concatenate(
        [
            fr[e]["ranks"]
            for fr in fold_results
            for e in EMOTIONS
        ]
    )

    aggregate = {
        "mrr": float(
            np.mean(
                1.0 / all_ranks
            )
        ),
        "top1": float(
            np.mean(
                all_ranks == 1
            )
        ),
        "top5": float(
            np.mean(
                all_ranks <= 5
            )
        ),
    }

    return aggregate, out


# ============================================================
# Historical stored values
# ============================================================

def load_historical():

    dev_report = json.loads(
        DEV_REPORT.read_text(
            encoding="utf-8"
        )
    )

    ext_report = json.loads(
        EXT_REPORT.read_text(
            encoding="utf-8"
        )
    )

    baseline = {
        row["method"]: row
        for row
        in dev_report[
            "baseline_aggregate"
        ]
    }

    expected = {
        "dev_raw":
            float(
                baseline[
                    "raw_2048"
                ][
                    "loeo_mrr_mean"
                ]
            ),

        "dev_corrected":
            float(
                baseline[
                    "corrected_2048"
                ][
                    "loeo_mrr_mean"
                ]
            ),

        "ext_raw":
            float(
                ext_report[
                    "results"
                ][
                    "raw_2048"
                ][
                    "loeo"
                ][
                    "mrr"
                ]
            ),

        "ext_corrected":
            float(
                ext_report[
                    "results"
                ][
                    "corrected_2048"
                ][
                    "loeo"
                ][
                    "mrr"
                ]
            ),

        "ext_multi128":
            float(
                ext_report[
                    "results"
                ][
                    "frozen_multivariate_128"
                ][
                    "loeo"
                ][
                    "mrr"
                ]
            ),
    }

    # Read exact historical gamma=.1, dim=128 CV result.
    found = None

    with DEV_GRID.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            if (
                abs(
                    float(
                        row["gamma"]
                    )
                    - GAMMA
                )
                < 1e-12
                and int(
                    float(
                        row["dim"]
                    )
                )
                == DIM
            ):
                found = row
                break

    if found is None:
        raise RuntimeError(
            "Historical gamma=.1 dim=128 "
            "row not found in aggregate_grid.csv"
        )

    expected[
        "dev_multi128"
    ] = float(
        found[
            "loeo_mrr_mean"
        ]
    )

    return expected


# ============================================================
# Reproduction checks
# ============================================================

def check(
    label,
    actual,
    expected,
    tol=REPRO_TOL,
):
    delta = (
        actual
        - expected
    )

    ok = (
        abs(delta)
        <= tol
    )

    print(
        f"{label:<32}"
        f" actual={actual:.12f}"
        f" expected={expected:.12f}"
        f" delta={delta:+.3e}"
        f"  {'OK' if ok else 'FAIL'}"
    )

    return ok


def print_table(
    title,
    methods,
):
    print()
    print("=" * 126)
    print(title)
    print("=" * 126)

    print(
        f"{'emotion':<12}"
        f"{'raw MRR':>10}"
        f"{'corr MRR':>11}"
        f"{'multi MRR':>12}"
        f"{'multi-raw':>12}"
        f"{'multi-corr':>12}"
        f"{'raw T1':>10}"
        f"{'corr T1':>10}"
        f"{'multi T1':>10}"
        f"{'raw T5':>10}"
        f"{'corr T5':>10}"
        f"{'multi T5':>10}"
    )

    for emotion in EMOTIONS:

        r = methods[
            "raw"
        ][emotion]

        c = methods[
            "corrected"
        ][emotion]

        m = methods[
            "multivariate"
        ][emotion]

        print(
            f"{emotion:<12}"
            f"{r['mrr']:>10.4f}"
            f"{c['mrr']:>11.4f}"
            f"{m['mrr']:>12.4f}"
            f"{m['mrr']-r['mrr']:>12.4f}"
            f"{m['mrr']-c['mrr']:>12.4f}"
            f"{r['top1']:>10.4f}"
            f"{c['top1']:>10.4f}"
            f"{m['top1']:>10.4f}"
            f"{r['top5']:>10.4f}"
            f"{c['top5']:>10.4f}"
            f"{m['top5']:>10.4f}"
        )


def print_matrix(
    title,
    M,
):
    print()
    print(title)

    print(
        f"{'':12s}"
        + "".join(
            f"{e[:8]:>11s}"
            for e in EMOTIONS
        )
    )

    for i, emotion in enumerate(
        EMOTIONS
    ):
        print(
            f"{emotion:<12}"
            + "".join(
                f"{M[i,j]:>11.4f}"
                for j in range(6)
            )
        )


def save_matrix(
    path,
    M,
):
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        w = csv.writer(f)

        w.writerow(
            ["emotion"]
            + EMOTIONS
        )

        for i, emotion in enumerate(
            EMOTIONS
        ):
            w.writerow(
                [emotion]
                + [
                    float(
                        M[i,j]
                    )
                    for j in range(6)
                ]
            )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 126)
    print(
        "STUDY1 35F - EMOTION-WISE RETRIEVAL V2"
    )
    print(
        "USES HISTORICAL MULTIVARIATE IMPLEMENTATION DIRECTLY"
    )
    print("=" * 126)

    Xdev = np.load(
        DEV
    ).astype(
        np.float64
    )

    Xext = np.load(
        EXT
    ).astype(
        np.float64
    )

    if Xdev.shape != (
        300,
        6,
        2048,
    ):
        raise RuntimeError(
            Xdev.shape
        )

    if Xext.shape != (
        100,
        6,
        2048,
    ):
        raise RuntimeError(
            Xext.shape
        )

    expected = (
        load_historical()
    )

    print()
    print("Historical expected MRR:")
    for k, v in expected.items():
        print(
            f"  {k:<18s}: "
            f"{v:.12f}"
        )

    # --------------------------------------------------------
    # Development 5-fold OOF
    # --------------------------------------------------------

    print()
    print("=" * 126)
    print(
        "1. DEVELOPMENT 300 - ORIGINAL 5-FOLD LOGIC"
    )
    print("=" * 126)

    raw_fold_per = []
    corr_fold_per = []
    multi_fold_per = []

    raw_orig_fold = []
    corr_orig_fold = []
    multi_orig_fold = []

    geometry_blocks = {
        "raw": {
            "same": [],
            "different": [],
        },
        "corrected": {
            "same": [],
            "different": [],
        },
        "multivariate": {
            "same": [],
            "different": [],
        },
    }

    for fi, (
        train_idx,
        valid_idx,
    ) in enumerate(
        get_cv_indices(),
        start=1,
    ):

        Xtrain = Xdev[
            train_idx
        ]

        Xvalid = Xdev[
            valid_idx
        ]

        prep = prepare_training(
            Xtrain
        )

        model = fit_gamma(
            prep,
            GAMMA,
        )

        raw = Xvalid

        corr = correct(
            Xvalid,
            prep,
        )

        multi = project(
            Xvalid,
            prep,
            model["basis"],
            DIM,
        )

        # Exact original aggregate function.
        Lraw = loeo(raw)
        Lcorr = loeo(corr)
        Lmulti = loeo(multi)

        raw_orig_fold.append(
            Lraw["mrr"]
        )
        corr_orig_fold.append(
            Lcorr["mrr"]
        )
        multi_orig_fold.append(
            Lmulti["mrr"]
        )

        # New split-by-emotion function.
        araw, praw = (
            loeo_by_emotion(
                raw
            )
        )

        acorr, pcorr = (
            loeo_by_emotion(
                corr
            )
        )

        amulti, pmulti = (
            loeo_by_emotion(
                multi
            )
        )

        # Internal validation: emotion split must recombine
        # exactly to historical loeo().
        for name, a, L in [
            (
                "raw",
                araw,
                Lraw,
            ),
            (
                "corrected",
                acorr,
                Lcorr,
            ),
            (
                "multivariate",
                amulti,
                Lmulti,
            ),
        ]:
            if abs(
                a["mrr"]
                - L["mrr"]
            ) > 1e-12:
                raise RuntimeError(
                    f"Fold {fi} "
                    f"{name} emotion split "
                    f"does not reproduce loeo(): "
                    f"{a['mrr']} vs "
                    f"{L['mrr']}"
                )

        raw_fold_per.append(
            praw
        )
        corr_fold_per.append(
            pcorr
        )
        multi_fold_per.append(
            pmulti
        )

        for name, Y in [
            (
                "raw",
                raw,
            ),
            (
                "corrected",
                corr,
            ),
            (
                "multivariate",
                multi,
            ),
        ]:
            s, d = (
                geometry_arrays(
                    Y
                )
            )

            geometry_blocks[
                name
            ][
                "same"
            ].append(s)

            geometry_blocks[
                name
            ][
                "different"
            ].append(d)

        print(
            f"fold{fi}: "
            f"raw={Lraw['mrr']:.6f} "
            f"corr={Lcorr['mrr']:.6f} "
            f"multi128={Lmulti['mrr']:.6f}"
        )

    dev_raw_agg, dev_raw_per = (
        combine_per_emotion(
            raw_fold_per
        )
    )

    dev_corr_agg, dev_corr_per = (
        combine_per_emotion(
            corr_fold_per
        )
    )

    dev_multi_agg, dev_multi_per = (
        combine_per_emotion(
            multi_fold_per
        )
    )

    # Historical report aggregated fold metrics.
    dev_raw_historical_style = float(
        np.mean(
            raw_orig_fold
        )
    )

    dev_corr_historical_style = float(
        np.mean(
            corr_orig_fold
        )
    )

    dev_multi_historical_style = float(
        np.mean(
            multi_orig_fold
        )
    )

    # --------------------------------------------------------
    # External: original implementation
    # --------------------------------------------------------

    print()
    print("=" * 126)
    print(
        "2. EXTERNAL 100 - ORIGINAL FROZEN LOGIC"
    )
    print("=" * 126)

    prep_full = (
        prepare_training(
            Xdev
        )
    )

    model_full = (
        fit_gamma(
            prep_full,
            GAMMA,
        )
    )

    ext_raw = Xext

    ext_corr = correct(
        Xext,
        prep_full,
    )

    ext_multi = project(
        Xext,
        prep_full,
        model_full[
            "basis"
        ],
        DIM,
    )

    L_ext_raw = loeo(
        ext_raw
    )

    L_ext_corr = loeo(
        ext_corr
    )

    L_ext_multi = loeo(
        ext_multi
    )

    ext_raw_agg, ext_raw_per = (
        loeo_by_emotion(
            ext_raw
        )
    )

    ext_corr_agg, ext_corr_per = (
        loeo_by_emotion(
            ext_corr
        )
    )

    ext_multi_agg, ext_multi_per = (
        loeo_by_emotion(
            ext_multi
        )
    )

    # Per-emotion splitter must reconstruct original aggregate.
    for name, A, L in [
        (
            "raw",
            ext_raw_agg,
            L_ext_raw,
        ),
        (
            "corrected",
            ext_corr_agg,
            L_ext_corr,
        ),
        (
            "multivariate",
            ext_multi_agg,
            L_ext_multi,
        ),
    ]:
        if abs(
            A["mrr"]
            - L["mrr"]
        ) > 1e-12:
            raise RuntimeError(
                f"External {name} "
                f"emotion split mismatch"
            )

    # --------------------------------------------------------
    # Frozen NPZ provenance audit
    # --------------------------------------------------------

    frozen = np.load(
        FROZEN_TRANSFORM
    )

    frozen_emotion_effect = (
        frozen[
            "emotion_effect"
        ].astype(
            np.float64
        )
    )

    frozen_grand = (
        frozen[
            "grand"
        ].astype(
            np.float64
        )
    )

    frozen_basis = (
        frozen[
            "basis"
        ].astype(
            np.float64
        )
    )

    ext_multi_saved = (
        (
            Xext
            - frozen_emotion_effect[
                None,
                :,
                :,
            ]
            - frozen_grand[
                None,
                None,
                :,
            ]
        )
        @ frozen_basis
    )

    saved_loeo = loeo(
        ext_multi_saved
    )

    print(
        "External original-code MRR : "
        f"{L_ext_multi['mrr']:.12f}"
    )

    print(
        "Saved frozen-NPZ MRR       : "
        f"{saved_loeo['mrr']:.12f}"
    )

    # Float32 archival transform can differ at tiny numerical level,
    # but ranking should ordinarily remain identical.
    saved_transform_delta = abs(
        saved_loeo["mrr"]
        - L_ext_multi["mrr"]
    )

    # --------------------------------------------------------
    # Reproduction gate
    # --------------------------------------------------------

    print()
    print("=" * 126)
    print(
        "3. HISTORICAL REPRODUCTION GATE"
    )
    print("=" * 126)

    checks = []

    checks.append(
        check(
            "DEV raw",
            dev_raw_historical_style,
            expected[
                "dev_raw"
            ],
        )
    )

    checks.append(
        check(
            "DEV corrected",
            dev_corr_historical_style,
            expected[
                "dev_corrected"
            ],
        )
    )

    checks.append(
        check(
            "DEV multi128",
            dev_multi_historical_style,
            expected[
                "dev_multi128"
            ],
        )
    )

    checks.append(
        check(
            "EXT raw",
            L_ext_raw[
                "mrr"
            ],
            expected[
                "ext_raw"
            ],
        )
    )

    checks.append(
        check(
            "EXT corrected",
            L_ext_corr[
                "mrr"
            ],
            expected[
                "ext_corrected"
            ],
        )
    )

    checks.append(
        check(
            "EXT multi128",
            L_ext_multi[
                "mrr"
            ],
            expected[
                "ext_multi128"
            ],
        )
    )

    if not all(checks):
        print()
        print("!" * 126)
        print(
            "REPRODUCTION FAILED. "
            "STOPPING BEFORE INTERPRETATION."
        )
        print("!" * 126)
        raise SystemExit(2)

    print()
    print(
        "ALL HISTORICAL RESULTS REPRODUCED."
    )

    # --------------------------------------------------------
    # Emotion-specific results
    # --------------------------------------------------------

    dev_methods = {
        "raw": dev_raw_per,
        "corrected":
            dev_corr_per,
        "multivariate":
            dev_multi_per,
    }

    ext_methods = {
        "raw": ext_raw_per,
        "corrected":
            ext_corr_per,
        "multivariate":
            ext_multi_per,
    }

    print_table(
        "4A. DEVELOPMENT 300 - EMOTION-SPECIFIC OOF RETRIEVAL",
        dev_methods,
    )

    print_table(
        "4B. EXTERNAL 100 - EMOTION-SPECIFIC RETRIEVAL",
        ext_methods,
    )

    # --------------------------------------------------------
    # Geometry
    # --------------------------------------------------------

    dev_geometry = {}

    for method in [
        "raw",
        "corrected",
        "multivariate",
    ]:
        dev_geometry[
            method
        ] = summarize_geometry(
            geometry_blocks[
                method
            ][
                "same"
            ],
            geometry_blocks[
                method
            ][
                "different"
            ],
        )

    ext_geometry = {}

    for method, Y in [
        (
            "raw",
            ext_raw,
        ),
        (
            "corrected",
            ext_corr,
        ),
        (
            "multivariate",
            ext_multi,
        ),
    ]:
        s, d = (
            geometry_arrays(
                Y
            )
        )

        ext_geometry[
            method
        ] = summarize_geometry(
            [s],
            [d],
        )

    print()
    print("=" * 126)
    print(
        "5. EXTERNAL CROSS-EMOTION GEOMETRY"
    )
    print("=" * 126)

    for method in [
        "raw",
        "corrected",
        "multivariate",
    ]:

        print_matrix(
            f"{method.upper()} - SAME-CANDIDATE COSINE",
            ext_geometry[
                method
            ][
                "same"
            ],
        )

        print_matrix(
            f"{method.upper()} - SAME MINUS DIFFERENT GAP",
            ext_geometry[
                method
            ][
                "gap"
            ],
        )

        print_matrix(
            f"{method.upper()} - D-PRIME",
            ext_geometry[
                method
            ][
                "dprime"
            ],
        )

    # --------------------------------------------------------
    # Save JSON-compatible structures
    # --------------------------------------------------------

    def clean_per_emotion(d):
        out = {}

        for emotion in EMOTIONS:
            out[
                emotion
            ] = {
                k: v
                for k, v
                in d[
                    emotion
                ].items()
                if k != "ranks"
            }

        return out

    report = {
        "status":
            "emotion-wise diagnostic analysis using historical frozen implementation",

        "configuration": {
            "gamma": GAMMA,
            "dimension": DIM,
            "emotions":
                EMOTIONS,
            "folds":
                FOLDS,
        },

        "historical_reproduction": {
            "expected":
                expected,

            "actual": {
                "dev_raw":
                    dev_raw_historical_style,
                "dev_corrected":
                    dev_corr_historical_style,
                "dev_multi128":
                    dev_multi_historical_style,
                "ext_raw":
                    L_ext_raw[
                        "mrr"
                    ],
                "ext_corrected":
                    L_ext_corr[
                        "mrr"
                    ],
                "ext_multi128":
                    L_ext_multi[
                        "mrr"
                    ],
                "ext_multi128_saved_npz":
                    saved_loeo[
                        "mrr"
                    ],
                "saved_npz_delta":
                    saved_transform_delta,
            },

            "passed":
                bool(
                    all(checks)
                ),
        },

        "development": {
            "aggregate": {
                "raw":
                    dev_raw_agg,
                "corrected":
                    dev_corr_agg,
                "multivariate":
                    dev_multi_agg,
            },

            "per_emotion": {
                "raw":
                    clean_per_emotion(
                        dev_raw_per
                    ),
                "corrected":
                    clean_per_emotion(
                        dev_corr_per
                    ),
                "multivariate":
                    clean_per_emotion(
                        dev_multi_per
                    ),
            },

            "geometry": {
                method: {
                    key:
                        value.tolist()
                    for key, value
                    in dev_geometry[
                        method
                    ].items()
                }
                for method
                in dev_geometry
            },
        },

        "external": {
            "aggregate": {
                "raw":
                    ext_raw_agg,
                "corrected":
                    ext_corr_agg,
                "multivariate":
                    ext_multi_agg,
            },

            "per_emotion": {
                "raw":
                    clean_per_emotion(
                        ext_raw_per
                    ),
                "corrected":
                    clean_per_emotion(
                        ext_corr_per
                    ),
                "multivariate":
                    clean_per_emotion(
                        ext_multi_per
                    ),
            },

            "geometry": {
                method: {
                    key:
                        value.tolist()
                    for key, value
                    in ext_geometry[
                        method
                    ].items()
                }
                for method
                in ext_geometry
            },
        },
    }

    json_path = (
        OUT
        / "emotionwise_retrieval_report.json"
    )

    json_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Compact retrieval CSV
    # --------------------------------------------------------

    csv_path = (
        OUT
        / "emotionwise_retrieval.csv"
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        w = csv.writer(f)

        w.writerow([
            "dataset",
            "emotion",
            "raw_mrr",
            "corrected_mrr",
            "multivariate_mrr",
            "multi_minus_raw_mrr",
            "multi_minus_corrected_mrr",
            "raw_top1",
            "corrected_top1",
            "multivariate_top1",
            "raw_top5",
            "corrected_top5",
            "multivariate_top5",
        ])

        for dataset, methods in [
            (
                "development_oof",
                dev_methods,
            ),
            (
                "external",
                ext_methods,
            ),
        ]:

            for emotion in EMOTIONS:

                r = methods[
                    "raw"
                ][emotion]

                c = methods[
                    "corrected"
                ][emotion]

                m = methods[
                    "multivariate"
                ][emotion]

                w.writerow([
                    dataset,
                    emotion,

                    r["mrr"],
                    c["mrr"],
                    m["mrr"],

                    m["mrr"]
                    - r["mrr"],

                    m["mrr"]
                    - c["mrr"],

                    r["top1"],
                    c["top1"],
                    m["top1"],

                    r["top5"],
                    c["top5"],
                    m["top5"],
                ])

    # --------------------------------------------------------
    # Matrix CSVs
    # --------------------------------------------------------

    for dataset, geometry in [
        (
            "development_oof",
            dev_geometry,
        ),
        (
            "external",
            ext_geometry,
        ),
    ]:

        for method in [
            "raw",
            "corrected",
            "multivariate",
        ]:

            for stat in [
                "same",
                "different",
                "gap",
                "dprime",
            ]:

                save_matrix(
                    OUT
                    / (
                        f"{dataset}_"
                        f"{method}_"
                        f"{stat}.csv"
                    ),
                    geometry[
                        method
                    ][stat],
                )

    print()
    print("=" * 126)
    print("6. SAVED")
    print("=" * 126)

    print(
        "JSON :",
        json_path,
    )

    print(
        "CSV  :",
        csv_path,
    )

    print(
        "DIR  :",
        OUT,
    )

    print()
    print("DONE")


if __name__ == "__main__":
    main()
