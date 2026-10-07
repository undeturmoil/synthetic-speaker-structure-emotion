#!/usr/bin/env python3
# coding: utf-8

import csv
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np


ROOT = Path("/factory")

EMBED = (
    ROOT / "campaigns/study1_35f300_v1/"
    "analysis/embeddings_300x6x2048.npy"
)

OUT = (
    ROOT / "campaigns/study1_35f300_v1/"
    "multivariate_cv"
)
OUT.mkdir(parents=True, exist_ok=True)

EMOTIONS = [
    "normal", "happy", "sad",
    "calm", "angry", "surprise",
]

GAMMAS = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
]

DIMS = [
    8,
    16,
    32,
    64,
    96,
    128,
    160,
    192,
    224,
    239,
]

# Interleaving prevents old/new generation blocks
# from forming separate validation folds.
FOLDS = [
    [0, 5, 10, 15, 20, 25],
    [1, 6, 11, 16, 21, 26],
    [2, 7, 12, 17, 22, 27],
    [3, 8, 13, 18, 23, 28],
    [4, 9, 14, 19, 24, 29],
]


def norm(x):
    d = np.linalg.norm(
        x,
        axis=-1,
        keepdims=True,
    )
    return x / np.maximum(d, 1e-12)


def rank(row, correct):
    order = np.argsort(-row)
    return int(
        np.where(order == correct)[0][0]
    ) + 1


def retrieval_matrix(sim):
    n = sim.shape[0]

    ranks = []
    margins = []

    for i in range(n):
        r = rank(sim[i], i)
        ranks.append(r)

        imp = np.delete(
            sim[i],
            i,
        ).max()

        margins.append(
            float(
                sim[i, i] - imp
            )
        )

    ranks = np.asarray(ranks)

    return {
        "top1": float(
            np.mean(ranks == 1)
        ),
        "top5": float(
            np.mean(ranks <= 5)
        ),
        "mrr": float(
            np.mean(1.0 / ranks)
        ),
        "margin": float(
            np.mean(margins)
        ),
    }


def loeo(X):
    Z = norm(X)

    n = Z.shape[0]

    ranks = []
    margins = []
    same = []
    cross = []

    for qe in range(6):
        other = [
            e for e in range(6)
            if e != qe
        ]

        enroll = norm(
            Z[:, other, :].mean(
                axis=1
            )
        )

        query = Z[:, qe, :]

        sim = (
            query
            @ enroll.T
        )

        diag = np.diag(sim)

        off = sim[
            ~np.eye(
                n,
                dtype=bool,
            )
        ]

        same.extend(
            diag.tolist()
        )
        cross.extend(
            off.tolist()
        )

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
        ranks
    )

    return {
        "top1": float(
            np.mean(ranks == 1)
        ),
        "top5": float(
            np.mean(ranks <= 5)
        ),
        "mrr": float(
            np.mean(1.0 / ranks)
        ),
        "margin": float(
            np.mean(margins)
        ),
        "gap": float(
            np.mean(same)
            - np.mean(cross)
        ),
    }


def pairwise(X):
    Z = norm(X)

    n = Z.shape[0]

    ranks = []
    margins = []
    same = []
    cross = []

    for ea, eb in combinations(
        range(6),
        2,
    ):
        sim = (
            Z[:, ea, :]
            @ Z[:, eb, :].T
        )

        diag = np.diag(sim)

        off = sim[
            ~np.eye(
                n,
                dtype=bool,
            )
        ]

        same.extend(
            diag.tolist()
        )
        cross.extend(
            off.tolist()
        )

        for matrix in (
            sim,
            sim.T,
        ):
            for i in range(n):
                r = rank(
                    matrix[i],
                    i,
                )
                ranks.append(r)

                imp = np.delete(
                    matrix[i],
                    i,
                ).max()

                margins.append(
                    float(
                        matrix[i, i]
                        - imp
                    )
                )

    ranks = np.asarray(
        ranks
    )

    return {
        "top1": float(
            np.mean(ranks == 1)
        ),
        "top5": float(
            np.mean(ranks <= 5)
        ),
        "mrr": float(
            np.mean(1.0 / ranks)
        ),
        "margin": float(
            np.mean(margins)
        ),
        "gap": float(
            np.mean(same)
            - np.mean(cross)
        ),
    }


def batch_indices(batches):
    idx = []

    for b in batches:
        for p in range(10):
            idx.append(
                b * 10 + p
            )

    return np.asarray(
        idx,
        dtype=int,
    )


def prepare_training(X):
    """
    X = N candidate x 6 emotions x 2048

    Returns everything that can be reused
    over all ridge values.
    """

    n, e, d = X.shape

    grand0 = X.mean(
        axis=(0, 1)
    )

    emean = X.mean(
        axis=0
    )

    emotion_effect = (
        emean
        - grand0[None, :]
    )

    Y = (
        X
        - emotion_effect[
            None,
            :,
            :
        ]
    )

    grand = Y.mean(
        axis=(0, 1)
    )

    cmean = Y.mean(
        axis=1
    )

    # Between-candidate matrix
    M = (
        cmean
        - grand[None, :]
    ).T

    M /= math.sqrt(
        n - 1
    )

    # Candidate x emotion residual
    residual = (
        Y
        - cmean[
            :,
            None,
            :
        ]
    )

    R = residual.reshape(
        n * e,
        d,
    )

    R /= math.sqrt(
        (n - 1)
        * (e - 1)
    )

    # Low-rank representation of W.
    RRt = R @ R.T

    # One eigendecomposition per fold,
    # reused for every gamma.
    s, Q = np.linalg.eigh(
        RRt
    )

    s = np.maximum(
        s,
        0.0,
    )

    RM = R @ M
    QtRM = Q.T @ RM

    mean_w = float(
        np.mean(
            np.sum(
                R * R,
                axis=0,
            )
        )
    )

    return {
        "emotion_effect":
            emotion_effect,

        "grand":
            grand,

        "M":
            M,

        "R":
            R,

        "s":
            s,

        "Q":
            Q,

        "QtRM":
            QtRM,

        "mean_w":
            mean_w,
    }


def fit_gamma(prep, gamma):
    """
    Generalized problem:

      B v = lambda (W + alpha I) v

    Uses Woodbury/sample-space algebra.

    Returned basis is Euclidean-orthonormal,
    removing arbitrary eigenvector scaling.
    """

    M = prep["M"]
    R = prep["R"]
    s = prep["s"]
    Q = prep["Q"]
    QtRM = prep["QtRM"]

    alpha = max(
        gamma
        * prep["mean_w"],
        1e-12,
    )

    # Solve:
    # (RR' + alpha I)^-1 RM
    solved = (
        Q
        @ (
            QtRM
            / (
                s[:, None]
                + alpha
            )
        )
    )

    # A^-1 M
    G = (
        M
        - R.T @ solved
    ) / alpha

    K = M.T @ G
    K = (
        K + K.T
    ) * 0.5

    values, U = np.linalg.eigh(
        K
    )

    order = np.argsort(
        -values
    )

    values = values[
        order
    ]

    U = U[
        :,
        order
    ]

    threshold = max(
        float(values[0])
        * 1e-10,
        1e-12,
    )

    keep = values > threshold

    values = values[
        keep
    ]

    U = U[
        :,
        keep
    ]

    # Generalized directions.
    V = G @ U

    # Remove arbitrary scaling and correlations.
    # QR preserves the nested span:
    # Q[:, :k] spans V[:, :k].
    Qv, _ = np.linalg.qr(
        V,
        mode="reduced",
    )

    return {
        "basis": Qv,
        "eigenvalues": values,
        "alpha": alpha,
    }


def correct(X, prep):
    return (
        X
        - prep[
            "emotion_effect"
        ][
            None,
            :,
            :
        ]
    )


def project(X, prep, basis, k):
    Y = (
        X
        - prep[
            "emotion_effect"
        ][
            None,
            :,
            :
        ]
        - prep[
            "grand"
        ][
            None,
            None,
            :
        ]
    )

    return (
        Y
        @ basis[
            :,
            :k,
        ]
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


def main():
    X = np.load(
        EMBED
    ).astype(
        np.float64
    )

    if X.shape != (
        300,
        6,
        2048,
    ):
        raise RuntimeError(
            X.shape
        )

    trial_rows = []
    baseline_rows = []

    all_batches = set(
        range(30)
    )

    for fi, valid_batches in enumerate(
        FOLDS,
        1,
    ):
        train_batches = sorted(
            all_batches
            - set(valid_batches)
        )

        train_idx = batch_indices(
            train_batches
        )

        valid_idx = batch_indices(
            valid_batches
        )

        Xtrain = X[
            train_idx
        ]

        Xvalid = X[
            valid_idx
        ]

        print()
        print("=" * 82)
        print(
            f"FOLD {fi} "
            f"train={len(train_idx)} "
            f"valid={len(valid_idx)}"
        )
        print(
            f"valid batches={valid_batches}"
        )
        print("=" * 82)

        prep = prepare_training(
            Xtrain
        )

        # Baselines
        for method, Y in {
            "raw_2048":
                Xvalid,

            "corrected_2048":
                correct(
                    Xvalid,
                    prep,
                ),
        }.items():

            L = loeo(Y)
            P = pairwise(Y)

            baseline_rows.append({
                "fold": fi,
                "method": method,
                "loeo_mrr": L["mrr"],
                "loeo_top1": L["top1"],
                "loeo_top5": L["top5"],
                "loeo_margin": L["margin"],
                "loeo_gap": L["gap"],
                "pair_mrr": P["mrr"],
                "pair_top1": P["top1"],
                "pair_margin": P["margin"],
                "pair_gap": P["gap"],
            })

            print(
                f"{method:24s} "
                f"MRR={L['mrr']:.4f} "
                f"Top1={L['top1']:.4f}"
            )

        for gamma in GAMMAS:
            model = fit_gamma(
                prep,
                gamma,
            )

            available = (
                model["basis"].shape[1]
            )

            print(
                f"gamma={gamma:<7g} "
                f"available_rank={available}"
            )

            for k in DIMS:
                if k > available:
                    continue

                Y = project(
                    Xvalid,
                    prep,
                    model["basis"],
                    k,
                )

                L = loeo(Y)
                P = pairwise(Y)

                trial_rows.append({
                    "fold": fi,
                    "gamma": gamma,
                    "dim": k,
                    "available_rank":
                        available,
                    "alpha":
                        model["alpha"],

                    "loeo_mrr":
                        L["mrr"],
                    "loeo_top1":
                        L["top1"],
                    "loeo_top5":
                        L["top5"],
                    "loeo_margin":
                        L["margin"],
                    "loeo_gap":
                        L["gap"],

                    "pair_mrr":
                        P["mrr"],
                    "pair_top1":
                        P["top1"],
                    "pair_margin":
                        P["margin"],
                    "pair_gap":
                        P["gap"],
                })

    write_csv(
        OUT / "trials.csv",
        trial_rows,
    )

    write_csv(
        OUT / "baselines.csv",
        baseline_rows,
    )

    # Aggregate every gamma/dim across five folds.
    aggregate = []

    for gamma in GAMMAS:
        for k in DIMS:
            rows = [
                r
                for r in trial_rows
                if r["gamma"] == gamma
                and r["dim"] == k
            ]

            if len(rows) != 5:
                continue

            out = {
                "gamma": gamma,
                "dim": k,
            }

            for metric in [
                "loeo_mrr",
                "loeo_top1",
                "loeo_top5",
                "loeo_margin",
                "loeo_gap",
                "pair_mrr",
                "pair_top1",
                "pair_margin",
                "pair_gap",
            ]:
                vals = np.asarray(
                    [
                        r[metric]
                        for r in rows
                    ],
                    dtype=float,
                )

                out[
                    metric + "_mean"
                ] = float(
                    vals.mean()
                )

                out[
                    metric + "_sd"
                ] = float(
                    vals.std(
                        ddof=1
                    )
                )

            aggregate.append(out)

    ranking = sorted(
        aggregate,
        key=lambda r: (
            r[
                "loeo_mrr_mean"
            ],
            r[
                "loeo_top1_mean"
            ],
            r[
                "pair_mrr_mean"
            ],
        ),
        reverse=True,
    )

    write_csv(
        OUT / "aggregate_grid.csv",
        ranking,
    )

    # Baseline aggregate
    base_agg = []

    for method in [
        "raw_2048",
        "corrected_2048",
    ]:
        rows = [
            r
            for r in baseline_rows
            if r["method"] == method
        ]

        d = {
            "method": method,
        }

        for metric in [
            "loeo_mrr",
            "loeo_top1",
            "loeo_margin",
            "pair_mrr",
        ]:
            vals = np.asarray(
                [
                    x[metric]
                    for x in rows
                ]
            )

            d[
                metric + "_mean"
            ] = float(
                vals.mean()
            )

            d[
                metric + "_sd"
            ] = float(
                vals.std(
                    ddof=1
                )
            )

        base_agg.append(d)

    # Best params within each individual fold.
    fold_best = []

    for fi in range(
        1,
        6,
    ):
        rows = [
            r
            for r in trial_rows
            if r["fold"] == fi
        ]

        best = max(
            rows,
            key=lambda r: (
                r["loeo_mrr"],
                r["loeo_top1"],
                r["pair_mrr"],
            ),
        )

        fold_best.append({
            "fold": fi,
            "gamma": best["gamma"],
            "dim": best["dim"],
            "loeo_mrr":
                best["loeo_mrr"],
            "loeo_top1":
                best["loeo_top1"],
            "pair_mrr":
                best["pair_mrr"],
        })

    write_csv(
        OUT / "fold_best.csv",
        fold_best,
    )

    best = ranking[0]

    report = {
        "status": (
            "DEVELOPMENT CV ONLY - "
            "not final test"
        ),

        "data": {
            "batch_seeds": 30,
            "candidates": 300,
            "emotions": 6,
            "wav_samples": 1800,
        },

        "cv": {
            "folds": FOLDS,
            "train_candidates_per_fold":
                240,
            "validation_candidates_per_fold":
                60,
        },

        "grid": {
            "gammas": GAMMAS,
            "dimensions": DIMS,
        },

        "best_global": best,

        "fold_best":
            fold_best,

        "baseline_aggregate":
            base_agg,

        "boundary_flags": {
            "gamma_at_upper_boundary":
                best["gamma"]
                == max(GAMMAS),

            "dimension_at_upper_boundary":
                best["dim"]
                == max(DIMS),
        },

        "method_note": (
            "Generalized eigen directions are "
            "Euclidean-QR orthonormalized before "
            "cosine retrieval, so evaluation reflects "
            "the selected subspace rather than arbitrary "
            "eigenvector scaling."
        ),
    }

    (
        OUT / "report.json"
    ).write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("35F300 MULTIVARIATE DEVELOPMENT CV")
    print("=" * 100)

    for b in base_agg:
        print(
            f"{b['method']:26s} "
            f"LOEO MRR={b['loeo_mrr_mean']:.4f} "
            f"Top1={b['loeo_top1_mean']:.4f} "
            f"pairMRR={b['pair_mrr_mean']:.4f}"
        )

    print()
    print("BEST MULTIVARIATE")
    print(
        f"gamma={best['gamma']} "
        f"dim={best['dim']}"
    )

    print(
        f"LOEO MRR="
        f"{best['loeo_mrr_mean']:.4f} "
        f"(SD={best['loeo_mrr_sd']:.4f})"
    )

    print(
        f"Top1="
        f"{best['loeo_top1_mean']:.4f} "
        f"pairMRR="
        f"{best['pair_mrr_mean']:.4f}"
    )

    print()
    print("PER-FOLD OPTIMA")

    for r in fold_best:
        print(
            f"fold{r['fold']}: "
            f"gamma={r['gamma']:<7g} "
            f"dim={r['dim']:<3d} "
            f"MRR={r['loeo_mrr']:.4f}"
        )

    print()
    print(
        "upper gamma boundary:",
        report[
            "boundary_flags"
        ][
            "gamma_at_upper_boundary"
        ],
    )

    print(
        "upper dim boundary  :",
        report[
            "boundary_flags"
        ][
            "dimension_at_upper_boundary"
        ],
    )

    print()
    print(
        "NOTE: this is development CV. "
        "The chosen parameters still require "
        "a completely new batch-seed validation panel."
    )

    print(
        "saved:",
        OUT / "report.json"
    )


if __name__ == "__main__":
    main()
