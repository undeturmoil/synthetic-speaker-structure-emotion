#!/usr/bin/env python3
# coding: utf-8

import json
from itertools import combinations, product
from pathlib import Path

import numpy as np


ROOT = Path("/factory")

VAL = (
    ROOT / "campaigns/study1_35f100_sealed_v1/"
    "analysis/embeddings_100x6x2048.npy"
)

FROZEN = (
    ROOT / "campaigns/study1_35f100_external_v1/"
    "frozen_validation/frozen_transform.npz"
)

OUT = (
    ROOT / "campaigns/study1_35f100_sealed_v1/"
    "inference"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

N_BOOT = 200_000
BOOT_SEED = 20261005


def normalize(x):
    d = np.linalg.norm(
        x,
        axis=-1,
        keepdims=True,
    )
    return x / np.maximum(
        d,
        1e-12,
    )


def ranks_from_similarity(sim):
    n = sim.shape[0]

    ranks = np.empty(
        n,
        dtype=int,
    )

    for i in range(n):
        order = np.argsort(
            -sim[i]
        )

        ranks[i] = (
            np.where(
                order == i
            )[0][0]
            + 1
        )

    return ranks


def loeo_query_scores(X):
    """
    Returns one record per:
      candidate x query emotion

    With N candidates:
      N * 6 records.
    """

    Z = normalize(X)
    n = Z.shape[0]

    candidate = []
    rr = []
    top1 = []
    top5 = []

    for qe in range(6):
        enroll_idx = [
            e
            for e in range(6)
            if e != qe
        ]

        enroll = normalize(
            Z[
                :,
                enroll_idx,
                :
            ].mean(axis=1)
        )

        query = Z[:, qe, :]

        sim = (
            query
            @ enroll.T
        )

        ranks = ranks_from_similarity(
            sim
        )

        candidate.extend(
            range(n)
        )

        rr.extend(
            1.0 / ranks
        )

        top1.extend(
            (ranks == 1).astype(float)
        )

        top5.extend(
            (ranks <= 5).astype(float)
        )

    return {
        "candidate":
            np.asarray(
                candidate,
                dtype=int,
            ),

        "mrr":
            np.asarray(
                rr,
                dtype=float,
            ),

        "top1":
            np.asarray(
                top1,
                dtype=float,
            ),

        "top5":
            np.asarray(
                top5,
                dtype=float,
            ),
    }


def pair_query_scores(X):
    """
    15 emotion pairs * both directions
    = 30 directed queries per candidate.
    """

    Z = normalize(X)
    n = Z.shape[0]

    candidate = []
    rr = []
    top1 = []

    for ea, eb in combinations(
        range(6),
        2,
    ):
        sim = (
            Z[:, ea, :]
            @ Z[:, eb, :].T
        )

        for matrix in (
            sim,
            sim.T,
        ):
            ranks = ranks_from_similarity(
                matrix
            )

            candidate.extend(
                range(n)
            )

            rr.extend(
                1.0 / ranks
            )

            top1.extend(
                (ranks == 1).astype(float)
            )

    return {
        "candidate":
            np.asarray(
                candidate,
                dtype=int,
            ),

        "mrr":
            np.asarray(
                rr,
                dtype=float,
            ),

        "top1":
            np.asarray(
                top1,
                dtype=float,
            ),
    }


def query_batch_means(scores):
    """
    External candidates are:
      local 0..9   -> seed batch30
      local 10..19 -> seed batch31
      ...
      local 90..99 -> seed batch39

    Full 100-candidate gallery is held fixed.
    """

    batch = (
        scores["candidate"]
        // 10
    )

    out = {}

    for metric in scores:
        if metric == "candidate":
            continue

        out[metric] = np.asarray([
            scores[metric][
                batch == b
            ].mean()
            for b in range(10)
        ])

    return out


def bootstrap_ci(values, boot_idx):
    boot = values[
        boot_idx
    ].mean(axis=1)

    lo, hi = np.percentile(
        boot,
        [2.5, 97.5],
    )

    return (
        float(values.mean()),
        float(lo),
        float(hi),
    )


def exact_signflip_p(diff):
    """
    Exact two-sided paired randomization test.

    With 10 independent batch replicates:
      2^10 = 1024 sign assignments.
    """

    diff = np.asarray(
        diff,
        dtype=float,
    )

    obs = abs(
        diff.mean()
    )

    values = []

    for signs in product(
        [-1.0, 1.0],
        repeat=len(diff),
    ):
        s = np.asarray(
            signs,
            dtype=float,
        )

        values.append(
            abs(
                np.mean(
                    diff * s
                )
            )
        )

    values = np.asarray(
        values
    )

    p = np.mean(
        values
        >= obs - 1e-15
    )

    return float(p)


def holm_adjust(ps):
    """
    Holm family-wise adjustment.
    """

    ps = np.asarray(
        ps,
        dtype=float,
    )

    m = len(ps)

    order = np.argsort(
        ps
    )

    adjusted = np.empty(
        m,
        dtype=float,
    )

    running = 0.0

    for rank, idx in enumerate(
        order
    ):
        value = (
            (m - rank)
            * ps[idx]
        )

        running = max(
            running,
            value,
        )

        adjusted[idx] = min(
            running,
            1.0,
        )

    return adjusted


def evaluate_method(X):
    return {
        "loeo":
            loeo_query_scores(
                X
            ),

        "pair":
            pair_query_scores(
                X
            ),
    }


def within_batch_metrics(X):
    """
    Independent replicate analysis.

    Each external seed batch is evaluated
    separately with its own 10-candidate gallery.

    This changes gallery size to N=10,
    but makes the 10 seed batches independent
    experimental replicates.
    """

    result = {
        "loeo_mrr": [],
        "loeo_top1": [],
        "loeo_top5": [],
        "pair_mrr": [],
        "pair_top1": [],
    }

    for b in range(10):
        Y = X[
            b * 10:
            (b + 1) * 10
        ]

        L = loeo_query_scores(
            Y
        )

        P = pair_query_scores(
            Y
        )

        result[
            "loeo_mrr"
        ].append(
            L["mrr"].mean()
        )

        result[
            "loeo_top1"
        ].append(
            L["top1"].mean()
        )

        result[
            "loeo_top5"
        ].append(
            L["top5"].mean()
        )

        result[
            "pair_mrr"
        ].append(
            P["mrr"].mean()
        )

        result[
            "pair_top1"
        ].append(
            P["top1"].mean()
        )

    return {
        k: np.asarray(
            v,
            dtype=float,
        )
        for k, v in result.items()
    }


def main():
    X = np.load(
        VAL
    ).astype(
        np.float64
    )

    if X.shape != (
        100,
        6,
        2048,
    ):
        raise RuntimeError(
            X.shape
        )

    frozen = np.load(
        FROZEN
    )

    gamma = float(
        frozen["gamma"][0]
    )

    dim = int(
        frozen["dimension"][0]
    )

    if gamma != 0.1:
        raise RuntimeError(
            f"unexpected gamma={gamma}"
        )

    if dim != 128:
        raise RuntimeError(
            f"unexpected dim={dim}"
        )

    effect = frozen[
        "emotion_effect"
    ].astype(
        np.float64
    )

    grand = frozen[
        "grand"
    ].astype(
        np.float64
    )

    basis = frozen[
        "basis"
    ].astype(
        np.float64
    )

    if basis.shape != (
        2048,
        128,
    ):
        raise RuntimeError(
            basis.shape
        )

    corrected = (
        X
        - effect[
            None,
            :,
            :
        ]
    )

    subspace = (
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

    methods = {
        "raw":
            X,

        "corrected":
            corrected,

        "multivariate":
            subspace,
    }

    rng = np.random.default_rng(
        BOOT_SEED
    )

    boot_idx = rng.integers(
        0,
        10,
        size=(
            N_BOOT,
            10,
        ),
    )

    report = {
        "frozen": {
            "gamma": gamma,
            "dimension": dim,
        },

        "bootstrap": {
            "clusters": 10,
            "replicates": N_BOOT,
            "seed": BOOT_SEED,
        },

        "full_gallery": {},
        "independent_batch": {},
    }

    # -------------------------------------------------
    # A. FULL 100-CANDIDATE GALLERY
    # -------------------------------------------------

    print()
    print("=" * 100)
    print(
        "A. FULL 100-CANDIDATE GALLERY"
    )
    print(
        "Batch-cluster bootstrap 95% CI "
        "(gallery held fixed)"
    )
    print("=" * 100)

    full_batch = {}

    for name, Y in methods.items():
        ev = evaluate_method(
            Y
        )

        Lb = query_batch_means(
            ev["loeo"]
        )

        Pb = query_batch_means(
            ev["pair"]
        )

        full_batch[name] = {
            "loeo_mrr":
                Lb["mrr"],

            "loeo_top1":
                Lb["top1"],

            "loeo_top5":
                Lb["top5"],

            "pair_mrr":
                Pb["mrr"],

            "pair_top1":
                Pb["top1"],
        }

        report[
            "full_gallery"
        ][name] = {}

        for metric, values in (
            full_batch[name]
        ).items():

            mean, lo, hi = (
                bootstrap_ci(
                    values,
                    boot_idx,
                )
            )

            report[
                "full_gallery"
            ][name][metric] = {
                "mean": mean,
                "ci95": [
                    lo,
                    hi,
                ],
            }

        x = report[
            "full_gallery"
        ][name]

        print(
            f"{name:14s} "
            f"LOEO MRR="
            f"{x['loeo_mrr']['mean']:.4f} "
            f"[{x['loeo_mrr']['ci95'][0]:.4f}, "
            f"{x['loeo_mrr']['ci95'][1]:.4f}] "
            f"Top1="
            f"{x['loeo_top1']['mean']:.4f}"
        )

    comparisons = [
        (
            "corrected_vs_raw",
            "corrected",
            "raw",
        ),

        (
            "multivariate_vs_corrected",
            "multivariate",
            "corrected",
        ),

        (
            "multivariate_vs_raw",
            "multivariate",
            "raw",
        ),
    ]

    print()
    print(
        "FULL-GALLERY PAIRED DIFFERENCE CIs"
    )

    report[
        "full_gallery"
    ][
        "differences"
    ] = {}

    for cname, a, b in comparisons:
        report[
            "full_gallery"
        ][
            "differences"
        ][cname] = {}

        for metric in [
            "loeo_mrr",
            "loeo_top1",
            "loeo_top5",
            "pair_mrr",
        ]:
            diff = (
                full_batch[a][metric]
                - full_batch[b][metric]
            )

            mean, lo, hi = (
                bootstrap_ci(
                    diff,
                    boot_idx,
                )
            )

            report[
                "full_gallery"
            ][
                "differences"
            ][cname][metric] = {
                "delta": mean,
                "ci95": [
                    lo,
                    hi,
                ],
            }

        r = report[
            "full_gallery"
        ][
            "differences"
        ][cname][
            "loeo_mrr"
        ]

        print(
            f"{cname:28s} "
            f"ΔMRR={r['delta']:+.4f} "
            f"[{r['ci95'][0]:+.4f}, "
            f"{r['ci95'][1]:+.4f}]"
        )

    # -------------------------------------------------
    # B. INDEPENDENT BATCH REPLICATES
    # -------------------------------------------------

    print()
    print("=" * 100)
    print(
        "B. 10 INDEPENDENT EXTERNAL BATCH REPLICATES"
    )
    print(
        "Each batch evaluated as its own "
        "10-candidate gallery"
    )
    print("=" * 100)

    independent = {
        name:
            within_batch_metrics(
                Y
            )
        for name, Y
        in methods.items()
    }

    print(
        "N=10 gallery chance: "
        "Top1=0.1000, "
        "random MRR=0.2929"
    )
    print()

    for name in methods:
        vals = independent[
            name
        ][
            "loeo_mrr"
        ]

        mean, lo, hi = (
            bootstrap_ci(
                vals,
                boot_idx,
            )
        )

        print(
            f"{name:14s} "
            f"LOEO MRR="
            f"{mean:.4f} "
            f"[{lo:.4f}, {hi:.4f}]"
        )

    metrics = [
        "loeo_mrr",
        "loeo_top1",
        "loeo_top5",
        "pair_mrr",
    ]

    tests = {}

    for metric in metrics:
        rows = []

        for cname, a, b in comparisons:
            diff = (
                independent[a][metric]
                - independent[b][metric]
            )

            mean, lo, hi = (
                bootstrap_ci(
                    diff,
                    boot_idx,
                )
            )

            p = exact_signflip_p(
                diff
            )

            rows.append({
                "comparison":
                    cname,

                "delta":
                    mean,

                "ci95": [
                    lo,
                    hi,
                ],

                "p_exact":
                    p,

                "batch_differences":
                    diff.tolist(),
            })

        adjusted = holm_adjust(
            [
                r["p_exact"]
                for r in rows
            ]
        )

        for r, q in zip(
            rows,
            adjusted,
        ):
            r[
                "p_holm_within_metric"
            ] = float(q)

        tests[metric] = rows

    report[
        "independent_batch"
    ][
        "tests"
    ] = tests

    print()
    print(
        "EXACT TWO-SIDED PAIRED SIGN-FLIP TEST"
    )
    print(
        "(Holm correction across 3 method "
        "contrasts within each metric)"
    )
    print()

    print(
        f"{'metric':12s} "
        f"{'comparison':28s} "
        f"{'delta':>9s} "
        f"{'95% CI':>23s} "
        f"{'p':>8s} "
        f"{'Holm':>8s}"
    )

    print("-" * 100)

    for metric in metrics:
        for r in tests[
            metric
        ]:
            lo, hi = r[
                "ci95"
            ]

            print(
                f"{metric:12s} "
                f"{r['comparison']:28s} "
                f"{r['delta']:+9.4f} "
                f"[{lo:+.4f}, {hi:+.4f}] "
                f"{r['p_exact']:8.4f} "
                f"{r['p_holm_within_metric']:8.4f}"
            )

    # Primary hypothesis:
    # multivariate > corrected on LOEO MRR.
    primary = next(
        r
        for r
        in tests["loeo_mrr"]
        if r["comparison"]
        == "multivariate_vs_corrected"
    )

    report[
        "primary_hypothesis"
    ] = {
        "endpoint":
            "LOEO MRR",

        "contrast":
            "multivariate - corrected",

        "test":
            "exact two-sided paired sign-flip "
            "across 10 independent external batches",

        **primary,
    }

    print()
    print("=" * 100)
    print(
        "PRIMARY CONFIRMATORY RESULT"
    )
    print("=" * 100)

    lo, hi = primary[
        "ci95"
    ]

    print(
        "Frozen multivariate vs corrected "
        "on LOEO MRR"
    )

    print(
        f"mean delta = "
        f"{primary['delta']:+.4f}"
    )

    print(
        f"95% cluster-bootstrap CI = "
        f"[{lo:+.4f}, {hi:+.4f}]"
    )

    print(
        f"exact two-sided p = "
        f"{primary['p_exact']:.6f}"
    )

    print(
        f"Holm p within LOEO-MRR family = "
        f"{primary['p_holm_within_metric']:.6f}"
    )

    path = (
        OUT
        / "inference_report.json"
    )

    path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("saved:", path)


if __name__ == "__main__":
    main()
