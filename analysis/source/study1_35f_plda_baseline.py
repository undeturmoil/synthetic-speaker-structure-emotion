#!/usr/bin/env python3
# coding: utf-8

import json
from pathlib import Path

import numpy as np


ROOT = Path("/factory")

DEV_PATH = (
    ROOT / "campaigns/study1_35f300_v1/"
    "analysis/embeddings_300x6x2048.npy"
)

EXT_PATH = (
    ROOT / "campaigns/study1_35f100_external_v1/"
    "analysis/embeddings_100x6x2048.npy"
)

OUT = (
    ROOT / "campaigns/study1_35f100_external_v1/"
    "plda_baseline"
)

OUT.mkdir(parents=True, exist_ok=True)

# Same batch-group folds used previously.
FOLD_BATCHES = [
    [0, 5, 10, 15, 20, 25],
    [1, 6, 11, 16, 21, 26],
    [2, 7, 12, 17, 22, 27],
    [3, 8, 13, 18, 23, 28],
    [4, 9, 14, 19, 24, 29],
]

MODES = [
    "raw",
    "corrected",
]

DIMS = [
    64,
    128,
    192,
    224,
]

ALPHAS = [
    0.0,
    0.01,
    0.05,
    0.10,
]


def l2norm(X):
    n = np.linalg.norm(
        X,
        axis=-1,
        keepdims=True,
    )
    return X / np.maximum(
        n,
        1e-12,
    )


def emotion_effect(X):
    grand = X.mean(
        axis=(0, 1)
    )
    emean = X.mean(
        axis=0
    )
    return emean - grand[None, :]


def apply_mode(X, mode, effect):
    if mode == "raw":
        return X.copy()

    if mode == "corrected":
        return (
            X
            - effect[
                None,
                :,
                :
            ]
        )

    raise ValueError(mode)


def fit_pca(X, max_dim):
    """
    X: speakers x emotions x features.

    Input vectors are L2-normalized before PCA,
    consistent with speaker-embedding geometry.
    """

    Y = l2norm(X)

    flat = Y.reshape(
        -1,
        Y.shape[-1],
    )

    mean = flat.mean(
        axis=0
    )

    centered = (
        flat
        - mean[None, :]
    )

    # Exact SVD. Done once per fold/mode.
    _, _, Vt = np.linalg.svd(
        centered,
        full_matrices=False,
    )

    basis = (
        Vt[:max_dim].T
    )

    return mean, basis


def project_pca(
    X,
    pca_mean,
    basis,
):
    Y = l2norm(X)

    P = (
        Y
        - pca_mean[
            None,
            None,
            :
        ]
    ) @ basis

    # Standard speaker-recognition style
    # length normalization after projection.
    return l2norm(P)


def psd_clip(C, floor_scale=1e-6):
    C = (
        C + C.T
    ) / 2.0

    eigval, eigvec = np.linalg.eigh(
        C
    )

    scale = max(
        float(
            np.mean(
                np.abs(eigval)
            )
        ),
        1e-12,
    )

    floor = (
        floor_scale
        * scale
    )

    eigval = np.maximum(
        eigval,
        floor,
    )

    return (
        eigvec
        @ np.diag(eigval)
        @ eigvec.T
    )


def shrink_cov(C, alpha):
    d = C.shape[0]

    iso = (
        np.trace(C)
        / d
    )

    return (
        (1.0 - alpha) * C
        + alpha
        * iso
        * np.eye(d)
    )


def estimate_bw(P):
    """
    Gaussian PLDA / two-covariance model:

       x_ci = mu + y_c + eps_ci

       y_c   ~ N(0, B)
       eps   ~ N(0, W)

    P shape:
       candidate x emotion x dimension
    """

    n, m, d = P.shape

    mu = P.mean(
        axis=(0, 1)
    )

    Z = (
        P
        - mu[
            None,
            None,
            :
        ]
    )

    speaker_mean = Z.mean(
        axis=1
    )

    residual = (
        Z
        - speaker_mean[
            :,
            None,
            :
        ]
    )

    R = residual.reshape(
        -1,
        d,
    )

    W = (
        R.T @ R
    ) / R.shape[0]

    # Observed covariance of speaker means:
    # Cov(mean speaker obs) = B + W/m
    M = (
        speaker_mean.T
        @ speaker_mean
    ) / n

    B = (
        M
        - W / m
    )

    W = psd_clip(W)
    B = psd_clip(B)

    return mu, B, W


def fit_plda_from_bw(
    mu,
    B,
    W,
    alpha,
):
    """
    Whiten W and diagonalize B in the whitened
    space. This yields independent PLDA dimensions:

        within covariance = I
        between covariance = diag(lambda)
    """

    B = shrink_cov(
        B,
        alpha,
    )

    W = shrink_cov(
        W,
        alpha,
    )

    B = psd_clip(B)
    W = psd_clip(W)

    ew, Uw = np.linalg.eigh(
        W
    )

    ew = np.maximum(
        ew,
        1e-10,
    )

    W_inv_sqrt = (
        Uw
        @ np.diag(
            1.0 / np.sqrt(ew)
        )
        @ Uw.T
    )

    Bw = (
        W_inv_sqrt
        @ B
        @ W_inv_sqrt
    )

    lam, U = np.linalg.eigh(
        (
            Bw + Bw.T
        ) / 2.0
    )

    order = np.argsort(
        lam
    )[::-1]

    lam = np.maximum(
        lam[order],
        0.0,
    )

    U = U[:, order]

    transform = (
        W_inv_sqrt
        @ U
    )

    return {
        "mu": mu,
        "transform": transform,
        "lambda": lam,
    }


def transform_plda(P, model):
    centered = (
        P
        - model["mu"][
            None,
            None,
            :
        ]
    )

    return (
        centered
        @ model["transform"]
    )


def plda_score_matrix(
    query,
    enrollment,
    lam,
    n_enroll,
):
    """
    query:
      nq x d

    enrollment:
      ne x d
      (mean of n_enroll observations)

    In diagonal PLDA space:

      Var(query)  = lambda + 1
      Var(mean)   = lambda + 1/n
      Cov(same)   = lambda
    """

    a = (
        lam + 1.0
    )

    b = (
        lam
        + 1.0 / n_enroll
    )

    det_same = (
        a * b
        - lam * lam
    )

    det_diff = (
        a * b
    )

    det_same = np.maximum(
        det_same,
        1e-12,
    )

    det_diff = np.maximum(
        det_diff,
        1e-12,
    )

    const = (
        -0.5
        * np.sum(
            np.log(
                det_same
                / det_diff
            )
        )
    )

    cq = (
        -0.5
        * (
            b / det_same
            - 1.0 / a
        )
    )

    ce = (
        -0.5
        * (
            a / det_same
            - 1.0 / b
        )
    )

    cross = (
        lam / det_same
    )

    q_term = (
        (query * query)
        @ cq
    )

    e_term = (
        (enrollment * enrollment)
        @ ce
    )

    cross_term = (
        (query * cross)
        @ enrollment.T
    )

    return (
        const
        + q_term[:, None]
        + e_term[None, :]
        + cross_term
    )


def cosine_score_matrix(
    query,
    enrollment,
):
    q = l2norm(query)
    e = l2norm(enrollment)

    return (
        q @ e.T
    )


def ranks_from_scores(S):
    n = S.shape[0]

    ranks = np.empty(
        n,
        dtype=int,
    )

    for i in range(n):
        order = np.argsort(
            -S[i]
        )

        ranks[i] = (
            np.where(
                order == i
            )[0][0]
            + 1
        )

    return ranks


def evaluate_loeo_plda(
    Z,
    lam,
):
    ranks = []

    for qe in range(6):
        other = [
            e
            for e in range(6)
            if e != qe
        ]

        query = Z[
            :,
            qe,
            :
        ]

        enrollment = Z[
            :,
            other,
            :
        ].mean(
            axis=1
        )

        S = plda_score_matrix(
            query,
            enrollment,
            lam,
            n_enroll=5,
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


def evaluate_loeo_cosine(P):
    ranks = []

    for qe in range(6):
        other = [
            e
            for e in range(6)
            if e != qe
        ]

        query = P[
            :,
            qe,
            :
        ]

        enrollment = P[
            :,
            other,
            :
        ].mean(
            axis=1
        )

        S = cosine_score_matrix(
            query,
            enrollment,
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


def fold_indices():
    folds = []

    batches = np.arange(30)

    for valid_batches in FOLD_BATCHES:
        valid_batches = np.asarray(
            valid_batches
        )

        train_batches = np.asarray([
            b
            for b in batches
            if b not in set(
                valid_batches.tolist()
            )
        ])

        train_idx = np.concatenate([
            np.arange(
                b * 10,
                b * 10 + 10,
            )
            for b in train_batches
        ])

        valid_idx = np.concatenate([
            np.arange(
                b * 10,
                b * 10 + 10,
            )
            for b in valid_batches
        ])

        folds.append(
            (
                train_idx,
                valid_idx,
            )
        )

    return folds


def main():
    X = np.load(
        DEV_PATH
    ).astype(
        np.float64
    )

    Xext = np.load(
        EXT_PATH
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

    if Xext.shape != (
        100,
        6,
        2048,
    ):
        raise RuntimeError(
            Xext.shape
        )

    cv = {}

    max_dim = max(
        DIMS
    )

    folds = fold_indices()

    print()
    print("=" * 105)
    print(
        "DEVELOPMENT-300 PLDA BASELINE CV"
    )
    print("=" * 105)

    for fold_no, (
        train_idx,
        valid_idx,
    ) in enumerate(
        folds,
        start=1,
    ):
        Xtr0 = X[
            train_idx
        ]

        Xva0 = X[
            valid_idx
        ]

        effect = emotion_effect(
            Xtr0
        )

        for mode in MODES:
            Xtr = apply_mode(
                Xtr0,
                mode,
                effect,
            )

            Xva = apply_mode(
                Xva0,
                mode,
                effect,
            )

            pca_mean, full_basis = (
                fit_pca(
                    Xtr,
                    max_dim,
                )
            )

            for dim in DIMS:
                basis = (
                    full_basis[
                        :,
                        :dim,
                    ]
                )

                Ptr = project_pca(
                    Xtr,
                    pca_mean,
                    basis,
                )

                Pva = project_pca(
                    Xva,
                    pca_mean,
                    basis,
                )

                plda_mu, B, W = (
                    estimate_bw(
                        Ptr
                    )
                )

                cosine_metrics = (
                    evaluate_loeo_cosine(
                        Pva
                    )
                )

                for alpha in ALPHAS:
                    model = fit_plda_from_bw(
                        plda_mu,
                        B,
                        W,
                        alpha,
                    )

                    Zva = transform_plda(
                        Pva,
                        model,
                    )

                    metrics = (
                        evaluate_loeo_plda(
                            Zva,
                            model["lambda"],
                        )
                    )

                    key = (
                        mode,
                        dim,
                        alpha,
                    )

                    if key not in cv:
                        cv[key] = {
                            "plda": [],
                            "pca_cosine": [],
                        }

                    cv[key][
                        "plda"
                    ].append(
                        metrics
                    )

                    cv[key][
                        "pca_cosine"
                    ].append(
                        cosine_metrics
                    )

        print(
            f"fold {fold_no}/5 complete"
        )

    rows = []

    for key, result in cv.items():
        mode, dim, alpha = key

        mrr = np.asarray([
            r["mrr"]
            for r in result["plda"]
        ])

        top1 = np.asarray([
            r["top1"]
            for r in result["plda"]
        ])

        top5 = np.asarray([
            r["top5"]
            for r in result["plda"]
        ])

        cos_mrr = np.asarray([
            r["mrr"]
            for r in result[
                "pca_cosine"
            ]
        ])

        rows.append({
            "mode":
                mode,

            "dimension":
                dim,

            "alpha":
                alpha,

            "mrr_mean":
                float(
                    mrr.mean()
                ),

            "mrr_sd":
                float(
                    mrr.std(
                        ddof=1
                    )
                ),

            "top1_mean":
                float(
                    top1.mean()
                ),

            "top5_mean":
                float(
                    top5.mean()
                ),

            "pca_cosine_mrr_mean":
                float(
                    cos_mrr.mean()
                ),
        })

    rows.sort(
        key=lambda x:
            x["mrr_mean"],
        reverse=True,
    )

    print()
    print("=" * 105)
    print(
        "TOP DEVELOPMENT CONFIGURATIONS"
    )
    print("=" * 105)

    print(
        f"{'mode':12s} "
        f"{'dim':>5s} "
        f"{'alpha':>7s} "
        f"{'PLDA MRR':>10s} "
        f"{'SD':>8s} "
        f"{'Top1':>8s} "
        f"{'PCAcos':>9s}"
    )

    print("-" * 75)

    for r in rows[:12]:
        print(
            f"{r['mode']:12s} "
            f"{r['dimension']:5d} "
            f"{r['alpha']:7.3f} "
            f"{r['mrr_mean']:10.4f} "
            f"{r['mrr_sd']:8.4f} "
            f"{r['top1_mean']:8.4f} "
            f"{r['pca_cosine_mrr_mean']:9.4f}"
        )

    best = rows[0]

    print()
    print(
        "SELECTED FROM DEVELOPMENT ONLY:"
    )
    print(
        f"mode={best['mode']}, "
        f"dim={best['dimension']}, "
        f"alpha={best['alpha']}"
    )

    # =================================================
    # Freeze on all development300.
    # External100 is not involved in selection.
    # =================================================

    effect_full = emotion_effect(
        X
    )

    Xtr = apply_mode(
        X,
        best["mode"],
        effect_full,
    )

    Xte = apply_mode(
        Xext,
        best["mode"],
        effect_full,
    )

    pca_mean, full_basis = fit_pca(
        Xtr,
        best["dimension"],
    )

    basis = full_basis[
        :,
        :best["dimension"],
    ]

    Ptr = project_pca(
        Xtr,
        pca_mean,
        basis,
    )

    Pte = project_pca(
        Xte,
        pca_mean,
        basis,
    )

    plda_mu, B, W = estimate_bw(
        Ptr
    )

    model = fit_plda_from_bw(
        plda_mu,
        B,
        W,
        best["alpha"],
    )

    Zte = transform_plda(
        Pte,
        model,
    )

    external_plda = (
        evaluate_loeo_plda(
            Zte,
            model["lambda"],
        )
    )

    external_cosine = (
        evaluate_loeo_cosine(
            Pte
        )
    )

    print()
    print("=" * 105)
    print(
        "EXTERNAL-100 FROZEN PLDA RESULT"
    )
    print("=" * 105)

    print(
        f"PLDA      : "
        f"MRR={external_plda['mrr']:.4f} "
        f"Top1={external_plda['top1']:.4f} "
        f"Top5={external_plda['top5']:.4f}"
    )

    print(
        f"PCA cosine: "
        f"MRR={external_cosine['mrr']:.4f} "
        f"Top1={external_cosine['top1']:.4f} "
        f"Top5={external_cosine['top5']:.4f}"
    )

    print()
    print(
        "REFERENCE EXISTING EXTERNAL RESULTS:"
    )
    print(
        "raw cosine             MRR=0.1680"
    )
    print(
        "emotion-corrected      MRR=0.1992"
    )
    print(
        "frozen multivariate128 MRR=0.2856"
    )

    # Save frozen model for later WAV bundle scoring.
    model_path = (
        OUT
        / "frozen_plda_model.npz"
    )

    np.savez(
        model_path,

        mode=np.asarray([
            best["mode"]
        ]),

        dimension=np.asarray([
            best["dimension"]
        ]),

        alpha=np.asarray([
            best["alpha"]
        ]),

        emotion_effect=
            effect_full.astype(
                np.float32
            ),

        pca_mean=
            pca_mean.astype(
                np.float32
            ),

        pca_basis=
            basis.astype(
                np.float32
            ),

        plda_mean=
            model["mu"].astype(
                np.float32
            ),

        plda_transform=
            model[
                "transform"
            ].astype(
                np.float32
            ),

        plda_lambda=
            model[
                "lambda"
            ].astype(
                np.float32
            ),
    )

    report = {
        "method":
            "full-rank Gaussian PLDA two-covariance baseline",

        "selection":
            "5-fold batch-group development300 CV only",

        "best_config":
            best,

        "top_configs":
            rows[:20],

        "external100": {
            "plda":
                external_plda,

            "same_PCA_space_cosine":
                external_cosine,
        },

        "existing_external_reference": {
            "raw_cosine_mrr":
                0.1680,

            "emotion_corrected_mrr":
                0.1992,

            "frozen_multivariate128_mrr":
                0.2856,
        },
    }

    report_path = (
        OUT
        / "plda_baseline_report.json"
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
    print(
        "saved model:",
        model_path,
    )

    print(
        "saved report:",
        report_path,
    )


if __name__ == "__main__":
    main()
