#!/usr/bin/env python3

from pathlib import Path
import csv
import json
import math

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


ROOT = Path("/factory")

CAMPAIGN = (
    ROOT
    / "campaigns"
    / "study1_manipulation_check_v1"
)

E2V = (
    CAMPAIGN
    / "emotion2vec_plus_large"
)

MANIFEST = (
    CAMPAIGN
    / "manifest_2400.csv"
)

EMBED = (
    E2V
    / "embeddings_2400x1024.npy"
)

SCORES = (
    E2V
    / "scores_2400x9.npy"
)

LABELS_JSON = (
    E2V
    / "labels.json"
)

OUT = (
    CAMPAIGN
    / "analysis_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


EMOTIONS = [
    "normal",
    "happy",
    "sad",
    "calm",
    "angry",
    "surprise",
]

EMOTION_TO_ID = {
    e: i
    for i, e
    in enumerate(EMOTIONS)
}

SEMANTIC_TARGET = {
    "normal": "neutral",
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "surprise": "surprised",
}

DEV_FOLDS = [
    [0, 5, 10, 15, 20, 25],
    [1, 6, 11, 16, 21, 26],
    [2, 7, 12, 17, 22, 27],
    [3, 8, 13, 18, 23, 28],
    [4, 9, 14, 19, 24, 29],
]

RNG_SEED = 777
N_BOOT = 5000
N_PERM = 10000

CHANCE = 1.0 / 6.0


# ============================================================
# Utility
# ============================================================

def read_manifest():
    with MANIFEST.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        rows = list(
            csv.DictReader(f)
        )

    if len(rows) != 2400:
        raise RuntimeError(
            f"Expected 2400 rows, got {len(rows)}"
        )

    return rows


def make_classifier():
    """
    Fixed a priori linear decoder.

    No tuning.
    No feature selection.
    No PCA.
    """
    return Pipeline([
        (
            "scale",
            StandardScaler(),
        ),
        (
            "clf",
            LogisticRegression(
                C=1.0,
                penalty="l2",
                solver="lbfgs",
                max_iter=5000,
                random_state=RNG_SEED,
            ),
        ),
    ])


def metrics(
    y_true,
    y_pred,
):
    return {
        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                )
            ),
    }


def candidate_cluster_bootstrap(
    y_true,
    y_pred,
    candidate_ids,
    n_boot=N_BOOT,
    seed=RNG_SEED,
):
    rng = np.random.default_rng(
        seed
    )

    unique = np.unique(
        candidate_ids
    )

    index_by_candidate = {
        c: np.flatnonzero(
            candidate_ids == c
        )
        for c in unique
    }

    out_bal = np.empty(
        n_boot,
        dtype=np.float64,
    )

    out_f1 = np.empty(
        n_boot,
        dtype=np.float64,
    )

    for b in range(
        n_boot
    ):
        sampled = rng.choice(
            unique,
            size=len(unique),
            replace=True,
        )

        idx = np.concatenate(
            [
                index_by_candidate[c]
                for c in sampled
            ]
        )

        yt = y_true[idx]
        yp = y_pred[idx]

        out_bal[b] = (
            balanced_accuracy_score(
                yt,
                yp,
            )
        )

        out_f1[b] = (
            f1_score(
                yt,
                yp,
                average="macro",
            )
        )

    return {
        "balanced_accuracy_ci95": [
            float(
                np.percentile(
                    out_bal,
                    2.5,
                )
            ),
            float(
                np.percentile(
                    out_bal,
                    97.5,
                )
            ),
        ],

        "macro_f1_ci95": [
            float(
                np.percentile(
                    out_f1,
                    2.5,
                )
            ),
            float(
                np.percentile(
                    out_f1,
                    97.5,
                )
            ),
        ],
    }


def within_candidate_permutation_test(
    y_true,
    y_pred,
    candidate_ids,
    n_perm=N_PERM,
    seed=RNG_SEED,
):
    """
    Shuffle the six instruction labels within each candidate.

    This preserves:
      - candidate structure
      - six-class balance
      - one observation per emotion per candidate

    Predictions stay frozen.
    """
    observed = (
        balanced_accuracy_score(
            y_true,
            y_pred,
        )
    )

    rng = np.random.default_rng(
        seed
    )

    unique = np.unique(
        candidate_ids
    )

    index_by_candidate = {
        c: np.flatnonzero(
            candidate_ids == c
        )
        for c in unique
    }

    ge = 0

    for _ in range(
        n_perm
    ):
        ypseudo = y_true.copy()

        for c in unique:
            idx = index_by_candidate[c]

            ypseudo[idx] = rng.permutation(
                ypseudo[idx]
            )

        score = (
            balanced_accuracy_score(
                ypseudo,
                y_pred,
            )
        )

        if score >= observed:
            ge += 1

    p = (
        ge + 1
    ) / (
        n_perm + 1
    )

    return {
        "observed":
            float(observed),

        "chance":
            float(CHANCE),

        "n_permutations":
            int(n_perm),

        "p_value":
            float(p),
    }


def bootstrap_mean_ci(
    values,
    n_boot=N_BOOT,
    seed=RNG_SEED,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    rng = np.random.default_rng(
        seed
    )

    n = len(values)

    means = np.empty(
        n_boot,
        dtype=np.float64,
    )

    for i in range(
        n_boot
    ):
        sample = rng.choice(
            values,
            size=n,
            replace=True,
        )

        means[i] = sample.mean()

    return [
        float(
            np.percentile(
                means,
                2.5,
            )
        ),
        float(
            np.percentile(
                means,
                97.5,
            )
        ),
    ]


# ============================================================
# Variance decomposition
# ============================================================

def variance_decomposition(
    X,
):
    """
    X: candidate x emotion x feature

    X_ce = mu + C_c + E_e + R_ce

    R_ce contains candidate x emotion interaction
    plus unseparated cell-level variation.
    """

    n, e, d = X.shape

    mu = X.mean(
        axis=(0, 1)
    )

    cmean = X.mean(
        axis=1
    )

    emean = X.mean(
        axis=0
    )

    C = (
        cmean
        - mu[None, :]
    )

    E = (
        emean
        - mu[None, :]
    )

    R = (
        X
        - mu[None, None, :]
        - C[:, None, :]
        - E[None, :, :]
    )

    total = (
        X
        - mu[
            None,
            None,
            :
        ]
    )

    ss_total = float(
        np.sum(
            total ** 2
        )
    )

    ss_candidate = float(
        e
        * np.sum(
            C ** 2
        )
    )

    ss_emotion = float(
        n
        * np.sum(
            E ** 2
        )
    )

    ss_residual = float(
        np.sum(
            R ** 2
        )
    )

    reconstructed = (
        ss_candidate
        + ss_emotion
        + ss_residual
    )

    return {
        "ss_total":
            ss_total,

        "ss_candidate":
            ss_candidate,

        "ss_emotion":
            ss_emotion,

        "ss_candidate_x_emotion_plus_unseparated":
            ss_residual,

        "candidate_pct":
            100.0
            * ss_candidate
            / ss_total,

        "emotion_pct":
            100.0
            * ss_emotion
            / ss_total,

        "candidate_x_emotion_plus_unseparated_pct":
            100.0
            * ss_residual
            / ss_total,

        "reconstruction_relative_error":
            abs(
                reconstructed
                - ss_total
            )
            / max(
                ss_total,
                1e-30,
            ),
    }


# ============================================================
# Semantic score check
# ============================================================

def semantic_analysis(
    rows,
    score_matrix,
    ser_labels,
    dataset,
):
    subset = [
        i
        for i, r
        in enumerate(rows)
        if r["dataset"] == dataset
    ]

    label_to_idx = {
        label: j
        for j, label
        in enumerate(
            ser_labels
        )
    }

    candidate_ids = sorted(
        {
            rows[i][
                "candidate_id"
            ]
            for i in subset
        }
    )

    row_lookup = {
        (
            rows[i][
                "candidate_id"
            ],
            rows[i][
                "emotion"
            ],
        ):
            i
        for i in subset
    }

    pred = np.argmax(
        score_matrix[
            subset
        ],
        axis=1,
    )

    pred_labels = [
        ser_labels[j]
        for j in pred
    ]

    result = {
        "mapped_conditions": {},
        "calm_descriptive": {},
    }

    # --------------------------------------------------------
    # Five pre-specified mappings
    # --------------------------------------------------------

    for condition, target in (
        SEMANTIC_TARGET.items()
    ):

        target_idx = (
            label_to_idx[
                target
            ]
        )

        condition_rows = [
            i
            for i in subset
            if rows[i][
                "emotion"
            ]
            == condition
        ]

        hits = []

        target_scores = []

        candidate_deltas = []

        for i in condition_rows:
            pred_label = ser_labels[
                int(
                    np.argmax(
                        score_matrix[i]
                    )
                )
            ]

            hits.append(
                pred_label
                == target
            )

            target_scores.append(
                float(
                    score_matrix[
                        i,
                        target_idx,
                    ]
                )
            )

        # Within-candidate specificity:
        #
        # target-class score on its intended condition
        # minus the mean target-class score on the
        # other five instruction conditions.
        for c in candidate_ids:

            intended_idx = (
                row_lookup[
                    (
                        c,
                        condition,
                    )
                ]
            )

            other_idx = [
                row_lookup[
                    (
                        c,
                        e,
                    )
                ]
                for e in EMOTIONS
                if e != condition
            ]

            intended_score = float(
                score_matrix[
                    intended_idx,
                    target_idx,
                ]
            )

            other_score = float(
                score_matrix[
                    other_idx,
                    target_idx,
                ].mean()
            )

            candidate_deltas.append(
                intended_score
                - other_score
            )

        result[
            "mapped_conditions"
        ][condition] = {
            "target_label":
                target,

            "n":
                len(
                    condition_rows
                ),

            "top1_hit_rate":
                float(
                    np.mean(
                        hits
                    )
                ),

            "target_score_mean":
                float(
                    np.mean(
                        target_scores
                    )
                ),

            "target_score_median":
                float(
                    np.median(
                        target_scores
                    )
                ),

            "within_candidate_target_score_delta_mean":
                float(
                    np.mean(
                        candidate_deltas
                    )
                ),

            "within_candidate_target_score_delta_ci95":
                bootstrap_mean_ci(
                    candidate_deltas
                ),
        }

    # --------------------------------------------------------
    # Calm remains unmapped a priori.
    # Report only observed model behavior.
    # --------------------------------------------------------

    calm_rows = [
        i
        for i in subset
        if rows[i][
            "emotion"
        ]
        == "calm"
    ]

    calm_pred = [
        ser_labels[
            int(
                np.argmax(
                    score_matrix[i]
                )
            )
        ]
        for i in calm_rows
    ]

    calm_counts = {
        label:
            int(
                sum(
                    p == label
                    for p in calm_pred
                )
            )
        for label
        in ser_labels
    }

    calm_mean_scores = {
        label:
            float(
                score_matrix[
                    calm_rows,
                    j,
                ].mean()
            )
        for j, label
        in enumerate(
            ser_labels
        )
    }

    result[
        "calm_descriptive"
    ] = {
        "n":
            len(
                calm_rows
            ),

        "predicted_label_counts":
            calm_counts,

        "mean_scores":
            calm_mean_scores,
    }

    return result


# ============================================================
# Data load
# ============================================================

print("=" * 110)
print(
    "STUDY1 - INDEPENDENT EMOTION MANIPULATION CHECK"
)
print("=" * 110)

rows = read_manifest()

X = np.load(
    EMBED
).astype(
    np.float64,
    copy=False,
)

S = np.load(
    SCORES
).astype(
    np.float64,
    copy=False,
)

ser_labels = json.loads(
    LABELS_JSON.read_text(
        encoding="utf-8"
    )
)

if X.shape != (
    2400,
    1024,
):
    raise RuntimeError(
        X.shape
    )

if S.shape != (
    2400,
    9,
):
    raise RuntimeError(
        S.shape
    )

if len(
    ser_labels
) != 9:
    raise RuntimeError(
        ser_labels
    )

print()
print(
    "Embeddings:",
    X.shape,
)

print(
    "SER scores:",
    S.shape,
)

print(
    "SER labels:",
    ser_labels,
)


# ============================================================
# Analysis arrays
# ============================================================

y = np.asarray(
    [
        EMOTION_TO_ID[
            r["emotion"]
        ]
        for r in rows
    ],
    dtype=int,
)

candidate = np.asarray(
    [
        r["candidate_id"]
        for r in rows
    ],
    dtype=object,
)

batch = np.asarray(
    [
        int(
            r["batch"]
        )
        for r in rows
    ],
    dtype=int,
)

dataset = np.asarray(
    [
        r["dataset"]
        for r in rows
    ],
    dtype=object,
)

dev_mask = (
    dataset
    == "development"
)

ext_mask = (
    dataset
    == "external"
)


# ============================================================
# Development grouped OOF
# ============================================================

print()
print("=" * 110)
print(
    "1. DEVELOPMENT 300 - GROUPED 5-FOLD EMOTION DECODING"
)
print("=" * 110)

dev_pred = np.full(
    len(rows),
    -1,
    dtype=int,
)

fold_results = []

for fold_no, valid_batches in enumerate(
    DEV_FOLDS,
    start=1,
):

    valid = (
        dev_mask
        & np.isin(
            batch,
            valid_batches,
        )
    )

    train = (
        dev_mask
        & ~np.isin(
            batch,
            valid_batches,
        )
    )

    if train.sum() != 1440:
        raise RuntimeError(
            f"fold{fold_no} train={train.sum()}"
        )

    if valid.sum() != 360:
        raise RuntimeError(
            f"fold{fold_no} valid={valid.sum()}"
        )

    model = make_classifier()

    model.fit(
        X[train],
        y[train],
    )

    pred = model.predict(
        X[valid]
    )

    dev_pred[
        valid
    ] = pred

    m = metrics(
        y[valid],
        pred,
    )

    fold_results.append(
        m
    )

    print(
        f"Fold {fold_no}: "
        f"BalAcc={m['balanced_accuracy']:.4f} "
        f"MacroF1={m['macro_f1']:.4f} "
        f"Accuracy={m['accuracy']:.4f}"
    )

if np.any(
    dev_pred[
        dev_mask
    ]
    < 0
):
    raise RuntimeError(
        "Incomplete OOF predictions"
    )

dev_true = y[
    dev_mask
]

dev_oof = dev_pred[
    dev_mask
]

dev_candidates = candidate[
    dev_mask
]

dev_metrics = metrics(
    dev_true,
    dev_oof,
)

dev_ci = (
    candidate_cluster_bootstrap(
        dev_true,
        dev_oof,
        dev_candidates,
    )
)

dev_perm = (
    within_candidate_permutation_test(
        dev_true,
        dev_oof,
        dev_candidates,
    )
)

dev_cm = confusion_matrix(
    dev_true,
    dev_oof,
    labels=np.arange(6),
)

dev_cm_norm = confusion_matrix(
    dev_true,
    dev_oof,
    labels=np.arange(6),
    normalize="true",
)


# ============================================================
# External frozen validation
# ============================================================

print()
print("=" * 110)
print(
    "2. EXTERNAL 100 - DEVELOPMENT-TRAINED FROZEN DECODER"
)
print("=" * 110)

final_model = make_classifier()

final_model.fit(
    X[
        dev_mask
    ],
    y[
        dev_mask
    ],
)

ext_pred = final_model.predict(
    X[
        ext_mask
    ]
)

ext_true = y[
    ext_mask
]

ext_candidates = candidate[
    ext_mask
]

ext_metrics = metrics(
    ext_true,
    ext_pred,
)

ext_ci = (
    candidate_cluster_bootstrap(
        ext_true,
        ext_pred,
        ext_candidates,
    )
)

ext_perm = (
    within_candidate_permutation_test(
        ext_true,
        ext_pred,
        ext_candidates,
    )
)

ext_cm = confusion_matrix(
    ext_true,
    ext_pred,
    labels=np.arange(6),
)

ext_cm_norm = confusion_matrix(
    ext_true,
    ext_pred,
    labels=np.arange(6),
    normalize="true",
)


# ============================================================
# Independent embedding variance decomposition
# ============================================================

def tensor_for_dataset(
    dataset_name,
):
    selected = [
        i
        for i, r
        in enumerate(rows)
        if r["dataset"]
        == dataset_name
    ]

    candidate_ids = sorted(
        {
            rows[i][
                "candidate_id"
            ]
            for i in selected
        },
        key=lambda x: (
            int(
                x[1:3]
            ),
            int(
                x[4:6]
            ),
        ),
    )

    lookup = {
        (
            rows[i][
                "candidate_id"
            ],
            rows[i][
                "emotion"
            ],
        ):
            i
        for i in selected
    }

    T = np.stack(
        [
            np.stack(
                [
                    X[
                        lookup[
                            (
                                c,
                                e,
                            )
                        ]
                    ]
                    for e
                    in EMOTIONS
                ],
                axis=0,
            )
            for c
            in candidate_ids
        ],
        axis=0,
    )

    return T


Xdev_tensor = (
    tensor_for_dataset(
        "development"
    )
)

Xext_tensor = (
    tensor_for_dataset(
        "external"
    )
)

dev_variance = (
    variance_decomposition(
        Xdev_tensor
    )
)

ext_variance = (
    variance_decomposition(
        Xext_tensor
    )
)


# ============================================================
# Semantic score checks
# ============================================================

dev_semantic = (
    semantic_analysis(
        rows,
        S,
        ser_labels,
        "development",
    )
)

ext_semantic = (
    semantic_analysis(
        rows,
        S,
        ser_labels,
        "external",
    )
)


# ============================================================
# SER score means: instruction x model class
# ============================================================

def score_mean_matrix(
    dataset_name,
):
    M = np.zeros(
        (
            6,
            len(
                ser_labels
            ),
        ),
        dtype=float,
    )

    for i, emotion in enumerate(
        EMOTIONS
    ):
        idx = [
            j
            for j, r
            in enumerate(rows)
            if (
                r["dataset"]
                == dataset_name
                and r[
                    "emotion"
                ]
                == emotion
            )
        ]

        M[i] = S[
            idx
        ].mean(
            axis=0
        )

    return M


dev_score_means = (
    score_mean_matrix(
        "development"
    )
)

ext_score_means = (
    score_mean_matrix(
        "external"
    )
)


# ============================================================
# Print
# ============================================================

print()
print("=" * 110)
print(
    "3. PRIMARY MANIPULATION CHECK"
)
print("=" * 110)

print()
print(
    f"Chance balanced accuracy = "
    f"{CHANCE:.4f}"
)

print()
print("DEVELOPMENT 300 OOF")
print(
    f"  Balanced accuracy : "
    f"{dev_metrics['balanced_accuracy']:.4f} "
    f"95% CI "
    f"[{dev_ci['balanced_accuracy_ci95'][0]:.4f}, "
    f"{dev_ci['balanced_accuracy_ci95'][1]:.4f}]"
)

print(
    f"  Macro-F1          : "
    f"{dev_metrics['macro_f1']:.4f} "
    f"95% CI "
    f"[{dev_ci['macro_f1_ci95'][0]:.4f}, "
    f"{dev_ci['macro_f1_ci95'][1]:.4f}]"
)

print(
    f"  Permutation p     : "
    f"{dev_perm['p_value']:.6f}"
)

print()
print("EXTERNAL 100")
print(
    f"  Balanced accuracy : "
    f"{ext_metrics['balanced_accuracy']:.4f} "
    f"95% CI "
    f"[{ext_ci['balanced_accuracy_ci95'][0]:.4f}, "
    f"{ext_ci['balanced_accuracy_ci95'][1]:.4f}]"
)

print(
    f"  Macro-F1          : "
    f"{ext_metrics['macro_f1']:.4f} "
    f"95% CI "
    f"[{ext_ci['macro_f1_ci95'][0]:.4f}, "
    f"{ext_ci['macro_f1_ci95'][1]:.4f}]"
)

print(
    f"  Permutation p     : "
    f"{ext_perm['p_value']:.6f}"
)


def print_cm(
    title,
    cm,
):
    print()
    print(title)

    print(
        f"{'true\\pred':<12}"
        + "".join(
            f"{e[:8]:>10s}"
            for e
            in EMOTIONS
        )
    )

    for i, emotion in enumerate(
        EMOTIONS
    ):
        print(
            f"{emotion:<12}"
            + "".join(
                f"{cm[i,j]:>10.3f}"
                for j
                in range(6)
            )
        )


print_cm(
    "DEVELOPMENT normalized confusion matrix",
    dev_cm_norm,
)

print_cm(
    "EXTERNAL normalized confusion matrix",
    ext_cm_norm,
)


print()
print("=" * 110)
print(
    "4. INDEPENDENT AFFECT-EMBEDDING EFFECT ACCOUNTING"
)
print("=" * 110)

for name, d in [
    (
        "Development300",
        dev_variance,
    ),
    (
        "External100",
        ext_variance,
    ),
]:
    print()
    print(name)
    print(
        f"  Candidate                   : "
        f"{d['candidate_pct']:.3f}%"
    )
    print(
        f"  Emotion                     : "
        f"{d['emotion_pct']:.3f}%"
    )
    print(
        f"  CxE + unseparated residual  : "
        f"{d['candidate_x_emotion_plus_unseparated_pct']:.3f}%"
    )
    print(
        f"  Reconstruction rel. error   : "
        f"{d['reconstruction_relative_error']:.3e}"
    )


print()
print("=" * 110)
print(
    "5. PRE-SPECIFIED SEMANTIC SER CHECK"
)
print("=" * 110)

for dataset_name, result in [
    (
        "Development300",
        dev_semantic,
    ),
    (
        "External100",
        ext_semantic,
    ),
]:
    print()
    print(dataset_name)

    print(
        f"{'condition':<12}"
        f"{'target':<12}"
        f"{'hit rate':>10}"
        f"{'mean score':>13}"
        f"{'delta':>11}"
        f"{'95% CI':>24}"
    )

    for condition in [
        "normal",
        "happy",
        "sad",
        "angry",
        "surprise",
    ]:
        d = (
            result[
                "mapped_conditions"
            ][condition]
        )

        ci = (
            d[
                "within_candidate_target_score_delta_ci95"
            ]
        )

        print(
            f"{condition:<12}"
            f"{d['target_label']:<12}"
            f"{d['top1_hit_rate']:>10.4f}"
            f"{d['target_score_mean']:>13.4f}"
            f"{d['within_candidate_target_score_delta_mean']:>11.4f}"
            f"  [{ci[0]:.4f}, {ci[1]:.4f}]"
        )

    print()
    print(
        "CALM: no a priori target class."
    )

    calm = (
        result[
            "calm_descriptive"
        ]
    )

    ordered = sorted(
        calm[
            "predicted_label_counts"
        ].items(),
        key=lambda x:
            x[1],
        reverse=True,
    )

    print(
        "  Predicted labels:",
        ", ".join(
            f"{k}={v}"
            for k, v
            in ordered
            if v > 0
        )
    )


# ============================================================
# Save confusion matrices
# ============================================================

def save_cm_csv(
    path,
    cm,
):
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        w = csv.writer(f)

        w.writerow(
            ["true"]
            + EMOTIONS
        )

        for i, emotion in enumerate(
            EMOTIONS
        ):
            w.writerow(
                [emotion]
                + [
                    float(
                        cm[i,j]
                    )
                    for j
                    in range(6)
                ]
            )


save_cm_csv(
    OUT
    / "development_confusion_normalized.csv",
    dev_cm_norm,
)

save_cm_csv(
    OUT
    / "external_confusion_normalized.csv",
    ext_cm_norm,
)


# ============================================================
# Save SER mean score matrices
# ============================================================

def save_score_matrix(
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
            ["instruction"]
            + ser_labels
        )

        for i, emotion in enumerate(
            EMOTIONS
        ):
            w.writerow(
                [emotion]
                + [
                    float(v)
                    for v
                    in M[i]
                ]
            )


save_score_matrix(
    OUT
    / "development_ser_score_means.csv",
    dev_score_means,
)

save_score_matrix(
    OUT
    / "external_ser_score_means.csv",
    ext_score_means,
)


# ============================================================
# Save OOF / external predictions
# ============================================================

prediction_csv = (
    OUT
    / "emotion_decoder_predictions.csv"
)

with prediction_csv.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:

    w = csv.writer(f)

    w.writerow([
        "dataset",
        "candidate_id",
        "batch",
        "position",
        "true_emotion",
        "predicted_emotion",
        "correct",
    ])

    for i, r in enumerate(
        rows
    ):
        if r["dataset"] == "development":
            pred_id = int(
                dev_pred[i]
            )
        else:
            # external indices are contiguous in the
            # filtered prediction vector, so resolve explicitly
            ext_indices = np.flatnonzero(
                ext_mask
            )

            ext_position = int(
                np.where(
                    ext_indices == i
                )[0][0]
            )

            pred_id = int(
                ext_pred[
                    ext_position
                ]
            )

        true_id = int(
            y[i]
        )

        w.writerow([
            r["dataset"],
            r["candidate_id"],
            r["batch"],
            r["position"],
            EMOTIONS[
                true_id
            ],
            EMOTIONS[
                pred_id
            ],
            int(
                true_id
                == pred_id
            ),
        ])


# ============================================================
# Report
# ============================================================

report = {
    "status":
        "FROZEN INDEPENDENT MANIPULATION CHECK",

    "design": {
        "affect_model":
            "emotion2vec+ large",

        "embedding_dimension":
            1024,

        "instruction_conditions":
            EMOTIONS,

        "chance_balanced_accuracy":
            CHANCE,

        "decoder":
            (
                "StandardScaler + "
                "multinomial logistic regression; "
                "L2, C=1.0, lbfgs; no tuning"
            ),

        "development_cv":
            DEV_FOLDS,

        "external_rule":
            (
                "train once on all development300 "
                "and apply once to external100"
            ),

        "bootstrap_unit":
            "candidate",

        "permutation":
            (
                "shuffle six instruction labels "
                "within candidate"
            ),

        "n_bootstrap":
            N_BOOT,

        "n_permutations":
            N_PERM,

        "calm_policy":
            (
                "no a priori SER target class; "
                "descriptive only"
            ),
    },

    "development": {
        "metrics":
            dev_metrics,

        "cluster_bootstrap":
            dev_ci,

        "permutation":
            dev_perm,

        "confusion_matrix":
            dev_cm.tolist(),

        "confusion_matrix_normalized":
            dev_cm_norm.tolist(),

        "variance_decomposition":
            dev_variance,

        "semantic":
            dev_semantic,
    },

    "external": {
        "metrics":
            ext_metrics,

        "cluster_bootstrap":
            ext_ci,

        "permutation":
            ext_perm,

        "confusion_matrix":
            ext_cm.tolist(),

        "confusion_matrix_normalized":
            ext_cm_norm.tolist(),

        "variance_decomposition":
            ext_variance,

        "semantic":
            ext_semantic,
    },

    "ser_labels":
        ser_labels,

    "development_ser_score_means":
        dev_score_means.tolist(),

    "external_ser_score_means":
        ext_score_means.tolist(),
}

report_path = (
    OUT
    / "manipulation_check_report.json"
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
print("=" * 110)
print(
    "6. SAVED"
)
print("=" * 110)

print(
    "Report:",
    report_path,
)

print(
    "Predictions:",
    prediction_csv,
)

print(
    "Output dir:",
    OUT,
)

print()
print("DONE")
