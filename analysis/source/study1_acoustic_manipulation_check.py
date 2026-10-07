#!/usr/bin/env python3

from pathlib import Path
import csv
import json
import os

import numpy as np
import soundfile as sf
import librosa

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
)


ROOT = Path("/factory")

CAMPAIGN = ROOT / "campaigns/study1_manipulation_check_v1"
MANIFEST = CAMPAIGN / "manifest_2400.csv"

OUT = CAMPAIGN / "acoustic_v1"
OUT.mkdir(parents=True, exist_ok=True)

FEATURE_NPY = OUT / "acoustic_features_2400x6.npy"
PARTIAL_NPY = OUT / "acoustic_features_2400x6.partial.npy"
DONE_NPY = OUT / "acoustic_done.partial.npy"

FEATURE_CSV = OUT / "acoustic_features_2400.csv"
REPORT_JSON = OUT / "acoustic_manipulation_report.json"

EMOTIONS = [
    "normal",
    "happy",
    "sad",
    "calm",
    "angry",
    "surprise",
]

E2I = {
    e: i
    for i, e in enumerate(EMOTIONS)
}

FEATURES = [
    "f0_median_hz",
    "f0_iqr_hz",
    "voiced_fraction",
    "rms_dbfs",
    "duration_sec",
    "spectral_centroid_hz",
]

DEV_FOLDS = [
    [0, 5, 10, 15, 20, 25],
    [1, 6, 11, 16, 21, 26],
    [2, 7, 12, 17, 22, 27],
    [3, 8, 13, 18, 23, 28],
    [4, 9, 14, 19, 24, 29],
]

SEED = 777
N_BOOT = 5000
N_PERM = 10000
CHANCE = 1.0 / 6.0


# ============================================================
# Load manifest
# ============================================================

with MANIFEST.open(
    "r",
    encoding="utf-8",
    newline="",
) as f:
    rows = list(csv.DictReader(f))

if len(rows) != 2400:
    raise RuntimeError(
        f"Expected 2400 rows, got {len(rows)}"
    )


dataset = np.asarray(
    [r["dataset"] for r in rows],
    dtype=object,
)

candidate = np.asarray(
    [r["candidate_id"] for r in rows],
    dtype=object,
)

batch = np.asarray(
    [int(r["batch"]) for r in rows],
    dtype=int,
)

emotion = np.asarray(
    [r["emotion"] for r in rows],
    dtype=object,
)

y = np.asarray(
    [E2I[e] for e in emotion],
    dtype=int,
)

dev_mask = dataset == "development"
ext_mask = dataset == "external"


# ============================================================
# Acoustic extraction
# ============================================================

def extract_features(path):
    x, sr = sf.read(
        path,
        dtype="float32",
        always_2d=False,
    )

    if x.ndim == 2:
        x = x.mean(axis=1)

    if not np.all(np.isfinite(x)):
        raise RuntimeError(
            f"Non-finite waveform: {path}"
        )

    duration = (
        len(x) / float(sr)
    )

    # --------------------------------------------------------
    # F0 and voiced fraction
    # Wide range deliberately chosen to avoid condition-
    # specific clipping of generated female voices.
    # --------------------------------------------------------

    f0, voiced_flag, _ = librosa.pyin(
        x,
        fmin=50.0,
        fmax=600.0,
        sr=sr,
        frame_length=1024,
        hop_length=240,
        center=True,
    )

    valid_f0 = (
        np.isfinite(f0)
        & voiced_flag
    )

    if np.any(valid_f0):
        f = f0[valid_f0]

        f0_median = float(
            np.median(f)
        )

        f0_iqr = float(
            np.percentile(f, 75)
            - np.percentile(f, 25)
        )
    else:
        f0_median = np.nan
        f0_iqr = np.nan

    voiced_fraction = float(
        np.mean(voiced_flag)
    )

    # --------------------------------------------------------
    # Whole-utterance RMS in dBFS
    # --------------------------------------------------------

    rms = float(
        np.sqrt(
            np.mean(
                np.square(
                    x.astype(np.float64)
                )
            )
        )
    )

    rms_dbfs = float(
        20.0
        * np.log10(
            max(
                rms,
                1e-12,
            )
        )
    )

    # --------------------------------------------------------
    # Spectral centroid
    # --------------------------------------------------------

    centroid = librosa.feature.spectral_centroid(
        y=x,
        sr=sr,
        n_fft=1024,
        hop_length=240,
    )[0]

    spectral_centroid = float(
        np.mean(
            centroid
        )
    )

    return np.asarray(
        [
            f0_median,
            f0_iqr,
            voiced_fraction,
            rms_dbfs,
            duration,
            spectral_centroid,
        ],
        dtype=np.float64,
    )


print("=" * 112)
print("3-F MODEL-INDEPENDENT ACOUSTIC MANIPULATION CHECK")
print("=" * 112)


if FEATURE_NPY.exists():
    X = np.load(
        FEATURE_NPY
    ).astype(
        np.float64,
        copy=False,
    )

    print()
    print(
        "Using frozen acoustic features:",
        FEATURE_NPY,
    )

else:
    if PARTIAL_NPY.exists():
        Xmem = np.load(
            PARTIAL_NPY,
            mmap_mode="r+",
        )

        done = np.load(
            DONE_NPY,
            mmap_mode="r+",
        )
    else:
        Xmem = np.lib.format.open_memmap(
            PARTIAL_NPY,
            mode="w+",
            dtype=np.float64,
            shape=(2400, 6),
        )

        done = np.lib.format.open_memmap(
            DONE_NPY,
            mode="w+",
            dtype=np.uint8,
            shape=(2400,),
        )

        done[:] = 0
        done.flush()

    print()
    print(
        "Extraction resume:",
        int(done.sum()),
        "/ 2400",
    )

    for i, r in enumerate(rows):
        if int(done[i]) == 1:
            continue

        wav = (
            ROOT
            / r["wav_path"]
        )

        feat = extract_features(
            wav
        )

        Xmem[i] = feat
        Xmem.flush()

        done[i] = 1
        done.flush()

        n = int(
            done.sum()
        )

        if (
            n == 1
            or n % 25 == 0
            or n == 2400
        ):
            print(
                f"[{n:4d}/2400] "
                f"{r['candidate_id']} "
                f"{r['emotion']:<9s} "
                f"F0={feat[0]:7.2f} "
                f"RMS={feat[3]:7.2f} dBFS "
                f"dur={feat[4]:6.2f}s"
            )

    if not np.all(
        done == 1
    ):
        raise RuntimeError(
            "Incomplete acoustic extraction."
        )

    X = np.asarray(
        Xmem
    ).copy()

    del Xmem
    del done

    np.save(
        FEATURE_NPY,
        X,
    )

    PARTIAL_NPY.unlink(
        missing_ok=True
    )

    DONE_NPY.unlink(
        missing_ok=True
    )


if X.shape != (2400, 6):
    raise RuntimeError(
        X.shape
    )


print()
print("Feature matrix:", X.shape)

for j, name in enumerate(FEATURES):
    print(
        f"  {name:<24s} "
        f"missing={np.sum(~np.isfinite(X[:,j]))}"
    )


# ============================================================
# Save feature table
# ============================================================

with FEATURE_CSV.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:

    w = csv.writer(f)

    w.writerow(
        [
            "dataset",
            "candidate_id",
            "batch",
            "position",
            "emotion",
            "wav_path",
        ]
        + FEATURES
    )

    for i, r in enumerate(rows):
        w.writerow(
            [
                r["dataset"],
                r["candidate_id"],
                r["batch"],
                r["position"],
                r["emotion"],
                r["wav_path"],
            ]
            + [
                float(v)
                if np.isfinite(v)
                else ""
                for v in X[i]
            ]
        )


# ============================================================
# Fixed acoustic decoder
# ============================================================

def make_model():
    return Pipeline([
        (
            "impute",
            SimpleImputer(
                strategy="median",
            ),
        ),
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
                random_state=SEED,
            ),
        ),
    ])


def calc_metrics(
    yt,
    yp,
):
    return {
        "accuracy":
            float(
                accuracy_score(
                    yt,
                    yp,
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    yt,
                    yp,
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    yt,
                    yp,
                    average="macro",
                )
            ),
    }


# ============================================================
# Development OOF
# ============================================================

print()
print("=" * 112)
print("1. DEVELOPMENT 300 - GROUPED 5-FOLD ACOUSTIC DECODING")
print("=" * 112)

dev_pred_all = np.full(
    2400,
    -1,
    dtype=int,
)

for fi, valid_batches in enumerate(
    DEV_FOLDS,
    start=1,
):

    train = (
        dev_mask
        & ~np.isin(
            batch,
            valid_batches,
        )
    )

    valid = (
        dev_mask
        & np.isin(
            batch,
            valid_batches,
        )
    )

    model = make_model()

    model.fit(
        X[train],
        y[train],
    )

    pred = model.predict(
        X[valid]
    )

    dev_pred_all[
        valid
    ] = pred

    m = calc_metrics(
        y[valid],
        pred,
    )

    print(
        f"Fold {fi}: "
        f"BalAcc={m['balanced_accuracy']:.4f} "
        f"MacroF1={m['macro_f1']:.4f}"
    )


dev_true = y[
    dev_mask
]

dev_pred = dev_pred_all[
    dev_mask
]

dev_candidate = candidate[
    dev_mask
]

if np.any(
    dev_pred < 0
):
    raise RuntimeError(
        "Incomplete OOF prediction."
    )


# ============================================================
# Frozen external
# ============================================================

print()
print("=" * 112)
print("2. EXTERNAL 100 - DEVELOPMENT-TRAINED FROZEN ACOUSTIC DECODER")
print("=" * 112)

final_model = make_model()

final_model.fit(
    X[dev_mask],
    y[dev_mask],
)

ext_pred = final_model.predict(
    X[ext_mask]
)

ext_true = y[
    ext_mask
]

ext_candidate = candidate[
    ext_mask
]


# ============================================================
# Candidate-cluster bootstrap
# ============================================================

def cluster_bootstrap(
    yt,
    yp,
    cid,
):
    rng = np.random.default_rng(
        SEED
    )

    unique = np.unique(
        cid
    )

    lookup = {
        c:
            np.flatnonzero(
                cid == c
            )
        for c in unique
    }

    bal = np.empty(
        N_BOOT
    )

    mf1 = np.empty(
        N_BOOT
    )

    for b in range(
        N_BOOT
    ):
        sampled = rng.choice(
            unique,
            size=len(unique),
            replace=True,
        )

        idx = np.concatenate(
            [
                lookup[c]
                for c in sampled
            ]
        )

        bal[b] = (
            balanced_accuracy_score(
                yt[idx],
                yp[idx],
            )
        )

        mf1[b] = (
            f1_score(
                yt[idx],
                yp[idx],
                average="macro",
            )
        )

    return {
        "balanced_accuracy_ci95": [
            float(
                np.percentile(
                    bal,
                    2.5,
                )
            ),
            float(
                np.percentile(
                    bal,
                    97.5,
                )
            ),
        ],

        "macro_f1_ci95": [
            float(
                np.percentile(
                    mf1,
                    2.5,
                )
            ),
            float(
                np.percentile(
                    mf1,
                    97.5,
                )
            ),
        ],
    }


# ============================================================
# Within-candidate permutation
# ============================================================

def permutation_test(
    yt,
    yp,
    cid,
):
    observed = float(
        balanced_accuracy_score(
            yt,
            yp,
        )
    )

    rng = np.random.default_rng(
        SEED
    )

    unique = np.unique(
        cid
    )

    lookup = {
        c:
            np.flatnonzero(
                cid == c
            )
        for c in unique
    }

    ge = 0

    for _ in range(
        N_PERM
    ):
        pseudo = yt.copy()

        for c in unique:
            idx = lookup[c]

            pseudo[idx] = (
                rng.permutation(
                    pseudo[idx]
                )
            )

        stat = (
            balanced_accuracy_score(
                pseudo,
                yp,
            )
        )

        if stat >= observed:
            ge += 1

    return float(
        (ge + 1)
        / (N_PERM + 1)
    )


dev_metrics = calc_metrics(
    dev_true,
    dev_pred,
)

ext_metrics = calc_metrics(
    ext_true,
    ext_pred,
)

dev_ci = cluster_bootstrap(
    dev_true,
    dev_pred,
    dev_candidate,
)

ext_ci = cluster_bootstrap(
    ext_true,
    ext_pred,
    ext_candidate,
)

dev_p = permutation_test(
    dev_true,
    dev_pred,
    dev_candidate,
)

ext_p = permutation_test(
    ext_true,
    ext_pred,
    ext_candidate,
)


# ============================================================
# Confusion
# ============================================================

dev_cm = confusion_matrix(
    dev_true,
    dev_pred,
    labels=np.arange(6),
    normalize="true",
)

ext_cm = confusion_matrix(
    ext_true,
    ext_pred,
    labels=np.arange(6),
    normalize="true",
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
            for e in EMOTIONS
        )
    )

    for i, e in enumerate(
        EMOTIONS
    ):
        print(
            f"{e:<12}"
            + "".join(
                f"{cm[i,j]:>10.3f}"
                for j in range(6)
            )
        )


# ============================================================
# Feature condition means
# ============================================================

def condition_stats(
    mask,
):
    out = {}

    for e in EMOTIONS:
        idx = (
            mask
            & (
                emotion == e
            )
        )

        out[e] = {}

        for j, feature in enumerate(
            FEATURES
        ):
            v = X[
                idx,
                j,
            ]

            v = v[
                np.isfinite(v)
            ]

            out[e][feature] = {
                "mean":
                    float(
                        np.mean(v)
                    ),

                "sd":
                    float(
                        np.std(
                            v,
                            ddof=1,
                        )
                    ),
            }

    return out


dev_stats = condition_stats(
    dev_mask
)

ext_stats = condition_stats(
    ext_mask
)


# ============================================================
# Feature-wise candidate / emotion decomposition
# ============================================================

def build_tensor(
    mask,
    feature_index,
):
    candidates = sorted(
        np.unique(
            candidate[
                mask
            ]
        ),
        key=lambda s: (
            int(s[1:3]),
            int(s[4:6]),
        ),
    )

    lookup = {}

    idxs = np.flatnonzero(
        mask
    )

    for i in idxs:
        lookup[
            (
                candidate[i],
                emotion[i],
            )
        ] = X[
            i,
            feature_index,
        ]

    # Use development median if missing.
    finite_dev = X[
        dev_mask,
        feature_index,
    ]

    finite_dev = finite_dev[
        np.isfinite(
            finite_dev
        )
    ]

    replacement = float(
        np.median(
            finite_dev
        )
    )

    T = np.zeros(
        (
            len(candidates),
            6,
        ),
        dtype=np.float64,
    )

    for ci, c in enumerate(
        candidates
    ):
        for ei, e in enumerate(
            EMOTIONS
        ):
            v = lookup[
                (c, e)
            ]

            if not np.isfinite(v):
                v = replacement

            T[
                ci,
                ei,
            ] = v

    return T


def decompose_scalar(
    T,
):
    n, e = T.shape

    grand = float(
        T.mean()
    )

    cmean = T.mean(
        axis=1
    )

    emean = T.mean(
        axis=0
    )

    C = (
        cmean
        - grand
    )

    E = (
        emean
        - grand
    )

    R = (
        T
        - grand
        - C[:, None]
        - E[None, :]
    )

    total = (
        T
        - grand
    )

    ss_total = float(
        np.sum(
            total ** 2
        )
    )

    ss_c = float(
        e
        * np.sum(
            C ** 2
        )
    )

    ss_e = float(
        n
        * np.sum(
            E ** 2
        )
    )

    ss_r = float(
        np.sum(
            R ** 2
        )
    )

    return {
        "candidate_pct":
            100
            * ss_c
            / ss_total,

        "emotion_pct":
            100
            * ss_e
            / ss_total,

        "candidate_x_emotion_plus_unseparated_pct":
            100
            * ss_r
            / ss_total,
    }


dev_effects = {}
ext_effects = {}

for j, feature in enumerate(
    FEATURES
):
    dev_effects[
        feature
    ] = decompose_scalar(
        build_tensor(
            dev_mask,
            j,
        )
    )

    ext_effects[
        feature
    ] = decompose_scalar(
        build_tensor(
            ext_mask,
            j,
        )
    )


# ============================================================
# Results print
# ============================================================

print()
print("=" * 112)
print("3. PRIMARY MODEL-INDEPENDENT ACOUSTIC CHECK")
print("=" * 112)

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
    f"{dev_p:.6f}"
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
    f"{ext_p:.6f}"
)

print_cm(
    "DEVELOPMENT normalized confusion matrix",
    dev_cm,
)

print_cm(
    "EXTERNAL normalized confusion matrix",
    ext_cm,
)


print()
print("=" * 112)
print("4. CONDITION MEANS")
print("=" * 112)

for dataset_name, stat in [
    (
        "Development300",
        dev_stats,
    ),
    (
        "External100",
        ext_stats,
    ),
]:

    print()
    print(dataset_name)

    print(
        f"{'emotion':<10}"
        f"{'F0 med':>10}"
        f"{'F0 IQR':>10}"
        f"{'voiced':>10}"
        f"{'RMS dB':>10}"
        f"{'duration':>11}"
        f"{'centroid':>11}"
    )

    for e in EMOTIONS:
        print(
            f"{e:<10}"
            f"{stat[e]['f0_median_hz']['mean']:>10.2f}"
            f"{stat[e]['f0_iqr_hz']['mean']:>10.2f}"
            f"{stat[e]['voiced_fraction']['mean']:>10.3f}"
            f"{stat[e]['rms_dbfs']['mean']:>10.2f}"
            f"{stat[e]['duration_sec']['mean']:>11.2f}"
            f"{stat[e]['spectral_centroid_hz']['mean']:>11.1f}"
        )


print()
print("=" * 112)
print("5. FEATURE-WISE EFFECT ACCOUNTING")
print("=" * 112)

print()
print(
    f"{'feature':<26}"
    f"{'DEV C%':>10}"
    f"{'DEV E%':>10}"
    f"{'DEV R%':>10}"
    f"{'EXT C%':>10}"
    f"{'EXT E%':>10}"
    f"{'EXT R%':>10}"
)

for feature in FEATURES:
    d = dev_effects[
        feature
    ]

    x = ext_effects[
        feature
    ]

    print(
        f"{feature:<26}"
        f"{d['candidate_pct']:>10.2f}"
        f"{d['emotion_pct']:>10.2f}"
        f"{d['candidate_x_emotion_plus_unseparated_pct']:>10.2f}"
        f"{x['candidate_pct']:>10.2f}"
        f"{x['emotion_pct']:>10.2f}"
        f"{x['candidate_x_emotion_plus_unseparated_pct']:>10.2f}"
    )


# ============================================================
# Save report
# ============================================================

report = {
    "status":
        "FROZEN MODEL-INDEPENDENT ACOUSTIC MANIPULATION CHECK",

    "features":
        FEATURES,

    "feature_extraction": {
        "f0_method":
            "librosa.pyin",

        "f0_range_hz":
            [
                50.0,
                600.0,
            ],

        "frame_length":
            1024,

        "hop_length":
            240,

        "rms":
            "whole-utterance RMS converted to dBFS",

        "spectral_centroid":
            "mean frame-level spectral centroid",
    },

    "decoder": {
        "pipeline":
            (
                "median imputation + StandardScaler + "
                "multinomial logistic regression"
            ),

        "C":
            1.0,

        "tuning":
            "none",

        "chance":
            CHANCE,

        "development_folds":
            DEV_FOLDS,

        "external_rule":
            (
                "fit once on all development300 "
                "and apply once to external100"
            ),
    },

    "development": {
        "metrics":
            dev_metrics,

        "cluster_bootstrap":
            dev_ci,

        "permutation_p":
            dev_p,

        "confusion_normalized":
            dev_cm.tolist(),

        "condition_stats":
            dev_stats,

        "feature_effect_accounting":
            dev_effects,
    },

    "external": {
        "metrics":
            ext_metrics,

        "cluster_bootstrap":
            ext_ci,

        "permutation_p":
            ext_p,

        "confusion_normalized":
            ext_cm.tolist(),

        "condition_stats":
            ext_stats,

        "feature_effect_accounting":
            ext_effects,
    },
}

REPORT_JSON.write_text(
    json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print()
print("=" * 112)
print("6. SAVED")
print("=" * 112)

print(
    "Features:",
    FEATURE_NPY,
)

print(
    "CSV     :",
    FEATURE_CSV,
)

print(
    "Report  :",
    REPORT_JSON,
)

print()
print("DONE")
