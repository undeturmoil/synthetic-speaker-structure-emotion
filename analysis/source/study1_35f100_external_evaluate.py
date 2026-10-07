#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np

from study1_35f300_multivariate_cv import (
    prepare_training,
    fit_gamma,
    correct,
    project,
    loeo,
    pairwise,
)


ROOT = Path("/factory")

DEV = (
    ROOT / "campaigns/study1_35f300_v1/"
    "analysis/embeddings_300x6x2048.npy"
)

VAL = (
    ROOT / "campaigns/study1_35f100_external_v1/"
    "analysis/embeddings_100x6x2048.npy"
)

OUT = (
    ROOT / "campaigns/study1_35f100_external_v1/"
    "frozen_validation"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

# FROZEN BEFORE EXTERNAL DATA ANALYSIS
GAMMA = 0.1
DIM = 128


def main():
    Xdev = np.load(
        DEV
    ).astype(
        np.float64
    )

    Xval = np.load(
        VAL
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

    if Xval.shape != (
        100,
        6,
        2048,
    ):
        raise RuntimeError(
            Xval.shape
        )

    print(
        "Fitting frozen transform "
        "on development 300 only..."
    )

    prep = prepare_training(
        Xdev
    )

    model = fit_gamma(
        prep,
        GAMMA,
    )

    if model[
        "basis"
    ].shape[1] < DIM:
        raise RuntimeError(
            "insufficient rank"
        )

    methods = {
        "raw_2048":
            Xval,

        "corrected_2048":
            correct(
                Xval,
                prep,
            ),

        "frozen_multivariate_128":
            project(
                Xval,
                prep,
                model[
                    "basis"
                ],
                DIM,
            ),
    }

    report = {
        "status":
            "FROZEN EXTERNAL VALIDATION",

        "development_candidates":
            300,

        "external_candidates":
            100,

        "gamma":
            GAMMA,

        "dimension":
            DIM,

        "selection_rule":
            "minimum dimension within 1-SE of development-CV optimum",

        "results": {},
    }

    print()
    print("=" * 88)
    print(
        "FROZEN EXTERNAL VALIDATION"
    )
    print("=" * 88)

    print(
        "Chance Top1 = 0.0100"
    )

    random_mrr = sum(
        1.0 / r
        for r in range(
            1,
            101
        )
    ) / 100.0

    print(
        f"Random MRR  = "
        f"{random_mrr:.4f}"
    )

    print()

    print(
        f"{'method':28s} "
        f"{'LOEO MRR':>10s} "
        f"{'Top1':>9s} "
        f"{'Top5':>9s} "
        f"{'pairMRR':>10s}"
    )

    print("-" * 72)

    for name, Y in methods.items():
        L = loeo(
            Y
        )

        P = pairwise(
            Y
        )

        report[
            "results"
        ][
            name
        ] = {
            "loeo":
                L,
            "pairwise":
                P,
        }

        print(
            f"{name:28s} "
            f"{L['mrr']:10.4f} "
            f"{L['top1']:9.4f} "
            f"{L['top5']:9.4f} "
            f"{P['mrr']:10.4f}"
        )

    raw = report[
        "results"
    ][
        "raw_2048"
    ][
        "loeo"
    ]

    cor = report[
        "results"
    ][
        "corrected_2048"
    ][
        "loeo"
    ]

    sub = report[
        "results"
    ][
        "frozen_multivariate_128"
    ][
        "loeo"
    ]

    report[
        "improvements"
    ] = {
        "corrected_vs_raw_mrr":
            cor["mrr"]
            - raw["mrr"],

        "subspace_vs_corrected_mrr":
            sub["mrr"]
            - cor["mrr"],

        "subspace_vs_raw_mrr":
            sub["mrr"]
            - raw["mrr"],

        "subspace_vs_corrected_top1":
            sub["top1"]
            - cor["top1"],
    }

    # Preserve the exact transform
    # used for this external validation.
    np.savez_compressed(
        OUT / "frozen_transform.npz",

        emotion_effect=
            prep[
                "emotion_effect"
            ].astype(
                np.float32
            ),

        grand=
            prep[
                "grand"
            ].astype(
                np.float32
            ),

        basis=
            model[
                "basis"
            ][
                :,
                :DIM
            ].astype(
                np.float32
            ),

        gamma=np.asarray(
            [GAMMA],
            dtype=np.float64,
        ),

        dimension=np.asarray(
            [DIM],
            dtype=np.int64,
        ),
    )

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
    print(
        "Frozen parameters: "
        f"gamma={GAMMA}, dim={DIM}"
    )

    print(
        "External data were not used "
        "for fitting or parameter selection."
    )

    print(
        "saved:",
        OUT / "report.json"
    )


if __name__ == "__main__":
    main()
