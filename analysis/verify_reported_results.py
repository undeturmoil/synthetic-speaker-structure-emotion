from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_json(relative_path: str) -> dict:
    path = ROOT / relative_path

    if not path.exists():
        raise FileNotFoundError(path)

    return json.loads(path.read_text(encoding="utf-8"))


def get_value(data: dict, path: str):
    obj = data

    for key in path.split("."):
        obj = obj[key]

    if isinstance(obj, bool) or not isinstance(obj, (int, float)):
        raise TypeError(
            f"{path}: expected numeric value, got {type(obj).__name__}"
        )

    return float(obj)


def check_rounded(
    *,
    group: str,
    file: str,
    path: str,
    reported: float,
    decimals: int,
) -> bool:
    data = load_json(file)
    actual = get_value(data, path)

    tolerance = 0.5 * (10 ** (-decimals)) + 1e-12
    error = abs(actual - reported)
    passed = error <= tolerance

    status = "PASS" if passed else "FAIL"

    print(
        f"{status:4s}  {group:42s} "
        f"reported={reported:.{decimals}f}  "
        f"actual={actual:.12g}"
    )

    if not passed:
        print(f"      file: {file}")
        print(f"      path: {path}")
        print(f"      tolerance: {tolerance}")
        print(f"      absolute error: {error}")

    return passed


CHECKS = [
    # ------------------------------------------------------------
    # Primary Development300 variance decomposition
    # ------------------------------------------------------------
    {
        "group": "Development candidate variance (%)",
        "file": "results/development/variance_decomposition_300_report.json",
        "path": "primary_candidate_emotion.candidate_pct",
        "reported": 23.517,
        "decimals": 3,
    },
    {
        "group": "Development emotion variance (%)",
        "file": "results/development/variance_decomposition_300_report.json",
        "path": "primary_candidate_emotion.emotion_pct",
        "reported": 14.616,
        "decimals": 3,
    },
    {
        "group": "Development remainder (%)",
        "file": "results/development/variance_decomposition_300_report.json",
        "path": (
            "primary_candidate_emotion."
            "candidate_by_emotion_plus_unseparated_pct"
        ),
        "reported": 61.867,
        "decimals": 3,
    },

    # ------------------------------------------------------------
    # Secondary Development300 B x P x E decomposition
    # ------------------------------------------------------------
    {
        "group": "B main effect (%)",
        "file": "results/development/variance_decomposition_300_report.json",
        "path": "secondary_batch_position_emotion.batch_seed.pct",
        "reported": 2.241,
        "decimals": 3,
    },
    {
        "group": "P main effect (%)",
        "file": "results/development/variance_decomposition_300_report.json",
        "path": "secondary_batch_position_emotion.position.pct",
        "reported": 0.627,
        "decimals": 3,
    },
    {
        "group": "E main effect (%)",
        "file": "results/development/variance_decomposition_300_report.json",
        "path": "secondary_batch_position_emotion.emotion.pct",
        "reported": 14.616,
        "decimals": 3,
    },
    {
        "group": "B x P effect (%)",
        "file": "results/development/variance_decomposition_300_report.json",
        "path": "secondary_batch_position_emotion.batch_x_position.pct",
        "reported": 20.649,
        "decimals": 3,
    },
    {
        "group": "B x E effect (%)",
        "file": "results/development/variance_decomposition_300_report.json",
        "path": "secondary_batch_position_emotion.batch_x_emotion.pct",
        "reported": 6.112,
        "decimals": 3,
    },
    {
        "group": "P x E effect (%)",
        "file": "results/development/variance_decomposition_300_report.json",
        "path": "secondary_batch_position_emotion.position_x_emotion.pct",
        "reported": 1.814,
        "decimals": 3,
    },
    {
        "group": "B x P x E + unseparated (%)",
        "file": "results/development/variance_decomposition_300_report.json",
        "path": (
            "secondary_batch_position_emotion."
            "batch_x_position_x_emotion_plus_unseparated.pct"
        ),
        "reported": 53.941,
        "decimals": 3,
    },

    # ------------------------------------------------------------
    # Original External100
    # ------------------------------------------------------------
    {
        "group": "External raw LOEO MRR",
        "file": "results/external100/frozen_validation_report.json",
        "path": "results.raw_2048.loeo.mrr",
        "reported": 0.1680,
        "decimals": 4,
    },
    {
        "group": "External corrected LOEO MRR",
        "file": "results/external100/frozen_validation_report.json",
        "path": "results.corrected_2048.loeo.mrr",
        "reported": 0.1992,
        "decimals": 4,
    },
    {
        "group": "External multivariate LOEO MRR",
        "file": "results/external100/frozen_validation_report.json",
        "path": "results.frozen_multivariate_128.loeo.mrr",
        "reported": 0.2856,
        "decimals": 4,
    },

    # ------------------------------------------------------------
    # Sealed100
    # ------------------------------------------------------------
    {
        "group": "Sealed raw full-gallery MRR",
        "file": "results/sealed100/inference_report.json",
        "path": "full_gallery.raw.loeo_mrr.mean",
        "reported": 0.1701,
        "decimals": 4,
    },
    {
        "group": "Sealed corrected full-gallery MRR",
        "file": "results/sealed100/inference_report.json",
        "path": "full_gallery.corrected.loeo_mrr.mean",
        "reported": 0.2078,
        "decimals": 4,
    },
    {
        "group": "Sealed multivariate full-gallery MRR",
        "file": "results/sealed100/inference_report.json",
        "path": "full_gallery.multivariate.loeo_mrr.mean",
        "reported": 0.2777,
        "decimals": 4,
    },
    {
        "group": "Sealed primary MRR delta",
        "file": "results/sealed100/inference_report.json",
        "path": "primary_hypothesis.delta",
        "reported": 0.1164,
        "decimals": 4,
    },
    {
        "group": "Sealed primary Holm p",
        "file": "results/sealed100/inference_report.json",
        "path": "primary_hypothesis.p_holm_within_metric",
        "reported": 0.005859,
        "decimals": 6,
    },

    # ------------------------------------------------------------
    # WavLM independent representation
    # ------------------------------------------------------------
    {
        "group": "WavLM MRR",
        "file": (
            "results/external100/"
            "wavlm_independent_validation.json"
        ),
        "path": "wavlm_operational_identity.mrr",
        "reported": 0.0833,
        "decimals": 4,
    },
    {
        "group": "WavLM Top-1",
        "file": (
            "results/external100/"
            "wavlm_independent_validation.json"
        ),
        "path": "wavlm_operational_identity.top1",
        "reported": 0.0333,
        "decimals": 4,
    },
    {
        "group": "WavLM d-prime",
        "file": (
            "results/external100/"
            "wavlm_independent_validation.json"
        ),
        "path": "wavlm_operational_identity.dprime",
        "reported": 0.0750,
        "decimals": 4,
    },

    # ------------------------------------------------------------
    # emotion2vec manipulation check
    # ------------------------------------------------------------
    {
        "group": "emotion2vec development balanced acc.",
        "file": (
            "results/manipulation_checks/"
            "manipulation_check_report.json"
        ),
        "path": "development.metrics.balanced_accuracy",
        "reported": 0.5017,
        "decimals": 4,
    },
    {
        "group": "emotion2vec development macro-F1",
        "file": (
            "results/manipulation_checks/"
            "manipulation_check_report.json"
        ),
        "path": "development.metrics.macro_f1",
        "reported": 0.4945,
        "decimals": 4,
    },
    {
        "group": "emotion2vec external balanced acc.",
        "file": (
            "results/manipulation_checks/"
            "manipulation_check_report.json"
        ),
        "path": "external.metrics.balanced_accuracy",
        "reported": 0.5167,
        "decimals": 4,
    },
    {
        "group": "emotion2vec external macro-F1",
        "file": (
            "results/manipulation_checks/"
            "manipulation_check_report.json"
        ),
        "path": "external.metrics.macro_f1",
        "reported": 0.5058,
        "decimals": 4,
    },

    # ------------------------------------------------------------
    # Acoustic manipulation check
    # ------------------------------------------------------------
    {
        "group": "Acoustic development balanced acc.",
        "file": (
            "results/manipulation_checks/"
            "acoustic_manipulation_report.json"
        ),
        "path": "development.metrics.balanced_accuracy",
        "reported": 0.4000,
        "decimals": 4,
    },
    {
        "group": "Acoustic development macro-F1",
        "file": (
            "results/manipulation_checks/"
            "acoustic_manipulation_report.json"
        ),
        "path": "development.metrics.macro_f1",
        "reported": 0.3880,
        "decimals": 4,
    },
    {
        "group": "Acoustic external balanced acc.",
        "file": (
            "results/manipulation_checks/"
            "acoustic_manipulation_report.json"
        ),
        "path": "external.metrics.balanced_accuracy",
        "reported": 0.4217,
        "decimals": 4,
    },
    {
        "group": "Acoustic external macro-F1",
        "file": (
            "results/manipulation_checks/"
            "acoustic_manipulation_report.json"
        ),
        "path": "external.metrics.macro_f1",
        "reported": 0.4124,
        "decimals": 4,
    },
]


def main() -> int:
    print("Reported-result audit")
    print("=" * 96)
    print()

    passed = 0
    failed = 0

    for check in CHECKS:
        try:
            ok = check_rounded(**check)
        except Exception as exc:
            ok = False
            print(
                f"FAIL  {check['group']:42s} "
                f"error={type(exc).__name__}: {exc}"
            )

        if ok:
            passed += 1
        else:
            failed += 1

    print()
    print("=" * 96)
    print(f"Checks passed: {passed}/{len(CHECKS)}")

    if failed:
        print(f"Checks failed: {failed}")
        print("REPORTED RESULT AUDIT: FAILED")
        return 1

    print("ALL REPORTED RESULT CHECKS: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
