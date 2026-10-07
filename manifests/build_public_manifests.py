from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "manifests"
SOURCE_DIR = MANIFEST_DIR / "source"

EMOTIONS = {
    "normal",
    "happy",
    "sad",
    "calm",
    "angry",
    "surprise",
}

PANELS = {
    "development": {
        "sources": [
            "study1_35f100_v1_manifest.json",
            "study1_35f300_extension_v1_manifest.json",
        ],
        "output": "development300.csv",
        "expected_candidates": 300,
        "expected_samples": 1800,
        "expected_batches": set(range(0, 30)),
        "expected_global_indices": set(range(0, 300)),
    },
    "external": {
        "sources": [
            "study1_35f100_external_v1_manifest.json",
        ],
        "output": "external100.csv",
        "expected_candidates": 100,
        "expected_samples": 600,
        "expected_batches": set(range(30, 40)),
        "expected_global_indices": set(range(300, 400)),
    },
    "sealed": {
        "sources": [
            "study1_35f100_sealed_v1_manifest.json",
        ],
        "output": "sealed100.csv",
        "expected_candidates": 100,
        "expected_samples": 600,
        "expected_batches": set(range(40, 50)),
        "expected_global_indices": set(range(400, 500)),
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest(filename: str) -> dict:
    path = SOURCE_DIR / filename
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def stable_value(records: list[dict], field: str):
    values = {record[field] for record in records}
    if len(values) != 1:
        raise ValueError(
            f"Inconsistent {field}: {sorted(values, key=str)}"
        )
    return next(iter(values))


def build_panel(panel_name: str, spec: dict) -> tuple[list[dict], dict]:
    grouped = defaultdict(list)
    source_metadata = []
    sample_count = 0

    for filename in spec["sources"]:
        manifest = load_manifest(filename)

        source_metadata.append(
            {
                "filename": filename,
                "schema": manifest.get("schema"),
                "campaign_id": manifest.get("campaign_id"),
                "sha256": sha256_file(SOURCE_DIR / filename),
            }
        )

        for sample in manifest.get("samples", []):
            sample_count += 1

            if sample.get("status") != "DONE":
                raise ValueError(
                    f"{filename}: non-DONE sample "
                    f"{sample.get('candidate_id')} "
                    f"{sample.get('emotion')}"
                )

            grouped[sample["candidate_id"]].append(
                {
                    "sample": sample,
                    "source_manifest": filename,
                    "source_campaign_id": manifest.get("campaign_id"),
                }
            )

    rows = []

    for candidate_id, items in grouped.items():
        records = [item["sample"] for item in items]

        if len(records) != 6:
            raise ValueError(
                f"{candidate_id}: expected 6 emotion records, "
                f"found {len(records)}"
            )

        emotions = {record["emotion"] for record in records}

        if emotions != EMOTIONS:
            raise ValueError(
                f"{candidate_id}: emotion set mismatch: {sorted(emotions)}"
            )

        if all("global_candidate_index" in record for record in records):
            global_candidate_index = stable_value(
                records, "global_candidate_index"
            )
        elif all("candidate_index" in record for record in records):
            global_candidate_index = stable_value(
                records, "candidate_index"
            )
        else:
            raise ValueError(
                f"{candidate_id}: neither global_candidate_index "
                f"nor candidate_index is available"
            )
        batch_index = stable_value(records, "batch_index")
        batch_seed = stable_value(records, "batch_seed")
        batch_position = stable_value(records, "batch_position")
        batch_size = stable_value(records, "batch_size")
        target_age = stable_value(records, "target_age")
        design_age = stable_value(records, "design_age")
        gender = stable_value(records, "gender")

        match = re.fullmatch(r"b(\d+)p(\d+)", candidate_id)

        if match is None:
            raise ValueError(
                f"Invalid candidate_id format: {candidate_id}"
            )

        id_batch = int(match.group(1))
        id_position = int(match.group(2))

        if id_batch != batch_index:
            raise ValueError(
                f"{candidate_id}: candidate ID batch does not match "
                f"batch_index={batch_index}"
            )

        if id_position != batch_position:
            raise ValueError(
                f"{candidate_id}: candidate ID position does not match "
                f"batch_position={batch_position}"
            )

        expected_global = batch_index * batch_size + batch_position

        if global_candidate_index != expected_global:
            raise ValueError(
                f"{candidate_id}: global_candidate_index="
                f"{global_candidate_index}, expected={expected_global}"
            )

        expected_seed = 1502000 + batch_index

        if batch_seed != expected_seed:
            raise ValueError(
                f"{candidate_id}: batch_seed={batch_seed}, "
                f"expected={expected_seed}"
            )

        source_manifests = sorted(
            {item["source_manifest"] for item in items}
        )
        source_campaigns = sorted(
            {item["source_campaign_id"] for item in items}
        )

        if len(source_manifests) != 1 or len(source_campaigns) != 1:
            raise ValueError(
                f"{candidate_id}: candidate spans multiple source manifests"
            )

        rows.append(
            {
                "panel": panel_name,
                "candidate_id": candidate_id,
                "global_candidate_index": global_candidate_index,
                "batch_index": batch_index,
                "batch_seed": batch_seed,
                "batch_position": batch_position,
                "batch_size": batch_size,
                "target_age": target_age,
                "design_age": design_age,
                "gender": gender,
                "emotion_count": len(emotions),
                "source_campaign_id": source_campaigns[0],
                "source_manifest": source_manifests[0],
            }
        )

    rows.sort(key=lambda row: row["global_candidate_index"])

    if len(rows) != spec["expected_candidates"]:
        raise ValueError(
            f"{panel_name}: expected {spec['expected_candidates']} candidates, "
            f"found {len(rows)}"
        )

    if sample_count != spec["expected_samples"]:
        raise ValueError(
            f"{panel_name}: expected {spec['expected_samples']} samples, "
            f"found {sample_count}"
        )

    actual_batches = {row["batch_index"] for row in rows}

    if actual_batches != spec["expected_batches"]:
        raise ValueError(
            f"{panel_name}: batch set mismatch"
        )

    actual_global_indices = {
        row["global_candidate_index"] for row in rows
    }

    if actual_global_indices != spec["expected_global_indices"]:
        raise ValueError(
            f"{panel_name}: global candidate index mismatch"
        )

    report = {
        "panel": panel_name,
        "candidate_count": len(rows),
        "sample_count": sample_count,
        "batch_indices": sorted(actual_batches),
        "batch_seeds": sorted({row["batch_seed"] for row in rows}),
        "first_candidate": rows[0]["candidate_id"],
        "last_candidate": rows[-1]["candidate_id"],
        "source_manifests": source_metadata,
        "validation": "PASS",
    }

    return rows, report


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "panel",
        "candidate_id",
        "global_candidate_index",
        "batch_index",
        "batch_seed",
        "batch_position",
        "batch_size",
        "target_age",
        "design_age",
        "gender",
        "emotion_count",
        "source_campaign_id",
        "source_manifest",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    all_rows = []
    reports = {}

    for panel_name, spec in PANELS.items():
        rows, report = build_panel(panel_name, spec)

        write_csv(MANIFEST_DIR / spec["output"], rows)

        reports[panel_name] = report
        all_rows.extend(rows)

        print(
            f"{panel_name}: "
            f"{report['candidate_count']} candidates, "
            f"{report['sample_count']} samples, "
            f"{report['first_candidate']} -> "
            f"{report['last_candidate']} : PASS"
        )

    candidate_ids = [row["candidate_id"] for row in all_rows]
    global_indices = [
        row["global_candidate_index"] for row in all_rows
    ]

    if len(candidate_ids) != 500:
        raise ValueError(
            f"Expected 500 candidates total, found {len(candidate_ids)}"
        )

    if len(set(candidate_ids)) != 500:
        raise ValueError("Duplicate candidate_id detected")

    if len(set(global_indices)) != 500:
        raise ValueError("Duplicate global_candidate_index detected")

    if set(global_indices) != set(range(500)):
        raise ValueError("Global candidate indices are not exactly 0-499")

    build_report = {
        "validation": "PASS",
        "total_candidates": 500,
        "total_samples": 3000,
        "expected_emotions": sorted(EMOTIONS),
        "candidate_definition": (
            "fixed batch-seed x batch-position stochastic candidate; "
            "not human speaker ground truth"
        ),
        "panels": reports,
    }

    report_path = MANIFEST_DIR / "manifest_build_report.json"

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(
            build_report,
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")

    print("TOTAL: 500 candidates, 3000 samples : PASS")
    print(f"Report: {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
