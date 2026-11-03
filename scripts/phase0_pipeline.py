"""Run the non-destructive Phase 0 EEG dataset audit."""

import argparse
import csv
import json
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd

try:
    from .inspect_eeg_datasets import DATASETS, run_review
except ImportError:
    from inspect_eeg_datasets import DATASETS, run_review


PROJECT_DIR = Path(__file__).resolve().parent.parent


def write_csv(path: Path, rows, fieldnames=None) -> None:
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_value(value: str) -> object:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def package_versions() -> dict[str, str]:
    versions = {}
    for package in ("mne", "numpy", "pandas", "matplotlib", "scipy"):
        try:
            module = __import__(package)
            versions[package] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[package] = "not installed"
    return versions


def build_subject_manifest(recordings: list[dict]) -> list[dict]:
    subjects = {}
    for recording in recordings:
        dataset_prefixed_id = (
            f"{recording.get('dataset')}/{recording.get('subject_id')}"
            if recording.get("dataset")
            else recording.get("subject_id", "")
        )
        key = (recording.get("dataset", ""), recording.get("subject_id", ""))
        subject = subjects.setdefault(key, {
            "dataset": recording.get("dataset", ""),
            "subject_id": recording.get("subject_id", ""),
            "dataset_subject_id": dataset_prefixed_id,
            "diagnosis_original": recording.get("diagnosis", ""),
            "age": recording.get("age", ""),
            "sex": recording.get("sex", ""),
            "mmse": recording.get("mmse", ""),
            "recording_count": 0,
            "recording_files": [],
            "conditions": set(),
            "quality_flags": set(),
            "statuses": set(),
        })
        subject["recording_count"] += 1
        subject["recording_files"].append(recording.get("file", ""))
        subject["conditions"].add(recording.get("condition", ""))
        subject["statuses"].add(recording.get("status", ""))
        flags = json_value(recording.get("quality_flags", "[]"))
        if isinstance(flags, list):
            subject["quality_flags"].update(flags)

    rows = []
    for subject in subjects.values():
        rows.append({
            key: ";".join(sorted(value)) if isinstance(value, set) else
            ";".join(value) if isinstance(value, list) else value
            for key, value in subject.items()
        })
    return rows


def build_qc_summary(recordings: list[dict]) -> list[dict]:
    rows = []
    for recording in recordings:
        flags = json_value(recording.get("quality_flags", "[]"))
        rows.append({
            "dataset": recording.get("dataset", ""),
            "subject_id": recording.get("subject_id", ""),
            "recording": recording.get("file", ""),
            "status": recording.get("status", ""),
            "short_recording": "short_recording" in (flags or []),
            "sample_counts_consistent": recording.get("sample_counts_consistent", ""),
            "channel_names": recording.get("channel_names", ""),
            "quality_flags": ";".join(flags) if isinstance(flags, list) else flags,
            "missing_channels": recording.get("missing_channels", ""),
            "channel_invalid_values": recording.get("channel_invalid_values", ""),
            "error": recording.get("error", ""),
            "exclusion_recommended": False,
            "exclusion_reason": "",
            "review_required": bool(flags) or recording.get("status") != "ok",
        })
    return rows


def build_channel_compatibility(recordings: list[dict]) -> list[dict]:
    aliases = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
    availability = {}
    for recording in recordings:
        names = json_value(recording.get("channel_names", "[]"))
        if not isinstance(names, list):
            continue
        dataset = recording.get("dataset", "")
        normalized = set()
        for name in names:
            cleaned = "".join(character for character in str(name).upper() if character.isalnum())
            normalized.add(aliases.get(cleaned, cleaned))
        for channel in normalized:
            availability.setdefault(channel, set()).add(dataset)
    return [{
        "standardized_channel": channel,
        "in_adftd": "ADFTD" in datasets,
        "in_adfsu": "ADFSU" in datasets,
        "recording_datasets": ";".join(sorted(datasets)),
    } for channel, datasets in sorted(availability.items())]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(PROJECT_DIR), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""


def write_markdown_reports(output: Path, summaries: list[dict]) -> None:
    by_dataset = {row["dataset"]: row for row in summaries}
    harmonisation = """# Harmonisation Plan

## Status

No final harmonisation protocol is recommended until acquisition metadata has
been verified for every cohort.

## Evidence-based starting configuration

- Keep raw recordings and participant-level provenance unchanged.
- Use 19 standard 10-20 scalp locations as the provisional common channel
  configuration.
- Normalize legacy temporal names T3, T4, T5, and T6 to T7, T8, P7, and P8.
- Do not infer ADFSU duration or sampling frequency from sample counts.
- Run input-source and montage ablations only after acquisition metadata is
  sufficient to define each candidate independently.

## Alternatives requiring separate evidence

1. Common-channel subset based on observed availability.
2. Montage-based comparison using verified coordinates.
3. Higher-density source-specific pipelines evaluated without cross-cohort tuning.

This plan deliberately does not prescribe a final resampling frequency,
reference, filter, epoch duration, or exclusion threshold.
"""
    (output / "harmonisation_plan.md").write_text(harmonisation, encoding="utf-8")

    lines = ["# Dataset Audit Report", "", "## Cohort summary", ""]
    for dataset in sorted(by_dataset):
        summary = by_dataset[dataset]
        lines.append(
            f"- **{dataset}:** {summary.get('patients', 0)} participants, "
            f"{summary.get('eeg_recordings', 0)} EEG recordings, "
            f"{summary.get('signal_sessions', 0)} signal sessions, "
            f"{summary.get('warnings', '[]')} warnings."
        )
    lines.extend([
        "",
        "## Readiness decision",
        "",
        "This audit establishes inventory, integrity, and metadata readiness. "
        "It does not establish model-ready EEG preprocessing readiness.",
        "",
        "See `dataset_readiness.json`, `qc_summary.csv`, "
        "`channel_compatibility.csv`, and `harmonisation_plan.md` for evidence.",
    ])
    (output / "dataset_audit_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adftd", type=Path, default=DATASETS["ADFTD"])
    parser.add_argument("--adfsu", type=Path, default=DATASETS["ADFSU"])
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "Processed" / "phase0_audit")
    parser.add_argument("--min-duration-seconds", type=float, default=60.0)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    summaries = run_review(
        {"ADFTD": args.adftd, "ADFSU": args.adfsu},
        output,
        min_duration_seconds=args.min_duration_seconds,
    )
    inventory = output / "recording_inventory.csv"
    recordings = pd.read_csv(inventory, dtype=str, keep_default_na=False).to_dict("records")

    write_csv(output / "recording_manifest.csv", recordings)
    write_csv(output / "subject_manifest.csv", build_subject_manifest(recordings))
    write_csv(output / "qc_summary.csv", build_qc_summary(recordings))
    write_csv(output / "channel_compatibility.csv", build_channel_compatibility(recordings))
    write_markdown_reports(output, summaries)

    metadata = {
        "status": "audit_complete_metadata_not_ready" if any(
            summary.get("status") != "ok" for summary in summaries
        ) else "audit_complete_review_required",
        "generated_from": "local files only",
        "datasets": summaries,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": package_versions(),
            "git_commit": git_commit(),
        },
        "unresolved": [
            "ADFSU sampling frequency",
            "ADFSU amplitude unit",
            "ADFSU reference",
            "ADFSU filtering and artifact handling",
            "ADFSU duration cannot be calculated without verified sampling frequency",
        ],
    }
    (output / "dataset_readiness.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output),
        "subjects": len(build_subject_manifest(recordings)),
        "recordings": len(recordings),
    }, indent=2))


if __name__ == "__main__":
    main()
