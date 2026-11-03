import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASETS = {
    "ADFTD": PROJECT_DIR / "datasets" / "ADFTD",
    "ADFSU": PROJECT_DIR / "datasets" / "ADFSU",
}
EEG_EXTENSIONS = {".set", ".edf", ".bdf", ".fif", ".vhdr"}

SUMMARY_COLUMNS = [
    "dataset",
    "path",
    "status",
    "total_files",
    "eeg_files",
    "eeg_recordings",
    "raw_recordings",
    "derivative_recordings",
    "text_signal_files",
    "signal_sessions",
    "session_directories",
    "empty_signal_sessions",
    "patients",
    "class_counts",
    "recording_class_counts",
    "conditions",
    "diagnosis_source",
    "metadata_files",
    "participants",
    "file_formats",
    "sampling_rates_hz",
    "channel_counts",
    "text_signal_channel_counts",
    "text_signal_sample_counts",
    "channel_names_consistent",
    "short_recordings",
    "sessions_with_inconsistent_sample_counts",
    "warnings",
]
RECORDING_COLUMNS = [
    "dataset",
    "file",
    "channel_files",
    "data_type",
    "source_type",
    "subject_id",
    "patient_id",
    "diagnosis",
    "diagnosis_source",
    "age",
    "sex",
    "mmse",
    "condition",
    "reference",
    "reference_source",
    "montage",
    "status",
    "sampling_rate_hz",
    "channel_count",
    "channel_names",
    "channel_sample_counts",
    "channel_invalid_values",
    "expected_channel_count",
    "missing_channels",
    "sample_count_min",
    "sample_count_max",
    "sample_counts_consistent",
    "duration_seconds",
    "annotation_count",
    "annotation_descriptions",
    "quality_flags",
    "error",
]
METADATA_COLUMNS = [
    "dataset",
    "file",
    "status",
    "row_count",
    "participant_id_column",
    "unique_participants",
    "duplicate_participant_ids",
    "missing_participant_ids",
    "label_column",
    "missing_labels",
    "label_counts",
    "error",
]


def read_eeg_recording(path: Path) -> Any:
    import mne

    readers = {
        ".set": mne.io.read_raw_eeglab,
        ".edf": mne.io.read_raw_edf,
        ".bdf": mne.io.read_raw_bdf,
        ".fif": mne.io.read_raw_fif,
        ".vhdr": mne.io.read_raw_brainvision,
    }
    reader = readers.get(path.suffix.lower())
    if reader is None:
        raise ValueError(f"Unsupported EEG format: {path.suffix}")
    return reader(path, preload=False, verbose="ERROR")


def inspect_recording(
    eeg_file: Path,
    dataset_path: Path,
    dataset_name: str,
    min_duration_seconds: float,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "dataset": dataset_name,
        "file": str(eeg_file.relative_to(dataset_path)),
        "channel_files": "",
        "data_type": "mne_recording",
        "source_type": "derivative"
        if "derivatives" in eeg_file.relative_to(dataset_path).parts
        else "raw",
        "subject_id": "",
        "patient_id": "",
        "diagnosis": "",
        "diagnosis_source": "",
        "age": "",
        "sex": "",
        "mmse": "",
        "condition": "",
        "reference": "",
        "reference_source": "",
        "montage": "",
        "status": "error",
        "sampling_rate_hz": "",
        "channel_count": "",
        "channel_names": "",
        "duration_seconds": "",
        "annotation_count": "",
        "annotation_descriptions": "",
        "quality_flags": "",
        "error": "",
    }
    if context:
        result.update(context)
    raw = None

    try:
        if eeg_file.stat().st_size == 0:
            raise ValueError(
                f"Recording file is empty (0 bytes): {eeg_file.name}"
            )
        raw = read_eeg_recording(eeg_file)
        sampling_rate = float(raw.info["sfreq"])
        duration = raw.n_times / sampling_rate
        channel_names = list(raw.ch_names)
        annotations = raw.annotations
        descriptions = Counter(str(item) for item in annotations.description)
        flags = []
        if duration < min_duration_seconds:
            flags.append("short_recording")

        result.update(
            status="ok",
            sampling_rate_hz=sampling_rate,
            channel_count=len(channel_names),
            channel_names=json.dumps(channel_names),
            duration_seconds=round(duration, 3),
            annotation_count=len(annotations),
            annotation_descriptions=json.dumps(dict(descriptions)),
            quality_flags=json.dumps(flags),
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        if eeg_file.stat().st_size == 0:
            result["quality_flags"] = json.dumps(["empty_recording_file"])
    finally:
        if raw is not None:
            try:
                raw.close()
            except Exception as exc:
                close_error = f"Could not close recording: {type(exc).__name__}: {exc}"
                result["error"] = "; ".join(
                    part for part in (result["error"], close_error) if part
                )
                result["status"] = "error"

    return result


def load_bids_group_labels(dataset_path: Path) -> dict[str, str]:
    groups_file = dataset_path / "participants.json"
    if not groups_file.is_file():
        return {}
    with groups_file.open("r", encoding="utf-8-sig") as stream:
        description = json.load(stream)
    group_metadata = next(
        (
            value
            for key, value in description.items()
            if key.casefold() == "group"
        ),
        {},
    )
    levels = group_metadata.get("Levels", {})
    return {str(code): str(label) for code, label in levels.items()}


def load_recording_context(
    eeg_file: Path,
    dataset_path: Path,
    participant_records: dict[str, dict[str, str]],
    group_labels: dict[str, str],
) -> tuple[dict[str, Any], list[str]]:
    relative_parts = eeg_file.relative_to(dataset_path).parts
    subject_match = re.search(r"sub-[^_]+", eeg_file.name, re.IGNORECASE)
    subject_id = subject_match.group(0) if subject_match else ""
    participant = participant_records.get(subject_id, {})
    task_match = re.search(r"(?:^|_)task-([^_]+)", eeg_file.name, re.IGNORECASE)
    condition = task_match.group(1) if task_match else ""
    if condition.casefold() == "eyesclosed":
        condition = "eyes-closed"
    elif condition.casefold() == "eyesopen":
        condition = "eyes-open"

    context: dict[str, Any] = {
        "subject_id": subject_id,
        "patient_id": subject_id,
        "diagnosis": group_labels.get(
            participant.get("Group", ""), participant.get("Group", "")
        ),
        "diagnosis_source": "participants.tsv"
        if participant.get("Group")
        else "",
        "age": participant.get("Age", ""),
        "sex": participant.get("Gender", ""),
        "mmse": participant.get("MMSE", ""),
        "condition": condition,
    }
    warnings = []
    sidecar_path: Path | None = eeg_file.with_suffix(".json")
    if "derivatives" in relative_parts:
        subject_dir = next(
            (part for part in relative_parts if part.startswith("sub-")), ""
        )
        raw_sidecar = (
            dataset_path / subject_dir / "eeg" / eeg_file.with_suffix(".json").name
        )
        if raw_sidecar.is_file():
            sidecar_path = raw_sidecar
        else:
            sidecar_path = None

    if sidecar_path is not None and sidecar_path.is_file():
        try:
            with sidecar_path.open("r", encoding="utf-8-sig") as stream:
                sidecar = json.load(stream)
            context["montage"] = sidecar.get("EEGPlacementScheme", "")
            if "derivatives" in relative_parts:
                readme_path = dataset_path / "README"
                readme = (
                    readme_path.read_text(encoding="utf-8-sig", errors="replace")
                    if readme_path.is_file()
                    else ""
                )
                reference_match = re.search(
                    r"re-referenced to\s+([A-Za-z0-9-]+)",
                    readme,
                    re.IGNORECASE,
                )
                if reference_match:
                    context["reference"] = reference_match.group(1)
                    context["reference_source"] = (
                        "dataset README derivative preprocessing description"
                    )
                else:
                    context["reference"] = sidecar.get("EEGReference", "")
                    context["reference_source"] = (
                        "raw BIDS sidecar; derivative reference not documented"
                    )
                    warnings.append(
                        f"derivative_reference_not_documented:{eeg_file.name}"
                    )
            else:
                context["reference"] = sidecar.get("EEGReference", "")
                context["reference_source"] = "BIDS EEG sidecar"
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            warnings.append(
                f"eeg_sidecar_read_error:{sidecar_path.name}:"
                f"{type(exc).__name__}"
            )
    else:
        warnings.append(f"eeg_sidecar_not_found:{eeg_file.name}")

    return context, warnings


def inspect_text_signal_session(
    patient_dir: Path,
    dataset_path: Path,
    dataset_name: str,
    expected_channel_names: set[str] | None = None,
) -> dict[str, Any]:
    relative_parts = patient_dir.relative_to(dataset_path).parts
    result: dict[str, Any] = {
        "dataset": dataset_name,
        "file": str(patient_dir.relative_to(dataset_path)),
        "channel_files": "[]",
        "data_type": "channel_text",
        "source_type": "raw",
        "subject_id": f"{relative_parts[-3]}/{patient_dir.name}"
        if len(relative_parts) >= 3
        else patient_dir.name,
        "patient_id": patient_dir.name,
        "diagnosis": relative_parts[-3] if len(relative_parts) >= 3 else "",
        "diagnosis_source": "diagnosis folder"
        if len(relative_parts) >= 3
        else "",
        "age": "",
        "sex": "",
        "mmse": "",
        "condition": relative_parts[-2] if len(relative_parts) >= 2 else "",
        "reference": "",
        "reference_source": "",
        "montage": "",
        "status": "ok",
        "sampling_rate_hz": "",
        "channel_count": 0,
        "channel_names": "[]",
        "channel_sample_counts": "{}",
        "channel_invalid_values": "{}",
        "expected_channel_count": len(expected_channel_names)
        if expected_channel_names
        else "",
        "missing_channels": "[]",
        "sample_count_min": "",
        "sample_count_max": "",
        "sample_counts_consistent": "",
        "duration_seconds": "",
        "annotation_count": "",
        "annotation_descriptions": "",
        "quality_flags": "[]",
        "error": "",
    }
    channel_counts: dict[str, int] = {}
    invalid_counts: dict[str, int] = {}
    errors = []

    channel_files = sorted(patient_dir.glob("*.txt"))
    result["channel_files"] = json.dumps(
        [str(path.relative_to(dataset_path)) for path in channel_files]
    )
    for channel_file in channel_files:
        channel = channel_file.stem
        sample_count = 0
        invalid_count = 0
        try:
            with channel_file.open("r", encoding="utf-8-sig") as stream:
                for line in stream:
                    for value in line.split():
                        try:
                            sample = float(value)
                            if not math.isfinite(sample):
                                invalid_count += 1
                            else:
                                sample_count += 1
                        except ValueError:
                            invalid_count += 1
            channel_counts[channel] = sample_count
            invalid_counts[channel] = invalid_count
            if invalid_count:
                errors.append(f"{channel_file.name}: {invalid_count} invalid values")
        except (OSError, UnicodeError) as exc:
            errors.append(f"{channel_file.name}: {type(exc).__name__}: {exc}")

    sample_counts = list(channel_counts.values())
    if not channel_files:
        errors.append("No channel text files found")
    sample_counts_consistent = len(set(sample_counts)) <= 1
    flags = []
    if not channel_counts:
        flags.append("no_channel_text_files")
    missing_channels = (
        sorted(expected_channel_names - channel_counts.keys())
        if expected_channel_names and channel_counts
        else []
    )
    if missing_channels:
        flags.append("channels_missing_vs_dataset_mode")
    if not sample_counts_consistent:
        flags.append("channel_sample_counts_vary")
    if any(invalid_counts.values()):
        flags.append("invalid_channel_values")
    result.update(
        status="error" if errors else "ok",
        channel_count=len(channel_counts),
        channel_names=json.dumps(sorted(channel_counts)),
        channel_sample_counts=json.dumps(channel_counts),
        channel_invalid_values=json.dumps(invalid_counts),
        missing_channels=json.dumps(missing_channels),
        sample_count_min=min(sample_counts) if sample_counts else "",
        sample_count_max=max(sample_counts) if sample_counts else "",
        sample_counts_consistent=sample_counts_consistent if sample_counts else "",
        quality_flags=json.dumps(flags),
        error="; ".join(errors),
    )
    return result


def inspect_metadata(
    metadata_file: Path,
    dataset_path: Path,
    dataset_name: str,
    group_labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "dataset": dataset_name,
        "file": str(metadata_file.relative_to(dataset_path)),
        "status": "error",
        "row_count": "",
        "participant_id_column": "",
        "unique_participants": "",
        "duplicate_participant_ids": "",
        "missing_participant_ids": "",
        "label_column": "",
        "missing_labels": "",
        "label_counts": "",
        "error": "",
    }

    try:
        with metadata_file.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            columns = [column.strip() for column in (reader.fieldnames or [])]
            rows = [
                {
                    str(key).strip(): (value or "").strip()
                    for key, value in row.items()
                    if key is not None
                }
                for row in reader
            ]

        participant_column = next(
            (column for column in columns if column.casefold() == "participant_id"),
            None,
        )
        label_column = next(
            (
                column
                for column in columns
                if column.casefold() in {"group", "diagnosis"}
            ),
            None,
        )
        participant_ids = (
            [row.get(participant_column, "") for row in rows]
            if participant_column
            else []
        )
        present_ids = [value for value in participant_ids if value]
        labels = (
            [row.get(label_column, "") for row in rows] if label_column else []
        )
        present_labels = [
            (group_labels or {}).get(value, value) for value in labels if value
        ]
        group_column = next(
            (column for column in columns if column.casefold() == "group"), None
        )
        participant_records = []
        if participant_column:
            for row in rows:
                participant_id = row.get(participant_column, "")
                if not participant_id:
                    continue
                group = row.get(group_column, "") if group_column else ""
                participant_records.append(
                    {
                        "participant_id": participant_id,
                        "Group": group,
                        "diagnosis": (group_labels or {}).get(group, group),
                        "Age": row.get(
                            next(
                                (column for column in columns if column.casefold() == "age"),
                                "",
                            ),
                            "",
                        ),
                        "Gender": row.get(
                            next(
                                (
                                    column
                                    for column in columns
                                    if column.casefold() in {"gender", "sex"}
                                ),
                                "",
                            ),
                            "",
                        ),
                        "MMSE": row.get(
                            next(
                                (column for column in columns if column.casefold() == "mmse"),
                                "",
                            ),
                            "",
                        ),
                    }
                )

        result.update(
            status="ok",
            row_count=len(rows),
            participant_id_column=participant_column or "",
            unique_participants=len(set(present_ids)),
            duplicate_participant_ids=len(present_ids) - len(set(present_ids)),
            missing_participant_ids=(
                len(rows) - len(present_ids) if participant_column else ""
            ),
            label_column=label_column or "",
            missing_labels=len(rows) - len(present_labels) if label_column else "",
            label_counts=json.dumps(dict(Counter(present_labels))),
            _participant_ids=present_ids,
            _participant_records=participant_records,
        )
    except (OSError, UnicodeError, csv.Error) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def scan_dataset(
    dataset_name: str,
    dataset_path: Path,
    min_duration_seconds: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    summary: dict[str, Any] = {
        "dataset": dataset_name,
        "path": str(dataset_path),
        "status": "ok",
        "total_files": 0,
        "eeg_files": 0,
        "eeg_recordings": 0,
        "raw_recordings": 0,
        "derivative_recordings": 0,
        "text_signal_files": 0,
        "signal_sessions": 0,
        "session_directories": 0,
        "empty_signal_sessions": 0,
        "patients": 0,
        "class_counts": "{}",
        "recording_class_counts": "{}",
        "conditions": "[]",
        "diagnosis_source": "unavailable",
        "metadata_files": 0,
        "participants": 0,
        "file_formats": "{}",
        "sampling_rates_hz": "[]",
        "channel_counts": "[]",
        "channel_names_consistent": "",
        "short_recordings": 0,
        "sessions_with_inconsistent_sample_counts": 0,
        "text_signal_channel_counts": "[]",
        "text_signal_sample_counts": "[]",
        "warnings": "[]",
    }
    recordings: list[dict[str, Any]] = []
    metadata_reports: list[dict[str, Any]] = []
    warnings: list[str] = []

    if not dataset_path.exists():
        summary["status"] = "missing"
        warnings.append("dataset_folder_not_found")
        summary["warnings"] = json.dumps(warnings)
        return summary, recordings, metadata_reports

    if not dataset_path.is_dir():
        summary["status"] = "not_a_directory"
        warnings.append("dataset_path_is_not_a_directory")
        summary["warnings"] = json.dumps(warnings)
        return summary, recordings, metadata_reports

    all_files = sorted(path for path in dataset_path.rglob("*") if path.is_file())
    extensions = Counter(path.suffix.lower() or "[no extension]" for path in all_files)
    eeg_files = [path for path in all_files if path.suffix.lower() in EEG_EXTENSIONS]
    metadata_files = [
        path for path in all_files if path.name.casefold() == "participants.tsv"
    ]
    patient_directories = sorted(
        {
            path
            for path in dataset_path.rglob("*")
            if path.is_dir()
            and re.fullmatch(r"(?:paciente|patient)[_-]?\w+", path.name, re.I)
        }
    )
    text_signal_files = [
        path
        for path in all_files
        if path.suffix.casefold() == ".txt"
        and any(path.parent == patient_dir for patient_dir in patient_directories)
    ]
    channel_layouts = Counter(
        tuple(sorted(channel_file.stem for channel_file in patient_dir.glob("*.txt")))
        for patient_dir in patient_directories
        if list(patient_dir.glob("*.txt"))
    )
    expected_channel_names = (
        set(channel_layouts.most_common(1)[0][0]) if channel_layouts else None
    )

    summary.update(
        total_files=len(all_files),
        eeg_files=len(eeg_files),
        eeg_recordings=len(eeg_files),
        text_signal_files=len(text_signal_files),
        signal_sessions=sum(
            bool(list(patient_dir.glob("*.txt")))
            for patient_dir in patient_directories
        ),
        session_directories=len(patient_directories),
        empty_signal_sessions=sum(
            not list(patient_dir.glob("*.txt")) for patient_dir in patient_directories
        ),
        metadata_files=len(metadata_files),
        file_formats=json.dumps(dict(extensions)),
    )

    patient_keys = set()
    conditions = set()
    for patient_dir in patient_directories:
        relative_parts = patient_dir.relative_to(dataset_path).parts
        diagnosis = relative_parts[-3] if len(relative_parts) >= 3 else ""
        condition = relative_parts[-2] if len(relative_parts) >= 2 else ""
        patient_keys.add(
            (
                diagnosis,
                f"{diagnosis}/{patient_dir.name}" if diagnosis else patient_dir.name,
            )
        )
        if condition:
            conditions.add(condition)
        report = inspect_text_signal_session(
            patient_dir,
            dataset_path,
            dataset_name,
            expected_channel_names,
        )
        recordings.append(report)
        if report["sample_counts_consistent"] is False:
            summary["sessions_with_inconsistent_sample_counts"] += 1
        if report["status"] == "error":
            if "no_channel_text_files" in json.loads(report["quality_flags"]):
                warnings.append(f"empty_signal_session:{report['file']}")
            else:
                warnings.append(f"channel_text_read_error:{report['file']}")
        for quality_flag in json.loads(report["quality_flags"]):
            if quality_flag == "channels_missing_vs_dataset_mode":
                warnings.append(f"channels_missing_vs_dataset_mode:{report['file']}")
            elif quality_flag == "invalid_channel_values":
                warnings.append(f"invalid_channel_values:{report['file']}")

    summary["patients"] = len(patient_keys)
    summary["conditions"] = json.dumps(sorted(conditions))
    if metadata_files and patient_directories:
        summary["diagnosis_source"] = "participants.tsv_and_folder_names"
    elif metadata_files:
        summary["diagnosis_source"] = "participants.tsv"
    elif patient_directories:
        summary["diagnosis_source"] = "folder_names"
    summary["text_signal_channel_counts"] = json.dumps(
        sorted(
            {
                report["channel_count"]
                for report in recordings
                if report["data_type"] == "channel_text"
                and report["channel_count"]
            }
        )
    )
    summary["text_signal_sample_counts"] = json.dumps(
        sorted(
            {
                sample_count
                for report in recordings
                if report["data_type"] == "channel_text"
                and report["channel_count"]
                for sample_count in json.loads(
                    report["channel_sample_counts"]
                ).values()
            }
        )
    )

    group_labels: dict[str, str] = {}
    if (dataset_path / "participants.json").is_file():
        try:
            group_labels = load_bids_group_labels(dataset_path)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            warnings.append(
                f"participant_description_read_error:{type(exc).__name__}"
            )

    participant_records: dict[str, dict[str, str]] = {}
    for metadata_file in metadata_files:
        report = inspect_metadata(
            metadata_file, dataset_path, dataset_name, group_labels
        )
        metadata_reports.append(report)
        if report["status"] == "error":
            warnings.append(f"metadata_read_error:{report['file']}")
        else:
            participant_records.update(
                {
                    participant["participant_id"]: participant
                    for participant in report.get("_participant_records", [])
                }
            )

    participant_ids = set()
    for report in metadata_reports:
        if report["status"] == "ok":
            participant_ids.update(report.get("_participant_ids", []))
            if not report["participant_id_column"]:
                warnings.append(f"participant_id_column_missing:{report['file']}")
            if not report["label_column"]:
                warnings.append(f"diagnosis_column_missing:{report['file']}")
            if report["missing_participant_ids"]:
                warnings.append(f"participant_ids_missing:{report['file']}")
            if report["duplicate_participant_ids"]:
                warnings.append(f"duplicate_participant_ids:{report['file']}")

    summary["participants"] = len(participant_ids)

    if not metadata_files and patient_directories:
        warnings.append("participant_demographics_not_available")
    elif not metadata_files:
        warnings.append("participants_tsv_not_found")

    for eeg_file in eeg_files:
        context, context_warnings = load_recording_context(
            eeg_file, dataset_path, participant_records, group_labels
        )
        warnings.extend(context_warnings)
        report = inspect_recording(
            eeg_file,
            dataset_path,
            dataset_name,
            min_duration_seconds,
            context,
        )
        recordings.append(report)
        if report["status"] == "error":
            warnings.append(f"recording_read_error:{report['file']}")
        elif "short_recording" in json.loads(report["quality_flags"]):
            summary["short_recordings"] += 1

    summary["raw_recordings"] = sum(
        report["source_type"] == "raw" and bool(report["channel_count"])
        for report in recordings
    )
    summary["derivative_recordings"] = sum(
        report["source_type"] == "derivative" and bool(report["channel_count"])
        for report in recordings
    )
    summary["eeg_recordings"] += summary["signal_sessions"]
    for report in recordings:
        if report["condition"]:
            conditions.add(report["condition"])
        if report["subject_id"] and report["diagnosis"]:
            patient_keys.add((report["diagnosis"], report["subject_id"]))
    summary["patients"] = len(
        {
            report["subject_id"]
            for report in recordings
            if report["subject_id"]
            and report["diagnosis"]
            and report["channel_count"]
        }
    )
    class_subjects: dict[str, set[str]] = {}
    for report in recordings:
        if (
            report["diagnosis"]
            and report["subject_id"]
            and report["channel_count"]
        ):
            class_subjects.setdefault(report["diagnosis"], set()).add(
                report["subject_id"]
            )
    summary["class_counts"] = json.dumps(
        {label: len(subjects) for label, subjects in sorted(class_subjects.items())}
    )
    recording_class_counts = Counter(
        report["diagnosis"]
        for report in recordings
        if report["diagnosis"] and report["channel_count"]
    )
    summary["recording_class_counts"] = json.dumps(
        dict(sorted(recording_class_counts.items()))
    )
    summary["conditions"] = json.dumps(sorted(conditions))

    successful_recordings = [
        report
        for report in recordings
        if report["status"] == "ok" and report["data_type"] == "mne_recording"
    ]
    sampling_rates = sorted(
        {report["sampling_rate_hz"] for report in successful_recordings}
    )
    channel_counts = sorted(
        {report["channel_count"] for report in successful_recordings}
    )
    channel_layouts = {
        report["channel_names"] for report in successful_recordings
    }
    summary["sampling_rates_hz"] = json.dumps(sampling_rates)
    summary["channel_counts"] = json.dumps(channel_counts)
    if successful_recordings:
        summary["channel_names_consistent"] = len(channel_layouts) == 1
        if len(sampling_rates) > 1:
            warnings.append("sampling_rate_varies")
        if len(channel_counts) > 1:
            warnings.append("channel_count_varies")
        if len(channel_layouts) > 1:
            warnings.append("channel_names_vary")
    elif patient_directories:
        text_channel_layouts = {
            report["channel_names"]
            for report in recordings
            if report["data_type"] == "channel_text" and report["channel_count"]
        }
        summary["channel_names_consistent"] = len(text_channel_layouts) == 1
    elif eeg_files:
        warnings.append("no_recordings_read_successfully")
    elif not eeg_files and not patient_directories:
        warnings.append("no_supported_eeg_recordings_found")

    if patient_directories:
        warnings.append("sampling_rate_not_available_for_channel_text_data")
    if summary["sessions_with_inconsistent_sample_counts"]:
        warnings.append("channel_sample_counts_vary")
    if summary["short_recordings"]:
        warnings.append("recordings_below_minimum_duration")

    summary["warnings"] = json.dumps(warnings)
    return summary, recordings, metadata_reports


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_review(
    datasets: dict[str, Path],
    results_dir: Path,
    min_duration_seconds: float = 1.0,
) -> list[dict[str, Any]]:
    summaries = []
    recordings = []
    metadata_reports = []

    for dataset_name, dataset_path in datasets.items():
        summary, dataset_recordings, dataset_metadata = scan_dataset(
            dataset_name, dataset_path, min_duration_seconds
        )
        summaries.append(summary)
        recordings.extend(dataset_recordings)
        metadata_reports.extend(dataset_metadata)
        for warning in json.loads(summary["warnings"]):
            print(f"WARNING [{dataset_name}]: {warning}", file=sys.stderr)

    write_csv(results_dir / "dataset_inventory.csv", summaries, SUMMARY_COLUMNS)
    write_csv(
        results_dir / "recording_inventory.csv", recordings, RECORDING_COLUMNS
    )
    write_csv(
        results_dir / "metadata_inventory.csv", metadata_reports, METADATA_COLUMNS
    )
    return summaries


def parse_dataset(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError(
            "datasets must use NAME=PATH, for example --dataset EEG=C:\\data\\eeg"
        )
    return name.strip(), Path(path.strip()).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inventory EEG datasets, metadata, and recording properties."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        type=parse_dataset,
        metavar="NAME=PATH",
        help="dataset to review; repeat to review multiple (replaces built-in defaults)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "results",
        help="directory for CSV reports (default: project results directory)",
    )
    parser.add_argument(
        "--min-duration-seconds",
        type=float,
        default=1.0,
        help="flag recordings shorter than this duration (default: 1 second)",
    )
    args = parser.parse_args()

    if args.min_duration_seconds < 0:
        parser.error("--min-duration-seconds must be non-negative")

    datasets = dict(args.dataset) if args.dataset else DATASETS
    summaries = run_review(datasets, args.output_dir, args.min_duration_seconds)

    print("\nDATASET INVENTORY")
    print("-" * 72)
    for summary in summaries:
        print(
            f"{summary['dataset']}: status={summary['status']}, "
            f"files={summary['total_files']}, EEG recordings="
            f"{summary['eeg_recordings']} "
            f"(raw={summary['raw_recordings']}, "
            f"derivative={summary['derivative_recordings']}), "
            f"patients={summary['patients']}, "
            f"subject class counts={summary['class_counts']}, "
            f"recording rows by class={summary['recording_class_counts']}"
        )
    print(f"\nReports saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
