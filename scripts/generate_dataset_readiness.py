"""Generate non-destructive dataset audit and readiness reports.

This script only reads source metadata and text EEG files. It does not modify
original recordings, resample signals, or create model-ready datasets.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def audit_adftd(root):
    participants_file = root / "participants.tsv"
    participants = pd.read_csv(participants_file, sep="\t", dtype=str)
    levels = {
        "A": "Alzheimer Disease Group",
        "F": "Frontotemporal Dementia Group",
        "C": "Healthy Group",
    }
    participants_file_json = root / "participants.json"
    if participants_file_json.is_file():
        try:
            levels.update({
                str(key): str(value)
                for key, value in json.loads(
                    participants_file_json.read_text(encoding="utf-8-sig")
                ).get("Group", {}).get("Levels", {}).items()
            })
        except (OSError, json.JSONDecodeError):
            pass

    raw = sorted(root.glob("sub-*/eeg/*.set"))
    derivatives = sorted(root.glob("derivatives/sub-*/eeg/*.set"))
    derivative_usable = sum(path.stat().st_size > 0 for path in derivatives)
    standardized = participants["Group"].map(levels)
    duration_values = []
    try:
        import mne
        for recording in raw:
            if recording.stat().st_size == 0:
                continue
            eeg = mne.io.read_raw_eeglab(recording, preload=False, verbose="ERROR")
            duration_values.append(
                {
                    "recording": recording.relative_to(root).as_posix(),
                    "participant_id": recording.parents[1].name,
                    "sampling_rate_hz": float(eeg.info["sfreq"]),
                    "channel_count": len(eeg.ch_names),
                    "channel_names": json_text(list(eeg.ch_names)),
                    "duration_seconds": round(eeg.n_times / eeg.info["sfreq"], 3),
                    "status": "ok",
                }
            )
            eeg.close()
    except Exception as exc:
        duration_values = [{
            "recording": "",
            "participant_id": "",
            "sampling_rate_hz": "",
            "channel_count": "",
            "channel_names": "",
            "duration_seconds": "",
            "status": f"{type(exc).__name__}: {exc}",
        }]

    return {
        "dataset_id": "ADFTD",
        "unique_participants": int(participants["participant_id"].nunique()),
        "raw_recordings": len(raw),
        "derivative_recordings": len(derivatives),
        "derivative_recordings_with_data": derivative_usable,
        "diagnosis_counts": json_text(
            standardized.value_counts().to_dict()
        ),
        "age_available": bool(participants["Age"].notna().any()),
        "sex_available": bool(participants["Gender"].notna().any()),
        "metadata_columns": json_text(list(participants.columns)),
        "condition_verified": "eyes closed",
        "electrode_montage": "10-20 (19 scalp channels)",
        "reference": "raw sidecars: A1 A2; README: derivatives re-referenced A1-A2",
        "original_sampling_frequency_hz": 500,
        "amplitude_unit": "MNE convention: volts; source README reports 10 microvolts/mm",
        "existing_filter": "raw: acquisition high-frequency 70 Hz, amplification time; README does not state a raw digital band-pass. derivatives: 0.5-45 Hz",
        "existing_artifact_handling": "raw exported without device annotations; derivatives use ASR and ICLabel IC rejection",
        "power_line_hz": "sidecar sub-001: 50",
        "recordings": duration_values,
    }


def audit_adfsu(root):
    sessions = []
    expected = None
    diagnosis_counts = Counter()
    diagnosis_patients = defaultdict(set)
    files_by_session = defaultdict(list)

    for folder in sorted(path for path in root.rglob("*") if path.is_dir()):
        files = sorted(folder.glob("*.txt"))
        if not files:
            continue
        relative = folder.relative_to(root)
        diagnosis = relative.parts[0] if len(relative.parts) >= 3 else "UNKNOWN"
        condition = relative.parts[1] if len(relative.parts) >= 3 else "UNKNOWN"
        patient = folder.name
        diagnosis_counts[diagnosis] += 1
        diagnosis_patients[diagnosis].add(patient)

        names = sorted(file.stem for file in files)
        lengths = []
        finite = True
        errors = []
        for file in files:
            try:
                values = np.loadtxt(file, dtype=float)
                lengths.append(len(values))
                finite = finite and np.isfinite(values).all()
            except (OSError, ValueError) as exc:
                errors.append(f"{file.name}: {type(exc).__name__}: {exc}")
        expected = Counter(names)
        sessions.append({
            "dataset_id": "ADFSU",
            "recording_id": relative.as_posix(),
            "participant_id": f"{diagnosis}/{patient}",
            "diagnosis_original": diagnosis,
            "recording_condition": condition,
            "status": "ok" if not errors else "error",
            "channel_count": len(names),
            "channel_names": json_text(names),
            "sample_counts": json_text(lengths),
            "sample_counts_consistent": len(set(lengths)) <= 1,
            "all_values_finite": finite,
            "error": "; ".join(errors),
        })
        files_by_session[relative] = files

    empty_directories = [
        path.relative_to(root).as_posix()
        for path in sorted(root.rglob("*"))
        if path.is_dir() and len(path.relative_to(root).parts) == 3 and not list(path.glob("*.txt"))
    ]
    channel_sets = Counter(json.dumps(
        sorted(file.stem for file in session_files)
    ) for session_files in files_by_session.values())
    return {
        "dataset_id": "ADFSU",
        "unique_dataset_prefixed_participants": sum(
            len(patients) for patients in diagnosis_patients.values()
        ),
        "participants_by_diagnosis": json_text(
            {diagnosis: len(patients) for diagnosis, patients in diagnosis_patients.items()}
        ),
        "sessions": len(sessions),
        "sessions_by_diagnosis": json_text(dict(diagnosis_counts)),
        "conditions": json_text(sorted({
            str(Path(s["recording_id"]).parent.name) for s in sessions
        })),
        "empty_session_directories": json_text(empty_directories),
        "channel_sets": json_text(dict(channel_sets)),
        "age_available": False,
        "sex_available": False,
        "sampling_frequency_hz": "UNKNOWN",
        "amplitude_unit": "UNKNOWN",
        "reference": "UNKNOWN",
        "electrode_montage": "International 10-20 claimed by local source notes; local channel evidence should be confirmed in source documentation",
        "existing_filter": "UNKNOWN",
        "power_line_hz": "UNKNOWN",
        "existing_artifact_handling": "UNKNOWN",
        "recordings": sessions,
    }


def write_csv(path, rows, fieldnames=None):
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adftd", type=Path, default=PROJECT_DIR / "datasets" / "ADFTD")
    parser.add_argument("--adfsu", type=Path, default=PROJECT_DIR / "datasets" / "ADFSU")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "Processed" / "dataset_readiness")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    adftd = audit_adftd(args.adftd)
    adfsu = audit_adfsu(args.adfsu)
    write_csv(args.output / "adftd_recording_audit.csv", adftd.pop("recordings"))
    write_csv(args.output / "adfsu_recording_audit.csv", adfsu.pop("recordings"))
    comparison = []
    properties = [
        "unique_participants", "diagnosis_counts", "age_available", "sex_available",
        "original_sampling_frequency_hz", "amplitude_unit", "reference",
        "electrode_montage", "existing_filter", "existing_artifact_handling",
        "power_line_hz",
    ]
    for prop in properties:
        comparison.append({
            "property": prop,
            "ADFTD": adftd.get(prop, "UNKNOWN"),
            "ADFSU": adfsu.get(
                prop if prop not in ("original_sampling_frequency_hz",) else "sampling_frequency_hz",
                "UNKNOWN",
            ),
            "verified": "yes" if prop in adftd and prop in adfsu else "partial",
        })
    write_csv(args.output / "dataset_comparison.csv", comparison)

    adftd_channels = {"Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
                      "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"}
    adfsu_channels = {
        "C3", "C4", "Cz", "F1", "F2", "F3", "F4", "F7", "F8", "Fp1", "Fp2",
        "Fz", "O1", "O2", "P3", "P4", "Pz", "T3", "T4", "T5", "T6"
    }
    aliases = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
    normalized_a = {aliases.get(ch, ch) for ch in adftd_channels}
    normalized_b = {aliases.get(ch, ch) for ch in adfsu_channels}
    common = sorted(normalized_a & normalized_b)
    write_csv(args.output / "channel_compatibility.csv", [
        {
            "channel": channel,
            "in_adftd": channel in normalized_a,
            "in_adfsu": channel in normalized_b,
            "standardized_name": channel,
            "notes": "T3/T4/T5/T6 are legacy aliases for T7/T8/P7/P8" if channel in set(aliases.values()) else "",
        }
        for channel in sorted(normalized_a | normalized_b)
    ])

    metadata = {
        "adftd": adftd,
        "adfsu": adfsu,
        "generated_from": "local files only",
        "unknowns_must_not_be_inferred": [
            "ADFSU sampling frequency",
            "ADFSU amplitude unit",
            "ADFSU reference",
            "ADFSU filter settings",
            "ADFSU power-line frequency",
        ],
        "provisional_common_channels": common,
    }
    (args.output / "dataset_readiness.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "ADFTD": {key: adftd[key] for key in ("unique_participants", "raw_recordings", "derivative_recordings")},
        "ADFSU": {key: adfsu[key] for key in ("unique_dataset_prefixed_participants", "sessions")},
    }, indent=2))


if __name__ == "__main__":
    main()
