import argparse
import re
from pathlib import Path

import mne


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET1 = PROJECT_DIR / "datasets" / "ADFTD"
DATASET2 = PROJECT_DIR / "datasets" / "ADFSU"


CHANNEL_ALIASES = {
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
}


def normalize_channel_name(channel_name: str) -> str:
    normalized = re.sub(r"[\s._-]+", "", channel_name).upper()
    return CHANNEL_ALIASES.get(normalized, normalized)


def participant_matches_filename(path: Path, participant_id: str) -> bool:
    participant_id = re.escape(participant_id.strip().casefold())
    filename = path.name.casefold()
    return re.search(rf"(^|[^0-9a-z]){participant_id}([^0-9a-z]|$)", filename) is not None


def show_dataset1_patient(patient_id: str) -> set[str]:
    print(f"\nDataset 1 - patient {patient_id}")
    recordings = sorted(
        path
        for path in DATASET1.rglob("*.set")
        if participant_matches_filename(path, patient_id)
        and "derivatives" not in path.relative_to(DATASET1).parts
    )

    if not recordings:
        print("  No recording found")
        return set()

    dataset_signals: set[str] = set()
    for recording in recordings:
        try:
            if recording.stat().st_size == 0:
                print(f"  ERROR reading {recording.relative_to(DATASET1)}: empty recording file (0 bytes)")
                continue
            raw = mne.io.read_raw_eeglab(recording, preload=False, verbose="ERROR")
            signal_names = sorted(
                {normalize_channel_name(name) for name in raw.ch_names},
                key=str.casefold,
            )
        except Exception as exc:
            print(f"  ERROR reading {recording.relative_to(DATASET1)}: {exc}")
            continue
        finally:
            if "raw" in locals():
                raw.close()
                del raw
        dataset_signals.update(signal_names)
        print(f"  Recording: {recording.relative_to(DATASET1)}")
        print(f"  Number of signals/channels: {len(signal_names)}")
        print(f"  Signal/channel names: {', '.join(signal_names)}")

    return dataset_signals


def show_dataset2_patient(patient_name: str, condition: str = "Eyes_closed") -> set[str]:
    print(f"\nDataset 2 - patient {patient_name}")
    patient_dirs = sorted(
        path
        for path in DATASET2.rglob("*")
        if path.is_dir() and path.name.casefold() == patient_name.casefold()
    )

    if not patient_dirs:
        print("  No patient folder found")
        return set()

    condition_key = re.sub(r"[\s_-]+", "", condition).casefold()
    dataset_signals: set[str] = set()
    for patient_dir in patient_dirs:
        folder_condition = re.sub(
            r"[\s_-]+", "", patient_dir.parent.name
        ).casefold()
        if condition and condition_key != folder_condition:
            continue
        signal_names = sorted(
            {normalize_channel_name(path.stem) for path in patient_dir.glob("*.txt")},
            key=str.casefold,
        )
        dataset_signals.update(signal_names)
        print(f"  Session: {patient_dir.relative_to(DATASET2)}")
        print(f"  Number of signals/channels: {len(signal_names)}")
        print(f"  Signal/channel names: {', '.join(signal_names)}")

    if not dataset_signals:
        print(f"  No matching channels found for condition '{condition}'")
    return dataset_signals


def print_signal_comparison(dataset1_signals: set[str], dataset2_signals: set[str]) -> None:
    common = sorted(dataset1_signals & dataset2_signals, key=str.casefold)
    only_dataset2 = sorted(dataset2_signals - dataset1_signals, key=str.casefold)
    only_dataset1 = sorted(dataset1_signals - dataset2_signals, key=str.casefold)

    print("\nSignal comparison")
    print(f"  Same signals ({len(common)}): {', '.join(common) if common else 'None'}")
    print(
        f"  In Dataset 2 but not Dataset 1 ({len(only_dataset2)}): "
        f"{', '.join(only_dataset2) if only_dataset2 else 'None'}"
    )
    print(
        f"  In Dataset 1 but not Dataset 2 ({len(only_dataset1)}): "
        f"{', '.join(only_dataset1) if only_dataset1 else 'None'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show the number and names of EEG signals for one patient in each dataset."
    )
    parser.add_argument("--dataset1-patient", default="sub-011")
    parser.add_argument("--dataset2-patient", default="Paciente1")
    parser.add_argument(
        "--dataset2-condition",
        default="Eyes_closed",
        help="Condition folder to compare, for example Eyes_closed or Eyes_open.",
    )
    args = parser.parse_args()

    dataset1_signals = show_dataset1_patient(args.dataset1_patient)
    dataset2_signals = show_dataset2_patient(
        args.dataset2_patient, args.dataset2_condition
    )
    print_signal_comparison(dataset1_signals, dataset2_signals)


if __name__ == "__main__":
    main()