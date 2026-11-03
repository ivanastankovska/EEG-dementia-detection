# EEG Dementia Detection - Phase 0 Dataset Audit

This repository contains a read-only audit and exploratory-analysis pipeline for
the ADFTD and ADFSU EEG datasets. Its purpose is to inspect dataset integrity,
identify acquisition metadata, quantify channel and recording differences, and
assess whether the two cohorts are ready for harmonisation.

The code does not train dementia classification models. It also does not
perform final filtering, artifact removal, epoching, resampling, subject
splitting, or harmonisation of the source recordings.

## Current datasets

- `datasets/ADFTD/`: ADFTD BIDS-style recordings and `participants.tsv`.
- `datasets/ADFSU/`: ADFSU channel text files arranged by diagnosis, condition,
  and patient.
- `datasets/` is ignored by Git because the EEG data is large and must not be
  uploaded to the repository.

Raw recordings must remain separate from generated reports and processed data.

## Code overview

| File | Purpose |
| --- | --- |
| `scripts/inspect_eeg_datasets.py` | Main dataset inventory and shared loading functions. Supports MNE-compatible EEG files and ADFSU text-channel sessions. |
| `scripts/dataset_review.py` | Compatibility entry point that can be run as either a script or imported package module. |
| `scripts/phase0_pipeline.py` | Consolidated Phase 0 pipeline producing subject, recording, QC, channel, readiness, and report outputs. |
| `scripts/generate_dataset_readiness.py` | Generates a focused ADFTD/ADFSU readiness comparison. |
| `scripts/compare_eeg_channels.py` | Compares channels for selected patients and normalises legacy temporal names. |
| `ADFTD_review_preprocess.py` | Conservative ADFTD recording audit and optional 200 Hz review resampling. It does not add filtering or artifact rejection. |
| `visualize_eeg.py` | Generates waveform, PSD, frequency-band, class-distribution, and channel figures. |
| `tests/` | Unit tests covering metadata parsing, recording inspection, text-channel sessions, empty files, and report generation. |

## Installation

Create and activate a virtual environment, then install the required packages.
For example:

```powershell
cd "C:\Users\Ivana\Desktop\EEG dementia detection"
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install mne numpy pandas scipy matplotlib
```

The audit scripts use Python and MNE-compatible recordings. ADFSU text-channel
files require NumPy; plots require Matplotlib and SciPy.

## Run the main audit

### Direct script execution

```powershell
python scripts\inspect_eeg_datasets.py
```

Custom paths and output location:

```powershell
python scripts\inspect_eeg_datasets.py `
  --dataset "ADFTD=C:\path\to\ADFTD" `
  --dataset "ADFSU=C:\path\to\ADFSU" `
  --output-dir results `
  --min-duration-seconds 60
```

### Consolidated Phase 0 pipeline

```powershell
python scripts\phase0_pipeline.py `
  --adftd datasets\ADFTD `
  --adfsu datasets\ADFSU `
  --output Processed\phase0_audit
```

The Phase 0 pipeline imports or executes the shared inventory functions and
writes the required audit products to the selected output directory.

## Generated reports

The inventory pipeline writes to `results/`:

- `dataset_inventory.csv`: one row per dataset.
- `recording_inventory.csv`: one row per recording or channel-text session.
- `metadata_inventory.csv`: participant metadata integrity.

The Phase 0 pipeline additionally writes:

- `subject_manifest.csv`: unique subject-level audit records.
- `recording_manifest.csv`: recording-level metadata and integrity checks.
- `channel_compatibility.csv`: normalised channel availability.
- `qc_summary.csv`: non-destructive quality-control review flags.
- `dataset_readiness.json`: provenance, environment versions, known values, and
  unresolved acquisition metadata.
- `harmonisation_plan.md`: provisional harmonisation evidence and alternatives.
- `dataset_audit_report.md`: concise dataset-readiness summary.

`Processed/` contains additional historical audit and preprocessing-review
outputs. These files should be treated as evidence generated from the local
data, not as final model-ready results.

## Quality-control behaviour

The audit reports corrupted, unreadable, or empty files without stopping the
entire run. In particular:

- Zero-byte `.set` files are flagged as `empty_recording_file`.
- MATLAB `MatReadError` is avoided by checking for empty files before invoking
  the MNE EEGLAB reader.
- Missing ADFSU channels, empty session folders, inconsistent channel lengths,
  invalid values, and missing acquisition metadata are recorded rather than
  silently corrected.
- QC flags are review indicators only; no recording is automatically excluded.

## Channel normalisation

The shared pipeline normalises text and punctuation in channel names and maps
legacy temporal names:

- T3 -> T7
- T4 -> T8
- T5 -> P7
- T6 -> P8

The mapping is recorded as an alias assumption and does not replace source
channel names in the underlying data.

## Known unresolved information

The following properties are unknown for the local ADFSU copy and must be
obtained from source documentation rather than inferred:

- Sampling frequency and signal duration in seconds.
- Amplitude units.
- EEG reference.
- Digital filtering and artifact handling.
- Line frequency.
- Participant/session provenance and demographics.
- Verified montage and recording conditions.

The local ADFTD derivative `.set` files are currently zero-byte placeholders.
Therefore, usable signal analysis in the local copy is based on the available
raw ADFTD recordings unless the derivative files are restored.

## Run tests

Run the unit tests from the repository root:

```powershell
python -m unittest discover -s tests -v
```

## Dataset-ready use and limitations

Before model experiments:

1. Restore or verify all EEG payloads.
2. Reconcile ADFSU participant and session identities.
3. Verify ADFSU acquisition metadata.
4. Validate demographics and diagnosis mappings.
5. Review the recording-level QC output.
6. Define harmonisation only after these properties are documented.
7. Keep development, calibration, and external test data separated at subject
   level in later experiments.
