# Dataset Readiness Audit

## Phase 0 pipeline

Run the consolidated local audit from the repository parent directory:

```powershell
python -m project.scripts.phase0_pipeline ^
  --adftd project/datasets/ADFTD ^
  --adfsu project/datasets/ADFSU ^
  --output project/Processed/phase0_audit
```

The pipeline creates subject- and recording-level manifests, QC and channel
reports, harmonisation guidance, a dataset-readiness JSON file, and a concise
audit report. It is read-only and does not filter, resample, epoch, or exclude
any source recording.

These reports describe the local copies of ADFTD and ADFSU without applying
harmonization or changing the source recordings. UNKNOWN fields require source
documentation and must not be inferred from sample counts.

## Project review

- `ADFTD_review_preprocess.py` audits and resamples the raw ADFTD derivatives
  or raw EEGLAB recordings. It is intentionally not a final preprocessing
  pipeline: filtering and artifact rejection are not applied there.
- `visualize_eeg.py` creates exploratory plots for ADFTD and ADFSU. Its ADFSU
  sampling frequency and unit must be supplied by the user because neither can
  safely be inferred locally.
- `scripts/inspect_eeg_datasets.py` is the common dataset inventory loader.
- Existing resampled `.npy` files and figures are review products; they are not
  evidence that model-ready preprocessing has been completed.

## ADFTD audit

- 88 unique participants, one raw recording per participant.
- 88 raw and 88 derivative `.set` paths; all 88 derivative `.set` files have
  zero bytes in the local copy, so only the 88 raw recordings are usable from
  these files without restoring the derivative payloads.
- Diagnosis: 36 Alzheimer disease, 23 frontotemporal dementia, 29 healthy.
- Age, sex, and MMSE are available in `participants.tsv`.
- Every recording is eyes closed.
- Raw sampling frequency: 500 Hz.
- 19 EEG channels following the documented international 10-20 layout.
- Reference information is available in recording sidecars and the README:
  raw sidecar records A1 A2, while derivatives are documented as re-referenced
  to A1-A2.
- The README documents acquisition high-frequency filtering at 70 Hz and a
  0.5-45 Hz derivative band-pass. The derivative files also use ASR and ICLabel
  component rejection.
- The existing resampling script processed all 88 recordings to 200 Hz and
  preserved a `(samples, channels)` NumPy layout.

## ADFSU audit

- 183 populated diagnosis/condition/patient sessions and 92 local diagnosis-
  qualified participant folders: 80 AD and 12 Healthy.
- 160 AD-condition sessions and 23 Healthy-condition sessions are populated;
  `Healthy/Eyes_open/Paciente5` is an empty directory.
- Patient names are reused across diagnosis folders and must be treated as
  dataset-prefixed IDs such as `AD/Paciente1`, not globally unique IDs.
- Both eye-open and eye-closed folders exist in most cases.
- Channel text files are complete and finite in 183 sessions, with equal
  channel lengths of 1024 samples.
- `Healthy/Eyes_closed/Paciente6` has 19 channels and lacks F1 and F2.
- Sampling frequency: UNKNOWN.
- Amplitude unit: UNKNOWN.
- Reference: UNKNOWN.
- Digital filtering and artifact handling: UNKNOWN.
- Demographics: not available locally.

## Harmonization readiness

The datasets share eyes-closed recordings and the 19 standard 10-20 scalp
locations after legacy T3/T4/T5/T6 names are mapped to T7/T8/P7/P8. F1/F2 are
additional in most ADFSU sessions but not ADFTD.

Do not perform final harmonization yet. Before it, obtain and verify:

1. ADFSU sampling frequency, amplitude unit, reference, filtering, montage,
   line frequency, and participant/session provenance.
2. The intended meaning of repeated `PacienteN` identifiers.
3. Resolution of the missing F1/F2 pair and the incomplete healthy session.
4. Whether ADFTD raw, derivative, or a documented processing-preserving source
   is appropriate for each future comparison.

## EDA readiness

- Possible now for ADFTD: age, sex, diagnosis counts, MMSE, duration, and PSD.
- Possible now for ADFSU: folder-derived class counts, condition counts,
  channel layout, and raw numeric trace quality.
- Not possible from local ADFSU data alone: age, sex, duration in seconds,
  validated spectral units, or verified sampling-frequency distributions.

## Cohort-shift readiness

Participant-level band-power and covariance analyses are not yet scientifically
safe. Required information includes verified ADFSU acquisition metadata, known
units, consistent reference and filtering, and resolution of participant versus
condition counting. Demographic confounding cannot be assessed for ADFSU until
demographics are available.

## Data-quality findings

- ADFTD: all 88 raw and 88 derivative recordings loaded in the existing review
  workflow; duplicate raw/derivative representations are not extra patients.
- ADFSU: 183 populated sessions are finite and internally equal in length.
- ADFSU: one session has a missing channel pair, one directory is empty, and
  patient numbers collide across diagnosis folders.
- Source frequency and unit metadata remain unknown for ADFSU.

## Recommended next steps

1. Obtain and document the ADFSU acquisition metadata from its source archive,
   wiki, paper, or project owner.
2. Reconcile the local ADFSU participant/session identifiers.
3. Use the generated CSV and JSON reports as the source of dataset-specific
   audit evidence.
4. Re-run the audit after adding metadata or correcting source files.
5. Only then design harmonization for sampling rate, common channels, filter
   range, epochs, referencing, and the third dataset.
