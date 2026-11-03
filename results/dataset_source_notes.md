# Stage 1 dataset notes

These notes describe the local dataset copies inventoried by `dataset_review.py`.
The per-recording data is in `recording_inventory.csv`; dataset totals are in
`dataset_inventory.csv`.

## OpenNeuro ds004504 (`datasets/dataset1`)

- Layout is BIDS. There are 88 raw `.set` recordings under `sub-*/eeg/` and
  88 preprocessed recordings under `derivatives/sub-*/eeg/`. Raw and derivative
  rows represent the same 88 subjects, so they are not 176 independent people.
- Diagnosis comes from `participants.tsv` `Group`; `participants.json` maps `A`
  to Alzheimer Disease Group, `F` to Frontotemporal Dementia Group, and `C` to
  Healthy Group. The file also provides `Age`, `Gender`, and `MMSE`.
- The local subject counts are 36 Alzheimer, 23 frontotemporal dementia, and
  29 healthy. All recordings are eyes-closed, 500 Hz, 19 channels, and use the
  10-20 placement scheme. Raw reference is read from the EEG sidecar; derivative
  reference (`A1-A2`) is taken from the dataset README's preprocessing
  description because derivative recordings do not have their own sidecars.
- MNE successfully read all 176 recording files. Durations vary; see the CSV
  for each recording's measured value.

## OSF 2v5md (`datasets/dataset2`)

- The local copy is organized as
  `diagnosis/condition/patient/channel.txt`, for example
  `AD/Eyes_closed/Paciente1/C3.txt`. Diagnosis labels are the top-level folder
  names (`AD`, `Healthy`); conditions are `Eyes_closed` and `Eyes_open`.
- Each channel text file contains one numeric EEG signal value per line. The
  files are channel time series, not XYZ electrode-coordinate tables. The local
  copy contains 3,841 such files and 1,024 samples per channel file.
- The local inventory finds 80 AD and 12 Healthy patient IDs, and 183 populated
  patient-condition sessions. There are 184 patient-condition directories;
  `Healthy/Eyes_open/Paciente5` is empty and is reported as a missing session.
  The local copy contains no participant spreadsheet, demographics, sampling
  rate, or reference metadata, so those fields remain unknown rather than being
  inferred.
- The most common local channel set has 21 names. One session,
  `Healthy/Eyes_closed/Paciente6`, has 19 and lacks `F1` and `F2` relative to
  that local majority.

## Source discrepancy to resolve before Stage 2

The [OSF project](https://osf.io/2v5md/) currently exposes one archive,
`EEG_data.zip`, and a project wiki. The wiki describes 160 probable AD patients,
24 healthy elderly participants, and 19 scalp loci in the international 10-20
system. The local folder names instead identify 80 AD IDs and 12 Healthy IDs,
each with two eye-condition folders (160 and 24 expected condition sessions).
That numerical match suggests the wiki totals may count condition recordings,
but its wording says patients; this inventory does **not** assume those are
independent subjects. The local 21-channel majority also differs from the
wiki's 19-locus description. Confirm the OSF dataset's participant/session
definition and channel convention with its source documentation before
combining datasets or designing subject-wise splits.
