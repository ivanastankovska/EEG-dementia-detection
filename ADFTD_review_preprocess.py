"""ADFTD dataset audit and conservative resampling (not model-ready epoching).

Run: python ADFTD_review_preprocess.py --dataset /path/to/ADFTD --output /path/to/output
Optional: --source derivatives | raw | auto (auto prefers derivatives)

Requires: mne, numpy, pandas. Does not filter or reject artifacts automatically.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import mne

LABELS = {'C': 0, 'A': 1, 'F': 2}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--source', choices=['auto','raw','derivatives'], default='auto')
    parser.add_argument('--target-fs', type=float, default=200.0)
    args = parser.parse_args()
    dataset = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(dataset / 'participants.tsv', sep='\t', dtype={'participant_id': str})
    required = {'participant_id','Group'}
    if not required.issubset(meta.columns):
        raise ValueError(f'Missing participant columns: {sorted(required - set(meta.columns))}')
    if meta.participant_id.isna().any() or meta.participant_id.duplicated().any():
        raise ValueError('Missing or duplicate participant IDs in participants.tsv')
    if not meta.Group.isin(LABELS).all():
        raise ValueError(f'Unknown diagnosis groups: {meta.loc[~meta.Group.isin(LABELS),"Group"].tolist()}')
    meta['label'] = meta.Group.map(LABELS).astype(int)
    meta.to_csv(output/'participant_labels.csv', index=False)

    derivatives = dataset/'derivatives'
    if args.source == 'derivatives' and not derivatives.exists():
        raise FileNotFoundError(derivatives)
    root = derivatives if args.source == 'derivatives' or (args.source == 'auto' and derivatives.exists()) else dataset
    files = sorted(root.glob('sub-*/eeg/*.set'))
    if not files:
        raise FileNotFoundError(f'No EEGLAB .set files found in {root}/sub-*/eeg/')
    lookup = meta.set_index('participant_id')
    rows = []
    for f in files:
        pid = f.parent.parent.name
        record = {'participant_id':pid,'status':'ok'}
        if f.stat().st_size == 0:
            record.update(
                status='error',
                error='ValueError: Recording file is empty (0 bytes)'
            )
            rows.append(record)
            print(f"\nSkipping empty recording: {f}")
            continue
        if pid not in lookup.index:
            record['status'] = 'missing_metadata'
            rows.append(record)
            continue
        record['Group'] = lookup.loc[pid,'Group']
        record['label'] = int(lookup.loc[pid,'label'])
        try:
            raw = mne.io.read_raw_eeglab(f, preload=False, verbose='ERROR')
            eeg_channels = mne.pick_types(raw.info, eeg=True, exclude=[])
            record.update(original_sfreq_hz=float(raw.info['sfreq']),
                          n_channels_total=len(raw.ch_names),
                          n_eeg_channels=len(eeg_channels),
                          channel_names=';'.join(raw.ch_names),
                          bads_marked=';'.join(raw.info['bads']),
                          duration_sec=float(raw.n_times/raw.info['sfreq']))
            if len(eeg_channels) == 0:
                raise ValueError('No EEG channels found')
            if raw.info['sfreq'] < args.target_fs:
                raise ValueError('Original sampling rate below requested target')
            raw.load_data(verbose='ERROR')
            raw.pick(picks=eeg_channels)
            # This does not add a band-pass, notch filter, or artifact rejection.
            raw.resample(args.target_fs, npad='auto', verbose='ERROR')
            data = raw.get_data().T.astype(np.float32)
            if not np.isfinite(data).all():
                raise ValueError('Non-finite samples in resampled EEG')
            dest = output/'recordings'/pid
            dest.mkdir(parents=True, exist_ok=True)
            out = dest/(f.stem + '_200hz.npy')
            np.save(out, data)
            record.update(target_sfreq_hz=args.target_fs,
                          output_samples=int(data.shape[0]), output_channels=int(data.shape[1]),
                          output_unit='volts (MNE EEG convention)')
        except Exception as exc:
            import traceback
            print(f"\nERROR reading: {f}")
            traceback.print_exc()
            record.update(
                status='error',
                error=f'{type(exc).__name__}: {exc}'
            )
        rows.append(record)
    report = pd.DataFrame(rows)
    report.to_csv(output/'recording_audit.csv', index=False)
    summary = {'source_root':str(root), 'total_participants_in_metadata':len(meta),
               'diagnosis_counts':meta.Group.value_counts().to_dict(),
               'recordings_found':len(files), 'recording_status_counts':report.status.value_counts().to_dict(),
               'unique_subjects_with_successful_recordings':int(report.loc[report.status=='ok','participant_id'].nunique()),
               'target_sfreq_hz':args.target_fs,
               'note':'Resampling only; filtering, artifact rejection, epoching, and split creation not performed.'}
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print('Review recording_audit.csv for errors, channel inconsistencies, and recording duration.')

if __name__ == '__main__':
    main()
