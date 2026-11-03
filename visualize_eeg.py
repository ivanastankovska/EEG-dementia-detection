"""Exploratory EEG figures for ADFTD (BIDS/EEGLAB) and ADFSU (channel TXT).

Run from a project containing datasets/ADFTD and datasets/ADFSU.
ADFSU sampling frequency MUST be verified from its documentation and passed via
--adfsu-sfreq; it is never inferred from sample counts.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.signal import welch


PROJECT_DIR = Path(__file__).resolve().parent


def normalize_channel(name):
    return re.sub(r'[^a-z0-9]', '', name.casefold())


def canonical_channel(name):
    aliases = {
        't3': 'T7',
        't4': 'T8',
        't5': 'P7',
        't6': 'P8',
    }
    normalized = normalize_channel(name)
    return aliases.get(normalized, normalized.upper())


def resolve_project_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_DIR / path


def load_adftd_metadata(root):
    file = root / 'participants.tsv'
    if not file.exists():
        print('WARNING: ADFTD participants.tsv not found; class distribution unavailable.')
        return pd.DataFrame()
    df = pd.read_csv(file, sep='\t')
    col = next((c for c in df if c.casefold() in ('group', 'diagnosis')), None)
    idcol = next((c for c in df if c.casefold() == 'participant_id'), None)
    if col is None or idcol is None:
        print('WARNING: ADFTD participant ID or diagnosis column missing.')
        return pd.DataFrame()
    levels = {}
    desc = root / 'participants.json'
    if desc.exists():
        try:
            meta = json.loads(desc.read_text(encoding='utf-8-sig'))
            entry = next((v for k, v in meta.items() if k.casefold() == col.casefold()), {})
            levels = entry.get('Levels', {})
        except (ValueError, OSError) as exc:
            print(f'WARNING: Cannot parse participants.json: {exc}')
    out = df[[idcol, col]].dropna().drop_duplicates(subset=[idcol]).copy()
    out.columns = ['subject', 'diagnosis']
    out['diagnosis'] = out['diagnosis'].astype(str).map(lambda v: levels.get(v, v))
    return out


def find_adftd_recordings(root):
    return sorted(p for p in root.rglob('*.set') if 'derivatives' not in p.relative_to(root).parts)


def find_adfsu_sessions(root):
    sessions = []
    for d in sorted(root.rglob('*')):
        if d.is_dir() and list(d.glob('*.txt')):
            parts = d.relative_to(root).parts
            if len(parts) >= 3:
                diagnosis, condition, patient = parts[-3:]
                sessions.append((d, diagnosis, condition, patient))
    return sessions


def load_adfsu_session(folder, sfreq):
    names, signals = [], []
    for file in sorted(folder.glob('*.txt')):
        try:
            values = np.loadtxt(file, dtype=float).reshape(-1)
        except (ValueError, OSError) as exc:
            print(f'WARNING: Could not read {file}: {exc}')
            continue
        if not len(values) or not np.all(np.isfinite(values)):
            print(f'WARNING: Skipping empty/nonfinite channel {file}')
            continue
        names.append(file.stem)
        signals.append(values)
    if not signals:
        raise ValueError(f'No valid channel files in {folder}')
    sizes = {len(x) for x in signals}
    if len(sizes) != 1:
        raise ValueError(f'Inconsistent channel lengths in {folder}: {sorted(sizes)}')
    return names, np.stack(signals), sfreq


def draw_waveform(data, names, sfreq, title, output, seconds, unit, multiplier=1.0):
    n = min(int(seconds * sfreq), data.shape[1])
    if n < 2:
        print(f'WARNING: Too few samples for {title}')
        return
    x = data[:min(12, len(names)), :n] * multiplier
    labels = names[:len(x)]
    scales = np.nanpercentile(np.abs(x - np.nanmedian(x, axis=1, keepdims=True)), 95, axis=1)
    spacing = max(float(np.nanmedian(scales) * 3), 1e-10)
    fig, ax = plt.subplots(figsize=(13, 7))
    t = np.arange(n) / sfreq
    for idx, signal in enumerate(x):
        ax.plot(t, signal - np.nanmedian(signal) + idx * spacing, lw=0.6)
    ax.set_yticks(np.arange(len(labels)) * spacing, labels)
    ax.invert_yaxis()
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel(f'Channels (offset vertically; input units: {unit})')
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def draw_psd(data, names, sfreq, title, output):
    if data.shape[1] < 8:
        return
    freq, power = welch(data, fs=sfreq, nperseg=min(int(sfreq * 4), data.shape[1]), axis=1)
    mask = (freq >= 0.5) & (freq <= min(45, sfreq / 2))
    fig, ax = plt.subplots(figsize=(10, 5))
    for ch, p in zip(names[:12], power[:12]):
        ax.plot(freq[mask], 10 * np.log10(np.maximum(p[mask], np.finfo(float).tiny)), lw=0.9, label=ch)
    ax.set(xlabel='Frequency (Hz)', ylabel='PSD (dB, relative to input unit²/Hz)', title=title)
    ax.legend(ncol=4, fontsize=7)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def draw_class_distribution(adftd_meta, adfsu_sessions, output):
    counts = Counter()
    if not adftd_meta.empty:
        for diagnosis, n in adftd_meta.groupby('diagnosis')['subject'].nunique().items():
            counts[('ADFTD', str(diagnosis))] = int(n)
    subjects = {(diag, patient) for _, diag, _, patient in adfsu_sessions}
    for diag, _ in subjects:
        counts[('ADFSU', diag)] += 1
    if not counts:
        return
    labels = [f'{dataset}\n{diagnosis}' for dataset, diagnosis in counts]
    values = list(counts.values())
    fig, ax = plt.subplots(figsize=(max(8, len(labels)*1.6), 5))
    bars = ax.bar(labels, values)
    ax.bar_label(bars)
    ax.set_ylabel('Unique subjects (as identified by dataset metadata/folders)')
    ax.set_title('Diagnosis distribution by cohort')
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def draw_channel_comparison(a_names, b_names, output):
    a = {normalize_channel(n): n for n in a_names}
    b = {normalize_channel(n): n for n in b_names}
    shared = sorted(a.keys() & b.keys())
    only_a = sorted(a.keys() - b.keys())
    only_b = sorted(b.keys() - a.keys())
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(['ADFTD only', 'Shared', 'ADFSU only'], [len(only_a), len(shared), len(only_b)])
    ax.bar_label(bars)
    ax.set_ylabel('Number of channel names')
    ax.set_title('Channel-name overlap (sample recordings only)')
    fig.text(0.03, 0.02, 'Shared: ' + (', '.join(a[k] for k in shared) or 'none'), fontsize=8, wrap=True)
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    fig.savefig(output, dpi=170)
    plt.close(fig)
    print('Shared channels in selected recordings:', [a[k] for k in shared])


def draw_frequency_bands(data, names, sfreq, title, output):
    if data.shape[1] < 8:
        return
    frequencies, power = welch(
        data, fs=sfreq, nperseg=min(int(sfreq * 4), data.shape[1]), axis=1
    )
    bands = {
        'Delta\n(0.5-4 Hz)': (0.5, 4),
        'Theta\n(4-8 Hz)': (4, 8),
        'Alpha\n(8-13 Hz)': (8, 13),
        'Beta\n(13-30 Hz)': (13, 30),
        'Gamma\n(30-45 Hz)': (30, 45),
    }
    values = []
    for low, high in bands.values():
        mask = (frequencies >= low) & (frequencies < high)
        values.append(np.mean(power[:, mask], axis=1) if np.any(mask) else np.full(data.shape[0], np.nan))

    fig, ax = plt.subplots(figsize=(11, 6))
    image = np.vstack([
        10 * np.log10(np.maximum(row[:12], np.finfo(float).tiny)) for row in values
    ])
    image -= np.nanmean(image, axis=1, keepdims=True)
    im = ax.imshow(image, aspect='auto', cmap='magma', interpolation='nearest')
    ax.set_xticks(np.arange(min(12, len(names))))
    ax.set_xticklabels(names[:min(12, len(names))], rotation=45, ha='right', fontsize=8)
    ax.set_yticks(np.arange(len(bands)))
    ax.set_yticklabels(list(bands.keys()))
    ax.set_xlabel('Selected channels')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label='Mean band power relative to band/channel mean (dB)')
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def draw_scalp_map(data, names, sfreq, title, output):
    if data.shape[0] < 4 or sfreq <= 0:
        return
    clean_names = [canonical_channel(name) for name in names]
    try:
        frequencies, power = welch(
            data, fs=sfreq, nperseg=min(int(sfreq * 4), data.shape[1]), axis=1
        )
        mask = (frequencies >= 8) & (frequencies <= 13)
        if not np.any(mask):
            return
        alpha_power = np.mean(power[:, mask], axis=1)
        montage = mne.channels.make_standard_montage('colin27_1020')
        info = mne.create_info(clean_names[:len(alpha_power)], sfreq, ch_types='eeg')
        info.set_montage(montage, on_missing='ignore')
        pos = np.array([info['chs'][idx]['loc'][:3] for idx in range(len(info['chs']))])
        valid = np.isfinite(pos).all(axis=1) & (np.linalg.norm(pos, axis=1) > 0)
        if np.count_nonzero(valid) < 4:
            return
        fig, ax = plt.subplots(figsize=(7, 6))
        mne.viz.plot_topomap(
            np.log10(np.maximum(alpha_power[valid], np.finfo(float).tiny)),
            pos[valid, :2],
            axes=ax,
            cmap='viridis',
            contours=0,
            show=False,
        )
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(output, dpi=170)
        plt.close(fig)
    except (RuntimeError, TypeError, ValueError) as exc:
        print(f'WARNING: Could not create scalp map for {title}: {exc}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adftd', type=Path, default=PROJECT_DIR / 'datasets' / 'ADFTD')
    parser.add_argument('--adfsu', type=Path, default=PROJECT_DIR / 'datasets' / 'ADFSU')
    parser.add_argument('--output', type=Path, default=PROJECT_DIR / 'results' / 'figures')
    parser.add_argument('--adfsu-sfreq', type=float, default=None,
                        help='Verified ADFSU sampling frequency (Hz); required for its waveform/PSD plots')
    parser.add_argument('--adfsu-unit', default='unknown', help='Verified amplitude unit, e.g. uV (do not guess)')
    parser.add_argument('--seconds', type=float, default=10)
    args = parser.parse_args()
    if args.adfsu_sfreq is not None and args.adfsu_sfreq <= 0:
        parser.error('--adfsu-sfreq must be positive')
    if args.seconds <= 0:
        parser.error('--seconds must be positive')
    args.adftd = resolve_project_path(args.adftd)
    args.adfsu = resolve_project_path(args.adfsu)
    args.output = resolve_project_path(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    adftd_files = find_adftd_recordings(args.adftd) if args.adftd.exists() else []
    adfsu_sessions = find_adfsu_sessions(args.adfsu) if args.adfsu.exists() else []
    meta = load_adftd_metadata(args.adftd) if args.adftd.exists() else pd.DataFrame()
    draw_class_distribution(meta, adfsu_sessions, args.output / 'class_distribution.png')
    a_names = b_names = []
    if adftd_files:
        try:
            with mne.io.read_raw_eeglab(adftd_files[0], preload=False, verbose='ERROR') as raw:
                a_names = raw.ch_names.copy()
                sfreq = float(raw.info['sfreq'])
                data = raw.get_data(picks='eeg', start=0, stop=min(raw.n_times, int(max(args.seconds, 30)*sfreq)))
                names = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, exclude=[])]
                draw_waveform(data, names, sfreq, f'ADFTD raw EEG: {adftd_files[0].name}',
                              args.output / 'adftd_raw_eeg.png', args.seconds, 'µV', 1e6)
                draw_psd(data * 1e6, names, sfreq, 'ADFTD PSD (µV²/Hz)', args.output / 'adftd_psd.png')
                draw_frequency_bands(
                    data * 1e6, names, sfreq, 'ADFTD frequency bands',
                    args.output / 'adftd_frequency_bands.png'
                )
                draw_scalp_map(
                    data * 1e6, names, sfreq, 'ADFTD alpha-band scalp map',
                    args.output / 'adftd_alpha_topomap.png'
                )
        except Exception as exc:
            print(f'WARNING: ADFTD visualization failed: {exc}')
    else:
        print('WARNING: No raw ADFTD .set recordings found.')
    if adfsu_sessions:
        folder, diagnosis, condition, patient = adfsu_sessions[0]
        b_names = [p.stem for p in sorted(folder.glob('*.txt'))]
        if args.adfsu_sfreq is None:
            print('NOTE: ADFSU waveform and PSD skipped; supply verified --adfsu-sfreq.')
        else:
            try:
                names, data, sfreq = load_adfsu_session(folder, args.adfsu_sfreq)
                draw_waveform(data, names, sfreq, f'ADFSU raw EEG: {diagnosis}/{condition}/{patient}',
                              args.output / 'adfsu_raw_eeg.png', args.seconds, args.adfsu_unit)
                draw_psd(data, names, sfreq, 'ADFSU PSD (input units²/Hz)', args.output / 'adfsu_psd.png')
                draw_frequency_bands(
                    data, names, sfreq, 'ADFSU frequency bands',
                    args.output / 'adfsu_frequency_bands.png'
                )
                draw_scalp_map(
                    data, names, sfreq, 'ADFSU alpha-band scalp map',
                    args.output / 'adfsu_alpha_topomap.png'
                )
            except Exception as exc:
                print(f'WARNING: ADFSU visualization failed: {exc}')
    else:
        print('WARNING: No ADFSU patient folders with .txt files found.')
    if a_names and b_names:
        draw_channel_comparison(a_names, b_names, args.output / 'channel_comparison.png')
    manifest = {
        'adftd_root': str(args.adftd),
        'adfsu_root': str(args.adfsu),
        'seconds_requested': args.seconds,
        'adfsu_sfreq_hz': args.adfsu_sfreq,
        'adfsu_unit': args.adfsu_unit,
        'figures': sorted(
            path.name for path in args.output.glob('*.png') if path.is_file()
        ),
    }
    (args.output / 'visualization_manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    print(f'Figures written to: {args.output.resolve()}')


if __name__ == '__main__':
    main()
