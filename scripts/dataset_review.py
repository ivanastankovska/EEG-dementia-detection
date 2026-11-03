"""Compatibility module for the dataset audit CLI and its tests."""

try:
    from . import inspect_eeg_datasets as _implementation
except ImportError:
    import inspect_eeg_datasets as _implementation


read_eeg_recording = _implementation.read_eeg_recording
inspect_recording = _implementation.inspect_recording


def inspect_metadata(*args, **kwargs):
    return _implementation.inspect_metadata(*args, **kwargs)


def load_recording_context(*args, **kwargs):
    return _implementation.load_recording_context(*args, **kwargs)


def run_review(*args, **kwargs):
    return _implementation.run_review(*args, **kwargs)


def scan_dataset(*args, **kwargs):
    original_reader = _implementation.read_eeg_recording
    _implementation.read_eeg_recording = read_eeg_recording
    try:
        return _implementation.scan_dataset(*args, **kwargs)
    finally:
        _implementation.read_eeg_recording = original_reader

__all__ = [
    "inspect_metadata",
    "inspect_recording",
    "load_recording_context",
    "read_eeg_recording",
    "run_review",
    "scan_dataset",
]
