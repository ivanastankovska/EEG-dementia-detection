import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import project.scripts.dataset_review as dataset_review


class FakeAnnotations:
    def __init__(self, descriptions):
        self.description = descriptions

    def __len__(self):
        return len(self.description)


class FakeRaw:
    def __init__(self, sampling_rate, channel_names, duration, descriptions=()):
        self.info = {"sfreq": sampling_rate}
        self.ch_names = channel_names
        self.n_times = int(sampling_rate * duration)
        self.annotations = FakeAnnotations(descriptions)
        self.closed = False

    def close(self):
        self.closed = True


class DatasetReviewTests(unittest.TestCase):
    def test_metadata_reports_duplicates_missing_ids_and_labels(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            metadata_file = root / "participants.tsv"
            metadata_file.write_text(
                "participant_id\tDiagnosis\tAge\tGender\tMMSE\n"
                "sub-01\tAD\t67\tF\t19\n"
                "sub-01\t\t\t\t\n"
                "\tControl\t\t\t\n",
                encoding="utf-8",
            )

            report = dataset_review.inspect_metadata(
                metadata_file,
                root,
                "test",
                {"AD": "Alzheimer Disease Group", "Control": "Healthy Group"},
            )

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["row_count"], 3)
        self.assertEqual(report["unique_participants"], 1)
        self.assertEqual(report["duplicate_participant_ids"], 1)
        self.assertEqual(report["missing_participant_ids"], 1)
        self.assertEqual(report["missing_labels"], 1)
        self.assertEqual(
            json.loads(report["label_counts"]),
            {"Alzheimer Disease Group": 1, "Healthy Group": 1},
        )
        self.assertEqual(report["_participant_records"][0]["Age"], "67")

    def test_bids_recording_context_joins_demographics_and_derivative_reference(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw_eeg = root / "sub-001" / "eeg"
            derivative_eeg = root / "derivatives" / "sub-001" / "eeg"
            raw_eeg.mkdir(parents=True)
            derivative_eeg.mkdir(parents=True)
            (raw_eeg / "sub-001_task-eyesclosed_eeg.json").write_text(
                json.dumps(
                    {"EEGReference": "A1 A2", "EEGPlacementScheme": "10-20"}
                ),
                encoding="utf-8",
            )
            (root / "README").write_text(
                "signals were re-referenced to A1-A2.", encoding="utf-8"
            )
            participant = {
                "Group": "A",
                "diagnosis": "Alzheimer Disease Group",
                "Age": "67",
                "Gender": "F",
                "MMSE": "19",
            }

            raw_context, raw_warnings = dataset_review.load_recording_context(
                raw_eeg / "sub-001_task-eyesclosed_eeg.set",
                root,
                {"sub-001": participant},
                {"A": "Alzheimer Disease Group"},
            )
            derivative_context, derivative_warnings = (
                dataset_review.load_recording_context(
                    derivative_eeg / "sub-001_task-eyesclosed_eeg.set",
                    root,
                    {"sub-001": participant},
                    {"A": "Alzheimer Disease Group"},
                )
            )

        self.assertEqual(raw_context["subject_id"], "sub-001")
        self.assertEqual(raw_context["diagnosis"], "Alzheimer Disease Group")
        self.assertEqual(raw_context["diagnosis_source"], "participants.tsv")
        self.assertEqual(raw_context["age"], "67")
        self.assertEqual(raw_context["sex"], "F")
        self.assertEqual(raw_context["mmse"], "19")
        self.assertEqual(raw_context["condition"], "eyes-closed")
        self.assertEqual(raw_context["reference"], "A1 A2")
        self.assertEqual(raw_context["montage"], "10-20")
        self.assertEqual(derivative_context["reference"], "A1-A2")
        self.assertIn("README", derivative_context["reference_source"])
        self.assertEqual(raw_warnings, [])
        self.assertEqual(derivative_warnings, [])

    def test_scan_inspects_every_recording_and_flags_variation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for name in ("first.edf", "second.bdf", "broken.set"):
                (root / name).write_bytes(b"test-recording")
            (root / "participants.tsv").write_text(
                "participant_id\tgroup\nsub-01\tAD\n",
                encoding="utf-8",
            )
            raw_by_name = {
                "first.edf": FakeRaw(128, ["C3", "C4"], 0.5, ["event", "event"]),
                "second.bdf": FakeRaw(256, ["C3"], 3),
            }

            def fake_reader(path):
                if path.name == "broken.set":
                    raise OSError("invalid recording")
                return raw_by_name[path.name]

            with patch.object(
                dataset_review,
                "read_eeg_recording",
                side_effect=fake_reader,
            ) as reader:
                summary, recordings, metadata = dataset_review.scan_dataset(
                    "test", root, min_duration_seconds=1
                )

        self.assertEqual(reader.call_count, 3)
        self.assertEqual(len(recordings), 3)
        self.assertEqual(len(metadata), 1)
        self.assertEqual(summary["participants"], 1)
        self.assertEqual(summary["short_recordings"], 1)
        self.assertFalse(summary["channel_names_consistent"])
        self.assertEqual(
            json.loads(summary["sampling_rates_hz"]), [128.0, 256.0]
        )
        self.assertIn("sampling_rate_varies", json.loads(summary["warnings"]))
        first_recording = next(
            recording for recording in recordings if recording["file"] == "first.edf"
        )
        self.assertEqual(
            json.loads(first_recording["annotation_descriptions"]), {"event": 2}
        )
        broken_recording = next(
            recording for recording in recordings if recording["file"] == "broken.set"
        )
        self.assertEqual(broken_recording["status"], "error")
        self.assertIn("OSError: invalid recording", broken_recording["error"])
        self.assertTrue(all(raw.closed for raw in raw_by_name.values()))

    def test_reader_uses_format_specific_edf_reader(self):
        expected = object()
        edf_reader = unittest.mock.Mock(return_value=expected)
        mne_module = SimpleNamespace(
            io=SimpleNamespace(
                read_raw_edf=edf_reader,
                read_raw_bdf=unittest.mock.Mock(),
                read_raw_eeglab=unittest.mock.Mock(),
                read_raw_fif=unittest.mock.Mock(),
                read_raw_brainvision=unittest.mock.Mock(),
            )
        )

        with patch.dict("sys.modules", {"mne": mne_module}):
            result = dataset_review.read_eeg_recording(Path("recording.edf"))

        self.assertIs(result, expected)
        edf_reader.assert_called_once_with(
            Path("recording.edf"), preload=False, verbose="ERROR"
        )

    def test_empty_eeg_file_is_reported_before_mne_matlab_reader(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            recording = root / "empty.set"
            recording.touch()

            with patch.object(
                dataset_review,
                "read_eeg_recording",
                side_effect=AssertionError("MNE must not be called"),
            ) as reader:
                report = dataset_review.inspect_recording(
                    recording, root, "test", min_duration_seconds=1
                )

        reader.assert_not_called()
        self.assertEqual(report["status"], "error")
        self.assertIn("empty (0 bytes)", report["error"])
        self.assertEqual(
            json.loads(report["quality_flags"]),
            ["empty_recording_file"],
        )

    def test_missing_dataset_is_reported_and_all_csvs_are_written(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output = root / "reports"
            summaries = dataset_review.run_review(
                {"missing": root / "does-not-exist"}, output
            )

            self.assertEqual(summaries[0]["status"], "missing")
            for filename in (
                "dataset_inventory.csv",
                "recording_inventory.csv",
                "metadata_inventory.csv",
            ):
                self.assertTrue((output / filename).is_file())

            with (output / "dataset_inventory.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["status"], "missing")
            self.assertIn(
                "dataset_folder_not_found", json.loads(row["warnings"])
            )

    def test_text_channel_sessions_are_counted_as_eeg_not_missing_recordings(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            patient_dir = root / "AD" / "Eyes_closed" / "Paciente1"
            patient_dir.mkdir(parents=True)
            for index in range(21):
                samples = "1.25\n2.5\n"
                if index == 20:
                    samples = "1.25\nbad-value\n"
                (patient_dir / f"Ch{index}.txt").write_text(
                    samples, encoding="utf-8"
                )

            summary, recordings, _ = dataset_review.scan_dataset(
                "coordinates", root, min_duration_seconds=1
            )

        self.assertEqual(summary["eeg_files"], 0)
        self.assertEqual(summary["eeg_recordings"], 1)
        self.assertEqual(summary["text_signal_files"], 21)
        self.assertEqual(summary["signal_sessions"], 1)
        self.assertEqual(summary["patients"], 1)
        self.assertEqual(json.loads(summary["class_counts"]), {"AD": 1})
        self.assertEqual(
            json.loads(summary["recording_class_counts"]), {"AD": 1}
        )
        self.assertEqual(json.loads(summary["conditions"]), ["Eyes_closed"])
        self.assertEqual(summary["diagnosis_source"], "folder_names")
        self.assertEqual(json.loads(summary["text_signal_channel_counts"]), [21])
        self.assertEqual(json.loads(summary["text_signal_sample_counts"]), [1, 2])
        self.assertEqual(len(recordings), 1)
        self.assertEqual(recordings[0]["data_type"], "channel_text")
        self.assertEqual(recordings[0]["source_type"], "raw")
        self.assertEqual(recordings[0]["subject_id"], "AD/Paciente1")
        self.assertEqual(recordings[0]["diagnosis_source"], "diagnosis folder")
        self.assertEqual(recordings[0]["diagnosis"], "AD")
        self.assertEqual(recordings[0]["sample_count_min"], 1)
        self.assertEqual(recordings[0]["sample_count_max"], 2)
        self.assertFalse(recordings[0]["sample_counts_consistent"])
        self.assertIn("invalid_channel_values", json.loads(recordings[0]["quality_flags"]))
        self.assertNotIn(
            "no_supported_eeg_recordings_found", json.loads(summary["warnings"])
        )

    def test_empty_patient_condition_folder_is_reported_as_missing_session(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            present = root / "Healthy" / "Eyes_closed" / "Paciente1"
            empty = root / "Healthy" / "Eyes_open" / "Paciente1"
            present.mkdir(parents=True)
            empty.mkdir(parents=True)
            (present / "C3.txt").write_text("1.0\n2.0\n", encoding="utf-8")

            summary, recordings, _ = dataset_review.scan_dataset(
                "coordinates", root, min_duration_seconds=1
            )

        self.assertEqual(summary["signal_sessions"], 1)
        self.assertEqual(summary["session_directories"], 2)
        self.assertEqual(summary["empty_signal_sessions"], 1)
        self.assertEqual(summary["eeg_recordings"], 1)
        missing_recording = next(
            recording for recording in recordings if recording["file"].endswith("Eyes_open\\Paciente1")
        )
        self.assertEqual(missing_recording["status"], "error")
        self.assertIn("no_channel_text_files", json.loads(missing_recording["quality_flags"]))
        self.assertEqual(json.loads(missing_recording["missing_channels"]), [])
        self.assertIn(
            "empty_signal_session:Healthy\\Eyes_open\\Paciente1",
            json.loads(summary["warnings"]),
        )


if __name__ == "__main__":
    unittest.main()
