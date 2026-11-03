import json
import tempfile
import unittest
from pathlib import Path

from project.scripts.phase0_pipeline import (
    build_channel_compatibility,
    build_qc_summary,
    build_subject_manifest,
)


class Phase0PipelineTests(unittest.TestCase):
    def setUp(self):
        self.recordings = [{
            "dataset": "ADFSU",
            "subject_id": "AD/Paciente1",
            "diagnosis": "AD",
            "age": "",
            "sex": "",
            "mmse": "",
            "condition": "Eyes_open",
            "file": "AD/Eyes_open/Paciente1",
            "status": "ok",
            "quality_flags": json.dumps(["short_recording"]),
            "channel_names": json.dumps(["T3", "C3"]),
        }]

    def test_subject_manifest_is_subject_level(self):
        subjects = build_subject_manifest(self.recordings)
        self.assertEqual(len(subjects), 1)
        self.assertEqual(subjects[0]["recording_count"], 1)
        self.assertEqual(subjects[0]["recording_files"], "AD/Eyes_open/Paciente1")

    def test_qc_summary_requires_review_but_no_automatic_exclusion(self):
        qc = build_qc_summary(self.recordings)
        self.assertTrue(qc[0]["review_required"])
        self.assertFalse(qc[0]["exclusion_recommended"])

    def test_channel_alias_is_normalized(self):
        channels = build_channel_compatibility(self.recordings)
        self.assertIn("T7", [row["standardized_channel"] for row in channels])


if __name__ == "__main__":
    unittest.main()
