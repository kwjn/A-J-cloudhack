"""Focused tests for anonymous face-reference decisions."""

import unittest

import numpy as np

from face_identity import FaceEvidence, SessionFaceVerifier


class SessionFaceVerifierTests(unittest.TestCase):
    def setUp(self):
        # These decision tests do not need to load the ONNX models.
        self.verifier = SessionFaceVerifier.__new__(SessionFaceVerifier)
        self.verifier.match_threshold = 0.45
        self.verifier.mismatch_threshold = 0.30
        self.verifier.consistency_threshold = 0.45
        self.verifier.reference = np.array([1.0, 0.0], dtype=np.float32)
        self.verifier.acquisition_samples = []

    def test_match_mismatch_and_ambiguous_bands(self):
        self.assertEqual(
            self.verifier.classify([1.0, 0.0]), FaceEvidence.MATCH
        )
        self.assertEqual(
            self.verifier.classify([0.0, 1.0]), FaceEvidence.MISMATCH
        )
        self.assertEqual(
            self.verifier.classify([0.4, 0.9165]), FaceEvidence.AMBIGUOUS
        )

    def test_reset_removes_all_memory_only_identity_state(self):
        self.verifier.acquisition_samples = [np.array([1.0, 0.0])]

        self.verifier.reset()

        self.assertIsNone(self.verifier.reference)
        self.assertEqual(self.verifier.acquisition_samples, [])


if __name__ == "__main__":
    unittest.main()
