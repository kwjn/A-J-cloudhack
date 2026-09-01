"""Focused tests for automatic signer-session identity handling."""

import unittest

from face_identity import FaceEvidence
from signer_session import SessionState, SignerSessionController


def pose(x=0.4, face=None):
    return {
        "landmarks": [],
        "shoulder_midpoint": (x, 0.5),
        "shoulder_width": 0.2,
        "body_region": (x - 0.1, 0.3, x + 0.1, 0.7),
        "body_center": (x, 0.5),
        "face": face,
    }


class FakeFaceVerifier:
    def __init__(self):
        self.reference = None
        self.samples = []
        self.reset_count = 0

    @property
    def reference_sample_count(self):
        return len(self.samples)

    def embeddings_for_observations(self, frame, observations):
        return {
            index: observation["face"]
            for index, observation in enumerate(observations)
            if observation["face"] is not None
        }

    def add_acquisition_sample(self, embedding):
        if self.samples and embedding != self.samples[0]:
            return False
        self.samples.append(embedding)
        return True

    def finalize_reference(self):
        self.reference = self.samples[0] if self.samples else None
        return self.reference is not None

    def classify(self, embedding):
        if embedding == self.reference:
            return FaceEvidence.MATCH
        if embedding == "ambiguous":
            return FaceEvidence.AMBIGUOUS
        return FaceEvidence.MISMATCH

    def reset_acquisition(self):
        if self.reference is None:
            self.samples = []

    def reset(self):
        self.reference = None
        self.samples = []
        self.reset_count += 1


class SignerSessionControllerTests(unittest.TestCase):
    def setUp(self):
        self.faces = FakeFaceVerifier()
        self.session = SignerSessionController(
            self.faces,
            stable_frames=3,
            reference_samples=2,
        )

    def auto_lock_a(self):
        result = None
        for frame_number in range(3):
            result = self.session.update(
                None,
                [pose(face="A")],
                current_time=10.0 + frame_number * 0.1,
            )
        self.assertEqual(result.state, SessionState.LOCKED)
        return result

    def test_stable_candidate_with_consistent_faces_auto_locks(self):
        result = self.auto_lock_a()

        self.assertIsNotNone(result.active_signer)
        self.assertTrue(self.session.tracker.locked)
        self.assertEqual(self.faces.reference, "A")

    def test_different_face_cannot_inherit_same_position(self):
        self.auto_lock_a()

        result = self.session.update(None, [pose(face="B")], 10.4)

        self.assertEqual(result.state, SessionState.TEMPORARILY_LOST)
        self.assertIsNone(result.active_signer)
        self.assertTrue(self.session.tracker.locked)

    def test_ambiguous_face_returns_no_active_signer(self):
        self.auto_lock_a()

        result = self.session.update(None, [pose(face="ambiguous")], 10.4)

        self.assertEqual(result.state, SessionState.TEMPORARILY_LOST)
        self.assertIsNone(result.active_signer)

    def test_one_unambiguous_pose_can_continue_during_face_loss(self):
        self.auto_lock_a()

        result = self.session.update(None, [pose(face=None)], 10.4)

        self.assertEqual(result.state, SessionState.LOCKED)
        self.assertIsNotNone(result.active_signer)
        self.assertEqual(self.faces.reference, "A")

    def test_multiple_faceless_candidates_are_not_used(self):
        self.auto_lock_a()

        result = self.session.update(
            None,
            [pose(0.4, None), pose(0.45, None)],
            10.4,
        )

        self.assertEqual(result.state, SessionState.TEMPORARILY_LOST)
        self.assertIsNone(result.active_signer)

    def test_timeout_clears_identity_and_returns_to_searching(self):
        self.auto_lock_a()

        result = self.session.update(None, [pose(face="B")], 12.21)

        self.assertEqual(result.state, SessionState.SEARCHING)
        self.assertFalse(self.session.tracker.locked)
        self.assertIsNone(self.faces.reference)
        self.assertEqual(self.session.acquisition_frame_count, 0)

    def test_manual_reset_clears_session(self):
        self.auto_lock_a()

        self.session.reset()

        self.assertEqual(self.session.state, SessionState.SEARCHING)
        self.assertFalse(self.session.tracker.locked)
        self.assertIsNone(self.faces.reference)


if __name__ == "__main__":
    unittest.main()
