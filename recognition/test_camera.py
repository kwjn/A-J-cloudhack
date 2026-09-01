"""Focused tests for the primary signer lock lifecycle."""

import unittest

from signer_session import PrimarySignerTracker


def pose(x, y=0.5, shoulder_width=0.2):
    """Build the positional fields used by PrimarySignerTracker."""
    return {
        "landmarks": [],
        "shoulder_midpoint": (x, y),
        "shoulder_width": shoulder_width,
        "body_region": (x - 0.1, y - 0.2, x + 0.1, y + 0.2),
        "body_center": (x, y),
    }


class PrimarySignerTrackerTests(unittest.TestCase):
    def setUp(self):
        self.tracker = PrimarySignerTracker(lost_timeout=2.0)
        self.signer_a = pose(0.40)
        self.tracker.lock(self.signer_a, current_time=10.0)

    def test_lock_stays_with_visible_signer_when_bystander_is_present(self):
        signer_b = pose(0.51)

        match = self.tracker.match(
            [signer_b, self.signer_a],
            current_time=10.1,
        )

        self.assertIs(match, self.signer_a)
        self.assertTrue(self.tracker.locked)

    def test_brief_loss_does_not_switch_to_spatially_inconsistent_bystander(self):
        signer_b = pose(0.80)

        match = self.tracker.match([signer_b], current_time=11.0)

        self.assertIsNone(match)
        self.assertTrue(self.tracker.locked)

        reacquired = self.tracker.match([self.signer_a], current_time=11.5)
        self.assertIs(reacquired, self.signer_a)
        self.assertTrue(self.tracker.locked)

    def test_timeout_automatically_resets_all_signer_state(self):
        self.tracker.hand_positions = {"Left": (0.4, 0.5)}
        self.tracker.hand_seen_times = {"Left": 10.0}

        match = self.tracker.match([], current_time=12.01)

        self.assertIsNone(match)
        self.assertFalse(self.tracker.locked)
        self.assertIsNone(self.tracker.shoulder_midpoint)
        self.assertIsNone(self.tracker.shoulder_width)
        self.assertIsNone(self.tracker.body_region)
        self.assertIsNone(self.tracker.body_center)
        self.assertEqual(self.tracker.velocity, (0.0, 0.0))
        self.assertIsNone(self.tracker.last_seen_time)
        self.assertEqual(self.tracker.hand_positions, {})
        self.assertEqual(self.tracker.hand_seen_times, {})

    def test_new_candidate_can_be_selected_after_timeout(self):
        signer_b = pose(0.52)
        self.tracker.match([], current_time=12.01)

        candidate = self.tracker.select_candidate([pose(0.85), signer_b])

        self.assertIs(candidate, signer_b)
        self.assertFalse(self.tracker.locked)

    def test_manual_reset_immediately_clears_lock(self):
        self.tracker.reset()

        self.assertFalse(self.tracker.locked)
        self.assertIsNone(self.tracker.last_seen_time)


if __name__ == "__main__":
    unittest.main()
