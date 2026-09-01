"""Automatic signer acquisition and session-local identity lifecycle."""

from dataclasses import dataclass
from enum import Enum
import math

from face_identity import FaceEvidence


# Initial values only. Calibrate them with the target camera and environment.
SIGNER_LOST_TIMEOUT_SECONDS = 2.0
ACQUISITION_STABLE_FRAMES = 12
FACE_REFERENCE_SAMPLE_COUNT = 5
TRACK_SMOOTHING = 0.25


def point_distance(first, second) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def smooth_point(previous, current):
    return tuple(
        old * (1.0 - TRACK_SMOOTHING) + new * TRACK_SMOOTHING
        for old, new in zip(previous, current)
    )


class PrimarySignerTracker:
    """Keep a conservative positional lock on one signer."""

    def __init__(self, lost_timeout=SIGNER_LOST_TIMEOUT_SECONDS):
        self.lost_timeout = lost_timeout
        self.reset()

    def reset(self) -> None:
        self.locked = False
        self.shoulder_midpoint = None
        self.shoulder_width = None
        self.body_region = None
        self.body_center = None
        self.velocity = (0.0, 0.0)
        self.last_seen_time = None
        self.hand_positions = {}
        self.hand_seen_times = {}

    def select_candidate(self, observations):
        if not observations:
            return None
        return min(
            observations,
            key=lambda item: point_distance(item["shoulder_midpoint"], (0.5, 0.5)),
        )

    def lock(self, observation, current_time) -> None:
        self.locked = True
        self.shoulder_midpoint = observation["shoulder_midpoint"]
        self.shoulder_width = observation["shoulder_width"]
        self.body_region = observation["body_region"]
        self.body_center = observation["body_center"]
        self.velocity = (0.0, 0.0)
        self.last_seen_time = current_time
        self.hand_positions = {}
        self.hand_seen_times = {}

    def match(self, observations, current_time):
        if not self.locked or self.last_seen_time is None:
            return None
        if current_time - self.last_seen_time > self.lost_timeout:
            self.reset()
            return None

        predicted_midpoint = (
            self.shoulder_midpoint[0] + self.velocity[0],
            self.shoulder_midpoint[1] + self.velocity[1],
        )
        maximum_shoulder_distance = max(0.08, self.shoulder_width * 0.80)
        body_width = self.body_region[2] - self.body_region[0]
        body_height = self.body_region[3] - self.body_region[1]
        maximum_body_distance = max(0.15, max(body_width, body_height) * 0.50)

        matches = []
        for observation in observations:
            shoulder_distance = point_distance(
                observation["shoulder_midpoint"], predicted_midpoint
            )
            body_distance = point_distance(
                observation["body_center"], self.body_center
            )
            scale_ratio = observation["shoulder_width"] / self.shoulder_width
            if shoulder_distance > maximum_shoulder_distance:
                continue
            if body_distance > maximum_body_distance:
                continue
            if not 0.70 <= scale_ratio <= 1.40:
                continue
            scale_change = abs(math.log(scale_ratio))
            score = shoulder_distance + 0.4 * body_distance + 0.1 * scale_change
            matches.append((score, observation))

        if not matches:
            return None

        observation = min(matches, key=lambda item: item[0])[1]
        previous_midpoint = self.shoulder_midpoint
        self.shoulder_midpoint = smooth_point(
            self.shoulder_midpoint, observation["shoulder_midpoint"]
        )
        movement = (
            observation["shoulder_midpoint"][0] - previous_midpoint[0],
            observation["shoulder_midpoint"][1] - previous_midpoint[1],
        )
        self.velocity = smooth_point(self.velocity, movement)
        self.body_center = smooth_point(self.body_center, observation["body_center"])
        self.body_region = smooth_point(self.body_region, observation["body_region"])
        self.shoulder_width = (
            self.shoulder_width * (1.0 - TRACK_SMOOTHING)
            + observation["shoulder_width"] * TRACK_SMOOTHING
        )
        self.last_seen_time = current_time
        return observation


class SessionState(Enum):
    SEARCHING = "SEARCHING"
    ACQUIRING = "ACQUIRING"
    LOCKED = "LOCKED"
    TEMPORARILY_LOST = "TEMPORARILY_LOST"


@dataclass
class SessionResult:
    state: SessionState
    active_signer: object = None
    acquisition_candidate: object = None


class SignerSessionController:
    """Own automatic acquisition, identity gating, loss, and reset."""

    def __init__(
        self,
        face_verifier,
        lost_timeout=SIGNER_LOST_TIMEOUT_SECONDS,
        stable_frames=ACQUISITION_STABLE_FRAMES,
        reference_samples=FACE_REFERENCE_SAMPLE_COUNT,
    ):
        self.face_verifier = face_verifier
        self.tracker = PrimarySignerTracker(lost_timeout)
        self.stable_frames = stable_frames
        self.reference_samples = reference_samples
        self.reset()

    def reset(self):
        self.tracker.reset()
        self.face_verifier.reset()
        self.state = SessionState.SEARCHING
        self.acquisition_candidate = None
        self.acquisition_frame_count = 0

    def close(self):
        self.reset()

    def update(self, frame, observations, current_time) -> SessionResult:
        embeddings = self.face_verifier.embeddings_for_observations(
            frame, observations
        )
        if self.tracker.locked:
            return self._update_locked(observations, embeddings, current_time)
        return self._update_acquisition(observations, embeddings, current_time)

    def _candidate_is_stable(self, candidate) -> bool:
        previous = self.acquisition_candidate
        if previous is None:
            return False
        shoulder_limit = max(0.05, previous["shoulder_width"] * 0.40)
        scale_ratio = candidate["shoulder_width"] / previous["shoulder_width"]
        return (
            point_distance(
                candidate["shoulder_midpoint"], previous["shoulder_midpoint"]
            )
            <= shoulder_limit
            and 0.80 <= scale_ratio <= 1.25
        )

    def _start_acquisition(self, candidate, embedding=None):
        self.face_verifier.reset_acquisition()
        self.acquisition_candidate = candidate
        self.acquisition_frame_count = 1
        self.state = SessionState.ACQUIRING
        if embedding is not None:
            self.face_verifier.add_acquisition_sample(embedding)

    def _update_acquisition(self, observations, embeddings, current_time):
        candidate = self.tracker.select_candidate(observations)
        if candidate is None:
            self.face_verifier.reset_acquisition()
            self.acquisition_candidate = None
            self.acquisition_frame_count = 0
            self.state = SessionState.SEARCHING
            return SessionResult(self.state)

        candidate_index = next(
            index for index, observation in enumerate(observations)
            if observation is candidate
        )
        embedding = embeddings.get(candidate_index)
        if not self._candidate_is_stable(candidate):
            self._start_acquisition(candidate, embedding)
            return SessionResult(self.state, acquisition_candidate=candidate)

        self.acquisition_candidate = candidate
        self.acquisition_frame_count += 1
        if embedding is not None:
            if not self.face_verifier.add_acquisition_sample(embedding):
                self._start_acquisition(candidate, embedding)
                return SessionResult(self.state, acquisition_candidate=candidate)

        if (
            self.acquisition_frame_count >= self.stable_frames
            and self.face_verifier.reference_sample_count >= self.reference_samples
            and self.face_verifier.finalize_reference()
        ):
            self.tracker.lock(candidate, current_time)
            self.state = SessionState.LOCKED
            return SessionResult(self.state, active_signer=candidate)

        return SessionResult(self.state, acquisition_candidate=candidate)

    def _update_locked(self, observations, embeddings, current_time):
        if current_time - self.tracker.last_seen_time > self.tracker.lost_timeout:
            self.reset()
            return SessionResult(self.state)

        face_matches = []
        no_face_candidates = []
        for index, observation in enumerate(observations):
            embedding = embeddings.get(index)
            if embedding is None:
                no_face_candidates.append(observation)
                continue
            evidence = self.face_verifier.classify(embedding)
            if evidence is FaceEvidence.MATCH:
                face_matches.append(observation)
            # MISMATCH and AMBIGUOUS are deliberately hard-rejected.

        eligible = face_matches
        if not eligible and len(no_face_candidates) == 1:
            eligible = no_face_candidates

        active_signer = self.tracker.match(eligible, current_time)
        if active_signer is None:
            self.state = SessionState.TEMPORARILY_LOST
            return SessionResult(self.state)

        self.state = SessionState.LOCKED
        return SessionResult(self.state, active_signer=active_signer)
