"""Focused tests for temporary two-intent prediction preprocessing."""

import unittest

import numpy as np

from predict import (
    EXPECTED_CLASSES,
    FEATURE_DIM,
    classify_sequence,
    load_model_artifact,
    preprocess_sequence,
)


class TemporaryPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact = load_model_artifact()

    def test_saved_model_loads_with_expected_classes(self):
        classes = tuple(str(item) for item in self.artifact["model"].classes_)
        self.assertEqual(classes, EXPECTED_CLASSES)

    def test_variable_sequence_lengths_have_training_input_dimension(self):
        expected_width = self.artifact["fixed_frame_count"] * FEATURE_DIM
        for frame_count in (1, 17, 30, 83):
            with self.subTest(frame_count=frame_count):
                sequence = np.zeros((frame_count, FEATURE_DIM), dtype=np.float32)
                processed = preprocess_sequence(self.artifact, sequence)
                self.assertEqual(processed.shape, (1, expected_width))

    def test_classification_returns_confidence_and_intent_metadata(self):
        sequence = np.zeros((21, FEATURE_DIM), dtype=np.float32)
        result = classify_sequence(self.artifact, sequence)

        self.assertIn(result["intent"], EXPECTED_CLASSES)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)
        self.assertIsInstance(result["text"], str)
        self.assertTrue(result["text"])
        self.assertIsInstance(result["critical"], bool)


if __name__ == "__main__":
    unittest.main()
