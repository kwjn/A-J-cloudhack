"""Focused tests for the local prediction JSON bridge."""

from pathlib import Path
import tempfile
import unittest

from services.recognition_bridge import (
    publish_prediction,
    read_latest_prediction,
    read_new_prediction,
)


PACK_RESULT = {
    "intent": "PACK",
    "text": "Please pack this.",
    "confidence": 0.87,
    "critical": False,
}


class RecognitionBridgeTests(unittest.TestCase):
    def test_publish_and_read_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest_prediction.json"
            published = publish_prediction(PACK_RESULT, path=path)
            loaded = read_latest_prediction(path=path)

            self.assertEqual(loaded, published)
            self.assertEqual(loaded["text"], "Please pack this.")
            self.assertEqual(loaded["confidence"], 0.87)
            self.assertTrue(loaded["prediction_id"])
            self.assertTrue(loaded["timestamp"])
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_missing_and_invalid_json_return_none(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest_prediction.json"
            self.assertIsNone(read_latest_prediction(path=path))
            path.write_text("{not valid json", encoding="utf-8")
            self.assertIsNone(read_latest_prediction(path=path))

    def test_prediction_id_prevents_duplicate_consumption(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest_prediction.json"
            published = publish_prediction(PACK_RESULT, path=path)

            first = read_new_prediction(path=path)
            duplicate = read_new_prediction(
                last_prediction_id=published["prediction_id"],
                path=path,
            )

            self.assertEqual(first, published)
            self.assertIsNone(duplicate)


if __name__ == "__main__":
    unittest.main()
