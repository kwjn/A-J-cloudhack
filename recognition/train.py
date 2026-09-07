"""Train a temporary ALLERGY-versus-PACK SgSL smoke-test classifier."""

from collections import Counter, defaultdict
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


CLASS_LABELS = ("ALLERGY", "PACK")
FEATURE_DIM = 144
FEATURE_VERSION = "v1"
FIXED_FRAMES = 30
MIN_SAMPLES_PER_CLASS = 5
TEST_SIZE = 0.33
RANDOM_SEED = 42

BASE_DIRECTORY = Path(__file__).resolve().parent
DATA_DIRECTORY = BASE_DIRECTORY / "data"
MODEL_PATH = BASE_DIRECTORY / "models" / "sgsl_classifier_test.joblib"


def warning(path: Path, reason: str) -> None:
    """Print a concise malformed-sample warning to stderr."""
    print(f"Warning: skipping {path}: {reason}", file=sys.stderr)


def scalar_value(archive, key: str):
    """Read a required scalar field from an npz archive."""
    if key not in archive:
        raise ValueError(f"missing {key}")
    value = archive[key]
    if value.size != 1:
        raise ValueError(f"{key} is not scalar")
    return value.item()


def load_sample(path: Path):
    """Load and validate one temporal feature sample."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "features" not in archive:
                raise ValueError("missing features")
            features = np.asarray(archive["features"])
            if features.ndim != 2:
                raise ValueError(f"features must be 2D, got shape {features.shape}")
            if features.shape[0] == 0:
                raise ValueError("feature sequence is empty")
            if features.shape[1] != FEATURE_DIM:
                raise ValueError(
                    f"features have {features.shape[1]} values per frame, "
                    f"expected {FEATURE_DIM}"
                )

            stored_feature_dim = scalar_value(archive, "feature_dim")
            if int(stored_feature_dim) != FEATURE_DIM:
                raise ValueError(
                    f"feature_dim is {stored_feature_dim}, expected {FEATURE_DIM}"
                )

            stored_feature_version = str(
                scalar_value(archive, "feature_version")
            )
            if stored_feature_version != FEATURE_VERSION:
                raise ValueError(
                    f"feature_version is {stored_feature_version!r}, "
                    f"expected {FEATURE_VERSION!r}"
                )

            if not np.issubdtype(features.dtype, np.number):
                raise ValueError("features are not numeric")
            if not np.isfinite(features).all():
                raise ValueError("features contain NaN or infinity")

            signer_id = (
                str(scalar_value(archive, "signer_id"))
                if "signer_id" in archive
                else None
            )
            return features.astype(np.float32, copy=False), signer_id
    except (OSError, ValueError, TypeError) as error:
        warning(path, str(error))
        return None


def resample_sequence(
    features: np.ndarray,
    fixed_frames: int = FIXED_FRAMES,
) -> np.ndarray:
    """Linearly resample a frames-by-features sequence to a fixed length."""
    if fixed_frames < 1:
        raise ValueError("fixed_frames must be positive")
    if features.shape[0] == fixed_frames:
        return features.astype(np.float32, copy=True)
    if features.shape[0] == 1:
        return np.repeat(features, fixed_frames, axis=0).astype(np.float32)

    target_positions = np.linspace(0, features.shape[0] - 1, fixed_frames)
    lower_indices = np.floor(target_positions).astype(np.int64)
    upper_indices = np.ceil(target_positions).astype(np.int64)
    weights = (target_positions - lower_indices).astype(np.float32)[:, None]
    return (
        features[lower_indices] * (1.0 - weights)
        + features[upper_indices] * weights
    ).astype(np.float32)


def load_dataset(data_directory: Path = DATA_DIRECTORY):
    """Load only the two temporary smoke-test classes."""
    examples = []
    labels = []
    signer_ids = defaultdict(set)
    usable_counts = Counter()

    for label in CLASS_LABELS:
        class_directory = data_directory / label
        for path in sorted(class_directory.glob("*.npz")):
            loaded = load_sample(path)
            if loaded is None:
                continue
            features, signer_id = loaded
            examples.append(resample_sequence(features).reshape(-1))
            labels.append(label)
            usable_counts[label] += 1
            if signer_id:
                signer_ids[label].add(signer_id)

        print(f"{label} usable samples: {usable_counts[label]}")
        if usable_counts[label] < MIN_SAMPLES_PER_CLASS:
            raise RuntimeError(
                f"{label} needs at least {MIN_SAMPLES_PER_CLASS} usable "
                f"samples; found {usable_counts[label]}"
            )

    return (
        np.asarray(examples, dtype=np.float32),
        np.asarray(labels),
        usable_counts,
        signer_ids,
    )


def train() -> None:
    """Train, evaluate, report, and save the temporary classifier."""
    examples, labels, usable_counts, signer_ids = load_dataset()

    print(f"Fixed temporal frames: {FIXED_FRAMES}")
    for label in CLASS_LABELS:
        represented = ", ".join(sorted(signer_ids[label])) or "not recorded"
        print(f"{label} signer IDs: {represented}")

    training_examples, test_examples, training_labels, test_labels = (
        train_test_split(
            examples,
            labels,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            stratify=labels,
        )
    )
    print(f"Training size: {len(training_labels)}")
    print(f"Test size: {len(test_labels)}")

    model = Pipeline(
        steps=(
            ("scaler", StandardScaler()),
            (
                "classifier",
                CalibratedClassifierCV(
                    estimator=SVC(
                        kernel="rbf",
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                    ),
                    method="sigmoid",
                    cv=3,
                ),
            ),
        )
    )
    model.fit(training_examples, training_labels)
    predictions = model.predict(test_examples)

    accuracy = accuracy_score(test_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels,
        predictions,
        labels=CLASS_LABELS,
        average="macro",
        zero_division=0,
    )
    matrix = confusion_matrix(test_labels, predictions, labels=CLASS_LABELS)

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Macro precision: {precision:.3f}")
    print(f"Macro recall: {recall:.3f}")
    print(f"Macro F1: {f1:.3f}")
    print(f"Confusion matrix (rows=true, columns=predicted; {CLASS_LABELS}):")
    print(matrix)
    print(
        "Warning: this same-dataset split is only a smoke test; its metrics "
        "are not evidence of generalisation to unseen signers."
    )

    artifact = {
        "artifact_type": "temporary_two_intent_smoke_test",
        "model": model,
        "classifier": model.named_steps["classifier"],
        "scaler": model.named_steps["scaler"],
        "preprocessing": {
            "method": "linear_interpolation",
            "fixed_frames": FIXED_FRAMES,
            "flatten": True,
        },
        "fixed_frame_count": FIXED_FRAMES,
        "feature_dimension": FEATURE_DIM,
        "feature_version": FEATURE_VERSION,
        "class_labels": list(CLASS_LABELS),
        "random_seed": RANDOM_SEED,
        "usable_sample_counts": dict(usable_counts),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved temporary model: {MODEL_PATH}")


if __name__ == "__main__":
    train()
