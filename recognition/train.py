"""Train and evaluate the final eight-intent SgSL classifier."""

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

from intents import INTENTS


CLASS_LABELS = tuple(INTENTS)
FEATURE_DIM = 144
FEATURE_VERSION = "v1"
FIXED_FRAMES = 30
MIN_SAMPLES_PER_CLASS = 5
TEST_SIZE = 0.33
RANDOM_SEED = 42
CALIBRATION_FOLDS = 3

BASE_DIRECTORY = Path(__file__).resolve().parent
DATA_DIRECTORY = BASE_DIRECTORY / "data"
MODEL_PATH = BASE_DIRECTORY / "models" / "sgsl_classifier.joblib"


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


def load_sample(path: Path, expected_intent=None):
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

            stored_intent = str(scalar_value(archive, "intent"))
            if stored_intent not in CLASS_LABELS:
                raise ValueError(f"unsupported intent {stored_intent!r}")
            if expected_intent is not None and stored_intent != expected_intent:
                raise ValueError(
                    f"stored intent is {stored_intent!r}, "
                    f"expected {expected_intent!r}"
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
    """Load every supported intent and report dataset statistics."""
    examples = []
    labels = []
    example_signers = []
    signer_ids = defaultdict(set)
    usable_counts = Counter()
    skipped_counts = Counter()
    frame_counts = defaultdict(list)

    for label in CLASS_LABELS:
        class_directory = data_directory / label
        for path in sorted(class_directory.glob("*.npz")):
            loaded = load_sample(path, expected_intent=label)
            if loaded is None:
                skipped_counts[label] += 1
                continue
            features, signer_id = loaded
            examples.append(resample_sequence(features).reshape(-1))
            labels.append(label)
            example_signers.append(signer_id)
            usable_counts[label] += 1
            frame_counts[label].append(features.shape[0])
            if signer_id:
                signer_ids[label].add(signer_id)

        counts = frame_counts[label]
        frame_summary = (
            f"min={min(counts)}, average={np.mean(counts):.2f}, max={max(counts)}"
            if counts
            else "unavailable"
        )
        represented = ", ".join(sorted(signer_ids[label])) or "not recorded"
        print(
            f"{label}: usable={usable_counts[label]}, "
            f"skipped={skipped_counts[label]}, signers={represented}, "
            f"frames({frame_summary})"
        )
        if usable_counts[label] < MIN_SAMPLES_PER_CLASS:
            raise RuntimeError(
                f"{label} needs at least {MIN_SAMPLES_PER_CLASS} usable "
                f"samples; found {usable_counts[label]}"
            )

    return (
        np.asarray(examples, dtype=np.float32),
        np.asarray(labels),
        np.asarray(example_signers, dtype=object),
        usable_counts,
        signer_ids,
        skipped_counts,
        frame_counts,
    )


def build_model() -> Pipeline:
    """Create the same standardized calibrated RBF SVM used in smoke testing."""
    return Pipeline(
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
                    cv=CALIBRATION_FOLDS,
                ),
            ),
        )
    )


def signer_aware_splits(labels, signers):
    """Return valid unseen-signer splits and reasons skipped splits are unsafe."""
    required_classes = set(CLASS_LABELS)
    valid_splits = []
    skipped_splits = {}
    for signer in sorted({item for item in signers if item}):
        test_indices = np.flatnonzero(signers == signer)
        train_indices = np.flatnonzero(signers != signer)
        missing_test_classes = required_classes.difference(labels[test_indices])
        missing_train_classes = required_classes.difference(labels[train_indices])
        if missing_test_classes or missing_train_classes:
            skipped_splits[signer] = (
                "missing test classes " + ", ".join(sorted(missing_test_classes))
                if missing_test_classes
                else "missing training classes "
                + ", ".join(sorted(missing_train_classes))
            )
            continue

        training_counts = Counter(labels[train_indices])
        too_small = [
            label
            for label in CLASS_LABELS
            if training_counts[label] < CALIBRATION_FOLDS
        ]
        if too_small:
            skipped_splits[signer] = (
                f"training side has fewer than {CALIBRATION_FOLDS} samples for "
                + ", ".join(too_small)
            )
            continue
        valid_splits.append((train_indices, test_indices, signer))
    return valid_splits, skipped_splits


def print_evaluation(test_labels, predictions) -> dict:
    """Print overall, per-intent, confusion, strongest, and weakest results."""
    accuracy = accuracy_score(test_labels, predictions)
    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            test_labels,
            predictions,
            labels=CLASS_LABELS,
            average="macro",
            zero_division=0,
        )
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        test_labels,
        predictions,
        labels=CLASS_LABELS,
        average=None,
        zero_division=0,
    )
    matrix = confusion_matrix(test_labels, predictions, labels=CLASS_LABELS)

    print(f"Overall accuracy: {accuracy:.3f}")
    print(f"Macro precision: {macro_precision:.3f}")
    print(f"Macro recall: {macro_recall:.3f}")
    print(f"Macro F1: {macro_f1:.3f}")
    print("Per-intent metrics:")
    for index, label in enumerate(CLASS_LABELS):
        print(
            f"  {label}: precision={precision[index]:.3f}, "
            f"recall={recall[index]:.3f}, F1={f1[index]:.3f}, "
            f"support={int(support[index])}"
        )

    print(f"Confusion matrix order: {CLASS_LABELS}")
    print(matrix)

    best_f1 = float(np.max(f1))
    weakest_f1 = float(np.min(f1))
    best = [CLASS_LABELS[index] for index in np.flatnonzero(f1 == best_f1)]
    weakest = [
        CLASS_LABELS[index] for index in np.flatnonzero(f1 == weakest_f1)
    ]
    print(f"Best-performing intents (F1={best_f1:.3f}): {', '.join(best)}")
    print(f"Weakest intents (F1={weakest_f1:.3f}): {', '.join(weakest)}")

    confused_pairs = []
    for true_index, true_label in enumerate(CLASS_LABELS):
        for predicted_index, predicted_label in enumerate(CLASS_LABELS):
            if true_index == predicted_index or matrix[true_index, predicted_index] == 0:
                continue
            confused_pairs.append(
                (
                    int(matrix[true_index, predicted_index]),
                    true_label,
                    predicted_label,
                )
            )
    confused_pairs.sort(reverse=True)
    if confused_pairs:
        highest_count = confused_pairs[0][0]
        common = [item for item in confused_pairs if item[0] == highest_count]
        print(
            "Most common confusions: "
            + ", ".join(
                f"{true_label} -> {predicted_label} ({count})"
                for count, true_label, predicted_label in common
            )
        )
    else:
        print("Most common confusions: none in this test split")

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_intent": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(CLASS_LABELS)
        },
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": list(CLASS_LABELS),
        "best_intents": best,
        "weakest_intents": weakest,
    }


def train() -> None:
    """Train, evaluate, and save the final eight-intent classifier."""
    (
        examples,
        labels,
        example_signers,
        usable_counts,
        signer_ids,
        skipped_counts,
        frame_counts,
    ) = load_dataset()

    print(f"Total usable samples: {len(labels)}")
    print(f"Total malformed/skipped samples: {sum(skipped_counts.values())}")
    print(f"Fixed temporal frames: {FIXED_FRAMES}")

    all_indices = np.arange(len(labels))
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels,
    )
    training_examples = examples[train_indices]
    test_examples = examples[test_indices]
    training_labels = labels[train_indices]
    test_labels = labels[test_indices]
    evaluation_strategy = "same-signer / same-dataset stratified evaluation"
    print(f"\nEvaluation strategy: {evaluation_strategy}")
    print(f"Training size: {len(training_labels)}")
    print(f"Test size: {len(test_labels)}")

    evaluation_model = build_model()
    evaluation_model.fit(training_examples, training_labels)
    predictions = evaluation_model.predict(test_examples)
    stratified_metrics = print_evaluation(test_labels, predictions)
    print(
        "Warning: these metrics are not evidence of unseen-signer "
        "generalisation."
    )

    unseen_signer_metrics = {}
    valid_splits, skipped_splits = signer_aware_splits(labels, example_signers)
    for held_out_signer, reason in skipped_splits.items():
        print(f"\nUnseen-signer evaluation held out {held_out_signer}: SKIPPED")
        print(f"Reason: {reason}")
    for signer_train_indices, signer_test_indices, held_out_signer in valid_splits:
        print(f"\nUnseen-signer evaluation: held out signer {held_out_signer}")
        print(f"Training size: {len(signer_train_indices)}")
        print(f"Test size: {len(signer_test_indices)}")
        signer_model = build_model()
        signer_model.fit(examples[signer_train_indices], labels[signer_train_indices])
        signer_predictions = signer_model.predict(examples[signer_test_indices])
        unseen_signer_metrics[held_out_signer] = print_evaluation(
            labels[signer_test_indices],
            signer_predictions,
        )

    represented_signers = {item for item in example_signers if item}
    if "C" in represented_signers:
        final_indices = np.flatnonzero(np.isin(example_signers, ("A", "B")))
        final_training_description = "signers A and B; signer C reserved"
    else:
        final_indices = all_indices
        final_training_description = "all usable samples"

    final_counts = Counter(labels[final_indices])
    insufficient_final_classes = [
        label
        for label in CLASS_LABELS
        if final_counts[label] < CALIBRATION_FOLDS
    ]
    if insufficient_final_classes:
        raise RuntimeError(
            "Final training data has too few samples for probability calibration: "
            + ", ".join(insufficient_final_classes)
        )

    print(f"\nFitting final saved model on {final_training_description}...")
    model = build_model()
    model.fit(examples[final_indices], labels[final_indices])
    model_classes = [str(label) for label in model.classes_]
    artifact = {
        "artifact_type": "final_eight_intent_classifier",
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
        "class_labels": model_classes,
        "supported_intents": list(CLASS_LABELS),
        "random_seed": RANDOM_SEED,
        "usable_sample_counts": dict(usable_counts),
        "skipped_sample_counts": dict(skipped_counts),
        "signer_ids": {
            label: sorted(signer_ids[label]) for label in CLASS_LABELS
        },
        "frame_count_statistics": {
            label: {
                "minimum": min(frame_counts[label]),
                "average": float(np.mean(frame_counts[label])),
                "maximum": max(frame_counts[label]),
            }
            for label in CLASS_LABELS
        },
        "final_training_description": final_training_description,
        "evaluation_strategy": evaluation_strategy,
        "evaluation_metrics": stratified_metrics,
        "unseen_signer_metrics": unseen_signer_metrics,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved final model: {MODEL_PATH}")


if __name__ == "__main__":
    train()
