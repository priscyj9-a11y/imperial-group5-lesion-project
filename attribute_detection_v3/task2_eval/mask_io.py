"""File-path helpers for Task 2 attribute masks.

THE PROBLEM:
Task 2 has five attributes per image. The evaluator must always compare the
prediction and ground truth for the same image ID and the same attribute.

WHY IT MATTERS:
The shared preprocessing code is responsible for loading masks, resizing them to
512x512, and converting them to 0/1. This file does not duplicate that work. It
only builds the expected file paths.

WHERE IT FITS:
evaluate_validation.py imports these helpers to find the correct ground-truth
mask and prediction mask before passing them to preprocessing.transforms.load_mask.
"""

from __future__ import annotations

from pathlib import Path


ATTRIBUTE_NAMES = [
    "pigment_network",
    "negative_network",
    "streaks",
    "milia_like_cyst",
    "globules",
]


def ground_truth_mask_path(data_root: Path, image_id: str, attribute: str) -> Path:
    """Return the real Task 2 mask path for one image and one attribute.

    Example:
        image_id = "000001"
        attribute = "globules"

    Result:
        data_root/task2_gt/000001_attribute_globules.png
    """
    # data_root is the dataset train folder, e.g.
    # C:/.../summer_school_project_train/train
    # Task 2 ground-truth masks live inside data_root/task2_gt/.
    return data_root / "task2_gt" / f"{image_id}_attribute_{attribute}.png"


def prediction_mask_path(prediction_root: Path, image_id: str, attribute: str) -> Path:
    """Return the expected predicted Task 2 mask path.

    Expected format:
        prediction_root / image_id / attribute.png

    Example:
        predictions/task2/000001/globules.png
    """
    # prediction_root is the folder Rita exports, e.g. predictions/task2/.
    # We expect one subfolder per image ID and one PNG per attribute.
    return prediction_root / image_id / f"{attribute}.png"
