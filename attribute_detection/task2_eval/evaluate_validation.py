"""Evaluate Task 2 predictions on the validation split.

THE PROBLEM:
Rita trains the model on the train split. I should evaluate on the validation
split, because those images were not used for learning.

WHY IT MATTERS:
Evaluating on train images can give scores that are too optimistic. Validation
images give a better estimate of how the model performs on unseen data.

WHERE IT FITS:
Once Rita exports predicted masks, run this script with:
    --splits-csv path/to/splits.csv
    --prediction-root path/to/predictions/task2

SPLIT FILE:
The team-generated data/splits.csv contains one row per image. Each row tells
whether the image belongs to train or val. This script uses only the rows marked
val, because validation images are used to evaluate the model after training.

PREPROCESSING RULE:
This file uses preprocessing/transforms.py directly. That is the same shared
preprocessing Rita uses for model inputs and masks, so the evaluation compares
masks in the same 512x512 format.

LOCAL SETUP NOTE:
This import will only work after the team repo contains:
    preprocessing/transforms.py
If that folder is not present locally yet, VS Code/Python will show
"ModuleNotFoundError: No module named 'preprocessing'". That is a setup issue,
not a metrics issue.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from attribute_detection.task2_eval.mask_io import (
    ATTRIBUTE_NAMES,
    ground_truth_mask_path,
    prediction_mask_path,
)
from attribute_detection.task2_eval.metrics import all_metrics

# Shared preprocessing from the group's Task 1/preprocessing code.
# We use load_mask() here instead of rewriting resize/conversion logic.
# It automatically resizes masks to DEFAULT_SIZE=512 and converts them to 0/1.
# Keep this import as the single source of truth for mask preprocessing.
from preprocessing.transforms import load_mask


def read_validation_ids(splits_csv: Path) -> list[str]:
    """Read splits.csv and keep only IDs where split == val.

    The CSV is expected to contain:
        image_id,split
        000001,val
        000002,train
    """
    validation_ids: list[str] = []

    # The split CSV is a normal text table created by create_splits.py:
    # image_id,split
    # 000001,val
    # 000002,train
    # DictReader lets us access each row by column name, e.g. row["split"].
    with splits_csv.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"image_id", "split"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError("splits.csv must contain columns: image_id, split")

        for row in reader:
            # Only validation images are evaluated. Train images were used for
            # learning, so they are skipped here.
            if row["split"].strip().lower() == "val":
                validation_ids.append(row["image_id"].strip())

    return validation_ids


def evaluate_one_pair(
    data_root: Path,
    prediction_root: Path,
    image_id: str,
    attribute: str,
) -> dict[str, object]:
    """Evaluate one predicted attribute mask against its real mask."""
    # Build the two filenames that belong together.
    # Example for image_id="000001" and attribute="globules":
    #   gt_path   -> train/task2_gt/000001_attribute_globules.png
    #   pred_path -> predictions/task2/000001/globules.png
    gt_path = ground_truth_mask_path(data_root, image_id, attribute)
    pred_path = prediction_mask_path(prediction_root, image_id, attribute)

    # This should be 0 once Rita's export is complete. Keeping the check gives a
    # clear CSV row instead of crashing if a file path is wrong during setup.
    if not pred_path.exists():
        return {
            "image_id": image_id,
            "attribute": attribute,
            "status": "missing_prediction",
            "gt_present": "",
            "pred_present": "",
            "dice": "",
            "iou": "",
            "precision": "",
            "recall": "",
        }

    # Use the shared preprocessing on BOTH masks. This makes the comparison fair:
    # both arrays are 512x512 and binary 0/1 before metrics are calculated.
    ground_truth = load_mask(gt_path)
    prediction = load_mask(pred_path)

    # If this ever fails, it means one side was not prepared using the shared
    # preprocessing rule. Pixel metrics cannot compare different-sized arrays.
    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"Shape mismatch for {image_id} {attribute}: "
            f"prediction {prediction.shape}, ground truth {ground_truth.shape}. "
            "Both should use preprocessing.transforms.load_mask."
        )

    # all_metrics returns a dictionary with Dice, IoU, precision, and recall.
    scores = all_metrics(prediction, ground_truth)

    return {
        "image_id": image_id,
        "attribute": attribute,
        "status": "ok",
        # max()==1 means "there is at least one white pixel", so the attribute
        # is present. int() stores True/False as 1/0 in the output CSV.
        "gt_present": int(ground_truth.max() == 1),
        "pred_present": int(prediction.max() == 1),
        # **scores expands the metrics dictionary into this result row.
        **scores,
    }


def write_rows(output_csv: Path, rows: list[dict[str, object]]) -> None:
    """Write per-image, per-attribute metrics to CSV."""
    # Create the output folder if it does not exist yet.
    # Example: outputs/task2_evaluation/
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # These are the CSV columns, in the exact order we want them to appear.
    fieldnames = [
        "image_id",
        "attribute",
        "status",
        "gt_present",
        "pred_present",
        "dice",
        "iou",
        "precision",
        "recall",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as file:
        # DictWriter turns dictionaries like {"image_id": "000001", "dice": 0.8}
        # into CSV rows under the matching column names.
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_attribute_summary(rows: list[dict[str, object]]) -> None:
    """Print mean scores per attribute, ignoring missing predictions."""
    # grouped will become:
    # {
    #   "globules": [row1, row2, ...],
    #   "streaks": [row1, row2, ...],
    # }
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)

    for row in rows:
        if row["status"] == "ok":
            grouped[str(row["attribute"])].append(row)

    print("\nValidation summary per attribute")
    for attribute in ATTRIBUTE_NAMES:
        attr_rows = grouped[attribute]
        if not attr_rows:
            print(f"  {attribute:<18}: no predictions found")
            continue

        # Each metric is first calculated per image+attribute pair.
        # Here we average those values to get one summary line per attribute.
        mean_dice = np.mean([float(row["dice"]) for row in attr_rows])
        mean_iou = np.mean([float(row["iou"]) for row in attr_rows])
        mean_precision = np.mean([float(row["precision"]) for row in attr_rows])
        mean_recall = np.mean([float(row["recall"]) for row in attr_rows])

        print(
            f"  {attribute:<18}: "
            f"Dice={mean_dice:.3f}, IoU={mean_iou:.3f}, "
            f"Precision={mean_precision:.3f}, Recall={mean_recall:.3f}"
        )


def main() -> None:
    # argparse defines the terminal inputs for this script. Required arguments
    # must be provided when running the command.
    parser = argparse.ArgumentParser(description="Evaluate Task 2 predictions on validation images.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"C:\Users\Ivana\Downloads\summer_school_project_train\train"),
        help="Folder containing images/ and task2_gt/.",
    )
    parser.add_argument(
        "--splits-csv",
        type=Path,
        required=True,
        help="Team-generated CSV from create_splits.py with columns image_id,split.",
    )
    parser.add_argument(
        "--prediction-root",
        type=Path,
        required=True,
        help="Folder containing predictions as prediction_root/image_id/attribute.png.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs") / "task2_evaluation" / "validation_scores.csv",
    )
    args = parser.parse_args()

    # Read the team split and keep only validation IDs.
    validation_ids = read_validation_ids(args.splits_csv)
    print(f"Validation IDs loaded: {len(validation_ids)}")

    # Build one result row for every validation image and every Task 2 attribute.
    # Example: 540 validation images x 5 attributes = 2700 evaluated pairs.
    rows: list[dict[str, object]] = []
    for image_id in validation_ids:
        for attribute in ATTRIBUTE_NAMES:
            rows.append(
                evaluate_one_pair(
                    args.data_root,
                    args.prediction_root,
                    image_id,
                    attribute,
                )
            )

    write_rows(args.output_csv, rows)
    print(f"Wrote detailed scores: {args.output_csv}")

    missing = sum(1 for row in rows if row["status"] == "missing_prediction")
    print(f"Missing prediction masks: {missing}")
    print_attribute_summary(rows)


if __name__ == "__main__":
    main()
