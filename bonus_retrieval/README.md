"""Option 3 RAG support: compare Task 2 predictions with CLIP-neighbour evidence.

THE GOAL:
This script is used after two files/folders exist:
1. bonus/test_bonus_rag_summary.csv from rag_attribute_summary.py
2. Task 2 predicted masks for the test images

It compares:
    attributes predicted for the query/test image
vs
    attributes observed in the retrieved training neighbours

The output is a CSV with a short support sentence for each query/test image.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from task2_eval.mask_io import ATTRIBUTE_NAMES, prediction_mask_path


def load_shared_mask(mask_path: Path):
    """Load a prediction mask using the group's shared preprocessing."""
    try:
        from preprocessing.transforms import load_mask
    except ImportError as error:
        raise ImportError(
            "Could not import preprocessing.transforms.load_mask.\n"
            "Add the team's shared preprocessing folder to the project first.\n"
            "Expected file: preprocessing/transforms.py"
        ) from error

    return load_mask(mask_path)


def predicted_attribute_present(prediction_root: Path, image_id: str, attribute: str) -> bool:
    """Check whether the Task 2 prediction mask says an attribute is present."""
    mask_path = prediction_mask_path(prediction_root, image_id, attribute)
    mask = load_shared_mask(mask_path)
    return bool(mask.max() == 1)


def read_option2_summary(summary_csv: Path) -> list[dict[str, str]]:
    """Read the Option 2 RAG summary CSV."""
    with summary_csv.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def neighbour_count(row: dict[str, str], attribute: str) -> int:
    """Read one attribute count from an Option 2 summary row."""
    return int(row[f"{attribute}_count"])


def build_support_sentence(
    predicted_present: list[str],
    predicted_absent: list[str],
    supported: list[str],
    weak_or_unsupported: list[str],
) -> str:
    """Create a short sentence comparing predictions with retrieval evidence."""
    parts = []

    if predicted_present:
        parts.append("Task 2 predicted present: " + ", ".join(predicted_present) + ".")
    else:
        parts.append("Task 2 did not predict any of the five attributes as present.")

    if supported:
        parts.append("Retrieval support was found for: " + ", ".join(supported) + ".")

    if weak_or_unsupported:
        parts.append("Weak or no retrieval support for: " + ", ".join(weak_or_unsupported) + ".")

    if predicted_absent:
        parts.append("Predicted absent: " + ", ".join(predicted_absent) + ".")

    return " ".join(parts)


def compare_predictions_with_retrieval(
    option2_rows: list[dict[str, str]],
    prediction_root: Path,
    output_csv: Path,
) -> None:
    """Write the Option 3 support CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "query_image",
        "predicted_present",
        "retrieval_supported",
        "weak_or_unsupported",
        "support_summary",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in option2_rows:
            query_id = row["query_image"]
            predicted_present = []
            predicted_absent = []
            supported = []
            weak_or_unsupported = []

            for attribute in ATTRIBUTE_NAMES:
                is_predicted = predicted_attribute_present(prediction_root, query_id, attribute)
                retrieval_count = neighbour_count(row, attribute)

                if is_predicted:
                    predicted_present.append(attribute)

                    if retrieval_count > 0:
                        supported.append(attribute)
                    else:
                        weak_or_unsupported.append(attribute)
                else:
                    predicted_absent.append(attribute)

            support_summary = build_support_sentence(
                predicted_present,
                predicted_absent,
                supported,
                weak_or_unsupported,
            )

            writer.writerow(
                {
                    "query_image": query_id,
                    "predicted_present": ";".join(predicted_present),
                    "retrieval_supported": ";".join(supported),
                    "weak_or_unsupported": ";".join(weak_or_unsupported),
                    "support_summary": support_summary,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Task 2 predictions with CLIP-neighbour evidence.")
    parser.add_argument(
        "--option2-summary-csv",
        type=Path,
        default=Path("bonus") / "test_bonus_rag_summary.csv",
        help="CSV created by bonus_retrieval.rag_attribute_summary.",
    )
    parser.add_argument(
        "--prediction-root",
        type=Path,
        required=True,
        help="Folder containing Task 2 predicted masks as prediction_root/image_id/attribute.png.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("bonus") / "test_bonus_rag_prediction_support.csv",
        help="Where to save the Option 3 support CSV.",
    )
    args = parser.parse_args()

    option2_rows = read_option2_summary(args.option2_summary_csv)
    compare_predictions_with_retrieval(option2_rows, args.prediction_root, args.output_csv)

    print(f"Query images compared: {len(option2_rows)}")
    print(f"Wrote Option 3 support CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
