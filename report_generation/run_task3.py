"""Run the complete Task 3 anchored findings pipeline."""

import argparse
from pathlib import Path

from .attribute_evidence import (
    extract_attribute_evidence,
    load_attribute_probabilities,
)
from .config import (
    DATASET_SPLIT,
    PIPELINE_MODEL_VERSION,
)
from .export_csv import (
    create_csv_row,
    save_rows,
)
from .generate_json import (
    build_json_record,
    save_json,
)
from .generate_report import (
    generate_findings_report,
    save_report,
)
from .lesion_features import (
    calculate_border_category,
    calculate_size_category,
    load_binary_mask,
)
from .validate_output import validate_json_and_report


# ------------------------------------------------------------------
# Project paths
# ------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

TEST_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "test"
)

ATTRIBUTE_CSV_PATH = (
    TEST_ROOT
    / "attribute_probabilities.csv"
)

TASK2_PREDICTION_DIR = (
    TEST_ROOT
    / "task2_predictions"
)

LESION_MASK_DIR = (
    TEST_ROOT
    / "lesion_masks"
)

JSON_DIR = (
    TEST_ROOT
    / "json"
)

REPORT_DIR = (
    TEST_ROOT
    / "reports"
)

LOG_DIR = (
    TEST_ROOT
    / "logs"
)

CSV_PATH = (
    REPORT_DIR
    / "findings_reports.csv"
)

SUMMARY_PATH = (
    LOG_DIR
    / "task3_validation_summary.txt"
)


# ------------------------------------------------------------------
# Input file handling
# ------------------------------------------------------------------

def find_lesion_mask(
    image_id: str,
) -> Path:
    """Find a Task 1 predicted lesion mask.

    Supports the common filename formats used by the group.
    """

    candidates = [
        LESION_MASK_DIR
        / f"{image_id}.png",

        LESION_MASK_DIR
        / f"{image_id}_predicted_mask.png",

        LESION_MASK_DIR
        / f"ISIC_{image_id}.png",

        LESION_MASK_DIR
        / f"ISIC_{image_id}_predicted_mask.png",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No Task 1 lesion mask was found for image {image_id}."
    )


# ------------------------------------------------------------------
# Output directory handling
# ------------------------------------------------------------------

def clear_generated_outputs() -> None:
    """Remove previous generated Task 3 outputs.

    Task 1 and Task 2 prediction inputs are not deleted.
    """

    JSON_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in JSON_DIR.glob("*.json"):
        path.unlink()

    for path in REPORT_DIR.glob("*.txt"):
        path.unlink()

    if CSV_PATH.exists():
        CSV_PATH.unlink()

    if SUMMARY_PATH.exists():
        SUMMARY_PATH.unlink()


def write_summary(
    total_images: int,
    successful_images: int,
    failures: dict[str, str],
) -> None:
    """Write a summary of the complete Task 3 run."""

    lines = [
        "Task 3 Processing Summary",
        "",
        f"Dataset split: {DATASET_SPLIT}",
        f"Expected images: {total_images}",
        f"Successfully processed: {successful_images}",
        f"Failed images: {len(failures)}",
        f"JSON files generated: {successful_images}",
        f"Written reports generated: {successful_images}",
        f"CSV rows generated: {successful_images}",
        "",
    ]

    if failures:
        lines.append("Failed image IDs:")

        for image_id, message in failures.items():
            lines.append(
                f"- {image_id}: {message}"
            )

    else:
        lines.append(
            "All images passed the Task 3 pipeline."
        )

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ------------------------------------------------------------------
# Main Task 3 pipeline
# ------------------------------------------------------------------

def run_pipeline(
    limit: int | None = None,
    clean: bool = False,
) -> None:
    """Run Task 3 on all test images or a limited subset."""

    if clean:
        clear_generated_outputs()

    dataframe = load_attribute_probabilities(
        ATTRIBUTE_CSV_PATH
    )

    if limit is not None:
        if limit <= 0:
            raise ValueError(
                "--limit must be greater than zero."
            )

        dataframe = dataframe.head(limit)

    total_images = len(dataframe)

    csv_rows = []
    failures: dict[str, str] = {}

    print(
        f"Starting Task 3 for {total_images} "
        f"{DATASET_SPLIT} images..."
    )

    for _, row in dataframe.iterrows():
        image_id = str(row["image_id"])

        try:
            print(f"Processing {image_id}")

            # ------------------------------------------------------
            # Task 1 evidence
            # ------------------------------------------------------

            lesion_mask_path = find_lesion_mask(
                image_id
            )

            lesion_mask = load_binary_mask(
                lesion_mask_path
            )

            area_ratio, size_category = (
                calculate_size_category(
                    lesion_mask
                )
            )

            border_score, border_category = (
                calculate_border_category(
                    lesion_mask
                )
            )

            # ------------------------------------------------------
            # Task 2 evidence
            # ------------------------------------------------------

            probabilities, statuses = (
                extract_attribute_evidence(
                    image_id=image_id,
                    probability_row=row,
                    prediction_root=TASK2_PREDICTION_DIR,
                    expected_shape=lesion_mask.shape,
                )
            )

            # ------------------------------------------------------
            # Structured JSON
            # ------------------------------------------------------

            json_record = build_json_record(
                image_id=image_id,
                split=DATASET_SPLIT,
                model_version=PIPELINE_MODEL_VERSION,
                probabilities=probabilities,
                statuses=statuses,
            )

            # ------------------------------------------------------
            # Written findings report
            # ------------------------------------------------------

            report_text = generate_findings_report(
                json_record=json_record,
                size_category=size_category,
                border_category=border_category,
            )

            # ------------------------------------------------------
            # Validation
            # ------------------------------------------------------

            validation_errors = (
                validate_json_and_report(
                    json_record=json_record,
                    report_text=report_text,
                    expected_statuses=statuses,
                )
            )

            if validation_errors:
                raise ValueError(
                    " | ".join(validation_errors)
                )

            # ------------------------------------------------------
            # Save individual outputs
            # ------------------------------------------------------

            save_json(
                record=json_record,
                output_path=(
                    JSON_DIR
                    / f"{image_id}.json"
                ),
            )

            save_report(
                report_text=report_text,
                output_path=(
                    REPORT_DIR
                    / f"{image_id}.txt"
                ),
            )

            # ------------------------------------------------------
            # Add row to combined CSV
            # ------------------------------------------------------

            csv_rows.append(
                create_csv_row(
                    json_record=json_record,
                    report_text=report_text,
                    lesion_area_ratio=area_ratio,
                    size_category=size_category,
                    border_irregularity=border_score,
                    border_category=border_category,
                )
            )

        except Exception as error:
            failures[image_id] = str(error)

            print(
                f"FAILED {image_id}: {error}"
            )

    # Save the combined CSV even if some images failed.
    save_rows(
        rows=csv_rows,
        output_path=CSV_PATH,
    )

    write_summary(
        total_images=total_images,
        successful_images=len(csv_rows),
        failures=failures,
    )

    print("")
    print("Task 3 run completed.")
    print(f"Successful: {len(csv_rows)}")
    print(f"Failed: {len(failures)}")
    print(f"Summary: {SUMMARY_PATH}")

    if failures:
        raise SystemExit(1)


# ------------------------------------------------------------------
# Command-line interface
# ------------------------------------------------------------------

def main() -> None:
    """Read command-line options and run Task 3."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate Task 3 JSON files, findings reports "
            "and a combined CSV."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Process only the first N images. "
            "Useful for testing."
        ),
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help=(
            "Remove previous generated JSON, reports, "
            "CSV and summary files before running."
        ),
    )

    args = parser.parse_args()

    run_pipeline(
        limit=args.limit,
        clean=args.clean,
    )


if __name__ == "__main__":
    main()