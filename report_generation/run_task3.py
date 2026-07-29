import json
from pathlib import Path

from export_csv import create_csv_row, save_rows
from generate_json import build_json_record, save_json
from generate_report import generate_findings_report, save_report
from lesion_features import (
    calculate_border_category,
    calculate_size_category,
    load_binary_mask,
)
from validate_output import validate_json_and_report


# Locate the project folders
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Input files and folders
MOCK_PATH = BASE_DIR / "mock_predictions.json"
LESION_MASK_DIR = PROJECT_ROOT / "outputs" / "lesion_masks"

# Output folders
JSON_DIR = PROJECT_ROOT / "outputs" / "json"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
CSV_PATH = REPORT_DIR / "findings_reports.csv"


def main() -> None:
    """Run the complete Task 3 pipeline for one mock image."""

    print("Starting Task 3 pipeline...")

    # 1. Load the mock Task 2 probabilities
    if not MOCK_PATH.exists():
        raise FileNotFoundError(
            f"Mock prediction file was not found: {MOCK_PATH}"
        )

    with MOCK_PATH.open("r", encoding="utf-8") as file:
        mock_data = json.load(file)

    image_id = mock_data["image_id"]

    print(f"Processing {image_id}")

    # 2. Find and load the Task 1 lesion mask
    lesion_mask_path = LESION_MASK_DIR / f"{image_id}.png"

    if not lesion_mask_path.exists():
        raise FileNotFoundError(
            f"Lesion mask was not found: {lesion_mask_path}"
        )

    lesion_mask = load_binary_mask(lesion_mask_path)

    # 3. Calculate lesion size information
    area_ratio, size_category = calculate_size_category(
        lesion_mask
    )

    # 4. Calculate lesion-border information
    border_score, border_category = calculate_border_category(
        lesion_mask
    )

    print(f"Lesion area ratio: {area_ratio:.4f}")
    print(f"Size category: {size_category}")
    print(f"Border score: {border_score:.4f}")
    print(f"Border category: {border_category}")

    # 5. Generate the structured JSON record
    json_record = build_json_record(
        image_id=image_id,
        split=mock_data["split"],
        model_version=mock_data["model_version"],
        probabilities=mock_data["probabilities"],
    )

    json_path = JSON_DIR / f"{image_id}.json"

    save_json(
        record=json_record,
        output_path=json_path,
    )

    print("JSON saved successfully.")

    # 6. Generate the controlled findings report
    report_text = generate_findings_report(
        json_record=json_record,
        size_category=size_category,
        border_category=border_category,
    )

    report_path = REPORT_DIR / f"{image_id}.txt"

    save_report(
        report_text=report_text,
        output_path=report_path,
    )

    print("Report saved successfully.")
    print(report_text)

    # 7. Create one row for the combined CSV
    csv_row = create_csv_row(
        json_record=json_record,
        report_text=report_text,
        lesion_area_ratio=area_ratio,
        size_category=size_category,
        border_irregularity=border_score,
        border_category=border_category,
    )

    # 8. Save the CSV
    save_rows(
        rows=[csv_row],
        output_path=CSV_PATH,
    )

    print("CSV saved successfully.")

    # 9. Check JSON-to-report consistency
    errors = validate_json_and_report(
        json_record=json_record,
        report_text=report_text,
    )

    if errors:
        print("Validation failed:")

        for error in errors:
            print(f"- {error}")

        raise RuntimeError(
            "Task 3 stopped because output validation failed."
        )

    print("Validation passed.")
    print("Task 3 completed successfully.")


if __name__ == "__main__":
    main()