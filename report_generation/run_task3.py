from lesion_features import (
    calculate_border_category,
    calculate_size_category,
    load_binary_mask,
)
import json
from pathlib import Path

from export_csv import create_csv_row, save_rows
from generate_json import build_json_record, save_json
from generate_report import generate_findings_report, save_report
from validate_output import validate_json_and_report


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

MOCK_PATH = BASE_DIR / "mock_predictions.json"

LESION_MASK_DIR = PROJECT_ROOT / "outputs" / "lesion_masks"
JSON_DIR = PROJECT_ROOT / "outputs" / "json"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
CSV_PATH = REPORT_DIR / "findings_reports.csv"


def main() -> None:
    """Run the complete Task 3 mock pipeline."""

    print("Starting Task 3 mock pipeline...")

    with MOCK_PATH.open("r", encoding="utf-8") as file:
        mock_data = json.load(file)

        image_id = mock_data["image_id"]
    lesion_mask_path = LESION_MASK_DIR / f"{image_id}.png"

    lesion_mask = load_binary_mask(lesion_mask_path)

    area_ratio, size_category = calculate_size_category(
        lesion_mask
    )

    border_score, border_category = calculate_border_category(
        lesion_mask
    )

    print(f"Processing {image_id}")

    json_record = build_json_record(
        image_id=image_id,
        split=mock_data["split"],
        model_version=mock_data["model_version"],
        probabilities=mock_data["probabilities"],
    )

    json_path = JSON_DIR / f"{image_id}.json"
    save_json(json_record, json_path)

    print("JSON saved successfully.")

    report_text = generate_findings_report(
        json_record=json_record,
        size_category=size_category,
        border_category=border_category,
    )

    report_path = REPORT_DIR / f"{image_id}.txt"
    save_report(report_text, report_path)

    print("Report saved successfully.")

    csv_row = create_csv_row(
        json_record=json_record,
        report_text=report_text,
    )

    save_rows([csv_row], CSV_PATH)

    print("CSV saved successfully.")

    errors = validate_json_and_report(
        json_record=json_record,
        report_text=report_text,
    )

    if errors:
        print("Validation failed:")

        for error in errors:
            print(f"- {error}")

        raise RuntimeError(
            "Task 3 stopped because validation failed."
        )

    print("Validation passed.")
    print("Task 3 completed successfully.")


if __name__ == "__main__":
    main()