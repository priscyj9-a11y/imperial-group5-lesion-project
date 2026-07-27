import json
from pathlib import Path
from typing import Any

import pandas as pd

from config import ATTRIBUTES


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

JSON_PATH = PROJECT_ROOT / "outputs" / "json" / "ISIC_000001.json"
REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "ISIC_000001.txt"
CSV_PATH = PROJECT_ROOT / "outputs" / "reports" / "findings_reports.csv"


def create_csv_row(
    json_record: dict[str, Any],
    report_text: str,
) -> dict[str, Any]:
    """Create one CSV row from a JSON record and findings report."""

    row = {
        "image_id": json_record["image_id"],
        "split": json_record["split"],
        "model_version": json_record["model_version"],
        "findings_report": report_text,
    }

    presence = json_record["outputs"]["presence"]

    for attribute in ATTRIBUTES:
        row[f"{attribute}_probability"] = presence[attribute]["prob"]
        row[f"{attribute}_status"] = presence[attribute]["status"]

    return row


def save_rows(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Save Task 3 results as a combined CSV file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(output_path, index=False)


def main() -> None:
    """Create a sample combined findings CSV."""

    with JSON_PATH.open("r", encoding="utf-8") as file:
        json_record = json.load(file)

    with REPORT_PATH.open("r", encoding="utf-8") as file:
        report_text = file.read().strip()

    row = create_csv_row(json_record, report_text)
    save_rows([row], CSV_PATH)

    print(f"CSV generated successfully: {CSV_PATH}")


if __name__ == "__main__":
    main()