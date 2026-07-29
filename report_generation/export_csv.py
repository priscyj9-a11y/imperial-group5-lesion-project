from pathlib import Path
from typing import Any

import pandas as pd

from config import ATTRIBUTES


def create_csv_row(
    json_record: dict[str, Any],
    report_text: str,
    lesion_area_ratio: float,
    size_category: str,
    border_irregularity: float,
    border_category: str,
) -> dict[str, Any]:
    """Create one row for the combined findings CSV."""

    row = {
        "image_id": json_record["image_id"],
        "split": json_record["split"],
        "model_version": json_record["model_version"],
        "lesion_area_ratio": lesion_area_ratio,
        "size_category": size_category,
        "border_irregularity": border_irregularity,
        "border_category": border_category,
        "findings_report": report_text,
    }

    presence = json_record["outputs"]["presence"]

    for attribute in ATTRIBUTES:
        row[f"{attribute}_probability"] = (
            presence[attribute]["prob"]
        )

        row[f"{attribute}_status"] = (
            presence[attribute]["status"]
        )

    return row


def save_rows(
    rows: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """Save all Task 3 rows into one combined CSV."""

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.DataFrame(rows)

    dataframe.to_csv(
        output_path,
        index=False,
    )