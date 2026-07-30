"""Create and save Task 3 structured JSON records."""

import json
from pathlib import Path
from typing import Any

from .config import ATTRIBUTES


VALID_STATUSES = {
    "present",
    "absent",
    "uncertain",
}


def build_json_record(
    image_id: str,
    split: str,
    model_version: str,
    probabilities: dict[str, float],
    statuses: dict[str, str],
) -> dict[str, Any]:
    """Build one strict Task 3 JSON record."""

    missing_probabilities = [
        attribute
        for attribute in ATTRIBUTES
        if attribute not in probabilities
    ]

    missing_statuses = [
        attribute
        for attribute in ATTRIBUTES
        if attribute not in statuses
    ]

    if missing_probabilities:
        raise ValueError(
            f"Missing probabilities: {missing_probabilities}"
        )

    if missing_statuses:
        raise ValueError(
            f"Missing statuses: {missing_statuses}"
        )

    presence: dict[str, dict[str, float | str]] = {}

    for attribute in ATTRIBUTES:
        probability = float(
            probabilities[attribute]
        )

        status = statuses[attribute]

        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"{attribute} probability is outside 0 to 1."
            )

        if status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid status for {attribute}: {status}"
            )

        presence[attribute] = {
            "prob": round(probability, 4),
            "status": status,
        }

    return {
        "image_id": image_id,
        "split": split,
        "model_version": model_version,
        "attributes_order": ATTRIBUTES,
        "outputs": {
            "presence": presence,
        },
    }


def save_json(
    record: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Save one Task 3 JSON record."""

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            record,
            file,
            indent=2,
        )