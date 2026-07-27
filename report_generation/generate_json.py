import json
from pathlib import Path
from typing import Any

from config import ATTRIBUTES
from status_mapping import probability_to_status


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MOCK_PATH = BASE_DIR / "mock_predictions.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "json"


def build_json_record(
    image_id: str,
    split: str,
    model_version: str,
    probabilities: dict[str, float],
) -> dict[str, Any]:
    """Build the structured Task 3 JSON record."""

    missing_attributes = [
        attribute
        for attribute in ATTRIBUTES
        if attribute not in probabilities
    ]

    if missing_attributes:
        raise ValueError(
            f"Missing attribute probabilities: {missing_attributes}"
        )

    presence = {}

    for attribute in ATTRIBUTES:
        probability = float(probabilities[attribute])

        presence[attribute] = {
            "prob": round(probability, 4),
            "status": probability_to_status(probability),
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
    output_path: Path,
) -> None:
    """Save a JSON record to a file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(record, file, indent=2)


def main() -> None:
    """Generate one Task 3 JSON file using mock predictions."""

    with MOCK_PATH.open("r", encoding="utf-8") as file:
        mock_data = json.load(file)

    record = build_json_record(
        image_id=mock_data["image_id"],
        split=mock_data["split"],
        model_version=mock_data["model_version"],
        probabilities=mock_data["probabilities"],
    )

    output_path = OUTPUT_DIR / f"{mock_data['image_id']}.json"

    save_json(record, output_path)

    print(f"JSON generated successfully: {output_path}")


if __name__ == "__main__":
    main()