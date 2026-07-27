import json
from pathlib import Path
from typing import Any

from config import ATTRIBUTES, ATTRIBUTE_VERBS, DISPLAY_NAMES
from status_mapping import probability_to_status


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

JSON_PATH = PROJECT_ROOT / "outputs" / "json" / "ISIC_000001.json"
REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "ISIC_000001.txt"


def validate_json_and_report(
    json_record: dict[str, Any],
    report_text: str,
) -> list[str]:
    """Check JSON structure and JSON-to-report consistency."""

    errors = []

    required_fields = {
        "image_id",
        "split",
        "model_version",
        "attributes_order",
        "outputs",
    }

    missing_fields = required_fields - json_record.keys()

    if missing_fields:
        errors.append(
            f"Missing top-level fields: {sorted(missing_fields)}"
        )
        return errors

    if json_record["attributes_order"] != ATTRIBUTES:
        errors.append("The attribute order is incorrect.")

    presence = json_record.get("outputs", {}).get("presence", {})

    for attribute in ATTRIBUTES:
        if attribute not in presence:
            errors.append(f"Missing attribute: {attribute}")
            continue

        result = presence[attribute]
        probability = result.get("prob")
        status = result.get("status")

        if not isinstance(probability, (int, float)):
            errors.append(
                f"{attribute} does not have a valid probability."
            )
            continue

        if not 0.0 <= probability <= 1.0:
            errors.append(
                f"{attribute} probability is outside the range 0 to 1."
            )
            continue

        expected_status = probability_to_status(float(probability))

        if status != expected_status:
            errors.append(
                f"{attribute} status '{status}' does not match "
                f"probability {probability}."
            )

        display_name = DISPLAY_NAMES[attribute].lower()
        verb = ATTRIBUTE_VERBS[attribute]

        if display_name not in report_text.lower():
            errors.append(
                f"The report is missing the term '{display_name}'."
            )

        expected_phrase = (
            f"{display_name} {verb} {status}"
        )

        if expected_phrase not in report_text.lower():
            errors.append(
                f"The written report does not match the JSON "
                f"for {attribute}."
            )

    unsupported_terms = [
        "melanoma",
        "malignant",
        "benign",
        "diagnosis",
        "cancerous",
    ]

    for term in unsupported_terms:
        if term in report_text.lower():
            errors.append(
                f"Unsupported diagnostic term found: '{term}'."
            )

    return errors


def main() -> None:
    """Validate the sample JSON and report."""

    with JSON_PATH.open("r", encoding="utf-8") as file:
        json_record = json.load(file)

    with REPORT_PATH.open("r", encoding="utf-8") as file:
        report_text = file.read().strip()

    errors = validate_json_and_report(
        json_record,
        report_text,
    )

    if errors:
        print("Validation failed:")

        for error in errors:
            print(f"- {error}")
    else:
        print("Validation passed.")


if __name__ == "__main__":
    main()