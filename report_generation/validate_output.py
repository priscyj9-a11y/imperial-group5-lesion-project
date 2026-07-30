"""Validate Task 3 JSON and findings-report consistency."""

import re
from typing import Any

from .config import (
    ATTRIBUTES,
    ATTRIBUTE_VERBS,
    DISPLAY_NAMES,
)


VALID_STATUSES = {
    "present",
    "absent",
    "uncertain",
}


def validate_json_and_report(
    json_record: dict[str, Any],
    report_text: str,
    expected_statuses: dict[str, str] | None = None,
) -> list[str]:
    """Return a list of validation errors.

    An empty list means the JSON and report passed validation.
    """

    errors: list[str] = []

    required_fields = {
        "image_id",
        "split",
        "model_version",
        "attributes_order",
        "outputs",
    }

    missing_fields = (
        required_fields
        - set(json_record.keys())
    )

    if missing_fields:
        errors.append(
            f"Missing JSON fields: {sorted(missing_fields)}"
        )

        return errors

    image_id = str(json_record["image_id"])

    if re.fullmatch(r"\d{6}", image_id) is None:
        errors.append(
            f"Image ID must contain six digits: {image_id}"
        )

    if json_record["split"] != "val":
        errors.append(
            f"Expected split 'val', received: {json_record['split']}"
        )

    if "mock" in str(json_record["model_version"]).lower():
        errors.append(
            "The final JSON still contains a mock model version."
        )

    if json_record["attributes_order"] != ATTRIBUTES:
        errors.append(
            "The JSON attribute order is incorrect."
        )

    presence = (
        json_record
        .get("outputs", {})
        .get("presence", {})
    )

    report_lower = report_text.lower()

    for attribute in ATTRIBUTES:
        if attribute not in presence:
            errors.append(
                f"Missing attribute: {attribute}"
            )
            continue

        probability = presence[attribute].get("prob")
        status = presence[attribute].get("status")

        if not isinstance(
            probability,
            (int, float),
        ):
            errors.append(
                f"{attribute} does not have a numeric probability."
            )

        elif not 0.0 <= probability <= 1.0:
            errors.append(
                f"{attribute} probability is outside 0 to 1."
            )

        if status not in VALID_STATUSES:
            errors.append(
                f"Invalid status for {attribute}: {status}"
            )

        if (
            expected_statuses is not None
            and status != expected_statuses.get(attribute)
        ):
            errors.append(
                f"{attribute} JSON status does not match "
                "the status derived from its prediction mask."
            )

        display_name = DISPLAY_NAMES[attribute].lower()
        verb = ATTRIBUTE_VERBS[attribute]

        if display_name not in report_lower:
            errors.append(
                f"The report is missing '{display_name}'."
            )

        expected_phrase = (
            f"{display_name} {verb} {status}"
        )

        if expected_phrase not in report_lower:
            errors.append(
                f"The report does not match the JSON "
                f"status for {attribute}."
            )

    unsupported_terms = {
        "melanoma",
        "malignant",
        "benign",
        "cancerous",
        "diagnosis",
    }

    for term in unsupported_terms:
        if term in report_lower:
            errors.append(
                f"Unsupported diagnostic term found: {term}"
            )

    return errors