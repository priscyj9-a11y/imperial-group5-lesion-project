from typing import Any

from config import (
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
    """Check structure, terminology and JSON-to-report consistency."""

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
        - json_record.keys()
    )

    if missing_fields:
        errors.append(
            f"Missing JSON fields: {sorted(missing_fields)}"
        )

        return errors

    if json_record["attributes_order"] != ATTRIBUTES:
        errors.append(
            "The JSON attribute order is incorrect."
        )

    presence = (
        json_record
        .get("outputs", {})
        .get("presence", {})
    )

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

        if display_name not in report_text.lower():
            errors.append(
                f"The report is missing '{display_name}'."
            )

        expected_phrase = (
            f"{display_name} {verb} {status}"
        )

        if expected_phrase not in report_text.lower():
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
        if term in report_text.lower():
            errors.append(
                f"Unsupported diagnostic term found: {term}"
            )

    return errors