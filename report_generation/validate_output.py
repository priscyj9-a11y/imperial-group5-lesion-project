"""Validate Task 3 JSON and findings-report consistency."""

import re
from typing import Any

from .config import (
    ATTRIBUTES,
    ATTRIBUTE_VERBS,
    DATASET_SPLIT,
    DISPLAY_NAMES,
)


VALID_STATUSES = {
    "present",
    "absent",
    "uncertain",
}


UNSUPPORTED_DIAGNOSTIC_TERMS = {
    "melanoma",
    "malignant",
    "benign",
    "cancerous",
    "diagnosis",
}


def validate_json_and_report(
    json_record: dict[str, Any],
    report_text: str,
    expected_statuses: dict[str, str] | None = None,
) -> list[str]:
    """Validate one JSON record and its generated report.

    Returns:
        A list of errors. An empty list means validation passed.
    """

    errors: list[str] = []

    # ------------------------------------------------------------------
    # Required top-level JSON structure
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Image ID validation
    # ------------------------------------------------------------------

    image_id = str(
        json_record["image_id"]
    )

    if re.fullmatch(r"\d{6}", image_id) is None:
        errors.append(
            f"Image ID must contain exactly six digits: {image_id}"
        )

    # ------------------------------------------------------------------
    # Dataset split validation
    # ------------------------------------------------------------------

    split = str(
        json_record["split"]
    )

    if split != DATASET_SPLIT:
        errors.append(
            f"Expected split '{DATASET_SPLIT}', "
            f"received '{split}'."
        )

    # ------------------------------------------------------------------
    # Model-version validation
    # ------------------------------------------------------------------

    model_version = str(
        json_record["model_version"]
    )

    if not model_version.strip():
        errors.append(
            "Model version is empty."
        )

    if "mock" in model_version.lower():
        errors.append(
            "The final JSON still contains a mock model version."
        )

    # ------------------------------------------------------------------
    # Attribute-order validation
    # ------------------------------------------------------------------

    if json_record["attributes_order"] != ATTRIBUTES:
        errors.append(
            "The JSON attribute order is incorrect."
        )

    # ------------------------------------------------------------------
    # Presence-output validation
    # ------------------------------------------------------------------

    outputs = json_record.get(
        "outputs",
        {},
    )

    if not isinstance(outputs, dict):
        errors.append(
            "The outputs field must be an object."
        )

        return errors

    presence = outputs.get(
        "presence",
        {},
    )

    if not isinstance(presence, dict):
        errors.append(
            "The outputs.presence field must be an object."
        )

        return errors

    missing_attributes = [
        attribute
        for attribute in ATTRIBUTES
        if attribute not in presence
    ]

    if missing_attributes:
        errors.append(
            f"Missing attributes: {missing_attributes}"
        )

    extra_attributes = [
        attribute
        for attribute in presence
        if attribute not in ATTRIBUTES
    ]

    if extra_attributes:
        errors.append(
            f"Unexpected attributes: {extra_attributes}"
        )

    # ------------------------------------------------------------------
    # Report validation
    # ------------------------------------------------------------------

    if not isinstance(report_text, str):
        errors.append(
            "The findings report must be text."
        )

        return errors

    if not report_text.strip():
        errors.append(
            "The findings report is empty."
        )

        return errors

    report_lower = report_text.lower()

    # ------------------------------------------------------------------
    # Per-attribute validation
    # ------------------------------------------------------------------

    for attribute in ATTRIBUTES:
        if attribute not in presence:
            continue

        evidence = presence[attribute]

        if not isinstance(evidence, dict):
            errors.append(
                f"{attribute} evidence must be an object."
            )
            continue

        probability = evidence.get(
            "prob"
        )

        status = evidence.get(
            "status"
        )

        # Probability must be numerical and within 0 to 1.
        if not isinstance(
            probability,
            (int, float),
        ):
            errors.append(
                f"{attribute} does not have a numeric probability."
            )

        elif not 0.0 <= float(probability) <= 1.0:
            errors.append(
                f"{attribute} probability is outside 0 to 1."
            )

        # Status must use one of the controlled terms.
        if status not in VALID_STATUSES:
            errors.append(
                f"Invalid status for {attribute}: {status}"
            )

        # Check that JSON status matches the mask-derived status.
        if expected_statuses is not None:
            expected_status = expected_statuses.get(
                attribute
            )

            if expected_status is None:
                errors.append(
                    f"No expected status was supplied for {attribute}."
                )

            elif status != expected_status:
                errors.append(
                    f"{attribute} JSON status '{status}' "
                    f"does not match mask-derived status "
                    f"'{expected_status}'."
                )

        # Check that the attribute name appears in the report.
        display_name = (
            DISPLAY_NAMES[attribute]
            .lower()
        )

        if display_name not in report_lower:
            errors.append(
                f"The report is missing '{display_name}'."
            )

        # Check that the report wording matches the JSON status.
        verb = ATTRIBUTE_VERBS[attribute]

        expected_phrase = (
            f"{display_name} {verb} {status}"
        )

        if expected_phrase not in report_lower:
            errors.append(
                f"The report does not match the JSON "
                f"status for {attribute}. "
                f"Expected phrase: '{expected_phrase}'."
            )

    # ------------------------------------------------------------------
    # Unsupported clinical-language validation
    # ------------------------------------------------------------------

    for term in UNSUPPORTED_DIAGNOSTIC_TERMS:
        if term in report_lower:
            errors.append(
                f"Unsupported diagnostic term found: '{term}'."
            )

    return errors