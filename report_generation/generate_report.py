"""Generate controlled Task 3 findings text."""

from pathlib import Path
from typing import Any

from .config import (
    ATTRIBUTES,
    ATTRIBUTE_VERBS,
    DISPLAY_NAMES,
)


def generate_findings_report(
    json_record: dict[str, Any],
    size_category: str,
    border_category: str,
) -> str:
    """Generate report text directly from the structured JSON."""

    presence = json_record["outputs"]["presence"]

    attribute_findings = []

    for attribute in ATTRIBUTES:
        status = presence[attribute]["status"]
        display_name = DISPLAY_NAMES[attribute]
        verb = ATTRIBUTE_VERBS[attribute]

        attribute_findings.append(
            f"{display_name} {verb} {status}"
        )

    attribute_text = "; ".join(
        attribute_findings
    )

    if (
        size_category == "undetermined"
        and border_category == "undetermined"
    ):
        lesion_description = (
            "Lesion size and border characteristics could not be "
            "determined because the Task 1 prediction mask was empty."
        )

    elif size_category == "undetermined":
        lesion_description = (
            "Lesion size could not be determined. "
            f"The predicted border is {border_category}."
        )

    elif border_category == "undetermined":
        lesion_description = (
            f"The predicted lesion is {size_category} in size. "
            "Border characteristics could not be determined."
        )

    else:
        lesion_description = (
            f"The predicted lesion is {size_category} in size "
            f"with {border_category} borders."
        )

    return (
        f"{lesion_description} "
        f"{attribute_text}."
    )


def save_report(
    report_text: str,
    output_path: str | Path,
) -> None:
    """Save one findings report."""

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        report_text,
        encoding="utf-8",
    )