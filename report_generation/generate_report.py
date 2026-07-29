from pathlib import Path
from typing import Any

from config import (
    ATTRIBUTES,
    ATTRIBUTE_VERBS,
    DISPLAY_NAMES,
)


def generate_findings_report(
    json_record: dict[str, Any],
    size_category: str,
    border_category: str,
) -> str:
    """Generate controlled report text directly from the JSON."""

    presence = json_record["outputs"]["presence"]

    findings = []

    for attribute in ATTRIBUTES:
        status = presence[attribute]["status"]
        display_name = DISPLAY_NAMES[attribute]
        verb = ATTRIBUTE_VERBS[attribute]

        findings.append(
            f"{display_name} {verb} {status}"
        )

    attribute_text = "; ".join(findings)

    return (
        f"The lesion is {size_category} in size with "
        f"{border_category} borders. "
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

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(report_text)