import json
from pathlib import Path
from typing import Any

from config import ATTRIBUTES, DISPLAY_NAMES


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

JSON_PATH = PROJECT_ROOT / "outputs" / "json" / "ISIC_000001.json"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"


def generate_findings_report(
    json_record: dict[str, Any],
    size_category: str = "moderate",
    border_category: str = "irregular",
) -> str:
    """Generate a controlled findings report from Task 3 JSON."""

    presence = json_record["outputs"]["presence"]
    findings = []

    for attribute in ATTRIBUTES:
        status = presence[attribute]["status"]
        display_name = DISPLAY_NAMES[attribute]

        findings.append(f"{display_name} is {status}")

    attribute_text = "; ".join(findings)

    return (
        f"The lesion is {size_category} in size with "
        f"{border_category} borders. "
        f"{attribute_text}."
    )


def save_report(report_text: str, output_path: Path) -> None:
    """Save a findings report as a text file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        file.write(report_text)


def main() -> None:
    """Generate one sample findings report."""

    with JSON_PATH.open("r", encoding="utf-8") as file:
        json_record = json.load(file)

    report_text = generate_findings_report(json_record)

    output_path = (
        REPORT_DIR / f"{json_record['image_id']}.txt"
    )

    save_report(report_text, output_path)

    print(report_text)
    print(f"Report saved successfully: {output_path}")


if __name__ == "__main__":
    main()