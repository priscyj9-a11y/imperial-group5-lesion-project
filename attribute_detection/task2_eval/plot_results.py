"""Create report charts from Task 2 validation scores.

THE PROBLEM:
evaluate_validation.py writes a detailed CSV with one row per validation image
and per attribute. That is useful, but too detailed for a report or presentation.

WHY IT MATTERS:
Person B needs clear figures showing which attributes are easy or difficult.
Charts make the class-imbalance and per-attribute performance easier to explain.

WHERE IT FITS:
Run this after evaluate_validation.py has created validation_scores.csv.

This file intentionally uses only Python's csv module plus Pillow. That avoids
requiring matplotlib/pandas in the Summerschool environment.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from attribute_detection.task2_eval.mask_io import ATTRIBUTE_NAMES


METRIC_NAMES = ["dice", "iou", "precision", "recall"]


def read_ok_rows(scores_csv: Path) -> list[dict[str, str]]:
    """Read validation_scores.csv and keep only rows with calculated metrics.

    Rows marked missing_prediction are skipped because they do not contain
    numeric Dice/IoU/precision/recall values.
    """
    with scores_csv.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return [row for row in reader if row.get("status") == "ok"]


def mean_by_attribute(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    """Calculate mean Dice/IoU/precision/recall for each attribute."""
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["attribute"]].append(row)

    summary: dict[str, dict[str, float]] = {}
    for attribute in ATTRIBUTE_NAMES:
        attr_rows = grouped.get(attribute, [])
        if not attr_rows:
            summary[attribute] = {metric: 0.0 for metric in METRIC_NAMES}
            continue

        summary[attribute] = {}
        for metric in METRIC_NAMES:
            values = [float(row[metric]) for row in attr_rows]
            summary[attribute][metric] = sum(values) / len(values)

    return summary


def _font(size: int = 16) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_bar_chart(
    values: dict[str, float],
    title: str,
    output_path: Path,
    y_label: str = "score",
) -> None:
    """Draw a simple 0-1 bar chart and save it as PNG."""
    width = 980
    height = 560
    margin_left = 170
    margin_right = 40
    margin_top = 80
    margin_bottom = 110
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(24)
    label_font = _font(15)
    small_font = _font(13)

    draw.text((margin_left, 25), title, fill="black", font=title_font)

    # Axes
    x0 = margin_left
    y0 = margin_top + chart_height
    draw.line((x0, margin_top, x0, y0), fill="black", width=2)
    draw.line((x0, y0, x0 + chart_width, y0), fill="black", width=2)
    draw.text((25, margin_top + 10), y_label, fill="black", font=label_font)

    # Horizontal guide lines from 0.0 to 1.0.
    for tick in range(0, 6):
        score = tick / 5
        y = y0 - int(score * chart_height)
        draw.line((x0 - 5, y, x0 + chart_width, y), fill=(220, 220, 220), width=1)
        draw.text((x0 - 48, y - 8), f"{score:.1f}", fill="black", font=small_font)

    bar_count = len(ATTRIBUTE_NAMES)
    gap = 30
    bar_width = int((chart_width - gap * (bar_count + 1)) / bar_count)
    colors = [
        (68, 114, 196),
        (112, 173, 71),
        (237, 125, 49),
        (165, 105, 189),
        (91, 155, 213),
    ]

    for index, attribute in enumerate(ATTRIBUTE_NAMES):
        value = values.get(attribute, 0.0)
        bar_height = int(value * chart_height)
        x1 = x0 + gap + index * (bar_width + gap)
        y1 = y0 - bar_height
        x2 = x1 + bar_width
        draw.rectangle((x1, y1, x2, y0), fill=colors[index])
        draw.text((x1, y1 - 22), f"{value:.3f}", fill="black", font=small_font)

        # Split long names over two lines to keep labels readable.
        label = attribute.replace("_", "\n")
        draw.multiline_text((x1, y0 + 12), label, fill="black", font=small_font, spacing=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def write_summary_csv(summary: dict[str, dict[str, float]], output_csv: Path) -> None:
    """Save one row per attribute with mean metric values."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["attribute", *METRIC_NAMES])
        for attribute in ATTRIBUTE_NAMES:
            writer.writerow([attribute, *[summary[attribute][metric] for metric in METRIC_NAMES]])


def main() -> None:
    parser = argparse.ArgumentParser(description="Create charts from Task 2 validation scores.")
    parser.add_argument(
        "--scores-csv",
        type=Path,
        default=Path("outputs") / "task2_evaluation" / "validation_scores.csv",
        help="CSV produced by evaluate_validation.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "task2_evaluation" / "charts",
        help="Folder where chart PNGs and summary CSV will be saved.",
    )
    args = parser.parse_args()

    rows = read_ok_rows(args.scores_csv)
    if not rows:
        raise ValueError(f"No usable rows found in {args.scores_csv}. Run evaluation first.")

    summary = mean_by_attribute(rows)
    write_summary_csv(summary, args.output_dir / "attribute_metric_summary.csv")

    for metric in METRIC_NAMES:
        draw_bar_chart(
            {attribute: summary[attribute][metric] for attribute in ATTRIBUTE_NAMES},
            title=f"Mean {metric.title()} by Attribute",
            output_path=args.output_dir / f"mean_{metric}_by_attribute.png",
            y_label=metric,
        )

    print(f"Charts written to: {args.output_dir}")


if __name__ == "__main__":
    main()
