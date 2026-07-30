"""Visual checks for the CLIP retrieval bonus.

THE GOAL:
After clip_retrieval.py creates bonus/test_bonus_clip.csv, this script makes
one contact-sheet image per query/test image.

WHAT THE CONTACT SHEET SHOWS:
The first panel is the query/test image. The next panels are the top-k training
neighbours found by CLIP, with their similarity scores.

WHY IT MATTERS:
The CSV is the audit file, but pictures are easier to explain in the report and
presentation. If the neighbours look visually similar to the query image, the
retrieval result is easier to trust.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def find_image_by_id(folder: Path, image_id: str) -> Path:
    """Find one image file in a folder using its ID.

    Example:
        image_id = "000001"
        this function tries 000001.jpg, 000001.jpeg, and 000001.png.

    This is useful because the CSV usually stores IDs, while the folder stores
    real filenames with extensions.
    """
    for extension in IMAGE_EXTENSIONS:
        candidate = folder / f"{image_id}{extension}"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not find image {image_id} in {folder}")


def read_retrieval_csv(csv_path: Path) -> dict[str, list[tuple[str, float]]]:
    """Read the CLIP retrieval CSV and group neighbours by query image.

    Returns a dictionary like:
        {
            "test_001": [("data/images/000245.jpg", 0.89), ...],
            "test_002": [("data/images/000812.jpg", 0.86), ...],
        }
    """
    grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            query_id = row["query_image"]
            neighbor_id = row["neighbor_id"]
            similarity = float(row["similarity"])
            grouped[query_id].append((neighbor_id, similarity))

    return grouped


def load_panel_image(path: Path, size: int) -> Image.Image:
    """Load one image and resize it for the contact sheet.

    This resize is only for display. It does not change the original image and
    it is not used for model training or evaluation.
    """
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((size, size), Image.Resampling.LANCZOS)

        panel = Image.new("RGB", (size, size), "white")
        x = (size - image.width) // 2
        y = (size - image.height) // 2
        panel.paste(image, (x, y))
        return panel


def draw_label(panel: Image.Image, label: str) -> Image.Image:
    """Add a small label above one image panel."""
    label_height = 40
    labelled = Image.new("RGB", (panel.width, panel.height + label_height), "white")
    labelled.paste(panel, (0, label_height))

    draw = ImageDraw.Draw(labelled)
    font = ImageFont.load_default()
    draw.text((8, 12), label, fill="black", font=font)
    return labelled


def make_contact_sheet(
    query_path: Path,
    neighbor_paths_and_scores: list[tuple[Path, float]],
    output_path: Path,
    panel_size: int,
) -> None:
    """Create one image showing the query and its retrieved neighbours."""
    panels = [draw_label(load_panel_image(query_path, panel_size), "Query/test image")]

    for rank, (neighbor_path, score) in enumerate(neighbor_paths_and_scores, start=1):
        label = f"Neighbour #{rank}: {neighbor_path.name} | score={score:.3f}"
        panels.append(draw_label(load_panel_image(neighbor_path, panel_size), label))

    gap = 12
    width = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
    height = max(panel.height for panel in panels)
    sheet = Image.new("RGB", (width, height), "white")

    x = 0
    for panel in panels:
        sheet.paste(panel, (x, 0))
        x += panel.width + gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise CLIP retrieval neighbours.")
    parser.add_argument(
        "--retrieval-csv",
        type=Path,
        default=Path("bonus") / "test_bonus_clip.csv",
        help="CSV created by bonus_retrieval.clip_retrieval.",
    )
    parser.add_argument(
        "--query-dir",
        type=Path,
        required=True,
        help="Folder containing query/test images.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path(r"C:\Users\Ivana\Downloads\summer_school_project_train\train\images"),
        help="Folder containing training/reference images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "bonus_retrieval_visuals",
        help="Folder where contact-sheet images will be saved.",
    )
    parser.add_argument("--panel-size", type=int, default=224)
    parser.add_argument(
        "--max-query-images",
        type=int,
        default=10,
        help="Maximum number of query images to visualise.",
    )
    args = parser.parse_args()

    grouped = read_retrieval_csv(args.retrieval_csv)

    for count, (query_id, neighbours) in enumerate(grouped.items()):
        if count >= args.max_query_images:
            break

        query_path = find_image_by_id(args.query_dir, query_id)
        neighbour_paths_and_scores = []

        for neighbor_id, score in neighbours:
            neighbor_filename = Path(neighbor_id.replace("\\", "/")).name
            neighbor_path = args.index_dir / neighbor_filename
            neighbour_paths_and_scores.append((neighbor_path, score))

        output_path = args.output_dir / f"{query_id}_retrieval.png"
        make_contact_sheet(query_path, neighbour_paths_and_scores, output_path, args.panel_size)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
