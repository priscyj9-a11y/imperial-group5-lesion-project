"""Visualise one image and its five Task 2 ground-truth masks.

THE PROBLEM:
Before evaluating a model, we need to understand what the true masks look like.
Some masks are all black because the attribute is absent.

WHY IT MATTERS:
i must explain model results per attribute. Visual checks make it clear
which attributes are present, absent, small, sparse, or difficult.

WHERE IT FITS:
Run this script first. It creates a contact sheet:
    original image + five ground-truth attribute masks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from preprocessing.transforms import load_image, load_mask
from attribute_detection_v3.task2_eval.mask_io import ATTRIBUTE_NAMES, ground_truth_mask_path


def _font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", 16)
    except OSError:
        return ImageFont.load_default()


def add_title(image: Image.Image, title: str) -> Image.Image:
    """Add a small title above a tile."""
    title_height = 34
    canvas = Image.new("RGB", (image.width, image.height + title_height), "white")
    canvas.paste(image, (0, title_height))

    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill="black", font=_font())
    return canvas


def mask_to_tile(mask: np.ndarray, title: str) -> Image.Image:
    """Convert a 0/1 mask into a labelled black/white tile."""
    image = Image.fromarray((mask * 255).astype(np.uint8), mode="L").convert("RGB")
    return add_title(image, title)


def save_ground_truth_contact_sheet(
    data_root: Path,
    image_id: str,
    output_path: Path,
) -> None:
    """Save the original photo and five Task 2 masks in one PNG.

    It uses preprocessing.transforms so visual checks match the same 512x512
    format used by the model and evaluation code.
    """
    image_path = data_root / "images" / f"{image_id}.jpg"
    photo_array = load_image(image_path)
    photo = Image.fromarray((photo_array * 255).astype(np.uint8), mode="RGB")
    photo_tile = add_title(photo, f"image {image_id}")

    tiles = [photo_tile]
    for attribute in ATTRIBUTE_NAMES:
        mask = load_mask(ground_truth_mask_path(data_root, image_id, attribute))
        status = "present" if mask.max() == 1 else "absent"
        tiles.append(mask_to_tile(mask, f"{attribute} ({status})"))

    columns = 3
    rows = 2
    tile_width = max(tile.width for tile in tiles)
    tile_height = max(tile.height for tile in tiles)
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")

    for index, tile in enumerate(tiles):
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        sheet.paste(tile, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise Task 2 ground-truth masks.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"C:\Users\Ivana\Downloads\summer_school_project_train\train"),
        help="Folder containing images/ and task2_gt/.",
    )
    parser.add_argument("--image-id", default="000001")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / "task2_evaluation" / "ground_truth_masks.png",
    )
    args = parser.parse_args()

    save_ground_truth_contact_sheet(args.data_root, args.image_id, args.output)
    print(f"Saved visualisation: {args.output}")


if __name__ == "__main__":
    main()
