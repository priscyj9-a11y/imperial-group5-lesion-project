"""Create visual error maps for Task 2 predictions.

THE PROBLEM:
Metrics give numbers, but they do not show where the model is wrong.

WHY IT MATTERS:
For the report/presentation, Person B should show examples of correct overlap,
false positives, and false negatives. This makes the evaluation easier to
understand visually.

WHERE IT FITS:
Run this after Rita exports predicted masks. It creates one PNG for a chosen
image ID and attribute.

COLOUR MEANING:
Green = correct overlap: prediction and ground truth both mark the attribute.
Red   = false positive: model predicts the attribute where GT is black.
Blue  = false negative: GT has the attribute but the model missed it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from preprocessing.transforms import load_image, load_mask
from attribute_detection.task2_eval.mask_io import ATTRIBUTE_NAMES, ground_truth_mask_path, prediction_mask_path


def _font(size: int = 16) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def add_title(image: Image.Image, title: str) -> Image.Image:
    """Add a title above one visual tile."""
    title_height = 34
    canvas = Image.new("RGB", (image.width, image.height + title_height), "white")
    canvas.paste(image, (0, title_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill="black", font=_font())
    return canvas


def mask_to_rgb(mask: np.ndarray) -> Image.Image:
    """Convert a 0/1 mask to black/white RGB for display."""
    return Image.fromarray((mask * 255).astype(np.uint8), mode="L").convert("RGB")


def make_error_map(prediction: np.ndarray, ground_truth: np.ndarray) -> Image.Image:
    """Create a colour error map comparing prediction and ground truth."""
    pred = prediction.astype(bool)
    gt = ground_truth.astype(bool)

    correct_overlap = np.logical_and(pred, gt)
    false_positive = np.logical_and(pred, np.logical_not(gt))
    false_negative = np.logical_and(np.logical_not(pred), gt)

    error = np.zeros((*prediction.shape, 3), dtype=np.uint8)
    error[correct_overlap] = [0, 180, 0]
    error[false_positive] = [220, 0, 0]
    error[false_negative] = [0, 90, 220]
    return Image.fromarray(error, mode="RGB")


def save_prediction_visualisation(
    data_root: Path,
    prediction_root: Path,
    image_id: str,
    attribute: str,
    output_path: Path,
) -> None:
    """Save original image, GT mask, prediction, and error map in one PNG."""
    if attribute not in ATTRIBUTE_NAMES:
        raise ValueError(f"Unknown attribute '{attribute}'. Expected one of: {ATTRIBUTE_NAMES}")

    image_path = data_root / "images" / f"{image_id}.jpg"
    gt_path = ground_truth_mask_path(data_root, image_id, attribute)
    pred_path = prediction_mask_path(prediction_root, image_id, attribute)

    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction not found: {pred_path}")

    # Use the shared preprocessing so all visuals match the 512x512 model format.
    image_array = load_image(image_path)
    ground_truth = load_mask(gt_path)
    prediction = load_mask(pred_path)

    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"Shape mismatch for {image_id} {attribute}: "
            f"prediction {prediction.shape}, ground truth {ground_truth.shape}"
        )

    image_tile = add_title(Image.fromarray((image_array * 255).astype(np.uint8), mode="RGB"), "image")
    gt_tile = add_title(mask_to_rgb(ground_truth), "ground truth")
    pred_tile = add_title(mask_to_rgb(prediction), "prediction")
    error_tile = add_title(make_error_map(prediction, ground_truth), "error map")

    tiles = [image_tile, gt_tile, pred_tile, error_tile]
    tile_width, tile_height = tiles[0].size
    sheet = Image.new("RGB", (tile_width * len(tiles), tile_height), "white")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, (index * tile_width, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise one Task 2 prediction against ground truth.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"C:\Users\Ivana\Downloads\summer_school_project_train\train"),
        help="Folder containing images/ and task2_gt/.",
    )
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--attribute", choices=ATTRIBUTE_NAMES, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / "task2_evaluation" / "prediction_visualisation.png",
    )
    args = parser.parse_args()

    save_prediction_visualisation(
        data_root=args.data_root,
        prediction_root=args.prediction_root,
        image_id=args.image_id,
        attribute=args.attribute,
        output_path=args.output,
    )
    print(f"Saved prediction visualisation: {args.output}")


if __name__ == "__main__":
    main()
