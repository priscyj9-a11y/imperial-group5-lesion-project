"""Visualise Task 1 lesion segmentation results.

For each image, draws 4 panels side by side:
  1. the original photo
  2. the ground-truth mask (the real answer)
  3. the predicted mask (what the model guessed)
  4. an error map showing where they agree and disagree

Also prints the Dice score per image and the average across all of them.
Dice is the main Task 1 metric: it measures how much the prediction and
the ground truth overlap (1.0 = perfect, 0.0 = no overlap).
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def load_binary(path: Path) -> np.ndarray:
    """Open a mask and return it as clean 0s and 1s."""
    # convert("L") = greyscale, then >127 forces a clean yes/no per pixel
    return (np.array(Image.open(path).convert("L")) > 127).astype(np.uint8)


def dice_score(truth: np.ndarray, pred: np.ndarray) -> float:
    """How much the two masks overlap. 1.0 = identical, 0.0 = no overlap.

    Dice = 2 * (pixels both agree are lesion) / (truth lesion + pred lesion).
    This is THE standard metric for segmentation and it's on our scoring rubric.
    """
    # where both masks say "lesion" (both are 1) -> the overlap
    overlap = np.sum((truth == 1) & (pred == 1))
    # total lesion pixels in each mask
    total = np.sum(truth == 1) + np.sum(pred == 1)
    # if both masks are completely empty, call that a perfect match (1.0)
    if total == 0:
        return 1.0
    return 2.0 * overlap / total


def make_error_map(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Build a colour image showing agreement vs disagreement.

    green  = both agree it's lesion (correct)
    red    = model said lesion but it's NOT (false positive)
    blue   = model missed a real lesion pixel (false negative)
    black  = both agree it's background
    """
    # start with an all-black colour image (H, W, 3 for red/green/blue)
    h, w = truth.shape
    error = np.zeros((h, w, 3), dtype=np.uint8)

    # green where both are lesion
    error[(truth == 1) & (pred == 1)] = [0, 255, 0]
    # red where model over-predicted (said lesion, wasn't)
    error[(truth == 0) & (pred == 1)] = [255, 0, 0]
    # blue where model missed (was lesion, said no)
    error[(truth == 1) & (pred == 0)] = [0, 0, 255]

    return error


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise Task 1 results.")
    parser.add_argument("--image-dir", type=Path, required=True,
                        help="Folder of original images")
    parser.add_argument("--truth-dir", type=Path, required=True,
                        help="Folder of ground-truth masks")
    parser.add_argument("--pred-dir", type=Path, required=True,
                        help="Folder of predicted masks")
    parser.add_argument("--output", type=Path, default=Path("outputs/comparison.png"),
                        help="Where to save the comparison figure")
    parser.add_argument("--n", type=int, default=5,
                        help="How many images to show")
    args = parser.parse_args()

    # grab the first n images to display
    image_paths = sorted(args.image_dir.glob("*.png"))[:args.n]

    # set up a grid: one row per image, 4 columns
    fig, axes = plt.subplots(len(image_paths), 4, figsize=(16, 4 * len(image_paths)))
    # if there's only one image, wrap axes so the indexing below still works
    if len(image_paths) == 1:
        axes = axes.reshape(1, 4)

    dice_scores = []

    for row, image_path in enumerate(image_paths):
        image_id = image_path.stem

        # build the matching mask paths
        truth_path = args.truth_dir / f"{image_id}_segmentation.png"
        pred_path = args.pred_dir / f"{image_id}_predicted_mask.png"

        # load everything
        image = np.array(Image.open(image_path).convert("RGB"))
        truth = load_binary(truth_path)
        pred = load_binary(pred_path)

        # score this pair
        dice = dice_score(truth, pred)
        dice_scores.append(dice)

        # panel 1: original photo
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"{image_id} - original")

        # panel 2: ground truth (grey = mask, cmap makes 0/1 black/white)
        axes[row, 1].imshow(truth, cmap="gray")
        axes[row, 1].set_title("ground truth")

        # panel 3: prediction
        axes[row, 2].imshow(pred, cmap="gray")
        axes[row, 2].set_title("prediction")

        # panel 4: error map with the Dice score in the title
        axes[row, 3].imshow(make_error_map(truth, pred))
        axes[row, 3].set_title(f"error map - Dice {dice:.3f}")

        # turn off the axis ticks on all four, they're just clutter
        for col in range(4):
            axes[row, col].axis("off")

    # make sure the output folder exists, then save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=100)

    # print the scores to the terminal too
    print(f"\nSaved comparison to: {args.output}")
    print(f"Average Dice over {len(dice_scores)} images: {np.mean(dice_scores):.3f}")
    print("(green=correct, red=over-predicted, blue=missed)")


if __name__ == "__main__":
    main()