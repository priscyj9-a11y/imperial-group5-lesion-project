"""Build a pre-resized cache of the dataset (run ONCE).

THE PROBLEM: our original images are up to 30 megapixels. Decoding them is
slow, and training re-decodes every image every epoch - hours per pass.
THE FIX: resize everything to 512x512 ONCE, save the small copies to disk,
and train from those. Decoding a 512px file is ~20x faster.

Run this once before training. Re-run only if you change --size.
Read-only on the original dataset - only writes to the cache folder.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

# reuse the EXACT same resize logic as the live pipeline, so cached images are
# identical to what load_image/load_mask would produce. no drift between them.
from transforms import resize_image, resize_mask

ATTRIBUTE_NAMES = [
    "pigment_network",
    "negative_network",
    "streaks",
    "milia_like_cyst",
    "globules",
]


def cache_one_image(src: Path, dst: Path, size: int) -> None:
    """Resize one colour photo and save it to the cache."""
    with Image.open(src) as img:
        # convert to RGB and resize with bilinear (same as the live loader)
        img = resize_image(img.convert("RGB"), size)
        # save as PNG so the cached image is lossless - no extra JPEG blurring
        img.save(dst)


def cache_one_mask(src: Path, dst: Path, size: int) -> None:
    """Resize one mask and save it to the cache."""
    with Image.open(src) as mask:
        # nearest-neighbour so the mask stays binary (same as live loader)
        mask = resize_mask(mask.convert("L"), size)
        mask.save(dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the pre-resized cache.")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Original dataset: images/, task1_gt/, task2_gt/")
    parser.add_argument("--cache-root", type=Path, required=True,
                        help="Where to write the resized copies")
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    # make the three cache subfolders, mirroring the original layout
    (args.cache_root / "images").mkdir(parents=True, exist_ok=True)
    (args.cache_root / "task1_gt").mkdir(parents=True, exist_ok=True)
    (args.cache_root / "task2_gt").mkdir(parents=True, exist_ok=True)

    # every image ID, from the images folder
    image_ids = sorted(p.stem for p in (args.data_root / "images").glob("*.jpg"))
    total = len(image_ids)

    for index, image_id in enumerate(image_ids, start=1):
        # --- colour photo: .jpg in, .png out ---
        cache_one_image(
            args.data_root / "images" / f"{image_id}.jpg",
            args.cache_root / "images" / f"{image_id}.png",
            args.size,
        )

        # --- lesion mask ---
        cache_one_mask(
            args.data_root / "task1_gt" / f"{image_id}_segmentation.png",
            args.cache_root / "task1_gt" / f"{image_id}_segmentation.png",
            args.size,
        )

        # --- five attribute masks ---
        for name in ATTRIBUTE_NAMES:
            fname = f"{image_id}_attribute_{name}.png"
            cache_one_mask(
                args.data_root / "task2_gt" / fname,
                args.cache_root / "task2_gt" / fname,
                args.size,
            )

        # progress every 200 images so we can see it's alive
        if index % 200 == 0 or index == total:
            print(f"  cached {index}/{total}")

    print(f"\nDone. Cache written to: {args.cache_root}")
    print("Point your training loader at this folder instead of the original.")


if __name__ == "__main__":
    main()