"""Count how often each of the five attributes actually appears.

An absent attribute is stored as an all-black mask, so "appears" means
"this mask has at least one white pixel".

Read-only on the dataset - only writes a small CSV.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


# spelled exactly as our FILES spell them (note: cyst is singular, the
# briefing's JSON schema uses the plural - don't let this bite you)
ATTRIBUTE_NAMES = [
    "pigment_network",
    "negative_network",
    "streaks",
    "milia_like_cyst",
    "globules",
]


def attribute_is_present(mask_path: Path) -> bool:
    """True if this attribute appears in this image.

    THE PROBLEM: we need to know whether an attribute is in an image, but
    every image has all five mask files whether the attribute is there or not.
    WHY IT MATTERS: a mask file existing does NOT mean the attribute exists.
    We confirmed earlier that absent attributes are stored as all-black masks,
    not missing files. So we have to look inside, not just count files.
    WHERE IT FITS: called once per attribute per image (13,500 times total).

    HOW WE TELL: if the brightest pixel in the whole mask is 0, the mask is
    entirely black and the attribute is absent. Anything brighter = present.
    """
    with Image.open(mask_path) as mask:
        # convert("L") = single greyscale channel, no colour.
        # np.array turns the picture into a grid of numbers.
        pixels = np.array(mask.convert("L"))
    # .max() finds the brightest pixel anywhere in the mask
    return bool(pixels.max() > 0)


def count_attributes(root: Path) -> tuple[dict[str, int], int]:
    """Go through every image and count how many have each attribute.

    THE PROBLEM: Task 2 has to detect five attributes, but we don't know how
    common each one is.
    WHY IT MATTERS: this is the class-imbalance problem. If streaks only
    appear in 6% of images, a model can score 94% by always predicting
    "absent" - looking great while having learned nothing. Ivana and Rita
    need these numbers to plan for it (weighted loss, thresholds, etc).
    WHERE IT FITS: the last dataset fact the team is missing.
    """
    images_dir = root / "images"
    if not images_dir.is_dir():
        # fail loudly on a typo'd path instead of silently reporting 0 images
        raise FileNotFoundError(f"Folder not found: {images_dir}")

    # .stem = filename without extension: '000001.jpg' -> '000001'
    # sorted() so the run is identical every time
    image_ids = sorted(p.stem for p in images_dir.glob("*.jpg"))

    # start every attribute's count at zero
    counts = {name: 0 for name in ATTRIBUTE_NAMES}

    for image_id in image_ids:
        for name in ATTRIBUTE_NAMES:
            mask_path = root / "task2_gt" / f"{image_id}_attribute_{name}.png"
            # try/except: if ONE file is corrupted we report it and keep going,
            # rather than losing the whole run 12,000 files in
            try:
                if attribute_is_present(mask_path):
                    counts[name] += 1
            except Exception as error:
                print(f"  could not read {mask_path.name}: {error}")

    # return the counts AND how many images we looked at, so we can do percentages
    return counts, len(image_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Count attribute frequencies.")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Folder containing images/, task1_gt/, task2_gt/")
    parser.add_argument("--output", type=Path,
                        default=Path("data/attribute_frequency.csv"),
                        help="Where to write the frequency table")
    args = parser.parse_args()

    counts, total = count_attributes(args.data_root)

    # make the output folder if it doesn't exist yet.
    # exist_ok=True means "don't error if it's already there"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # WHY A CSV: the team can open it in Excel, and it goes in the report.
    # newline="" stops Windows adding blank rows between lines.
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["attribute", "present", "absent", "percent_present"])
        for name in ATTRIBUTE_NAMES:
            present = counts[name]
            absent = total - present
            percent = round(100 * present / total, 1)
            writer.writerow([name, present, absent, percent])

    # also print it so we can read the result immediately
    print(f"Total images: {total}\n")
    for name in ATTRIBUTE_NAMES:
        present = counts[name]
        percent = round(100 * present / total, 1)
        # :<18 pads the name to 18 characters so the columns line up.
        # :>5 pads the number to 5 characters, right-aligned.
        print(f"  {name:<18}: present in {present:>5} images  ({percent}%)")
    print(f"\nWritten to: {args.output}")
    print("\nLOW PERCENTAGES = HARD TO DETECT. Task 2 will need to handle this.")


if __name__ == "__main__":
    main()