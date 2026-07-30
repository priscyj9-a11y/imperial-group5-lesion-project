"""My dataset inspection script.

Just looks at the dataset and tells me what's in it.
Doesn't change, move or delete anything.
"""
import argparse              # lets us pass the dataset path in from the terminal
from collections import Counter   # counts how many times stuff shows up
from pathlib import Path     # handles file paths, don't have to mess with backslashes
from PIL import Image   # Pillow: opens image files so we can look inside them
import numpy as np # NumPy: reads image pixels as a grid of numbers

# the five attributes, spelled how the files spell them
# note: it's "cyst" not "cysts" here, the briefing uses the plural, ours doesn't
# DO NOT CHANGE ! ! ! 
ATTRIBUTE_NAMES = [
    "pigment_network",
    "negative_network",
    "streaks",
    "milia_like_cyst",
    "globules",
]


def extract_id(path: Path) -> str:
    """Pull the 6-digit ID off the front of a filename."""
    # .stem = filename minus the extension
    #   '000001_segmentation.png' becomes '000001_segmentation'
    # .split("_") breaks it at the underscores -> ['000001', 'segmentation']
    # [0] grabs the first bit, which is the ID
    # works for all 3 of my folders since they all start with the number
    return path.stem.split("_")[0]


def inspect_one_image(path: Path) -> dict:
    """Open a single image and report its size and mode. Never crashes.

    THE PROBLEM: any image file might be corrupted or unreadable, and we
    especially can't trust the test set since we've never seen it.
    WHY IT MATTERS: if one bad file crashes the run, everything after it is
    lost. We'd rather flag the bad file by name and keep going.
    WHERE IT FITS: the smallest building block for looking inside files.
    Everything below calls this.

    Returns a dict describing the file. If it can't be opened, the dict has
    ok=False and the reason, instead of throwing an error.
    """
    # try/except means: attempt the risky thing, and if it fails, handle it
    # calmly instead of crashing.
    try:
        # Image.open loads the file. "with" auto-closes it when we're done.
        with Image.open(path) as img:
            # img.size is (width, height). img.mode is the colour type:
            #   "RGB"  = colour, three channels (our photos)
            #   "L"    = greyscale, one channel (our masks, probably)
            return {
                "id": extract_id(path),
                "ok": True,          # we opened it successfully
                "width": img.size[0],
                "height": img.size[1],
                "mode": img.mode,
            }
    except Exception as error:
        # Any failure at all lands here. We record WHY, but don't crash.
        return {
            "id": extract_id(path),
            "ok": False,
            "error": str(error),
        }

def check_image_mask_pairs(root: Path, sample_ids: list[str]) -> None:
    """Open each image and its lesion mask, report size mismatches.

    THE PROBLEM: a photo and its lesion mask must be the exact same size,
    or they don't line up pixel-for-pixel.
    WHY IT MATTERS: the model overlays the mask on the photo to learn. If the
    sizes differ, the answer sheet describes a slightly different picture than
    the one the model sees -- silently wrong labels.
    WHERE IT FITS: opens a sample of pairs using inspect_one_image and reports
    any mismatches, plus the range of image sizes for the later resize step.

    Prints a report. Modifies nothing.
    """
    mismatches = []          # pairs whose sizes don't match
    unreadable = []          # files we couldn't open at all
    widths = []              # every image width we see, to find the range
    heights = []             # every image height we see

    # go through each sampled ID one at a time
    for image_id in sample_ids:
        # build the two file paths for this ID
        image_path = root / "images" / f"{image_id}.jpg"
        mask_path = root / "task1_gt" / f"{image_id}_segmentation.png"

        # open both, safely, using the function from before
        image_info = inspect_one_image(image_path)
        mask_info = inspect_one_image(mask_path)

        # if either failed to open, record it and skip the size comparison
        if not image_info["ok"] or not mask_info["ok"]:
            unreadable.append(image_id)
            continue    # "continue" jumps to the next ID in the loop

        # collect the image's size so we can report the overall range later
        widths.append(image_info["width"])
        heights.append(image_info["height"])

        # compare (width, height) of image vs mask
        image_size = (image_info["width"], image_info["height"])
        mask_size = (mask_info["width"], mask_info["height"])
        if image_size != mask_size:
            mismatches.append((image_id, image_size, mask_size))

    # --- print the report ---
    print(f"\nChecked {len(sample_ids)} image/mask pairs:")
    print(f"  unreadable files: {len(unreadable)}")
    print(f"  size mismatches : {len(mismatches)}")
    if mismatches:
        # show the first 5 so we can see the pattern without flooding output
        for image_id, img_sz, msk_sz in mismatches[:5]:
            print(f"    {image_id}: image {img_sz} vs mask {msk_sz}")

    # report the size range only if we actually opened some images
    if widths:
        print(f"  image width range : {min(widths)} to {max(widths)}")
        print(f"  image height range: {min(heights)} to {max(heights)}")





def check_mask_values(root: Path, sample_ids: list[str]) -> None:
    """Check lesion masks are binary, and find all-black attribute masks.

    THE PROBLEM: we've assumed masks are pure black-and-white but never
    checked. A mask should hold only 0 (background) and 255 (the thing).
    Stray grey values like 127 would quietly break our lesion/not-lesion logic.
    WHY IT MATTERS: it also answers how an ABSENT attribute is stored. An
    all-black attribute mask (only 0, no 255) means "not present in this image".
    Counting those tells us how sparse each attribute is.
    WHERE IT FITS: the final inspection check. After this the "look inside the
    files" stage is done.

    Prints a report. Modifies nothing.
    """
    # a set collects distinct values and throws away repeats automatically.
    # we'll gather every pixel value we ever see across the lesion masks here.
    lesion_values = set()

    # count how many attribute masks are completely black, per attribute.
    # start every attribute at 0.
    blank_counts = {name: 0 for name in ATTRIBUTE_NAMES}

    for image_id in sample_ids:
        # --- lesion mask ---
        mask_path = root / "task1_gt" / f"{image_id}_segmentation.png"
        info = inspect_one_image(mask_path)
        if info["ok"]:
            # open the image again, convert to "L" (greyscale), read as numbers
            with Image.open(mask_path) as img:
                pixels = np.array(img.convert("L"))
            # np.unique lists the distinct values in the grid, e.g. [0 255].
            # .update adds them to our running set.
            lesion_values.update(np.unique(pixels).tolist())

        # --- five attribute masks ---
        for name in ATTRIBUTE_NAMES:
            attr_path = root / "task2_gt" / f"{image_id}_attribute_{name}.png"
            attr_info = inspect_one_image(attr_path)
            if attr_info["ok"]:
                with Image.open(attr_path) as img:
                    pixels = np.array(img.convert("L"))
                # .max() is the brightest pixel. If it's 0, the whole mask is
                # black -> this attribute is absent for this image.
                if pixels.max() == 0:
                    blank_counts[name] += 1

    # --- report ---
    print(f"\nMask values (from {len(sample_ids)} sampled cases):")
    # sorted() so the values print in order, e.g. [0, 255]
    print(f"  distinct lesion-mask values: {sorted(lesion_values)}")
    print(f"  (expect [0, 255] if masks are clean binary)")

    print(f"\nAll-black (absent) attribute masks, out of {len(sample_ids)}:")
    for name in ATTRIBUTE_NAMES:
        # how many were blank, and roughly what fraction
        n = blank_counts[name]
        print(f"  {name:<18}: {n} absent")





def collect_ids(folder: Path, extension: str) -> list[str]:
    """Go through a folder and give me back a list of every ID in it."""

    # if anyone typos the path It basically shouts at you and doesnt just say : "found 0 files " 
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    # glob("*.png") = every file ending in .png
    # sorted() so the order is the same every time we run it
    # the [ ... for ... ] thing runs extract_id on each file and collects the results
    return [extract_id(p) for p in sorted(folder.glob(f"*{extension}"))]


def report_duplicates(label: str, ids: list[str]) -> None:
    """Shout if any ID turns up more than once."""

    # Counter gives me {'000001': 1, '000002': 2, ...}
    # keep only the ones where the count is bigger than 1
    repeats = [i for i, n in Counter(ids).items() if n > 1]

    # empty list counts as False, so this only prints when something's wrong
    if repeats:
        # [:5] = only shows me the first 5 so it doesn't spam the terminal
        print(f"  WARNING {label}: {len(repeats)} duplicate IDs, e.g. {repeats[:5]}")


def main() -> None:
    # set up the --data-root argument so the path isn't hardcoded
    # (hardcoding it would break for everyone else on the team)
    parser = argparse.ArgumentParser(description="Inspect the dataset (read-only).")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Folder containing images/, task1_gt/, task2_gt/")
    args = parser.parse_args()

    # argparse swaps the dash for an underscore: --data-root -> args.data_root
    root: Path = args.data_root

    # the / just glues folder names onto the path
    # root / "images" = C:\...\train\images
    image_ids = collect_ids(root / "images", ".jpg")
    lesion_ids = collect_ids(root / "task1_gt", ".png")
    attribute_ids = collect_ids(root / "task2_gt", ".png")

    # basic counts first
    print(f"Dataset root: {root}")
    print(f"  raw images     : {len(image_ids)}")
    print(f"  lesion masks   : {len(lesion_ids)}")
    print(f"  attribute masks: {len(attribute_ids)}")

    # silent if everything's fine
    report_duplicates("images", image_ids)
    report_duplicates("lesion masks", lesion_ids)

    # sets drop duplicates and are fast for comparing
    images = set(image_ids)
    lesions = set(lesion_ids)

    # \n = blank line before this
    print(f"\nUnique IDs: images={len(images)}, lesion masks={len(lesions)}")

    # minus on sets = "in the left one but not the right one"
    # the counts already matched, but that doesn't prove they're the SAME ids
    # both of these need to be 0
    print(f"  images with no lesion mask: {len(images - lesions)}")
    print(f"  lesion masks with no image: {len(lesions - images)}")

    # every ID should show up 5 times in task2_gt, once per attribute
    counts = Counter(attribute_ids)

    # keep the ones that aren't 5
    # using len(ATTRIBUTE_NAMES) instead of just writing 5, in case that changes
    wrong = {i: c for i, c in counts.items() if c != len(ATTRIBUTE_NAMES)}

    print(f"\nIDs without exactly 5 attribute masks: {len(wrong)}")
    if wrong:
        print(f"  examples: {list(wrong.items())[:5]}")
    
    # and catch any image with no attribute masks at all
    print(f"IDs in images but absent from task2_gt: {len(images - set(counts))}")
    
    
    # pick a sample to inspect the file CONTENTS...
    sample_ids = sorted(images)[:100]

    check_image_mask_pairs(root, sample_ids)
    check_mask_values(root, sample_ids)

# only runs main() if I run this file directly
# if something imports it later, main() won't fire on its own
if __name__ == "__main__":
    main()