"""Resizing (and later augmentation) for images and masks.

These get called while training, one image at a time.
Nothing here writes to disk - the original dataset is never touched.
"""

from pathlib import Path
from PIL import Image
import numpy as np


# the size everything gets squashed to.
# 512 is the standard for this ISIC dataset - big enough to keep the small
# structures (globules, milia-like cysts), small enough to train in 2 weeks.
# debug at 256 if things are slow, try 768 later if we have time to spare.
DEFAULT_SIZE = 512


def resize_image(img: Image.Image, size: int = DEFAULT_SIZE) -> Image.Image:
    """Resize a COLOUR PHOTO to size x size using bilinear.

    THE PROBLEM: our images range from 639px to 6688px wide. A neural network
    needs every input to be the exact same shape, so everything must be
    squashed to one fixed size.
    WHY BILINEAR: it blends neighbouring pixels together when shrinking, which
    keeps the photo looking smooth and natural. That's what we want for a
    photo - smooth colour gradients are real information about the skin.
    WHERE IT FITS: called on the raw .jpg every time an image is loaded.
    """
    # .resize takes (width, height) as a pair, hence (size, size) for a square
    # Image.BILINEAR is the blending method described above
    return img.resize((size, size), Image.BILINEAR)


def resize_mask(mask: Image.Image, size: int = DEFAULT_SIZE) -> Image.Image:
    """Resize a MASK to size x size using nearest-neighbour.

    THE PROBLEM: masks must stay binary. Only 0 (background) and 255 (the thing)
    are allowed - we proved they're clean [0, 255] in inspect_dataset.py.
    WHY NOT BILINEAR: blending would average a 0 and a 255 into greys like 127.
    Then "is this pixel lesion or not?" has no clean answer and our ground truth
    is quietly corrupted. This is the classic mistake in segmentation projects.
    WHY NEAREST: it just copies the closest existing pixel, no maths, no
    blending. A 0 stays 0, a 255 stays 255. Edges get slightly blocky, which
    is a fine price to pay for keeping the labels correct.
    WHERE IT FITS: called on every lesion mask AND every attribute mask.
    """
    # Image.NEAREST = copy the closest pixel, don't blend anything
    return mask.resize((size, size), Image.NEAREST)


def load_image(path: Path, size: int = DEFAULT_SIZE) -> np.ndarray:
    """Load a colour photo from disk, resize it, return it as numbers.

    WHERE IT FITS: this is what the Dataset class will call for the .jpg.
    Returns a grid of shape (size, size, 3) - the 3 is red/green/blue.
    """
    with Image.open(path) as img:
        # .convert("RGB") forces 3 colour channels.
        # some files sneak in as greyscale or RGBA (with transparency) and
        # would give the model the wrong number of channels. this normalises it.
        img = img.convert("RGB")
        img = resize_image(img, size)
        # np.array turns the picture into a grid of numbers we can do maths on
        return np.array(img)


def load_mask(path: Path, size: int = DEFAULT_SIZE) -> np.ndarray:
    """Load a mask from disk, resize it, return it as 0s and 1s.

    WHY 0 AND 1 INSTEAD OF 0 AND 255: models expect labels as 0/1, not 0/255.
    We convert at the very end, AFTER resizing, so the nearest-neighbour step
    still had clean values to work with.
    WHERE IT FITS: called for the lesion mask and each attribute mask.
    """
    with Image.open(path) as mask:
        # .convert("L") = single greyscale channel. a mask has no colour,
        # and this guarantees one channel even if the file was saved oddly.
        mask = mask.convert("L")
        mask = resize_mask(mask, size)
        array = np.array(mask)

    # "> 127" asks each pixel "are you bright?" giving True/False.
    # .astype(np.uint8) turns True into 1 and False into 0.
    # using 127 as the cutoff rather than "== 255" is defensive: if a mask ever
    # does contain a stray grey, this still forces a clean yes/no answer.
    return (array > 127).astype(np.uint8)


# ---------------------------------------------------------------
# self-test: run this file directly to check the transforms on MANY images.
# testing one image only proves the code runs. testing a batch proves it holds
# up across the real variety in our dataset (639px to 6688px wide, portrait
# and landscape, etc).
# ---------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test the transforms on a batch.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--n", type=int, default=50, help="how many cases to test")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    args = parser.parse_args()

    # grab the first n image files so we test a real spread, not just one.
    # sorted() so we get the same sample every run.
    image_paths = sorted((args.data_root / "images").glob("*.jpg"))[:args.n]

    bad_shapes = []      # anything that didn't come out size x size
    bad_values = []      # any mask that isn't purely 0s and 1s
    empty_masks = []     # masks that lost ALL their white after resizing

    for image_path in image_paths:
        # .stem is the filename without the extension, so '000001.jpg' -> '000001'
        image_id = image_path.stem
        mask_path = args.data_root / "task1_gt" / f"{image_id}_segmentation.png"

        image = load_image(image_path, args.size)
        mask = load_mask(mask_path, args.size)

        # check both came out the right shape
        if image.shape != (args.size, args.size, 3):
            bad_shapes.append((image_id, "image", image.shape))
        if mask.shape != (args.size, args.size):
            bad_shapes.append((image_id, "mask", mask.shape))

        # check the mask contains ONLY 0s and 1s, nothing in between.
        # .issubset({0, 1}) asks "is everything in here either a 0 or a 1?"
        values = set(np.unique(mask).tolist())
        if not values.issubset({0, 1}):
            bad_values.append((image_id, sorted(values)))

        # a mask of all zeros means the lesion vanished during resizing.
        # shouldn't happen for lesions (they're big), but worth catching.
        if mask.max() == 0:
            empty_masks.append(image_id)

    # all three counts should be 0
    print(f"Tested {len(image_paths)} cases at {args.size}x{args.size}")
    print(f"  wrong shapes    : {len(bad_shapes)}   {bad_shapes[:3]}")
    print(f"  non-binary masks: {len(bad_values)}   {bad_values[:3]}")
    print(f"  empty masks     : {len(empty_masks)}  {empty_masks[:3]}")