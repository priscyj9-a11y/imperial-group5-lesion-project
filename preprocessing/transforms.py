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


# ============================================================================
# OPTIONAL: HAIR REMOVAL (DullRazor)  -- referenced on slide 34
# ============================================================================
# NOT part of the default pipeline. Commented out on purpose so it doesn't
# silently change the images everyone trains on.
#
# THE PROBLEM: some dermoscopy photos have body hair over the lesion. The dark
# strands add fake edges that can confuse the segmentation model.
# WHAT IT DOES: finds the thin dark hair lines, then paints over them using the
# surrounding skin colour (this is called "inpainting").
# HOW TO DECIDE IF WE NEED IT: train a baseline WITHOUT it first, look at where
# the model fails. If it's clearly tripping on hairy images, turn this on and
# measure whether the Dice score actually improves. Don't add it blind.
#
# RULES IF YOU TURN IT ON:
#   - Apply to the PHOTO only, never the mask (the mask has no hair).
#   - Apply to BOTH train and test, or the model sees something it never
#     learned during training.
#
# TO USE IT: pip install opencv-python, then uncomment and call remove_hair()
# inside load_image() BEFORE resizing.
#
# import cv2
#
# def remove_hair(image: np.ndarray) -> np.ndarray:
#     """Remove hair strands from a dermoscopy photo (DullRazor method).
#
#     Input:  an RGB image as a numpy array (values 0-255, before normalising).
#     Output: the same image with hair painted over.
#     """
#     # STEP 1: convert to greyscale. hair shows up as dark lines regardless of
#     # colour, so we only need brightness to find it.
#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#
#     # STEP 2: build a "kernel" - a small shape the filter slides over the image.
#     # (17, 17) is a square sized to match typical hair thickness. bigger catches
#     # thicker hair but risks grabbing real structures.
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
#
#     # STEP 3: blackhat morphology. this highlights thin DARK features (hair)
#     # sitting on a lighter background (skin), and ignores the big smooth areas.
#     blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
#
#     # STEP 4: threshold. turn the highlighted hair into a clean black/white
#     # mask: white = "this is hair", black = "leave alone". 10 is the cutoff
#     # brightness; tune it if it misses hair or grabs too much.
#     _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
#
#     # STEP 5: inpaint. paint over every white (hair) pixel using the colours
#     # around it. cv2.INPAINT_TELEA is a standard fill method. the "1" is how
#     # far around each pixel it looks for colour to copy.
#     clean = cv2.inpaint(image, hair_mask, 1, cv2.INPAINT_TELEA)
#
#     return clean
# ============================================================================

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
    """Load a colour photo from disk, resize it, return it as numbers (0.0-1.0).

  
    """
    with Image.open(path) as img:
        # .convert("RGB") forces 3 colour channels.
        # some files sneak in as greyscale or RGBA (with transparency) and
        # would give the model the wrong number of channels. this normalises it.
        img = img.convert("RGB")
        img = resize_image(img, size)
        # NORMALISATION - the team's shared standard.
        # THE PROBLEM: raw pixels are 0-255. Those numbers are too big for a
        # model to learn from comfortably.
        # WHY IT MATTERS: everyone must use the SAME normalisation or our
        # results aren't comparable. This is that one agreed place.
        # WHAT WE DO: divide by 255 so every pixel lands between 0.0 and 1.0.
        # .astype(np.float32) makes them decimals - models expect decimals.

        return np.array(img).astype(np.float32) / 255.0 


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
    
    # match both .jpg (original dataset) and .png (the resized cache), so this
    # same test works whether we point it at the originals or the cache
    image_paths = sorted(
        list((args.data_root / "images").glob("*.jpg"))
        + list((args.data_root / "images").glob("*.png"))
    )[:args.n]

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