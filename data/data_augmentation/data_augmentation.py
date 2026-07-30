from pathlib import Path
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode


IMG_SIZE = 512


INTERP_IMAGE = InterpolationMode.BILINEAR
INTERP_MASK = InterpolationMode.NEAREST
#input
def load_input(path: Path, is_mask: bool = False) -> Image.Image:
    """Open a file from disk and return it as a PIL Image, ready to augment.

    is_mask=False -> converts to RGB (colour photo)
    is_mask=True  -> converts to L (single-channel mask)
    Assumes the file is ALREADY resized to IMG_SIZE x IMG_SIZE.
    """
    img = Image.open(path)
    img = img.convert("L") if is_mask else img.convert("RGB")
    return img

#output
def save_output(img: Image.Image, out_dir: Path, filename: str) -> Path:
    """Save an augmented PIL Image to out_dir/filename, creating out_dir if needed.

    Returns the path it was saved to.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / filename
    img.save(save_path)
    return save_path

#flip
def flip_horizontal(img: Image.Image) -> Image.Image:
    """Mirror the image left-right."""
    return F.hflip(img)


def flip_vertical(img: Image.Image) -> Image.Image:
    """Mirror the image top-bottom."""
    return F.vflip(img)


#rotate
ROTATE_ANGLES = [90]  #counterclockwise

def rotate(img: Image.Image, angle: float, interpolation: InterpolationMode) -> Image.Image:

    return F.rotate(img, angle, interpolation=interpolation)

#shift
SHIFT_OFFSETS = [(20, 20)]  # fill in the (dx, dy) pixel offsets you've decided on


def shift(img: Image.Image, dx: int, dy: int, interpolation: InterpolationMode) -> Image.Image:
    return F.affine(
        img,
        angle=0,
        translate=(dx, dy),
        scale=1.0,
        shear=0,
        interpolation=interpolation,
        fill=0,
    )

#zoom
ZOOM_FACTORS = [1.2]


def zoom(img: Image.Image, factor: float, interpolation: InterpolationMode) -> Image.Image:
    w, h = img.size
    new_w, new_h = int(w * factor), int(h * factor)
    resized = F.resize(img, [new_h, new_w], interpolation=interpolation)

    if factor >= 1.0:
        
        top = (new_h - h) // 2
        left = (new_w - w) // 2
        return F.crop(resized, top, left, h, w)
    else:
        
        canvas = Image.new(img.mode, (w, h), 0)
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        canvas.paste(resized, (left, top))
        return canvas


#brightness and contrast
BRIGHTNESS_FACTORS = [1.3]  
CONTRAST_FACTORS = [0.7] 

def adjust_brightness(img: Image.Image, factor: float) -> Image.Image:
    return F.adjust_brightness(img, factor)


def adjust_contrast(img: Image.Image, factor: float) -> Image.Image:
    return F.adjust_contrast(img, factor)

import random


def random_augment(img: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Randomly apply flip/rotate/shift/zoom to BOTH img and mask together,
    then randomly apply brightness/contrast to img ONLY.

    Call this ONLY on training data. Never on validation data - see
    process_paired_folder's docstring and the team README for why.
    """
    # --- geometric transforms: same random choice applied to both ---
    if random.random() < 0.5:
        img, mask = flip_horizontal(img), flip_horizontal(mask)

    if random.random() < 0.5:
        img, mask = flip_vertical(img), flip_vertical(mask)

    if random.random() < 0.5:
        angle = random.choice(ROTATE_ANGLES)
        img = rotate(img, angle, interpolation=INTERP_IMAGE)
        mask = rotate(mask, angle, interpolation=INTERP_MASK)

    if random.random() < 0.3:
        dx, dy = random.choice(SHIFT_OFFSETS)
        img = shift(img, dx, dy, interpolation=INTERP_IMAGE)
        mask = shift(mask, dx, dy, interpolation=INTERP_MASK)

    if random.random() < 0.3:
        factor = random.choice(ZOOM_FACTORS)
        img = zoom(img, factor, interpolation=INTERP_IMAGE)
        mask = zoom(mask, factor, interpolation=INTERP_MASK)

    # --- photo-only transforms: mask is NEVER touched here ---
    if random.random() < 0.5:
        factor = random.choice(BRIGHTNESS_FACTORS)
        img = adjust_brightness(img, factor)

    if random.random() < 0.5:
        factor = random.choice(CONTRAST_FACTORS)
        img = adjust_contrast(img, factor)

    return img, mask



def find_mask_path(image_path: Path, mask_dir: Path, mask_suffix: str = "_segmentation") -> Path:
    image_id = image_path.stem
    return mask_dir / f"{image_id}{mask_suffix}.png"


# ---------------------------------------------------------------
# batch entry point: process every image + its matching mask together
# every augmented pair gets the SAME transform applied and the SAME
# output filename tag, so image and mask stay lined up.
# ---------------------------------------------------------------
def process_paired_folder(
    image_dir: Path,
    mask_dir: Path,
    output_image_dir: Path,
    output_mask_dir: Path,
    mask_suffix: str = "_segmentation",
):
    """Loop over every image in image_dir, flip it AND its matching mask
    together, save both to their respective output folders under matching
    filenames."""
    valid_exts = (".jpg", ".jpeg", ".png")
    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in valid_exts)

    for image_path in image_paths:
        mask_path = find_mask_path(image_path, mask_dir, mask_suffix)
        if not mask_path.exists():
            print(f"skipping {image_path.name}: no matching mask at {mask_path}")
            continue

        img = load_input(image_path, is_mask=False)
        mask = load_input(mask_path, is_mask=True)
        base_name = image_path.stem

        # --- horizontal flip: same transform, applied to both ---
        img_h = flip_horizontal(img)
        mask_h = flip_horizontal(mask)
        save_output(img_h, output_image_dir, f"{base_name}_flip_horizontal.png")
        save_output(mask_h, output_mask_dir, f"{base_name}{mask_suffix}_flip_horizontal.png")

        # --- vertical flip: same transform, applied to both ---
        img_v = flip_vertical(img)
        mask_v = flip_vertical(mask)
        save_output(img_v, output_image_dir, f"{base_name}_flip_vertical.png")
        save_output(mask_v, output_mask_dir, f"{base_name}{mask_suffix}_flip_vertical.png")

        # --- rotate: same angle applied to both, but different interpolation ---
        for angle in ROTATE_ANGLES:
            img_r = rotate(img, angle, interpolation=INTERP_IMAGE)
            mask_r = rotate(mask, angle, interpolation=INTERP_MASK)
            save_output(img_r, output_image_dir, f"{base_name}_rotate_{angle}.png")
            save_output(mask_r, output_mask_dir, f"{base_name}{mask_suffix}_rotate_{angle}.png")

        # --- shift: same offset applied to both, different interpolation ---
        for dx, dy in SHIFT_OFFSETS:
            img_s = shift(img, dx, dy, interpolation=INTERP_IMAGE)
            mask_s = shift(mask, dx, dy, interpolation=INTERP_MASK)
            save_output(img_s, output_image_dir, f"{base_name}_shift_dx{dx}_dy{dy}.png")
            save_output(mask_s, output_mask_dir, f"{base_name}{mask_suffix}_shift_dx{dx}_dy{dy}.png")

        # --- zoom: same factor applied to both, different interpolation ---
        for factor in ZOOM_FACTORS:
            img_z = zoom(img, factor, interpolation=INTERP_IMAGE)
            mask_z = zoom(mask, factor, interpolation=INTERP_MASK)
            save_output(img_z, output_image_dir, f"{base_name}_zoom_{factor}.png")
            save_output(mask_z, output_mask_dir, f"{base_name}{mask_suffix}_zoom_{factor}.png")

        # --- brightness: PHOTO ONLY. mask is saved unchanged alongside it,
        # so the pair still lines up under the same filename tag. ---
        for factor in BRIGHTNESS_FACTORS:
            img_b = adjust_brightness(img, factor)
            save_output(img_b, output_image_dir, f"{base_name}_brightness_{factor}.png")
            save_output(mask, output_mask_dir, f"{base_name}{mask_suffix}_brightness_{factor}.png")

        # --- contrast: PHOTO ONLY. same reasoning as brightness. ---
        for factor in CONTRAST_FACTORS:
            img_c = adjust_contrast(img, factor)
            save_output(img_c, output_image_dir, f"{base_name}_contrast_{factor}.png")
            save_output(mask, output_mask_dir, f"{base_name}{mask_suffix}_contrast_{factor}.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Augment a folder of resized images + masks together.")
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output-image-dir", type=Path, required=True)
    parser.add_argument("--output-mask-dir", type=Path, required=True)
    parser.add_argument("--mask-suffix", type=str, default="_segmentation")
    args = parser.parse_args()

    process_paired_folder(
        args.image_dir,
        args.mask_dir,
        args.output_image_dir,
        args.output_mask_dir,
        mask_suffix=args.mask_suffix,
    )