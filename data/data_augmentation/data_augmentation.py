from pathlib import Path
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode


# same target size as resize_image.py - augmentation should never change this
IMG_SIZE = 512


INTERP_IMAGE = InterpolationMode.BILINEAR
INTERP_MASK = InterpolationMode.NEAREST



# INPUT: read a single image (or mask) off disk as a PIL Image
def load_input(path: Path, is_mask: bool = False) -> Image.Image:
    """
    is_mask=False -> converts to RGB (colour photo)
    is_mask=True  -> converts to L (single-channel mask)
    """
    img = Image.open(path)
    img = img.convert("L") if is_mask else img.convert("RGB")
    return img


# OUTPUT: save a single augmented PIL Image back to disk
def save_output(img: Image.Image, out_dir: Path, filename: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / filename
    img.save(save_path)
    return save_path


# FLIP
def flip_horizontal(img: Image.Image) -> Image.Image:
    """Reflect the image left-right."""
    return F.hflip(img)


def flip_vertical(img: Image.Image) -> Image.Image:
    """Reflect the image top-bottom."""
    return F.vflip(img)

# ROTATE

ROTATE_ANGLES = [90]  # fill in the angles you've already decided on


def rotate(img: Image.Image, angle: float, interpolation: InterpolationMode) -> Image.Image:
    """Rotate img by angle degrees, using the given interpolation mode.
    """
    return F.rotate(img, angle, interpolation=interpolation)



# SHIFT

SHIFT_OFFSETS = [(50,50)]  # fill in the (dx, dy) pixel offsets you've decided on


def shift(img: Image.Image, dx: int, dy: int, interpolation: InterpolationMode) -> Image.Image:
    """Translate img by (dx, dy) pixels, using the given interpolation mode.

    Call with interpolation=INTERP_IMAGE for photos,
    interpolation=INTERP_MASK for masks.
    """
    return F.affine(
        img,
        angle=0,
        translate=(dx, dy),
        scale=1.0,
        shear=0,
        interpolation=interpolation,
        fill=0,
    )


# ZOOM

ZOOM_FACTORS = [1.1] 


def zoom(img: Image.Image, factor: float, interpolation: InterpolationMode) -> Image.Image:
    w, h = img.size
    new_w, new_h = int(w * factor), int(h * factor)
    resized = F.resize(img, [new_h, new_w], interpolation=interpolation)

    if factor >= 1.0:
        # zoomed in: crop back to original size from the centre
        top = (new_h - h) // 2
        left = (new_w - w) // 2
        return F.crop(resized, top, left, h, w)
    else:
        # zoomed out: paste centred onto a blank (0) canvas of original size
        canvas = Image.new(img.mode, (w, h), 0)
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        canvas.paste(resized, (left, top))
        return canvas

#brightness
BRIGHTNESS_FACTORS = [1.3] 
CONTRAST_FACTORS = [1.3]    

def adjust_brightness(img: Image.Image, factor: float) -> Image.Image:
    """factor=1.0 is unchanged, >1 brighter, <1 darker. Photos only."""
    return F.adjust_brightness(img, factor)


def adjust_contrast(img: Image.Image, factor: float) -> Image.Image:
    """factor=1.0 is unchanged, >1 more contrast, <1 less. Photos only."""
    return F.adjust_contrast(img, factor)


def find_mask_path(image_path: Path, mask_dir: Path, mask_suffix: str = "_segmentation") -> Path:
    image_id = image_path.stem
    return mask_dir / f"{image_id}{mask_suffix}.png"

def process_paired_folder(
    image_dir: Path,
    mask_dir: Path,
    output_image_dir: Path,
    output_mask_dir: Path,
    mask_suffix: str = "_segmentation",):
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

        #horizontal flip
        img_h = flip_horizontal(img)
        mask_h = flip_horizontal(mask)
        save_output(img_h, output_image_dir, f"{base_name}_flip_horizontal.png")
        save_output(mask_h, output_mask_dir, f"{base_name}{mask_suffix}_flip_horizontal.png")

        #vertical flip
        img_v = flip_vertical(img)
        mask_v = flip_vertical(mask)
        save_output(img_v, output_image_dir, f"{base_name}_flip_vertical.png")
        save_output(mask_v, output_mask_dir, f"{base_name}{mask_suffix}_flip_vertical.png")

        #rotate
        for angle in ROTATE_ANGLES:
            img_r = rotate(img, angle, interpolation=INTERP_IMAGE)
            mask_r = rotate(mask, angle, interpolation=INTERP_MASK)
            save_output(img_r, output_image_dir, f"{base_name}_rotate_{angle}.png")
            save_output(mask_r, output_mask_dir, f"{base_name}{mask_suffix}_rotate_{angle}.png")

        #shift
        for dx, dy in SHIFT_OFFSETS:
            img_s = shift(img, dx, dy, interpolation=INTERP_IMAGE)
            mask_s = shift(mask, dx, dy, interpolation=INTERP_MASK)
            save_output(img_s, output_image_dir, f"{base_name}_shift_dx{dx}_dy{dy}.png")
            save_output(mask_s, output_mask_dir, f"{base_name}{mask_suffix}_shift_dx{dx}_dy{dy}.png")

        #zoom
        for factor in ZOOM_FACTORS:
            img_z = zoom(img, factor, interpolation=INTERP_IMAGE)
            mask_z = zoom(mask, factor, interpolation=INTERP_MASK)
            save_output(img_z, output_image_dir, f"{base_name}_zoom_{factor}.png")
            save_output(mask_z, output_mask_dir, f"{base_name}{mask_suffix}_zoom_{factor}.png")

        # brightness
        for factor in BRIGHTNESS_FACTORS:
            img_b = adjust_brightness(img, factor)
            save_output(img_b, output_image_dir, f"{base_name}_brightness_{factor}.png")
            save_output(mask, output_mask_dir, f"{base_name}{mask_suffix}_brightness_{factor}.png")

        # contrast
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