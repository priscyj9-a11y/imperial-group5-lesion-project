
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from unet_model import UNet

IMG_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------
# DATASET: turns a folder of images + masks into (tensor, tensor) pairs
# ---------------------------------------------------------------
class LesionDataset(Dataset):
    """Pairs each image with its matching mask by filename.

    Expects: image_dir/{id}.jpg  and  mask_dir/{id}{mask_suffix}.png
    (same convention as resize_image.py / augmentation.py)
    """

    def __init__(self, image_dir: Path, mask_dir: Path, mask_suffix: str = "_segmentation"):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.mask_suffix = mask_suffix

        valid_exts = (".jpg", ".jpeg", ".png")
        self.image_paths = sorted(
            p for p in self.image_dir.iterdir() if p.suffix.lower() in valid_exts
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        mask_path = self.mask_dir / f"{image_path.stem}{self.mask_suffix}.png"

        # ---- image: RGB, scaled to [0, 1], shape (3, H, W) ----
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # (H,W,3) -> (3,H,W)

        # ---- mask: single channel, binary 0/1, shape (1, H, W) ----
        mask = Image.open(mask_path).convert("L")
        mask_array = (np.array(mask) > 127).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)  # (H,W) -> (1,H,W)

        return img_tensor, mask_tensor


# ---------------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------------
def train(
    image_dir: Path,
    mask_dir: Path,
    checkpoint_path: Path,
    epochs: int = 20,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
):
    """Train a fresh UNet and save its weights to checkpoint_path."""
    dataset = LesionDataset(image_dir, mask_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    # BCEWithLogitsLoss expects raw logits (matches UNet's output) and
    # applies sigmoid internally - more numerically stable than doing
    # sigmoid + BCELoss as two separate steps.
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for images, masks in loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        avg_loss = running_loss / len(dataset)
        print(f"epoch {epoch}/{epochs}  loss: {avg_loss:.4f}")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"saved model weights to {checkpoint_path}")


# ---------------------------------------------------------------
# INFERENCE: predict a mask for ONE new image
# ---------------------------------------------------------------
def load_trained_model(checkpoint_path: Path) -> UNet:
    """Load a UNet with saved weights, ready for prediction (not training)."""
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()  # turns off dropout/batchnorm training behaviour
    return model


def predict_mask(model: UNet, image_path: Path, threshold: float = 0.5) -> Image.Image:
   
    img = Image.open(image_path).convert("RGB")
    if img.size != (IMG_SIZE, IMG_SIZE):
        raise ValueError(
            f"expected a {IMG_SIZE}x{IMG_SIZE} image, got {img.size} - "
            f"resize it first (see resize_image.py)"
        )

    img_array = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(DEVICE)  # (1,3,H,W)

    with torch.no_grad():  # no gradients needed for inference - saves memory
        logits = model(img_tensor)
        probs = torch.sigmoid(logits)  # logits -> probabilities in [0, 1]

    prob_array = probs.squeeze().cpu().numpy()  # (1,1,H,W) -> (H,W)
    binary_mask = (prob_array > threshold).astype(np.uint8) * 255

    return Image.fromarray(binary_mask, mode="L")


def predict_folder(model: UNet, image_dir: Path, output_dir: Path, threshold: float = 0.5):
    """Predict masks for every image in image_dir, save results to output_dir
    with matching filenames: {image_id}_predicted_mask.png"""
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_exts = (".jpg", ".jpeg", ".png")
    image_paths = sorted(p for p in Path(image_dir).iterdir() if p.suffix.lower() in valid_exts)

    for image_path in image_paths:
        mask = predict_mask(model, image_path, threshold=threshold)
        out_path = output_dir / f"{image_path.stem}_predicted_mask.png"
        mask.save(out_path)
        print(f"predicted: {image_path.name} -> {out_path.name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or run the lesion segmentation U-Net.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--image-dir", type=Path, required=True)
    train_parser.add_argument("--mask-dir", type=Path, required=True)
    train_parser.add_argument("--checkpoint", type=Path, required=True)
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--lr", type=float, default=1e-4)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--checkpoint", type=Path, required=True)
    predict_parser.add_argument("--image-dir", type=Path, required=True)
    predict_parser.add_argument("--output-dir", type=Path, required=True)
    predict_parser.add_argument("--threshold", type=float, default=0.5)

    args = parser.parse_args()

    if args.command == "train":
        train(
            args.image_dir,
            args.mask_dir,
            args.checkpoint,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )
    elif args.command == "predict":
        loaded_model = load_trained_model(args.checkpoint)
        predict_folder(loaded_model, args.image_dir, args.output_dir, threshold=args.threshold)