"""
Export Task 2 predicted attribute masks for the validation split.

Takes the saved task2_model.pth (trained weights), runs it on the validation images only (never-seen-during-training, so the score is meaningful), 
and writes out predicted masks in the exact folder format Ivana's evaluation code expects.
"""

from __future__ import annotations
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image #Pillow's Image class used to write PNG files to disk

from attribute_detection.model import UNetMultiHead
from attribute_detection.task2_eval.mask_io import ATTRIBUTE_NAMES, prediction_mask_path
from preprocessing.transforms import load_image, DEFAULT_SIZE

#The same fixed normalization numbers used during training — has to match exactly, or the model would see differently-scaled input than what it learned on.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def read_validation_ids(splits_csv: Path) -> list[str]:
    """Same logic as evaluate_validation.py's reader, kept local so this
    script has no dependency on that file specifically."""
    validation_ids: list[str] = []
    with splits_csv.open("r", newline="", encoding="utf-8") as file: #Opens the CSV
        reader = csv.DictReader(file) #wraps it in a reader that lets you access columns by name (row["image_id"]) instead of by position.
        
        #safety check confirming the CSV actually has both expected column headers before trying to use them.
        required = {"image_id", "split"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError("splits.csv must contain columns: image_id, split")
        
        #Goes through every row; if its split column says "val" (after cleaning up case/whitespace), grab that row's image_id, zero-pad it to 6 digits, and add it to the list. Hand back the completed list.
        for row in reader:
            if row["split"].strip().lower() == "val":
                validation_ids.append(row["image_id"].strip().zfill(6))
    return validation_ids


def load_model(checkpoint_path: Path, device: torch.device) -> UNetMultiHead:

    #Creates a fresh model with random weights (pretrained=False here because we're about to immediately overwrite every weight with your actual trained checkpoint, so downloading VGG16's ImageNet weights first would be wasted time).
    model = UNetMultiHead(pretrained=False, freeze_encoder=False, num_attributes=len(ATTRIBUTE_NAMES))
    
    #Reads the checkpoint file off disk. map_location=device handles a subtle cross-device issue: if you trained on GPU but are now running this on a CPU-only machine (or vice versa), this makes sure the loaded tensors land on the correct device rather than crashing or ending up in the wrong place.
    loaded = torch.load(checkpoint_path, map_location=device)

    #handles the checkpoint-format: newer checkpoints are a small dictionary bundle (with model_state_dict, epoch, etc.), while older ones were just the raw weights directly. isinstance(loaded, dict) and "model_state_dict" in loaded checks which format this particular file is (if it's the new bundle, dig into it and print a helpful message showing which epoch it came from; otherwise, assume the whole loaded object is the weights themselves (the old format))
    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        state_dict = loaded["model_state_dict"]
        print(f"  Loaded checkpoint from epoch {loaded.get('epoch', '?')}, "
              f"best_mean_dice={loaded.get('best_mean_dice', '?')}")
    else:
        state_dict = loaded  # old-format checkpoint: the file itself IS the state_dict

    #opies every learned number from the checkpoint into the fresh model — this is the real "loading" step; everything before it was just figuring out which numbers to load.
    model.load_state_dict(state_dict)

    #Moves the now-loaded model onto the target device, switches it into evaluation mode (disabling training-only behaviors like dropout/batchnorm-in-training-mode), and hands it back.
    model.to(device)
    model.eval()  
    return model


def predict_one_image(model: UNetMultiHead, image_path: Path, size: int, device: torch.device) -> np.ndarray:
    """Returns predicted attribute masks as a [5, H, W] binary NumPy array."""
    
    #Loads and resizes one real image
    image_array = load_image(image_path, size)  # [H, W, 3] float in [0, 1], resized to `size`

    #Converts to a tensor, reorders from [H, W, 3] to [3, H, W] (channels-first), converts to floating point
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()  # [3, H, W]

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std  # ImageNet normalisation, same as training
    #Note: .view(3, 1, 1) reshapes the flat list of 3 numbers into a shape that lines up correctly against the image's [3, H, W] shape (one mean/std value per color channel, broadcasting across every pixel).
    #Subtracting the mean and dividing by the standard deviation is the actual normalization math — same operation TF.normalize performs in dataset.py, just written out manually here instead of using that helper function.

    #unsqueeze(0) inserts a new dimension at the front, turning [3, H, W] into [1, 3, H, W] — the model always expects a batch dimension, even for a single image; 1 here just means "a batch of exactly one." Then moves it to the correct device.
    image_tensor = image_tensor.unsqueeze(0).to(device)  

    #Runs the image through the model, without tracking gradients (we're not training, just predicting).
    with torch.no_grad():  
        preds = model(image_tensor) 

    #Thresholds every value to a hard 0 or 1 — turning "probability the attribute is present" into a definite yes/no decision.
    preds = (preds > 0.5).float() 
    preds = preds.squeeze(0).cpu().numpy()  # [5, H, W] removing batch dimension  -> back to NumPy on CPU since we're only dealing with one image at a time
    #plain NumPy array necessary because the next step (saving as an image file) uses Pillow, which works with NumPy arrays, not PyTorch tensors.
    return preds.astype(np.uint8) #Converts the data type to 8-bit unsigned integers (values 0–255) — the standard format for image pixel data, required by Pillow to correctly interpret and save this as a proper image file.


def save_prediction_masks(preds: np.ndarray, prediction_root: Path, image_id: str) -> None:
    """Saves one PNG per attribute, matching mask_io.prediction_mask_path()."""
    for index, attribute in enumerate(ATTRIBUTE_NAMES): #Loops through all 5 attribute names, along with their position (index) in that list — since preds is a [5, H, W] array, index tells us which "slice" of that array corresponds to which attribute.
        
        #Pulls out just this one attribute's 2D mask — preds[index] gives you a [H, W] slice from the full [5, H, W] array.
        mask = preds[index]  

        #Builds the correct output file path using the shared helper function, then makes sure the containing folder actually exists before trying to save into it (prediction_root/image_id/ needs to exist as a real folder before you can save a file inside it).
        out_path = prediction_mask_path(prediction_root, image_id, attribute)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as standard 0/255 grayscale PNG — same convention as the real
        # ground-truth mask files, so load_mask() reads predictions the same
        # way it reads ground truth.
        #Image.fromarray(...) converts the NumPy array into an actual Pillow image object; mode="L" specifies "this is a single-channel grayscale image," not color. .save(out_path) writes it to disk as a PNG.
        Image.fromarray(mask * 255, mode="L").save(out_path)


def main() -> None:

    #Sets up 6 command-line arguments, matching the same pattern we saw in train.py. Unlike train.py, this file's --resume-equivalent situation doesn't exist
    parser = argparse.ArgumentParser(description="Export Task 2 predicted attribute masks for validation images.")
    parser.add_argument("--data-root", type=Path, required=True, help="Folder containing images/, task1_gt/, task2_gt/.")
    parser.add_argument("--splits-csv", type=Path, required=True, help="CSV with image_id,split columns.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained model .pth file (e.g. task2_model.pth).")
    parser.add_argument("--prediction-root", type=Path, required=True, help="Output folder: prediction_root/image_id/attribute.png")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help="Resize target — must match what the checkpoint was trained on.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    #Converts to an actual device object, and prints it - confirms whether we're running on GPU or CPU. 
    device = torch.device(args.device)
    print(f"Using device: {device}")

    #Gets the list of validation image IDs, and reports how many there are.
    validation_ids = read_validation_ids(args.splits_csv)
    print(f"Validation IDs to export: {len(validation_ids)}")

    #Loads trained model once — importantly, outside the loop, since re-loading it for every single image would be enormously wasteful.
    model = load_model(args.checkpoint, device)

    #Loops through every validation ID, with a counter i starting at 1
    for i, image_id in enumerate(validation_ids, start=1):

        #Builds the expected image path; if it genuinely doesn't exist, print a warning and skip to the next image rather than crashing the entire export over one missing file.
        image_path = args.data_root / "images" / f"{image_id}.jpg"
        if not image_path.exists():
            print(f"  [skip] missing image: {image_path}")
            continue

        #Runs the model on this image, then saves all 5 resulting masks to disk.
        preds = predict_one_image(model, image_path, args.size, device)
        save_prediction_masks(preds, args.prediction_root, image_id)


        #i % 50 == 0 is true every 50th image (50, 100, 150, ...) — so you get a progress update periodically
        #i == len(validation_ids) additionally guarantees you always see a final confirmation on the very last image, even if the total count isn't a clean multiple of 50.
        if i % 50 == 0 or i == len(validation_ids):
            print(f"  Exported {i}/{len(validation_ids)}")

    #Final confirmation once the entire loop finishes.
    print(f"Done. Predictions written to: {args.prediction_root}")


if __name__ == "__main__":
    main()