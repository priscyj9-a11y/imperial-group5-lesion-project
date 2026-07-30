"""
Minimal training entry point for Task 2 attribute segmentation.

The loop tying the pipeline together: load data → run model → score with loss → adjust weights → repeat, 
for however many epochs you ask for, then saves the trained weights to task2_model.pth.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from attribute_detection_v3.dataset import AttributeDataset
from attribute_detection_v3.loss import DiceBCELoss, dice_score
from attribute_detection_v3.model import UNetMultiHead
from attribute_detection_v3.task2_eval.mask_io import ATTRIBUTE_NAMES
from preprocessing.attribute_stats import attribute_is_present  # reused so weighting matches the
                                                                  # same presence definition attribute_stats.py
                                                                  # already validated (full-res, no resize)


def parse_args() -> argparse.Namespace: #Creates an argument parser: turns text we type after python3 -m attribute_detection_v3.train into actual Python values.
    parser = argparse.ArgumentParser(description="Train Task 2 attribute segmentation model")

    #Two arguments we must supply, or the script refuses to run at all: where data lives, and where the train/val split list is.
    parser.add_argument("--data-root", type=Path, required=True, help="Dataset root containing images/, task1_gt/, task2_gt/")
    parser.add_argument("--splits-csv", type=Path, required=True, help="CSV with image_id,split columns.")

    #Three optional arguments with sensible defaults if not specified: how many times to loop over the whole dataset, how many images per batch, and the optimizer's learning rate (how big a step to take each time weights get adjusted).
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)

    #Automatically picks GPU if one's available, otherwise falls back to CPU (possible to override with --device cuda or --device cpu).
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    #Optional: if you want to resume training from a previous checkpoint, give its path here. Otherwise, training starts from scratch.
    parser.add_argument("--resume", type=Path, default=None, help="Path to checkpoint to resume training from (e.g. task2_model.pth).")

    #Optional: where to save the trained model and any intermediate checkpoints. Defaults to outputs/task2_training/.
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "task2_training")
    
    #Optional: learning rate scheduler patience (epochs before reducing LR). Set to 0 to disable.
    parser.add_argument("--scheduler-patience", type=int, default=3, help="Epochs of no improvement before reducing learning rate (0 to disable scheduler)")
    
    #Optional: early stopping patience (stop training if val_loss doesn't improve for N epochs). Set to 0 to disable.
    parser.add_argument("--early-stop-patience", type=int, default=5, help="Epochs of no improvement before stopping training (0 to disable early stopping)")

    #Optional: weight the loss by inverse attribute prevalence, so rare attributes (streaks,
    #negative_network) cost more to get wrong. Separate lever from oversampling - see
    #compute_attribute_weights() docstring. Default ON; pass --no-attribute-weighting to
    #A/B test against an unweighted run (e.g. to compare against last night's results).
    parser.add_argument("--attribute-weighting", action=argparse.BooleanOptionalAction, default=True,
                        help="Weight loss by inverse attribute prevalence. Use --no-attribute-weighting to disable.")
    return parser.parse_args()


def compute_attribute_weights(data_root: Path, image_ids: list[str], attributes: list[str]) -> torch.Tensor:
    """Dampened inverse-prevalence loss weights (sqrt), one per attribute.

    THE PROBLEM: oversampling (in build_train_dataset) duplicates whole IMAGES
    that contain a rare attribute, but per-pixel, even a duplicated image is
    still mostly background (streaks/negative_network cover very few pixels
    even when "present" - see the briefing's Sparse Attribute Masks slide).
    So oversampling alone doesn't fix the per-PIXEL imbalance that BCE sees.
    Weighting the loss itself is a separate, complementary fix: it directly
    tells the loss function "get this attribute wrong and it costs more",
    regardless of how many images contain it.

    WHY sqrt AND NOT RAW 1/prevalence: raw inverse-prevalence was tried first
    and collapsed pigment_network's Dice to ~0 by epoch 2 (see training log)
    while val_loss kept "improving" - the ~11x weight spread it produced was
    aggressive enough to make the model sacrifice a common, real attribute
    entirely in favour of rare ones. sqrt(1/prevalence) keeps rare attributes
    weighted higher without swinging that hard - see the weights= line below.

    WHY COMPUTE FROM image_ids RATHER THAN HARDCODE: hardcoding numbers from
    an earlier attribute_stats.py run risks silently going stale if the
    dataset or split changes. Computing it fresh, on the UNIQUE (pre-
    oversampling) training IDs only, keeps this honest and avoids leaking
    validation-set statistics into a training-time decision.

    Uses attribute_is_present() from preprocessing/attribute_stats.py directly
    (full resolution, no resize) rather than transforms.load_mask(), so a
    very sparse attribute mask can't have its only positive pixels destroyed
    by downsampling before we even count it as present.
    """
    counts = {name: 0 for name in attributes}
    for image_id in image_ids:
        for name in attributes:
            mask_path = data_root / "task2_gt" / f"{image_id}_attribute_{name}.png"
            if mask_path.exists() and attribute_is_present(mask_path):
                counts[name] += 1

    total = len(image_ids)

    # sanity check: if every attribute came back with zero real positives,
    # something upstream is broken (wrong path, ID format mismatch, etc) -
    # fail loudly here rather than silently returning uniform, meaningless
    # weights the way this function used to.
    if all(count == 0 for count in counts.values()):
        raise RuntimeError(
            "compute_attribute_weights found zero positive masks for every "
            "attribute. This almost always means the image IDs don't match "
            "the files on disk (e.g. missing zero-padding) or data_root is "
            "wrong - check with a manual mask_path.exists() before trusting "
            "any weights computed from this."
        )

    # clamp to avoid a divide-by-zero if some (but not all) attribute never appears at all
    prevalence = torch.tensor(
        [max(counts[name], 1) / total for name in attributes], dtype=torch.float32
    )
    # sqrt(1/prevalence) rather than raw 1/prevalence: raw inverse-prevalence
    # scales linearly with rarity, which for this dataset produces an ~11x
    # spread between the most and least common attribute (pigment_network
    # ~0.19 vs streaks ~2.17). That's aggressive enough to actively suppress
    # pigment_network - a first training run with the raw formula collapsed
    # pigment_network's Dice to ~0 by epoch 2 while "improving" val_loss,
    # because BCE on mostly-background pixels still looks good even when a
    # whole head has flatlined. sqrt keeps the same ranking (rare attributes
    # still weighted higher) but compresses the spread to roughly 3-4x,
    # which still gives streaks/negative_network a real boost without
    # sacrificing the common attributes to get there.
    weights = torch.sqrt(1.0 / prevalence)
    # normalise so weights average to 1.0 across attributes - keeps the overall
    # loss scale comparable to the unweighted (attribute_weights=None) case,
    # matching the convention already used in loss.py's own smoke test
    weights = weights / weights.sum() * len(attributes)
    return weights


def build_train_dataset(data_root: Path, splits_csv: Path):
    import pandas as pd
    from pathlib import Path

    #Reads the whole splits.csv into memory. .copy() makes an independent copy (avoids accidentally modifying data another part of the program might still be using) 
    split_df = pd.read_csv(splits_csv)
    split_df = split_df.copy()

    #Cleans up the split column: forces every value to text, strips stray whitespace, lowercases everything so "Train", "train ", and "TRAIN" all become "train".
    split_df["split"] = split_df["split"].astype(str).str.strip().str.lower()

    #Filters to only rows where split equals "train", grabs just the image_id column from those rows, converts to strings, and turns it into a plain Python list.
    train_ids = split_df.loc[split_df["split"] == "train", "image_id"].astype(str).tolist()

    #safety check: if list came back completely empty, stop immediately with a clear error, rather than silently training on zero images.
    if not train_ids:
        raise ValueError(f"No training IDs found in {splits_csv}. Check the split column values.")

    # Builds a small DataFrame containing just the image_id column, filtered to the training IDs, with .drop_duplicates() removing any accidental repeats.
    train_df = split_df.loc[split_df["image_id"].astype(str).isin(train_ids), ["image_id"]].drop_duplicates()

    # The image files on disk use zero-padded 6-digit names (e.g. 000001.jpg),
    # while the split CSV may contain plain integers (e.g. 1). We normalise the
    # IDs before writing the training CSV so the dataset loader can find them.
    train_df["image_id"] = train_df["image_id"].astype(str).str.strip().str.zfill(6)

    # Oversample images with rare/fragile attributes by duplicating them.
    # milia_like_cyst added after diagnosing a real failure: it went completely dead
    # (bit-identical val Dice for 17 straight epochs) in a run where it was the only
    # attribute helped by NEITHER oversampling NOR an above-average loss weight (its
    # weight, 0.81, sits below the 1.0 average since it's not rare enough by image-count
    # to earn much boost - yet its actual positive pixels per image are still small,
    # per the briefing's own sparse-attribute-mask examples). This gives it the same
    # extra training exposure already working for negative_network/streaks.
    rare_attributes = ["negative_network", "streaks", "milia_like_cyst"]  # Present in <25% of images
    oversampled_ids = list(train_df["image_id"])
    
    for image_id in train_df["image_id"]:
        # Check if this image has any rare attribute present
        for attr in rare_attributes:
            mask_path = data_root / "task2_gt" / f"{image_id}_attribute_{attr}.png"
            if mask_path.exists():
                # Load the mask and check if it has any positive pixels
                from preprocessing.transforms import load_mask
                mask = load_mask(mask_path, 512)
                if mask.sum() > 0:  # Attribute is present in this image
                    oversampled_ids.append(image_id)  # Duplicate this training example
                    break
    
    # Create DataFrame with oversampled IDs
    train_df_oversampled = pd.DataFrame({"image_id": oversampled_ids})

    #Writes this cleaned-up, training-only list out to a brand new CSV file, train_ids.csv, saved in the data folder. index=False means "don't add an extra unnamed column with row numbers."
    train_df_oversampled.to_csv(data_root / "train_ids.csv", index=False)
    
    #creates and returns an actual AttributeDataset object, pointing at the freshly-written CSV file, the data root, and the list of attribute names. This is what the training loop will actually use to load images and masks.
    return AttributeDataset(csv_file=data_root / "train_ids.csv", data_root=data_root, attributes=ATTRIBUTE_NAMES, augment=True)


def build_val_dataset(data_root: Path, splits_csv: Path):
    """Same logic as build_train_dataset, but for the validation split."""
    import pandas as pd

    split_df = pd.read_csv(splits_csv)
    split_df = split_df.copy()
    split_df["split"] = split_df["split"].astype(str).str.strip().str.lower()
    val_ids = split_df.loc[split_df["split"] == "val", "image_id"].astype(str).tolist()

    if not val_ids:
        raise ValueError(f"No validation IDs found in {splits_csv}. Check the split column values.")

    val_df = split_df.loc[split_df["image_id"].astype(str).isin(val_ids), ["image_id"]].drop_duplicates()
    val_df["image_id"] = val_df["image_id"].astype(str).str.strip().str.zfill(6)
    val_df.to_csv(data_root / "val_ids.csv", index=False)
    return AttributeDataset(csv_file=data_root / "val_ids.csv", data_root=data_root, attributes=ATTRIBUTE_NAMES)


def main() -> None:

    #Read the command-line arguments, then create the output folder if it doesn't already exist (parents=True: create any missing parent folders too; exist_ok=True: don't complain if it's already there).
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    #Converts the device string ("cuda" or "cpu") into an actual PyTorch device object.
    device = torch.device(args.device)

    #Builds both datasets using functions defined above
    train_dataset = build_train_dataset(args.data_root, args.splits_csv)
    val_dataset = build_val_dataset(args.data_root, args.splits_csv)

    #Wraps each dataset in a DataLoader, which handles grouping samples into batches.
    #shuffle=True for training (see different random orderings each epoch — helps the model not learn spurious patterns based on data order); 
    #shuffle=False for validation (no need to shuffle, since we're just measuring, not learning). 
    #num_workers=2 means 2 background processes load/resize images in parallel, ready before the GPU needs them (fixes earlier Colab bottleneck where num_workers=0 left the GPU idle waiting on the CPU)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    #Creates the actual model, using pretrained VGG16 weights, with the encoder unfrozen (trainable), 5 output heads, then moves it onto whichever device we picked.
    model = UNetMultiHead(pretrained=True, freeze_encoder=False, num_attributes=len(ATTRIBUTE_NAMES)).to(device)

    #Creates the optimizer: applies weight updates based on gradients. Adam is a specific, very commonly used update algorithm.
    # model.parameters() hands it every learnable weight in the whole model, so it knows what to update.
    #weight_decay applies L2 regularization: a small penalty proportional to each weight's own magnitude,
    #added to every update. This discourages weights from growing arbitrarily large just to fit training
    #data noise, which is one standard lever against the overfitting seen after epoch ~4 in the last run
    #(train_loss kept dropping while val Dice plateaued/worsened).
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    #Creates the loss function. Attribute weighting is a separate lever from
    #the oversampling already applied in build_train_dataset - see
    #compute_attribute_weights() for why both are needed. Toggle with
    #--no-attribute-weighting to A/B test against an unweighted run.
    if args.attribute_weighting:
        # .unique() recovers the original train IDs, undoing the duplicate
        # rows oversampling added - we want prevalence over real images,
        # not artificially inflated by how many times each was duplicated.
        # .astype(str).str.zfill(6) matters: pandas silently reads the zero-padded
        # "000001" strings in train_ids.csv back as int64 (dropping the leading
        # zeros) unless we force them back to text here - same fix dataset.py's
        # __getitem__ already applies per-item. Skipping this makes every
        # mask_path.exists() check below fail silently, which is exactly what
        # produced the all-1.0 weights bug.
        unique_train_ids = train_dataset.df["image_id"].astype(str).str.strip().str.zfill(6).unique().tolist()
        print("Computing attribute weights from training split...")
        attribute_weights = compute_attribute_weights(args.data_root, unique_train_ids, ATTRIBUTE_NAMES)
        print("  attribute weights:", dict(zip(ATTRIBUTE_NAMES, attribute_weights.tolist())))
    else:
        attribute_weights = None
        print("Attribute weighting disabled (--no-attribute-weighting).")

    criterion = DiceBCELoss(attribute_weights=attribute_weights).to(device)

    #Creates a learning rate scheduler: reduces LR if validation loss plateaus for N epochs (helps with noisy attributes like negative_network)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.scheduler_patience
    ) if args.scheduler_patience > 0 else None

    #Sets up tracking variables and the two file paths checkpoints will be saved to.
    best_mean_dice = 0.0
    start_epoch = 0
    last_checkpoint_path = args.output_dir / "task2_model.pth"
    best_checkpoint_path = args.output_dir / "task2_model_best.pth"
    
    #Early stopping tracking: stop if val_loss plateaus for N epochs
    best_dice_for_early_stop = 0.0
    early_stop_counter = 0

    # --- Resume from a previous checkpoint, if one was given ---
    if args.resume is not None:
        #Loads the saved checkpoint bundle, restores the model's weights and the 
        #optimizer's internal state (Adam tracks per-weight momentum (this restores that too, not just the raw weights), 
        #figures out which epoch to continue from (+1, since we don't want to redo the epoch that was already saved), and remembers what the best score was so far.
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1  # continue from the epoch AFTER the one saved
        best_mean_dice = checkpoint["best_mean_dice"]
        print(f"  Resuming at epoch {start_epoch + 1}, best_mean_dice so far = {best_mean_dice:.4f}")
 
        #afeguard: if you're resuming from, say, epoch 15, but only asked for --epochs 10 total, there's nothing left to do
        if start_epoch >= args.epochs:
            print(f"  Nothing to do: checkpoint is already at epoch {start_epoch}, "
                  f"which meets or exceeds --epochs {args.epochs}. Increase --epochs to continue training.")
            return

    for epoch in range(args.epochs): #loop over epochs
        # --- Training loop ---
        model.train() #mode switch for layers that behave differently during training vs. evaluation (e.g. dropout, batchnorm)
        running_loss = 0.0 #resets a running total back to zero at the start of every epoch

        for batch_idx, batch in enumerate(train_loader): #train_loader hands out one batch at a time (e.g. 8 images bundled together, per --batch-size). 
        #enumerate(...) gives a counter alongside each batch: batch_idx starts at 0 and goes up by 1 each time. 
        #batch_idx isn't actually used anywhere inside the loop (available e.g. for printing progress every N batches).

            #batch is a dictionary.
            images = batch["image"].to(device) #batch["image"] grabs the tensor of images
            targets = batch["attributes"].to(device) #batch["attributes"] grabs the tensor of ground-truth masks
            #.to(device moves each tensor onto whichever device you're using (GPU or CPU) — this has to happen before feeding them into the model, since the model itself is already sitting on that same device.

            #PyTorch accumulates gradients by default. Every time we call .backward(), it adds to whatever gradient values already exist, rather than replacing them. 
            #If we skipped this line, gradients from the previous batch would still be sitting there, getting mixed in with the new ones, corrupting your training. 
            #This line wipes the slate clean before computing anything new.
            optimizer.zero_grad()

            #Forward pass: run the model on the batch of input images, producing predicted masks. Then compute the loss by comparing those predictions to the ground-truth targets.
            preds = model(images)

            #COmputing loss with DiceBCELoss. Returns two things: the single scalar loss (for training), and a per-attribute breakdown (for logging).
            #_ throws away the per-attribute breakdown, since we don't actually use it for the training step (only validation logging later)
            loss, _ = criterion(preds, targets)

            #Backward pass: PyTorch has been silently tracking every mathematical operation that led from images all the way to this final loss number. 
            #.backward() walks back through all of that, calculating exactly how much each individual weight in the entire model contributed to the current loss and in which direction adjusting it would make the loss go down. 
            #These calculations get stored as .grad on each weight tensor. Nothing has actually changed yet at this point  (only calculates what changes should happen).
            loss.backward()

            #Gradient clipping: caps the total gradient norm across all weights at max_norm=1.0 BEFORE the
            #optimizer applies them. Added specifically to address a real failure seen in training: one
            #output head's gradient can occasionally spike large enough in a single batch to push its
            #weights into saturation (predictions permanently pinned near 0 or 1 everywhere), after which
            #that head effectively stops learning while the rest of the model trains normally - this is a
            #standard, safe fix for exactly that kind of single-batch instability.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            #model's weights get updated: using the gradients from .backward(), combined with the learning rate (args.lr) and Adam's specific update rule, every single weight shifts a small amount in the direction that should reduce the loss.
            optimizer.step()
            
            #loss is currently a PyTorch tensor (even though it holds just one number). .item() extracts that single number as a plain Python float. We add it onto our running total, so that after the whole epoch, we can compute an average.
            running_loss += loss.item()

        #len(train_loader) tells you how many batches existed this epoch. Dividing the total by that count gives you the average loss per batch for this whole epoch.
        train_loss = running_loss / max(1, len(train_loader))
        #Note: max(1, len(train_loader)) is a safety check. If the training set was empty (len(train_loader) == 0), it avoids a divide-by-zero crash by using 1 instead. This is unlikely to happen in practice

        # --- Validate ---
        model.eval() #switch to evaluation mode
        val_running_loss = 0.0

        #Two trackers, reset at the start of every epoch's validation pass: a running sum of Dice scores (one number per attribute, so this is a small tensor with 5 slots, not just one number), 
        #and a count of how many validation images we've processed so far.
        dice_sum = torch.zeros(len(ATTRIBUTE_NAMES))
        num_val_samples = 0


        with torch.no_grad(): #tells PyTorch "we're not training right now, so don't bother tracking gradients or building the computation graph". This saves memory and speeds up validation.
            
            #Same overall shape as training, but notice what's missing: no optimizer.zero_grad(), no loss.backward(), no optimizer.step(). We only compute predictions and measure how good they are — nothing about the model changes during validation.
            for batch in val_loader:
                images = batch["image"].to(device)
                targets = batch["attributes"].to(device)

                #forward pass and loss computation
                preds = model(images)
                loss, _ = criterion(preds, targets)
                val_running_loss += loss.item()

                #grabs the first dimension of the images tensor's shape (batch dimension). This tells us how many actual images were in this specific batch (the very last batch of an epoch is often smaller than the rest, if total image count doesn't divide evenly by batch_size).
                batch_size = images.size(0)

                #returns a small tensor of 5 numbers — the Dice score for this batch alone, one per attribute. .cpu() moves it off the GPU (if it was there) back onto CPU, since we're just accumulating numbers now, not doing GPU math.
                #Multiplying by batch_size weights this batch's contribution correctly — a batch of 8 images should count more toward the final average than, say, a leftover final batch of only 3 images. 
                #Adding it onto dice_sum accumulates this across every batch in the whole validation set.
                dice_sum += dice_score(preds, targets).cpu() * batch_size

                #running total of how many actual validation images have been processed, across all batches — needed to properly average at the end.
                num_val_samples += batch_size

        #Same averaging idea as train_loss — total validation loss divided by number of validation batches.
        val_loss = val_running_loss / max(1, len(val_loader))

        #divides by num_val_samples (total images), not len(val_loader) (total batches) — because dice_sum was already weighted by batch_size earlier, this correctly computes a true per-image average across all 540 validation images, not skewed by batch size.
        dice_per_attribute = dice_sum / max(1, num_val_samples)

        #dice_per_attribute is a tensor of 5 separate scores (one per attribute). .mean() collapses those 5 numbers into a single average. .item() pulls that single number out as a plain Python float.
        mean_dice = dice_per_attribute.mean().item()

        #Printing results for this epoch: the average training loss, average validation loss, and the mean Dice score across all attributes. Then prints a dictionary showing each attribute's individual Dice score.
        print(f"epoch {epoch + 1}/{args.epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} mean_val_dice={mean_dice:.4f}")
        print("  val dice per attribute:", dict(zip(ATTRIBUTE_NAMES, dice_per_attribute.tolist())))

        #Step the learning rate scheduler if enabled: reduce LR if validation loss hasn't improved for 'patience' epochs
        if scheduler is not None:
            scheduler.step(mean_dice)

        #Early stopping: stop if mean_val_dice hasn't improved for N epochs.
        #Tracks mean_dice, NOT val_loss - a real run showed val_loss can keep improving
        #for the entire 20 epochs (BCE on mostly-background pixels keeps looking "better"
        #even after a head has effectively stopped learning) while mean_val_dice - the
        #metric that actually matters for scoring - peaked early and got worse afterward.
        #An early-stop check on val_loss would never have triggered in that exact run.
        if args.early_stop_patience > 0:
            if mean_dice > best_dice_for_early_stop:
                best_dice_for_early_stop = mean_dice
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            
            if early_stop_counter >= args.early_stop_patience:
                print(f"Early stopping: mean_val_dice hasn't improved for {args.early_stop_patience} epochs. Stopping training.")
                break

        #Bundle everything needed to properly resume later:model weights, optimizer state, which epoch this is, and the best score (using this epoch's score if it's actually better than the previous best, otherwise keeping the old best).
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_mean_dice": best_mean_dice if mean_dice <= best_mean_dice else mean_dice,
        }

        #Always saves this bundle as the "latest" checkpoint, every single epoch: this is what --resume would load from if training gets interrupted.
        torch.save(checkpoint, last_checkpoint_path)

        #Separately, only if this epoch actually beat the previous best score, updates best_mean_dice and saves a second copy specifically as the "best" checkpoint.
        if mean_dice > best_mean_dice:
            best_mean_dice = mean_dice
            torch.save(checkpoint, best_checkpoint_path)
            print(f"  -> new best model saved (mean_val_dice={mean_dice:.4f})")

    print(f"Training complete. Last epoch saved to {last_checkpoint_path}")
    print(f"Best epoch (mean_val_dice={best_mean_dice:.4f}) saved to {best_checkpoint_path}")

    


if __name__ == "__main__":
    main()
