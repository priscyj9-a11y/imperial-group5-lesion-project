"""Create the fixed train/validation split for the whole team.

Writes one CSV that everyone reads, so all five of us train and evaluate on
exactly the same images. Read-only on the dataset - only writes the CSV.
"""

import argparse
import csv
import random
from pathlib import Path


def collect_image_ids(images_dir: Path) -> list[str]:
    """Get every image ID from the images folder, sorted.

    WHY SORTED: the order files come back from the OS isn't guaranteed to be
    the same on every machine. sorting first means everyone starts from the
    identical list, which is what makes the seed reproducible.
    """
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Folder not found: {images_dir}")
    # .stem = filename without extension, so '000001.jpg' -> '000001'
    return sorted(p.stem for p in images_dir.glob("*.jpg"))


def make_splits(image_ids: list[str], val_fraction: float, seed: int) -> dict[str, str]:
    """Randomly assign each image ID to 'train' or 'val'.

    THE PROBLEM: we need a validation set, but it has to be the SAME one for
    everyone, every time, on every machine.
    WHY A FIXED SEED: random.shuffle with a fixed seed produces the exact same
    shuffle every run. Same seed = same split = comparable results. Change the
    seed and everyone's numbers stop matching.
    WHY SPLIT BY IMAGE ID: each ID owns 1 photo + 6 masks. Splitting by ID
    keeps a case whole. If the same lesion appeared in both train and val, the
    model would have already seen the answer -- that's data leakage, and it
    makes validation scores meaningless.
    """
    # copy the list so we don't shuffle the caller's original
    shuffled = list(image_ids)

    # seed the random generator, THEN shuffle. this order matters.
    random.seed(seed)
    random.shuffle(shuffled)

    # how many go to validation. int() rounds down.
    n_val = int(len(shuffled) * val_fraction)

    # the first n_val become validation, the rest become training
    val_ids = set(shuffled[:n_val])

    # build a dictionary: {'000001': 'train', '000002': 'val', ...}
    return {i: ("val" if i in val_ids else "train") for i in image_ids}


def write_splits(assignments: dict[str, str], output_path: Path) -> None:
    """Write the split to a CSV everyone on the team can read."""
    # make the folder if it doesn't exist yet. parents=True makes any missing
    # parent folders too. exist_ok=True means "don't error if it's already there".
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # newline="" is required on Windows or the CSV gets blank lines between rows
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "split"])   # the header row
        # sorted() so the file is in ID order and easy for a human to read
        for image_id in sorted(assignments):
            writer.writerow([image_id, assignments[image_id]])


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the fixed train/val split.")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Folder containing images/, task1_gt/, task2_gt/")
    parser.add_argument("--output", type=Path, default=Path("data/splits.csv"),
                        help="Where to write the split CSV")
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="Fraction held out for validation (0.2 = 20%%)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed. DO NOT CHANGE - everyone uses 42.")
    args = parser.parse_args()

    image_ids = collect_image_ids(args.data_root / "images")
    assignments = make_splits(image_ids, args.val_fraction, args.seed)
    write_splits(assignments, args.output)

    # count how many landed in each split so we can eyeball it
    n_train = sum(1 for v in assignments.values() if v == "train")
    n_val = sum(1 for v in assignments.values() if v == "val")

    print(f"Total images : {len(image_ids)}")
    print(f"  train      : {n_train}")
    print(f"  val        : {n_val}")
    print(f"  seed       : {args.seed}")
    print(f"Written to   : {args.output}")

    # safety check: every ID must appear exactly once, in exactly one split.
    # this is the leakage check - if it ever fails, stop and investigate.
    assert n_train + n_val == len(image_ids), "IDs went missing!"
    assert len(assignments) == len(set(image_ids)), "duplicate IDs!"
    print("Leakage check: OK - every image is in exactly one split")


if __name__ == "__main__":
    main()