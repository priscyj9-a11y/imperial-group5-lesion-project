"""Reorganise exported predictions into the submission deliverable format.

THE PROBLEM:
predictions.py (correctly, for working with Ivana's evaluation code) writes:
    prediction_root/{image_id}/{attribute}.png

But the project briefing's deliverables list asks for:
    "Attribute Masks: PNG files in 5 separate folders"
i.e. one folder PER ATTRIBUTE, containing all images:
    submission_root/{attribute}/{image_id}.png

WHERE IT FITS:
Run this AFTER predictions.py has already exported predictions in the
per-image-ID format. This does not re-run the model — it only copies
already-generated PNGs into the submission folder layout.
"""

from __future__ import annotations #Purely a compatibility detail, doesn't affect behavior (lets you write type hints like list[str] (lowercase list))

import argparse #For reading command-line arguments
import shutil #Python's standard library for file operations like copying
from pathlib import Path

#Pulls in the list of 5 attribute names, and the function that knows how to build a prediction's file path — both borrowed from mask_io.py
from attribute_detection.task2_eval.mask_io import ATTRIBUTE_NAMES, prediction_mask_path


def reorganise_for_submission(prediction_root: Path, submission_root: Path, image_ids: list[str]) -> None:
    """Copy predictions into the submission folder layout."""
    #Takes: where the existing per-image predictions already live, where the new per-attribute folders should be created, and which image IDs to process.
    # Loops over all 5 attribute names, creating a folder for each one upfront — e.g. submission_root/pigment_network/, submission_root/streaks/, etc. 
    #parents=True means "create any missing parent folders too, not just this one." exist_ok=True means "don't complain if the folder's already there" — without it, re-running this script a second time would crash on the second attempt.
    for attribute in ATTRIBUTE_NAMES:
        (submission_root / attribute).mkdir(parents=True, exist_ok=True)

    #Two trackers: a list to collect any files that turned out to be missing, and a counter for how many files successfully got copied — both purely for the summary printed at the end.
    missing = []
    copied = 0

    for image_id in image_ids: #A nested loop — for every image, check every one of its 5 attributes.
        for attribute in ATTRIBUTE_NAMES:
            source_path = prediction_mask_path(prediction_root, image_id, attribute) #Builds the existing path where predictions.py already saved this specific image+attribute's prediction — using the shared helper function, so this always matches exactly what predictions.py actually wrote, with zero risk of the two files disagreeing about the naming convention.
            dest_path = submission_root / attribute / f"{image_id}.png" #Builds the new path — this time, attribute name first (folder), then image ID (filename) — the reversed structure the Project Briefing asks for.

            if not source_path.exists(): #If the expected prediction file genuinely isn't there (maybe predictions.py hasn't been run for every image yet), record it as missing and continue — skip straight to the next iteration of the loop, without crashing the whole script over one missing file.
                missing.append(str(source_path))
                continue

            #Actually copies the file from the old location to the new one. copy2 (as opposed to plain copy) also preserves the original file's timestamp metadata — a small nicety, not essential here, but harmless. Importantly: this copies, it does not move or delete — your original per-image-ID predictions stay completely untouched, so Ivana's evaluation scripts keep working exactly as before.
            shutil.copy2(source_path, dest_path)  # copy2 preserves timestamps, doesn't move/delete the original
            copied += 1

    #A final summary. Note it warns about missing files rather than crashing — a deliberate choice, so one incomplete image doesn't prevent you from getting the other 2699 correctly copied files. missing[:5] just means "show at most the first 5 examples," so the terminal doesn't get flooded if hundreds were missing.
    print(f"Copied {copied} files into: {submission_root}")
    if missing:
        print(f"WARNING: {len(missing)} expected prediction files were missing, e.g.:")
        for path in missing[:5]:
            print(f"  {path}")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more")


def read_validation_ids(splits_csv: Path) -> list[str]:
    """Read splits.csv and keep only IDs where split == val."""
    import csv

    validation_ids: list[str] = [] #An empty list that'll collect image IDs, with a type hint saying "this will hold strings"
    with splits_csv.open("r", newline="", encoding="utf-8") as file: #Opens the CSV file for reading, with newline="" to avoid Python's automatic newline translation (which can break CSV parsing), and UTF-8 encoding to handle any non-ASCII characters in the file.
        reader = csv.DictReader(file) #DictReader reads the CSV file and lets you access each row as a dictionary, where the keys are the column names (e.g. row["image_id"]).
        
        #A safety check — confirms the CSV actually has both required column headers before trying to use them; if not, fails immediately with a clear error message rather than a confusing crash later.~
        required = {"image_id", "split"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError("splits.csv must contain columns: image_id, split")

        #Goes through every row; if that row's split column says "val" (after stripping whitespace and lowercasing, to be tolerant of things like "Val " or "VAL"), grab its image_id, clean it up, zero-pad it to 6 digits, and add it to the list.
        for row in reader:
            if row["split"].strip().lower() == "val":
                validation_ids.append(row["image_id"].strip().zfill(6))
    return validation_ids


def main() -> None:
    #Sets up 3 required command-line arguments. required=True means the script refuses to run at all if you forget to supply one of them — you'll get a clear error rather than a mysterious crash later.
    parser = argparse.ArgumentParser(description="Reorganise predictions into 5-folders-per-attribute submission format.")
    parser.add_argument("--prediction-root", type=Path, required=True, help="Existing predictions from predictions.py: prediction_root/image_id/attribute.png")
    parser.add_argument("--submission-root", type=Path, required=True, help="Output folder: submission_root/attribute/image_id.png")
    parser.add_argument("--splits-csv", type=Path, required=True, help="CSV with image_id,split columns, used to list which images to include.")
    args = parser.parse_args()

    #Reads the validation IDs, prints a heads-up of how many are about to be processed, then calls the actual reorganizing function with everything it needs.
    image_ids = read_validation_ids(args.splits_csv)
    print(f"Reorganising predictions for {len(image_ids)} validation images...")

    reorganise_for_submission(args.prediction_root, args.submission_root, image_ids)


if __name__ == "__main__":
    main()