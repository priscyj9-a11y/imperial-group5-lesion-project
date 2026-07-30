"""Simple RAG-style attribute summary for the CLIP bonus.

THE GOAL:
After CLIP retrieves similar training images, this script checks the real Task 2
ground-truth masks of those retrieved neighbours.

WHY THIS IS RAG-STYLE:
Retrieval = CLIP gives us similar training images.
Augmentation = we read reliable information from those neighbours: their real
Task 2 attribute masks.
Generation = we write a short evidence sentence for each query/test image.

IMPORTANT:
This does not need Task 2 predictions yet. Tomorrow, when the group has Task 2
predictions for the test set, this script can be extended to compare:
    predicted test attributes vs attributes found in retrieved neighbours.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from task2_eval.mask_io import ATTRIBUTE_NAMES, ground_truth_mask_path


def load_shared_mask(mask_path: Path):
    """Load a mask using the group's shared preprocessing function.

    The shared file preprocessing/transforms.py is expected to contain
    load_mask(). It is not copied inside this bonus folder because the team
    should use one common preprocessing rule everywhere.

    The import happens inside this function so that `--help` still works even
    before the shared preprocessing folder has been added locally.
    """
    try:
        from preprocessing.transforms import load_mask
    except ImportError as error:
        raise ImportError(
            "Could not import preprocessing.transforms.load_mask.\n"
            "Add the team's shared preprocessing folder to the project first.\n"
            "Expected file: preprocessing/transforms.py"
        ) from error

    return load_mask(mask_path)


def image_id_from_neighbor(neighbor_id: str) -> str:
    """Extract the training image ID from the neighbour value in the CSV.

    The retrieval CSV may store neighbours like:
        data/images\\000245.jpg

    For the Task 2 masks, we only need:
        000245
    """
    cleaned = neighbor_id.replace("\\", "/")
    return Path(cleaned).stem


def read_retrieval_csv(csv_path: Path) -> dict[str, list[str]]:
    """Read the CLIP retrieval CSV and group neighbour IDs by query image.

    Returns:
        {
            "test_001": ["000245", "001876", "000982"],
            "test_002": ["000104", "002111", "000732"],
        }
    """
    grouped: dict[str, list[str]] = defaultdict(list)

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            query_id = row["query_image"]
            neighbour_image_id = image_id_from_neighbor(row["neighbor_id"])
            grouped[query_id].append(neighbour_image_id)

    return grouped


def is_attribute_present(data_root: Path, image_id: str, attribute: str) -> bool:
    """Check whether one attribute is present in one training image.

    A Task 2 attribute mask is binary:
        0 = attribute absent / background
        1 = attribute present

    If the mask contains at least one pixel equal to 1, the attribute is present.
    """
    mask_path = ground_truth_mask_path(data_root, image_id, attribute)
    mask = load_shared_mask(mask_path)
    return bool(mask.max() == 1)


def count_neighbour_attributes(data_root: Path, neighbour_ids: list[str]) -> dict[str, int]:
    """Count how often each attribute appears in the retrieved neighbours."""
    counts = {attribute: 0 for attribute in ATTRIBUTE_NAMES}

    for image_id in neighbour_ids:
        for attribute in ATTRIBUTE_NAMES:
            if is_attribute_present(data_root, image_id, attribute):
                counts[attribute] += 1

    return counts


def build_evidence_sentence(counts: dict[str, int], total_neighbours: int) -> str:
    """Create a short sentence summarising the retrieved-neighbour evidence."""
    present_parts = []
    absent_parts = []

    for attribute in ATTRIBUTE_NAMES:
        count = counts[attribute]
        part = f"{attribute} {count}/{total_neighbours}"

        if count > 0:
            present_parts.append(part)
        else:
            absent_parts.append(attribute)

    if present_parts:
        evidence = "Retrieved-neighbour evidence: " + ", ".join(present_parts) + "."
    else:
        evidence = "Retrieved-neighbour evidence: none of the five attributes appeared in the retrieved neighbours."

    if absent_parts:
        evidence += " Not observed: " + ", ".join(absent_parts) + "."

    return evidence


def write_summary_csv(
    output_csv: Path,
    retrieval_groups: dict[str, list[str]],
    data_root: Path,
) -> None:
    """Write one RAG-style evidence summary per query/test image."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "query_image",
        "neighbour_ids",
        "n_neighbours",
        *[f"{attribute}_count" for attribute in ATTRIBUTE_NAMES],
        "evidence_summary",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for query_id, neighbour_ids in retrieval_groups.items():
            counts = count_neighbour_attributes(data_root, neighbour_ids)
            evidence_summary = build_evidence_sentence(counts, len(neighbour_ids))

            row = {
                "query_image": query_id,
                "neighbour_ids": ";".join(neighbour_ids),
                "n_neighbours": len(neighbour_ids),
                "evidence_summary": evidence_summary,
            }

            for attribute in ATTRIBUTE_NAMES:
                row[f"{attribute}_count"] = counts[attribute]

            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create RAG-style attribute summaries from CLIP neighbours.")
    parser.add_argument(
        "--retrieval-csv",
        type=Path,
        default=Path("bonus") / "test_bonus_clip.csv",
        help="CSV created by bonus_retrieval.clip_retrieval.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"C:\Users\Ivana\Downloads\summer_school_project_train\train"),
        help="Training dataset folder containing images/, task1_gt/, and task2_gt/.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("bonus") / "test_bonus_rag_summary.csv",
        help="Where to save the RAG-style summary CSV.",
    )
    args = parser.parse_args()

    retrieval_groups = read_retrieval_csv(args.retrieval_csv)
    write_summary_csv(args.output_csv, retrieval_groups, args.data_root)

    print(f"Query images summarised: {len(retrieval_groups)}")
    print(f"Wrote RAG-style summary CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
