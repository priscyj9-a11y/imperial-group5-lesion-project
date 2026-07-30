"""CLIP retrieval bonus for the dermoscopy project.

THE GOAL:
For each query/test image, find the most visually similar training images using
CLIP embeddings, then save the neighbour IDs in a CSV audit file.

We do not train a new model. CLIP is already pretrained, so this script only:
1. loads images,
2. converts them into CLIP feature vectors,
3. compares vectors with cosine similarity,
4. writes the top-k nearest neighbours.

EXPECTED OUTPUT:
bonus/test_bonus_clip.csv with columns:
    query_image,neighbor_id,similarity

This matches the bonus audit-file idea from the project briefing.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def collect_images(folder: Path) -> list[Path]:
    """Collect image files from a folder in a stable order.

    THE PROBLEM:
    We need to embed every training image and every query/test image.

    WHY SORTED:
    Sorting makes the output reproducible: the same files are processed in the
    same order on every machine.
    """
    if not folder.is_dir():
        raise FileNotFoundError(f"Image folder not found: {folder}")

    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_id_from_path(path: Path) -> str:
    """Return the image ID without extension.

    Example:
        000001.jpg -> 000001
    """
    return path.stem


def load_clip_model(model_name: str, pretrained: str, device: torch.device):
    """Load a pretrained CLIP model through open_clip.

    IMPORTANT:
    This requires the package open_clip_torch. If it is not installed, the script
    stops with a clear message instead of failing mysteriously.
    """
    try:
        import open_clip
    except ImportError as error:
        raise ImportError(
            "open_clip_torch is required for the bonus retrieval script.\n"
            "Install it in the Summerschool environment with:\n"
            "pip install open_clip_torch"
        ) from error

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
    )
    model = model.to(device)
    model.eval()
    return model, preprocess


def encode_images(
    image_paths: list[Path],
    model,
    preprocess,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Convert images into normalized CLIP embeddings.

    Each embedding is a vector representing the image content. Similar images
    should have vectors pointing in similar directions.
    """
    embeddings: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            batch_images = []

            for path in batch_paths:
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    batch_images.append(preprocess(image))

            batch_tensor = torch.stack(batch_images).to(device)
            features = model.encode_image(batch_tensor)

            # Normalize so dot product becomes cosine similarity.
            features = features / features.norm(dim=-1, keepdim=True)
            embeddings.append(features.cpu().numpy())

            print(f"Encoded {min(start + batch_size, len(image_paths))}/{len(image_paths)} images")

    return np.concatenate(embeddings, axis=0)


def top_k_neighbors(
    query_embeddings: np.ndarray,
    index_embeddings: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Find top-k most similar index images for each query image.

    Cosine similarity is a number from roughly -1 to 1.
    Higher means more similar. Because embeddings are normalized, matrix
    multiplication gives cosine similarity.
    """
    similarities = query_embeddings @ index_embeddings.T

    # argsort returns indices from low to high, so we take the last top_k and
    # reverse them to get high-to-low similarity order.
    neighbor_indices = np.argsort(similarities, axis=1)[:, -top_k:][:, ::-1]
    neighbor_scores = np.take_along_axis(similarities, neighbor_indices, axis=1)
    return neighbor_indices, neighbor_scores


def write_bonus_csv(
    output_csv: Path,
    query_paths: list[Path],
    index_paths: list[Path],
    neighbor_indices: np.ndarray,
    neighbor_scores: np.ndarray,
    neighbor_prefix: str,
) -> None:
    """Write the final bonus audit CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["query_image", "neighbor_id", "similarity"])

        for query_row, query_path in enumerate(query_paths):
            query_id = image_id_from_path(query_path)

            for rank in range(neighbor_indices.shape[1]):
                index_path = index_paths[int(neighbor_indices[query_row, rank])]
                score = float(neighbor_scores[query_row, rank])

                # The example_result.zip uses paths like data/images\001573.jpg.
                # Keep this prefix configurable in case the team chooses a
                # slightly different submission path.
                neighbor_id = f"{neighbor_prefix}{index_path.name}"
                writer.writerow([query_id, neighbor_id, score])


def main() -> None:
    parser = argparse.ArgumentParser(description="Create CLIP retrieval CSV for the project bonus.")
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path(r"C:\Users\Ivana\Downloads\summer_school_project_train\train\images"),
        help="Folder of training/reference images to search through.",
    )
    parser.add_argument(
        "--query-dir",
        type=Path,
        required=True,
        help="Folder of query/test images that need nearest neighbours.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("bonus") / "test_bonus_clip.csv",
        help="Where to save the bonus audit CSV.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of neighbours per query image.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument(
        "--neighbor-prefix",
        default=r"data/images\\",
        help="Prefix written before each neighbour filename in the CSV.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    index_paths = collect_images(args.index_dir)
    query_paths = collect_images(args.query_dir)
    print(f"Reference/train images: {len(index_paths)}")
    print(f"Query/test images     : {len(query_paths)}")

    model, preprocess = load_clip_model(args.model_name, args.pretrained, device)

    print("\nEncoding reference/train images...")
    index_embeddings = encode_images(index_paths, model, preprocess, device, args.batch_size)

    print("\nEncoding query/test images...")
    query_embeddings = encode_images(query_paths, model, preprocess, device, args.batch_size)

    neighbor_indices, neighbor_scores = top_k_neighbors(
        query_embeddings,
        index_embeddings,
        args.top_k,
    )

    write_bonus_csv(
        args.output_csv,
        query_paths,
        index_paths,
        neighbor_indices,
        neighbor_scores,
        args.neighbor_prefix,
    )

    print(f"\nWrote bonus retrieval CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
