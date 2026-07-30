"""Load Task 2 probabilities and predicted attribute masks."""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .config import (
    ATTRIBUTES,
    TASK2_CSV_COLUMNS,
    TASK2_FILE_NAMES,
)
from .status_mapping import mask_to_status


def normalise_image_id(value: object) -> str:
    """Convert image IDs into the canonical six-digit format."""

    image_id = str(value).strip()

    if image_id.upper().startswith("ISIC_"):
        image_id = image_id[5:]

    if image_id.endswith(".0"):
        image_id = image_id[:-2]

    return image_id.zfill(6)


def load_attribute_probabilities(
    csv_path: str | Path,
) -> pd.DataFrame:
    """Load and validate Rita's Task 2 probability CSV."""

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Attribute probability CSV was not found: {csv_path}"
        )

    dataframe = pd.read_csv(
        csv_path,
        dtype={"image_id": str},
    )

    if "image_id" not in dataframe.columns:
        raise ValueError(
            "The probability CSV does not contain an image_id column."
        )

    dataframe["image_id"] = dataframe["image_id"].map(
        normalise_image_id
    )

    required_columns = [
        "image_id",
        *TASK2_CSV_COLUMNS.values(),
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing Task 2 CSV columns: {missing_columns}"
        )

    if dataframe["image_id"].duplicated().any():
        duplicate_ids = dataframe.loc[
            dataframe["image_id"].duplicated(),
            "image_id",
        ].tolist()

        raise ValueError(
            f"Duplicate image IDs found: {duplicate_ids}"
        )

    for column in TASK2_CSV_COLUMNS.values():
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="raise",
        )

        if not dataframe[column].between(0.0, 1.0).all():
            raise ValueError(
                f"{column} contains values outside 0 to 1."
            )

    return dataframe


def load_binary_attribute_mask(
    mask_path: str | Path,
    expected_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Load one Task 2 predicted attribute mask."""

    mask_path = Path(mask_path)

    if not mask_path.exists():
        raise FileNotFoundError(
            f"Task 2 mask was not found: {mask_path}"
        )

    mask = np.array(
        Image.open(mask_path).convert("L")
    )

    binary_mask = mask > 127

    if (
        expected_shape is not None
        and binary_mask.shape != expected_shape
    ):
        raise ValueError(
            f"Mask shape mismatch for {mask_path}. "
            f"Received {binary_mask.shape}; "
            f"expected {expected_shape}."
        )

    return binary_mask


def extract_attribute_evidence(
    image_id: str,
    probability_row: pd.Series,
    prediction_root: str | Path,
    expected_shape: tuple[int, int],
) -> tuple[dict[str, float], dict[str, str]]:
    """Extract Task 2 scores and controlled report statuses."""

    prediction_root = Path(prediction_root)

    probabilities: dict[str, float] = {}
    statuses: dict[str, str] = {}

    for attribute in ATTRIBUTES:
        csv_column = TASK2_CSV_COLUMNS[attribute]
        file_name = TASK2_FILE_NAMES[attribute]

        probability = float(
            probability_row[csv_column]
        )

        mask_path = (
            prediction_root
            / image_id
            / f"{file_name}.png"
        )

        mask = load_binary_attribute_mask(
            mask_path,
            expected_shape=expected_shape,
        )

        has_positive_pixels = bool(mask.any())

        probabilities[attribute] = probability

        statuses[attribute] = mask_to_status(
            attribute=attribute,
            has_positive_pixels=has_positive_pixels,
        )

    return probabilities, statuses