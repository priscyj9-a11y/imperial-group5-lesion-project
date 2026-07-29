from pathlib import Path

import cv2
import numpy as np

from config import (
    IRREGULAR_BORDER_THRESHOLD,
    MODERATE_LESION_MAX_RATIO,
    SMALL_LESION_MAX_RATIO,
)


def load_binary_mask(
    mask_path: str | Path,
) -> np.ndarray:
    """Load a lesion mask and convert it into binary values 0 and 1."""

    mask_path = Path(mask_path)

    if not mask_path.exists():
        raise FileNotFoundError(
            f"Lesion mask was not found: {mask_path}"
        )

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if mask is None:
        raise ValueError(
            f"Lesion mask could not be opened: {mask_path}"
        )

    return (mask > 127).astype(np.uint8)


def calculate_size_category(
    mask: np.ndarray,
) -> tuple[float, str]:
    """Calculate lesion area ratio and its controlled size description."""

    if mask.size == 0:
        raise ValueError("The lesion mask is empty.")

    lesion_pixels = int(mask.sum())
    area_ratio = lesion_pixels / mask.size

    if area_ratio < SMALL_LESION_MAX_RATIO:
        category = "small"
    elif area_ratio <= MODERATE_LESION_MAX_RATIO:
        category = "moderate"
    else:
        category = "large"

    return float(area_ratio), category


def calculate_border_category(
    mask: np.ndarray,
) -> tuple[float, str]:
    """Calculate a simple contour-based border irregularity score."""

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        raise ValueError(
            "No lesion contour was found in the predicted mask."
        )

    largest_contour = max(
        contours,
        key=cv2.contourArea,
    )

    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(
        largest_contour,
        closed=True,
    )

    if area <= 0:
        raise ValueError(
            "The predicted lesion contour has no valid area."
        )

    irregularity = (
        perimeter**2
    ) / (
        4 * np.pi * area
    )

    category = (
        "irregular"
        if irregularity >= IRREGULAR_BORDER_THRESHOLD
        else "regular"
    )

    return float(irregularity), category