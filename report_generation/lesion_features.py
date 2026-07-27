from pathlib import Path

import cv2
import numpy as np


def load_binary_mask(
    mask_path: str | Path,
) -> np.ndarray:
    """Load a lesion mask and convert it into binary values."""

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if mask is None:
        raise FileNotFoundError(
            f"Could not load lesion mask: {mask_path}"
        )

    return (mask > 127).astype(np.uint8)


def calculate_size_category(
    mask: np.ndarray,
) -> tuple[float, str]:
    """Calculate the lesion area ratio and size category."""

    if mask.size == 0:
        raise ValueError("The lesion mask is empty.")

    lesion_pixels = int(mask.sum())
    area_ratio = lesion_pixels / mask.size

    if area_ratio < 0.08:
        size_category = "small"
    elif area_ratio <= 0.25:
        size_category = "moderate"
    else:
        size_category = "large"

    return float(area_ratio), size_category


def calculate_border_category(
    mask: np.ndarray,
) -> tuple[float, str]:
    """Calculate a simple border irregularity score."""

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        raise ValueError(
            "No lesion contour was found in the mask."
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
            "The lesion contour has no valid area."
        )

    irregularity = (
        perimeter**2
    ) / (
        4 * np.pi * area
    )

    border_category = (
        "irregular"
        if irregularity >= 1.60
        else "regular"
    )

    return float(irregularity), border_category