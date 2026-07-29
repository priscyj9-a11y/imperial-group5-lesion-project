"""Calculate lesion features from Task 1 predicted masks."""

from pathlib import Path

import cv2
import numpy as np

from .config import (
    IRREGULAR_BORDER_THRESHOLD,
    MODERATE_LESION_MAX_RATIO,
    SMALL_LESION_MAX_RATIO,
)


def load_binary_mask(
    mask_path: str | Path,
) -> np.ndarray:
    """Load a Task 1 lesion mask as a binary 0/1 array."""

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
    """Calculate lesion area ratio and controlled size category."""

    if mask.size == 0:
        raise ValueError(
            "The lesion-mask array contains no pixels."
        )

    lesion_pixels = int(mask.sum())
    area_ratio = lesion_pixels / mask.size

    # An empty Task 1 prediction is not the same as a small lesion.
    if lesion_pixels == 0:
        return 0.0, "undetermined"

    if area_ratio < SMALL_LESION_MAX_RATIO:
        size_category = "small"

    elif area_ratio <= MODERATE_LESION_MAX_RATIO:
        size_category = "moderate"

    else:
        size_category = "large"

    return float(area_ratio), size_category


def calculate_border_category(
    mask: np.ndarray,
) -> tuple[float | None, str]:
    """Calculate contour-based border irregularity.

    Empty or invalid Task 1 masks are marked as undetermined.
    """

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None, "undetermined"

    largest_contour = max(
        contours,
        key=cv2.contourArea,
    )

    contour_area = cv2.contourArea(
        largest_contour
    )

    contour_perimeter = cv2.arcLength(
        largest_contour,
        closed=True,
    )

    if contour_area <= 0 or contour_perimeter <= 0:
        return None, "undetermined"

    irregularity_score = (
        contour_perimeter**2
    ) / (
        4 * np.pi * contour_area
    )

    border_category = (
        "irregular"
        if irregularity_score >= IRREGULAR_BORDER_THRESHOLD
        else "regular"
    )

    return float(irregularity_score), border_category