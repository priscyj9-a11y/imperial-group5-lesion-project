"""Evaluation metrics for Task 2 attribute masks.

my job is to compare predicted attribute masks with the real
ground-truth masks. This file contains the scores used for that comparison.

THE PROBLEM:
The model outputs a black/white mask for each dermoscopic attribute, but we
need a numeric score to say how close that predicted mask is to the real mask.

WHY IT MATTERS:
Task 2 attributes are highly imbalanced. Some attributes, such as streaks and
negative_network, are absent in most images. A single global score can hide
poor performance on rare attributes, so we calculate metrics per attribute.

WHERE IT FITS:
evaluate_validation.py imports these functions and applies them to every
predicted mask vs ground-truth mask pair.
"""

from __future__ import annotations

import numpy as np


def _as_bool(mask: np.ndarray) -> np.ndarray:
    """Convert a mask to True/False pixels.

    1 or white pixels become True.
    0 or black pixels become False.
    """
    return mask.astype(bool)


def dice_score(prediction: np.ndarray, ground_truth: np.ndarray, eps: float = 1e-7) -> float:
    """Measure overlap using Dice score.

    Dice answers:
        How much do the predicted mask and the real mask overlap?

    Score meaning:
        1.0 = perfect match
        0.0 = no overlap

    Special case:
        If both masks are empty, the prediction is correct for an absent
        attribute, so we return 1.0.
    """
    pred = _as_bool(prediction)
    gt = _as_bool(ground_truth)

    intersection = np.logical_and(pred, gt).sum()
    total_white_pixels = pred.sum() + gt.sum()

    if total_white_pixels == 0:
        return 1.0

    return float((2 * intersection + eps) / (total_white_pixels + eps))


def iou_score(prediction: np.ndarray, ground_truth: np.ndarray, eps: float = 1e-7) -> float:
    """Measure overlap using IoU, also called Jaccard index.

    IoU means Intersection over Union.

    Intersection:
        Pixels that are white in both prediction and ground truth.

    Union:
        Pixels that are white in prediction OR ground truth.
    """
    pred = _as_bool(prediction)
    gt = _as_bool(ground_truth)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return 1.0

    return float((intersection + eps) / (union + eps))


def precision_score(prediction: np.ndarray, ground_truth: np.ndarray, eps: float = 1e-7) -> float:
    """Measure how reliable the model's white pixels are.

    Precision answers:
        When the model predicts an attribute pixel, how often is it correct?

    Low precision usually means the model predicts too much white area.
    """
    pred = _as_bool(prediction)
    gt = _as_bool(ground_truth)

    true_positive = np.logical_and(pred, gt).sum()
    predicted_positive = pred.sum()

    if predicted_positive == 0:
        return 1.0 if gt.sum() == 0 else 0.0

    return float((true_positive + eps) / (predicted_positive + eps))


def recall_score(prediction: np.ndarray, ground_truth: np.ndarray, eps: float = 1e-7) -> float:
    """Measure how much of the real attribute the model found.

    Recall answers:
        Of all real attribute pixels, how many did the model detect?

    Low recall usually means the model misses true attribute regions.
    """
    pred = _as_bool(prediction)
    gt = _as_bool(ground_truth)

    true_positive = np.logical_and(pred, gt).sum()
    real_positive = gt.sum()

    if real_positive == 0:
        return 1.0 if pred.sum() == 0 else 0.0

    return float((true_positive + eps) / (real_positive + eps))


def all_metrics(prediction: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
    """Return all Task 2 metrics for one predicted/real mask pair."""
    return {
        "dice": dice_score(prediction, ground_truth),
        "iou": iou_score(prediction, ground_truth),
        "precision": precision_score(prediction, ground_truth),
        "recall": recall_score(prediction, ground_truth),
    }
