"""Convert Task 2 mask evidence into controlled report statuses."""

from .config import MASK_STATUS_POLICY


def mask_to_status(
    attribute: str,
    has_positive_pixels: bool,
) -> str:
    """Convert one binary attribute mask into a report status."""

    if attribute not in MASK_STATUS_POLICY:
        raise ValueError(
            f"Unknown attribute: {attribute}"
        )

    mask_state = (
        "positive"
        if has_positive_pixels
        else "negative"
    )

    return MASK_STATUS_POLICY[attribute][mask_state]