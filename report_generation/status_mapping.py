from config import MASK_STATUS_POLICY


def mask_to_status(
    attribute: str,
    has_positive_pixels: bool,
) -> str:
    """Convert a real Task 2 binary-mask result into a report status."""

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


if __name__ == "__main__":
    print(mask_to_status("pigment_network", True))
    print(mask_to_status("pigment_network", False))
    print(mask_to_status("negative_network", True))
    print(mask_to_status("streaks", False))