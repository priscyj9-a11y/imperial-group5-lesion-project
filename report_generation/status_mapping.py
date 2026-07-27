from config import ABSENT_THRESHOLD, PRESENT_THRESHOLD

def probability_to_status(probability: float) -> str:
    """Convert an attribute probability into a report status."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"Probability must be between 0 and 1, received {probability}."
        )

    if probability >= PRESENT_THRESHOLD:
        return "present"

    if probability <= ABSENT_THRESHOLD:
        return "absent"

    return "uncertain"

if __name__ == "__main__":
    print(probability_to_status(0.81))  # expected: present
    print(probability_to_status(0.12))  # expected: absent
    print(probability_to_status(0.51))  # expected: uncertain