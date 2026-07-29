"""Shared configuration for Task 3 report generation."""


# Canonical names used in Task 3 JSON and CSV outputs
ATTRIBUTES = [
    "pigment_network",
    "negative_network",
    "streaks",
    "milia_like_cysts",
    "globules",
]


# Names used in the written report
DISPLAY_NAMES = {
    "pigment_network": "Pigment network",
    "negative_network": "negative network",
    "streaks": "streaks",
    "milia_like_cysts": "milia-like cysts",
    "globules": "globules",
}


# Grammar used in the written report
ATTRIBUTE_VERBS = {
    "pigment_network": "is",
    "negative_network": "is",
    "streaks": "are",
    "milia_like_cysts": "are",
    "globules": "are",
}


# Rita's actual Task 2 filenames
TASK2_FILE_NAMES = {
    "pigment_network": "pigment_network",
    "negative_network": "negative_network",
    "streaks": "streaks",
    "milia_like_cysts": "milia_like_cyst",
    "globules": "globules",
}


# Rita's actual probability CSV column names
TASK2_CSV_COLUMNS = {
    "pigment_network": "pigment_network",
    "negative_network": "negative_network",
    "streaks": "streaks",
    "milia_like_cysts": "milia_like_cyst",
    "globules": "globules",
}


# Thresholds Rita used to create the binary PNG masks
# These have already been applied, so Task 3 does not apply them again.
TASK2_PIXEL_THRESHOLDS = {
    "pigment_network": 0.50,
    "negative_network": 0.50,
    "streaks": 0.50,
    "milia_like_cysts": 0.50,
    "globules": 0.30,
}


# Reliability-aware reporting policy
MASK_STATUS_POLICY = {
    "pigment_network": {
        "positive": "present",
        "negative": "absent",
    },
    "negative_network": {
        "positive": "present",
        "negative": "uncertain",
    },
    "streaks": {
        "positive": "uncertain",
        "negative": "uncertain",
    },
    "milia_like_cysts": {
        "positive": "uncertain",
        "negative": "uncertain",
    },
    "globules": {
        "positive": "present",
        "negative": "uncertain",
    },
}


# Trained model information
TASK2_MODEL_VERSION = "task2_model_best"


# Faisal's Task 1 mask filename format
TASK1_MASK_SUFFIX = "_predicted_mask.png"