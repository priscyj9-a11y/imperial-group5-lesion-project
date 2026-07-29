"""Shared configuration for Task 3 report generation."""


# Canonical names used in Task 3 JSON and CSV outputs.
ATTRIBUTES = [
    "pigment_network",
    "negative_network",
    "streaks",
    "milia_like_cysts",
    "globules",
]


# Names used in the written findings report.
DISPLAY_NAMES = {
    "pigment_network": "Pigment network",
    "negative_network": "negative network",
    "streaks": "streaks",
    "milia_like_cysts": "milia-like cysts",
    "globules": "globules",
}


# Grammar used in the generated report.
ATTRIBUTE_VERBS = {
    "pigment_network": "is",
    "negative_network": "is",
    "streaks": "are",
    "milia_like_cysts": "are",
    "globules": "are",
}


# Rita's actual Task 2 filenames.
TASK2_FILE_NAMES = {
    "pigment_network": "pigment_network",
    "negative_network": "negative_network",
    "streaks": "streaks",
    "milia_like_cysts": "milia_like_cyst",
    "globules": "globules",
}


# Rita's probability CSV column names.
TASK2_CSV_COLUMNS = {
    "pigment_network": "pigment_network",
    "negative_network": "negative_network",
    "streaks": "streaks",
    "milia_like_cysts": "milia_like_cyst",
    "globules": "globules",
}


# Pixel thresholds already used by Task 2 to create the binary masks.
# Task 3 records these for documentation but does not threshold the PNGs again.
TASK2_PIXEL_THRESHOLDS = {
    "pigment_network": 0.50,
    "negative_network": 0.50,
    "streaks": 0.50,
    "milia_like_cysts": 0.50,
    "globules": 0.30,
}


# Conservative reliability-aware reporting policy.
#
# Pigment network had the strongest validation performance.
# The remaining attributes showed weak or failed generalisation, so they
# are reported as uncertain instead of turning model failure into certainty.
MASK_STATUS_POLICY = {
    "pigment_network": {
        "positive": "present",
        "negative": "absent",
    },
    "negative_network": {
        "positive": "uncertain",
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
        "positive": "uncertain",
        "negative": "uncertain",
    },
}


# Model information included in the JSON.
PIPELINE_MODEL_VERSION = "task1_unet__task2_vgg16_unet_best"
DATASET_SPLIT = "validation"


# Lesion-description rules currently used by Task 3.
SMALL_LESION_MAX_RATIO = 0.08
MODERATE_LESION_MAX_RATIO = 0.25
IRREGULAR_BORDER_THRESHOLD = 1.60