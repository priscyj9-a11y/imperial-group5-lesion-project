"""Shared configuration for Task 3 report generation."""


# ------------------------------------------------------------------
# Canonical attribute names required by the Task 3 JSON schema
# ------------------------------------------------------------------

ATTRIBUTES = [
    "pigment_network",
    "negative_network",
    "streaks",
    "milia_like_cysts",
    "globules",
]


# ------------------------------------------------------------------
# Names used in the written findings report
# ------------------------------------------------------------------

DISPLAY_NAMES = {
    "pigment_network": "Pigment network",
    "negative_network": "negative network",
    "streaks": "streaks",
    "milia_like_cysts": "milia-like cysts",
    "globules": "globules",
}


# ------------------------------------------------------------------
# Grammar used in the findings report
# ------------------------------------------------------------------

ATTRIBUTE_VERBS = {
    "pigment_network": "is",
    "negative_network": "is",
    "streaks": "are",
    "milia_like_cysts": "are",
    "globules": "are",
}


# ------------------------------------------------------------------
# Rita's Task 2 mask filenames
#
# Task 3 requires the plural JSON name milia_like_cysts.
# Rita's files use the singular name milia_like_cyst.
# ------------------------------------------------------------------

TASK2_FILE_NAMES = {
    "pigment_network": "pigment_network",
    "negative_network": "negative_network",
    "streaks": "streaks",
    "milia_like_cysts": "milia_like_cyst",
    "globules": "globules",
}


# ------------------------------------------------------------------
# Rita's actual probability CSV columns
# ------------------------------------------------------------------

TASK2_CSV_COLUMNS = {
    "pigment_network": "pigment_network",
    "negative_network": "negative_network",
    "streaks": "streaks",
    "milia_like_cysts": "milia_like_cyst",
    "globules": "globules",
}


# ------------------------------------------------------------------
# Pixel thresholds already used by Task 2 when saving the PNG masks
#
# Task 3 does not apply these thresholds again. They are recorded here
# for documentation.
# ------------------------------------------------------------------

TASK2_PIXEL_THRESHOLDS = {
    "pigment_network": 0.50,
    "negative_network": 0.50,
    "streaks": 0.50,
    "milia_like_cysts": 0.50,
    "globules": 0.30,
}


# ------------------------------------------------------------------
# Reliability-aware reporting policy
#
# "positive" means the predicted mask contains white pixels.
# "negative" means the predicted mask is completely black.
#
# Pigment network had the strongest validation performance.
# The other attributes are treated conservatively because their
# validation performance was weak or failed.
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# JSON metadata
# ------------------------------------------------------------------

PIPELINE_MODEL_VERSION = "task1_unet__task2_model_best"
DATASET_SPLIT = "val"


# ------------------------------------------------------------------
# Lesion feature thresholds
# ------------------------------------------------------------------

SMALL_LESION_MAX_RATIO = 0.08
MODERATE_LESION_MAX_RATIO = 0.25
IRREGULAR_BORDER_THRESHOLD = 1.60