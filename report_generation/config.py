"""Shared configuration for Task 3 report generation."""


ATTRIBUTES = [
    "pigment_network",
    "negative_network",
    "streaks",
    "milia_like_cysts",
    "globules",
]


DISPLAY_NAMES = {
    "pigment_network": "Pigment network",
    "negative_network": "negative network",
    "streaks": "streaks",
    "milia_like_cysts": "milia-like cysts",
    "globules": "globules",
}


ATTRIBUTE_VERBS = {
    "pigment_network": "is",
    "negative_network": "is",
    "streaks": "are",
    "milia_like_cysts": "are",
    "globules": "are",
}


TASK2_FILE_NAMES = {
    "pigment_network": "pigment_network",
    "negative_network": "negative_network",
    "streaks": "streaks",
    "milia_like_cysts": "milia_like_cyst",
    "globules": "globules",
}


TASK2_CSV_COLUMNS = {
    "pigment_network": "pigment_network",
    "negative_network": "negative_network",
    "streaks": "streaks",
    "milia_like_cysts": "milia_like_cyst",
    "globules": "globules",
}


TASK2_PIXEL_THRESHOLDS = {
    "pigment_network": 0.50,
    "negative_network": 0.50,
    "streaks": 0.50,
    "milia_like_cysts": 0.50,
    "globules": 0.30,
}


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


PIPELINE_MODEL_VERSION = (
    "task1_unet__task2_model_best"
)

DATASET_SPLIT = "test"


SMALL_LESION_MAX_RATIO = 0.08
MODERATE_LESION_MAX_RATIO = 0.25
IRREGULAR_BORDER_THRESHOLD = 1.60