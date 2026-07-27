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

PRESENT_THRESHOLD = 0.60
ABSENT_THRESHOLD = 0.40