# Preprocessing

Owner: Faisal — branch `faisal-data-preprocessing`

Everything here prepares the dataset so all five of us work from the same
foundation. If you're starting a task, read this first.

---

## Quick start

```powershell
# 1. make the shared environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. load data in your code
from preprocessing.transforms import load_image, load_mask
```

The dataset is NOT in this repo. Keep it separately on your own machine.

---

## The three rules

1. **Use `data/splits.csv`.** Don't make your own split. Never train on `val`.
2. **Import from `transforms.py`.** Don't write your own loading or resizing.
3. **Never commit anything from `project_data/`.**

---

## Dataset layout

```
project_data/
└── train/
    ├── images/      2700 files   000001.jpg
    ├── task1_gt/    2700 files   000001_segmentation.png
    └── task2_gt/   13500 files   000001_attribute_globules.png
```

Every file starts with the same 6-digit image ID. Split the filename at the
first underscore to get it.

The five attributes, spelled as they appear in filenames:
`pigment_network`, `negative_network`, `streaks`, `milia_like_cyst`, `globules`

> **Careful:** files use `milia_like_cyst` (singular), the briefing's JSON
> schema uses `milia_like_cysts` (plural). Task 3 needs to map between them.

---

## How to load data

```python
from pathlib import Path
from preprocessing.transforms import load_image, load_mask

image = load_image(Path("path/to/000001.jpg"))
# -> 512x512x3, float32, values 0.0 to 1.0

mask = load_mask(Path("path/to/000001_segmentation.png"))
# -> 512x512, values 0 or 1
```

**Images** are resized with bilinear (smooth blending — right for photos) and
divided by 255 so values land between 0.0 and 1.0.

**Masks** are resized with nearest-neighbour (no blending) and returned as 0/1.

> **Why masks are different:** bilinear would average a 0 and a 255 into greys
> like 127, so "is this pixel lesion or not?" has no clean answer and our ground
> truth is silently corrupted. Every Dice/IoU score after that would be wrong.
> Masks are labels, not pictures — they are never normalised.

---

## The split — `data/splits.csv`

| | |
|---|---|
| train | 2160 images (80%) |
| val | 540 images (20%) |
| seed | 42 — **do not change** |

Two columns: `image_id`, `split`. Filter by the `split` column.

> **Why it's shared:** if we each pick our own validation set, our scores aren't
> comparable — one person might just get easier images.
>
> **Why split by image ID:** each ID owns 1 photo + 6 masks. Splitting by ID
> keeps a case whole. If the same lesion landed in both train and val, the model
> would have already seen the answer — that's data leakage, and validation
> becomes meaningless.

---

## Attribute frequencies — `data/attribute_frequency.csv`

Checked all five attributes across all 2700 images.

| attribute | present | absent | % present |
|---|---|---|---|
| pigment_network | 1651 | 1049 | 61.1% |
| negative_network | 201 | 2499 | 7.4% |
| streaks | 142 | 2558 | 5.3% |
| milia_like_cyst | 574 | 2126 | 21.3% |
| globules | 610 | 2090 | 22.6% |

> **No mask files are missing** — every image has all five. When an attribute
> isn't in the photo, its mask is simply all black. Absent ≠ broken file.
>
> **Class imbalance warning for Task 2:** streaks appear in only 5.3% of images.
> A model that always predicts "absent" scores 94.7% accuracy while learning
> nothing. Accuracy is a useless metric here — use per-attribute Dice, and
> expect to need weighted loss or tuned thresholds for streaks and negative
> network.

---

## Augmentation warning

If you flip, rotate, or crop an image, you **must** apply the identical
transform to its mask. Flip one without the other and the ground truth points
at the wrong pixels — training breaks silently and you won't notice until your
scores are bad.

Never apply random augmentation to `val` data.

---

## Scripts

| Script | What it does |
|---|---|
| `inspect_dataset.py` | Read-only check that every image has its masks. Modifies nothing. |
| `transforms.py` | Loading, resizing, normalisation. Import from this. |
| `create_splits.py` | Generates `data/splits.csv`. Already run — don't re-run unless needed. |
| `attribute_stats.py` | Generates `data/attribute_frequency.csv`. Already run. |

```powershell
python preprocessing\inspect_dataset.py --data-root "<YOUR_PATH>\project_data\train"
python preprocessing\transforms.py --data-root "<YOUR_PATH>\project_data\train" --n 50
```

---

## Known issue — speed

A full pass over 2700 images takes ~7 minutes, because our originals are up to
30 megapixels and decoding them is slow (the resizing itself is fast). This
repeats every training epoch.

**Fix:** a pre-resized cache — save 512×512 copies to disk once, load those
instead. Keeps full 512 quality. Not built yet; will add before training scales
up. Flag it if it's slowing you down.

---

## Status

- [x] Dataset inspected — 2700 clean cases, no missing or duplicate IDs
- [x] Masks confirmed binary
- [x] Resize + normalisation, verified across all 2700
- [x] Fixed train/val split
- [x] Attribute frequency table
- [ ] Pre-resized cache (speed)
- [ ] Integration pipeline
- [ ] Submission checker