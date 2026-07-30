# Attribute Detection (Task 2 — Model Side - Person A)

Owner: Rita — `attribute_detection/`

This folder trains the multi-attribute segmentation model for Task 2 and
exports predicted masks. It picks up where `preprocessing/` leaves off:
`preprocessing/` confirms and prepares the data, this folder trains a model
on it and produces predictions Ivana's `task2_eval/` folder can score.

> **What this folder is for:** given the confirmed, resizable dataset,
> load it into PyTorch, train a U-Net that predicts all five attribute
> masks at once, and export those predictions in the format the
> evaluation code and the final submission both expect.

---

## Files

| File | What it does |
|---|---|
| `dataset.py` | `AttributeDataset` — loads one image + its 5 attribute masks per sample, resized via `preprocessing/transforms.py`, ImageNet-normalised for the pretrained encoder. |
| `model.py` | `UNetMultiHead` — U-Net with a pretrained VGG16 encoder and 5 independent 1×1-conv sigmoid heads, one per attribute. |
| `loss.py` | `DiceBCELoss` — combined BCE + Dice loss, computed per attribute so rare attributes (streaks, negative_network) can be up-weighted. Also exposes `dice_score()` for validation metrics. |
| `train.py` | Training loop: builds train/val datasets from `splits.csv`, trains `UNetMultiHead`, saves the last epoch (`task2_model.pth`) and the best epoch by mean validation Dice (`task2_model_best.pth`). |
| `predictions.py` | Loads a trained checkpoint, runs it on the validation images, saves predicted masks as `prediction_root/{image_id}/{attribute}.png`. |
| `submission_format.py` | Reorganises `predictions.py`'s output into the briefing's required deliverable layout: `submission_root/{attribute}/{image_id}.png` (5 folders, one per attribute). |
---

## How the pieces fit together

```
splits.csv ──► dataset.py (AttributeDataset)
                     │
                     ▼
              model.py (UNetMultiHead)
                     │
              loss.py (DiceBCELoss)
                     │
                     ▼
             train.py  ──► task2_model_best.pth
                                    │
                                    ▼
                           predictions.py
                                    │
                    prediction_root/{image_id}/{attribute}.png
                                    │
                                    ▼
                         submission_format.py
                                    │
                    submission_root/{attribute}/{image_id}.png
```

---

## Dependencies

- `preprocessing/transforms.py` — `load_image`, `load_mask`, `DEFAULT_SIZE`. Must be importable as `preprocessing.transforms`.
- `preprocessing/create_splits.py` output — `splits.csv` (image_id, split columns). Everyone on the team must use the same file (seed 42), or train/val numbers won't be comparable.
- `torch`, `torchvision`, `pandas`, `numpy`, `Pillow`.
- `task2_eval/` (Ivana's evaluation folder) — **not currently present in this repo.** `predictions.py` imports `task2_eval.mask_io` to guarantee the training and evaluation code agree on attribute names/order. Until `task2_eval/` is merged in, `predictions.py` will fail with `ModuleNotFoundError`.

---

## Running it

Train:
```bash
python -m attribute_detection.train \
    --data-root /path/to/train \
    --splits-csv /path/to/data/splits.csv \
    --epochs 20 \
    --batch-size 8 \
    --output-dir outputs/task2_training
```

Export predictions on the validation split:
```bash
python -m attribute_detection.predictions \
    --data-root /path/to/train \
    --splits-csv /path/to/data/splits.csv \
    --checkpoint outputs/task2_training/task2_model_best.pth \
    --prediction-root predictions/task2
```

Reorganise into the submission format:
```bash
python -m attribute_detection.submission_format \
    --prediction-root predictions/task2 \
    --submission-root submission/task2_attributes \
    --splits-csv /path/to/data/splits.csv
```

---

## Known issues / TODO

- **No data augmentation.** `dataset.py`'s docstring flags this directly (`MISSING AUGMENTATION`). Training currently runs on unaugmented images/masks only. Given the class imbalance in the attributes (streaks 5.3%, negative_network 7.4% — see `preprocessing/attribute_stats.py`), augmentation would likely help the rare classes most.
- **No hair removal or colour constancy.** The briefing's preprocessing slide lists DullRazor hair removal and Shades-of-Gray colour constancy alongside resize/normalise; `transforms.py` only does resize + divide-by-255. Not required for scoring, but worth noting as a limitation in the technical report.
- **95% Hausdorff distance is not computed for Task 2** — this is expected, the scoring criteria mark it as Task 1 only.

---

## Status

- [x] Dataset class (`AttributeDataset`)
- [x] Model (`UNetMultiHead`, VGG16 encoder, 5 heads)
- [x] Loss (`DiceBCELoss`, per-attribute, supports rare-class weighting)
- [x] Training loop with best-checkpoint saving
- [x] Prediction export
- [x] Submission-format reorganisation
