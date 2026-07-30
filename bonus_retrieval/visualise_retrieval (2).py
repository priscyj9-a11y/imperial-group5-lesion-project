# Bonus Retrieval - CLIP

This folder contains a simple implementation of the optional bonus:

```text
CLIP semantic retrieval + audit CSV
```

The goal is to find visually similar training images for each query/test image.
The output is:

```text
bonus/test_bonus_clip.csv
```

with columns:

```text
query_image,neighbor_id,similarity
```

## What The Code Does

`clip_retrieval.py`:

1. loads a pretrained CLIP model,
2. encodes all reference/training images into embeddings,
3. encodes all query/test images into embeddings,
4. compares query embeddings with training embeddings using cosine similarity,
5. keeps the top-k nearest neighbours,
6. writes the CSV required for the bonus audit trail.

## Environment

This script uses:

```text
torch
Pillow
numpy
open_clip_torch
```

Your `Summerschool` environment already has:

```text
torch
Pillow
numpy
```

It may still need:

```powershell
pip install open_clip_torch
```

`open_clip_torch` has been installed in Ivana's local `Summerschool`
environment. If another teammate runs this code on their own computer, they may
need to install it too.

## Command

Run this after the test/query images are available:

```powershell
& "C:\Users\Ivana\anaconda3\envs\Summerschool\python.exe" -m bonus_retrieval.clip_retrieval --query-dir path\to\test\images --output-csv bonus\test_bonus_clip.csv
```

By default, the reference images are read from:

```text
C:\Users\Ivana\Downloads\summer_school_project_train\train\images
```

If the team stores the training images somewhere else, pass:

```powershell
--index-dir path\to\train\images
```

## Visual Check

The CSV is useful for submission, but the report/presentation may also need a
figure. `visualise_retrieval.py` creates contact sheets showing:

```text
query/test image | neighbour #1 | neighbour #2 | neighbour #3
```

Run it after `bonus/test_bonus_clip.csv` has been created:

```powershell
& "C:\Users\Ivana\anaconda3\envs\Summerschool\python.exe" -m bonus_retrieval.visualise_retrieval --query-dir path\to\test\images --retrieval-csv bonus\test_bonus_clip.csv
```

By default, it saves example images in:

```text
outputs/bonus_retrieval_visuals
```

This visualisation is only for checking and explaining the retrieval results.
It does not train CLIP and it does not change the CSV.

## RAG-Style Attribute Summary

`rag_attribute_summary.py` is the first simple RAG step.

It uses:

```text
CLIP retrieved neighbours
+ real Task 2 masks of those training neighbours
= short evidence summary
```

For each query/test image, it counts how many retrieved neighbours contain each
of the five Task 2 attributes:

```text
pigment_network
negative_network
streaks
milia_like_cyst
globules
```

Run it after `bonus/test_bonus_clip.csv` exists:

```powershell
& "C:\Users\Ivana\anaconda3\envs\Summerschool\python.exe" -m bonus_retrieval.rag_attribute_summary --retrieval-csv bonus\test_bonus_clip.csv --output-csv bonus\test_bonus_rag_summary.csv
```

The output CSV contains:

```text
query_image
neighbour_ids
attribute counts
evidence_summary
```

This is Option 2. It does not require Task 2 predictions for the test images.
When Task 2 predictions are available, this can be extended to Option 3 by
comparing the predicted test attributes with the attribute counts found in the
retrieved neighbours.

## Option 3: Prediction Support Report

`rag_prediction_support.py` is prepared for tomorrow, when Task 2 predicted masks
for the test images are available.

It compares:

```text
Task 2 predicted attributes for the query/test image
vs
attributes observed in the CLIP retrieved training neighbours
```

Expected prediction folder format:

```text
prediction_root/
    test_001/
        pigment_network.png
        negative_network.png
        streaks.png
        milia_like_cyst.png
        globules.png
```

Run it after `test_bonus_rag_summary.csv` and Task 2 predictions both exist:

```powershell
& "C:\Users\Ivana\anaconda3\envs\Summerschool\python.exe" -m bonus_retrieval.rag_prediction_support --prediction-root path\to\predictions\task2 --option2-summary-csv bonus\test_bonus_rag_summary.csv
```

The output is:

```text
bonus/test_bonus_rag_prediction_support.csv
```

This CSV explains whether the retrieved-neighbour evidence supports the
attributes predicted by the Task 2 model.

## Output Example

```text
query_image,neighbor_id,similarity
000001,data/images\001573.jpg,0.9403
000001,data/images\002395.jpg,0.9362
000001,data/images\002506.jpg,0.9352
```

## Simple Explanation

CLIP turns each image into a vector. Images that look semantically similar have
similar vectors. The script compares vectors and saves the most similar training
images for every test image.
