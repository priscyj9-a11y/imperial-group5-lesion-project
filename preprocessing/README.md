# Preprocessing

Owner: Faisal — branch `faisal-data-preprocessing`

This folder handles dataset checking and preparation for Task 1 (lesion
segmentation) and Task 2 (attribute detection).

> **What this folder is for:** before anyone trains a model, someone has to
> confirm the data is actually correct and organised. That's what lives here.
> If this part is wrong, every model trained afterwards is wrong too, and it
> usually isn't obvious until days later.

---

## Dataset location

The dataset is **not** in this repository and must never be committed to it.
Keep it in a separate folder on your own machine. All scripts take the path
as a command-line argument so nobody's personal path is hardcoded.

> **Why:** the dataset is thousands of medical images. Pushing it to GitHub
> would blow past size limits and is bad practice for patient imaging.
> Keeping it outside the repo means Git literally cannot see it, so nobody
> can commit it by accident. And passing the path as an argument means the
> scripts work on everyone's machine, not just mine.

Expected structure:

```
project_data/
└── train/
    ├── images/      2700 files   000001.jpg
    ├── task1_gt/    2700 files   000001_segmentation.png
    └── task2_gt/   13500 files   000001_attribute_globules.png
```

> `images` are the photos the model looks at. `task1_gt` and `task2_gt` are
> the answer sheets we compare its guesses against. "gt" = ground truth.
> Photos are `.jpg`, masks are `.png` — masks must stay lossless or the
> lesion boundary gets blurred and our answer sheets become slightly wrong.

---

## Naming convention

Every file starts with the same 6-digit image ID. Split the filename at the
first underscore to get it.

| Folder | Example | ID |
|---|---|---|
| `images` | `000001.jpg` | `000001` |
| `task1_gt` | `000001_segmentation.png` | `000001` |
| `task2_gt` | `000001_attribute_streaks.png` | `000001` |

> **Why this matters:** the ID is the only thing linking a photo to its
> answer sheets. Everything downstream — pairing, splitting, evaluation —
> depends on extracting it the same way every time.

The five attributes, spelled as they appear in the filenames:

- `pigment_network`
- `negative_network`
- `streaks`
- `milia_like_cyst`
- `globules`

**Careful:** the files use `milia_like_cyst` (singular) but the project
briefing's JSON schema uses `milia_like_cysts` (plural). Whoever builds the
report output in Task 3 needs to know this.

> A one-letter mismatch like this won't crash anything. It'll just quietly
> produce a report that fails the schema check on submission day.

---

## Scripts

### `inspect_dataset.py`

Read-only check that every image has its matching masks. Modifies nothing.

> **What it does, in plain terms:** reads all the filenames, pulls the ID off
> the front of each one, and checks that photo `000001` has a lesion mask
> `000001` and five attribute masks `000001`. Then reports anything that
> doesn't line up. It never opens, edits, moves or deletes a file.

Run it:

```powershell
python preprocessing\inspect_dataset.py --data-root "<YOUR_PATH>\project_data\train"
```

Expected output — every mismatch count should be `0`:

```
  raw images     : 2700
  lesion masks   : 2700
  attribute masks: 13500

Unique IDs: images=2700, lesion masks=2700
  images with no lesion mask: 0
  lesion masks with no image: 0

IDs without exactly 5 attribute masks: 0
IDs in images but absent from task2_gt: 0
```

> **Why this check exists:** the totals matching (2700 / 2700 / 13500) does
> not prove the *same* IDs are in each folder. One missing image plus one
> duplicate elsewhere gives identical totals while hiding a broken pair. This
> script compares the actual ID sets, so a photo paired with the wrong answer
> sheet can't silently reach training.

---

## Status

- [x] Dataset structure inspected — passed, no missing or duplicate IDs
- [ ] Inspect mask pixel values (are masks really binary?)
- [ ] Check whether absent attributes are blank masks or missing files
- [ ] Build dataset manifest (CSV)
- [ ] Create train/val splits with a fixed seed

> **What's left, briefly:** we've confirmed the *files* are correct. We have
> not yet looked *inside* them. Next we open a few masks and check the pixel
> values are actually just black and white, and work out how an absent
> attribute is stored — all-black mask, or something else.

---

## Notes for the team

- There is no `val` or `test` folder. We create the validation split ourselves
  from `train`. The real test set is released 30 July.
- Split by **image ID**, never by file, or the same lesion ends up in both
  train and validation and our scores become meaningless.

  > This woudl be data leakage. If the model sees an image during training
  > and again during validation, our validation score measures memorisation,
  > not skill — and it'll look great right up until the real test set lands.




##  note to Faisal ( Self ): Resizing is the main issue to solve at the moment, Build an algorithm in the preprocessing file to solve this issue. 

- Never commit anything from `project_data/`.