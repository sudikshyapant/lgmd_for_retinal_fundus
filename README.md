# Concept Discovery on Retinal Fundus Images via Language-Guided Matrix Decomposition

An adaptation of *Interpretable Concept Discovery via Language-Guided Matrix
Decomposition* (LGMD, ECCV 2026, #11398) to **retinal fundus disease classification**.
It does post-hoc concept discovery: instead of NMF's free coefficient matrix, it uses
**CLIP-guided semantic activations**, so each learned basis vector maps to a **named,
human-readable clinical concept** (`Ā ≈ S Wᵀ`, with `S` fixed from CLIP and `W` learned).

The goal here: given a classifier trained to grade fundus images into 7 disease classes,
explain *which clinical findings* (microaneurysms, optic-disc cupping, drusen, …) drive
its predictions, using a vocabulary a clinician would recognize.

## What this fork changes vs. the original LGMD repo

The method (CLIP maps `S`, the decomposition, ICE/CRAFT/FACE baselines, the Acc / C-Ins /
MSE metrics) is unchanged. What's adapted for the medical domain:

- **Dataset.** Local retinal-fundus images in `ImageFolder` layout (`<split>/<class>/*`).
- 7 disease classes (e.g. Cataract, Diabetic Retinopathy,
  Glaucoma, Hypertensive Retinopathy, Myopia, AMD, Normal) **resolved at runtime from the
  train folder**.
- **The classifier is ours, and it's fine-tuned.** LGMD explains a *trained classifier*,
  so we fine-tune **ResNet-34** (ImageNet-initialized) to a 7-way fundus head on a
  subset of the train split — see [`src/train_backbone.py`](src/train_backbone.py). An
  ImageNet head would explain nothing about fundus disease.
- **BiomedCLIP** Concept maps come from **BiomedCLIP**
  (`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`, via `open_clip`), which
  understands clinical terminology. Generic CLIP ViT-B/16 is kept as a fallback
  (`CONFIG["clip_backend"] = "openai"`) for comparison.
- **Shared concept bank (default).** With only 7 disease classes drawn from one clinical
  lexicon, concepts are a single **shared bank** used for every class — so the same
  concept's activation is comparable *across* diseases ("do hard exudates fire for both
  DR and hypertensive retinopathy?"). The paper's **per-class** mode is still available.
- **One backbone.** ResNet-34 only. The paper also reports MobileNetV2; we don't run it.

The CLIP red-circle prompting (suppl. A1.6), the shared 224 crop that aligns the encoder
grid with the CLIP grid, and the `[suppl]` informed-default hyperparameters in
[`src/config.py`](src/config.py) all carry over unchanged.

## Data layout

The dataset is expected on disk as (Kaggle paths shown):

```
/kaggle/input/retinal-fundus-image-50k/Retinal Fundus Images/
├── train/<ClassName>/*.jpg
├── val/<ClassName>/*.jpg
└── test/<ClassName>/*.jpg
```

Point `CONFIG["data_root"]` (and `train_dir` / `val_dir` / `test_dir`) at your root. Run
`config.fundus_class_names()` once to print the exact class-folder names.

## Setup & running (Kaggle / Colab notebook)

```python
!pip -q install -r requirements.txt
```

Then, in the notebook (after the `sys.path` setup cell that adds `src/`):

```python
import config
config.fundus_class_names()        # confirm the 7 class-folder names

import train_backbone
train_backbone.train()             # fine-tune ResNet34 -> cache/resnet34_fundus.pt

import runner
results, agg = runner.run_all()              # all classes (concept discovery + metrics)
results, agg = runner.run_all(["Glaucoma"])  # a single class
runner.make_figures()                        # concept-overlay figures (default: cataract)
```

`run_all` is **fault-tolerant**: a class that errors (too few `val` images, or — in
per_class mode — fewer than `r` concepts surviving the filter) is printed and skipped,
and the run continues. Every completed class is cached per-class, so finished work is
never lost and a rerun resumes for free. The showcase/figure class defaults to
**cataract** (`CONFIG["class_name"]` / `FIGURE_CLASSES`) — the fundus analogue of the
original's "tabby cat".

`train_backbone.train()` must run first: it produces the weights
(`CONFIG["backbone_weights"]`) that `model_utils.load_backbone` reads. BiomedCLIP is
public and downloads automatically; no token is needed for it. (`config.get_secret`
still supports an `HF_TOKEN` via Colab Secrets / env var / gitignored `secrets.json` if
you later use a gated model.)

## Knobs you'll likely touch (all in `src/config.py`)

- **`n_per_class`** — images per class drawn from `train/` to fine-tune the backbone.
  Start at `3000`; lower it (e.g. `500`) for a quick smoke run, raise to use more data.
  Also overridable per call: `train_backbone.train(n_per_class=500)`.
- **`n_train` / `n_val`** — images/class used by the *LGMD* stage to fit the concept
  basis `W` (from `train/`) and to run inference + metrics (from `val/`). Defaults 100 / 50.
- **`concept_mode`** — `"shared"` (default) or `"per_class"`.
- **`clip_backend`** — `"biomedclip"` (default) or `"openai"`.
- **`train_epochs` / `train_lr` / `train_batch_size`** — backbone fine-tuning.

## Concept vocabulary

Concepts are read from [`concept_vocab.json`](concept_vocab.json) — no LLM call, so runs
are reproducible and offline. Behavior depends on `CONFIG["concept_mode"]`:

- **`shared` (default).** Every value in the file is **flattened into one ordered,
  de-duplicated bank** used for *all* classes (the grouping — e.g. `optic_disc_and_cup`,
  `retinal_vasculature` — is purely for human readability and is ignored by the
  pipeline). Concepts are used **verbatim**: no lexical or CLIP reduction, so `r` = bank
  size and the basis columns are identical and comparable across classes. To add/remove a
  concept, just edit the bank; the file's hash auto-invalidates the concept cache.
- **`per_class`.** Paper-faithful. The file is keyed by **class name** (snake_case), each
  mapping to an over-provided candidate list filtered down to `r = 25` in two stages:
  - *Lexical (suppl. A1.3):* keep 1–4 word concepts; drop generic filler and concepts
    that just repeat the class name — unless they carry a clinical visual attribute.
  - *CLIP semantic (suppl. A1.4):* rank by BiomedCLIP similarity to the class images, then
    greedily keep diverse concepts (drop any too similar, `> 0.80`, to one already kept).
  A class-keyed example is in
  [`concept_vocab.per_class.json`](concept_vocab.per_class.json) — to use it, copy it over
  `concept_vocab.json` and set `CONFIG["concept_mode"] = "per_class"`.

## Caching

Heavy artifacts (backbone weights, activations, CLIP maps `S`, the learned basis `W`,
baseline bases, metric tables) are cached under `cache/` and `results/`. Each filename
embeds a short hash of the `CONFIG` values it depends on, so changing a relevant knob
rebuilds only the affected caches. Delete the files to force recomputation.

## Notes

- **Baselines compared fairly.** ICE, CRAFT, and FACE differ only in how the basis `W` is
  learned (NMF / recursive NMF / KL-regularized NMF). All else — backbone, preprocessing,
  splits, number of concepts, and the non-negative inference used for reconstruction — is
  identical, as the paper requires (Sec 4).
- **C-Ins** is the normalized area under the concept-insertion curve (Sec 4.2): how fast
  the correct-class prediction is restored as top concepts are added.
- **Backbone and CLIP share one 224 crop**, so the encoder's 7×7 feature cells and the
  CLIP red-circle cells cover identical pixels — keeping the concept heatmaps spatially
  accurate.
