# Concept Discovery on Retinal Fundus Images (LGMD)

Post-hoc concept discovery for a 5-class retinal-fundus classifier. A fine-tuned **DenseNet-121**
is explained with **named clinical concepts** via CLIP-guided matrix decomposition
(`Ā ≈ S Wᵀ`), using **FLAIR** (a foundation language-image model of the retina) for the
concept maps `S`.

## Data

Dataset: **ODIR-5K** (Ocular Disease Intelligent Recognition), restricted to five single-label
classes — **Normal, Diabetes (DR), Glaucoma, Cataract, AMD** (ODIR codes N / D / G / C / A).
The reference DenseNet-121 reaches ~**0.755 balanced accuracy** on this split.

ODIR-5K ships as a flat image folder plus a per-eye label table, **not** an ImageFolder tree,
so [`src/odir_prep.py`](src/odir_prep.py) converts it once into the layout the pipeline expects:

```
<data_root>/                        # cache/odir5k_imagefolder (built by odir_prep.prepare)
├── train/<Class>/*.jpg
├── val/<Class>/*.jpg
└── test/<Class>/*.jpg
```

`odir_prep` derives each eye's label from its diagnostic keywords, keeps only images that are
unambiguously one of the five classes (mixed/other diagnoses are dropped), and writes a
stratified 70/15/15 split. The notebook's **section 0** downloads ODIR-5K via `kagglehub`
(cached) and calls `odir_prep.prepare(...)`, which sets `CONFIG["data_root"]`. On Kaggle the
mounted copy is used instead of downloading. Kaggle credentials (`KAGGLE_USERNAME`, `KAGGLE_KEY`)
come from `config.get_secret` — Colab Secrets, env vars, or a gitignored `secrets.json` (see
`secrets.json.example`).

## Run (notebook, after the `sys.path` cell that adds `src/`)

```python
!pip -q install -r requirements.txt

import config, odir_prep
# Section 0 downloads ODIR-5K (kagglehub) and builds the 5-class ImageFolder:
odir_prep.prepare(raw_root=<download dir>)   # sets CONFIG["data_root"]
config.fundus_class_names(refresh=True)      # confirm the 5 class-folder names

import train_backbone
train_backbone.train()          # fine-tune DenseNet-121 -> cache/densenet121_odir.pt

import runner
results, agg = runner.run_all()              # all classes (cached, fault-tolerant)
results, agg = runner.run_all(["Glaucoma"])  # one class
runner.make_figures()                        # concept overlays (default: cataract)
```

`train_backbone.train()` must run first (it produces the weights `load_backbone` reads).
A class that errors is printed and skipped; completed classes are cached and resume free.

## Lesion-localization check (ground-truth masks)

`runner.run_all` reports two correctness counts per class: **diagnosed** (how many val
images the backbone classified correctly — every concept heatmap is drawn on one of these)
and **concept-preserved** (of those, how many keep the diagnosis after concept
reconstruction). See `runner.diagnosis_summary(results)`.

To check whether the heatmaps land on the *right lesions*, [`src/lesion_eval.py`](src/lesion_eval.py)
scores the Diabetes concept basis against **IDRiD** (Indian Diabetic Retinopathy Image
Dataset) pixel lesion masks (microaneurysms / haemorrhages / hard & soft exudates / optic disc):

```python
import lesion_eval
lesion_eval.run_localization()          # downloads IDRiD via kagglehub
```

Each DR concept is mapped to a lesion mask type by keyword (`concept_lesion_type`); vessel /
neovascular concepts have no mask and are skipped. For every (image, concept) pair it
computes a **pointing-game** hit (heatmap peak inside the mask) and **mass-in-mask**
(fraction of heatmap energy on the lesion), then reports how many concepts localize
(hit on ≥ half their images) and the overall heatmap hit rate. The Kaggle slug is
`lesion_eval.IDRID_SLUG` (override with `run_localization(root=...)` if it moves).

## Key knobs (`src/config.py`)

- `backbone` — `"densenet121"` (default; feat_dim 1024) or `"resnet34"` / `"resnet50"` / `"mobilenet_v2"`.
- `n_per_class` — max train images/class for the backbone (default 2000; most ODIR classes are smaller).
- `n_train` / `n_val` — images/class for fitting `W` / evaluating (default 100 / 50).
- `concept_mode` — `"per_class"` (default) or `"shared"`.
- `r` — concepts kept per class in per_class mode (default 25).
- `flair_weights` — local FLAIR checkpoint path, or `None` to download the pretrained retina weights.

## Concept vocabulary

[`concept_vocab.json`](concept_vocab.json), keyed by class name (snake_case, ~50 candidates
each), filtered to `r` at run time. The five keys — `cataract`, `diabetic_retinopathy`,
`glaucoma`, `amd`, `normal_fundus` — map to the ODIR folder names via `config.resolve_vocab_key`
(`Diabetes` ↔ `diabetic_retinopathy`, `Normal` ↔ `normal_fundus`, `AMD` ↔ `amd`); unmatched
classes are skipped. Editing the file invalidates the concept cache.
