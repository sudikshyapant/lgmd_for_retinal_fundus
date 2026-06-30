# Concept Discovery on Retinal Fundus Images (LGMD)

Post-hoc concept discovery for a 7-class retinal-fundus classifier. A fine-tuned ResNet-34
is explained with **named clinical concepts** via CLIP-guided matrix decomposition
(`Ā ≈ S Wᵀ`), using **BiomedCLIP** for the concept maps `S`.

## Data

Dataset: [Retinal Fundus Image 50k](https://www.kaggle.com/datasets/gautamrajiitk/retinal-fundus-image-50k)
(~50k images, 7 classes, ImageFolder layout):

```
<data_root>/
├── train/<ClassName>/*.jpg
├── val/<ClassName>/*.jpg
└── test/<ClassName>/*.jpg
```

The notebook's **section 0** downloads it once via `kagglehub` (cached) and sets
`CONFIG["data_root"]` automatically — on Kaggle the mounted `/kaggle/input` copy is used
instead, no download. Kaggle only serves the full archive, so the ~4 GB pull is unavoidable,
but we then **use only `CONFIG["n_per_class"]` images/class** (default **2000**) for backbone
training, i.e. a 2k/class subset of the 50k. Kaggle credentials (`KAGGLE_USERNAME`,
`KAGGLE_KEY`) come from `config.get_secret` — Colab Secrets, env vars, or a gitignored
`secrets.json` (see `secrets.json.example`). Classes are read from the train folders.

## Run (notebook, after the `sys.path` cell that adds `src/`)

```python
!pip -q install -r requirements.txt

import config
# Section 0 of the notebook downloads the dataset (kagglehub) and sets CONFIG["data_root"].
config.fundus_class_names()     # confirm the 7 class-folder names

import train_backbone
train_backbone.train()          # fine-tune ResNet34 -> cache/resnet34_fundus.pt

import runner
results, agg = runner.run_all()              # all classes (cached, fault-tolerant)
results, agg = runner.run_all(["Glaucoma"])  # one class
runner.make_figures()                        # concept overlays (default: cataract)
```

`train_backbone.train()` must run first (it produces the weights `load_backbone` reads).
A class that errors is printed and skipped; completed classes are cached and resume free.

## Key knobs (`src/config.py`)

- `n_per_class` — train images/class for the backbone (default 2000; subset of the 50k).
- `n_train` / `n_val` — images/class for fitting `W` / evaluating (default 100 / 50).
- `concept_mode` — `"per_class"` (default) or `"shared"`.
- `r` — concepts kept per class in per_class mode (default 25).
- `clip_backend` — `"biomedclip"` (default) or `"openai"`.

## Concept vocabulary

[`concept_vocab.json`](concept_vocab.json), keyed by class name (snake_case, ~50 candidates
each), filtered to `r` at run time. Keys must match the dataset folder names (case/spacing
normalized, e.g. `Diabetic Retinopathy` ↔ `diabetic_retinopathy`); unmatched classes are
skipped. Editing the file invalidates the concept cache.
