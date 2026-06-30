# Concept Discovery on Retinal Fundus Images (LGMD)

Post-hoc concept discovery for a 7-class retinal-fundus classifier. A fine-tuned ResNet-34
is explained with **named clinical concepts** via CLIP-guided matrix decomposition
(`Ā ≈ S Wᵀ`), using **BiomedCLIP** for the concept maps `S`.

## Data layout

ImageFolder format (Kaggle paths shown):

```
/kaggle/input/retinal-fundus-image-50k/Retinal Fundus Images/
├── train/<ClassName>/*.jpg
├── val/<ClassName>/*.jpg
└── test/<ClassName>/*.jpg
```

Set `CONFIG["data_root"]` to your root. Classes are read from the train folders.

## Run (notebook, after the `sys.path` cell that adds `src/`)

```python
!pip -q install -r requirements.txt

import config
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

- `n_per_class` — train images/class for the backbone (default 3000).
- `n_train` / `n_val` — images/class for fitting `W` / evaluating (default 100 / 50).
- `concept_mode` — `"per_class"` (default) or `"shared"`.
- `r` — concepts kept per class in per_class mode (default 25).
- `clip_backend` — `"biomedclip"` (default) or `"openai"`.

## Concept vocabulary

[`concept_vocab.json`](concept_vocab.json), keyed by class name (snake_case, ~50 candidates
each), filtered to `r` at run time. Keys must match the dataset folder names (case/spacing
normalized, e.g. `Diabetic Retinopathy` ↔ `diabetic_retinopathy`); unmatched classes are
skipped. Editing the file invalidates the concept cache.
