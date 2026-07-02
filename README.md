# Concept Discovery on Retinal Fundus Images (LGMD)

Post-hoc concept discovery for a **5-grade diabetic-retinopathy classifier**. **RETFound
(DINOv2 ViT-L/14)** — a retinal foundation model — is used as a **frozen** encoder with a
**linear DR-grading head** on top, and that classifier is explained with **named clinical
concepts** via CLIP-guided matrix decomposition (`Ā ≈ S Wᵀ`), using **FLAIR** (a foundation
language-image model of the retina) for the concept maps `S`.

## Data

Dataset: **IDRiD** (Indian Diabetic Retinopathy Image Dataset), **Disease Grading** part —
image-level DR severity grade 0–4:

| grade | folder | meaning |
|---|---|---|
| 0 | `0_no_dr` | No DR |
| 1 | `1_mild_npdr` | Mild NPDR (microaneurysms only) |
| 2 | `2_moderate_npdr` | Moderate NPDR |
| 3 | `3_severe_npdr` | Severe NPDR (ETDRS 4-2-1) |
| 4 | `4_proliferative_dr` | Proliferative DR |

The numeric prefix keeps torchvision `ImageFolder`'s alphabetical ordering equal to grade
order (and to the classifier's output index). IDRiD ships as flat image folders + label CSVs,
**not** an ImageFolder tree, so [`src/dr_prep.py`](src/dr_prep.py) converts it once:

```
<data_root>/                         # cache/dr_grading_imagefolder (built by dr_prep.prepare)
├── train/<grade>/*.jpg
├── val/<grade>/*.jpg
└── test/<grade>/*.jpg
```

`dr_prep` reads the `Retinopathy grade` column and, since IDRiD has no official validation
set, uses IDRiD's **Testing Set** as `test/` and stratified-splits its **Training Set** 80/20
into `train/` + `val/`. The notebook's **section 0** downloads IDRiD via `kagglehub` (slug
`mariaherrerot/idrid-dataset`) and calls `dr_prep.prepare()`, which sets `CONFIG["data_root"]`.
Kaggle credentials (`KAGGLE_USERNAME`, `KAGGLE_KEY`) come from `config.get_secret` — Colab
Secrets, env vars, or a gitignored `secrets.json` (see `secrets.json.example`).

The grades are imbalanced (Mild NPDR is scarce). A grade whose head diagnoses too few val
images correctly falls back to evaluating on **all** its val images (`min_eval_images`), so
it still produces metrics instead of being skipped.

## RETFound checkpoint

The DINOv2 architecture is fetched from `torch.hub` (`facebookresearch/dinov2`), but the
RETFound pretrained **encoder** weights are not public here — download them and place the file
at `CONFIG["retfound_weights"]` (default `cache/retfound_dinov2_cfp.pth`). Use
`CONFIG["retfound_arch"] = "dinov2_vitl14_reg"` if your checkpoint is the register-token
variant. Only the linear head is trained; it is saved separately to `CONFIG["backbone_weights"]`.

## Run (notebook, after the `sys.path` cell that adds `src/`)

```python
!pip -q install -r requirements.txt

import config, dr_prep
# Section 0 downloads IDRiD (kagglehub) and builds the 5-grade ImageFolder:
dr_prep.prepare()                            # sets CONFIG["data_root"]
config.fundus_class_names(refresh=True)      # confirm the 5 grade-folder names

import train_backbone
train_backbone.train()          # linear-probe the head -> cache/retfound_dinov2_head_dr.pt

import runner
results, agg, failures = runner.run_all()                 # all grades (cached, fault-tolerant)
results, agg, failures = runner.run_all(["3_severe_npdr"]) # one grade
runner.grounding_table()                                  # FLAIR concept grounding: table + figure
runner.make_figures()                                     # concept overlays (grades 2/3/4)
```

`train_backbone.train()` must run first (it produces the head weights `load_backbone` reads);
it needs the RETFound encoder checkpoint in place. A grade that errors is printed and skipped;
completed grades are cached and resume free.

## Concept grounding (FLAIR)

`runner.grounding_table()` measures how well FLAIR visually grounds each concept via the
localized similarity maps `S` — **peak** (strongest localized cell, averaged over images) and
**mean** (diffuse presence) — **without dropping any concept**. It prints a per-grade table,
saves `results/concept_grounding_<run_tag>.json`, and renders a sorted bar/heatmap figure
(`viz.plot_concept_grounding`). It needs only FLAIR + the cached `S`, so it runs without the
trained head.

## Lesion-localization check (ground-truth masks)

`runner.run_all` reports two counts per grade: **diagnosed** (val images the backbone
classified correctly) and **concept-preserved** (of the evaluated images, how many keep the
prediction after concept reconstruction). See `runner.diagnosis_summary(results)`.

To check whether the heatmaps land on the *right lesions*,
[`src/lesion_eval.py`](src/lesion_eval.py) scores a grade's concept basis against **IDRiD**
segmentation pixel masks (microaneurysms / haemorrhages / hard & soft exudates / optic disc):

```python
import lesion_eval
lesion_eval.run_localization()          # downloads IDRiD masks via kagglehub
```

Each concept is mapped to a lesion mask type by keyword (`concept_lesion_type`); vessel /
neovascular concepts have no mask and are skipped. For every (image, concept) pair it computes
a **pointing-game** hit (heatmap peak inside the mask) and **mass-in-mask** (fraction of
heatmap energy on the lesion), then reports how many concepts localize and the overall hit
rate. The Kaggle slug is `lesion_eval.IDRID_SLUG` (override with `run_localization(root=...)`).

## Key knobs (`src/config.py`)

- `backbone` — `"retfound_dinov2"` (default; frozen ViT-L/14 + linear head) or the conv
  backbones `"densenet121"` / `"resnet34"` / `"resnet50"` / `"mobilenet_v2"`.
- `retfound_weights` / `retfound_arch` — RETFound encoder checkpoint path / DINOv2 arch.
- `grid` — probing grid (default 8: DINOv2's 16×16 tokens pooled 2×2).
- `n_train` / `n_val` — images/grade for fitting `W` / evaluating (caps; default 100 / 50).
- `min_eval_images` — below this many correctly-diagnosed val images, evaluate on all val
  images (default 5).
- `concept_mode` — `"per_class"` (default) or `"shared"`.
- `concept_curated` — use each grade's vocab verbatim (default `True`); `False` runs the
  paper's two-stage lexical+CLIP filter down to `r` concepts.
- `flair_weights` — local FLAIR checkpoint path, or `None` to download the pretrained weights.

## Concept vocabulary

[`concept_vocab.json`](concept_vocab.json), keyed per grade (`class_0_no_dr` …
`class_4_proliferative_dr`), grouped by lesion category with optional `description`/`notes`
metadata (ignored by the pipeline). With `concept_curated` (default) each grade's concepts are
used **verbatim** — variable count, no filtering, no fixed-`r` skip. Keys resolve to the grade
folders via `config.resolve_vocab_key` (tolerant of a `class_`/`grade_` prefix and of bare
`0`–`4` folders via `CONCEPT_ALIASES`). Editing the file invalidates the concept cache.
