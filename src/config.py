"""Central configuration for the LGMD project.

A single CONFIG dict holds every hyperparameter. The module auto-detects whether
it runs on Google Colab and, if so, mounts Google Drive so that all heavy /
constant artifacts (activations, CLIP maps, learned basis, results, visualizations)
persist across sessions — while the code itself is re-cloned from GitHub each time.
"""

import hashlib
import json
import os
import re

_THIS = os.path.dirname(os.path.abspath(__file__))   # .../lgmd/src
REPO_ROOT = os.path.dirname(_THIS)                   # .../lgmd


def _in_colab():
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


IN_COLAB = _in_colab()

# --- Persistent storage root -------------------------------------------------
# On Colab everything persistent lives on Drive; locally it lives in the repo.
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE_DIR = "/content/drive/MyDrive/lgmd"
else:
    BASE_DIR = REPO_ROOT

CACHE_DIR = os.path.join(BASE_DIR, "cache")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
VIZ_DIR = os.path.join(BASE_DIR, "viz")
for _d in (CACHE_DIR, RESULTS_DIR, VIZ_DIR):
    os.makedirs(_d, exist_ok=True)


# Stored concept vocabulary: a JSON table keyed by class name (replaces LLM
# generation). Travels with the repo so runs are reproducible and offline. Each
# class maps to candidate concepts (over-provided; filtered down to r in concepts.py).
CONCEPT_VOCAB_PATH = os.path.join(REPO_ROOT, "concept_vocab.json")


def _vocab_hash():
    """Short hash of the vocabulary file so edits to it invalidate concept caches."""
    if not os.path.exists(CONCEPT_VOCAB_PATH):
        return "missing"
    with open(CONCEPT_VOCAB_PATH, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:8]


# Generic filler terms removed during concept filtering (rule i; suppl. A1.3 lists
# "animal", "object", "scene" as examples). Kept deliberately minimal: a concept is
# dropped if *any* of its words is filler, so domain qualifiers that legitimately appear
# in real concepts ("tessellated fundus", "healthy retina") must NOT be listed here.
_FILLER_TERMS = ["object", "scene", "thing", "things", "background",
                 "image", "photo", "picture", "stuff", "item", "appearance"]

# Visual-attribute lexicon (suppl. A1.3): a concept carrying any of these words is
# preserved even when it partially overlaps the class name. Re-tuned for retinal
# fundus pathology — the localized lesions, structures, colors and morphologies a
# clinician reads off a fundus image (so e.g. "diabetic macular edema" survives the
# class-name-overlap rule for the Diabetic Retinopathy class).
_ATTRIBUTE_TERMS = [
    # color / tone
    "red", "yellow", "white", "pale", "dark", "bright", "orange", "grey", "gray",
    "creamy", "waxy", "silver", "copper", "reddish", "yellowish",
    # lesions / morphology
    "hemorrhage", "hemorrhages", "haemorrhage", "microaneurysm", "microaneurysms",
    "exudate", "exudates", "drusen", "neovascularization", "neovascular",
    "aneurysm", "edema", "oedema", "cotton", "wool", "spot", "spots", "dot", "blot",
    "flame", "ring", "deposit", "deposits", "opacity", "opacities", "lesion",
    "atrophy", "atrophic", "scar", "scarring", "fibrosis", "occlusion", "leakage",
    "pigment", "pigmentation", "pigmented", "hyperpigmentation",
    # optic disc / cup / vessels
    "disc", "disk", "cup", "cupping", "rim", "notch", "notching", "pallor",
    "vessel", "vessels", "vascular", "artery", "arteriolar", "vein", "venous",
    "tortuous", "tortuosity", "nicking", "narrowing", "attenuated", "dilated",
    "beading", "collateral",
    # location / region
    "macula", "macular", "foveal", "fovea", "peripheral", "peripapillary",
    "temporal", "nasal", "central", "diffuse", "focal",
    # texture / shape
    "round", "linear", "punctate", "irregular", "well", "defined", "blurred",
    "swollen", "raised", "flat", "crescent", "tessellated", "tigroid", "myopic",
]


# --- backbone switch ---------------------------------------------------------
# Pick the encoder here; arch / checkpoint filename / probing grid all derive from it so
# the RETFound variants stay mutually consistent. RETFound comes in two flavors:
#   retfound_mae    — MAE ViT-L/16 (patch 16 -> 14x14 native tokens); the original
#                     RETFound (Nature 2023). CFP encoder: RETFound_mae_natureCFP.pth.
#   retfound_dinov2 — DINOv2 ViT-L/14 (patch 14 -> 16x16 native tokens). No CFP-specific
#                     checkpoint exists publicly; the MEH weight is the color-fundus one.
# The torchvision conv backbones (densenet121/resnet34/resnet50/mobilenet_v2) bypass this
# block entirely (they set their own arch/grid in model_utils).
_BACKBONE = "retfound_mae"   # retfound_mae | retfound_dinov2 | densenet121 | resnet34 | resnet50 | mobilenet_v2
_IS_MAE = _BACKBONE == "retfound_mae"
_RETFOUND_ARCH = "vit_large_patch16_224" if _IS_MAE else "dinov2_vitl14"
_RETFOUND_CKPT = "RETFound_mae_natureCFP.pth" if _IS_MAE else "retfound_dinov2_cfp.pth"
# Native token grid = img_size / patch: 14 for MAE (224/16), 16 for DINOv2 (224/14).
# Probing grid must divide it so GAP == mean-over-tokens stays exact: 7 for MAE, 8 for DINOv2.
_GRID = 7 if _IS_MAE else 8


# NOTE: keys tagged [suppl] are NOT specified in the paper body — their exact values
# live in the paper's supplementary material. The values here are informed defaults;
# change them in this one place once the supplementary is available.
CONFIG = {
    # --- data -----------------------------------------------------------------
    # DR severity grading: 5 ordinal classes 0-4 (No DR, Mild NPDR, Moderate NPDR,
    # Severe NPDR, Proliferative DR). Arrange the dataset as <split>/<grade>/*.jpg for
    # torchvision ImageFolder. IMPORTANT: ImageFolder assigns class indices by sorting
    # the folder names alphabetically, and the classifier's output index must match. Name
    # the folders with a numeric prefix so alphabetical order == grade order — the
    # convention used throughout this config (and in concept_vocab.json):
    #     0_no_dr  1_mild_npdr  2_moderate_npdr  3_severe_npdr  4_proliferative_dr
    # (readable names like "Mild"/"Moderate"/"No DR" sort in the WRONG order.) The active
    # class for concept discovery is set by select_class(name); class_index is its position
    # in that sorted list.
    "data_root": os.path.join(CACHE_DIR, "dr_grading_imagefolder"),  # <split>/<grade>/*.jpg
    "train_dir": "train",
    "val_dir": "val",
    "test_dir": "test",
    "class_name": "2_moderate_npdr",  # default showcase grade (resolved by select_class)
    "class_index": None,
    # KNOB: max images per class drawn from the train split to train/fine-tune the backbone
    # classifier; lower it only for a quick smoke run.
    "n_per_class": 2000,
    "n_train": 100,                # images/class (train split) used to fit the concept basis W
    "n_val": 50,                   # images/class (val split) used for inference + metrics
    # Evaluation normally uses only val images the backbone diagnoses correctly. For a scarce
    # grade whose weak head gets fewer than this many right (e.g. Mild NPDR, ~4 val images),
    # fall back to ALL val images so the grade still produces metrics instead of being skipped
    # — reconstruction faithfulness then leans on prediction agreement (label-independent).
    "min_eval_images": 5,
    "seed": 0,

    # --- backbone (encoder f + classifier head g) -----------------------------
    # The classifier LGMD explains. Default: RETFound (MAE ViT-L/16) used as a FROZEN
    # foundation encoder with a linear DR-grading head probed on top (train_backbone.py).
    # The encoder is NOT fine-tuned — only the Linear(1024 -> num_classes) head is trained,
    # and backbone_weights stores just that head. The torchvision conv backbones
    # (densenet121/resnet34/resnet50/mobilenet_v2) still work via model_utils._BACKBONES.
    # Select the backbone at the _BACKBONE switch above (not here).
    "backbone": _BACKBONE,
    "num_classes": 5,              # 5 DR severity grades (checked against the folders)
    "feat_dim": 1024,              # p — encoder channels (ViT-L / densenet121 both = 1024)
    # Trained linear head only; per-backbone filename so a DINOv2-trained head is never
    # loaded onto the MAE encoder (they share the Linear shape but live in different feature spaces).
    "backbone_weights": os.path.join(CACHE_DIR, f"{_BACKBONE}_head_dr.pt"),
    # RETFound foundation encoder architecture: MAE built via timm (vit_large_patch16_224),
    # DINOv2 fetched from torch.hub (facebookresearch/dinov2; use 'dinov2_vitl14_reg' for the
    # register-token variant). SSL weights loaded from retfound_weights.
    "retfound_arch": _RETFOUND_ARCH,
    "retfound_img_size": 224,      # MAE: 224/16 = 14x14 native tokens; DINOv2: 224/14 = 16x16
    "retfound_weights": os.path.join(CACHE_DIR, _RETFOUND_CKPT),  # RETFound SSL encoder ckpt
    # Backbone preprocessing: "clip_shared_224" shares CLIP's exact 224 crop so encoder
    # feature cells and CLIP red-circle cells cover identical pixels (aligns A_bar <-> S).
    # Part of the activation cache key so changing it recomputes activations.
    "backbone_preprocess": "clip_shared_224",
    "grid": _GRID,                 # h = w — encoder / CLIP probing grid. Exact 2x2 pool of the
                                   # native token grid: 7x7 for MAE (14), 8x8 for DINOv2 (16); A1.6.

    # --- backbone training (train_backbone.py) --------------------------------
    "train_epochs": 15,            # fine-tuning epochs on the n_per_class subset
    "train_lr": 3e-4,              # Adam learning rate
    "train_batch_size": 64,
    "train_weight_decay": 1e-4,

    # --- FLAIR VLM (localized similarity maps) --------------------------------
    # The image-text model behind the semantic maps S: FLAIR "A Foundation
    # LAnguage-Image model of the Retina" (github.com/jusiro/FLAIR), a CLIP trained on
    # fundus images, so its image-text space is domain-matched to retinal pathology.
    "flair_weights": None,         # FLAIR checkpoint path; None -> download pretrained weights
    "prompt_template": "a fundus photograph showing {}",  # medical text prompt for concepts
    "circle_radius": round(112 / _GRID),  # [suppl] red-circle radius (px) — half a grid cell
                                   # (224/grid/2): 16 for the 7x7 MAE grid, 14 for the 8x8 DINOv2 grid
    "circle_width": 3,             # [suppl] red-circle stroke width (px) — "thin" outline, value unspecified
    "circle_color": "red",         # red localization marker (suppl. A1.6)

    # --- concepts -------------------------------------------------------------
    # "shared": one fixed clinical concept bank (flattened from concept_vocab.json,
    #   any grouping is cosmetic) used for every class -> r = bank size, concepts used
    #   verbatim (no lexical/CLIP reduction). Best for cross-disease comparison.
    # "per_class": paper-faithful — each class draws its own r concepts from a
    #   class-keyed vocab via the two-stage lexical+CLIP filter (see concept_vocab.per_class.json).
    "concept_mode": "per_class",
    # per_class + concept_curated: use each grade's hand-curated concept list verbatim
    # (variable count = however many are listed; no lexical/CLIP filtering, no fixed-r skip).
    # Suits the grouped per-grade concept_vocab.json. Set False to run the paper's two-stage
    # lexical+CLIP filter down to exactly `r` concepts instead.
    "concept_curated": True,
    "r": 25,                       # per_class + non-curated: concepts per class (paper fixes r = 25)
    # per_class only: if fewer than r concepts survive filtering, skip the class (raise
    # InsufficientConcepts so a multi-class run records it and continues) instead of
    # proceeding with a short basis. Recover via concepts.add_concepts + re-evaluate.
    "concept_skip_if_short": True,
    "concept_vocab_path": CONCEPT_VOCAB_PATH,  # stored concept table (no LLM)
    "concept_vocab_hash": _vocab_hash(),       # content hash for cache invalidation
    "concept_word_min": 2,         # lexical filter: 2-3 words (suppl. A1.3) — drops bare single-word
    "concept_word_max": 3,         # lesion names and long phrases; keeps e.g. "cotton wool spots"
    "concept_filler_terms": _FILLER_TERMS,        # generic filler removed, rule i (suppl. A1.3)
    "concept_attribute_terms": _ATTRIBUTE_TERMS,  # attribute words exempt from class-name overlap (suppl. A1.3)
    "concept_proto_images": 100,   # up to N class images for CLIP relevance ranking (suppl. A1.4)
    "dedup_threshold": 1.0,        # CLIP-text cosine sim above which concepts are near-dups.
                                   # 1.0 = disabled: no concept is ever dropped as a near-dup
                                   # (all pass). Only used in non-curated per_class mode.

    # --- optimization ---------------------------------------------------------
    # NOTE: PGD step sizes are NOT free params — the paper fixes them via the spectral
    # norm (Lipschitz constant), so they are computed, not configured here.
    "pgd_iters": 500,              # [suppl] PGD iterations to fit the basis W
    "infer_iters": 50,              # [suppl] PGD refinement iterations at inference

    # --- baselines ------------------------------------------------------------
    "craft_levels": 2,             # [suppl] CRAFT recursive-NMF depth
    "nmf_max_iter": 2000,          # NMF solver iterations (raised to reach convergence)
    "face_lambda": 1.0,            # [suppl] FACE KL-regularization weight
    "face_iters": 300,             # [suppl] FACE optimization iterations
    "face_lr": 1e-2,               # [suppl] FACE Adam learning rate

    # --- metrics --------------------------------------------------------------
    "cins_metric": "prob",         # [suppl] C-Ins 'model performance': "prob" | "accuracy"

    # --- run tag --------------------------------------------------------------
    # Namespaces this run's outputs. It is mixed into every cache_name() hash (so all
    # derived caches — activations, S, W, metrics — rebuild fresh) and prefixes every saved
    # figure filename (so new figures never overwrite an old run's). Bump this string to
    # force a clean run without hand-deleting caches. The trained backbone weights are a
    # fixed path (not cache-keyed), so changing run_tag never triggers a retrain.
    "run_tag": "dr_grading_run1",

    # --- paths ----------------------------------------------------------------
    "cache_dir": CACHE_DIR,
    "results_dir": RESULTS_DIR,
    "viz_dir": VIZ_DIR,
}


# Cache invalidation: each artifact's filename embeds a short hash of the CONFIG
# values it depends on, so changing any of those values rebuilds only the affected
# caches. Call sites name the dependency groups (see below) that apply.
_CACHE_DEPS = {
    "data":  ["class_index", "class_name", "n_train", "n_val", "seed"],          # which images
    "model": ["backbone", "backbone_preprocess", "num_classes", "backbone_weights",   # encoder + head
              "retfound_arch", "retfound_weights"],
    "con":   ["concept_vocab_hash", "concept_mode", "concept_curated",            # concept vocab
              "concept_filler_terms", "concept_attribute_terms",
              "concept_word_min", "concept_word_max",
              "concept_proto_images", "dedup_threshold", "r", "class_name",
              "flair_weights"],
    "clip":  ["flair_weights",                                                   # VLM maps S
              "prompt_template", "circle_radius",
              "circle_width", "circle_color", "grid"],
    "pgd":   ["pgd_iters"],                                                    # basis W fit
    "infer": ["infer_iters", "min_eval_images"],                              # inference + eval set
    "base":  ["craft_levels", "face_lambda", "face_iters", "face_lr"],         # baselines
    "cins":  ["cins_metric"],                                                  # C-Ins metric
}


def cache_name(base, ext, *groups):
    """Build a cache filename '<base>_<hash><ext>'.

    The hash covers the CONFIG values in the named dependency groups, so editing any
    of them yields a new filename (and a fresh cache) instead of reusing a stale one.
    """
    keys = sorted({k for g in groups for k in _CACHE_DEPS[g]})
    payload = {k: CONFIG[k] for k in keys}
    # A non-empty run_tag busts every cache into a fresh namespace. An empty run_tag adds
    # nothing to the payload, so the hash exactly matches the original pre-tag caches —
    # set CONFIG["run_tag"] = "" to switch back to the old artifacts.
    tag = CONFIG.get("run_tag", "")
    if tag:
        payload["run_tag"] = tag
    blob = json.dumps(payload, sort_keys=True, default=str)
    digest = hashlib.md5(blob.encode()).hexdigest()[:8]
    return f"{base}_{digest}{ext}"


def get_secret(name):
    """Fetch a secret (API key / token) without ever committing it.

    Lookup order: Colab Secrets -> environment variable -> gitignored secrets.json.
    """
    if IN_COLAB:
        try:
            from google.colab import userdata
            val = userdata.get(name)
            if val:
                return val
        except Exception:
            pass
    if name in os.environ:
        return os.environ[name]
    spath = os.path.join(REPO_ROOT, "secrets.json")
    if os.path.exists(spath):
        with open(spath) as f:
            val = json.load(f).get(name)
            if val:
                return val
    raise KeyError(
        f"Secret '{name}' not found. Set it in Colab Secrets, an env var, "
        f"or a (gitignored) secrets.json at the repo root."
    )


# ---------------------------------------------------------------------------
# Fundus class registry (resolved from the dataset's own folders)
# ---------------------------------------------------------------------------
# The 5 DR-grade classes are the sub-directories of <data_root>/<train_dir>. We read
# them straight off disk rather than hardcoding, so the class list and index ordering
# can never drift from the dataset (and match torchvision ImageFolder, which sorts the
# class folders alphabetically — that ordering is what the trained classifier predicts).

_CLASS_NAMES = None


def _canon(name):
    """snake_case key for a class folder name, e.g. 'Diabetic Retinopathy' -> 'diabetic_retinopathy'."""
    return re.sub(r"[\s\-]+", "_", name.strip().lower())


# The canonical convention (folder name == concept_vocab.json key) is the numeric-prefixed
# grade name, e.g. "2_moderate_npdr" — with that, resolve_vocab_key matches directly and no
# alias is needed. This table only exists so alternative folder spellings still resolve:
# bare numeric folders ("0".."4"), short forms ("mild", "pdr"), or a "Proliferative DR
# (PDR)" folder whose parentheses survive canonicalization. Both sides are compared
# canonically (snake_case); each value must be an actual key in concept_vocab.json.
CONCEPT_ALIASES = {
    # grade 0 — No DR
    "0": "0_no_dr", "no_dr": "0_no_dr", "nodr": "0_no_dr", "normal": "0_no_dr", "no_dr_grade": "0_no_dr",
    # grade 1 — Mild NPDR
    "1": "1_mild_npdr", "mild": "1_mild_npdr", "mild_npdr": "1_mild_npdr", "mild_dr": "1_mild_npdr",
    # grade 2 — Moderate NPDR
    "2": "2_moderate_npdr", "moderate": "2_moderate_npdr", "moderate_npdr": "2_moderate_npdr",
    "moderate_dr": "2_moderate_npdr",
    # grade 3 — Severe NPDR
    "3": "3_severe_npdr", "severe": "3_severe_npdr", "severe_npdr": "3_severe_npdr",
    "severe_dr": "3_severe_npdr",
    # grade 4 — Proliferative DR
    "4": "4_proliferative_dr", "pdr": "4_proliferative_dr", "proliferative_dr": "4_proliferative_dr",
    "proliferative": "4_proliferative_dr", "proliferative_dr_pdr": "4_proliferative_dr",
    "proliferative_dr_(pdr)": "4_proliferative_dr",
}


def resolve_vocab_key(class_name, available_keys):
    """Map a dataset class/folder name to its concept_vocab.json key.

    Tries an exact canonical match first, then the CONCEPT_ALIASES table. Returns the
    matching key from `available_keys`, or None if nothing matches.
    """
    by_canon = {}
    for k in available_keys:
        ck_k = _canon(k)
        by_canon.setdefault(ck_k, k)
        # tolerate a leading "class_"/"grade_" prefix on the vocab key, so a key like
        # "class_0_no_dr" is reachable from the folder "0_no_dr".
        by_canon.setdefault(re.sub(r"^(class|grade)_", "", ck_k), k)
    ck = _canon(class_name)
    if ck in by_canon:
        return by_canon[ck]
    aliased = CONCEPT_ALIASES.get(ck)
    if aliased is not None:
        return by_canon.get(_canon(aliased))
    return None


def fundus_class_names(refresh=False):
    """Sorted disease-class folder names under <data_root>/<train_dir>.

    Cached after first call; pass refresh=True to re-list. Run this once at the start of
    a session to confirm the exact folder names (concept_vocab.json keys must match their
    canonical/snake_case form — see concepts.py).
    """
    global _CLASS_NAMES
    if _CLASS_NAMES is None or refresh:
        train = os.path.join(CONFIG["data_root"], CONFIG["train_dir"])
        if not os.path.isdir(train):
            raise FileNotFoundError(
                f"Train directory not found: {train}. Set CONFIG['data_root']/"
                f"['train_dir'] to your dataset's ImageFolder root."
            )
        _CLASS_NAMES = sorted(
            d for d in os.listdir(train) if os.path.isdir(os.path.join(train, d))
        )
    return _CLASS_NAMES


# Qualitative concept overlays are saved only for these classes (the figure subset) —
# the showcase grades. Matched canonically against the actual folder names, so case/
# spacing need not be exact. These must be real folder names (not vocab aliases).
FIGURE_CLASSES = ["2_moderate_npdr", "3_severe_npdr", "4_proliferative_dr"]


def select_class(name):
    """Make `name` the active class: set class_name / class_index.

    `name` may be the exact folder name or any case/spacing variant (matched canonically).
    Subsequent cache_name() calls key on these, so each class gets its own caches.
    Returns the resolved class index (position in the sorted class list).
    """
    names = fundus_class_names()
    canon = {_canon(n): n for n in names}
    folder = canon.get(_canon(name))
    if folder is None:
        raise KeyError(f"{name!r} not among dataset classes. Available: {names}")
    CONFIG["class_name"] = folder
    CONFIG["class_index"] = names.index(folder)
    return CONFIG["class_index"]
