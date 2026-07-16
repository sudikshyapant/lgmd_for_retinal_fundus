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


# Stored concept vocabulary: a JSON table keyed by grade. It travels with the repo so
# runs are reproducible and offline (no LLM). Each grade's concepts are used verbatim.
CONCEPT_VOCAB_PATH = os.path.join(REPO_ROOT, "concept_vocab.json")


def _vocab_hash():
    """Short hash of the vocabulary file so edits to it invalidate concept caches."""
    if not os.path.exists(CONCEPT_VOCAB_PATH):
        return "missing"
    with open(CONCEPT_VOCAB_PATH, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:8]


# --- backbone ----------------------------------------------------------------
# The classifier LGMD explains: RETFound (MAE ViT-L/16), the original RETFound (Nature
# 2023). It is used as a FROZEN foundation encoder with a linear DR-grading head probed on
# top (train_backbone.py) — only the Linear(1024 -> num_classes) head is trained. The MAE
# encoder is built via timm (vit_large_patch16_224, patch 16 -> 14x14 native tokens); its
# SSL weights are loaded from retfound_weights (RETFound_mae_natureCFP.pth).
# The probing grid must divide the 14x14 native token grid so GAP == mean-over-tokens stays
# exact: 7x7 (a 2x2 pool of the 14x14 tokens).


# NOTE: keys tagged [suppl] are NOT specified in the paper body — their exact values
# live in the paper's supplementary material. The values here are informed defaults.
CONFIG = {
    # --- data -----------------------------------------------------------------
    # DR severity grading: 5 ordinal classes 0-4 (No DR, Mild NPDR, Moderate NPDR,
    # Severe NPDR, Proliferative DR). Arrange the dataset as <split>/<grade>/*.jpg for
    # torchvision ImageFolder. IMPORTANT: ImageFolder assigns class indices by sorting
    # the folder names alphabetically, and the classifier's output index must match. Name
    # the folders with a numeric prefix so alphabetical order == grade order:
    #     0_no_dr  1_mild_npdr  2_moderate_npdr  3_severe_npdr  4_proliferative_dr
    # The active grade for concept discovery is set by select_class(name); class_index is
    # its position in that sorted list.
    "data_root": os.path.join(CACHE_DIR, "dr_grading_imagefolder"),  # <split>/<grade>/*.jpg
    "train_dir": "train",
    "val_dir": "val",
    "test_dir": "test",
    "class_name": "2_moderate_npdr",  # default showcase grade (resolved by select_class)
    "class_index": None,
    # KNOB: max images per class drawn from the train split to train the linear head;
    # lower it only for a quick smoke run.
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
    # RETFound MAE ViT-L/16, frozen encoder + linear DR-grading head. The encoder is NOT
    # fine-tuned; backbone_weights stores just the trained head.
    "backbone": "retfound_mae",
    "num_classes": 5,              # 5 DR severity grades (checked against the folders)
    "feat_dim": 1024,              # p — encoder channels (ViT-L)
    "backbone_weights": os.path.join(CACHE_DIR, "retfound_mae_head_dr.pt"),  # trained head only
    "retfound_arch": "vit_large_patch16_224",   # timm MAE architecture
    "retfound_img_size": 224,      # 224/16 = 14x14 native tokens
    "retfound_weights": os.path.join(CACHE_DIR, "RETFound_mae_natureCFP.pth"),  # SSL encoder ckpt
    # Backbone preprocessing: "clip_shared_224" shares FLAIR's exact 224 crop so encoder
    # feature cells and FLAIR red-circle cells cover identical pixels (aligns A_bar <-> S).
    # Part of the activation cache key so changing it recomputes activations.
    "backbone_preprocess": "clip_shared_224",
    "grid": 7,                     # h = w — encoder / FLAIR probing grid. Exact 2x2 pool of
                                   # the 14x14 native token grid.

    # --- backbone training (train_backbone.py) --------------------------------
    "train_epochs": 15,            # linear-probe epochs on the n_per_class subset
    "train_lr": 3e-4,              # Adam learning rate
    "train_batch_size": 64,
    "train_weight_decay": 1e-4,

    # --- FLAIR VLM (localized similarity maps) --------------------------------
    # The image-text model behind the semantic maps S: FLAIR "A Foundation LAnguage-Image
    # model of the Retina" (github.com/jusiro/FLAIR), a CLIP trained on fundus images, so
    # its image-text space is domain-matched to retinal pathology.
    "flair_weights": None,         # FLAIR checkpoint path; None -> download pretrained weights
    "prompt_template": "a fundus photograph showing {}",  # medical text prompt for concepts
    "circle_radius": round(112 / 7),  # [suppl] red-circle radius (px) — half a 7x7 grid cell
    "circle_width": 3,             # [suppl] red-circle stroke width (px) — "thin" outline
    "circle_color": "red",         # red localization marker (suppl. A1.6)

    # --- concepts -------------------------------------------------------------
    # Each grade's hand-curated clinical concepts (concept_vocab.json) are used VERBATIM:
    # every listed concept becomes a basis column, in order — no LLM, no lexical/CLIP
    # filtering, variable count per grade.
    "concept_vocab_path": CONCEPT_VOCAB_PATH,  # stored concept table (no LLM)
    "concept_vocab_hash": _vocab_hash(),       # content hash for cache invalidation

    # --- optimization ---------------------------------------------------------
    # NOTE: PGD step sizes are NOT free params — the paper fixes them via the spectral
    # norm (Lipschitz constant), so they are computed, not configured here.
    "pgd_iters": 500,              # [suppl] PGD iterations to fit the basis W
    "infer_iters": 50,             # [suppl] PGD refinement iterations at inference

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
    # figure filename. Bump this string to force a clean run without hand-deleting caches.
    # The trained backbone weights are a fixed path (not cache-keyed), so changing run_tag
    # never triggers a retrain.
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
    "con":   ["concept_vocab_hash", "class_name"],                                # concept vocab
    "clip":  ["flair_weights", "prompt_template", "circle_radius",                # VLM maps S
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
    """snake_case key for a class folder name, e.g. '2 Moderate NPDR' -> '2_moderate_npdr'."""
    return re.sub(r"[\s\-]+", "_", name.strip().lower())


# The canonical convention (folder name == concept_vocab.json key) is the numeric-prefixed
# grade name, e.g. "2_moderate_npdr". This table only exists so alternative folder spellings
# still resolve: bare numeric folders ("0".."4"), short forms ("mild", "pdr"), etc. Both
# sides are compared canonically (snake_case); each value must be an actual grade folder.
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


# Qualitative concept overlays are saved only for these grades (the figure subset) — the
# showcase grades. Matched canonically against the actual folder names, so case/spacing
# need not be exact. These must be real folder names (not vocab aliases).
FIGURE_CLASSES = ["2_moderate_npdr", "3_severe_npdr", "4_proliferative_dr"]


def select_class(name):
    """Make `name` the active grade: set class_name / class_index.

    `name` may be the exact folder name or any case/spacing variant (matched canonically).
    Subsequent cache_name() calls key on these, so each grade gets its own caches.
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
