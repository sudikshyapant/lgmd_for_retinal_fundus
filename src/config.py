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


# NOTE: keys tagged [suppl] are NOT specified in the paper body — their exact values
# live in the paper's supplementary material. The values here are informed defaults;
# change them in this one place once the supplementary is available.
CONFIG = {
    # --- data -----------------------------------------------------------------
    # ODIR-5K (Ocular Disease Intelligent Recognition), restricted to 5 single-label
    # classes: Normal, Diabetes (DR), Glaucoma, Cataract, AMD. The raw ODIR-5K download
    # is a flat image folder + a per-eye label table, so odir_prep.py first arranges it
    # into the ImageFolder layout (<split>/<class>/*.jpg) the rest of the pipeline expects.
    # The *active* class for concept discovery is set by select_class(name); class_index
    # is its position in the sorted class list (matching torchvision ImageFolder ordering,
    # so it lines up with the trained classifier's output index).
    "data_root": os.path.join(CACHE_DIR, "odir5k_imagefolder"),  # built by odir_prep.prepare()
    "train_dir": "train",
    "val_dir": "val",
    "test_dir": "test",
    "class_name": "Diabetes",      # default showcase class (resolved by select_class; the
    "class_index": None,           # fundus analogue of the old "tabby cat" default)
    # KNOB: max images per class drawn from the train split to train/fine-tune the backbone
    # classifier. ODIR-5K is far smaller than the old 50k set, so most classes have well
    # under this cap and are used in full; lower it only for a quick smoke run.
    "n_per_class": 2000,
    "n_train": 100,                # images/class (train split) used to fit the concept basis W
    "n_val": 50,                   # images/class (val split) used for inference + metrics
    "seed": 0,

    # --- backbone (encoder f + classifier head g) -----------------------------
    # The classifier LGMD explains. DenseNet-121 fine-tuned to a num_classes-way fundus
    # head (see train_backbone.py); weights loaded from backbone_weights. Reference setup
    # reaches ~0.755 balanced accuracy on the 5-class ODIR-5K split. Other registered
    # backbones (resnet34/50, mobilenet_v2) still work via model_utils._BACKBONES.
    "backbone": "densenet121",     # "densenet121" | "resnet34" | "resnet50" | "mobilenet_v2"
    "num_classes": 5,              # ODIR-5K classes N/D/G/C/A (resolved/checked against the folders)
    "feat_dim": 1024,              # p — encoder channels (densenet121 = 1024)
    "backbone_weights": os.path.join(CACHE_DIR, "densenet121_odir.pt"),  # trained 5-way weights
    # Backbone preprocessing: "clip_shared_224" shares CLIP's exact 224 crop so encoder
    # feature cells and CLIP red-circle cells cover identical pixels (aligns A_bar <-> S).
    # Part of the activation cache key so changing it recomputes activations.
    "backbone_preprocess": "clip_shared_224",
    "grid": 7,                     # h = w — encoder / CLIP probing grid, 7x7 (suppl. A1.6)

    # --- backbone training (train_backbone.py) --------------------------------
    "train_epochs": 15,            # fine-tuning epochs on the n_per_class subset
    "train_lr": 3e-4,              # Adam learning rate
    "train_batch_size": 64,
    "train_weight_decay": 1e-4,

    # --- CLIP (localized similarity maps) -------------------------------------
    # Default backend is BiomedCLIP (medical image-text), loaded via open_clip; set
    # clip_backend="openai" to fall back to generic CLIP ViT-B/16 for comparison.
    "clip_backend": "biomedclip",  # "biomedclip" | "openai"
    "clip_model": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",  # open_clip id
    "clip_model_openai": "openai/clip-vit-base-patch16",  # fallback HF CLIP ViT-B/16
    "prompt_template": "a fundus photograph showing {}",  # medical text prompt for concepts
    "circle_radius": 16,           # [suppl] red-circle radius (px) — suppl. A1.6 fixes a radius but gives no value
    "circle_width": 3,             # [suppl] red-circle stroke width (px) — "thin" outline, value unspecified
    "circle_color": "red",         # red localization marker (suppl. A1.6)

    # --- concepts -------------------------------------------------------------
    # "shared": one fixed clinical concept bank (flattened from concept_vocab.json,
    #   any grouping is cosmetic) used for every class -> r = bank size, concepts used
    #   verbatim (no lexical/CLIP reduction). Best for cross-disease comparison.
    # "per_class": paper-faithful — each class draws its own r concepts from a
    #   class-keyed vocab via the two-stage lexical+CLIP filter (see concept_vocab.per_class.json).
    "concept_mode": "per_class",
    "r": 25,                       # per_class only: concepts per class (paper fixes r = 25)
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
    "dedup_threshold": 0.95,       # CLIP-text cosine sim above which concepts are near-dups (suppl. A1.4)

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
    "run_tag": "diabetes_run2",

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
    "model": ["backbone", "backbone_preprocess", "num_classes", "backbone_weights"],  # encoder + head
    "con":   ["concept_vocab_hash", "concept_mode", "concept_filler_terms",      # concept vocab
              "concept_attribute_terms", "concept_word_min", "concept_word_max",
              "concept_proto_images", "dedup_threshold", "r", "class_name",
              "clip_backend", "clip_model"],
    "clip":  ["clip_backend", "clip_model", "prompt_template", "circle_radius",  # CLIP maps S
              "circle_width", "circle_color", "grid"],
    "pgd":   ["pgd_iters"],                                                    # basis W fit
    "infer": ["infer_iters"],                                                  # inference
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
# The 7 disease classes are the sub-directories of <data_root>/<train_dir>. We read
# them straight off disk rather than hardcoding, so the class list and index ordering
# can never drift from the dataset (and match torchvision ImageFolder, which sorts the
# class folders alphabetically — that ordering is what the trained classifier predicts).

_CLASS_NAMES = None


def _canon(name):
    """snake_case key for a class folder name, e.g. 'Diabetic Retinopathy' -> 'diabetic_retinopathy'."""
    return re.sub(r"[\s\-]+", "_", name.strip().lower())


# Dataset folders don't always match the concept_vocab.json keys verbatim — AMD and the
# normal class are the usual offenders (e.g. a folder named "AMD" or "Age related Macular
# Degeneration" vs. the key "amd"; "Normal" vs. "normal_fundus"). Map any such folder name
# to its vocab key here; both sides are compared canonically (snake_case). Extend this if
# fundus_class_names() shows a folder whose name doesn't canonically equal its vocab key.
CONCEPT_ALIASES = {
    # AMD variants -> "amd"
    "amd": "amd",
    "armd": "amd",
    "age_related_macular_degeneration": "amd",
    "age_related_macular_degeneration_amd": "amd",
    "macular_degeneration": "amd",
    # normal / healthy variants -> "normal_fundus"
    "normal": "normal_fundus",
    "normal_fundus": "normal_fundus",
    "healthy": "normal_fundus",
    "no_disease": "normal_fundus",
    # diabetic-retinopathy variants -> "diabetic_retinopathy" (ODIR labels this "Diabetes")
    "diabetes": "diabetic_retinopathy",
    "diabetic": "diabetic_retinopathy",
    "diabetic_retinopathy": "diabetic_retinopathy",
    "dr": "diabetic_retinopathy",
}


def resolve_vocab_key(class_name, available_keys):
    """Map a dataset class/folder name to its concept_vocab.json key.

    Tries an exact canonical match first, then the CONCEPT_ALIASES table. Returns the
    matching key from `available_keys`, or None if nothing matches.
    """
    by_canon = {_canon(k): k for k in available_keys}
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
# the fundus analogue of the paper's showcase rows. Matched canonically against the
# actual folder names, so case/spacing need not be exact ("cataract" ~ "Cataract").
FIGURE_CLASSES = ["Diabetes"]


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
