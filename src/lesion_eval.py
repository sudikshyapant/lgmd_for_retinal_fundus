"""Lesion-mask localization eval: do the concept heatmaps land on real lesions?

The main pipeline only checks whether the concept reconstruction preserves the *diagnosis*.
This module adds a ground-truth *localization* check using IDRiD (Indian Diabetic
Retinopathy Image Dataset), whose images carry pixel masks for microaneurysms /
haemorrhages / hard & soft exudates / optic disc. We run a DR grade's concept basis on
those images, turn each concept's coefficients into a heatmap, and score how well it falls
inside the matching lesion mask (mass-in-mask + pointing-game). Result: how many concept
heatmaps are correctly localized.
"""

import os

import numpy as np
from PIL import Image

import flair_maps
import lgmd
import metrics
import model_utils
import runner
from config import CONFIG
from data_utils import clip_preprocess
from viz import _heatmap

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
LESION_TYPES = ("MA", "HE", "EX", "SE", "OD")

# IDRiD via kagglehub (override with run_localization(root=...) if the slug moves).
IDRID_SLUG = "aaryapatel98/indian-diabetic-retinopathy-image-dataset"

# Ordered keyword rules, first match wins. Maps a concept -> lesion type: soft-exudate /
# cotton-wool cues are checked before the generic "exudate" so "cotton wool spots" -> SE
# (not EX); vessel / neovascular concepts have no mask and fall through to None.
_LESION_RULES = [
    ("SE", ["cotton wool", "cotton", "soft exudate", "wool", "ischemic", "ischaemic"]),
    ("MA", ["microaneurysm", "aneurysm"]),
    ("HE", ["hemorrhage", "haemorrhage", "bleeding", "blood", "boat", "blot"]),
    ("EX", ["hard exudate", "exudate", "exudation", "lipid"]),
    ("OD", ["optic disc", "optic disk", "cupping", "disc", "disk", "optic nerve"]),
]
_MASK_SUFFIX = {"_ma": "MA", "_he": "HE", "_ex": "EX", "_se": "SE", "_od": "OD"}


def concept_lesion_type(concept):
    """Lesion-mask type (MA/HE/EX/SE/OD) a concept names, or None if none applies."""
    c = concept.lower()
    for t, kws in _LESION_RULES:
        if any(k in c for k in kws):
            return t
    return None


def _mask_type(path):
    """(lesion_type, image_stem) if `path` is an IDRiD lesion mask, else (None, None).

    IDRiD masks are suffix-named per type: IDRiD_01_EX.tif, IDRiD_01_OD.tif, ...
    """
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    for suf, t in _MASK_SUFFIX.items():
        if stem.endswith(suf):
            return t, stem[: -len(suf)]
    return None, None


def load_records(root, limit=None):
    """Walk the IDRiD root -> [(image_path, {type: mask_path})].

    An image is matched to its masks by stem. Images with no lesion mask are dropped
    (nothing to score). Deterministic order; `limit` keeps the first N records.
    """
    imgs, masks = {}, {}                                 # stem->img ; (stem,type)->mask
    for r, _d, files in os.walk(root):
        for f in files:
            if not f.lower().endswith(_IMG_EXTS):
                continue
            p = os.path.join(r, f)
            t, stem = _mask_type(p)
            if t:
                masks[(stem, t)] = p
            else:
                imgs.setdefault(os.path.splitext(f)[0].lower(), p)

    records = []
    for stem, ipath in sorted(imgs.items()):
        m = {t: masks[(stem, t)] for t in LESION_TYPES if (stem, t) in masks}
        if m:
            records.append((ipath, m))
    return records[:limit] if limit else records


def download(root=None):
    """Return the IDRiD root: `root` if given, else a kagglehub download."""
    if root:
        return root
    import kagglehub
    return kagglehub.dataset_download(IDRID_SLUG)


def _load_mask(path, size=224):
    """Load a lesion mask and align it to the heatmap's 224 crop (same geometry)."""
    m = clip_preprocess(Image.open(path).convert("L"), size)
    return np.array(m) > 0


def run_localization(root=None, class_name="2_moderate_npdr", limit=None,
                     require_correct=True, localized_frac=0.5, print_table=True):
    """Score how well the class's concept heatmaps localize on IDRiD lesion masks.

    Runs the (cached) concept basis for `class_name` on IDRiD, then for every (image,
    concept) pair whose concept maps to an available lesion mask computes a pointing-game
    hit + mass-in-mask. With require_correct, only images the backbone still diagnoses as
    `class_name` are scored (the heatmaps we'd actually trust). A concept counts as
    "localized" when it hits on >= `localized_frac` of its scored images.

    Returns a summary dict (per-concept stats + heatmap/concept totals); prints a table.
    """
    model, transform = model_utils.load_backbone()
    vlm = flair_maps.VLM()
    concepts, W, _ = runner.class_basis(class_name, model, transform, vlm)
    label = CONFIG["class_index"]

    root = download(root)
    records = load_records(root, limit=limit)
    if not records:
        raise RuntimeError(f"No IDRiD image+mask records found under {root}.")

    imgs = [Image.open(p).convert("RGB") for p, _ in records]
    Z = model_utils.extract_activations(model, transform, imgs, desc="IDRiD act")
    if require_correct:
        keep = (model_utils.logits_from_Z(model, Z).argmax(-1) == label).tolist()
        records = [rec for rec, k in zip(records, keep) if k]
        Z = Z[[i for i, k in enumerate(keep) if k]]
        if not records:
            raise RuntimeError(f"No IDRiD image was diagnosed as '{class_name}'.")

    grid, r = CONFIG["grid"], len(concepts)
    S = lgmd.infer(lgmd.unfold(Z), W).reshape(len(records), grid * grid, r)

    per_concept = {}
    for i, (_ipath, mdict) in enumerate(records):
        masks = {t: _load_mask(p) for t, p in mdict.items()}
        for k, concept in enumerate(concepts):
            t = concept_lesion_type(concept)
            if t is None or t not in masks:
                continue
            sc = metrics.heatmap_localization(_heatmap(S[i, :, k], grid), masks[t])
            d = per_concept.setdefault(concept, {"type": t, "scored": 0, "hits": 0, "mass": 0.0})
            d["scored"] += 1
            d["hits"] += int(sc["hit"])
            d["mass"] += sc["mass_in_mask"]

    scored = sum(d["scored"] for d in per_concept.values())
    hits = sum(d["hits"] for d in per_concept.values())
    localized = sum(1 for d in per_concept.values()
                    if d["scored"] and d["hits"] / d["scored"] >= localized_frac)
    summary = {
        "dataset": "IDRiD", "class": class_name, "n_images": len(records),
        "per_concept": per_concept,
        "heatmaps": {"scored": scored, "hits": hits,
                     "hit_rate": hits / scored if scored else 0.0,
                     "mean_mass": sum(d["mass"] for d in per_concept.values()) / scored if scored else 0.0},
        "concepts": {"scored": len(per_concept), "localized": localized},
    }
    if print_table:
        _print(summary)
    return summary


def _print(summary):
    """Per-concept + total localization table (mirrors runner.diagnosis_summary)."""
    h, c = summary["heatmaps"], summary["concepts"]
    print(f"[{summary['dataset']}] {summary['class']}: {summary['n_images']} diagnosed images, "
          f"{c['localized']}/{c['scored']} concepts localized, "
          f"{h['hits']}/{h['scored']} heatmaps hit ({h['hit_rate']:.0%}), "
          f"mean mass-in-mask {h['mean_mass']:.2f}")
    print(f"{'Concept':<28}{'Type':>5}{'Hit rate':>12}{'Mean mass':>12}")
    for concept, d in sorted(summary["per_concept"].items(),
                             key=lambda kv: kv[1]["hits"] / max(1, kv[1]["scored"]), reverse=True):
        rate = f"{d['hits']}/{d['scored']} ({d['hits'] / d['scored']:.0%})" if d["scored"] else "-"
        mass = f"{d['mass'] / d['scored']:.2f}" if d["scored"] else "-"
        print(f"{concept[:28]:<28}{d['type']:>5}{rate:>12}{mass:>12}")
