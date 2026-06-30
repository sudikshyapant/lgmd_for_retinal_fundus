"""Local retinal-fundus image loading (ImageFolder layout) + CLIP geometric preprocessing.

The dataset lives on disk as <data_root>/<split>/<class>/*.jpg. Concept discovery is
per-class and needs only a small sample, so we list a class folder and read N images —
no streaming, no download. (Backbone *training* draws a larger n_per_class subset; see
train_backbone.py, which uses torchvision ImageFolder over the whole train split.)
"""

import os
import random

from PIL import Image

from config import CONFIG

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".ppm", ".webp")


def class_dir(class_name, split="train_dir"):
    """Absolute path to <data_root>/<CONFIG[split]>/<class_name>."""
    return os.path.join(CONFIG["data_root"], CONFIG[split], class_name)


def list_class_files(class_name, split="train_dir"):
    """Sorted list of image file paths for one class in one split."""
    d = class_dir(class_name, split)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"Class folder not found: {d}")
    files = sorted(f for f in os.listdir(d) if f.lower().endswith(_IMG_EXTS))
    return [os.path.join(d, f) for f in files]


def load_class_images(class_name, n_total, split="train_dir", seed=None):
    """Load up to `n_total` RGB images of `class_name` from `split`.

    A fixed `seed` shuffles deterministically before taking the first n_total (so the
    sample is reproducible but not just the alphabetical head). Returns fewer images
    than requested — with a warning — if the folder has fewer than n_total.
    """
    files = list_class_files(class_name, split)
    if seed is not None:
        random.Random(seed).shuffle(files)
    if len(files) < n_total:
        print(f"[warn] {class_name}/{CONFIG[split]}: requested {n_total} images, "
              f"only {len(files)} available — using all of them.")
    files = files[:n_total]
    return [Image.open(p).convert("RGB") for p in files]


def clip_preprocess(img, size=224):
    """CLIP's deterministic geometric preprocessing: resize shortest side + center crop.

    Returns a `size`x`size` RGB PIL image, so we can draw red circles on it before
    handing it to the CLIP normalizer (resize/crop then become no-ops).
    """
    w, h = img.size
    scale = size / min(w, h)
    img = img.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    return img.crop((left, top, left + size, top + size))
