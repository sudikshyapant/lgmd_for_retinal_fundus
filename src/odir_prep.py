"""Turn the raw ODIR-5K download into a 5-class train/val/test ImageFolder.

ODIR-5K (Ocular Disease Intelligent Recognition) ships as a flat pile of fundus images
plus a per-patient label table — not the `<split>/<class>/*.jpg` ImageFolder layout the
rest of this pipeline (train_backbone, data_utils, torchvision.ImageFolder) assumes. This
module bridges the two: it reads the label table, derives a *single* per-eye label from
that eye's diagnostic keywords, keeps only images that are unambiguously one of the five
target classes, and copies them into a fresh ImageFolder tree with a stratified split.

Five classes (ODIR codes -> folder names, chosen so config.resolve_vocab_key maps each to
its concept_vocab.json key):
    N -> Normal     (normal_fundus)      D -> Diabetes   (diabetic_retinopathy)
    G -> Glaucoma   (glaucoma)           C -> Cataract   (cataract)
    A -> AMD        (amd)

Usage (from the notebook's dataset cell, after the kagglehub download):
    import odir_prep
    odir_prep.prepare(raw_root=<kagglehub download dir>)   # -> sets CONFIG["data_root"]

Idempotent: a completed build drops a `.prepared.json` marker; re-running is a no-op
unless the target/split/seed changed (or force=True).
"""

import csv
import json
import os
import random
import shutil

from config import CONFIG

# ODIR code -> ImageFolder class-folder name (see resolve_vocab_key / CONCEPT_ALIASES).
CODE_TO_FOLDER = {
    "N": "Normal",
    "D": "Diabetes",
    "G": "Glaucoma",
    "C": "Cataract",
    "A": "AMD",
}

# Per-eye diagnostic-keyword -> ODIR code. ODIR's keyword strings are per eye (unlike the
# patient-level N..O one-hots, which conflate both eyes), so they give the cleanest single
# label. Substrings are matched case-insensitively; an eye is kept only if its keywords
# resolve to exactly one of these five codes (artifacts like "lens dust" / "low image
# quality" are ignored, genuinely mixed diagnoses are dropped — see _label_from_keywords).
_DISEASE_KEYWORDS = {
    "D": ["diabetic retinopathy", "proliferative retinopathy"],
    "G": ["glaucoma"],
    "C": ["cataract"],
    "A": ["age-related macular degeneration", "age related macular degeneration",
          "macular degeneration"],
}
_NORMAL_KEYWORD = "normal fundus"

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


# --------------------------------------------------------------------------------------
# Locating the raw download
# --------------------------------------------------------------------------------------
def find_raw_root(start=None):
    """Find the ODIR-5K root: the nearest directory (at/under `start`) holding the CSV.

    Tries the common Kaggle mount first, then walks `start`. Returns the directory that
    contains the label CSV (full_df.csv or similar); raises with guidance if none found.
    """
    candidates = []
    if start:
        candidates.append(start)
    candidates += [
        "/kaggle/input/ocular-disease-recognition-odir5k",
        "/kaggle/input/odir5k",
    ]
    for base in candidates:
        if base and os.path.isdir(base):
            hit = _find_label_csv(base)
            if hit:
                return os.path.dirname(hit)
    raise FileNotFoundError(
        "Could not locate the ODIR-5K label CSV (e.g. full_df.csv). Pass "
        "odir_prep.prepare(raw_root=<download dir>) with the folder returned by "
        "kagglehub.dataset_download('andrewmvd/ocular-disease-recognition-odir5k') "
        "(or your ODIR-5K mount)."
    )


def _find_label_csv(base):
    """Deepest-first search for a plausible ODIR label CSV under `base`."""
    preferred = None
    for root, _dirs, files in os.walk(base):
        for f in files:
            if not f.lower().endswith(".csv"):
                continue
            if f.lower() == "full_df.csv":
                return os.path.join(root, f)          # canonical community export
            if preferred is None and ("odir" in f.lower() or "data" in f.lower()):
                preferred = os.path.join(root, f)
    return preferred


def _index_images(base):
    """Map image basename -> absolute path for every image under `base` (first wins)."""
    index = {}
    for root, _dirs, files in os.walk(base):
        for f in files:
            if f.lower().endswith(_IMG_EXTS):
                index.setdefault(f, os.path.join(root, f))
    return index


# --------------------------------------------------------------------------------------
# Labeling
# --------------------------------------------------------------------------------------
def _label_from_keywords(keywords):
    """Map one eye's diagnostic-keyword string to a single ODIR code, or None if ambiguous.

    Keep the eye only when its keywords name exactly one disease among {D,G,C,A}, or name
    "normal fundus" with no disease. Anything mixed / unmatched / outside the five (e.g.
    hypertensive, myopia) returns None so the class folders stay clean and single-label.
    """
    kw = (keywords or "").lower()
    codes = {code for code, subs in _DISEASE_KEYWORDS.items()
             if any(s in kw for s in subs)}
    if len(codes) == 1:
        return next(iter(codes))
    if not codes and _NORMAL_KEYWORD in kw:
        return "N"
    return None


def _labels_from_list(raw):
    """Map a per-image `labels` field (e.g. "['N']") to a single code, or None."""
    codes = [c for c in ("N", "D", "G", "C", "A") if f"'{c}'" in (raw or "")]
    return codes[0] if len(codes) == 1 else None


def _records(csv_path):
    """Yield (image_filename, code) pairs from the label CSV, dropping ambiguous/other.

    Handles both common schemas: per-patient rows (Left/Right-Fundus + per-eye keyword
    columns) and the per-image community export (filename + labels).
    """
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        cols = {c.lower(): c for c in (reader.fieldnames or [])}

        def col(*names):
            for n in names:
                if n.lower() in cols:
                    return cols[n.lower()]
            return None

        f_file = col("filename")
        f_labels = col("labels")
        f_lfun, f_rfun = col("Left-Fundus"), col("Right-Fundus")
        f_lkw = col("Left-Diagnostic Keywords", "Left-Diagnostic-Keywords")
        f_rkw = col("Right-Diagnostic Keywords", "Right-Diagnostic-Keywords")

        per_image = f_file and f_labels
        per_patient = f_lfun and f_rfun and f_lkw and f_rkw
        if not (per_image or per_patient):
            raise ValueError(
                f"{os.path.basename(csv_path)} has neither a per-image (filename+labels) "
                f"nor a per-patient (Left/Right-Fundus + keyword) schema. Columns: "
                f"{reader.fieldnames}"
            )

        for row in reader:
            if per_image:
                code = _labels_from_list(row.get(f_labels))
                if code:
                    yield row[f_file], code
            else:
                for f_fun, f_kw in ((f_lfun, f_lkw), (f_rfun, f_rkw)):
                    code = _label_from_keywords(row.get(f_kw))
                    if code and row.get(f_fun):
                        yield row[f_fun], code


# --------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------
def _split(items, fracs, seed):
    """Deterministically split a list into (train, val, test) by `fracs`."""
    items = list(items)
    random.Random(seed).shuffle(items)
    n = len(items)
    n_tr = int(round(fracs[0] * n))
    n_va = int(round(fracs[1] * n))
    return items[:n_tr], items[n_tr:n_tr + n_va], items[n_tr + n_va:]


def prepare(raw_root=None, out_root=None, splits=(0.70, 0.15, 0.15),
            seed=None, force=False, link=False):
    """Build the 5-class ODIR-5K ImageFolder and point CONFIG['data_root'] at it.

    Args:
        raw_root: folder holding the ODIR-5K download (auto-detected if None).
        out_root: where to write <split>/<class>/*.jpg (default CONFIG['data_root']).
        splits:   (train, val, test) fractions, per class (stratified).
        seed:     shuffle seed (default CONFIG['seed']).
        force:    rebuild even if a matching .prepared.json marker exists.
        link:     hardlink images instead of copying (saves disk; falls back to copy).

    Returns the out_root path (also stored in CONFIG['data_root']).
    """
    out_root = out_root or CONFIG["data_root"]
    seed = CONFIG["seed"] if seed is None else seed
    marker = os.path.join(out_root, ".prepared.json")
    want = {"classes": sorted(CODE_TO_FOLDER.values()), "splits": list(splits), "seed": seed}

    if not force and os.path.exists(marker):
        with open(marker) as fh:
            have = json.load(fh)
        if {k: have.get(k) for k in want} == want:
            CONFIG["data_root"] = out_root
            print(f"[odir_prep] reusing existing build at {out_root} "
                  f"({have.get('total', '?')} images).")
            return out_root

    raw_root = raw_root or find_raw_root()
    csv_path = _find_label_csv(raw_root)
    if not csv_path:
        raise FileNotFoundError(f"No ODIR label CSV under {raw_root}.")
    images = _index_images(raw_root)
    print(f"[odir_prep] label table: {csv_path}")
    print(f"[odir_prep] indexed {len(images)} images under {raw_root}")

    # Group image paths by class, skipping any label row whose image file is missing.
    by_code, missing = {c: [] for c in CODE_TO_FOLDER}, 0
    for fname, code in _records(csv_path):
        path = images.get(fname) or images.get(os.path.basename(fname))
        if path is None:
            missing += 1
            continue
        by_code[code].append(path)
    if missing:
        print(f"[odir_prep] {missing} labeled rows had no matching image file (skipped).")

    # Fresh build: wipe any stale tree so removed images don't linger.
    if os.path.isdir(out_root):
        shutil.rmtree(out_root)
    counts = {}
    for code, paths in by_code.items():
        folder = CODE_TO_FOLDER[code]
        tr, va, te = _split(paths, splits, seed)
        for split_name, split_paths in (("train", tr), ("val", va), ("test", te)):
            dest_dir = os.path.join(out_root, split_name, folder)
            os.makedirs(dest_dir, exist_ok=True)
            for src in split_paths:
                dst = os.path.join(dest_dir, os.path.basename(src))
                if link:
                    try:
                        os.link(src, dst)
                        continue
                    except OSError:
                        pass
                shutil.copy2(src, dst)
        counts[folder] = {"train": len(tr), "val": len(va), "test": len(te),
                          "total": len(paths)}

    total = sum(c["total"] for c in counts.values())
    with open(marker, "w") as fh:
        json.dump({**want, "counts": counts, "total": total}, fh, indent=2)

    print(f"[odir_prep] built {out_root}: {total} images across "
          f"{len(counts)} classes")
    for folder in sorted(counts):
        c = counts[folder]
        print(f"    {folder:10s} train={c['train']:4d} val={c['val']:3d} "
              f"test={c['test']:3d} (total {c['total']})")
    if total == 0:
        raise RuntimeError(
            "No images were placed — the CSV parsed but nothing matched the five classes "
            "or the image filenames. Check that raw_root points at the ODIR-5K root."
        )
    CONFIG["data_root"] = out_root
    return out_root


if __name__ == "__main__":
    prepare()
