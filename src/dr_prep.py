"""Turn the IDRiD "Disease Grading" download into a 5-grade train/val/test ImageFolder.

IDRiD (Indian Diabetic Retinopathy Image Dataset) ships its grading sub-dataset as a flat
image folder (Training Set / Testing Set) plus two label CSVs (Image name -> Retinopathy
grade 0..4 + a DME risk we ignore) — not the `<split>/<grade>/*.jpg` ImageFolder layout the
rest of this pipeline (train_backbone, data_utils, torchvision.ImageFolder) assumes. This
module bridges the two.

Split policy (IDRiD has no official validation set): IDRiD's Testing Set becomes `test/`,
and its Training Set is stratified-split (per grade) into `train/` + `val/`.

Grade -> folder name (chosen so config.resolve_vocab_key maps each to its concept_vocab.json
key; numeric prefix keeps ImageFolder's alphabetical order == grade order == classifier index):
    0 -> 0_no_dr     1 -> 1_mild_npdr     2 -> 2_moderate_npdr
    3 -> 3_severe_npdr                     4 -> 4_proliferative_dr

Usage (from the notebook's dataset cell):
    import dr_prep
    dr_prep.prepare()                       # kagglehub download -> sets CONFIG["data_root"]
    dr_prep.prepare(raw_root=<extracted IDRiD dir>)   # or point at a local/Drive copy

Idempotent: a completed build drops a `.prepared.json` marker; re-running is a no-op unless
the split/seed changed (or force=True).
"""

import csv
import json
import os
import random
import shutil

from config import CONFIG

# IDRiD Disease Grading on Kaggle (includes the 'Retinopathy grade' label CSVs + original
# images). Override via prepare(slug=...) if your copy lives elsewhere. Note lesion_eval.py
# uses its own (segmentation-mask) slug; the two downloads are independent.
IDRID_SLUG = "mariaherrerot/idrid-dataset"

# Retinopathy grade (0..4) -> ImageFolder folder name.
GRADE_TO_FOLDER = {
    0: "0_no_dr",
    1: "1_mild_npdr",
    2: "2_moderate_npdr",
    3: "3_severe_npdr",
    4: "4_proliferative_dr",
}

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


# --------------------------------------------------------------------------------------
# Locating the raw download
# --------------------------------------------------------------------------------------
def _download(slug):
    """Fetch the IDRiD download dir via kagglehub (cached after the first call)."""
    import kagglehub
    path = kagglehub.dataset_download(slug)
    print(f"[dr_prep] kagglehub: {slug} -> {path}")
    return path


def _index_images(base):
    """Map both filename and extension-less stem -> absolute path for images under `base`.

    IDRiD CSVs reference images by stem (e.g. 'IDRiD_001'); files are 'IDRiD_001.jpg', so we
    index by stem too. Grading-folder images ('grading' / 'original images' in the path) are
    preferred to avoid collisions with any segmentation tree, but if the download uses a
    different layout we still fall back to any matching image so lookups don't fail.
    """
    grading, other = {}, {}
    for root, _dirs, files in os.walk(base):
        is_grading = "grading" in root.lower() or "original images" in root.lower()
        target = grading if is_grading else other
        for f in files:
            if f.lower().endswith(_IMG_EXTS):
                path = os.path.join(root, f)
                target.setdefault(f, path)
                target.setdefault(os.path.splitext(f)[0], path)
    merged = dict(other)          # non-grading images as fallback...
    merged.update(grading)        # ...overridden by grading-folder images where present
    return merged


def _find_grading_csvs(base):
    """Locate IDRiD's grading label CSVs by content (header carries 'Retinopathy grade').

    Returns {"train": path, "test": path}; the two are told apart by 'train'/'test' in the
    file path. Raises if the grading CSVs aren't in this download.
    """
    found = {}
    for root, _dirs, files in os.walk(base):
        for f in files:
            if not f.lower().endswith(".csv"):
                continue
            path = os.path.join(root, f)
            if not _has_grade_header(path):
                continue
            low = path.lower()
            if "test" in low:
                found.setdefault("test", path)
            elif "train" in low:
                found.setdefault("train", path)
    if "train" not in found or "test" not in found:
        raise FileNotFoundError(
            "Could not find IDRiD 'Disease Grading' label CSVs (with a 'Retinopathy grade' "
            f"column, split into train/test) under {base}. Found: {found or 'none'}. Make "
            "sure the download includes the 'B. Disease Grading' part, or pass "
            "prepare(slug=<a slug that contains it>) / prepare(raw_root=<local IDRiD dir>)."
        )
    print(f"[dr_prep] grading labels: train={os.path.basename(found['train'])!r} "
          f"test={os.path.basename(found['test'])!r}")
    return found


def _has_grade_header(csv_path):
    """True if the CSV's header contains a 'Retinopathy grade' column."""
    try:
        with open(csv_path, newline="", encoding="utf-8-sig") as fh:
            header = next(csv.reader(fh), [])
    except (OSError, UnicodeDecodeError):
        return False
    return any("retinopathy grade" in (c or "").strip().lower() for c in header)


# --------------------------------------------------------------------------------------
# Labeling
# --------------------------------------------------------------------------------------
def _read_grades(csv_path):
    """Yield (image_stem, grade:int) from an IDRiD grading CSV, skipping blank/bad rows."""
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        cols = {(c or "").strip().lower(): c for c in (reader.fieldnames or [])}
        f_name = cols.get("image name")
        f_grade = cols.get("retinopathy grade")
        if not (f_name and f_grade):
            raise ValueError(
                f"{os.path.basename(csv_path)} lacks 'Image name'/'Retinopathy grade' "
                f"columns. Columns: {reader.fieldnames}"
            )
        for row in reader:
            name = (row.get(f_name) or "").strip()
            raw = (row.get(f_grade) or "").strip()
            if not name or raw == "":
                continue
            try:
                grade = int(float(raw))
            except ValueError:
                continue
            if grade in GRADE_TO_FOLDER:
                yield os.path.splitext(name)[0], grade


def _resolve(stem, images):
    """Absolute image path for a CSV stem, trying stem then stem+ext then basename."""
    return images.get(stem) or images.get(os.path.basename(stem))


# --------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------
def _place(paths, split_name, folder, out_root, link):
    """Copy/hardlink `paths` into <out_root>/<split_name>/<folder>/."""
    dest_dir = os.path.join(out_root, split_name, folder)
    os.makedirs(dest_dir, exist_ok=True)
    for src in paths:
        dst = os.path.join(dest_dir, os.path.basename(src))
        if link:
            try:
                os.link(src, dst)
                continue
            except OSError:
                pass
        shutil.copy2(src, dst)


def prepare(raw_root=None, slug=None, out_root=None, val_frac=0.2,
            seed=None, force=False, link=True):
    """Build the 5-grade IDRiD ImageFolder and point CONFIG['data_root'] at it.

    Args:
        raw_root: extracted IDRiD dir (skips the kagglehub download if given).
        slug:     Kaggle dataset slug (default IDRID_SLUG); used only when raw_root is None.
        out_root: where to write <split>/<grade>/*.jpg (default CONFIG['data_root']).
        val_frac: fraction of IDRiD's Training Set held out (per grade) as val/.
        seed:     shuffle seed for the train/val carve (default CONFIG['seed']).
        force:    rebuild even if a matching .prepared.json marker exists.
        link:     hardlink images instead of copying (saves disk; falls back to copy).

    Returns the out_root path (also stored in CONFIG['data_root']).
    """
    out_root = out_root or CONFIG["data_root"]
    seed = CONFIG["seed"] if seed is None else seed
    marker = os.path.join(out_root, ".prepared.json")
    want = {"source": "idrid_grading", "folders": sorted(GRADE_TO_FOLDER.values()),
            "val_frac": val_frac, "seed": seed}

    if not force and os.path.exists(marker):
        with open(marker) as fh:
            have = json.load(fh)
        if {k: have.get(k) for k in want} == want:
            CONFIG["data_root"] = out_root
            print(f"[dr_prep] reusing existing build at {out_root} "
                  f"({have.get('total', '?')} images).")
            return out_root

    raw_root = raw_root or _download(slug or IDRID_SLUG)
    csvs = _find_grading_csvs(raw_root)
    images = _index_images(raw_root)
    print(f"[dr_prep] indexed {len(set(images.values()))} grading images under {raw_root}")

    # IDRiD Training Set -> stratified train/val; IDRiD Testing Set -> test.
    train_by_grade = {g: [] for g in GRADE_TO_FOLDER}
    test_by_grade = {g: [] for g in GRADE_TO_FOLDER}
    missing = 0
    for split_csv, bucket in (("train", train_by_grade), ("test", test_by_grade)):
        for stem, grade in _read_grades(csvs[split_csv]):
            path = _resolve(stem, images)
            if path is None:
                missing += 1
                continue
            bucket[grade].append(path)
    if missing:
        print(f"[dr_prep] {missing} labeled rows had no matching image file (skipped).")

    # Fresh build: wipe any stale tree so removed images don't linger.
    if os.path.isdir(out_root):
        shutil.rmtree(out_root)

    rng = random.Random(seed)
    counts = {}
    for grade, folder in GRADE_TO_FOLDER.items():
        tr_all = list(train_by_grade[grade])
        rng.shuffle(tr_all)
        n_val = int(round(val_frac * len(tr_all)))
        va, tr = tr_all[:n_val], tr_all[n_val:]
        te = test_by_grade[grade]
        _place(tr, "train", folder, out_root, link)
        _place(va, "val", folder, out_root, link)
        _place(te, "test", folder, out_root, link)
        counts[folder] = {"train": len(tr), "val": len(va), "test": len(te),
                          "total": len(tr) + len(va) + len(te)}

    total = sum(c["total"] for c in counts.values())
    with open(marker, "w") as fh:
        json.dump({**want, "counts": counts, "total": total}, fh, indent=2)

    print(f"[dr_prep] built {out_root}: {total} images across {len(counts)} grades")
    for folder in sorted(counts):
        c = counts[folder]
        print(f"    {folder:20s} train={c['train']:4d} val={c['val']:3d} "
              f"test={c['test']:3d} (total {c['total']})")
    if total == 0:
        raise RuntimeError(
            "No images were placed — grading CSVs parsed but no image files matched. Check "
            "that raw_root points at an IDRiD download containing the grading Original Images."
        )
    CONFIG["data_root"] = out_root
    return out_root


if __name__ == "__main__":
    prepare()
