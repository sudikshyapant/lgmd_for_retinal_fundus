"""Drive the LGMD pipeline across the 40-class ImageNet benchmark.

The backbone and CLIP are class-independent, so they are loaded once and reused. For
each class we select it (config.select_class sets class_name / synset / index), then run
data -> activations -> concepts -> S -> W -> inference -> metrics + baselines. Every heavy
artifact is cached per class via cache_name() (the "data"/"con" groups key on the active
class). Qualitative overlays are produced only for FIGURE_CLASSES, mirroring the paper's
figures (cat, bald eagle, library, electric guitar).

Usage (from the notebook, after the sys.path setup cell):
    import runner
    results, agg = runner.run_all()                 # all 40 classes
    results, agg = runner.run_all(["tabby cat"])    # a subset
"""

import os

import utils
import data_utils
import model_utils
import concepts as concept_mod
import clip_maps
import lgmd
import baselines
import metrics
import viz
from config import (CONFIG, FIGURE_CLASSES, cache_name, select_class,
                    fundus_class_names, _canon)

DEVICE = model_utils.DEVICE


def run_class(name, model, transform, clip, make_figures=False):
    """Run the full LGMD pipeline for one class and return its metrics + concepts.

    `model`, `transform`, `clip` are the shared (class-independent) backbone and CLIP.
    Set `make_figures` to also save concept-overlay images for this class.
    """
    select_class(name)                                  # sets class_name / index
    CDIR, RDIR, bb = CONFIG["cache_dir"], CONFIG["results_dir"], CONFIG["backbone"]
    cls, label, seed = CONFIG["class_name"], CONFIG["class_index"], CONFIG["seed"]

    # 1. data — fit the concept basis on train-split images, evaluate on val-split images
    train_imgs = data_utils.load_class_images(cls, CONFIG["n_train"], "train_dir", seed)
    val_imgs = data_utils.load_class_images(cls, CONFIG["n_val"], "val_dir", seed)

    # 2. encoder activations (cache key: data + model)
    Z_train = utils.cached(os.path.join(CDIR, cache_name("Z_train", ".pt", "data", "model")),
        lambda: model_utils.extract_activations(model, transform, train_imgs, desc=f"{name} train act"))
    Z_val = utils.cached(os.path.join(CDIR, cache_name("Z_val", ".pt", "data", "model")),
        lambda: model_utils.extract_activations(model, transform, val_imgs, desc=f"{name} val act"))
    A_train = lgmd.unfold(Z_train)

    # 3. concepts — shared bank, or per-class two-stage filter (cache key: con)
    concept_list = concept_mod.get_concepts(clip, images=train_imgs)

    # 4. localized CLIP similarity maps S (cache key: data + con + clip)
    S_train = utils.cached(os.path.join(CDIR, cache_name("S_train", ".pt", "data", "con", "clip")),
        lambda: clip_maps.build_S(train_imgs, concept_list, clip))

    # 5. fit semantic concept basis W via PGD (cache key: data + model + con + clip + pgd)
    W = utils.cached(os.path.join(CDIR, cache_name("W", ".pt", "data", "model", "con", "clip", "pgd")),
        lambda: lgmd.fit_basis(A_train, S_train))

    # 6. inference on correctly-classified val samples (Sec 4)
    orig_logits_full = model_utils.logits_from_Z(model, Z_val)
    keep = orig_logits_full.argmax(-1) == label
    n_diag_total = int(keep.numel())                    # val images seen by the backbone
    n_diag_correct = int(keep.sum())                    # correctly diagnosed -> get heatmaps
    Z_val = Z_val[keep]
    val_imgs = [im for im, k in zip(val_imgs, keep.tolist()) if k]
    A_val = lgmd.unfold(Z_val)
    orig_logits = orig_logits_full[keep]
    S_hat = lgmd.infer(A_val, W)
    A_hat = lgmd.reconstruct(S_hat, W, Z_val.shape)

    def _lgmd_metrics():
        recon_logits = model_utils.logits_from_Z(model, A_hat)
        return {
            **metrics.predictive_preservation(orig_logits, recon_logits, label),
            "kl": metrics.kl_logits(orig_logits, recon_logits),
            "recon_err": metrics.recon_error(A_val, lgmd.unfold(A_hat)),
        }

    lgmd_metrics = utils.cached_json(
        os.path.join(RDIR, cache_name("lgmd_metrics", ".json",
                                      "data", "model", "con", "clip", "pgd", "infer")),
        _lgmd_metrics)

    # 7. baseline comparison: ICE / CRAFT / FACE vs LGMD (Acc + C-Ins)
    R = len(concept_list)                               # basis columns = number of concepts
    face_head = lambda a: model_utils.classify_pooled(model, a.to(DEVICE))
    head_fn = lambda Z: model_utils.logits_from_Z(model, Z)

    # Fit the baseline bases once and cache them (reused for metrics AND figures).
    def _bases():
        return {
            "ICE":   baselines.fit_ice(A_train, R),
            "CRAFT": baselines.fit_craft(A_train, R),
            "FACE":  baselines.fit_face(A_train, R, face_head, Z_train.shape),
            "LGMD":  W,
        }
    bases = utils.cached(
        os.path.join(CDIR, cache_name("bases", ".pt", "data", "model", "con", "clip", "pgd", "base")),
        _bases)

    def _comparison():
        out = {}
        for bn, Wb in bases.items():
            Sb = lgmd.infer(A_val, Wb)
            Ab = lgmd.reconstruct(Sb, Wb, Z_val.shape)
            lg = model_utils.logits_from_Z(model, Ab)
            cur = metrics.faithfulness_curves(Sb, Wb, Z_val.shape, head_fn, label)
            pp = metrics.predictive_preservation(orig_logits, lg, label)
            out[bn] = {
                "Acc": pp["recon_acc"],
                "C-Ins": metrics.insertion_auc(cur["insertion"]),
                "agreement": pp["agreement"],
                "kl": metrics.kl_logits(orig_logits, lg),
                "recon_err": metrics.recon_error(A_val, lgmd.unfold(Ab)),
            }
        return out

    comparison = utils.cached_json(
        os.path.join(RDIR, cache_name("comparison", ".json",
                                      "data", "model", "con", "clip", "pgd", "infer", "base", "cins")),
        _comparison)

    # counts behind the heatmaps: images the backbone diagnosed correctly (every overlay is
    # drawn on one of these), and how many of those keep the diagnosis after concept recon.
    n_recon_correct = round(lgmd_metrics["recon_acc"] * n_diag_correct)
    result = {
        "class": cls, "index": label,
        "concepts": concept_list, "lgmd": lgmd_metrics, "comparison": comparison,
        "diagnosed": {
            "correct": n_diag_correct, "total": n_diag_total,
            "acc": n_diag_correct / n_diag_total if n_diag_total else 0.0,
        },
        "concept_preserved": {
            "correct": n_recon_correct, "of": n_diag_correct,
            "acc": lgmd_metrics["recon_acc"],
        },
    }

    # 8. collect per-class artifacts the multi-class figures need (LGMD coeffs + every
    #    method's inferred coefficient maps on the val images), only for figure classes.
    if make_figures:
        result["figure"] = {
            "images": val_imgs,
            "concepts": concept_list,
            "S_hat": S_hat,
            "method_maps": {bn: lgmd.infer(A_val, Wb) for bn, Wb in bases.items()},
        }
    return result


def aggregate(results):
    """Mean Acc / C-Ins per method across the run's classes."""
    methods = ["ICE", "CRAFT", "FACE", "LGMD"]
    agg = {m: {} for m in methods}
    for m in methods:
        for k in ("Acc", "C-Ins"):
            vals = [r["comparison"][m][k] for r in results.values() if m in r["comparison"]]
            agg[m][k] = sum(vals) / len(vals) if vals else None
    return agg


def diagnosis_summary(results, print_table=True):
    """Per-class + total counts: images diagnosed correctly, and of those how many keep the
    diagnosis after concept reconstruction. Returns {class: {...}, "TOTAL": {...}}."""
    rows, dt, dc, ct, cc = {}, 0, 0, 0, 0
    for cls, r in results.items():
        dg, cp = r.get("diagnosed"), r.get("concept_preserved")
        if not dg:
            continue
        rows[cls] = {"diag_correct": dg["correct"], "diag_total": dg["total"],
                     "diag_acc": dg["acc"], "cp_correct": cp["correct"],
                     "cp_of": cp["of"], "cp_acc": cp["acc"]}
        dt += dg["total"]; dc += dg["correct"]; ct += cp["of"]; cc += cp["correct"]
    rows["TOTAL"] = {"diag_correct": dc, "diag_total": dt,
                     "diag_acc": dc / dt if dt else 0.0, "cp_correct": cc,
                     "cp_of": ct, "cp_acc": cc / ct if ct else 0.0}

    if print_table:
        print(f"{'Class':<16}{'Diagnosed':>16}{'Concept-preserved':>20}")
        for cls, v in rows.items():
            diag = f"{v['diag_correct']}/{v['diag_total']} ({v['diag_acc']:.0%})"
            cp = f"{v['cp_correct']}/{v['cp_of']} ({v['cp_acc']:.0%})"
            print(f"{cls:<16}{diag:>16}{cp:>20}")
    return rows


def metric_table(results, metric="Acc", methods=("LGMD", "FACE", "ICE", "CRAFT")):
    """Per-class metric table for one metric, shaped for viz.render_metric_table.

    Returns {backbone: {class: {method: value}, ..., "Average": {method: mean}}}. Rows are
    the classes that produced a `comparison` block (plus an Average row); columns are the
    baseline methods vs. LGMD. Feed one call per metric ("Acc", "C-Ins") to get a table
    that reports every retinal-disease class instead of only the aggregate mean.
    """
    bb = CONFIG["backbone"]
    per_class = {}
    for cls, r in results.items():
        comp = r.get("comparison", {})
        row = {m: comp[m][metric] for m in methods if m in comp and metric in comp[m]}
        if row:
            per_class[cls] = row
    if per_class:                                       # Average row across the classes
        avg = {}
        for m in methods:
            vals = [row[m] for row in per_class.values() if m in row]
            if vals:
                avg[m] = sum(vals) / len(vals)
        per_class["Average"] = avg
    return {bb: per_class}


def run_all(classes=None, figure_classes=None):
    """Run the pipeline over `classes` (default: every dataset class).

    Returns (results, aggregate, failures): `failures` maps each skipped class name to its
    error string, printed as a numbered list so the recovery cell can reference a class by
    its number.

    Loads the backbone + CLIP once and reuses them. Overlays are saved only for classes in
    `figure_classes` (default: FIGURE_CLASSES). A class that errors out (e.g. too few val
    images, or — in per_class mode — fewer than r concepts surviving filtering) is printed
    and skipped rather than aborting the whole run; every other class still completes and is
    cached, so a rerun resumes for free.
    """
    classes = classes if classes is not None else fundus_class_names()
    # Figure overlays: explicit list, else FIGURE_CLASSES, else auto-pick the first class.
    fig_src = (figure_classes if figure_classes is not None
               else (FIGURE_CLASSES if FIGURE_CLASSES is not None else classes[:1]))
    figset = {_canon(x) for x in fig_src}   # canonical -> tolerant of case/spacing

    model, transform = model_utils.load_backbone()
    clip = clip_maps.CLIP()

    results, failures = {}, {}
    for i, name in enumerate(classes, 1):
        try:
            res = run_class(name, model, transform, clip, make_figures=_canon(name) in figset)
            results[name] = res
            dg, cp = res["diagnosed"], res["concept_preserved"]
            print(f"[{i}/{len(classes)}] {name}: diagnosed {dg['correct']}/{dg['total']} "
                  f"({dg['acc']:.0%}), concept-preserved {cp['correct']}/{cp['of']}; "
                  f"LGMD={res['lgmd']}")
        except Exception as e:
            failures[name] = f"{type(e).__name__}: {e}"
            print(f"[{i}/{len(classes)}] {name}: SKIPPED — {failures[name]}")
    if failures:
        print(f"\n[run_all] completed {len(results)}/{len(classes)} classes; "
              f"skipped {len(failures)} — recover these in the next cell by number:")
        for j, (fname, err) in enumerate(failures.items()):
            print(f"  [{j}] {fname} — {err}")
    return results, aggregate(results), failures


def make_figures(classes=None):
    """Render the qualitative concept figures across the showcase classes.

    Runs the pipeline for `classes` (default FIGURE_CLASSES, e.g. cataract), then renders
    three multi-class figures:
      - fig1_concept_heatmaps_grid.png : input image + top named-concept heatmaps per class
      - fig2_baseline_grid.png         : Data Samples / ICE / CRAFT / FACE (no LGMD)
      - fig2_lgmd_panel.png            : LGMD named concepts by top-activating patches
    Classes that error out are printed and skipped. Returns the per-class figure payload.
    """
    classes = list(classes) if classes is not None else (
        list(FIGURE_CLASSES) if FIGURE_CLASSES is not None else fundus_class_names()[:1])
    model, transform = model_utils.load_backbone()
    clip = clip_maps.CLIP()

    per_class = {}
    for i, name in enumerate(classes, 1):
        print(f"[fig {i}/{len(classes)}] {name}")
        try:
            per_class[name] = run_class(name, model, transform, clip, make_figures=True)["figure"]
        except Exception as e:
            print(f"[fig {i}/{len(classes)}] {name}: SKIPPED — {type(e).__name__}: {e}")
    if not per_class:
        print("[make_figures] no classes succeeded; nothing to plot.")
        return per_class

    done = list(per_class)
    viz.plot_concept_heatmaps_grid(per_class, done)
    viz.plot_baseline_grid(per_class, done)
    viz.plot_lgmd_panel(per_class, done)
    return per_class
