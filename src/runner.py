"""Drive the LGMD pipeline across the 5 DR-severity grades.

The backbone and VLM are class-independent, so they are loaded once and reused. For
each grade we select it (config.select_class sets class_name / index), then run
data -> activations -> concepts -> S -> W -> inference -> metrics + baselines. Every heavy
artifact is cached per grade via cache_name() (the "data"/"con" groups key on the active
grade). Qualitative overlays are produced only for FIGURE_CLASSES.

Usage (from the notebook, after the sys.path setup cell):
    import runner
    results, agg, failures = runner.run_all()                  # all grades
    results, agg, failures = runner.run_all(["3_severe_npdr"]) # a subset
"""

import os

import torch

import utils
import data_utils
import model_utils
import concepts as concept_mod
import flair_maps
import lgmd
import baselines
import metrics
import viz
from config import (CONFIG, FIGURE_CLASSES, cache_name, select_class,
                    fundus_class_names, _canon)

DEVICE = model_utils.DEVICE


def nonneg_offset(A):
    """Per-channel min of activations A (N, p) -> shift vector (p,).

    NMF needs a non-negative matrix, but RETFound's ViT tokens are signed. Subtracting this
    per-channel min moves the activations into the non-negative orthant for factorization;
    lgmd.reconstruct adds it back so the classifier head still reads original-space features
    (the shift is a constant offset, reproduced exactly, so predictions are unchanged).
    """
    return A.min(dim=0).values


def class_basis(name, model, transform, vlm):
    """Steps 1-5 of run_class: fit (or load) one class's concept basis.

    Returns (concepts, W, Z_train, offset) — the class concept list, its learned semantic
    basis W, the train-split activations (shape reused for baselines), and the non-negative
    shift applied before factorization. Cache keys are identical to run_class's, so run_class
    and the lesion-localization eval share the same artifacts.
    """
    select_class(name)                                  # sets class_name / index
    CDIR, seed, cls = CONFIG["cache_dir"], CONFIG["seed"], CONFIG["class_name"]

    # 1. data + 2. encoder activations (cache key: data + model)
    train_imgs = data_utils.load_class_images(cls, CONFIG["n_train"], "train_dir", seed)
    Z_train = utils.cached(os.path.join(CDIR, cache_name("Z_train", ".pt", "data", "model")),
        lambda: model_utils.extract_activations(model, transform, train_imgs, desc=f"{name} train act"))

    # 3. concepts (cache key: con) + 4. localized VLM similarity maps S (data + con + clip)
    concept_list = concept_mod.get_concepts(vlm, images=train_imgs)
    S_train = utils.cached(os.path.join(CDIR, cache_name("S_train", ".pt", "data", "con", "clip")),
        lambda: flair_maps.build_S(train_imgs, concept_list, vlm))

    # 5. shift to the non-negative orthant, then fit the concept basis W via PGD
    #    (cache key: data + model + con + clip + pgd). The offset is deterministic from
    #    Z_train (data + model), so it is recomputed rather than separately cached.
    A_train = lgmd.unfold(Z_train)
    offset = nonneg_offset(A_train)
    W = utils.cached(os.path.join(CDIR, cache_name("W", ".pt", "data", "model", "con", "clip", "pgd")),
        lambda: lgmd.fit_basis(A_train - offset, S_train))
    return concept_list, W, Z_train, offset


def run_class(name, model, transform, vlm, make_figures=False):
    """Run the full LGMD pipeline for one class and return its metrics + concepts.

    `model`, `transform`, `vlm` are the shared (class-independent) backbone and VLM.
    Set `make_figures` to also save concept-overlay images for this class.
    """
    concept_list, W, Z_train, offset = class_basis(name, model, transform, vlm)
    CDIR, RDIR, bb = CONFIG["cache_dir"], CONFIG["results_dir"], CONFIG["backbone"]
    cls, label, seed = CONFIG["class_name"], CONFIG["class_index"], CONFIG["seed"]
    A_train = lgmd.unfold(Z_train) - offset             # non-negative feature space (all methods)

    # val data + activations (cache key: data + model)
    val_imgs = data_utils.load_class_images(cls, CONFIG["n_val"], "val_dir", seed)
    Z_val = utils.cached(os.path.join(CDIR, cache_name("Z_val", ".pt", "data", "model")),
        lambda: model_utils.extract_activations(model, transform, val_imgs, desc=f"{name} val act"))

    # 6. inference on the evaluation samples (Sec 4). Normally we explain only the val images
    #    the backbone diagnoses correctly. But a scarce grade's weak head may get too few (or
    #    zero) right, which would leave an empty eval set and skip the grade. So when fewer
    #    than min_eval_images are correct we fall back to ALL val images and rely on
    #    prediction *agreement* (label-independent) rather than correct-diagnosis preservation.
    orig_logits_full = model_utils.logits_from_Z(model, Z_val)
    keep = orig_logits_full.argmax(-1) == label
    n_diag_total = int(keep.numel())                    # val images seen by the backbone
    n_diag_correct = int(keep.sum())                    # correctly diagnosed
    eval_on_all = n_diag_correct < CONFIG["min_eval_images"]
    eval_mask = torch.ones_like(keep) if eval_on_all else keep
    if int(eval_mask.sum()) == 0:
        raise RuntimeError(f"{cls}: no val images to evaluate (val split is empty).")
    if eval_on_all:
        print(f"[thin grade] {cls}: {n_diag_correct}/{n_diag_total} val images diagnosed "
              f"correctly (< {CONFIG['min_eval_images']}); evaluating on all "
              f"{int(eval_mask.sum())} val images (agreement-based).")
    Z_val = Z_val[eval_mask]
    val_imgs = [im for im, k in zip(val_imgs, eval_mask.tolist()) if k]
    n_eval = len(val_imgs)                               # images actually scored below
    A_val = lgmd.unfold(Z_val) - offset                 # shift into the non-negative space
    orig_logits = orig_logits_full[eval_mask]
    S_hat = lgmd.infer(A_val, W)
    A_hat = lgmd.reconstruct(S_hat, W, Z_val.shape, offset)   # offset added back -> original space

    def _lgmd_metrics():
        recon_logits = model_utils.logits_from_Z(model, A_hat)
        return {
            **metrics.predictive_preservation(orig_logits, recon_logits, label),
            "kl": metrics.kl_logits(orig_logits, recon_logits),
            "recon_err": metrics.recon_error(A_val, lgmd.unfold(A_hat) - offset),
        }

    lgmd_metrics = utils.cached_json(
        os.path.join(RDIR, cache_name("lgmd_metrics", ".json",
                                      "data", "model", "con", "clip", "pgd", "infer")),
        _lgmd_metrics)

    # 7. baseline comparison: ICE / CRAFT / FACE vs LGMD (Acc + C-Ins). All baselines fit in
    #    the same shifted non-negative space as LGMD. face_head adds the offset back so FACE's
    #    KL term is computed against the original classifier's logits.
    R = len(concept_list)                               # basis columns = number of concepts
    offset_dev = offset.to(DEVICE)
    face_head = lambda a: model_utils.classify_pooled(model, a.to(DEVICE) + offset_dev)
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
            Ab = lgmd.reconstruct(Sb, Wb, Z_val.shape, offset)   # original space
            lg = model_utils.logits_from_Z(model, Ab)
            cur = metrics.insertion_curve(Sb, Wb, Z_val.shape, head_fn, label, offset=offset)
            pp = metrics.predictive_preservation(orig_logits, lg, label)
            out[bn] = {
                "Acc": pp["recon_acc"],
                "C-Ins": metrics.insertion_auc(cur),
                "agreement": pp["agreement"],
                "kl": metrics.kl_logits(orig_logits, lg),
                "recon_err": metrics.recon_error(A_val, lgmd.unfold(Ab) - offset),
            }
        return out

    comparison = utils.cached_json(
        os.path.join(RDIR, cache_name("comparison", ".json",
                                      "data", "model", "con", "clip", "pgd", "infer", "base", "cins")),
        _comparison)

    # counts behind the heatmaps: the evaluation images (diagnosed-correct, or — for a thin
    # grade in the eval_on_all fallback — all val images) and how many keep the true-grade
    # prediction after concept reconstruction. n_eval >= 1 here, so recon_acc is never nan.
    recon_acc = lgmd_metrics["recon_acc"]
    n_recon_correct = round(recon_acc * n_eval)
    result = {
        "class": cls, "index": label,
        "concepts": concept_list, "lgmd": lgmd_metrics, "comparison": comparison,
        "eval_on_all": eval_on_all, "n_eval": n_eval,
        "diagnosed": {
            "correct": n_diag_correct, "total": n_diag_total,
            "acc": n_diag_correct / n_diag_total if n_diag_total else 0.0,
        },
        "concept_preserved": {
            "correct": n_recon_correct, "of": n_eval,
            "acc": recon_acc,
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


def _grounding_for_class(name, vlm):
    """FLAIR visual-grounding score per concept for one grade (uses the cached S_train).

    Reuses the exact S_train cache key as class_basis, so if the grade has already been run
    this just loads S; otherwise it builds it. Needs only the VLM (not the trained backbone).
    Returns (concept_list, {"peak": tensor, "mean": tensor}).
    """
    select_class(name)
    CDIR, seed, cls = CONFIG["cache_dir"], CONFIG["seed"], CONFIG["class_name"]
    train_imgs = data_utils.load_class_images(cls, CONFIG["n_train"], "train_dir", seed)
    concept_list = concept_mod.get_concepts(vlm, images=train_imgs)
    S_train = utils.cached(os.path.join(CDIR, cache_name("S_train", ".pt", "data", "con", "clip")),
        lambda: flair_maps.build_S(train_imgs, concept_list, vlm))
    scores = metrics.concept_grounding(S_train, len(train_imgs), CONFIG["grid"])
    return concept_list, scores


def grounding_table(classes=None, print_table=True, save=True, plot=True, score="peak"):
    """Per-concept FLAIR visual-grounding table AND figure for each grade.

    For every grade, scores how visually grounded each concept is via FLAIR's localized
    similarity maps S — peak (strongest localized cell, averaged over images) and mean
    (diffuse presence). No concept is dropped; this is a diagnostic run alongside the basis
    fit. Emits three artifacts: a printed per-grade table, a saved JSON, and a sorted
    bar/heatmap figure (viz). Returns {grade: {concept: {"peak": float, "mean": float}}}.
    """
    classes = classes if classes is not None else fundus_class_names()
    vlm = flair_maps.VLM()
    out = {}
    for name in classes:
        try:
            concepts, sc = _grounding_for_class(name, vlm)
        except Exception as e:
            print(f"[grounding] {name}: SKIPPED — {type(e).__name__}: {e}")
            continue
        cls = CONFIG["class_name"]
        rows = {c: {"peak": float(sc["peak"][k]), "mean": float(sc["mean"][k])}
                for k, c in enumerate(concepts)}
        out[cls] = rows
        if print_table:
            _print_grounding(cls, rows)
    if save and out:
        import json
        tag = CONFIG.get("run_tag", "") or "default"
        path = os.path.join(CONFIG["results_dir"], f"concept_grounding_{tag}.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[grounding] saved {path}")
    if plot and out:
        viz.plot_concept_grounding(out, score=score)
    return out


def _print_grounding(cls, rows):
    """Per-concept grounding table for one grade, sorted by peak (most-grounded first)."""
    print(f"\n[grounding] {cls} — FLAIR visual grounding per concept "
          f"(peak = strongest localized cell mean over images):")
    print(f"{'Concept':<34}{'Peak':>8}{'Mean':>8}")
    for c, d in sorted(rows.items(), key=lambda kv: kv[1]["peak"], reverse=True):
        print(f"{c[:34]:<34}{d['peak']:>8.3f}{d['mean']:>8.3f}")


def run_all(classes=None, figure_classes=None):
    """Run the pipeline over `classes` (default: every DR grade).

    Returns (results, aggregate, failures): `failures` maps each skipped class name to its
    error string, printed as a numbered list so the recovery cell can reference a class by
    its number.

    Loads the backbone + VLM once and reuses them. Overlays are saved only for classes in
    `figure_classes` (default: FIGURE_CLASSES). A grade that errors out (e.g. too few val
    images, or no concepts listed for it) is printed and skipped rather than aborting the
    whole run; every other grade still completes and is cached, so a rerun resumes for free.
    """
    classes = classes if classes is not None else fundus_class_names()
    # Figure overlays: explicit list, else FIGURE_CLASSES, else auto-pick the first class.
    fig_src = (figure_classes if figure_classes is not None
               else (FIGURE_CLASSES if FIGURE_CLASSES is not None else classes[:1]))
    figset = {_canon(x) for x in fig_src}   # canonical -> tolerant of case/spacing

    model, transform = model_utils.load_backbone()
    vlm = flair_maps.VLM()

    results, failures = {}, {}
    for i, name in enumerate(classes, 1):
        try:
            res = run_class(name, model, transform, vlm, make_figures=_canon(name) in figset)
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

    Runs the pipeline for `classes` (default FIGURE_CLASSES, e.g. moderate NPDR), then renders
    three multi-class figures:
      - fig1_concept_heatmaps_grid.png : input image + top named-concept heatmaps per class
      - fig2_baseline_grid.png         : Data Samples / ICE / CRAFT / FACE (no LGMD)
      - fig2_lgmd_panel.png            : LGMD named concepts by top-activating patches
    Classes that error out are printed and skipped. Returns the per-class figure payload.
    """
    classes = list(classes) if classes is not None else (
        list(FIGURE_CLASSES) if FIGURE_CLASSES is not None else fundus_class_names()[:1])
    model, transform = model_utils.load_backbone()
    vlm = flair_maps.VLM()

    per_class = {}
    for i, name in enumerate(classes, 1):
        print(f"[fig {i}/{len(classes)}] {name}")
        try:
            per_class[name] = run_class(name, model, transform, vlm, make_figures=True)["figure"]
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
