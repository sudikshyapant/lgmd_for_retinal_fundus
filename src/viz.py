"""Publication-quality visualization for LGMD concept discovery (paper Sec 3.6 / Eq. 6).

Project the inferred semantic coefficients S_hat back into the spatial domain, normalize
each concept map to [0, 1], upsample, and overlay it on the input image. Because each
column of S_hat corresponds to a named concept, every heatmap is human-interpretable.

All figures and tables share one design system (`_init_style()` + the constants below):
a single sans-serif family, a fixed type scale, one heatmap colormap, one accent colour
for discovered concepts, 300-DPI output on a white canvas, and consistent table styling.
Keeping every figure on the same system is what makes the panels read as one coherent set.
"""

import math
import os
import re
import textwrap

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm

from config import CONFIG
from data_utils import clip_preprocess

# --- design system ---------------------------------------------------------
SAVE_DPI = 300                     # publication raster resolution
HEAT_CMAP = "viridis"              # one perceptually-uniform colormap everywhere
HEAT_ALPHA = 0.55                  # overlay opacity for every heatmap
CONCEPT_COLOR = "#1f4e79"          # deep blue: discovered / named LGMD concepts
CONTOUR_COLOR = "#d62728"          # red: ICE active-region outline
ICE_CONTOUR_LEVEL = 0.72           # activation threshold for the ICE outline; higher ->
                                   # tighter, smaller circle (tuned for small fundus lesions)
HEADER_BG = "#e8edf4"             # table header fill
STRIPE_BG = "#f6f8fb"             # table zebra-stripe fill
EDGE_COLOR = "#d7dee8"            # table grid lines

# type scale (points)
FS_SUPTITLE = 14
FS_TITLE = 11
FS_LABEL = 10
FS_CONCEPT = 8
FS_TICK = 8

WRAP = 16                          # default character width for wrapped labels


def _init_style():
    """Apply the shared rcParams once so every figure/table is visually consistent."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": SAVE_DPI,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": FS_LABEL,
        "axes.titlesize": FS_TITLE,
        "axes.titleweight": "medium",
        "axes.linewidth": 0.8,
        "figure.titlesize": FS_SUPTITLE,
        "figure.titleweight": "semibold",
        "image.interpolation": "nearest",
    })


_init_style()


# --- shared helpers --------------------------------------------------------
_PRETTY_CLASS = {
    "diabetes": "Diabetic Retinopathy",
    "diabetic_retinopathy": "Diabetic Retinopathy",
    "amd": "AMD",
    "normal": "Normal",
    "normal_fundus": "Normal",
    "cataract": "Cataract",
    "glaucoma": "Glaucoma",
}


def _pretty_class(name):
    """Dataset folder name -> clean clinical display label (e.g. 'Diabetes' -> 'Diabetic Retinopathy')."""
    key = re.sub(r"[\s\-]+", "_", str(name).strip().lower())
    return _PRETTY_CLASS.get(key, str(name).replace("_", " ").title())


def _wrap(text, width=WRAP):
    """Soft-wrap a label onto stacked lines so long phrases never clip the axes/block edge."""
    text = str(text)
    return "\n".join(textwrap.wrap(text, width=width)) or text


def _concept_title(name, width=WRAP):
    """Quoted, wrapped concept label used as an axes title."""
    return _wrap(f'“{name}”', width=width)


def _save(fig, out_name, pad=0.12):
    """Save a figure to viz_dir at the shared DPI, display it, and return the path."""
    path = os.path.join(CONFIG["viz_dir"], out_name)
    fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight", pad_inches=pad)
    plt.show()
    print(f"[viz] saved {path}")
    return path


def _heatmap(coeffs_k, grid, size=224):
    """One concept's coefficients (h*w,) -> normalized [0,1] heatmap upsampled to `size`."""
    m = coeffs_k.reshape(grid, grid)
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    m = F.interpolate(m[None, None], size=size, mode="bilinear", align_corners=False)
    return m[0, 0].numpy()


def _side_label(ax, text, color="black", size=FS_LABEL, wrap=None):
    """Rotated row/block label pinned at an axis's left margin (survives axis('off'))."""
    label = _wrap(text, wrap) if wrap else text
    ax.annotate(label, xy=(0, 0.5), xytext=(-10, 0), xycoords="axes fraction",
                textcoords="offset points", ha="right", va="center", rotation=90,
                fontsize=size, color=color)


def _add_heat_colorbar(fig, axes, label="normalized concept activation"):
    """Attach a single shared colorbar describing the [0,1] heatmap scale."""
    sm = plt.cm.ScalarMappable(cmap=HEAT_CMAP, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.026, pad=0.02)
    cbar.set_label(label, fontsize=FS_TICK)
    cbar.ax.tick_params(labelsize=max(FS_TICK - 1, 6))
    cbar.outline.set_linewidth(0.5)
    return cbar


def _style_table(tbl, n_header):
    """Give a matplotlib table a clean, consistent look: shaded bold header, zebra rows."""
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(EDGE_COLOR)
        cell.set_linewidth(0.6)
        if r == 0:                                   # header row
            cell.set_facecolor(HEADER_BG)
            cell.set_text_props(fontweight="bold")
        elif r % 2 == 0:                             # zebra striping for readability
            cell.set_facecolor(STRIPE_BG)
        if c == 0 and r > 0:                         # row label column
            cell.set_text_props(fontweight="semibold")
    tbl.auto_set_column_width(range(n_header))


def save_concept_overlays(images, S_hat, concepts, out_name="overlays.png", max_images=4):
    """Save a grid of per-concept heatmaps overlaid on the first few validation images."""
    grid, r = CONFIG["grid"], len(concepts)
    n_show = min(max_images, len(images))
    S_hat = S_hat.reshape(len(images), grid * grid, r)

    fig, axes = plt.subplots(n_show, r + 1, figsize=(1.9 * (r + 1), 1.9 * n_show),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.12})
    axes = np.atleast_2d(axes)
    for i in tqdm(range(n_show), desc="rendering overlays"):
        img = clip_preprocess(images[i])
        axes[i, 0].imshow(img)
        if i == 0:
            axes[i, 0].set_title("input", fontsize=FS_CONCEPT)
        axes[i, 0].axis("off")
        for k in range(r):
            axes[i, k + 1].imshow(img)
            axes[i, k + 1].imshow(_heatmap(S_hat[i, :, k], grid), cmap=HEAT_CMAP,
                                  alpha=HEAT_ALPHA, vmin=0, vmax=1)
            if i == 0:
                axes[i, k + 1].set_title(_concept_title(concepts[k], width=12),
                                         fontsize=6)
            axes[i, k + 1].axis("off")
    return _save(fig, out_name)


def _per_image_concept_scores(S):
    """(n*h*w, r) coefficient matrix -> (n, r) per-image concept scores.

    Each image's score for a concept is the total activation mass over its h*w spatial
    locations, so one box in the Fig-4 plot summarizes a concept across all images.
    """
    grid, r = CONFIG["grid"], S.shape[1]
    hw = grid * grid
    n = S.shape[0] // hw
    return S.reshape(n, hw, r).sum(1).numpy()       # (n, r)


def plot_score_distributions(S_before, S_after, out_name="fig4_score_distributions.png"):
    """Fig 4: per-concept distribution of activation scores before vs after optimization.

    'Before' = CLIP-similarity initialization S; 'after' = the optimized coefficients
    S_hat. One box per concept index summarizes that concept's score across all images.
    Before optimization the scores largely reflect the CLIP-similarity init; after, the
    reconstruction objective reshapes them to match encoder activations — i.e. the
    concept maps are driven by the model's internal representations, not CLIP biases.
    """
    b = _per_image_concept_scores(S_before)         # (n, r)
    a = _per_image_concept_scores(S_after)
    r = b.shape[1]
    pos = list(range(r))

    fig, ax = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for axis, data, title, color in (
        (ax[0], b, "Before optimization (CLIP-extracted)", "#9aa7b5"),
        (ax[1], a, "After optimization (learned)", CONCEPT_COLOR),
    ):
        bp = axis.boxplot([data[:, k] for k in range(r)], positions=pos, widths=0.6,
                          showfliers=False, patch_artist=True,
                          medianprops=dict(color="white", linewidth=1.2))
        for box in bp["boxes"]:
            box.set(facecolor=color, edgecolor=color, alpha=0.85)
        for art in bp["whiskers"] + bp["caps"]:
            art.set(color=color, linewidth=0.9)
        axis.set_title(title)
        axis.set_xlabel("concept index")
        axis.set_xticks(pos)
        axis.set_xticklabels(pos, fontsize=max(FS_TICK - 1, 6))
        axis.grid(axis="y", alpha=0.3, linewidth=0.6)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    ax[0].set_ylabel("concept score")
    fig.tight_layout()
    return _save(fig, out_name)


def plot_concept_heatmaps(images, S_hat, concepts, img_index=0, which=None, top_k=3,
                          out_name="fig1_concept_heatmaps.png", alpha=HEAT_ALPHA,
                          cmap=HEAT_CMAP):
    """Fig 1: one input image + its top named-concept heatmaps in a single clean row.

    `which` selects concepts by name (e.g. ["pointy ears", "green eyes", "whiskers"]);
    if None, the `top_k` concepts with the most activation mass on this image are shown.
    Each heatmap is normalized to [0,1], upsampled, and overlaid with `cmap`/`alpha`.
    """
    grid, r = CONFIG["grid"], len(concepts)
    S = S_hat.reshape(len(images), grid * grid, r)
    if which is None:
        mass = torch.as_tensor(S[img_index]).sum(0)         # (r,)
        sel = torch.argsort(mass, descending=True)[:top_k].tolist()
    else:
        sel = [concepts.index(c) for c in which]

    img = clip_preprocess(images[img_index])
    fig, axes = plt.subplots(1, len(sel) + 1, figsize=(2.7 * (len(sel) + 1), 3.0),
                             gridspec_kw={"wspace": 0.06})
    axes = np.atleast_1d(axes)
    axes[0].imshow(img)
    axes[0].set_title("input image")
    axes[0].axis("off")
    for ax, k in zip(axes[1:], sel):
        ax.imshow(img)
        ax.imshow(_heatmap(S[img_index, :, k], grid), cmap=cmap, alpha=alpha, vmin=0, vmax=1)
        ax.set_title(_concept_title(concepts[k]), fontsize=FS_CONCEPT, color=CONCEPT_COLOR)
        ax.axis("off")
    _add_heat_colorbar(fig, list(axes))
    return _save(fig, out_name)


def plot_baseline_comparison(images, method_maps, out_name="fig3_baseline_comparison.png",
                             concepts_by_method=None, n_images=3, alpha=HEAT_ALPHA,
                             cmap=HEAT_CMAP):
    """Fig 3: top-concept overlays per method across a few sample images.

    `method_maps` maps a method name (e.g. "ICE", "CRAFT", "FACE", "LGMD") to its
    inferred coefficient matrix S of shape (n*h*w, K). For each sample image we overlay
    the method's single most-activated component. NMF baselines have no concept names,
    so their components are labeled "comp #i"; pass `concepts_by_method={"LGMD": names}`
    to title LGMD's row with the discovered concept names instead (paper Fig 3 message:
    baselines give coarse unnamed regions, LGMD gives named fine-grained concepts).
    """
    grid = CONFIG["grid"]
    hw = grid * grid
    n_show = min(n_images, len(images))
    rows = ["Data Samples"] + list(method_maps.keys())
    concepts_by_method = concepts_by_method or {}

    fig, axes = plt.subplots(len(rows), n_show, figsize=(2.7 * n_show, 2.7 * len(rows)),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.14})
    axes = np.atleast_2d(axes)
    imgs = [clip_preprocess(images[i]) for i in range(n_show)]

    for j in range(n_show):
        axes[0, j].imshow(imgs[j])
        axes[0, j].axis("off")

    for row, name in enumerate(method_maps, start=1):
        S = method_maps[name]
        K = S.shape[1]
        Smap = S.reshape(-1, hw, K)
        names = concepts_by_method.get(name)
        for j in range(n_show):
            top = int(torch.as_tensor(Smap[j]).sum(0).argmax())
            axes[row, j].imshow(imgs[j])
            axes[row, j].imshow(_heatmap(Smap[j, :, top], grid), cmap=cmap, alpha=alpha,
                                vmin=0, vmax=1)
            axes[row, j].axis("off")
            label = _concept_title(names[top]) if names else f"comp #{top}"
            axes[row, j].set_title(label, fontsize=FS_CONCEPT,
                                   color=CONCEPT_COLOR if names else "black")

    # row labels (axis("off") hides ylabel, so annotate at the left margin instead)
    for row, name in enumerate(rows):
        _side_label(axes[row, 0], name, size=FS_LABEL)
    return _save(fig, out_name)


def _crop_patch(img224, cell_idx, grid, patch_cells):
    """Crop a (patch_cells x patch_cells)-cell window centered on a grid cell."""
    side = img224.size[0]
    cell = side / grid
    i, j = divmod(int(cell_idx), grid)
    cx, cy = (j + 0.5) * cell, (i + 0.5) * cell
    half = patch_cells * cell / 2
    box = (max(0, round(cx - half)), max(0, round(cy - half)),
           min(side, round(cx + half)), min(side, round(cy + half)))
    return img224.crop(box)


def plot_concept_patches(images, S, concepts=None, top_concepts=3, n_patches=3,
                         patch_cells=3, style="crop", out_name="fig3_concept_patches.png",
                         title=None):
    """Fig 3 (example-patch style): show each concept by its top-activating image regions.

    A concept (column of S, shape (n*h*w, K)) is summarized by the images where it fires
    strongest, taken at its peak spatial cell — example regions instead of a heatmap,
    matching paper Fig 3. Rows are concepts (chosen by total activation mass), columns
    are the top-activating images for that concept.

    style:
      "crop"    — crop the patch around the peak cell (CRAFT / FACE / LGMD; CRAFT/FACE
                  pool spatially, so a cropped exemplar is the natural view).
      "contour" — keep the full image and trace a red outline around the active region
                  (ICE, which localizes on the pre-pool feature map -> "red circle regions").

    `concepts` (names) labels the rows for LGMD; if None rows are unnamed components
    ('comp #k'), as for the NMF baselines.
    """
    grid = CONFIG["grid"]
    hw = grid * grid
    S = torch.as_tensor(S)
    K = S.shape[1]
    Sr = S.reshape(len(images), hw, K)

    chosen = torch.argsort(Sr.sum(dim=(0, 1)), descending=True)[:top_concepts].tolist()
    imgs224 = [clip_preprocess(im) for im in images]
    n_show = min(n_patches, len(images))

    fig, axes = plt.subplots(len(chosen), n_show, figsize=(2.1 * n_show, 2.1 * len(chosen)),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.08})
    axes = np.atleast_2d(axes)
    fig.subplots_adjust(left=0.22)                     # room for rotated concept labels
    if title:
        fig.suptitle(title)

    for row, k in enumerate(chosen):
        per_img = Sr[:, :, k]                                   # (n, hw)
        top_imgs = torch.argsort(per_img.max(dim=1).values,
                                 descending=True)[:n_show].tolist()
        for col, im_idx in enumerate(top_imgs):
            ax = axes[row, col]
            cell = int(per_img[im_idx].argmax())
            if style == "contour":                             # ICE: red region outline
                ax.imshow(imgs224[im_idx])
                ax.contour(_heatmap(Sr[im_idx, :, k], grid),
                           levels=[ICE_CONTOUR_LEVEL], colors=CONTOUR_COLOR, linewidths=1.5)
            else:                                              # crop exemplar patch
                ax.imshow(_crop_patch(imgs224[im_idx], cell, grid, patch_cells))
            ax.axis("off")
        raw = concepts[k] if concepts else f"comp #{k}"
        _side_label(axes[row, 0], raw, color=CONCEPT_COLOR if concepts else "black",
                    size=FS_CONCEPT, wrap=WRAP)
    return _save(fig, out_name)


def render_metric_table(table, out_name, methods=("OURS", "FACE", "ICE", "CRAFT"),
                        backbones=("ResNet", "MobileNet"), value_fmt="{:.2f}",
                        higher_is_better=True, title=None):
    """Tables 1 & 2: per-category metric table with best bold / second-best underlined.

    `table` is nested: table[backbone][category][method] = float. Rows are categories
    (plus an "Average" row if present); within each backbone block the best value in a
    row is bolded and the second-best underlined (paper Tables 1-2 convention). Renders
    a matplotlib table image; ties are broken by first occurrence.
    """
    cats = list(next(iter(table.values())).keys())

    def _rank(cat, bb):
        """Per-row rank of each method's value: 0 = best, 1 = second-best."""
        vals = [table[bb][cat].get(m, float("nan")) for m in methods]
        order = np.argsort(vals)
        if higher_is_better:
            order = order[::-1]
        return vals, {int(idx): pos for pos, idx in enumerate(order)}

    def fmt_row(cat, bb):
        vals, rank = _rank(cat, bb)
        cells = []
        for i, v in enumerate(vals):
            s = value_fmt.format(v)
            if rank.get(i) == 0:
                s = r"$\mathbf{" + s + "}$"           # best -> bold
            elif rank.get(i) == 1:
                s = r"$\underline{" + s + "}$"        # second -> underlined
            cells.append(s)
        return cells

    col_labels = [f"{bb}:{m}" for bb in backbones for m in methods]
    cell_text = []
    for cat in cats:
        row = [_pretty_class(cat) if cat != "Average" else cat]
        for bb in backbones:
            row += fmt_row(cat, bb)
        cell_text.append(row)

    # plain-text version to stdout (best marked '*', second-best '^')
    def text_row(cat):
        row = [cat]
        for bb in backbones:
            vals, rank = _rank(cat, bb)
            for i, v in enumerate(vals):
                mark = "*" if rank.get(i) == 0 else "^" if rank.get(i) == 1 else " "
                row.append(value_fmt.format(v) + mark)
        return row

    header = ["Category"] + col_labels
    text_rows = [text_row(cat) for cat in cats]
    widths = [max(len(r[i]) for r in [header] + text_rows) for i in range(len(header))]
    if title:
        print(title)
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(header)))
    for r in text_rows:
        print("  ".join(c.ljust(widths[i]) for i, c in enumerate(r)))
    print("(* best, ^ second-best)")

    fig, ax = plt.subplots(figsize=(1.5 * (len(col_labels) + 1), 0.55 * (len(cats) + 1)))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=FS_TITLE, loc="left", pad=12)
    tbl = ax.table(cellText=cell_text, colLabels=header, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(FS_CONCEPT)
    tbl.scale(1, 1.5)
    _style_table(tbl, len(header))
    return _save(fig, out_name)


def render_class_metric_table(class_name, comparison,
                              metrics=("Acc", "C-Ins", "agreement", "kl", "recon_err"),
                              methods=("LGMD", "FACE", "ICE", "CRAFT"),
                              higher_is_better=None, value_fmt="{:.3f}", out_name=None):
    """One class's methods x metrics table (best per metric bold, second-best underlined).

    `comparison` is results[class]["comparison"] = {method: {metric: value}}. Rows are the
    methods, columns are every metric used. Within each metric column the best method is
    bolded and the second-best underlined, using `higher_is_better[metric]` for the
    direction — Acc / C-Ins / agreement are up, kl / recon_err are down. An arrow after
    each header marks its direction. Renders + saves a styled table (and prints a
    plain-text version), mirroring render_metric_table.
    """
    higher = {"Acc": True, "C-Ins": True, "agreement": True, "kl": False, "recon_err": False}
    if higher_is_better:
        higher.update(higher_is_better)
    # keep only metrics that at least one method actually reports for this class
    metrics = [m for m in metrics if any(m in comparison.get(meth, {}) for meth in methods)]

    def rank(metric):
        """Per-column rank of each method: 0 = best, 1 = second-best (dir from `higher`)."""
        vals = [comparison.get(meth, {}).get(metric, float("nan")) for meth in methods]
        order = np.argsort(vals)
        if higher.get(metric, True):
            order = order[::-1]
        return {int(idx): pos for pos, idx in enumerate(order)}

    ranks = {metric: rank(metric) for metric in metrics}
    cell_text, text_rows = [], []
    for i, meth in enumerate(methods):
        row, trow = [meth], [meth]
        for metric in metrics:
            v = comparison.get(meth, {}).get(metric, float("nan"))
            s = value_fmt.format(v)
            r = ranks[metric].get(i)
            mark = "*" if r == 0 else "^" if r == 1 else " "
            trow.append(s + mark)
            if r == 0:
                s = r"$\mathbf{" + s + "}$"           # best -> bold
            elif r == 1:
                s = r"$\underline{" + s + "}$"        # second -> underlined
            row.append(s)
        cell_text.append(row)
        text_rows.append(trow)

    disp_class = _pretty_class(class_name)
    arrow = {m: "↑" if higher.get(m, True) else "↓" for m in metrics}
    header = ["Method"] + [f"{m} {arrow[m]}" for m in metrics]
    text_header = ["Method"] + list(metrics)
    widths = [max(len(r[i]) for r in [text_header] + text_rows) for i in range(len(text_header))]
    print(f"{disp_class} — per-metric performance (dir: " +
          ", ".join(f"{m}{arrow[m]}" for m in metrics) + ")")
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(text_header)))
    for r in text_rows:
        print("  ".join(c.ljust(widths[i]) for i, c in enumerate(r)))
    print("(* best, ^ second-best)")

    fig, ax = plt.subplots(figsize=(1.6 * (len(metrics) + 1), 0.55 * (len(methods) + 1)))
    ax.axis("off")
    ax.set_title(f"{disp_class} — per-metric performance", fontsize=FS_TITLE, loc="left", pad=12)
    tbl = ax.table(cellText=cell_text, colLabels=header, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(FS_CONCEPT)
    tbl.scale(1, 1.5)
    _style_table(tbl, len(header))
    return _save(fig, out_name or f"table_{class_name.lower()}_metrics.png")


# ---------------------------------------------------------------------------
# Multi-class composites — reproduce the paper's figures across FIGURE_CLASSES
# ---------------------------------------------------------------------------
# Each takes `per_class`: {class_name: {"images": [PIL], "concepts": [str],
# "S_hat": LGMD coeffs (n*h*w, r), "method_maps": {"ICE"/"CRAFT"/"FACE"/"LGMD": (n*h*w, K)}}}.
# Classes are laid out top-to-bottom in the given order, so the caller controls ordering.
# runner.make_figures() builds this dict for the showcase classes.

def _representative_index(S):
    """Index of the image carrying the most total concept-activation mass."""
    return int(S.sum(dim=(1, 2)).argmax()) if S.shape[0] else 0


def plot_concept_heatmaps_grid(per_class, classes=None, top_k=3, which_by_class=None,
                               out_name="fig1_concept_heatmaps_grid.png",
                               alpha=HEAT_ALPHA, cmap=HEAT_CMAP):
    """Fig 1 (multi-class): one row per class = [input image | top-k named concept heatmaps].

    For each class, a representative image is chosen and its strongest LGMD concepts are
    overlaid as heatmaps (or pass `which_by_class={cls: [names]}` to fix the concepts).
    Rows follow the order of `classes` (default: per_class insertion order).
    """
    classes = classes or list(per_class)
    grid = CONFIG["grid"]
    which_by_class = which_by_class or {}

    fig, axes = plt.subplots(len(classes), top_k + 1,
                             figsize=(2.7 * (top_k + 1), 2.7 * len(classes)),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.14})
    axes = np.atleast_2d(axes)
    fig.subplots_adjust(left=0.16)                     # room for rotated class labels
    for row, name in enumerate(classes):
        d = per_class[name]
        imgs, concepts = d["images"], d["concepts"]
        r = len(concepts)
        S = torch.as_tensor(d["S_hat"]).reshape(len(imgs), grid * grid, r)
        idx = _representative_index(S)
        which = which_by_class.get(name)
        if which:
            sel = [concepts.index(c) for c in which][:top_k]
        else:
            sel = torch.argsort(S[idx].sum(0), descending=True)[:top_k].tolist()

        img = clip_preprocess(imgs[idx])
        axes[row, 0].imshow(img)
        axes[row, 0].axis("off")
        if row == 0:
            axes[row, 0].set_title("input image")
        _side_label(axes[row, 0], _pretty_class(name), size=FS_LABEL, wrap=12)
        for col, k in enumerate(sel, start=1):
            axes[row, col].imshow(img)
            axes[row, col].imshow(_heatmap(S[idx, :, k], grid), cmap=cmap, alpha=alpha,
                                  vmin=0, vmax=1)
            axes[row, col].set_title(_concept_title(concepts[k]), fontsize=FS_CONCEPT,
                                     color=CONCEPT_COLOR)
            axes[row, col].axis("off")
        for col in range(len(sel) + 1, top_k + 1):           # hide unused cells
            axes[row, col].axis("off")

    _add_heat_colorbar(fig, axes)
    return _save(fig, out_name)


def plot_baseline_grid(per_class, classes=None, methods=("ICE", "CRAFT", "FACE"),
                       n_images=3, patch_cells=3, out_name="fig2_baseline_grid.png"):
    """Fig 2 top (multi-class, NO LGMD): rows = classes, column groups = Data Samples + methods.

    Each group shows `n_images` per class. ICE traces a red contour around its active region
    (it localizes on the feature map); CRAFT/FACE show cropped exemplar patches at the peak
    cell of their dominant component (they pool spatially). This is the paper's "baselines
    give coarse, unnamed regions" panel. Rows follow the order of `classes`.
    """
    classes = classes or list(per_class)
    grid = CONFIG["grid"]
    hw = grid * grid
    groups = ["Data Samples"] + list(methods)
    ncols = len(groups) * n_images

    fig, axes = plt.subplots(len(classes), ncols,
                             figsize=(1.5 * ncols, 1.7 * len(classes)),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.14})
    axes = np.atleast_2d(axes)
    fig.subplots_adjust(left=0.12)                     # room for rotated class labels
    for row, name in enumerate(classes):
        d = per_class[name]
        imgs224 = [clip_preprocess(im) for im in d["images"]]
        # Data Samples
        for j in range(n_images):
            ax = axes[row, j]
            if j < len(imgs224):
                ax.imshow(imgs224[j])
            ax.axis("off")
        # method groups
        for gi, m in enumerate(methods, start=1):
            S = torch.as_tensor(d["method_maps"][m])
            K = S.shape[1]
            Sr = S.reshape(len(imgs224), hw, K)
            top = int(Sr.sum(dim=(0, 1)).argmax())            # dominant component
            per_img = Sr[:, :, top]
            order = torch.argsort(per_img.max(1).values, descending=True)[:n_images].tolist()
            for j in range(n_images):
                ax = axes[row, gi * n_images + j]
                if j < len(order):
                    im_idx = order[j]
                    if m == "ICE":
                        ax.imshow(imgs224[im_idx])
                        ax.contour(_heatmap(Sr[im_idx, :, top], grid),
                                   levels=[ICE_CONTOUR_LEVEL], colors=CONTOUR_COLOR,
                                   linewidths=1.5)
                    else:
                        cell = int(per_img[im_idx].argmax())
                        ax.imshow(_crop_patch(imgs224[im_idx], cell, grid, patch_cells))
                ax.axis("off")
        # class label at the left margin
        _side_label(axes[row, 0], _pretty_class(name), size=FS_LABEL, wrap=12)

    # group headers centered over each block (top row)
    for gi, g in enumerate(groups):
        center = axes[0, gi * n_images + n_images // 2]
        center.set_title(g, fontsize=FS_TITLE)
    return _save(fig, out_name)


def plot_lgmd_panel(per_class, classes=None, top_concepts=3, n_patches=3, patch_cells=3,
                    ncols_blocks=2, out_name="fig2_lgmd_panel.png"):
    """Fig 2 bottom (multi-class, LGMD): one block per class, each showing its top named
    concepts by their top-activating image patches — the paper's "Language Guided Concept
    Discovery" panel. Concept names label each patch row. Blocks fill left-to-right,
    top-to-bottom in the order of `classes`.
    """
    classes = classes or list(per_class)
    grid = CONFIG["grid"]
    hw = grid * grid
    nrows_blocks = math.ceil(len(classes) / ncols_blocks)

    fig = plt.figure(figsize=(5.4 * ncols_blocks, 1.15 * top_concepts * nrows_blocks + 1))
    subfigs = np.atleast_1d(fig.subfigures(nrows_blocks, ncols_blocks)).ravel()
    for sf in subfigs:                                 # keep unused blocks blank & clean
        sf.set_facecolor("white")
    for bi, name in enumerate(classes):
        sf = subfigs[bi]
        sf.suptitle(_pretty_class(name), fontsize=FS_TITLE, fontweight="semibold")
        d = per_class[name]
        imgs224 = [clip_preprocess(im) for im in d["images"]]
        concepts = d["concepts"]
        S = torch.as_tensor(d["S_hat"]).reshape(len(imgs224), hw, len(concepts))
        chosen = torch.argsort(S.sum(dim=(0, 1)), descending=True)[:top_concepts].tolist()

        # reserve left margin so the rotated concept labels have room and are not clipped
        axs = np.atleast_2d(sf.subplots(top_concepts, n_patches,
                                        gridspec_kw={"left": 0.32, "wspace": 0.05,
                                                     "hspace": 0.08}))
        for r_i, k in enumerate(chosen):
            per_img = S[:, :, k]
            order = torch.argsort(per_img.max(1).values, descending=True)[:n_patches].tolist()
            for c_i in range(n_patches):
                ax = axs[r_i, c_i]
                if c_i < len(order):
                    im_idx = order[c_i]
                    cell = int(per_img[im_idx].argmax())
                    ax.imshow(_crop_patch(imgs224[im_idx], cell, grid, patch_cells))
                ax.axis("off")
            # wrap long clinical phrases onto stacked lines so the (rotated) text stays
            # within its row instead of overflowing the block edge
            _side_label(axs[r_i, 0], concepts[k], color=CONCEPT_COLOR,
                        size=FS_CONCEPT, wrap=14)
    return _save(fig, out_name)
