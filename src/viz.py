"""Concept heatmap visualization (paper Sec 3.6 / Eq. 6).

Project the inferred semantic coefficients S_hat back into the spatial domain, normalize
each concept map to [0, 1], upsample, and overlay it on the input image. Because each
column of S_hat corresponds to a named concept, every heatmap is human-interpretable.
"""

import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm

from config import CONFIG
from data_utils import clip_preprocess


def _heatmap(coeffs_k, grid, size=224):
    """One concept's coefficients (h*w,) -> normalized [0,1] heatmap upsampled to `size`."""
    m = coeffs_k.reshape(grid, grid)
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    m = F.interpolate(m[None, None], size=size, mode="bilinear", align_corners=False)
    return m[0, 0].numpy()


def save_concept_overlays(images, S_hat, concepts, out_name="overlays.png", max_images=4):
    """Save a grid of per-concept heatmaps overlaid on the first few validation images."""
    grid, r = CONFIG["grid"], len(concepts)
    n_show = min(max_images, len(images))
    S_hat = S_hat.reshape(len(images), grid * grid, r)

    fig, axes = plt.subplots(n_show, r + 1, figsize=(2 * (r + 1), 2 * n_show))
    axes = np.atleast_2d(axes)
    for i in tqdm(range(n_show), desc="rendering overlays"):
        img = clip_preprocess(images[i])
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("input", fontsize=6)
        axes[i, 0].axis("off")
        for k in range(r):
            axes[i, k + 1].imshow(img)
            axes[i, k + 1].imshow(_heatmap(S_hat[i, :, k], grid), cmap="jet", alpha=0.5)
            axes[i, k + 1].set_title(concepts[k], fontsize=5)
            axes[i, k + 1].axis("off")

    path = os.path.join(CONFIG["viz_dir"], out_name)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"[viz] saved {path}")
    return path


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

    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for axis, data, title in (
        (ax[0], b, "Before Optimization (CLIP extracted)"),
        (ax[1], a, "After Optimization (Learned)"),
    ):
        axis.boxplot([data[:, k] for k in range(r)], positions=pos,
                     widths=0.6, showfliers=False)
        axis.set_title(title)
        axis.set_xlabel("Concept index")
        axis.set_ylabel("Concept score (mean)")
        axis.set_xticks(pos)
        axis.set_xticklabels(pos, fontsize=7)
        axis.grid(axis="y", alpha=0.3)

    path = os.path.join(CONFIG["viz_dir"], out_name)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"[viz] saved {path}")
    return path


def plot_concept_heatmaps(images, S_hat, concepts, img_index=0, which=None, top_k=3,
                          out_name="fig1_concept_heatmaps.png", alpha=0.6, cmap="viridis"):
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
    fig, axes = plt.subplots(1, len(sel) + 1, figsize=(3 * (len(sel) + 1), 3))
    axes = np.atleast_1d(axes)
    axes[0].imshow(img)
    axes[0].set_title("Input Image")
    axes[0].axis("off")
    for ax, k in zip(axes[1:], sel):
        ax.imshow(img)
        ax.imshow(_heatmap(S[img_index, :, k], grid), cmap=cmap, alpha=alpha)
        ax.set_title(f'"{concepts[k]}"')
        ax.axis("off")

    path = os.path.join(CONFIG["viz_dir"], out_name)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"[viz] saved {path}")
    return path


def plot_baseline_comparison(images, method_maps, out_name="fig3_baseline_comparison.png",
                             concepts_by_method=None, n_images=3, alpha=0.6, cmap="viridis"):
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

    fig, axes = plt.subplots(len(rows), n_show, figsize=(3 * n_show, 3 * len(rows)))
    axes = np.atleast_2d(axes)
    imgs = [clip_preprocess(images[i]) for i in range(n_show)]

    for j in range(n_show):
        axes[0, j].imshow(imgs[j])
        axes[0, j].axis("off")
    axes[0, 0].set_ylabel("Data Samples", fontsize=10, rotation=90)

    for row, name in enumerate(method_maps, start=1):
        S = method_maps[name]
        K = S.shape[1]
        Smap = S.reshape(-1, hw, K)
        names = concepts_by_method.get(name)
        for j in range(n_show):
            top = int(torch.as_tensor(Smap[j]).sum(0).argmax())
            axes[row, j].imshow(imgs[j])
            axes[row, j].imshow(_heatmap(Smap[j, :, top], grid), cmap=cmap, alpha=alpha)
            axes[row, j].axis("off")
            label = f'"{names[top]}"' if names else f"comp #{top}"
            axes[row, j].set_title(label, fontsize=8)
        axes[row, 0].set_ylabel(name, fontsize=10)

    # row labels (axis("off") hides ylabel, so annotate at the left margin instead)
    for row, name in enumerate(rows):
        axes[row, 0].annotate(name, xy=(0, 0.5), xytext=(-10, 0),
                              xycoords="axes fraction", textcoords="offset points",
                              ha="right", va="center", rotation=90, fontsize=11)

    path = os.path.join(CONFIG["viz_dir"], out_name)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"[viz] saved {path}")
    return path


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

    fig, axes = plt.subplots(len(chosen), n_show,
                             figsize=(2.2 * n_show, 2.2 * len(chosen)))
    axes = np.atleast_2d(axes)
    if title:
        fig.suptitle(title, fontsize=12)

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
                           levels=[0.6], colors="red", linewidths=1.5)
            else:                                              # crop exemplar patch
                ax.imshow(_crop_patch(imgs224[im_idx], cell, grid, patch_cells))
            ax.axis("off")
        label = concepts[k] if concepts else f"comp #{k}"
        axes[row, 0].annotate(label, xy=(0, 0.5), xytext=(-12, 0),
                              xycoords="axes fraction", textcoords="offset points",
                              ha="right", va="center", rotation=90, fontsize=10,
                              color="tab:blue" if concepts else "black")

    path = os.path.join(CONFIG["viz_dir"], out_name)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"[viz] saved {path}")
    return path


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
        row = [cat]
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

    fig, ax = plt.subplots(figsize=(1.4 * (len(col_labels) + 1), 0.5 * (len(cats) + 1)))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10, loc="left", pad=12)
    tbl = ax.table(cellText=cell_text, colLabels=["Category"] + col_labels,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)

    path = os.path.join(CONFIG["viz_dir"], out_name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[viz] saved {path}")
    return path


# ---------------------------------------------------------------------------
# Multi-class composites — reproduce the paper's figures across FIGURE_CLASSES
# ---------------------------------------------------------------------------
# Each takes `per_class`: {class_name: {"images": [PIL], "concepts": [str],
# "S_hat": LGMD coeffs (n*h*w, r), "method_maps": {"ICE"/"CRAFT"/"FACE"/"LGMD": (n*h*w, K)}}}.
# runner.make_figures() builds this dict for the paper's reference classes.

def _representative_index(S):
    """Index of the image carrying the most total concept-activation mass."""
    return int(S.sum(dim=(1, 2)).argmax()) if S.shape[0] else 0


def plot_concept_heatmaps_grid(per_class, classes=None, top_k=3, which_by_class=None,
                               out_name="fig1_concept_heatmaps_grid.png",
                               alpha=0.6, cmap="viridis"):
    """Fig 1 (multi-class): one row per class = [input image | top-k named concept heatmaps].

    For each class, a representative image is chosen and its strongest LGMD concepts are
    overlaid as heatmaps (or pass `which_by_class={cls: [names]}` to fix the concepts).
    """
    classes = classes or list(per_class)
    grid = CONFIG["grid"]
    which_by_class = which_by_class or {}

    fig, axes = plt.subplots(len(classes), top_k + 1,
                             figsize=(3 * (top_k + 1), 3 * len(classes)))
    axes = np.atleast_2d(axes)
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
            axes[row, 0].set_title("Input Image")
        axes[row, 0].annotate(name, xy=(0, 0.5), xytext=(-12, 0), xycoords="axes fraction",
                              textcoords="offset points", ha="right", va="center",
                              rotation=90, fontsize=11)
        for col, k in enumerate(sel, start=1):
            axes[row, col].imshow(img)
            axes[row, col].imshow(_heatmap(S[idx, :, k], grid), cmap=cmap, alpha=alpha)
            axes[row, col].set_title(f'"{concepts[k]}"', fontsize=9)
            axes[row, col].axis("off")
        for col in range(len(sel) + 1, top_k + 1):           # hide unused cells
            axes[row, col].axis("off")

    path = os.path.join(CONFIG["viz_dir"], out_name)
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.show()
    print(f"[viz] saved {path}")
    return path


def plot_baseline_grid(per_class, classes=None, methods=("ICE", "CRAFT", "FACE"),
                       n_images=3, patch_cells=3, out_name="fig2_baseline_grid.png"):
    """Fig 2 top (multi-class, NO LGMD): rows = classes, column groups = Data Samples + methods.

    Each group shows `n_images` per class. ICE traces a red contour around its active region
    (it localizes on the feature map); CRAFT/FACE show cropped exemplar patches at the peak
    cell of their dominant component (they pool spatially). This is the paper's "baselines
    give coarse, unnamed regions" panel.
    """
    classes = classes or list(per_class)
    grid = CONFIG["grid"]
    hw = grid * grid
    groups = ["Data Samples"] + list(methods)
    ncols = len(groups) * n_images

    fig, axes = plt.subplots(len(classes), ncols,
                             figsize=(1.5 * ncols, 1.8 * len(classes)))
    axes = np.atleast_2d(axes)
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
                                   levels=[0.6], colors="red", linewidths=1.5)
                    else:
                        cell = int(per_img[im_idx].argmax())
                        ax.imshow(_crop_patch(imgs224[im_idx], cell, grid, patch_cells))
                ax.axis("off")
        # class label at the left margin
        axes[row, 0].annotate(name, xy=(0, 0.5), xytext=(-12, 0), xycoords="axes fraction",
                              textcoords="offset points", ha="right", va="center",
                              rotation=90, fontsize=11)

    # group headers centered over each block (top row)
    for gi, g in enumerate(groups):
        center = axes[0, gi * n_images + n_images // 2]
        center.set_title(g, fontsize=11)

    path = os.path.join(CONFIG["viz_dir"], out_name)
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.show()
    print(f"[viz] saved {path}")
    return path


def plot_lgmd_panel(per_class, classes=None, top_concepts=3, n_patches=3, patch_cells=3,
                    ncols_blocks=2, out_name="fig2_lgmd_panel.png"):
    """Fig 2 bottom (multi-class, LGMD): one block per class, each showing its top named
    concepts by their top-activating image patches — the paper's "Language Guided Concept
    Discovery" panel. Concept names label each patch row.
    """
    classes = classes or list(per_class)
    grid = CONFIG["grid"]
    hw = grid * grid
    nrows_blocks = math.ceil(len(classes) / ncols_blocks)

    fig = plt.figure(figsize=(5.2 * ncols_blocks, 1.1 * top_concepts * nrows_blocks + 1))
    subfigs = np.atleast_1d(fig.subfigures(nrows_blocks, ncols_blocks)).ravel()
    for bi, name in enumerate(classes):
        sf = subfigs[bi]
        sf.suptitle(name, fontsize=12)
        d = per_class[name]
        imgs224 = [clip_preprocess(im) for im in d["images"]]
        concepts = d["concepts"]
        S = torch.as_tensor(d["S_hat"]).reshape(len(imgs224), hw, len(concepts))
        chosen = torch.argsort(S.sum(dim=(0, 1)), descending=True)[:top_concepts].tolist()

        axs = np.atleast_2d(sf.subplots(top_concepts, n_patches))
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
            axs[r_i, 0].annotate(concepts[k], xy=(0, 0.5), xytext=(-10, 0),
                                 xycoords="axes fraction", textcoords="offset points",
                                 ha="right", va="center", rotation=90, fontsize=9,
                                 color="tab:blue")
    # any unused blocks (when len(classes) < grid slots) simply render empty

    path = os.path.join(CONFIG["viz_dir"], out_name)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.show()
    print(f"[viz] saved {path}")
    return path
