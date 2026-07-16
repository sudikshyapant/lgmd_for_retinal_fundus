"""Backbone encoder/head split and activation extraction.

The predictor g(f(x)) is split into:
  - encoder f: image -> spatial feature map Z (n, p, h, w)
  - head    g: Z -> logits, via global average pooling + the classifier

The backbone is RETFound (MAE ViT-L/16): a frozen foundation encoder whose patch tokens
form the (n, p, grid, grid) map, with a linear DR-grading head on top (see retfound.py).
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ImageNet normalization stats (RETFound/MAE preprocessing).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _retfound_transform():
    """RETFound preprocessing: FLAIR-shared 224 crop + ImageNet ToTensor/Normalize.

    We reuse `clip_preprocess` (resize shortest-side -> 224, center crop) for geometry so
    the encoder feature cell (i, j) and the FLAIR red-circle cell (i, j) cover the same
    pixels — otherwise the rows of A_bar and S (both unfolded row-major) would describe
    slightly different image regions.
    """
    from torchvision.transforms import Compose, Normalize, ToTensor

    from data_utils import clip_preprocess

    normalize = Compose([ToTensor(), Normalize(_IMAGENET_MEAN, _IMAGENET_STD)])
    return lambda img: normalize(clip_preprocess(img))


def build_backbone(pretrained=True):
    """Construct the RETFound MAE backbone with a fresh `num_classes`-way head.

    The RETFound SSL encoder is always loaded (it *is* the foundation model); `pretrained`
    is accepted for call-site symmetry but does not gate the encoder load. Returns
    (model_on_DEVICE, FLAIR-aligned transform).
    """
    import retfound
    from config import CONFIG
    model = retfound.build(CONFIG["num_classes"], load_pretrained=True)
    return model.to(DEVICE), _retfound_transform()


def load_backbone():
    """Load the DR-trained backbone (num_classes-way head) for the LGMD pipeline.

    Reads the trained head from CONFIG['backbone_weights'] — the file produced by
    train_backbone.py. Raises a clear error if those weights don't exist yet.
    """
    import os
    import torch
    from config import CONFIG
    wpath = CONFIG["backbone_weights"]
    if not os.path.exists(wpath):
        raise FileNotFoundError(
            f"Trained backbone weights not found at {wpath}. Run train_backbone.train() "
            f"first to linear-probe the head on the fundus dataset."
        )
    model, transform = build_backbone(pretrained=False)
    state = torch.load(wpath, map_location=DEVICE)
    model.head.load_state_dict(state)   # linear probe: only the head was trained/saved
    return model.eval(), transform


def encoder(model, x):
    """f: input images -> spatial feature map Z (n, p, grid, grid)."""
    return model.feature_map(x)


def classify_pooled(model, a):
    """g restricted to its final layer: globally-pooled features a (n, p) -> logits.

    Grad-enabled (used directly, and by FACE's KL term).
    """
    return model.classify(a)


def head(model, z):
    """g: spatial feature map -> logits, via GAP + the linear classifier."""
    a = torch.flatten(F.adaptive_avg_pool2d(z, 1), 1)    # global average pooling
    return classify_pooled(model, a)


@torch.no_grad()
def extract_activations(model, transform, images, batch_size=16, desc="activations"):
    """Run the encoder over images, returning Z (n, p, grid, grid) on CPU (p = feat_dim)."""
    feats = []
    for i in tqdm(range(0, len(images), batch_size), desc=desc):
        batch = torch.stack([transform(im) for im in images[i:i + batch_size]]).to(DEVICE)
        feats.append(encoder(model, batch).cpu())
    return torch.cat(feats, 0)


@torch.no_grad()
def logits_from_Z(model, Z, batch_size=32):
    """Classifier logits from a (possibly reconstructed) feature map Z (n, p, h, w)."""
    out = []
    for i in range(0, len(Z), batch_size):
        out.append(head(model, Z[i:i + batch_size].to(DEVICE)).cpu())
    return torch.cat(out, 0)


@torch.no_grad()
def logits_from_pooled(model, A_pooled):
    """Classifier logits from already globally-pooled features A (n, p)."""
    return classify_pooled(model, A_pooled.to(DEVICE)).cpu()
