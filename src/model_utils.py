"""Backbone encoder/head split and activation extraction.

The predictor g(f(x)) is split into:
  - encoder f: image -> spatial feature map Z (n, p, h, w)
  - head    g: Z -> logits, via global average pooling + the pretrained classifier

Supports DenseNet-121 (the active backbone for the ODIR-5K classifier) plus the paper's
ResNet34 / MobileNetV2 and ResNet50; all share the same encoder/head abstraction so the
rest of the pipeline is backbone-agnostic.
"""

import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_BACKBONES = {
    "densenet121":  (torchvision.models.densenet121,  torchvision.models.DenseNet121_Weights.IMAGENET1K_V1),
    "resnet34":     (torchvision.models.resnet34,     torchvision.models.ResNet34_Weights.IMAGENET1K_V1),
    "resnet50":     (torchvision.models.resnet50,     torchvision.models.ResNet50_Weights.IMAGENET1K_V2),
    "mobilenet_v2": (torchvision.models.mobilenet_v2, torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2),
}


def _aligned_transform(weights):
    """Backbone preprocessing that shares CLIP's exact 224 crop.

    The encoder feature cell (i, j) and the CLIP red-circle cell (i, j) must cover the
    same pixels, or the rows of A_bar and S (both unfolded row-major) describe slightly
    different image regions and S picks up spatial slop. We reuse `clip_preprocess`
    (resize shortest-side -> 224, center crop) for geometry, then apply ONLY the weights'
    ToTensor + Normalize -- deliberately skipping the weights' own resize/crop, which
    would re-zoom to the backbone's native (e.g. 256->224) field of view.
    """
    from torchvision.transforms import Compose, Normalize, ToTensor

    from data_utils import clip_preprocess

    base = weights.transforms()                          # exposes ImageNet mean/std
    normalize = Compose([ToTensor(), Normalize(base.mean, base.std)])
    return lambda img: normalize(clip_preprocess(img))


def _replace_head(model, name, num_classes):
    """Swap the pretrained 1000-way classifier for a `num_classes`-way fundus head."""
    import torch.nn as nn
    if name.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name.startswith("densenet"):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif name == "mobilenet_v2":
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"unknown backbone {name!r}")
    return model


def build_backbone(name=None, pretrained=True):
    """Construct the backbone architecture with a fresh `num_classes`-way head.

    `pretrained=True` initializes from ImageNet weights (used by train_backbone.py to
    fine-tune); `pretrained=False` gives a bare architecture (used when loading our own
    trained weights). Returns (model_on_DEVICE, CLIP-aligned transform).
    """
    from config import CONFIG
    name = name or CONFIG["backbone"]
    ctor, weights = _BACKBONES[name]
    model = ctor(weights=weights if pretrained else None)
    _replace_head(model, name, CONFIG["num_classes"])
    return model.to(DEVICE), _aligned_transform(weights)


def load_backbone(name=None):
    """Load the fundus-trained backbone (num_classes-way head) for the LGMD pipeline.

    Reads weights from CONFIG['backbone_weights'] — the file produced by
    train_backbone.py. Raises a clear error if those weights don't exist yet.
    """
    import os
    import torch
    from config import CONFIG
    name = name or CONFIG["backbone"]
    wpath = CONFIG["backbone_weights"]
    if not os.path.exists(wpath):
        raise FileNotFoundError(
            f"Trained backbone weights not found at {wpath}. Run train_backbone.train() "
            f"first to fine-tune the backbone on the fundus dataset."
        )
    model, transform = build_backbone(name, pretrained=False)
    model.load_state_dict(torch.load(wpath, map_location=DEVICE))
    return model.eval(), transform


def _is_mobilenet(model):
    return isinstance(model, torchvision.models.MobileNetV2)


def _is_densenet(model):
    return isinstance(model, torchvision.models.DenseNet)


def encoder(model, x):
    """f: input images -> spatial feature map Z (n, p, h, w)."""
    if _is_mobilenet(model):
        return model.features(x)                         # (n, 1280, 7, 7)
    if _is_densenet(model):
        # DenseNet's forward applies a final ReLU between features and pooling, so we
        # fold it into the encoder — then head() = GAP + classifier reproduces model(x)
        # exactly (and the feature map is the non-negative post-ReLU activation).
        return F.relu(model.features(x), inplace=False)  # (n, 1024, 7, 7)
    # ResNet family
    x = model.conv1(x); x = model.bn1(x); x = model.relu(x); x = model.maxpool(x)
    x = model.layer1(x); x = model.layer2(x); x = model.layer3(x); x = model.layer4(x)
    return x                                              # (n, 512|2048, 7, 7)


def classify_pooled(model, a):
    """g restricted to its final layer: globally-pooled features a (n, p) -> logits.

    Grad-enabled and architecture-aware (used directly, and by FACE's KL term).
    """
    if _is_mobilenet(model) or _is_densenet(model):
        return model.classifier(a)
    return model.fc(a)


def head(model, z):
    """g: spatial feature map -> logits, via GAP + the pretrained classifier."""
    a = torch.flatten(F.adaptive_avg_pool2d(z, 1), 1)    # global average pooling
    return classify_pooled(model, a)


@torch.no_grad()
def extract_activations(model, transform, images, batch_size=16, desc="activations"):
    """Run the encoder over images, returning Z (n, p, 7, 7) on CPU (p = feat_dim)."""
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
