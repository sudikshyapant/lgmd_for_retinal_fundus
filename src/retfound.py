"""RETFound backbones (DINOv2 ViT-L/14 and MAE ViT-L/16) for the LGMD pipeline.

RETFound ships in two foundation-model flavors, both usable here as a *frozen* encoder
with a small DR-grading head linear-probed on top (CONFIG['num_classes'] logits):

  - retfound_dinov2 — a DINOv2 ViT-L/14 (patch 14, 16x16 native token grid at 224),
      architecture fetched via torch.hub from facebookresearch/dinov2. Patch tokens come
      from forward_features(x)['x_norm_patchtokens'] (already LayerNorm-normalized).
  - retfound_mae — the original RETFound (Nature 2023): an MAE-pretrained ViT-L/16
      (patch 16, 14x14 native token grid at 224), architecture built via timm
      (vit_large_patch16_224). Patch tokens come from forward_features(x)[:, 1:] — timm
      applies the encoder's final LayerNorm, so these too are normalized tokens.

Encoder / head split expected by the rest of the pipeline (see model_utils):
  - encoder f: image -> normalized patch tokens, reshaped to a (n, d, g0, g0) map
      (g0 = img_size / patch), then average-pooled to CONFIG['grid'].
  - head    g: global-average-pool the map -> linear head. Because the tokens are already
      LayerNorm-normalized, no extra fc_norm is needed; GAP-then-linear reproduces a
      mean-patch-token linear probe *exactly* whenever g0 is a multiple of CONFIG['grid']
      (uniform pooling -> mean-of-cells == mean-of-tokens).

Weights:
  CONFIG['retfound_weights'] must point at the RETFound pretrained *encoder* checkpoint
  matching CONFIG['backbone'] (a DINOv2 ViT-L/14 state dict, or the MAE ViT-L/16
  RETFound_mae_natureCFP state dict). The trained linear head is saved/loaded separately
  (CONFIG['backbone_weights']).
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CONFIG

_HUB_REPO = "facebookresearch/dinov2"


class _RETFoundBase(nn.Module):
    """Frozen RETFound foundation encoder + a linear DR-grading head.

    Subclasses set self.backbone (the architecture), self.embed_dim, self.patch, and
    implement patch_tokens(); the encoder/head split and weight loading are shared.
    """

    def _init_head(self, num_classes):
        self.head = nn.Linear(self.embed_dim, num_classes)

    # --- weights ----------------------------------------------------------
    def load_encoder_weights(self, path):
        """Load the RETFound pretrained encoder state dict into self.backbone."""
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"RETFound encoder weights not found at {path}. Set "
                f"CONFIG['retfound_weights'] to the downloaded checkpoint for "
                f"CONFIG['backbone']={CONFIG['backbone']!r}."
            )
        sd = torch.load(path, map_location="cpu")
        sd = sd.get("model", sd.get("teacher", sd.get("state_dict", sd)))  # unwrap common containers
        sd = {k.replace("backbone.", "").replace("module.", ""): v for k, v in sd.items()}
        # Keep only keys that exist in the target with a matching shape — robust across
        # checkpoint layouts (drops decoder/head/mask tokens and any size-mismatched pos_embed).
        tgt = self.backbone.state_dict()
        sd = {k: v for k, v in sd.items() if k in tgt and v.shape == tgt[k].shape}
        missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
        print(f"[retfound] loaded encoder from {os.path.basename(path)}: "
              f"{len(sd)} tensors matched, {len(missing)} missing, {len(unexpected)} unexpected")
        return self

    # --- encoder / head split --------------------------------------------
    def patch_tokens(self, x):
        """Normalized patch tokens, (n, P, d) with P = g0*g0. Implemented per backbone."""
        raise NotImplementedError

    def feature_map(self, x):
        """encoder f: (n, d, grid, grid), pooled from the native g0 x g0 token grid."""
        t = self.patch_tokens(x)                          # (n, P, d)
        n, P, d = t.shape
        g0 = int(round(P ** 0.5))
        z = t.transpose(1, 2).reshape(n, d, g0, g0)       # (n, d, g0, g0)
        grid = CONFIG["grid"]
        if grid != g0:
            if g0 % grid != 0:
                raise ValueError(
                    f"CONFIG['grid']={grid} must divide the native token grid {g0} "
                    f"(img_size/patch) so GAP == mean-over-tokens stays exact. "
                    f"Set grid to a divisor of {g0} (e.g. {g0 // 2})."
                )
            z = F.avg_pool2d(z, kernel_size=g0 // grid)    # exact uniform pooling
        return z

    def classify(self, a):
        """g's final layer: globally-pooled features a (n, d) -> logits."""
        return self.head(a)

    def forward(self, x):
        """Full predictor: mean patch token -> linear head (== head(feature_map(x)))."""
        return self.head(self.patch_tokens(x).mean(dim=1))


class RETFoundDINOv2(_RETFoundBase):
    """RETFound (DINOv2 ViT-L/14): torch.hub frozen encoder + a linear DR-grading head."""

    def __init__(self, num_classes, arch=None, img_size=None):
        super().__init__()
        arch = arch or CONFIG.get("retfound_arch", "dinov2_vitl14")
        self.img_size = img_size or CONFIG.get("retfound_img_size", 224)
        # pretrained=False: fetch the *architecture* only; RETFound weights are loaded
        # separately via load_encoder_weights (keeps this offline-friendly once cached).
        self.backbone = torch.hub.load(_HUB_REPO, arch, pretrained=False)
        self.embed_dim = self.backbone.embed_dim          # 1024 for ViT-L
        self.patch = self.backbone.patch_size             # 14
        self._init_head(num_classes)

    def patch_tokens(self, x):
        """LayerNorm-normalized DINOv2 patch tokens, (n, P, d) with P = g0*g0."""
        return self.backbone.forward_features(x)["x_norm_patchtokens"]


class RETFoundMAE(_RETFoundBase):
    """RETFound (MAE ViT-L/16): timm vit_large_patch16 frozen encoder + a linear head."""

    def __init__(self, num_classes, arch=None, img_size=None):
        super().__init__()
        import timm
        arch = arch or CONFIG.get("retfound_arch", "vit_large_patch16_224")
        self.img_size = img_size or CONFIG.get("retfound_img_size", 224)
        # num_classes=0 drops timm's own classifier; we only use forward_features (the
        # architecture), then load RETFound's MAE encoder weights via load_encoder_weights.
        self.backbone = timm.create_model(
            arch, pretrained=False, num_classes=0, img_size=self.img_size)
        self.embed_dim = self.backbone.embed_dim          # 1024 for ViT-L
        self.patch = self.backbone.patch_embed.patch_size[0]   # 16
        self._init_head(num_classes)

    def patch_tokens(self, x):
        """Normalized patch tokens, (n, P, d). timm's forward_features applies the encoder's
        final LayerNorm and returns the cls token at index 0, so drop it (P = (img/16)^2)."""
        t = self.backbone.forward_features(x)             # (n, 1 + P, d), norm applied
        return t[:, 1:, :]


def build(num_classes, load_pretrained=True):
    """Construct the RETFound backbone for CONFIG['backbone']; optionally load its weights."""
    cls = RETFoundMAE if CONFIG["backbone"] == "retfound_mae" else RETFoundDINOv2
    model = cls(num_classes)
    if load_pretrained:
        model.load_encoder_weights(CONFIG["retfound_weights"])
    return model
