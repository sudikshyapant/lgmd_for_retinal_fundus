"""RETFound (DINOv2 ViT-Large) backbone for the LGMD pipeline.

RETFound's DINOv2 variant is a DINOv2 ViT-L/14 self-supervised foundation model
pretrained on retinal images. Here it is used as a *frozen* encoder with a small
DR-grading head linear-probed on top (CONFIG['num_classes'] logits).

Encoder / head split expected by the rest of the pipeline (see model_utils):
  - encoder f: image -> DINOv2 patch tokens, reshaped to a (n, d, g0, g0) map
      (g0 = img_size / patch = 224/14 = 16), then average-pooled to CONFIG['grid'].
  - head    g: global-average-pool the map -> linear head. DINOv2 already
      LayerNorm-normalizes its patch tokens (`x_norm_patchtokens`), so no extra
      fc_norm is needed; GAP-then-linear reproduces a mean-patch-token linear probe
      *exactly* whenever g0 is a multiple of CONFIG['grid'] (uniform 2x2 pooling ->
      mean-of-cells == mean-of-tokens).

Weights:
  CONFIG['retfound_weights'] must point at the RETFound-DINOv2 pretrained *encoder*
  checkpoint (a DINOv2 ViT-L/14 state dict). The architecture itself is fetched via
  torch.hub from facebookresearch/dinov2 (CONFIG['retfound_arch'], default
  'dinov2_vitl14'; use 'dinov2_vitl14_reg' for the register-token variant). The
  trained linear head is saved/loaded separately (CONFIG['backbone_weights']).
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CONFIG

_HUB_REPO = "facebookresearch/dinov2"


class RETFoundDINOv2(nn.Module):
    """DINOv2 ViT-L/14 frozen encoder + a linear DR-grading head."""

    def __init__(self, num_classes, arch=None, img_size=None):
        super().__init__()
        arch = arch or CONFIG.get("retfound_arch", "dinov2_vitl14")
        self.img_size = img_size or CONFIG.get("retfound_img_size", 224)
        # pretrained=False: fetch the *architecture* only; RETFound weights are loaded
        # separately via load_encoder_weights (keeps this offline-friendly once cached).
        self.backbone = torch.hub.load(_HUB_REPO, arch, pretrained=False)
        self.embed_dim = self.backbone.embed_dim          # 1024 for ViT-L
        self.patch = self.backbone.patch_size             # 14
        self.grid_native = self.img_size // self.patch     # 16 at 224
        self.head = nn.Linear(self.embed_dim, num_classes)

    # --- weights ----------------------------------------------------------
    def load_encoder_weights(self, path):
        """Load the RETFound-DINOv2 pretrained encoder state dict into self.backbone."""
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"RETFound-DINOv2 encoder weights not found at {path}. Set "
                f"CONFIG['retfound_weights'] to the downloaded DINOv2 ViT-L/14 checkpoint."
            )
        sd = torch.load(path, map_location="cpu")
        sd = sd.get("model", sd.get("teacher", sd.get("state_dict", sd)))  # unwrap common containers
        sd = {k.replace("backbone.", "").replace("module.", ""): v for k, v in sd.items()}
        missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
        kept = len(sd) - len(unexpected)
        print(f"[retfound] loaded encoder from {os.path.basename(path)}: "
              f"{kept} tensors matched, {len(missing)} missing, {len(unexpected)} unexpected")
        return self

    # --- encoder / head split --------------------------------------------
    def patch_tokens(self, x):
        """LayerNorm-normalized DINOv2 patch tokens, (n, P, d) with P = g0*g0."""
        return self.backbone.forward_features(x)["x_norm_patchtokens"]

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


def build(num_classes, load_pretrained=True):
    """Construct RETFoundDINOv2; optionally load the RETFound encoder weights."""
    model = RETFoundDINOv2(num_classes)
    if load_pretrained:
        model.load_encoder_weights(CONFIG["retfound_weights"])
    return model
