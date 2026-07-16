"""RETFound (MAE ViT-L/16) backbone for the LGMD pipeline.

RETFound (Nature 2023) is an MAE-pretrained ViT-L/16 (patch 16, 14x14 native token grid at
224). Here it is used as a *frozen* foundation encoder with a small DR-grading head
linear-probed on top (CONFIG['num_classes'] logits). The architecture is built via timm
(vit_large_patch16_224); the RETFound SSL encoder weights are loaded from
CONFIG['retfound_weights'] (RETFound_mae_natureCFP.pth). The trained linear head is saved /
loaded separately (CONFIG['backbone_weights']).

Encoder / head split expected by the rest of the pipeline (see model_utils):
  - encoder f: image -> normalized patch tokens, reshaped to a (n, d, g0, g0) map
      (g0 = img_size / patch = 14), then average-pooled to CONFIG['grid'] (7).
  - head    g: global-average-pool the map -> linear head. Because timm applies the
      encoder's final LayerNorm, the tokens are already normalized, so GAP-then-linear
      reproduces a mean-patch-token linear probe *exactly* whenever g0 is a multiple of
      CONFIG['grid'] (uniform pooling -> mean-of-cells == mean-of-tokens).
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CONFIG


class RETFoundMAE(nn.Module):
    """Frozen RETFound (MAE ViT-L/16) encoder + a linear DR-grading head."""

    def __init__(self, num_classes, arch=None, img_size=None):
        super().__init__()
        import timm
        arch = arch or CONFIG.get("retfound_arch", "vit_large_patch16_224")
        self.img_size = img_size or CONFIG.get("retfound_img_size", 224)
        # num_classes=0 drops timm's own classifier; we only use forward_features (the
        # architecture), then load RETFound's MAE encoder weights via load_encoder_weights.
        self.backbone = timm.create_model(
            arch, pretrained=False, num_classes=0, img_size=self.img_size)
        self.embed_dim = self.backbone.embed_dim               # 1024 for ViT-L
        self.patch = self.backbone.patch_embed.patch_size[0]   # 16
        self.head = nn.Linear(self.embed_dim, num_classes)

    # --- weights ----------------------------------------------------------
    def load_encoder_weights(self, path):
        """Load the RETFound pretrained encoder state dict into self.backbone."""
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"RETFound encoder weights not found at {path}. Set "
                f"CONFIG['retfound_weights'] to the downloaded RETFound_mae_natureCFP checkpoint."
            )
        # weights_only=False: the RETFound checkpoint bundles an argparse.Namespace (its
        # training args), which PyTorch 2.6's default safe loader (weights_only=True) rejects.
        # Safe here — this is the trusted official RETFound encoder file. The try/except keeps
        # it working on older torch (<1.13) that lacks the weights_only argument.
        try:
            sd = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
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
        """Normalized patch tokens, (n, P, d) with P = g0*g0. timm's forward_features applies
        the encoder's final LayerNorm and returns the cls token at index 0, so drop it."""
        t = self.backbone.forward_features(x)                  # (n, 1 + P, d), norm applied
        return t[:, 1:, :]

    def feature_map(self, x):
        """encoder f: (n, d, grid, grid), pooled from the native g0 x g0 token grid."""
        t = self.patch_tokens(x)                               # (n, P, d)
        n, P, d = t.shape
        g0 = int(round(P ** 0.5))
        z = t.transpose(1, 2).reshape(n, d, g0, g0)            # (n, d, g0, g0)
        grid = CONFIG["grid"]
        if grid != g0:
            if g0 % grid != 0:
                raise ValueError(
                    f"CONFIG['grid']={grid} must divide the native token grid {g0} "
                    f"(img_size/patch) so GAP == mean-over-tokens stays exact. "
                    f"Set grid to a divisor of {g0} (e.g. {g0 // 2})."
                )
            z = F.avg_pool2d(z, kernel_size=g0 // grid)        # exact uniform pooling
        return z

    def classify(self, a):
        """g's final layer: globally-pooled features a (n, d) -> logits."""
        return self.head(a)

    def forward(self, x):
        """Full predictor: mean patch token -> linear head (== head(feature_map(x)))."""
        return self.head(self.patch_tokens(x).mean(dim=1))


def build(num_classes, load_pretrained=True):
    """Construct the RETFound MAE backbone; optionally load its SSL encoder weights."""
    model = RETFoundMAE(num_classes)
    if load_pretrained:
        model.load_encoder_weights(CONFIG["retfound_weights"])
    return model
