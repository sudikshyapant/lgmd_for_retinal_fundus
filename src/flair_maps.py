"""Localized CLIP similarity maps S via red-circle prompting (paper Sec 3.3).

For each image and each cell of an (h x w) grid aligned to the encoder resolution,
we overlay a small red circle, encode the whole image with CLIP, and take the cosine
similarity to each concept's text embedding. The result is the fixed, language-aligned
coefficient matrix S used in the reconstruction A_bar ~ S W^T.
"""

import torch
import torch.nn.functional as F
from PIL import ImageDraw
from tqdm import tqdm

from config import CONFIG
from data_utils import clip_preprocess

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class VLM:
    """FLAIR image/text wrapper — the retina vision-language model behind the maps S.

    FLAIR ("A Foundation LAnguage-Image model of the Retina", github.com/jusiro/FLAIR)
    is a CLIP pre-trained on fundus images, so its shared image-text space is
    domain-matched to retinal pathology. Both encoders return L2-normalized embeddings
    in that space (FLAIR's norm_features default), so cosine similarity is a dot product.

    Images arrive having ALREADY had the 224 geometric crop + red circle applied
    upstream; FLAIR's own preprocessing then resizes uniformly, so the red circle's
    relative position — and thus the encoder/VLM grid alignment — is preserved.
    """

    def __init__(self):
        from flair import FLAIRModel
        # from_checkpoint downloads the pretrained retina weights unless a local
        # weights_path is given.
        self.model = FLAIRModel(from_checkpoint=True,
                                weights_path=CONFIG["flair_weights"])
        self.model = self.model.to(DEVICE).eval()

    @torch.no_grad()
    def embed_text(self, texts):
        # preprocess_text tokenizes the raw prompts (no caption template — that is only
        # applied by compute_text_embeddings, which we bypass to keep our own
        # prompt_template the sole templating); text_model returns normalized embeds.
        ids, mask = self.model.preprocess_text(list(texts))
        e = self.model.text_model(ids.to(DEVICE), mask.to(DEVICE))
        return F.normalize(e, dim=-1).cpu()

    @torch.no_grad()
    def embed_images(self, pil_images):
        import numpy as np
        # Images are already 224x224 (geometric preprocessing done upstream). FLAIR's
        # preprocess_image expects numpy H×W×C uint8 and resizes/normalizes to the
        # model's native input. Preprocess per image, then batch — vision_model returns
        # L2-normalized embeddings in the shared image-text space.
        px = torch.cat([self.model.preprocess_image(np.asarray(im))
                        for im in pil_images], dim=0)
        e = self.model.vision_model(px.to(DEVICE))
        return F.normalize(e, dim=-1).cpu()


def _grid_variants(img224, grid, radius):
    """Return grid*grid copies of the image, each with a circle marker at one cell center.

    Cells are visited row-major (i over h, j over w) so variant index = i*grid + j,
    matching the row-major spatial unfolding of the encoder feature map.
    """
    cell = img224.size[0] / grid
    width, color = CONFIG["circle_width"], CONFIG["circle_color"]
    variants = []
    for i in range(grid):           # row  (h)
        for j in range(grid):       # col  (w)
            v = img224.copy()
            cx, cy = (j + 0.5) * cell, (i + 0.5) * cell
            ImageDraw.Draw(v).ellipse(
                [cx - radius, cy - radius, cx + radius, cy + radius],
                outline=color, width=width,
            )
            variants.append(v)
    return variants


def build_S(images, concepts, vlm):
    """Build the semantic activation matrix S of shape (n*h*w, r).

    Each row is a spatial location; each column is a named concept. Values are the
    (non-negative) image-text cosine similarities under red-circle localization.
    """
    grid, radius = CONFIG["grid"], CONFIG["circle_radius"]
    prompts = [CONFIG["prompt_template"].format(c) for c in concepts]
    text_emb = vlm.embed_text(prompts)                  # (r, d)
    rows = []
    for img in tqdm(images, desc="VLM similarity maps S"):
        variants = _grid_variants(clip_preprocess(img), grid, radius)
        img_emb = vlm.embed_images(variants)            # (h*w, d)
        sim = img_emb @ text_emb.T                       # (h*w, r) cosine similarity
        rows.append(sim)
    return torch.cat(rows, 0).clamp(min=0)               # S in R_+^{(nhw) x r}
