"""Baseline concept decompositions for comparison (paper Sec 4): ICE, CRAFT, FACE.

Per the paper, all methods use identical backbones, preprocessing, data splits, and
concept count r. Each baseline learns a concept basis W (p, r) from the training
activations; validation reconstruction reuses LGMD's non-negative inference
(`lgmd.infer`), so the only difference between methods is the learned basis.
"""

import numpy as np
import torch
from sklearn.decomposition import NMF

from config import CONFIG

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _np(A):
    return A.numpy() if torch.is_tensor(A) else A


def _nmf(A, r):
    model = NMF(n_components=r, init="nndsvda", max_iter=CONFIG["nmf_max_iter"], random_state=CONFIG["seed"])
    U = model.fit_transform(np.clip(_np(A), 0, None))   # NMF needs non-negative input
    return U, model.components_.T                        # U (N, r), W (p, r)


def fit_ice(A, r):
    """ICE (AAAI'21): plain NMF concepts with an invertible activation mapping."""
    return torch.tensor(_nmf(A, r)[1], dtype=torch.float32)   # W (p, r)


def fit_craft(A, r, n_levels=None):
    """CRAFT (CVPR'23): recursive NMF.

    After an initial factorization, re-factorize the support of the most active
    concept to refine the basis — CRAFT's recursive concept extraction. This yields
    a basis distinct from ICE's single-level NMF.
    """
    n_levels = n_levels if n_levels is not None else CONFIG["craft_levels"]
    A_np = np.clip(_np(A), 0, None)
    U, W = _nmf(A_np, r)
    for _ in range(n_levels - 1):
        dom = U.sum(0).argmax()                          # most active concept
        mask = U[:, dom] > U[:, dom].mean()              # its spatial support
        if mask.sum() < r:
            break
        _, W = _nmf(A_np[mask], r)                        # refine basis on that region
    return torch.tensor(W, dtype=torch.float32)


def fit_face(A, r, head_logits, shape, lam=None, iters=None, lr=None):
    """FACE: NMF reconstruction + KL alignment to the classifier's predictions.

    Adds a KL(p_orig || p_recon) term between logits on original vs reconstructed
    (globally pooled) activations, aligning the decomposition with the decision surface.

    head_logits: callable mapping pooled features (n, p) -> logits (n, c).
    shape:       (n, p, h, w) so spatial rows can be pooled back to per-image features.
    """
    lam = lam if lam is not None else CONFIG["face_lambda"]
    iters = iters if iters is not None else CONFIG["face_iters"]
    lr = lr if lr is not None else CONFIG["face_lr"]
    n, p, h, w = shape
    A = A.to(DEVICE)
    # warm-start from a plain NMF factorization
    nmf = NMF(n_components=r, init="nndsvda", max_iter=CONFIG["nmf_max_iter"], random_state=CONFIG["seed"])
    U0 = torch.tensor(nmf.fit_transform(np.clip(_np(A.cpu()), 0, None)), dtype=torch.float32)
    W0 = torch.tensor(nmf.components_.T, dtype=torch.float32)

    U = U0.to(DEVICE).requires_grad_(True)
    W = W0.to(DEVICE).requires_grad_(True)
    opt = torch.optim.Adam([U, W], lr=lr)

    def pool(M):                                        # (N, p) -> per-image GAP (n, p)
        return M.reshape(n, h, w, p).permute(0, 3, 1, 2).mean((2, 3))

    target = torch.log_softmax(head_logits(pool(A)).to(DEVICE), -1).detach()
    for _ in range(iters):
        opt.zero_grad()
        A_hat = U @ W.T
        recon = 0.5 * ((A - A_hat) ** 2).sum()
        pred = torch.log_softmax(head_logits(pool(A_hat)).to(DEVICE), -1)
        kl = torch.nn.functional.kl_div(pred, target, log_target=True, reduction="batchmean")
        (recon + lam * kl).backward()
        opt.step()
        with torch.no_grad():
            U.clamp_(min=0)
            W.clamp_(min=0)
    return W.detach().cpu()
