"""LGMD core: fit the semantic concept basis W and run NNLS inference.

Training (Sec 3.4):   min_{W>=0} 1/2 ||A_bar - S W^T||_F^2     (S fixed, learn W)
Inference (Sec 3.5):  min_{S_hat>=0} 1/2 ||S_hat W^T - A_bar||_F^2  (W fixed, learn S_hat)

NMF assumes a NON-NEGATIVE activation matrix A_bar. Conv+ReLU encoders give that for free,
but a ViT encoder (RETFound) emits signed, LayerNorm-normalized tokens. The caller (runner)
therefore shifts activations to the non-negative orthant by their per-channel min before
fitting, and `reconstruct` adds that offset back so the head still sees original-space
features. So the A passed to fit_basis / infer here is already the shifted, non-negative one.
"""

import torch
from sklearn.decomposition._nmf import _initialize_nmf
from tqdm import tqdm

from config import CONFIG

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def unfold(Z):
    """Z (n, p, h, w) -> A_bar (n*h*w, p), row-major over spatial locations."""
    n, p, h, w = Z.shape
    return Z.permute(0, 2, 3, 1).reshape(n * h * w, p)


def _nndsvd_init(A, r):
    """Initialize the basis W (p, r) with NNDSVD for stable, accelerated PGD (Sec 3.4).

    NNDSVD factorizes A ~ W0 @ H0 with H0 of shape (r, p); our basis is W = H0^T.
    """
    _, H0 = _initialize_nmf(A.numpy(), r, init="nndsvda")
    return torch.tensor(H0.T, dtype=torch.float32)


def fit_basis(A, S):
    """Learn W >= 0 minimizing 1/2 ||A - S W^T||_F^2 via projected gradient descent.

    The step size 1/L uses the Lipschitz constant L = ||S^T S||_2, and W is projected
    back to the non-negative orthant after each step (Eq. 2).
    """
    r = S.shape[1]
    W = _nndsvd_init(A, r).to(DEVICE)
    A, S = A.to(DEVICE), S.to(DEVICE)

    StS = S.T @ S                                       # (r, r)
    StA = S.T @ A                                       # (r, p)
    eta = 1.0 / torch.linalg.matrix_norm(StS, ord=2)    # 1 / Lipschitz constant

    pbar = tqdm(range(CONFIG["pgd_iters"]), desc="fit W (PGD)")
    for it in pbar:
        grad = (StS @ W.T - StA).T                      # dL/dW, shape (p, r)
        W = torch.clamp(W - eta * grad, min=0)          # project to W >= 0
        if it % 25 == 0:
            loss = 0.5 * torch.linalg.matrix_norm(A - S @ W.T) ** 2
            pbar.set_postfix(recon_loss=float(loss))
    return W.cpu()


def infer(A, W, refine=True):
    """Estimate non-negative semantic coefficients S_hat for new activations.

    Fast init via the projected normal equation (Eq. 4), optional PGD refinement.
    """
    A, W = A.to(DEVICE), W.to(DEVICE)
    WtW = W.T @ W                                       # (r, r)
    S_hat = torch.clamp(A @ W @ torch.linalg.pinv(WtW), min=0)   # Eq. 4
    if refine:
        AW = A @ W
        eta = 1.0 / torch.linalg.matrix_norm(WtW, ord=2)
        for _ in range(CONFIG["infer_iters"]):
            S_hat = torch.clamp(S_hat - eta * (S_hat @ WtW - AW), min=0)
    return S_hat.cpu()


def reconstruct(S_hat, W, shape, offset=None):
    """A_hat = S_hat W^T (+ offset), reshaped back to a spatial feature map (n, p, h, w).

    `offset` (per-channel, shape (p,)) is the non-negative shift the activations were
    reduced by before factorization (see runner). Adding it back lands the reconstruction
    in the ORIGINAL activation space — the space the classifier head reads — so the head
    can be applied directly with no change. Omit it to stay in the shifted space.
    """
    n, p, h, w = shape
    A_hat = S_hat @ W.T
    if offset is not None:
        A_hat = A_hat + offset                          # per-channel broadcast over locations
    return A_hat.reshape(n, h, w, p).permute(0, 3, 1, 2)
