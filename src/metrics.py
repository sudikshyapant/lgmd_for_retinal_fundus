"""Predictive-preservation and faithfulness metrics.

Predictive preservation: does running the *reconstructed* activations through the
classifier head reproduce the original predictions?  Faithfulness: KL between logits,
relative reconstruction error, and C-Deletion / C-Insertion curves over concepts.
"""

import torch
import torch.nn.functional as F

from config import CONFIG


@torch.no_grad()
def predictive_preservation(orig_logits, recon_logits, label):
    """Accuracy on original vs reconstructed activations, plus prediction agreement."""
    orig_pred = orig_logits.argmax(-1)
    recon_pred = recon_logits.argmax(-1)
    return {
        "orig_acc": float((orig_pred == label).float().mean()),
        "recon_acc": float((recon_pred == label).float().mean()),
        "agreement": float((orig_pred == recon_pred).float().mean()),
    }


@torch.no_grad()
def kl_logits(orig_logits, recon_logits):
    """Mean KL(p_orig || p_recon) between softmax predictions (lower = more faithful)."""
    p = F.log_softmax(orig_logits, -1)
    q = F.log_softmax(recon_logits, -1)
    return float(F.kl_div(q, p, log_target=True, reduction="batchmean"))


def recon_error(A, A_hat):
    """Relative Frobenius reconstruction error ||A - A_hat||_F / ||A||_F."""
    return float(torch.linalg.matrix_norm(A - A_hat) / torch.linalg.matrix_norm(A))


def mse(A, A_hat):
    """Mean squared reconstruction error (Table 4 MSE column)."""
    return float(((A - A_hat) ** 2).mean())


@torch.no_grad()
def faithfulness_curves(S_hat, W, shape, head_fn, label, metric=None):
    """C-Deletion and C-Insertion curves over concepts, ordered by importance.

    Importance = total activation mass per concept. Deletion removes concepts most-
    important-first (steep early drop = faithful); insertion adds them in the same
    order (steep early rise = faithful).

    `metric` ("prob" | "accuracy", default CONFIG["cins_metric"]) selects how "model
    performance" is measured at each step: mean true-class probability, or accuracy.
    """
    n, p, h, w = shape
    metric = metric or CONFIG["cins_metric"]
    importance = S_hat.sum(0)                           # (r,)
    order = torch.argsort(importance, descending=True)

    def true_prob(S):
        A_hat = (S @ W.T).reshape(n, h, w, p).permute(0, 3, 1, 2)
        logits = head_fn(A_hat)
        if metric == "accuracy":
            return float((logits.argmax(-1) == label).float().mean())
        return float(F.softmax(logits, -1)[:, label].mean())

    # deletion: start full, zero out concepts one by one
    deletion, S = [], S_hat.clone()
    deletion.append(true_prob(S))
    for k in order.tolist():
        S[:, k] = 0
        deletion.append(true_prob(S))

    # insertion: start empty, add concepts most-important-first
    insertion, S = [], torch.zeros_like(S_hat)
    insertion.append(true_prob(S))
    for k in order.tolist():
        S[:, k] = S_hat[:, k]
        insertion.append(true_prob(S))

    return {"deletion": deletion, "insertion": insertion}


def insertion_auc(curve):
    """C-Ins scalar: normalized area under the insertion curve (in [0, 1]).

    Higher = model performance is restored faster as top-ranked concepts are added.
    """
    c = torch.tensor(curve, dtype=torch.float32)
    return float(c.mean())
