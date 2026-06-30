"""Small shared helpers: reproducibility and load-or-compute caching."""

import json
import os
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cached(path, compute_fn):
    """Return the artifact at `path`, computing + saving it on first use.

    Used for the "constant" artifacts (activations, CLIP maps S, basis W) that are
    expensive to compute and persisted to Google Drive on Colab.
    """
    if os.path.exists(path):
        print(f"[cache] loaded {os.path.basename(path)}")
        return torch.load(path)
    obj = compute_fn()
    torch.save(obj, path)
    print(f"[cache] saved {os.path.basename(path)}")
    return obj


def cached_json(path, compute_fn):
    """Load-or-compute for JSON-serializable results (metric tables).

    Saves human-readable JSON to Drive so reruns reuse computed results instead of
    refitting baselines / re-running inference.
    """
    if os.path.exists(path):
        print(f"[cache] loaded {os.path.basename(path)}")
        with open(path) as f:
            return json.load(f)
    obj = compute_fn()
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"[cache] saved {os.path.basename(path)}")
    return obj
