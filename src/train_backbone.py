"""Train the DR-grading classifier head that LGMD then explains.

LGMD explains a *trained classifier's* decisions, so before any concept discovery we need a
backbone whose head predicts the DR severity grades. This is a LINEAR PROBE: the RETFound
(MAE ViT-L/16) foundation encoder is frozen and only a fresh Linear(feat_dim -> num_classes)
head is trained (so CONFIG['backbone_weights'] stores just that head). It trains on an
`n_per_class` subset of the train split, validates on the val split, and saves the best
weights to CONFIG['backbone_weights'] — exactly where model_utils.load_backbone reads them.

Usage (from the notebook, after the sys.path setup cell):
    import train_backbone
    train_backbone.train()                 # uses CONFIG knobs (n_per_class, epochs, lr...)
    train_backbone.train(n_per_class=500)  # quick smoke run on a smaller subset
"""

import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import (Compose, Normalize, RandomHorizontalFlip,
                                     RandomResizedCrop, ToTensor)
from tqdm import tqdm

import model_utils
import utils
from config import CONFIG

DEVICE = model_utils.DEVICE


def _train_transform():
    """Light augmentation for training (random crop + flip), ImageNet normalization."""
    return Compose([
        RandomResizedCrop(224, scale=(0.7, 1.0)),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(model_utils._IMAGENET_MEAN, model_utils._IMAGENET_STD),
    ])


def _subset_indices(dataset, n_per_class, seed):
    """Pick up to `n_per_class` sample indices per class (deterministic with `seed`)."""
    by_class = {}
    for idx, (_, y) in enumerate(dataset.samples):
        by_class.setdefault(y, []).append(idx)
    rng = random.Random(seed)
    chosen = []
    for y, idxs in sorted(by_class.items()):
        rng.shuffle(idxs)
        chosen.extend(idxs[:n_per_class])
        if len(idxs) < n_per_class:
            print(f"[warn] class idx {y}: only {len(idxs)} train images (< {n_per_class}).")
    return chosen


@torch.no_grad()
def _evaluate(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def train(n_per_class=None, epochs=None, lr=None, batch_size=None, seed=None):
    """Linear-probe the DR-grading head on the fundus subset; save the best-val-acc head.

    All args default to the corresponding CONFIG knobs. Returns the path to the saved
    weights (CONFIG['backbone_weights']).
    """
    n_per_class = n_per_class if n_per_class is not None else CONFIG["n_per_class"]
    epochs      = epochs      if epochs      is not None else CONFIG["train_epochs"]
    lr          = lr          if lr          is not None else CONFIG["train_lr"]
    batch_size  = batch_size  if batch_size  is not None else CONFIG["train_batch_size"]
    seed        = seed        if seed        is not None else CONFIG["seed"]
    utils.set_seed(seed)

    # Train split: ImageFolder with augmentation, subsetted to n_per_class/class.
    train_root = os.path.join(CONFIG["data_root"], CONFIG["train_dir"])
    full_train = datasets.ImageFolder(train_root, transform=_train_transform())
    nc = len(full_train.classes)
    if nc != CONFIG["num_classes"]:
        print(f"[info] dataset has {nc} classes ({full_train.classes}); "
              f"setting CONFIG['num_classes'] = {nc}.")
        CONFIG["num_classes"] = nc
    subset = Subset(full_train, _subset_indices(full_train, n_per_class, seed))
    train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)

    # Model: frozen RETFound SSL encoder + fresh head. Built once; its FLAIR-aligned
    # center-crop transform is what LGMD will also use at eval time.
    model, eval_transform = model_utils.build_backbone(pretrained=True)

    # Val split: same eval transform as above (and as LGMD later).
    val_root = os.path.join(CONFIG["data_root"], CONFIG["val_dir"])
    val_ds = datasets.ImageFolder(val_root, transform=eval_transform)
    assert val_ds.classes == full_train.classes, "train/val class folders differ"
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    # Linear probe: freeze the foundation encoder, train only the head.
    for p in model.backbone.parameters():
        p.requires_grad = False
    print(f"[train] RETFound linear probe: encoder frozen, training head "
          f"({sum(p.numel() for p in model.head.parameters())} params).")
    opt = torch.optim.Adam(model.head.parameters(), lr=lr,
                           weight_decay=CONFIG["train_weight_decay"])
    loss_fn = nn.CrossEntropyLoss()

    wpath = CONFIG["backbone_weights"]
    best_acc = -1.0
    for ep in range(1, epochs + 1):
        model.train()
        model.backbone.eval()   # frozen encoder: no drop-path / train-mode noise
        running = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {ep}/{epochs}")
        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * y.numel()
            pbar.set_postfix(loss=loss.item())
        val_acc = _evaluate(model, val_loader)
        print(f"epoch {ep}: train_loss={running / len(subset):.4f}  val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.head.state_dict(), wpath)   # save only the trained head
            print(f"  [save] new best val_acc={best_acc:.4f} -> {os.path.basename(wpath)}")

    print(f"done. best val_acc={best_acc:.4f}; weights at {wpath}")
    return wpath
