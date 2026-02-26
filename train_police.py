"""
train_police.py

Train a binary police vehicle classifier from a small dataset.
Uses MobileNetV2 backbone (frozen) with a trainable head, heavy
augmentation to stretch ~10 positive examples.

Expects:
    dataset/police/       — positive examples (even just ~10)
    dataset/car/          — negatives
    dataset/truck/        — negatives
    dataset/bus/          — negatives
    dataset/motorcycle/   — negatives

Usage:
    python train_police.py --dataset ./dataset
    python train_police.py --dataset ./dataset --epochs 60 --neg-cap 200
    python train_police.py --dataset ./dataset --unfreeze  # fine-tune backbone too

Output:
    police_classifier.pt  — saved model weights
    training_log.csv      — loss/accuracy per epoch
"""

import os
import random
import csv
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

class PoliceDataset(Dataset):
    """Binary dataset: police (1) vs not-police (0)."""

    def __init__(self, samples, transform=None):
        """samples: list of (path, label) where label is 0 or 1."""
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def collect_samples(dataset_dir,
                    confirmed_folder="police_confirmed",
                    unconfirmed_folder="police",
                    negative_folders=("car", "truck", "bus", "motorcycle"),
                    neg_cap=None):
    """
    Gather (path, label) pairs.

    Positives    — everything in confirmed_folder (police_confirmed/).
    Hard negatives — files in unconfirmed_folder (police/) whose filename is
                     NOT already in confirmed_folder: these are look-alike
                     vehicles that were reviewed and rejected, so they are
                     valuable hard negatives.
    Soft negatives — car/truck/bus/motorcycle, optionally capped.
    """
    positives = []
    negatives = []

    # ── Confirmed police → positives ──────────────────────────────────────
    confirmed_dir = os.path.join(dataset_dir, confirmed_folder)
    confirmed_names = set()
    if os.path.isdir(confirmed_dir):
        for f in os.listdir(confirmed_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                positives.append((os.path.join(confirmed_dir, f), 1))
                confirmed_names.add(f)

    # ── Unconfirmed police (rejected candidates) → hard negatives ─────────
    unconfirmed_dir = os.path.join(dataset_dir, unconfirmed_folder)
    hard_neg = 0
    if os.path.isdir(unconfirmed_dir):
        for f in os.listdir(unconfirmed_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png")) and f not in confirmed_names:
                negatives.append((os.path.join(unconfirmed_dir, f), 0))
                hard_neg += 1

    # ── Soft negatives (car/truck/etc.) ───────────────────────────────────
    for folder in negative_folders:
        d = os.path.join(dataset_dir, folder)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                negatives.append((os.path.join(d, f), 0))

    random.shuffle(negatives)
    if neg_cap and len(negatives) > neg_cap:
        negatives = negatives[:neg_cap]

    return positives, negatives, hard_neg


# ══════════════════════════════════════════════════════════════════════════════
# AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
# Aggressive augmentation is critical with ~10 positives. Every training
# epoch sees a different distorted version of each image.

def get_train_transform():
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=15, translate=(0.15, 0.15),
                                scale=(0.75, 1.3), shear=10),
        transforms.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.4, hue=0.08),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])


def get_val_transform():
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

def build_model(unfreeze_backbone=False):
    """
    MobileNetV2 backbone (pretrained) + small binary head.
    Backbone is frozen by default — we only train the head.
    """
    backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    if not unfreeze_backbone:
        for param in backbone.features.parameters():
            param.requires_grad = False

    in_features = backbone.classifier[1].in_features
    backbone.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 1),
    )

    return backbone


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Collect samples ───────────────────────────────────────────────────
    positives, negatives, hard_neg = collect_samples(
        args.dataset,
        confirmed_folder=args.confirmed_folder,
        unconfirmed_folder=args.unconfirmed_folder,
        negative_folders=args.neg_folders.split(","),
        neg_cap=args.neg_cap,
    )

    print(f"Positives (confirmed):          {len(positives)}")
    print(f"Hard negatives (unconfirmed):   {hard_neg}")
    print(f"Negatives total (after cap):    {len(negatives)}")

    if len(positives) == 0:
        print("No positive samples found. Check --dataset and --pos-folders.")
        return
    if len(negatives) == 0:
        print("No negative samples found. Check --neg-folders.")
        return

    # ── Split: hold out ~20% for validation (min 1 positive) ─────────────
    random.shuffle(positives)
    random.shuffle(negatives)

    n_val_pos = max(1, len(positives) // 5)
    n_val_neg = max(2, len(negatives) // 5)

    val_samples = positives[:n_val_pos] + negatives[:n_val_neg]
    train_samples = positives[n_val_pos:] + negatives[n_val_neg:]

    # If very few positives, duplicate them in training set so the sampler
    # has enough to work with
    train_pos = [s for s in train_samples if s[1] == 1]
    train_neg = [s for s in train_samples if s[1] == 0]

    if len(train_pos) < 5:
        # Repeat positives to at least 5x so augmentation has material
        repeats = max(1, 10 // len(train_pos))
        train_pos = train_pos * repeats

    train_samples = train_pos + train_neg
    random.shuffle(train_samples)

    print(f"Train: {len(train_pos)} pos + {len(train_neg)} neg = {len(train_samples)}")
    print(f"Val:   {n_val_pos} pos + {n_val_neg} neg = {len(val_samples)}")

    # ── Weighted sampler: balance positives and negatives per batch ────────
    labels = [s[1] for s in train_samples]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    weight_per_class = {0: 1.0 / n_neg, 1: 1.0 / n_pos}
    sample_weights = [weight_per_class[l] for l in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_samples),
                                     replacement=True)

    # ── Dataloaders ───────────────────────────────────────────────────────
    train_ds = PoliceDataset(train_samples, get_train_transform())
    val_ds   = PoliceDataset(val_samples,   get_val_transform())

    num_workers = 4 if torch.cuda.is_available() else 0
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                          num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=torch.cuda.is_available())

    # ── Model, loss, optimizer ────────────────────────────────────────────
    model = build_model(unfreeze_backbone=args.unfreeze).to(device)
    criterion = nn.BCEWithLogitsLoss()

    # Only optimise parameters that require grad
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\nTraining for {args.epochs} epochs (lr={args.lr}, "
          f"{'backbone unfrozen' if args.unfreeze else 'backbone frozen'})...\n")

    log = []
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # ── Train ─────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for imgs, labels in train_dl:
            imgs = imgs.to(device)
            labels = labels.float().to(device)

            logits = model(imgs).squeeze(-1)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            train_correct += (preds == labels.long()).sum().item()
            train_total += len(labels)

        scheduler.step()

        # ── Validate ──────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_tp = val_fp = val_fn = val_tn = 0

        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs = imgs.to(device)
                labels = labels.float().to(device)

                logits = model(imgs).squeeze(-1)
                loss = criterion(logits, labels)

                val_loss += loss.item() * len(labels)
                preds = (torch.sigmoid(logits) >= 0.5).long()
                val_correct += (preds == labels.long()).sum().item()
                val_total += len(labels)

                for p, t in zip(preds, labels.long()):
                    if p == 1 and t == 1: val_tp += 1
                    if p == 1 and t == 0: val_fp += 1
                    if p == 0 and t == 1: val_fn += 1
                    if p == 0 and t == 0: val_tn += 1

        t_loss = train_loss / max(1, train_total)
        t_acc  = train_correct / max(1, train_total)
        v_loss = val_loss / max(1, val_total)
        v_acc  = val_correct / max(1, val_total)
        precision = val_tp / max(1, val_tp + val_fp)
        recall    = val_tp / max(1, val_tp + val_fn)

        log.append({
            "epoch": epoch, "train_loss": round(t_loss, 4),
            "train_acc": round(t_acc, 4), "val_loss": round(v_loss, 4),
            "val_acc": round(v_acc, 4), "precision": round(precision, 4),
            "recall": round(recall, 4), "tp": val_tp, "fp": val_fp,
            "fn": val_fn, "tn": val_tn,
        })

        marker = ""
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), args.output)
            marker = " ← saved"

        if epoch % 5 == 0 or epoch == 1 or marker:
            print(f"  [{epoch:3d}/{args.epochs}]  "
                  f"train {t_loss:.4f}/{t_acc:.1%}  "
                  f"val {v_loss:.4f}/{v_acc:.1%}  "
                  f"P={precision:.2f} R={recall:.2f}  "
                  f"(tp={val_tp} fp={val_fp} fn={val_fn} tn={val_tn}){marker}")

    # ── Save final if never saved ─────────────────────────────────────
    if best_val_acc == 0.0:
        torch.save(model.state_dict(), args.output)

    # ── Write training log ────────────────────────────────────────────
    log_path = args.output.replace(".pt", "_log.csv")
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=log[0].keys())
        w.writeheader()
        w.writerows(log)

    print(f"\nModel saved to {args.output}")
    print(f"Log saved to {log_path}")
    print(f"Best val accuracy: {best_val_acc:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     type=str,   default="./dataset")
    parser.add_argument("--output",      type=str,   default="police_classifier.pt")
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch-size",  type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--neg-cap",     type=int,   default=300,
                        help="Max negatives to use (prevents extreme imbalance)")
    parser.add_argument("--confirmed-folder",   type=str, default="police_confirmed",
                        help="Folder of confirmed positives")
    parser.add_argument("--unconfirmed-folder", type=str, default="police",
                        help="Folder of candidates; those not in confirmed become hard negatives")
    parser.add_argument("--neg-folders", type=str,   default="car,truck,bus,motorcycle",
                        help="Comma-separated folder names for soft negatives")
    parser.add_argument("--unfreeze",    action="store_true",
                        help="Also fine-tune backbone (use with more data)")
    args = parser.parse_args()
    train(args)
