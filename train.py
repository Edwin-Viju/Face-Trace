import csv
import json
import time
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import math

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
RESULTS_DIR   = Path("../results")
CHECKPOINTS   = Path("../checkpoints")
TRAIN_CSV     = RESULTS_DIR / "train_paths.csv"
CHECKPOINTS.mkdir(exist_ok=True)

EPOCHS        = 20
BATCH_SIZE    = 32
LR            = 1e-4
RANDOM_SEED   = 42
NUM_WORKERS   = 0       # 0 = main thread only (required on Windows)
IMG_SIZE      = 112     # ArcFace standard input size

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

print("=" * 55)
print("  DAY 3 — FINE TUNING")
print("=" * 55)

# ─────────────────────────────────────────
# STEP 1 — Read train CSV + build label map
# ─────────────────────────────────────────
# We need to convert person names to integers
# because PyTorch works with numbers not strings
# ananthu=0, Firoz=1, mujeeb=2, etc.
print("\n[1/6] Loading training data...")

train_records = []
with open(TRAIN_CSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        train_records.append(row)

# Build label map: person name → integer
persons = sorted(list(set(r["person"] for r in train_records)))
label_map = {name: idx for idx, name in enumerate(persons)}
NUM_CLASSES = len(persons)

print(f"      Total train images : {len(train_records)}")
print(f"      Number of classes  : {NUM_CLASSES}")
print(f"      Label map          : {label_map}")

# Check class distribution
counts = Counter(r["person"] for r in train_records)
print(f"\n      Images per person:")
for p, c in sorted(counts.items()):
    print(f"        {p:10s} : {c}")

# ─────────────────────────────────────────
# STEP 2 — Dataset class
# ─────────────────────────────────────────
# PyTorch needs a Dataset class that knows:
#   __len__  → how many images total
#   __getitem__ → how to load image number i
class FaceDataset(Dataset):
    def __init__(self, records, label_map, transform=None):
        self.records   = records
        self.label_map = label_map
        self.transform = transform

        # Filter out unreadable files upfront
        valid = []
        for r in self.records:
            if Path(r["path"]).exists():
                valid.append(r)
        self.records = valid
        print(f"      Valid images confirmed: {len(self.records)}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        label  = self.label_map[record["person"]]

        # Load image with PIL (works better with torchvision transforms)
        try:
            img = Image.open(record["path"]).convert("RGB")
        except Exception:
            # If image is corrupt, return a blank image
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))

        if self.transform:
            img = self.transform(img)

        return img, label

# ─────────────────────────────────────────
# STEP 3 — Transforms (augmentation)
# ─────────────────────────────────────────
# Augmentation = artificially creating variations of training images
# This prevents overfitting — model doesn't just memorize images
# it learns to recognize faces despite small changes
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    # randomly flip image left-right 50% of the time
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    # randomly change brightness/contrast to simulate lighting changes
    transforms.RandomRotation(degrees=10),
    # randomly rotate up to 10 degrees
    transforms.ToTensor(),
    # convert PIL image to PyTorch tensor (0-255 → 0.0-1.0)
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # normalize to [-1, 1] range — ArcFace expects this
])

# ─────────────────────────────────────────
# STEP 4 — Handle class imbalance
# ─────────────────────────────────────────
# Suresh has 6362 images, Ruvais has 927
# Without balancing, model will be biased toward suresh
# WeightedRandomSampler gives rarer classes more chances to be sampled
print("\n[2/6] Building dataset and handling class imbalance...")

dataset = FaceDataset(train_records, label_map, transform=train_transform)

# Compute weight for each sample
# weight = 1 / count of that person's images
# rarer person → higher weight → sampled more often
class_counts = [counts[p] for p in persons]
class_weights = [1.0 / c for c in class_counts]
sample_weights = [class_weights[label_map[r["person"]]]
                  for r in dataset.records]

sampler = torch.utils.data.WeightedRandomSampler(
    weights     = sample_weights,
    num_samples = len(sample_weights),
    replacement = True
)

loader = DataLoader(
    dataset,
    batch_size  = BATCH_SIZE,
    sampler     = sampler,
    num_workers = NUM_WORKERS
)

print(f"      Dataset size     : {len(dataset)}")
print(f"      Batches per epoch: {len(loader)}")

# ─────────────────────────────────────────
# STEP 5 — Load InsightFace backbone
# ─────────────────────────────────────────
# We load the iresnet50 backbone from InsightFace
# This is the feature extractor — outputs 512-d embeddings
# We freeze early layers and only train the last 2 blocks
print("\n[3/6] Loading InsightFace backbone...")

from insightface.model_zoo import model_zoo
from insightface.app import FaceAnalysis
import insightface

# Load pretrained model to extract backbone weights
app = FaceAnalysis(name="buffalo_l",
                   providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(320, 320))

# Get the recognition model (ArcFace backbone)
rec_model = app.models['recognition']

# ─────────────────────────────────────────
# ArcFace Loss Implementation
# ─────────────────────────────────────────
class ArcFaceLoss(nn.Module):
    """
    ArcFace loss adds an angular margin between classes.
    This makes embeddings of different people more separated.
    margin=0.5 means 0.5 radians extra angle between classes
    scale=32 controls how sharp the decision boundary is
    (we use 32 not 64 because our dataset is small)
    """
    def __init__(self, embedding_size, num_classes, margin=0.5, scale=32):
        super().__init__()
        self.scale  = scale
        self.margin = margin
        # Learnable weight matrix — one vector per class
        self.weight = nn.Parameter(
            torch.FloatTensor(num_classes, embedding_size)
        )
        nn.init.xavier_uniform_(self.weight)
        self.cos_m  = math.cos(margin)
        self.sin_m  = math.sin(margin)
        self.th     = math.cos(math.pi - margin)
        self.mm     = math.sin(math.pi - margin) * margin

    def forward(self, embeddings, labels):
        # Normalize embeddings and weights
        cos_theta = torch.nn.functional.linear(
            torch.nn.functional.normalize(embeddings),
            torch.nn.functional.normalize(self.weight)
        )
        cos_theta = cos_theta.clamp(-1, 1)
        sin_theta = torch.sqrt(1.0 - cos_theta ** 2)

        # Add margin to the correct class angle
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        cos_theta_m = torch.where(
            cos_theta > self.th,
            cos_theta_m,
            cos_theta - self.mm
        )

        # One-hot encode labels and apply margin
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * cos_theta_m) + ((1.0 - one_hot) * cos_theta)
        output *= self.scale

        return nn.CrossEntropyLoss()(output, labels.long())

# ─────────────────────────────────────────
# Simple CNN Backbone (reliable on CPU)
# ─────────────────────────────────────────
# Instead of extracting InsightFace weights (complex),
# we use a pretrained ResNet50 from torchvision
# Then replace its final layer with a 512-d embedding layer
# This is standard practice for fine-tuning face recognition
import torchvision.models as tv_models

print("      Loading ResNet50 backbone (pretrained on ImageNet)...")
backbone = tv_models.resnet50(
    weights=tv_models.ResNet50_Weights.IMAGENET1K_V1
)

# Replace final classification layer with 512-d embedding layer
backbone.fc = nn.Sequential(
    nn.Linear(backbone.fc.in_features, 512),
    nn.BatchNorm1d(512)
)

# Freeze all layers except last 2 blocks + new FC
for name, param in backbone.named_parameters():
    param.requires_grad = False

# Unfreeze layer3, layer4 and fc
for name, param in backbone.named_parameters():
    if any(x in name for x in ["layer3", "layer4", "fc"]):
        param.requires_grad = True

trainable = sum(p.numel() for p in backbone.parameters()
                if p.requires_grad)
total     = sum(p.numel() for p in backbone.parameters())
print(f"      Trainable params : {trainable:,} / {total:,}")

device = torch.device("cpu")
backbone = backbone.to(device)

arcface_loss = ArcFaceLoss(
    embedding_size = 512,
    num_classes    = NUM_CLASSES,
    margin         = 0.5,
    scale          = 32
).to(device)

# ─────────────────────────────────────────
# STEP 6 — Training loop
# ─────────────────────────────────────────
print("\n[4/6] Starting training...")
print(f"      Epochs     : {EPOCHS}")
print(f"      Batch size : {BATCH_SIZE}")
print(f"      LR         : {LR}")
print(f"      Device     : CPU")
print()

optimizer = optim.AdamW(
    list(backbone.parameters()) +
    list(arcface_loss.parameters()),
    lr=LR, weight_decay=1e-4
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS
)

loss_history = []
best_loss    = float("inf")
best_epoch   = 0

for epoch in range(1, EPOCHS + 1):
    backbone.train()
    epoch_loss  = 0.0
    batch_count = 0
    start_time  = time.time()

    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass — get 512-d embeddings
        embeddings = backbone(imgs)

        # Compute ArcFace loss
        loss = arcface_loss(embeddings, labels)

        # Backward pass — compute gradients
        loss.backward()

        # Update weights
        optimizer.step()

        epoch_loss  += loss.item()
        batch_count += 1

    scheduler.step()

    avg_loss   = epoch_loss / batch_count
    elapsed    = time.time() - start_time
    loss_history.append(avg_loss)

    print(f"  Epoch [{epoch:02d}/{EPOCHS}] "
          f"Loss: {avg_loss:.4f}  "
          f"Time: {elapsed:.1f}s  "
          f"LR: {scheduler.get_last_lr()[0]:.6f}")

    # Save checkpoint every 5 epochs
    if epoch % 5 == 0:
        ckpt_path = CHECKPOINTS / f"checkpoint_epoch_{epoch:02d}.pth"
        torch.save({
            "epoch"     : epoch,
            "model_state": backbone.state_dict(),
            "optimizer" : optimizer.state_dict(),
            "loss"      : avg_loss
        }, ckpt_path)
        print(f"             Checkpoint saved → {ckpt_path.name}")

    # Save best model
    if avg_loss < best_loss:
        best_loss  = avg_loss
        best_epoch = epoch
        torch.save({
            "epoch"      : epoch,
            "model_state": backbone.state_dict(),
            "loss"       : avg_loss,
            "label_map"  : label_map
        }, CHECKPOINTS / "best_model.pth")

print(f"\n  Best model: Epoch {best_epoch} with loss {best_loss:.4f}")
print(f"  Saved → checkpoints/best_model.pth")

# ─────────────────────────────────────────
# STEP 7 — Plot training loss curve
# ─────────────────────────────────────────
print("\n[5/6] Plotting training loss curve...")

plt.figure(figsize=(8, 4))
plt.plot(range(1, EPOCHS + 1), loss_history,
         color="#378ADD", lw=2, marker="o", markersize=4)
plt.xlabel("Epoch")
plt.ylabel("ArcFace Loss")
plt.title("Training Loss Curve")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "training_loss_curve.png", dpi=150)
plt.close()
print("      Saved → results/training_loss_curve.png")

# ─────────────────────────────────────────
# STEP 8 — Save training log
# ─────────────────────────────────────────
print("\n[6/6] Saving training log...")

training_log = {
    "epochs"        : EPOCHS,
    "batch_size"    : BATCH_SIZE,
    "learning_rate" : LR,
    "optimizer"     : "AdamW",
    "scheduler"     : "CosineAnnealingLR",
    "loss_function" : "ArcFace (margin=0.5, scale=32)",
    "backbone"      : "ResNet50 pretrained ImageNet",
    "frozen_layers" : "layer1, layer2",
    "trained_layers": "layer3, layer4, fc",
    "augmentation"  : "HorizontalFlip, ColorJitter, Rotation(10deg)",
    "best_epoch"    : best_epoch,
    "best_loss"     : round(best_loss, 4),
    "loss_history"  : [round(l, 4) for l in loss_history]
}

with open(RESULTS_DIR / "training_log.json", "w") as f:
    json.dump(training_log, f, indent=2)

print("      Saved → results/training_log.json")

print("\n" + "=" * 55)
print("  TRAINING COMPLETE")
print("=" * 55)
print(f"  Best epoch : {best_epoch}")
print(f"  Best loss  : {best_loss:.4f}")
print(f"  Checkpoints saved in : checkpoints/")
print("=" * 55)
print("\n  Day 3 complete.")