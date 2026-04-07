import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as transforms
import cv2
import random
from PIL import Image
from insightface.app import FaceAnalysis

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
RESULTS_DIR   = Path("../results")
CHECKPOINTS   = Path("../checkpoints")
EVAL_CSV      = RESULTS_DIR / "eval_paths.csv"
RANDOM_SEED   = 42
MAX_PAIRS     = 5000
IMG_SIZE      = 112

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

print("=" * 55)
print("  DAY 4 — POST FINE-TUNING EVALUATION")
print("=" * 55)

# ─────────────────────────────────────────
# STEP 1 — Load fine-tuned model
# ─────────────────────────────────────────
# We load the same ResNet50 architecture
# then fill it with the weights we trained
print("\n[1/5] Loading fine-tuned model...")

checkpoint = torch.load(
    CHECKPOINTS / "best_model.pth",
    map_location="cpu"
)

# Rebuild the exact same architecture as train.py
backbone = tv_models.resnet50(weights=None)
backbone.fc = nn.Sequential(
    nn.Linear(backbone.fc.in_features, 512),
    nn.BatchNorm1d(512)
)

backbone.load_state_dict(checkpoint["model_state"])
backbone.eval()  # evaluation mode — disables dropout, batchnorm uses running stats

label_map  = checkpoint["label_map"]
best_epoch = checkpoint["epoch"]
best_loss  = checkpoint["loss"]

print(f"      Loaded checkpoint from epoch {best_epoch}")
print(f"      Training loss at checkpoint : {best_loss:.4f}")
print(f"      Identities : {list(label_map.keys())}")

# ─────────────────────────────────────────
# STEP 2 — Load InsightFace for face detection only
# ─────────────────────────────────────────
# We still use InsightFace to DETECT and ALIGN faces
# But embedding extraction is done by our fine-tuned model
print("\n[2/5] Loading InsightFace detector...")
app = FaceAnalysis(name="buffalo_l",
                   providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(320, 320))
app.models['detection'].det_thresh = 0.3
print("      Detector ready.")

# ─────────────────────────────────────────
# Transform for fine-tuned model
# ─────────────────────────────────────────
# Same normalization as training — very important
# If you normalize differently at eval, embeddings will be wrong
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# ─────────────────────────────────────────
# STEP 3 — Extract embeddings using fine-tuned model
# ─────────────────────────────────────────
print("\n[3/5] Extracting embeddings with fine-tuned model...")
print("      (Using same eval images as Day 2 baseline)")

eval_records = []
with open(EVAL_CSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        eval_records.append(row)

print(f"      Total eval images: {len(eval_records)}")

embeddings = []
labels     = []
skipped    = 0
processed  = 0

def get_embedding(img_bgr, app, backbone, transform):
    """
    Detect face with InsightFace → crop → 
    pass through fine-tuned ResNet50 → return 512-d embedding
    """
    faces = app.get(img_bgr)
    if len(faces) == 0:
        return None

    # Take largest face
    face = max(faces, key=lambda f: (
        (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    ))

    # Get aligned face crop from InsightFace
    # kps = 5 facial keypoints (eyes, nose, mouth corners)
    # InsightFace uses these to align the face to standard position
    from insightface.utils import face_align
    aligned = face_align.norm_crop(img_bgr, landmark=face.kps)
    # aligned is a 112x112 BGR image, standardized face position

    # Convert BGR → RGB → PIL → tensor
    aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    pil_img     = Image.fromarray(aligned_rgb)
    tensor      = transform(pil_img).unsqueeze(0)
    # unsqueeze(0) adds batch dimension: (3,112,112) → (1,3,112,112)

    with torch.no_grad():
        # no_grad = don't compute gradients (saves memory at eval time)
        embedding = backbone(tensor)
        # L2 normalize so cosine similarity = dot product
        embedding = torch.nn.functional.normalize(embedding, dim=1)

    return embedding.squeeze(0).numpy()  # back to (512,) numpy array

for i, record in enumerate(eval_records):
    if i % 100 == 0:
        print(f"      Processing {i}/{len(eval_records)}...")

    img = cv2.imread(record["path"])
    if img is None:
        skipped += 1
        continue

    emb = get_embedding(img, app, backbone, eval_transform)
    if emb is None:
        skipped += 1
        continue

    embeddings.append(emb)
    labels.append(record["person"])
    processed += 1

print(f"\n      Embeddings extracted : {processed}")
print(f"      Images skipped       : {skipped}")

embeddings = np.array(embeddings)
labels     = np.array(labels)

np.save(RESULTS_DIR / "finetuned_embeddings.npy", embeddings)
np.save(RESULTS_DIR / "finetuned_labels.npy",     labels)
print(f"      Saved finetuned_embeddings.npy")

# ─────────────────────────────────────────
# STEP 4 — Build same pairs + compute similarity
# ─────────────────────────────────────────
print("\n[4/5] Computing similarity scores...")

persons_unique     = list(set(labels))
indices_by_person  = {p: np.where(labels == p)[0].tolist()
                      for p in persons_unique}

positive_pairs = []
negative_pairs = []

for person in persons_unique:
    idxs  = indices_by_person[person]
    pairs = list(combinations(idxs, 2))
    positive_pairs.extend(pairs)

for _ in range(len(positive_pairs) * 3):
    p1, p2 = random.sample(persons_unique, 2)
    i1 = random.choice(indices_by_person[p1])
    i2 = random.choice(indices_by_person[p2])
    negative_pairs.append((i1, i2))

random.shuffle(positive_pairs)
random.shuffle(negative_pairs)
positive_pairs = positive_pairs[:MAX_PAIRS]
negative_pairs = negative_pairs[:MAX_PAIRS]

print(f"      Positive pairs : {len(positive_pairs)}")
print(f"      Negative pairs : {len(negative_pairs)}")

def compute_similarities(pairs, embeddings):
    return [float(np.dot(embeddings[i], embeddings[j]))
            for i, j in pairs]

pos_scores = compute_similarities(positive_pairs, embeddings)
neg_scores = compute_similarities(negative_pairs, embeddings)

all_scores = pos_scores + neg_scores
all_labels = [1] * len(pos_scores) + [0] * len(neg_scores)

# ─────────────────────────────────────────
# STEP 5 — Plot before vs after comparison
# ─────────────────────────────────────────
print("\n[5/5] Generating before vs after comparison plots...")

# Load baseline scores for comparison
baseline_metrics = json.load(
    open(RESULTS_DIR / "baseline_metrics.json")
)

fpr_ft, tpr_ft, thresholds_ft = roc_curve(all_labels, all_scores)
roc_auc_ft = auc(fpr_ft, tpr_ft)

precision_ft, recall_ft, _ = precision_recall_curve(all_labels, all_scores)
pr_auc_ft = auc(recall_ft, precision_ft)

best_idx_ft    = np.argmax(tpr_ft - fpr_ft)
best_thresh_ft = float(thresholds_ft[best_idx_ft])
far_1_idx_ft   = np.argmin(np.abs(fpr_ft - 0.01))
tar_at_far1_ft = float(tpr_ft[far_1_idx_ft])

# ── Load baseline embeddings for distribution plot ──
baseline_emb = np.load(RESULTS_DIR / "baseline_embeddings.npy")
baseline_lbl = np.load(RESULTS_DIR / "baseline_labels.npy")

b_persons = list(set(baseline_lbl))
b_indices  = {p: np.where(baseline_lbl == p)[0].tolist()
              for p in b_persons}

b_pos, b_neg = [], []
for person in b_persons:
    idxs  = b_indices[person]
    pairs = list(combinations(idxs, 2))
    b_pos.extend(pairs)
for _ in range(len(b_pos)):
    p1, p2 = random.sample(b_persons, 2)
    b_neg.append((
        random.choice(b_indices[p1]),
        random.choice(b_indices[p2])
    ))

random.shuffle(b_pos); random.shuffle(b_neg)
b_pos = b_pos[:MAX_PAIRS]; b_neg = b_neg[:MAX_PAIRS]
b_pos_scores = compute_similarities(b_pos, baseline_emb)
b_neg_scores = compute_similarities(b_neg, baseline_emb)

# ── Plot 1: Similarity Distribution — Before vs After ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(b_pos_scores, bins=60, alpha=0.6,
             color="#1D9E75", label="Same person")
axes[0].hist(b_neg_scores, bins=60, alpha=0.6,
             color="#D85A30", label="Different person")
axes[0].set_title("Baseline (pretrained)")
axes[0].set_xlabel("Cosine Similarity")
axes[0].set_ylabel("Number of Pairs")
axes[0].legend()

axes[1].hist(pos_scores, bins=60, alpha=0.6,
             color="#1D9E75", label="Same person")
axes[1].hist(neg_scores, bins=60, alpha=0.6,
             color="#D85A30", label="Different person")
axes[1].set_title("After fine-tuning")
axes[1].set_xlabel("Cosine Similarity")
axes[1].set_ylabel("Number of Pairs")
axes[1].legend()

fig.suptitle("Similarity Distribution — Before vs After Fine-tuning",
             fontsize=13)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "comparison_similarity_distribution.png",
            dpi=150)
plt.close()
print("      Saved comparison_similarity_distribution.png")

# ── Plot 2: ROC Curve — Before vs After ──
baseline_all_lbl = [1]*len(b_pos_scores) + [0]*len(b_neg_scores)
baseline_all_scr = b_pos_scores + b_neg_scores
fpr_bl, tpr_bl, _ = roc_curve(baseline_all_lbl, baseline_all_scr)
auc_bl = auc(fpr_bl, tpr_bl)

plt.figure(figsize=(7, 6))
plt.plot(fpr_bl, tpr_bl, color="#B4B2A9", lw=2,
         label=f"Baseline AUC = {auc_bl:.4f}", linestyle="--")
plt.plot(fpr_ft, tpr_ft, color="#378ADD", lw=2,
         label=f"Fine-tuned AUC = {roc_auc_ft:.4f}")
plt.plot([0,1],[0,1], color="#D3D1C7", lw=1, linestyle=":")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Before vs After Fine-tuning")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "comparison_roc_curve.png", dpi=150)
plt.close()
print("      Saved comparison_roc_curve.png")

# ── Plot 3: PR Curve — Before vs After ──
prec_bl, rec_bl, _ = precision_recall_curve(
    baseline_all_lbl, baseline_all_scr
)
pr_auc_bl = auc(rec_bl, prec_bl)

plt.figure(figsize=(7, 6))
plt.plot(rec_bl, prec_bl, color="#B4B2A9", lw=2,
         label=f"Baseline AUC = {pr_auc_bl:.4f}", linestyle="--")
plt.plot(recall_ft, precision_ft, color="#7F77DD", lw=2,
         label=f"Fine-tuned AUC = {pr_auc_ft:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve — Before vs After Fine-tuning")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "comparison_pr_curve.png", dpi=150)
plt.close()
print("      Saved comparison_pr_curve.png")

# ── Save final metrics ──
finetuned_metrics = {
    "model"           : "ResNet50 fine-tuned with ArcFace loss",
    "checkpoint_epoch": best_epoch,
    "checkpoint_loss" : best_loss,
    "embeddings"      : int(processed),
    "skipped"         : int(skipped),
    "positive_pairs"  : len(positive_pairs),
    "negative_pairs"  : len(negative_pairs),
    "roc_auc"         : round(roc_auc_ft, 4),
    "pr_auc"          : round(pr_auc_ft, 4),
    "best_threshold"  : round(best_thresh_ft, 4),
    "tar_at_far_1pct" : round(tar_at_far1_ft, 4),
}

with open(RESULTS_DIR / "finetuned_metrics.json", "w") as f:
    json.dump(finetuned_metrics, f, indent=2)

# ── Print final comparison ──
print("\n" + "=" * 55)
print("  BEFORE vs AFTER COMPARISON")
print("=" * 55)
print(f"  {'Metric':<25} {'Baseline':>10} {'Fine-tuned':>12}")
print(f"  {'-'*47}")
print(f"  {'ROC AUC':<25} "
      f"{baseline_metrics['roc_auc']:>10.4f} "
      f"{roc_auc_ft:>12.4f}")
print(f"  {'PR AUC':<25} "
      f"{baseline_metrics['pr_auc']:>10.4f} "
      f"{pr_auc_ft:>12.4f}")
print(f"  {'Best Threshold':<25} "
      f"{baseline_metrics['best_threshold']:>10.4f} "
      f"{best_thresh_ft:>12.4f}")
print(f"  {'TAR @ FAR=1%':<25} "
      f"{baseline_metrics['tar_at_far_1pct']:>10.4f} "
      f"{tar_at_far1_ft:>12.4f}")
print(f"  {'Embeddings extracted':<25} "
      f"{baseline_metrics['total_embeddings']:>10} "
      f"{processed:>12}")
print("=" * 55)
print("\n  Day 4 complete.")