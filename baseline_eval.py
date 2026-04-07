import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # needed for Windows without display issues

from pathlib import Path
from itertools import combinations
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import normalize
import insightface
from insightface.app import FaceAnalysis
import cv2
import random

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
RESULTS_DIR  = Path("../results")
EVAL_CSV     = RESULTS_DIR / "eval_paths.csv"
RANDOM_SEED  = 42
MAX_PAIRS    = 5000   # cap pairs to keep it fast on CPU
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("=" * 55)
print("  DAY 2 — BASELINE EVALUATION")
print("=" * 55)

# ─────────────────────────────────────────
# STEP 1 — Load InsightFace
# ─────────────────────────────────────────
# FaceAnalysis loads the pretrained ArcFace model
# ctx_id = -1 means CPU (0 would mean GPU)
# det_size is the image size used for face detection
print("\n[1/5] Loading InsightFace pretrained model...")
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(320, 320))
app.models['detection'].det_thresh = 0.3
print("      Model loaded successfully.")

# ─────────────────────────────────────────
# STEP 2 — Read eval CSV
# ─────────────────────────────────────────
print("\n[2/5] Reading eval dataset...")
eval_records = []
with open(EVAL_CSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        eval_records.append(row)

print(f"      Total eval images: {len(eval_records)}")

# ─────────────────────────────────────────
# STEP 3 — Extract embeddings
# ─────────────────────────────────────────
# For each eval image:
#   - Load image with OpenCV
#   - Run InsightFace detection + embedding extraction
#   - If a face is detected, save the 512-d embedding
#   - If no face detected, skip and log it
print("\n[3/5] Extracting embeddings from eval images...")
print("      (This will take a while on CPU — go grab a chai)")

embeddings  = []   # list of 512-d numpy arrays
labels      = []   # person name for each embedding
skipped     = 0
processed   = 0

for i, record in enumerate(eval_records):
    img_path = record["path"]
    person   = record["person"]

    # Progress update every 100 images
    if i % 100 == 0:
        print(f"      Processing {i}/{len(eval_records)}...")

    img = cv2.imread(img_path)
    if img is None:
        skipped += 1
        continue

    # InsightFace expects BGR format (which OpenCV uses by default)
    faces = app.get(img)

    if len(faces) == 0:
        # No face detected in this image — skip it
        skipped += 1
        continue

    # If multiple faces detected, take the largest one
    # (largest bounding box area = most prominent face)
    face = max(faces, key=lambda f: (
        (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    ))

    embedding = face.normed_embedding   # already L2-normalized 512-d vector
    embeddings.append(embedding)
    labels.append(person)
    processed += 1

print(f"\n      Done.")
print(f"      Embeddings extracted : {processed}")
print(f"      Images skipped       : {skipped} (no face detected)")

embeddings = np.array(embeddings)   # shape: (N, 512)
labels     = np.array(labels)

# Save for later use
np.save(RESULTS_DIR / "baseline_embeddings.npy", embeddings)
np.save(RESULTS_DIR / "baseline_labels.npy",     labels)
print(f"      Saved baseline_embeddings.npy and baseline_labels.npy")

# ─────────────────────────────────────────
# STEP 4 — Build pairs + compute similarity
# ─────────────────────────────────────────
# Positive pairs: two images of the SAME person
# Negative pairs: two images of DIFFERENT people
# We compute cosine similarity for each pair
# cosine similarity = dot product of two L2-normalized vectors
print("\n[4/5] Building pairs and computing similarity scores...")

persons_unique = list(set(labels))
indices_by_person = {p: np.where(labels == p)[0].tolist()
                     for p in persons_unique}

positive_pairs = []   # (idx1, idx2) same person
negative_pairs = []   # (idx1, idx2) different people

# Build positive pairs — all combinations within each person
for person in persons_unique:
    idxs = indices_by_person[person]
    pairs = list(combinations(idxs, 2))
    positive_pairs.extend(pairs)

# Build negative pairs — random cross-person pairs
for _ in range(len(positive_pairs) * 3):
    p1, p2 = random.sample(persons_unique, 2)
    i1 = random.choice(indices_by_person[p1])
    i2 = random.choice(indices_by_person[p2])
    negative_pairs.append((i1, i2))

# Cap pairs to MAX_PAIRS each to keep computation fast
random.shuffle(positive_pairs)
random.shuffle(negative_pairs)
positive_pairs = positive_pairs[:MAX_PAIRS]
negative_pairs = negative_pairs[:MAX_PAIRS]

print(f"      Positive pairs (same person)     : {len(positive_pairs)}")
print(f"      Negative pairs (different person): {len(negative_pairs)}")

# Compute cosine similarity
# Since embeddings are already L2-normalized,
# cosine similarity = simple dot product
def compute_similarities(pairs, embeddings):
    scores = []
    for i, j in pairs:
        score = np.dot(embeddings[i], embeddings[j])
        scores.append(float(score))
    return scores

pos_scores = compute_similarities(positive_pairs, embeddings)
neg_scores = compute_similarities(negative_pairs, embeddings)

all_scores = pos_scores + neg_scores
all_labels = [1] * len(pos_scores) + [0] * len(neg_scores)
# 1 = same person (positive), 0 = different person (negative)

# ─────────────────────────────────────────
# STEP 5 — Plot and save metrics
# ─────────────────────────────────────────
print("\n[5/5] Plotting results and saving metrics...")

# ── Plot 1: Similarity Score Distribution ──
plt.figure(figsize=(8, 4))
plt.hist(pos_scores, bins=60, alpha=0.6, color="#1D9E75",
         label="Same person (positive)")
plt.hist(neg_scores, bins=60, alpha=0.6, color="#D85A30",
         label="Different person (negative)")
plt.xlabel("Cosine Similarity Score")
plt.ylabel("Number of Pairs")
plt.title("Baseline — Similarity Score Distribution")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "baseline_similarity_distribution.png", dpi=150)
plt.close()
print("      Saved baseline_similarity_distribution.png")

# ── Plot 2: ROC Curve ──
fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="#378ADD", lw=2,
         label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color="#B4B2A9", lw=1, linestyle="--",
         label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Baseline — ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "baseline_roc_curve.png", dpi=150)
plt.close()
print("      Saved baseline_roc_curve.png")

# ── Plot 3: Precision-Recall Curve ──
precision, recall, _ = precision_recall_curve(all_labels, all_scores)
pr_auc = auc(recall, precision)

plt.figure(figsize=(6, 5))
plt.plot(recall, precision, color="#7F77DD", lw=2,
         label=f"PR curve (AUC = {pr_auc:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Baseline — Precision-Recall Curve")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "baseline_pr_curve.png", dpi=150)
plt.close()
print("      Saved baseline_pr_curve.png")

# ── Save metrics to JSON ──
# Find best threshold — the one that maximizes TPR - FPR
best_idx       = np.argmax(tpr - fpr)
best_threshold = float(thresholds[best_idx])
best_tpr       = float(tpr[best_idx])
best_fpr       = float(fpr[best_idx])

# TAR @ FAR=1% — how many same-person pairs accepted when only 1% false accepts
far_1_idx   = np.argmin(np.abs(fpr - 0.01))
tar_at_far1 = float(tpr[far_1_idx])

metrics = {
    "model"              : "InsightFace buffalo_l (pretrained, no fine-tuning)",
    "total_embeddings"   : int(processed),
    "skipped_images"     : int(skipped),
    "positive_pairs"     : len(positive_pairs),
    "negative_pairs"     : len(negative_pairs),
    "roc_auc"            : round(roc_auc, 4),
    "pr_auc"             : round(pr_auc, 4),
    "best_threshold"     : round(best_threshold, 4),
    "tar_at_far_1pct"    : round(tar_at_far1, 4),
}

with open(RESULTS_DIR / "baseline_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("      Saved baseline_metrics.json")

# ── Print final summary ──
print("\n" + "=" * 55)
print("  BASELINE RESULTS SUMMARY")
print("=" * 55)
print(f"  Embeddings extracted : {processed}")
print(f"  Images skipped       : {skipped}")
print(f"  ROC AUC              : {roc_auc:.4f}")
print(f"  PR  AUC              : {pr_auc:.4f}")
print(f"  Best threshold       : {best_threshold:.4f}")
print(f"  TAR @ FAR=1%         : {tar_at_far1:.4f}")
print("=" * 55)
print("\n  Day 2 complete.")