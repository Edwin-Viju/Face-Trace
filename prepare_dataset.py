import os
import random
import shutil
import csv
from pathlib import Path

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
DATA_DIR      = Path("../data")
RESULTS_DIR   = Path("../results")
LQ_EVAL_RATIO = 0.2      # 20% of LQ goes to eval
RANDOM_SEED   = 42
RESULTS_DIR.mkdir(exist_ok=True)
random.seed(RANDOM_SEED)

# ─────────────────────────────────────────
# VALID IMAGE EXTENSIONS
# ─────────────────────────────────────────
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def get_images(folder: Path):
    return [f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in IMG_EXTS]

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
persons = sorted([p for p in DATA_DIR.iterdir() if p.is_dir()])

print("=" * 55)
print("  DATASET PREPARATION")
print("=" * 55)
print(f"  Persons found : {len(persons)}")
print(f"  LQ eval split : {int(LQ_EVAL_RATIO*100)}%")
print(f"  Random seed   : {RANDOM_SEED}")
print("=" * 55)

train_records = []   # [path, person_name, quality]
eval_records  = []

for person in persons:
    hq_dir = person / "high_quality"
    lq_dir = person / "low_quality"

    hq_images = get_images(hq_dir) if hq_dir.exists() else []
    lq_images = get_images(lq_dir) if lq_dir.exists() else []

    # ── HQ: 100% goes to train ──────────────
    for img in hq_images:
        train_records.append({
            "path"   : str(img),
            "person" : person.name,
            "quality": "high"
        })

    # ── LQ: 80% train / 20% eval ───────────
    random.shuffle(lq_images)
    n_eval  = max(1, int(len(lq_images) * LQ_EVAL_RATIO))
    lq_eval  = lq_images[:n_eval]
    lq_train = lq_images[n_eval:]

    for img in lq_train:
        train_records.append({
            "path"   : str(img),
            "person" : person.name,
            "quality": "low"
        })

    for img in lq_eval:
        eval_records.append({
            "path"   : str(img),
            "person" : person.name,
            "quality": "low"
        })

    print(f"\n  {person.name}")
    print(f"    HQ → train          : {len(hq_images)}")
    print(f"    LQ → train          : {len(lq_train)}")
    print(f"    LQ → eval           : {len(lq_eval)}")
    print(f"    Total train contrib : {len(hq_images) + len(lq_train)}")

# ─────────────────────────────────────────
# SAVE CSVs
# ─────────────────────────────────────────
def save_csv(records, filepath):
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "person", "quality"])
        writer.writeheader()
        writer.writerows(records)

save_csv(train_records, RESULTS_DIR / "train_paths.csv")
save_csv(eval_records,  RESULTS_DIR / "eval_paths.csv")

# ─────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────
print("\n" + "=" * 55)
print("  FINAL SUMMARY")
print("=" * 55)
print(f"  Total TRAIN images : {len(train_records)}")
hq_train = sum(1 for r in train_records if r["quality"] == "high")
lq_train = sum(1 for r in train_records if r["quality"] == "low")
print(f"    ├── HQ images   : {hq_train}")
print(f"    └── LQ images   : {lq_train}")
print(f"  Total EVAL  images : {len(eval_records)}")
print("=" * 55)
print("\n  Saved → results/train_paths.csv")
print("  Saved → results/eval_paths.csv")

# ─────────────────────────────────────────
# SAVE SPLIT LOG FOR REPORT
# ─────────────────────────────────────────
with open(RESULTS_DIR / "split_log.txt", "w", encoding="utf-8") as f:
    f.write("DATASET SPLIT LOG\n")
    f.write("=" * 55 + "\n")
    f.write("Strategy:\n")
    f.write("  high_quality → 100% TRAIN\n")
    f.write("  low_quality  → 80% TRAIN / 20% EVAL\n")
    f.write(f"  Seed: {RANDOM_SEED}\n\n")
    f.write(f"Total persons : {len(persons)}\n")
    f.write(f"Total train   : {len(train_records)} "
            f"(HQ:{hq_train} + LQ:{lq_train})\n")
    f.write(f"Total eval    : {len(eval_records)} (LQ only)\n\n")
    f.write("Per-person breakdown:\n")
    for person in persons:
        t = [r for r in train_records if r["person"] == person.name]
        e = [r for r in eval_records  if r["person"] == person.name]
        hq = sum(1 for r in t if r["quality"] == "high")
        lq = sum(1 for r in t if r["quality"] == "low")
        f.write(f"  {person.name} | train: {len(t)} "
                f"(HQ:{hq} LQ:{lq}) | eval: {len(e)}\n")

print("  Saved → results/split_log.txt")
print("\n  Day 1 complete.")