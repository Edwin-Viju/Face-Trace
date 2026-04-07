# FaceTrace — Face Verification System

An end-to-end face verification system implementing baseline evaluation, ArcFace-based fine-tuning, and a live interactive demo. Built as part of the AI Developer Intern Evaluation Task for **Jezt Technologies, Kerala Startup Mission, Kochi**.

---

## Results at a Glance

| Metric | Baseline | Fine-tuned | Change |
|---|---|---|---|
| ROC AUC | 0.7683 | **0.9378** | +22.1% |
| PR AUC | 0.7858 | **0.9407** | +19.7% |
| TAR @ FAR=1% | 0.1634 | **0.4988** | +205% |
| Best Threshold | 0.2905 | **0.5705** | — |

---

## Project Structure

```
face_recognition_project/
├── data/                        # Dataset (8 identity folders)
│   └── <person_name>/
│       ├── high_quality/        # Clean, controlled face photos
│       └── low_quality/         # Blurry, masked, distorted photos
├── src/
│   ├── prepare_dataset.py       # Dataset exploration and split
│   ├── baseline_eval.py         # Pretrained model evaluation
│   ├── train.py                 # ArcFace fine-tuning
│   ├── evaluate_finetuned.py    # Post fine-tuning evaluation
│   └── umap_viz.py              # Embedding visualisation
├── app/
│   └── app.py                   # FaceTrace Streamlit web app
├── checkpoints/                 # Saved model weights
│   ├── checkpoint_epoch_05.pth
│   ├── checkpoint_epoch_10.pth
│   ├── checkpoint_epoch_15.pth
│   ├── checkpoint_epoch_20.pth
│   └── best_model.pth           # Best model (epoch 20)
├── results/                     # All plots and metrics
│   ├── train_paths.csv
│   ├── eval_paths.csv
│   ├── split_log.txt
│   ├── baseline_embeddings.npy
│   ├── baseline_labels.npy
│   ├── baseline_metrics.json
│   ├── baseline_roc_curve.png
│   ├── baseline_pr_curve.png
│   ├── baseline_similarity_distribution.png
│   ├── finetuned_embeddings.npy
│   ├── finetuned_labels.npy
│   ├── finetuned_metrics.json
│   ├── training_loss_curve.png
│   ├── training_log.json
│   ├── comparison_roc_curve.png
│   ├── comparison_pr_curve.png
│   ├── comparison_similarity_distribution.png
│   ├── umap_baseline.png
│   ├── umap_finetuned.png
│   └── umap_comparison.png
├── report/
│   └── FaceTrace_Report_Edwin_Viju.pdf
└── README.md
```

---

## Setup

### Requirements

- Python 3.10+
- Windows / Linux / macOS
- CPU only (no GPU required)

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/facetrace.git
cd facetrace

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio
pip install insightface onnxruntime
pip install opencv-python
pip install scikit-learn matplotlib seaborn
pip install umap-learn streamlit
pip install pandas numpy tqdm pillow
```

---

## Reproducing Results

Run the scripts in order from the `src/` directory:

### Step 1 — Prepare dataset

```bash
cd src
python prepare_dataset.py
```

Explores the dataset, applies the split strategy, and saves `train_paths.csv` and `eval_paths.csv` to `results/`.

**Split strategy:**
- `high_quality/` → 100% training
- `low_quality/` → 80% training, 20% evaluation
- Fixed seed (42) for full reproducibility
- Zero overlap between train and eval sets

### Step 2 — Baseline evaluation

```bash
python baseline_eval.py
```

Loads pretrained InsightFace buffalo_l (ArcFace backbone), extracts 512-d embeddings from all evaluation images, computes pairwise cosine similarity, and generates ROC curve, PR curve, and similarity distribution plots.

**Detection settings:** `det_thresh=0.3`, `det_size=320x320`

### Step 3 — Fine-tuning

```bash
python train.py
```

Fine-tunes a ResNet-50 backbone (pretrained on ImageNet) using ArcFace loss on the training set. Saves checkpoints every 5 epochs and the best model to `checkpoints/best_model.pth`.

**Training time:** approximately 7 hours on CPU.

**Key hyperparameters:**

| Parameter | Value |
|---|---|
| Backbone | ResNet-50 (ImageNet pretrained) |
| Loss | ArcFace (margin=0.5, scale=32) |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR |
| Epochs | 20 |
| Batch size | 32 |
| Frozen layers | layer1, layer2 |
| Trained layers | layer3, layer4, FC |

### Step 4 — Post fine-tuning evaluation

```bash
python evaluate_finetuned.py
```

Loads `best_model.pth`, extracts embeddings using the fine-tuned model on the same evaluation set, recomputes all metrics, and generates before vs after comparison plots.

### Step 5 — UMAP visualisation

```bash
python umap_viz.py
```

Projects 512-dimensional embeddings to 2D using UMAP and generates cluster visualisation plots for both baseline and fine-tuned models.

---

## FaceTrace Demo App

```bash
cd app
streamlit run app.py
```

Opens at `http://localhost:8501`. Upload two face photos to compare similarity scores from both the baseline and fine-tuned model side by side.

**Features:**
- Side-by-side baseline vs fine-tuned comparison
- Cosine similarity score with progress bar
- Same / Different person verdict
- Adjustable threshold slider
- Aligned face crop preview

---

## Dataset

The dataset consists of 8 identities with two image quality tiers per identity:

| Person | HQ Train | LQ Train | LQ Eval | Total |
|---|---|---|---|---|
| Firoz | 758 | 987 | 197 | 1,942 |
| Ruvais | 128 | 799 | 199 | 1,126 |
| ananthu | 845 | 3,023 | 755 | 4,623 |
| mujeeb | 648 | 794 | 198 | 1,640 |
| sebin | 411 | 1,424 | 356 | 2,191 |
| shinjil | 273 | 1,230 | 307 | 1,810 |
| suresh | 838 | 5,524 | 1,381 | 7,743 |
| thomas | 567 | 673 | 168 | 1,408 |
| **Total** | **4,468** | **14,454** | **3,561** | **22,483** |

> The dataset was provided by Jezt Technologies for evaluation purposes only. It is not redistributed here.

---

## Data Integrity

- The evaluation set was never accessed during training.
- The split was applied before any model was loaded or run.
- All metrics are computed exclusively on held-out evaluation images.
- A signed declaration confirming this is included in the report.

---

## Real-world Application

FaceTrace is designed as a proof-of-concept toward solving the missing persons identification problem in India, where over 8.7 lakh people go missing annually and nearly half remain untraced. The system demonstrates that a low-resource, CPU-based face verification pipeline can meaningfully outperform pretrained baselines on challenging real-world image conditions — the same conditions present in CCTV footage, shelter photographs, and field-captured images.

---

## Submitted by

**Edwin Viju**  
B.Tech Computer Science, 2nd Year  
Amal Jyothi College of Engineering, Kottayam  
Vice Chair, IEEE SSCS Chapter  

**Submitted to:** Jezt Technologies, Kerala Startup Mission, Kochi  
**Contact:** midhun@jezt.in
