import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import umap

RESULTS_DIR = Path("../results")

print("=" * 55)
print("  UMAP VISUALIZATION")
print("=" * 55)

# ── 8 distinct colors, one per person ──
COLORS = [
    "#378ADD", "#1D9E75", "#D85A30", "#7F77DD",
    "#BA7517", "#D4537E", "#639922", "#888780"
]

def plot_umap(embeddings, labels, title, filename):
    print(f"\n  Generating UMAP for: {title}")
    print(f"  Embeddings shape: {embeddings.shape}")

    # Fit UMAP — reduces 512-d to 2-d
    # n_neighbors: how many nearby points to consider
    # min_dist: how tightly to pack clusters
    reducer = umap.UMAP(
        n_neighbors = 15,
        min_dist    = 0.1,
        n_components= 2,
        random_state= 42
    )
    embedding_2d = reducer.fit_transform(embeddings)

    # Get unique persons and assign colors
    persons = sorted(list(set(labels)))
    color_map = {p: COLORS[i] for i, p in enumerate(persons)}

    plt.figure(figsize=(10, 8))

    for person in persons:
        mask = labels == person
        plt.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            c      = color_map[person],
            label  = person,
            alpha  = 0.6,
            s      = 25,        # dot size
            edgecolors = 'none'
        )

    plt.title(title, fontsize=14, fontweight='normal')
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(loc="best", fontsize=9, markerscale=1.5)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  Saved → results/{filename}")

# ── Load baseline embeddings ──
print("\n[1/3] Loading embeddings...")
baseline_emb = np.load(RESULTS_DIR / "baseline_embeddings.npy")
baseline_lbl = np.load(RESULTS_DIR / "baseline_labels.npy")
finetuned_emb = np.load(RESULTS_DIR / "finetuned_embeddings.npy")
finetuned_lbl = np.load(RESULTS_DIR / "finetuned_labels.npy")

print(f"  Baseline  embeddings : {baseline_emb.shape}")
print(f"  Fine-tuned embeddings: {finetuned_emb.shape}")

# ── Plot baseline UMAP ──
print("\n[2/3] Plotting baseline UMAP...")
plot_umap(
    baseline_emb,
    baseline_lbl,
    "Baseline — Embedding Space (Pretrained InsightFace)",
    "umap_baseline.png"
)

# ── Plot fine-tuned UMAP ──
print("\n[3/3] Plotting fine-tuned UMAP...")
plot_umap(
    finetuned_emb,
    finetuned_lbl,
    "After Fine-tuning — Embedding Space (ResNet50 + ArcFace)",
    "umap_finetuned.png"
)

# ── Side by side comparison ──
print("\n  Generating side-by-side comparison...")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
persons = sorted(list(set(baseline_lbl)))
color_map = {p: COLORS[i] for i, p in enumerate(persons)}

for ax, emb, lbl, title in [
    (axes[0], baseline_emb,  baseline_lbl,
     "Baseline (Pretrained InsightFace)"),
    (axes[1], finetuned_emb, finetuned_lbl,
     "After Fine-tuning (ResNet50 + ArcFace)")
]:
    reducer = umap.UMAP(
        n_neighbors  = 15,
        min_dist     = 0.1,
        n_components = 2,
        random_state = 42
    )
    emb_2d = reducer.fit_transform(emb)

    for person in persons:
        mask = lbl == person
        ax.scatter(
            emb_2d[mask, 0],
            emb_2d[mask, 1],
            c          = color_map[person],
            label      = person,
            alpha      = 0.6,
            s          = 25,
            edgecolors = 'none'
        )

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.legend(fontsize=8, markerscale=1.5)

fig.suptitle(
    "Embedding Space — Before vs After Fine-tuning",
    fontsize=14
)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "umap_comparison.png", dpi=150)
plt.close()
print("  Saved → results/umap_comparison.png")

print("\n" + "=" * 55)
print("  UMAP complete.")
print("  Check results/ for:")
print("    umap_baseline.png")
print("    umap_finetuned.png")
print("    umap_comparison.png")
print("=" * 55)