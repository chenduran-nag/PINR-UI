"""
eda_section9_domain_gap.py
──────────────────────────
WHAT: Quantify the visual domain gap between the three datasets.
      Uses feature-level comparison (colour moments, texture moments)
      and visual t-SNE/PCA embedding of image features to show how
      distinct each dataset is in feature space.
WHY:  If datasets occupy separate regions of feature space:
        • A model trained on one may not generalise to another
        • Per-dataset stratified splitting is critical (not random global split)
        • The diversity of the combined set is a training advantage
      This justifies the three-dataset strategy and the stratified split.

TOOLS: cv2, numpy, sklearn (PCA, TSNE), matplotlib, seaborn
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from eda_utils import DATASETS, OUTPUT_DIR, list_images, save_fig

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
import random


SAMPLE_N = 150
TARGET   = 64    # small size for feature extraction (speed)


def sample_files(folder, n=SAMPLE_N, seed=42):
    random.seed(seed)
    files = list_images(folder)
    return random.sample(files, min(n, len(files)))


def extract_features(fpath):
    """
    Lightweight feature vector per image:
    RGB colour moments (mean, std, skew × 3 channels) = 9
    HSV moments = 9
    LF energy ratio = 1
    Edge density = 1
    Total = 20 features
    """
    from scipy.stats import skew as sp_skew
    img_bgr = cv2.imread(fpath)
    if img_bgr is None: return None
    img_rs  = cv2.resize(img_bgr, (TARGET, TARGET)).astype(np.float32)/255.
    img_rgb = cv2.cvtColor((img_rs*255).astype(np.uint8), cv2.COLOR_BGR2RGB).astype(np.float32)/255.
    img_hsv = cv2.cvtColor((img_rs*255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    img_hsv[:,:,0] /= 179.; img_hsv[:,:,1:] /= 255.
    img_gray = cv2.cvtColor((img_rs*255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)/255.

    feats = []
    for ch in range(3):
        px = img_rgb[:,:,ch].ravel()
        feats += [px.mean(), px.std(), float(sp_skew(px))]
    for ch in range(3):
        px = img_hsv[:,:,ch].ravel()
        feats += [px.mean(), px.std(), float(sp_skew(px))]

    # LF ratio
    f = np.fft.fftshift(np.fft.fft2(img_gray))
    mag = np.abs(f)**2
    h, w = mag.shape; cy, cx = h//2, w//2
    r = int(min(h,w)*0.1)
    Y, X = np.ogrid[:h,:w]
    mask = (X-cx)**2 + (Y-cy)**2 <= r**2
    feats.append(float(mag[mask].sum()/mag.sum()))

    # edge density
    gx = cv2.Sobel(img_gray*255, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray*255, cv2.CV_64F, 0, 1, ksize=3)
    feats.append(float(np.sqrt(gx**2+gy**2).mean()))

    return np.array(feats, dtype=np.float32)


def run():
    print("\n" + "="*60)
    print("SECTION 9 - INTER-DATASET DOMAIN GAP")
    print("="*60)

    all_feats, all_labels, all_roles = [], [], []

    for ds_name, info in DATASETS.items():
        for role, folder in (("raw", info["raw"]), ("ref", info["ref"])):
            for fpath in tqdm(sample_files(folder), desc=f"{ds_name}/{role}", leave=False):
                feat = extract_features(fpath)
                if feat is not None:
                    all_feats.append(feat)
                    all_labels.append(ds_name)
                    all_roles.append(role)

    X = np.array(all_feats)
    labels = np.array(all_labels)
    roles  = np.array(all_roles)

    # clean NaN / Inf
    mask = np.all(np.isfinite(X), axis=1)
    X, labels, roles = X[mask], labels[mask], roles[mask]

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # ── 9a: PCA 2-D embedding ─────────────────────────────────────────────────
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_s)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # coloured by dataset
    ax = axes[0]
    for ds_name, info in DATASETS.items():
        mask_ds = labels == ds_name
        ax.scatter(X_pca[mask_ds, 0], X_pca[mask_ds, 1],
                   alpha=0.4, s=18, color=info["color"], label=ds_name)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title("S9 — PCA: Coloured by Dataset\n(Separation = domain gap)")
    ax.legend()

    # coloured by role
    ax = axes[1]
    for role, color, marker in (("raw","#E74C3C","o"), ("ref","#2ECC71","^")):
        mask_r = roles == role
        ax.scatter(X_pca[mask_r, 0], X_pca[mask_r, 1],
                   alpha=0.35, s=18, color=color, marker=marker,
                   label=f"{'Degraded' if role=='raw' else 'Reference'}")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title("S9 — PCA: Coloured by Role\n(Degraded vs Reference overlap)")
    ax.legend()

    fig.suptitle("S9 — PCA Feature Space Projection", fontsize=13)
    save_fig(fig, "s9a_pca_embedding.png")

    # ── 9b: t-SNE embedding (best for non-linear separation) ─────────────────
    print("  Running t-SNE (this may take ~60s)…")
    tsne = TSNE(n_components=2, perplexity=40, random_state=42,
                max_iter=1000, verbose=0)
    X_tsne = tsne.fit_transform(X_s)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    for ds_name, info in DATASETS.items():
        mask_ds = labels == ds_name
        ax.scatter(X_tsne[mask_ds, 0], X_tsne[mask_ds, 1],
                   alpha=0.45, s=18, color=info["color"], label=ds_name)
    ax.set_title("S9 — t-SNE: Coloured by Dataset")
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    ax.legend()

    ax = axes[1]
    for role, color, marker in (("raw","#E74C3C","o"), ("ref","#2ECC71","^")):
        mask_r = roles == role
        ax.scatter(X_tsne[mask_r, 0], X_tsne[mask_r, 1],
                   alpha=0.35, s=18, color=color, marker=marker,
                   label=f"{'Degraded' if role=='raw' else 'Reference'}")
    ax.set_title("S9 — t-SNE: Coloured by Role")
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    ax.legend()

    fig.suptitle("S9 — t-SNE Feature Space (non-linear)\n"
                 "Clusters = distinct visual domains per dataset", fontsize=13)
    save_fig(fig, "s9b_tsne_embedding.png")

    # ── 9c: feature-level pairwise dataset distance ───────────────────────────
    ds_names = list(DATASETS.keys())
    n_ds = len(ds_names)
    dist_matrix = np.zeros((n_ds, n_ds))
    for i, ds_i in enumerate(ds_names):
        for j, ds_j in enumerate(ds_names):
            fi = X_s[labels == ds_i]
            fj = X_s[labels == ds_j]
            # Euclidean distance between centroids
            dist_matrix[i, j] = np.linalg.norm(fi.mean(axis=0) - fj.mean(axis=0))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(dist_matrix, cmap="YlOrRd")
    ax.set_xticks(range(n_ds)); ax.set_yticks(range(n_ds))
    ax.set_xticklabels(ds_names); ax.set_yticklabels(ds_names)
    for i in range(n_ds):
        for j in range(n_ds):
            ax.text(j, i, f"{dist_matrix[i,j]:.2f}",
                    ha="center", va="center", fontsize=11, color="black")
    plt.colorbar(im, ax=ax, label="Centroid distance (normalised features)")
    ax.set_title("S9 — Pairwise Dataset Distance in Feature Space\n"
                 "(Higher = more domain gap)")
    save_fig(fig, "s9c_dataset_distance_matrix.png")

    # ── 9d: PCA variance explained scree plot ─────────────────────────────────
    pca_full = PCA(n_components=min(10, X_s.shape[1]), random_state=42)
    pca_full.fit(X_s)
    evr = pca_full.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(1, len(evr)+1), evr*100, color="#4C72B0", alpha=0.8)
    ax.plot(range(1, len(evr)+1), np.cumsum(evr)*100,
            color="red", marker="o", lw=2, label="Cumulative")
    ax.axhline(90, color="gray", lw=1, ls="--", label="90% threshold")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title("S9 — PCA Scree Plot\n(How many feature dimensions carry most info)")
    ax.legend()
    save_fig(fig, "s9d_pca_scree.png")

    print("\n  Domain gap (centroid distances):")
    for i, ds_i in enumerate(ds_names):
        for j, ds_j in enumerate(ds_names):
            if i < j:
                print(f"  {ds_i} ↔ {ds_j}: {dist_matrix[i,j]:.3f}")
    print("\n  ✔ Section 9 complete.")


if __name__ == "__main__":
    run()
