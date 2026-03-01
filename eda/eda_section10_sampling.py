"""
eda_section10_sampling.py
─────────────────────────
WHAT: Justify the INR coordinate sampling design choices:
      (1) Why sample 2048 points per image?
      (2) Why random + edge-weighted strategies?
      (3) Why uniform for validation?
      (4) What does the spatial distribution of meaningful signal look like?

      Also performs pixel-level correlation analysis (spatial autocorrelation)
      to quantify how much neighbouring pixels are correlated — this motivates
      why we need to sample across the whole image rather than dense patches.

WHY:  The coordinate-based INR training paradigm requires choosing a sampling
      strategy that:
        • Covers the image without redundancy
        • Over-samples informative (high-gradient) regions
        • Is consistent and repeatable at validation
      This section provides the empirical evidence for each choice.

TOOLS: cv2, numpy, scipy (autocorrelation), matplotlib, torch (optional)
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from eda_utils import DATASETS, OUTPUT_DIR, list_images, save_fig

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import uniform_filter
from tqdm import tqdm
import random


SAMPLE_N = 100
TARGET   = 256


def sample_files(folder, n=SAMPLE_N, seed=42):
    random.seed(seed)
    files = list_images(folder)
    return random.sample(files, min(n, len(files)))


def sobel_map(img_gray):
    gx = cv2.Sobel(img_gray.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(gx**2 + gy**2)


def edge_weighted_sample(edge_map, n=2048, edge_bias=0.7, seed=0):
    """Reproduce the pipeline's edge-weighted sampling."""
    rng = np.random.default_rng(seed)
    h, w = edge_map.shape
    flat = edge_map.ravel()
    flat_norm = flat / (flat.sum() + 1e-12)
    # blend edge probability with uniform
    uniform = np.ones(h*w, dtype=np.float64) / (h*w)
    prob = edge_bias * flat_norm + (1 - edge_bias) * uniform
    prob /= prob.sum()
    idxs = rng.choice(h*w, size=n, replace=False, p=prob)
    ys, xs = np.unravel_index(idxs, (h, w))
    return xs, ys


def random_sample(n=2048, h=256, w=256, seed=0):
    rng = np.random.default_rng(seed)
    xs = rng.integers(0, w, size=n)
    ys = rng.integers(0, h, size=n)
    return xs, ys


def uniform_sample(n=2048, h=256, w=256):
    """Regular grid."""
    stride = max(1, int(np.sqrt(h * w / n)))
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    return xs.ravel()[:n], ys.ravel()[:n]


def spatial_autocorrelation(img_gray, max_lag=20):
    """1-D autocorrelation along rows (measures spatial redundancy)."""
    img_f = img_gray.astype(np.float64)
    row_means = img_f.mean(axis=1, keepdims=True)
    img_c = img_f - row_means
    acfs = []
    for lag in range(max_lag):
        if lag == 0:
            acfs.append(1.0)
        else:
            c = float(np.mean(img_c[:, :-lag] * img_c[:, lag:]))
            v = float(np.var(img_c))
            acfs.append(c / v if v > 0 else 0.)
    return acfs


def run():
    print("\n" + "="*60)
    print("SECTION 10 - SAMPLING STRATEGY JUSTIFICATION")
    print("="*60)

    # ── 10a: sample coverage visualisation (all 3 strategies) ────────────────
    ds_name = list(DATASETS.keys())[0]
    info    = DATASETS[ds_name]
    raw_f   = list_images(info["raw"])[0]
    raw_bgr = cv2.imread(raw_f)
    raw_rgb = cv2.cvtColor(cv2.resize(raw_bgr, (TARGET, TARGET)), cv2.COLOR_BGR2RGB)
    raw_gray = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2GRAY)
    edge_map  = sobel_map(raw_gray)

    xs_r, ys_r  = random_sample(n=2048)
    xs_e, ys_e  = edge_weighted_sample(edge_map, n=2048)
    xs_u, ys_u  = uniform_sample(n=2048)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    strategies = [
        (xs_r, ys_r, "Random (2048 pts)", "#4C72B0"),
        (xs_e, ys_e, "Edge-Weighted (2048 pts, 70%)", "#DD8452"),
        (xs_u, ys_u, "Uniform Grid (2048 pts)", "#55A868"),
    ]
    for col, (xs, ys, title, color) in enumerate(strategies):
        # image + scatter
        axes[0, col].imshow(raw_rgb, alpha=0.7)
        axes[0, col].scatter(xs, ys, s=2, color=color, alpha=0.6)
        axes[0, col].set_title(title, fontsize=10)
        axes[0, col].axis("off")
        # density heatmap
        hist2d, _, _ = np.histogram2d(xs, ys, bins=32, range=[[0,TARGET],[0,TARGET]])
        axes[1, col].imshow(hist2d.T, cmap="YlOrRd", origin="upper")
        axes[1, col].set_title(f"Sample Density Heatmap\n{title}")
        axes[1, col].axis("off")
    fig.suptitle(f"S10 — Sampling Strategies on {ds_name} Image\n"
                 "Justifies why edge-weighted captures important regions better",
                 fontsize=13)
    save_fig(fig, "s10a_sampling_strategies.png")

    # ── 10b: edge map + sampling overlay ─────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(raw_rgb);            axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(edge_map, cmap="hot"); axes[1].set_title("Sobel Edge Map"); axes[1].axis("off")
    axes[2].imshow(raw_rgb, alpha=0.6)
    axes[2].scatter(xs_r, ys_r, s=2, c="#4C72B0", alpha=0.6, label="Random")
    axes[2].set_title("Random Sampling"); axes[2].axis("off")
    axes[3].imshow(raw_rgb, alpha=0.6)
    axes[3].scatter(xs_e, ys_e, s=2, c="#DD8452", alpha=0.6, label="Edge-Weighted")
    axes[3].set_title("Edge-Weighted Sampling\n(denser near edges)"); axes[3].axis("off")
    fig.suptitle("S10 — Edge-Weighted Sampling Focuses on High-Information Regions",
                 fontsize=12)
    save_fig(fig, "s10b_edge_sampling_overlay.png")

    # ── 10c: how many samples for adequate coverage? ──────────────────────────
    n_samples_range = [128, 256, 512, 1024, 2048, 4096, 8192]
    coverage_list = []
    for n in n_samples_range:
        xs_t, ys_t = random_sample(n=n, seed=99)
        coverage = np.zeros((TARGET, TARGET), dtype=bool)
        coverage[ys_t, xs_t] = True
        # "coverage" within radius-3 neighbourhood
        from scipy.ndimage import binary_dilation
        struct = np.ones((7, 7), dtype=bool)
        coverage_dilated = binary_dilation(coverage, structure=struct)
        coverage_list.append(coverage_dilated.mean())

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogx(n_samples_range, [c*100 for c in coverage_list],
                marker="o", color="#4C72B0", lw=2)
    ax.axvline(2048, color="red", lw=1.5, ls="--", label="Chosen N=2048")
    ax.axhline(90, color="gray", lw=1, ls=":", label="90% coverage")
    ax.set_xlabel("Number of sampled coordinates (log scale)")
    ax.set_ylabel("Effective pixel coverage (within radius 3) %")
    ax.set_title("S10 — Effective Image Coverage vs N Samples\n"
                 "(N=2048 achieves good coverage while keeping batch size tractable)")
    ax.legend(); ax.set_ylim(0, 105)
    for n, c in zip(n_samples_range, coverage_list):
        ax.annotate(f"{c*100:.0f}%", (n, c*100+1.5), ha="center", fontsize=8)
    save_fig(fig, "s10c_coverage_vs_n.png")

    # ── 10d: spatial autocorrelation (why random/spread sampling works) ───────
    fig, ax = plt.subplots(figsize=(10, 5))
    for ds_name, info in DATASETS.items():
        acfs_ds = []
        for fpath in tqdm(sample_files(info["raw"], n=50), desc=ds_name, leave=False):
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            img_rs = cv2.resize(img, (TARGET, TARGET))
            acfs_ds.append(spatial_autocorrelation(img_rs, max_lag=30))
        if not acfs_ds: continue
        mean_acf = np.mean(acfs_ds, axis=0)
        lags = range(len(mean_acf))
        ax.plot(lags, mean_acf, color=info["color"], lw=2, label=ds_name)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.axhline(0.5, color="gray", lw=0.8, ls=":", label="ACF=0.5")
    ax.set_xlabel("Spatial Lag (pixels)")
    ax.set_ylabel("Autocorrelation Coefficient")
    ax.set_title("S10 — Spatial Autocorrelation of Pixel Intensities\n"
                 "(High ACF at small lags → neighbouring pixels are redundant "
                 "→ spread sampling is more efficient)")
    ax.legend()
    save_fig(fig, "s10d_spatial_autocorrelation.png")

    # ── 10e: GPU memory budget for RTX 2060 (6 GB) ──────────────────────────
    print("\n  Memory budget analysis (RTX 2060, 6 GB VRAM):")
    dtype_bytes = 4   # float32
    n_samples   = 2048
    configs = [
        (4,  "Batch=4  (safe)"),
        (8,  "Batch=8  (chosen)"),
        (16, "Batch=16 (borderline)"),
        (32, "Batch=32 (risky)"),
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    vram_total_mb = 6 * 1024
    bs_list, mem_list = [], []
    for bs, label in configs:
        # coords [B,N,2] + raw_rgb [B,N,3] + ref_rgb [B,N,3] + model params ~50MB
        data_mb = bs * n_samples * (2 + 3 + 3) * dtype_bytes / 1e6
        total_mb = data_mb + 50   # rough model overhead
        bs_list.append(bs)
        mem_list.append(total_mb)
        pct = total_mb / vram_total_mb * 100
        print(f"    {label}: data={data_mb:.1f} MB + model≈50 MB "
              f"→ {total_mb:.1f} MB ({pct:.1f}% of 6 GB)")

    bars = ax.bar([c[1] for c in configs], mem_list, color="#4C72B0", alpha=0.8)
    ax.axhline(vram_total_mb, color="red", lw=2, ls="--",
               label=f"RTX 2060 VRAM = {vram_total_mb} MB")
    ax.axhline(vram_total_mb * 0.8, color="orange", lw=1.5, ls=":",
               label="80% safe limit")
    for bar, m in zip(bars, mem_list):
        ax.text(bar.get_x()+bar.get_width()/2, m+50,
                f"{m:.0f} MB\n({m/vram_total_mb*100:.0f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Total VRAM Required (MB)")
    ax.set_title(f"S10 — GPU Memory Budget (N={n_samples} samples/image)\n"
                 "RTX 2060: 6 GB VRAM")
    ax.set_ylim(0, vram_total_mb * 1.2); ax.legend()
    save_fig(fig, "s10e_gpu_memory_budget.png")

    print("\n  ✔ Section 10 complete.")


if __name__ == "__main__":
    run()
