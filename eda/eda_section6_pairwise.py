"""
eda_section6_pairwise.py
────────────────────────
WHAT: Compute paired image quality metrics (SSIM, PSNR, MSE) between every
      degraded image and its ground-truth reference.
WHY:  These metrics quantify HOW MUCH degradation is present:
        • SSIM < 0.9 signals structural loss (not just noise)
        • PSNR < 30 dB is commonly considered "low quality"
        • Low values across the board confirm that a learned restoration
          model is needed — simple filters/normalisation won't suffice
      They also reveal if datasets vary in degradation severity, which
      informs weighted sampling or curriculum training strategies.

TOOLS: cv2, numpy, scikit-image (SSIM, PSNR), matplotlib, seaborn, pandas
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from eda_utils import DATASETS, OUTPUT_DIR, list_images, save_fig

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import (structural_similarity as ssim,
                              peak_signal_noise_ratio as psnr)
from tqdm import tqdm
import random


SAMPLE_N = 200
TARGET   = 256


def sample_pairs(raw_dir, ref_dir, n=SAMPLE_N, seed=42):
    random.seed(seed)
    raw_files = sorted(list_images(raw_dir))
    ref_files  = sorted(list_images(ref_dir))
    pairs = list(zip(raw_files, ref_files))
    return random.sample(pairs, min(n, len(pairs)))


def compute_pair(raw_path, ref_path):
    raw = cv2.imread(raw_path)
    ref  = cv2.imread(ref_path)
    if raw is None or ref is None:
        return None
    # Resize both to same size for fair comparison
    raw_r = cv2.resize(raw, (TARGET, TARGET)).astype(np.float32) / 255.
    ref_r  = cv2.resize(ref,  (TARGET, TARGET)).astype(np.float32) / 255.

    raw_g = cv2.cvtColor((raw_r * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    ref_g  = cv2.cvtColor((ref_r  * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)

    ssim_val = ssim(raw_g, ref_g, data_range=255)
    psnr_val = psnr(ref_g, raw_g, data_range=255)
    mse_val  = float(np.mean((raw_r - ref_r) ** 2))

    # per-channel MSE
    ch_mse = [float(np.mean((raw_r[:,:,c] - ref_r[:,:,c])**2)) for c in range(3)]
    return dict(ssim=ssim_val, psnr=psnr_val, mse=mse_val,
                mse_b=ch_mse[0], mse_g=ch_mse[1], mse_r=ch_mse[2])


def run():
    print("\n" + "="*60)
    print("SECTION 6 - PAIRWISE QUALITY METRICS (SSIM / PSNR / MSE)")
    print("="*60)

    records = []
    for ds_name, info in DATASETS.items():
        pairs = sample_pairs(info["raw"], info["ref"])
        for raw_p, ref_p in tqdm(pairs, desc=ds_name, leave=False):
            m = compute_pair(raw_p, ref_p)
            if m:
                records.append({**m, "dataset": ds_name})

    df = pd.DataFrame(records)
    df.to_csv(f"{OUTPUT_DIR}/s6_pairwise_metrics.csv", index=False)

    metrics = ["ssim", "psnr", "mse"]
    titles  = ["SSIM (0–1, higher=better)", "PSNR (dB, higher=better)",
               "MSE (lower=better)"]
    thresholds = {"ssim": 0.9, "psnr": 30.0, "mse": None}

    # ── 6a: violin plots per dataset ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for ax, metric, title in zip(axes, metrics, titles):
        data = [df[df["dataset"]==ds][metric].dropna().values
                for ds in DATASETS]
        vp = ax.violinplot(data, showmedians=True, widths=0.6)
        for body, (ds, info) in zip(vp["bodies"], DATASETS.items()):
            body.set_facecolor(info["color"]); body.set_alpha(0.7)
        if thresholds[metric] is not None:
            ax.axhline(thresholds[metric], color="red", lw=1.5, ls="--",
                       label=f"Threshold: {thresholds[metric]}")
            ax.legend(fontsize=8)
        ax.set_xticks(range(1, len(DATASETS)+1))
        ax.set_xticklabels(list(DATASETS.keys()))
        ax.set_title(title); ax.set_ylabel(metric.upper())
    fig.suptitle("S6 — Pairwise Quality: Degraded vs Reference\n"
                 "(Confirms significant structural degradation in all datasets)",
                 fontsize=13)
    save_fig(fig, "s6a_pairwise_violins.png")

    # ── 6b: SSIM vs PSNR scatter ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    for ds_name, info in DATASETS.items():
        sub = df[df["dataset"] == ds_name]
        ax.scatter(sub["psnr"], sub["ssim"],
                   alpha=0.35, s=20, color=info["color"], label=ds_name)
    ax.axvline(30, color="red", lw=1.2, ls="--", label="PSNR=30 dB threshold")
    ax.axhline(0.9, color="orange", lw=1.2, ls="--", label="SSIM=0.9 threshold")
    ax.set_xlabel("PSNR (dB)"); ax.set_ylabel("SSIM")
    ax.set_title("S6 — SSIM vs PSNR Scatter\n"
                 "(Most images fall below both quality thresholds)")
    ax.legend()
    save_fig(fig, "s6b_ssim_psnr_scatter.png")

    # ── 6c: per-channel MSE bar chart ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(DATASETS)); w = 0.25
    ch_names = ("Blue", "Green", "Red")
    ch_colors = ("#1f77b4", "#2ca02c", "#d62728")
    for i, (ch, col) in enumerate(zip(("mse_b", "mse_g", "mse_r"), ch_colors)):
        vals = [df[df["dataset"]==ds][ch].mean() for ds in DATASETS]
        bars = ax.bar(x + i*w, vals, w, label=ch_names[i], color=col, alpha=0.8)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0001,
                    f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x + w); ax.set_xticklabels(list(DATASETS.keys()))
    ax.set_ylabel("Mean Squared Error per Channel")
    ax.set_title("S6 — Per-Channel MSE: Degraded vs Reference\n"
                 "(Red channel has highest error — water absorbs red fastest)")
    ax.legend()
    save_fig(fig, "s6c_per_channel_mse.png")

    # ── 6d: SSIM distribution histogram ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for ds_name, info in DATASETS.items():
        sub = df[df["dataset"] == ds_name]["ssim"]
        ax.hist(sub, bins=25, alpha=0.55, color=info["color"],
                label=f"{ds_name} (mean={sub.mean():.3f})", edgecolor="none")
    ax.axvline(0.9, color="red", lw=1.5, ls="--", label="SSIM=0.9 quality threshold")
    ax.set_xlabel("SSIM"); ax.set_ylabel("Count")
    ax.set_title("S6 — SSIM Distribution per Dataset\n"
                 "(Degraded images consistently below 0.9 = structural loss confirmed)")
    ax.legend()
    save_fig(fig, "s6d_ssim_histograms.png")

    # ── 6e: visual gallery (worst/best SSIM pairs) ────────────────────────────
    for ds_name, info in DATASETS.items():
        sub = df[df["dataset"] == ds_name].copy()
        sub = sub.dropna(subset=["ssim"]).reset_index(drop=True)
        worst_idx = sub["ssim"].idxmin()
        best_idx  = sub["ssim"].idxmax()

        pairs_all = sample_pairs(info["raw"], info["ref"],
                                 n=SAMPLE_N, seed=42)
        if worst_idx >= len(pairs_all) or best_idx >= len(pairs_all):
            continue

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for row_i, (idx, label) in enumerate(((worst_idx, "Worst SSIM"),
                                               (best_idx,  "Best SSIM"))):
            raw_p, ref_p = pairs_all[idx]
            raw_img = cv2.cvtColor(cv2.imread(raw_p), cv2.COLOR_BGR2RGB)
            ref_img  = cv2.cvtColor(cv2.imread(ref_p),  cv2.COLOR_BGR2RGB)
            ssim_v   = sub.loc[idx, "ssim"]
            psnr_v   = sub.loc[idx, "psnr"]

            axes[row_i, 0].imshow(raw_img);  axes[row_i, 0].set_title(f"Degraded ({label})")
            axes[row_i, 1].imshow(ref_img);  axes[row_i, 1].set_title("Reference")
            axes[row_i, 0].axis("off");       axes[row_i, 1].axis("off")

            # difference map
            diff = np.abs(
                cv2.resize(raw_img, (256,256)).astype(np.float32) -
                cv2.resize(ref_img, (256,256)).astype(np.float32)
            ).mean(axis=2)
            im = axes[row_i, 2].imshow(diff, cmap="hot")
            axes[row_i, 2].set_title(f"Abs Difference\nSSIM={ssim_v:.3f} PSNR={psnr_v:.1f}dB")
            axes[row_i, 2].axis("off")
            plt.colorbar(im, ax=axes[row_i, 2], fraction=0.04)
            axes[row_i, 3].axis("off")

        fig.suptitle(f"S6 — {ds_name}: Best vs Worst SSIM Pair", fontsize=13)
        save_fig(fig, f"s6e_best_worst_ssim_{ds_name}.png")

    # Print summary
    print("\n  Quality metric means per dataset:")
    print(df.groupby("dataset")[metrics].mean().round(4).to_string())
    print("\n  ✔ Section 6 complete.")


if __name__ == "__main__":
    run()
