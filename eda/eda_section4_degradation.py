"""
eda_section4_degradation.py
───────────────────────────
WHAT: Quantify the specific physical degradation present in underwater images.
      Measures: contrast, haze/scattering (dark channel prior), noise level
      (Laplacian variance proxy), local contrast (Michelson), dynamic range.
WHY:  Underwater imaging degrades through four physics-driven phenomena:
        1. Absorption  → colour shift (Section 3)
        2. Scattering  → haze / veiling light (dark channel prior)
        3. Low contrast → washed-out fine detail (grayscale std)
        4. Backscatter → noise at longer distances
      Measuring these before preprocessing justifies every metric stored
      in the degradation_metrics JSON files.

TOOLS: cv2, numpy, scipy, matplotlib, seaborn, pandas
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from eda_utils import DATASETS, OUTPUT_DIR, list_images, save_fig

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random


SAMPLE_N = 300


def sample_files(folder, n=SAMPLE_N, seed=42):
    random.seed(seed)
    files = list_images(folder)
    return random.sample(files, min(n, len(files)))


# ── Metric functions ──────────────────────────────────────────────────────────

def contrast_rms(img_gray):
    """RMS contrast = std of normalised grayscale pixels."""
    return float(img_gray.astype(np.float32).std() / 255.0)


def dark_channel_haze(img_bgr, patch=15):
    """
    Dark Channel Prior (He et al. 2009).
    In haze-free images, at least one channel per patch is near zero.
    High dark channel mean → heavy haze / scattering / veiling light.
    """
    img_f = img_bgr.astype(np.float32) / 255.0
    min_rgb = img_f.min(axis=2)                      # min across channels
    kernel  = np.ones((patch, patch), np.uint8)
    dark    = cv2.erode(min_rgb, kernel)              # local minimum
    return float(dark.mean())


def laplacian_blur(img_gray):
    """
    Laplacian variance as sharpness proxy.
    Low value = blurry (typical in hazy underwater scenes).
    """
    lap = cv2.Laplacian(img_gray, cv2.CV_64F)
    return float(lap.var())


def michelson_contrast(img_gray):
    """Local Michelson contrast (I_max - I_min) / (I_max + I_min)."""
    mn, mx = float(img_gray.min()), float(img_gray.max())
    denom = mx + mn
    return float((mx - mn) / denom) if denom > 0 else 0.0


def dynamic_range(img_gray):
    """
    Effective dynamic range (2nd–98th percentile, avoids outlier pixels).
    Low value = crushed histogram = low contrast.
    """
    lo = np.percentile(img_gray, 2)
    hi = np.percentile(img_gray, 98)
    return float((hi - lo) / 255.0)


def compute_all(fpath):
    img_bgr  = cv2.imread(fpath)
    if img_bgr is None:
        return None
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return dict(
        contrast   = contrast_rms(img_gray),
        haze       = dark_channel_haze(img_bgr),
        sharpness  = laplacian_blur(img_gray),
        michelson  = michelson_contrast(img_gray),
        dyn_range  = dynamic_range(img_gray),
    )


def run():
    print("\n" + "="*60)
    print("SECTION 4 - DEGRADATION METRICS")
    print("="*60)

    records = []
    for ds_name, info in DATASETS.items():
        for role, folder in (("raw", info["raw"]), ("ref", info["ref"])):
            for fpath in tqdm(sample_files(folder), desc=f"{ds_name}/{role}", leave=False):
                m = compute_all(fpath)
                if m:
                    records.append({**m, "dataset": ds_name, "role": role})

    df = pd.DataFrame(records)
    df.to_csv(f"{OUTPUT_DIR}/s4_degradation_metrics.csv", index=False)

    metrics = ["contrast", "haze", "sharpness", "michelson", "dyn_range"]
    labels  = ["RMS Contrast", "Haze (Dark Channel)",
               "Sharpness (Laplacian Var)", "Michelson Contrast",
               "Dynamic Range (2–98 pct)"]

    # ── 4a: violin plots raw vs ref per metric ────────────────────────────────
    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 6))
    for ax, metric, label in zip(axes, metrics, labels):
        data_plot, tick_labels, palette = [], [], []
        for ds_name, info in DATASETS.items():
            raw_v = df[(df["dataset"]==ds_name) & (df["role"]=="raw")][metric].dropna()
            ref_v = df[(df["dataset"]==ds_name) & (df["role"]=="ref")][metric].dropna()
            data_plot += [raw_v.values, ref_v.values]
            tick_labels += [f"{ds_name}\nDeg", f"{ds_name}\nRef"]
            palette += [info["color"], "#aaaaaa"]
        vp = ax.violinplot(data_plot, showmedians=True, widths=0.7)
        for i, (body, col) in enumerate(zip(vp["bodies"], palette)):
            body.set_facecolor(col); body.set_alpha(0.6)
        ax.set_xticks(range(1, len(tick_labels)+1))
        ax.set_xticklabels(tick_labels, fontsize=7)
        ax.set_title(label, fontsize=9)
    fig.suptitle("S4 — Degradation Metrics: Degraded vs Reference\n"
                 "(All datasets, sampled images)", fontsize=13)
    save_fig(fig, "s4a_degradation_violins.png")

    # ── 4b: dark channel prior visual demo ────────────────────────────────────
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for row_idx, ds_name in enumerate(DATASETS.keys()):
        info = DATASETS[ds_name]
        raw_f = list_images(info["raw"])[0]
        ref_f = list_images(info["ref"])[0]
        raw_bgr = cv2.imread(raw_f); ref_bgr = cv2.imread(ref_f)
        if raw_bgr is None or ref_bgr is None: continue
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        ref_rgb  = cv2.cvtColor(ref_bgr,  cv2.COLOR_BGR2RGB)

        # dark channel maps
        def dc_map(img_bgr, patch=15):
            img_f = img_bgr.astype(np.float32)/255.
            min_rgb = img_f.min(axis=2)
            kernel = np.ones((patch, patch), np.uint8)
            return cv2.erode(min_rgb, kernel)

        dc_raw = dc_map(raw_bgr); dc_ref = dc_map(ref_bgr)

        axes[row_idx, 0].imshow(raw_rgb);  axes[row_idx, 0].set_title(f"{ds_name} Degraded")
        axes[row_idx, 1].imshow(ref_rgb);  axes[row_idx, 1].set_title(f"{ds_name} Reference")
        im = axes[row_idx, 2].imshow(dc_raw, cmap="hot", vmin=0, vmax=0.5)
        axes[row_idx, 2].set_title(f"Dark Ch (Deg) mean={dc_raw.mean():.3f}")
        im = axes[row_idx, 3].imshow(dc_ref, cmap="hot", vmin=0, vmax=0.5)
        axes[row_idx, 3].set_title(f"Dark Ch (Ref) mean={dc_ref.mean():.3f}")
        plt.colorbar(im, ax=axes[row_idx, 3], fraction=0.03)
        for ax in axes[row_idx]: ax.axis("off")

    fig.suptitle("S4 — Dark Channel Prior: Haze Visualisation\n"
                 "Bright regions in dark channel = heavy haze/scattering",
                 fontsize=13)
    save_fig(fig, "s4b_dark_channel_demo.png")

    # ── 4c: haze vs contrast scatter (degraded only) ──────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    raw_df = df[df["role"] == "raw"]
    for ds_name, info in DATASETS.items():
        sub = raw_df[raw_df["dataset"] == ds_name]
        ax.scatter(sub["haze"], sub["contrast"],
                   alpha=0.35, s=18, color=info["color"], label=ds_name)
    ax.set_xlabel("Haze Score (Dark Channel Mean)")
    ax.set_ylabel("RMS Contrast")
    ax.set_title("S4 — Haze vs Contrast in Degraded Images\n"
                 "(High haze → low contrast: inverse relationship)")
    ax.legend()
    save_fig(fig, "s4c_haze_vs_contrast.png")

    # ── 4d: per-dataset degradation delta (raw - ref) ────────────────────────
    delta_rows = []
    for ds_name in DATASETS:
        r = df[(df["dataset"]==ds_name) & (df["role"]=="raw")][metrics].mean()
        f = df[(df["dataset"]==ds_name) & (df["role"]=="ref")][metrics].mean()
        delta_rows.append((ds_name, *(r - f).values))
    delta_df = pd.DataFrame(delta_rows, columns=["dataset"] + metrics)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics)); w = 0.25
    for i, (ds_name, info) in enumerate(DATASETS.items()):
        row = delta_df[delta_df["dataset"]==ds_name].iloc[0]
        vals = [row[m] for m in metrics]
        ax.bar(x + i*w, vals, w, label=ds_name, color=info["color"], alpha=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x + w); ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Mean(Degraded) − Mean(Reference)")
    ax.set_title("S4 — Degradation Delta per Dataset\n"
                 "(Positive = degraded image is worse; negative = cleaner)")
    ax.legend()
    save_fig(fig, "s4d_degradation_delta.png")

    # ── 4e: sharpness (blurriness) KDE ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for ds_name, info in DATASETS.items():
        raw_s = df[(df["dataset"]==ds_name)&(df["role"]=="raw")]["sharpness"].clip(0, 5000)
        ref_s = df[(df["dataset"]==ds_name)&(df["role"]=="ref")]["sharpness"].clip(0, 5000)
        sns.kdeplot(raw_s, ax=ax, label=f"{ds_name} Degraded",
                    color=info["color"], fill=True, alpha=0.25, linewidth=2)
        sns.kdeplot(ref_s, ax=ax, label=f"{ds_name} Reference",
                    color=info["color"], fill=False, linewidth=2, linestyle="--")
    ax.set_xlabel("Laplacian Variance (Sharpness Proxy)")
    ax.set_title("S4 — Sharpness Distribution: Degraded vs Reference\n"
                 "(Degraded images are significantly blurrier)")
    ax.legend(fontsize=8)
    save_fig(fig, "s4e_sharpness_kde.png")

    print("\n  Summary statistics:")
    print(df.groupby(["dataset","role"])[metrics].mean().round(4).to_string())
    print("\n  ✔ Section 4 complete.")


if __name__ == "__main__":
    run()
