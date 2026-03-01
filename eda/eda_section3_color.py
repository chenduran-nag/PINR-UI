"""
eda_section3_color.py
─────────────────────
WHAT: Full colour analysis of degraded vs reference images.
      Covers RGB, HSV, LAB colour spaces, channel histograms,
      mean channel values, colour cast severity.
WHY:  Underwater light absorption causes predictable colour shifts:
        red attenuates fastest (~5 m), then green (~20 m), blue lasts longest.
      Quantifying this:
        • Motivates colour cast as a degradation metric
        • Shows which channels are most affected
        • Explains why standard RGB normalisation alone isn't enough

TOOLS: cv2, numpy, matplotlib, seaborn, scipy
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from eda_utils import DATASETS, OUTPUT_DIR, list_images, load_rgb, save_fig

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm import tqdm
import random


SAMPLE_N = 200   # images per dataset for heavy computations


def sample_files(folder, n=SAMPLE_N, seed=42):
    random.seed(seed)
    files = list_images(folder)
    return random.sample(files, min(n, len(files)))


def channel_stats(fpath, role="raw"):
    img = cv2.imread(fpath)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    r_mean, g_mean, b_mean = (img_rgb[:,:,c].mean() for c in range(3))
    cast = float(np.std([r_mean, g_mean, b_mean]))   # colour cast = imbalance
    # HSV saturation and value (brightness)
    img_hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    img_hsv /= np.array([179., 255., 255.])
    sat  = img_hsv[:,:,1].mean()
    bri  = img_hsv[:,:,2].mean()
    # LAB a* and b* encode green-red and blue-yellow
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab).astype(np.float32) / 255.0
    a_mean  = img_lab[:,:,1].mean()
    b_chan  = img_lab[:,:,2].mean()
    return dict(r=r_mean, g=g_mean, b=b_mean,
                cast=cast, sat=sat, bri=bri,
                a_lab=a_mean, b_lab=b_chan, role=role)


def run():
    print("\n" + "="*60)
    print("SECTION 3 - COLOUR SPACE ANALYSIS")
    print("="*60)

    all_stats = []
    per_ds_raw = {}; per_ds_ref = {}

    for ds_name, info in DATASETS.items():
        raw_files = sample_files(info["raw"])
        ref_files = sample_files(info["ref"])

        raw_stats, ref_stats = [], []
        for fpath in tqdm(raw_files, desc=f"{ds_name}/raw", leave=False):
            s = channel_stats(fpath, "raw")
            if s: raw_stats.append(s); all_stats.append({**s, "dataset": ds_name})
        for fpath in tqdm(ref_files, desc=f"{ds_name}/ref", leave=False):
            s = channel_stats(fpath, "ref")
            if s: ref_stats.append(s); all_stats.append({**s, "dataset": ds_name})

        per_ds_raw[ds_name] = raw_stats
        per_ds_ref[ds_name] = ref_stats

    import pandas as pd
    df = pd.DataFrame(all_stats)
    df.to_csv(f"{OUTPUT_DIR}/s3_colour_stats.csv", index=False)

    # ── 3a: mean R G B bar chart raw vs ref per dataset ───────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    channels = ("r", "g", "b"); ch_colors = ("#d62728", "#2ca02c", "#1f77b4")
    for ax, ds_name in zip(axes, DATASETS.keys()):
        raw = per_ds_raw[ds_name]; ref = per_ds_ref[ds_name]
        x = np.arange(3); w = 0.35
        raw_means = [np.mean([s[c] for s in raw]) for c in channels]
        ref_means = [np.mean([s[c] for s in ref]) for c in channels]
        b1 = ax.bar(x - w/2, raw_means, w, color=ch_colors, alpha=0.6,
                    label="Degraded", edgecolor="white")
        b2 = ax.bar(x + w/2, ref_means, w, color=ch_colors, alpha=1.0,
                    label="Reference", edgecolor="white", hatch="//")
        ax.set_xticks(x); ax.set_xticklabels(["R", "G", "B"])
        ax.set_ylim(0, 0.7); ax.set_title(ds_name)
        ax.set_ylabel("Mean pixel intensity")
        if ax == axes[0]: ax.legend()
    fig.suptitle("S3 — Mean RGB Channel Values: Degraded vs Reference\n"
                 "(Note blue/green dominance in degraded images)", fontsize=13)
    save_fig(fig, "s3a_mean_rgb_comparison.png")

    # ── 3b: colour cast severity box plots ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    data_cast = []
    labels_cast = []
    colors_cast = []
    for ds_name, info in DATASETS.items():
        raw_cast = [s["cast"] for s in per_ds_raw[ds_name]]
        ref_cast = [s["cast"] for s in per_ds_ref[ds_name]]
        data_cast += [raw_cast, ref_cast]
        labels_cast += [f"{ds_name}\nDegraded", f"{ds_name}\nReference"]
        colors_cast += [info["color"], "#aaaaaa"]
    bp = ax.boxplot(data_cast, patch_artist=True, labels=labels_cast,
                    widths=0.5, medianprops=dict(color="white", lw=2))
    for patch, c in zip(bp["boxes"], colors_cast):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    ax.set_ylabel("Colour Cast Score (std of channel means)")
    ax.set_title("S3 — Colour Cast Severity: Degraded vs Reference\n"
                 "(Higher = more imbalanced channels = more colour degradation)")
    ax.tick_params(axis='x', labelsize=8)
    save_fig(fig, "s3b_colour_cast_boxplot.png")

    # ── 3c: HSV saturation & brightness ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, title in zip(axes,
        ("sat", "bri"), ("Saturation (HSV-S)", "Brightness (HSV-V)")):
        for ds_name, info in DATASETS.items():
            raw_v = [s[metric] for s in per_ds_raw[ds_name]]
            ref_v = [s[metric] for s in per_ds_ref[ds_name]]
            ax.hist(raw_v, bins=30, alpha=0.5, color=info["color"],
                    label=f"{ds_name} degraded", edgecolor="none")
            ax.hist(ref_v, bins=30, alpha=0.5, color=info["color"],
                    label=f"{ds_name} reference", edgecolor="none",
                    histtype="step", linewidth=2)
        ax.set_xlabel(title); ax.set_ylabel("Count")
        ax.set_title(f"S3 — {title}")
        ax.legend(fontsize=7)
    fig.suptitle("S3 — HSV Saturation & Brightness Distributions", y=1.02)
    save_fig(fig, "s3c_hsv_saturation_brightness.png")

    # ── 3d: LAB colour gamut scatter (a* vs b*) ──────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, ds_name, info in zip(axes, DATASETS.keys(), DATASETS.values()):
        raw_a = [s["a_lab"] for s in per_ds_raw[ds_name]]
        raw_b = [s["b_lab"] for s in per_ds_raw[ds_name]]
        ref_a = [s["a_lab"] for s in per_ds_ref[ds_name]]
        ref_b = [s["b_lab"] for s in per_ds_ref[ds_name]]
        ax.scatter(raw_a, raw_b, alpha=0.4, s=15,
                   color=info["color"], label="Degraded")
        ax.scatter(ref_a, ref_b, alpha=0.4, s=15,
                   color="#555555", label="Reference", marker="^")
        ax.axhline(0.5, color="gray", lw=0.5, ls="--")
        ax.axvline(0.5, color="gray", lw=0.5, ls="--")
        ax.set_xlabel("a* (green↔red)"); ax.set_ylabel("b* (blue↔yellow)")
        ax.set_title(ds_name); ax.legend(fontsize=8)
    fig.suptitle("S3 — LAB Colour Gamut (a* vs b*)\n"
                 "Degraded images cluster in blue-green region", fontsize=13)
    save_fig(fig, "s3d_lab_gamut_scatter.png")

    # ── 3e: pixel-level channel histograms on 3 sample pairs ─────────────────
    for ds_name, info in DATASETS.items():
        raw_files = list_images(info["raw"])
        ref_files  = list_images(info["ref"])
        if not raw_files or not ref_files:
            continue
        raw_img = cv2.cvtColor(cv2.imread(raw_files[0]), cv2.COLOR_BGR2RGB)
        ref_img  = cv2.cvtColor(cv2.imread(ref_files[0]),  cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(2, 4, figsize=(16, 7))
        # top: images
        axes[0, 0].imshow(raw_img);  axes[0, 0].set_title("Degraded"); axes[0,0].axis("off")
        axes[0, 3].imshow(ref_img);  axes[0, 3].set_title("Reference"); axes[0,3].axis("off")
        axes[0, 1].axis("off"); axes[0, 2].axis("off")
        # bottom: histograms
        ch_names  = ("Red", "Green", "Blue")
        ch_colors = ("#d62728", "#2ca02c", "#1f77b4")
        for i, (ch, col) in enumerate(zip(ch_names, ch_colors)):
            axes[1, i].hist(raw_img[:,:,i].ravel(), bins=256,
                            color=col, alpha=0.6, label="Degraded")
            axes[1, i].hist(ref_img[:,:,i].ravel(), bins=256,
                            color=col, alpha=0.3, label="Reference",
                            histtype="step", lw=2)
            axes[1, i].set_title(f"{ch} channel"); axes[1, i].legend(fontsize=7)
        axes[1, 3].axis("off")
        fig.suptitle(f"S3 — Channel Histograms: {ds_name} Sample Pair", fontsize=13)
        save_fig(fig, f"s3e_channel_hist_{ds_name}.png")

    print("\n  ✔ Section 3 complete.")


if __name__ == "__main__":
    run()
