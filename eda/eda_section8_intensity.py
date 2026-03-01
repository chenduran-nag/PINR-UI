"""
eda_section8_intensity.py
─────────────────────────
WHAT: Analyse pixel intensity distributions — global and local.
      Covers: brightness histograms, pixel value range utilisation,
      dark/light pixel ratios, histogram entropy, cumulative distribution
      comparison, and mean image composites.
WHY:  Before choosing how to normalise or preprocess:
        • We must know if images are under-/over-exposed
        • Histogram entropy tells us if images are information-dense
        • Clipped pixels (0 or 255) indicate exposure problems
        • Mean composite reveals systematic spatial brightness patterns
          (e.g., surface light from above, darker bottom)
        • This informs normalisation strategy: [0,1] float or [-1,1] centred

TOOLS: cv2, numpy, scipy, matplotlib
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from eda_utils import DATASETS, OUTPUT_DIR, list_images, save_fig

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy as sp_entropy
from tqdm import tqdm
import random


SAMPLE_N = 200
TARGET   = 256


def sample_files(folder, n=SAMPLE_N, seed=42):
    random.seed(seed)
    files = list_images(folder)
    return random.sample(files, min(n, len(files)))


def histogram_entropy(img_gray):
    hist, _ = np.histogram(img_gray.ravel(), bins=256, range=(0, 256), density=True)
    return float(sp_entropy(hist + 1e-12))


def clipped_fraction(img_gray, lo=5, hi=250):
    total = img_gray.size
    dark  = float((img_gray <= lo).sum() / total)
    bright = float((img_gray >= hi).sum() / total)
    return dark, bright


def run():
    print("\n" + "="*60)
    print("SECTION 8 - PIXEL INTENSITY & BRIGHTNESS")
    print("="*60)

    records = []
    mean_imgs = {ds: {"raw": np.zeros((TARGET, TARGET, 3)),
                      "ref": np.zeros((TARGET, TARGET, 3)),
                      "cnt": {"raw": 0, "ref": 0}}
                 for ds in DATASETS}

    # Accumulate global histograms per dataset/role
    global_hists = {ds: {"raw": np.zeros(256), "ref": np.zeros(256)}
                    for ds in DATASETS}

    for ds_name, info in DATASETS.items():
        for role, folder in (("raw", info["raw"]), ("ref", info["ref"])):
            for fpath in tqdm(sample_files(folder), desc=f"{ds_name}/{role}", leave=False):
                img_bgr  = cv2.imread(fpath)
                if img_bgr is None: continue
                img_rs   = cv2.resize(img_bgr, (TARGET, TARGET))
                img_gray = cv2.cvtColor(img_rs, cv2.COLOR_BGR2GRAY)

                # global hist
                hist, _ = np.histogram(img_gray.ravel(), bins=256, range=(0, 256))
                global_hists[ds_name][role] += hist

                # mean image composite
                img_rgb = cv2.cvtColor(img_rs, cv2.COLOR_BGR2RGB).astype(np.float32)/255.
                mean_imgs[ds_name][role] += img_rgb
                mean_imgs[ds_name]["cnt"][role] += 1

                # per-image metrics
                ent = histogram_entropy(img_gray)
                dk, br = clipped_fraction(img_gray)
                records.append(dict(
                    dataset=ds_name, role=role,
                    brightness=float(img_gray.mean()),
                    ent=ent, dark_clip=dk, bright_clip=br,
                    pct10=float(np.percentile(img_gray, 10)),
                    pct90=float(np.percentile(img_gray, 90)),
                ))

    import pandas as pd
    df = pd.DataFrame(records)
    df.to_csv(f"{OUTPUT_DIR}/s8_intensity_stats.csv", index=False)

    # ── 8a: global histogram comparison (degraded vs reference) ──────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, ds_name, info in zip(axes, DATASETS.keys(), DATASETS.values()):
        raw_h = global_hists[ds_name]["raw"]
        ref_h  = global_hists[ds_name]["ref"]
        bins = np.arange(256)
        ax.fill_between(bins, raw_h / raw_h.sum(), alpha=0.5,
                        color=info["color"], label="Degraded")
        ax.fill_between(bins, ref_h / ref_h.sum(), alpha=0.3,
                        color="#444444", label="Reference", linewidth=0)
        ax.step(bins, ref_h / ref_h.sum(), color="#111111", lw=1.5)
        ax.set_xlabel("Pixel intensity (0–255)"); ax.set_ylabel("Normalised freq.")
        ax.set_title(ds_name); ax.legend(fontsize=8)
    fig.suptitle("S8 — Pixel Intensity Histograms: Degraded vs Reference\n"
                 "(Degraded = narrower range, leftward shift = underexposed/dark)",
                 fontsize=12)
    save_fig(fig, "s8a_global_histograms.png")

    # ── 8b: brightness & entropy box plots ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, metric, title in zip(axes,
        ("brightness", "ent"),
        ("Mean Brightness (0–255)", "Histogram Entropy (higher=richer)")):
        data, labels, palette = [], [], []
        for ds_name, info in DATASETS.items():
            raw_v = df[(df["dataset"]==ds_name)&(df["role"]=="raw")][metric].values
            ref_v = df[(df["dataset"]==ds_name)&(df["role"]=="ref")][metric].values
            data += [raw_v, ref_v]
            labels += [f"{ds_name}\nDeg", f"{ds_name}\nRef"]
            palette += [info["color"], "#888888"]
        bp = ax.boxplot(data, patch_artist=True, labels=labels,
                        medianprops=dict(color="white", lw=2), widths=0.55)
        for patch, c in zip(bp["boxes"], palette):
            patch.set_facecolor(c); patch.set_alpha(0.75)
        ax.set_title(title)
    fig.suptitle("S8 — Brightness & Histogram Entropy", fontsize=12)
    save_fig(fig, "s8b_brightness_entropy.png")

    # ── 8c: clipped pixel fractions ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, title, color in zip(axes,
        ("dark_clip", "bright_clip"),
        ("Dark-Clipped Pixels (≤5)", "Bright-Clipped Pixels (≥250)"),
        ("#333399", "#cc3300")):
        for i, (ds_name, info) in enumerate(DATASETS.items()):
            raw_v = df[(df["dataset"]==ds_name)&(df["role"]=="raw")][metric].values
            ref_v = df[(df["dataset"]==ds_name)&(df["role"]=="ref")][metric].values
            x = i * 2
            ax.boxplot([raw_v, ref_v], positions=[x, x+0.7],
                       widths=0.5, patch_artist=True,
                       boxprops=dict(facecolor=info["color"], alpha=0.7),
                       medianprops=dict(color="white"))
        ax.set_xticks([0.35, 2.35, 4.35])
        ax.set_xticklabels(list(DATASETS.keys()))
        ax.set_ylabel("Fraction of pixels")
        ax.set_title(title)
    fig.suptitle("S8 — Clipped Pixel Fractions (Exposure Analysis)", fontsize=12)
    save_fig(fig, "s8c_clipped_pixels.png")

    # ── 8d: CDF comparison ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, ds_name, info in zip(axes, DATASETS.keys(), DATASETS.values()):
        raw_h = global_hists[ds_name]["raw"].astype(float)
        ref_h  = global_hists[ds_name]["ref"].astype(float)
        raw_cdf = np.cumsum(raw_h) / raw_h.sum()
        ref_cdf  = np.cumsum(ref_h) / ref_h.sum()
        ax.plot(raw_cdf, color=info["color"], lw=2, label="Degraded")
        ax.plot(ref_cdf, color="#333333", lw=2, ls="--", label="Reference")
        ax.set_xlabel("Pixel intensity"); ax.set_ylabel("CDF")
        ax.set_title(f"{ds_name}"); ax.legend(fontsize=8)
        ax.axhline(0.5, color="gray", lw=0.5, ls=":")
    fig.suptitle("S8 — Cumulative Distribution Function\n"
                 "(Shift of degraded CDF reveals under-exposure / colour bias)",
                 fontsize=12)
    save_fig(fig, "s8d_cdf_comparison.png")

    # ── 8e: mean image composites ────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    for row, ds_name in enumerate(DATASETS.keys()):
        for col, role in enumerate(("raw", "ref")):
            cnt = mean_imgs[ds_name]["cnt"][role]
            if cnt == 0: continue
            mean_img = np.clip(mean_imgs[ds_name][role] / cnt, 0, 1)
            label = "Degraded" if role == "raw" else "Reference"
            axes[row, col].imshow(mean_img)
            axes[row, col].set_title(f"{ds_name} — {label}\n(Mean of {cnt} images)")
            axes[row, col].axis("off")
    fig.suptitle("S8 — Mean Image Composites\n"
                 "(Reveals systematic spatial brightness patterns & colour bias)",
                 fontsize=13)
    save_fig(fig, "s8e_mean_composites.png")

    print("\n  Summary:")
    print(df.groupby(["dataset","role"])[["brightness","ent","dark_clip","bright_clip"]].mean().round(4).to_string())
    print("\n  ✔ Section 8 complete.")


if __name__ == "__main__":
    run()
