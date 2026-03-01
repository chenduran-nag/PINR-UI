"""
eda_section2_resolution.py
──────────────────────────
WHAT: Measure the resolution (W × H) and aspect ratio of every raw image.
WHY:  Neural networks require fixed-size inputs. This section tells us:
        • How diverse the resolutions are → justifies resizing
        • Whether images are portrait/landscape/square → informs crop strategy
        • Min/max resolution → choose a safe target (256 × 256)
        • How many images need upscaling vs downscaling → prefer crop for quality

TOOLS: cv2 (fast header-only reading), numpy, matplotlib, seaborn, pandas
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from eda_utils import DATASETS, OUTPUT_DIR, list_images, save_fig

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from tqdm import tqdm


def get_dims(fpath):
    """Read width, height without decoding pixel data (fast)."""
    img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, None
    h, w = img.shape[:2]
    return w, h


def run():
    print("\n" + "="*60)
    print("SECTION 2 - RESOLUTION & ASPECT RATIO")
    print("="*60)

    records = []
    for ds_name, info in DATASETS.items():
        files = list_images(info["raw"])
        for fpath in tqdm(files, desc=ds_name, leave=False):
            w, h = get_dims(fpath)
            if w is None:
                continue
            records.append({
                "dataset": ds_name,
                "width": w, "height": h,
                "aspect": round(w / h, 4),
                "mpix": round(w * h / 1e6, 3),
                "filename": os.path.basename(fpath),
            })

    df = pd.DataFrame(records)
    df.to_csv(f"{OUTPUT_DIR}/s2_resolutions.csv", index=False)

    TARGET = 256
    df["action"] = np.where(
        (df["width"] >= TARGET) & (df["height"] >= TARGET), "Center Crop",
        "Pad + Resize"
    )

    # ── 2a: summary statistics ────────────────────────────────────────────────
    print("\n  Resolution summary per dataset:")
    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]
        print(f"\n  {ds}  (n={len(sub):,})")
        for col in ("width", "height", "aspect", "mpix"):
            vals = sub[col]
            print(f"    {col:8s}  min={vals.min():.0f}  "
                  f"max={vals.max():.0f}  mean={vals.mean():.1f}  "
                  f"std={vals.std():.1f}")

    # ── 2b: scatter W × H coloured by dataset ────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    for ds_name, info in DATASETS.items():
        sub = df[df["dataset"] == ds_name]
        ax.scatter(sub["width"], sub["height"],
                   alpha=0.25, s=8, color=info["color"], label=ds_name)
    ax.axvline(TARGET, color="red", lw=1.5, ls="--", label=f"Target {TARGET}px")
    ax.axhline(TARGET, color="red", lw=1.5, ls="--")
    # diagonal = square
    lim = max(df["width"].max(), df["height"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color="gray", lw=0.8, ls=":", label="Square (W=H)")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Width (px)"); ax.set_ylabel("Height (px)")
    ax.set_title("S2 — Raw Image Resolutions (all datasets)\nRed lines = 256 px target")
    ax.legend(markerscale=3)
    save_fig(fig, "s2a_resolution_scatter.png")

    # ── 2c: histograms for width and height ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col, label in zip(axes, ("width", "height"), ("Width (px)", "Height (px)")):
        for ds_name, info in DATASETS.items():
            sub = df[df["dataset"] == ds_name]
            ax.hist(sub[col], bins=60, alpha=0.55,
                    color=info["color"], label=ds_name, edgecolor="none")
        ax.axvline(TARGET, color="red", lw=1.5, ls="--", label=f"{TARGET} target")
        ax.set_xlabel(label); ax.set_ylabel("Count")
        ax.legend(fontsize=8)
    axes[0].set_title("S2 — Width Distribution")
    axes[1].set_title("S2 — Height Distribution")
    fig.suptitle("S2 — Resolution Histograms (log-scaled if needed)", y=1.01)
    save_fig(fig, "s2b_resolution_histograms.png")

    # ── 2d: aspect ratio KDE ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for ds_name, info in DATASETS.items():
        sub = df[df["dataset"] == ds_name]
        sns.kdeplot(sub["aspect"], ax=ax, label=ds_name,
                    color=info["color"], fill=True, alpha=0.3, linewidth=1.5)
    ax.axvline(1.0, color="black", lw=1.2, ls="--", label="Square AR=1")
    ax.axvline(4/3, color="red", lw=1.2, ls=":", label="4:3 AR=1.33")
    ax.set_xlabel("Aspect ratio (W/H)")
    ax.set_title("S2 — Aspect Ratio Distribution (KDE)")
    ax.legend()
    save_fig(fig, "s2c_aspect_ratio_kde.png")

    # ── 2e: crop vs pad decision map ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    colors_map = {"Center Crop": "#4C72B0", "Pad + Resize": "#DD8452"}
    for action, grp in df.groupby("action"):
        ax.scatter(grp["width"], grp["height"],
                   alpha=0.25, s=7,
                   color=colors_map[action], label=action)
    ax.axvline(TARGET, color="black", lw=1, ls="--")
    ax.axhline(TARGET, color="black", lw=1, ls="--")
    ax.set_xlabel("Width (px)"); ax.set_ylabel("Height (px)")
    ax.set_title(f"S2 — Crop vs Pad Decision at {TARGET}×{TARGET} Target")
    counts_action = df["action"].value_counts()
    ax.legend(title="Strategy",
              labels=[f"{k}: {v:,} imgs" for k, v in counts_action.items()])
    save_fig(fig, "s2d_crop_vs_pad_map.png")

    # ── 2f: resolution diversity heatmap (2-D histogram) ─────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    h2d, xedges, yedges = np.histogram2d(df["width"], df["height"], bins=40)
    im = ax.imshow(np.log1p(h2d.T), origin="lower", aspect="auto",
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="log(count+1)")
    ax.axvline(TARGET, color="white", lw=1.5, ls="--")
    ax.axhline(TARGET, color="white", lw=1.5, ls="--")
    ax.set_xlabel("Width"); ax.set_ylabel("Height")
    ax.set_title("S2 — 2-D Resolution Density (all datasets combined)")
    save_fig(fig, "s2e_resolution_density_heatmap.png")

    print(f"\n  Images needing crop:  {(df['action']=='Center Crop').sum():,}")
    print(f"  Images needing pad:   {(df['action']=='Pad + Resize').sum():,}")
    print("\n  ✔ Section 2 complete.")


if __name__ == "__main__":
    run()
