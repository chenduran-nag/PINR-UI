"""
eda_section7_edges_texture.py
──────────────────────────────
WHAT: Analyse edge density, texture richness, and spatial content distribution
      across the raw datasets. Uses Sobel, Canny, LBP (Local Binary Patterns),
      and GLCM (Gray-Level Co-occurrence Matrix) texture descriptors.
WHY:  This section answers the question: "Where is the important information
      in these images, and how is it distributed spatially?"
        • High edge density in reference but not raw → haze destroys edges
        • Non-uniform spatial distribution of edges → uniform pixel sampling
          wastes capacity on blank/uniform regions
        • Texture richness differences between datasets → explains why
          edge-weighted coordinate sampling with 70% bias was chosen

TOOLS: cv2 (Sobel, Canny), scikit-image (LBP, GLCM, Sobel), numpy, matplotlib
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from eda_utils import DATASETS, OUTPUT_DIR, list_images, save_fig

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import random
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops


SAMPLE_N = 150
TARGET   = 256


def sample_files(folder, n=SAMPLE_N, seed=42):
    random.seed(seed)
    files = list_images(folder)
    return random.sample(files, min(n, len(files)))


def sobel_edge_density(img_gray):
    """Mean Sobel gradient magnitude (edge density)."""
    img_f = img_gray.astype(np.float64)
    gx = cv2.Sobel(img_f, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_f, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    return float(mag.mean()), mag


def canny_edge_fraction(img_gray):
    """Fraction of pixels flagged as edges by Canny."""
    edges = cv2.Canny(img_gray, 50, 150)
    return float(edges.mean() / 255.0), edges


def lbp_uniformity(img_gray):
    """LBP histogram entropy (texture complexity)."""
    lbp = local_binary_pattern(img_gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
    hist = hist + 1e-12
    entropy = -float(np.sum(hist * np.log2(hist)))
    return entropy


def glcm_contrast(img_gray):
    """GLCM contrast property (texture coarseness)."""
    img_q = (img_gray // 32).astype(np.uint8)   # quantise to 8 levels
    glcm  = graycomatrix(img_q, distances=[1], angles=[0], levels=8,
                          symmetric=True, normed=True)
    return float(graycoprops(glcm, "contrast")[0, 0])


def run():
    print("\n" + "="*60)
    print("SECTION 7 - EDGE, TEXTURE & SPATIAL CONTENT")
    print("="*60)

    records = []
    spatial_acc = {ds: {"raw": np.zeros((TARGET, TARGET)),
                        "ref": np.zeros((TARGET, TARGET)),
                        "cnt": {"raw": 0, "ref": 0}}
                   for ds in DATASETS}

    for ds_name, info in DATASETS.items():
        for role, folder in (("raw", info["raw"]), ("ref", info["ref"])):
            for fpath in tqdm(sample_files(folder), desc=f"{ds_name}/{role}", leave=False):
                img_bgr  = cv2.imread(fpath)
                if img_bgr is None: continue
                img_rs   = cv2.resize(img_bgr,  (TARGET, TARGET))
                img_gray = cv2.cvtColor(img_rs, cv2.COLOR_BGR2GRAY)

                ed, edge_map = sobel_edge_density(img_gray)
                cf, canny_map = canny_edge_fraction(img_gray)
                lbp_e = lbp_uniformity(img_gray)
                glcm_c = glcm_contrast(img_gray)

                # accumulate normalised edge map for spatial heatmap
                edge_norm = edge_map / (edge_map.max() + 1e-8)
                spatial_acc[ds_name][role] += edge_norm
                spatial_acc[ds_name]["cnt"][role] += 1

                records.append(dict(
                    dataset=ds_name, role=role,
                    edge_density=ed, canny_fraction=cf,
                    lbp_entropy=lbp_e, glcm_contrast=glcm_c
                ))

    import pandas as pd
    df = pd.DataFrame(records)
    df.to_csv(f"{OUTPUT_DIR}/s7_edge_texture.csv", index=False)

    # ── 7a: edge density box plots ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, metric, title in zip(axes,
        ("edge_density", "canny_fraction"),
        ("Sobel Edge Density (mean gradient mag)", "Canny Edge Fraction")):
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
        ax.set_title(title); ax.set_ylabel(metric)
    fig.suptitle("S7 — Edge Density: Degraded vs Reference\n"
                 "(Reference images consistently have higher edge content → "
                 "haze destroys edges)", fontsize=12)
    save_fig(fig, "s7a_edge_density_boxplot.png")

    # ── 7b: texture metrics (LBP entropy & GLCM) ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, metric, title in zip(axes,
        ("lbp_entropy", "glcm_contrast"),
        ("LBP Entropy (texture complexity)", "GLCM Contrast (coarseness)")):
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
    fig.suptitle("S7 — Texture Analysis: LBP Entropy & GLCM Contrast\n"
                 "(Degraded images have lower texture complexity)", fontsize=12)
    save_fig(fig, "s7b_texture_metrics.png")

    # ── 7c: spatial edge heatmaps ─────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    for row, ds_name in enumerate(DATASETS.keys()):
        for col, role in enumerate(("raw", "ref")):
            cnt = spatial_acc[ds_name]["cnt"][role]
            if cnt == 0: continue
            heatmap = spatial_acc[ds_name][role] / cnt
            im = axes[row, col].imshow(heatmap, cmap="inferno")
            label = "Degraded" if role == "raw" else "Reference"
            axes[row, col].set_title(f"{ds_name} — {label}\n"
                                     f"(mean edge density={heatmap.mean():.3f})")
            axes[row, col].axis("off")
            plt.colorbar(im, ax=axes[row, col], fraction=0.04)
    fig.suptitle("S7 — Spatial Edge Heatmaps\n"
                 "Shows WHERE edges concentrate → justifies edge-weighted sampling",
                 fontsize=13)
    save_fig(fig, "s7c_spatial_edge_heatmaps.png")

    # ── 7d: sample pair with edge overlays ───────────────────────────────────
    for ds_name, info in DATASETS.items():
        raw_f = list_images(info["raw"])[0]
        ref_f  = list_images(info["ref"])[0]
        raw_bgr = cv2.imread(raw_f); ref_bgr = cv2.imread(ref_f)
        if raw_bgr is None or ref_bgr is None: continue
        raw_rs  = cv2.resize(raw_bgr, (TARGET, TARGET))
        ref_rs   = cv2.resize(ref_bgr,  (TARGET, TARGET))
        raw_gray = cv2.cvtColor(raw_rs,  cv2.COLOR_BGR2GRAY)
        ref_gray  = cv2.cvtColor(ref_rs,  cv2.COLOR_BGR2GRAY)

        _, edge_raw = sobel_edge_density(raw_gray)
        _, edge_ref  = sobel_edge_density(ref_gray)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes[0,0].imshow(cv2.cvtColor(raw_rs, cv2.COLOR_BGR2RGB)); axes[0,0].set_title("Degraded"); axes[0,0].axis("off")
        axes[0,1].imshow(edge_raw, cmap="hot");  axes[0,1].set_title("Edge Map (Degraded)"); axes[0,1].axis("off")
        # overlay
        overlay_raw = cv2.cvtColor(raw_rs, cv2.COLOR_BGR2RGB).copy()
        edge_norm   = (edge_raw / edge_raw.max() * 255).astype(np.uint8)
        overlay_raw[:,:,0] = np.clip(overlay_raw[:,:,0].astype(int) + edge_norm.astype(int)*0.5, 0, 255)
        axes[0,2].imshow(overlay_raw); axes[0,2].set_title("Overlay"); axes[0,2].axis("off")

        axes[1,0].imshow(cv2.cvtColor(ref_rs, cv2.COLOR_BGR2RGB));  axes[1,0].set_title("Reference"); axes[1,0].axis("off")
        axes[1,1].imshow(edge_ref, cmap="hot");   axes[1,1].set_title("Edge Map (Reference)"); axes[1,1].axis("off")
        overlay_ref = cv2.cvtColor(ref_rs, cv2.COLOR_BGR2RGB).copy()
        edge_norm2  = (edge_ref / edge_ref.max() * 255).astype(np.uint8)
        overlay_ref[:,:,0] = np.clip(overlay_ref[:,:,0].astype(int) + edge_norm2.astype(int)*0.5, 0, 255)
        axes[1,2].imshow(overlay_ref); axes[1,2].set_title("Overlay"); axes[1,2].axis("off")

        fig.suptitle(f"S7 — Edge Detection Overlay: {ds_name}\n"
                     f"Degraded edge density: {edge_raw.mean():.2f} | "
                     f"Reference: {edge_ref.mean():.2f}", fontsize=12)
        save_fig(fig, f"s7d_edge_overlay_{ds_name}.png")

    # ── 7e: edge density vs colour cast (interaction) ────────────────────────
    import pandas as pd
    col_df = pd.read_csv(f"{OUTPUT_DIR}/s3_colour_stats.csv") if \
        os.path.exists(f"{OUTPUT_DIR}/s3_colour_stats.csv") else None

    print("\n  Edge density means:")
    print(df.groupby(["dataset","role"])[["edge_density","canny_fraction",
                                          "lbp_entropy"]].mean().round(4).to_string())
    print("\n  ✔ Section 7 complete.")


if __name__ == "__main__":
    run()
