"""
eda_section11_gallery_summary.py
──────────────────────────────────
WHAT: Generate a visual summary gallery showing representative image pairs
      from each dataset with all key metrics annotated, plus a final
      cross-section summary table that consolidates EDA findings into
      a single reference table mapping each finding → pipeline decision.

WHY:  This is the final EDA deliverable — a human-readable summary that
      ties every quantitative finding back to a concrete design choice
      in the data pipeline.

TOOLS: cv2, matplotlib, pandas
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from eda_utils import DATASETS, OUTPUT_DIR, list_images, save_fig

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import pandas as pd
import random


def sample_files(folder, n=6, seed=0):
    random.seed(seed)
    files = list_images(folder)
    return random.sample(files, min(n, len(files)))


def annotate_image(ax, img_rgb, title, metrics_text):
    ax.imshow(img_rgb)
    ax.set_title(title, fontsize=8, pad=3)
    ax.axis("off")
    ax.text(0.02, 0.02, metrics_text,
            transform=ax.transAxes, fontsize=6,
            verticalalignment='bottom',
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.55),
            color="white")


def quick_metrics(fpath):
    img_bgr = cv2.imread(fpath)
    if img_bgr is None: return "N/A"
    img_rs  = cv2.resize(img_bgr, (128, 128))
    img_rgb = cv2.cvtColor(img_rs, cv2.COLOR_BGR2RGB).astype(np.float32)/255.
    img_gray = cv2.cvtColor(img_rs, cv2.COLOR_BGR2GRAY)
    r, g, b  = img_rgb[:,:,0].mean(), img_rgb[:,:,1].mean(), img_rgb[:,:,2].mean()
    cast     = np.std([r, g, b])
    contrast = img_gray.std() / 255.
    haze     = float(img_rs.astype(np.float32).min(axis=2).mean()/255.)
    h_orig, w_orig = cv2.imread(fpath).shape[:2]
    return (f"Res: {w_orig}×{h_orig}\n"
            f"R:{r:.2f} G:{g:.2f} B:{b:.2f}\n"
            f"Cast:{cast:.3f} Contrast:{contrast:.3f}\n"
            f"Haze:{haze:.3f}")


def run():
    print("\n" + "="*60)
    print("SECTION 11 - GALLERY & SUMMARY")
    print("="*60)

    # ── 11a: representative pairs gallery ─────────────────────────────────────
    N_PAIRS = 4
    fig = plt.figure(figsize=(20, 14))
    gs  = gridspec.GridSpec(len(DATASETS)*2, N_PAIRS*2, figure=fig,
                            hspace=0.4, wspace=0.1)

    for row_ds, (ds_name, info) in enumerate(DATASETS.items()):
        raw_files = sample_files(info["raw"], n=N_PAIRS)
        ref_files  = sample_files(info["ref"],  n=N_PAIRS)

        for col, (raw_f, ref_f) in enumerate(zip(raw_files, ref_files)):
            raw_img = cv2.cvtColor(cv2.resize(cv2.imread(raw_f), (160,120)), cv2.COLOR_BGR2RGB)
            ref_img  = cv2.cvtColor(cv2.resize(cv2.imread(ref_f),  (160,120)), cv2.COLOR_BGR2RGB)

            ax_raw = fig.add_subplot(gs[row_ds*2,     col*2: col*2+2])
            ax_ref  = fig.add_subplot(gs[row_ds*2 + 1, col*2: col*2+2])

            annotate_image(ax_raw, raw_img,
                           f"{ds_name} Degraded #{col+1}",
                           quick_metrics(raw_f))
            annotate_image(ax_ref, ref_img,
                           f"{ds_name} Reference #{col+1}",
                           quick_metrics(ref_f))

    fig.suptitle("S11 — Representative Image Pairs Gallery\n"
                 "(All three datasets: degraded on top, reference below)",
                 fontsize=14, y=1.01)
    save_fig(fig, "s11a_representative_gallery.png")

    # ── 11b: findings → decisions table ──────────────────────────────────────
    decisions = [
        ("Dataset imbalance\n(UIEB 890, SUIM-E 1525, EUVP 9023)",
         "Per-dataset stratified 70/15/15 split before merging"),
        ("Resolution heterogeneity\n(varies from ~200px to >1000px)",
         "Smart resize: center-crop if ≥256, pad+resize if <256 → 256×256"),
        ("Aspect ratios near 1:1 for majority",
         "Square crop is safe; minimal content loss"),
        ("Blue/green channel dominance in degraded images",
         "Colour cast (std of channel means) stored as degradation metric"),
        ("High dark channel values in degraded images",
         "Dark channel prior used as haze density metric"),
        ("Degraded images: lower Laplacian variance (blurry)",
         "Laplacian var stored as sharpness metric in JSON"),
        ("Low-frequency energy ratio higher in degraded images",
         "FFT LF ratio stored as degradation metric"),
        ("SSIM < 0.9 and PSNR < 30 dB across all datasets",
         "Confirms need for supervised neural restoration (not just filtering)"),
        ("Edges concentrated in non-uniform spatial regions",
         "Edge-weighted sampling (70% bias) for training coordinates"),
        ("High spatial autocorrelation at small pixel lags",
         "Spread sampling (not dense patches) avoids redundancy"),
        ("N=2048 samples achieves ~90% effective coverage",
         "2048 samples per image chosen as training sample count"),
        ("RTX 2060 (6 GB VRAM): batch×2048×8 features ≈ 100–200 MB",
         "Batch size 8 chosen; coordinate generation on CPU to avoid CUDA issues"),
        ("Datasets occupy distinct feature-space clusters (PCA/t-SNE)",
         "Multi-dataset training increases diversity; stratified split preserves each"),
        ("Reference images have 2–3× higher edge density than degraded",
         "Edge-weighted sampling trains network to prioritise edge restoration"),
    ]

    df = pd.DataFrame(decisions, columns=["EDA Finding", "Pipeline Decision"])
    df.to_csv(f"{OUTPUT_DIR}/s11_findings_decisions.csv", index=False)

    # Render as table figure
    fig, ax = plt.subplots(figsize=(18, len(decisions) * 0.7 + 2))
    ax.axis("off")
    col_widths = [0.45, 0.55]
    tbl = ax.table(
        cellText=df.values.tolist(),
        colLabels=["📊 EDA Finding", "⚙️  Pipeline Decision"],
        cellLoc="left", loc="center",
        colWidths=col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.2)
    # Style header
    for col in range(2):
        tbl[(0, col)].set_facecolor("#2C3E50")
        tbl[(0, col)].set_text_props(color="white", fontweight="bold")
    # Alternate row colours
    for row in range(1, len(decisions)+1):
        colour = "#EAF2FB" if row % 2 == 0 else "#FDFEFE"
        for col in range(2):
            tbl[(row, col)].set_facecolor(colour)
    ax.set_title("S11 — EDA Findings → Pipeline Design Decisions",
                 fontsize=14, pad=20, fontweight="bold")
    save_fig(fig, "s11b_findings_decisions_table.png")

    print(f"\n  Saved findings table → {OUTPUT_DIR}/s11_findings_decisions.csv")
    print("\n  ✔ Section 11 complete.")
    print("\n" + "="*60)
    print("ALL EDA SECTIONS COMPLETE")
    print(f"All outputs saved to: {OUTPUT_DIR}/")
    print("="*60)


if __name__ == "__main__":
    run()
