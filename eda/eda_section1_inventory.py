"""
eda_section1_inventory.py
─────────────────────────
WHAT: Catalog every image in the raw datasets.
WHY:  Before touching a single pixel you must know what you have:
      counts per dataset, file formats, naming conventions,
      whether raw↔reference pairs are 1-to-1, and whether any
      files are unreadable / corrupted.

TOOLS: os, cv2, PIL, pandas, matplotlib
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from eda_utils import DATASETS, OUTPUT_DIR, VALID_EXTS, list_images, save_fig

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from collections import Counter
from tqdm import tqdm


def run():
    print("\n" + "="*60)
    print("SECTION 1 - DATASET INVENTORY")
    print("="*60)

    rows = []
    format_counter = Counter()
    corrupt = []

    for ds_name, info in DATASETS.items():
        for role in ("raw", "ref"):
            folder = info[role]
            files  = list_images(folder)
            for fpath in tqdm(files, desc=f"{ds_name}/{role}", leave=False):
                ext = os.path.splitext(fpath)[1].lower()
                format_counter[ext] += 1
                # corruption check via PIL
                ok = True
                try:
                    with Image.open(fpath) as im:
                        im.verify()
                except Exception as e:
                    ok = False
                    corrupt.append((fpath, str(e)))
                rows.append({
                    "dataset": ds_name,
                    "role":    role,
                    "filename": os.path.basename(fpath),
                    "ext":     ext,
                    "readable": ok,
                })

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUTPUT_DIR}/s1_inventory.csv", index=False)

    # ── 1a: pair counts ───────────────────────────────────────────────────────
    counts = (df[df["readable"]]
              .groupby(["dataset", "role"])
              .size()
              .unstack(fill_value=0)
              .rename(columns={"raw": "Degraded (raw)", "ref": "Reference (ref)"}))
    print("\n  Image pair counts:")
    print(counts.to_string())

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(counts))
    w = 0.35
    b1 = ax.bar(x - w/2, counts["Degraded (raw)"],  w, label="Degraded",  color="#4C72B0")
    b2 = ax.bar(x + w/2, counts["Reference (ref)"], w, label="Reference", color="#55A868")
    for b in (b1, b2):
        for rect in b:
            ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 30,
                    f"{int(rect.get_height()):,}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(counts.index)
    ax.set_ylabel("Number of images")
    ax.set_title("S1 — Image Count per Dataset (Raw vs Reference)")
    ax.legend(); ax.set_ylim(0, counts.values.max() * 1.15)
    save_fig(fig, "s1a_pair_counts.png")

    # ── 1b: dataset proportion pie ────────────────────────────────────────────
    totals = df[df["role"] == "raw"].groupby("dataset").size()
    fig, ax = plt.subplots(figsize=(6, 6))
    wedge_colors = [DATASETS[d]["color"] for d in totals.index]
    wedges, texts, autotexts = ax.pie(
        totals.values, labels=totals.index, autopct="%1.1f%%",
        colors=wedge_colors, startangle=140,
        wedgeprops=dict(linewidth=1.2, edgecolor="white"))
    for at in autotexts: at.set_fontsize(10)
    ax.set_title(f"S1 — Dataset Proportions\n(Total: {totals.sum():,} pairs)")
    save_fig(fig, "s1b_dataset_proportions.png")

    # ── 1c: file-format distribution ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    exts = list(format_counter.keys())
    cnts = [format_counter[e] for e in exts]
    ax.bar(exts, cnts, color="#4C72B0", edgecolor="white")
    for i, (e, c) in enumerate(zip(exts, cnts)):
        ax.text(i, c + 20, str(c), ha="center", fontsize=9)
    ax.set_xlabel("File extension")
    ax.set_ylabel("Count")
    ax.set_title("S1 — File Format Distribution Across All Datasets")
    save_fig(fig, "s1c_file_formats.png")

    # ── 1d: pairing completeness ──────────────────────────────────────────────
    print("\n  Pairing completeness check:")
    for ds_name, info in DATASETS.items():
        raw_names = {os.path.splitext(os.path.basename(f))[0]
                     for f in list_images(info["raw"])}
        ref_names = {os.path.splitext(os.path.basename(f))[0]
                     for f in list_images(info["ref"])}
        matched   = raw_names & ref_names
        raw_only  = raw_names - ref_names
        ref_only  = ref_names - raw_names
        print(f"  {ds_name}: {len(matched)} matched, "
              f"{len(raw_only)} raw-only, {len(ref_only)} ref-only")

    # ── 1e: corruption report ─────────────────────────────────────────────────
    print(f"\n  Corrupted / unreadable files: {len(corrupt)}")
    for p, e in corrupt[:10]:
        print(f"    {p}: {e}")

    print("\n  ✔ Section 1 complete.")


if __name__ == "__main__":
    run()
