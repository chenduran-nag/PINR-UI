"""
eda_utils.py - Shared utilities for all EDA sections.
Defines dataset paths, color palette, and helper functions.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 130,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

# ── Dataset paths ─────────────────────────────────────────────────────────────
DATASETS = {
    "UIEB": {
        "raw": "data/raw/UIEB/raw",
        "ref": "data/raw/UIEB/reference",
        "color": "#4C72B0",
    },
    "SUIM-E": {
        "raw": "data/raw/SUIM-E/raw",
        "ref": "data/raw/SUIM-E/reference",
        "color": "#DD8452",
    },
    "EUVP": {
        "raw": "data/raw/EUVP/raw",
        "ref": "data/raw/EUVP/reference",
        "color": "#55A868",
    },
}

OUTPUT_DIR = "eda/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def list_images(folder):
    """Return sorted list of valid image file paths in a folder."""
    if not os.path.isdir(folder):
        return []
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in VALID_EXTS
    ])


def load_rgb(path, size=None):
    """Load image as float32 RGB array, optionally resize."""
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if size is not None:
        img = cv2.resize(img, (size, size))
    return img.astype(np.float32) / 255.0


def save_fig(fig, name, close=True):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  [SAVED] {path}")
    if close:
        plt.close(fig)
