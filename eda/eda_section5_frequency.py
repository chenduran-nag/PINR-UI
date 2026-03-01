"""
eda_section5_frequency.py
─────────────────────────
WHAT: Analyse the spatial frequency content of raw vs reference images
      using 2-D Fast Fourier Transform (FFT).
WHY:  Haze, scattering, and blur all attenuate high-frequency signal.
      Degraded underwater images therefore concentrate energy in low
      frequencies. Quantifying this:
        • Justifies the LF-energy-ratio degradation metric
        • Shows that the network must learn to restore high-frequency
          texture (edges, fine structure) — motivating edge-weighted sampling
        • Reveals whether datasets differ in baseline frequency content

TOOLS: numpy (fft), cv2, scipy, matplotlib
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from eda_utils import DATASETS, OUTPUT_DIR, list_images, save_fig

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import ndimage
from tqdm import tqdm
import random


SAMPLE_N = 200


def sample_files(folder, n=SAMPLE_N, seed=42):
    random.seed(seed)
    files = list_images(folder)
    return random.sample(files, min(n, len(files)))


def fft_magnitude(img_gray, log=True):
    f  = np.fft.fft2(img_gray.astype(np.float64))
    fs = np.fft.fftshift(f)
    mag = np.abs(fs)
    return np.log1p(mag) if log else mag


def low_freq_ratio(img_gray, cutoff=0.1):
    """Fraction of total FFT energy contained in a circle of radius cutoff."""
    f  = np.fft.fft2(img_gray.astype(np.float64))
    fs = np.fft.fftshift(f)
    mag = np.abs(fs) ** 2
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    r = int(min(h, w) * cutoff)
    Y, X = np.ogrid[:h, :w]
    mask = (X - cx)**2 + (Y - cy)**2 <= r**2
    return float(mag[mask].sum() / mag.sum())


def radial_psd(img_gray, n_bins=64):
    """Azimuthally-averaged power spectral density (radial profile)."""
    f  = np.fft.fft2(img_gray.astype(np.float64))
    fs = np.fft.fftshift(f)
    psd = (np.abs(fs) ** 2)
    h, w = psd.shape
    cy, cx = h // 2, w // 2
    Y, X = np.mgrid[:h, :w]
    R = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(int)
    r_max = R.max()
    bins  = np.linspace(0, r_max, n_bins + 1)
    profile = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (R >= bins[i]) & (R < bins[i+1])
        if mask.sum() > 0:
            profile[i] = psd[mask].mean()
    return profile, bins[:-1]


def run():
    print("\n" + "="*60)
    print("SECTION 5 - FREQUENCY DOMAIN ANALYSIS (FFT)")
    print("="*60)

    lf_ratios = {}   # {dataset: {"raw": [...], "ref": [...]}}
    radial_profiles = {}

    for ds_name, info in DATASETS.items():
        lf_ratios[ds_name] = {"raw": [], "ref": []}
        radial_profiles[ds_name] = {"raw": None, "ref": None}

        for role, folder in (("raw", info["raw"]), ("ref", info["ref"])):
            files = sample_files(folder)
            ratios = []
            psd_acc = None
            for fpath in tqdm(files, desc=f"{ds_name}/{role}", leave=False):
                img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                img_rs = cv2.resize(img, (256, 256))
                ratios.append(low_freq_ratio(img_rs))
                prof, bins = radial_psd(img_rs)
                psd_acc = prof if psd_acc is None else psd_acc + prof
            lf_ratios[ds_name][role] = ratios
            if psd_acc is not None:
                radial_profiles[ds_name][role] = (psd_acc / len(files), bins)

    # ── 5a: LF ratio histogram (all datasets together) ────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for ds_name, info in DATASETS.items():
        raw_r = lf_ratios[ds_name]["raw"]
        ref_r = lf_ratios[ds_name]["ref"]
        ax.hist(raw_r, bins=30, alpha=0.5, color=info["color"],
                label=f"{ds_name} Degraded", edgecolor="none")
        ax.hist(ref_r, bins=30, alpha=0.3, color=info["color"],
                label=f"{ds_name} Reference", histtype="step", lw=2)
    ax.set_xlabel("Low-Frequency Energy Ratio (cutoff=10%)")
    ax.set_ylabel("Count")
    ax.set_title("S5 — FFT Low-Frequency Energy Ratio: Degraded vs Reference\n"
                 "(Higher = more energy in LF = more blur/haze)")
    ax.legend(fontsize=8)
    save_fig(fig, "s5a_lf_ratio_histogram.png")

    # ── 5b: mean LF ratio bar chart ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ds_names = list(DATASETS.keys())
    x = np.arange(len(ds_names)); w = 0.35
    raw_means = [np.mean(lf_ratios[d]["raw"]) for d in ds_names]
    ref_means = [np.mean(lf_ratios[d]["ref"]) for d in ds_names]
    b1 = ax.bar(x - w/2, raw_means, w, label="Degraded",
                color=[DATASETS[d]["color"] for d in ds_names], alpha=0.7)
    b2 = ax.bar(x + w/2, ref_means, w, label="Reference",
                color=[DATASETS[d]["color"] for d in ds_names], alpha=0.35, hatch="//")
    for b in (b1, b2):
        for rect in b:
            ax.text(rect.get_x() + rect.get_width()/2,
                    rect.get_height() + 0.002, f"{rect.get_height():.3f}",
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(ds_names)
    ax.set_ylabel("Mean Low-Frequency Ratio")
    ax.set_title("S5 — Mean LF Energy Ratio per Dataset\n"
                 "(Degraded consistently higher → more blur)")
    ax.legend(); ax.set_ylim(0, max(max(raw_means), max(ref_means)) * 1.2)
    save_fig(fig, "s5b_lf_ratio_mean_bar.png")

    # ── 5c: radial PSD profiles (log scale) ──────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, ds_name, info in zip(axes, DATASETS.keys(), DATASETS.values()):
        for role, ls in (("raw", "-"), ("ref", "--")):
            prof_data = radial_profiles[ds_name][role]
            if prof_data is None: continue
            prof, bins = prof_data
            norm = prof / prof.max()
            ax.semilogy(bins / bins.max(), norm, ls=ls, lw=2,
                        color=info["color"],
                        label=f"{'Degraded' if role=='raw' else 'Reference'}")
        ax.set_xlabel("Normalised Spatial Frequency")
        ax.set_ylabel("Normalised PSD (log)")
        ax.set_title(ds_name)
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1)
    fig.suptitle("S5 — Radial Power Spectral Density\n"
                 "(Degraded images fall off faster at high frequencies)",
                 fontsize=13)
    save_fig(fig, "s5c_radial_psd.png")

    # ── 5d: 2-D FFT spectrum visualisation on sample pairs ────────────────────
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for row, ds_name in enumerate(DATASETS.keys()):
        info = DATASETS[ds_name]
        raw_f = list_images(info["raw"])[0]
        ref_f  = list_images(info["ref"])[0]
        raw_g  = cv2.resize(cv2.imread(raw_f, cv2.IMREAD_GRAYSCALE), (256, 256))
        ref_g  = cv2.resize(cv2.imread(ref_f,  cv2.IMREAD_GRAYSCALE), (256, 256))

        axes[row, 0].imshow(raw_g, cmap="gray"); axes[row, 0].set_title(f"{ds_name} Degraded"); axes[row,0].axis("off")
        axes[row, 1].imshow(fft_magnitude(raw_g), cmap="inferno")
        axes[row, 1].set_title("FFT Spectrum (Degraded)\nEnergy → centre"); axes[row,1].axis("off")
        axes[row, 2].imshow(ref_g,  cmap="gray"); axes[row, 2].set_title(f"{ds_name} Reference"); axes[row,2].axis("off")
        axes[row, 3].imshow(fft_magnitude(ref_g), cmap="inferno")
        axes[row, 3].set_title("FFT Spectrum (Reference)\nMore spread"); axes[row,3].axis("off")

    fig.suptitle("S5 — 2-D FFT Magnitude Spectra\n"
                 "Degraded: energy concentrated at centre (low-freq dominant)\n"
                 "Reference: energy spreads outward (richer high-freq detail)",
                 fontsize=12)
    save_fig(fig, "s5d_fft_spectrum_grid.png")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n  LF ratio summary (mean ± std):")
    for ds_name in DATASETS:
        for role in ("raw", "ref"):
            vals = lf_ratios[ds_name][role]
            if vals:
                print(f"  {ds_name:8s} {role:3s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    print("\n  ✔ Section 5 complete.")


if __name__ == "__main__":
    run()
