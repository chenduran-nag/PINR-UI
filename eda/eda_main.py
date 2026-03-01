"""
eda_main.py — Master EDA Runner
================================
Run this from the project root with:
    python eda/eda_main.py

Executes all 11 EDA sections in order. Each section can also be run
independently: e.g. python eda/eda_section3_color.py

Requirements:
    pip install numpy opencv-python matplotlib seaborn pillow scipy
                scikit-image scikit-learn pandas tqdm

All outputs are saved to eda/outputs/
"""

import os
import sys
import time
import traceback

# Ensure we can import eda_utils regardless of where we run from
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
os.chdir(os.path.dirname(SCRIPT_DIR))   # set cwd to project root

os.makedirs(os.path.join(SCRIPT_DIR, "outputs"), exist_ok=True)

SECTIONS = [
    ("eda_section1_inventory",        "Dataset Inventory & Audit"),
    ("eda_section2_resolution",       "Resolution & Aspect Ratio"),
    ("eda_section3_color",            "Colour Space Analysis"),
    ("eda_section4_degradation",      "Degradation Metrics"),
    ("eda_section5_frequency",        "Frequency Domain (FFT)"),
    ("eda_section6_pairwise",         "Pairwise Quality (SSIM/PSNR/MSE)"),
    ("eda_section7_edges_texture",    "Edge, Texture & Spatial Content"),
    ("eda_section8_intensity",        "Pixel Intensity & Brightness"),
    ("eda_section9_domain_gap",       "Inter-Dataset Domain Gap"),
    ("eda_section10_sampling",        "Sampling Strategy Justification"),
    ("eda_section11_gallery_summary", "Gallery & Summary Table"),
]

BANNER = """
============================================================
UNDERWATER IMAGE RESTORATION - COMPLETE EDA PIPELINE
Physics-Informed Neural Restoration Project
============================================================
"""

print(BANNER)
print(f"  Working directory : {os.getcwd()}")
print(f"  EDA scripts       : {SCRIPT_DIR}")
print(f"  Outputs           : {os.path.join(SCRIPT_DIR, 'outputs')}")
print(f"  Sections to run   : {len(SECTIONS)}\n")

start_all = time.time()
failed = []

for i, (module_name, description) in enumerate(SECTIONS, 1):
    print(f"\n{'-'*60}")
    print(f"  [{i:02d}/{len(SECTIONS)}]  {description}")
    print(f"{'-'*60}")
    t0 = time.time()
    try:
        # Dynamic import and run
        import importlib
        mod = importlib.import_module(module_name)
        mod.run()
        elapsed = time.time() - t0
        print(f"  [{elapsed:.1f}s]")
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  FAILED: {module_name}")
        print(f"  Error: {e}")
        traceback.print_exc()
        failed.append((module_name, str(e)))

total_elapsed = time.time() - start_all

print(f"\n{'='*60}")
print(f"  EDA COMPLETE - Total time: {total_elapsed:.0f}s")
print(f"  Sections completed: {len(SECTIONS) - len(failed)}/{len(SECTIONS)}")
if failed:
    print(f"  Failed sections:")
    for name, err in failed:
        print(f"    - {name}: {err}")
print(f"\n  All outputs in: eda/outputs/")
print(f"{'='*60}")
