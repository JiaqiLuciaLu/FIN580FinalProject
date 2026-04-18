"""
Phase E: Generate figures and tables from Phase D outputs.

Usage:
    PYTHONPATH=. python src/notebooks/run_phase_e.py
"""

import matplotlib
matplotlib.use("Agg")

from src.code.plots import (
    figure_10a, figure_10b, figure_10c, figure_10d,
    figure_11, figure_12, FIGURE_DIR,
)

print("=== Phase E: Figures ===\n")

print("Generating Figure 10a (Validation SR vs K)...")
figure_10a()
print("  -> saved")

print("Generating Figure 10b (Training SR heatmap)...")
figure_10b()
print("  -> saved")

print("Generating Figure 10c (Testing SR vs K)...")
figure_10c()
print("  -> saved")

print("Generating Figure 10d (Testing SR heatmap)...")
figure_10d()
print("  -> saved")

print("Generating Figure 11 (Combined SDF weight map)...")
figure_11()
print("  -> saved")

print("Generating Figure 12 (Per-portfolio weight maps)...")
figure_12()
print("  -> saved")

print(f"\nDone. All figures in {FIGURE_DIR}")
