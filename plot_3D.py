import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import os

# ============================================================
# 3D TRAJECTORY PLOT - COMPARE MODELS ACROSS FH/WS
# ============================================================

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = "FH_WS"

FH_VALUES = [30, 60, 90, 120]
WS_VALUES = [60, 90, 120, 150]

# Format: (subfolder, results_filename, label, color, linestyle)
MODELS = [
    ("Baseline",    "Baseline_Trajectory_Results.csv",    "Baseline",    "red",    "--"),
    ("Best_1-Feat", "Best_1-Feat_Trajectory_Results.csv", "Best 1-Feat", "green",  "-."),
    ("Best_2-Feat", "Best_2-Feat_Trajectory_Results.csv", "Best 2-Feat", "blue",   ":"),
    ("Best_3-Feat", "Best_3-Feat_Trajectory_Results.csv", "Best 3-Feat", "orange", (0, (3, 1, 1, 1))),
    ("Best_4-Feat", "Best_4-Feat_Trajectory_Results.csv", "Best 4-Feat", "purple", (0, (5, 2))),
]

FIGURE_SIZE = (20, 8.5)
SAVE_DPI = 300
OUTPUT_DIR = "Plots/3D_Trajectory_Plots"

# Fraction of timesteps that belong to the complex path (first segment).
# 0.5 = 50/50 split. Increase if the complex path is longer than the linear one.
COMPLEX_RATIO = 0.5

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_results_path(base_dir, model_subfolder, fh, ws, results_filename):
    return os.path.join(base_dir, f"FH{fh}_WS{ws}", model_subfolder, results_filename)


def extract_first_step(df, fh):
    """Extract the first-step prediction from each window (t+1 only, no overlap noise)."""
    n_windows = len(df) // fh
    idx = np.arange(n_windows) * fh
    return df.iloc[idx].reset_index(drop=True)


def plot_path(ax, gt, models_data, title):
    """Plot GT (red) and all model predictions on a 3D axes."""
    ax.plot(gt['true_x'], gt['true_y'], gt['true_z'],
            color='red', linewidth=1.4, linestyle='-', solid_capstyle='round',
            label='Ground Truth (RTK)')

    for label, data, color, linestyle in models_data:
        ax.plot(data['pred_x'], data['pred_y'], data['pred_z'],
                color=color, linewidth=0.9, linestyle=linestyle,
                dash_capstyle='round',
                label=label)

    ax.set_xlabel('Longitude (dd)', fontsize=11, labelpad=10)
    ax.set_ylabel('Latitude (dd)',  fontsize=11, labelpad=10)
    ax.set_zlabel('Altitude (m)',   fontsize=11, labelpad=10)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=18)

    ax.legend(fontsize=10, loc='upper left',
              frameon=True, facecolor='white', edgecolor='lightgray', framealpha=0.95)
    ax.tick_params(labelsize=9, pad=4)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.view_init(elev=25, azim=-55)


def plot_3d_for_config(fh, ws, base_dir, models, output_dir):
    print(f"\nFH={fh}, WS={ws}:")

    # Load and extract first-step data for each model
    loaded = []
    for model_subfolder, results_filename, label, color, linestyle in models:
        csv_path = get_results_path(base_dir, model_subfolder, fh, ws, results_filename)
        try:
            df = pd.read_csv(csv_path)
            first = extract_first_step(df, fh)
            loaded.append((label, first, color, linestyle))
            print(f"  ✓ {label}: {len(first)} timesteps")
        except FileNotFoundError:
            print(f"  ✗ {label}: File not found - {csv_path}")
        except Exception as e:
            print(f"  ✗ {label}: Error - {e}")

    if not loaded:
        print(f"  No valid models for FH={fh}, WS={ws}")
        return False

    n_pts = len(loaded[0][1])
    mid = int(n_pts * COMPLEX_RATIO)
    complex_sl = slice(0, mid)
    linear_sl  = slice(mid, n_pts)

    gt = loaded[0][1]  # GT columns (true_x/y/z) are the same across models

    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor':   'white',
        'font.family':      'sans-serif',
        'font.size':        11,
    })

    fig = plt.figure(figsize=FIGURE_SIZE)

    fh_seconds = fh * 0.1

    # (a) Linear path — first half
    ax1 = fig.add_subplot(121, projection='3d')
    plot_path(ax1,
              gt.iloc[complex_sl],
              [(lbl, d.iloc[complex_sl], col, ls) for lbl, d, col, ls in loaded],
              f"(a) Trajectory Prediction - Linear Path\n(FH={fh_seconds:.0f}s, WS={ws})")

    # (b) Complex path — second half
    ax2 = fig.add_subplot(122, projection='3d')
    plot_path(ax2,
              gt.iloc[linear_sl],
              [(lbl, d.iloc[linear_sl], col, ls) for lbl, d, col, ls in loaded],
              f"(b) Trajectory Prediction - Complex Path\n(FH={fh_seconds:.0f}s, WS={ws})")

    plt.tight_layout(w_pad=5)

    filename = f"3D_Trajectory_FH{fh}_WS{ws}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=SAVE_DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  → Saved: {filepath}")
    return True


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}/")
    print(f"Base directory: {BASE_DIR}/")

    print("\n" + "="*70)
    print("GENERATING 3D TRAJECTORY PLOTS FOR EACH FH/WS COMBINATION")
    print("="*70)

    successful = 0
    for fh in FH_VALUES:
        for ws in WS_VALUES:
            if plot_3d_for_config(fh, ws, BASE_DIR, MODELS, OUTPUT_DIR):
                successful += 1

    print("\n" + "="*70)
    print(f"DONE! Generated {successful} plots in {OUTPUT_DIR}/")
    print("="*70)


if __name__ == "__main__":
    main()
