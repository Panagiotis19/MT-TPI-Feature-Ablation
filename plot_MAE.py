import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ============================================================
# MAE COMPARISON PLOT ALONG X, Y, Z AXES
# Creates separate MAE plots for each FH and WS combination
# ============================================================

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = "FH_WS"

FH_VALUES = [30, 60, 90, 120]
WS_VALUES = [60, 90, 120, 150]

MODELS = [
    ("Baseline",    "Baseline_Trajectory_Results.csv",    "Baseline",    "red"),
    ("Best_1-Feat", "Best_1-Feat_Trajectory_Results.csv", "Best 1-Feat", "green"),
    ("Best_2-Feat", "Best_2-Feat_Trajectory_Results.csv", "Best 2-Feat", "blue"),
    ("Best_3-Feat", "Best_3-Feat_Trajectory_Results.csv", "Best 3-Feat", "orange"),
    ("Best_4-Feat", "Best_4-Feat_Trajectory_Results.csv", "Best 4-Feat", "purple"),
]

FIGURE_SIZE = (10, 7)
SAVE_DPI = 300
OUTPUT_DIR = "Plots/MAE_Plots"

BAR_WIDTH = 0.18
SHOW_VALUES = True

CONVERT_LATLON_TO_METERS = True
REFERENCE_LATITUDE = 35.0


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_results_path(base_dir, model_subfolder, fh, ws, results_filename):
    return os.path.join(base_dir, f"FH{fh}_WS{ws}", model_subfolder, results_filename)


def latlon_error_to_meters(lon_error_deg, lat_error_deg, ref_lat=REFERENCE_LATITUDE):
    meters_per_degree_lat = 111320
    meters_per_degree_lon = 111320 * np.cos(np.radians(ref_lat))

    lon_error_m = lon_error_deg * meters_per_degree_lon
    lat_error_m = lat_error_deg * meters_per_degree_lat

    return lon_error_m, lat_error_m


def plot_mae_for_config(fh, ws, base_dir, models, output_dir):

    print(f"\nFH={fh}, WS={ws}:")

    axes_labels = ['X (Longitude)', 'Y (Latitude)', 'Z (Altitude)']
    x_positions = np.arange(len(axes_labels))

    plt.figure(figsize=FIGURE_SIZE)

    mae_results = {}

    for model_subfolder, results_filename, label, color in models:

        csv_path = get_results_path(base_dir, model_subfolder, fh, ws, results_filename)

        try:
            df = pd.read_csv(csv_path)

            abs_error_x = np.abs(df['pred_x'] - df['true_x'])
            abs_error_y = np.abs(df['pred_y'] - df['true_y'])
            abs_error_z = np.abs(df['pred_z'] - df['true_z'])

            if CONVERT_LATLON_TO_METERS:
                abs_error_x_m, abs_error_y_m = latlon_error_to_meters(
                    abs_error_x, abs_error_y
                )
                mae_x = np.mean(abs_error_x_m)
                mae_y = np.mean(abs_error_y_m)
            else:
                mae_x = np.mean(abs_error_x)
                mae_y = np.mean(abs_error_y)

            mae_z = np.mean(abs_error_z)

            mae_results[label] = {
                'values': [mae_x, mae_y, mae_z],
                'color': color
            }

            print(f"  ✓ {label}:")
            print(f"      MAE X: {mae_x:.4f} m")
            print(f"      MAE Y: {mae_y:.4f} m")
            print(f"      MAE Z: {mae_z:.4f} m")
            print(f"      Avg MAE: {(mae_x+mae_y+mae_z)/3:.4f} m")

        except FileNotFoundError:
            print(f"  ✗ {label}: File not found - {csv_path}")
            continue
        except Exception as e:
            print(f"  ✗ {label}: Error - {e}")
            continue

    if not mae_results:
        print("  No valid results.")
        plt.close()
        return False

    # Plot bars
    num_models = len(mae_results)
    total_width = BAR_WIDTH * num_models
    offsets = np.linspace(-total_width/2 + BAR_WIDTH/2,
                          total_width/2 - BAR_WIDTH/2,
                          num_models)

    for idx, (label, data) in enumerate(mae_results.items()):
        bars = plt.bar(x_positions + offsets[idx],
                       data['values'],
                       BAR_WIDTH,
                       label=label,
                       color=data['color'])

        if SHOW_VALUES:
            for bar, val in zip(bars, data['values']):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2.,
                         height,
                         f'{val:.2f}',
                         ha='center',
                         va='bottom',
                         fontsize=8)

    # Formatting
    plt.xlabel('Axis', fontsize=12)
    plt.ylabel('MAE (m)', fontsize=12)

    fh_seconds = fh * 0.1
    plt.title(f'MAE Comparison\n(Forecast Horizon={fh_seconds:.0f}s, Window Size={ws})',
              fontsize=14)

    plt.xticks(x_positions, axes_labels)
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, axis='y', alpha=0.3)
    plt.ylim(bottom=0)

    plt.tight_layout()

    filename = f"MAE_FH{fh}_WS{ws}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=SAVE_DPI, bbox_inches='tight')

    print(f"  → Saved: {filepath}")

    plt.close()
    return True


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("="*70)
    print("GENERATING MAE PLOTS FOR EACH FH/WS COMBINATION")
    print("="*70)

    for fh in FH_VALUES:
        for ws in WS_VALUES:
            plot_mae_for_config(fh, ws, BASE_DIR, MODELS, OUTPUT_DIR)

    print("\n" + "="*70)
    print("DONE!")
    print("="*70)


if __name__ == "__main__":
    main()