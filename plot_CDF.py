import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ============================================================
# CDF COMPARISON PLOT - BY FORECAST HORIZON (FH) AND WINDOW SIZE (WS)
# ============================================================
# Creates separate CDF plots for each FH and WS combination
# 
# File structure:
# - Baseline: FH_WS/FH{fh}_WS{ws}/Baseline/Baseline_Trajectory_Results.csv
# - Best N-Feat: FH_WS/FH{fh}_WS{ws}/Best_N-Feat/Best_N-Feat_Trajectory_Results.csv
# ============================================================

# ============================================================
# CONFIGURATION
# ============================================================

# Base directory where all FH_WS results are stored
BASE_DIR = "FH_WS"

# FH and WS combinations to plot
FH_VALUES = [30, 60, 90, 120]  # Forecast horizons (in timesteps)
WS_VALUES = [60, 90, 120, 150]  # Window sizes

# Models to compare (will be applied to each FH/WS combination)
# Format: (subfolder_pattern, results_file_pattern, label, color, linestyle)
MODELS = [
    ("Baseline",    "Baseline_Trajectory_Results.csv",    "Baseline",    "red",    "-"),
    ("Best_1-Feat", "Best_1-Feat_Trajectory_Results.csv", "Best 1-Feat", "green",  "--"),
    ("Best_2-Feat", "Best_2-Feat_Trajectory_Results.csv", "Best 2-Feat", "blue",   "-."),
    ("Best_3-Feat", "Best_3-Feat_Trajectory_Results.csv", "Best 3-Feat", "orange", ":"),
    ("Best_4-Feat", "Best_4-Feat_Trajectory_Results.csv", "Best 4-Feat", "purple", "-"),
]

# ============================================================
# PLOT CONFIGURATION
# ============================================================
FIGURE_SIZE = (10, 7)
SAVE_DPI = 300
OUTPUT_DIR = "Plots/CDF_Plots"

# X-axis limit (set to None for auto, or a number like 2.0)
X_LIMIT = None

# Percentile lines to draw
PERCENTILE_LINES = [95]

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def compute_cdf(errors):
    """Compute CDF from error values."""
    sorted_errors = np.sort(errors)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    return sorted_errors, cdf


def get_results_path(base_dir, model_subfolder, fh, ws, results_filename):
    """Construct the full path to a results file."""
    # Path: BASE_DIR/FH{fh}_WS{ws}/model_subfolder/results_filename
    return os.path.join(base_dir, f"FH{fh}_WS{ws}", model_subfolder, results_filename)


def plot_cdf_for_config(fh, ws, base_dir, models, output_dir):
    """Create a CDF plot for a specific FH/WS configuration."""
    
    print(f"\nFH={fh}, WS={ws}:")
    
    plt.figure(figsize=FIGURE_SIZE)
    
    percentile_values = {}
    valid_results = 0
    
    for model_subfolder, results_filename, label, color, linestyle in models:
        csv_path = get_results_path(base_dir, model_subfolder, fh, ws, results_filename)
        
        try:
            # Read the CSV file
            df = pd.read_csv(csv_path)
            errors = df['euclidean_error_m'].values
            
            # Compute CDF
            sorted_errors, cdf = compute_cdf(errors)
            
            # Plot CDF
            plt.plot(sorted_errors, cdf, label=label, color=color, 
                    linestyle=linestyle, linewidth=2)
            
            # Calculate percentiles
            percentile_values[label] = {}
            for p in PERCENTILE_LINES:
                pval = np.percentile(errors, p)
                percentile_values[label][p] = pval
            
            # Print statistics
            print(f"  ✓ {label}:")
            print(f"      Mean: {errors.mean():.4f} m, Median: {np.median(errors):.4f} m, 95th: {percentile_values[label][95]:.4f} m")
            
            valid_results += 1
                
        except FileNotFoundError:
            print(f"  ✗ {label}: File not found - {csv_path}")
            continue
        except Exception as e:
            print(f"  ✗ {label}: Error - {e}")
            continue
    
    if valid_results == 0:
        print(f"  No valid results for FH={fh}, WS={ws}")
        plt.close()
        return False
    
    # Draw percentile reference lines
    for p in PERCENTILE_LINES:
        plt.axhline(y=p/100, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        plt.text(0.02, p/100 + 0.02, f'{p}%', fontsize=9, color='gray')
    
    # Formatting
    plt.xlabel('Euclidean Distance (m)', fontsize=12)
    plt.ylabel('CDF', fontsize=12)
    
    # Convert FH to seconds (assuming 10 Hz = 0.1s per timestep)
    fh_seconds = fh * 0.1
    plt.title(f'CDF of Trajectory Prediction Error\n(Forecast Horizon={fh_seconds:.0f}s, Window Size={ws})', fontsize=14)
    
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    
    if X_LIMIT:
        plt.xlim(0, X_LIMIT)
    
    plt.tight_layout()
    
    # Save plot
    filename = f"CDF_FH{fh}_WS{ws}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=SAVE_DPI, bbox_inches='tight')
    print(f"  → Saved: {filepath}")
    
    plt.close()
    return True

# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}/")
    print(f"Base directory: {BASE_DIR}/")
    
    print("\n" + "="*70)
    print("GENERATING INDIVIDUAL CDF PLOTS FOR EACH FH/WS COMBINATION")
    print("="*70)
    
    # Generate individual plots for each FH/WS combination
    for fh in FH_VALUES:
        for ws in WS_VALUES:
            plot_cdf_for_config(fh, ws, BASE_DIR, MODELS, OUTPUT_DIR)

    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)


if __name__ == "__main__":
    main()