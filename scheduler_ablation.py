#!/usr/bin/env python3
"""
Flexible Feature Ablation Scheduler
====================================
Runs Transformer_physicsinformed_pytorch.py for any combination of experiments:
baseline (no removals), single-feature removals, and multi-feature removals.

HOW TO DECLARE EXPERIMENTS
--------------------------
Edit the EXPERIMENTS section below. Each variable holds a list of tests for
that removal category. Each LINE in the list is a SEPARATE experiment.

  Baseline  = []                           # empty = run baseline (no removals)
  Feat_1    = ['wind_speed',               # test 1: remove wind_speed
               'wind_angle']               # test 2: remove wind_angle
  Feat_2    = ['wind_speed, wind_angle',   # test 1: remove wind_speed AND wind_angle
               'wind_speed, angular_x']    # test 2: remove wind_speed AND angular_x
  Feat_3    = ['wind_speed, wind_angle, battery_voltage']
  Feat_4    = ['wind_speed, wind_angle, battery_voltage, battery_current']

Rules:
  - For baseline: just include [] (an empty list).
  - For 1-feat: each string is a single feature name (one test per line).
  - For 2+ feat: each string contains feature names separated by ', '.
  - Set any variable to None to skip that category entirely.

OUTPUT STRUCTURE
----------------
    Baseline/
        Baseline_best_model.pt
        Baseline_final_model.pt
        ...
    Remove_Tests/
        Remove_1_feature/
            Removed_wind_speed/
                wind_speed_best_model.pt
                ...
            Removed_wind_angle/
                ...
        Remove_2_features/
            Removed_wind_speed_wind_angle/
                ...

Usage:
    python scheduler_ablation.py
"""

import os
import sys
import re
import subprocess
import time
from datetime import datetime, timedelta

# ============================================================================
# Configuration
# ============================================================================

# Path to the training script (must be in the same directory)
TRAINING_SCRIPT = "Transformer_physicsinformed_pytorch.py"

# Log file
BATCH_LOG_FILE = "ablation_scheduler_log.txt"

# Stop on first error or continue?
STOP_ON_ERROR = False

# ============================================================================
# All 20 input features (for reference / validation)
# ============================================================================
ALL_FEATURES = [
    'wind_speed', 'wind_angle', 'battery_voltage', 'battery_current',
    'position_x', 'position_y', 'position_z',
    'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w',
    'velocity_x', 'velocity_y', 'velocity_z',
    'angular_x', 'angular_y', 'angular_z',
    'linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z',
]

# ============================================================================
# >>>>>>>>>>>  EXPERIMENT DECLARATIONS — EDIT HERE  <<<<<<<<<<<<<
# ============================================================================
# Set any variable to None to skip that category.
# Set to [] for baseline (no removals).
# Each string in the list is ONE experiment.
# For multi-feature removals, separate features with ', ' inside the string.
#
# EXAMPLES:
#   Baseline = []                              # run baseline once
#   Baseline = None                            # skip baseline
#   Feat_1   = Feat_1 = ['wind_speed', 
#                      'wind_angle', 
#                      'battery_voltage', 
#                      'battery_current',
#                      'position_y', 
#                      'position_z', 
#                      'orientation_y', 
#                      'orientation_z', 
#                      'orientation_w', 
#                      'velocity_x', 
#                      'velocity_y', 
#                      'velocity_z', 
#                      'angular_x', 
#                      'angular_y', 
#                      'angular_z', 
#                      'linear_acceleration_x', 
#                      'linear_acceleration_y', 
#                      'linear_acceleration_z',]    # two 1-feat experiments
#   Feat_2   = ['position_y, orientation_z',
        #    'position_y, orientation_y',
        #    'position_y, wind_angle',
        #    'position_y, velocity_z',
        #    'position_y, wind_speed',
        #    'position_y, orientation_x',
        #    'position_y, linear_acceleration_y',
        #    'position_y, orientation_w',
        #    'position_y, angular_z',
        #    'orientation_z, orientation_y',
        #    'orientation_z, wind_angle',
        #    'orientation_z, velocity_z',
        #    'orientation_z, wind_speed',
        #    'orientation_z, orientation_x',
        #    'orientation_z, linear_acceleration_y',
        #    'orientation_z, orientation_w',
        #    'orientation_z, angular_z',
        #    'orientation_y, wind_angle',
        #    'orientation_y, velocity_z',
        #    'orientation_y, wind_speed',
        #    'orientation_y, orientation_x',
        #    'orientation_y, linear_acceleration_y',
        #    'orientation_y, orientation_w',
        #    'orientation_y, angular_z',
        #    'wind_angle, velocity_z',
        #    'wind_angle, wind_speed',
        #    'wind_angle, orientation_x',
        #    'wind_angle, linear_acceleration_y',
        #    'wind_angle, orientation_w',
        #    'wind_angle, angular_z',
        #    'velocity_z, wind_speed',
        #    'velocity_z, orientation_x',
        #    'velocity_z, linear_acceleration_y',
        #    'velocity_z, orientation_w',
        #    'velocity_z, angular_z',
        #    'wind_speed, orientation_x',
        #    'wind_speed, linear_acceleration_y',
        #    'wind_speed, orientation_w',
        #    'wind_speed, angular_z',
        #    'orientation_x, linear_acceleration_y',
        #    'orientation_x, orientation_w',
        #    'orientation_x, angular_z',
        #    'linear_acceleration_y, orientation_w',
        #    'linear_acceleration_y, angular_z',
        #    'orientation_w, angular_z']
#  Feat_3 = [
    # --- Pair rank #1: orientation_z, orientation_y (Δ MAE -0.0002) ---
#     'orientation_z, orientation_y, position_y',
#     'orientation_z, orientation_y, wind_angle',
#     'orientation_z, orientation_y, velocity_z',
#     'orientation_z, orientation_y, wind_speed',
#     'orientation_z, orientation_y, orientation_x',
#     'orientation_z, orientation_y, linear_acceleration_y',
#     'orientation_z, orientation_y, orientation_w',
#     'orientation_z, orientation_y, angular_z',
#     # --- Pair rank #2: wind_angle, wind_speed (Δ MAE -0.0001) ---
#     'wind_angle, wind_speed, position_y',
#     'wind_angle, wind_speed, orientation_z',
#     'wind_angle, wind_speed, orientation_y',
#     'wind_angle, wind_speed, velocity_z',
#     'wind_angle, wind_speed, orientation_x',
#     'wind_angle, wind_speed, linear_acceleration_y',
#     'wind_angle, wind_speed, orientation_w',
#     'wind_angle, wind_speed, angular_z',
#     # --- Pair rank #3: velocity_z, linear_acceleration_y (Δ MAE +0.0000) ---
#     'velocity_z, linear_acceleration_y, position_y',
#     'velocity_z, linear_acceleration_y, orientation_z',
#     'velocity_z, linear_acceleration_y, orientation_y',
#     'velocity_z, linear_acceleration_y, wind_angle',
#     'velocity_z, linear_acceleration_y, wind_speed',
#     'velocity_z, linear_acceleration_y, orientation_x',
#     'velocity_z, linear_acceleration_y, orientation_w',
#     'velocity_z, linear_acceleration_y, angular_z',
#     # --- Pair rank #4: wind_speed, angular_z (Δ MAE +0.0000) ---
#     'wind_speed, angular_z, position_y',
#     'wind_speed, angular_z, orientation_z',
#     'wind_speed, angular_z, orientation_y',
#     'wind_speed, angular_z, velocity_z',
#     'wind_speed, angular_z, orientation_x',
#     'wind_speed, angular_z, linear_acceleration_y',
#     'wind_speed, angular_z, orientation_w',
#     # wind_angle skipped — same as Pair #2 + angular_z
#     # --- Pair rank #5: orientation_x, angular_z (Δ MAE +0.0001) ---
#     'orientation_x, angular_z, position_y',
#     'orientation_x, angular_z, orientation_z',
#     'orientation_x, angular_z, orientation_y',
#     'orientation_x, angular_z, wind_angle',
#     'orientation_x, angular_z, velocity_z',
#     'orientation_x, angular_z, linear_acceleration_y',
#     'orientation_x, angular_z, orientation_w',
#     # wind_speed skipped — same as Pair #4 + orientation_x
#     # --- Pair rank #6: orientation_x, linear_acceleration_y (Δ MAE +0.0001) ---
#     'orientation_x, linear_acceleration_y, position_y',
#     'orientation_x, linear_acceleration_y, orientation_z',
#     'orientation_x, linear_acceleration_y, orientation_y',
#     'orientation_x, linear_acceleration_y, wind_angle',
#     'orientation_x, linear_acceleration_y, wind_speed',
#     'orientation_x, linear_acceleration_y, orientation_w',
#     # velocity_z skipped — same as Pair #3 + orientation_x
#     # angular_z skipped — same as Pair #5 + linear_acceleration_y
#     # --- Pair rank #7: orientation_y, wind_angle (Δ MAE +0.0001) ---
#     'orientation_y, wind_angle, position_y',
#     'orientation_y, wind_angle, velocity_z',
#     'orientation_y, wind_angle, orientation_x',
#     'orientation_y, wind_angle, linear_acceleration_y',
#     'orientation_y, wind_angle, orientation_w',
#     'orientation_y, wind_angle, angular_z',
#     # orientation_z skipped — same as Pair #1 + wind_angle
#     # wind_speed skipped — same as Pair #2 + orientation_y
#     # --- Pair rank #8: wind_angle, linear_acceleration_y (Δ MAE +0.0001) ---
#     'wind_angle, linear_acceleration_y, position_y',
#     'wind_angle, linear_acceleration_y, orientation_z',
#     'wind_angle, linear_acceleration_y, orientation_w',
#     'wind_angle, linear_acceleration_y, angular_z',
#     # orientation_y skipped — same as Pair #7 + linear_acceleration_y
#     # velocity_z skipped — same as Pair #3 + wind_angle
#     # wind_speed skipped — same as Pair #2 + linear_acceleration_y
#     # orientation_x skipped — same as Pair #6 + wind_angle
#     # --- Pair rank #9: wind_speed, linear_acceleration_y (Δ MAE +0.0001) ---
#     'wind_speed, linear_acceleration_y, position_y',
#     'wind_speed, linear_acceleration_y, orientation_z',
#     'wind_speed, linear_acceleration_y, orientation_y',
#     'wind_speed, linear_acceleration_y, orientation_w',
#     # wind_angle skipped — same as Pair #2 + linear_acceleration_y
#     # velocity_z skipped — same as Pair #3 + wind_speed
#     # orientation_x skipped — same as Pair #6 + wind_speed
#     # angular_z skipped — same as Pair #4 + linear_acceleration_y
#     # --- Pair rank #10: wind_speed, orientation_x (Δ MAE +0.0001) ---
#     'wind_speed, orientation_x, position_y',
#     'wind_speed, orientation_x, orientation_z',
#     'wind_speed, orientation_x, orientation_y',
#     'wind_speed, orientation_x, velocity_z',
#     'wind_speed, orientation_x, orientation_w',
#     # wind_angle skipped — same as Pair #2 + orientation_x
#     # linear_acceleration_y skipped — same as Pair #6 + wind_speed
#     # angular_z skipped — same as Pair #4 + orientation_x
# ]

# Feat_4 = [
#     # --- Triple rank #1: wind_angle, linear_acceleration_y, orientation_w ---
#     'wind_angle, linear_acceleration_y, orientation_w, position_y',
#     'wind_angle, linear_acceleration_y, orientation_w, orientation_z',
#     'wind_angle, linear_acceleration_y, orientation_w, orientation_y',
#     'wind_angle, linear_acceleration_y, orientation_w, velocity_z',
#     'wind_angle, linear_acceleration_y, orientation_w, wind_speed',
#     'wind_angle, linear_acceleration_y, orientation_w, orientation_x',
#     'wind_angle, linear_acceleration_y, orientation_w, angular_z',
#     # --- Triple rank #2: orientation_y, wind_angle, angular_z ---
#     'orientation_y, wind_angle, angular_z, position_y',
#     'orientation_y, wind_angle, angular_z, orientation_z',
#     'orientation_y, wind_angle, angular_z, velocity_z',
#     'orientation_y, wind_angle, angular_z, wind_speed',
#     'orientation_y, wind_angle, angular_z, orientation_x',
#     'orientation_y, wind_angle, angular_z, linear_acceleration_y',
#     'orientation_y, wind_angle, angular_z, orientation_w',
#     # --- Triple rank #3: wind_speed, angular_z, linear_acceleration_y ---
#     'wind_speed, angular_z, linear_acceleration_y, position_y',
#     'wind_speed, angular_z, linear_acceleration_y, orientation_z',
#     'wind_speed, angular_z, linear_acceleration_y, orientation_y',
#     'wind_speed, angular_z, linear_acceleration_y, wind_angle',
#     'wind_speed, angular_z, linear_acceleration_y, velocity_z',
#     'wind_speed, angular_z, linear_acceleration_y, orientation_x',
#     'wind_speed, angular_z, linear_acceleration_y, orientation_w',
#     # --- Triple rank #4: velocity_z, linear_acceleration_y, orientation_x ---
#     'velocity_z, linear_acceleration_y, orientation_x, position_y',
#     'velocity_z, linear_acceleration_y, orientation_x, orientation_z',
#     'velocity_z, linear_acceleration_y, orientation_x, orientation_y',
#     'velocity_z, linear_acceleration_y, orientation_x, wind_angle',
#     'velocity_z, linear_acceleration_y, orientation_x, wind_speed',
#     'velocity_z, linear_acceleration_y, orientation_x, orientation_w',
#     'velocity_z, linear_acceleration_y, orientation_x, angular_z',
#     # --- Triple rank #5: orientation_y, wind_angle, orientation_x ---
#     'orientation_y, wind_angle, orientation_x, position_y',
#     'orientation_y, wind_angle, orientation_x, orientation_z',
#     'orientation_y, wind_angle, orientation_x, velocity_z',
#     'orientation_y, wind_angle, orientation_x, wind_speed',
#     'orientation_y, wind_angle, orientation_x, linear_acceleration_y',
#     'orientation_y, wind_angle, orientation_x, orientation_w',
#     # --- Triple rank #6: orientation_x, angular_z, orientation_w ---
#     'orientation_x, angular_z, orientation_w, position_y',
#     'orientation_x, angular_z, orientation_w, orientation_z',
#     'orientation_x, angular_z, orientation_w, orientation_y',
#     'orientation_x, angular_z, orientation_w, wind_angle',
#     'orientation_x, angular_z, orientation_w, velocity_z',
#     'orientation_x, angular_z, orientation_w, wind_speed',
#     'orientation_x, angular_z, orientation_w, linear_acceleration_y',
#     # --- Triple rank #7: orientation_x, linear_acceleration_y, wind_angle ---
#     'orientation_x, linear_acceleration_y, wind_angle, position_y',
#     'orientation_x, linear_acceleration_y, wind_angle, orientation_z',
#     'orientation_x, linear_acceleration_y, wind_angle, wind_speed',
#     'orientation_x, linear_acceleration_y, wind_angle, angular_z',
#     # --- Triple rank #8: wind_angle, wind_speed, angular_z ---
#     'wind_angle, wind_speed, angular_z, position_y',
#     'wind_angle, wind_speed, angular_z, orientation_z',
#     'wind_angle, wind_speed, angular_z, velocity_z',
#     'wind_angle, wind_speed, angular_z, orientation_x',
#     'wind_angle, wind_speed, angular_z, orientation_w',
#     # --- Triple rank #9: velocity_z, linear_acceleration_y, orientation_y ---
#     'velocity_z, linear_acceleration_y, orientation_y, position_y',
#     'velocity_z, linear_acceleration_y, orientation_y, orientation_z',
#     'velocity_z, linear_acceleration_y, orientation_y, wind_angle',
#     'velocity_z, linear_acceleration_y, orientation_y, wind_speed',
#     'velocity_z, linear_acceleration_y, orientation_y, orientation_w',
#     'velocity_z, linear_acceleration_y, orientation_y, angular_z',
#     # --- Triple rank #10: wind_angle, wind_speed, orientation_w ---
#     'wind_angle, wind_speed, orientation_w, position_y',
#     'wind_angle, wind_speed, orientation_w, orientation_z',
#     'wind_angle, wind_speed, orientation_w, orientation_y',
#     'wind_angle, wind_speed, orientation_w, velocity_z',
#     'wind_angle, wind_speed, orientation_w, orientation_x',
# ]
# ============================================================================

Baseline = None

Feat_1 = None

Feat_2 = None

Feat_3 = None

Feat_4 = None

# ============================================================================
# END OF EXPERIMENT DECLARATIONS
# ============================================================================


# ============================================================================
# Build experiment list from declarations
# ============================================================================
def parse_experiments():
    """
    Parse the experiment declarations into a unified list.
    Each entry is a list of feature names to exclude (empty list = baseline).
    """
    experiments = []

    # Baseline
    if Baseline is not None:
        experiments.append([])  # empty list = no features excluded

    # 1-feature removals
    if Feat_1 is not None:
        for feat_str in Feat_1:
            feat_str = feat_str.strip()
            if feat_str:
                experiments.append([feat_str])

    # 2-feature removals
    if Feat_2 is not None:
        for feat_str in Feat_2:
            feats = [f.strip() for f in feat_str.split(',') if f.strip()]
            if len(feats) != 2:
                print(f"WARNING: Expected 2 features in '{feat_str}', got {len(feats)}. Skipping.")
                continue
            experiments.append(feats)

    # 3-feature removals
    if Feat_3 is not None:
        for feat_str in Feat_3:
            feats = [f.strip() for f in feat_str.split(',') if f.strip()]
            if len(feats) != 3:
                print(f"WARNING: Expected 3 features in '{feat_str}', got {len(feats)}. Skipping.")
                continue
            experiments.append(feats)

    # 4-feature removals
    if Feat_4 is not None:
        for feat_str in Feat_4:
            feats = [f.strip() for f in feat_str.split(',') if f.strip()]
            if len(feats) != 4:
                print(f"WARNING: Expected 4 features in '{feat_str}', got {len(feats)}. Skipping.")
                continue
            experiments.append(feats)

    return experiments

    


def validate_features(experiments):
    """Check that all feature names are valid."""
    valid = True
    for exp in experiments:
        for feat in exp:
            if feat not in ALL_FEATURES:
                print(f"ERROR: Unknown feature '{feat}' in experiment {exp}")
                valid = False
    return valid


# ============================================================================
# Helpers
# ============================================================================
def format_duration(seconds):
    return str(timedelta(seconds=int(seconds)))


def experiment_label(excluded_features):
    """Human-readable label for an experiment."""
    if not excluded_features:
        return "Baseline (all 20 features)"
    return f"Remove {len(excluded_features)}: {' + '.join(excluded_features)}"


def create_temp_script(excluded_features):
    """
    Create a temporary copy of the training script with EXCLUDED_FEATURES
    set to the specified list.
    """
    with open(TRAINING_SCRIPT, 'r') as f:
        content = f.read()

    if excluded_features:
        features_str = "['" + "', '".join(excluded_features) + "']"
    else:
        features_str = "[]"

    # Replace the EXCLUDED_FEATURES definition
    pattern = r'EXCLUDED_FEATURES\s*=\s*\[[\s\S]*?\]'
    replacement = f'EXCLUDED_FEATURES = {features_str}'
    new_content = re.sub(pattern, replacement, content, count=1)

    temp_script = "_temp_ablation_run.py"
    with open(temp_script, 'w') as f:
        f.write(new_content)

    return temp_script


def run_experiment(excluded_features, exp_num, total_exp):
    """Run a single ablation experiment."""
    label = experiment_label(excluded_features)

    print("\n" + "=" * 70)
    print(f"EXPERIMENT {exp_num}/{total_exp}")
    print(f"  {label}")
    print(f"  Remaining features: {20 - len(excluded_features)}")
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")

    temp_script = create_temp_script(excluded_features)
    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, temp_script],
            check=True,
        )
        success = True
        error_msg = None
    except subprocess.CalledProcessError as e:
        success = False
        error_msg = str(e)
    except Exception as e:
        success = False
        error_msg = str(e)
    finally:
        if os.path.exists(temp_script):
            os.remove(temp_script)

    duration = time.time() - start_time

    return {
        'label': label,
        'excluded': excluded_features,
        'success': success,
        'duration': duration,
        'error': error_msg,
    }


# ============================================================================
# Main
# ============================================================================
def main():
    if not os.path.exists(TRAINING_SCRIPT):
        print(f"ERROR: Training script '{TRAINING_SCRIPT}' not found!")
        print("Make sure scheduler_ablation.py is in the same directory as")
        print(f"  {TRAINING_SCRIPT}")
        sys.exit(1)

    # Parse and validate
    experiments = parse_experiments()

    if not experiments:
        print("ERROR: No experiments declared! Edit the EXPERIMENT DECLARATIONS section.")
        sys.exit(1)

    if not validate_features(experiments):
        print("\nERROR: Invalid feature names detected. Please fix and re-run.")
        sys.exit(1)

    total = len(experiments)

    # Count by category
    n_baseline = sum(1 for e in experiments if len(e) == 0)
    n_1feat    = sum(1 for e in experiments if len(e) == 1)
    n_2feat    = sum(1 for e in experiments if len(e) == 2)
    n_3feat    = sum(1 for e in experiments if len(e) == 3)
    n_4feat    = sum(1 for e in experiments if len(e) == 4)

    print("=" * 70)
    print("FEATURE ABLATION SCHEDULER")
    print("=" * 70)
    print(f"Training script:     {TRAINING_SCRIPT}")
    print(f"Total experiments:   {total}")
    if n_baseline: print(f"  Baseline:          {n_baseline}")
    if n_1feat:    print(f"  1-feat removals:   {n_1feat}")
    if n_2feat:    print(f"  2-feat removals:   {n_2feat}")
    if n_3feat:    print(f"  3-feat removals:   {n_3feat}")
    if n_4feat:    print(f"  4-feat removals:   {n_4feat}")
    print(f"Stop on error:       {STOP_ON_ERROR}")
    print(f"Started at:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    print("\nQueued experiments:")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i:3d}. {experiment_label(exp)}")
    print()

    # Run experiments
    results = []
    batch_start = time.time()

    for i, excluded in enumerate(experiments, 1):
        result = run_experiment(excluded, i, total)
        results.append(result)

        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        print(f"\n{status} — {result['label']} — Duration: {format_duration(result['duration'])}")

        if not result['success']:
            print(f"  Error: {result['error']}")
            if STOP_ON_ERROR:
                print("\nStopping batch (STOP_ON_ERROR=True)")
                break

        # Time estimate
        if i < total:
            avg = (time.time() - batch_start) / i
            remaining = avg * (total - i)
            print(f"  Estimated remaining: {format_duration(remaining)}")

    # ---- Summary ----
    batch_duration = time.time() - batch_start
    successful = sum(1 for r in results if r['success'])
    failed     = sum(1 for r in results if not r['success'])

    summary = f"""
{'=' * 70}
ABLATION BATCH COMPLETE
{'=' * 70}
Total experiments:  {total}
Successful:         {successful}
Failed:             {failed}
Total duration:     {format_duration(batch_duration)}
Finished at:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RESULTS:
{'-' * 70}
"""
    for i, r in enumerate(results, 1):
        icon = "✓" if r['success'] else "✗"
        summary += f"  {icon} {i:3d}. {r['label']:<50} ({format_duration(r['duration'])})\n"
        if not r['success']:
            summary += f"        Error: {r['error']}\n"

    summary += "=" * 70
    print(summary)

    with open(BATCH_LOG_FILE, 'w') as f:
        f.write(summary)
    print(f"\nLog saved to: {BATCH_LOG_FILE}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()