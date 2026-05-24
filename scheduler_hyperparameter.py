#!/usr/bin/env python3
"""
Hyperparameter Sweep Scheduler
================================
Runs Transformer_physicsinformed_pytorch.py for the Baseline and the best
1/2/3/4-feature-removal configurations across every combination of
Forecast Horizon and Window Size.

Configurations tested
---------------------
  Baseline   : all 20 features
  Best 1-Feat: remove position_y
  Best 2-Feat: remove orientation_z, orientation_y
  Best 3-Feat: remove wind_angle, linear_acceleration_y, orientation_w
  Best 4-Feat: remove orientation_x, linear_acceleration_y, wind_angle, angular_z

Grid
----
  FORECAST_HORIZONS : [30, 60, 90, 120]
  WINDOW_SIZES      : [60, 90, 120, 150]
  Total runs        : 5 configs × 9 FH/WS pairs = 45  (WS > FH, FH=30/WS=90 excluded)

Usage
-----
    python scheduler_hyperparameter.py
"""

import os
import sys
import re
import json
import subprocess
import time
from datetime import datetime, timedelta

# ============================================================================
# Configuration
# ============================================================================

TRAINING_SCRIPT   = "Transformer_physicsinformed_pytorch.py"
BATCH_LOG_FILE    = "hyperparameter_scheduler_log.txt"
PROGRESS_FILE     = "hyperparameter_progress.json"   # resume checkpoint
STOP_ON_ERROR     = False

# ============================================================================
# Sweep grid
# ============================================================================

FORECAST_HORIZONS = [30, 60, 90, 120]
WINDOW_SIZES      = [60, 90, 120, 150]

# ============================================================================
# Experiment configurations (excluded features per run)
# ============================================================================

CONFIGURATIONS = [
    {
        'name':     'Baseline',
        'excluded': [],
    },
    {
        'name':     'Best_1-Feat',
        'excluded': ['position_y'],
    },
    {
        'name':     'Best_2-Feat',
        'excluded': ['orientation_z', 'orientation_y'],
    },
    {
        'name':     'Best_3-Feat',
        'excluded': ['orientation_y', 'wind_angle', 'angular_z'],
    },
    {
        'name':     'Best_4-Feat',
        'excluded': ['orientation_y', 'wind_angle', 'angular_z', 'linear_acceleration_y'],
    },
]

# ============================================================================
# Helpers
# ============================================================================

# ============================================================================
# Checkpoint helpers
# ============================================================================

def checkpoint_key(config, fh, ws):
    """Unique string key identifying one experiment."""
    return f"{config['name']}__FH{fh}__WS{ws}"


def load_progress():
    """Return the set of already-completed experiment keys."""
    if not os.path.exists(PROGRESS_FILE):
        return {}
    with open(PROGRESS_FILE) as f:
        return json.load(f)


def save_progress(progress):
    """Persist the completed-experiments dict to disk."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


# ============================================================================
# Helpers
# ============================================================================

def format_duration(seconds):
    return str(timedelta(seconds=int(seconds)))


def experiment_label(config, fh, ws):
    excl = config['excluded']
    feat_part = "all 20 features" if not excl else f"-{', -'.join(excl)}"
    return f"{config['name']} | FH={fh} WS={ws} | {feat_part}"


def create_temp_script(config, fh, ws):
    """
    Write a temporary training script with EXCLUDED_FEATURES, forecast_horizon,
    WINDOW_SIZE, OUTPUT_DIR, and the file prefix patched for this experiment.

    Output structure produced:
        FH_WS/
            FH{fh}_WS{ws}/
                {config['name']}/
                    {config['name']}_best_model.pt
                    {config['name']}_final_model.pt
                    {config['name']}_Summary.txt
                    {config['name']}_Trajectory_Results.csv
                    {config['name']}_Classification_Results.csv
                    {config['name']}_Confusion_Matrix.png
    """
    with open(TRAINING_SCRIPT, 'r') as f:
        content = f.read()

    name = config['name']
    excl = config['excluded']

    # 1. Patch EXCLUDED_FEATURES
    features_str = "['" + "', '".join(excl) + "']" if excl else "[]"
    content = re.sub(
        r'EXCLUDED_FEATURES\s*=\s*\[[\s\S]*?\]',
        f'EXCLUDED_FEATURES = {features_str}',
        content, count=1
    )

    # 2. Patch forecast_horizon (lowercase variable in training script)
    content = re.sub(
        r'(forecast_horizon\s*=\s*)\d+',
        rf'\g<1>{fh}',
        content, count=1
    )

    # 3. Patch WINDOW_SIZE
    content = re.sub(
        r'(WINDOW_SIZE\s*=\s*)\d+',
        rf'\g<1>{ws}',
        content, count=1
    )

    # 4. Override OUTPUT_DIR — replace the entire if/else block that builds it
    #    with a direct assignment to the target folder.
    output_dir = os.path.join("FH_WS", f"FH{fh}_WS{ws}", name).replace("\\", "/")
    content = re.sub(
        r'if not EXCLUDED_FEATURES:.*?OUTPUT_DIR\s*=\s*os\.path\.join\(.*?\)',
        f'OUTPUT_DIR = "{output_dir}"',
        content, count=1, flags=re.DOTALL
    )

    # 5. Override the file prefix inside out_path() so files are named
    #    {config_name}_best_model.pt etc. instead of using EXCLUDED_FEATURES.
    content = re.sub(
        r'(def out_path\(filename\):.*?)'
        r'if not EXCLUDED_FEATURES:.*?prefix\s*=\s*"_"\.join\(EXCLUDED_FEATURES\)',
        rf'\1prefix = "{name}"',
        content, count=1, flags=re.DOTALL
    )

    temp_script = "_temp_hyperparam_run.py"
    with open(temp_script, 'w') as f:
        f.write(content)
    return temp_script


def run_experiment(config, fh, ws, exp_num, total_exp):
    """Run a single training experiment and return the result dict."""
    label = experiment_label(config, fh, ws)

    print("\n" + "=" * 70)
    print(f"EXPERIMENT {exp_num}/{total_exp}")
    print(f"  Config   : {config['name']}")
    print(f"  Excluded : {config['excluded'] if config['excluded'] else 'none'}")
    print(f"  FH={fh}  WS={ws}  ({20 - len(config['excluded'])} features)")
    print(f"  Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")

    temp_script = create_temp_script(config, fh, ws)
    start_time  = time.time()

    # Per-experiment log file so we can diagnose failures after the fact.
    log_dir = "hyperparameter_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir, f"{config['name']}__FH{fh}__WS{ws}.log"
    )
    print(f"  Log file : {log_path}\n")

    try:
        with open(log_path, 'w', buffering=1) as log_f:
            # Tee stdout/stderr: subprocess writes line-by-line to log_f, and
            # we also echo each line to the parent stdout in real time.
            proc = subprocess.Popen(
                [sys.executable, '-u', temp_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                text=True,
            )
            for line in proc.stdout:
                sys.stdout.write(line)
                log_f.write(line)
            returncode = proc.wait()

        if returncode != 0:
            # Extract the final traceback/error lines from the log for quick view.
            tail = ""
            try:
                with open(log_path, 'r') as lf:
                    lines = lf.readlines()
                tail = "".join(lines[-20:]).rstrip()
            except Exception:
                pass
            success   = False
            error_msg = (
                f"exit {returncode}. "
                f"Log: {log_path}"
                + (f"\nLast 20 lines:\n{tail}" if tail else "")
            )
        else:
            success   = True
            error_msg = None
    except Exception as e:
        success   = False
        error_msg = f"{type(e).__name__}: {e}. Log: {log_path}"
    finally:
        if os.path.exists(temp_script):
            os.remove(temp_script)

    duration = time.time() - start_time
    return {
        'label':    label,
        'config':   config['name'],
        'fh':       fh,
        'ws':       ws,
        'excluded': config['excluded'],
        'success':  success,
        'duration': duration,
        'error':    error_msg,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    if not os.path.exists(TRAINING_SCRIPT):
        print(f"ERROR: Training script '{TRAINING_SCRIPT}' not found!")
        sys.exit(1)

    # Build the full experiment queue: iterate config → FH → WS
    # WS must be strictly greater than FH (10 valid pairs per config).
    # FH=30, WS=90 is excluded — already run in the ablation study.
    queue = [
        (cfg, fh, ws)
        for cfg in CONFIGURATIONS
        for fh  in FORECAST_HORIZONS
        for ws  in WINDOW_SIZES
        if ws > fh and not (fh == 30 and ws == 90)
    ]
    total = len(queue)

    print("=" * 70)
    print("HYPERPARAMETER SWEEP SCHEDULER")
    print("=" * 70)
    print(f"Training script : {TRAINING_SCRIPT}")
    print(f"Configurations  : {len(CONFIGURATIONS)}")
    for cfg in CONFIGURATIONS:
        excl = ', '.join(cfg['excluded']) if cfg['excluded'] else 'none'
        print(f"  {cfg['name']:<14} excluded: {excl}")
    print(f"Forecast Horizons: {FORECAST_HORIZONS}")
    print(f"Window Sizes     : {WINDOW_SIZES}")
    print(f"Total runs       : {total}")
    print(f"Stop on error    : {STOP_ON_ERROR}")
    print(f"Started at       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ---- Resume checkpoint ----
    progress = load_progress()
    if progress:
        done_count = sum(1 for v in progress.values() if v['success'])
        print(f"\nResuming from checkpoint: {len(progress)} recorded, "
              f"{done_count} successful, "
              f"{sum(1 for v in progress.values() if not v['success'])} failed.")
        print(f"Checkpoint file: {PROGRESS_FILE}")
    else:
        print(f"\nNo checkpoint found — starting fresh.")

    print("\nQueued experiments:")
    for i, (cfg, fh, ws) in enumerate(queue, 1):
        key  = checkpoint_key(cfg, fh, ws)
        prev = progress.get(key)
        if prev and prev['success']:
            tag = "  [DONE]"
        elif prev:
            tag = "  [RETRY — previous run failed]"
        else:
            tag = ""
        print(f"  {i:3d}. {experiment_label(cfg, fh, ws)}{tag}")
    print()

    results     = []
    batch_start = time.time()
    ran_count   = 0

    for i, (cfg, fh, ws) in enumerate(queue, 1):
        key = checkpoint_key(cfg, fh, ws)

        prev = progress.get(key)
        if prev and prev['success']:
            print(f"  SKIP {i}/{total} (already done): {experiment_label(cfg, fh, ws)}")
            results.append(prev)
            continue

        ran_count += 1
        result = run_experiment(cfg, fh, ws, i, total)
        results.append(result)

        # Save to checkpoint immediately after each run
        progress[key] = result
        save_progress(progress)

        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        print(f"\n{status} — {result['label']} — {format_duration(result['duration'])}")

        if not result['success']:
            print(f"  Error: {result['error']}")
            if STOP_ON_ERROR:
                print("\nStopping (STOP_ON_ERROR=True)")
                break

        remaining_queue = total - i
        if remaining_queue > 0 and ran_count > 0:
            avg       = (time.time() - batch_start) / ran_count
            remaining = avg * remaining_queue
            print(f"  Estimated remaining: {format_duration(remaining)}")

    # ---- Final summary ----
    batch_duration = time.time() - batch_start
    successful     = sum(1 for r in results if r['success'])
    failed         = sum(1 for r in results if not r['success'])

    summary = f"""
{'=' * 70}
HYPERPARAMETER SWEEP COMPLETE
{'=' * 70}
Total runs:     {total}
Successful:     {successful}
Failed:         {failed}
Total duration: {format_duration(batch_duration)}
Finished at:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RESULTS:
{'-' * 70}
"""
    for i, r in enumerate(results, 1):
        icon = "✓" if r['success'] else "✗"
        summary += f"  {icon} {i:3d}. {r['label']:<60} ({format_duration(r['duration'])})\n"
        if not r['success']:
            summary += f"        Error: {r['error']}\n"

    summary += "=" * 70
    print(summary)

    with open(BATCH_LOG_FILE, 'w') as f:
        f.write(summary)
    print(f"\nLog saved to: {BATCH_LOG_FILE}")

    if failed == 0 and successful == total:
        # All done — remove checkpoint so a re-run starts fresh
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)
            print(f"All experiments complete. Checkpoint '{PROGRESS_FILE}' removed.")
    else:
        print(f"\nCheckpoint saved to '{PROGRESS_FILE}'.")
        print("Re-run this script to continue from where it stopped.")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
