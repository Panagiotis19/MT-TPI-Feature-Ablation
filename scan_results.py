"""Scan UAV trajectory experiment results and emit a single all_results.json.

Walks Baseline/, Remove_Tests/, FH_WS/ recursively. A leaf experiment is any
directory containing *_Summary.txt + *_Trajectory_Results.csv +
*_Classification_Results.csv. For each experiment we parse the summary, then
condense the (possibly >1M-row) CSVs into JSON-friendly aggregates: per-axis
MAE, an error CDF (100 quantile points), per-class F1, and a downsampled
trajectory (500 points) suitable for plotting.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["IDLE_HOVER", "ASCEND", "TURN", "HMSL", "DESCEND"]

# Canonical feature list, used to split joined "Removed_a_b_c" folder names
# back into individual feature names (since features themselves contain
# underscores like "linear_acceleration_y").
FEATURES = [
    "angular_x", "angular_y", "angular_z",
    "battery_current", "battery_voltage",
    "linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z",
    "orientation_w", "orientation_x", "orientation_y", "orientation_z",
    "position_x", "position_y", "position_z",
    "velocity_x", "velocity_y", "velocity_z",
    "wind_angle", "wind_speed",
]

TRAJECTORY_SAMPLES = 500
CDF_POINTS = 100


def split_feature_string(joined: str) -> list[str]:
    """Split 'linear_acceleration_y_angular_z' into ['linear_acceleration_y','angular_z']."""
    if not joined:
        return []
    feats: list[str] = []
    s = joined
    while s:
        match = next((f for f in FEATURES if s == f or s.startswith(f + "_")), None)
        if match is None:
            # Unknown chunk — bail out, return what we have plus the remainder raw.
            feats.append(s)
            break
        feats.append(match)
        s = s[len(match):].lstrip("_")
    return feats


# ---------------------------------------------------------------------------
# Summary parsing
# ---------------------------------------------------------------------------

NUM_RE = r"([-+]?\d*\.?\d+)"


def parse_summary(path: Path) -> dict:
    text = path.read_text()

    def grab(pattern: str, cast=float, default=None):
        m = re.search(pattern, text)
        if not m:
            return default
        try:
            return cast(m.group(1))
        except (TypeError, ValueError):
            return default

    excluded_raw_match = re.search(r"Excluded Features:\s*(.+)", text)
    excluded_raw = excluded_raw_match.group(1).strip() if excluded_raw_match else ""
    if excluded_raw.lower().startswith("none"):
        excluded = []
    else:
        excluded = [f.strip() for f in excluded_raw.split(",") if f.strip()]

    summary: dict = {
        "excluded_features": excluded,
        "fh": grab(r"Forecast Horizon:\s*" + NUM_RE, int),
        "ws": grab(r"Window Size:\s*" + NUM_RE, int),
        "n_features": grab(r"Number of Features:\s*" + NUM_RE, int),
        "val_accuracy": grab(r"Val Accuracy:\s*" + NUM_RE + r"\s*%", lambda x: float(x) / 100.0),
        "val_mae": grab(r"Val MAE:\s*" + NUM_RE),
        "val_loss": grab(r"Val Loss:\s*" + NUM_RE),
        "epochs": grab(r"Epochs:\s*" + NUM_RE, int),
        "macro_precision": grab(r"\(Macro\):\s*\n\s*Precision:\s*" + NUM_RE),
        "macro_recall": grab(r"\(Macro\):.*?Recall:\s*" + NUM_RE, default=None),
        "macro_f1": grab(r"\(Macro\):.*?F1-Score:\s*" + NUM_RE, default=None),
        "weighted_precision": grab(r"\(Weighted\):.*?Precision:\s*" + NUM_RE, default=None),
        "weighted_recall": grab(r"\(Weighted\):.*?Recall:\s*" + NUM_RE, default=None),
        "weighted_f1": grab(r"\(Weighted\):.*?F1-Score:\s*" + NUM_RE, default=None),
        "test_mean_euclidean": grab(r"Mean Euclidean Error:\s*" + NUM_RE),
        "test_accuracy": grab(r"Test Accuracy:\s*" + NUM_RE + r"\s*%", lambda x: float(x) / 100.0),
    }

    # Re-grab macro/weighted with re.DOTALL since simple .*? above does not cross newlines by default.
    for key, pattern in [
        ("macro_recall",       r"\(Macro\)[\s\S]*?Recall:\s*"   + NUM_RE),
        ("macro_f1",           r"\(Macro\)[\s\S]*?F1-Score:\s*" + NUM_RE),
        ("weighted_precision", r"\(Weighted\)[\s\S]*?Precision:\s*" + NUM_RE),
        ("weighted_recall",    r"\(Weighted\)[\s\S]*?Recall:\s*"    + NUM_RE),
        ("weighted_f1",        r"\(Weighted\)[\s\S]*?F1-Score:\s*"  + NUM_RE),
    ]:
        m = re.search(pattern, text)
        if m:
            summary[key] = float(m.group(1))

    per_class_acc: dict[str, float] = {}
    for cls in CLASSES:
        m = re.search(rf"{cls}:\s*" + NUM_RE + r"\s*%", text)
        if m:
            per_class_acc[cls] = float(m.group(1)) / 100.0
    summary["per_class_accuracy"] = per_class_acc

    timestamp_errors: dict[str, float] = {}
    for m in re.finditer(r"(t\+\d+s):\s*" + NUM_RE + r"\s*m", text):
        timestamp_errors[m.group(1)] = float(m.group(2))
    summary["timestamp_errors"] = timestamp_errors

    return summary


# ---------------------------------------------------------------------------
# CSV aggregation
# ---------------------------------------------------------------------------

def aggregate_trajectory(csv_path: Path) -> dict:
    df = pd.read_csv(
        csv_path,
        usecols=["pred_x", "pred_y", "pred_z", "true_x", "true_y", "true_z", "euclidean_error_m"],
        dtype=np.float32,
    )
    n = len(df)

    pred = df[["pred_x", "pred_y", "pred_z"]].to_numpy()
    true = df[["true_x", "true_y", "true_z"]].to_numpy()
    abs_err = np.abs(pred - true)
    mae_per_axis = {
        "x": float(abs_err[:, 0].mean()),
        "y": float(abs_err[:, 1].mean()),
        "z": float(abs_err[:, 2].mean()),
    }

    eucl = df["euclidean_error_m"].to_numpy()
    qs = np.linspace(0.0, 1.0, CDF_POINTS)
    cdf_x = np.quantile(eucl, qs).astype(float)
    cdf = {"x": [round(v, 5) for v in cdf_x.tolist()],
           "y": [round(v, 5) for v in qs.tolist()]}

    if n <= TRAJECTORY_SAMPLES:
        idx = np.arange(n)
    else:
        idx = np.linspace(0, n - 1, TRAJECTORY_SAMPLES).astype(int)
    sample = df.iloc[idx]
    trajectory = {
        col: [round(v, 5) for v in sample[col].astype(float).tolist()]
        for col in ["pred_x", "pred_y", "pred_z", "true_x", "true_y", "true_z"]
    }

    return {
        "mae_per_axis": mae_per_axis,
        "mean_euclidean": float(eucl.mean()),
        "cdf": cdf,
        "trajectory": trajectory,
        "n_points": int(n),
    }


def aggregate_classification(csv_path: Path) -> dict:
    pred_cols = [f"pred_{c}" for c in CLASSES]
    true_cols = [f"true_{c}" for c in CLASSES]
    df = pd.read_csv(csv_path, usecols=pred_cols + true_cols, dtype=np.float32)

    pred = df[pred_cols].to_numpy()
    true = df[true_cols].to_numpy()

    # The truth columns can be multi-hot in this dataset, so per-class one-vs-rest
    # F1 with a 0.5 threshold is the most defensible interpretation.
    pred_bin = pred >= 0.5
    true_bin = true >= 0.5

    per_class_f1: dict[str, float] = {}
    per_class_precision: dict[str, float] = {}
    per_class_recall: dict[str, float] = {}
    for i, cls in enumerate(CLASSES):
        tp = float(np.sum(pred_bin[:, i] & true_bin[:, i]))
        fp = float(np.sum(pred_bin[:, i] & ~true_bin[:, i]))
        fn = float(np.sum(~pred_bin[:, i] & true_bin[:, i]))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_class_precision[cls] = precision
        per_class_recall[cls] = recall
        per_class_f1[cls] = f1

    return {
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
    }


# ---------------------------------------------------------------------------
# Walk + classify experiments
# ---------------------------------------------------------------------------

def find_experiments(root: Path) -> list[dict]:
    experiments: list[dict] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        sums = [f for f in filenames if f.endswith("_Summary.txt")]
        trajs = [f for f in filenames if f.endswith("_Trajectory_Results.csv")]
        clfs = [f for f in filenames if f.endswith("_Classification_Results.csv")]
        if not (sums and trajs and clfs):
            continue
        # Pair files by their common prefix.
        for s in sums:
            prefix = s[: -len("_Summary.txt")]
            traj = f"{prefix}_Trajectory_Results.csv"
            clf = f"{prefix}_Classification_Results.csv"
            if traj in trajs and clf in clfs:
                experiments.append({
                    "dir": Path(dirpath),
                    "prefix": prefix,
                    "summary_path": Path(dirpath) / s,
                    "traj_path": Path(dirpath) / traj,
                    "clf_path": Path(dirpath) / clf,
                })
    return experiments


def classify_experiment(exp: dict, root: Path) -> dict | None:
    rel = exp["dir"].relative_to(root)
    parts = rel.parts
    if not parts:
        return None
    top = parts[0]
    meta: dict = {
        "dir_rel": str(rel),
        "prefix": exp["prefix"],
    }
    if top == "Baseline":
        meta.update({
            "group": "baseline",
            "config_name": "Baseline",
            "n_removed": 0,
            "removed_features": [],
        })
    elif top == "Remove_Tests" and len(parts) >= 3:
        m = re.match(r"Remove_(\d+)_features?", parts[1])
        if not m:
            return None
        n_removed = int(m.group(1))
        leaf = parts[2]
        joined = leaf[len("Removed_"):] if leaf.startswith("Removed_") else leaf
        removed = split_feature_string(joined)
        meta.update({
            "group": "ablation",
            "n_removed": n_removed,
            "removed_features": removed,
            "config_name": leaf,
        })
    elif top == "FH_WS" and len(parts) >= 3:
        m = re.match(r"FH(\d+)_WS(\d+)", parts[1])
        if not m:
            return None
        meta.update({
            "group": "hyperparameter",
            "fh_ws": parts[1],
            "fh": int(m.group(1)),
            "ws": int(m.group(2)),
            "config_name": parts[2],
        })
    else:
        return None
    return meta


def build_label(meta: dict, summary: dict) -> str:
    if meta["group"] == "baseline":
        return "Baseline"
    if meta["group"] == "ablation":
        feats = ", ".join(meta["removed_features"])
        return f"-{meta['n_removed']}: {feats}"
    if meta["group"] == "hyperparameter":
        excl = summary.get("excluded_features", [])
        suffix = "Baseline" if not excl else f"−{len(excl)}: {', '.join(excl)}"
        return f"FH{meta['fh']}/WS{meta['ws']} • {meta['config_name']} ({suffix})"
    return meta.get("config_name", "?")


def process(exp: dict, root: Path) -> dict | None:
    meta = classify_experiment(exp, root)
    if meta is None:
        return None
    summary = parse_summary(exp["summary_path"])
    if meta["group"] != "hyperparameter":
        meta["fh"] = summary.get("fh")
        meta["ws"] = summary.get("ws")
    traj_metrics = aggregate_trajectory(exp["traj_path"])
    clf_metrics = aggregate_classification(exp["clf_path"])

    record = {
        "id": meta["dir_rel"].replace(os.sep, "/"),
        "label": build_label(meta, summary),
        "config_name": meta["config_name"],
        "group": meta["group"],
        "n_removed": meta.get("n_removed", 0),
        "removed_features": meta.get("removed_features", summary.get("excluded_features", [])),
        "fh": meta.get("fh"),
        "ws": meta.get("ws"),
        "summary": summary,
        "metrics": {
            "mae_per_axis": traj_metrics["mae_per_axis"],
            "mean_euclidean": traj_metrics["mean_euclidean"],
            "n_points": traj_metrics["n_points"],
            **clf_metrics,
        },
        "cdf": traj_metrics["cdf"],
        "trajectory": traj_metrics["trajectory"],
    }
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Project root containing Baseline/, Remove_Tests/, FH_WS/")
    parser.add_argument("--out", default="all_results.json", help="Output JSON path")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    experiments = find_experiments(root)
    print(f"Discovered {len(experiments)} experiments under {root}", file=sys.stderr)

    out: dict = {
        "feature_list": FEATURES,
        "classes": CLASSES,
        "baseline": None,
        "ablation": {f"remove_{n}": [] for n in (1, 2, 3, 4)},
        "hyperparameter": {},
    }

    for i, exp in enumerate(experiments, 1):
        try:
            record = process(exp, root)
        except Exception as e:
            print(f"  [skip] {exp['dir']}: {e}", file=sys.stderr)
            continue
        if record is None:
            continue
        print(f"  [{i:>3}/{len(experiments)}] {record['group']:<14} {record['id']}", file=sys.stderr)
        if record["group"] == "baseline":
            out["baseline"] = record
        elif record["group"] == "ablation":
            key = f"remove_{record['n_removed']}"
            out["ablation"].setdefault(key, []).append(record)
        elif record["group"] == "hyperparameter":
            fh_ws = f"FH{record['fh']}_WS{record['ws']}"
            out["hyperparameter"].setdefault(fh_ws, []).append(record)

    for bucket in out["ablation"].values():
        bucket.sort(key=lambda r: r["config_name"])
    for bucket in out["hyperparameter"].values():
        bucket.sort(key=lambda r: r["config_name"])

    out_path = Path(args.out)
    out_path.write_text(json.dumps(out, separators=(",", ":")))
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {out_path} ({size_mb:.1f} MB)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
