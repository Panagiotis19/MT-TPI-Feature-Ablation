"""
PyTorch conversion of Transformer_physicsinformed.py (TensorFlow/Keras).

Architecture, hyperparameters, data pipeline, training logic, and evaluation
are kept identical.  Comments marked [TF→PT] highlight the mapping.

Feature ablation support: set EXCLUDED_FEATURES to remove input features.
The scheduler (scheduler_ablation.py) will regex-replace this line.
"""

import os
import math
import copy
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from haversine import haversine  # kept for parity, unused in logic

# ============================================================================
# Feature ablation configuration
# The scheduler replaces this line via regex.  DO NOT change the format.
# ============================================================================
EXCLUDED_FEATURES = []

# ============================================================================
# All 20 input feature names (in column order after 'flight')
# ============================================================================
ALL_INPUT_FEATURES = [
    'wind_speed', 'wind_angle', 'battery_voltage', 'battery_current',
    'position_x', 'position_y', 'position_z',
    'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w',
    'velocity_x', 'velocity_y', 'velocity_z',
    'angular_x', 'angular_y', 'angular_z',
    'linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z',
]

# Determine which features are kept and their indices within the 20 input cols
KEPT_FEATURES = [f for f in ALL_INPUT_FEATURES if f not in EXCLUDED_FEATURES]
# Indices into the 20-column input block (cols index 1..20 in the full cols list)
KEPT_INDICES = [ALL_INPUT_FEATURES.index(f) for f in KEPT_FEATURES]
NUM_INPUT_FEATURES = len(KEPT_FEATURES)

print(f"Excluded features: {EXCLUDED_FEATURES}")
print(f"Kept features ({NUM_INPUT_FEATURES}): {KEPT_FEATURES}")

# ============================================================================
# Output directory setup
# Mirrors the structure expected by the scheduler:
#   Baseline/
#   Remove_Tests/Remove_1_feature/Removed_<name>/
#   Remove_Tests/Remove_2_features/Removed_<name1>_<name2>/
# ============================================================================
if not EXCLUDED_FEATURES:
    OUTPUT_DIR = "Baseline"
else:
    n_removed = len(EXCLUDED_FEATURES)
    folder_name = "Removed_" + "_".join(EXCLUDED_FEATURES)
    OUTPUT_DIR = os.path.join("Remove_Tests", f"Remove_{n_removed}_feature{'s' if n_removed > 1 else ''}", folder_name)

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")

# Helper to build output paths
def out_path(filename):
    if not EXCLUDED_FEATURES:
        prefix = "Baseline"
    else:
        prefix = "_".join(EXCLUDED_FEATURES)
    return os.path.join(OUTPUT_DIR, f"{prefix}_{filename}")

# ============================================================================
# GPU configuration
# ============================================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(0.4, device=0)
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ============================================================================
# Physics-informed loss  (unchanged)
# ============================================================================
DRONE_MASS = 6.3
GRAVITY    = 9.81


def quad_res(pos, vel, thrusts, dt):
    v_n = vel[:, 1:, :]
    v_c = vel[:, :-1, :]
    afd = (v_n - v_c) / dt
    tfc = thrusts[:, 2:, :] / DRONE_MASS
    am = torch.cat([
        torch.zeros_like(afd[..., :2]),
        tfc - GRAVITY
    ], dim=-1)
    return torch.mean((afd - am) ** 2)


class PhysicsInformed2(nn.Module):
    def __init__(self, dt=0.1, lam=0.02, alpha_z=20.0, beta_xy=10.0):
        super().__init__()
        self.dt = dt
        self.lambda_phys = nn.Parameter(torch.tensor(lam,    dtype=torch.float32))
        self.alpha_z     = nn.Parameter(torch.tensor(alpha_z, dtype=torch.float32))
        self.beta_xy     = nn.Parameter(torch.tensor(beta_xy, dtype=torch.float32))

    def forward(self, y_true, y_pred):
        axis_wt = torch.tensor([1.0, 1.0, 10.0],
                               dtype=y_true.dtype, device=y_true.device)
        mse = torch.mean(((y_true - y_pred) * axis_wt) ** 2)

        vel = (y_pred[:, 1:, :] - y_pred[:, :-1, :]) / self.dt
        b, T, _ = y_pred.shape
        thrusts = torch.zeros((b, T, 1), dtype=y_pred.dtype, device=y_pred.device)

        phys_res = quad_res(y_pred, vel, thrusts, self.dt)
        z_mae    = torch.mean(torch.abs(y_true[..., 2] - y_pred[..., 2]))
        xy_mae   = torch.mean(torch.abs(y_true[..., :2] - y_pred[..., :2]))

        return mse + self.lambda_phys * phys_res + self.alpha_z * z_mae + self.beta_xy * xy_mae


# ============================================================================
# Data loading & sliding window
# ============================================================================
df = pd.read_csv("Weather_Dataset_final.csv", delimiter=";")
cols = [
    'flight',
    'wind_speed', 'wind_angle', 'battery_voltage', 'battery_current',
    'position_x', 'position_y', 'position_z',
    'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w',
    'velocity_x', 'velocity_y', 'velocity_z',
    'angular_x', 'angular_y', 'angular_z',
    'linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z',
    'IDLE_HOVER', 'ASCEND', 'TURN', 'HMSL', 'DESCEND'
]

forecast_horizon = 30
WINDOW_SIZE      = 90


def sliding_window(data, window, horizon):
    """
    Build sliding windows as float32 to halve RAM vs numpy's default float64.
    Input features: columns at KEPT_INDICES (within the 20 input cols = data[:,1:21])
    Targets:        position x,y,z = data[:,5:8] and flight modes = data[:,21:26]
    """
    n = len(data) - window - horizon + 1
    if n <= 0:
        return (np.empty((0, window, NUM_INPUT_FEATURES), dtype=np.float32),
                np.empty((0, horizon, 3),               dtype=np.float32),
                np.empty((0, horizon, 5),               dtype=np.float32))

    X     = np.empty((n, window, NUM_INPUT_FEATURES), dtype=np.float32)
    y_traj = np.empty((n, horizon, 3),                dtype=np.float32)
    y_clf  = np.empty((n, horizon, 5),                dtype=np.float32)

    for i in range(n):
        X[i]      = data[i:i+window, 1:21][:, KEPT_INDICES]
        y_traj[i] = data[i+window:i+window+horizon, 5:8]
        y_clf[i]  = data[i+window:i+window+horizon, 21:26]
    return X, y_traj, y_clf


X_list, Ytraj_list, Yclf_list = [], [], []
for fid in df['flight'].unique():
    sub = df[df['flight'] == fid][cols].values
    Xi, Yi, Ci = sliding_window(sub, WINDOW_SIZE, forecast_horizon)
    if Xi.size:
        X_list.append(Xi);  Ytraj_list.append(Yi);  Yclf_list.append(Ci)
del df, sub  # free the raw pandas DataFrame — we only need the sliding windows

X_all     = np.concatenate(X_list, axis=0)
Ytraj_all = np.concatenate(Ytraj_list, axis=0)
Yclf_all  = np.concatenate(Yclf_list, axis=0)
del X_list, Ytraj_list, Yclf_list
gc.collect()

# --- Scaling (cast back to float32 after sklearn's float64 output) ---
mm_x = MinMaxScaler();  mm_y = MinMaxScaler()
b, w, f = X_all.shape    # f == NUM_INPUT_FEATURES
X_scaled = mm_x.fit_transform(X_all.reshape(-1, f)).astype(np.float32, copy=False).reshape(b, w, f)
del X_all
Y_scaled = mm_y.fit_transform(Ytraj_all.reshape(-1, 3)).astype(np.float32, copy=False).reshape(b, forecast_horizon, 3)
del Ytraj_all
gc.collect()

# --- Train / Val / Test split (70 / 15 / 15) ---
i1, i2 = int(0.7 * b), int(0.85 * b)
X_train, X_val, X_test       = X_scaled[:i1], X_scaled[i1:i2], X_scaled[i2:]
y_train, y_val, y_test       = Y_scaled[:i1], Y_scaled[i1:i2], Y_scaled[i2:]
clf_train, clf_val, clf_test = Yclf_all[:i1],  Yclf_all[i1:i2], Yclf_all[i2:]

# ============================================================================
# DataLoaders — use torch.from_numpy to SHARE memory with the numpy arrays
# (no extra copy). X_scaled / Y_scaled / Yclf_all are already float32.
# ============================================================================
def make_loader(X, y_traj, y_clf, batch_size=64, shuffle=False):
    ds = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(X)),
        torch.from_numpy(np.ascontiguousarray(y_traj)),
        torch.from_numpy(np.ascontiguousarray(y_clf)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

train_loader = make_loader(X_train, y_train, clf_train, batch_size=64, shuffle=True)
val_loader   = make_loader(X_val,   y_val,   clf_val,   batch_size=64, shuffle=False)
test_loader  = make_loader(X_test,  y_test,  clf_test,  batch_size=64, shuffle=False)


# ============================================================================
# Model definition  (UNCHANGED architecture)
# d_model is now NUM_INPUT_FEATURES instead of hardcoded 20
# ============================================================================
class PositionalEncoding:
    @staticmethod
    def build(seq_len, d_model):
        pos = np.arange(seq_len)[:, None]
        div = 1.0 / np.power(10000, (2 * (np.arange(d_model) // 2)) / d_model)
        angles = pos * div[None, :]
        pe = np.concatenate([np.sin(angles[:, 0::2]),
                             np.cos(angles[:, 1::2])], axis=-1)
        return torch.tensor(pe[None, :, :], dtype=torch.float32)


class PiTransformerModel(nn.Module):
    def __init__(self, seq_len=90, d_model=20, horizon=30, num_classes=5,
                 head_size=256, num_heads=4, ff_dim=512, num_blocks=4,
                 dropout=0.1):
        super().__init__()
        self.horizon    = horizon
        self.num_blocks = num_blocks

        self.register_buffer('pe', PositionalEncoding.build(seq_len, d_model))

        self.attn_norms = nn.ModuleList()
        self.attns      = nn.ModuleList()
        self.attn_drops = nn.ModuleList()
        self.ff_norms   = nn.ModuleList()
        self.ff1s       = nn.ModuleList()
        self.ff2s       = nn.ModuleList()

        for _ in range(num_blocks):
            self.attn_norms.append(nn.LayerNorm(d_model, eps=1e-6))
            self.attns.append(nn.MultiheadAttention(
                embed_dim=d_model, num_heads=num_heads, batch_first=True
            ))
            self.attn_drops.append(nn.Dropout(dropout))
            self.ff_norms.append(nn.LayerNorm(d_model, eps=1e-6))
            self.ff1s.append(nn.Linear(d_model, ff_dim))
            self.ff2s.append(nn.Linear(ff_dim, d_model))

        self.traj_dense = nn.Linear(d_model, 3)
        self.clf_dense  = nn.Linear(d_model, num_classes)

    def forward(self, x):
        for i in range(self.num_blocks):
            x = x + self.pe
            a = self.attn_norms[i](x)
            a, _ = self.attns[i](a, a, a)
            x = x + self.attn_drops[i](a)

            f = self.ff_norms[i](x)
            f = F.relu(self.ff1s[i](f))
            f = self.ff2s[i](f)
            x = x + f

        traj = self.traj_dense(x)
        traj = traj[:, -self.horizon:, :]

        clf = self.clf_dense(x)
        clf = F.softmax(clf, dim=-1)
        clf = clf[:, -self.horizon:, :]

        return {"trajectory": traj, "classification": clf}


# ============================================================================
# Instantiate model + loss + optimizer
# ============================================================================
num_classes = clf_train.shape[-1]   # 5

# d_model adapts to the number of kept features
# num_heads must evenly divide d_model — find a valid value
def find_num_heads(d_model, preferred=4):
    """Return the largest divisor of d_model that is <= preferred."""
    for h in range(min(preferred, d_model), 0, -1):
        if d_model % h == 0:
            return h
    return 1

actual_num_heads = find_num_heads(NUM_INPUT_FEATURES, preferred=4)
print(f"d_model={NUM_INPUT_FEATURES}, num_heads={actual_num_heads}")

model = PiTransformerModel(
    seq_len=WINDOW_SIZE, d_model=NUM_INPUT_FEATURES, horizon=forecast_horizon,
    num_classes=num_classes, head_size=256, num_heads=actual_num_heads,
    ff_dim=512, num_blocks=4, dropout=0.1
).to(device)

pin_loss = PhysicsInformed2(dt=0.1, lam=0.02, alpha_z=20.0, beta_xy=10.0).to(device)

def categorical_crossentropy(y_true, y_pred, eps=1e-7):
    y_pred = torch.clamp(y_pred, eps, 1.0 - eps)
    return -torch.mean(torch.sum(y_true * torch.log(y_pred), dim=-1))

optimizer = torch.optim.Adam(
    list(model.parameters()) + list(pin_loss.parameters()),
    lr=5e-5
)

TRAJ_WEIGHT = 0.4
CLF_WEIGHT  = 0.6

# ============================================================================
# Training utilities
# ============================================================================
EPOCHS    = 50
PATIENCE  = 10
BEST_PATH  = out_path("best_model.pt")
FINAL_PATH = out_path("final_model.pt")


def run_epoch(loader, training=True):
    if training:
        model.train()
    else:
        model.eval()

    total_loss  = 0.0
    total_mae   = 0.0
    total_correct = 0
    total_samples = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for xb, yb_traj, yb_clf in loader:
            xb      = xb.to(device)
            yb_traj = yb_traj.to(device)
            yb_clf  = yb_clf.to(device)

            preds = model(xb)
            traj_out = preds["trajectory"]
            clf_out  = preds["classification"]

            loss_traj = pin_loss(yb_traj, traj_out)
            loss_clf  = categorical_crossentropy(yb_clf, clf_out)
            loss = TRAJ_WEIGHT * loss_traj + CLF_WEIGHT * loss_clf

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(pin_loss.parameters()),
                    max_norm=1.0
                )
                optimizer.step()

                with torch.no_grad():
                    pin_loss.lambda_phys.data.clamp_(0.0, 1.0)
                    pin_loss.alpha_z.data.clamp_(0.0, 50.0)
                    pin_loss.beta_xy.data.clamp_(0.0, 50.0)

            batch_size_actual = xb.size(0)
            total_loss += loss.item() * batch_size_actual

            with torch.no_grad():
                mae = torch.mean(torch.abs(yb_traj - traj_out)).item()
                total_mae += mae * batch_size_actual

                pred_labels = clf_out.reshape(-1, clf_out.size(-1)).argmax(dim=-1)
                true_labels = yb_clf.reshape(-1, yb_clf.size(-1)).argmax(dim=-1)
                total_correct += (pred_labels == true_labels).sum().item()
                total_samples += pred_labels.numel()

    n = total_samples // forecast_horizon
    avg_loss = total_loss / n if n > 0 else 0
    avg_mae  = total_mae / n if n > 0 else 0
    avg_acc  = total_correct / total_samples if total_samples > 0 else 0

    return avg_loss, avg_mae, avg_acc


# ============================================================================
# Training loop
# ============================================================================
best_val_mae = float('inf')
best_val_loss = float('inf')
best_val_acc  = 0.0
best_epoch    = 0
patience_ctr  = 0
best_model_state = None
best_loss_state  = None

for epoch in range(1, EPOCHS + 1):
    train_loss, train_mae, train_acc = run_epoch(train_loader, training=True)
    val_loss, val_mae, val_acc       = run_epoch(val_loader,   training=False)

    print(f"Epoch {epoch:3d}/{EPOCHS} — "
          f"train_loss: {train_loss:.4f}  train_traj_mae: {train_mae:.4f}  train_acc: {train_acc:.4f} | "
          f"val_loss: {val_loss:.4f}  val_traj_mae: {val_mae:.4f}  val_acc: {val_acc:.4f}")

    if val_mae < best_val_mae:
        best_val_mae  = val_mae
        best_val_loss = val_loss
        best_val_acc  = val_acc
        best_epoch    = epoch
        patience_ctr  = 0
        best_model_state = copy.deepcopy(model.state_dict())
        best_loss_state  = copy.deepcopy(pin_loss.state_dict())
        torch.save({
            'model_state_dict': best_model_state,
            'loss_state_dict':  best_loss_state,
        }, BEST_PATH)
        print(f"  ↳ val_traj_mae improved to {val_mae:.6f} — saved {BEST_PATH}")
    else:
        patience_ctr += 1
        print(f"  ↳ no improvement ({patience_ctr}/{PATIENCE})")

    if patience_ctr >= PATIENCE:
        print(f"Early stopping at epoch {epoch}.")
        break

if best_model_state is not None:
    model.load_state_dict(best_model_state)
    pin_loss.load_state_dict(best_loss_state)
    print("Restored best model weights.")

del best_model_state, best_loss_state

print("Training complete!")

torch.save({
    'model_state_dict': model.state_dict(),
    'loss_state_dict':  pin_loss.state_dict(),
}, FINAL_PATH)
print(f"Final model saved to {FINAL_PATH}")


# ============================================================================
# Free training/validation memory before the evaluation phase.
# WS=150 runs were hitting OOM because pandas/numpy peaks during evaluation
# stacked on top of still-resident training arrays and scalers' duplicates.
# ============================================================================
del train_loader, val_loader
del X_train, X_val, y_train, y_val, clf_train, clf_val
del X_scaled, Y_scaled, Yclf_all
del X_test, y_test, clf_test
gc.collect()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
elif torch.cuda.is_available():
    torch.cuda.empty_cache()


# ============================================================================
# Evaluation & saving results (streamed to disk, batch-by-batch)
# ============================================================================
classes       = ['IDLE_HOVER', 'ASCEND', 'TURN', 'HMSL', 'DESCEND']
num_classes_eval = len(classes)

traj_csv_path = out_path('Trajectory_Results.csv')
clf_csv_path  = out_path('Classification_Results.csv')

traj_cols = ['pred_x', 'pred_y', 'pred_z',
             'true_x', 'true_y', 'true_z', 'euclidean_error_m']
clf_cols  = [f'pred_{c}' for c in classes] + [f'true_{c}' for c in classes]

# Truncate and write headers
pd.DataFrame(columns=traj_cols).to_csv(traj_csv_path, index=False)
pd.DataFrame(columns=clf_cols).to_csv(clf_csv_path,  index=False)

cm = np.zeros((num_classes_eval, num_classes_eval), dtype=np.int64)

ts_indices = [(sec, idx) for sec, idx in [(1, 9), (2, 19), (3, 29)]
              if idx < forecast_horizon]
ts_sum = {sec: 0.0 for sec, _ in ts_indices}
ts_cnt = {sec: 0   for sec, _ in ts_indices}

euclid_sum  = 0.0
euclid_cnt  = 0
acc_correct = 0
acc_total   = 0

model.eval()
with torch.no_grad():
    for xb, yb_traj, yb_clf in test_loader:
        xb = xb.to(device)
        preds = model(xb)
        traj_pred_batch = preds['trajectory'].cpu().numpy()      # (B, FH, 3)
        clf_pred_batch  = preds['classification'].cpu().numpy()  # (B, FH, C)
        yb_traj_np = yb_traj.numpy()
        yb_clf_np  = yb_clf.numpy()

        B = traj_pred_batch.shape[0]
        FH = traj_pred_batch.shape[1]

        tp = mm_y.inverse_transform(traj_pred_batch.reshape(-1, 3))
        tt = mm_y.inverse_transform(yb_traj_np.reshape(-1, 3))
        euclid_b = np.linalg.norm(tp - tt, axis=1)

        # Append trajectory rows
        pd.DataFrame(
            np.column_stack([tp, tt, euclid_b[:, None]]),
            columns=traj_cols,
        ).to_csv(traj_csv_path, mode='a', header=False, index=False,
                 float_format='%.6f')

        # Append classification rows
        cp = clf_pred_batch.reshape(-1, num_classes_eval)
        ct = yb_clf_np.reshape(-1, num_classes_eval)
        pd.DataFrame(
            np.column_stack([cp, ct]),
            columns=clf_cols,
        ).to_csv(clf_csv_path, mode='a', header=False, index=False,
                 float_format='%.6f')

        # Confusion matrix accumulator
        pl = cp.argmax(axis=1)
        tl = ct.argmax(axis=1)
        np.add.at(cm, (tl, pl), 1)

        acc_correct += int((pl == tl).sum())
        acc_total   += pl.size

        # Timestamp-level errors
        euclid_win = euclid_b.reshape(B, FH)
        for sec, idx in ts_indices:
            ts_sum[sec] += float(euclid_win[:, idx].sum())
            ts_cnt[sec] += B

        euclid_sum += float(euclid_b.sum())
        euclid_cnt += int(euclid_b.size)

        del traj_pred_batch, clf_pred_batch, yb_traj_np, yb_clf_np
        del tp, tt, euclid_b, cp, ct, pl, tl, euclid_win, preds, xb

gc.collect()

# --- Confusion matrix plot ---
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix")
plt.savefig(out_path('Confusion_Matrix.png'))
plt.close()

# --- Per-class accuracy (from confusion matrix diagonal) ---
per_class_acc = {}
for idx, cls_name in enumerate(classes):
    cls_total = cm[idx].sum()
    if cls_total > 0:
        per_class_acc[cls_name] = cm[idx, idx] / cls_total * 100.0
    else:
        per_class_acc[cls_name] = 0.0

# --- Macro and weighted precision / recall / F1, derived from confusion matrix ---
_tp = np.diag(cm).astype(np.float64)
_fp = cm.sum(axis=0) - _tp
_fn = cm.sum(axis=1) - _tp
_support = cm.sum(axis=1).astype(np.float64)

_precision = np.where(_tp + _fp > 0, _tp / (_tp + _fp), 0.0)
_recall    = np.where(_tp + _fn > 0, _tp / (_tp + _fn), 0.0)
_f1        = np.where(_precision + _recall > 0,
                      2 * _precision * _recall / (_precision + _recall), 0.0)

prec_macro = float(_precision.mean())
rec_macro  = float(_recall.mean())
f1_macro   = float(_f1.mean())

_total_support = _support.sum()
if _total_support > 0:
    prec_weighted = float((_precision * _support).sum() / _total_support)
    rec_weighted  = float((_recall    * _support).sum() / _total_support)
    f1_weighted   = float((_f1        * _support).sum() / _total_support)
else:
    prec_weighted = rec_weighted = f1_weighted = 0.0

# --- Test accuracy ---
test_acc = (acc_correct / acc_total * 100.0) if acc_total > 0 else 0.0

# --- Timestamp-level trajectory errors (t+1s, t+2s, t+3s) ---
timestamp_errors = {}
for sec, _ in ts_indices:
    timestamp_errors[sec] = (ts_sum[sec] / ts_cnt[sec]) if ts_cnt[sec] > 0 else float('nan')
for sec in [1, 2, 3]:
    timestamp_errors.setdefault(sec, float('nan'))

mean_euclid = (euclid_sum / euclid_cnt) if euclid_cnt > 0 else 0.0

# --- Build Summary.txt ---
if not EXCLUDED_FEATURES:
    excluded_str = "None (Baseline)"
else:
    excluded_str = ", ".join(EXCLUDED_FEATURES)

saved_files = [
    BEST_PATH,
    FINAL_PATH,
    out_path('Trajectory_Results.csv'),
    out_path('Classification_Results.csv'),
    out_path('Confusion_Matrix.png'),
]

summary_lines = []
summary_lines.append("=" * 60)
summary_lines.append("RESULTS SUMMARY")
summary_lines.append("=" * 60)
summary_lines.append(f"Excluded Features: {excluded_str}")
summary_lines.append("")
summary_lines.append("HYPERPARAMETERS:")
summary_lines.append(f"  Forecast Horizon: {forecast_horizon}")
summary_lines.append(f"  Window Size:      {WINDOW_SIZE}")
summary_lines.append("")
summary_lines.append("ABLATION STUDY RESULTS:")
summary_lines.append(f"  Number of Features: {NUM_INPUT_FEATURES}")
summary_lines.append(f"  Val Accuracy:       {best_val_acc * 100:.2f}%")
summary_lines.append(f"  Val MAE:            {best_val_mae:.4f}")
summary_lines.append(f"  Val Loss:           {best_val_loss:.4f}")
summary_lines.append(f"  Epochs:             {best_epoch}")
for cls_name in classes:
    summary_lines.append(f"  {cls_name + ':':<20s}{per_class_acc[cls_name]:.1f}%")
summary_lines.append("")
summary_lines.append("CLASSIFICATION PERFORMANCE (Macro):")
summary_lines.append(f"  Precision: {prec_macro:.4f}")
summary_lines.append(f"  Recall:    {rec_macro:.4f}")
summary_lines.append(f"  F1-Score:  {f1_macro:.4f}")
summary_lines.append("")
summary_lines.append("CLASSIFICATION PERFORMANCE (Weighted):")
summary_lines.append(f"  Precision: {prec_weighted:.4f}")
summary_lines.append(f"  Recall:    {rec_weighted:.4f}")
summary_lines.append(f"  F1-Score:  {f1_weighted:.4f}")
summary_lines.append("")
summary_lines.append("TIMESTAMP-LEVEL TRAJECTORY ERRORS:")
for sec in [1, 2, 3]:
    summary_lines.append(f"  t+{sec}s: {timestamp_errors[sec]:.4f} m")
summary_lines.append("")
summary_lines.append("TEST METRICS:")
summary_lines.append(f"  Mean Euclidean Error: {mean_euclid:.4f} m")
summary_lines.append(f"  Test Accuracy:        {test_acc:.2f}%")
summary_lines.append("")
summary_lines.append("FILES SAVED:")
for fpath in saved_files:
    summary_lines.append(f"  - {fpath}")
summary_lines.append("=" * 60)

summary_text = "\n".join(summary_lines)

# Write Summary.txt
summary_path = out_path('Summary.txt')
with open(summary_path, 'w') as fout:
    fout.write(summary_text)

# Also print to console
print(summary_text)
print(f"\nEvaluation complete — results saved to {OUTPUT_DIR}/")