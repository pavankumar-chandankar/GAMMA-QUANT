#!/usr/bin/env python3
# ------------------------------------------------------------------------
# Within-subject cascade debug for a single sbj_X
#
# For one subject (default: sbj_1):
#   1) Load that subject's CSV for 50 Hz, 25 Hz, 12.5 Hz configs
#   2) Split its samples into train/val (chronologically)
#   3) Normalize like models/train.py (using training-part stats)
#   4) Build InertialDataset for train/val
#   5) Train DeepConvLSTM for each frequency (local train_one_epoch /
#      validate_one_epoch here, same style as models/train.py)
#   6) Run cascade (12.5 -> 25 -> 50 Hz) on the held-out part
#   7) Only AFTER prediction, compute accuracy / F1 vs labels
# ------------------------------------------------------------------------

import argparse
import os
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from utils.os_utils import load_config
from utils.torch_utils import InertialDataset, init_weights, worker_init_reset_seed
from models.DeepConvLSTM import DeepConvLSTM

# ------------------------------------------------------------------------
# Label dict (same one you posted)
# ------------------------------------------------------------------------
DEFAULT_LABEL_DICT = {
    'null': 0,
    'jogging': 1,
    'jogging (rotating arms)': 2,
    'jogging (skipping)': 3,
    'jogging (sidesteps)': 4,
    'jogging (butt-kicks)': 5,
    'stretching (triceps)': 6,
    'stretching (lunging)': 7,
    'stretching (shoulders)': 8,
    'stretching (hamstrings)': 9,
    'stretching (lumbar rotation)': 10,
    'push-ups': 11,
    'push-ups (complex)': 12,
    'sit-ups': 13,
    'sit-ups (complex)': 14,
    'burpees': 15,
    'lunges': 16,
    'lunges (complex)': 17,
    'bench-dips': 18,
}

# Frequency groupings for cascade logic
LOW_FREQ_CLASSES = [6, 7, 8, 9, 10]       # stretching
MID_FREQ_CLASSES = [16, 17, 18]           # lunges, bench-dips etc.

# ------------------------------------------------------------------------
# Local copy of train_one_epoch / validate_one_epoch (from models/train.py)
# ------------------------------------------------------------------------

def train_one_epoch(loader, network, opt, criterion, device=None):
    """
    Trains the network for one epoch.
    Same style as models/train.py, but local and using 'device' explicitly.
    """
    losses, preds, gt = [], [], []
    network.train()

    for _, (inputs, targets) in enumerate(loader):
        if device is not None:
            inputs = inputs.to(device)
            targets = targets.to(device)

        output = network(inputs)
        batch_loss = criterion(output, targets.long())
        opt.zero_grad()
        batch_loss.backward()
        opt.step()

        losses.append(batch_loss.item())

        batch_preds = np.argmax(output.detach().cpu().numpy(), axis=-1)
        batch_gt = targets.detach().cpu().numpy().flatten()
        preds = np.concatenate((preds, batch_preds)) if len(preds) > 0 else batch_preds
        gt = np.concatenate((gt, batch_gt)) if len(gt) > 0 else batch_gt

    return network, losses, preds, gt


def validate_one_epoch(loader, network, criterion, device=None):
    """
    Validates the network for one epoch.
    Same style as models/train.py, but local and using 'device' explicitly.
    """
    losses, preds, gt = [], [], []

    network.eval()
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(loader):
            if device is not None:
                inputs = inputs.to(device)
                targets = targets.to(device)

            output = network(inputs)
            batch_loss = criterion(output, targets.long())
            losses.append(batch_loss.item())

            batch_preds = np.argmax(output.detach().cpu().numpy(), axis=-1)
            batch_gt = targets.detach().cpu().numpy().flatten()
            preds = np.concatenate((preds, batch_preds)) if len(preds) > 0 else batch_preds
            gt = np.concatenate((gt, batch_gt)) if len(gt) > 0 else batch_gt

    return losses, preds, gt

# ------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------

def load_subject_data(cfg: dict, subject_idx: int, label_dict: Dict[str, int]) -> np.ndarray:
    """
    Load a single subject CSV (for a given frequency config), replace label
    strings by ints using label_dict, and return as numpy array.

    Shape: (N_samples, input_dim + 2)  [time, channels..., label]
    """
    sens_folder = cfg['dataset']['sens_folder']
    input_dim = cfg['dataset']['input_dim']

    sbj_name = f"sbj_{subject_idx}"
    path = os.path.join(sens_folder, sbj_name + ".csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Subject file not found for {sbj_name}: {path}")

    df = pd.read_csv(path, index_col=False, low_memory=False)
    # Map label strings to ints and fill NaNs with 0, using infer_objects to silence FutureWarning
    df = df.replace({"label": label_dict}).infer_objects(copy=False).fillna(0)
    data = df.to_numpy().astype(np.float32)

    if data.shape[1] != input_dim + 2:
        raise RuntimeError(
            f"{sbj_name} ({path}) has {data.shape[1]} columns, "
            f"but cfg['dataset']['input_dim'] + 2 = {input_dim + 2}"
        )

    return data


def split_and_normalize(
    data: np.ndarray,
    train_frac: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chronologically split a subject's data into train/val and normalize
    both parts using *train* min/max, just like train.py.

    data shape: (N, input_dim + 2), where columns 1:-1 are sensor channels.
    """
    n = data.shape[0]
    n_train = int(n * train_frac)
    if n_train <= 0 or n_train >= n:
        raise ValueError(f"Invalid train_frac={train_frac}, got n={n}, n_train={n_train}")

    train_data = data[:n_train].copy()
    val_data   = data[n_train:].copy()

    # Normalize to [-1, 1] using training stats
    min_train = np.min(train_data[:, 1:-1])
    max_train = np.max(train_data[:, 1:-1])

    train_data[:, 1:-1] = 2 * (train_data[:, 1:-1] - min_train) / (max_train - min_train) - 1
    val_data[:, 1:-1]   = 2 * (val_data[:, 1:-1]   - min_train) / (max_train - min_train) - 1

    return train_data.astype(np.float32), val_data.astype(np.float32)


def build_net_from_dataset_and_cfg(
    train_dataset: InertialDataset,
    cfg: dict,
    num_classes: int,
    quant_bits: int = 4,
) -> DeepConvLSTM:
    """
    Build DeepConvLSTM with the same shape logic as in models/train.py,
    but *force* num_classes to global label space (0..18) so that
    CrossEntropyLoss always has correct n_classes.
    """
    conv_kernels = cfg['model']['conv_kernels']
    conv_kernel_size = cfg['model']['conv_kernel_size']
    lstm_units = cfg['model']['lstm_units']
    lstm_layers = cfg['model']['lstm_layers']
    dropout = cfg['model']['dropout']

    net = DeepConvLSTM(
        train_dataset.channels,
        num_classes,
        train_dataset.window_size,
        conv_kernels,
        conv_kernel_size,
        lstm_units,
        lstm_layers,
        dropout,
        gamma_quant="quantize_and_gamma",
        quant_bits=quant_bits,
        apply_gamma="global",
        gamma_type="s_shaped",
    )
    return net


def compute_metrics_window_level(
    preds: np.ndarray,
    gts: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    Same style as in models/train.py:

        conf_mat = confusion_matrix(v_gt, v_preds, normalize='true')
        v_acc = conf_mat.diagonal()
        v_prec = precision_score(..., average=None)
        v_rec = recall_score(..., average=None)
        v_f1 = f1_score(..., average=None)

    Then macro-average. Returns percentages (0..100).
    """
    assert preds.shape == gts.shape
    if preds.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    conf_mat = confusion_matrix(gts, preds, normalize='true')
    v_acc = conf_mat.diagonal()

    v_prec = precision_score(gts, preds, average=None, zero_division=0)
    v_rec  = recall_score(gts, preds, average=None, zero_division=0)
    v_f1   = f1_score(gts, preds, average=None, zero_division=0)

    acc  = float(np.nanmean(v_acc) * 100.0)
    prec = float(np.nanmean(v_prec) * 100.0)
    rec  = float(np.nanmean(v_rec) * 100.0)
    f1   = float(np.nanmean(v_f1) * 100.0)

    return acc, prec, rec, f1


def train_single_freq_model(
    cfg: dict,
    train_data: np.ndarray,
    val_data: np.ndarray,
    device: torch.device,
    num_classes: int,
    label: str,
) -> Tuple[torch.nn.Module, InertialDataset, InertialDataset]:
    """
    Train a DeepConvLSTM for a given frequency config on a single subject:
      - create InertialDataset(train/val)
      - DataLoader
      - local train_one_epoch / validate_one_epoch (same logic as models/train.py)
      - track best val F1 and keep that checkpoint in memory

    NOTE: We *ignore* cfg['train_cfg']['weighted_loss'] here to avoid
    class-weight length mismatches (we force global 19-class outputs).
    """
    window_size = cfg['dataset']['window_size']
    window_overlap = cfg['dataset']['window_overlap']

    train_dataset = InertialDataset(train_data, window_size, window_overlap, model=cfg['name'])
    val_dataset   = InertialDataset(val_data,   window_size, window_overlap, model=cfg['name'])

    # DataLoaders (same shuffle rule as models/train.py)
    if cfg['name'] in ["shallow_deepconvlstm", "deepconvcontext"]:
        shuffle_train = False
    else:
        shuffle_train = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['loader']['train_batch_size'],
        shuffle=shuffle_train,
        num_workers=4,
        worker_init_fn=worker_init_reset_seed,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['loader']['test_batch_size'],
        shuffle=False,
        num_workers=4,
        worker_init_fn=worker_init_reset_seed,
    )

    # Network + optimizer + criterion
    net = build_net_from_dataset_and_cfg(train_dataset, cfg, num_classes=num_classes)
    net = init_weights(net, cfg['train_cfg']['weight_init'])
    net.to(device)

    opt = torch.optim.Adam(
        net.parameters(),
        lr=cfg['train_cfg']['lr'],
        weight_decay=cfg['train_cfg']['weight_decay'],
    )
    if cfg['train_cfg']['lr_step'] > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(
            opt,
            step_size=cfg['train_cfg']['lr_step'],
            gamma=cfg['train_cfg']['lr_decay'],
        )
    else:
        scheduler = None

    # IMPORTANT: no class weighting here (to avoid length mismatch)
    criterion = nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_state: Optional[Dict[str, torch.Tensor]] = None

    num_epochs = cfg['train_cfg']['epochs']
    print(f"\n=== Training {label} model for {num_epochs} epochs ===")
    for epoch in range(num_epochs):
        net, t_losses, _, _ = train_one_epoch(train_loader, net, opt, criterion, device=device)
        v_losses, v_preds, v_gt = validate_one_epoch(val_loader, net, criterion, device=device)

        acc, prec, rec, f1 = compute_metrics_window_level(np.array(v_preds), np.array(v_gt))

        print(
            f"[{label}] Epoch {epoch+1:03d}/{num_epochs:03d}  "
            f"train loss {np.nanmean(t_losses):.3f}  "
            f"val loss {np.nanmean(v_losses):.3f}  "
            f"Acc {acc:5.2f}%  Prec {prec:5.2f}%  Rec {rec:5.2f}%  F1 {f1:5.2f}%"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}

        if scheduler is not None:
            scheduler.step()

    if best_state is not None:
        net.load_state_dict(best_state)
        net.to(device)

    print(f"[{label}] Best val F1: {best_f1:5.2f}%")
    return net, train_dataset, val_dataset


def run_model_on_dataset(
    net: torch.nn.Module,
    dataset: InertialDataset,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run a model on each window of a dataset (batch_size=1).
    Returns (preds, gts) as numpy arrays of shape (N,).
    """
    net.eval()
    preds: List[int] = []
    gts: List[int] = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            x_np, y = dataset[idx]
            x = torch.tensor(x_np, dtype=torch.float32, device=device)
            if x.ndim == 2:
                x = x.unsqueeze(0)

            logits = net(x)
            probs = F.softmax(logits, dim=-1)[0]
            _, pred = torch.max(probs, dim=-1)

            preds.append(int(pred.item()))
            gts.append(int(y))

    return np.array(preds, dtype=np.int64), np.array(gts, dtype=np.int64)


def run_cascade_on_val(
    net_50: torch.nn.Module,
    net_25: torch.nn.Module,
    net_12: torch.nn.Module,
    ds_50: InertialDataset,
    ds_25: InertialDataset,
    ds_12: InertialDataset,
    device: torch.device,
    thr_12: float = 0.9,
    thr_25: float = 0.9,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Run the 12.5 -> 25 -> 50 Hz cascade on the *same* val windows across
    all three datasets (using min length).

    IMPORTANT: During prediction we do NOT look at labels. Labels are
    only used afterwards to compute metrics.
    """
    net_50.eval()
    net_25.eval()
    net_12.eval()

    n_12 = len(ds_12)
    n_25 = len(ds_25)
    n_50 = len(ds_50)
    n = min(n_12, n_25, n_50)

    if not (n_12 == n_25 == n_50):
        print(
            f"Warning: window counts differ (12.5 Hz: {n_12}, 25 Hz: {n_25}, 50 Hz: {n_50}). "
            f"Using first {n} windows."
        )

    cascade_preds: List[int] = []
    gts: List[int] = []

    used_12 = 0
    used_25 = 0
    used_50 = 0

    with torch.no_grad():
        for idx in range(n):
            x12_np, y12 = ds_12[idx]
            x25_np, y25 = ds_25[idx]
            x50_np, y50 = ds_50[idx]

            # Ground-truth from 50 Hz dataset
            y = int(y50)
            gts.append(y)

            x12 = torch.tensor(x12_np, dtype=torch.float32, device=device)
            x25 = torch.tensor(x25_np, dtype=torch.float32, device=device)
            x50 = torch.tensor(x50_np, dtype=torch.float32, device=device)

            if x12.ndim == 2:
                x12 = x12.unsqueeze(0)
            if x25.ndim == 2:
                x25 = x25.unsqueeze(0)
            if x50.ndim == 2:
                x50 = x50.unsqueeze(0)

            # Step 1: 12.5 Hz
            logits12 = net_12(x12)
            p12 = F.softmax(logits12, dim=-1)[0]
            conf12, pred12 = torch.max(p12, dim=-1)
            conf12 = float(conf12.item())
            pred12 = int(pred12.item())

            if conf12 >= thr_12 and pred12 in LOW_FREQ_CLASSES:
                final_pred = pred12
                used_12 += 1
            else:
                # Step 2: 25 Hz
                logits25 = net_25(x25)
                p25 = F.softmax(logits25, dim=-1)[0]
                conf25, pred25 = torch.max(p25, dim=-1)
                conf25 = float(conf25.item())
                pred25 = int(pred25.item())

                if conf25 >= thr_25 and pred25 in MID_FREQ_CLASSES:
                    final_pred = pred25
                    used_25 += 1
                else:
                    # Step 3: 50 Hz
                    logits50 = net_50(x50)
                    p50 = F.softmax(logits50, dim=-1)[0]
                    _, pred50 = torch.max(p50, dim=-1)
                    final_pred = int(pred50.item())
                    used_50 += 1

            cascade_preds.append(final_pred)

    cascade_preds = np.array(cascade_preds, dtype=np.int64)
    gts = np.array(gts, dtype=np.int64)

    total = float(n)
    frac_12 = used_12 / total if total > 0 else 0.0
    frac_25 = used_25 / total if total > 0 else 0.0
    frac_50 = used_50 / total if total > 0 else 0.0

    eff_rate = 12.5 * frac_12 + 25.0 * frac_25 + 50.0 * frac_50
    rel_power = eff_rate / 50.0 if total > 0 else 0.0
    saving = 1.0 - rel_power

    stats = {
        "frac_12": frac_12,
        "frac_25": frac_25,
        "frac_50": frac_50,
        "eff_rate": eff_rate,
        "rel_power": rel_power,
        "saving": saving,
        "n_windows": n,
        "used_12": used_12,
        "used_25": used_25,
        "used_50": used_50,
    }

    return cascade_preds, gts, stats

# ------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Within-subject train/test for sbj_X with three freq models and cascade evaluation."
    )
    p.add_argument(
        "--cfg_50",
        type=str,
        default="configs/deepconvlstm/wear_loso.yaml",
        help="Config file for 50 Hz model.",
    )
    p.add_argument(
        "--cfg_25",
        type=str,
        default="configs/deepconvlstm/wear_loso_25hz.yaml",
        help="Config file for 25 Hz model.",
    )
    p.add_argument(
        "--cfg_12_5",
        type=str,
        default="configs/deepconvlstm/wear_loso_12_5hz.yaml",
        help="Config file for 12.5 Hz model.",
    )
    p.add_argument(
        "--subject",
        type=int,
        default=1,
        help="Subject index X to use (sbj_X). Default: 1 (sbj_1).",
    )
    p.add_argument(
        "--train_frac",
        type=float,
        default=0.8,
        help="Fraction of samples from this subject used for training (chronological).",
    )
    p.add_argument(
        "--thr_12",
        type=float,
        default=0.9,
        help="Confidence threshold for 12.5 Hz predictions (softmax max).",
    )
    p.add_argument(
        "--thr_25",
        type=float,
        default=0.9,
        help="Confidence threshold for 25 Hz predictions (softmax max).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use (default: cpu). Use cuda only if your GPU is supported.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    cfg_50 = load_config(args.cfg_50)
    cfg_25 = load_config(args.cfg_25)
    cfg_12 = load_config(args.cfg_12_5)

    # Take label_dict from config if present, otherwise fall back to the default.
    label_dict = cfg_50.get("label_dict", DEFAULT_LABEL_DICT)
    num_classes = max(label_dict.values()) + 1  # should be 19 (0..18)

    print(f"\n=== Within-subject experiment for sbj_{args.subject} ===")
    print(f"train_frac={args.train_frac}, thr_12={args.thr_12}, thr_25={args.thr_25}")

    # 1) Load single subject data for each frequency
    data_50 = load_subject_data(cfg_50, args.subject, label_dict)
    data_25 = load_subject_data(cfg_25, args.subject, label_dict)
    data_12 = load_subject_data(cfg_12, args.subject, label_dict)

    print(f"50 Hz data shape (sbj_{args.subject}): {data_50.shape}")
    print(f"25 Hz data shape (sbj_{args.subject}): {data_25.shape}")
    print(f"12.5 Hz data shape (sbj_{args.subject}): {data_12.shape}")

    # 2) Split + normalize
    train_50, val_50 = split_and_normalize(data_50, args.train_frac)
    train_25, val_25 = split_and_normalize(data_25, args.train_frac)
    train_12, val_12 = split_and_normalize(data_12, args.train_frac)

    # 3) Train three models (using same train/val logic as models/train.py)
    net_50, train_ds_50, val_ds_50 = train_single_freq_model(
        cfg_50, train_50, val_50, device, num_classes=num_classes, label="50 Hz"
    )
    net_25, train_ds_25, val_ds_25 = train_single_freq_model(
        cfg_25, train_25, val_25, device, num_classes=num_classes, label="25 Hz"
    )
    net_12, train_ds_12, val_ds_12 = train_single_freq_model(
        cfg_12, train_12, val_12, device, num_classes=num_classes, label="12.5 Hz"
    )

    # 4) Baseline 50 Hz eval on val windows
    print("\n=== Baseline 50 Hz (within-subject) on val windows ===")
    preds_50, gts_50 = run_model_on_dataset(net_50, val_ds_50, device)
    acc_50, prec_50, rec_50, f1_50 = compute_metrics_window_level(preds_50, gts_50)
    print(f"Acc: {acc_50:5.2f} %  Prec: {prec_50:5.2f} %  Rec: {rec_50:5.2f} %  F1: {f1_50:5.2f} %")

    # 5) Cascade (12.5 -> 25 -> 50) eval on same val windows
    print("\n=== Cascade (12.5 -> 25 -> 50 Hz) on val windows ===")
    cascade_preds, cascade_gts, stats = run_cascade_on_val(
        net_50, net_25, net_12,
        val_ds_50, val_ds_25, val_ds_12,
        device,
        thr_12=args.thr_12,
        thr_25=args.thr_25,
    )
    acc_c, prec_c, rec_c, f1_c = compute_metrics_window_level(cascade_preds, cascade_gts)

    print(f"Cascade Acc: {acc_c:5.2f} %  Prec: {prec_c:5.2f} %  Rec: {rec_c:5.2f} %  F1: {f1_c:5.2f} %")
    print(
        f"Windows at 12.5 Hz: {stats['frac_12']*100:5.1f}% ({stats['used_12']} / {stats['n_windows']})\n"
        f"Windows at 25.0 Hz: {stats['frac_25']*100:5.1f}% ({stats['used_25']} / {stats['n_windows']})\n"
        f"Windows at 50.0 Hz: {stats['frac_50']*100:5.1f}% ({stats['used_50']} / {stats['n_windows']})\n"
        f"Effective avg. rate: {stats['eff_rate']:5.2f} Hz\n"
        f"Relative power vs 50 Hz: {stats['rel_power']*100:5.1f} %\n"
        f"Approx. power saving: {stats['saving']*100:5.1f} %"
    )


if __name__ == "__main__":
    main()
