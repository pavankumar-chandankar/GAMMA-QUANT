#!/usr/bin/env python3
# ------------------------------------------------------------------------
# Cascade evaluation using the SAME data loading / preprocessing as train.py
#
# For each LOSO split:
#   - Rebuild train_data, val_data exactly like run_inertial_network:
#       * CSVs from cfg['dataset']['sens_folder']
#       * .replace({"label": label_dict}).fillna(0)
#       * normalize with min/max of training data to [-1, 1]
#   - Build InertialDataset for val_data for:
#       * 50 Hz config
#       * 25 Hz config
#       * 12.5 Hz config
#   - Load checkpoints from:
#         logs/deepconvlstm/50hz/loso_sbj_k/best_loso_sbj_k.pth.tar
#         logs/deepconvlstm/25hz/loso_sbj_k/best_loso_sbj_k.pth.tar
#         logs/deepconvlstm/12_5hz/loso_sbj_k/best_loso_sbj_k.pth.tar
#   - Run cascade: 12.5 Hz -> 25 Hz -> 50 Hz
#
# This script does NOT modify models/train.py.
# ------------------------------------------------------------------------

import argparse
import os
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from utils.os_utils import load_config
from utils.torch_utils import InertialDataset
from models.DeepConvLSTM import DeepConvLSTM

# ------------------------------------------------------------------------
# Default label_dict (same mapping you sent)
# In training, main.py probably injects this into cfg["label_dict"].
# Here we recreate it if it's missing.
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

# ------------------------------------------------------------------------
# Frequency-based class grouping
# ------------------------------------------------------------------------

# Low-frequency classes (stretching)
LOW_FREQ_CLASSES = [6, 7, 8, 9, 10]

# Example mid-frequency classes
MID_FREQ_CLASSES = [16, 17, 18]


# ------------------------------------------------------------------------
# Data loading helper: copy logic from run_inertial_network
# ------------------------------------------------------------------------

def build_train_val_data_for_split(
    cfg: dict,
    split_idx: int,
    num_subjects: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reproduce the data loading & normalization from run_inertial_network,
    but for a specific LOSO split.

    - train_sbjs: all subjects except sbj_<split_idx>
    - val_sbjs:   [sbj_<split_idx>]
    - Same CSV loading, label replacement, fillna, and normalization.
    """
    input_dim = cfg['dataset']['input_dim']
    sens_folder = cfg['dataset']['sens_folder']

    # If main.py didn't inject label_dict into cfg, fall back to DEFAULT_LABEL_DICT
    label_dict = cfg.get('label_dict', DEFAULT_LABEL_DICT)

    # LOSO subjects
    all_sbjs = [f"sbj_{i}" for i in range(num_subjects)]
    val_sbjs = [f"sbj_{split_idx}"]
    train_sbjs = [s for s in all_sbjs if s not in val_sbjs]

    # Allocate arrays
    train_data = np.empty((0, input_dim + 2))
    val_data = np.empty((0, input_dim + 2))

    # Training subjects
    for t_sbj in train_sbjs:
        t_path = os.path.join(sens_folder, t_sbj + '.csv')
        if not os.path.isfile(t_path):
            raise FileNotFoundError(f"Training subject file not found: {t_path}")
        t_df = pd.read_csv(t_path, index_col=False, low_memory=False)
        # Same as train.py, plus infer_objects to silence FutureWarning
        t_df = t_df.replace({"label": label_dict}).infer_objects(copy=False).fillna(0)
        t_data = t_df.to_numpy()
        if t_data.shape[1] != input_dim + 2:
            raise RuntimeError(
                f"[{t_sbj}] Expected {input_dim+2} cols, got {t_data.shape[1]}"
            )
        train_data = np.append(train_data, t_data, axis=0)

    # Validation subjects (only one in LOSO)
    for v_sbj in val_sbjs:
        v_path = os.path.join(sens_folder, v_sbj + '.csv')
        if not os.path.isfile(v_path):
            raise FileNotFoundError(f"Validation subject file not found: {v_path}")
        v_df = pd.read_csv(v_path, index_col=False, low_memory=False)
        v_df = v_df.replace({"label": label_dict}).infer_objects(copy=False).fillna(0)
        v_data = v_df.to_numpy()
        if v_data.shape[1] != input_dim + 2:
            raise RuntimeError(
                f"[{v_sbj}] Expected {input_dim+2} cols, got {v_data.shape[1]}"
            )
        val_data = np.append(val_data, v_data, axis=0)

    # Normalize using training data stats (exactly like train.py)
    min_train = np.min(train_data[:, 1:-1])
    max_train = np.max(train_data[:, 1:-1])

    train_data[:, 1:-1] = 2 * (train_data[:, 1:-1] - min_train) / (max_train - min_train) - 1
    val_data[:, 1:-1]   = 2 * (val_data[:, 1:-1]   - min_train) / (max_train - min_train) - 1

    train_data = train_data.astype(np.float32)
    val_data = val_data.astype(np.float32)

    return train_data, val_data


# ------------------------------------------------------------------------
# Model construction and checkpoint loading
# ------------------------------------------------------------------------

def build_model_from_cfg(
    cfg: dict,
    num_classes_override: Optional[int] = None,
    quant_bits: int = 4,
) -> DeepConvLSTM:
    """
    Build DeepConvLSTM with same hyper-params as training.
    Gamma quantization is set to the 4-bit setup you used.
    """
    input_dim = cfg['dataset']['input_dim']
    if num_classes_override is None:
        num_classes = cfg['dataset']['num_classes']
    else:
        num_classes = num_classes_override

    window_size = cfg['dataset']['window_size']
    conv_kernels = cfg['model']['conv_kernels']
    conv_kernel_size = cfg['model']['conv_kernel_size']
    lstm_units = cfg['model']['lstm_units']
    lstm_layers = cfg['model']['lstm_layers']
    dropout = cfg['model']['dropout']

    conv_type = cfg['model'].get('conv_type', 'standard')
    conv_rank = cfg['model'].get('conv_rank', 8)
    conv_mlp_hidden_dim = cfg['model'].get('conv_mlp_hidden_dim', 32)

    net = DeepConvLSTM(
        input_dim,
        num_classes,
        window_size,
        conv_kernels,
        conv_kernel_size,
        lstm_units,
        lstm_layers,
        dropout,
        gamma_quant="quantize_and_gamma",
        quant_bits=quant_bits,
        apply_gamma="global",
        gamma_type="s_shaped",
        conv_type=conv_type,
        conv_rank=conv_rank,
        conv_mlp_hidden_dim=conv_mlp_hidden_dim,
    )
    return net


def load_model_for_freq(
    freq_tag: str,
    cfg: dict,
    ckpt_root: str,
    split_idx: int,
    device: torch.device,
) -> torch.nn.Module:
    """
    freq_tag: "50hz", "25hz", "12_5hz"
    ckpt path: logs/deepconvlstm/<freq_tag>/loso_sbj_k/best_loso_sbj_k.pth.tar
    """
    split_name = f"loso_sbj_{split_idx}"
    ckpt_dir = os.path.join(ckpt_root, freq_tag, split_name)
    ckpt_path = os.path.join(ckpt_dir, f"best_{split_name}.pth.tar")

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found for freq {freq_tag}, split {split_name}: {ckpt_path}"
        )

    state = torch.load(ckpt_path, map_location=device)
    classifier_weight = state["state_dict"]["classifier.weight"]
    num_classes_ckpt = classifier_weight.shape[0]

    net = build_model_from_cfg(cfg, num_classes_override=num_classes_ckpt)
    net.load_state_dict(state["state_dict"])
    net.to(device)
    net.eval()
    return net


# ------------------------------------------------------------------------
# Metrics (match train.py style: confusion_matrix + macro P/R/F1)
# ------------------------------------------------------------------------

def compute_metrics(
    preds: np.ndarray,
    gts: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    Use same idea as train.py:

        conf_mat = confusion_matrix(v_gt, v_preds, normalize='true')
        v_acc = conf_mat.diagonal()
        v_prec = precision_score(..., average=None)
        v_rec = recall_score(..., average=None)
        v_f1 = f1_score(..., average=None)
        then macro-average them.

    Returns (acc, prec, rec, f1) in percentages.
    """
    assert preds.shape == gts.shape
    if preds.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    # confusion_matrix automatically handles whatever labels are present
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


# ------------------------------------------------------------------------
# Baseline prediction (single model on a dataset)
# ------------------------------------------------------------------------

def run_model_on_dataset(
    net: torch.nn.Module,
    dataset: InertialDataset,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run a model on every window in a dataset (batch_size = 1).
    Returns: preds, gts (both np arrays of shape (N,))
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
            elif x.ndim == 3 and x.shape[0] != 1:
                x = x.unsqueeze(0)

            logits = net(x)
            if logits.ndim == 1:
                probs = F.softmax(logits, dim=0)
            else:
                probs = F.softmax(logits, dim=-1)[0]

            _, pred = torch.max(probs, dim=-1)
            preds.append(int(pred.item()))
            gts.append(int(y))

    return np.array(preds, dtype=np.int64), np.array(gts, dtype=np.int64)


# ------------------------------------------------------------------------
# Cascade for one LOSO split
# ------------------------------------------------------------------------

def run_cascade_for_split(
    split_idx: int,
    cfg_50: dict,
    cfg_25: dict,
    cfg_12: dict,
    ckpt_root: str,
    num_subjects: int,
    device: torch.device,
    thr_12: float = 0.9,
    thr_25: float = 0.9,
) -> Dict[str, float]:
    """
    For one LOSO split:
      - Rebuild train/val data with EXACT same logic as run_inertial_network.
      - Build InertialDataset for val_data for 50/25/12.5 Hz.
      - Load three pretrained models.
      - Run cascade 12.5 -> 25 -> 50.
    """
    print(f"\n=== Split loso_sbj_{split_idx} ===")

    # EXACT same data pipeline as train.py, but per config
    _, val_50 = build_train_val_data_for_split(cfg_50, split_idx, num_subjects)
    _, val_25 = build_train_val_data_for_split(cfg_25, split_idx, num_subjects)
    _, val_12 = build_train_val_data_for_split(cfg_12, split_idx, num_subjects)

    # Build datasets (same InertialDataset as in training)
    ds_50 = InertialDataset(
        val_50,
        cfg_50['dataset']['window_size'],
        cfg_50['dataset']['window_overlap'],
        model=cfg_50['name'],
    )
    ds_25 = InertialDataset(
        val_25,
        cfg_25['dataset']['window_size'],
        cfg_25['dataset']['window_overlap'],
        model=cfg_25['name'],
    )
    ds_12 = InertialDataset(
        val_12,
        cfg_12['dataset']['window_size'],
        cfg_12['dataset']['window_overlap'],
        model=cfg_12['name'],
    )

    n_12 = len(ds_12)
    n_25 = len(ds_25)
    n_50 = len(ds_50)
    n = min(n_12, n_25, n_50)

    if not (n_12 == n_25 == n_50):
        print(
            f"Warning: window counts differ (12.5 Hz: {n_12}, "
            f"25 Hz: {n_25}, 50 Hz: {n_50}). Using first {n} windows."
        )

    # Load models
    net_50 = load_model_for_freq("50hz", cfg_50, ckpt_root, split_idx, device)
    net_25 = load_model_for_freq("25hz", cfg_25, ckpt_root, split_idx, device)
    net_12 = load_model_for_freq("12_5hz", cfg_12, ckpt_root, split_idx, device)

    # Baseline 50 Hz
    print("Running baseline 50 Hz model on validation windows...")
    preds_50_all, gts_all = run_model_on_dataset(net_50, ds_50, device)
    preds_50 = preds_50_all[:n]
    gts = gts_all[:n]

    acc_50, prec_50, rec_50, f1_50 = compute_metrics(preds_50, gts)
    print(f"Baseline 50 Hz acc vs labels: {acc_50:5.2f} % (F1 {f1_50:5.2f} %)")

    # Cascade 12.5 -> 25 -> 50
    net_50.eval()
    net_25.eval()
    net_12.eval()

    cascade_preds: List[int] = []
    used_12 = 0
    used_25 = 0
    used_50 = 0

    print("Running cascade (12.5 -> 25 -> 50 Hz) on validation windows...")

    with torch.no_grad():
        for idx in range(n):
            x12_np, y12 = ds_12[idx]
            x25_np, y25 = ds_25[idx]
            x50_np, y50 = ds_50[idx]

            # GT label is same reference as baseline
            y = int(y50)

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
                    # Step 3: 50 Hz fallback
                    logits50 = net_50(x50)
                    p50 = F.softmax(logits50, dim=-1)[0]
                    _, pred50 = torch.max(p50, dim=-1)
                    final_pred = int(pred50.item())
                    used_50 += 1

            cascade_preds.append(final_pred)

    cascade_preds = np.array(cascade_preds, dtype=np.int64)
    gts_cascade = gts

    acc_c, prec_c, rec_c, f1_c = compute_metrics(cascade_preds, gts_cascade)

    print(f"Cascade acc vs labels:        {acc_c:5.2f} % (F1 {f1_c:5.2f} %)")

    total = n
    frac_12 = used_12 / total if total > 0 else 0.0
    frac_25 = used_25 / total if total > 0 else 0.0
    frac_50 = used_50 / total if total > 0 else 0.0

    eff_rate = 12.5 * frac_12 + 25.0 * frac_25 + 50.0 * frac_50
    rel_power = eff_rate / 50.0 if total > 0 else 0.0
    saving = 1.0 - rel_power

    print(f"Windows at 12.5 Hz: {frac_12 * 100:5.1f} %  ({used_12} / {total})")
    print(f"Windows at 25.0 Hz: {frac_25 * 100:5.1f} %  ({used_25} / {total})")
    print(f"Windows at 50.0 Hz: {frac_50 * 100:5.1f} %  ({used_50} / {total})")
    print(f"Effective avg. rate:       {eff_rate:5.2f} Hz")
    print(f"Relative power vs 50 Hz:   {rel_power * 100:5.1f} %")
    print(f"Approx. power saving:      {saving * 100:5.1f} %")

    return {
        "baseline_acc": acc_50,
        "baseline_f1": f1_50,
        "cascade_acc": acc_c,
        "cascade_f1": f1_c,
        "eff_rate": eff_rate,
        "saving": saving * 100.0,
    }


# ------------------------------------------------------------------------
# CLI + main
# ------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cascade evaluation using pretrained 12.5/25/50 Hz models with train.py-style data pipeline."
    )
    parser.add_argument(
        "--cfg_50",
        type=str,
        default="configs/deepconvlstm/wear_loso.yaml",
        help="Config file for 50 Hz model.",
    )
    parser.add_argument(
        "--cfg_25",
        type=str,
        default="configs/deepconvlstm/wear_loso_25hz.yaml",
        help="Config file for 25 Hz model.",
    )
    parser.add_argument(
        "--cfg_12_5",
        type=str,
        default="configs/deepconvlstm/wear_loso_12_5hz.yaml",
        help="Config file for 12.5 Hz model.",
    )
    parser.add_argument(
        "--ckpt_root",
        type=str,
        default="logs/deepconvlstm",
        help="Root directory where freq-specific logs are stored.",
    )
    parser.add_argument(
        "--num_subjects",
        type=int,
        default=22,
        help="Number of subjects (sbj_0..sbj_{num_subjects-1}).",
    )
    parser.add_argument(
        "--start_split",
        type=int,
        default=0,
        help="First LOSO split index (inclusive).",
    )
    parser.add_argument(
        "--end_split",
        type=int,
        default=3,
        help="Last LOSO split index (inclusive).",
    )
    parser.add_argument(
        "--thr_12",
        type=float,
        default=0.9,
        help="Confidence threshold for 12.5 Hz prediction.",
    )
    parser.add_argument(
        "--thr_25",
        type=float,
        default=0.9,
        help="Confidence threshold for 25 Hz prediction.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cfg_50 = load_config(args.cfg_50)
    cfg_25 = load_config(args.cfg_25)
    cfg_12 = load_config(args.cfg_12_5)

    all_baseline_acc = []
    all_baseline_f1 = []
    all_cascade_acc = []
    all_cascade_f1 = []
    all_eff_rate = []
    all_saving = []

    for split_idx in range(args.start_split, args.end_split + 1):
        metrics = run_cascade_for_split(
            split_idx,
            cfg_50,
            cfg_25,
            cfg_12,
            args.ckpt_root,
            args.num_subjects,
            device,
            thr_12=args.thr_12,
            thr_25=args.thr_25,
        )
        all_baseline_acc.append(metrics["baseline_acc"])
        all_baseline_f1.append(metrics["baseline_f1"])
        all_cascade_acc.append(metrics["cascade_acc"])
        all_cascade_f1.append(metrics["cascade_f1"])
        all_eff_rate.append(metrics["eff_rate"])
        all_saving.append(metrics["saving"])

    if len(all_baseline_acc) > 0:
        print("\n=== Overall over splits {}..{} ===".format(
            args.start_split, args.end_split
        ))
        print("Mean baseline acc:  {:5.2f} %".format(float(np.mean(all_baseline_acc))))
        print("Mean baseline F1:   {:5.2f} %".format(float(np.mean(all_baseline_f1))))
        print("Mean cascade acc:   {:5.2f} %".format(float(np.mean(all_cascade_acc))))
        print("Mean cascade F1:    {:5.2f} %".format(float(np.mean(all_cascade_f1))))
        print("Mean eff. rate:     {:5.2f} Hz".format(float(np.mean(all_eff_rate))))
        print("Mean power saving:  {:5.2f} %".format(float(np.mean(all_saving))))


if __name__ == "__main__":
    main()
