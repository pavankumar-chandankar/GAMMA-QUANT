#!/usr/bin/env python3
# ------------------------------------------------------------------------
# Multi-model ensemble over LOSO splits for a *new* test subject.
#
# Idea:
#   - Use multiple LOSO models (e.g., loso_sbj_1 .. loso_sbj_21) as
#     "experts" and evaluate their ensemble on a separate subject, e.g. sbj_22.
#
# For each frequency (12.5, 25, 50 Hz) and each split in train_splits:
#   1) Compute min/max from that split's training subjects.
#   2) Normalize the *test subject's* CSV using those min/max.
#   3) Segment into windows (window_size / window_overlap from cfg).
#   4) Load checkpoint logs/deepconvlstm/{freq_tag}/loso_sbj_K/best_loso_sbj_K.pth.tar
#   5) Run model -> per-window probability over classes.
#
# Then:
#   - For each frequency, average probabilities across splits:
#       probs_ens_freq = mean(probs_freq_splits, axis=0)
#       preds_ens_freq = argmax(probs_ens_freq, axis=-1)
#   - Compare preds_ens_freq to true labels of test subject:
#       Acc / macro Prec / macro Rec / macro F1
#
# Strictly:
#   - 12.5 Hz ensemble uses 12.5 Hz windows & model.
#   - 25.0 Hz ensemble uses 25 Hz windows & model.
#   - 50.0 Hz ensemble uses 50 Hz windows & model.
# ------------------------------------------------------------------------

import argparse
import os
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from utils.os_utils import load_config
from utils.torch_utils import InertialDataset
from models.DeepConvLSTM import DeepConvLSTM


# ------------------------------------------------------------------------
# Label mapping (your label_dict)
# ------------------------------------------------------------------------

LABEL_DICT: Dict[str, int] = {
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
    'bench-dips': 18
}


def encode_labels_inplace(data: np.ndarray) -> np.ndarray:
    """
    Ensure last column (labels) is numeric by mapping strings via LABEL_DICT.
    """
    labels = data[:, -1]
    if np.issubdtype(labels.dtype, np.number):
        return data

    mapped = np.zeros(labels.shape[0], dtype=np.float32)
    for i, val in enumerate(labels):
        mapped[i] = float(LABEL_DICT.get(str(val), 0))
    data[:, -1] = mapped
    return data


# ------------------------------------------------------------------------
# Min/max and test data construction for a given split & subject
# ------------------------------------------------------------------------

def compute_minmax_for_split(cfg: dict, train_split_idx: int, num_subjects: int) -> Tuple[float, float]:
    """
    Compute min/max over training subjects of a LOSO split for a given frequency config.
    """
    sens_folder = cfg["dataset"]["sens_folder"]
    all_sbjs = [f"sbj_{i}" for i in range(num_subjects)]
    val_sbj = all_sbjs[train_split_idx]
    train_sbjs = [s for s in all_sbjs if s != val_sbj]

    train_data_list = []
    for t_sbj in train_sbjs:
        t_path = os.path.join(sens_folder, t_sbj + ".csv")
        if not os.path.isfile(t_path):
            raise FileNotFoundError(f"Training subject file not found: {t_path}")
        t_df = pd.read_csv(t_path, index_col=False, low_memory=False)
        t_df = t_df.fillna(0)
        t_data = t_df.to_numpy()
        t_data = encode_labels_inplace(t_data)
        train_data_list.append(t_data)

    if not train_data_list:
        raise RuntimeError("No training data found for min/max computation.")

    train_data = np.concatenate(train_data_list, axis=0)
    min_train = float(np.min(train_data[:, 1:-1]))
    max_train = float(np.max(train_data[:, 1:-1]))
    return min_train, max_train


def build_test_data_for_subject(cfg: dict, test_subject_idx: int, min_train: float, max_train: float) -> np.ndarray:
    """
    Build normalized test data for a given subject under one frequency config.
    Normalization: [-1,1] using provided min_train / max_train.
    """
    sens_folder = cfg["dataset"]["sens_folder"]
    input_dim = cfg["dataset"]["input_dim"]

    sbj_name = f"sbj_{test_subject_idx}"
    t_path = os.path.join(sens_folder, sbj_name + ".csv")
    if not os.path.isfile(t_path):
        raise FileNotFoundError(f"Test subject file not found: {t_path}")

    df = pd.read_csv(t_path, index_col=False, low_memory=False)
    df = df.fillna(0)
    data = df.to_numpy()
    data = encode_labels_inplace(data)

    data[:, 1:-1] = 2 * (data[:, 1:-1] - min_train) / (max_train - min_train) - 1
    data = data.astype(np.float32)

    if data.shape[1] != input_dim + 2:
        raise RuntimeError(
            f"Expected data with {input_dim + 2} columns, got {data.shape[1]}"
        )
    return data


# ------------------------------------------------------------------------
# Model construction
# ------------------------------------------------------------------------

def build_model_from_cfg(cfg: dict, num_classes: int) -> DeepConvLSTM:
    """
    Build DeepConvLSTM matching the cfg, with num_classes outputs.
    """
    input_dim = cfg["dataset"]["input_dim"]
    window_size = cfg["dataset"]["window_size"]
    conv_kernels = cfg["model"]["conv_kernels"]
    conv_kernel_size = cfg["model"]["conv_kernel_size"]
    lstm_units = cfg["model"]["lstm_units"]
    lstm_layers = cfg["model"]["lstm_layers"]
    dropout = cfg["model"]["dropout"]

    conv_type = cfg["model"].get("conv_type", "standard")
    conv_rank = cfg["model"].get("conv_rank", 8)
    conv_mlp_hidden_dim = cfg["model"].get("conv_mlp_hidden_dim", 32)

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
        quant_bits=4,
        apply_gamma="global",
        gamma_type="s_shaped",
        conv_type=conv_type,
        conv_rank=conv_rank,
        conv_mlp_hidden_dim=conv_mlp_hidden_dim,
    )
    return net


# ------------------------------------------------------------------------
# Inference returning *probabilities* (for ensembling)
# ------------------------------------------------------------------------

def run_model_return_probs(
    net: torch.nn.Module,
    dataset: InertialDataset,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run model on dataset, return:
      - probs: shape (N_windows, num_classes)
      - gts:   shape (N_windows,)
    """
    net.eval()
    probs_list: List[np.ndarray] = []
    gts_list: List[int] = []

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
                p = F.softmax(logits, dim=0).cpu().numpy()
            else:
                p = F.softmax(logits, dim=-1)[0].cpu().numpy()

            probs_list.append(p)
            gts_list.append(int(y))

    probs = np.stack(probs_list, axis=0)
    gts = np.array(gts_list, dtype=np.int64)
    return probs, gts


# ------------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------------

def compute_metrics(
    preds: np.ndarray,
    gts: np.ndarray,
    num_classes: int,
) -> Tuple[float, float, float, float]:
    """
    Compute accuracy and macro-averaged precision/recall/F1.
    """
    assert preds.shape == gts.shape
    N = preds.shape[0]
    if N == 0:
        return 0.0, 0.0, 0.0, 0.0

    acc = float(np.mean(preds == gts) * 100.0)

    precisions = []
    recalls = []
    f1s = []

    for c in range(num_classes):
        tp = np.sum((preds == c) & (gts == c))
        fp = np.sum((preds == c) & (gts != c))
        fn = np.sum((preds != c) & (gts == c))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    macro_prec = float(np.mean(precisions) * 100.0)
    macro_rec = float(np.mean(recalls) * 100.0)
    macro_f1 = float(np.mean(f1s) * 100.0)

    return acc, macro_prec, macro_rec, macro_f1


# ------------------------------------------------------------------------
# Single-frequency ensemble over multiple splits for one test subject
# ------------------------------------------------------------------------

def ensemble_for_frequency(
    cfg: dict,
    freq_tag: str,
    ckpt_root: str,
    train_splits: List[int],
    test_subject_idx: int,
    num_subjects: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For a given frequency config and freq_tag ("12_5hz", "25hz", "50hz"),
    build a probability ensemble across all train_splits on test_subject_idx.

    Returns:
      - probs_ens:  (N_windows, num_classes)
      - gts_final:  (N_windows,)
    """
    sens_folder = cfg["dataset"]["sens_folder"]
    input_dim = cfg["dataset"]["input_dim"]
    window_size = cfg["dataset"]["window_size"]
    window_overlap = cfg["dataset"]["window_overlap"]

    # We'll collect per-split probabilities and labels
    probs_splits: List[np.ndarray] = []
    gts_splits: List[np.ndarray] = []

    num_classes_global: int = -1
    N_global: int = -1

    for split_idx in train_splits:
        # 1) Min/max for this split
        min_train, max_train = compute_minmax_for_split(cfg, split_idx, num_subjects)

        # 2) Test data for this split (subject sbj_test)
        test_data = build_test_data_for_subject(cfg, test_subject_idx, min_train, max_train)

        # 3) Dataset for this split (windowing is same for all splits -> same N)
        ds = InertialDataset(test_data, window_size, window_overlap, model=cfg["name"])

        # 4) Load model checkpoint
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

        if num_classes_global < 0:
            num_classes_global = num_classes_ckpt
        elif num_classes_ckpt != num_classes_global:
            raise RuntimeError(
                f"Split {split_idx} has num_classes={num_classes_ckpt}, "
                f"expected {num_classes_global}"
            )

        model = build_model_from_cfg(cfg, num_classes=num_classes_ckpt)
        model.load_state_dict(state["state_dict"])
        model.to(device)

        # 5) Run model -> probabilities and ground-truth labels
        probs, gts = run_model_return_probs(model, ds, device)

        if N_global < 0:
            N_global = probs.shape[0]
        elif probs.shape[0] != N_global:
            raise RuntimeError(
                f"Split {split_idx} produced {probs.shape[0]} windows, "
                f"expected {N_global} (train_splits must share windowing)."
            )

        probs_splits.append(probs)
        gts_splits.append(gts)

    # Stack over splits and average probabilities
    probs_all = np.stack(probs_splits, axis=0)   # (num_splits, N, num_classes)
    probs_ens = np.mean(probs_all, axis=0)       # (N, num_classes)

    # Check that all gts are identical across splits
    for i in range(1, len(gts_splits)):
        if not np.array_equal(gts_splits[i], gts_splits[0]):
            raise RuntimeError(
                "Ground-truth labels differ across splits for the same test subject "
                "(check label mapping!)."
            )
    gts_final = gts_splits[0]

    return probs_ens, gts_final


# ------------------------------------------------------------------------
# CLI + main
# ------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-model LOSO ensemble for a new test subject (per frequency)."
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
        default=23,
        help="Total number of subjects with CSVs (sbj_0..sbj_{num_subjects-1}). "
             "If you want sbj_22 as test, set num_subjects >= 23.",
    )
    parser.add_argument(
        "--train_splits",
        type=int,
        nargs="+",
        default=list(range(1, 22)),
        help="List of LOSO split indices to use as ensemble members "
             "(e.g., 1 2 3 ... 21).",
    )
    parser.add_argument(
        "--test_subject",
        type=int,
        default=22,
        help="Index of subject used as test (e.g., 22 for sbj_22).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Train splits (ensemble): {args.train_splits}")
    print(f"Test subject:            sbj_{args.test_subject}")

    cfg_50 = load_config(args.cfg_50)
    cfg_25 = load_config(args.cfg_25)
    cfg_12 = load_config(args.cfg_12_5)

    num_classes = len(LABEL_DICT)

    # 50 Hz ensemble
    print("\n=== 50 Hz ensemble ===")
    probs_50, gts_50 = ensemble_for_frequency(
        cfg_50, "50hz", args.ckpt_root, args.train_splits, args.test_subject, args.num_subjects, device
    )
    preds_50 = np.argmax(probs_50, axis=-1)
    acc_50, prec_50, rec_50, f1_50 = compute_metrics(preds_50, gts_50, num_classes)
    print(f"Acc:  {acc_50:5.2f} %")
    print(f"Prec: {prec_50:5.2f} %")
    print(f"Rec:  {rec_50:5.2f} %")
    print(f"F1:   {f1_50:5.2f} %")

    # 25 Hz ensemble
    print("\n=== 25 Hz ensemble ===")
    probs_25, gts_25 = ensemble_for_frequency(
        cfg_25, "25hz", args.ckpt_root, args.train_splits, args.test_subject, args.num_subjects, device
    )
    preds_25 = np.argmax(probs_25, axis=-1)
    acc_25, prec_25, rec_25, f1_25 = compute_metrics(preds_25, gts_25, num_classes)
    print(f"Acc:  {acc_25:5.2f} %")
    print(f"Prec: {prec_25:5.2f} %")
    print(f"Rec:  {rec_25:5.2f} %")
    print(f"F1:   {f1_25:5.2f} %")

    # 12.5 Hz ensemble
    print("\n=== 12.5 Hz ensemble ===")
    probs_12, gts_12 = ensemble_for_frequency(
        cfg_12, "12_5hz", args.ckpt_root, args.train_splits, args.test_subject, args.num_subjects, device
    )
    preds_12 = np.argmax(probs_12, axis=-1)
    acc_12, prec_12, rec_12, f1_12 = compute_metrics(preds_12, gts_12, num_classes)
    print(f"Acc:  {acc_12:5.2f} %")
    print(f"Prec: {prec_12:5.2f} %")
    print(f"Rec:  {rec_12:5.2f} %")
    print(f"F1:   {f1_12:5.2f} %")


if __name__ == "__main__":
    main()
