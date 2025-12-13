import argparse
import os
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from utils.os_utils import load_config
from utils.torch_utils import InertialDataset
from models.DeepConvLSTM import DeepConvLSTM


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
    labels = data[:, -1]
    if np.issubdtype(labels.dtype, np.number):
        return data

    mapped = np.zeros(labels.shape[0], dtype=np.float32)
    for i, val in enumerate(labels):
        mapped[i] = float(LABEL_DICT.get(str(val), 0))

    data[:, -1] = mapped
    return data


def compute_minmax_for_split(
    cfg: dict,
    train_split_idx: int,
    num_subjects: int,
) -> Tuple[float, float]:
    sens_folder = cfg["dataset"]["sens_folder"]

    all_sbjs = [f"sbj_{i}" for i in range(num_subjects)]
    val_sbj = all_sbjs[train_split_idx]
    train_sbjs = [s for s in all_sbjs if s != val_sbj]

    train_data_list = []

    for t_sbj in train_sbjs:
        t_path = os.path.join(sens_folder, t_sbj + ".csv")
        t_df = pd.read_csv(t_path, index_col=False, low_memory=False)
        t_df = t_df.fillna(0)
        t_data = t_df.to_numpy()
        t_data = encode_labels_inplace(t_data)
        train_data_list.append(t_data)

    if len(train_data_list) == 0:
        raise RuntimeError("No training subjects found for min/max computation.")

    train_data = np.concatenate(train_data_list, axis=0)
    min_train = float(np.min(train_data[:, 1:-1]))
    max_train = float(np.max(train_data[:, 1:-1]))

    return min_train, max_train


def build_test_data_for_subject(
    cfg: dict,
    test_subject_idx: int,
    min_train: float,
    max_train: float,
) -> np.ndarray:
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

def build_model_from_cfg(
    cfg: dict,
    num_classes_override: Optional[int] = None,
    quant_bits: int = 4,
) -> DeepConvLSTM:
    input_dim = cfg["dataset"]["input_dim"]
    if num_classes_override is None:
        num_classes = cfg["dataset"]["num_classes"]
    else:
        num_classes = num_classes_override

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
        quant_bits=quant_bits,
        apply_gamma="global",
        gamma_type="s_shaped",
        conv_type=conv_type,
        conv_rank=conv_rank,
        conv_mlp_hidden_dim=conv_mlp_hidden_dim,
    )
    return net



def run_model_on_dataset(
    net: torch.nn.Module,
    dataset: InertialDataset,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run a model on every window in a dataset (batch_size = 1).

    Returns:
        preds: predicted class index per window
        confs: max softmax confidence per window
        gts:   ground-truth label from the dataset
    """
    net.eval()
    preds = []
    confs = []
    gts = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            x_np, y = dataset[idx]  # y: scalar label
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

            conf, pred = torch.max(probs, dim=-1)
            preds.append(int(pred.item()))
            confs.append(float(conf.item()))
            gts.append(int(y))

    return np.array(preds), np.array(confs), np.array(gts)


# ------------------------------------------------------------------------
# Simple macro metrics: Acc / Prec / Rec / F1
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

        if tp + fp > 0:
            prec = tp / (tp + fp)
        else:
            prec = 0.0

        if tp + fn > 0:
            rec = tp / (tp + fn)
        else:
            rec = 0.0

        if prec + rec > 0:
            f1 = 2.0 * prec * rec / (prec + rec)
        else:
            f1 = 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    macro_prec = float(np.mean(precisions) * 100.0)
    macro_rec = float(np.mean(recalls) * 100.0)
    macro_f1 = float(np.mean(f1s) * 100.0)

    return acc, macro_prec, macro_rec, macro_f1


# ------------------------------------------------------------------------
# Cascade simulation for one train/test pair
# ------------------------------------------------------------------------

def simulate_cascade_cross_subject(
    train_split_idx: int,
    test_subject_idx: int,
    cfg_12: dict,
    cfg_25: dict,
    cfg_50: dict,
    ckpt_root: str,
    num_subjects: int,
    class_to_pref: Dict[int, str],
    threshold_up_low: float,
    threshold_up_mid: float,
    threshold_down: float,
    down_stable_windows: int,
    device: torch.device,
) -> Dict[str, float]:
    """
    Simulate the cascade when checkpoints are taken from LOSO split
    'train_split_idx', but evaluated on a different test subject
    'test_subject_idx'.
    """
    print(f"\n=== Cross-subject cascade: train split loso_sbj_{train_split_idx}, "
          f"test subject sbj_{test_subject_idx} ===")

    # 1) Compute min/max for each frequency based on training subjects of train_split_idx
    min12, max12 = compute_minmax_for_split(cfg_12, train_split_idx, num_subjects)
    min25, max25 = compute_minmax_for_split(cfg_25, train_split_idx, num_subjects)
    min50, max50 = compute_minmax_for_split(cfg_50, train_split_idx, num_subjects)

    # 2) Build test data for the chosen subject at each frequency
    test_12 = build_test_data_for_subject(cfg_12, test_subject_idx, min12, max12)
    test_25 = build_test_data_for_subject(cfg_25, test_subject_idx, min25, max25)
    test_50 = build_test_data_for_subject(cfg_50, test_subject_idx, min50, max50)

    # 3) Datasets
    ds_12 = InertialDataset(
        test_12,
        cfg_12["dataset"]["window_size"],
        cfg_12["dataset"]["window_overlap"],
        model=cfg_12["name"],
    )
    ds_25 = InertialDataset(
        test_25,
        cfg_25["dataset"]["window_size"],
        cfg_25["dataset"]["window_overlap"],
        model=cfg_25["name"],
    )
    ds_50 = InertialDataset(
        test_50,
        cfg_50["dataset"]["window_size"],
        cfg_50["dataset"]["window_overlap"],
        model=cfg_50["name"],
    )

    # Equalize length by truncating to the minimum number of windows
    N = min(len(ds_12), len(ds_25), len(ds_50))
    if len(ds_12) != len(ds_25) or len(ds_25) != len(ds_50):
        print(
            f"Warning: window counts differ for test subject "
            f"(12.5 Hz: {len(ds_12)}, 25 Hz: {len(ds_25)}, 50 Hz: {len(ds_50)}). "
            f"Using first {N} windows."
        )

    # 4) Build models and load checkpoints from train_split_idx
    split_name = f"loso_sbj_{train_split_idx}"

    def load_model_for_freq(freq_tag: str, cfg: dict) -> torch.nn.Module:
        ckpt_dir = os.path.join(ckpt_root, freq_tag, split_name)
        ckpt_path = os.path.join(ckpt_dir, f"best_{split_name}.pth.tar")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint not found for freq {freq_tag}, split {split_name}: {ckpt_path}"
            )

        state = torch.load(ckpt_path, map_location=device)
        classifier_weight = state["state_dict"]["classifier.weight"]
        num_classes_ckpt = classifier_weight.shape[0]

        model = build_model_from_cfg(cfg, num_classes_override=num_classes_ckpt)
        model.load_state_dict(state["state_dict"])
        model.to(device)
        model.eval()
        return model

    net_12 = load_model_for_freq("12_5hz", cfg_12)
    net_25 = load_model_for_freq("25hz", cfg_25)
    net_50 = load_model_for_freq("50hz", cfg_50)

    # 5) Precompute predictions and confidences for each frequency
    print("Running models on test subject windows...")
    preds_12, confs_12, gts_12 = run_model_on_dataset(net_12, ds_12, device)
    preds_25, confs_25, gts_25 = run_model_on_dataset(net_25, ds_25, device)
    preds_50, confs_50, gts_50 = run_model_on_dataset(net_50, ds_50, device)

    preds_12 = preds_12[:N]
    confs_12 = confs_12[:N]
    preds_25 = preds_25[:N]
    confs_25 = confs_25[:N]
    preds_50 = preds_50[:N]
    confs_50 = confs_50[:N]
    gts = gts_50[:N]  # ground-truth from 50 Hz dataset for test subject

    # 6) Baseline 50 Hz metrics vs labels
    num_classes = len(LABEL_DICT)
    base_acc, base_prec, base_rec, base_f1 = compute_metrics(preds_50, gts, num_classes)

    # 7) Simulate cascade
    current_level = "low"  # start at 12.5 Hz
    last_pred_class = None
    stable_count = 0

    level_history: List[str] = []
    cascade_preds: List[int] = []

    def level_rank(level: str) -> int:
        if level == "low":
            return 0
        if level == "mid":
            return 1
        if level == "high":
            return 2
        raise ValueError(f"Unknown level '{level}'")

    fs_map = {"low": 12.5, "mid": 25.0, "high": 50.0}

    for t in range(N):
        # Select prediction and confidence for the current level
        if current_level == "low":
            pred_t = preds_12[t]
            conf_t = confs_12[t]
        elif current_level == "mid":
            pred_t = preds_25[t]
            conf_t = confs_25[t]
        else:
            pred_t = preds_50[t]
            conf_t = confs_50[t]

        cascade_preds.append(pred_t)
        level_history.append(current_level)

        # Preferred level for this predicted class
        pref_level = class_to_pref.get(pred_t, "high")

        # Upward transitions (escalate if not confident enough)
        new_level = current_level
        if current_level == "low" and conf_t < threshold_up_low:
            new_level = "mid"
        elif current_level == "mid" and conf_t < threshold_up_mid:
            new_level = "high"

        # Downward transitions: only if no upward move and confidence high
        if new_level == current_level and conf_t >= threshold_down:
            curr_rank = level_rank(current_level)
            pref_rank = level_rank(pref_level)

            if pref_rank < curr_rank:
                # Potential fallback candidate
                if last_pred_class == pred_t:
                    stable_count += 1
                else:
                    stable_count = 1
                    last_pred_class = pred_t

                if stable_count >= down_stable_windows:
                    new_level = pref_level
                    stable_count = 0
            else:
                # Preferred level not lower, reset stability
                stable_count = 0
                last_pred_class = pred_t
        else:
            # Upward move or confidence too low for fallback
            stable_count = 0
            last_pred_class = pred_t

        current_level = new_level

    cascade_preds = np.array(cascade_preds)
    level_history = np.array(level_history)

    # 8) Cascade metrics vs labels
    cas_acc, cas_prec, cas_rec, cas_f1 = compute_metrics(cascade_preds, gts, num_classes)

    # 9) Frequency usage & power
    n_low = np.sum(level_history == "low")
    n_mid = np.sum(level_history == "mid")
    n_high = np.sum(level_history == "high")

    frac_low = n_low / float(N)
    frac_mid = n_mid / float(N)
    frac_high = n_high / float(N)

    f_eff = fs_map["low"] * frac_low + fs_map["mid"] * frac_mid + fs_map["high"] * frac_high
    rel_power = f_eff / fs_map["high"]
    saving = (1.0 - rel_power) * 100.0

    print("\n--- Baseline 50 Hz (test subject) ---")
    print(f"Acc:  {base_acc:5.2f} %")
    print(f"Prec: {base_prec:5.2f} %")
    print(f"Rec:  {base_rec:5.2f} %")
    print(f"F1:   {base_f1:5.2f} %")

    print("\n--- Cascade (12.5 -> 25 -> 50 Hz) ---")
    print(f"Acc:  {cas_acc:5.2f} %")
    print(f"Prec: {cas_prec:5.2f} %")
    print(f"Rec:  {cas_rec:5.2f} %")
    print(f"F1:   {cas_f1:5.2f} %")

    print("\n--- Frequency usage & power ---")
    print(f"Windows at 12.5 Hz: {frac_low * 100:5.1f} %  ({n_low} / {N})")
    print(f"Windows at 25.0 Hz: {frac_mid * 100:5.1f} %  ({n_mid} / {N})")
    print(f"Windows at 50.0 Hz: {frac_high * 100:5.1f} %  ({n_high} / {N})")
    print(f"Effective avg. rate:       {f_eff:5.2f} Hz")
    print(f"Relative power vs 50 Hz:   {rel_power * 100:5.1f} %")
    print(f"Approx. power saving:      {saving:5.1f} %")

    return {
        "baseline_acc": base_acc,
        "baseline_prec": base_prec,
        "baseline_rec": base_rec,
        "baseline_f1": base_f1,
        "cascade_acc": cas_acc,
        "cascade_prec": cas_prec,
        "cascade_rec": cas_rec,
        "cascade_f1": cas_f1,
        "frac_low": frac_low,
        "frac_mid": frac_mid,
        "frac_high": frac_high,
        "f_eff": f_eff,
        "rel_power": rel_power,
        "saving": saving,
    }


# ------------------------------------------------------------------------
# CLI + main
# ------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-subject 3-level frequency cascade simulation for WEAR."
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
        help="Number of subjects (sbj_0..sbj_{num_subjects-1} used for training).",
    )
    parser.add_argument(
        "--train_split",
        type=int,
        default=5,
        help="Which LOSO split to use for checkpoints (e.g., 5 for loso_sbj_5).",
    )
    parser.add_argument(
        "--test_subject",
        type=int,
        default=21,
        help="Index of the subject to use as test set (e.g., 21 for sbj_21).",
    )
    parser.add_argument(
        "--threshold_up_low",
        type=float,
        default=0.70,
        help="Confidence threshold for low->mid escalation.",
    )
    parser.add_argument(
        "--threshold_up_mid",
        type=float,
        default=0.75,
        help="Confidence threshold for mid->high escalation.",
    )
    parser.add_argument(
        "--threshold_down",
        type=float,
        default=0.90,
        help="Confidence threshold for fallback to lower preferred level.",
    )
    parser.add_argument(
        "--down_stable_windows",
        type=int,
        default=3,
        help="Number of consecutive stable predictions before fallback.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Train split (checkpoints): loso_sbj_{args.train_split}")
    print(f"Test subject:              sbj_{args.test_subject}")

    # Load configs
    cfg_50 = load_config(args.cfg_50)
    cfg_25 = load_config(args.cfg_25)
    cfg_12 = load_config(args.cfg_12_5)

    # Preferred frequency per class (by numeric label ID).
    # Levels: "low"  -> 12.5 Hz
    #         "mid"  -> 25.0 Hz
    #         "high" -> 50.0 Hz
    CLASS_TO_PREF: Dict[int, str] = {
        0: "low",   # 'null'

        # Jogging variants -> high frequency, highly dynamic
        1: "high",  # 'jogging'
        2: "high",  # 'jogging (rotating arms)'
        3: "high",  # 'jogging (skipping)'
        4: "high",  # 'jogging (sidesteps)'
        5: "high",  # 'jogging (butt-kicks)'

        # Stretching -> low frequency, mostly static / slow
        6: "low",   # 'stretching (triceps)'
        7: "low",   # 'stretching (lunging)'
        8: "low",   # 'stretching (shoulders)'
        9: "low",   # 'stretching (hamstrings)'
        10: "low",  # 'stretching (lumbar rotation)'

        # Strength / core – simple variants -> mid
        11: "mid",  # 'push-ups'
        13: "mid",  # 'sit-ups'
        16: "mid",  # 'lunges'
        18: "mid",  # 'bench-dips'

        # Strength / core – complex or explosive -> high
        12: "high", # 'push-ups (complex)'
        14: "high", # 'sit-ups (complex)'
        15: "high", # 'burpees'
        17: "high", # 'lunges (complex)'
    }

    _ = simulate_cascade_cross_subject(
        train_split_idx=args.train_split,
        test_subject_idx=args.test_subject,
        cfg_12=cfg_12,
        cfg_25=cfg_25,
        cfg_50=cfg_50,
        ckpt_root=args.ckpt_root,
        num_subjects=args.num_subjects,
        class_to_pref=CLASS_TO_PREF,
        threshold_up_low=args.threshold_up_low,
        threshold_up_mid=args.threshold_up_mid,
        threshold_down=args.threshold_down,
        down_stable_windows=args.down_stable_windows,
        device=device,
    )


if __name__ == "__main__":
    main()
