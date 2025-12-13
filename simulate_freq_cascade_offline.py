import argparse
import os
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from utils.os_utils import load_config
from utils.torch_utils import InertialDataset
from models.DeepConvLSTM import DeepConvLSTM

# Label mapping
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

STRETCH_CLASSES = [6, 7, 8, 9, 10]


def encode_labels_inplace(data: np.ndarray) -> np.ndarray:
    labels = data[:, -1]
    if np.issubdtype(labels.dtype, np.number):
        return data

    mapped = np.zeros(labels.shape[0], dtype=np.float32)
    for i, val in enumerate(labels):
        mapped[i] = float(LABEL_DICT.get(str(val), 0))
    data[:, -1] = mapped
    return data


def build_train_val_data(
    cfg: dict,
    split_idx: int,
    num_subjects: int,
) -> Tuple[np.ndarray, np.ndarray]:
    sens_folder = cfg["dataset"]["sens_folder"]
    input_dim = cfg["dataset"]["input_dim"]

    all_sbjs = [f"sbj_{i}" for i in range(num_subjects)]
    val_sbjs = [all_sbjs[split_idx]]
    train_sbjs = [s for s in all_sbjs if s not in val_sbjs]

    train_data_list = []
    val_data_list = []

    # Training subjects
    for t_sbj in train_sbjs:
        t_path = os.path.join(sens_folder, t_sbj + ".csv")
        t_df = pd.read_csv(t_path, index_col=False, low_memory=False)
        t_df = t_df.fillna(0)
        t_data = t_df.to_numpy()
        t_data = encode_labels_inplace(t_data)
        train_data_list.append(t_data)

    # Validation subject
    for v_sbj in val_sbjs:
        v_path = os.path.join(sens_folder, v_sbj + ".csv")
        v_df = pd.read_csv(v_path, index_col=False, low_memory=False)
        v_df = v_df.fillna(0)
        v_data = v_df.to_numpy()
        v_data = encode_labels_inplace(v_data)
        val_data_list.append(v_data)

    train_data = (
        np.concatenate(train_data_list, axis=0)
        if train_data_list
        else np.empty((0, input_dim + 2))
    )
    val_data = (
        np.concatenate(val_data_list, axis=0)
        if val_data_list
        else np.empty((0, input_dim + 2))
    )

    min_train = np.min(train_data[:, 1:-1])
    max_train = np.max(train_data[:, 1:-1])

    train_data[:, 1:-1] = 2 * (train_data[:, 1:-1] - min_train) / (max_train - min_train) - 1
    val_data[:, 1:-1] = 2 * (val_data[:, 1:-1] - min_train) / (max_train - min_train) - 1

    train_data = train_data.astype(np.float32)
    val_data = val_data.astype(np.float32)

    return train_data, val_data


# Model construction deepconvlsstm

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
) -> Tuple[np.ndarray, np.ndarray]:
    net.eval()
    preds = []
    gts = []

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

    return np.array(preds), np.array(gts)

def compute_stretch_metrics(
    preds: np.ndarray,
    gts: np.ndarray,
    stretch_classes,
) -> Tuple[float, float, float, float, int]:
    mask = np.isin(gts, stretch_classes)
    preds_s = preds[mask]
    gts_s = gts[mask]
    n = preds_s.shape[0]

    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0

    acc = float(np.mean(preds_s == gts_s) * 100.0)

    precisions = []
    recalls = []
    f1s = []

    unique_gts = set(gts_s.tolist())
    for c in stretch_classes:
        if c not in unique_gts:
            continue

        tp = np.sum((preds_s == c) & (gts_s == c))
        fp = np.sum((preds_s == c) & (gts_s != c))
        fn = np.sum((preds_s != c) & (gts_s == c))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if prec + rec > 0:
            f1 = 2.0 * prec * rec / (prec + rec)
        else:
            f1 = 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    if len(precisions) == 0:
        return acc, 0.0, 0.0, 0.0, n

    macro_prec = float(np.mean(precisions) * 100.0)
    macro_rec = float(np.mean(recalls) * 100.0)
    macro_f1 = float(np.mean(f1s) * 100.0)

    return acc, macro_prec, macro_rec, macro_f1, n


def evaluate_split_12_5hz(
    split_idx: int,
    cfg_12: dict,
    ckpt_root: str,
    num_subjects: int,
    device: torch.device,
) -> Dict[str, float]:

    _, val_12 = build_train_val_data(cfg_12, split_idx, num_subjects)

    ds_12 = InertialDataset(
        val_12,
        cfg_12["dataset"]["window_size"],
        cfg_12["dataset"]["window_overlap"],
        model=cfg_12["name"],
    )

    split_name = f"loso_sbj_{split_idx}"
    ckpt_dir = os.path.join(ckpt_root, "12_5hz", split_name)
    ckpt_path = os.path.join(ckpt_dir, f"best_{split_name}.pth.tar")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found for 12.5 Hz, split {split_name}: {ckpt_path}"
        )

    state = torch.load(ckpt_path, map_location=device)
    classifier_weight = state["state_dict"]["classifier.weight"]
    num_classes_ckpt = classifier_weight.shape[0]

    net_12 = build_model_from_cfg(cfg_12, num_classes_override=num_classes_ckpt)
    net_12.load_state_dict(state["state_dict"])
    net_12.to(device)
    net_12.eval()

    print("Running 12.5 Hz model on validation windows...")
    preds_12, gts_12 = run_model_on_dataset(net_12, ds_12, device)

    acc, prec, rec, f1, n = compute_stretch_metrics(preds_12, gts_12, STRETCH_CLASSES)

    print(f"Number of stretching windows: {n}")
    print(f"Stretch-only Acc:  {acc:5.2f} %")
    print(f"Stretch-only Prec: {prec:5.2f} %")
    print(f"Stretch-only Rec:  {rec:5.2f} %")
    print(f"Stretch-only F1:   {f1:5.2f} %")

    return {
        "n": n,
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate 12.5 Hz DeepConvLSTM on stretching activities only."
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
        default=21,
        help="Last LOSO split index (inclusive).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cfg_12 = load_config(args.cfg_12_5)

    all_acc = []
    all_prec = []
    all_rec = []
    all_f1 = []
    total_n = 0

    for split_idx in range(args.start_split, args.end_split + 1):
        metrics = evaluate_split_12_5hz(
            split_idx=split_idx,
            cfg_12=cfg_12,
            ckpt_root=args.ckpt_root,
            num_subjects=args.num_subjects,
            device=device,
        )
        if metrics["n"] == 0:
            print(f"Warning: no stretching windows for split {split_idx}, skipping in averages.")
            continue

        total_n += metrics["n"]
        all_acc.append(metrics["acc"])
        all_prec.append(metrics["prec"])
        all_rec.append(metrics["rec"])
        all_f1.append(metrics["f1"])

    if len(all_acc) == 0:
        print("\nNo splits had stretching windows. Nothing to summarize.")
        return

    print("\n=== Overall stretching-only metrics over splits {}..{} ===".format(
        args.start_split, args.end_split
    ))
    print(f"Total stretching windows: {total_n}")
    print("Mean Acc:  {:5.2f} %".format(float(np.mean(all_acc))))
    print("Mean Prec: {:5.2f} %".format(float(np.mean(all_prec))))
    print("Mean Rec:  {:5.2f} %".format(float(np.mean(all_rec))))
    print("Mean F1:   {:5.2f} %".format(float(np.mean(all_f1))))


if __name__ == "__main__":
    main()
