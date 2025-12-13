import os
import time
from typing import Dict, List, Tuple

import numpy as np

import pandas as pd

def _encode_labels_to_int(df, cfg):
    """
    Ensure label column is integer class id.
    Supports cfg["label_dict"] as either {id: name} or {name: id}.
    Also moves label column to the last position (required by apply_sliding_window()).
    """
    # pick label column
    if "label" in df.columns:
        label_col = "label"
    elif "activity" in df.columns:
        label_col = "activity"
    else:
        label_col = df.columns[-1]

    # build mapping if labels are strings/objects
    if df[label_col].dtype == object:
        label_dict = cfg.get("label_dict", None)

        if label_dict is None:
            # fallback: factorize (NOT recommended for final, but prevents crash)
            codes, uniques = pd.factorize(df[label_col])
            df[label_col] = codes
        else:
            # detect orientation
            keys = list(label_dict.keys())
            vals = list(label_dict.values())

            keys_intlike = all(isinstance(k, (int,)) or (isinstance(k, str) and k.isdigit()) for k in keys)
            vals_str = all(isinstance(v, str) for v in vals)

            if keys_intlike and vals_str:
                # {id: name} -> invert to {name: id}
                name_to_id = {v: int(k) for k, v in label_dict.items()}
            else:
                # assume {name: id}
                name_to_id = {str(k): int(v) for k, v in label_dict.items()}

            df[label_col] = df[label_col].map(name_to_id)

            if df[label_col].isna().any():
                bad = df.loc[df[label_col].isna(), label_col].unique()
                raise ValueError(f"Unmapped labels found in column '{label_col}': {bad}")

            df[label_col] = df[label_col].astype(int)

    # move label column to end (apply_sliding_window expects last column = label)
    if label_col != df.columns[-1]:
        cols = [c for c in df.columns if c != label_col] + [label_col]
        df = df[cols]

    return df
import pandas as pd
import torch
import torch.nn as nn
from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader, Dataset

from models.DeepConvLSTM import DeepConvLSTM
from models.multifreq_deepconvlstm import MultiFreqDeepConvLSTM, FREQ_FOR_CLASS
from utils.data_utils import apply_sliding_window
from utils.logging_utils import classification_scores
from utils.torch_utils import (
    fix_random_seed,
    init_weights,
    save_checkpoint,
    worker_init_reset_seed,
)


# ---------------------------------------------------------------------------
# Simple dataset for a single frequency (already windowed)
# ---------------------------------------------------------------------------

class InertialWindowDataset(Dataset):
    """
    Dataset for a single frequency.

    features : (N, T, C)
    labels   : (N,)
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray, model_name: str = "deepconvlstm") -> None:
        assert features.ndim == 3, "features must be (N, T, C)"
        assert labels.ndim == 1, "labels must be (N,)"
        assert features.shape[0] == labels.shape[0], "features/labels mismatch"

        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.uint8)
        self.model_name = model_name

        self.window_size = self.features.shape[1]
        self.channels = self.features.shape[2]
        self.classes = int(labels.max()) + 1

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int):
        x = self.features[index]  # (T, C)
        y = self.labels[index]
        return x, y


class InertialMultiFreqDataset(Dataset):
    """
    Dataset that provides aligned windows at 25, 12 and 6 Hz.

    x_25 : (N, T_25, C)
    x_12 : (N, T_12, C)
    x_6  : (N, T_6, C)
    y    : (N,)
    """

    def __init__(
        self,
        x_25: np.ndarray,
        x_12: np.ndarray,
        x_6: np.ndarray,
        y: np.ndarray,
    ) -> None:
        assert x_25.shape[0] == x_12.shape[0] == x_6.shape[0] == y.shape[0], "All inputs must have same length."
        self.x_25 = x_25.astype(np.float32)
        self.x_12 = x_12.astype(np.float32)
        self.x_6 = x_6.astype(np.float32)
        self.y = y.astype(np.uint8)

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int):
        return self.x_25[idx], self.x_12[idx], self.x_6[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Build windowed dataset + 70/20/10 split for a single frequency
# ---------------------------------------------------------------------------

def build_single_freq_splits(
    cfg: Dict,
    sbj_list: List[str],
    freq_tag: str,
    seed: int,
) -> Tuple[InertialWindowDataset, InertialWindowDataset, Tuple[np.ndarray, np.ndarray]]:
    """
    Reads all subjects for a given frequency, normalizes, applies sliding windows,
    and returns train/val datasets plus held-out test windows (10%).

    70/20/10 split is done on the window index level.
    """
    sens_folder = cfg["dataset"]["sens_folder"]

    all_rows: List[np.ndarray] = []

    for sbj in sbj_list:
        csv_path = os.path.join(sens_folder, sbj + ".csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Sensor CSV not found: {csv_path}")

        df = pd.read_csv(csv_path, low_memory=False)
        df = _encode_labels_to_int(df, cfg)
        if "label_dict" in cfg:
            df = df.replace({"label": cfg["label_dict"]})
        df = df.fillna(0)

        all_rows.append(df.to_numpy())

    data = np.concatenate(all_rows, axis=0)

    # Normalize to [-1, 1] using global min/max
    feat_cols = slice(1, -1)  # [subject_id | sensors... | label]
    feat_min = data[:, feat_cols].min()
    feat_max = data[:, feat_cols].max()
    denom = (feat_max - feat_min) if (feat_max - feat_min) != 0 else 1.0
    data[:, feat_cols] = 2.0 * (data[:, feat_cols] - feat_min) / denom - 1.0

    print(f"[{freq_tag}] All data shape: {data.shape}")
    print(f"[{freq_tag}] Min/Max after normalization: {data[:, feat_cols].min():.2f}, {data[:, feat_cols].max():.2f}")

    # Sliding windows (same style as original)
    window_size = cfg["dataset"]["window_size"]
    window_overlap = cfg["dataset"]["window_overlap"]
    no_context_windows = cfg["dataset"].get("no_context_windows", 0)

    ids, windows, labels = apply_sliding_window(
        data,
        window_size=window_size,
        window_overlap=window_overlap,
        no_context_windows=no_context_windows,
    )

    # Remove subject id channel from windows
    windows = windows[:, :, 1:].astype(np.float32)  # (N, T, C)
    # --- filter invalid labels (e.g., -1 -> 255 when cast to uint8) ---
    labels = labels.astype(np.int64)
    n_classes = len(cfg.get('label_dict', {})) or (int(labels.max()) + 1)
    valid = (labels >= 0) & (labels < n_classes)
    if not valid.all():
        windows = windows[valid]
        labels = labels[valid]
    labels = labels.astype(np.uint8)

    n_windows = windows.shape[0]
    print(f"[{freq_tag}] #windows after sliding window: {n_windows}")

    # 70 / 20 / 10 split
    rng = np.random.RandomState(seed)
    indices = np.arange(n_windows)
    rng.shuffle(indices)

    n_train = int(0.7 * n_windows)
    n_val = int(0.2 * n_windows)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    x_train, y_train = windows[train_idx], labels[train_idx]
    x_val, y_val = windows[val_idx], labels[val_idx]
    x_test, y_test = windows[test_idx], labels[test_idx]

    print(f"[{freq_tag}] Split sizes (train / val / test): {len(train_idx)} / {len(val_idx)} / {len(test_idx)}")

    train_ds = InertialWindowDataset(x_train, y_train, model_name=cfg.get("name", "deepconvlstm"))
    val_ds = InertialWindowDataset(x_val, y_val, model_name=cfg.get("name", "deepconvlstm"))

    return train_ds, val_ds, (x_test, y_test)


# ---------------------------------------------------------------------------
# Training single-frequency expert with 70/20/10 split
# ---------------------------------------------------------------------------

def run_inertial_expert_702010(
    cfg: Dict,
    sbj_list: List[str],
    log_dir: str,
    freq_tag: str,
    seed: int,
    test_split_dir: str,
    device: torch.device,
    gamma_quant=None,
    quant_bits: int = 4,
    gamma_type: str = "s_shaped",
    apply_gamma: str = "global",
    resume: str = "",
    ckpt_freq: int = -1,
) -> str:
    """
    Train one DeepConvLSTM Gamma-Quant expert for a given frequency, using a global 70/20/10 split.

    Returns best checkpoint path.
    """
    os.makedirs(log_dir, exist_ok=True)
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # seed (like original main.py)
    fix_random_seed(cfg.get("init_rand_seed", seed), include_cuda=True)

    # Build datasets + held-out test
    train_ds, val_ds, (x_test, y_test) = build_single_freq_splits(cfg, sbj_list, freq_tag, seed)

    # Save test split for later multi-frequency cascade evaluation
    os.makedirs(test_split_dir, exist_ok=True)
    test_npz_path = os.path.join(test_split_dir, f"test_windows_{freq_tag}.npz")
    np.savez_compressed(test_npz_path, x=x_test, y=y_test)
    print(f"[{freq_tag}] Saved held-out test windows to: {test_npz_path}")

    # DataLoaders
    batch_size = cfg["loader"]["train_batch_size"]
    test_batch_size = cfg["loader"]["test_batch_size"]
    num_workers = cfg["loader"].get("num_workers", 0)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_reset_seed,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_reset_seed,
    )

    # Model definition
    channels = train_ds.channels
    n_classes = train_ds.classes
    window_size = train_ds.window_size
    net_cfg = cfg["model"]

    net = DeepConvLSTM(
        channels=channels,
        classes=n_classes,
        window_size=window_size,
        conv_kernels=net_cfg["conv_kernels"],
        conv_kernel_size=net_cfg["conv_kernel_size"],
        lstm_units=net_cfg["lstm_units"],
        lstm_layers=net_cfg["lstm_layers"],
        dropout=net_cfg["dropout"],
        gamma_quant=gamma_quant,
        quant_bits=quant_bits,
        apply_gamma=apply_gamma,
        gamma_type=gamma_type,
        conv_type=net_cfg.get("conv_type", "standard"),
        conv_rank=net_cfg.get("conv_rank", 2),
        conv_mlp_hidden_dim=net_cfg.get("conv_mlp_hidden_dim", 64),
    )

    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Number of learnable parameters for DeepConvLSTM ({freq_tag}): {n_params}")

    # weight init from config
    net.apply(lambda m: init_weights(m, cfg["train_cfg"].get("weight_init", "xavier_normal")))
    net.to(device)

    criterion = nn.CrossEntropyLoss()

    # class weights (like original)
    if cfg["train_cfg"].get("weighted_loss", False):
        class_weights = compute_class_weight(
            "balanced",
            classes=np.unique(train_ds.labels) + 1,
            y=train_ds.labels + 1,
        )
        class_weights = torch.tensor(class_weights).float().to(device)
        criterion.weight = class_weights
        print(f"[{freq_tag}] Class weights: {class_weights.cpu().numpy()}")

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=cfg["train_cfg"]["lr"],
        weight_decay=cfg["train_cfg"]["weight_decay"],
    )

    scheduler = None
    if cfg["train_cfg"].get("lr_step", 0) > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg["train_cfg"]["lr_step"],
            gamma=cfg["train_cfg"]["lr_decay"],
        )

    start_epoch = 1
    best_f1 = -1.0
    best_ckpt_path = ""

    # Resume if requested
    if resume:
        if os.path.isfile(resume):
            print(f"[{freq_tag}] Resuming from checkpoint {resume}")
            checkpoint = torch.load(resume, map_location=device)
            if "state_dict" in checkpoint:
                net.load_state_dict(checkpoint["state_dict"])
            else:
                net.load_state_dict(checkpoint)
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_f1 = checkpoint.get("best_f1", -1.0)
        else:
            print(f"[{freq_tag}] WARNING: resume checkpoint not found: {resume}")

    epochs = cfg["train_cfg"]["epochs"]

    # Training loop
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        # Train
        net.train()
        train_loss_sum = 0.0
        n_train_samples = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = net(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * x.size(0)
            n_train_samples += x.size(0)

        train_loss = train_loss_sum / max(1, n_train_samples)

        # Validate
        net.eval()
        val_loss_sum = 0.0
        n_val_samples = 0
        all_true: List[int] = []
        all_pred: List[int] = []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                logits = net(x)
                loss = criterion(logits, y)

                val_loss_sum += loss.item() * x.size(0)
                n_val_samples += x.size(0)

                preds = torch.argmax(logits, dim=1)
                all_true.extend(y.cpu().numpy().tolist())
                all_pred.extend(preds.cpu().numpy().tolist())

        val_loss = val_loss_sum / max(1, n_val_samples)
        all_true_arr = np.array(all_true, dtype=np.int64)
        all_pred_arr = np.array(all_pred, dtype=np.int64)

        _, _, _, f1_per_class = classification_scores(
            all_true_arr,
            all_pred_arr,
            no_classes=n_classes,
        )
        macro_f1 = np.nanmean(f1_per_class) * 100.0
        acc = (all_true_arr == all_pred_arr).mean() * 100.0

        if scheduler is not None:
            scheduler.step()

        dt = time.time() - t0
        print(
            f"[{freq_tag}] Epoch {epoch:03d}/{epochs:03d}  "
            f"train loss {train_loss:.4f}  val loss {val_loss:.4f}  "
            f"Acc {acc:6.2f}%  Macro-F1 {macro_f1:6.2f}%  "
            f"({dt:.1f}s)"
        )

        # Save best checkpoint
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            state = {
                "epoch": epoch,
                "state_dict": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_f1": best_f1,
                "config": cfg,  # store full config for later reconstruction
            }
            ckpt_name = f"best_{freq_tag}.pth.tar"
            save_checkpoint(state, False, ckpt_dir, ckpt_name)
            best_ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            print(f"[{freq_tag}] New best model (Macro-F1={best_f1:.2f}%), saved to {best_ckpt_path}")

    return best_ckpt_path


# ---------------------------------------------------------------------------
# Multi-frequency cascade evaluation (6 / 12 / 25 Hz)
# ---------------------------------------------------------------------------

def test_multifreq_cascade(
    cfg: Dict,
    ckpt_6hz: str,
    ckpt_12hz: str,
    ckpt_25hz: str,
    test_split_dir: str,
    device: torch.device,
    gamma_quant=None,
    quant_bits: int = 4,
    gamma_type: str = "s_shaped",
    apply_gamma: str = "global",
) -> None:
    """
    Evaluate the multi-frequency cascade on the held-out 10% windows.

    IMPORTANT:
      The 6/12/25 Hz test splits have different #windows because of different
      window sizes / overlaps. We align them **by label** instead:

      For each class c, we take min(count_6[c], count_12[c], count_25[c])
      windows from each frequency, pair them, and build triples
      (x_6, x_12, x_25, y=c).
    """
    # ------------------------------------------------------------------
    # 1) Load test splits
    # ------------------------------------------------------------------
    path_6 = os.path.join(test_split_dir, "test_windows_6hz.npz")
    path_12 = os.path.join(test_split_dir, "test_windows_12hz.npz")
    path_25 = os.path.join(test_split_dir, "test_windows_25hz.npz")

    if not (os.path.isfile(path_6) and os.path.isfile(path_12) and os.path.isfile(path_25)):
        raise FileNotFoundError(
            f"Expected test split files not found in {test_split_dir}. "
            f"Please run training for 6/12/25 Hz first."
        )

    data_6 = np.load(path_6)
    data_12 = np.load(path_12)
    data_25 = np.load(path_25)

    x_6, y_6 = data_6["x"], data_6["y"]
    x_12, y_12 = data_12["x"], data_12["y"]
    x_25, y_25 = data_25["x"], data_25["y"]

    # ------------------------------------------------------------------
    # 2) Align windows across frequencies by label
    # ------------------------------------------------------------------
    n_classes = len(FREQ_FOR_CLASS)
    x6_list: List[np.ndarray] = []
    x12_list: List[np.ndarray] = []
    x25_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    for c in range(n_classes):
        idx6 = np.where(y_6 == c)[0]
        idx12 = np.where(y_12 == c)[0]
        idx25 = np.where(y_25 == c)[0]
        m = min(len(idx6), len(idx12), len(idx25))
        if m == 0:
            continue
        x6_list.append(x_6[idx6[:m]])
        x12_list.append(x_12[idx12[:m]])
        x25_list.append(x_25[idx25[:m]])
        y_list.append(np.full(m, c, dtype=np.uint8))

    if not x6_list:
        raise RuntimeError("Could not align any windows across frequencies by label.")

    x_6_aligned = np.concatenate(x6_list, axis=0)
    x_12_aligned = np.concatenate(x12_list, axis=0)
    x_25_aligned = np.concatenate(x25_list, axis=0)
    y_aligned = np.concatenate(y_list, axis=0)

    print(
        f"[cascade] Aligned windows per freq: "
        f"6hz={x_6_aligned.shape[0]}, 12hz={x_12_aligned.shape[0]}, 25hz={x_25_aligned.shape[0]}"
    )

    test_ds = InertialMultiFreqDataset(
        x_25=x_25_aligned,
        x_12=x_12_aligned,
        x_6=x_6_aligned,
        y=y_aligned,
    )

    test_batch_size = cfg["loader"]["test_batch_size"]
    num_workers = cfg["loader"].get("num_workers", 0)

    test_loader = DataLoader(
        test_ds,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_reset_seed,
    )

    # ------------------------------------------------------------------
    # 3) Helper: build each expert from its own checkpoint CONFIG
    # ------------------------------------------------------------------
    def build_net_from_ckpt(ckpt_path: str, x_example: np.ndarray, tag: str) -> DeepConvLSTM:
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint for {tag} not found: {ckpt_path}")
        print(f"[cascade] Loading {tag} checkpoint from {ckpt_path}")

        state = torch.load(ckpt_path, map_location=device, weights_only=False)

        # Prefer the config stored in the checkpoint; fall back to global cfg if missing
        ckpt_cfg = state.get("config", cfg)
        model_cfg = ckpt_cfg["model"]

        labels_cfg = ckpt_cfg.get("labels", cfg.get("labels", None))
        if labels_cfg is not None:
            n_classes_local = len(labels_cfg)
        else:
            # fallback to size of FREQ_FOR_CLASS (should be same)
            n_classes_local = len(FREQ_FOR_CLASS)

        channels = x_example.shape[2]
        window_size = x_example.shape[1]

        net = DeepConvLSTM(
            channels=channels,
            classes=n_classes_local,
            window_size=window_size,
            conv_kernels=model_cfg["conv_kernels"],
            conv_kernel_size=model_cfg["conv_kernel_size"],
            lstm_units=model_cfg["lstm_units"],
            lstm_layers=model_cfg["lstm_layers"],
            dropout=model_cfg["dropout"],
            # Gamma-Quant settings (same as training)
            gamma_quant=gamma_quant,
            quant_bits=quant_bits,
            apply_gamma=apply_gamma,
            gamma_type=gamma_type,
            # continuous conv / low-rank options if present
            conv_type=model_cfg.get("conv_type", "standard"),
            conv_rank=model_cfg.get("conv_rank", 2),
            conv_mlp_hidden_dim=model_cfg.get("conv_mlp_hidden_dim", 64),
        )

        # Load weights
        if "state_dict" in state:
            net.load_state_dict(state["state_dict"])
        else:
            net.load_state_dict(state)

        net.to(device)
        net.eval()
        return net

    # Build 3 experts with their OWN configs (kernels, windows, etc.)
    net_6 = build_net_from_ckpt(ckpt_6hz, x_6_aligned, "6 Hz")
    net_12 = build_net_from_ckpt(ckpt_12hz, x_12_aligned, "12 Hz")
    net_25 = build_net_from_ckpt(ckpt_25hz, x_25_aligned, "25 Hz")

    # ------------------------------------------------------------------
    # 4) Wrap them in the multi-frequency oracle model
    # ------------------------------------------------------------------
    multi_net = MultiFreqDeepConvLSTM(
        model_6hz=net_6,
        model_12hz=net_12,
        model_25hz=net_25,
        freq_for_class=FREQ_FOR_CLASS,
        history_len=cfg.get("cascade_history_len", 5),
        stability_threshold=cfg.get("cascade_stability_threshold", 0.8),
        min_confidence=cfg.get("cascade_min_confidence", 0.6),
    )
    multi_net.to(device)
    multi_net.eval()
    multi_net.reset_history()

    # ------------------------------------------------------------------
    # 5) Evaluation loop
    # ------------------------------------------------------------------
    all_true: List[int] = []
    all_pred: List[int] = []
    all_freq: List[str] = []

    with torch.no_grad():
        for x25, x12, x6, y in test_loader:
            x25 = x25.to(device, non_blocking=True)
            x12 = x12.to(device, non_blocking=True)
            x6 = x6.to(device, non_blocking=True)

            logits, chosen_freqs = multi_net.step(
                x_25hz=x25,
                x_12hz=x12,
                x_6hz=x6,
            )

            preds = torch.argmax(logits, dim=1)

            all_true.extend(y.numpy().tolist())
            all_pred.extend(preds.cpu().numpy().tolist())
            all_freq.extend(chosen_freqs)

    y_true = np.array(all_true, dtype=np.int64)
    y_pred = np.array(all_pred, dtype=np.int64)

    # We know we have 19 classes; FREQ_FOR_CLASS also has 19 entries
    _, _, _, f1_per_class = classification_scores(y_true, y_pred, no_classes=len(FREQ_FOR_CLASS))
    macro_f1 = np.nanmean(f1_per_class) * 100.0
    acc = (y_true == y_pred).mean() * 100.0

    print("----------------------------------------------------------------")
    print("Multi-frequency cascade evaluation on held-out 10% windows")
    print(f"Accuracy:  {acc:6.2f}%")
    print(f"Macro F1:  {macro_f1:6.2f}%")
    print("Per-class F1 (aligned with FREQ_FOR_CLASS):")
    for cls_id, f1_c in enumerate(f1_per_class):
        print(f"  Class {cls_id:02d}: F1 = {f1_c * 100.0:6.2f}%   (preferred freq: {FREQ_FOR_CLASS[cls_id]})")

    unique, counts = np.unique(np.array(all_freq), return_counts=True)
    print("\nFrequency usage in cascade (on test set):")
    for f, c in zip(unique, counts):
        print(f"  {f}: {c} windows ({100.0 * c / len(all_freq):5.1f}%)")
    print("----------------------------------------------------------------")
