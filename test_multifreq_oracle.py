import argparse
import inspect
import re
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from models.multi_freq_deepconvlstm import build_multifreq_from_single_cfg


# ============================================================
# Preferred frequency per class (NON-NULL only)
# repo expects length = (num_classes - 1) because has_null=True
# ============================================================

# Classes 1..18 (18 entries). Class 0 (null) uses fallback->25hz.
FREQ_FOR_CLASS_NON_NULL_STR = [
    "25hz",  # 1 bench-dips
    "25hz",  # 2 burpees
    "6hz",   # 3 jogging
    "25hz",  # 4 jogging (butt-kicks)
    "25hz",  # 5 jogging (rotating arms)
    "25hz",  # 6 jogging (sidesteps)
    "25hz",  # 7 jogging (skipping)
    "12hz",  # 8 lunges
    "12hz",  # 9 lunges (complex)
    "25hz",  # 10 push-ups
    "25hz",  # 11 push-ups (complex)
    "25hz",  # 12 sit-ups
    "12hz",  # 13 sit-ups (complex)
    "6hz",   # 14 stretching (hamstrings)
    "6hz",   # 15 stretching (lumbar rotation)
    "12hz",  # 16 stretching (lunging)
    "12hz",  # 17 stretching (shoulders)
    "12hz",  # 18 stretching (triceps)
]

# Numeric mapping required by MultiFreqDeepConvLSTM(torch.long)
# Convention used in your repo: 0=6hz, 1=12hz, 2=25hz
FREQ_TO_IDX = {"6hz": 0, "12hz": 1, "25hz": 2}
FREQ_FOR_CLASS_NON_NULL_IDX = [FREQ_TO_IDX[s] for s in FREQ_FOR_CLASS_NON_NULL_STR]


def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_network_key(cfg: dict) -> dict:
    """
    build_multifreq_from_single_cfg expects cfg["network"].
    Your YAML has cfg["model"].
    """
    if "network" in cfg:
        return cfg
    if "model" in cfg and isinstance(cfg["model"], dict):
        cfg["network"] = dict(cfg["model"])
        return cfg
    raise KeyError(
        "Config missing 'network' and 'model'. "
        f"Top-level keys: {list(cfg.keys())}"
    )


def ensure_dataset_keys(cfg: dict, n_classes_fallback: int) -> dict:
    """
    build_multifreq_from_single_cfg expects cfg['dataset']['n_classes'].
    Your YAML has cfg['dataset']['num_classes'].
    """
    if "dataset" not in cfg or not isinstance(cfg["dataset"], dict):
        raise KeyError(
            "Config missing top-level 'dataset' dict. "
            f"Top-level keys: {list(cfg.keys())}"
        )

    ds = cfg["dataset"]

    if "n_classes" not in ds:
        if "num_classes" in ds:
            ds["n_classes"] = int(ds["num_classes"])
        else:
            ds["n_classes"] = int(n_classes_fallback)

    cfg["dataset"] = ds
    return cfg


def load_npz_windows(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    keys = list(data.keys())
    if "x" in keys and "y" in keys:
        return data["x"], data["y"]
    if "windows" in keys and "labels" in keys:
        return data["windows"], data["labels"]
    if len(keys) >= 2:
        return data[keys[0]], data[keys[1]]
    raise RuntimeError(f"{npz_path} has keys={keys}, expected at least 2 arrays.")


def safe_torch_load_cpu(path: str):
    # PyTorch 2.6+: default weights_only=True can break older ckpts containing numpy scalars
    return torch.load(path, map_location="cpu", weights_only=False)


def infer_lstm_layers_from_state_dict(state_dict: dict) -> int:
    """
    Infer LSTM num_layers from keys present in state_dict:
    - num_layers=1 => has lstm.weight_ih_l0 only
    - num_layers=2 => also has lstm.weight_ih_l1
    """
    pat = re.compile(r"^lstm\.weight_ih_l(\d+)$")
    max_idx = -1
    for k in state_dict.keys():
        m = pat.match(k)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return (max_idx + 1) if max_idx >= 0 else 1


def ckpt_has_gamma(state_dict: dict) -> bool:
    for k in state_dict.keys():
        if k.startswith("gamma_function.") or ".gamma_function." in k:
            return True
    return False


def forward_logits(model: torch.nn.Module, x: torch.Tensor):
    try:
        return model(x)
    except Exception:
        if x.ndim == 3:
            return model(x.unsqueeze(1))
        raise


def confusion_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1

    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = (2 * tp + fp + fn).astype(np.float64)
    f1 = np.where(denom > 0, (2 * tp) / denom, 0.0)
    macro_f1 = float(np.mean(f1))
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    return acc, macro_f1, f1


def build_container(cfg: dict, conv_type, conv_rank, conv_mlp_hidden_dim,
                    gamma_quant=None, quant_bits=None, gamma_type=None, apply_gamma=None):
    """
    Build MultiFreq container with numeric freq_for_class for NON-NULL classes only.
    """
    kwargs = dict(
        cfg=cfg,
        freq_for_class=FREQ_FOR_CLASS_NON_NULL_IDX,  # IMPORTANT: length 18
        conv_type=conv_type,
        conv_rank=conv_rank,
    )

    sig = inspect.signature(build_multifreq_from_single_cfg)

    if "conv_mlp_hidden_dim" in sig.parameters and conv_mlp_hidden_dim is not None:
        kwargs["conv_mlp_hidden_dim"] = conv_mlp_hidden_dim

    # pass gamma args only if builder supports them
    if "gamma_quant" in sig.parameters:
        kwargs["gamma_quant"] = gamma_quant
    if "quant_bits" in sig.parameters:
        kwargs["quant_bits"] = quant_bits
    if "gamma_type" in sig.parameters:
        kwargs["gamma_type"] = gamma_type
    if "apply_gamma" in sig.parameters:
        kwargs["apply_gamma"] = apply_gamma

    return build_multifreq_from_single_cfg(**kwargs)


def preferred_freq_for_class_id(class_id: int) -> str:
    """
    class_id is 0..18.
    - 0 is null => fallback 25hz
    - 1..18 use mapping (index class_id-1)
    """
    if class_id <= 0:
        return "25hz"
    idx = class_id - 1
    if 0 <= idx < len(FREQ_FOR_CLASS_NON_NULL_STR):
        return FREQ_FOR_CLASS_NON_NULL_STR[idx]
    return "25hz"


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--config-25hz", required=True, type=str)
    ap.add_argument("--config-12hz", required=True, type=str)
    ap.add_argument("--config-6hz", required=True, type=str)

    ap.add_argument("--test-split-dir", default="./data/wear/multifreq_test", type=str)

    ap.add_argument("--ckpt-25hz", required=True, type=str)
    ap.add_argument("--ckpt-12hz", required=True, type=str)
    ap.add_argument("--ckpt-6hz", required=True, type=str)

    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--batch-size", default=512, type=int)

    ap.add_argument("--conv-type", default="continuous", choices=["continuous", "standard"])
    ap.add_argument("--conv-rank", default=8, type=int)
    ap.add_argument("--conv-mlp-hidden-dim", default=32, type=int)

    ap.add_argument("--history-len", default=5, type=int)
    ap.add_argument("--conf-thresh", default=0.60, type=float)

    # Support BOTH dash and underscore variants
    ap.add_argument("--gamma-quant", "--gamma_quant", dest="gamma_quant", default=None, type=str)
    ap.add_argument("--quant-bits", "--quant_bits", dest="quant_bits", default=4, type=int)
    ap.add_argument("--gamma-type", "--gamma_type", dest="gamma_type", default="s_shaped", type=str)
    ap.add_argument("--apply-gamma", "--apply_gamma", dest="apply_gamma", default="global", type=str)

    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")

    # WEAR: 19 classes (0..18) when has_null=True
    n_classes = 19

    cfg25 = ensure_dataset_keys(ensure_network_key(load_cfg(args.config_25hz)), n_classes)
    cfg12 = ensure_dataset_keys(ensure_network_key(load_cfg(args.config_12hz)), n_classes)
    cfg6  = ensure_dataset_keys(ensure_network_key(load_cfg(args.config_6hz)),  n_classes)

    # ---- Load checkpoints FIRST (CPU) to infer the exact architecture ----
    ckpt25 = safe_torch_load_cpu(args.ckpt_25hz)
    ckpt12 = safe_torch_load_cpu(args.ckpt_12hz)
    ckpt6  = safe_torch_load_cpu(args.ckpt_6hz)

    sd25, sd12, sd6 = ckpt25["state_dict"], ckpt12["state_dict"], ckpt6["state_dict"]

    l25 = infer_lstm_layers_from_state_dict(sd25)
    l12 = infer_lstm_layers_from_state_dict(sd12)
    l6  = infer_lstm_layers_from_state_dict(sd6)

    g25 = ckpt_has_gamma(sd25)
    g12 = ckpt_has_gamma(sd12)
    g6  = ckpt_has_gamma(sd6)
    need_gamma = g25 or g12 or g6

    print(f"[ckpt] inferred lstm_layers: 25hz={l25}, 12hz={l12}, 6hz={l6}")
    print(f"[ckpt] gamma_function in ckpt: 25hz={g25}, 12hz={g12}, 6hz={g6}")

    # Force cfgs to match checkpoints (this fixes missing lstm.*_l1 keys)
    cfg25["network"]["lstm_layers"] = int(l25)
    cfg12["network"]["lstm_layers"] = int(l12)
    cfg6["network"]["lstm_layers"]  = int(l6)

    # If checkpoint includes gamma params but user did not pass --gamma-quant,
    # auto-enable a default to make model instantiate gamma_function.
    gamma_quant = args.gamma_quant
    if need_gamma and gamma_quant is None:
        gamma_quant = "quantize_and_gamma"
        print(f"[ckpt] --gamma-quant not provided; enabling gamma_quant='{gamma_quant}' to match checkpoints.")

    # Build three containers (each with its own cfg), then combine experts
    multi25 = build_container(
        cfg25, args.conv_type, args.conv_rank, args.conv_mlp_hidden_dim,
        gamma_quant=gamma_quant, quant_bits=args.quant_bits,
        gamma_type=args.gamma_type, apply_gamma=args.apply_gamma
    )
    multi12 = build_container(
        cfg12, args.conv_type, args.conv_rank, args.conv_mlp_hidden_dim,
        gamma_quant=gamma_quant, quant_bits=args.quant_bits,
        gamma_type=args.gamma_type, apply_gamma=args.apply_gamma
    )
    multi6 = build_container(
        cfg6, args.conv_type, args.conv_rank, args.conv_mlp_hidden_dim,
        gamma_quant=gamma_quant, quant_bits=args.quant_bits,
        gamma_type=args.gamma_type, apply_gamma=args.apply_gamma
    )

    # Use the 25hz container as the “main” wrapper; swap in 12/6 experts
    multi_model = multi25
    multi_model.model_12 = multi12.model_12
    multi_model.model_6  = multi6.model_6

    multi_model.to(device)
    multi_model.eval()
    if hasattr(multi_model, "reset_history"):
        multi_model.reset_history()

    # Load state dicts STRICTLY now (should match after fixes)
    multi_model.model_25.load_state_dict(sd25, strict=True)
    multi_model.model_12.load_state_dict(sd12, strict=True)
    multi_model.model_6.load_state_dict(sd6, strict=True)

    test_dir = Path(args.test_split_dir)
    x25, y25 = load_npz_windows(str(test_dir / "test_windows_25hz.npz"))
    x12, y12 = load_npz_windows(str(test_dir / "test_windows_12hz.npz"))
    x6,  y6  = load_npz_windows(str(test_dir / "test_windows_6hz.npz"))

    n = min(len(x6), len(x12), len(x25))
    x6,  y6  = x6[:n],  y6[:n]
    x12, y12 = x12[:n], y12[:n]
    x25, y25 = x25[:n], y25[:n]
    y = y25.astype(np.int64)

    print(f"[oracle] Aligned windows per freq: 6hz={len(x6)}, 12hz={len(x12)}, 25hz={len(x25)}")

    x6_t  = torch.from_numpy(x6).float()
    x12_t = torch.from_numpy(x12).float()
    x25_t = torch.from_numpy(x25).float()

    logits6_all, logits12_all, logits25_all = [], [], []
    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            xb6  = x6_t[start:end].to(device, non_blocking=True)
            xb12 = x12_t[start:end].to(device, non_blocking=True)
            xb25 = x25_t[start:end].to(device, non_blocking=True)

            logits6_all.append(forward_logits(multi_model.model_6,  xb6).detach().cpu())
            logits12_all.append(forward_logits(multi_model.model_12, xb12).detach().cpu())
            logits25_all.append(forward_logits(multi_model.model_25, xb25).detach().cpu())

    logits6  = torch.cat(logits6_all,  dim=0)[:n]
    logits12 = torch.cat(logits12_all, dim=0)[:n]
    logits25 = torch.cat(logits25_all, dim=0)[:n]

    pred6  = F.softmax(logits6,  dim=1).numpy().argmax(axis=1)
    pred12 = F.softmax(logits12, dim=1).numpy().argmax(axis=1)
    pred25 = F.softmax(logits25, dim=1).numpy().argmax(axis=1)

    # Oracle-by-history routing
    hist = deque(maxlen=max(1, args.history_len))
    y_pred = np.zeros((n,), dtype=np.int64)
    usage = {"6hz": 0, "12hz": 0, "25hz": 0}

    for i in range(n):
        chosen = "25hz"  # fallback when uncertain

        if len(hist) > 0:
            cnt = Counter(hist)
            maj_cls, maj_votes = cnt.most_common(1)[0]
            maj_ratio = maj_votes / len(hist)

            if maj_ratio >= args.conf_thresh:
                chosen = preferred_freq_for_class_id(int(maj_cls))
            else:
                chosen = "25hz"

        if chosen == "6hz":
            yhat = int(pred6[i]); usage["6hz"] += 1
        elif chosen == "12hz":
            yhat = int(pred12[i]); usage["12hz"] += 1
        else:
            yhat = int(pred25[i]); usage["25hz"] += 1

        y_pred[i] = yhat
        hist.append(yhat)

    acc, macro_f1, per_class_f1 = confusion_f1(y, y_pred, n_classes)

    print("\nMulti-frequency ORACLE-by-history evaluation on held-out windows")
    print(f"Accuracy:   {acc*100:6.2f}%")
    print(f"Macro F1:   {macro_f1*100:6.2f}%")

    print("Per-class F1 (with preferred freq):")
    print(f"  Class 00 (null): F1 = {per_class_f1[0]*100:6.2f}%   (preferred: fallback->25hz)")
    for c in range(1, n_classes):
        pref = preferred_freq_for_class_id(c)
        print(f"  Class {c:02d}: F1 = {per_class_f1[c]*100:6.2f}%   (preferred: {pref})")

    total = sum(usage.values())
    print("\nFrequency usage (routing decisions):")
    for k in ["12hz", "25hz", "6hz"]:
        v = usage[k]
        pct = (100.0 * v / total) if total else 0.0
        print(f"  {k}: {v} windows ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
