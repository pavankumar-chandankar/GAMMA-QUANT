import argparse
import os
import sys
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from models.DeepConvLSTM import DeepConvLSTM

# ---------------------------------------------------------------------------
# Configuration for label names (from your label_dict) and "low-movement" set
# ---------------------------------------------------------------------------

LABEL_DICT = {
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

# Inverse mapping: id -> name
ID2LABEL = {v: k for k, v in LABEL_DICT.items()}

# You can change this set to decide which activities are "low-movement" and
# should preferably use the low-frequency/cheap model.
LOW_MOVEMENT_CLASSES = {
    LABEL_DICT['null'],
    LABEL_DICT['stretching (triceps)'],
    LABEL_DICT['stretching (lunging)'],
    LABEL_DICT['stretching (shoulders)'],
    LABEL_DICT['stretching (hamstrings)'],
    LABEL_DICT['stretching (lumbar rotation)'],
}

# Everything else is considered high-movement (for class-based bias)
HIGH_MOVEMENT_CLASSES = set(ID2LABEL.keys()) - LOW_MOVEMENT_CLASSES


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def load_yaml_config(path: str) -> dict:
    """Load YAML configuration file (e.g. configs/deepconvlstm/wear_loso.yaml)."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_model_from_config(cfg: dict,
                            quantize: bool = False,
                            quant_bits: int = 8,
                            apply_gamma: bool = False,
                            gamma_type: str = "y-quant") -> DeepConvLSTM:
    """
    Build a DeepConvLSTM model from your WEAR config dict.
    """
    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]

    input_dim = dataset_cfg["input_dim"]
    num_classes = dataset_cfg["num_classes"]

    conv_kernels = model_cfg.get("conv_kernels", 64)
    conv_kernel_size = model_cfg.get("conv_kernel_size", 9)
    lstm_units = model_cfg.get("lstm_units", 128)
    lstm_layers = model_cfg.get("lstm_layers", 1)
    dropout = model_cfg.get("dropout", 0.5)

    model = DeepConvLSTM(
        input_channels=input_dim,
        num_classes=num_classes,
        conv_kernels=conv_kernels,
        conv_kernel_size=conv_kernel_size,
        lstm_units=lstm_units,
        lstm_layers=lstm_layers,
        dropout=dropout,
        quantize=quantize,
        quant_bits=quant_bits,
        apply_gamma=apply_gamma,
        gamma_type=gamma_type,
    )
    return model


def load_checkpoint_into_model(model: torch.nn.Module,
                               ckpt_path: str,
                               device: torch.device) -> None:
    """
    Load 'state_dict' from a training checkpoint into the model.
    Assumes the checkpoint is a dict with key 'state_dict'.
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    # Strip "module." prefix if present (from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)


def load_stream(path: str, input_dim: int) -> np.ndarray:
    """
    Load a continuous inertial data stream from .npy or .csv.
    Expected shape: [T, input_dim].
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        data = np.load(path)
    elif ext == ".csv":
        data = np.loadtxt(path, delimiter=",")
    else:
        raise ValueError(f"Unsupported stream file extension: {ext}")

    if data.ndim != 2 or data.shape[1] < input_dim:
        raise ValueError(
            f"Expected stream shape [T, {input_dim}] (or more channels), got {data.shape}"
        )

    # If there are more columns than input_dim, keep only the first input_dim.
    if data.shape[1] > input_dim:
        data = data[:, :input_dim]

    return data.astype(np.float32)


def compute_motion_intensity(window: np.ndarray) -> float:
    """
    Simple motion intensity measure for a window.
    window: [W, C], where C=input_dim.
    We compute the mean L2 norm across time.
    """
    # L2 per time step
    norms = np.linalg.norm(window, axis=1)
    return float(np.mean(norms))


def prepare_window_tensor(window: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert window [W, C] to tensor [1, C, W, 1] as DeepConvLSTM expects.
    """
    # window: [W, C]
    x = torch.from_numpy(window.astype(np.float32))  # [W, C]
    x = x.unsqueeze(0)          # [1, W, C]
    x = x.permute(0, 2, 1)      # [1, C, W]
    x = x.unsqueeze(-1)         # [1, C, W, 1]
    return x.to(device)


def choose_model_gating(motion_intensity: float,
                        last_pred: Optional[int],
                        threshold: float) -> str:
    """
    Decide whether to use 'low' or 'high' model based on:
      - motion_intensity: float
      - last_pred: last predicted class id (or None)
      - threshold: scalar threshold for low vs high motion

    Logic:
      1. If last_pred is a low-movement class and motion is not too large,
         bias towards 'low' model.
      2. Otherwise, use motion threshold.
    """
    # Bias towards low model if we are in a low-movement class
    if last_pred is not None and last_pred in LOW_MOVEMENT_CLASSES:
        # Slightly relaxed threshold when already in low-movement class
        if motion_intensity < 1.2 * threshold:
            return "low"

    # Default gating purely on motion
    if motion_intensity < threshold:
        return "low"
    else:
        return "high"


# ---------------------------------------------------------------------------
# Main gated inference routine
# ---------------------------------------------------------------------------

def run_gated_inference(config_path: str,
                        ckpt_low: str,
                        ckpt_high: str,
                        stream_path: str,
                        gating_threshold: float,
                        window_size_override: Optional[int] = None,
                        window_step: Optional[int] = None,
                        quantize_low: bool = True,
                        quantize_high: bool = True,
                        quant_bits_low: int = 4,
                        quant_bits_high: int = 4,
                        apply_gamma: bool = True,
                        gamma_type: str = "y-quant",
                        device_str: str = "cuda") -> None:
    """
    Run inference on a continuous stream with gating between low and high models.

    Args:
        config_path: path to wear_loso.yaml
        ckpt_low: checkpoint path for low-frequency/cheap model
        ckpt_high: checkpoint path for high-frequency/expensive model
        stream_path: path to .npy / .csv inertial data [T, input_dim]
        gating_threshold: motion intensity threshold
        window_size_override: if not None, use this window size instead of config
        window_step: stride between windows (default: equal to window_size)
        quantize_low/high, quant_bits_*, apply_gamma, gamma_type: control quantization
        device_str: "cuda" or "cpu"
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load config and read dataset + model parameters
    cfg = load_yaml_config(config_path)
    dataset_cfg = cfg["dataset"]

    input_dim = int(dataset_cfg["input_dim"])
    cfg_window_size = int(dataset_cfg["window_size"])
    window_size = int(window_size_override) if window_size_override is not None else cfg_window_size

    # Default: non-overlapping windows
    if window_step is None:
        window_step = window_size

    print(f"Input dim: {input_dim}")
    print(f"Window size: {window_size}, step: {window_step}")

    # Build and load both models
    print("Building low-cost model...")
    model_low = build_model_from_config(
        cfg,
        quantize=quantize_low,
        quant_bits=quant_bits_low,
        apply_gamma=apply_gamma,
        gamma_type=gamma_type,
    ).to(device)
    load_checkpoint_into_model(model_low, ckpt_low, device)
    model_low.eval()

    print("Building high-cost model...")
    model_high = build_model_from_config(
        cfg,
        quantize=quantize_high,
        quant_bits=quant_bits_high,
        apply_gamma=apply_gamma,
        gamma_type=gamma_type,
    ).to(device)
    load_checkpoint_into_model(model_high, ckpt_high, device)
    model_high.eval()

    # Load the continuous stream
    stream = load_stream(stream_path, input_dim=input_dim)
    T = stream.shape[0]
    print(f"Loaded stream of length T={T} samples.")

    # Sliding-window gating inference
    all_preds: List[int] = []
    all_model_choices: List[str] = []
    last_pred: Optional[int] = None

    with torch.no_grad():
        t = 0
        window_index = 0
        while t + window_size <= T:
            window = stream[t:t + window_size, :]  # [W, C]
            motion_intensity = compute_motion_intensity(window)

            model_choice = choose_model_gating(
                motion_intensity=motion_intensity,
                last_pred=last_pred,
                threshold=gating_threshold,
            )

            if model_choice == "low":
                model = model_low
            else:
                model = model_high

            x = prepare_window_tensor(window, device)  # [1, C, W, 1]
            logits = model(x)                          # [1, num_classes]
            probs = F.softmax(logits, dim=-1)
            pred = int(torch.argmax(probs, dim=-1).item())

            all_preds.append(pred)
            all_model_choices.append(model_choice)
            last_pred = pred

            # Print some debug info
            print(
                f"[window {window_index:04d}] t={t:06d} "
                f"motion={motion_intensity:.4f} "
                f"model={model_choice:4s} "
                f"pred={pred:2d} ({ID2LABEL.get(pred, 'UNKNOWN')})"
            )

            window_index += 1
            t += window_step

    # Simple usage statistics (how often each model was used)
    total_windows = len(all_model_choices)
    if total_windows == 0:
        print("No windows processed.")
        return

    low_count = sum(1 for m in all_model_choices if m == "low")
    high_count = sum(1 for m in all_model_choices if m == "high")

    print("\n=== Gating summary ===")
    print(f"Total windows: {total_windows}")
    print(f"Low model used : {low_count} windows ({100.0 * low_count / total_windows:.2f}%)")
    print(f"High model used: {high_count} windows ({100.0 * high_count / total_windows:.2f}%)")
    print("Note: You can log power / MACs externally per model to estimate savings.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference-time gating between low- and high-cost DeepConvLSTM models."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config (e.g. ./configs/deepconvlstm/wear_loso.yaml)",
    )
    parser.add_argument(
        "--ckpt_low",
        type=str,
        required=True,
        help="Checkpoint path for low-cost model (e.g. logs/.../low_model_best.pth.tar)",
    )
    parser.add_argument(
        "--ckpt_high",
        type=str,
        required=True,
        help="Checkpoint path for high-cost model (e.g. logs/.../high_model_best.pth.tar)",
    )
    parser.add_argument(
        "--stream_path",
        type=str,
        required=True,
        help="Path to continuous inertial stream (.npy or .csv) with shape [T, input_dim].",
    )
    parser.add_argument(
        "--gating_threshold",
        type=float,
        default=0.5,
        help="Motion intensity threshold separating low vs high model.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=None,
        help="Override window_size (otherwise taken from config).",
    )
    parser.add_argument(
        "--window_step",
        type=int,
        default=None,
        help="Sliding step between windows (default: equal to window_size).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device string, e.g. 'cuda' or 'cpu'.",
    )
    # Quantization options (if your checkpoints are quantized models)
    parser.add_argument(
        "--quantize_low",
        action="store_true",
        help="Instantiate low model with quantization layers enabled.",
    )
    parser.add_argument(
        "--quantize_high",
        action="store_true",
        help="Instantiate high model with quantization layers enabled.",
    )
    parser.add_argument(
        "--quant_bits_low",
        type=int,
        default=2,
        help="Bit-width for low model (if quantize_low is set).",
    )
    parser.add_argument(
        "--quant_bits_high",
        type=int,
        default=2,
        help="Bit-width for high model (if quantize_high is set).",
    )
    parser.add_argument(
        "--gamma_type",
        type=str,
        default="y-quant",
        help="Gamma type (matches training, e.g. 'y-quant').",
    )
    parser.add_argument(
        "--no_gamma",
        action="store_true",
        help="Disable gamma application (apply_gamma=False).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    apply_gamma = not args.no_gamma

    run_gated_inference(
        config_path=args.config,
        ckpt_low=args.ckpt_low,
        ckpt_high=args.ckpt_high,
        stream_path=args.stream_path,
        gating_threshold=args.gating_threshold,
        window_size_override=args.window_size,
        window_step=args.window_step,
        quantize_low=args.quantize_low,
        quantize_high=args.quantize_high,
        quant_bits_low=args.quant_bits_low,
        quant_bits_high=args.quant_bits_high,
        apply_gamma=apply_gamma,
        gamma_type=args.gamma_type,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
