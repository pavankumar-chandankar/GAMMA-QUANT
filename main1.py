import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Set

import torch
import yaml

from models.train1 import run_inertial_expert_702010, test_multifreq_cascade


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def prepare_labels_and_subjects(cfg: Dict) -> List[str]:
    """
    Build label_dict and subject list from all annotation JSON files
    listed in cfg["anno_json"].

    - label_dict is constructed from the first JSON's label_dict:
      if it's a dict, we sort by id; if it's a list, we use it as-is.
    - subject list is the union of all "database" keys across all JSONs.
    """
    anno_list = cfg.get("anno_json", [])
    if not anno_list:
        raise ValueError("Config file must contain an 'anno_json' entry with at least one annotation file.")

    all_subjects: Set[str] = set()
    labels: List[str] = []

    first_label_dict_done = False

    for anno_path in anno_list:
        if not os.path.isfile(anno_path):
            raise FileNotFoundError(f"Annotation JSON not found: {anno_path}")

        with open(anno_path, "r") as f:
            anno_file = json.load(f)

        db = anno_file["database"]
        all_subjects.update(db.keys())

        if not first_label_dict_done:
            raw_ld = anno_file.get("label_dict", None)
            if raw_ld is None:
                raise KeyError(f"'label_dict' not found in annotation file {anno_path}")

            if isinstance(raw_ld, dict):
                # assume mapping label_name -> id, sort by id
                labels = [name for name, _ in sorted(raw_ld.items(), key=lambda kv: kv[1])]
            else:
                # assume list of label names
                labels = list(raw_ld)

            if cfg.get("has_null", False):
                labels = ["null"] + labels

            label_dict = {lab: idx for idx, lab in enumerate(labels)}
            cfg["labels"] = labels
            cfg["label_dict"] = label_dict
            first_label_dict_done = True

    if not first_label_dict_done:
        raise RuntimeError("Could not build label_dict from any annotation file.")

    sbj_list = sorted(list(all_subjects))
    return sbj_list


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gamma-Quant Inertial – multi-frequency experts + oracle cascade (main1)."
    )

    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--mode", type=str, default="train_expert", choices=["train_expert", "test_multifreq"])

    # For training individual experts
    parser.add_argument("--freq-tag", type=str, default=None, help="Frequency tag: e.g. 6hz, 12hz, 25hz.")
    parser.add_argument(
        "--test-split-dir",
        type=str,
        default="./data/wear/multifreq_test",
        help="Directory where test windows (10% split) are stored.",
    )
    parser.add_argument("--ckpt-freq", type=int, default=-1, help="(Kept for compatibility, currently unused.)")
    parser.add_argument("--resume", type=str, default="", help="Optional path to resume checkpoint.")

    # Gamma-Quant arguments (underscore style, like your original main.py)
    parser.add_argument(
        "--gamma_quant",
        default=None,
        type=str,
        help="Gamma-Quant mode: e.g. 'quantize', 'quantize_and_calibrate', 'quantize_and_gamma', or None.",
    )
    parser.add_argument(
        "--quant_bits",
        default=4,
        type=int,
        help="Number of bits for quantization (default: 4).",
    )
    parser.add_argument(
        "--gamma_type",
        default="s_shaped",
        type=str,
        help="Type of gamma function to use (e.g. 's_shaped', 'id').",
    )
    parser.add_argument(
        "--apply_gamma",
        default="global",
        type=str,
        help="Apply gamma function 'global' or 'local' (default: global).",
    )

    # For multi-frequency cascade evaluation
    parser.add_argument("--ckpt-6hz", type=str, default="", help="Checkpoint for 6 Hz expert.")
    parser.add_argument("--ckpt-12hz", type=str, default="", help="Checkpoint for 12 Hz expert.")
    parser.add_argument("--ckpt-25hz", type=str, default="", help="Checkpoint for 25 Hz expert.")

    args = parser.parse_args()

    cfg = load_config(args.config)
    sbj_list = prepare_labels_and_subjects(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cfg_name = cfg.get("name", "deepconvlstm")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.mode == "train_expert":
        if args.freq_tag is None:
            raise ValueError("--freq-tag must be provided in train_expert mode (e.g. 6hz, 12hz, 25hz).")

        log_dir = os.path.join("logs", cfg_name, f"{args.freq_tag}_702010", ts)
        print(f"Logging to: {log_dir}")

        best_ckpt = run_inertial_expert_702010(
            cfg=cfg,
            sbj_list=sbj_list,
            log_dir=log_dir,
            freq_tag=args.freq_tag,
            seed=args.seed,
            test_split_dir=args.test_split_dir,
            device=device,
            gamma_quant=args.gamma_quant,
            quant_bits=args.quant_bits,
            gamma_type=args.gamma_type,
            apply_gamma=args.apply_gamma,
            resume=args.resume,
            ckpt_freq=args.ckpt_freq,
        )

        print("\n------------------------------------------------------------")
        print(f"Finished training expert for {args.freq_tag}.")
        print(f"Best checkpoint saved at: {best_ckpt}")
        print("Now train the other frequencies (6hz, 12hz) with the same command,")
        print("then run test_multifreq mode to evaluate the cascade.")
        print("------------------------------------------------------------\n")

    elif args.mode == "test_multifreq":
        if not (args.ckpt_6hz and args.ckpt_12hz and args.ckpt_25hz):
            raise ValueError("In test_multifreq mode you must provide --ckpt-6hz, --ckpt-12hz, and --ckpt-25hz.")

        print("Running multi-frequency cascade evaluation on held-out 10% windows...")
        test_multifreq_cascade(
            cfg=cfg,
            ckpt_6hz=args.ckpt_6hz,
            ckpt_12hz=args.ckpt_12hz,
            ckpt_25hz=args.ckpt_25hz,
            test_split_dir=args.test_split_dir,
            device=device,
            gamma_quant=args.gamma_quant,
            quant_bits=args.quant_bits,
            gamma_type=args.gamma_type,
            apply_gamma=args.apply_gamma,
        )


if __name__ == "__main__":
    main()
