import argparse
import glob
import os
from datetime import datetime

import torch
import yaml

from models.train2 import (
    run_inertial_expert_702010_continuous,
    test_multifreq_cascade_continuous,
)

# Your repo has fix_random_seed (you hit this earlier)
from utils.torch_utils import fix_random_seed
from utils.os_utils import mkdir_if_missing


def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def list_subjects_from_folder(sens_folder: str):
    csvs = sorted(glob.glob(os.path.join(sens_folder, "*.csv")))
    sbjs = [os.path.splitext(os.path.basename(p))[0] for p in csvs]
    if len(sbjs) == 0:
        raise RuntimeError(f"No subject CSVs found in: {sens_folder}")
    return sbjs


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--config", required=True, type=str)
    p.add_argument("--seed", default=1, type=int)

    p.add_argument("--mode", default="train_expert",
                   choices=["train_expert", "test_multifreq"])

    p.add_argument("--freq-tag", default=None, type=str,
                   help="One of: 6hz / 12hz / 25hz (only for train_expert)")

    p.add_argument("--test-split-dir", default="./data/wear/multifreq_test", type=str)

    p.add_argument("--ckpt-freq", default=-1, type=int)
    p.add_argument("--resume", default="", type=str)

    # Accept BOTH underscore + hyphen versions (you used underscore in your commands)
    p.add_argument("--gamma_quant", "--gamma-quant", dest="gamma_quant",
                   default=None, type=str)
    p.add_argument("--quant_bits", "--quant-bits", dest="quant_bits",
                   default=4, type=int)
    p.add_argument("--gamma_type", "--gamma-type", dest="gamma_type",
                   default="id", type=str)
    p.add_argument("--apply_gamma", "--apply-gamma", dest="apply_gamma",
                   default="global", type=str)

    p.add_argument("--ckpt-6hz", default="", type=str)
    p.add_argument("--ckpt-12hz", default="", type=str)
    p.add_argument("--ckpt-25hz", default="", type=str)

    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_cfg(args.config)

    fix_random_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    mkdir_if_missing(args.test_split_dir)

    if args.mode == "train_expert":
        if args.freq_tag is None:
            raise ValueError("--freq-tag is required for mode=train_expert (6hz/12hz/25hz)")

        # Build subject list from sens_folder
        sens_folder = cfg["dataset"]["sens_folder"]
        sbj_list = list_subjects_from_folder(sens_folder)

        # Logging style consistent with your previous run:
        # logs/deepconvlstm/<freq_tag>_702010/<timestamp>/
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join("logs", "deepconvlstm", f"{args.freq_tag}_702010", ts)
        mkdir_if_missing(log_dir)

        print(f"Logging to: {log_dir}")

        best_ckpt = run_inertial_expert_702010_continuous(
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
            ckpt_freq=args.ckpt_freq,
            resume=args.resume,
        )

        print(f"[{args.freq_tag}] Done. Best checkpoint: {best_ckpt}")

    else:
        # test_multifreq
        if not (args.ckpt_6hz and args.ckpt_12hz and args.ckpt_25hz):
            raise ValueError("--ckpt-6hz/--ckpt-12hz/--ckpt-25hz are required for mode=test_multifreq")

        print("Running multi-frequency cascade evaluation on held-out 10% windows...")

        test_multifreq_cascade_continuous(
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
