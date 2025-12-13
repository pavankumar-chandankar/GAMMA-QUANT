# ------------------------------------------------------------------------
# Offline simulation of dynamic sampling policy for WEAR / inertial data
# ------------------------------------------------------------------------
# This script estimates potential power savings (and reports basic
# accuracy numbers) for the activity-specific sampling policy defined in
# utils.sampling_policy, using the saved validation predictions produced
# by models.train.run_inertial_network.
#
# It does NOT actually resample the raw signals; instead, it assumes that
# each window occupies the same duration and uses the mapping
# activity -> target sampling rate to compute an *idealised* effective
# sampling rate over time.
# ------------------------------------------------------------------------

import argparse
import os

import numpy as np

from utils.sampling_policy import get_target_sampling_rate, FS_50_CLASSES, FS_25_CLASSES, FS_12_5_CLASSES


def compute_stats(labels: np.ndarray, desc: str):
    """
    Compute how much time (in terms of windows) would be spent at each
    sampling rate according to the policy, and estimate the relative
    power savings vs. always sampling at 50 Hz.

    Parameters
    ----------
    labels : np.ndarray
        1D array of activity ids.
    desc : str
        Description string (e.g. "ground-truth" or "predicted").
    """
    counts = {}
    for lab in labels:
        fs = get_target_sampling_rate(int(lab))
        counts[fs] = counts.get(fs, 0) + 1

    total_windows = len(labels)
    if total_windows == 0:
        print(f"[{desc}] No labels provided.")
        return

    print(f"\n=== Dynamic sampling stats ({desc}) ===")
    for fs in sorted(counts.keys()):
        frac = counts[fs] / total_windows
        print(f"  {fs:4.1f} Hz: {counts[fs]:6d} windows ({frac*100:5.1f}%)")

    # Effective average sampling rate (weighted by window count)
    avg_fs = sum(fs * n for fs, n in counts.items()) / float(total_windows)
    relative_power = avg_fs / 50.0  # vs. always 50 Hz
    saving = 1.0 - relative_power

    print(f"\n  Effective average sampling rate: {avg_fs:5.2f} Hz")
    print(f"  Relative power (vs 50 Hz):        {relative_power*100:5.1f}%")
    print(f"  Approx. power saving:             {saving*100:5.1f}%\n")


def main():
    parser = argparse.ArgumentParser(
        description="Offline simulation of dynamic sampling policy using saved validation predictions."
    )
    parser.add_argument(
        "--results_dir",
        required=True,
        type=str,
        help="Path to log directory for a run (e.g. logs/deepconvlstm/<timestamp>_runid).",
    )
    parser.add_argument(
        "--split_name",
        required=True,
        type=str,
        help="Split name used when saving results (e.g. 'loso_sbj_0').",
    )
    args = parser.parse_args()

    preds_path = os.path.join(args.results_dir, "unprocessed_results", "v_preds_" + args.split_name + ".npy")
    gt_path = os.path.join(args.results_dir, "unprocessed_results", "v_gt_" + args.split_name + ".npy")

    if not os.path.isfile(preds_path) or not os.path.isfile(gt_path):
        raise FileNotFoundError(
            f"Could not find v_preds_/v_gt_ files for split '{args.split_name}' in {args.results_dir}/unprocessed_results"
        )

    preds = np.load(preds_path)
    gt = np.load(gt_path)

    if preds.shape != gt.shape:
        print(f"Warning: preds shape {preds.shape} and gt shape {gt.shape} differ.")

    # Basic accuracy (at sample level)
    acc = float((preds == gt).sum()) / float(len(gt))
    print(f"Sample-wise accuracy (baseline 50 Hz model, no resampling): {acc*100:5.2f}%")

    # Stats under ideal policy (using ground-truth labels)
    compute_stats(gt, desc="ground-truth (ideal policy)")

    # Stats under policy driven by model predictions
    compute_stats(preds, desc="predicted (realistic policy)")


if __name__ == "__main__":
    main()
