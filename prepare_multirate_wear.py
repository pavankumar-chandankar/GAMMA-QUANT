# ------------------------------------------------------------------------
# Prepare multi-rate versions of WEAR inertial CSVs
# ------------------------------------------------------------------------
# Reads original 50 Hz CSVs (./data/wear/raw/inertial by default),
# creates downsampled copies for 25 Hz and 12.5 Hz by simple decimation.
#
# Assumptions about CSVs:
#   - Same structure as used by Gamma-Quant:
#       * first column: time / sample index
#       * last column: "label"
#       * all other columns: sensor channels
#   - We keep columns identical; we only drop rows (decimate), then
#     reset the time/index in the first column to 0..N-1 so it matches
#     the new sampling rate in the configs.
#
# After running this, you will have:
#   ./data/wear/raw/inertial_25hz
#   ./data/wear/raw/inertial_12_5hz
# each containing sbj_*.csv compatible with the existing pipeline.
# ------------------------------------------------------------------------

import argparse
import os
from typing import Tuple

import pandas as pd


def downsample_df(df: pd.DataFrame, factor: int) -> pd.DataFrame:
    """
    Simple decimation: keep every `factor`-th row.

    We also reset the first column to 0..N-1 to be consistent with the
    new sampling rate used in the configs.
    """
    if factor <= 0:
        raise ValueError(f"Downsampling factor must be > 0, got {factor}.")

    df_ds = df.iloc[::factor].reset_index(drop=True).copy()

    # Reset first column to 0..N-1 (assumes first column is time/index)
    first_col = df_ds.columns[0]
    df_ds[first_col] = range(len(df_ds))

    return df_ds


def process_folder(
    src_folder: str,
    dst_folder_25: str,
    dst_folder_12_5: str,
    factor_25: int = 2,
    factor_12_5: int = 4,
) -> None:
    """
    Create 25 Hz and 12.5 Hz versions of all CSVs in src_folder.

    Parameters
    ----------
    src_folder : str
        Folder with original 50 Hz CSVs (e.g., ./data/wear/raw/inertial).
    dst_folder_25 : str
        Output folder for 25 Hz CSVs.
    dst_folder_12_5 : str
        Output folder for 12.5 Hz CSVs.
    factor_25 : int
        Downsampling factor for 25 Hz (2 for 50->25).
    factor_12_5 : int
        Downsampling factor for 12.5 Hz (4 for 50->12.5).
    """
    os.makedirs(dst_folder_25, exist_ok=True)
    os.makedirs(dst_folder_12_5, exist_ok=True)

    csv_files = sorted(
        f for f in os.listdir(src_folder) if f.lower().endswith(".csv")
    )

    print(f"Found {len(csv_files)} CSV files in {src_folder}")

    for fname in csv_files:
        src_path = os.path.join(src_folder, fname)
        print(f"Processing {fname} ...", flush=True)

        df = pd.read_csv(src_path, index_col=False, low_memory=False)

        # 25 Hz
        df_25 = downsample_df(df, factor_25)
        dst_25_path = os.path.join(dst_folder_25, fname)
        df_25.to_csv(dst_25_path, index=False)

        # 12.5 Hz
        df_12_5 = downsample_df(df, factor_12_5)
        dst_12_5_path = os.path.join(dst_folder_12_5, fname)
        df_12_5.to_csv(dst_12_5_path, index=False)

        print(
            f"  50 Hz rows: {len(df):7d} | 25 Hz rows: {len(df_25):7d} | 12.5 Hz rows: {len(df_12_5):7d}"
        )

    print("Done. Downsampled CSVs written to:")
    print(f"  25 Hz   -> {dst_folder_25}")
    print(f"  12.5 Hz -> {dst_folder_12_5}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create 25 Hz and 12.5 Hz versions of WEAR inertial CSVs."
    )
    parser.add_argument(
        "--src",
        type=str,
        default="./data/wear/raw/inertial",
        help="Source folder with original 50 Hz CSVs.",
    )
    parser.add_argument(
        "--dst_25",
        type=str,
        default="./data/wear/raw/inertial_25hz",
        help="Destination folder for 25 Hz CSVs.",
    )
    parser.add_argument(
        "--dst_12_5",
        type=str,
        default="./data/wear/raw/inertial_12_5hz",
        help="Destination folder for 12.5 Hz CSVs.",
    )
    parser.add_argument(
        "--factor_25",
        type=int,
        default=2,
        help="Downsampling factor for 25 Hz (default 2: 50->25).",
    )
    parser.add_argument(
        "--factor_12_5",
        type=int,
        default=4,
        help="Downsampling factor for 12.5 Hz (default 4: 50->12.5).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_folder(
        src_folder=args.src,
        dst_folder_25=args.dst_25,
        dst_folder_12_5=args.dst_12_5,
        factor_25=args.factor_25,
        factor_12_5=args.factor_12_5,
    )
