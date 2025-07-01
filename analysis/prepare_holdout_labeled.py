#!/usr/bin/env python3
"""Create labelled hold-out CSVs.

Given a holdout_<MODALITY>.csv containing subject IDs (one per line) and the
corresponding raw data CSV for that modality, this utility finds the diagnosis
label for each subject (mapped to CN/AD/MCI) and writes
  holdout_<MODALITY>_labeled.csv
with columns: subject,label.
"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

LABEL_MAP = {"SMC": "CN", "Dementia": "AD", "MCI": "MCI", "EMCI": "MCI", "LMCI": "MCI", "CN": "CN", "AD": "AD"}


def detect_label_column(df: pd.DataFrame):
    best_idx = None
    best_count = 0
    for i, col in enumerate(df.columns):
        uniques = set(df[col].dropna().astype(str).unique())
        count = len(uniques & set(LABEL_MAP.keys()))
        if count > best_count:
            best_idx = i
            best_count = count
    if best_idx is None:
        raise RuntimeError("Could not detect diagnosis column in data CSV")
    return best_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--modality', required=True, choices=['AV45','TAU'])
    ap.add_argument('--holdout_csv', required=True)
    ap.add_argument('--data_csv', required=True)
    ap.add_argument('--out_csv', help='Output CSV path; default: same dir with _labeled suffix')
    args = ap.parse_args()

    holdout_ids = pd.read_csv(args.holdout_csv, header=None)[0].astype(str).tolist()

    df = pd.read_csv(args.data_csv, header=None)
    dx_col_idx = detect_label_column(df)

    labeled_rows = []
    for _, row in df.iterrows():
        subj = str(row.iloc[0])
        if subj not in holdout_ids:
            continue
        raw_label = str(row.iloc[dx_col_idx])
        label = LABEL_MAP.get(raw_label, raw_label)
        labeled_rows.append({'subject': subj, 'label': label})

    if not labeled_rows:
        print("No hold-out subjects matched in data CSV.")
        return

    out_csv = args.out_csv or str(Path(args.holdout_csv).with_name(f"holdout_{args.modality}_labeled.csv"))
    pd.DataFrame(labeled_rows).to_csv(out_csv, index=False)
    print(f"Saved {out_csv} with {len(labeled_rows)} entries.")


if __name__ == '__main__':
    main() 