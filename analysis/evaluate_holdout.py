#!/usr/bin/env python3
"""Evaluate trained autoencoders on hold-out subjects.

For each modality (AV45/TAU) and each diagnosis label (AD/CN/MCI),
this script loads every trained model for the modality, computes the
average SSIM on the corresponding hold-out subjects, and generates a
bar-plot comparing models.

Outputs (per modality):
  analysis/
    holdout_eval_{modality}.csv       – metrics table
    holdout_plot_{modality}.png       – bar plot per label
"""
from pathlib import Path
import argparse, json, re
from typing import List
import torch, numpy as np, pandas as pd
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / 'code'))
from dataset import NiftiDataset
from monai_ae import MyAutoencoderKL
from utils import center_crop

LABELS = ['AD', 'CN', 'MCI']


def collect_model_dirs(results_dir: Path, modality: str) -> List[Path]:
    return sorted([p for p in (results_dir).iterdir()
                   if p.is_dir() and p.name.startswith(f"{modality}_")])

def load_best_model(model_dir: Path, device: str = 'cpu') -> torch.nn.Module:
    ckpt = model_dir / 'best_model.pt'
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} not found")
    # weights_only may fail; use map_location
    state = torch.load(ckpt, map_location=device, weights_only=False) if torch.__version__ >= '2.6' else torch.load(ckpt, map_location=device)
    model = MyAutoencoderKL().to(device)
    model.load_state_dict(state['state_dict'])
    model.eval()
    return model

def eval_ssim(model, loader, device: str):
    scores = []
    with torch.no_grad():
        for x, _, _ in loader:
            x = x.float().to(device)
            out = model(x)
            out_crop = center_crop(out.cpu(), (91, 109, 91))
            x_crop = center_crop(x.cpu(), (91, 109, 91))
            s = ssim(x_crop.squeeze(), out_crop.squeeze(), data_range=1.0)
            scores.append(s)
    return float(np.mean(scores)) if scores else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', required=True, help='Path to /home/exp1.encoder/results')
    ap.add_argument('--csv_dir', required=True, help='Path to source_data with *_data*.csv')
    ap.add_argument('--analysis_dir', default='analysis', help='Output directory for plots/csv')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    csv_dir = Path(args.csv_dir)
    analysis_dir = Path(args.analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    for modality in ['AV45', 'TAU']:
        labeled_csv = results_dir / f'holdout_{modality}_labeled.csv'
        if labeled_csv.exists():
            df_labeled = pd.read_csv(labeled_csv)
            subjects_by_label = {lbl: set(df_labeled[df_labeled['label'] == lbl]['subject'].astype(str).tolist()) for lbl in LABELS}
            subjects_holdout = set(df_labeled['subject'].astype(str).tolist())
        else:
            holdout_csv = results_dir / f'holdout_{modality}.csv'
            if not holdout_csv.exists():
                print(f"[skip] {holdout_csv} missing and no labeled csv")
                continue
            subjects_holdout = set(pd.read_csv(holdout_csv, header=None)[0].astype(str).tolist())
            subjects_by_label = None  # will infer later

        # pick the main data csv (underscores/dash)
        candidates = [f"{modality}_data_micai_fixed.csv", f"{modality}-data_micai_fixed.csv", f"{modality}_data.csv", f"{modality}-data.csv"]
        data_csv = None
        for c in candidates:
            if (csv_dir / c).exists():
                data_csv = csv_dir / c; break
        if data_csv is None:
            print(f"[skip] no data csv for {modality}")
            continue

        model_dirs = collect_model_dirs(results_dir, modality)
        rows = []
        device = args.device
        for label in LABELS:
            if subjects_by_label is not None:
                wanted_subjects = subjects_by_label.get(label, set())
            else:
                wanted_subjects = subjects_holdout

            ds = NiftiDataset(str(data_csv), labels=label, mode='eval', crop_size=(96,128,96), orig_size=(91,109,91), exclude_subjects=None)
            mask = [s in wanted_subjects for s in ds.subjects]
            if not any(mask):
                continue
            hold_ds_paths = [p for p, m in zip(ds.paths, mask) if m]
            if not hold_ds_paths:
                continue
            hold_ds = NiftiDataset(str(data_csv), labels=label, mode='eval', crop_size=(96,128,96), orig_size=(91,109,91), exclude_subjects=set())
            # override to only holdout paths
            hold_ds.paths = hold_ds_paths
            hold_ds.subjects = [Path(p).stem for p in hold_ds_paths]
            loader = DataLoader(hold_ds, batch_size=1, shuffle=False)

            for md in model_dirs:
                try:
                    model = load_best_model(md, device)
                    score = eval_ssim(model, loader, device)
                    tag = md.name.replace(f"{modality}_", '')
                    rows.append({'modality': modality, 'label': label, 'model': tag, 'ssim': score})
                except Exception as e:
                    print(f"[warn] failed {md}: {e}")
                    continue
        if not rows:
            continue
        df = pd.DataFrame(rows)
        csv_out = analysis_dir / f'holdout_eval_{modality}.csv'
        df.to_csv(csv_out, index=False)
        # plotting
        plt.figure(figsize=(8,4))
        sns.barplot(data=df, x='model', y='ssim', hue='label')
        plt.title(f'Hold-out SSIM – {modality}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(analysis_dir / f'holdout_plot_{modality}.png', dpi=300)
        plt.close()
        print(f"Saved {csv_out} and plot for {modality}")

if __name__ == '__main__':
    main() 