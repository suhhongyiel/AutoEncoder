import argparse
import subprocess
from pathlib import Path
import itertools
import random
import pandas as pd
import re
import torch

MODALITIES = ['AV45', 'TAU']
SINGLE_LABELS = ['CN', 'AD', 'MCI']
COMBOS = [
    ('CN',), ('AD',), ('MCI',),
    ('CN','AD','MCI'),
    ('CN','AD'), ('AD','MCI'), ('CN','MCI')
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--amp', action='store_true', help='Pass --amp to train_ae for mixed precision')
    
    args = parser.parse_args()
    csv_dir = Path(args.csv_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)
    for modality in MODALITIES:
        # holdout 10 subjects if not already saved
        holdout_path = out_dir / f'holdout_{modality}.csv'
        if not holdout_path.exists():
            candidates = [
                f"{modality}_data_micai_fixed.csv",
                f"{modality}-data_micai_fixed.csv",
                f"{modality}_data.csv",
                f"{modality}-data.csv"
            ]
            csv_base = None
            for cand in candidates:
                p = csv_dir / cand
                if p.exists():
                    csv_base = p; break
            if csv_base is None:
                print(f"[skip] No CSV for holdout in {modality}")
                continue
            df = pd.read_csv(csv_base, header=None)
            subjects = df[0].astype(str).unique().tolist()
            sample = rng.sample(subjects, min(10, len(subjects)))
            pd.Series(sample).to_csv(holdout_path, index=False, header=False)
            print(f"[holdout] Saved {holdout_path}")

    for modality, labels in itertools.product(MODALITIES, COMBOS):
        label_tag = '-'.join(labels)
        run_dir = out_dir / f'{modality}_{label_tag}'
        run_dir.mkdir(exist_ok=True)
        log_path = run_dir / 'train.log'
        # Resolve csv path with various naming conventions
        candidates = [
            f"{modality}_data_micai_fixed.csv",
            f"{modality}-data_micai_fixed.csv",
            f"{modality}_data.csv",
            f"{modality}-data.csv"
        ]
        csv_path = None
        for cand in candidates:
            p = csv_dir / cand
            if p.exists():
                csv_path = p
                break
        if csv_path is None:
            print(f"[skip] No CSV for {modality}: tried {candidates}")
            continue
        cmd = [
            'python', 'train_ae.py',
            '--csv', str(csv_path),
            '--labels', *labels,
            '--modality', modality,
            '--out_dir', str(run_dir),
            '--epochs', str(args.epochs),
            '--batch', str(args.batch),
            '--device', args.device
        ]
        if args.amp:
            cmd.append('--amp')
        cmd.extend(['--exclude_csv', str(out_dir / f'holdout_{modality}.csv')])
        print(' '.join(cmd))
        with open(log_path, 'w') as logf:
            subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
        # After run, extract metrics
        best_file = run_dir / 'best_model.pt'
        val_loss = None; val_ssim = None
        if best_file.exists():
            try:
                meta = torch.load(best_file, map_location='cpu', weights_only=False)
                val_loss = meta.get('val_loss'); val_ssim = meta.get('val_ssim')
            except Exception:
                pass
        if val_loss is None or val_ssim is None:
            # fallback: parse last occurrence in train.log
            if log_path.exists():
                text = Path(log_path).read_text().splitlines()
                for line in reversed(text):
                    m = re.search(r"val_loss=([0-9.eE+-]+).*val_ssim=([0-9.eE+-]+)", line)
                    if m:
                        val_loss = float(m.group(1)); val_ssim = float(m.group(2)); break
        if val_loss is not None and val_ssim is not None:
            with open(out_dir / 'summary.csv', 'a') as sf:
                sf.write(f"{modality},{label_tag},{val_loss:.6f},{val_ssim:.4f}\n")

if __name__ == '__main__':
    main() 