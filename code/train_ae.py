import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from torch.nn import MSELoss
from monai.optimizers import Novograd
from monai.utils import set_determinism
from tqdm import tqdm
import numpy as np
from dataset import NiftiDataset, center_crop_to_orig
from monai_ae import MyAutoencoderKL
from utils import minmax_scale, center_crop
from skimage.metrics import structural_similarity as ssim

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--labels', nargs='+', required=True, help='One or more labels e.g. CN AD')
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--modality', type=str, required=True, choices=['AV45','TAU'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--amp', action='store_true', help='Use mixed precision (FP16) to reduce memory')
    parser.add_argument('--exclude_csv', type=str, help='CSV of subject IDs to exclude from training')
    args = parser.parse_args()

    exclude_set = set()
    if args.exclude_csv and Path(args.exclude_csv).exists():
        import pandas as pd
        exclude_set = set(pd.read_csv(args.exclude_csv, header=None)[0].astype(str).tolist())

    set_determinism(42)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    ds = NiftiDataset(args.csv, args.labels, crop_size=(96,128,96), orig_size=(91,109,91), mode="train", exclude_subjects=exclude_set)
    n_val = max(1, int(0.1 * len(ds)))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # Model
    model = MyAutoencoderKL().to(args.device)
    optimizer = Novograd(model.parameters(), lr=1e-3)
    criterion = MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and args.device.startswith('cuda'))

    best_val_loss = float('inf')
    best_ssim = -1.0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for x, _, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train]"):
            x = x.float().to(args.device)
            x = minmax_scale(x)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.amp and args.device.startswith('cuda')):
                out = model(x)
                loss = criterion(out, x)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0
        val_ssim = 0
        with torch.no_grad():
            for x, _, _ in val_loader:
                x = x.float().to(args.device)
                x = minmax_scale(x)
                out = model(x)
                # Center crop output to (91,109,91)
                out_crop = center_crop(out.cpu(), (91,109,91))
                x_crop = center_crop(x.cpu(), (91,109,91))
                with torch.cuda.amp.autocast(enabled=args.amp and args.device.startswith('cuda')):
                    loss = criterion(torch.tensor(out_crop), torch.tensor(x_crop))
                val_loss += loss.item()
                ss = ssim(x_crop.squeeze(), out_crop.squeeze(), data_range=1.0)
                val_ssim += ss
        n_val_samples = len(val_loader.dataset)
        val_loss /= n_val_samples
        val_ssim /= n_val_samples
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_ssim={val_ssim:.4f}")
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ssim = val_ssim
            fname = out_dir / "best_model.pt"
            torch.save({"state_dict": model.state_dict(), "val_loss": best_val_loss, "val_ssim": best_ssim}, fname)
            print(f"[saved] {fname.name} (val_loss={val_loss:.4f}, val_ssim={val_ssim:.4f})")

if __name__ == '__main__':
    main() 