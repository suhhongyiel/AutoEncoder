import argparse
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import matplotlib.pyplot as plt
from dataset import NiftiDataset, center_crop_to_orig
from monai_ae import MyAutoencoderKL
from utils import minmax_scale, center_crop

def save_slice_png(vol, out_path, slice_idx=None, title=None):
    arr = vol.squeeze()
    if slice_idx is None:
        slice_idx = arr.shape[-1] // 2
    plt.figure(figsize=(4,4))
    plt.imshow(arr[..., slice_idx], cmap='gray')
    if title:
        plt.title(title)
    plt.axis('off')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = NiftiDataset(args.csv, label=None, crop_size=(96,128,96), orig_size=(91,109,91), mode="eval")
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    model = MyAutoencoderKL(img_size=(96,128,96)).to(args.device)
    model.load_state_dict(torch.load(args.model, map_location=args.device))
    model.eval()
    rows = []
    for i, (x, subject, path) in enumerate(loader):
        x = x.float().to(args.device)
        x = minmax_scale(x)
        with torch.no_grad():
            out = model(x)
        out_crop = center_crop(out.cpu(), (91,109,91)).squeeze().numpy()
        x_crop = center_crop(x.cpu(), (91,109,91)).squeeze().numpy()
        mse = np.mean((out_crop - x_crop) ** 2)
        p = psnr(x_crop, out_crop, data_range=1.0)
        s = ssim(x_crop, out_crop, data_range=1.0)
        rows.append({'subject': subject[0], 'file': path[0], 'mse': mse, 'psnr': p, 'ssim': s})
        if i < 10:
            save_slice_png(x_crop, out_dir / f'{subject[0]}_input.png', title='Input')
            save_slice_png(out_crop, out_dir / f'{subject[0]}_recon.png', title='Recon')
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'metrics.csv', index=False)
    print('[saved]', out_dir / 'metrics.csv')
    # summary
    summary = df[['mse','psnr','ssim']].mean().to_dict()
    with open(out_dir / 'summary.txt','w') as f:
        for k,v in summary.items():
            f.write(f'{k}: {v}\n')
    print('[saved]', out_dir / 'summary.txt')

if __name__ == '__main__':
    main() 