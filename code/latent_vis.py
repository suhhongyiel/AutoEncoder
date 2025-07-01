import argparse
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from dataset import NiftiDataset
from monai_ae import MyAutoencoderKL
from utils import minmax_scale

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
    latents = []
    subjects = []
    for x, subject, _ in loader:
        x = x.float().to(args.device)
        x = minmax_scale(x)
        with torch.no_grad():
            z = model.encode(x)
        latents.append(z.cpu().numpy().reshape(-1))
        subjects.append(subject[0])
    latents = np.stack(latents)
    np.save(out_dir / 'latents.npy', latents)
    print('[saved]', out_dir / 'latents.npy')
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    emb = tsne.fit_transform(latents)
    plt.figure(figsize=(6,6))
    plt.scatter(emb[:,0], emb[:,1], s=10)
    for i, subj in enumerate(subjects):
        if i < 20:
            plt.text(emb[i,0], emb[i,1], subj, fontsize=6)
    plt.title('Latent t-SNE')
    plt.tight_layout()
    plt.savefig(out_dir / 'tsne.png', dpi=200)
    print('[saved]', out_dir / 'tsne.png')

if __name__ == '__main__':
    main() 