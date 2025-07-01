#!/usr/bin/env python3
"""Visualise hold-out reconstructions.

For 각 모달리티(AV45/TAU)와 라벨(AD/CN/MCI)별 최고 SSIM 모델을 골라
hold-out 샘플 최대 5개를 시각화한다.

결과 PNG 는 analysis/holdout_vis_{modality}_{label}.png 에 저장된다.
"""
from pathlib import Path
import argparse, re, numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
import seaborn as sns

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / 'code'))
from dataset import NiftiDataset
from monai_ae import MyAutoencoderKL
from utils import center_crop

LABELS = ['AD', 'CN', 'MCI']


def load_best_model(model_dir: Path, device='cpu'):
    ckpt = model_dir / 'best_model.pt'
    state = torch.load(ckpt, map_location=device, weights_only=False) if torch.__version__ >= '2.6' else torch.load(ckpt, map_location=device)
    model = MyAutoencoderKL().to(device)
    model.load_state_dict(state['state_dict'])
    model.eval()
    return model


def mid_slice(vol):
    # vol shape C,Z,Y,X or Z,Y,X
    arr = vol.squeeze()
    z = arr.shape[0]//2
    return arr[z, :, :]


def visualise(modality, label, best_model_dir, data_csv, subjects_holdout, out_png, device):
    ds = NiftiDataset(str(data_csv), labels=label, mode='eval', crop_size=(96,128,96), orig_size=(91,109,91))
    # keep only holdout ids
    mask = [s in subjects_holdout for s in ds.subjects]
    paths = [p for p, m in zip(ds.paths, mask) if m][:5]  # max 5 samples
    if not paths:
        return False
    ds.paths = paths
    ds.subjects = [Path(p).stem for p in paths]
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    model = load_best_model(best_model_dir, device)

    cols = 3
    rows = len(paths)
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    if rows==1:
        axes = np.expand_dims(axes,0)
    for i,(x, subj, pth) in enumerate(loader):
        x = x.float().to(device)
        with torch.no_grad():
            out = model(x)
        out_np = out.cpu().numpy()[0,0]  # Z,Y,X
        x_np = x.cpu().numpy()[0,0]
        x_crop = center_crop(x_np, (91,109,91))
        out_crop = center_crop(out_np, (91,109,91))
        score = ssim(x_crop.squeeze(), out_crop.squeeze(), data_range=1.0)
        axes[i,0].imshow(mid_slice(x_crop), cmap='gray'); axes[i,0].set_title('Original'); axes[i,0].axis('off')
        axes[i,1].imshow(mid_slice(out_crop), cmap='gray'); axes[i,1].set_title('Reconstruction'); axes[i,1].axis('off')
        diff = np.abs(mid_slice(x_crop)-mid_slice(out_crop))
        axes[i,2].imshow(diff, cmap='hot'); axes[i,2].set_title(f'SSIM={score:.3f}'); axes[i,2].axis('off')
    # Add label annotation on the left of originals
    axes[0,0].set_ylabel(label, rotation=0, ha='right', va='center', fontsize=12, color='red')
    # shorter title to avoid overlap
    plt.suptitle(f'{modality} – {label}', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_png, dpi=300)
    plt.close()
    return True


def collect_model_dirs(results_dir: Path, modality: str):
    return sorted([p for p in results_dir.iterdir() if p.is_dir() and p.name.startswith(f"{modality}_")])


def compare_ad_across_models(modality: str, res_dir: Path, data_csv: Path, holdout_ids: set, out_png: Path, device='cpu', max_subjects: int = 3):
    """Plot AD images after reconstruction by every encoder.

    Grid layout:
        rows    = subjects (up to max_subjects)
        columns = Original + one per model (n_models)
    """
    # Collect AD hold-out subjects
    ds = NiftiDataset(str(data_csv), labels='AD', mode='eval', crop_size=(96,128,96), orig_size=(91,109,91))
    mask = [s in holdout_ids for s in ds.subjects]
    paths = [p for p, m in zip(ds.paths, mask) if m]
    # if not enough AD hold-out subjects, fallback to first AD subjects overall
    if not paths:
        paths = ds.paths[:max_subjects]
    else:
        paths = paths[:max_subjects]
    if not paths:
        return False  # still none
    subjects = [Path(p).stem for p in paths]
    # prepare loader just for originals
    ds.paths = paths
    ds.subjects = subjects
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    # Load all models once
    model_dirs = collect_model_dirs(res_dir, modality)
    models = {}
    for md in model_dirs:
        try:
            tag = md.name.replace(f"{modality}_", '')
            models[tag] = load_best_model(md, device)
        except Exception as e:
            print(f"[warn] skip {md}: {e}")
    if not models:
        return False

    n_rows = len(paths)
    n_cols = 1 + len(models)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    if n_rows==1:
        axes = np.expand_dims(axes,0)
    if n_cols==1:
        axes = np.expand_dims(axes,1)

    # Original + recon grid per subject row
    for row_idx, (x, subj, pth) in enumerate(loader):
        x_np = x.numpy()[0,0]
        x_crop = center_crop(x_np, (91,109,91))
        axes[row_idx, 0].imshow(mid_slice(x_crop), cmap='gray')
        axes[row_idx, 0].set_title('Original') if row_idx==0 else None
        axes[row_idx, 0].set_ylabel(f'Subj {subj[0]}', rotation=0, ha='right', va='center')
        axes[row_idx, 0].axis('off')

        for col_idx, (tag, model) in enumerate(models.items(), start=1):
            with torch.no_grad():
                out = model(x.to(device))
            out_np = out.cpu().numpy()[0,0]
            out_crop = center_crop(out_np, (91,109,91))
            # compute SSIM
            score = ssim(x_crop.squeeze(), out_crop.squeeze(), data_range=1.0)
            axes[row_idx, col_idx].imshow(mid_slice(out_crop), cmap='gray')
            if row_idx==0:
                axes[row_idx, col_idx].set_title(f'{tag}\n{score:.3f}', fontsize=9)
            else:
                axes[row_idx, col_idx].set_title(f'{score:.3f}', fontsize=8)
            axes[row_idx, col_idx].axis('off')

    plt.suptitle(f'{modality} – AD reconstructions across models', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_png, dpi=300)
    plt.close()
    return True


def compare_labels_across_models(modality: str, res_dir: Path, data_csv: Path, holdout_by_label: dict, out_png: Path, device='cpu'):
    """Pick one hold-out subject per label (AD, CN, MCI) and show reconstructions across all models.

    Grid: rows = labels, columns = Original + models
    """
    model_dirs = collect_model_dirs(res_dir, modality)
    if not model_dirs:
        return False
    models = {md.name.replace(f"{modality}_", ''): load_best_model(md, device) for md in model_dirs}

    n_rows = len(LABELS)
    n_cols = 1 + len(models)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    if n_rows==1:
        axes = np.expand_dims(axes,0)
    if n_cols==1:
        axes = np.expand_dims(axes,1)

    for row_idx, label in enumerate(LABELS):
        # choose first subject for this label
        subj_set = holdout_by_label.get(label, set())
        if not subj_set:
            continue
        chosen_subj = sorted(list(subj_set))[0]
        # load path of this subject via dataset
        ds = NiftiDataset(str(data_csv), labels=label, mode='eval', crop_size=(96,128,96), orig_size=(91,109,91))
        idx = next((i for i,s in enumerate(ds.subjects) if s == chosen_subj), None)
        if idx is None:
            continue
        x, _, _ = ds[idx]
        x_np = x.numpy()[0]
        x_crop = center_crop(x_np, (91,109,91))
        ax0 = axes[row_idx,0]
        ax0.imshow(mid_slice(x_crop), cmap='gray')
        if row_idx==0:
            ax0.set_title('Original')
        # put label text slightly above the image
        ax0.text(0.5, 1.05, label, ha='center', va='bottom', transform=ax0.transAxes, fontsize=12, color='red')
        ax0.axis('off')

        for col_idx,(tag,model) in enumerate(models.items(), start=1):
            with torch.no_grad():
                out = model(x.unsqueeze(0).to(device))
            out_np = out.cpu().numpy()[0,0]
            out_crop = center_crop(out_np, (91,109,91))
            score = ssim(x_crop.squeeze(), out_crop.squeeze(), data_range=1.0)
            axes[row_idx,col_idx].imshow(mid_slice(out_crop), cmap='gray')
            if row_idx==0:
                axes[row_idx,col_idx].set_title(f'{tag}\n{score:.3f}', fontsize=9)
            else:
                axes[row_idx,col_idx].set_title(f'{score:.3f}', fontsize=8)
            axes[row_idx,col_idx].axis('off')

    plt.suptitle(f'{modality} – One sample per label across models', fontsize=14)
    plt.tight_layout(rect=[0,0,1,0.94])
    plt.savefig(out_png,dpi=300)
    plt.close()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', required=True)
    ap.add_argument('--csv_dir', required=True)
    ap.add_argument('--analysis_dir', default='analysis')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    res_dir = Path(args.results_dir)
    csv_dir = Path(args.csv_dir)
    out_dir = Path(args.analysis_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for modality in ['AV45','TAU']:
        eval_csv = out_dir / f'holdout_eval_{modality}.csv'
        if not eval_csv.exists():
            print(f'skip {modality} eval csv missing'); continue
        df = pd.read_csv(eval_csv)
        # load labeled holdout subjects if available
        labeled_csv = res_dir / f'holdout_{modality}_labeled.csv'
        if labeled_csv.exists():
            df_lab = pd.read_csv(labeled_csv)
            holdout_by_label = {lbl: set(df_lab[df_lab['label']==lbl]['subject'].astype(str).tolist()) for lbl in LABELS}
            holdout_ids_all = set(df_lab['subject'].astype(str).tolist())
        else:
            holdout_ids_all = set(pd.read_csv(res_dir / f'holdout_{modality}.csv', header=None)[0].astype(str).tolist())
            holdout_by_label = {lbl: holdout_ids_all for lbl in LABELS}
        # resolve data csv
        candidates = [f"{modality}_data_micai_fixed.csv", f"{modality}-data_micai_fixed.csv", f"{modality}_data.csv", f"{modality}-data.csv"]
        data_csv=None
        for c in candidates:
            if (csv_dir/ c).exists(): data_csv = csv_dir/ c; break
        if data_csv is None:
            continue
        for label in LABELS:
            sub = df[df['label']==label]
            if sub.empty: continue
            best_row = sub.sort_values('ssim', ascending=False).iloc[0]
            model_dir = res_dir / f"{modality}_{best_row['model']}"
            out_png = out_dir / f'holdout_vis_{modality}_{label}.png'
            ok = visualise(modality,label,model_dir,data_csv,holdout_by_label[label],out_png,args.device)
            if ok:
                print('saved',out_png)

        # --- extra: compare AD across all models ---
        compare_png = out_dir / f'holdout_AD_compare_{modality}.png'
        data_csv_for_mod = data_csv  # from earlier loop
        if data_csv_for_mod is not None:
            ok2 = compare_ad_across_models(modality, res_dir, data_csv_for_mod, holdout_by_label['AD'], compare_png, args.device)
            if ok2:
                print('saved', compare_png)

        # ----- new combined label comparison -----
        label_cmp_png = out_dir / f'holdout_label_compare_{modality}.png'
        if data_csv is not None:
            ok3 = compare_labels_across_models(modality, res_dir, data_csv, holdout_by_label, label_cmp_png, args.device)
            if ok3:
                print('saved', label_cmp_png)

if __name__=='__main__':
    main() 