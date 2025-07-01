import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
import nibabel as nib
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, RandSpatialCropd, SpatialPadd, CenterSpatialCropd

LABEL_MAP = {"SMC": "CN", "Dementia": "AD", "MCI": "MCI", "EMCI": "MCI", "LMCI": "MCI"}

class NiftiDataset(Dataset):
    def __init__(self, csv_path, labels, crop_size=(96,128,96), orig_size=(91,109,91), mode="train", exclude_subjects=None):
        self.df = pd.read_csv(csv_path, header=None)
        if isinstance(labels, str):
            labels = [labels]
        self.labels = labels
        exclude_subjects = set(exclude_subjects or [])
        # --- Detect diagnosis column ---
        if 'DX' in self.df.columns:
            label_series = self.df['DX'].astype(str)
        else:
            # choose column whose set intersects LABEL_MAP keys the most
            best_idx = None
            best_count = 0
            for i, col in enumerate(self.df.columns):
                uniques = set(self.df[col].dropna().astype(str).unique())
                count = len(uniques & set(LABEL_MAP.keys()))
                if count > best_count:
                    best_idx = i; best_count = count
            if best_idx is None:
                raise RuntimeError('Could not locate diagnosis column')
            label_series = self.df.iloc[:, best_idx].astype(str)
        # Map raw diagnosis to canonical label and filter
        mapped = label_series.map(lambda x: LABEL_MAP.get(x, x))
        self.df = self.df[mapped.isin(labels)]
        # Find NIfTI path column (guess by .nii)
        nii_col = self.df.apply(lambda col: col.astype(str).str.contains('.nii', na=False).any(), axis=0)
        nii_idx = np.where(nii_col)[0][0]
        self.paths = self.df[nii_idx].tolist()
        self.subjects = self.df[0].tolist()  # assume first col is subject
        if exclude_subjects:
            mask = ~self.df[0].astype(str).isin(exclude_subjects)
            self.df = self.df[mask]
            self.paths = list(np.array(self.paths)[mask.values])
            self.subjects = list(np.array(self.subjects)[mask.values])
        self.crop_size = crop_size
        self.orig_size = orig_size
        self.mode = mode
        self.transforms = Compose([
            LoadImaged(keys=["img"]),
            EnsureChannelFirstd(keys=["img"]),
            SpatialPadd(keys=["img"], spatial_size=crop_size),
            RandSpatialCropd(keys=["img"], roi_size=crop_size, random_size=False) if mode=="train" else CenterSpatialCropd(keys=["img"], roi_size=crop_size),
        ])
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        path = self.paths[idx]
        subject = self.subjects[idx]
        data = {"img": path}
        data = self.transforms(data)
        img = data["img"].astype(np.float32)
        # Per-volume min–max scaling to 0–1
        img_min = img.min()
        img_max = img.max()
        if img_max - img_min > 0:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)
        img = torch.from_numpy(img) if not torch.is_tensor(img) else img
        return img, subject, str(path)

def center_crop_to_orig(img, orig_size=(91,109,91)):
    """Center crop a tensor/ndarray to original size."""
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    c = [ (s - o)//2 for s,o in zip(img.shape[-3:], orig_size) ]
    slices = tuple(slice(cc, cc+oo) for cc,oo in zip(c, orig_size))
    return img[..., slices[0], slices[1], slices[2]] 