import numpy as np
import torch

def minmax_scale(img, eps=1e-8):
    minv = img.min()
    maxv = img.max()
    return (img - minv) / (maxv - minv + eps)

def crop_or_pad(img, target_shape):
    """Crop or pad a 3D numpy array or torch tensor to target_shape."""
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    out = np.zeros(target_shape, dtype=img.dtype)
    src = [slice(0, min(s, t)) for s, t in zip(img.shape, target_shape)]
    dst = [slice(0, min(s, t)) for s, t in zip(img.shape, target_shape)]
    out[tuple(dst)] = img[tuple(src)]
    return out

def center_crop(img, crop_shape):
    """Center crop a 3D numpy array or torch tensor to crop_shape."""
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    c = [ (s - c)//2 for s, c in zip(img.shape, crop_shape) ]
    slices = tuple(slice(cc, cc+cs) for cc,cs in zip(c, crop_shape))
    return img[slices] 