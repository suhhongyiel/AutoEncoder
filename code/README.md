# Encoder/Decoder Analysis for Conditional Diffusion in Medical Imaging

This folder contains reference code and guidelines for analysing the **encoder** and **decoder** used inside Latent-Diffusion models (LDM) applied to medical images (e.g. PET, MRI).

Why this matters
----------------
* In generic vision LDMs (e.g. Stable Diffusion) the VAE encoder-decoder is trained on large, diverse datasets and is assumed to reconstruct most inputs faithfully.
* In medical imaging **domain gaps** and **pathology-specific signals** are subtle.  If the VAE is trained only on cognitively-normal (CN) brains it may fail to represent Alzheimer (AD) pattern and leak pathology information into the noise space, corrupt conditional synthesis, or bias downstream evaluation.

Goals of this analysis
----------------------
1. **Reconstruction fidelity** – can the current encoder/decoder reconstruct CN *and* AD images with comparable error?
2. **Latent distribution shift** – does the latent space separate pathologies (desired?) or ignore them (problematic for conditional guidance)?
3. **Condition leakage** – does the latent already contain enough information to classify pathology without the diffusion process?
4. **Robustness** – evaluate the effect of noise perturbation or small adversarial perturbations on latent representations.

Workflow overview
-----------------
```
1. Prepare file lists                 (already done: data/*.csv)
2.  └─> load images (NIfTI)           ┐  see `analysis_encoder.py`
3.      encode with VAE / AE          │
4.      reconstruct & compute metrics │  PSNR, SSIM, MSE, NMSE, etc.
5.      embed latents (t-SNE, UMAP)   │
6.      train simple classifier        ┘  logistic regression / SVM
7. Visualise & report
```

Key scripts
-----------
* `analysis_encoder.py` – command-line tool that executes steps 2-6 for a CSV list of images and a VAE checkpoint.
* `data_io.py` – utilities for reading NIfTI, normalising volumes and batching.
* `model_io.py` – wrappers for loading the encoder and decoder.
* `visualise.py` – helper plotting functions (t-SNE, violin plots, etc.).

Quickstart
----------
```bash
# (1) create a Python virtualenv if desired
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# (2) run analysis for one model/variant
python analysis_encoder.py \
    --csv ../data/AV45-DDPM-proposed-AV451-1_files.csv \
    --vae_ckpt /path/to/vae.ckpt \
    --out_dir out/AV45-DDPM-proposed-AV451-1
```

The script will save:
* `metrics.csv` – reconstruction metrics per image
* `latent.npy`  – latent vectors (N × D)
* `tsne.png`    – 2-D embedding coloured by pathology/condition

Extending / next steps
----------------------
* Replace the simple logistic regression with a more sophisticated classifier or representation-similarity analysis (RSA).
* Investigate training a **domain-balanced VAE** (train on CN+AD) and compare results.
* Explore injecting condition tokens into the VAE rather than only at the diffusion UNet. 