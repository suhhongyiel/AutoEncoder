import torch
from monai.networks.nets import AutoEncoder

class MyAutoencoderKL(torch.nn.Module):
    """
    Wrapper for MONAI's AutoEncoder (not AutoencoderKL) for 3D medical images.
    This is a simple VAE-like autoencoder (no KL loss, just reconstruction).
    """
    def __init__(self, spatial_dims=3, in_channels=1, out_channels=1,
                 channels=(32, 64, 128, 256), strides=(2, 2, 2, 2),
                 num_res_units=2):
        super().__init__()
        self.ae = AutoEncoder(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units
        )
    def forward(self, x):
        return self.ae(x)
    def encode(self, x):
        # Only returns the encoded feature (not a distribution)
        return self.ae.encode(x)
    def decode(self, z):
        return self.ae.decode(z) 