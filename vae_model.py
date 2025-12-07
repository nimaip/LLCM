from monai.networks.nets import VarAutoEncoder
import torch
import torch.nn.functional as F
import config

def get_vae_model():
    """
    Initializes the 3D VAE model as specified in the plan.
    """
    # Calculate latent spatial dimensions based on ROI_SIZE and strides
    downsample_factor = 2 ** len(config.VAE_STRIDES)
    latent_depth = config.ROI_SIZE[0] // downsample_factor
    latent_height = config.ROI_SIZE[1] // downsample_factor
    latent_width = config.ROI_SIZE[2] // downsample_factor
    # latent_size is the total number of elements: channels * depth * height * width
    latent_size = config.LATENT_CHANNELS * latent_depth * latent_height * latent_width
    
    model = VarAutoEncoder(
        spatial_dims=3,
        in_shape=(1, *config.ROI_SIZE),  # (channels, depth, height, width)
        out_channels=1,       
        latent_size=latent_size,  # Total number of latent elements (int)
        channels=config.VAE_CHANNELS,
        strides=config.VAE_STRIDES,
        num_res_units=2,       
        norm="BATCH",
    )
    return model

def encode_vae(vae, x):
    """
    Encodes input through VAE encoder and returns latent sample in spatial format.
    Handles different MONAI VarAutoEncoder API versions.

    Returns:
        latent: Spatial latent tensor (B, C, D, H, W) where C=LATENT_CHANNELS
    """
    try:
        # Try encode_forward method (newer API)
        mu, logvar = vae.encode_forward(x)
    except AttributeError:
        try:
            # Try encoder attribute
            mu, logvar = vae.encoder(x)
        except (AttributeError, TypeError):
            # Fallback: use forward and extract mu/logvar
            _, mu, logvar = vae(x)

    # Reparameterization trick
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    latent_flat = mu + eps * std  # Shape: (B, latent_size)

    # Reshape to spatial format for U-Net processing
    # latent_size = LATENT_CHANNELS * depth * height * width
    batch_size = latent_flat.shape[0]
    downsample_factor = 2 ** len(config.VAE_STRIDES)
    latent_shape = (
        batch_size,
        config.LATENT_CHANNELS,
        config.ROI_SIZE[0] // downsample_factor,
        config.ROI_SIZE[1] // downsample_factor,
        config.ROI_SIZE[2] // downsample_factor
    )
    latent = latent_flat.view(*latent_shape)
    return latent

def decode_vae(vae, latent):
    """
    Decodes latent through VAE decoder.
    Handles different MONAI VarAutoEncoder API versions.

    Args:
        vae: The VAE model
        latent: Either a flattened latent vector (B, latent_size) or spatial latent (B, C, D, H, W)
    """
    # If latent is spatial (5D), flatten it
    if latent.dim() == 5:
        batch_size = latent.shape[0]
        latent = latent.view(batch_size, -1)  # Flatten to (B, latent_size)

    try:
        # Try decode_forward method (newer API)
        return vae.decode_forward(latent)
    except AttributeError:
        try:
            # Try decoder attribute
            return vae.decoder(latent)
        except (AttributeError, TypeError):
            # Fallback: reconstruct from latent (may not work directly)
            raise RuntimeError("Could not find decode method. Check MONAI VarAutoEncoder API.")

def compute_kl_loss(mu, logvar):
    """
    Computes KL divergence loss: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    """
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl_loss

if __name__ == "__main__":
    model = get_vae_model()
    print(model)
    print(f"VAE Latent Channels: {config.LATENT_CHANNELS}")
    print(f"VAE Downsample Strides: {config.VAE_STRIDES}")