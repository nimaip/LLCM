from monai.networks.nets import VarAutoEncoder
import torch
import torch.nn.functional as F
import config

def get_vae_model():
    """
    Initializes the 3D VAE model as specified in the plan.
    """
    model = VarAutoEncoder(
        spatial_dims=3,
        in_channels=1,         
        out_channels=1,       
        channels=config.VAE_CHANNELS,
        strides=config.VAE_STRIDES,
        latent_channels=config.LATENT_CHANNELS,
        num_res_units=2,       
        norm="BATCH",
    )
    return model

def encode_vae(vae, x):
    """
    Encodes input through VAE encoder and returns latent sample.
    Handles different MONAI VarAutoEncoder API versions.
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
    latent = mu + eps * std
    return latent

def decode_vae(vae, latent):
    """
    Decodes latent through VAE decoder.
    Handles different MONAI VarAutoEncoder API versions.
    """
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