"""
unet_wrapper.py

Wrapper around MONAI UNet to add:
1. Timestep embedding (sinusoidal encoding)
2. Cross-attention conditioning from QPI encoder
3. Proper consistency model architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import UNet
import config
import math


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding as used in diffusion models."""

    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps):
        """
        Args:
            timesteps: (batch_size,) tensor of timestep indices
        Returns:
            embeddings: (batch_size, dim) tensor of timestep embeddings
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(0, half_dim, dtype=torch.float32) / half_dim
        ).to(timesteps.device)
        args = timesteps[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer."""

    def __init__(self, num_channels, cond_dim):
        super().__init__()
        self.scale = nn.Linear(cond_dim, num_channels)
        self.shift = nn.Linear(cond_dim, num_channels)

    def forward(self, x, cond):
        """
        Args:
            x: (B, C, D, H, W) - feature map
            cond: (B, cond_dim) - conditioning vector
        Returns:
            modulated: (B, C, D, H, W) - FiLM-modulated features
        """
        scale = self.scale(cond)[:, :, None, None, None]
        shift = self.shift(cond)[:, :, None, None, None]
        return x * (1 + scale) + shift


class ConsistencyUNet(nn.Module):
    """
    U-Net wrapper for Latent Consistency Model.

    Adds timestep conditioning and QPI context conditioning to base U-Net.
    Based on the paper's approach where the model learns consistency across
    different noise levels of the same clean data.

    Uses multi-layer FiLM conditioning for better context integration.
    """

    def __init__(self):
        super().__init__()

        # Base U-Net for processing latents
        self.unet = UNet(
            spatial_dims=3,
            in_channels=config.LATENT_CHANNELS,  # Just latent channels, no context concatenation
            out_channels=config.LATENT_CHANNELS,
            channels=(32, 64),  # 2 levels for small latent space
            strides=(2,),  # Single downsample by 2
            num_res_units=2,
            norm="BATCH",
        )

        # Timestep embedding
        time_embed_dim = 256
        self.time_embed = TimestepEmbedding(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Combined conditioning dimension
        cond_dim = time_embed_dim + config.QPI_CONTEXT_DIM

        # Multi-layer FiLM conditioning
        # Apply at input, intermediate features, and output
        self.film_input = FiLMLayer(config.LATENT_CHANNELS, cond_dim)
        self.film_mid = FiLMLayer(config.LATENT_CHANNELS, cond_dim)
        self.film_output = FiLMLayer(config.LATENT_CHANNELS, cond_dim)

    def forward(self, latent, timesteps, qpi_context):
        """
        Args:
            latent: (B, C, D, H, W) - noisy latent representation
            timesteps: (B,) - timestep indices
            qpi_context: (B, QPI_CONTEXT_DIM) - encoded QPI features

        Returns:
            output: (B, C, D, H, W) - predicted clean latent
        """
        # Get timestep embeddings
        t_emb = self.time_embed(timesteps)  # (B, time_embed_dim)
        t_emb = self.time_mlp(t_emb)  # (B, time_embed_dim)

        # Concatenate timestep and context conditioning
        combined_cond = torch.cat([t_emb, qpi_context], dim=1)  # (B, time_embed_dim + QPI_CONTEXT_DIM)

        # Apply FiLM at input
        conditioned_input = self.film_input(latent, combined_cond)

        # Pass through U-Net
        unet_output = self.unet(conditioned_input)

        # Apply FiLM at output for additional conditioning
        conditioned_output = self.film_output(unet_output, combined_cond)

        return conditioned_output


def get_consistency_unet():
    """Factory function to create ConsistencyUNet."""
    return ConsistencyUNet()


if __name__ == "__main__":
    # Test the model
    print("--- Testing ConsistencyUNet ---")
    model = get_consistency_unet()

    batch_size = 2
    downsample_factor = 2 ** len(config.VAE_STRIDES)
    latent_shape = (
        batch_size,
        config.LATENT_CHANNELS,
        config.ROI_SIZE[0] // downsample_factor,
        config.ROI_SIZE[1] // downsample_factor,
        config.ROI_SIZE[2] // downsample_factor
    )

    dummy_latent = torch.randn(latent_shape)
    dummy_timesteps = torch.randint(0, 1000, (batch_size,))
    dummy_context = torch.randn(batch_size, config.QPI_CONTEXT_DIM)

    print(f"Input latent shape: {dummy_latent.shape}")
    print(f"Timesteps: {dummy_timesteps.shape}")
    print(f"QPI context shape: {dummy_context.shape}")

    output = model(dummy_latent, dummy_timesteps, dummy_context)
    print(f"Output shape: {output.shape}")
    assert output.shape == dummy_latent.shape, "Output shape mismatch!"
    print("\nConsistencyUNet test passed!")