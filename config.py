import torch
import sys

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Configuration
ROI_SIZE = (32, 128, 128)  # (Depth, Height, Width) - Must be divisible by 16 for VAE
SAMPLES_PER_VOLUME = 4     # Number of random crops per volume
NUM_WORKERS = 0            # Set to 0 for Windows compatibility (use 4+ on Linux with GPU)
BATCH_SIZE = 2             # Increase to 4-8 if GPU memory allows

# VAE Configuration
VAE_MODEL_PATH = "vae.pth"
VAE_CHANNELS = (16, 32, 64, 128)
VAE_STRIDES = (2, 2, 2, 2)
LATENT_CHANNELS = 8
VAE_EPOCHS = 50            # Production: 50-100 epochs for good compression
VAE_LR = 1e-4              # Learning rate

# VAE Loss Weights
L1_WEIGHT = 1.0            # Reconstruction loss
KL_WEIGHT = 1e-6           # KL divergence (latent regularization)
PERCEPTUAL_WEIGHT = 0.1    # LPIPS perceptual loss

# LLCM Configuration
QPI_ENCODER_PATH = "qpi_encoder.pth"
LLCM_EMA_PATH = "llcm_ema.pth"
LLCM_EPOCHS = 100          # Production: 100-200 epochs for consistency learning
LLCM_LR = 1e-4             # Learning rate
QPI_CONTEXT_DIM = 256      # QPI encoder output dimension
LLCM_UNET_CHANNELS = (32, 64, 128, 256)
LLCM_UNET_STRIDES = (2, 2, 2, 2)
NUM_TRAIN_TIMESTEPS = 1000 # Diffusion timesteps
EMA_DECAY = 0.999          # Teacher model EMA decay

INFERENCE_TIMESTEP = 0  # LCM uses timestep 0 for 1-step inference


def validate_config():
    """
    Validates configuration parameters for consistency and correctness.
    Raises ValueError if any validation fails.
    """
    errors = []
    
    # Check ROI_SIZE dimensions are divisible by downsample factor
    downsample_factor = 2 ** len(VAE_STRIDES)
    for i, dim in enumerate(ROI_SIZE):
        if dim % downsample_factor != 0:
            errors.append(
                f"ROI_SIZE[{i}] = {dim} must be divisible by downsample_factor = {downsample_factor} "
                f"(2^{len(VAE_STRIDES)} = {downsample_factor})"
            )
    
    # Check INFERENCE_TIMESTEP is valid
    if INFERENCE_TIMESTEP < 0 or INFERENCE_TIMESTEP >= NUM_TRAIN_TIMESTEPS:
        errors.append(
            f"INFERENCE_TIMESTEP = {INFERENCE_TIMESTEP} must be in range [0, {NUM_TRAIN_TIMESTEPS})"
        )
    
    # Check channel/strides consistency
    if len(VAE_CHANNELS) != len(VAE_STRIDES):
        errors.append(
            f"VAE_CHANNELS length ({len(VAE_CHANNELS)}) must match VAE_STRIDES length ({len(VAE_STRIDES)})"
        )
    
    if len(LLCM_UNET_CHANNELS) != len(LLCM_UNET_STRIDES):
        errors.append(
            f"LLCM_UNET_CHANNELS length ({len(LLCM_UNET_CHANNELS)}) must match "
            f"LLCM_UNET_STRIDES length ({len(LLCM_UNET_STRIDES)})"
        )
    
    # Check positive values
    if BATCH_SIZE <= 0:
        errors.append(f"BATCH_SIZE = {BATCH_SIZE} must be positive")
    if NUM_WORKERS < 0:
        errors.append(f"NUM_WORKERS = {NUM_WORKERS} must be non-negative")
    if SAMPLES_PER_VOLUME <= 0:
        errors.append(f"SAMPLES_PER_VOLUME = {SAMPLES_PER_VOLUME} must be positive")
    if VAE_EPOCHS <= 0:
        errors.append(f"VAE_EPOCHS = {VAE_EPOCHS} must be positive")
    if LLCM_EPOCHS <= 0:
        errors.append(f"LLCM_EPOCHS = {LLCM_EPOCHS} must be positive")
    if NUM_TRAIN_TIMESTEPS <= 0:
        errors.append(f"NUM_TRAIN_TIMESTEPS = {NUM_TRAIN_TIMESTEPS} must be positive")
    if not (0 < EMA_DECAY < 1):
        errors.append(f"EMA_DECAY = {EMA_DECAY} must be in (0, 1)")
    if QPI_CONTEXT_DIM <= 0:
        errors.append(f"QPI_CONTEXT_DIM = {QPI_CONTEXT_DIM} must be positive")
    if LATENT_CHANNELS <= 0:
        errors.append(f"LATENT_CHANNELS = {LATENT_CHANNELS} must be positive")
    
    # Check learning rates
    if VAE_LR <= 0:
        errors.append(f"VAE_LR = {VAE_LR} must be positive")
    if LLCM_LR <= 0:
        errors.append(f"LLCM_LR = {LLCM_LR} must be positive")
    
    # Check loss weights
    if L1_WEIGHT < 0:
        errors.append(f"L1_WEIGHT = {L1_WEIGHT} must be non-negative")
    if KL_WEIGHT < 0:
        errors.append(f"KL_WEIGHT = {KL_WEIGHT} must be non-negative")
    if PERCEPTUAL_WEIGHT < 0:
        errors.append(f"PERCEPTUAL_WEIGHT = {PERCEPTUAL_WEIGHT} must be non-negative")
    
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)
    
    return True


# Auto-validate on import (can be disabled if needed)
if __name__ != "__main__":
    try:
        validate_config()
    except ValueError as e:
        print(f"Warning: Configuration validation failed: {e}")
        print("Continuing anyway, but errors may occur during execution.")