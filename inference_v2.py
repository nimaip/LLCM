"""
inference_v2.py

Corrected inference for LLCM.

Key insight from paper: The consistency model is trained to predict
the clean data (x0) from noisy versions at different timesteps.
For 1-step inference, we:
1. Start with noisy latent from a high noise timestep
2. Use the model to directly predict clean latent (consistency property)
3. Decode to image space

For conditional generation (QPI → DAPI):
- The QPI context guides the generation through FiLM conditioning
"""

import torch
import config
import vae_model
from unet_wrapper import get_consistency_unet
import llcm_model
from diffusers.schedulers import LMSDiscreteScheduler
import numpy as np
import tifffile
from train_vae import setup_dummy_data
import data_pipeline
import sys


@torch.no_grad()
def run_inference(qpi_patch):
    """
    Runs 1-step LLCM inference on a single QPI patch.

    Args:
        qpi_patch: (1, 1, D, H, W) QPI input

    Returns:
        Generated DAPI patch as numpy array
    """
    print(f"--- Starting Phase 3: 1-Step LLCM Inference (Corrected) ---")
    device = config.DEVICE

    try:
        print("Loading all models...")

        # Load VAE
        vae = vae_model.get_vae_model().to(device)
        try:
            vae.load_state_dict(torch.load(config.VAE_MODEL_PATH, map_location=device))
        except FileNotFoundError:
            raise FileNotFoundError(f"VAE model not found at {config.VAE_MODEL_PATH}")
        vae.eval()

        # Load QPI Encoder
        qpi_encoder = llcm_model.QPIEncoder().to(device)
        try:
            qpi_encoder.load_state_dict(torch.load(config.QPI_ENCODER_PATH, map_location=device))
        except FileNotFoundError:
            raise FileNotFoundError(f"QPI encoder not found at {config.QPI_ENCODER_PATH}")
        qpi_encoder.eval()

        # Load LLCM (teacher/EMA model)
        llcm_ema = get_consistency_unet().to(device)
        try:
            llcm_ema.load_state_dict(torch.load(config.LLCM_EMA_PATH, map_location=device), strict=False)
        except FileNotFoundError:
            raise FileNotFoundError(f"LLCM model not found at {config.LLCM_EMA_PATH}")
        llcm_ema.eval()

        # Prepare input
        qpi_patch = qpi_patch.to(device)
        if qpi_patch.dim() == 4:
            qpi_patch = qpi_patch.unsqueeze(0)

        if qpi_patch.dim() != 5:
            raise ValueError(f"Expected 5D tensor (B, C, D, H, W), got {qpi_patch.dim()}D tensor")

        print("Running 1-step generation...")

        # 1. Encode QPI context (full 256D vector)
        qpi_context = qpi_encoder(qpi_patch)  # (batch, 256)

        # 2. Start from pure noise (matching training distribution)
        # During training, the model learns: noisy_DAPI_latent → clean_DAPI_latent
        # So during inference, we should start from pure noise and let the QPI context guide generation

        # Get latent shape from QPI encoding (just for dimensions)
        qpi_latent = vae_model.encode_vae(vae, qpi_patch)  # (batch, 8, 2, 8, 8)

        # Start from pure Gaussian noise (matching training)
        initial_noise = torch.randn_like(qpi_latent)

        # Use a high timestep for 1-step inference (as trained)
        # The consistency model should denoise in a single step from high noise
        inference_timestep = config.NUM_TRAIN_TIMESTEPS - 100  # Near-max noise
        timesteps = torch.full((qpi_patch.shape[0],), inference_timestep, device=device, dtype=torch.long)

        # Compute noise scaling
        scheduler = LMSDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=config.NUM_TRAIN_TIMESTEPS
        )
        scheduler.set_timesteps(config.NUM_TRAIN_TIMESTEPS)

        betas = scheduler.betas.to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Scale the initial noise to the appropriate noise level
        alpha_prod = alphas_cumprod[inference_timestep].view(1, 1, 1, 1, 1)
        noisy_latent = torch.sqrt(alpha_prod) * initial_noise + torch.sqrt(1 - alpha_prod) * initial_noise

        # 3. Predict clean DAPI latent using consistency model
        # The QPI context guides the denoising to produce DAPI output
        predicted_dapi_latent = llcm_ema(noisy_latent, timesteps, qpi_context)

        # 4. Decode to image space
        virtual_stain_patch = vae_model.decode_vae(vae, predicted_dapi_latent)

        print("--- Inference Complete ---")
        return virtual_stain_patch.cpu().numpy()

    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    try:
        # Use 3d_data if it exists, otherwise fall back to dummy_data
        import os
        if os.path.exists("3d_data") and os.path.exists(os.path.join("3d_data", "qpi")):
            data_dir = "3d_data"
            print(f"Using real data from: {data_dir}")
        else:
            data_dir = "dummy_data"
            print("Setting up dummy data for inference test...")
            setup_dummy_data(data_dir, for_vae=False)

        test_loader = data_pipeline.get_dataloader(data_dir, for_vae=False)

        if len(test_loader) == 0:
            raise ValueError("No test data found.")

        test_batch = next(iter(test_loader))

        # Handle batch structure
        if isinstance(test_batch, list):
            sample_qpi_patch = test_batch[0]["qpi"]
        else:
            sample_qpi_patch = test_batch["qpi"][0:1]  # Take first sample

        print(f"Loaded sample QPI patch with shape: {sample_qpi_patch.shape}")

        generated_patch = run_inference(sample_qpi_patch)

        print(f"Generated virtual stain patch shape: {generated_patch.shape}")

        output_filename = "virtual_stain_output_v2.tif"
        generated_patch = generated_patch.squeeze()
        tifffile.imwrite(output_filename, generated_patch)
        print(f"Saved generated patch to {output_filename}")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Could not load models. Please run train_vae.py and train_llcm_v2.py first.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInference interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred during inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)