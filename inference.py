import torch
import config
import vae_model
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
    Runs the full 1-step inference pipeline on a single QPI patch.
    """
    print(f"--- Starting Phase 3: 1-Step Inference ---")
    device = config.DEVICE
    
    try:
        print("Loading all models...")
        vae = vae_model.get_vae_model().to(device)
        try:
            vae.load_state_dict(torch.load(config.VAE_MODEL_PATH, map_location=device))
        except FileNotFoundError:
            raise FileNotFoundError(f"VAE model not found at {config.VAE_MODEL_PATH}. Please run train_vae.py first.")
        except Exception as e:
            raise RuntimeError(f"Error loading VAE model: {e}")
        vae.eval()
        
        qpi_encoder = llcm_model.QPIEncoder().to(device)
        try:
            qpi_encoder.load_state_dict(torch.load(config.QPI_ENCODER_PATH, map_location=device))
        except FileNotFoundError:
            raise FileNotFoundError(f"QPI encoder not found at {config.QPI_ENCODER_PATH}. Please run train_llcm.py first.")
        except Exception as e:
            raise RuntimeError(f"Error loading QPI encoder: {e}")
        qpi_encoder.eval()
        
        llcm_ema = llcm_model.get_llcm_unet().to(device)
        try:
            llcm_ema.load_state_dict(torch.load(config.LLCM_EMA_PATH, map_location=device), strict=False)
        except FileNotFoundError:
            raise FileNotFoundError(f"LLCM model not found at {config.LLCM_EMA_PATH}. Please run train_llcm.py first.")
        except Exception as e:
            raise RuntimeError(f"Error loading LLCM model: {e}")
        llcm_ema.eval()
        
        scheduler = LMSDiscreteScheduler(
            beta_start=0.00085, 
            beta_end=0.012, 
            beta_schedule="scaled_linear", 
            num_train_timesteps=config.NUM_TRAIN_TIMESTEPS
        )
        scheduler.set_timesteps(config.NUM_TRAIN_TIMESTEPS)
        
        qpi_patch = qpi_patch.to(device)
        if qpi_patch.dim() == 4:
            qpi_patch = qpi_patch.unsqueeze(0)
        
        if qpi_patch.dim() != 5:
            raise ValueError(f"Expected 5D tensor (B, C, D, H, W), got {qpi_patch.dim()}D tensor")
        
        downsample_factor = 2 ** len(config.VAE_STRIDES)
        latent_shape = (
            qpi_patch.shape[0],
            config.LATENT_CHANNELS,
            config.ROI_SIZE[0] // downsample_factor,
            config.ROI_SIZE[1] // downsample_factor,
            config.ROI_SIZE[2] // downsample_factor
        )

        print("Running 1-step generation...")

        # Encode QPI context
        qpi_context = qpi_encoder(qpi_patch)  # (batch, QPI_CONTEXT_DIM)
        # Reshape context to match latent spatial dimensions and add as channel
        # Get latent spatial shape
        latent_spatial = (
            config.ROI_SIZE[0] // downsample_factor,
            config.ROI_SIZE[1] // downsample_factor,
            config.ROI_SIZE[2] // downsample_factor
        )
        # Broadcast context to spatial dimensions: (batch, 1, D, H, W)
        qpi_context_spatial = qpi_context[:, 0:1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # Take first dim, expand
        qpi_context_spatial = qpi_context_spatial.expand(-1, -1, *latent_spatial)

        initial_noise = torch.randn(latent_shape).to(device)

        # Use inference timestep
        timestep = torch.tensor([config.INFERENCE_TIMESTEP], device=device, dtype=torch.long)

        # Concatenate context as additional channel (matching training)
        noise_with_context = torch.cat([initial_noise, qpi_context_spatial], dim=1)

        # UNet forward - MONAI UNet only takes input, doesn't use timestep
        predicted_latent = llcm_ema(noise_with_context)
        
        # Decode using VAE helper
        virtual_stain_patch = vae_model.decode_vae(vae, predicted_latent)
        
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
        
        output_filename = "virtual_stain_output.tif"
        generated_patch = generated_patch.squeeze() 
        tifffile.imwrite(output_filename, generated_patch)
        print(f"Saved generated patch to {output_filename}")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Could not load models. Please run train_vae.py and train_llcm.py first.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInference interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred during inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)