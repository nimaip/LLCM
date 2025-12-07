"""
train_llcm_v2.py

Corrected LLCM training implementation based on the paper.

Key fixes:
1. Proper timestep conditioning via sinusoidal embeddings
2. Full QPI context (256D) via FiLM conditioning
3. Separate teacher model with EMA weights
4. Consistency training: predict x0 from different noise levels
"""

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from diffusers.schedulers import LMSDiscreteScheduler
from tqdm import tqdm
import config
import data_pipeline
import vae_model
from unet_wrapper import get_consistency_unet
import llcm_model  # For QPIEncoder
from train_vae import setup_dummy_data
import sys
import copy


class EMAModel:
    """Exponential Moving Average for model parameters."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.model = model
        # Create a copy of the model for EMA
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad = False

    def update(self):
        """Update EMA parameters."""
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)

    def get_model(self):
        """Get the EMA model."""
        return self.ema_model


def train_llcm():
    print(f"--- Starting Phase 2: LLCM Training (Corrected) ---")
    print(f"Using device: {config.DEVICE}")

    try:
        # --- 1. Setup ---
        device = config.DEVICE

        # Load data
        import os
        if os.path.exists("3d_data") and os.path.exists(os.path.join("3d_data", "qpi")):
            data_dir = "3d_data"
            print(f"Using real data from: {data_dir}")
        else:
            data_dir = "dummy_data"
            setup_dummy_data(data_dir, for_vae=False)
            print(f"Using dummy data from: {data_dir}")

        print(f"Loading QPI & DAPI data from: {data_dir}")
        train_loader = data_pipeline.get_dataloader(data_dir, for_vae=False)

        if len(train_loader) == 0:
            raise ValueError("No training data found. Check data directory.")

        # Load frozen VAE
        print(f"Loading frozen VAE from: {config.VAE_MODEL_PATH}")
        vae = vae_model.get_vae_model().to(device)
        try:
            vae.load_state_dict(torch.load(config.VAE_MODEL_PATH, map_location=device))
        except FileNotFoundError:
            raise FileNotFoundError(f"{config.VAE_MODEL_PATH} not found. Please run train_vae.py first.")
        vae.eval()

        # Initialize models
        print("Initializing LLCM models (Student U-Net, Teacher/EMA, QPI Encoder)")
        qpi_encoder = llcm_model.QPIEncoder().to(device)
        student_unet = get_consistency_unet().to(device)

        # Create EMA model
        ema = EMAModel(student_unet, decay=config.EMA_DECAY)
        teacher_unet = ema.get_model()

        # Optimizer
        optimizer = torch.optim.Adam(
            list(student_unet.parameters()) + list(qpi_encoder.parameters()),
            lr=config.LLCM_LR
        )

        # Scheduler for noise scheduling
        scheduler = LMSDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=config.NUM_TRAIN_TIMESTEPS
        )
        scheduler.set_timesteps(config.NUM_TRAIN_TIMESTEPS)

        scaler = GradScaler()

        print("Starting training...")
        for epoch in range(config.LLCM_EPOCHS):
            student_unet.train()
            qpi_encoder.train()
            epoch_loss = 0

            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                              desc=f"Epoch {epoch+1}/{config.LLCM_EPOCHS}")

            for step, batch in progress_bar:
                try:
                    # Handle batch structure
                    if isinstance(batch, list):
                        qpi_patches = torch.cat([b["qpi"] for b in batch], dim=0).to(device)
                        dapi_patches = torch.cat([b["dapi"] for b in batch], dim=0).to(device)
                    else:
                        qpi_patches = batch["qpi"].to(device)
                        dapi_patches = batch["dapi"].to(device)

                    if qpi_patches.shape[0] == 0 or dapi_patches.shape[0] == 0:
                        continue

                    optimizer.zero_grad()

                    with autocast():
                        with torch.no_grad():
                            # Encode DAPI to latent space (clean data x0)
                            dapi_latent = vae_model.encode_vae(vae, dapi_patches)

                        # Encode QPI context (full 256D vector)
                        qpi_context = qpi_encoder(qpi_patches)  # (batch, 256)

                        # Sample timesteps
                        # For consistency training: sample pairs of adjacent timesteps
                        batch_size = dapi_latent.shape[0]
                        max_t_idx = len(scheduler.timesteps) - 2
                        t_indices = torch.randint(0, max_t_idx, (batch_size,), device=device)

                        # Get timestep values (not indices)
                        t = scheduler.timesteps[t_indices]
                        t_prime = scheduler.timesteps[t_indices + 1]

                        # Compute noise scaling factors
                        betas = scheduler.betas.to(device)
                        alphas = 1.0 - betas
                        alphas_cumprod = torch.cumprod(alphas, dim=0)

                        # Map timestep values to indices in alphas_cumprod
                        timestep_values = scheduler.timesteps.cpu().numpy()
                        timestep_to_idx = {int(tv): config.NUM_TRAIN_TIMESTEPS - 1 - i
                                         for i, tv in enumerate(timestep_values)}

                        t_idx = torch.tensor([timestep_to_idx.get(int(ts.item()), 0) for ts in t], device=device)
                        t_prime_idx = torch.tensor([timestep_to_idx.get(int(ts.item()), 0) for ts in t_prime], device=device)

                        t_idx = torch.clamp(t_idx, 0, len(alphas_cumprod) - 1)
                        t_prime_idx = torch.clamp(t_prime_idx, 0, len(alphas_cumprod) - 1)

                        alpha_prod_t = alphas_cumprod[t_idx].view(-1, 1, 1, 1, 1)
                        alpha_prod_t_prime = alphas_cumprod[t_prime_idx].view(-1, 1, 1, 1, 1)

                        # Add noise to clean latent
                        noise = torch.randn_like(dapi_latent)
                        noisy_t = torch.sqrt(alpha_prod_t) * dapi_latent + torch.sqrt(1 - alpha_prod_t) * noise
                        noisy_t_prime = torch.sqrt(alpha_prod_t_prime) * dapi_latent + torch.sqrt(1 - alpha_prod_t_prime) * noise

                        # Student prediction from higher noise level (t)
                        student_output = student_unet(noisy_t, t, qpi_context)

                        # Teacher prediction from lower noise level (t_prime)
                        with torch.no_grad():
                            teacher_output = teacher_unet(noisy_t_prime, t_prime, qpi_context)

                        # Consistency loss: both should predict the same clean latent
                        loss = F.mse_loss(student_output, teacher_output)

                    scaler.scale(loss).backward()

                    # Gradient clipping for stability
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(student_unet.parameters()) + list(qpi_encoder.parameters()),
                        max_norm=1.0
                    )

                    scaler.step(optimizer)
                    scaler.update()

                    # Update EMA
                    ema.update()

                    epoch_loss += loss.item()
                    progress_bar.set_postfix({"MSE Loss": f"{loss.item():.4f}"})

                except Exception as e:
                    print(f"Error in training step {step}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            avg_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else 0
            print(f"Epoch {epoch+1} Average MSE Loss: {avg_loss:.4f}")

            # Save models
            if (epoch + 1) % 10 == 0 or (epoch + 1) == config.LLCM_EPOCHS:
                torch.save(teacher_unet.state_dict(), config.LLCM_EMA_PATH)
                torch.save(qpi_encoder.state_dict(), config.QPI_ENCODER_PATH)
                if (epoch + 1) == config.LLCM_EPOCHS:
                    print(f"Phase 2 complete. Final models saved:")
                else:
                    print(f"Checkpoint saved at epoch {epoch+1}:")
                print(f"  - LLCM EMA: {config.LLCM_EMA_PATH}")
                print(f"  - QPI Encoder: {config.QPI_ENCODER_PATH}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error in LLCM training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    train_llcm()