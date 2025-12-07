import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from diffusers.schedulers import LMSDiscreteScheduler
# EMA implementation
class SimpleEMA:
    """Simple EMA implementation for model parameters."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.model = model
        self.shadow = {}
        self.backup = {}
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def step(self):
        """Update EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self):
        """Apply shadow parameters to model."""
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self):
        """Return state dict with EMA weights."""
        return {name: self.shadow[name].clone() for name in self.shadow}
from tqdm import tqdm
import config
import data_pipeline
import vae_model
import llcm_model
from train_vae import setup_dummy_data
import sys
import numpy as np 

def train_llcm():
    print(f"--- Starting Phase 2: LLCM Training ---")
    print(f"Using device: {config.DEVICE}")
    
    try:
        # --- 1. Setup ---
        device = config.DEVICE
        # Use 3d_data if it exists, otherwise fall back to dummy_data
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
        
        print(f"Loading frozen VAE from: {config.VAE_MODEL_PATH}")
        vae = vae_model.get_vae_model().to(device)
        try:
            vae.load_state_dict(torch.load(config.VAE_MODEL_PATH, map_location=device))
        except FileNotFoundError:
            raise FileNotFoundError(f"{config.VAE_MODEL_PATH} not found. Please run train_vae.py first.")
        except Exception as e:
            raise RuntimeError(f"Error loading VAE model: {e}")
        vae.eval() 
        
        print("Initializing LLCM models (Student, Teacher/EMA, QPI Encoder)")
        qpi_encoder = llcm_model.QPIEncoder().to(device)
        llcm_unet = llcm_model.get_llcm_unet().to(device)
        
        ema_unet = SimpleEMA(llcm_unet, decay=config.EMA_DECAY)
        
        optimizer = torch.optim.Adam(
            list(llcm_unet.parameters()) + list(qpi_encoder.parameters()), 
            lr=config.LLCM_LR
        )
        
        scheduler = LMSDiscreteScheduler(
            beta_start=0.00085, 
            beta_end=0.012, 
            beta_schedule="scaled_linear", 
            num_train_timesteps=config.NUM_TRAIN_TIMESTEPS
        )
        # Set timesteps for scheduler
        scheduler.set_timesteps(config.NUM_TRAIN_TIMESTEPS)
        
        scaler = GradScaler()
        
        print("Starting training...")
        for epoch in range(config.LLCM_EPOCHS):
            llcm_unet.train()
            qpi_encoder.train()
            epoch_loss = 0
            
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{config.LLCM_EPOCHS}")
            
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
                            # Use VAE encoding helper
                            dapi_latent = vae_model.encode_vae(vae, dapi_patches)
                        
                        # Encode QPI context
                        qpi_context = qpi_encoder(qpi_patches)  # (batch, QPI_CONTEXT_DIM)
                        # Reshape context to match latent spatial dimensions and add as channel
                        # Get latent spatial shape
                        latent_spatial = dapi_latent.shape[2:]  # (D, H, W)
                        # Broadcast context to spatial dimensions: (batch, 1, D, H, W)
                        qpi_context_spatial = qpi_context[:, 0:1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # Take first dim, expand
                        qpi_context_spatial = qpi_context_spatial.expand(-1, -1, *latent_spatial)
                        
                        # Sample timesteps
                        max_t_idx = len(scheduler.timesteps) - 2  # Leave room for t_prime
                        t_indices = torch.randint(0, max_t_idx, (dapi_latent.shape[0],), device=device)
                        t = scheduler.timesteps[t_indices]
                        t_prime = scheduler.timesteps[t_indices + 1]
                        
                        # Add noise manually for consistency training
                        # For LMSDiscreteScheduler, we need to compute noise scaling from betas
                        # Get betas and compute alphas_cumprod
                        betas = scheduler.betas
                        alphas = 1.0 - betas
                        alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)
                        
                        # Map timestep values to indices in alphas_cumprod
                        # scheduler.timesteps contains actual timestep values (e.g., 999, 998, ...)
                        # We need to map these to indices in the original schedule (0 to NUM_TRAIN_TIMESTEPS-1)
                        # Since timesteps are in descending order, reverse the mapping
                        timestep_values = scheduler.timesteps.cpu().numpy()
                        # Create reverse mapping: timestep value -> index in alphas_cumprod
                        # timestep_values[0] is the largest (e.g., 999), maps to index NUM_TRAIN_TIMESTEPS-1
                        timestep_to_idx = {int(tv): config.NUM_TRAIN_TIMESTEPS - 1 - i 
                                          for i, tv in enumerate(timestep_values)}
                        
                        # Get indices for current timesteps
                        t_idx = torch.tensor([timestep_to_idx.get(int(ts.item()), 0) for ts in t], device=device)
                        t_prime_idx = torch.tensor([timestep_to_idx.get(int(ts.item()), 0) for ts in t_prime], device=device)
                        
                        # Clamp indices to valid range
                        t_idx = torch.clamp(t_idx, 0, len(alphas_cumprod) - 1)
                        t_prime_idx = torch.clamp(t_prime_idx, 0, len(alphas_cumprod) - 1)
                        
                        alpha_prod_t = alphas_cumprod[t_idx].view(-1, 1, 1, 1, 1)
                        alpha_prod_t_prime = alphas_cumprod[t_prime_idx].view(-1, 1, 1, 1, 1)
                        
                        noise = torch.randn_like(dapi_latent)
                        noisy_t = torch.sqrt(alpha_prod_t) * dapi_latent + torch.sqrt(1 - alpha_prod_t) * noise
                        noisy_t_prime = torch.sqrt(alpha_prod_t_prime) * dapi_latent + torch.sqrt(1 - alpha_prod_t_prime) * noise
                        
                        # Concatenate context as additional channel
                        noisy_t_with_context = torch.cat([noisy_t, qpi_context_spatial], dim=1)
                        noisy_t_prime_with_context = torch.cat([noisy_t_prime, qpi_context_spatial], dim=1)
                        
                        # UNet forward - MONAI UNet only takes input (no timestep support)
                        student_output = llcm_unet(noisy_t_with_context)

                        with torch.no_grad():
                            # Use EMA model for teacher
                            ema_unet.apply_shadow()
                            teacher_output = llcm_unet(noisy_t_prime_with_context)
                            ema_unet.restore()
                        
                        loss = F.mse_loss(student_output, teacher_output)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    ema_unet.step()
                    
                    epoch_loss += loss.item()
                    progress_bar.set_postfix({"MSE Loss": f"{loss.item():.4f}"})
                except Exception as e:
                    print(f"Error in training step {step}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            avg_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else 0
            print(f"Epoch {epoch+1} Average MSE Loss: {avg_loss:.4f}")

            # Save models periodically (every 10 epochs) and at the end
            if (epoch + 1) % 10 == 0 or (epoch + 1) == config.LLCM_EPOCHS:
                # Save EMA model
                torch.save(ema_unet.state_dict(), config.LLCM_EMA_PATH)
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