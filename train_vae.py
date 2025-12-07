import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import lpips 
import config
import data_pipeline
import vae_model
from tqdm import tqdm
import os
import sys

def train_vae():
    print(f"--- Starting Phase 1: VAE Training ---")
    print(f"Using device: {config.DEVICE}")
    
    try:
        device = config.DEVICE
        # Use 3d_data if it exists, otherwise fall back to dummy_data
        import os
        if os.path.exists("3d_data") and os.path.exists(os.path.join("3d_data", "dapi")):
            data_dir = "3d_data"
            print(f"Using real data from: {data_dir}")
        else:
            data_dir = "dummy_data"
            setup_dummy_data(data_dir, for_vae=True)
            print(f"Using dummy data from: {data_dir}") 
        print(f"Loading DAPI data from: {data_dir}")
        train_loader = data_pipeline.get_dataloader(data_dir, for_vae=True)
        
        if len(train_loader) == 0:
            raise ValueError("No training data found. Check data directory.")
        
        model = vae_model.get_vae_model().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.VAE_LR)
        
        try:
            perceptual_loss_fn = lpips.LPIPS(net='alex', spatial=True).to(device)
            print("LPIPS (Perceptual Loss) loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load LPIPS. {e}. Training without perceptual loss.")
            perceptual_loss_fn = None
            
        scaler = GradScaler()
        
        print("Starting training...")
        for epoch in range(config.VAE_EPOCHS):
            model.train()
            epoch_loss = 0
            epoch_l1_loss = 0
            epoch_kl_loss = 0
            epoch_p_loss = 0
            
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{config.VAE_EPOCHS}")
            
            for step, batch in progress_bar:
                try:
                    # Handle batch structure - MONAI DataLoader returns list of dicts
                    if isinstance(batch, list):
                        dapi_patches = torch.cat([b["dapi"] for b in batch], dim=0).to(device)
                    else:
                        dapi_patches = batch["dapi"].to(device)
                    
                    if dapi_patches.shape[0] == 0:
                        continue
                    
                    optimizer.zero_grad()
                    
                    with autocast():
                        # MONAI VarAutoEncoder returns (reconstruction, mu, logvar, z)
                        # where z is the sampled latent (we don't need it for training)
                        result = model(dapi_patches)
                        if len(result) == 4:
                            reconstructed_dapi, mu, logvar, _ = result
                        elif len(result) == 3:
                            reconstructed_dapi, mu, logvar = result
                        else:
                            raise ValueError(f"Unexpected number of return values: {len(result)}")
                        
                        l1_loss = F.l1_loss(reconstructed_dapi, dapi_patches)
                        kl_loss = vae_model.compute_kl_loss(mu, logvar).mean()
                        
                        total_loss = config.L1_WEIGHT * l1_loss + config.KL_WEIGHT * kl_loss
                        
                        p_loss = torch.tensor(0.0, device=device)
                        if perceptual_loss_fn:
                            try:
                                mid_slice = dapi_patches.shape[2] // 2
                                rec_slice = reconstructed_dapi[:, :, mid_slice, :, :].repeat(1, 3, 1, 1)
                                dapi_slice = dapi_patches[:, :, mid_slice, :, :].repeat(1, 3, 1, 1)
                                p_loss = perceptual_loss_fn(rec_slice, dapi_slice).mean()
                                total_loss += config.PERCEPTUAL_WEIGHT * p_loss
                            except Exception as e:
                                print(f"Warning: Perceptual loss computation failed: {e}")

                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    epoch_loss += total_loss.item()
                    epoch_l1_loss += l1_loss.item()
                    epoch_kl_loss += kl_loss.item()
                    epoch_p_loss += p_loss.item()
                    
                    progress_bar.set_postfix({
                        "Total Loss": f"{total_loss.item():.4f}",
                        "L1": f"{l1_loss.item():.4f}",
                        "KL": f"{kl_loss.item():.4f}",
                        "LPIPS": f"{p_loss.item():.4f}"
                    })
                except Exception as e:
                    print(f"Error in training step {step}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                    
            avg_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else 0
            print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f} [L1: {epoch_l1_loss/len(train_loader):.4f}, KL: {epoch_kl_loss/len(train_loader):.4f}, LPIPS: {epoch_p_loss/len(train_loader):.4f}]")

        torch.save(model.state_dict(), config.VAE_MODEL_PATH)
        print(f"Phase 1 complete. VAE model saved to {config.VAE_MODEL_PATH}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error in VAE training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def setup_dummy_data(data_dir, for_vae=False):
    """Creates dummy .tif files for testing the pipeline."""
    import numpy as np
    import tifffile
    
    qpi_dir = os.path.join(data_dir, "qpi")
    dapi_dir = os.path.join(data_dir, "dapi")
    os.makedirs(qpi_dir, exist_ok=True)
    os.makedirs(dapi_dir, exist_ok=True)
    
    for i in range(1, 3):
        fname = f"volume_{i:03d}.tif"
        dapi_path = os.path.join(dapi_dir, fname)
        qpi_path = os.path.join(qpi_dir, fname)
        
        vol_shape = (config.ROI_SIZE[0] + 10, config.ROI_SIZE[1] + 10, config.ROI_SIZE[2] + 10)
        
        if not os.path.exists(dapi_path):
            print(f"Creating dummy DAPI: {dapi_path}")
            dummy_dapi = np.random.rand(*vol_shape).astype(np.float32)
            tifffile.imwrite(dapi_path, dummy_dapi)
            
        if not for_vae and not os.path.exists(qpi_path):
            print(f"Creating dummy QPI: {qpi_path}")
            dummy_qpi = np.random.rand(*vol_shape).astype(np.float32)
            tifffile.imwrite(qpi_path, dummy_qpi)

if __name__ == "__main__":
    train_vae()