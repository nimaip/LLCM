"""
evaluate.py

Evaluation script to calculate accuracy metrics for generated images.
Computes PSNR, SSIM, and LPIPS scores comparing generated DAPI to ground truth.
"""

import torch
import numpy as np
import config
import vae_model
import llcm_model
from diffusers.schedulers import LMSDiscreteScheduler
import data_pipeline
import inference
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
import sys
from tqdm import tqdm


def calculate_psnr(img1, img2, data_range=2.0):
    """Calculate PSNR between two images."""
    # Ensure images are in range [-1, 1] or [0, 1]
    img1 = np.clip(img1, -1, 1)
    img2 = np.clip(img2, -1, 1)
    # Convert to [0, 1] for PSNR calculation
    img1_norm = (img1 + 1) / 2.0
    img2_norm = (img2 + 1) / 2.0
    return peak_signal_noise_ratio(img1_norm, img2_norm, data_range=1.0)


def calculate_ssim(img1, img2):
    """Calculate SSIM between two images."""
    img1 = np.clip(img1, -1, 1)
    img2 = np.clip(img2, -1, 1)
    # Convert to [0, 1] for SSIM calculation
    img1_norm = (img1 + 1) / 2.0
    img2_norm = (img2 + 1) / 2.0
    
    # SSIM expects 2D or 3D images, handle 5D tensors by taking middle slice
    if img1_norm.ndim == 5:
        # Take middle Z-slice
        mid_z = img1_norm.shape[2] // 2
        img1_norm = img1_norm[0, 0, mid_z, :, :]
        img2_norm = img2_norm[0, 0, mid_z, :, :]
    elif img1_norm.ndim == 4:
        img1_norm = img1_norm[0, 0, :, :]
        img2_norm = img2_norm[0, 0, :, :]
    
    return structural_similarity(img1_norm, img2_norm, data_range=1.0)


def calculate_lpips(img1, img2, lpips_fn):
    """Calculate LPIPS (perceptual similarity) between two images."""
    device = config.DEVICE
    
    # Convert to torch tensors
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1).float()
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2).float()
    
    # Ensure images are in range [-1, 1]
    img1 = torch.clamp(img1, -1, 1)
    img2 = torch.clamp(img2, -1, 1)
    
    # LPIPS expects RGB images, so we need to handle grayscale
    # Take middle Z-slice and convert to RGB
    if img1.dim() == 5:
        mid_z = img1.shape[2] // 2
        img1_slice = img1[0, 0, mid_z, :, :].unsqueeze(0)  # (1, H, W)
        img2_slice = img2[0, 0, mid_z, :, :].unsqueeze(0)
    elif img1.dim() == 4:
        img1_slice = img1[0, 0, :, :].unsqueeze(0)
        img2_slice = img2[0, 0, :, :].unsqueeze(0)
    else:
        img1_slice = img1.unsqueeze(0)
        img2_slice = img2.unsqueeze(0)
    
    # Convert to RGB by repeating the channel
    img1_rgb = img1_slice.repeat(3, 1, 1).unsqueeze(0).to(device)  # (1, 3, H, W)
    img2_rgb = img2_slice.repeat(3, 1, 1).unsqueeze(0).to(device)
    
    # Normalize to [-1, 1] for LPIPS (already done, but ensure)
    img1_rgb = torch.clamp(img1_rgb, -1, 1)
    img2_rgb = torch.clamp(img2_rgb, -1, 1)
    
    with torch.no_grad():
        lpips_score = lpips_fn(img1_rgb, img2_rgb)
    
    return lpips_score.item()


def evaluate_model(test_data_dir="3d_data", num_samples=5):
    """
    Evaluate the trained model on test data.
    
    Args:
        test_data_dir: Directory containing test data
        num_samples: Number of samples to evaluate
    """
    print(f"\n{'='*60}")
    print(f"--- Phase 4: Model Evaluation ---")
    print(f"{'='*60}\n")
    
    device = config.DEVICE
    
    # Load test data
    print(f"Loading test data from: {test_data_dir}")
    try:
        test_loader = data_pipeline.get_dataloader(test_data_dir, for_vae=False)
    except Exception as e:
        print(f"Error loading test data: {e}")
        print("Falling back to dummy data...")
        from train_vae import setup_dummy_data
        setup_dummy_data("dummy_data", for_vae=False)
        test_loader = data_pipeline.get_dataloader("dummy_data", for_vae=False)
    
    if len(test_loader) == 0:
        raise ValueError("No test data found.")
    
    # Initialize LPIPS
    try:
        lpips_fn = lpips.LPIPS(net='alex', spatial=False).to(device)
        print("LPIPS metric loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load LPIPS. {e}")
        lpips_fn = None
    
    # Collect metrics
    psnr_scores = []
    ssim_scores = []
    lpips_scores = []
    
    print(f"\nEvaluating on {num_samples} samples...")
    
    sample_count = 0
    for batch_idx, batch in enumerate(test_loader):
        if sample_count >= num_samples:
            break
        
        try:
            # Handle batch structure
            if isinstance(batch, list):
                qpi_patches = torch.cat([b["qpi"] for b in batch], dim=0)
                dapi_patches = torch.cat([b["dapi"] for b in batch], dim=0)
            else:
                qpi_patches = batch["qpi"]
                dapi_patches = batch["dapi"]
            
            # Process each sample in the batch
            for i in range(min(qpi_patches.shape[0], num_samples - sample_count)):
                qpi_patch = qpi_patches[i:i+1].to(device)
                dapi_gt = dapi_patches[i:i+1].to(device)
                
                # Generate prediction
                print(f"\nProcessing sample {sample_count + 1}/{num_samples}...")
                generated_dapi = inference.run_inference(qpi_patch)
                
                # Convert to numpy if needed
                if isinstance(generated_dapi, torch.Tensor):
                    generated_dapi = generated_dapi.cpu().numpy()
                if isinstance(dapi_gt, torch.Tensor):
                    dapi_gt = dapi_gt.cpu().numpy()
                
                # Calculate metrics
                psnr = calculate_psnr(dapi_gt, generated_dapi)
                ssim = calculate_ssim(dapi_gt, generated_dapi)
                
                psnr_scores.append(psnr)
                ssim_scores.append(ssim)
                
                if lpips_fn:
                    lpips_score = calculate_lpips(dapi_gt, generated_dapi, lpips_fn)
                    lpips_scores.append(lpips_score)
                
                print(f"  PSNR: {psnr:.4f} dB")
                print(f"  SSIM: {ssim:.4f}")
                if lpips_fn:
                    print(f"  LPIPS: {lpips_score:.4f}")
                
                sample_count += 1
                
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Evaluation Summary ({sample_count} samples)")
    print(f"{'='*60}")
    print(f"PSNR (Peak Signal-to-Noise Ratio):")
    print(f"  Mean: {np.mean(psnr_scores):.4f} dB")
    print(f"  Std:  {np.std(psnr_scores):.4f} dB")
    print(f"  Min:  {np.min(psnr_scores):.4f} dB")
    print(f"  Max:  {np.max(psnr_scores):.4f} dB")
    
    print(f"\nSSIM (Structural Similarity Index):")
    print(f"  Mean: {np.mean(ssim_scores):.4f}")
    print(f"  Std:  {np.std(ssim_scores):.4f}")
    print(f"  Min:  {np.min(ssim_scores):.4f}")
    print(f"  Max:  {np.max(ssim_scores):.4f}")
    
    if lpips_scores:
        print(f"\nLPIPS (Perceptual Similarity - lower is better):")
        print(f"  Mean: {np.mean(lpips_scores):.4f}")
        print(f"  Std:  {np.std(lpips_scores):.4f}")
        print(f"  Min:  {np.min(lpips_scores):.4f}")
        print(f"  Max:  {np.max(lpips_scores):.4f}")
    
    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"{'='*60}\n")
    
    return {
        'psnr': {'mean': np.mean(psnr_scores), 'std': np.std(psnr_scores)},
        'ssim': {'mean': np.mean(ssim_scores), 'std': np.std(ssim_scores)},
        'lpips': {'mean': np.mean(lpips_scores), 'std': np.std(lpips_scores)} if lpips_scores else None
    }


if __name__ == "__main__":
    try:
        metrics = evaluate_model(test_data_dir="3d_data", num_samples=5)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Could not load models. Please run train_vae.py and train_llcm.py first.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

