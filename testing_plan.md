Project LLCM: Testing & Validation Plan

This document outlines the testing strategy for each phase of the 3-Phase implementation, aligned with the "Validation & Success Metrics" from the LCM Plan.pdf.

Phase 1: VAE ("Compressor") Testing

Objective: Ensure the 3D VAE can compress and reconstruct DAPI patches with high fidelity. This is critical, as a "blurry VAE" caps the maximum quality of the final model.

Success Criteria:

Quantitative: Training losses converge.

L1 Loss (Reconstruction) should steadily decrease and plateau.

KL Divergence should remain stable and small (e.g., < 0.01) after initial warmup.

LPIPS Loss (Perceptual) should steadily decrease, indicating improved visual sharpness.

Qualitative: Visual inspection of (input_dapi, reconstructed_dapi) pairs shows minimal-to-no blurriness. Nuclei structures should be sharp and clear.

Test Procedure:

Run the train_vae.py script.

Monitor Logs: Use tensorboard (or simple print logs) to track L1, KL, and LPIPS losses.

Visual Check: Add a callback to the training script to periodically save a validation patch and its reconstruction (e.g., epoch_10_input.tif, epoch_10_recon.tif).

Inspect: Open the saved patches in Fiji/ImageJ and compare.

Risk & Mitigation:

Risk: "Blurry VAE."

Mitigation: This is why the LPIPS (Perceptual Loss) is included. If results are blurry, increase config.PERCEPTUAL_WEIGHT and/or config.L1_WEIGHT relative to config.KL_WEIGHT.

Phase 2: LLCM ("Translator") Testing

Objective: Ensure the conditional U-Net (LLCM) successfully trains to translate a QPI context and noisy latent into a clean DAPI latent.

Success Criteria:

Quantitative: The MSE Loss (student vs. teacher) steadily decreases and converges.

Process: The training script completes without errors, and the final llcm_ema.pth and qpi_encoder.pth models are saved.

Test Procedure:

Ensure vae.pth from Phase 1 exists.

Run the train_llcm.py script.

Monitor Logs: Track the MSE Loss. A stable, decreasing loss indicates the student model is successfully learning to emulate the teacher model (i.e., the consistency objective is working).

Verify Checkpoints: Confirm that llcm_ema.pth and qpi_encoder.pth are created at the end of training.

Phase 3: End-to-End Inference & Validation

Objective: Validate that the 1-step generator (inference.py) produces high-fidelity virtual DAPI stains and, most importantly, solves the "missing nuclei" problem.

Test Procedure:

Use a held-out test set of QPI volumes (not used in training).

Run inference.py (or a modified batch-processing script) to generate virtual_dapi for the entire test set.

Compare the virtual_dapi outputs against the ground_truth_dapi inputs.

Success Metrics:

1. Baseline Pixel Metrics (vs. cGAN)

PSNR (Peak Signal-to-Noise Ratio): Must match or exceed the cGAN baseline.

SSIM (Structural Similarity Index): Must match or exceed the cGAN baseline.

LPIPS (Perceptual Similarity): Must be significantly lower than the cGAN baseline. A lower LPIPS score means the image is perceptually more realistic and less blurry.

2. Primary Diagnostic Metric (Nuclei-F1 Score)

This is the primary goal and directly measures the "missing nuclei" failure mode.

Procedure:

Ground Truth Segmentation: Use a standard, pre-trained segmentation algorithm (e.g., Cellpose) on the ground_truth_dapi test volumes. This gives the "ground truth" nuclei count and masks.

Prediction Segmentation: Run the exact same Cellpose model (with the same parameters) on the virtual_dapi outputs from our model.

Compare: Using the segmentation masks, calculate the object-level Precision, Recall, and F1-Score.

Final Success:

The cGAN will have a low Recall (it "misses nuclei").

Our LLCM must achieve a high Recall and F1-Score, demonstrating that it is reliably detecting and generating nuclei structures present in the QPI input.

Inference speed must be real-time (1-step per patch), matching the cGAN's performance.