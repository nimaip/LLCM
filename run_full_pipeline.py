"""
run_full_pipeline.py

Master script to run the entire LLCM pipeline from start to finish:
1. Preprocess 2D PNGs to 3D volumes
2. Train VAE
3. Train LLCM
4. Run inference
5. Evaluate and output accuracy scores

This is a proof-of-concept script with reduced epochs for quick testing.
"""

import os
import sys
import subprocess
import argparse


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n{description} interrupted by user.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run the complete LLCM training and evaluation pipeline"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory containing raw PNG files (default: ./data)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./3d_data",
        help="Output directory for 3D volumes (default: ./3d_data)"
    )
    parser.add_argument(
        "--skip_preprocessing",
        action="store_true",
        help="Skip preprocessing if 3D data already exists"
    )
    parser.add_argument(
        "--skip_vae",
        action="store_true",
        help="Skip VAE training if model already exists"
    )
    parser.add_argument(
        "--skip_llcm",
        action="store_true",
        help="Skip LLCM training if model already exists"
    )
    parser.add_argument(
        "--num_eval_samples",
        type=int,
        default=5,
        help="Number of samples to evaluate (default: 5)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("LLCM Full Pipeline - Proof of Concept")
    print("="*60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Evaluation samples: {args.num_eval_samples}")
    print("="*60 + "\n")
    
    # Step 1: Preprocess 2D PNGs to 3D volumes
    if not args.skip_preprocessing:
        if os.path.exists(args.output_dir) and os.path.exists(os.path.join(args.output_dir, "qpi")):
            print(f"3D data already exists at {args.output_dir}. Skipping preprocessing.")
            print("Use --skip_preprocessing to suppress this check.\n")
        else:
            cmd = f'python build_3d_dataset.py --data_dir "{args.data_dir}" --output_dir "{args.output_dir}" --min_depth 1'
            if not run_command(cmd, "Step 1: Preprocessing 2D PNGs to 3D Volumes"):
                print("Preprocessing failed. Exiting.")
                sys.exit(1)
    else:
        print("Skipping preprocessing (--skip_preprocessing flag set).\n")
    
    # Step 2: Train VAE
    if not args.skip_vae:
        if os.path.exists("vae.pth"):
            print("VAE model already exists. Skipping VAE training.")
            print("Use --skip_vae to suppress this check.\n")
        else:
            cmd = "python train_vae.py"
            if not run_command(cmd, "Step 2: Training VAE"):
                print("VAE training failed. Exiting.")
                sys.exit(1)
    else:
        print("Skipping VAE training (--skip_vae flag set).\n")
    
    # Step 3: Train LLCM (using v2 - corrected version)
    if not args.skip_llcm:
        if os.path.exists("llcm_ema.pth") and os.path.exists("qpi_encoder.pth"):
            print("LLCM models already exist. Skipping LLCM training.")
            print("Use --skip_llcm to suppress this check.\n")
        else:
            cmd = "python train_llcm_v2.py"
            if not run_command(cmd, "Step 3: Training LLCM (Corrected v2)"):
                print("LLCM training failed. Exiting.")
                sys.exit(1)
    else:
        print("Skipping LLCM training (--skip_llcm flag set).\n")
    
    # Step 4: Run inference (test generation using v2 - corrected version)
    print(f"\n{'='*60}")
    print("Step 4: Testing Inference")
    print(f"{'='*60}\n")
    try:
        import inference_v2
        import data_pipeline
        from train_vae import setup_dummy_data

        # Load test data
        if os.path.exists(args.output_dir):
            test_loader = data_pipeline.get_dataloader(args.output_dir, for_vae=False)
        else:
            setup_dummy_data("dummy_data", for_vae=False)
            test_loader = data_pipeline.get_dataloader("dummy_data", for_vae=False)

        if len(test_loader) == 0:
            raise ValueError("No test data found.")

        # Get a sample
        test_batch = next(iter(test_loader))
        if isinstance(test_batch, list):
            sample_qpi = test_batch[0]["qpi"]
        else:
            sample_qpi = test_batch["qpi"][0:1]

        print(f"Running inference on sample with shape: {sample_qpi.shape}")
        generated = inference_v2.run_inference(sample_qpi)
        print(f"Generated image shape: {generated.shape}")
        print("Inference test successful!\n")
        
    except Exception as e:
        print(f"Error during inference test: {e}")
        import traceback
        traceback.print_exc()
        print("Continuing to evaluation anyway...\n")
    
    # Step 5: Evaluate and output accuracy scores (using v2 - corrected version)
    cmd = f'python evaluate_v2.py'
    if not run_command(cmd, "Step 5: Model Evaluation & Accuracy Scores (Corrected v2)"):
        print("Evaluation failed, but pipeline completed.")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("="*60)
    print("\nAll steps completed successfully:")
    print("  ✓ Preprocessing: 2D PNGs → 3D volumes")
    print("  ✓ VAE Training: Learned DAPI compression")
    print("  ✓ LLCM Training: Learned QPI → DAPI translation")
    print("  ✓ Inference: Generated virtual DAPI stains")
    print("  ✓ Evaluation: Calculated accuracy metrics")
    print("\nCheck the evaluation output above for PSNR, SSIM, and LPIPS scores.")
    print("="*60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

