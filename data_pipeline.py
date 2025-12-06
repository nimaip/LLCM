import os
import glob
import torch
from monai.data import DataLoader, CacheDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    RandSpatialCropSamplesd,
    EnsureTyped,
    ToTensord,
)
import config

def get_train_transforms():
    """
    Defines the MONAI transformation pipeline for training.
    
    Handles variable-depth 3D volumes by extracting fixed-size crops.
    RandSpatialCropSamplesd automatically handles volumes with different Z-depths
    by cropping random sub-volumes of the specified ROI_SIZE.
    """
    return Compose([
        LoadImaged(keys=["qpi", "dapi"]),
        EnsureChannelFirstd(keys=["qpi", "dapi"]),
        ScaleIntensityRanged(
            keys=["qpi", "dapi"],
            a_min=0, a_max=65535,  # Assumes 16-bit input range (0-65535)
            b_min=-1.0, b_max=1.0,
            clip=True,
        ),
        # CRITICAL: Extract fixed-size 3D chunks from variable-depth stacks.
        # If a stack has Z-depth=10 and roi_size Z=8, it picks a random start point.
        # This enables training on volumes with different Z-depths.
        RandSpatialCropSamplesd(
            keys=["qpi", "dapi"],
            roi_size=config.ROI_SIZE,  # (Depth, Height, Width) = (8, 128, 128)
            num_samples=config.SAMPLES_PER_VOLUME,
            random_center=True,
            random_size=False,  # Fixed size crops
        ),
        EnsureTyped(keys=["qpi", "dapi"], dtype=torch.float32),
        ToTensord(keys=["qpi", "dapi"]),
    ])

def get_vae_transforms():
    """Defines the MONAI transformation pipeline for VAE training (DAPI only)."""
    return Compose([
        LoadImaged(keys=["dapi"]),
        EnsureChannelFirstd(keys=["dapi"]),
        ScaleIntensityRanged(
            keys=["dapi"],
            a_min=0, a_max=65535,
            b_min=-1.0, b_max=1.0,
            clip=True,
        ),
        RandSpatialCropSamplesd(
            keys=["dapi"],
            roi_size=config.ROI_SIZE,
            num_samples=config.SAMPLES_PER_VOLUME,
            random_center=True,
            random_size=False,
        ),
        EnsureTyped(keys=["dapi"], dtype=torch.float32),
        ToTensord(keys=["dapi"]),
    ])

def get_dataloader(data_dir, for_vae=False):
    """
    Creates a MONAI DataLoader.
    Assumes data_dir contains two subfolders: 'qpi' and 'dapi'
    with matching filenames (e.g., stack_*.tif or volume_*.tif).
    
    Supports variable-depth 3D volumes created by build_3d_dataset.py.
    The volumes are automatically cropped to fixed-size patches during training.
    """
    # Support both .tif and .nii.gz files
    qpi_files = sorted(glob.glob(os.path.join(data_dir, "qpi", "*.tif")) + 
                       glob.glob(os.path.join(data_dir, "qpi", "*.nii.gz")))
    dapi_files = sorted(glob.glob(os.path.join(data_dir, "dapi", "*.tif")) + 
                        glob.glob(os.path.join(data_dir, "dapi", "*.nii.gz")))

    if not dapi_files:
        raise ValueError(f"No DAPI files found in {data_dir}/dapi")
    
    if not for_vae and not qpi_files:
        raise ValueError(f"No QPI files found in {data_dir}/qpi")
    
    if not for_vae and len(qpi_files) != len(dapi_files):
        raise ValueError("Mismatch between QPI and DAPI file counts. Ensure they are paired.")

    if for_vae:
        data_dicts = [{"dapi": dapi_file} for dapi_file in dapi_files]
        transforms = get_vae_transforms()
    else:
        data_dicts = [
            {"qpi": qpi_file, "dapi": dapi_file}
            for qpi_file, dapi_file in zip(qpi_files, dapi_files)
        ]
        transforms = get_train_transforms()

    dataset = CacheDataset(
        data=data_dicts,
        transform=transforms,
        cache_rate=1.0,  # 1.0 means cache all data
        num_workers=config.NUM_WORKERS,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    return loader