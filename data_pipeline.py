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
    """Defines the MONAI transformation pipeline for training."""
    return Compose([
        LoadImaged(keys=["qpi", "dapi"]),
        EnsureChannelFirstd(keys=["qpi", "dapi"]),
        ScaleIntensityRanged(
            keys=["qpi", "dapi"],
            a_min=0, a_max=65535,  
            b_min=-1.0, b_max=1.0,
            clip=True,
        ),
        RandSpatialCropSamplesd(
            keys=["qpi", "dapi"],
            roi_size=config.ROI_SIZE,
            num_samples=config.SAMPLES_PER_VOLUME,
            random_center=True,
            random_size=False,
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
    with matching filenames, e.g., volume_001.tif
    """
    qpi_files = sorted(glob.glob(os.path.join(data_dir, "qpi", "*.tif")))
    dapi_files = sorted(glob.glob(os.path.join(data_dir, "dapi", "*.tif")))

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