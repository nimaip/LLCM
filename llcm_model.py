import torch
import torch.nn as nn
from monai.networks.nets import UNet
import config

class QPIEncoder(nn.Module):
    """
    A simple 3D CNN to encode the QPI patch into a flat context vector.
    This replaces the "3D ResNet" with a more lightweight equivalent.
    """
    def __init__(self, in_channels=1, out_dim=config.QPI_CONTEXT_DIM):
        super().__init__()
        channels = [16, 32, 64, 128]
        strides = 2
        
        layers = []
        current_channels = in_channels
        for ch in channels:
            layers.append(
                nn.Sequential(
                    nn.Conv3d(current_channels, ch, kernel_size=3, stride=strides, padding=1),
                    nn.BatchNorm3d(ch),
                    nn.ReLU(inplace=True),
                )
            )
            current_channels = ch
            
        self.encoder = nn.ModuleList(layers)
        self.pool = nn.AdaptiveAvgPool3d(1) 
        self.fc = nn.Linear(channels[-1], out_dim)
        
    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        x = self.pool(x)
        x = torch.flatten(x, 1) 
        x = self.fc(x) 
        return x

def get_llcm_unet():
    """
    Initializes the 3D U-Net for the LLCM.
    This U-Net operates in the VAE's latent space.
    """
    # MONAI UNet doesn't support cross_attention_dim directly
    # We'll concatenate the context to the input instead
    # So in_channels = LATENT_CHANNELS + 1 (for context channel)
    # With ROI_SIZE=(32,128,128) and VAE downsample=16, latent shape is (2, 8, 8)
    # Use 2 levels of U-Net to avoid over-downsampling the small latent
    model = UNet(
        spatial_dims=3,
        in_channels=config.LATENT_CHANNELS + 1,  # +1 for context channel
        out_channels=config.LATENT_CHANNELS,
        channels=(32, 64),  # 2 levels for small latent space
        strides=(2,),  # Single downsample by 2
        num_res_units=2,
        norm="BATCH",
    )
    return model

if __name__ == "__main__":
    print("--- Testing QPI Encoder ---")
    encoder = QPIEncoder()
    dummy_qpi = torch.randn(config.BATCH_SIZE, 1, *config.ROI_SIZE)
    context = encoder(dummy_qpi)
    print(f"Input QPI shape: {dummy_qpi.shape}")
    print(f"Output context shape: {context.shape}")
    assert context.shape == (config.BATCH_SIZE, config.QPI_CONTEXT_DIM)
    
    print("\n--- Testing LLCM U-Net ---")
    llcm_unet = get_llcm_unet()
    
    downsample_factor = 2 ** len(config.VAE_STRIDES)
    latent_shape = (
        config.ROI_SIZE[0] // downsample_factor,
        config.ROI_SIZE[1] // downsample_factor,
        config.ROI_SIZE[2] // downsample_factor
    )
    
    dummy_latent = torch.randn(config.BATCH_SIZE, config.LATENT_CHANNELS, *latent_shape)
    dummy_timestep = torch.randint(0, 1000, (config.BATCH_SIZE,))
    
    print(f"Input latent shape: {dummy_latent.shape}")
    print(f"Input context shape: {context.shape}")
    
    output_latent = llcm_unet(dummy_latent, dummy_timestep, encoder_hidden_states=context)
    print(f"Output latent shape: {output_latent.shape}")
    assert output_latent.shape == dummy_latent.shape
    print("\nModel definitions are correct.")