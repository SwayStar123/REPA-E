import torch
import torch.nn as nn

def build_mlp(hidden_size, z_dim):
    return nn.Sequential(
            nn.Linear(hidden_size, z_dim),
            nn.SiLU(),
            nn.Linear(z_dim, z_dim),
        )

def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)

class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3

class Block(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()
    def forward(self, x):
        return self.fuse(self.conv(x) + self.skip(x))

class DownsampleBlock(nn.Module):
    def __init__(self, hidden_dim=64, num_blocks=3):
        super().__init__()
        self.conv = conv(hidden_dim, hidden_dim, stride=2, bias=False)
        self.blocks = nn.Sequential(
            *[Block(hidden_dim, hidden_dim) for _ in range(num_blocks)]
        )
    def forward(self, x):
        x = self.conv(x)
        x = self.blocks(x)
        return x

class UpsampleBlock(nn.Module):
    def __init__(self, hidden_dim=64, num_blocks=3):
        super().__init__()
        self.blocks = nn.Sequential(
            *[Block(hidden_dim, hidden_dim) for _ in range(num_blocks)]
        )
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = conv(hidden_dim, hidden_dim, bias=False)
    def forward(self, x):
        x = self.blocks(x)
        x = self.upsample(x)
        x = self.conv(x)
        return x

class Patchify(nn.Module):
    """Convert image to patches (reduces spatial resolution)"""
    def __init__(self, in_channels, out_channels, patch_size):
        super().__init__()
        self.patch_size = patch_size
        # Use strided conv to reduce resolution and increase channels
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=patch_size, 
            stride=patch_size
        )
    
    def forward(self, x):
        return self.conv(x)

class Unpatchify(nn.Module):
    """Convert patches back to image (increases spatial resolution)"""
    def __init__(self, in_channels, out_channels, patch_size):
        super().__init__()
        self.patch_size = patch_size
        # Use transposed conv to increase resolution
        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self, x):
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, latent_channels=4, hidden_dim=64, num_downsample_blocks=3, num_blocks_per_downsample=3, patch_size=16, align_at_block=None):
        super().__init__()
        hd = hidden_dim
        self.inp = nn.Sequential(*[
            Patchify(3, hd, patch_size), 
            Block(hd, hd),
        ])

        self.alignment_proj = build_mlp(hidden_dim, 768)
        
        # If align_at_block is specified, we'll extract features at that specific downsample block
        self.align_at_block = align_at_block if align_at_block is not None else 0

        self.down_blocks = nn.ModuleList([
            DownsampleBlock(hd, num_blocks_per_downsample) for _ in range(num_downsample_blocks)
        ])

        self.out = conv(hd, latent_channels)
        
    def forward(self, x):
        x = self.inp(x)

        # Extract alignment features at the specified block
        align_features = None
        for i, block in enumerate(self.down_blocks):
            x = block(x)
            if i == self.align_at_block:
                align_features = x
        
        # If no specific block is set, use the input features (before downsampling)
        if align_features is None:
            align_features = x
            
        align_proj = self.alignment_proj(align_features.permute(0, 2, 3, 1))
        
        x = self.out(x)

        return x, align_proj

class Decoder(nn.Module):
    def __init__(self, latent_channels=4, hidden_dim=64, num_upsample_blocks=3, num_blocks_per_upsample=3, patch_size=16, align_at_block=None):
        super().__init__()
        hd = hidden_dim
        self.inp = nn.Sequential(*[
            Clamp(), 
            conv(latent_channels, hd), 
            nn.ReLU(),
        ])

        # If align_at_block is specified, we'll extract features at that specific upsample block
        self.align_at_block = align_at_block if align_at_block is not None else 0

        self.up_blocks = nn.ModuleList([
            UpsampleBlock(hd, num_blocks_per_upsample) for _ in range(num_upsample_blocks)
        ])

        self.alignment_layer = build_mlp(hidden_dim, 768)

        self.out = nn.Sequential(*[
            Block(hd, hd),
            Unpatchify(hd, 3, patch_size)
        ])

    def forward(self, z):
        z = self.inp(z)
        
        # Extract alignment features at the specified block
        align_features = None
        for i, block in enumerate(self.up_blocks):
            z = block(z)
            if i == self.align_at_block:
                align_features = z

        # If no specific block is set, use final features
        if align_features is None:
            align_features = z

        align_proj = self.alignment_layer(align_features.permute(0, 2, 3, 1))

        z = self.out(z)

        return z, align_proj

class MeiKai(nn.Module):
    def __init__(self, latent_channels=4, hidden_dim=64, num_blocks=3, blocks_per_stage=3, patch_size=8, encoder_align_at_block=None, decoder_align_at_block=None):
        super().__init__()
        self.encoder = Encoder(latent_channels, hidden_dim, num_blocks, blocks_per_stage, patch_size, encoder_align_at_block)
        self.decoder = Decoder(latent_channels, hidden_dim, num_blocks, blocks_per_stage, patch_size, decoder_align_at_block)

def AE_F32D256(**kwargs):
    """
    Autoencoder with f=32 compression factor (256x256 -> 8x8 latents).
    Alignment happens at f=16 (16x16 feature maps) for both encoder and decoder.
    
    Architecture:
    - Encoder: Patchify(4) -> 64x64 -> Downsample -> 32x32 -> Downsample -> 16x16 [ALIGN] -> Downsample -> 8x8
    - Decoder: 8x8 -> Upsample -> 16x16 [ALIGN] -> Upsample -> 32x32 -> Upsample -> 64x64 -> Unpatchify(4) -> 256x256
    
    For num_blocks=3 with patch_size=4:
    - Encoder aligns at downsample block 1 (after 2nd downsample, 16x16 resolution)
    - Decoder aligns at upsample block 0 (after 1st upsample, 16x16 resolution)
    """
    return MeiKai(
        latent_channels=256,
        hidden_dim=768,
        num_blocks=3,
        blocks_per_stage=2,
        patch_size=4,
        encoder_align_at_block=1,  # Align at second downsample block (16x16 resolution)
        decoder_align_at_block=0,  # Align at first upsample block (16x16 resolution)
        **kwargs
    )

@torch.no_grad()
def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device", dev)
    
    # Create a random 256x256 RGB image
    random_image = torch.rand(1, 3, 256, 256).to(dev)
    print(f"Input image shape: {random_image.shape}")
    
    # Create the autoencoder
    ae = MeiKai(hidden_dim=768, latent_channels=768, num_blocks=1, blocks_per_stage=2, patch_size=16).to(dev)
    # print parameters
    def nparams(m): 
        return sum(p.numel() for p in m.parameters())
    print(f"  Encoder: {nparams(ae.encoder):,}")
    print(f"  Decoder: {nparams(ae.decoder):,}")
    
    # Encode the image
    latent = ae.encoder(random_image)[0]
    print(f"latent latent shape: {latent.shape}")
    
    # Decode the latent
    decoded = ae.decoder.step(latent, random_image)
    print(f"Decoded image shape: {decoded.shape}")
    
    # Verify shapes match
    assert random_image.shape == decoded.shape, f"Shape mismatch! Input: {random_image.shape}, Output: {decoded.shape}"
    print("✓ Shapes match! Autoencoder test passed.")

    from thop import profile

    # Calculate FLOPs
    encoder_flops, _ = profile(ae.encoder, inputs=(random_image,))
    decoder_flops, _ = profile(ae.decoder, inputs=(latent, random_image, 50, 1.0))
    
    print(f"Encoder: {encoder_flops / 1e9:.2f} GFLOPs")
    print(f"Decoder: {decoder_flops / 1e9:.2f} GFLOPs")
    print(f"Total:   {(encoder_flops + decoder_flops) / 1e9:.2f} GFLOPs")



if __name__ == "__main__":
    main()