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
        self.pool = nn.AvgPool2d(2, 2)
        self.conv = conv(hidden_dim, hidden_dim, stride=2, bias=False)
        self.blocks = nn.Sequential(
            *[Block(hidden_dim, hidden_dim) for _ in range(num_blocks)]
        )
    def forward(self, x):
        # x = self.pool(x)
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
    def __init__(self, latent_channels=4, hidden_dim=64, num_downsample_blocks=3, num_blocks_per_downsample=3, patch_size=16):
        super().__init__()
        hd = hidden_dim
        self.inp = nn.Sequential(*[
            Patchify(3, hd, patch_size), 
            Block(hd, hd),
        ])

        self.alignment_proj = build_mlp(hidden_dim, 768)

        self.down_blocks = nn.Sequential(*[
            DownsampleBlock(hd, num_blocks_per_downsample) for _ in range(num_downsample_blocks)
        ])

        self.out = conv(hd, latent_channels)
    def forward(self, x):
        x = self.inp(x)

        align_proj = self.alignment_proj(x.permute(0, 2, 3, 1))
        
        x = self.down_blocks(x)
        x = self.out(x)

        return x, align_proj

class Decoder(nn.Module):
    def __init__(self, latent_channels=4, hidden_dim=64, num_upsample_blocks=3, num_blocks_per_upsample=3, patch_size=16):
        super().__init__()
        hd = hidden_dim
        self.inp = nn.Sequential(*[
            Clamp(), 
            conv(latent_channels, hd), 
            nn.ReLU(),
        ])

        self.up_blocks = nn.Sequential(*[
            UpsampleBlock(hd, num_blocks_per_upsample) for _ in range(num_upsample_blocks)
        ])

        self.alignment_layer = build_mlp(hidden_dim, 768)

        self.out = nn.Sequential(*[
            Block(hd, hd),
            Unpatchify(hd, 3, patch_size)
        ])

    def forward(self, z):
        z = self.inp(z)
        z = self.up_blocks(z)

        align_proj = self.alignment_layer(z.permute(0, 2, 3, 1))

        z = self.out(z)

        return z, align_proj

class MeiKai(nn.Module):
    def __init__(self, latent_channels=4, hidden_dim=64, num_blocks=3, blocks_per_stage=3, patch_size=8):
        super().__init__()
        self.encoder = Encoder(latent_channels, hidden_dim, num_blocks, blocks_per_stage, patch_size)
        self.decoder = Decoder(latent_channels, hidden_dim, num_blocks, blocks_per_stage, patch_size)

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