"""
Ref:
    https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/autoencoder.py
"""

from dictdot import dictdot
import numpy as np
import torch
import torch.nn as nn


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch=128,
        out_ch=3,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=3,
        resolution=256,
        z_channels=16,
        double_z=True,
        align_at_level=None,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.align_at_level = align_at_level

        # downsampling
        self.conv_in = torch.nn.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        
        # Alignment projection layer (project features to 768 dims for DINO alignment)
        if self.align_at_level is not None:
            align_channels = ch * ch_mult[self.align_at_level]
            self.alignment_proj = nn.Sequential(
                nn.Linear(align_channels, 768),
                nn.SiLU(),
                nn.Linear(768, 768),
            )
        else:
            self.alignment_proj = None

    def forward(self, x, return_align_proj=False):
        # assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        align_features = None
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
            
            # Extract alignment features at specified level
            if return_align_proj and self.align_at_level == i_level:
                align_features = hs[-1]

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        
        if return_align_proj and align_features is not None:
            # Project alignment features: [B, C, H, W] -> [B, H, W, C] -> [B, H*W, C] -> [B, H*W, 768]
            b, c, h_feat, w_feat = align_features.shape
            align_proj = align_features.permute(0, 2, 3, 1).reshape(b, h_feat * w_feat, c)
            align_proj = self.alignment_proj(align_proj)
            return h, align_proj
        
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch=128,
        out_ch=3,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=3,
        resolution=256,
        z_channels=16,
        give_pre_end=False,
        align_at_level=None,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.align_at_level = align_at_level

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        # print(
        #     "Working with z of shape {} = {} dimensions.".format(
        #         self.z_shape, np.prod(self.z_shape)
        #     )
        # )

        # z to block_in
        self.conv_in = torch.nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )
        
        # Alignment projection layer (project features to 768 dims for DINO alignment)
        if self.align_at_level is not None:
            align_channels = ch * ch_mult[self.align_at_level]
            self.alignment_proj = nn.Sequential(
                nn.Linear(align_channels, 768),
                nn.SiLU(),
                nn.Linear(768, 768),
            )
        else:
            self.alignment_proj = None

    def forward(self, z, return_align_proj=False):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        align_features = None
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            
            # Extract alignment features BEFORE upsampling (to get current resolution)
            if return_align_proj and self.align_at_level == i_level:
                align_features = h
            
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        
        if return_align_proj and align_features is not None:
            # Project alignment features: [B, C, H, W] -> [B, H, W, C] -> [B, H*W, C] -> [B, H*W, 768]
            b, c, h_feat, w_feat = align_features.shape
            align_proj = align_features.permute(0, 2, 3, 1).reshape(b, h_feat * w_feat, c)
            align_proj = self.alignment_proj(align_proj)
            return h, align_proj
        
        return h


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.parameters.device
            )

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape, device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3],
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self):
        return self.mean


class AutoencoderKL(nn.Module):
    def __init__(self, embed_dim, ch_mult, use_variational=True, encoder_align_at_level=None, decoder_align_at_level=None, patch_size=1, ch=128):
        super().__init__()
        self.use_variational = use_variational
        self.encoder_align_at_level = encoder_align_at_level
        self.decoder_align_at_level = decoder_align_at_level
        self.patch_size = patch_size
        self.ch = ch
        
        # Patchify/unpatchify layers for patch_size > 1
        # When patchifying, we need to increase channels to preserve information
        if self.patch_size > 1:
            # Patchify: expand channels to compensate for spatial reduction
            # Similar to meikai.py approach
            self.patch_embed = torch.nn.Conv2d(
                3, ch, kernel_size=patch_size, stride=patch_size, padding=0
            )
            self.patch_unembed = torch.nn.ConvTranspose2d(
                ch, 3, kernel_size=patch_size, stride=patch_size, padding=0
            )
            # Encoder and decoder need to work with 'ch' channels instead of 3
            self.encoder = Encoder(ch_mult=ch_mult, z_channels=embed_dim, align_at_level=encoder_align_at_level, in_channels=ch)
            self.decoder = Decoder(ch_mult=ch_mult, z_channels=embed_dim, align_at_level=decoder_align_at_level, out_ch=ch)
        else:
            self.encoder = Encoder(ch_mult=ch_mult, z_channels=embed_dim, align_at_level=encoder_align_at_level)
            self.decoder = Decoder(ch_mult=ch_mult, z_channels=embed_dim, align_at_level=decoder_align_at_level)
        
        mult = 2 if self.use_variational else 1
        self.quant_conv = torch.nn.Conv2d(2 * embed_dim, mult * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, embed_dim, 1)
    
    def patchify(self, x):
        """Convert image to patches if patch_size > 1"""
        if self.patch_size > 1:
            return self.patch_embed(x)
        return x
    
    def unpatchify(self, x):
        """Convert patches back to image if patch_size > 1"""
        if self.patch_size > 1:
            return self.patch_unembed(x)
        return x

    def encode(self, x, return_align_proj=False):
        # Patchify input if patch_size > 1
        x = self.patchify(x)
        
        if return_align_proj and self.encoder_align_at_level is not None:
            h, align_proj = self.encoder(x, return_align_proj=True)
        else:
            h = self.encoder(x, return_align_proj=False)
            align_proj = None
        
        moments = self.quant_conv(h)
        if not self.use_variational:
            moments = torch.cat((moments, torch.ones_like(moments)), 1)
        posterior = DiagonalGaussianDistribution(moments)
        
        if return_align_proj:
            return posterior, align_proj
        return posterior

    def decode(self, z, return_align_proj=False):
        # NOTE: We wrap the output in a dict to be consistent with the output
        z = self.post_quant_conv(z)
        if return_align_proj and self.decoder_align_at_level is not None:
            dec, align_proj = self.decoder(z, return_align_proj=True)
            # Unpatchify output if patch_size > 1
            dec = self.unpatchify(dec)
            return dictdot(dict(sample=dec)), align_proj
        else:
            dec = self.decoder(z, return_align_proj=False)
            # Unpatchify output if patch_size > 1
            dec = self.unpatchify(dec)
            return dictdot(dict(sample=dec))

    def forward(self, x, return_recon=True, return_align_proj=False):
        if return_align_proj:
            posterior, encoder_align_proj = self.encode(x, return_align_proj=True)
        else:
            posterior = self.encode(x, return_align_proj=False)
            encoder_align_proj = None
        
        z = posterior.sample()

        recon = None
        decoder_align_proj = None
        if return_recon:
            if return_align_proj:
                recon_dict, decoder_align_proj = self.decode(z, return_align_proj=True)
                recon = recon_dict.sample
            else:
                recon = self.decode(z, return_align_proj=False).sample
        
        if return_align_proj:
            return posterior, z, recon, encoder_align_proj, decoder_align_proj
        return posterior, z, recon


# Predefined VAE architectures
def VAE_F8D4(**kwargs):
    # [B, 4, 32, 32]
    return AutoencoderKL(embed_dim=4, ch_mult=[1, 2, 4, 4], use_variational=True, patch_size=1, **kwargs)


def VAE_F16D32(**kwargs):
    # [B, 32, 16, 16] (used in VA-VAE and our model)
    # For 256x256 input with 5 levels (4 downsamples): 256→128→64→32→16
    # Encoder: align at level 3 (after downsample: 32→16, so features are 16x16)
    # Decoder: starts at 16x16, processes levels in reverse [4,3,2,1,0]
    #   - Level 4: blocks at 16x16, extract here for 16x16 features
    return AutoencoderKL(
        embed_dim=32, 
        ch_mult=[1, 1, 2, 2, 4], 
        use_variational=True,
        encoder_align_at_level=3,  # Encoder: extract after level 3 downsample → 16x16
        decoder_align_at_level=4,  # Decoder: extract at level 4 before upsample → 16x16
        patch_size=1,
        **kwargs
    )


def VAE_F32D256(**kwargs):
    # [B, 256, 8, 8] (f=32 compression, matching MeiKai AE_F32D256)
    # For 256x256 input with 4 levels (3 downsamples): (due to patch_size=4) 64→32→16→8
    # Encoder: align at level 1 (after downsample: 32→16, so features are 16x16 in latent space)
    # Decoder: starts at 8x8, processes levels in reverse [3,2,1,0]
    #   - Level 3: blocks at 8x8, then upsample → 16x16
    #   - Level 2: blocks at 16x16, extract here for 16x16 features
    return AutoencoderKL(
        embed_dim=256, 
        ch_mult=[2, 2, 4, 4], 
        use_variational=True,
        encoder_align_at_level=1,  # Encoder: extract after level 1 downsample → 16x16
        decoder_align_at_level=2,  # Decoder: extract at level 2 before upsample → 16x16
        patch_size=4,
        ch=256,  # Use 256 channels after patchify to preserve information
        **kwargs
    )


vae_models = {
    "f8d4": VAE_F8D4, # [B, 4, 32, 32]
    "f16d32": VAE_F16D32, # [B, 32, 16, 16]
    "f32d256": VAE_F32D256, # [B, 256, 8, 8]
}
