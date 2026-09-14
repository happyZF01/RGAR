"""Single-scale decoder with configurable upsampling factor."""

import torch
import torch.nn as nn
from CGIC.modules.vqvae.decoder import (
    ResnetBlock,
    AttnBlock,
    Upsample,
    Normalize,
    nonlinearity
)


class DecoderSingle(nn.Module):
    """
    Single-scale decoder supporting 4x, 8x, or 16x upsampling.

    Upsampling factors:
    - 4x: 2 upsampling layers (64 -> 256)
    - 8x: 3 upsampling layers (32 -> 256)
    - 16x: 4 upsampling layers (16 -> 256)
    """

    def __init__(self,
                 downsample_factor=16,
                 ch=128,
                 out_ch=3,
                 ch_mult=(1, 2, 2, 4, 4),
                 num_res_blocks=2,
                 attn_resolutions=[32],
                 dropout=0.0,
                 resamp_with_conv=True,
                 in_channels=3,
                 resolution=256,
                 z_channels=4,
                 zq_ch=4,
                 give_pre_end=False,
                 add_conv=False,
                 **ignorekwargs):
        super().__init__()

        if downsample_factor not in [4, 8, 16]:
            raise ValueError(f"downsample_factor must be 4, 8, or 16, got {downsample_factor}")

        self.downsample_factor = downsample_factor
        self.ch = ch
        self.temb_ch = 0
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # Calculate number of upsampling layers
        # We add one extra resolution stage because the first stage does not upsample.
        num_upsample_layers = {4: 2, 8: 3, 16: 4}[downsample_factor]
        self.num_resolutions = num_upsample_layers + 1

        # Adjust ch_mult to match number of resolution stages
        ch_mult = ch_mult[:self.num_resolutions]

        # Compute in_ch_mult for upsampling path
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // downsample_factor

        # Input from latent
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # Middle blocks
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            zq_ch=zq_ch,
            add_conv=add_conv
        )
        self.mid.attn_1 = AttnBlock(block_in, zq_ch=zq_ch, add_conv=add_conv)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            zq_ch=zq_ch,
            add_conv=add_conv
        )

        # Upsampling blocks
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]

            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(
                    in_channels=block_in,
                    out_channels=block_out,
                    temb_channels=self.temb_ch,
                    dropout=dropout,
                    zq_ch=zq_ch,
                    add_conv=add_conv
                ))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in, zq_ch=zq_ch, add_conv=add_conv))

            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # Prepend to match the reversed order

        # Output
        self.norm_out = Normalize(block_in, zq_ch, add_conv=add_conv)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z, zq=None):
        """
        Decode latent representation to image.

        Args:
            z: Latent representation [B, z_channels, H, W]
            zq: Quantized representation for SpatialNorm (optional) [B, zq_ch, H, W]

        Returns:
            dec: Reconstructed image [B, out_ch, H_orig, W_orig]
        """
        # If zq is not provided, use z
        if zq is None:
            zq = z

        # Timestep embedding (not used)
        temb = None

        # Convert to block_in channels
        h = self.conv_in(z)

        # Middle
        h = self.mid.block_1(h, temb, zq)
        h = self.mid.attn_1(h, zq)
        h = self.mid.block_2(h, temb, zq)

        # Upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb, zq)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, zq)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # Output
        if self.give_pre_end:
            return h

        h = self.norm_out(h, zq)
        h = nonlinearity(h)
        h = self.conv_out(h)

        return h
