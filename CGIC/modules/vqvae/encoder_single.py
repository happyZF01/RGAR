"""Single-scale encoder with configurable downsampling factor."""

import torch
import torch.nn as nn
from CGIC.modules.vqvae.vqvae_blocks import (
    ResnetBlock,
    AttnBlock,
    Downsample,
    Normalize,
    nonlinearity
)


class EncoderSingle(nn.Module):
    """
    Single-scale encoder supporting 4x, 8x, or 16x downsampling.

    Downsampling factors:
    - 4x: 2 downsampling layers (256 -> 64)
    - 8x: 3 downsampling layers (256 -> 32)
    - 16x: 4 downsampling layers (256 -> 16)
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
                 double_z=False,
                 **ignore_kwargs):
        super().__init__()

        if downsample_factor not in [4, 8, 16]:
            raise ValueError(f"downsample_factor must be 4, 8, or 16, got {downsample_factor}")

        self.downsample_factor = downsample_factor
        self.ch = ch
        self.temb_ch = 0
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # Calculate number of downsampling layers needed
        # 4x = 2 layers, 8x = 3 layers, 16x = 4 layers
        # We add one extra resolution stage because the last stage does not downsample.
        num_downsample_layers = {4: 2, 8: 3, 16: 4}[downsample_factor]
        self.num_resolutions = num_downsample_layers + 1

        # Adjust ch_mult to match number of resolution stages
        ch_mult = ch_mult[:self.num_resolutions]

        # Input convolution
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        # Downsampling blocks
        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()

        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]

            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(
                    in_channels=block_in,
                    out_channels=block_out,
                    temb_channels=self.temb_ch,
                    dropout=dropout
                ))
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

        # Middle blocks
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout
        )

        # Output
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(
            block_in,
            2*z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x):
        """
        Encode input image to latent representation.

        Args:
            x: Input image [B, C, H, W]

        Returns:
            h: Latent representation [B, z_channels, H/downsample_factor, W/downsample_factor]
        """
        # Timestep embedding (not used, set to None)
        temb = None

        # Downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # Middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # Output
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        return h
