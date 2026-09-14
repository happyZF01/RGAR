"""Inference-only RGAR model definition."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from CGIC.modules.channel import TreePathChannel
from CGIC.modules.vqvae.decoder_single import DecoderSingle
from CGIC.modules.vqvae.encoder_single import EncoderSingle
from CGIC.modules.vqvae.quantize import VectorizedTreeQuantizer


class RGARInferenceModel(nn.Module):
    """Frozen depth-10 Tree-VQ image model with the RGAR receiver."""

    def __init__(self, channel: str = "awgn") -> None:
        super().__init__()
        if channel not in {"awgn", "fading-perfect", "fading-estimated"}:
            raise ValueError(f"Unsupported channel: {channel}")

        ddconfig = {
            "double_z": False,
            "z_channels": 4,
            "resolution": 256,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 128,
            "ch_mult": [1, 2, 2, 4, 4],
            "num_res_blocks": 2,
            "attn_resolutions": [32],
            "dropout": 0.0,
        }
        self.encoder = EncoderSingle(downsample_factor=4, **ddconfig)
        self.decoder = DecoderSingle(
            downsample_factor=4, zq_ch=4, **ddconfig
        )
        self.quantize = VectorizedTreeQuantizer(
            max_depth=10,
            e_dim=4,
            beta=0.25,
            alpha_decay=1.0,
            rate_ema_decay=0.99,
            rate_prob_floor=1.0e-6,
            rate_softmax_temperature=1.0,
            init_on_first_batch=False,
            kmeans_iters=10,
        )
        self.quant_conv = nn.Conv2d(4, 4, 1)
        self.post_quant_conv = nn.Conv2d(4, 4, 1)

        fading = channel.startswith("fading")
        estimated = channel == "fading-estimated"
        self.tree_path_channel = TreePathChannel(
            enabled=True,
            channel_type="fading" if fading else "awgn",
            modulation="gray16qam",
            snr_min_db=-5,
            snr_max_db=20,
            snr_step_db=1,
            validation_snr_db=5.0,
            csi_mode="estimated" if estimated else "perfect",
            pilot_symbols_per_image=32 if estimated else 0,
            split_parent_child=True,
            parent_depth=4,
            parent_reliable_bit_mapping=True,
            parent_repetition_factor=3,
            snr_adaptive_parent_power=False,
            adaptive_parent_ir_enabled=True,
            adaptive_parent_ir_policy="topk_blockwise",
            adaptive_parent_ir_min_repetition_factor=2,
            adaptive_parent_ir_max_snr_db=10.0,
            adaptive_parent_ir_r2_threshold=0.95,
            adaptive_parent_ir_r3_threshold=0.85,
            adaptive_parent_ir_pilot_weight=0.25,
            adaptive_parent_ir_block_tokens=128,
            adaptive_parent_ir_feedback_repetitions=3,
        )

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        source = checkpoint.get("state_dict", checkpoint)
        target = self.state_dict()
        compatible = {
            name: value
            for name, value in source.items()
            if name in target and target[name].shape == value.shape
        }
        required_prefixes = (
            "encoder.",
            "decoder.",
            "quantize.",
            "quant_conv.",
            "post_quant_conv.",
        )
        missing = [
            name
            for name in target
            if name.startswith(required_prefixes) and name not in compatible
        ]
        allowed_missing = {
            "quantize.branch_count_ema",
            "quantize.branch_count_updates",
        }
        missing = [name for name in missing if name not in allowed_missing]
        if missing:
            raise RuntimeError(f"Checkpoint is missing inference weights: {missing[:8]}")
        self.load_state_dict(compatible, strict=False)

    @torch.inference_mode()
    def encode_path(self, image: torch.Tensor):
        latent = self.quant_conv(self.encoder(image))
        quantized, _, _, info = self.quantize(latent, return_path_info=True)
        return latent, quantized, info["path_bits"]

    @torch.inference_mode()
    def decode_quantized(self, quantized: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.post_quant_conv(quantized))

    @torch.inference_mode()
    def decode_indices(
        self, latent: torch.Tensor, received_indices: torch.Tensor
    ) -> torch.Tensor:
        batch, _, height, width = latent.shape
        quantized = F.embedding(
            received_indices.reshape(-1), self.quantize.codebook
        )
        quantized = quantized.view(batch, height, width, -1)
        quantized = rearrange(
            quantized, "b h w c -> b c h w"
        ).contiguous()
        return self.decode_quantized(quantized)
