"""Digital channels for full-depth Tree VQ path bits."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn


class TreePathChannel(nn.Module):
    """Transmit Tree VQ path bits with optional soft parent repetition UEP."""

    _GRAY_BITS = ((0, 0), (0, 1), (1, 1), (1, 0))

    def __init__(
        self,
        enabled: bool = False,
        channel_type: str = "awgn",
        modulation: str = "gray16qam",
        snr_min_db: int = 0,
        snr_max_db: int = 30,
        snr_step_db: int = 1,
        validation_snr_db: float = 15.0,
        csi_mode: str = "perfect",
        pilot_symbols_per_image: int = 8,
        pilot_snr_offset_db: float = 0.0,
        split_parent_child: bool = False,
        parent_depth: int = 4,
        parent_reliable_bit_mapping: bool = False,
        explicit_depth_order: list[int] | None = None,
        parent_repetition_factor: int = 1,
        snr_adaptive_parent_power: bool = False,
        parent_repetition_power_max: float = 1.0,
        parent_power_full_until_db: float = 5.0,
        parent_power_unity_from_db: float = 20.0,
        soft_child_mmse_enabled: bool = False,
        soft_child_mmse_topk: int = 4,
        soft_child_mmse_temperature_scale: float = 1.0,
        semantic_medoid_bayes_enabled: bool = False,
        parent_hamming84_softml_enabled: bool = False,
        adaptive_parent_ir_enabled: bool = False,
        adaptive_parent_ir_policy: str = "adaptive",
        adaptive_parent_ir_min_repetition_factor: int = 1,
        adaptive_parent_ir_max_snr_db: float = 10.0,
        adaptive_parent_ir_r2_threshold: float = 0.95,
        adaptive_parent_ir_r3_threshold: float = 0.85,
        adaptive_parent_ir_pilot_weight: float = 0.25,
        adaptive_parent_ir_block_tokens: int = 256,
        adaptive_parent_ir_feedback_repetitions: int = 3,
        adaptive_parent_ir_topk_request_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.channel_type = str(channel_type).lower()
        self.modulation = str(modulation).lower()
        self.snr_min_db = int(snr_min_db)
        self.snr_max_db = int(snr_max_db)
        self.snr_step_db = int(snr_step_db)
        self.validation_snr_db = float(validation_snr_db)
        self.csi_mode = str(csi_mode).lower()
        self.pilot_symbols_per_image = int(pilot_symbols_per_image)
        self.pilot_snr_offset_db = float(pilot_snr_offset_db)
        self.split_parent_child = bool(split_parent_child)
        self.parent_depth = int(parent_depth)
        self.parent_reliable_bit_mapping = bool(parent_reliable_bit_mapping)
        self.explicit_depth_order = (
            tuple(int(value) for value in explicit_depth_order)
            if explicit_depth_order is not None
            else None
        )
        self.parent_repetition_factor = int(parent_repetition_factor)
        self.snr_adaptive_parent_power = bool(snr_adaptive_parent_power)
        self.parent_repetition_power_max = float(parent_repetition_power_max)
        self.parent_power_full_until_db = float(parent_power_full_until_db)
        self.parent_power_unity_from_db = float(parent_power_unity_from_db)
        self.soft_child_mmse_enabled = bool(soft_child_mmse_enabled)
        self.soft_child_mmse_topk = int(soft_child_mmse_topk)
        self.soft_child_mmse_temperature_scale = float(soft_child_mmse_temperature_scale)
        self.semantic_medoid_bayes_enabled = bool(semantic_medoid_bayes_enabled)
        self.parent_hamming84_softml_enabled = bool(parent_hamming84_softml_enabled)
        self.adaptive_parent_ir_enabled = bool(adaptive_parent_ir_enabled)
        self.adaptive_parent_ir_policy = str(adaptive_parent_ir_policy).lower()
        self.adaptive_parent_ir_min_repetition_factor = int(
            adaptive_parent_ir_min_repetition_factor
        )
        self.adaptive_parent_ir_max_snr_db = float(
            adaptive_parent_ir_max_snr_db
        )
        self.adaptive_parent_ir_r2_threshold = float(
            adaptive_parent_ir_r2_threshold
        )
        self.adaptive_parent_ir_r3_threshold = float(
            adaptive_parent_ir_r3_threshold
        )
        self.adaptive_parent_ir_pilot_weight = float(
            adaptive_parent_ir_pilot_weight
        )
        self.adaptive_parent_ir_block_tokens = int(adaptive_parent_ir_block_tokens)
        self.adaptive_parent_ir_feedback_repetitions = int(
            adaptive_parent_ir_feedback_repetitions
        )
        self.adaptive_parent_ir_topk_request_rate = float(
            adaptive_parent_ir_topk_request_rate
        )

        if self.channel_type not in {"awgn", "fading"}:
            raise ValueError(f"channel_type must be 'awgn' or 'fading', got {channel_type}")
        if self.modulation != "gray16qam":
            raise NotImplementedError("A1 currently supports only Gray-16QAM")
        if self.snr_step_db <= 0 or self.snr_max_db < self.snr_min_db:
            raise ValueError("invalid SNR range")
        if self.csi_mode not in {"perfect", "estimated"}:
            raise ValueError(f"csi_mode must be 'perfect' or 'estimated', got {csi_mode}")
        if self.channel_type == "fading" and self.csi_mode == "estimated" and self.pilot_symbols_per_image < 1:
            raise ValueError("estimated CSI requires at least one pilot symbol per image")
        if self.parent_depth < 1:
            raise ValueError("parent_depth must be positive")
        if self.parent_reliable_bit_mapping and not self.split_parent_child:
            raise ValueError("parent_reliable_bit_mapping requires split_parent_child=true")
        if self.explicit_depth_order is not None and not self.parent_reliable_bit_mapping:
            raise ValueError(
                "explicit_depth_order requires parent_reliable_bit_mapping=true"
            )
        if self.parent_repetition_factor < 1:
            raise ValueError("parent_repetition_factor must be at least 1")
        if self.parent_repetition_factor > 1 and not self.parent_reliable_bit_mapping:
            raise ValueError(
                "parent_repetition_factor > 1 requires parent_reliable_bit_mapping=true"
            )
        if self.snr_adaptive_parent_power and self.parent_repetition_factor <= 1:
            raise ValueError(
                "snr_adaptive_parent_power requires parent_repetition_factor > 1"
            )
        if self.parent_repetition_power_max < 1.0:
            raise ValueError("parent_repetition_power_max must be at least 1.0")
        if self.parent_power_unity_from_db <= self.parent_power_full_until_db:
            raise ValueError(
                "parent_power_unity_from_db must exceed parent_power_full_until_db"
            )
        if self.soft_child_mmse_enabled and not self.parent_reliable_bit_mapping:
            raise ValueError("soft_child_mmse_enabled requires parent_reliable_bit_mapping=true")
        if self.semantic_medoid_bayes_enabled and not self.parent_reliable_bit_mapping:
            raise ValueError("semantic_medoid_bayes_enabled requires parent_reliable_bit_mapping=true")
        if self.parent_hamming84_softml_enabled:
            if not self.parent_reliable_bit_mapping:
                raise ValueError(
                    "parent_hamming84_softml_enabled requires parent_reliable_bit_mapping=true"
                )
            if self.parent_depth != 4:
                raise ValueError("parent Hamming(8,4) requires parent_depth=4")
            if self.parent_repetition_factor != 2:
                raise ValueError(
                    "parent Hamming(8,4) replaces the matched r2 redundancy budget"
                )
            if self.snr_adaptive_parent_power:
                raise ValueError(
                    "parent Hamming(8,4) comparison requires fixed unit symbol power"
                )
        if self.soft_child_mmse_topk < 1:
            raise ValueError("soft_child_mmse_topk must be positive")
        if self.soft_child_mmse_temperature_scale <= 0.0:
            raise ValueError("soft_child_mmse_temperature_scale must be positive")
        if self.adaptive_parent_ir_enabled:
            if self.parent_repetition_factor != 3:
                raise ValueError(
                    "adaptive parent IR requires parent_repetition_factor=3 "
                    "to simulate at most two incremental copies"
                )
            if self.parent_hamming84_softml_enabled:
                raise ValueError("adaptive parent IR does not stack Hamming")
            if self.snr_adaptive_parent_power:
                raise ValueError(
                    "adaptive parent IR phase A requires fixed unit symbol power"
                )
            if self.adaptive_parent_ir_policy not in {
                "adaptive",
                "adaptive_blockwise",
                "fixed_r2",
                "fixed_r3",
                "random_blockwise",
                "topk_blockwise",
                "random_topk_blockwise",
            }:
                raise ValueError(
                    "adaptive_parent_ir_policy must be adaptive, adaptive_blockwise, "
                    "fixed_r2, fixed_r3, random_blockwise, topk_blockwise, or "
                    "random_topk_blockwise"
                )
            if self.adaptive_parent_ir_min_repetition_factor not in {1, 2}:
                raise ValueError(
                    "adaptive_parent_ir_min_repetition_factor must be 1 or 2"
                )
            if self.adaptive_parent_ir_max_snr_db < self.snr_min_db:
                raise ValueError(
                    "adaptive_parent_ir_max_snr_db must cover the minimum SNR"
                )
            for name, threshold in (
                ("r2", self.adaptive_parent_ir_r2_threshold),
                ("r3", self.adaptive_parent_ir_r3_threshold),
            ):
                if not 0.0 <= threshold <= 1.0:
                    raise ValueError(f"adaptive parent IR {name} threshold must be in [0,1]")
            if self.adaptive_parent_ir_pilot_weight < 0.0:
                raise ValueError(
                    "adaptive_parent_ir_pilot_weight must be nonnegative"
                )
            if self.adaptive_parent_ir_block_tokens < 1:
                raise ValueError("adaptive_parent_ir_block_tokens must be positive")
            if self.adaptive_parent_ir_feedback_repetitions < 1:
                raise ValueError(
                    "adaptive_parent_ir_feedback_repetitions must be positive"
                )
            if not 0.0 <= self.adaptive_parent_ir_topk_request_rate <= 1.0:
                raise ValueError(
                    "adaptive_parent_ir_topk_request_rate must be in [0,1]"
                )

        gray_bits = torch.tensor(self._GRAY_BITS, dtype=torch.long)
        gray_levels = torch.tensor([-3.0, -1.0, 1.0, 3.0], dtype=torch.float32)
        self.register_buffer("gray_bits", gray_bits, persistent=False)
        self.register_buffer("gray_levels", gray_levels, persistent=False)
        labels = torch.arange(16, dtype=torch.long)
        shifts = torch.tensor([3, 2, 1, 0], dtype=torch.long)
        constellation_bits = labels[:, None].bitwise_right_shift(shifts).bitwise_and(1)
        constellation = self._modulate(constellation_bits.view(16, 1, 4)).view(16, 2)
        self.register_buffer("constellation_bits", constellation_bits, persistent=False)
        self.register_buffer("constellation", constellation, persistent=False)
        hamming_messages = torch.arange(16, dtype=torch.long)
        hamming_messages = hamming_messages[:, None].bitwise_right_shift(shifts).bitwise_and(1)
        hamming_parity_matrix = 1 - torch.eye(4, dtype=torch.long)
        hamming_parity = (hamming_messages @ hamming_parity_matrix).remainder(2)
        self.register_buffer("hamming84_messages", hamming_messages, persistent=False)
        self.register_buffer("hamming84_parity", hamming_parity, persistent=False)

    def _sample_snr_db(self, device: torch.device, training: bool) -> torch.Tensor:
        if not training:
            return torch.tensor(self.validation_snr_db, device=device, dtype=torch.float32)
        count = (self.snr_max_db - self.snr_min_db) // self.snr_step_db + 1
        snr_index = torch.randint(0, count, (), device=device)
        return (self.snr_min_db + snr_index * self.snr_step_db).to(torch.float32)

    def _modulate(self, packed_bits: torch.Tensor) -> torch.Tensor:
        # packed_bits: [B, S, 4], with two Gray bits per real dimension.
        real_label = 2 * packed_bits[..., 0] + packed_bits[..., 1]
        imag_label = 2 * packed_bits[..., 2] + packed_bits[..., 3]
        # Binary labels 00,01,10,11 map to Gray levels -3,-1,+3,+1.
        label_to_level = self.gray_levels.new_tensor([-3.0, -1.0, 3.0, 1.0])
        real = label_to_level[real_label]
        imag = label_to_level[imag_label]
        return torch.stack([real, imag], dim=-1) / torch.sqrt(real.new_tensor(10.0))

    def _demodulate(self, symbols: torch.Tensor) -> torch.Tensor:
        levels = self.gray_levels.to(device=symbols.device, dtype=symbols.dtype)
        levels = levels / torch.sqrt(symbols.new_tensor(10.0))
        real_index = torch.argmin((symbols[..., 0, None] - levels) ** 2, dim=-1)
        imag_index = torch.argmin((symbols[..., 1, None] - levels) ** 2, dim=-1)
        real_bits = self.gray_bits[real_index]
        imag_bits = self.gray_bits[imag_index]
        return torch.cat([real_bits, imag_bits], dim=-1)

    @staticmethod
    def _modulate_qpsk(bits: torch.Tensor) -> torch.Tensor:
        """Map paired parent bits to unit-energy QPSK symbols."""
        if bits.shape[-1] != 2:
            raise ValueError(f"QPSK bits must end in pairs, got {tuple(bits.shape)}")
        levels = bits.to(torch.float32).mul(2.0).sub(1.0)
        return levels / torch.sqrt(levels.new_tensor(2.0))

    def _pack_parent_repetitions(
        self, parent_bits: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        """Return extra QPSK copies and the number of valid parent bits."""
        bit_count = int(parent_bits.shape[1])
        padded_count = ((bit_count + 1) // 2) * 2
        padded = torch.nn.functional.pad(parent_bits, (0, padded_count - bit_count), value=0)
        one_copy = self._modulate_qpsk(padded.view(parent_bits.shape[0], -1, 2))
        extra_copies = self.parent_repetition_factor - 1
        if extra_copies == 0:
            return one_copy[:, :0], bit_count
        return one_copy.repeat(1, extra_copies, 1), bit_count

    def _pack_parent_hamming84_parity(
        self, parent_bits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode four parent bits per token into four parity bits at the r2 budget."""
        parent_blocks = parent_bits.view(parent_bits.shape[0], -1, 4)
        parity_bits = (
            parent_blocks.sum(dim=-1, keepdim=True) - parent_blocks
        ).remainder(2)
        parity_symbols = self._modulate_qpsk(
            parity_bits.reshape(parent_bits.shape[0], -1, 2)
        )
        return parity_symbols, parity_bits.reshape(parent_bits.shape[0], -1)

    def _decode_parent_hamming84_softml(
        self,
        data_prob_one: torch.Tensor,
        parity_prob_one: torch.Tensor,
    ) -> torch.Tensor:
        """Choose the most likely systematic Hamming(8,4) codeword per token."""
        batch_size = data_prob_one.shape[0]
        data_probability = data_prob_one.view(batch_size, -1, 4)
        parity_probability = parity_prob_one.view(batch_size, -1, 4)
        probability = torch.cat([data_probability, parity_probability], dim=-1)
        probability = probability.clamp(1e-7, 1.0 - 1e-7)

        messages = self.hamming84_messages.to(device=probability.device)
        parity = self.hamming84_parity.to(device=probability.device)
        codewords = torch.cat([messages, parity], dim=-1).to(probability.dtype)
        log_probability = (
            codewords.view(1, 1, 16, 8) * probability.unsqueeze(-2).log()
            + (1.0 - codewords).view(1, 1, 16, 8)
            * (1.0 - probability).unsqueeze(-2).log()
        ).sum(dim=-1)
        chosen = log_probability.argmax(dim=-1)
        return messages[chosen].reshape(batch_size, -1)

    def _symbol_power_allocation(
        self,
        snr_db: torch.Tensor,
        data_symbol_count: int,
        redundancy_symbol_count: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Allocate parent-repetition power while keeping frame-average power at one."""
        one = snr_db.new_tensor(1.0)
        if not self.snr_adaptive_parent_power or redundancy_symbol_count == 0:
            return one, one

        span = self.parent_power_unity_from_db - self.parent_power_full_until_db
        low_snr_weight = (
            (self.parent_power_unity_from_db - snr_db) / span
        ).clamp(0.0, 1.0)
        redundancy_power = one + (
            self.parent_repetition_power_max - 1.0
        ) * low_snr_weight
        total_symbol_count = data_symbol_count + redundancy_symbol_count
        data_power = (
            total_symbol_count - redundancy_symbol_count * redundancy_power
        ) / data_symbol_count
        if torch.any(data_power <= 0.0):
            raise ValueError(
                "parent repetition power leaves non-positive power for data symbols"
            )
        return data_power, redundancy_power

    @staticmethod
    def _complex_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1],
             a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]],
            dim=-1,
        )

    @staticmethod
    def _zero_force_equalize(y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        denominator = (h[..., 0] ** 2 + h[..., 1] ** 2).clamp_min(1e-8)
        return torch.stack(
            [(y[..., 0] * h[..., 0] + y[..., 1] * h[..., 1]) / denominator,
             (y[..., 1] * h[..., 0] - y[..., 0] * h[..., 1]) / denominator],
            dim=-1,
        )

    def _apply_channel(
        self,
        symbols: torch.Tensor,
        snr_db: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        snr_linear = torch.pow(symbols.new_tensor(10.0), snr_db / 10.0)
        noise_std = torch.sqrt(1.0 / (2.0 * snr_linear))
        noise = torch.randn_like(symbols) * noise_std

        if self.channel_type == "awgn":
            noise_variance = symbols.new_ones((symbols.shape[0], 1)) / snr_linear
            return (
                symbols + noise,
                symbols.new_zeros(()),
                noise_variance,
                symbols.new_zeros((symbols.shape[0],)),
            )

        batch_size = symbols.shape[0]
        h = torch.randn((batch_size, 1, 2), device=symbols.device, dtype=symbols.dtype)
        h = h / torch.sqrt(symbols.new_tensor(2.0))
        received = self._complex_multiply(symbols, h) + noise

        if self.csi_mode == "perfect":
            h_hat = h
            csi_nmse = symbols.new_zeros(())
            pilot_uncertainty = symbols.new_zeros((batch_size, 1))
        else:
            pilot_snr_db = snr_db + self.pilot_snr_offset_db
            pilot_snr_linear = torch.pow(symbols.new_tensor(10.0), pilot_snr_db / 10.0)
            pilot_noise_std = torch.sqrt(1.0 / (2.0 * pilot_snr_linear))
            pilot_noise = torch.randn(
                (batch_size, self.pilot_symbols_per_image, 2),
                device=symbols.device,
                dtype=symbols.dtype,
            ) * pilot_noise_std
            pilot_observations = h + pilot_noise
            h_hat = pilot_observations.mean(dim=1, keepdim=True)
            csi_nmse = torch.mean((h_hat - h) ** 2) / torch.mean(h ** 2).clamp_min(1e-8)
            pilot_residual = pilot_observations - h_hat
            residual_power = (
                pilot_residual.square().sum(dim=-1).mean(dim=1, keepdim=True)
            )
            h_hat_observed_power = h_hat.square().sum(dim=-1).clamp_min(1e-6)
            pilot_uncertainty = residual_power / h_hat_observed_power

        h_hat_power = (h_hat[..., 0] ** 2 + h_hat[..., 1] ** 2).clamp_min(1e-6)
        noise_variance = (1.0 / snr_linear) / h_hat_power
        if self.csi_mode == "estimated":
            noise_variance = noise_variance + pilot_uncertainty
        return (
            self._zero_force_equalize(received, h_hat),
            csi_nmse,
            noise_variance,
            pilot_uncertainty.squeeze(-1),
        )

    def _qam_bit_probabilities(
        self, symbols: torch.Tensor, noise_variance: torch.Tensor
    ) -> torch.Tensor:
        """Return analytic bit marginals for equalized Gray-16QAM symbols."""
        constellation = self.constellation.to(device=symbols.device, dtype=symbols.dtype)
        distances = ((symbols.unsqueeze(-2) - constellation) ** 2).sum(dim=-1)
        temperature = (
            noise_variance.to(device=symbols.device, dtype=symbols.dtype).unsqueeze(-1)
            * self.soft_child_mmse_temperature_scale
        ).clamp_min(1e-5)
        posterior = torch.softmax(-distances / temperature, dim=-1)
        bits = self.constellation_bits.to(device=symbols.device, dtype=posterior.dtype)
        return posterior @ bits

    def _soft_child_leaf_candidates(
        self,
        parent_bits: torch.Tensor,
        child_prob_one: torch.Tensor,
        depth: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Enumerate each parent subtree and keep top-k child posterior leaves."""
        child_depth = depth - self.parent_depth
        candidate_count = 1 << child_depth
        values = torch.arange(candidate_count, device=child_prob_one.device, dtype=torch.long)
        shifts = torch.arange(child_depth - 1, -1, -1, device=child_prob_one.device)
        suffix_bits = values[:, None].bitwise_right_shift(shifts).bitwise_and(1)
        probability = child_prob_one.clamp(1e-6, 1.0 - 1e-6)
        log_probability = (
            suffix_bits.view(1, 1, 1, candidate_count, child_depth)
            * probability.unsqueeze(-2).log()
            + (1 - suffix_bits).view(1, 1, 1, candidate_count, child_depth)
            * (1.0 - probability).unsqueeze(-2).log()
        ).sum(dim=-1)
        topk = min(self.soft_child_mmse_topk, candidate_count)
        topk_log_probability, child_values = torch.topk(log_probability, k=topk, dim=-1)
        weights = torch.softmax(topk_log_probability, dim=-1)
        parent_shifts = torch.arange(
            self.parent_depth - 1, -1, -1, device=parent_bits.device
        )
        parent_value = (parent_bits.to(torch.long) * (1 << parent_shifts)).sum(dim=-1)
        binary_leaf_value = parent_value.unsqueeze(-1) * candidate_count + child_values
        leaf_indices = binary_leaf_value + ((1 << depth) - 1)
        entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=-1).mean()
        hard_mass = weights[..., 0].mean()
        return leaf_indices, weights, entropy, hard_mass

    @staticmethod
    def path_bits_to_leaf_indices(path_bits: torch.Tensor) -> torch.Tensor:
        if path_bits.ndim != 4:
            raise ValueError(f"path_bits must be [D,B,H,W], got {tuple(path_bits.shape)}")
        _, batch_size, height, width = path_bits.shape
        indices = torch.zeros((batch_size, height, width), device=path_bits.device, dtype=torch.long)
        for depth in range(path_bits.shape[0]):
            indices = 2 * indices + 1 + path_bits[depth].to(torch.long)
        return indices

    def _pack_parent_into_reliable_positions(
        self, token_bits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Place parent bits in the empirically reliable Gray-16QAM positions 0/2."""
        batch_size, height, width, depth = token_bits.shape
        token_count = height * width
        parent_bits = token_bits[..., :self.parent_depth].reshape(batch_size, -1)
        child_bits = token_bits[..., self.parent_depth:].reshape(batch_size, -1)
        bit_count = token_count * depth
        padded_count = ((bit_count + 3) // 4) * 4

        if self.explicit_depth_order is not None:
            if len(self.explicit_depth_order) != depth:
                raise ValueError(
                    "explicit_depth_order must contain exactly one entry per tree depth"
                )
            if set(self.explicit_depth_order) != set(range(depth)):
                raise ValueError(
                    "explicit_depth_order must be a zero-based depth permutation"
                )

            order = torch.tensor(
                self.explicit_depth_order, device=token_bits.device, dtype=torch.long
            )
            ordered_bits = token_bits.index_select(-1, order).reshape(batch_size, -1)
            packed = torch.nn.functional.pad(
                ordered_bits, (0, padded_count - bit_count), value=0
            )
            valid = (
                torch.arange(padded_count, device=token_bits.device)[None, :]
                < bit_count
            )

            positions_by_depth = torch.empty(
                depth, device=token_bits.device, dtype=torch.long
            )
            positions_by_depth[order] = torch.arange(
                depth, device=token_bits.device
            )
            token_offsets = (
                torch.arange(token_count, device=token_bits.device)[:, None] * depth
            )
            parent_slots = (
                token_offsets + positions_by_depth[:self.parent_depth][None, :]
            ).reshape(-1)
            child_slots = (
                token_offsets + positions_by_depth[self.parent_depth:][None, :]
            ).reshape(-1)

            parent_positions = parent_slots.remainder(4)
            parent_reliable = (parent_positions == 0) | (parent_positions == 2)
            if not bool(parent_reliable.all()):
                raise ValueError(
                    "explicit_depth_order must keep every parent bit in "
                    "Gray-16QAM reliable positions 0/2"
                )

            symbols = self._modulate(packed.view(batch_size, -1, 4))
            return packed, valid, symbols, parent_slots, child_slots

        slots = torch.arange(padded_count, device=token_bits.device)
        reliable = (slots.remainder(4) == 0) | (slots.remainder(4) == 2)
        parent_slots = slots[reliable][:parent_bits.shape[1]]
        if parent_slots.numel() != parent_bits.shape[1]:
            raise ValueError("not enough reliable QAM positions for all parent bits")

        remaining = torch.ones(padded_count, device=token_bits.device, dtype=torch.bool)
        remaining[parent_slots] = False
        child_slots = slots[remaining][:child_bits.shape[1]]
        if child_slots.numel() != child_bits.shape[1]:
            raise ValueError("not enough remaining QAM positions for all child bits")

        packed = token_bits.new_zeros((batch_size, padded_count))
        packed[:, parent_slots] = parent_bits
        packed[:, child_slots] = child_bits
        valid = torch.zeros((1, padded_count), device=token_bits.device, dtype=torch.bool)
        valid[:, parent_slots] = True
        valid[:, child_slots] = True
        symbols = self._modulate(packed.view(batch_size, -1, 4))
        return packed, valid, symbols, parent_slots, child_slots

    def forward(
        self,
        path_bits: torch.Tensor,
        training: bool,
        snr_db: torch.Tensor | float | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if path_bits.ndim != 4:
            raise ValueError(f"path_bits must be [D,B,H,W], got {tuple(path_bits.shape)}")

        depth, batch_size, height, width = path_bits.shape
        if self.split_parent_child and self.parent_depth >= depth:
            raise ValueError(f"parent_depth={self.parent_depth} must be smaller than tree depth={depth}")

        def pack_stream(stream_bits: torch.Tensor):
            bit_count = int(stream_bits.shape[1])
            padded_count = ((bit_count + 3) // 4) * 4
            padded = torch.nn.functional.pad(stream_bits, (0, padded_count - bit_count), value=0)
            valid = torch.arange(padded_count, device=path_bits.device)[None, :] < bit_count
            return padded, valid, self._modulate(padded.view(batch_size, -1, 4))

        token_bits = path_bits.permute(1, 2, 3, 0).to(torch.long)
        tx_bits = token_bits.reshape(batch_size, -1)
        if self.parent_reliable_bit_mapping:
            (
                tx_bits_padded,
                valid_mask,
                tx_symbols,
                parent_slots,
                child_slots,
            ) = self._pack_parent_into_reliable_positions(token_bits)
            parent_bits = token_bits[..., :self.parent_depth].reshape(batch_size, -1)
        elif self.split_parent_child:
            parent_bits = token_bits[..., :self.parent_depth].reshape(batch_size, -1)
            child_bits = token_bits[..., self.parent_depth:].reshape(batch_size, -1)
            parent_padded, parent_valid, parent_symbols = pack_stream(parent_bits)
            child_padded, child_valid, child_symbols = pack_stream(child_bits)
            tx_bits_padded = torch.cat([parent_padded, child_padded], dim=1)
            valid_mask = torch.cat([parent_valid, child_valid], dim=1)
            tx_symbols = torch.cat([parent_symbols, child_symbols], dim=1)
        else:
            tx_bits_padded, valid_mask, tx_symbols = pack_stream(tx_bits)
        if snr_db is None:
            snr_db_tensor = self._sample_snr_db(path_bits.device, training=training)
        else:
            snr_db_tensor = torch.as_tensor(snr_db, device=path_bits.device, dtype=torch.float32)

        data_symbol_count = int(tx_symbols.shape[1])
        redundancy_symbol_count = 0
        parent_bit_count = 0
        parity_bits = None
        if self.parent_repetition_factor > 1:
            if self.parent_hamming84_softml_enabled:
                redundancy_symbols, parity_bits = self._pack_parent_hamming84_parity(
                    parent_bits
                )
                parent_bit_count = int(parent_bits.shape[1])
            else:
                redundancy_symbols, parent_bit_count = self._pack_parent_repetitions(
                    parent_bits
                )
            redundancy_symbol_count = int(redundancy_symbols.shape[1])
            data_symbol_power, redundancy_symbol_power = self._symbol_power_allocation(
                snr_db_tensor, data_symbol_count, redundancy_symbol_count
            )
            data_symbol_gain = torch.sqrt(data_symbol_power)
            redundancy_symbol_gain = torch.sqrt(redundancy_symbol_power)
            physical_symbols = torch.cat(
                [
                    tx_symbols * data_symbol_gain,
                    redundancy_symbols * redundancy_symbol_gain,
                ],
                dim=1,
            )
        else:
            data_symbol_power, redundancy_symbol_power = self._symbol_power_allocation(
                snr_db_tensor, data_symbol_count, redundancy_symbol_count
            )
            data_symbol_gain = torch.sqrt(data_symbol_power)
            redundancy_symbol_gain = torch.sqrt(redundancy_symbol_power)
            physical_symbols = tx_symbols

        (
            rx_physical_symbols,
            csi_nmse,
            effective_noise_variance,
            receiver_pilot_uncertainty,
        ) = self._apply_channel(physical_symbols, snr_db_tensor)
        rx_data_physical = rx_physical_symbols[:, :data_symbol_count]
        rx_redundancy_physical = rx_physical_symbols[:, data_symbol_count:]
        rx_symbols = rx_data_physical / data_symbol_gain
        receiver_bit_probabilities = self._qam_bit_probabilities(
            rx_symbols, effective_noise_variance / data_symbol_power
        ).reshape(batch_size, -1)
        receiver_bit_confidence = (
            2.0 * (receiver_bit_probabilities - 0.5).abs()
        ).clamp(0.0, 1.0)
        receiver_qam_confidence = receiver_bit_confidence.mean(dim=1)
        receiver_parent_confidence = path_bits.new_zeros(
            (batch_size,), dtype=torch.float32
        )
        adaptive_r2_request_rate = path_bits.new_zeros((), dtype=torch.float32)
        adaptive_r3_request_rate = path_bits.new_zeros((), dtype=torch.float32)
        adaptive_feedback_bits_per_image = path_bits.new_zeros(
            (), dtype=torch.float32
        )
        adaptive_block_count = path_bits.new_zeros((), dtype=torch.float32)
        adaptive_requested_block_count = path_bits.new_zeros(
            (), dtype=torch.float32
        )
        actual_redundancy_symbols_per_image = path_bits.new_tensor(
            float(redundancy_symbol_count), dtype=torch.float32
        )
        mean_parent_repetition_factor = path_bits.new_tensor(
            float(self.parent_repetition_factor), dtype=torch.float32
        )
        raw_parent_ber = path_bits.new_zeros((), dtype=torch.float32)
        parent_ecc_correction_rate = path_bits.new_zeros((), dtype=torch.float32)
        parent_parity_ber = path_bits.new_zeros((), dtype=torch.float32)
        if self.parent_reliable_bit_mapping:
            receiver_parent_confidence = receiver_bit_confidence[
                :, parent_slots
            ].mean(dim=1)
            rx_bits_padded = self._demodulate(rx_symbols).reshape(batch_size, -1)
            parent_rx_raw = rx_bits_padded[:, parent_slots]
            raw_parent_ber = (
                parent_rx_raw != parent_bits
            ).to(torch.float32).mean()
            if self.parent_repetition_factor > 1:
                if self.parent_hamming84_softml_enabled:
                    data_noise_variance = effective_noise_variance / data_symbol_power
                    data_prob_one = self._qam_bit_probabilities(
                        rx_symbols, data_noise_variance
                    ).reshape(batch_size, -1)[:, parent_slots]
                    parity_rx_symbols = (
                        rx_redundancy_physical / redundancy_symbol_gain
                    )
                    parity_noise_variance = (
                        effective_noise_variance / redundancy_symbol_power
                    )
                    parity_coordinate = parity_rx_symbols.reshape(batch_size, -1)[
                        :, :parent_bit_count
                    ]
                    qpsk_amplitude = torch.rsqrt(
                        parity_coordinate.new_tensor(2.0)
                    )
                    parity_llr = (
                        4.0
                        * qpsk_amplitude
                        * parity_coordinate
                        / parity_noise_variance
                    )
                    parity_prob_one = torch.sigmoid(parity_llr)
                    parent_rx = self._decode_parent_hamming84_softml(
                        data_prob_one, parity_prob_one
                    )
                    parity_rx = (parity_coordinate >= 0.0).to(torch.long)
                    parent_parity_ber = (
                        parity_rx != parity_bits
                    ).to(torch.float32).mean()
                    parent_ecc_correction_rate = (
                        parent_rx != parent_rx_raw
                    ).to(torch.float32).mean()
                else:
                    symbol_indices = torch.div(parent_slots, 4, rounding_mode="floor")
                    axis_indices = torch.div(
                        parent_slots.remainder(4), 2, rounding_mode="floor"
                    )
                    data_evidence = rx_data_physical[:, symbol_indices, axis_indices]
                    extra_copies = self.parent_repetition_factor - 1
                    qpsk_symbols_per_copy = redundancy_symbol_count // extra_copies
                    repeated_evidence = rx_redundancy_physical.view(
                        batch_size, extra_copies, qpsk_symbols_per_copy, 2
                    ).reshape(batch_size, extra_copies, -1)[..., :parent_bit_count]

                    # Normalize each observation by its expected absolute coordinate.
                    data_scale = torch.sqrt(data_evidence.new_tensor(10.0)) / 2.0
                    repeat_scale = torch.sqrt(repeated_evidence.new_tensor(2.0))
                    normalized_data_evidence = (
                        data_evidence * data_scale * data_symbol_gain
                    )
                    normalized_repeated_evidence = (
                        repeated_evidence
                        * repeat_scale
                        * redundancy_symbol_gain
                    )
                    if self.adaptive_parent_ir_enabled:
                        noise_scale = torch.sqrt(
                            effective_noise_variance
                        ).clamp_min(1e-6)
                        pilot_risk = receiver_pilot_uncertainty / (
                            receiver_pilot_uncertainty + 0.05
                        )
                        data_confidence = torch.tanh(
                            normalized_data_evidence.abs() / noise_scale
                        ).mean(dim=1)
                        effective_data_confidence = data_confidence * torch.exp(
                            -self.adaptive_parent_ir_pilot_weight * pilot_risk
                        )
                        if self.adaptive_parent_ir_policy == "fixed_r2":
                            request_r2 = torch.ones(
                                batch_size,
                                device=path_bits.device,
                                dtype=torch.bool,
                            )
                            request_r3 = torch.zeros_like(request_r2)
                        elif self.adaptive_parent_ir_policy == "fixed_r3":
                            request_r2 = torch.ones(
                                batch_size,
                                device=path_bits.device,
                                dtype=torch.bool,
                            )
                            request_r3 = torch.ones_like(request_r2)
                        elif self.adaptive_parent_ir_policy in {
                            "adaptive_blockwise",
                            "random_blockwise",
                            "topk_blockwise",
                            "random_topk_blockwise",
                        }:
                            # Keep r2 as a fixed floor.  The policy only decides
                            # which parent-token blocks receive the third copy.
                            ir_allowed = (
                                snr_db_tensor
                                <= self.adaptive_parent_ir_max_snr_db
                            ).reshape(1).expand(batch_size)
                            request_r2 = torch.ones(
                                batch_size,
                                device=path_bits.device,
                                dtype=torch.bool,
                            )
                            token_count = parent_bit_count // self.parent_depth
                            block_tokens = min(
                                self.adaptive_parent_ir_block_tokens, token_count
                            )
                            if token_count % block_tokens:
                                raise ValueError(
                                    "parent token count must divide the configured "
                                    "adaptive IR block size"
                                )
                            block_bits = block_tokens * self.parent_depth
                            block_count = token_count // block_tokens
                            adaptive_block_count = path_bits.new_tensor(
                                float(block_count), dtype=torch.float32
                            )
                            evidence_after_r2 = (
                                normalized_data_evidence
                                + normalized_repeated_evidence[:, 0]
                            )
                            bit_confidence = torch.tanh(
                                evidence_after_r2.abs() / noise_scale
                            )
                            block_confidence = bit_confidence.reshape(
                                batch_size, block_count, block_bits
                            ).mean(dim=-1)
                            effective_block_confidence = block_confidence * torch.exp(
                                -self.adaptive_parent_ir_pilot_weight
                                * pilot_risk[:, None]
                            )
                            if self.adaptive_parent_ir_policy in {
                                "topk_blockwise",
                                "random_topk_blockwise",
                            }:
                                requested = int(
                                    round(
                                        block_count
                                        * self.adaptive_parent_ir_topk_request_rate
                                    )
                                )
                                request_count = torch.full(
                                    (batch_size, 1),
                                    requested,
                                    device=path_bits.device,
                                    dtype=torch.long,
                                )
                                request_count = request_count * ir_allowed[:, None]
                                confidence_rank = torch.argsort(
                                    effective_block_confidence, dim=-1
                                ).argsort(dim=-1)
                                request_blocks = (
                                    confidence_rank < request_count
                                ) & ir_allowed[:, None]
                            else:
                                risky_blocks = (
                                    effective_block_confidence
                                    < self.adaptive_parent_ir_r3_threshold
                                ) & ir_allowed[:, None]
                                request_count = risky_blocks.sum(
                                    dim=-1, keepdim=True
                                )
                                request_blocks = risky_blocks
                            if self.adaptive_parent_ir_policy in {
                                "random_blockwise",
                                "random_topk_blockwise",
                            }:
                                random_rank = torch.argsort(
                                    torch.rand_like(block_confidence), dim=-1
                                ).argsort(dim=-1)
                                request_blocks = (
                                    random_rank < request_count
                                ) & ir_allowed[:, None]
                            adaptive_requested_block_count = (
                                request_blocks.to(torch.float32).sum(dim=-1).mean()
                            )
                            request_r3 = request_blocks.repeat_interleave(
                                block_bits, dim=-1
                            )
                            adaptive_feedback_bits_per_image = (
                                ir_allowed.to(torch.float32)
                                * float(block_count)
                                * float(
                                    self.adaptive_parent_ir_feedback_repetitions
                                )
                            )
                        else:
                            ir_allowed = (
                                snr_db_tensor
                                <= self.adaptive_parent_ir_max_snr_db
                            )
                            if self.adaptive_parent_ir_min_repetition_factor == 2:
                                request_r2 = torch.ones(
                                    batch_size,
                                    device=path_bits.device,
                                    dtype=torch.bool,
                                )
                            else:
                                request_r2 = (
                                    effective_data_confidence
                                    < self.adaptive_parent_ir_r2_threshold
                                ) & ir_allowed
                            evidence_after_r2 = (
                                normalized_data_evidence
                                + request_r2[:, None].to(
                                    normalized_repeated_evidence.dtype
                                )
                                * normalized_repeated_evidence[:, 0]
                            )
                            confidence_after_r2 = torch.tanh(
                                evidence_after_r2.abs() / noise_scale
                            ).mean(dim=1)
                            effective_confidence_after_r2 = (
                                confidence_after_r2
                                * torch.exp(
                                    -self.adaptive_parent_ir_pilot_weight
                                    * pilot_risk
                                )
                            )
                            request_r3 = request_r2 & (
                                effective_confidence_after_r2
                                < self.adaptive_parent_ir_r3_threshold
                            ) & ir_allowed
                            if self.adaptive_parent_ir_min_repetition_factor == 2:
                                adaptive_feedback_bits_per_image = (
                                    ir_allowed.to(torch.float32)
                                )
                            else:
                                adaptive_feedback_bits_per_image = (
                                    ir_allowed.to(torch.float32)
                                    * (
                                        1.0
                                        + request_r2.to(torch.float32).mean()
                                    )
                                )

                        adaptive_r2_request_rate = request_r2.to(
                            torch.float32
                        ).mean()
                        adaptive_r3_request_rate = request_r3.to(
                            torch.float32
                        ).mean()
                        actual_redundancy_symbols_per_image = (
                            adaptive_r2_request_rate
                            + adaptive_r3_request_rate
                        ) * float(qpsk_symbols_per_copy)
                        mean_parent_repetition_factor = (
                            1.0
                            + adaptive_r2_request_rate
                            + adaptive_r3_request_rate
                        )
                        request_r3_evidence = (
                            request_r3
                            if request_r3.ndim == 2
                            else request_r3[:, None]
                        )
                        combined_evidence = (
                            normalized_data_evidence
                            + request_r2[:, None].to(
                                normalized_repeated_evidence.dtype
                            )
                            * normalized_repeated_evidence[:, 0]
                            + request_r3_evidence.to(
                                normalized_repeated_evidence.dtype
                            )
                            * normalized_repeated_evidence[:, 1]
                        )
                    else:
                        combined_evidence = (
                            normalized_data_evidence
                            + normalized_repeated_evidence.sum(dim=1)
                        )
                    receiver_parent_confidence = torch.tanh(
                        combined_evidence.abs()
                        / torch.sqrt(effective_noise_variance).clamp_min(1e-6)
                    ).mean(dim=1)
                    parent_rx = (combined_evidence >= 0.0).to(torch.long)
                rx_bits_padded[:, parent_slots] = parent_rx
            else:
                parent_rx = parent_rx_raw
            child_rx = rx_bits_padded[:, child_slots]
            soft_leaf_indices = None
            soft_leaf_weights = None
            soft_receiver_entropy = path_bits.new_zeros((), dtype=torch.float32)
            soft_receiver_hard_mass = path_bits.new_ones((), dtype=torch.float32)
            if self.soft_child_mmse_enabled or self.semantic_medoid_bayes_enabled:
                data_noise_variance = effective_noise_variance / data_symbol_power
                bit_probabilities = self._qam_bit_probabilities(rx_symbols, data_noise_variance)
                child_prob_one = bit_probabilities.reshape(batch_size, -1)[:, child_slots]
                child_prob_one = child_prob_one.view(
                    batch_size, height, width, depth - self.parent_depth
                )
                parent_token_bits = parent_rx.view(
                    batch_size, height, width, self.parent_depth
                )
                (
                    soft_leaf_indices, soft_leaf_weights, soft_receiver_entropy,
                    soft_receiver_hard_mass,
                ) = self._soft_child_leaf_candidates(
                    parent_token_bits, child_prob_one, depth
                )
            rx_token_bits = torch.cat(
                [
                    parent_rx.view(batch_size, height, width, self.parent_depth),
                    child_rx.view(batch_size, height, width, depth - self.parent_depth),
                ],
                dim=-1,
            )
            rx_bits = rx_token_bits.reshape(batch_size, -1)
        elif self.split_parent_child:
            parent_symbol_count = parent_symbols.shape[1]
            parent_rx_padded = self._demodulate(rx_symbols[:, :parent_symbol_count]).reshape(batch_size, -1)
            child_rx_padded = self._demodulate(rx_symbols[:, parent_symbol_count:]).reshape(batch_size, -1)
            parent_rx = parent_rx_padded[:, :parent_bits.shape[1]]
            child_rx = child_rx_padded[:, :child_bits.shape[1]]
            rx_token_bits = torch.cat(
                [
                    parent_rx.view(batch_size, height, width, self.parent_depth),
                    child_rx.view(batch_size, height, width, depth - self.parent_depth),
                ],
                dim=-1,
            )
            rx_bits_padded = torch.cat([parent_rx_padded, child_rx_padded], dim=1)
            rx_bits = rx_token_bits.reshape(batch_size, -1)
        else:
            rx_bits_padded = self._demodulate(rx_symbols).reshape(batch_size, -1)
            rx_bits = rx_bits_padded[:, :tx_bits.shape[1]]
            rx_token_bits = rx_bits.view(batch_size, height, width, depth)
        rx_path_bits = rx_token_bits.permute(3, 0, 1, 2).contiguous()

        tx_leaf = self.path_bits_to_leaf_indices(path_bits)
        rx_leaf = self.path_bits_to_leaf_indices(rx_path_bits)
        bit_errors = tx_bits != rx_bits
        depth_ber = (path_bits != rx_path_bits).to(torch.float32).mean(dim=(1, 2, 3))

        qam_position_ber = []
        padded_errors = tx_bits_padded != rx_bits_padded
        for position in range(4):
            position_mask = valid_mask[:, position::4]
            position_errors = padded_errors[:, position::4]
            qam_position_ber.append(
                position_errors[position_mask.expand_as(position_errors)].to(torch.float32).mean()
            )

        pilot_symbols = 0
        if self.channel_type == "fading" and self.csi_mode == "estimated":
            pilot_symbols = self.pilot_symbols_per_image

        metrics = {
            "snr_db": snr_db_tensor.detach(),
            "ber": bit_errors.to(torch.float32).mean().detach(),
            "ser": (tx_leaf != rx_leaf).to(torch.float32).mean().detach(),
            "parent_ber": depth_ber[:min(self.parent_depth, depth)].mean().detach(),
            "child_ber": depth_ber[min(self.parent_depth, depth):].mean().detach()
            if self.parent_depth < depth else depth_ber.new_zeros(()),
            "ber_by_depth": depth_ber.detach(),
            "ber_by_qam_bit_position": torch.stack(qam_position_ber).detach(),
            "csi_nmse": csi_nmse.detach(),
            "receiver_pilot_uncertainty": receiver_pilot_uncertainty.detach(),
            "receiver_qam_confidence": receiver_qam_confidence.detach(),
            "receiver_parent_confidence": receiver_parent_confidence.detach(),
            "source_bits_per_token": path_bits.new_tensor(float(depth), dtype=torch.float32),
            "data_symbols_per_image": path_bits.new_tensor(float(data_symbol_count), dtype=torch.float32),
            "redundancy_symbols_per_image": actual_redundancy_symbols_per_image.detach(),
            "max_redundancy_symbols_per_image": path_bits.new_tensor(
                float(redundancy_symbol_count), dtype=torch.float32
            ),
            "pilot_symbols_per_image": path_bits.new_tensor(float(pilot_symbols), dtype=torch.float32),
            "raw_parent_ber": raw_parent_ber.detach(),
            "parent_hamming84_softml_enabled": path_bits.new_tensor(
                float(self.parent_hamming84_softml_enabled), dtype=torch.float32
            ),
            "parent_ecc_correction_rate": parent_ecc_correction_rate.detach(),
            "parent_parity_ber": parent_parity_ber.detach(),
            "parent_repetition_factor": path_bits.new_tensor(
                float(self.parent_repetition_factor), dtype=torch.float32
            ),
            "mean_parent_repetition_factor": mean_parent_repetition_factor.detach(),
            "adaptive_parent_ir_enabled": path_bits.new_tensor(
                float(self.adaptive_parent_ir_enabled), dtype=torch.float32
            ),
            "adaptive_parent_ir_min_repetition_factor": path_bits.new_tensor(
                float(self.adaptive_parent_ir_min_repetition_factor),
                dtype=torch.float32,
            ),
            "adaptive_parent_ir_max_snr_db": path_bits.new_tensor(
                float(self.adaptive_parent_ir_max_snr_db),
                dtype=torch.float32,
            ),
            "adaptive_r2_request_rate": adaptive_r2_request_rate.detach(),
            "adaptive_r3_request_rate": adaptive_r3_request_rate.detach(),
            "adaptive_feedback_bits_per_image": adaptive_feedback_bits_per_image.detach(),
            "adaptive_block_count": adaptive_block_count.detach(),
            "adaptive_requested_block_count": (
                adaptive_requested_block_count.detach()
            ),
            "adaptive_parent_ir_topk_request_rate": path_bits.new_tensor(
                float(self.adaptive_parent_ir_topk_request_rate),
                dtype=torch.float32,
            ),
            "adaptive_parent_ir_feedback_repetitions": path_bits.new_tensor(
                float(self.adaptive_parent_ir_feedback_repetitions),
                dtype=torch.float32,
            ),
            "explicit_depth_order_enabled": path_bits.new_tensor(
                float(self.explicit_depth_order is not None), dtype=torch.float32
            ),
            "data_symbol_power": data_symbol_power.detach(),
            "redundancy_symbol_power": redundancy_symbol_power.detach(),
            "average_symbol_power": (
                (
                    data_symbol_count * data_symbol_power
                    + actual_redundancy_symbols_per_image
                    * redundancy_symbol_power
                )
                / (
                    data_symbol_count
                    + actual_redundancy_symbols_per_image
                ).clamp_min(1.0)
            ).detach(),
            "received_leaf_indices": rx_leaf,
            "soft_receiver_enabled": path_bits.new_tensor(
                float(self.soft_child_mmse_enabled), dtype=torch.float32
            ),
            "semantic_medoid_bayes_enabled": path_bits.new_tensor(
                float(self.semantic_medoid_bayes_enabled), dtype=torch.float32
            ),
            "soft_receiver_entropy": soft_receiver_entropy.detach()
            if self.parent_reliable_bit_mapping else path_bits.new_zeros((), dtype=torch.float32),
            "soft_receiver_hard_mass": soft_receiver_hard_mass.detach()
            if self.parent_reliable_bit_mapping else path_bits.new_ones((), dtype=torch.float32),
        }
        if self.parent_reliable_bit_mapping and (
            self.soft_child_mmse_enabled or self.semantic_medoid_bayes_enabled
        ):
            metrics["soft_leaf_indices"] = soft_leaf_indices
            metrics["soft_leaf_weights"] = soft_leaf_weights.detach()
        return rx_path_bits, metrics
