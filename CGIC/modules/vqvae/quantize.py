import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from torch import einsum
from typing import Optional, Tuple
import importlib


def _tree_num_nodes(max_depth: int) -> int:
    if max_depth < 0:
        raise ValueError(f"max_depth must be >= 0, got {max_depth}")
    return (1 << (max_depth + 1)) - 1


def _load_tree_vq_ext():
    try:
        return importlib.import_module("tree_vq_ext.tree_vq_ext._C")
    except Exception:
        return None


_TREE_VQ_EXT = _load_tree_vq_ext()


def _cuda_tree_search(inputs: torch.Tensor, codebook: torch.Tensor, max_depth: int) -> torch.Tensor:
    if _TREE_VQ_EXT is None:
        raise RuntimeError("tree_vq_ext is not available")
    if hasattr(_TREE_VQ_EXT, "tree_search_cuda_wrapper"):
        return _TREE_VQ_EXT.tree_search_cuda_wrapper(inputs, codebook, int(max_depth))
    if hasattr(_TREE_VQ_EXT, "tree_search"):
        return _TREE_VQ_EXT.tree_search(inputs, codebook, int(max_depth))
    raise AttributeError("tree_vq_ext missing tree_search or tree_search_cuda_wrapper")


def _default_shared_memory_limit(device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    props = torch.cuda.get_device_properties(device)
    for attr_name in ("shared_memory_per_block", "sharedMemPerBlock"):
        shared_limit = getattr(props, attr_name, None)
        if shared_limit is not None:
            return int(shared_limit)
    return 0


@torch.no_grad()
def _kmeans2(
    data: torch.Tensor,
    num_iters: int,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Runs a small K=2 k-means on `data`.

    Returns:
        centers: (2, C)
        labels: (N,) in {0,1}
        counts: (2,)
    """
    if data.ndim != 2:
        raise ValueError(f"data must be 2D (N,C), got shape {tuple(data.shape)}")
    num_points, dim = data.shape
    if num_points == 0:
        raise ValueError("kmeans2 received empty data")
    if num_points == 1:
        centers = torch.cat([data, data], dim=0)
        labels = torch.zeros((1,), device=data.device, dtype=torch.long)
        counts = torch.tensor([1, 0], device=data.device, dtype=torch.long)
        return centers, labels, counts

    if generator is None:
        generator = torch.Generator(device=data.device)
        generator.manual_seed(0)

    init_idx = torch.randint(0, num_points, (2,), generator=generator, device=data.device)
    centers = data[init_idx].clone()

    for _ in range(int(num_iters)):
        d0 = torch.mean((data - centers[0]) ** 2, dim=1)
        d1 = torch.mean((data - centers[1]) ** 2, dim=1)
        labels = (d1 < d0).long()

        counts0 = int((labels == 0).sum().item())
        counts1 = num_points - counts0
        if counts0 == 0 or counts1 == 0:
            labels = torch.randint(0, 2, (num_points,), generator=generator, device=data.device)
            counts0 = int((labels == 0).sum().item())
            counts1 = num_points - counts0
            if counts0 == 0 or counts1 == 0:
                labels = torch.zeros((num_points,), device=data.device, dtype=torch.long)
                counts0 = num_points
                counts1 = 0

        if counts0 > 0:
            centers[0] = data[labels == 0].mean(dim=0)
        if counts1 > 0:
            centers[1] = data[labels == 1].mean(dim=0)

    counts = torch.stack([(labels == 0).sum(), (labels == 1).sum()]).to(torch.long)
    return centers, labels, counts


@torch.no_grad()
def kmeans_init_tree(
    codebook: torch.Tensor,
    data: torch.Tensor,
    max_depth: int,
    num_iters: int = 10,
    generator: Optional[torch.Generator] = None,
) -> None:
    """Initializes a heap-layout binary tree codebook via recursive K=2 k-means.

    Critical detail (for pruning quality): every node stores the centroid of its assigned
    data subset before splitting.
    """
    if codebook.ndim != 2:
        raise ValueError(f"codebook must be 2D, got shape {tuple(codebook.shape)}")
    if data.ndim != 2:
        raise ValueError(f"data must be 2D, got shape {tuple(data.shape)}")
    if codebook.shape[1] != data.shape[1]:
        raise ValueError("codebook dim mismatch with data")

    expected_nodes = _tree_num_nodes(max_depth)
    if codebook.shape[0] != expected_nodes:
        raise ValueError(
            f"codebook has {codebook.shape[0]} nodes but max_depth={max_depth} requires {expected_nodes}"
        )
    if data.shape[0] == 0:
        raise ValueError("kmeans_init_tree requires non-empty data")

    if generator is None:
        generator = torch.Generator(device=data.device)
        generator.manual_seed(0)

    def _recurse(node_idx: int, subset: torch.Tensor, depth: int) -> None:
        codebook[node_idx].copy_(subset.mean(dim=0))
        if depth >= max_depth:
            return

        if subset.shape[0] < 2:
            left_idx = 2 * node_idx + 1
            right_idx = 2 * node_idx + 2
            if left_idx < codebook.shape[0]:
                codebook[left_idx].copy_(codebook[node_idx])
            if right_idx < codebook.shape[0]:
                codebook[right_idx].copy_(codebook[node_idx])
            return

        centers, labels, _ = _kmeans2(subset, num_iters=num_iters, generator=generator)

        left_idx = 2 * node_idx + 1
        right_idx = 2 * node_idx + 2
        if left_idx >= codebook.shape[0] or right_idx >= codebook.shape[0]:
            return
        codebook[left_idx].copy_(centers[0])
        codebook[right_idx].copy_(centers[1])

        _recurse(left_idx, subset[labels == 0], depth + 1)
        _recurse(right_idx, subset[labels == 1], depth + 1)

    _recurse(0, data, depth=0)

class VectorQuantize2(nn.Module):
    def __init__(self,
                n_e,
                e_dim,
                beta, 
                remap=None, 
                unknown_index="random",
                sane_index_shape=False, 
                legacy=True
                ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

        counter_dict = nn.ParameterDict({str(i): nn.Parameter(torch.zeros(1)) for i in range(n_e)})
        counter_dict.requires_grad_(False)
        if torch.cuda.is_available():
            counter_dict = counter_dict.cuda()
        self.embedding_counter = counter_dict

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)
    
    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)
    
    def forward(self, z):
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flatten = z.view(-1, self.e_dim)

        d = torch.sum(z_flatten ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flatten, rearrange(self.embedding.weight, 'n d -> d n'))
        
        # multi-grain z_indices calculation
        z_indices = torch.argmin(d, dim=1)
        if self.training:
            # Parallel counter update using bincount
            unique_indices, counts = torch.unique(z_indices, return_counts=True)
            for idx, count in zip(unique_indices, counts):
                self.embedding_counter[str(idx.item())].data += count.item()

        z_q = self.embedding(z_indices).view(z.shape)

        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        return z_q, loss, z_indices


class VectorizedTreeQuantizer(nn.Module):
    """Pruned Tree-Structured VQ (PTSVQ) with a heap-layout codebook.

    - Codebook is a complete binary tree stored in a flat array: node i -> children 2i+1, 2i+2.
    - Search is greedy (binary decision at each depth).
    - Supports BFOS-style pruning using a rate proxy R(d)=d (depth). Real bit-rate should
      be measured via an entropy coder on the chosen symbolization.
    """

    def __init__(
        self,
        max_depth: int,
        e_dim: int,
        beta: float = 0.25,
        alpha_decay: float = 1.0,
        rate_ema_decay: float = 0.99,
        rate_prob_floor: float = 1e-6,
        rate_softmax_temperature: float = 1.0,
        init_on_first_batch: bool = True,
        kmeans_iters: int = 10,
        use_cuda_tree: bool = True,
    ):
        super().__init__()
        self.max_depth = int(max_depth)
        self.e_dim = int(e_dim)
        self.beta = float(beta)
        self.alpha_decay = float(alpha_decay)
        self.rate_ema_decay = float(rate_ema_decay)
        self.rate_prob_floor = float(rate_prob_floor)
        self.rate_softmax_temperature = float(rate_softmax_temperature)
        self.init_on_first_batch = bool(init_on_first_batch)
        self.kmeans_iters = int(kmeans_iters)
        self.use_cuda_tree = bool(use_cuda_tree)

        self.n_e = _tree_num_nodes(self.max_depth)
        self.codebook = nn.Parameter(torch.empty(self.n_e, self.e_dim))
        self.reset_parameters()

        self.register_buffer("_initialized", torch.tensor(False), persistent=False)

        alphas = torch.ones(self.max_depth + 1, dtype=torch.float32)
        if self.alpha_decay != 1.0:
            depth_ids = torch.arange(self.max_depth + 1, dtype=torch.float32)
            alphas = (self.alpha_decay ** depth_ids).to(torch.float32)
        self.register_buffer("alphas", alphas, persistent=False)

        branch_count_ema = torch.ones(self.max_depth, 2, dtype=torch.float32)
        self.register_buffer("branch_count_ema", branch_count_ema)
        self.register_buffer("branch_count_updates", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def reset_parameters(self) -> None:
        self.codebook.data.uniform_(-1.0 / max(1, self.n_e), 1.0 / max(1, self.n_e))

    @torch.no_grad()
    def _maybe_init(self, z_flat: torch.Tensor) -> None:
        if not self.init_on_first_batch:
            return
        if bool(self._initialized.item()):
            return

        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            gathered = [torch.empty_like(z_flat) for _ in range(world_size)]
            dist.all_gather(gathered, z_flat)
            data = torch.cat(gathered, dim=0)
            if dist.get_rank() == 0:
                g = torch.Generator(device=data.device)
                g.manual_seed(0)
                kmeans_init_tree(self.codebook.data, data, max_depth=self.max_depth, num_iters=self.kmeans_iters, generator=g)
            dist.broadcast(self.codebook.data, src=0)
        else:
            g = torch.Generator(device=z_flat.device)
            g.manual_seed(0)
            kmeans_init_tree(self.codebook.data, z_flat, max_depth=self.max_depth, num_iters=self.kmeans_iters, generator=g)

        self._initialized.fill_(True)

    def _compute_path(self, z_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (indices_by_depth, path_bits, distortion_by_depth).

        - indices_by_depth: (D+1, N)
        - path_bits: (D, N) where 0=left, 1=right
        - distortion_by_depth: (D+1, N), using per-vector MSE (mean over channels)
        """
        num_points, dim = z_flat.shape
        if dim != self.e_dim:
            raise ValueError(f"Expected last dim {self.e_dim}, got {dim}")

        curr = torch.zeros((num_points,), device=z_flat.device, dtype=torch.long)
        indices_by_depth = [curr]
        path_bits = []
        distortions = []

        v_curr = F.embedding(curr, self.codebook)
        distortions.append(torch.mean((z_flat - v_curr) ** 2, dim=1))

        for _ in range(self.max_depth):
            left = 2 * curr + 1
            right = 2 * curr + 2

            v_left = F.embedding(left, self.codebook)
            v_right = F.embedding(right, self.codebook)

            dist_left = torch.mean((z_flat - v_left) ** 2, dim=1)
            dist_right = torch.mean((z_flat - v_right) ** 2, dim=1)

            go_right = dist_right < dist_left
            bit = go_right.to(torch.long)
            path_bits.append(bit)
            curr = torch.where(go_right, right, left)
            indices_by_depth.append(curr)

            v_curr = F.embedding(curr, self.codebook)
            distortions.append(torch.mean((z_flat - v_curr) ** 2, dim=1))

        indices_by_depth_t = torch.stack(indices_by_depth, dim=0)
        path_bits_t = torch.stack(path_bits, dim=0) if path_bits else torch.empty((0, num_points), device=z_flat.device, dtype=torch.long)
        distortions_t = torch.stack(distortions, dim=0)
        return indices_by_depth_t, path_bits_t, distortions_t

    def _can_use_cuda_tree(self, z_flat: torch.Tensor) -> bool:
        if not self.use_cuda_tree or _TREE_VQ_EXT is None:
            return False
        if self.max_depth < 1:
            return False
        if not (z_flat.is_cuda and self.codebook.is_cuda):
            return False
        if z_flat.dtype != torch.float32 or self.codebook.dtype != torch.float32:
            return False
        codebook_bytes = int(self.n_e) * int(self.e_dim) * int(self.codebook.element_size())
        if codebook_bytes > _default_shared_memory_limit(z_flat.device):
            return False
        return True

    @torch.no_grad()
    def reset_rate_statistics(self) -> None:
        """Reset EMA branch histograms used by the training-time rate proxy."""
        if self.branch_count_ema.numel() == 0:
            return
        self.branch_count_ema.fill_(1.0)
        self.branch_count_updates.zero_()

    def get_branch_probabilities(self) -> torch.Tensor:
        """Return per-depth branch probabilities used by the EMA-hist rate model."""
        if self.branch_count_ema.numel() == 0:
            return self.branch_count_ema

        counts = self.branch_count_ema.to(dtype=torch.float32)
        probs = counts + float(self.rate_prob_floor)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(float(self.rate_prob_floor))
        return probs

    @torch.no_grad()
    def update_branch_ema(self, path_bits: torch.Tensor) -> None:
        """Update per-depth branch usage histograms from hard routed path bits."""
        if self.branch_count_ema.numel() == 0 or path_bits.numel() == 0:
            return

        if path_bits.ndim == 4:
            bits = path_bits.reshape(path_bits.shape[0], -1).to(torch.float32)
        elif path_bits.ndim == 2:
            bits = path_bits.to(torch.float32)
        else:
            raise ValueError(f"path_bits must be [D,N] or [D,B,H,W], got {tuple(path_bits.shape)}")

        count_right = bits.sum(dim=1)
        count_total = torch.full_like(count_right, float(bits.shape[1]))
        batch_counts = torch.stack([count_total - count_right, count_right], dim=-1)

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_counts, op=dist.ReduceOp.SUM)

        self.branch_count_ema.mul_(self.rate_ema_decay).add_(batch_counts * (1.0 - self.rate_ema_decay))
        self.branch_count_updates.add_(1)

    def _build_depth_mask(
        self,
        depth_count: int,
        token_count: int,
        device: torch.device,
        effective_depth: Optional[torch.Tensor] = None,
        upto_depth: Optional[int] = None,
    ) -> torch.Tensor:
        if effective_depth is not None:
            depth_limit = effective_depth.reshape(-1).to(device=device, dtype=torch.long)
            if depth_limit.numel() != token_count:
                raise ValueError(
                    f"effective_depth has {depth_limit.numel()} elements but token_count={token_count}"
                )
            depth_ids = torch.arange(depth_count, device=device, dtype=torch.long).unsqueeze(1)
            return depth_ids < depth_limit.unsqueeze(0)

        if upto_depth is None:
            upto_depth = depth_count
        upto_depth = int(max(0, min(depth_count, upto_depth)))
        if upto_depth == 0:
            return torch.zeros((depth_count, token_count), device=device, dtype=torch.bool)
        depth_ids = torch.arange(depth_count, device=device, dtype=torch.long).unsqueeze(1)
        return depth_ids < upto_depth

    def estimate_rate_from_path_bits(
        self,
        path_bits: torch.Tensor,
        effective_depth: Optional[torch.Tensor] = None,
        upto_depth: Optional[int] = None,
    ) -> torch.Tensor:
        """Estimate hard routed code length in bits/token from EMA branch probabilities."""
        if path_bits.numel() == 0:
            return path_bits.new_zeros((), dtype=torch.float32)

        if path_bits.ndim == 4:
            bits = path_bits.reshape(path_bits.shape[0], -1).to(torch.long)
        elif path_bits.ndim == 2:
            bits = path_bits.to(torch.long)
        else:
            raise ValueError(f"path_bits must be [D,N] or [D,B,H,W], got {tuple(path_bits.shape)}")

        depth_count, token_count = bits.shape
        mask = self._build_depth_mask(
            depth_count=depth_count,
            token_count=token_count,
            device=bits.device,
            effective_depth=effective_depth,
            upto_depth=upto_depth,
        )
        probs = self.get_branch_probabilities().to(bits.device)[:depth_count]
        bit_cost = -torch.log2(probs.gather(1, bits.clamp(min=0, max=1)))
        total_cost = (bit_cost * mask.to(bit_cost.dtype)).sum(dim=0)
        return total_cost.mean()

    def estimate_soft_rate(
        self,
        z: torch.Tensor,
        path_indices_by_depth: torch.Tensor,
        effective_depth: Optional[torch.Tensor] = None,
        upto_depth: Optional[int] = None,
    ) -> torch.Tensor:
        """Differentiable EMA-hist rate proxy using soft left/right assignments."""
        if self.max_depth <= 0:
            return z.new_zeros(())
        if path_indices_by_depth.ndim != 4:
            raise ValueError(
                f"path_indices_by_depth must be [D+1,B,H,W], got {tuple(path_indices_by_depth.shape)}"
            )

        z_hw = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flat = z_hw.view(-1, self.e_dim)
        parent_indices = path_indices_by_depth[:-1].reshape(path_indices_by_depth.shape[0] - 1, -1).to(torch.long)

        depth_count, token_count = parent_indices.shape
        mask = self._build_depth_mask(
            depth_count=depth_count,
            token_count=token_count,
            device=z.device,
            effective_depth=effective_depth,
            upto_depth=upto_depth,
        )
        if not torch.any(mask):
            return z.new_zeros(())

        probs = self.get_branch_probabilities().to(z.device)[:depth_count]
        code_lengths = -torch.log2(probs)
        temperature = max(float(self.rate_softmax_temperature), 1e-6)

        total_cost = z.new_zeros((token_count,), dtype=z.dtype)
        for depth in range(depth_count):
            if not bool(mask[depth].any()):
                continue

            curr = parent_indices[depth]
            left = 2 * curr + 1
            right = 2 * curr + 2

            z_left = F.embedding(left, self.codebook)
            z_right = F.embedding(right, self.codebook)
            dist_left = torch.mean((z_flat - z_left) ** 2, dim=1)
            dist_right = torch.mean((z_flat - z_right) ** 2, dim=1)
            logits = torch.stack([-dist_left / temperature, -dist_right / temperature], dim=-1)
            soft_assign = torch.softmax(logits, dim=-1)
            cost_depth = torch.sum(soft_assign * code_lengths[depth].view(1, 2), dim=-1)
            total_cost = total_cost + cost_depth * mask[depth].to(cost_depth.dtype)

        return total_cost.mean()

    def nominal_rate_from_depth(
        self,
        effective_depth: Optional[torch.Tensor] = None,
        upto_depth: Optional[int] = None,
    ) -> float:
        """Return nominal branch-bit count per token from depth only."""
        if effective_depth is not None:
            return float(effective_depth.to(torch.float32).mean().item())
        if upto_depth is None:
            upto_depth = self.max_depth
        return float(max(0, min(int(upto_depth), self.max_depth)))

    def _greedy_path_at_depth(self, z_flat: torch.Tensor, target_depth: int):
        """Greedily route each token to a fixed target depth."""
        num_points = z_flat.shape[0]
        curr = torch.zeros((num_points,), device=z_flat.device, dtype=torch.long)
        indices_by_depth = [curr]
        path_bits = []

        for _ in range(int(target_depth)):
            left = 2 * curr + 1
            right = 2 * curr + 2

            v_left = F.embedding(left, self.codebook)
            v_right = F.embedding(right, self.codebook)
            dist_left = torch.mean((z_flat - v_left) ** 2, dim=1)
            dist_right = torch.mean((z_flat - v_right) ** 2, dim=1)

            go_right = dist_right < dist_left
            bit = go_right.to(torch.long)
            curr = torch.where(go_right, right, left)

            path_bits.append(bit)
            indices_by_depth.append(curr)

        indices_by_depth_t = torch.stack(indices_by_depth, dim=0)
        if path_bits:
            path_bits_t = torch.stack(path_bits, dim=0)
        else:
            path_bits_t = torch.empty((0, num_points), device=z_flat.device, dtype=torch.long)
        return indices_by_depth_t, path_bits_t

    def _beam_search_path_at_depth(
        self,
        z_flat: torch.Tensor,
        target_depth: int,
        beam_width: int,
        rerank: bool = False,
    ):
        """Beam-search tokens to a fixed target depth.

        Path score is the accumulated child distortion along the routed path.
        When ``rerank`` is ``True``, final beam candidates are reranked by direct
        token-to-centroid distortion at ``target_depth``.
        """
        num_points = z_flat.shape[0]
        beam_width = max(1, int(beam_width))

        curr_indices = torch.zeros((num_points, 1), device=z_flat.device, dtype=torch.long)
        curr_scores = torch.zeros((num_points, 1), device=z_flat.device, dtype=z_flat.dtype)
        curr_bits = torch.empty((0, num_points, 1), device=z_flat.device, dtype=torch.long)
        curr_nodes = torch.zeros((1, num_points, 1), device=z_flat.device, dtype=torch.long)

        for _ in range(int(target_depth)):
            left = 2 * curr_indices + 1
            right = 2 * curr_indices + 2

            z_flat_expanded = z_flat.unsqueeze(1)
            dist_left = torch.mean((z_flat_expanded - F.embedding(left, self.codebook)) ** 2, dim=2)
            dist_right = torch.mean((z_flat_expanded - F.embedding(right, self.codebook)) ** 2, dim=2)

            expanded_scores = torch.cat([curr_scores + dist_left, curr_scores + dist_right], dim=1)
            expanded_indices = torch.cat([left, right], dim=1)
            expanded_bits = torch.cat(
                [
                    torch.zeros_like(left, dtype=torch.long),
                    torch.ones_like(right, dtype=torch.long),
                ],
                dim=1,
            )

            beam_now = min(beam_width, expanded_scores.shape[1])
            top_scores, top_pos = torch.topk(expanded_scores, k=beam_now, dim=1, largest=False)
            curr_scores = top_scores
            curr_indices = expanded_indices.gather(1, top_pos)
            chosen_bits = expanded_bits.gather(1, top_pos)
            parent_pos = torch.div(top_pos, 2, rounding_mode='floor')

            if curr_bits.numel() == 0:
                curr_bits = chosen_bits.unsqueeze(0)
                curr_nodes = torch.cat(
                    [
                        curr_nodes.expand(-1, -1, beam_now),
                        curr_indices.unsqueeze(0),
                    ],
                    dim=0,
                )
            else:
                gather_bits = parent_pos.unsqueeze(0).expand(curr_bits.shape[0], -1, -1)
                curr_bits = torch.gather(curr_bits, 2, gather_bits)
                curr_bits = torch.cat([curr_bits, chosen_bits.unsqueeze(0)], dim=0)

                gather_nodes = parent_pos.unsqueeze(0).expand(curr_nodes.shape[0], -1, -1)
                curr_nodes = torch.gather(curr_nodes, 2, gather_nodes)
                curr_nodes = torch.cat([curr_nodes, curr_indices.unsqueeze(0)], dim=0)

        if rerank and curr_indices.shape[1] > 1:
            final_dist = torch.mean(
                (z_flat.unsqueeze(1) - F.embedding(curr_indices, self.codebook)) ** 2,
                dim=2,
            )
            best_pos = torch.argmin(final_dist, dim=1, keepdim=True)
            final_scores = final_dist.gather(1, best_pos)
        else:
            best_pos = torch.argmin(curr_scores, dim=1, keepdim=True)
            final_scores = curr_scores.gather(1, best_pos)

        selected_indices = curr_indices.gather(1, best_pos).squeeze(1)
        if curr_bits.numel() == 0:
            selected_bits = torch.empty((0, num_points), device=z_flat.device, dtype=torch.long)
        else:
            gather_bits = best_pos.unsqueeze(0).expand(curr_bits.shape[0], -1, -1)
            selected_bits = torch.gather(curr_bits, 2, gather_bits).squeeze(2)
        gather_nodes = best_pos.unsqueeze(0).expand(curr_nodes.shape[0], -1, -1)
        selected_nodes = torch.gather(curr_nodes, 2, gather_nodes).squeeze(2)

        candidate_info = {
            "final_candidate_scores": curr_scores.detach(),
            "selected_score": final_scores.squeeze(1).detach(),
        }
        return selected_nodes, selected_bits, candidate_info

    def infer_at_depth(
        self,
        z: torch.Tensor,
        target_depth: int,
        search_method: str = "greedy",
        beam_width: int = 1,
        return_path: bool = False,
    ):
        """Run depth-truncated tree inference for variable-rate evaluation.

        Args:
            z: Input latent tensor ``[B, C, H, W]``.
            target_depth: Fixed truncation depth. ``0`` selects the root.
            search_method: ``greedy``, ``beam``, or ``beam_rerank``.
            beam_width: Beam size for beam-based search methods.
            return_path: When ``True``, also returns depth/path metadata.

        Returns:
            ``(z_q, indices_flat, path_info)`` where ``z_q`` has shape ``[B, C, H, W]`` and
            ``indices_flat`` indexes the selected node at ``target_depth``.
        """
        target_depth = int(max(0, min(int(target_depth), self.max_depth)))
        search_method = str(search_method).lower()
        z_hw = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flat = z_hw.view(-1, self.e_dim)
        batch_size, _, height, width = z.shape

        if search_method == "greedy":
            indices_by_depth, path_bits = self._greedy_path_at_depth(z_flat, target_depth=target_depth)
            selected_indices = indices_by_depth[-1]
            candidate_info = None
        elif search_method in {"beam", "beam_rerank"}:
            indices_by_depth, path_bits, candidate_info = self._beam_search_path_at_depth(
                z_flat,
                target_depth=target_depth,
                beam_width=int(beam_width),
                rerank=(search_method == "beam_rerank"),
            )
            selected_indices = indices_by_depth[-1]
        else:
            raise ValueError(f"Unknown search_method: {search_method}")

        z_sel = F.embedding(selected_indices, self.codebook)
        z_q = z_flat + (z_sel - z_flat).detach()
        z_q = z_q.view(z_hw.shape)
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if not return_path:
            return z_q, selected_indices, None

        centroids_by_depth = F.embedding(indices_by_depth, self.codebook)
        centroids_by_depth = centroids_by_depth.view(target_depth + 1, batch_size, height, width, self.e_dim)
        centroids_by_depth = rearrange(centroids_by_depth, 'd b h w c -> d b c h w').contiguous()
        path_info = {
            "path_indices_by_depth": indices_by_depth.view(target_depth + 1, batch_size, height, width),
            "centroids_by_depth": centroids_by_depth,
            "selected_centroids": centroids_by_depth[-1],
            "path_bits": path_bits.view(target_depth, batch_size, height, width) if target_depth > 0 else path_bits,
            "effective_depth_per_token": torch.full(
                (batch_size, height, width),
                target_depth,
                device=z.device,
                dtype=torch.long,
            ),
            "nominal_bits_per_token": float(target_depth),
            "rate_info": {
                "nominal_bits_per_token": float(target_depth),
                "search_method": search_method,
                "beam_width": int(beam_width),
            },
            "candidate_info": candidate_info,
        }
        return z_q, selected_indices, path_info

    def forward(
        self,
        z: torch.Tensor,
        return_path_info: bool = False,
    ):
        """Quantize a latent map with the leaf centroid.

        Args:
            z: Input latent tensor of shape ``[B, C, H, W]``.
            return_path_info: When ``True``, also returns greedy routing metadata for
                hierarchical supervision:

                - ``path_indices_by_depth``: ``[D+1, B, H, W]``
                - ``centroids_by_depth``: ``[D+1, B, C, H, W]``
                - ``leaf_centroids``: ``[B, C, H, W]``
                - ``path_bits``: ``[D, B, H, W]``
                - ``effective_depth_per_token``: ``[B, H, W]``
                - ``rate_info``: lightweight depth/rate metadata

        Returns:
            If ``return_path_info`` is ``False``: ``(z_q, loss, leaf_indices)``.
            Otherwise: ``(z_q, loss, leaf_indices, path_info)``.
        """
        z_hw = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flat = z_hw.view(-1, self.e_dim)
        batch_size, _, height, width = z.shape

        if self.training:
            self._maybe_init(z_flat.detach())

        if not self.training and self._can_use_cuda_tree(z_flat) and not return_path_info:
            leaf_indices = _cuda_tree_search(z_flat, self.codebook, self.max_depth)
            z_leaf = F.embedding(leaf_indices, self.codebook)
            z_q = z_flat + (z_leaf - z_flat).detach()
            z_q = z_q.view(z_hw.shape)
            z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
            loss = z_leaf.new_zeros(())
            return z_q, loss, leaf_indices

        indices_by_depth, path_bits, _ = self._compute_path(z_flat)
        leaf_indices = indices_by_depth[-1]
        z_leaf = F.embedding(leaf_indices, self.codebook)

        # Multi-level codebook loss (updates nodes along the greedy path).
        codebook_loss = z_leaf.new_zeros(())
        z_sg = z_flat.detach()
        for depth in range(self.max_depth + 1):
            depth_indices = indices_by_depth[depth]
            z_depth = F.embedding(depth_indices, self.codebook)
            codebook_loss = codebook_loss + self.alphas[depth] * torch.mean((z_depth - z_sg) ** 2)

        # Commitment loss (updates encoder, against all nodes along the path).
        commit_loss = z_leaf.new_zeros(())
        for depth in range(self.max_depth + 1):
            depth_indices = indices_by_depth[depth]
            z_depth = F.embedding(depth_indices, self.codebook)
            commit_loss = commit_loss + self.alphas[depth] * torch.mean((z_flat - z_depth.detach()) ** 2)
        
        commit_loss = self.beta * commit_loss
        loss = codebook_loss + commit_loss

        z_q = z_flat + (z_leaf - z_flat).detach()
        z_q = z_q.view(z_hw.shape)
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        if not return_path_info:
            return z_q, loss, leaf_indices

        centroids_by_depth = F.embedding(indices_by_depth, self.codebook)
        centroids_by_depth = centroids_by_depth.view(self.max_depth + 1, batch_size, height, width, self.e_dim)
        centroids_by_depth = rearrange(centroids_by_depth, 'd b h w c -> d b c h w').contiguous()

        path_info = {
            "path_indices_by_depth": indices_by_depth.view(self.max_depth + 1, batch_size, height, width),
            "centroids_by_depth": centroids_by_depth,
            "leaf_centroids": centroids_by_depth[-1],
            "path_bits": path_bits.view(self.max_depth, batch_size, height, width) if self.max_depth > 0 else path_bits,
            "effective_depth_per_token": torch.full(
                (batch_size, height, width),
                int(self.max_depth),
                device=z.device,
                dtype=torch.long,
            ),
            "rate_info": {
                "nominal_bits_per_token": self.nominal_rate_from_depth(upto_depth=self.max_depth),
            },
        }
        return z_q, loss, leaf_indices, path_info

    @torch.no_grad()
    def get_optimal_subtree(
        self,
        z: torch.Tensor,
        lambda_val: float,
        return_path: bool = False,
    ):
        """BFOS-style pruning using rate proxy R(d)=d.

        The pruning decision uses per-step improvement S_d = D_parent - D_child and keeps
        descending while S_d > lambda_val.
        """
        z_hw = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flat = z_hw.view(-1, self.e_dim)

        indices_by_depth, path_bits, distortions = self._compute_path(z_flat)
        if self.max_depth == 0:
            effective_depth = torch.zeros((z_flat.shape[0],), device=z_flat.device, dtype=torch.long)
            final_indices = indices_by_depth[0]
        else:
            # BFOS-style pruning: minimize D + lambda * R
            # distortions: [D+1, N]
            # rates: [D+1, 1] -> 0, 1, ..., D
            rates = torch.arange(self.max_depth + 1, device=z_flat.device, dtype=distortions.dtype).unsqueeze(1)
            costs = distortions + float(lambda_val) * rates
            effective_depth = torch.argmin(costs, dim=0) # [N]
            final_indices = indices_by_depth.gather(0, effective_depth.unsqueeze(0)).squeeze(0)

        z_sel = F.embedding(final_indices, self.codebook)
        z_q = z_flat + (z_sel - z_flat).detach()
        z_q = z_q.view(z_hw.shape)
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if return_path:
            return z_q, final_indices, effective_depth, path_bits
        return z_q, final_indices, effective_depth

    @staticmethod
    def build_stop_bit_symbols(path_bits: torch.Tensor, effective_depth: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Builds a variable-length symbol stream with alphabet {0,1,STOP}.

        For each position i, emits `effective_depth[i]` branch bits then a STOP symbol.
        """
        stop_symbol = 2
        max_depth, num_points = path_bits.shape if path_bits.ndim == 2 else (0, int(effective_depth.numel()))
        if max_depth == 0:
            symbols = torch.full((num_points,), stop_symbol, device=effective_depth.device, dtype=torch.long)
            return symbols, 3

        depth = effective_depth.clamp(min=0, max=max_depth)
        lengths = depth + 1
        total_len = int(lengths.sum().item())
        symbols = torch.empty((total_len,), device=effective_depth.device, dtype=torch.long)

        offsets = torch.cumsum(lengths, dim=0) - lengths
        arange = torch.arange(max_depth, device=effective_depth.device, dtype=torch.long).unsqueeze(1)
        take = arange < depth.unsqueeze(0)

        # Fill bits
        bit_values = path_bits[take]
        bit_positions = (offsets.unsqueeze(0) + arange)[take]
        symbols[bit_positions] = bit_values

        # Fill STOP
        stop_positions = offsets + depth
        symbols[stop_positions] = stop_symbol
        return symbols, 3
