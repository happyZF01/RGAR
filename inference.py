#!/usr/bin/env python3
"""Run frozen RGAR image transmission over AWGN or Rayleigh fading."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from rgar import RGARInferenceModel


EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Image or image directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--channel",
        choices=("awgn", "fading-perfect", "fading-estimated"),
        default="awgn",
    )
    parser.add_argument(
        "--method",
        choices=("llr-topk", "fixed-r2", "fixed-r3"),
        default="llr-topk",
    )
    parser.add_argument("--snrs", default="0,5,10,15,20")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def image_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        item for item in path.rglob("*") if item.suffix.lower() in EXTENSIONS
    )


def load_image(path: Path, size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    scale = size / min(width, height)
    resized = (max(size, round(width * scale)), max(size, round(height * scale)))
    image = image.resize(resized, Image.Resampling.LANCZOS)
    left = (image.width - size) // 2
    top = (image.height - size) // 2
    image = image.crop((left, top, left + size, top + size))
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def save_image(tensor: torch.Tensor, path: Path) -> None:
    array = tensor.detach().clamp(-1, 1).add(1).mul(127.5)
    array = array.byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(array).save(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def topk_rate(snr: float) -> float:
    points = ((0.0, 0.8125), (5.0, 0.5625), (10.0, 0.1875))
    if snr <= points[0][0]:
        return points[0][1]
    if snr >= points[-1][0]:
        return points[-1][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= snr <= x1:
            return y0 + (snr - x0) * (y1 - y0) / (x1 - x0)
    raise RuntimeError("Invalid SNR schedule")


def configure_method(model: RGARInferenceModel, method: str, snr: float) -> None:
    channel = model.tree_path_channel
    channel.adaptive_parent_ir_policy = {
        "llr-topk": "topk_blockwise",
        "fixed-r2": "fixed_r2",
        "fixed-r3": "fixed_r3",
    }[method]
    channel.adaptive_parent_ir_topk_request_rate = topk_rate(snr)


def scalar(metrics: dict, name: str) -> float:
    value = metrics.get(name, 0.0)
    if torch.is_tensor(value):
        return float(value.detach().float().mean().cpu())
    return float(value)


def main() -> None:
    args = parse_args()
    paths = image_paths(Path(args.input))
    if not paths:
        raise RuntimeError(f"No images found under {args.input}")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = RGARInferenceModel(args.channel)
    model.load_checkpoint(args.checkpoint)
    model.to(device).eval()
    snrs = [float(item) for item in args.snrs.split(",")]
    rows = []

    for image_index, path in enumerate(paths):
        target = load_image(path, args.image_size).unsqueeze(0).to(device)
        latent, _, path_bits = model.encode_path(target)
        pixels = target.shape[-2] * target.shape[-1]
        for snr in snrs:
            configure_method(model, args.method, snr)
            set_seed(args.seed + image_index + 10000 * int(round(10 * snr)))
            _, metrics = model.tree_path_channel(
                path_bits, training=False, snr_db=snr
            )
            reconstruction = model.decode_indices(
                latent, metrics["received_leaf_indices"]
            )
            mse = F.mse_loss(
                (reconstruction + 1) / 2, (target + 1) / 2
            ).clamp_min(1.0e-12)
            psnr = float((-10 * torch.log10(mse)).cpu())
            suffix = str(snr).replace("-", "m").replace(".", "p")
            save_image(
                reconstruction[0], output / f"{path.stem}_snr_{suffix}dB.png"
            )
            data = scalar(metrics, "data_symbols_per_image")
            redundancy = scalar(metrics, "redundancy_symbols_per_image")
            pilot = scalar(metrics, "pilot_symbols_per_image")
            rows.append(
                {
                    "image": path.name,
                    "channel": args.channel,
                    "method": args.method,
                    "snr_db": snr,
                    "psnr": psnr,
                    "forward_cpp": (data + redundancy + pilot) / pixels,
                    "feedback_bits_per_pixel": scalar(
                        metrics, "adaptive_feedback_bits_per_image"
                    )
                    / pixels,
                    "ber": scalar(metrics, "ber"),
                    "parent_ber": scalar(metrics, "parent_ber"),
                }
            )
            print(f"{path.name} SNR={snr:g} dB PSNR={psnr:.3f}")

    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
