"""Input-only folder restoration. Metrics are computed separately by main_test.py."""
from __future__ import annotations
import config as settings  # configure CPU libraries before Torch
import argparse
import hashlib
import json
from pathlib import Path

import torch
from tqdm import tqdm
from models import FPIRNet
from utils import (autocast_context, device_for, image_map, load_image, load_weights,
                   save_image, sha256_file, write_json, environment_report)


def positions(length, tile, overlap):
    if length <= tile:
        return [0]
    values = list(range(0, length - tile + 1, tile - overlap))
    if values[-1] != length - tile:
        values.append(length - tile)
    return values


@torch.inference_mode()
def tiled_restore(model, image, iterations=4, tile=384, overlap=64, precision="bf16"):
    if tile < 1 or overlap < 0 or overlap >= tile:
        raise ValueError("Require tile > overlap >= 0")
    height, width = image.shape[-2:]
    if height <= tile and width <= tile:
        with autocast_context(image.device, precision):
            return model(image, iterations=iterations)["restored"].float()
    tile_h, tile_w = min(tile, height), min(tile, width)
    ys, xs = positions(height, tile_h, overlap), positions(width, tile_w, overlap)
    wy = torch.hann_window(tile_h, periodic=False, device=image.device).clamp_min(1e-3)
    wx = torch.hann_window(tile_w, periodic=False, device=image.device).clamp_min(1e-3)
    weight = (wy[:, None] * wx[None, :])[None, None]
    output, norm = torch.zeros_like(image, dtype=torch.float32), torch.zeros_like(image[:, :1], dtype=torch.float32)
    for top in ys:
        for left in xs:
            patch = image[..., top:top + tile_h, left:left + tile_w]
            with autocast_context(image.device, precision):
                restored = model(patch, iterations=iterations)["restored"].float()
            output[..., top:top + tile_h, left:left + tile_w] += restored * weight
            norm[..., top:top + tile_h, left:left + tile_w] += weight
    return output / norm.clamp_min(1e-6)


def export_checkpoint(source, destination, cfg, trusted=False, raw=False):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        raise ValueError("Export destination must differ from the training checkpoint")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite: {destination}")
    model = FPIRNet(cfg)
    payload, origin = load_weights(model, source, prefer_ema=not raw, trusted=trusted)
    count = sum(p.numel() for p in model.parameters())
    if count != 700271:
        raise ValueError(f"Final architecture should have 700271 parameters, got {count}")
    # No optimizer, logs or machine-specific absolute paths in the published file.
    release = {
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "architecture": "FPIR-Net", "iterations": 4, "fixed_step": 0.1268,
        "parameter_count": count, "source_weight_key": origin,
        "source_sha256": sha256_file(source),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".pth.tmp")
    torch.save(release, temporary)
    check = FPIRNet(cfg)
    load_weights(check, temporary)
    temporary.replace(destination)
    print(f"Exported {origin} -> {destination}\nParameters: {count}\nSHA256: {sha256_file(destination)}")


def main():
    defaults = settings.INFERENCE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=settings.CHECKPOINT)
    parser.add_argument("--input", "--input_root", dest="input_root", default=defaults["input_root"])
    parser.add_argument("--output", "--output_root", dest="output_root", default=defaults["output_root"])
    for name in ("iterations", "tile", "overlap"):
        parser.add_argument("--" + name, type=int, default=defaults[name])
    parser.add_argument("--device", default=defaults["device"])
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default=defaults["precision"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--raw-weights", action="store_true", help="Do not select EMA in a training checkpoint")
    parser.add_argument("--trusted-checkpoint", action="store_true")
    parser.add_argument("--export-checkpoint", metavar="SOURCE", help="Export EMA weights only, then exit (no image inference)")
    parser.add_argument("--export-output", default=settings.CHECKPOINT)
    args = parser.parse_args()
    cfg = settings.get_config()
    if args.export_checkpoint:
        export_checkpoint(args.export_checkpoint, args.export_output, cfg,
                          trusted=args.trusted_checkpoint, raw=args.raw_weights)
        return
    if args.iterations < 1 or not 0 <= args.overlap < args.tile:
        raise ValueError("Require iterations >= 1 and tile > overlap >= 0")
    source, target = Path(args.input_root).resolve(), Path(args.output_root).resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("Input and output directories must be separate, non-nested directories")
    images = image_map(source)
    signature = {
        "input_root": str(source), "checkpoint_sha256": sha256_file(args.checkpoint),
        "iterations": args.iterations, "tile": args.tile, "overlap": args.overlap,
        "precision": args.precision, "raw_weights": args.raw_weights,
        "source_fingerprint": hashlib.sha256(json.dumps([
            (k, p.stat().st_size, p.stat().st_mtime_ns) for k, p in images.items()
        ]).encode()).hexdigest(),
    }
    protocol_path = target / "inference.json"
    if args.resume:
        if not protocol_path.is_file():
            raise ValueError("Resume requires inference.json from the first run")
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        if previous["signature"] != signature:
            raise ValueError("Resume protocol/source changed. Use a new output directory.")
    elif target.exists() and any(target.rglob("*.png")):
        raise FileExistsError("Output already contains PNG images. Use --resume or a new output directory.")
    device = device_for(args.device)
    model = FPIRNet(cfg).to(device).eval()
    _, weight_source = load_weights(model, args.checkpoint, prefer_ema=not args.raw_weights,
                                    trusted=args.trusted_checkpoint)
    report = {"signature": signature, "weight_source": weight_source, "environment": environment_report(), "complete": False}
    write_json(report, protocol_path)
    processed = skipped = 0
    with torch.inference_mode():
        for key, path in tqdm(images.items(), desc="Inference", total=len(images)):
            image = load_image(path)
            output = target / (key + ".png")
            if args.resume and output.is_file():
                try:
                    if load_image(output).shape == image.shape:
                        skipped += 1
                        continue
                except RuntimeError:
                    pass  # Rewrite corrupted prediction; never ignore a corrupt input.
            restored = tiled_restore(model, image.unsqueeze(0).to(device), args.iterations,
                                     args.tile, args.overlap, args.precision)
            save_image(restored, output)
            processed += 1
    report.update(complete=True, processed=processed, resume_skipped=skipped, total=len(images))
    write_json(report, protocol_path)
    print(f"Saved {processed} images; reused {skipped}; output={target}")


if __name__ == "__main__":
    main()
