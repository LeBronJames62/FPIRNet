"""Train/validate FPIR-Net. This is NOT the saved-image test entry point."""
from __future__ import annotations
import config as settings  # CPU runtime settings before NumPy / Torch
import argparse
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Tuple
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import build_loader
from losses import restoration_loss, psnr, ssim
from models import FPIRNet
from simulation import TurbulenceSynthesizer
from utils import (ModelEMA, autocast_context, read_checkpoint, save_training_checkpoint,
                   seed_everything, setup_logger, write_json, device_for, environment_report)

def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)

def sample_conditions(batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    severity = 0.08 + 0.89 * torch.rand(batch_size, device=device)
    exposure = 0.10 + 0.85 * torch.rand(batch_size, device=device)
    return severity, exposure

def fixed_conditions(indices: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    index = indices.to(device=device, dtype=torch.float32)
    base = torch.remainder(index * 37.0 + 11.0, 997.0) / 996.0
    severity = 0.08 + 0.89 * torch.remainder(base * 1.73 + 0.19, 1.0)
    exposure = 0.10 + 0.85 * torch.remainder(base * 2.31 + 0.07, 1.0)
    return severity, exposure

def synthesize(clean, synthesizer, severity, exposure, seed: int):
    generator = torch.Generator(device=clean.device)
    generator.manual_seed(int(seed))
    return synthesizer(clean, severity=severity, exposure=exposure, generator=generator)

@torch.inference_mode()
def validate(cfg: Dict[str, Any], model, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    precision = str(cfg.get("train", {}).get("precision", "bf16"))
    synthesizer = TurbulenceSynthesizer(cfg.get("simulation", {})).to(device).eval()
    iterations = int(cfg.get("model", {}).get("default_iterations", 4))
    values = defaultdict(list)
    max_batches = cfg["data"]["val"].get("max_batches")
    for batch_index, batch in enumerate(tqdm(loader, desc="Validation", leave=False)):
        if max_batches is not None and batch_index >= int(max_batches):
            break
        clean = batch["clean"].to(device, non_blocking=True)
        indices = batch["index"] if isinstance(batch["index"], torch.Tensor) else torch.tensor(batch["index"])
        severity, exposure = fixed_conditions(indices, device)
        degraded = synthesize(
            clean,
            synthesizer,
            severity,
            exposure,
            seed=int(cfg.get("val_seed", 42)) * 100000 + batch_index * 101 + 23,
        )
        observation = degraded["input"]
        with autocast_context(device, precision):
            restored = model(observation, iterations=iterations)["restored"].float()
        values["input_psnr"].append(psnr(observation.float(), clean.float()).cpu())
        values["psnr"].append(psnr(restored, clean.float()).cpu())
        values["ssim"].append(ssim(restored, clean.float()).cpu())
    if not values["psnr"]:
        raise RuntimeError("Validation loader produced no samples")
    return {
        key: float(torch.cat(items).mean())
        for key, items in values.items()
    }

def train(
    cfg: Dict[str, Any],
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    run_dir: str | Path,
    resume: str | None = None,
    trusted: bool = False,
) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(cfg, run_dir / "config.json")
    logger = setup_logger(run_dir / "train.log", name=f"fpir_{run_dir.name}")
    train_cfg = cfg.get("train", {})
    precision = str(train_cfg.get("precision", "bf16"))
    max_steps = int(train_cfg.get("max_steps", 5000))
    epochs = int(train_cfg.get("epochs", 6))
    accumulation = int(train_cfg.get("accumulation_steps", 4))
    learning_rate = float(train_cfg.get("learning_rate", 2e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    warmup = int(train_cfg.get("warmup_steps", 250))
    minimum_ratio = float(train_cfg.get("min_lr_ratio", 0.05))
    clip = float(train_cfg.get("clip_grad_norm", 1.0))
    log_interval = int(train_cfg.get("log_interval", 25))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.99),
        weight_decay=weight_decay,
    )

    def schedule(step: int) -> float:
        if step < warmup:
            return max(1e-4, (step + 1) / max(1, warmup))
        progress = (step - warmup) / max(1, max_steps - warmup)
        return minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule)
    scaler = make_grad_scaler(device.type == "cuda" and precision == "fp16")
    ema = ModelEMA(model, decay=float(train_cfg.get("ema_decay", 0.999)))
    synthesizer = TurbulenceSynthesizer(cfg.get("simulation", {})).to(device)
    global_step = 0
    start_epoch = 0
    best_score = -float("inf")

    if resume:
        payload = read_checkpoint(resume, trusted=trusted, map_location=device)
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        if payload.get("scaler"):
            scaler.load_state_dict(payload["scaler"])
        global_step = int(payload.get("global_step", 0))
        start_epoch = int(payload.get("epoch", -1)) + 1
        best_score = float(payload.get("best_score", best_score))
        if payload.get("ema"):
            ema.shadow = {key: value.to(device) for key, value in payload["ema"].items()}
        logger.info("Resumed %s at optimizer step %d", resume, global_step)

    if global_step >= max_steps:
        logger.info("Requested optimizer-step budget already completed.")
        return

    model.train()
    optimizer.zero_grad(set_to_none=True)
    running = defaultdict(float)
    stop = False
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for iteration, batch in enumerate(progress):
            clean = batch["clean"].to(device, non_blocking=True)
            severity, exposure = sample_conditions(clean.shape[0], device)
            degraded = synthesize(
                clean,
                synthesizer,
                severity,
                exposure,
                seed=int(cfg.get("seed", 42)) * 1000000 + epoch * max(1, len(train_loader)) + iteration,
            )
            observation = degraded["input"]
            with autocast_context(device, precision):
                output = model(observation, iterations=model.default_iterations)
                losses = restoration_loss(cfg, output["restored"], clean)
                scaled_loss = losses["total"] / accumulation
            scaler.scale(scaled_loss).backward()
            for key, value in losses.items():
                running[key] += float(value.detach())

            if (iteration + 1) % accumulation == 0 or iteration + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                ema.update(model)

                if global_step % log_interval == 0:
                    denominator = max(1, log_interval * accumulation)
                    message = {key: value / denominator for key, value in running.items()}
                    running.clear()
                    logger.info(
                        "epoch=%d step=%d loss=%.6f image=%.6f ssim=%.6f lr=%.3e",
                        epoch + 1,
                        global_step,
                        message["total"],
                        message["image"],
                        message["ssim"],
                        optimizer.param_groups[0]["lr"],
                    )
                    progress.set_postfix(loss=f"{message['total']:.4f}")
                if global_step >= max_steps:
                    stop = True
                    break

        backup = ema.copy_to(model)
        metrics = validate(cfg, model, val_loader, device)
        ema.restore(model, backup)
        write_json({"epoch": epoch + 1, "step": global_step, **metrics}, run_dir / "validation.json")
        logger.info("Validation: %s", " ".join(f"{k}={v:.6f}" for k, v in metrics.items()))

        score = float(metrics["psnr"])
        if score > best_score:
            best_score = score
            save_training_checkpoint(
                run_dir / "best.pth",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                global_step,
                best_score,
                cfg,
                ema_state=ema.state_dict(),
            )
            write_json(metrics, run_dir / "best_metrics.json")
        save_training_checkpoint(
            run_dir / "latest.pth",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            global_step,
            best_score,
            cfg,
            ema_state=ema.state_dict(),
        )
        logger.info("Epoch finished in %.1f min", (time.time() - epoch_start) / 60.0)
        model.train()
        if stop:
            break
    logger.info("Training complete. best_psnr=%.6f", best_score)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", default=None)
    parser.add_argument("--val-root", default=None)
    parser.add_argument("--output", default="ckpt")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--accumulation", type=int, default=None)
    parser.add_argument("--trusted-checkpoint", action="store_true")
    args = parser.parse_args()
    cfg = settings.get_config()
    for part, root in (("train", args.train_root), ("val", args.val_root)):
        if root:
            cfg["data"][part]["roots"] = [root]
        if args.workers is not None:
            cfg["data"][part]["num_workers"] = args.workers
        if args.patch_size is not None:
            cfg["data"][part]["patch_size"] = args.patch_size
    if args.batch_size is not None:
        cfg["data"]["train"]["batch_size"] = args.batch_size
    for key, value in (("max_steps", args.max_steps), ("epochs", args.epochs),
                       ("precision", args.precision), ("accumulation_steps", args.accumulation)):
        if value is not None:
            cfg["train"][key] = value
    if min(cfg["train"]["max_steps"], cfg["train"]["epochs"], cfg["train"]["accumulation_steps"]) < 1:
        raise ValueError("Training steps, epochs and accumulation must be positive")
    root = Path(args.output)
    if not args.resume and any((root / name).exists() for name in ("best.pth", "latest.pth")):
        raise FileExistsError("Output contains a checkpoint. Use a new --output or explicit --resume.")
    root.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg["seed"], cfg.get("deterministic", False))
    device = device_for(args.device)
    model = FPIRNet(cfg).to(device)
    train_loader = build_loader(cfg["data"]["train"], True)
    val_loader = build_loader(cfg["data"]["val"], False)
    if len(train_loader) == 0:
        raise ValueError("Empty training loader: reduce batch size or check the data roots.")
    write_json(environment_report(), root / "environment.json")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}; K={model.default_iterations}")
    train(cfg, model, train_loader, val_loader, device, root,
          resume=args.resume, trusted=args.trusted_checkpoint)


if __name__ == "__main__":
    main()
