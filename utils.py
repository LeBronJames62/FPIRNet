"""Shared image I/O, checkpoint handling and training utilities."""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import platform
import random
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(root, recursive=True):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    items = root.rglob("*") if recursive else root.glob("*")
    return sorted(p for p in items if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def image_map(root):
    root = Path(root)
    result = {}
    for path in list_images(root):
        key = path.relative_to(root).with_suffix("").as_posix()
        if key in result:
            raise ValueError(f"Duplicate extension-insensitive key: {root} / {key}")
        result[key] = path
    if not result:
        raise ValueError(f"No images found: {root}")
    return result


def load_image(path, exif=False):
    try:
        with Image.open(path) as image:
            image.load()
            if exif:
                image = ImageOps.exif_transpose(image)
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    except Exception as exc:
        raise RuntimeError(f"Cannot decode {path}: {exc}") from exc
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def save_image(tensor, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = tensor.detach().float().cpu().clamp(0, 1)
    if value.ndim == 4:
        if value.shape[0] != 1:
            raise ValueError("save_image expects one image")
        value = value[0]
    array = (value.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
    temporary = path.with_name(path.stem + ".partial.png")
    Image.fromarray(array).save(temporary, format="PNG")
    with Image.open(temporary) as check:
        check.load()
    temporary.replace(path)


def write_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    temporary.replace(path)


def write_csv(rows, path):
    rows = list(rows)
    if not rows:
        raise ValueError(f"Refusing to write empty metrics: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def device_for(name):
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Select --device cpu explicitly for a CPU run.")
    return device


def autocast_context(device, precision):
    if precision not in {"fp32", "bf16", "fp16"}:
        raise ValueError(f"Unsupported precision: {precision}")
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16)


def environment_report():
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": str(torch.__version__), "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
    }


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def read_checkpoint(path, trusted=False, map_location="cpu"):
    if not Path(path).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # Unsafe pickle loading is opt-in and only for the user's own training file.
    try:
        return torch.load(path, map_location=map_location, weights_only=not trusted)
    except Exception as exc:
        if not trusted:
            raise RuntimeError(
                f"Could not load {path} with weights_only=True. Check the file. "
                "For your own trusted legacy training checkpoint only, add --trusted-checkpoint."
            ) from exc
        raise


def extract_model_state(payload, prefer_ema=True):
    if not isinstance(payload, dict):
        raise ValueError("Expected a checkpoint dictionary")
    if payload and all(isinstance(value, torch.Tensor) for value in payload.values()):
        return dict(payload), "raw_state_dict"
    if "state_dict" in payload:
        state, source = dict(payload["state_dict"]), "state_dict"
    elif "model" in payload:
        state, source = dict(payload["model"]), "model"
        if prefer_ema and payload.get("ema"):
            state.update(payload["ema"])
            source = "model+ema"
    else:
        raise KeyError("Checkpoint needs state_dict, model(+ema), or a plain tensor dictionary")
    if not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Invalid model state dictionary")
    return state, source


def load_weights(model, path, prefer_ema=True, trusted=False):
    payload = read_checkpoint(path, trusted=trusted)
    if isinstance(payload, dict) and "fixed_step" in payload:
        expected = float(model.update_operator.fixed_step)
        if abs(float(payload["fixed_step"]) - expected) > 1e-12:
            raise ValueError("Checkpoint fixed_step does not match config.py")
    state, source = extract_model_state(payload, prefer_ema)
    model.load_state_dict(state, strict=True)
    return payload, source


def save_training_checkpoint(path, model, optimizer, scheduler, scaler,
                             epoch, global_step, best_score, config, ema_state=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(), "ema": ema_state,
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(), "epoch": int(epoch),
        "global_step": int(global_step), "best_score": float(best_score), "config": config,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def setup_logger(path, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    for old in logger.handlers[:]:
        logger.removeHandler(old)
        old.close()
    formatter = logging.Formatter("%(asctime)s | %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(path, encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
        self.shadow = OrderedDict(
            (k, v.detach().clone()) for k, v in model.state_dict().items()
            if torch.is_floating_point(v)
        )

    @torch.no_grad()
    def update(self, model):
        state = model.state_dict()
        for key, value in self.shadow.items():
            value.mul_(self.decay).add_(state[key].detach(), alpha=1.0 - self.decay)

    def state_dict(self):
        return self.shadow

    @torch.no_grad()
    def copy_to(self, model):
        state = model.state_dict()
        backup = {k: state[k].detach().clone() for k in self.shadow}
        for key, value in self.shadow.items():
            state[key].copy_(value)
        return backup

    @staticmethod
    @torch.no_grad()
    def restore(model, backup):
        state = model.state_dict()
        for key, value in backup.items():
            state[key].copy_(value)
