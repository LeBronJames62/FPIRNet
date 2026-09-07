from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def ssim_map(x: torch.Tensor, y: torch.Tensor, window: int = 11) -> torch.Tensor:
    pad = window // 2
    mu_x = F.avg_pool2d(x, window, stride=1, padding=pad)
    mu_y = F.avg_pool2d(y, window, stride=1, padding=pad)
    sigma_x = F.avg_pool2d(x * x, window, stride=1, padding=pad) - mu_x.square()
    sigma_y = F.avg_pool2d(y * y, window, stride=1, padding=pad) - mu_y.square()
    sigma_xy = F.avg_pool2d(x * y, window, stride=1, padding=pad) - mu_x * mu_y
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    return numerator / denominator.clamp_min(1e-8)


def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    mse = (pred - target).square().flatten(1).mean(dim=1)
    return -10.0 * torch.log10(mse.clamp_min(eps))


def ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ssim_map(pred, target).flatten(1).mean(dim=1)



def charbonnier(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((pred - target).square() + eps * eps).mean()


def restoration_loss(
    cfg: Dict,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    loss_cfg = cfg.get("loss", {})
    image_loss = charbonnier(prediction, target)
    structural_loss = 1.0 - ssim_map(prediction, target).mean()
    total = (
        float(loss_cfg.get("lambda_image", 1.0)) * image_loss
        + float(loss_cfg.get("lambda_ssim", 0.1)) * structural_loss
    )
    return {"total": total, "image": image_loss, "ssim": structural_loss}
