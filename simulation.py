from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_base_grid(batch: int, height: int, width: int, device, dtype) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)


def warp_image(image: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = image.shape
    if displacement.shape[-2:] != (height, width):
        displacement = F.interpolate(
            displacement,
            size=(height, width),
            mode="bilinear",
            align_corners=True,
        )
    grid = make_base_grid(batch, height, width, image.device, image.dtype).clone()
    grid[..., 0] += displacement[:, 0] * (2.0 / max(1, width - 1))
    grid[..., 1] += displacement[:, 1] * (2.0 / max(1, height - 1))
    return F.grid_sample(
        image,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


class LocalOTFBlur(nn.Module):
    def __init__(self, window_size: int = 28, stride: int = 14, eps: float = 1e-6) -> None:
        super().__init__()
        if window_size < 8 or stride < 1 or stride > window_size:
            raise ValueError("Require window_size >= 8 and 1 <= stride <= window_size")
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.eps = float(eps)
        one = torch.hann_window(self.window_size, periodic=False)
        window = torch.outer(one, one).clamp_min(1e-3)
        self.register_buffer("window", window, persistent=False)
        fy = torch.fft.fftfreq(self.window_size)
        fx = torch.fft.fftfreq(self.window_size)
        wy, wx = torch.meshgrid(fy, fx, indexing="ij")
        self.register_buffer("freq_x", wx, persistent=False)
        self.register_buffer("freq_y", wy, persistent=False)

    def _padding(self, height: int, width: int) -> Tuple[int, int]:
        patch, stride = self.window_size, self.stride
        target_h = max(height, patch)
        target_w = max(width, patch)
        remainder_h = (target_h - patch) % stride
        remainder_w = (target_w - patch) % stride
        if remainder_h:
            target_h += stride - remainder_h
        if remainder_w:
            target_w += stride - remainder_w
        return target_h - height, target_w - width

    @staticmethod
    def _pad(value: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
        if not pad_h and not pad_w:
            return value
        height, width = value.shape[-2:]
        mode = "reflect" if height > 1 and width > 1 and pad_h < height and pad_w < width else "replicate"
        return F.pad(value, (0, pad_w, 0, pad_h), mode=mode)

    def _unfold(self, value: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = value.shape
        patch = self.window_size
        columns = F.unfold(value, kernel_size=patch, stride=self.stride)
        count = columns.shape[-1]
        return columns.transpose(1, 2).reshape(batch, count, channels, patch, patch)

    def _fold(self, patches: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
        batch, count, channels, patch, _ = patches.shape
        columns = patches.reshape(batch, count, channels * patch * patch).transpose(1, 2)
        return F.fold(columns, output_size=output_size, kernel_size=patch, stride=self.stride)

    def _normalizer(self, batch: int, output_size: Tuple[int, int], count: int, device) -> torch.Tensor:
        patch = self.window_size
        squared = self.window.float().square().reshape(1, patch * patch, 1)
        columns = squared.expand(batch, -1, count).to(device=device)
        return F.fold(columns, output_size=output_size, kernel_size=patch, stride=self.stride).clamp_min(self.eps)

    def _local_beta(self, beta: torch.Tensor) -> torch.Tensor:
        patches = self._unfold(beta.float())
        weight = self.window.float().square().view(1, 1, 1, self.window_size, self.window_size)
        return (patches * weight).sum(dim=(-2, -1)) / weight.sum().clamp_min(self.eps)

    def _otf(self, local_beta: torch.Tensor) -> torch.Tensor:
        attenuation = local_beta[..., 0].clamp_min(1e-4)
        anisotropy = local_beta[..., 1]
        orientation = local_beta[..., 2]
        wx = self.freq_x.float().view(1, 1, self.window_size, self.window_size)
        wy = self.freq_y.float().view(1, 1, self.window_size, self.window_size)
        cosine = torch.cos(orientation).unsqueeze(-1).unsqueeze(-1)
        sine = torch.sin(orientation).unsqueeze(-1).unsqueeze(-1)
        axis1 = cosine * wx + sine * wy
        axis2 = -sine * wx + cosine * wy
        exp_rho = torch.exp(anisotropy).unsqueeze(-1).unsqueeze(-1)
        exp_minus_rho = torch.exp(-anisotropy).unsqueeze(-1).unsqueeze(-1)
        quadratic = (exp_rho * axis1.square() + exp_minus_rho * axis2.square()).clamp_min(1e-8)
        return torch.exp(-attenuation.unsqueeze(-1).unsqueeze(-1) * quadratic.pow(5.0 / 6.0))

    def forward(self, image: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        original_dtype = image.dtype
        batch, _, height, width = image.shape
        pad_h, pad_w = self._padding(height, width)
        image_pad = self._pad(image.float(), pad_h, pad_w)
        beta_pad = self._pad(beta.float(), pad_h, pad_w)
        patches = self._unfold(image_pad)
        window = self.window.float().view(1, 1, 1, self.window_size, self.window_size)
        spectrum = torch.fft.fft2(patches * window, dim=(-2, -1))
        otf = self._otf(self._local_beta(beta_pad))
        output_patches = torch.fft.ifft2(
            spectrum * otf.unsqueeze(2), dim=(-2, -1)
        ).real * window
        padded_size = image_pad.shape[-2:]
        normalizer = self._normalizer(batch, padded_size, patches.shape[1], image.device)
        output = self._fold(output_patches, padded_size) / normalizer
        return output[..., :height, :width].to(original_dtype)


def _standardize(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dims = tuple(range(2, value.ndim))
    value = value - value.mean(dim=dims, keepdim=True)
    return value / value.square().mean(dim=dims, keepdim=True).sqrt().clamp_min(eps)


def _powerlaw_field(
    batch: int,
    channels: int,
    height: int,
    width: int,
    device,
    generator: Optional[torch.Generator],
    amplitude_exponent: float,
    outer_scale: float,
) -> torch.Tensor:
    noise = torch.randn(
        (batch, channels, height, width),
        device=device,
        generator=generator,
        dtype=torch.float32,
    )
    spectrum = torch.fft.fft2(noise)
    fy = torch.fft.fftfreq(height, device=device).view(1, 1, height, 1)
    fx = torch.fft.fftfreq(width, device=device).view(1, 1, 1, width)
    radius2 = fx.square() + fy.square() + float(outer_scale) ** 2
    filt = radius2.pow(-float(amplitude_exponent) / 2.0)
    filt[..., 0, 0] = 0.0
    return _standardize(torch.fft.ifft2(spectrum * filt).real)


class TurbulenceSynthesizer(nn.Module):
    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.blur = LocalOTFBlur(
            window_size=int(cfg.get("otf_window", 28)),
            stride=int(cfg.get("otf_stride", 14)),
        )
        self.max_tilt = float(cfg.get("max_tilt", 8.0))
        self.a_min = float(cfg.get("a_min", 0.15))
        self.a_max = float(cfg.get("a_max", 11.0))
        self.rho_max = float(cfg.get("rho_max", 0.75))
        self.a_spatial_modulation = float(cfg.get("a_spatial_modulation", 0.3))
        self.noise_std_max = float(cfg.get("noise_std_max", 0.0))
        self.scintillation = float(cfg.get("scintillation", 0.0))
        self.color_jitter = float(cfg.get("color_jitter", 0.0))
        self.residual_blur_prob = float(cfg.get("residual_blur_prob", 0.0))
        self.isotropic_blur = bool(cfg.get("isotropic_blur", False))

    def forward(
        self,
        clean: torch.Tensor,
        severity: torch.Tensor,
        exposure: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        clean_float = clean.float().clamp(0.0, 1.0)
        batch, _, height, width = clean.shape
        device = clean.device
        severity = severity.to(device=device, dtype=torch.float32).reshape(batch, 1, 1, 1)
        exposure = exposure.to(device=device, dtype=torch.float32).reshape(batch, 1, 1, 1)

        phase = _powerlaw_field(batch, 1, height, width, device, generator, 11.0 / 6.0, 0.025)
        grad_x = 0.5 * (torch.roll(phase, -1, -1) - torch.roll(phase, 1, -1))
        grad_y = 0.5 * (torch.roll(phase, -1, -2) - torch.roll(phase, 1, -2))
        displacement = _standardize(torch.cat((grad_x, grad_y), dim=1))
        tilt_rms = 0.25 + severity * (self.max_tilt - 0.25)
        displacement = self.max_tilt * torch.tanh(displacement * tilt_rms / max(self.max_tilt, 1e-3))

        noise = _powerlaw_field(batch, 3, height, width, device, generator, 2.2, 0.05)
        center = self.a_min + severity * (1.5 + 5.5 * exposure)
        attenuation = center * (1.0 + self.a_spatial_modulation * torch.tanh(noise[:, 0:1]))
        attenuation = attenuation.clamp(self.a_min, self.a_max)
        anisotropy = self.rho_max * severity * torch.tanh(noise[:, 1:2])
        orientation = (math.pi / 2.0) * torch.tanh(noise[:, 2:3])
        if self.isotropic_blur:
            anisotropy = torch.zeros_like(anisotropy)
            orientation = torch.zeros_like(orientation)
        beta = torch.cat((attenuation, anisotropy, orientation), dim=1)

        degraded = self.blur(warp_image(clean_float, displacement), beta)
        if self.scintillation > 0:
            illumination = _powerlaw_field(batch, 1, height, width, device, generator, 2.5, 0.025)
            degraded = degraded * torch.exp(self.scintillation * severity * illumination).clamp(0.85, 1.15)
        if self.color_jitter > 0:
            gain = 1.0 + self.color_jitter * torch.randn((batch, 3, 1, 1), device=device, generator=generator)
            degraded = degraded * gain
        if self.residual_blur_prob > 0:
            selector = torch.rand((batch, 1, 1, 1), device=device, generator=generator)
            mild = F.avg_pool2d(degraded, 3, stride=1, padding=1)
            mix = (selector < self.residual_blur_prob).to(degraded.dtype) * 0.15
            degraded = degraded * (1.0 - mix) + mild * mix
        if self.noise_std_max > 0:
            sigma = self.noise_std_max * severity * torch.rand((batch, 1, 1, 1), device=device, generator=generator)
            degraded = degraded + sigma * torch.randn(degraded.shape, device=device, generator=generator)

        return {
            "input": degraded.clamp(0.0, 1.0).to(clean.dtype),
            "clean": clean,
            "severity": severity.flatten(),
            "exposure": exposure.flatten(),
        }
