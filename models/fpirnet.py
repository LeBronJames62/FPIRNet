from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        activation: bool = True,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = nn.GroupNorm(_groups(out_channels), out_channels)
        self.act = nn.SiLU(inplace=True) if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.block = nn.Sequential(
            ConvGNAct(channels, hidden, 3),
            nn.Conv2d(hidden, channels, 3, padding=1),
            nn.GroupNorm(_groups(channels), channels),
        )
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.block(x)


class TinyUNet(nn.Module):
    """Compact two-level U-Net used by the shared recurrent operator."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 20,
        blocks: int = 1,
    ) -> None:
        super().__init__()
        base = int(base_channels)
        count = int(blocks)
        self.stem = ConvGNAct(in_channels, base, 3)
        self.enc1 = nn.Sequential(*[ResidualBlock(base) for _ in range(count)])
        self.down1 = ConvGNAct(base, base * 2, 3, stride=2)
        self.enc2 = nn.Sequential(*[ResidualBlock(base * 2) for _ in range(count)])
        self.down2 = ConvGNAct(base * 2, base * 4, 3, stride=2)
        self.mid = nn.Sequential(ResidualBlock(base * 4), ResidualBlock(base * 4))
        self.up2 = ConvGNAct(base * 4 + base * 2, base * 2, 3)
        self.dec2 = nn.Sequential(*[ResidualBlock(base * 2) for _ in range(count)])
        self.up1 = ConvGNAct(base * 2 + base, base, 3)
        self.dec1 = nn.Sequential(*[ResidualBlock(base) for _ in range(count)])
        self.head = nn.Conv2d(base, out_channels, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(self.stem(x))
        enc2 = self.enc2(self.down1(enc1))
        middle = self.mid(self.down2(enc2))
        up2 = F.interpolate(middle, size=enc2.shape[-2:], mode="bilinear", align_corners=False)
        up2 = self.dec2(self.up2(torch.cat((up2, enc2), dim=1)))
        up1 = F.interpolate(up2, size=enc1.shape[-2:], mode="bilinear", align_corners=False)
        up1 = self.dec1(self.up1(torch.cat((up1, enc1), dim=1)))
        return self.head(up1)



class SharedUpdateOperator(nn.Module):
    """One observation-anchored bounded update.

    The final published checkpoint was trained with a nine-channel stem whose
    third RGB slot is always zero. Only ``x^k`` and ``y`` carry information.
    The zero slot is retained here solely for exact checkpoint/state-dict
    compatibility with the released model.
    """

    def __init__(
        self,
        base_channels: int = 20,
        blocks: int = 1,
        max_step: float = 0.5,
        fixed_step: float = 0.1268,
    ) -> None:
        super().__init__()
        self.net = TinyUNet(9, 4, base_channels=base_channels, blocks=blocks)
        self.max_step = float(max_step)
        self.fixed_step = float(fixed_step)
        if not 0.0 < self.fixed_step <= self.max_step:
            raise ValueError("Require 0 < fixed_step <= max_step")

        # Kept as a frozen parameter to preserve the exact published checkpoint
        # key layout. It is ignored because the final model uses fixed eta.
        ratio = min(max(self.fixed_step / self.max_step, 1e-4), 1.0 - 1e-4)
        self.raw_step = nn.Parameter(
            torch.tensor(math.log(ratio / (1.0 - ratio))),
            requires_grad=False,
        )

    def context(self, state: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
        if state.shape != observation.shape:
            raise ValueError(
                f"State/observation shape mismatch: {state.shape} vs {observation.shape}"
            )
        zero_slot = torch.zeros_like(state)
        return torch.cat((state, observation, zero_slot), dim=1)

    def forward(self, state: torch.Tensor, observation: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw = self.net(self.context(state, observation))
        residual = torch.tanh(raw[:, :3])
        gate_logit = raw[:, 3:4]
        gate = torch.sigmoid(gate_logit)
        step = state.new_tensor(self.fixed_step)
        update = step * gate * residual
        next_state = (state + update).clamp(0.0, 1.0)
        return {
            "next": next_state,
            "residual": residual,
            "gate_logit": gate_logit,
            "gate": gate,
            "update": update,
            "step": step,
        }


class FPIRNet(nn.Module):
    """Final four-stage weight-shared FPIR-Net."""

    def __init__(self, cfg: Dict | None = None) -> None:
        super().__init__()
        cfg = {} if cfg is None else cfg
        model_cfg = cfg.get("model", cfg)
        self.default_iterations = int(model_cfg.get("default_iterations", 4))
        if self.default_iterations < 1:
            raise ValueError("default_iterations must be >= 1")
        self.update_operator = SharedUpdateOperator(
            base_channels=int(model_cfg.get("base_channels", 20)),
            blocks=int(model_cfg.get("blocks", 1)),
            max_step=float(model_cfg.get("max_step", 0.5)),
            fixed_step=float(model_cfg.get("fixed_step", 0.1268)),
        )

    def forward(
        self,
        observation: torch.Tensor,
        iterations: int | None = None,
    ) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        count = self.default_iterations if iterations is None else int(iterations)
        if count < 1:
            raise ValueError("iterations must be >= 1")

        state = observation
        states: List[torch.Tensor] = [state]
        updates: List[torch.Tensor] = []
        gates: List[torch.Tensor] = []
        gate_logits: List[torch.Tensor] = []
        residuals: List[torch.Tensor] = []
        for _ in range(count):
            result = self.update_operator(state, observation)
            state = result["next"]
            states.append(state)
            updates.append(result["update"])
            gates.append(result["gate"])
            gate_logits.append(result["gate_logit"])
            residuals.append(result["residual"])

        return {
            "restored": state,
            "states": states,
            "updates": updates,
            "gates": gates,
            "gate_logits": gate_logits,
            "residuals": residuals,
            "step": observation.new_tensor(self.update_operator.fixed_step),
        }
