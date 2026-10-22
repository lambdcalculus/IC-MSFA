import torch
import torch.nn as nn
from torch import Tensor

from msfa_layer import PatternGenerator
from .encoders import Encoder

__all__ = ["MosaicModel"]


class MosaicModel(nn.Module):
    """Encoder (pattern generator + mixing) followed by a decoder (the demosaicking network).

    Args:
        encoder (Encoder): owns the pattern generator.
        decoder (nn.Module): takes the encoder's output (B, L, H, W), returns the cube.
    """
    def __init__(self, encoder: Encoder, decoder: nn.Module):
        super().__init__()
        # encoder registered first so parameters() order matches legacy
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, cube: Tensor) -> Tensor:
        return self.decoder(self.encoder(cube))

    @property
    def generator(self) -> PatternGenerator:
        return self.encoder.generator

    def param_groups(self, lr: float,
                     pattern_lr: float | None = None,
                     pattern_eps: float | None = None) -> list[dict]:
        """Optimizer groups: the pattern's parameters, then the decoder's, optionally
        with different `lr` or `eps` for each."""
        pattern = [p for p in self.encoder.parameters() if p.requires_grad]
        decoder = [p for p in self.decoder.parameters() if p.requires_grad]
        groups: list[dict] = []
        if pattern:
            g: dict = {"name": "pattern", "params": pattern,
                       "lr": lr if pattern_lr is None else pattern_lr}
            if pattern_eps is not None:
                g["eps"] = pattern_eps
            groups.append(g)
        groups.append({"name": "decoder", "params": decoder, "lr": lr})
        return groups

    @torch.no_grad()
    def pattern_record(self) -> dict:
        """Pattern information for checkpoints (not dependent on model layout)."""
        gen = self.generator
        return {
            "pattern_hard": gen.hard().cpu(),
            "pattern_soft": gen.probs().detach().cpu() if gen.soft_pattern else None,
            "encoder": type(self.encoder).__name__,
            "generator": type(gen).__name__,
        }
