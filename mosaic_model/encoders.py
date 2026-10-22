import warnings

from torch import Tensor
import torch.nn as nn

import transforms.functional as F
from transforms.util import _ensure_batch
from msfa_layer import PatternGenerator, FixedPattern

__all__ = ["Encoder", "MaskedCube", "BandSum", "UnsoundEncoding"]


class UnsoundEncoding(UserWarning):
    """A generator/encoder pair whose pattern learning may not work as intended."""

class Encoder(nn.Module):
    """Mixes the per-band observed cube with a generated pattern into the decoder's input.

    For one-hot patterns, an encoder's output must be band selection
    according to the pattern. Where encoders differ is in what they do with a soft pattern
    and in the gradient they give the pattern, described by two flags:

        scalar_bottleneck: under a soft pattern, the output depends on one number per
            pixel, so there is a bottleneck. Generators with `soft_forward` need it.
        linear_in_mask: the output is linear in the pattern, so its gradient is a
            first-order estimate of swapping a pixel's band. Straight-through generators
            (`soft_pattern` without `soft_forward`) need it.

    Subclasses set the flags and implement `mix`.

    Args:
        generator (PatternGenerator): produces this step's pattern tile (L, P, P).
    """
    scalar_bottleneck: bool = False
    linear_in_mask: bool = False

    def __init__(self, generator: PatternGenerator):
        super().__init__()
        self.generator = generator
        self._check(generator)

    def _check(self, generator: PatternGenerator) -> None:
        enc, gen = type(self).__name__, type(generator).__name__
        if generator.soft_forward and not self.scalar_bottleneck:
            warnings.warn(f"{enc} with {gen}: a soft pattern gives no bottleneck here, so the "
                          "reconstruction loss has difficulty ranking patterns (legacy FAPESP setup)",
                          UnsoundEncoding, stacklevel=3)
        if generator.soft_pattern and not generator.soft_forward and not self.linear_in_mask:
            warnings.warn(f"{enc} with {gen}: the straight-through gradient through this "
                          "encoder matches no real pixel swap", UnsoundEncoding, stacklevel=3)

    def mix(self, cube: Tensor, pattern: Tensor) -> Tensor:
        """Combine a batched cube (B, L, H, W) with a pattern tile (L, P, P)."""
        raise NotImplementedError

    @_ensure_batch
    def forward(self, cube: Tensor) -> Tensor:
        return self.mix(cube, self.generator())

    def freeze(self) -> None:
        """Replace the generator with its hard pattern (the freeze event)."""
        self.generator = FixedPattern(self.generator.hard())

class MaskedCube(Encoder):
    """Cube times the tiled pattern (`apply_msfa`): the FAPESP runs' input."""
    scalar_bottleneck = False # many scalars per pixel
    linear_in_mask = True

    def mix(self, cube: Tensor, pattern: Tensor) -> Tensor:
        return F.apply_msfa(cube, pattern)


class BandSum(Encoder):
    """Sum over bands of cube times pattern, scattered back with the pattern."""
    scalar_bottleneck = True
    linear_in_mask = False # quadratic in the pattern

    def mix(self, cube: Tensor, pattern: Tensor) -> Tensor:
        return F.apply_msfa_sum_scatter(cube, pattern)
