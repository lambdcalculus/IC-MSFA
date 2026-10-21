from typing import Literal

import torch
from torch import Tensor

from .generators import PatternGenerator, SoftmaxGenerator
from .util import _init_raw, _entropy

class L1Generator(PatternGenerator):
    """Learnable soft pattern, given by band-wise L1 normalization of underlying weights.
    Legacy: reproduces NormalizedMSFA for `norm='l1'`. Not for new work.

    Args:
        n_bands (int): number of bands L; the tile is L x L.
        init (str): "uniform" (U(0, 1)) or "constant".
        rng (torch.Generator | None): generator for the init draw; None uses global RNG.
    """
    soft_pattern = True
    soft_forward = True

    def __init__(self, n_bands: int,
                 init: Literal["uniform", "constant"] = "uniform",
                 rng: torch.Generator | None = None):
        super().__init__()
        self.raw = _init_raw(n_bands, init, rng)

    def probs(self) -> Tensor:
        return torch.abs(self.raw) / torch.abs(self.raw).sum(dim=0)

    def forward(self) -> Tensor:
        return self.probs()

    def regularizers(self) -> dict[str, Tensor]:
        return {"entropy": _entropy(self.probs())}

    def extra_repr(self) -> str:
        return f"n_bands={self.raw.shape[0]}"

def normalized_generator(n_bands: int,
                         norm: Literal["l1", "softmax"] = "l1",
                         init: Literal["uniform", "constant"] = "uniform",
                         rng: torch.Generator | None = None) -> PatternGenerator:
    """Drop-in replacement for NormalizedMSFA. Only for legacy code."""
    match norm:
        case "softmax":
            return SoftmaxGenerator(n_bands, init=init, rng=rng)
        case "l1":
            return L1Generator(n_bands, init=init, rng=rng)
        case _:
            raise ValueError(f"unknown norm {norm!r}, expected 'l1' or 'softmax'")
