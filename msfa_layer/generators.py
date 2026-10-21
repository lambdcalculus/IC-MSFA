from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .util import _init_raw, _entropy

__all__ = ["PatternGenerator", "FixedPattern", "SoftmaxGenerator"]

class PatternGenerator(nn.Module):
    """Base class for MSFA generator component. Generates (hard or soft) MSFA for this step,
    with shape (L, P, P). The pattern is later applied to the image by an Encoder (mosaic_model/encoders.py).

    Subclasses set `soft_pattern` if it uses a soft pattern internally, and `soft_forward` if
    it outputs a soft pattern rather than a binarized one.
    """
    soft_pattern: bool = False
    soft_forward: bool = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.soft_forward and not cls.soft_pattern:
            raise TypeError(f"{cls.__name__}: soft_forward=True requires soft_pattern=True")

    def forward(self) -> Tensor:
        """The tile used in this step's forward pass (may carry a gradient)."""
        raise NotImplementedError

    def probs(self) -> Tensor:
        """Soft view of the pattern."""
        raise NotImplementedError

    @torch.no_grad()
    def hard(self) -> Tensor:
        """One-hot pattern. Does not accumulate gradients."""
        p = self.probs()
        return torch.zeros_like(p).scatter_(0, p.argmax(dim=0, keepdim=True), 1.0)

    def regularizers(self) -> dict[str, Tensor]:
        """Unweighted regularization loss terms this generator contributes."""
        return {}

    def on_epoch(self, epoch: int, total_epochs: int) -> None:
        """Called before each training epoch for any epoch-dependent changes. Epochs start from 1."""
        pass

class FixedPattern(PatternGenerator):
    """Generates a constant one-hot pattern.

    Args:
        pattern (Tensor): fixed square pattern to use (L x P x P).
    """
    soft_pattern = False
    soft_forward = False
    pattern: Tensor

    def __init__(self, pattern: Tensor | np.ndarray):
        super().__init__()
        p = torch.as_tensor(pattern, dtype=torch.float32).detach().clone()
        if p.ndim != 3 or p.shape[1] != p.shape[2]:
            raise ValueError(f"pattern must be (L, P, P), got shape {tuple(p.shape)}")
        if not (((p == 0) | (p == 1)).all() and (p.sum(dim=0) == 1).all()):
            raise ValueError("pattern must be one-hot over its first (band) axis")
        self.register_buffer("pattern", p)

    def forward(self) -> Tensor:
        return self.pattern

    def probs(self) -> Tensor:
        return self.pattern

    @torch.no_grad()
    def hard(self) -> Tensor:
        return self.pattern.clone()

    def extra_repr(self) -> str:
        L, P, _ = self.pattern.shape
        return f"n_bands={L}, tile={P}"

class SoftmaxGenerator(PatternGenerator):
    """
    Learnable soft pattern, given by band-wise softmax of underlying weights (L, L, L).
    Optional annealing as in Chakrabarti 2016 by changing the inverse temperature over
    epochs.

    Args:
        n_bands (int): number of bands L; the tile is L x L.
        init (str): "normal" (N(0, init_std^2)), "uniform" (U(0, 1), as NormalizedMSFA),
            or "constant" (all ones, a uniform pattern).
        init_std (float): standard deviation for init="normal". Lower means patterns take
            less epochs to change, but a higher lr can also affect this.
        alpha_end (float | None): anneal the inverse temperature alpha from 1 to this over
            the joint phase, quadratically.
            None: alpha stays 1.
        rng (torch.Generator | None): generator for the init draw; None uses the global RNG.
    """
    soft_pattern = True
    soft_forward = True
    alpha: Tensor

    def __init__(self, n_bands: int,
                 init: Literal["normal", "uniform", "constant"] = "normal",
                 init_std: float = 0.05,
                 alpha_end: Optional[float] = None,
                 rng: torch.Generator | None = None):
        super().__init__()
        if init == "normal":
            self.raw = nn.Parameter(torch.empty(n_bands, n_bands, n_bands))
            nn.init.normal_(self.raw, 0.0, init_std, generator=rng)
        else:
            self.raw = _init_raw(n_bands, init, rng)
        self.alpha_end = alpha_end
        self.register_buffer("alpha", torch.tensor(1.0))

    def probs(self) -> Tensor:
        return nn.functional.softmax(self.alpha * self.raw, dim=0)

    def forward(self) -> Tensor:
        return self.probs()

    def regularizers(self) -> dict[str, Tensor]:
        return {"entropy": _entropy(self.probs())}

    def on_epoch(self, epoch: int, total_epochs: int) -> None:
        if self.alpha_end is None:
            return
        frac = (epoch - 1) / max(total_epochs - 1, 1)
        self.alpha.fill_(1 + (self.alpha_end - 1) * frac ** 2)

    def extra_repr(self) -> str:
        s = f"n_bands={self.raw.shape[0]}"
        return s + (f", alpha_end={self.alpha_end}" if self.alpha_end is not None else "")

