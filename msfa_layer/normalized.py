from typing import Literal, Optional
import torch
import torch.nn as nn
from torch.nn.utils import parametrize
import transforms.functional as F

from torch import Tensor
from transforms.util import _ensure_batch

class _NormalizedL1(nn.Module):
    """Module for parametrizing NormalizedMSFA."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.abs(input) / torch.abs(input).sum(dim=0)

class _NormalizedSoftmax(nn.Module):
    """Module for parametrizing NormalizedMSFA."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return nn.functional.softmax(input, dim=0)

class NormalizedMSFA(nn.Module):
    """
    Optimizable MSFA layer with restriction that sum along first dimension must be 1.

    `measurement` picks the forward model: "masked" multiplies the cube by the MSFA
    (`apply_msfa`); "sum" sums over bands first, one reading per pixel, and scatters it back
    (`apply_msfa_sum`, proposal eq. 2). Both agree for a binary MSFA.
    """
    def __init__(self, n_bands: int,
                 norm: Literal["l1", "softmax"] = "l1",
                 init: Literal["uniform", "constant"] = "uniform",
                 generator: Optional[torch.Generator] = None,
                 measurement: Literal["masked", "sum"] = "masked"):
        super().__init__()

        if measurement not in ("masked", "sum"):
            raise ValueError(f"unknown measurement {measurement!r}, expected 'masked' or 'sum'")
        self.measurement = measurement

        self.msfa = nn.Parameter(torch.empty(n_bands, n_bands, n_bands), requires_grad=True)
        match init:
            case "uniform":
                nn.init.uniform_(self.msfa, 0, 1, generator=generator)
            case "constant":
                nn.init.ones_(self.msfa)

        match norm:
            case "l1":
                parametrize.register_parametrization(self, "msfa", _NormalizedL1())
            case "softmax":
                parametrize.register_parametrization(self, "msfa", _NormalizedSoftmax())

    @_ensure_batch
    def forward(self, img: Tensor) -> Tensor:
        """
        Args:
            img (torch.Tensor): input tensor (number of channels must match MSFA)

        Returns:
            torch.Tensor: input tensor after filtering through MSFA (same shape as input)
        """
        match self.measurement:
            case "masked":
                img = F.apply_msfa(img, self.msfa)
            case "sum":
                img = F.apply_msfa_sum(img, self.msfa)
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_bands={self.msfa.shape[0]}, measurement={self.measurement!r})"
