import torch.nn as nn
from torch import Tensor
from typing import List

from . import functional as F
from .util import _ensure_batch

# Uncommented right now because this is unnecessary until I make a transform that is not
# a nn.Module. Until then, using nn.Sequential is enough to compose transforms.
# # class Compose:
# #     """
# #     Composes a list of transforms into one.
# #     """
# #     def __init__(self, transforms: List[Callable]) -> None:
# #         self.transforms = transforms

# #     def __call__(self, img: Tensor) -> Tensor:
# #         for transform in self.transforms:
# #             img = transform(img)
# #         return img

# #     def __str__(self) -> str:
# #         msg = "Composed transform of:"
# #         for i, transform in enumerate(self.transforms):
# #             msg += f"\n({i}): {str(transform)}"
# #         return msg

class Select(nn.Module):
    """Selects the specified bands/channels.
    
    Args:
        bands (List[int]): list of bands/channels to select.
    """
    def __init__(self, bands: List[int]) -> None:
        super().__init__()
        self.bands = bands

    @_ensure_batch
    def forward(self, img: Tensor) -> Tensor:
        """
        Args:
            img (torch.Tensor): input tensor

        Returns:
            torch.Tensor: tensor with only the selected bands/channels
        """
        img = F.select(img, self.bands)
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(bands={self.bands})"

class MSFA(nn.Module):
    """Applies color filtering with a MSFA.

    Args:
        msfa (torch.Tensor): MSFA to use for filtering
    """
    def __init__(self, msfa: Tensor):
        super().__init__()
        # TODO: implement a class with requires_grad=True
        self.msfa = nn.Parameter(msfa, requires_grad=False)

    @_ensure_batch
    def forward(self, img: Tensor) -> Tensor:
        """
        Args:
            img (torch.Tensor): input tensor (number of channels must match MSFA)

        Returns:
            torch.Tensor: input tensor after filtering through MSFA (same shape as input)
        """
        img = F.apply_msfa(img, self.msfa)
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(msfa={self.msfa})"

class Flatten(nn.Module):
    """Flattens a sparse multi-band image into a 2D image, resembling raw sensor data.
    This is intended to be used with images that have gone through an MSFA filter."""
    def __init__(self):
        super().__init__()

    @_ensure_batch
    def forward(self, img: Tensor) -> Tensor:
        """
        Args:
            img (torch.Tensor): tensor to flatten

        Returns:
            torch.Tensor: flattened input (has singleton channel dimension)
        """
        img = F.flatten(img)
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class Unflatten(nn.Module):
    """Unflattens 2D images into the appropriate channels based on a MSFA.
    This is intended to be used with images that are (or emulate) raw sensor data.

    Args:
        msfa (torch.Tensor): MSFA to use.
    """
    def __init__(self, msfa: Tensor) -> None:
        super().__init__()
        self.msfa = nn.Parameter(msfa, requires_grad=False)

    @_ensure_batch
    def forward(self, img: Tensor) -> Tensor:
        """
        Args:
            img (torch.Tensor): tensor to unflatten (must have singleton channel dimension)

        Returns:
            torch.Tensor: unflattened tensor (has as many channels as the MSFA)
        """
        img = F.unflatten(img, self.msfa)
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(msfa={self.msfa})"
