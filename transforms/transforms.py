import torch.nn as nn

from . import functional as F
from .util import _ensure_batch, load_mat, load_csv

from torch import Tensor
from collections.abc import Callable, Sequence

class Compose:
    """Composes a list of transforms into one.

    Prefer `torch.nn.Sequential` if all transforms being used are subclasses of `torch.nn.Module`.

    Args:
        transforms (sequence of Callable[[Tensor], Tensor]): list of transforms to compose.
    """

    def __init__(self, transforms: Sequence[Callable[[Tensor], Tensor]]) -> None:
        self.transforms = transforms

    def __call__(self, img: Tensor) -> Tensor:
        for transform in self.transforms:
            img = transform(img)
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(transforms={self.transforms})"

class Illuminant(nn.Module):
    """Applies an illuminant.

    Applies a CIE illuminant <https://en.wikipedia.org/wiki/Standard_illuminant> to
    acquire luminance data from reflectance data.

    Should be used on inputs that match the number of elements in the illuminant.

    Args:
        illum: illuminant to use. Can directly load a torch.Tensor, or load from a
        .mat or .csv file.
    """

    def __init__(self, illum: Tensor | str) -> None:
        super().__init__()
        if isinstance(illum, str):
            if illum.endswith(".mat"):
                illum = load_mat(illum)
            elif illum.endswith(".csv"):
                illum = load_csv(illum)
            else:
                raise ValueError(f"{illum} is not a .mat or .csv file.")

        self.illum = nn.Parameter(illum, requires_grad=False)

    @_ensure_batch
    def forward(self, img: Tensor) -> Tensor:
        """
        Args:
            img: reflectance data (B x C x W x H)

        Returns:
            luminance data (B x C x W x H)
        """
        return F.apply_illuminant(img, self.illum)

    def __repr__(self):
        return f"{self.__class__.__name__}(illum={self.illum})"

class FilterSensitivity(nn.Module):
    """Applies filter sensitivity.

    This is used to emulate ground-truth data for MSFA cameras. See
    `apply_sensitivity` for more details.

    TODO: allow for loading from files

    Args:
        sens: filter sensitivities (N x C)
    """

    def __init__(self, sens: Tensor | str, transpose: bool = False) -> None:
        super().__init__()
        if isinstance(sens, str):
            if sens.endswith(".mat"):
                sens = load_mat(sens)
            elif sens.endswith(".csv"):
                sens = load_csv(sens)
            else:
                raise ValueError(f"{sens} is not a .mat or .csv file.")

        if transpose:
            sens = sens.T
        self.sens = nn.Parameter(sens, requires_grad=False)

    @_ensure_batch
    def forward(self, img: Tensor) -> Tensor:
        """
        Args:
            img: luminance data (B x C x W x H)

        Returns:
            ground-truth camera data/filter response data (B x N x W x H)
        """
        return F.apply_sensitivity(img, self.sens)

    def __repr__(self):
        return f"{self.__class__.__name__}(illum={self.illum})"

class Select(nn.Module):
    """Selects the specified bands/channels.
    
    Args:
        bands (sequence of int): list of bands/channels to select.
    """

    def __init__(self, bands: Sequence[int]) -> None:
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

    This is intended to be used with images that have gone through a MSFA filter.
    """

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
